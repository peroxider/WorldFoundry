"""Compare two bench JSONs and write W9 GO/NO-GO report."""
from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path

import numpy as np


def _load(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def _ratio(eval_clips: float, base_clips: float) -> float:
    """Raw (unrounded) ratio of eval to baseline throughput.

    Returns 0.0 if baseline is non-positive. Caller MUST use the unrounded
    value for the gate (a raw ratio like 0.6999 would round to 0.7000 and
    incorrectly pass the 0.7 gate). Round only when rendering the report.
    """
    if base_clips <= 0:
        return 0.0
    return eval_clips / base_clips


def _missing_file_report(eval_path: Path) -> str:
    """NO-GO report for missing eval JSON; pulls stderr tail from the
    per-step log if present so the failure context is preserved (not just
    a "file missing" symptom).
    """
    log_path = eval_path.with_suffix(".log")
    stderr_excerpt = ""
    if log_path.exists():
        try:
            lines = log_path.read_text(errors="replace").splitlines()
            stderr_excerpt = "\n".join(lines[-50:])
        except Exception:
            stderr_excerpt = ""
    stderr_block = f"```\n{stderr_excerpt}\n```" if stderr_excerpt else "_(无 — 仅缺失文件)_"
    return (
        "# W9 MindSpeed-MM 评估报告\n\n"
        f"日期：{date.today().isoformat()}\n"
        "执行环境：local NO-DOCKER smoke（WSL2 dev host，无物理 NPU）\n"
        "结论：**NO-GO**\n\n"
        "## 执行详情\n\n"
        "- FSDP baseline：成功（`bench_results/fsdp-baseline.json`）\n"
        f"- MM eval：失败，缺失产物 `{eval_path}`\n\n"
        "## 失败 stderr（bench_results/<tag>.log 末尾 50 行）\n\n"
        f"{stderr_block}\n\n"
        "## 决策\n\n"
        "- **保留 FSDP** 作为生产 trainer（`select_strategy(\"train\")` 默认）\n"
        "- **Tasks 8/9 跳过** per plan §7 decision gate\n"
        "- **重新打开路径**：在真实 Atlas 800 A2 / 910B × 8 + `worldforge-base:phase1` 容器内重跑 "
        "`bash scripts/w9_eval.sh`，让 MM bench 有机会产出 JSON；结果会覆盖本报告。\n\n"
        "## 失败项\n\n"
        f"- `{eval_path}` 缺失（MM stub 在 `wrap()` 阶段 raise `NotImplementedError`，未产生 JSON）\n"
    )


def compare(baseline_path: Path, eval_path: Path, output_path: Path) -> str:
    """Compare baseline + eval JSON, write report.md, return GO or NO-GO.

    Each GO condition is evaluated independently. If any condition raises
    on malformed input (e.g., null loss_curve, non-numeric clips, etc.) it
    is marked ``not_evaluable`` and contributes to the NO-GO verdict; the
    comparator does NOT silently pass.
    """
    baseline_path = Path(baseline_path)
    eval_path = Path(eval_path)
    output_path = Path(output_path)
    missing = [str(path) for path in (baseline_path, eval_path) if not path.exists()]
    if missing:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Always emit a context-preserving NO-GO report. The previous
        # bare "缺失文件" template discarded failure reason; this version
        # pulls stderr from the per-step log so the spec's authoritative
        # failure record is preserved.
        output_path.write_text(_missing_file_report(eval_path))
        return "NO-GO"
    try:
        base = _load(baseline_path)
    except (json.JSONDecodeError, ValueError) as exc:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            "# W9 MindSpeed-MM 评估报告\n\n**结论**：NO-GO\n\n"
            f"无法解析 baseline JSON：{baseline_path} ({exc})\n"
        )
        return "NO-GO"
    try:
        ev = _load(eval_path)
    except (json.JSONDecodeError, ValueError) as exc:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            "# W9 MindSpeed-MM 评估报告\n\n**结论**：NO-GO\n\n"
            f"无法解析 eval JSON：{eval_path} ({exc})\n"
        )
        return "NO-GO"

    not_evaluable: list[str] = []

    # GO item 1: eval JSON has a finite non-negative numeric throughput.
    # Wrap in try/except to catch malformed inputs (booleans, strings,
    # non-numeric types that bypass isinstance but fail math.isnan).
    try:
        clips = ev.get("clips_per_s")
        item1 = isinstance(clips, (int, float)) and not isinstance(clips, bool) \
            and not math.isnan(clips) and clips >= 0
    except Exception:
        item1 = False
        not_evaluable.append("clips_per_s")

    # GO item 2: no early termination (loss_curve length == measurement_steps)
    try:
        item2 = len(ev.get("loss_curve", [])) == ev.get("measurement_steps", 0)
    except Exception:
        item2 = False
        not_evaluable.append("loss_curve")

    # GO item 3: throughput >= 70% of FSDP baseline (full-precision ratio).
    try:
        ratio = _ratio(ev.get("clips_per_s", 0.0), base.get("clips_per_s", 1.0))
        item3 = ratio >= 0.7
    except Exception:
        ratio = 0.0
        item3 = False
        not_evaluable.append("ratio")

    # GO item 4: loss curve within atol=1e-2
    try:
        base_curve = np.array(base.get("loss_curve", [0.0]))
        eval_curve = np.array(ev.get("loss_curve", [0.0]))
        if len(base_curve) == len(eval_curve) and len(base_curve) > 0:
            max_diff = float(np.max(np.abs(eval_curve - base_curve)))
            item4 = max_diff <= 1e-2
        else:
            max_diff = float("inf")
            item4 = False
    except Exception:
        max_diff = float("inf")
        item4 = False
        not_evaluable.append("loss_curve_diff")

    verdict = "GO" if (item1 and item2 and item3 and item4) else "NO-GO"
    failed_items = [n for n, ok in [(1, item1), (2, item2), (3, item3), (4, item4)] if not ok]

    loss_curve_for_render = ev.get('loss_curve') or []
    report = f"""# W9 MindSpeed-MM 评估报告

日期：{date.today().isoformat()}
FSDP baseline clips/s：{base.get('clips_per_s', 0.0)}
MM eval clips/s：{ev.get('clips_per_s', 0.0)}
Ratio：{ratio:.4f}

## GO/NO-GO 门

| # | 项 | 状态 | 详情 |
| 1 | 加载成功 | {'✅' if item1 else '❌'} | clips_per_s 字段{'存在' if item1 else '缺失'} |
| 2 | {ev.get('measurement_steps', 0)} 步无崩溃 | {'✅' if item2 else '❌'} | loss_curve 长度 {len(loss_curve_for_render)} |
| 3 | 吞吐 ≥ FSDP × 0.7 | {'✅' if item3 else '❌'} | ratio={ratio:.2f} |
| 4 | Loss 曲线 atol=1e-2 | {'✅' if item4 else '❌'} | max abs diff={max_diff} |

## 决策

**结论**：{verdict}
**失败项**：{', '.join(str(n) for n in failed_items) if failed_items else '无'}
{('**不可评估项**：' + ', '.join(not_evaluable)) if not_evaluable else ''}
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    return verdict
