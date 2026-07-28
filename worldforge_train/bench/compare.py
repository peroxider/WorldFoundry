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
    if base_clips <= 0:
        return 0.0
    return round(eval_clips / base_clips, 4)


def compare(baseline_path: Path, eval_path: Path, output_path: Path) -> str:
    """Compare baseline + eval JSON, write report.md, return GO or NO-GO."""
    baseline_path = Path(baseline_path)
    eval_path = Path(eval_path)
    output_path = Path(output_path)
    missing = [str(path) for path in (baseline_path, eval_path) if not path.exists()]
    if missing:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("# W9 MindSpeed-MM 评估报告\n\n**结论**：NO-GO\n\n缺失文件：" + ", ".join(missing) + "\n")
        return "NO-GO"
    base = _load(baseline_path)
    ev = _load(eval_path)

    # GO item 1: eval JSON has a finite non-negative numeric throughput.
    clips = ev.get("clips_per_s")
    item1 = isinstance(clips, (int, float)) and not math.isnan(clips) and clips >= 0

    # GO item 2: no early termination (loss_curve length == measurement_steps)
    item2 = len(ev.get("loss_curve", [])) == ev.get("measurement_steps", 0)

    # GO item 3: throughput >= 70% of FSDP baseline
    ratio = _ratio(ev.get("clips_per_s", 0.0), base.get("clips_per_s", 1.0))
    item3 = ratio >= 0.7

    # GO item 4: loss curve within atol=1e-2
    base_curve = np.array(base.get("loss_curve", [0.0]))
    eval_curve = np.array(ev.get("loss_curve", [0.0]))
    if len(base_curve) == len(eval_curve) and len(base_curve) > 0:
        max_diff = float(np.max(np.abs(eval_curve - base_curve)))
        item4 = max_diff <= 1e-2
    else:
        max_diff = float("inf")
        item4 = False

    verdict = "GO" if (item1 and item2 and item3 and item4) else "NO-GO"
    failed_items = [n for n, ok in [(1, item1), (2, item2), (3, item3), (4, item4)] if not ok]

    report = f"""# W9 MindSpeed-MM 评估报告

日期：{date.today().isoformat()}
FSDP baseline clips/s：{base.get('clips_per_s', 0.0)}
MM eval clips/s：{ev.get('clips_per_s', 0.0)}
Ratio：{ratio}

## GO/NO-GO 门

| # | 项 | 状态 | 详情 |
| 1 | 加载成功 | {'✅' if item1 else '❌'} | clips_per_s 字段{'存在' if item1 else '缺失'} |
| 2 | {ev.get('measurement_steps', 0)} 步无崩溃 | {'✅' if item2 else '❌'} | loss_curve 长度 {len(ev.get('loss_curve', []))} |
| 3 | 吞吐 ≥ FSDP × 0.7 | {'✅' if item3 else '❌'} | ratio={ratio:.2f} |
| 4 | Loss 曲线 atol=1e-2 | {'✅' if item4 else '❌'} | max abs diff={max_diff} |

## 决策

**结论**：{verdict}
**失败项**：{', '.join(str(n) for n in failed_items) if failed_items else '无'}
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    return verdict