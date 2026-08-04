"""Aggregate W22.7 arm artifacts into the experiment report.

Reads bench_results/w22_7/arm_{a,b}/{metrics.jsonl,throughput.json} and
writes a markdown report to docs/superpowers/specs/. Any missing artifact
becomes a ``TBD`` placeholder — the report skeleton is valid before the
Atlas window produces numbers (design §9.2 acceptance gates included).

Usage:
    python scripts/w22_7_build_report.py [--results bench_results/w22_7]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from worldforge_train.eval.jepa_probing import read_jsonl

REPORT_PATH = Path(
    "docs/superpowers/specs/2026-08-04-w22-7-ab-experiment-report.md"
)
GATE_COSINE = 0.6  # design §9.2 acceptance: arm B recon cosine >= 0.6 @ 2k steps


def _load_arm(results: Path, arm: str) -> tuple[list[dict], dict | None]:
    metrics = read_jsonl(results / arm / "metrics.jsonl")
    tp_file = results / arm / "throughput.json"
    throughput = json.loads(tp_file.read_text()) if tp_file.is_file() else None
    return metrics, throughput


def _load_pilot_verdict(results: Path, arm: str) -> dict | None:
    f = results / arm / "pilot" / "verdict.json"
    return json.loads(f.read_text()) if f.is_file() else None


def _fmt(v, nd=4):
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "TBD"


def _last(metrics: list[dict], key: str):
    return metrics[-1][key] if metrics and key in metrics[-1] else None


def build_report(results: Path, out: Path = REPORT_PATH) -> Path:
    ma, ta = _load_arm(results, "a")
    mb, tb = _load_arm(results, "b")
    va = _load_pilot_verdict(results, "a")
    vb = _load_pilot_verdict(results, "b")

    def _pilot_row(v, arm):
        if not v:
            return f"| {arm} | TBD（pilot 未跑）| — | — |"
        fails = [c for c in v["checks"] if c["status"] == "fail"]
        warns = [c["name"] for c in v["checks"] if c["status"] == "warn"]
        fail_txt = ", ".join(c["name"] for c in fails) or "—"
        warn_txt = ", ".join(warns) or "—"
        icon = "✅" if v["verdict"] == "GO" else "⛔"
        return (f"| {arm} | {icon} {v['verdict']}（{v['n_steps']} 步）| "
                f"{fail_txt} | {warn_txt} |")

    def step_s(t):
        return _fmt(t.get("avg_step_s"), 6) if t else "TBD"

    def wall(t):
        return _fmt(t.get("wall_time_s"), 1) if t else "TBD"

    def tgt_frac(t):
        v = (t or {}).get("target_forward_fraction")
        return f"{v:.1%}" if isinstance(v, float) else "TBD"

    def ema_frac(t):
        v = (t or {}).get("ema_fraction")
        return f"{v:.1%}" if isinstance(v, float) else "TBD"

    cos_b = _last(mb, "recon_cosine")
    var_first_b = mb[0]["token_variance"] if mb else None
    var_last_b = _last(mb, "token_variance")
    gate_cos = (
        "✅ PASS" if isinstance(cos_b, float) and cos_b >= GATE_COSINE
        else ("❌ FAIL" if isinstance(cos_b, float) else "TBD")
    )
    gate_var = "TBD"
    if isinstance(var_first_b, float) and isinstance(var_last_b, float):
        gate_var = (
            "✅ PASS" if var_last_b >= 0.5 * var_first_b else "❌ FAIL"
        )
    # Data provenance: dry-run numbers prove plumbing only, never physics.
    modes = {t.get("mode") for t in (ta, tb) if t}
    provenance = (
        "⚠️ 当前数字来自 CPU dry-run 自检（随机初始化小模型 + 合成噪声，"
        "仅验证管线，不构成实验结论）"
        if modes == {"dry"} else
        "Atlas 正式跑数据" if modes == {"atlas"} else
        "⚠️ 混合/未知来源，请勿直接引用"
    )

    lines = [
        "# W22.7 A/B 对照实验报告 — 代理损失 vs 真 JEPA 目标",
        "",
        "**状态**：🟡 骨架已生成，数字待 Atlas 硬件窗口（带 TBD 处即未填）",
        "**依据**：设计文档 §9.2（`2026-08-03-w22-jepa-predictor-ema-design.md`）",
        f"**数据来源**：{provenance}",
        "",
        "- Arm A：`jepa.enabled: false`（自蒸馏代理损失基线）",
        "- Arm B：`jepa.enabled: true`（predictor + EMA target encoder，smooth-L1，cosine 0.996→1.0）",
        "- 数据：固定子集 manifest（两臂完全相同样本+mask 序列）；同步数/batch/grad_accum/lr",
        "",
        "## 0. Pilot 早停检查（正式跑前置门禁）",
        "",
        "| 臂 | 判定 | 失败项 | 警告项 |",
        "|----|------|--------|--------|",
        _pilot_row(va, "A"),
        _pilot_row(vb, "B"),
        "",
        "> 证据数值见 `bench_results/w22_7/{a,b}/pilot/verdict.json`；"
        "NO-GO 时 .sh 已中止正式跑。",
        "",
        "## 1. Probing 指标（最终 checkpoint step）",
        "",
        "| 指标 | Arm A | Arm B | 验收门槛 | 判定 |",
        "|------|-------|-------|----------|------|",
        f"| 掩码重建 cosine | {_fmt(_last(ma, 'recon_cosine'))}（≈随机水平）| {_fmt(cos_b)} | ≥ {GATE_COSINE} | {gate_cos} |",
        f"| token 方差（坍塌检查）| {_fmt(_last(ma, 'token_variance'), 6)} | {_fmt(var_last_b, 6)} | B 臂不衰减（≥首点的 50%）| {gate_var} |",
        f"| 有效秩 | {_fmt(_last(ma, 'effective_rank'), 2)} | {_fmt(_last(mb, 'effective_rank'), 2)} | B > A（参考）| TBD |",
        f"| 线性 probing 准确率 | {_fmt(_last(ma, 'linear_probe_acc'), 3)} | {_fmt(_last(mb, 'linear_probe_acc'), 3)} | B 显著优于 A（至少一项下游指标）| TBD |",
        f"| k-NN 一致性 | {_fmt(_last(ma, 'knn_hit_rate'), 3)} | {_fmt(_last(mb, 'knn_hit_rate'), 3)} | 参考 | TBD |",
        "",
        "## 2. 吞吐与开销（§12-②③ 补充实测项）",
        "",
        "| 项 | Arm A | Arm B |",
        "|----|-------|-------|",
        f"| 总 wall time (s) | {wall(ta)} | {wall(tb)} |",
        f"| 平均 optimizer step (s) | {step_s(ta)} | {step_s(tb)} |",
        f"| target 双前向占比 | — | {tgt_frac(tb)} |",
        f"| EMA update（含 FSDP gather）占比 | — | {ema_frac(tb)} |",
        f"| EMA gather 耗时 (s，FSDP 路径）| — | {_fmt((tb or {}).get('timing', {}).get('ema_gather_s'), 4) if tb else 'TBD'} |",
        "",
        "**决策点**：若 EMA gather 占比显著，切 `jepa.ema.update_every > 1`（§12-② 缓解杆）后重测本表。",
        "",
        "## 3. 曲线数据",
        "",
        "逐 checkpoint step 的指标历史见 `bench_results/w22_7/arm_{a,b}/metrics.jsonl`；",
        "loss 曲线见 `bench_results/w22_7/arm_{a,b}/throughput.json` 的 `loss_curve` 字段。",
        "",
        "## 4. 结论",
        "",
        "TBD — 待 Atlas 正式跑完成后填写（GO/NO-GO：B 臂是否满足全部验收门槛）。",
        "",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[W22.7] report written: {out}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=Path("bench_results/w22_7"))
    ap.add_argument("--out", type=Path, default=REPORT_PATH)
    args = ap.parse_args()
    build_report(args.results, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
