"""Aggregate W22.7 arm artifacts into the experiment report.

Reads bench_results/w22_7/arm_{a,b,c}/{metrics.jsonl,throughput.json} and
writes a markdown report to docs/superpowers/specs/. Any missing artifact
becomes a ``TBD`` placeholder — the report skeleton is valid before the
Atlas window produces numbers (review §2.2 acceptance gates included).

Usage:
    python scripts/w22_7_build_report.py [--results bench_results/w22_7]

Acceptance gates (review §2.2, 2026-08-05; replaces design §9.2's
absolute-value gates that were too sensitive to init / scale):

    masked_reconstruction_cosine: B − A ≥ 0.20  AND  B − C ≥ 0.15
    linear_probe_accuracy:        B − A 95% CI low > 0  (bootstrap)
    token_variance:               B/A ≥ 0.85 at the largest common step ≤ 200
    knn_hit_rate:                 B − A ≥ 0.10

NPU profile (review §2.3, mandatory in throughput.json):

    ema_gather_peak_mb:           reported; > 4096 → recommend
                                  ``jepa.ema.update_every`` ≥ 2
    ema_gather_s_avg:             timing["ema_gather_s"] / timing["ema_updates"]
    target_forward_fraction:      timing["target_forward_s"] /
                                  timing["loss_fn_s"]; budget < 0.4
    context_forward_s:            timing["context_forward_s"] (baseline)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from worldfoundry_train.eval.jepa_probing import read_jsonl

REPORT_PATH = Path(
    "docs/superpowers/specs/2026-08-04-w22-7-ab-experiment-report.md"
)

# Delta gates (review §2.2). Centralized so tests and report agree on the
# numbers; bumping any threshold here updates the report's delta gate table.
GATE_COS_DELTA_AB = 0.20
GATE_COS_DELTA_BC = 0.15
GATE_KNN_DELTA_AB = 0.10
GATE_VAR_RATIO_BA = 0.85
GATE_VAR_STEP_CAP = 200
NPU_GATHER_BUDGET_MB = 4096
NPU_GATHER_LATENCY_MS = 50.0
NPU_TARGET_FORWARD_FRAC = 0.40
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20260805  # reproducible CI; matches addendum spec date


# ---------------------------------------------------------------------------
# IO + per-arm probing helpers
# ---------------------------------------------------------------------------

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


def _paired_delta_series(
    m_a: list[dict], m_b: list[dict], key: str
) -> list[tuple[int, float]]:
    """Pair arm A and arm B metric values by step and return deltas (B − A).

    Returns a list of ``(step, delta)`` for every step present in both
    arms' metrics.jsonl that exposes ``key``. Empty list if no overlap
    (then the bootstrap CI is undefined and the gate stays TBD).
    """
    if not m_a or not m_b:
        return []
    sa = {r["step"]: r[key] for r in m_a if key in r and "step" in r}
    sb = {r["step"]: r[key] for r in m_b if key in r and "step" in r}
    return [(s, sb[s] - sa[s]) for s in sorted(set(sa) & set(sb))]


def _bootstrap_ci(
    deltas: list[float], n_boot: int = BOOTSTRAP_RESAMPLES, alpha: float = 0.05,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, float] | None:
    """Mean and (1-α) CI on ``mean(deltas)`` via percentile bootstrap.

    Returns ``(mean, ci_low, ci_high)`` or ``None`` if deltas is empty.
    Bootstrap is the standard non-parametric CI for small paired samples
    (review §2.2 gate: pass iff ``ci_low > 0``).
    """
    arr = np.asarray(deltas, dtype=np.float64)
    if arr.size == 0:
        return None
    rng = np.random.default_rng(seed)
    n = arr.size
    idx = rng.integers(0, n, size=(n_boot, n))
    means = arr[idx].mean(axis=1)
    return (
        float(arr.mean()),
        float(np.quantile(means, alpha / 2)),
        float(np.quantile(means, 1 - alpha / 2)),
    )


def _variance_ratio_at_step(
    m_a: list[dict], m_b: list[dict], step_cap: int = GATE_VAR_STEP_CAP,
    key: str = "token_variance",
) -> tuple[int, float, float] | None:
    """B/A token_variance ratio at the largest common step ≤ ``step_cap``.

    Review §2.2 intent: B shouldn't collapse relative to A. Returns the
    step, the A value, and the ratio, or None if no step qualifies.
    """
    sa = {r["step"]: r[key] for r in m_a if key in r and "step" in r}
    sb = {r["step"]: r[key] for r in m_b if key in r and "step" in r}
    common = sorted(set(sa) & set(sb))
    candidate = [s for s in common if s <= step_cap]
    if not candidate:
        return None
    s = candidate[-1]
    a = sa[s]
    b = sb[s]
    if a == 0:
        return None
    return s, a, b / a


# ---------------------------------------------------------------------------
# Mark-down report builder
# ---------------------------------------------------------------------------

def build_report(results: Path, out: Path = REPORT_PATH) -> Path:
    ma, ta = _load_arm(results, "a")
    mb, tb = _load_arm(results, "b")
    mc, tc = _load_arm(results, "c")
    va = _load_pilot_verdict(results, "a")
    vb = _load_pilot_verdict(results, "b")
    vc = _load_pilot_verdict(results, "c")

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

    def ema_gather_avg_ms(t):
        timing = (t or {}).get("timing") or {}
        gather = timing.get("ema_gather_s")
        updates = timing.get("ema_updates")
        if isinstance(gather, (int, float)) and isinstance(updates, (int, float)) and updates > 0:
            return f"{gather * 1000 / updates:.1f} ms（{int(updates)} 次）"
        return "TBD"

    def ema_gather_peak_mb(t):
        v = ((t or {}).get("timing") or {}).get("ema_gather_peak_mb")
        if isinstance(v, (int, float)):
            over = " ⚠️ 超过 4GB 预算" if v > NPU_GATHER_BUDGET_MB else ""
            return f"{v:.0f} MB{over}"
        return "TBD"

    # --- delta gates (review §2.2) -------------------------------------------------
    cos_a = _last(ma, "recon_cosine")
    cos_b = _last(mb, "recon_cosine")
    cos_c = _last(mc, "recon_cosine")
    delta_ab_ok = (
        isinstance(cos_a, float) and isinstance(cos_b, float)
        and cos_b - cos_a >= GATE_COS_DELTA_AB
    )
    delta_bc_ok = (
        isinstance(cos_b, float) and isinstance(cos_c, float)
        and cos_b - cos_c >= GATE_COS_DELTA_BC
    )

    probe_deltas = [d for _s, d in _paired_delta_series(ma, mb, "linear_probe_acc")]
    probe_ci = _bootstrap_ci(probe_deltas)
    probe_pass = (
        probe_ci is not None and probe_ci[1] > 0.0
    )

    var_ratio = _variance_ratio_at_step(ma, mb)
    var_pass = var_ratio is not None and var_ratio[2] >= GATE_VAR_RATIO_BA

    knn_deltas = [d for _s, d in _paired_delta_series(ma, mb, "knn_hit_rate")]
    knn_mean = float(np.mean(knn_deltas)) if knn_deltas else None
    knn_pass = knn_mean is not None and knn_mean >= GATE_KNN_DELTA_AB

    def verdict(b: bool | None) -> str:
        return "✅ PASS" if b else ("❌ FAIL" if b is False else "TBD")

    cos_ab_icon = verdict(delta_ab_ok)
    cos_bc_icon = verdict(delta_bc_ok)
    probe_icon = verdict(probe_pass)
    var_icon = verdict(var_pass)
    knn_icon = verdict(knn_pass)

    var_step_str = f"step {var_ratio[0]}" if var_ratio else "TBD"
    var_a_str = _fmt(var_ratio[1], 4) if var_ratio else "TBD"
    var_ratio_str = _fmt(var_ratio[2], 4) if var_ratio else "TBD"

    cos_ab_str = (
        _fmt(cos_b - cos_a, 4) if isinstance(cos_a, float) and isinstance(cos_b, float)
        else "TBD"
    )
    cos_bc_str = (
        _fmt(cos_b - cos_c, 4) if isinstance(cos_b, float) and isinstance(cos_c, float)
        else "TBD"
    )

    probe_mean_str = _fmt(probe_ci[0], 4) if probe_ci else "TBD"
    probe_lo_str = _fmt(probe_ci[1], 4) if probe_ci else "TBD"
    probe_hi_str = _fmt(probe_ci[2], 4) if probe_ci else "TBD"

    knn_str = _fmt(knn_mean, 4) if knn_mean is not None else "TBD"

    # --- NPU profile actions (review §2.3) -----------------------------------------
    ema_gather_ms = (tb or {}).get("timing", {}).get("ema_gather_s") if tb else None
    ema_updates = (tb or {}).get("timing", {}).get("ema_updates") if tb else None
    gather_avg_ms = (
        ema_gather_ms * 1000 / ema_updates
        if isinstance(ema_gather_ms, (int, float)) and isinstance(ema_updates, (int, float))
        and ema_updates > 0 else None
    )
    ema_gather_action = ""
    if isinstance(gather_avg_ms, (int, float)) and gather_avg_ms > NPU_GATHER_LATENCY_MS:
        ema_gather_action = (
            "⚠️ 超过 50ms 预算：建议 `jepa.ema.update_every` ≥ 2 后重测本表。"
        )
    tgt_frac_val = (tb or {}).get("target_forward_fraction") if tb else None
    if isinstance(tgt_frac_val, float) and tgt_frac_val >= NPU_TARGET_FORWARD_FRAC:
        tgt_frac_action = (
            "⚠️ ≥ 40% 预算：考虑 target encoder 主干共享引用（v2 优化，§7 可延后项）。"
        )
    else:
        tgt_frac_action = ""

    # --- Data provenance ---
    modes = {t.get("mode") for t in (ta, tb, tc) if t}
    if modes == {"dry"}:
        provenance = (
            "⚠️ 当前数字来自 CPU dry-run 自检（随机初始化小模型 + 合成噪声，"
            "仅验证管线，不构成实验结论）"
        )
    elif modes == {"atlas"}:
        provenance = "Atlas 正式跑数据"
    else:
        provenance = "⚠️ 混合/未知来源，请勿直接引用"

    lines = [
        "# W22.7 A/B/C 对照实验报告 — 代理损失 vs 真 JEPA vs 零遮挡健全性",
        "",
        "**状态**：🟡 骨架已生成，数字待 Atlas 硬件窗口（带 TBD 处即未填）",
        "**依据**：设计文档 §9.2（`2026-08-03-w22-jepa-predictor-ema-design.md`）",
        "**修订**：2026-08-05 addendum（`2026-08-05-w22-design-review-addendum.md`）— 验收门槛改为差值，新增 NPU profile 必填项",
        f"**数据来源**：{provenance}",
        "",
        "- Arm A：`jepa.enabled: false`（自蒸馏代理损失基线）",
        "- Arm B：`jepa.enabled: true`，`data.mask_ratio: 0.75`（predictor + EMA target encoder，smooth-L1，cosine 0.996→1.0）",
        "- Arm C：`jepa.enabled: true`，`data.mask_ratio: 0.0`（零遮挡健全性检查 — 证明 mask 信号被学到）",
        "- 数据：固定子集 manifest（三臂完全相同样本+mask 序列）；同步数/batch/grad_accum/lr",
        "",
        "## 0. Pilot 早停检查（正式跑前置门禁）",
        "",
        "| 臂 | 判定 | 失败项 | 警告项 |",
        "|----|------|--------|--------|",
        _pilot_row(va, "A"),
        _pilot_row(vb, "B"),
        _pilot_row(vc, "C"),
        "",
        "> 证据数值见 `bench_results/w22_7/{a,b,c}/pilot/verdict.json`；"
        "NO-GO 时 .sh 已中止正式跑。",
        "",
        "## 1. Probing 指标 + 修订后的 Δ-table 验收（review §2.2）",
        "",
        "| 指标 | Arm A | Arm B | Arm C | Δ(B−A) | Δ(B−C) | 验收门槛（差值） | 判定 |",
        "|------|-------|-------|-------|--------|--------|------------------|------|",
        f"| 掩码重建 cosine | {_fmt(cos_a)}（≈随机水平）| {_fmt(cos_b)} | {_fmt(cos_c)} | {cos_ab_str} | {cos_bc_str} | B−A ≥ {GATE_COS_DELTA_AB}；B−C ≥ {GATE_COS_DELTA_BC} | {cos_ab_icon} / {cos_bc_icon} |",
        f"| token 方差 ratio ({var_step_str}) | — / — | ratio={var_ratio_str} (A={var_a_str}) | — | — | — | B/A ≥ {GATE_VAR_RATIO_BA}（step ≤ {GATE_VAR_STEP_CAP}）| {var_icon} |",
        f"| 有效秩 | {_fmt(_last(ma, 'effective_rank'), 2)} | {_fmt(_last(mb, 'effective_rank'), 2)} | {_fmt(_last(mc, 'effective_rank'), 2)} | — | — | 参考 | TBD |",
        f"| 线性 probing（Δ bootstrap 95% CI）| {_fmt(_last(ma, 'linear_probe_acc'), 3)} | {_fmt(_last(mb, 'linear_probe_acc'), 3)} | — | mean={probe_mean_str} ci=[{probe_lo_str}, {probe_hi_str}] | — | ci_low > 0 | {probe_icon} |",
        f"| k-NN 一致性 | {_fmt(_last(ma, 'knn_hit_rate'), 3)} | {_fmt(_last(mb, 'knn_hit_rate'), 3)} | — | mean={knn_str} | — | B−A mean ≥ {GATE_KNN_DELTA_AB} | {knn_icon} |",
        "",
        "> Bootstrap CI 由 `_paired_delta_series + _bootstrap_ci`（n_boot=2000, seed=20260805）算出；空集时阈值回到 TBD。",
        "",
        "## 2. 吞吐与开销",
        "",
        "| 项 | Arm A | Arm B | Arm C |",
        "|----|-------|-------|-------|",
        f"| 总 wall time (s) | {wall(ta)} | {wall(tb)} | {wall(tc)} |",
        f"| 平均 optimizer step (s) | {step_s(ta)} | {step_s(tb)} | {step_s(tc)} |",
        f"| target 双前向占比 | — | {tgt_frac(tb)} | — |",
        f"| EMA update（含 FSDP gather）占比 | — | {ema_frac(tb)} | — |",
        f"| EMA gather 耗时 (s，FSDP 路径）| — | {_fmt((tb or {}).get('timing', {}).get('ema_gather_s'), 4) if tb else 'TBD'} | — |",
        "",
        "**决策点**：若 EMA gather 占比显著，切 `jepa.ema.update_every > 1`（§12-② 缓解杆）后重测本表。",
        "",
        "## 3. NPU profile 报告（review §2.3，必填）",
        "",
        "| 字段 | 测量方式 | Arm B 当前值 | 门槛 / 动作 |",
        "|------|----------|--------------|-------------|",
        f"| `ema_gather_peak_mb` | `torch.npu.max_memory_allocated()` EMA 期间峰值 | {ema_gather_peak_mb(tb)} | {NPU_GATHER_BUDGET_MB} MB 预算；超阈 → 推荐 `update_every` ≥ 2 |",
        f"| `ema_gather_s_avg` | `timing.ema_gather_s / timing.ema_updates` | {ema_gather_avg_ms(tb)} | < {NPU_GATHER_LATENCY_MS:.0f} ms（8-card HCCL 设计估）|",
        f"| `target_forward_fraction` | `target_forward_s / loss_fn_s` | {tgt_frac(tb)} | < {NPU_TARGET_FORWARD_FRAC:.2f} |",
        f"| `context_forward_s` | 同上 | {_fmt((tb or {}).get('timing', {}).get('context_forward_s'), 4) if tb else 'TBD'} | 基线对照 |",
        "",
        f"**决策提示**：{ema_gather_action or tgt_frac_action or '无动作 — 当前所有 NPU profile 项在预算内。'}",
        "",
        "## 4. warm-up window 分析（review §3 M4 强制项）",
        "",
        "实验报告中必须包含 step 0–100 的 loss 序列，以及 step 200/500/1000/2000 的对照表",
        "（详见 throughput.json 的 `loss_curve` 与 metrics.jsonl 的 `step`）。",
        "B 臂 step 0–100 期间 predictor 末层 MLP zero-init（`predictor.py:101-103`），起始 loss ≈ 0 不是收敛，",
        "结论需基于 step ≥ 200 的曲线对比。",
        "",
        "## 5. 曲线数据",
        "",
        "逐 checkpoint step 的指标历史见 `bench_results/w22_7/arm_{a,b,c}/metrics.jsonl`；",
        "loss 曲线见 `bench_results/w22_7/arm_{a,b,c}/throughput.json` 的 `loss_curve` 字段。",
        "",
        "## 6. 结论",
        "",
        "TBD — 待 Atlas 正式跑完成后填写。所有 Δ-table 验收 + NPU profile 项满足后才能标记 GO。",
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
