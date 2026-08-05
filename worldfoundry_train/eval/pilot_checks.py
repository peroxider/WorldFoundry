"""Pilot-mode early-stop checks for the W22.7 experiment driver (design §9.2
+ §12-②③ measurement items).

Pure functions over per-step series so the verdict logic is unit-testable
without any training run. The driver (``scripts/w22_7_run_arm.py --mode
pilot``) collects the series, calls :func:`evaluate_pilot`, and writes the
returned dict to ``verdict.json``.

Verdict semantics:
  - ``GO``     : no NO-GO check fired (warnings allowed).
  - ``NO-GO``  : at least one hard check fired; the .sh aborts the formal run.

Hard gates (NO-GO):
  - numeric health: any NaN/Inf loss;
  - memory leak: last-window peak-memory growth beyond tolerance AND a
    positive last-window slope (skipped when no memory series, e.g. CPU);
  - EMA stuck (arm B): fp32 EMA master unchanged after training.

Soft warnings (verdict stays GO):
  - throughput degradation: last-window median step time > 1.5x first window;
  - loss not decreasing (finite but flat/rising).
"""
from __future__ import annotations

import math
from typing import Any

__all__ = ["evaluate_pilot", "GO", "NO_GO"]

GO = "GO"
NO_GO = "NO-GO"

# Tolerances (documented in verdict.json evidence so Atlas reruns can audit).
MEM_REL_GROWTH_TOL = 0.10   # last-50 median > first-50 median * 1.10 -> suspect
MEM_SLOPE_REL_TOL = 1e-3    # last-50 slope > 0.1% of mean per step -> leaking
THROUGHPUT_DEGRADE_FACTOR = 1.5  # last-50 median > 1.5x first-50 -> warning


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def _slope(xs: list[float]) -> float:
    """Least-squares slope (units per step) of a 1-D series."""
    n = len(xs)
    if n < 2:
        return 0.0
    xm = (n - 1) / 2.0
    ym = sum(xs) / n
    num = sum((i - xm) * (y - ym) for i, y in enumerate(xs))
    den = sum((i - xm) ** 2 for i in range(n))
    return num / den if den else 0.0


def _window(series: list[float], window: int) -> tuple[list[float], list[float]]:
    """(first_window, last_window); overlaps allowed when series is short."""
    w = max(1, min(window, len(series) // 2 or 1))
    return series[:w], series[-w:]


def evaluate_pilot(
    *,
    arm: str,
    loss_series: list[float],
    step_time_series: list[float],
    memory_series_mb: list[float] | None = None,
    memory_source: str = "none",
    timing: dict[str, float] | None = None,
    ema_moved: bool | None = None,
    window: int = 50,
) -> dict[str, Any]:
    """Evaluate pilot early-stop checks. Returns the verdict dict.

    Args:
        arm: "a" | "b".
        loss_series: per-optimizer-step losses.
        step_time_series: per-optimizer-step wall seconds.
        memory_series_mb: per-step peak memory in MB, or None when the
            platform has no gauge (memory check is then "skipped").
        memory_source: "npu_peak" | "cpu_rss_peak" | "none" (evidence field).
        timing: WORLDFORGE_JEPA_TIMING component dict (arm B).
        ema_moved: arm B only — did the fp32 EMA master change during the run?
        window: tail/head window size for trend checks.
    """
    checks: list[dict[str, Any]] = []

    def add(name: str, status: str, evidence: dict, action: str = "") -> None:
        checks.append({
            "name": name, "status": status,
            "evidence": evidence, "action": action,
        })

    # --- numeric health (hard) ---
    finite = all(math.isfinite(v) for v in loss_series)
    add(
        "numeric_health",
        "pass" if finite else "fail",
        {"finite": finite, "n_steps": len(loss_series),
         "first_loss": loss_series[0] if loss_series else None,
         "last_loss": loss_series[-1] if loss_series else None},
        "" if finite else "loss 出现 NaN/Inf；检查 lr/混合精度/mask 全空批次后再试",
    )
    if finite and len(loss_series) >= 2:
        first_w, last_w = _window(loss_series, window)
        decreasing = _median(last_w) < _median(first_w)
        add(
            "loss_decreasing",
            "pass" if decreasing else "warn",
            {"first_window_median": _median(first_w),
             "last_window_median": _median(last_w)},
            "" if decreasing else "pilot 短跑 loss 未下降；正式跑前确认 lr/warmup，非阻塞",
        )

    # --- memory leak (hard; skipped without a gauge) ---
    if memory_series_mb is None or len(memory_series_mb) < 4:
        add("memory_growth", "skipped",
            {"reason": "no memory gauge on this platform",
             "memory_source": memory_source})
    else:
        first_w, last_w = _window(memory_series_mb, window)
        m_first, m_last = _median(first_w), _median(last_w)
        rel_growth = (m_last - m_first) / max(m_first, 1e-9)
        slope = _slope(last_w)
        rel_slope = slope / max(m_last, 1e-9)
        leaking = rel_growth > MEM_REL_GROWTH_TOL and rel_slope > MEM_SLOPE_REL_TOL
        add(
            "memory_growth",
            "fail" if leaking else "pass",
            {"memory_source": memory_source,
             "first_window_median_mb": round(m_first, 2),
             "last_window_median_mb": round(m_last, 2),
             "relative_growth": round(rel_growth, 4),
             "last_window_slope_mb_per_step": round(slope, 4),
             "tolerances": {"rel_growth": MEM_REL_GROWTH_TOL,
                            "rel_slope_per_step": MEM_SLOPE_REL_TOL}},
            "显存持续增长（泄漏/OOM 前兆）：检查 target encoder 副本、EMA gather "
            "瞬态分配与激活缓存；不要进入正式跑" if leaking else "",
        )

    # --- throughput (soft) + component breakdown ---
    if step_time_series:
        first_w, last_w = _window(step_time_series, window)
        t_first, t_last = _median(first_w), _median(last_w)
        cv = (
            (lambda sd: sd / t_last if t_last else 0.0)(
                (sum((v - t_last) ** 2 for v in last_w) / len(last_w)) ** 0.5
            )
        )
        degraded = t_last > THROUGHPUT_DEGRADE_FACTOR * t_first
        evidence: dict[str, Any] = {
            "first_window_median_s": round(t_first, 6),
            "last_window_median_s": round(t_last, 6),
            "last_window_cv": round(cv, 4),
            "degrade_factor": THROUGHPUT_DEGRADE_FACTOR,
        }
        if timing:
            total = sum(step_time_series) or 1.0
            evidence["target_forward_fraction"] = round(
                timing.get("target_forward_s", 0.0) / total, 4)
            evidence["ema_update_fraction"] = round(
                timing.get("ema_update_s", 0.0) / total, 4)
            evidence["ema_gather_s"] = round(timing.get("ema_gather_s", 0.0), 4)
        add(
            "throughput_stability",
            "warn" if degraded else "pass",
            evidence,
            "吞吐持续恶化：profile target 双前向与 EMA gather；考虑 "
            "ema_update_every>1 或 target 降帧" if degraded else "",
        )

    # --- EMA movement (arm B, hard) ---
    if arm == "b":
        if ema_moved is None:
            add("ema_movement", "skipped", {"reason": "no EMA snapshot taken"})
        else:
            add(
                "ema_movement",
                "pass" if ema_moved else "fail",
                {"ema_moved": ema_moved},
                "" if ema_moved else "EMA master 未移动：检查 post_step_hook "
                "接线（应每 optimizer step 一次）与 update_every 配置",
            )

    hard_fail = any(c["status"] == "fail" for c in checks)
    verdict = {
        "arm": arm,
        "verdict": NO_GO if hard_fail else GO,
        "n_steps": len(loss_series),
        "checks": checks,
        "summary": (
            "存在硬门槛失败，中止正式跑并人工介入" if hard_fail
            else "全部硬门槛通过；警告项见 checks"
        ),
    }
    return verdict
