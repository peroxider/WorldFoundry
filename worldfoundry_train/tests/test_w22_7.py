"""W22.7 three-arm experiment prep tests (review §2, 2026-08-05).

Covers:
- C-arm YAML config (mask_ratio=0, jepa.enabled=true).
- ``scripts/w22_7_run_arm.py`` helpers (``_dry_cfg``, ``default_arm_config``,
  ``--arm`` choices).
- ``scripts/w22_7_build_report.py`` Δ-table helpers (paired delta series,
  bootstrap CI, variance ratio at step cap) and the NPU profile slot.
- ``worldfoundry_train.jepa_objective`` peak memory capture in
  ``after_step()`` (review §2.3: ``timing.ema_gather_peak_mb``).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

import scripts.w22_7_build_report as build_report_mod
import scripts.w22_7_run_arm as run_arm_mod
from worldfoundry_train.jepa_objective import (
    JEPAObjective,
    _PeakMemoryAPI,
    _peak_memory_api,
)
from worldfoundry_train.model.ema import EMAUpdater
from worldfoundry_train.model.predictor import JEPAPredictor


# ---------------------------------------------------------------------------
# C-arm YAML config (review §2.1)
# ---------------------------------------------------------------------------

CFG_DIR = Path("worldfoundry_train/configs")


def test_arm_c_yaml_mask_ratio_zero_with_jepa_enabled():
    """C-arm YAML: jepa.enabled=true AND data.mask_ratio=0.0.

    The whole point of arm C is "predictor sees zero masked tokens"; one of
    the two switches broken means the run degrades into arm B and the
    sanity check collapses.
    """
    cfg = yaml.safe_load((CFG_DIR / "w22_7_arm_c_mask0.yaml").read_text())
    assert cfg["jepa"]["enabled"] is True
    assert cfg["data"]["mask_ratio"] == 0.0


def test_arm_c_yaml_matches_arm_b_everywhere_but_mask_ratio():
    """C-arm differs from B-arm only in data.mask_ratio.

    Catches accidental field drift introduced by hand-editing the new YAML
    instead of regenerating from B.
    """
    a = yaml.safe_load((CFG_DIR / "w22_7_arm_b_jepa.yaml").read_text())
    c = yaml.safe_load((CFG_DIR / "w22_7_arm_c_mask0.yaml").read_text())
    # The single allowed diff lives in data.mask_ratio.
    assert a["data"]["mask_ratio"] != c["data"]["mask_ratio"]
    assert c["data"]["mask_ratio"] == 0.0
    # All other fields identical.
    a2 = {k: v for k, v in a.items() if k != "data"}
    c2 = {k: v for k, v in c.items() if k != "data"}
    assert a2 == c2
    a_d = {k: v for k, v in a["data"].items() if k != "mask_ratio"}
    c_d = {k: v for k, v in c["data"].items() if k != "mask_ratio"}
    assert a_d == c_d


# ---------------------------------------------------------------------------
# run_arm.py helpers (review §2.1 plumbing)
# ---------------------------------------------------------------------------

def test_default_arm_config_suffix_table():
    """arm -> config-path resolver covers all three arms including c.

    C's filename suffix ``mask0`` differs from B's ``jepa``; an inline
    ternary that handles ``a`` vs not-a silently sends ``c`` to a
    nonexistent ``arm_c_jepa.yaml`` and the run dies on FileNotFoundError.
    """
    assert run_arm_mod.default_arm_config("a") == Path(
        "worldfoundry_train/configs/w22_7_arm_a_proxy.yaml"
    )
    assert run_arm_mod.default_arm_config("b") == Path(
        "worldfoundry_train/configs/w22_7_arm_b_jepa.yaml"
    )
    assert run_arm_mod.default_arm_config("c") == Path(
        "worldfoundry_train/configs/w22_7_arm_c_mask0.yaml"
    )


def test_default_arm_config_unknown_arm_raises():
    """argparse already guards choices; the function itself must too."""
    with pytest.raises(ValueError, match="unknown arm"):
        run_arm_mod.default_arm_config("z")


def test_dry_cfg_c_uses_zero_mask_ratio_and_jepa_enabled():
    """``_dry_cfg('c')`` mirrors the YAML config in CPU dry mode."""
    cfg = run_arm_mod._dry_cfg("c")
    assert cfg["data"]["mask_ratio"] == 0.0
    assert cfg["jepa"]["enabled"] is True
    # Predictor + EMA + loss sub-dicts present so the trainer build path
    # finds them; without these the trainer throws on missing fields.
    assert "predictor" in cfg["jepa"]
    assert "ema" in cfg["jepa"]
    assert "loss" in cfg["jepa"]


def test_dry_cfg_a_and_b_unchanged():
    """Smoke: arms A and B keep their pre-review shape (regression guard)."""
    a = run_arm_mod._dry_cfg("a")
    b = run_arm_mod._dry_cfg("b")
    assert a["jepa"]["enabled"] is False
    assert b["jepa"]["enabled"] is True
    assert b["data"]["mask_ratio"] == 0.5  # unchanged from before


# ---------------------------------------------------------------------------
# build_report.py math helpers
# ---------------------------------------------------------------------------

def test_paired_delta_series_pairs_by_step():
    a = [{"step": 1, "linear_probe_acc": 0.50},
         {"step": 2, "linear_probe_acc": 0.55}]
    b = [{"step": 1, "linear_probe_acc": 0.60},
         {"step": 3, "linear_probe_acc": 0.65}]  # step 3 missing in A
    out = build_report_mod._paired_delta_series(a, b, "linear_probe_acc")
    assert len(out) == 1 and out[0][0] == 1
    assert out[0][1] == pytest.approx(0.10)  # 0.60 - 0.50 (float)
    assert build_report_mod._paired_delta_series([], b, "x") == []
    assert build_report_mod._paired_delta_series(a, [], "x") == []


def test_bootstrap_ci_positive_mean_low_above_zero():
    """All-positive deltas -> CI low > 0 (passes the linear-probe gate)."""
    rng = np.random.default_rng(0)
    deltas = list(rng.uniform(0.01, 0.05, size=8))
    mean, lo, hi = build_report_mod._bootstrap_ci(deltas)
    assert mean > 0.0
    assert lo > 0.0
    assert hi > mean  # CI widens around the mean


def test_bootstrap_ci_negative_mean_high_below_zero():
    """Mostly-negative deltas -> CI high < 0."""
    deltas = [-0.02, -0.05, -0.01, -0.03, -0.04, -0.02, -0.01, -0.03]
    mean, lo, hi = build_report_mod._bootstrap_ci(deltas)
    assert mean < 0.0
    assert hi < 0.0


def test_bootstrap_ci_empty_returns_none():
    assert build_report_mod._bootstrap_ci([]) is None


def test_variance_ratio_at_step_picks_largest_common_le_cap():
    a = [{"step": 100, "token_variance": 1.0},
         {"step": 200, "token_variance": 1.0},
         {"step": 500, "token_variance": 1.0}]
    b = [{"step": 100, "token_variance": 0.90},
         {"step": 200, "token_variance": 0.88},
         {"step": 500, "token_variance": 0.50}]
    s, val_a, ratio = build_report_mod._variance_ratio_at_step(a, b)
    assert s == 200  # largest ≤ 200 cap
    assert val_a == 1.0
    assert ratio == pytest.approx(0.88)


def test_variance_ratio_returns_none_when_no_step_under_cap():
    a = [{"step": 500, "token_variance": 1.0}]
    b = [{"step": 500, "token_variance": 0.5}]
    assert build_report_mod._variance_ratio_at_step(a, b, step_cap=200) is None


# ---------------------------------------------------------------------------
# build_report.py markdown output (review §2.2 + §2.3)
# ---------------------------------------------------------------------------

def _fake_metrics(steps, *, recon, probe, var, knn, erank):
    out = []
    for s in steps:
        out.append({"step": s, "arm": "b",
                    "token_variance": var, "effective_rank": erank,
                    "linear_probe_acc": probe, "knn_hit_rate": knn,
                    "recon_cosine": recon})
    return out


def _write_fake_arm_artifacts(
    results: Path, arm: str, *, recon=0.4, probe=0.55, var=0.95,
    knn=0.50, erank=60.0,
):
    """Write a minimal metrics.jsonl + throughput.json for one arm."""
    arm_dir = results / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    metrics = _fake_metrics(
        [50, 100, 150, 200, 250], recon=recon, probe=probe, var=var,
        knn=knn, erank=erank,
    )
    (arm_dir / "metrics.jsonl").write_text(
        "\n".join(json.dumps(r) for r in metrics) + "\n"
    )
    (arm_dir / "throughput.json").write_text(json.dumps({
        "label": f"fake {arm}", "arm": arm, "mode": "dry",
        "jepa_enabled": arm in ("b", "c"),
        "optimizer_steps": 5, "wall_time_s": 12.0, "avg_step_s": 0.5,
        "loss_curve": [1.0, 0.8, 0.6, 0.5, 0.4],
        "timing": {
            "loss_fn_s": 0.4, "context_forward_s": 0.1,
            "target_forward_s": 0.1, "ema_update_s": 0.05,
            "ema_gather_s": 0.02, "ema_updates": 5,
            "ema_gather_peak_mb": 320.0,
        },
        "target_forward_fraction": 0.25,
        "ema_fraction": 0.10,
    }))


def test_build_report_writes_c_arm_rows_and_delta_table(tmp_path):
    """Three arms -> report contains C-arm row + Δ-table + NPU profile."""
    _write_fake_arm_artifacts(
        tmp_path, "a", recon=0.05, probe=0.50, var=1.00, knn=0.50, erank=40.0,
    )
    _write_fake_arm_artifacts(
        tmp_path, "b", recon=0.40, probe=0.65, var=0.90, knn=0.80, erank=60.0,
    )
    _write_fake_arm_artifacts(
        tmp_path, "c", recon=0.10, probe=0.55, var=0.95, knn=0.55, erank=50.0,
    )
    out = build_report_mod.build_report(tmp_path, tmp_path / "report.md")
    text = out.read_text()
    # C-arm headline + delta table
    assert "Arm C" in text
    assert "Δ(B−A)" in text and "Δ(B−C)" in text
    # Gates are present and the deltas land correctly (B−A cosine = 0.35).
    assert "B−A ≥ 0.2" in text
    assert "B−C ≥ 0.15" in text
    # NPU profile section is mandatory per §2.3.
    assert "NPU profile" in text
    assert "ema_gather_peak_mb" in text
    assert "320 MB" in text  # from fake timing dict
    # Bootstrap CI lands in the probe row.
    assert "bootstrap 95% CI" in text


def test_build_report_handles_missing_c_gracefully(tmp_path):
    """Old results dir without C-arm -> TBD everywhere, no crash."""
    _write_fake_arm_artifacts(tmp_path, "a", recon=0.05)
    _write_fake_arm_artifacts(tmp_path, "b", recon=0.40)
    out = build_report_mod.build_report(tmp_path, tmp_path / "report.md")
    text = out.read_text()
    assert "TBD" in text
    # C-arm row present but with TBD values from empty metrics list.
    assert "Arm C | TBD" not in text  # we show recon_cosine field as TBD
    assert "| TBD " in text  # plenty of TBD cells


# ---------------------------------------------------------------------------
# JEPAObjective peak-memory capture (review §2.3)
# ---------------------------------------------------------------------------

def _toy_jeapa(seed=0, with_timing=True):
    """Minimal JEPAObjective with two tiny linear params so EMA works."""
    torch.manual_seed(seed)
    online = torch.nn.Linear(2, 2)
    online.weight.requires_grad_(True)
    online.bias.requires_grad_(True)
    target = torch.nn.Linear(2, 2)
    target.weight.requires_grad_(False)
    target.bias.requires_grad_(False)
    with torch.no_grad():
        target.weight.copy_(online.weight)
        target.bias.copy_(online.bias)
    pred = JEPAPredictor(
        num_patches=2, grid_size=(1, 1, 2), encoder_dim=2,
        predictor_dim=2, depth=1, num_heads=1,
    )
    ema = EMAUpdater(
        list(online.parameters()), list(target.parameters()),
        momentum_start=0.9, momentum_end=0.99,
        schedule="cosine", schedule_steps=10, names=["w", "b"],
    )
    obj = JEPAObjective(
        pred, target, ema, loss_type="smooth_l1", beta=1.0,
    )
    if not with_timing:
        obj.timing = None
    return obj


def test_after_step_with_timing_captures_peak_memory(monkeypatch):
    """When timing is on and a peak-memory API exists, after_step writes the key.

    Use a fake PeakMemoryAPI to avoid relying on CUDA/NPU availability.
    """
    obj = _toy_jeapa(seed=1, with_timing=True)
    assert obj.timing is not None

    peak_log = []
    reset_calls = []
    fake = _PeakMemoryAPI(name="fake", reset=lambda: reset_calls.append(1),
                          peak=lambda: peak_log.append(1) or 1024 * 1024 * 100)
    monkeypatch.setattr(
        "worldfoundry_train.jepa_objective._peak_memory_api",
        lambda: fake,
    )
    obj.after_step()
    assert reset_calls, "peak reset was not called"
    assert peak_log, "peak read was not called"
    assert "ema_gather_peak_mb" in obj.timing
    assert obj.timing["ema_gather_peak_mb"] == pytest.approx(100.0, abs=1e-3)


def test_after_step_without_timing_is_noop(monkeypatch):
    obj = _toy_jeapa(seed=2, with_timing=False)

    # Patch _peak_memory_api to raise — if it were ever called it would
    # propagate here and fail the test. Succeed = wrapper never invoked.
    def _boom():
        raise RuntimeError("should not be called when timing is None")

    monkeypatch.setattr(
        "worldfoundry_train.jepa_objective._peak_memory_api", _boom,
    )
    obj.after_step()  # must not raise


def test_peak_memory_api_returns_none_when_no_device(monkeypatch):
    """On a pure-CPU box the helper must return None, not raise."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    # NPU may or may not be importable; force the import to fail too.
    monkeypatch.setattr(
        "worldfoundry_train.jepa_objective.torch.npu",
        None, raising=False,
    )
    assert _peak_memory_api() is None
