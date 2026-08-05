"""Run one arm of the W22.7 A/B experiment and compute probing metrics.

Modes:
  --mode dry    CPU self-check: tiny random-init VJEPA2 + synthetic fixed
                batches, ~12 optimizer steps. Proves the skeleton end-to-end:
                training runs, checkpoints land, four probing metrics compute,
                throughput JSON + metrics JSONL are written. NOT a physics
                result.
  --mode atlas  Real run from the arm YAML config (ViT-L 1.2B + LoRA +
                subset manifest). Single-process entry; the 8-card hccl/FSDP
                launch is wrapped by scripts/w22_7_ab_experiment.sh via
                torchrun (NPU device migration is W20 — see the .sh TODO).

Artifacts (under --out, default bench_results/w22_7):
  <arm>/ckpt/step_*.pt      trainer checkpoints (save_every_n)
  <arm>/metrics.jsonl       one record per checkpoint step, four metrics
  <arm>/throughput.json     wall time, avg step time, component breakdown
                            (bench_results/*.json style)

Usage:
    python scripts/w22_7_run_arm.py --arm a --mode dry
    python scripts/w22_7_run_arm.py --arm b --mode dry
    python scripts/w22_7_run_arm.py --arm b --mode pilot --backend dry
    python scripts/w22_7_run_arm.py --arm b --mode pilot --backend atlas
    python scripts/w22_7_run_arm.py --arm b --mode atlas \
        --config worldfoundry_train/configs/w22_7_arm_b_jepa.yaml --steps 2000

Pilot mode (--mode pilot): short run (~200 steps, env W22_7_PILOT_STEPS)
with per-step memory/throughput instrumentation and an early-stop verdict
(GO/NO-GO -> bench_results/w22_7/<arm>/pilot/verdict.json, exit code 3 on
NO-GO). Artifacts are isolated under <arm>/pilot/.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

# Throughput instrumentation (default OFF everywhere else; the experiment
# driver opts in). Must be set before JEPAObjective is constructed.
os.environ["WORLDFORGE_JEPA_TIMING"] = "1"

from worldfoundry_train.checkpoint import load_checkpoint  # noqa: E402
from worldfoundry_train.eval.jepa_probing import (  # noqa: E402
    append_jsonl, collapse_metrics, knn_hit_rate, linear_probe_accuracy,
    masked_reconstruction_cosine,
)
from worldfoundry_train.jepa_objective import JEPAObjective  # noqa: E402
from worldfoundry_train.model.lora_wrapper import build_lora_model  # noqa: E402
from worldfoundry_train.model.predictor import JEPAPredictor  # noqa: E402
from worldfoundry_train.parallel.fsdp import FSDPStrategy  # noqa: E402
from worldfoundry_train.trainer import run_training  # noqa: E402

SEED = 42
# Dry-run geometry: tiny VJEPA2, grid (1, 14, 14) -> N=196, hidden 64.
DRY_N, DRY_GRID, DRY_DIM = 196, (1, 14, 14), 64

# Pilot step-count defaults (review M5, 2026-08-05): single source of truth at
# module scope so the docstring "env W22_7_PILOT_STEPS, default 200 atlas /
# 40 dry" cannot drift from main()'s inline fallback. Override at runtime
# with the W22_7_PILOT_STEPS environment variable.
DEFAULT_PILOT_STEPS_ATLAS = 200
DEFAULT_PILOT_STEPS_DRY = 40


# ---------------------------------------------------------------------------
# dry-mode model + deterministic data
# ---------------------------------------------------------------------------

def _tiny_vjepa2():
    from transformers import VJEPA2Config, VJEPA2Model

    cfg = VJEPA2Config(
        hidden_size=DRY_DIM, num_hidden_layers=2, num_attention_heads=2,
        image_size=224, patch_size=16, tubelet_size=1, in_chans=3,
        frames_per_clip=1, drop_path_rate=0.0,
    )
    return VJEPA2Model(cfg)


def _dry_batch(i: int) -> dict[str, torch.Tensor]:
    """Deterministic per-index batch — identical sequence across arms."""
    rng = np.random.default_rng(i)
    return {
        "pixel_values_videos": torch.from_numpy(
            rng.random((1, 1, 3, 224, 224), dtype=np.float32)
        ),
        "bool_masked_pos": torch.from_numpy(
            rng.integers(0, 2, size=(1, DRY_N), dtype=bool)
        ),
    }


class _DryDataset:
    """Trainer requires a dataset for len()/grid derivation."""

    def __init__(self, n: int = 64):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng(i + 1)
        return {
            "frames": rng.random((1, 3, 224, 224), dtype=np.float32),
            "mask": rng.integers(0, 2, size=DRY_GRID, dtype=bool),
        }


def _dry_cfg(arm: str) -> dict:
    cfg = {
        "data": {"manifest": "/dry", "mask_tubelet_size": [1, 14, 14],
                 "mask_ratio": 0.5},
        "lora": {"r": 4, "alpha": 8, "dropout": 0.0, "bias": "none"},
        "training": {"per_device_batch": 1, "epochs": 1, "lr": 1e-3,
                     "log_every_n": 100},
        "jepa": {"enabled": arm == "b"},
    }
    if arm == "b":
        cfg["jepa"].update({
            "predictor": {"embed_dim": 32, "depth": 2, "num_heads": 2},
            "ema": {"momentum_start": 0.9, "momentum_end": 0.99,
                    "schedule": "cosine", "schedule_steps": None,
                    "update_every": 1},
            "loss": {"type": "smooth_l1", "beta": 1.0, "num_masks": 1},
        })
    return cfg


# ---------------------------------------------------------------------------
# probing set (synthetic, deterministic, shared across arms)
# ---------------------------------------------------------------------------

def _probe_set(n_classes: int = 4, clips_per_class: int = 6, seed: int = 777):
    """Synthetic labeled clips: class prototypes + independent noise draws.

    Returns (videos, labels, videos_seg2): ``videos_seg2`` is a second
    "time segment" of the same class (independent noise) used as the k-NN
    key set — the synthetic stand-in for "same video, different segment"
    (design §9.2-4). On Atlas this is replaced by real subset clips (hook:
    W22_7_PROBE_MANIFEST for a labeled probing manifest).
    """
    g = torch.Generator().manual_seed(seed)
    proto = torch.randn(n_classes, 1, 3, 224, 224, generator=g)
    v1, v2, labels = [], [], []
    for c in range(n_classes):
        for _ in range(clips_per_class):
            v1.append(proto[c] + 0.1 * torch.randn(proto[c].shape, generator=g))
            v2.append(proto[c] + 0.1 * torch.randn(proto[c].shape, generator=g))
            labels.append(c)
    return torch.stack(v1), torch.tensor(labels), torch.stack(v2)


@torch.no_grad()
def _encode(encoder, videos: torch.Tensor) -> torch.Tensor:
    """(M, 1, 3, H, W) -> (M, N, D) token features, no_grad, skip_predictor."""
    feats = []
    for i in range(videos.shape[0]):
        out = encoder(
            pixel_values_videos=videos[i : i + 1], skip_predictor=True
        ).last_hidden_state
        feats.append(out[0])
    return torch.stack(feats)  # (M, N, D)


def _compute_metrics(arm: str, step: int, encoder, objective, probe) -> dict:
    """Four §9.2 metrics for one checkpoint on the synthetic probe set."""
    videos, labels, videos_seg2 = probe
    feats = _encode(encoder, videos)          # (M, N, D) token-level
    feats2 = _encode(encoder, videos_seg2)
    clip = feats.mean(dim=1)                  # (M, D) clip-level
    clip2 = feats2.mean(dim=1)

    record: dict = {"step": step, "arm": arm}
    record.update(collapse_metrics(feats))

    # Linear probe: stratified even/odd split (clips are ordered by class —
    # a half/half split would put unseen classes in the test set).
    idx = torch.arange(clip.shape[0])
    tr, te = idx[::2], idx[1::2]
    record["linear_probe_acc"] = linear_probe_accuracy(
        clip[tr], labels[tr], clip[te], labels[te],
    )
    # k-NN consistency: seg-1 queries vs seg-2 keys, same class = hit.
    record["knn_hit_rate"] = knn_hit_rate(clip, clip2, labels, labels, k=1)

    # Masked reconstruction cosine. Arm B: the restored predictor + EMA
    # target encoder. Arm A: no predictor exists — use a fresh
    # random-init predictor against the encoder's own no-grad features to
    # establish the chance level (design §9.2: A ≈ random).
    g = torch.Generator().manual_seed(SEED)
    mask = torch.rand(1, DRY_N, generator=g) < 0.5
    h_ctx = _encode(encoder, videos[:1])
    if objective is not None:
        predictor = objective.predictor
        with torch.no_grad():
            h_tgt = objective.target_encoder(
                pixel_values_videos=videos[:1], skip_predictor=True
            ).last_hidden_state
    else:
        predictor = JEPAPredictor(
            num_patches=DRY_N, grid_size=DRY_GRID, encoder_dim=DRY_DIM,
            predictor_dim=32, depth=2, num_heads=2,
        )  # fresh random init -> chance level
        h_tgt = h_ctx.detach()
    with torch.no_grad():
        pred = predictor(h_ctx.detach(), mask)
        record["recon_cosine"] = masked_reconstruction_cosine(
            pred, h_tgt[0][mask[0]]
        )
    return record


# ---------------------------------------------------------------------------
# pilot mode: short run + early-stop checks (memory / throughput / numeric)
# ---------------------------------------------------------------------------

def _make_memory_gauge():
    """Return (sample_fn, source). sample_fn() -> per-step peak MB.

    NPU: torch.npu.max_memory_allocated with per-step peak reset.
    CPU dry: process peak RSS (monotonic; the leak check additionally
    requires a positive last-window slope, so warm-up growth alone does not
    false-positive). The source string lands in verdict evidence.
    """
    try:
        import torch_npu  # noqa: F401

        if torch.npu.is_available():
            def sample_npu() -> float:
                mb = torch.npu.max_memory_allocated() / 2**20
                torch.npu.reset_peak_memory_stats()
                return mb

            return sample_npu, "npu_peak"
    except Exception:
        pass

    import resource

    def sample_rss() -> float:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    return sample_rss, "cpu_rss_peak"


class _StepProbe:
    """step_callback sink: per-step wall dt + peak memory."""

    def __init__(self):
        self.sample_mem, self.mem_source = _make_memory_gauge()
        self.dt: list[float] = []
        self.mem_mb: list[float] = []
        self._t_prev: float | None = None

    def start(self) -> None:
        self._t_prev = time.perf_counter()

    def __call__(self, step: int, loss: float) -> None:
        now = time.perf_counter()
        if self._t_prev is not None:
            self.dt.append(now - self._t_prev)
        self._t_prev = now
        self.mem_mb.append(self.sample_mem())


def run_arm_pilot(arm: str, out_dir: Path, steps: int, backend: str) -> dict:
    """Short pilot run + early-stop verdict. Returns the verdict dict.

    Artifacts go to <out_dir>/<arm>/pilot/ so formal-run artifacts are never
    polluted. backend: "dry" (tiny CPU model, proves the harness) or
    "atlas" (real YAML config + real weights).
    """
    from worldfoundry_train.eval.pilot_checks import evaluate_pilot

    import worldfoundry_train.trainer as trainer_mod

    pilot_dir = out_dir / arm / "pilot"
    pilot_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    captured: list[JEPAObjective] = []
    original_cls = trainer_mod.JEPAObjective.from_online_model

    def _spy(cls, *args, **kwargs):
        obj = original_cls(*args, **kwargs)
        captured.append(obj)
        return obj

    trainer_mod.JEPAObjective.from_online_model = classmethod(_spy)
    original_ds = trainer_mod._build_dataset
    probe = _StepProbe()
    try:
        if backend == "dry":
            cfg = _dry_cfg(arm)
            trainer_mod._build_dataset = lambda c: _DryDataset()
            run_kwargs: dict = dict(
                base_model=_tiny_vjepa2(), num_steps=steps,
                batch_fn=_dry_batch,
            )
        else:
            import yaml

            default_cfg = Path(
                f"worldfoundry_train/configs/w22_7_arm_{arm}_"
                f"{'proxy' if arm == 'a' else 'jepa'}.yaml"
            )
            cfg = yaml.safe_load(default_cfg.read_text())
            manifest = Path(cfg["data"]["manifest"])
            if not manifest.is_file():
                raise FileNotFoundError(
                    f"subset manifest missing: {manifest} — run "
                    f"scripts/w22_7_make_subset.py first"
                )
            run_kwargs = dict(num_steps=steps)

        probe.start()
        t0 = time.perf_counter()
        losses = run_training(
            cfg, FSDPStrategy(), logger=lambda s, _loss: None,
            step_callback=probe, **run_kwargs,
        )
        wall = time.perf_counter() - t0
    finally:
        trainer_mod.JEPAObjective.from_online_model = original_cls
        trainer_mod._build_dataset = original_ds

    # EMA movement (arm B): compare fp32 master before/after.
    ema_moved = None
    timing = None
    if arm == "b" and captured:
        obj = captured[0]
        timing = dict(obj.timing or {})
        # A master that never updated equals the target's initial params;
        # any update with m<1 changes it (online != target at init is false
        # — they start equal — so movement means updates applied AND the
        # online weights moved, which any nonzero loss guarantees here).
        ema_moved = obj.ema.step > 0 and any(
            not torch.equal(buf, p.detach().to(torch.float32))
            for buf, p in zip(
                obj.ema.state_dict()["master"], obj.ema.online_params
            )
        )

    verdict = evaluate_pilot(
        arm=arm,
        loss_series=losses,
        step_time_series=probe.dt,
        memory_series_mb=probe.mem_mb,
        memory_source=probe.mem_source,
        timing=timing,
        ema_moved=ema_moved,
    )

    (pilot_dir / "throughput.json").write_text(json.dumps({
        "label": f"w22_7 arm {arm} pilot ({backend}; early-stop checks)",
        "arm": arm, "mode": "pilot", "backend": backend,
        "jepa_enabled": arm == "b",
        "optimizer_steps": len(losses),
        "wall_time_s": round(wall, 4),
        "avg_step_s": round(wall / max(len(losses), 1), 6),
        "step_time_series_s": [round(v, 6) for v in probe.dt],
        "memory_series_mb": [round(v, 2) for v in probe.mem_mb],
        "memory_source": probe.mem_source,
        "timing": timing,
        "loss_curve": [round(v, 6) for v in losses],
    }, indent=2))
    (pilot_dir / "verdict.json").write_text(
        json.dumps(verdict, indent=2, ensure_ascii=False)
    )

    mark = "✅" if verdict["verdict"] == "GO" else "⛔"
    print(f"[W22.7 PILOT] arm {arm} ({backend}): {mark} {verdict['verdict']} "
          f"— {verdict['summary']}")
    for c in verdict["checks"]:
        icon = {"pass": "  ✓", "warn": "  ⚠", "fail": "  ✗",
                "skipped": "  -"}[c["status"]]
        print(f"{icon} {c['name']}: {c['status']}"
              + (f" — {c['action']}" if c["action"] else ""))
    print(f"[W22.7 PILOT] artifacts: {pilot_dir}/{{throughput,verdict}}.json")
    return verdict


# ---------------------------------------------------------------------------
# arm driver
# ---------------------------------------------------------------------------

def _restore_for_probing(arm: str, ckpt: Path, cfg: dict):
    """Rebuild model (+ JEPA objective for arm B) and load a checkpoint."""
    torch.manual_seed(SEED)
    base = _tiny_vjepa2()
    lora = build_lora_model(
        base, r=int(cfg["lora"]["r"]), alpha=int(cfg["lora"]["alpha"]),
        dropout=float(cfg["lora"]["dropout"]),
    )
    model = lora.peft_model
    objective = None
    if arm == "b":
        jepa = cfg["jepa"]
        pred_cfg = jepa["predictor"]
        predictor = JEPAPredictor(
            num_patches=DRY_N, grid_size=DRY_GRID, encoder_dim=DRY_DIM,
            predictor_dim=int(pred_cfg["embed_dim"]),
            depth=int(pred_cfg["depth"]), num_heads=int(pred_cfg["num_heads"]),
        )
        objective = JEPAObjective.from_online_model(
            model, predictor,
            loss_type=jepa["loss"]["type"], beta=float(jepa["loss"]["beta"]),
            ema_momentum_start=float(jepa["ema"]["momentum_start"]),
            ema_momentum_end=float(jepa["ema"]["momentum_end"]),
            ema_schedule=jepa["ema"]["schedule"], ema_schedule_steps=10_000,
        )
    trainable = [p for p in model.parameters() if p.requires_grad]
    if objective is not None:
        # Match the trainer's JEPA two-group layout (LoRA + predictor) so the
        # saved optimizer state loads; probing itself never steps the optimizer.
        optimizer = torch.optim.AdamW(
            [{"params": trainable},
             {"params": list(objective.predictor.parameters())}],
            lr=1e-3,
        )
    else:
        optimizer = torch.optim.AdamW(trainable, lr=1e-3)
    load_checkpoint(ckpt, model, optimizer, jepa_objective=objective)
    model.eval()
    return model, objective


def run_arm_dry(arm: str, out_dir: Path, steps: int, save_every_n: int) -> dict:
    import worldfoundry_train.trainer as trainer_mod

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    cfg = _dry_cfg(arm)

    captured: list[JEPAObjective] = []
    original_cls = trainer_mod.JEPAObjective.from_online_model

    def _spy(cls, *args, **kwargs):
        obj = original_cls(*args, **kwargs)
        captured.append(obj)
        return obj

    original_ds = trainer_mod._build_dataset
    trainer_mod.JEPAObjective.from_online_model = classmethod(_spy)
    trainer_mod._build_dataset = lambda c: _DryDataset()
    ckpt_dir = out_dir / arm / "ckpt"
    try:
        t0 = time.perf_counter()
        losses = run_training(
            cfg, FSDPStrategy(), base_model=_tiny_vjepa2(),
            num_steps=steps, batch_fn=_dry_batch,
            logger=lambda s, _loss: None,
            checkpoint_dir=ckpt_dir, save_every_n=save_every_n,
        )
        wall = time.perf_counter() - t0
    finally:
        trainer_mod.JEPAObjective.from_online_model = original_cls
        trainer_mod._build_dataset = original_ds

    # --- probing per checkpoint step ---
    probe = _probe_set()
    metrics_path = out_dir / arm / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()
    records = []
    for ckpt in sorted(ckpt_dir.glob("step_*.pt"),
                       key=lambda p: int(p.stem.split("_")[1])):
        step = int(ckpt.stem.split("_")[1])
        model, objective = _restore_for_probing(arm, ckpt, cfg)
        rec = _compute_metrics(arm, step, model, objective, probe)
        append_jsonl(metrics_path, rec)
        records.append(rec)

    # --- throughput json (bench_results style) ---
    timing = dict(captured[0].timing) if captured else {}
    throughput = {
        "label": f"w22_7 arm {arm} dry-run (CPU; not a hardware baseline)",
        "arm": arm, "mode": "dry",
        "jepa_enabled": arm == "b",
        "optimizer_steps": len(losses),
        "wall_time_s": round(wall, 4),
        "avg_step_s": round(wall / max(len(losses), 1), 6),
        "timing": timing,
        "target_forward_fraction": (
            round(timing.get("target_forward_s", 0.0) / wall, 4)
            if timing else None
        ),
        "ema_fraction": (
            round(timing.get("ema_update_s", 0.0) / wall, 4)
            if timing else None
        ),
        "loss_curve": [round(v, 6) for v in losses],
    }
    tp_path = out_dir / arm / "throughput.json"
    tp_path.parent.mkdir(parents=True, exist_ok=True)
    tp_path.write_text(json.dumps(throughput, indent=2))

    print(f"[W22.7] arm {arm} dry-run done: steps={len(losses)} "
          f"wall={wall:.2f}s metrics={metrics_path} throughput={tp_path}")
    for rec in records:
        print(f"  step={rec['step']:>4} recon_cos={rec['recon_cosine']:.4f} "
              f"probe_acc={rec['linear_probe_acc']:.3f} "
              f"token_var={rec['token_variance']:.6f} "
              f"erank={rec['effective_rank']:.2f} knn={rec['knn_hit_rate']:.3f}")
    return throughput


def run_arm_atlas(arm: str, config: Path, out_dir: Path, steps: int | None,
                  save_every_n: int) -> None:
    """Real arm run from YAML. Single-process entry; torchrun wrapping for
    8-card hccl lives in scripts/w22_7_ab_experiment.sh (NPU migration W20).
    """
    import yaml

    cfg = yaml.safe_load(config.read_text())
    manifest = Path(cfg["data"]["manifest"])
    if not manifest.is_file():
        raise FileNotFoundError(
            f"subset manifest missing: {manifest} — run "
            f"scripts/w22_7_make_subset.py first"
        )
    ckpt_dir = out_dir / arm / "ckpt"
    t0 = time.perf_counter()
    losses = run_training(
        cfg, FSDPStrategy(), num_steps=steps,
        checkpoint_dir=ckpt_dir,
        save_every_n=save_every_n or int(cfg["training"].get("save_every_n", 0)),
    )
    wall = time.perf_counter() - t0
    throughput = {
        "label": f"w22_7 arm {arm} atlas run",
        "arm": arm, "mode": "atlas",
        "jepa_enabled": bool(cfg.get("jepa", {}).get("enabled", False)),
        "optimizer_steps": len(losses),
        "wall_time_s": round(wall, 4),
        "avg_step_s": round(wall / max(len(losses), 1), 6),
        "loss_curve": losses,
        # NOTE: per-component timing + per-checkpoint probing on real weights
        # is wired the same way as dry mode once W20 (NPU device migration)
        # lands; see scripts/w22_7_ab_experiment.sh TODO block.
    }
    tp_path = out_dir / arm / "throughput.json"
    tp_path.parent.mkdir(parents=True, exist_ok=True)
    tp_path.write_text(json.dumps(throughput, indent=2))
    print(f"[W22.7] arm {arm} atlas run done: steps={len(losses)} "
          f"wall={wall:.1f}s -> {tp_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, choices=["a", "b"])
    ap.add_argument("--mode", default="dry", choices=["dry", "atlas", "pilot"])
    ap.add_argument("--backend", default=None, choices=["dry", "atlas"],
                    help="pilot only: execution backend (default: atlas — the "
                         "real intent; pass dry for CPU harness validation)")
    ap.add_argument("--config", type=Path, default=None,
                    help="arm YAML (atlas mode; default configs/w22_7_arm_<X>_*.yaml)")
    ap.add_argument("--steps", type=int, default=None,
                    help="micro-batch steps (dry default 12; pilot default "
                         "W22_7_PILOT_STEPS=200 atlas / 40 dry; atlas default = cfg epochs)")
    ap.add_argument("--save-every-n", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("bench_results/w22_7"))
    args = ap.parse_args()

    if args.mode == "dry":
        run_arm_dry(args.arm, args.out, args.steps or 12,
                    args.save_every_n or 4)
    elif args.mode == "pilot":
        backend = args.backend or "atlas"
        default_steps = int(os.environ.get(
            "W22_7_PILOT_STEPS",
            str(DEFAULT_PILOT_STEPS_ATLAS if backend == "atlas"
                else DEFAULT_PILOT_STEPS_DRY)))
        verdict = run_arm_pilot(args.arm, args.out,
                                args.steps or default_steps, backend)
        return 0 if verdict["verdict"] == "GO" else 3
    else:
        default_cfg = Path(
            f"worldfoundry_train/configs/w22_7_arm_{args.arm}_"
            f"{'proxy' if args.arm == 'a' else 'jepa'}.yaml"
        )
        run_arm_atlas(args.arm, args.config or default_cfg, args.out,
                      args.steps, args.save_every_n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
