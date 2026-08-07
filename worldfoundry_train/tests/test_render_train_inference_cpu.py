"""RTI v1 tests: synthetic render -> tiny JEPA train -> pixel inference (CPU).

Covers the five modules that make the CPU render->train->inference loop
work without the HF V-JEPA 2 1.2B download:

- ``worldfoundry_train.data.synthetic_video``    (Render)
- ``worldfoundry_train.data.synthetic_manifest`` (Render)
- ``worldfoundry_train.model.tiny_jepa``         (Train / Inference)
- ``worldfoundry_train.ema_compat``              (Train)
- ``scripts/render_train_inference_cpu.py``      (orchestrator, subprocess)

Everything here is CPU-only and sized so the whole file runs in well
under a minute (32x32 clips, depth-2 encoders, <=6 optimizer steps).
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from worldfoundry_train.data.synthetic_manifest import (
    SyntheticManifestSpec,
    build_synthetic_manifest,
)
from worldfoundry_train.data.synthetic_video import (
    write_bouncing_ball_video,
    write_rotating_stripes_video,
    write_synthetic_clip,
)
from worldfoundry_train.ema_compat import (
    TinyEMAUpdater,
    load_tiny_checkpoint,
    save_tiny_checkpoint,
)
from worldfoundry_train.model.tiny_jepa import (
    TinyEncoderConfig,
    TinyJEPAPredictor,
    TinyPixelDecoder,
    TinyVJEPA2Encoder,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

# Small enough to keep every test sub-second on one CPU thread.
TINY_CFG = TinyEncoderConfig(image_size=32, patch_size=8, embed_dim=32, depth=2, num_heads=2)


# ---------------------------------------------------------------------------
# Render phase
# ---------------------------------------------------------------------------


def test_bouncing_ball_is_deterministic(tmp_path: Path) -> None:
    a = write_bouncing_ball_video(tmp_path / "a.mp4", num_frames=4, height=32, width=32, seed=7)
    b = write_bouncing_ball_video(tmp_path / "b.mp4", num_frames=4, height=32, width=32, seed=7)
    assert a.read_bytes() == b.read_bytes()


def test_rotating_stripes_differs_by_seed(tmp_path: Path) -> None:
    a = write_rotating_stripes_video(tmp_path / "a.mp4", num_frames=4, height=32, width=32, seed=1)
    b = write_rotating_stripes_video(tmp_path / "b.mp4", num_frames=4, height=32, width=32, seed=2)
    assert a.read_bytes() != b.read_bytes()


def test_synthetic_clip_routes_by_index(tmp_path: Path) -> None:
    """Even index -> bouncing ball, odd -> rotating stripes (byte-comparable)."""
    even = write_synthetic_clip(tmp_path, 0, num_frames=4, height=32, width=32, seed=3)
    odd = write_synthetic_clip(tmp_path, 1, num_frames=4, height=32, width=32, seed=3)
    assert even.name == "clip_0000.mp4"
    assert odd.name == "clip_0001.mp4"
    expect_even = write_bouncing_ball_video(
        tmp_path / "ref_even.mp4", num_frames=4, height=32, width=32, seed=3 * 1000 + 0,
    )
    expect_odd = write_rotating_stripes_video(
        tmp_path / "ref_odd.mp4", num_frames=4, height=32, width=32, seed=3 * 1000 + 1,
    )
    assert even.read_bytes() == expect_even.read_bytes()
    assert odd.read_bytes() == expect_odd.read_bytes()


def test_video_rejects_bad_geometry(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        write_bouncing_ball_video(tmp_path / "x.mp4", num_frames=1, height=32, width=32)
    with pytest.raises(ValueError):
        write_bouncing_ball_video(tmp_path / "x.mp4", num_frames=4, height=4, width=4)


def test_manifest_schema_and_row_count(tmp_path: Path) -> None:
    spec = SyntheticManifestSpec(
        out_dir=tmp_path, num_clips=3, frames_per_clip=4, height=32, width=32, seed=5,
    )
    manifest = build_synthetic_manifest(spec)
    with manifest.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    # Vjepa2Dataset asserts on this exact column order.
    assert header == ["video_path", "num_frames", "height", "width"]
    assert len(rows) == 3
    for row in rows:
        video_path = Path(row[0])
        assert video_path.is_absolute() and video_path.is_file()
        assert row[1:] == ["4", "32", "32"]


def test_build_synthetic_manifest_rejects_bad_inputs(tmp_path: Path) -> None:
    for bad in (
        dict(num_clips=0, frames_per_clip=4, height=32, width=32),
        dict(num_clips=2, frames_per_clip=1, height=32, width=32),
        dict(num_clips=2, frames_per_clip=4, height=4, width=32),
    ):
        with pytest.raises(ValueError):
            build_synthetic_manifest(SyntheticManifestSpec(out_dir=tmp_path, seed=0, **bad))


def test_video_loads_via_vjepa2_dataset(tmp_path: Path) -> None:
    """The Render output must be consumable by the existing dataset unchanged."""
    from worldfoundry_train.data.dataset import Vjepa2Dataset
    from worldfoundry_train.data.tubelet_mask import tubelet_mask

    spec = SyntheticManifestSpec(
        out_dir=tmp_path, num_clips=2, frames_per_clip=4, height=32, width=32, seed=11,
    )
    manifest = build_synthetic_manifest(spec)
    grid_hw = 32 // 8

    dataset = Vjepa2Dataset(
        manifest_path=manifest,
        mask_fn=lambda grid, rng: tubelet_mask(
            grid, tubelet_size=(2, grid_hw, grid_hw), mask_ratio=0.5, rng=rng,
        ),
        patch_size=8,
        mask_count=1,
    )
    assert len(dataset) == 2
    item = dataset[0]
    assert item["frames"].shape == (4, 3, 32, 32)
    assert item["frames"].dtype == np.float32
    assert 0.0 <= float(item["frames"].min()) and float(item["frames"].max()) <= 1.0
    assert item["mask"].shape == (4, grid_hw, grid_hw)


# ---------------------------------------------------------------------------
# Tiny model stack
# ---------------------------------------------------------------------------


def test_tiny_encoder_forward_shape() -> None:
    enc = TinyVJEPA2Encoder(TINY_CFG)
    x = torch.rand(2, 4, 3, 32, 32)
    out = enc(x, skip_predictor=True)
    n_tokens = 4 * TINY_CFG.grid_h * TINY_CFG.grid_w
    assert out.last_hidden_state.shape == (2, n_tokens, TINY_CFG.embed_dim)


def test_tiny_encoder_rejects_wrong_shape() -> None:
    enc = TinyVJEPA2Encoder(TINY_CFG)
    with pytest.raises(ValueError):
        enc(torch.rand(2, 3, 32, 32))  # missing time axis
    with pytest.raises(ValueError):
        enc(torch.rand(2, 4, 3, 40, 40))  # wrong spatial size


def test_tiny_decoder_round_trip_shape() -> None:
    enc = TinyVJEPA2Encoder(TINY_CFG)
    dec = TinyPixelDecoder(TINY_CFG)
    x = torch.rand(1, 4, 3, 32, 32)
    pixels = dec(enc(x).last_hidden_state, num_frames=4)
    assert pixels.shape == x.shape
    assert float(pixels.detach().min()) >= 0.0 and float(pixels.detach().max()) <= 1.0


def test_tiny_decoder_rejects_mismatched_seq_length() -> None:
    dec = TinyPixelDecoder(TINY_CFG)
    n_tokens = 4 * TINY_CFG.grid_h * TINY_CFG.grid_w
    with pytest.raises(ValueError):
        dec(torch.rand(1, n_tokens, TINY_CFG.embed_dim), num_frames=3)


def test_tiny_predictor_returns_masked_positions() -> None:
    enc = TinyVJEPA2Encoder(TINY_CFG)
    pred = TinyJEPAPredictor(TINY_CFG)
    h = enc(torch.rand(1, 4, 3, 32, 32)).last_hidden_state
    n_tokens = h.shape[1]
    mask = torch.zeros(1, n_tokens, dtype=torch.bool)
    mask[0, : n_tokens // 2] = True
    out = pred(h, mask)
    assert out.shape == (n_tokens // 2, TINY_CFG.embed_dim)


def test_tiny_stack_takes_a_real_gradient_step() -> None:
    """One JEPA+recon step must actually move the loss (Train phase contract)."""
    from worldfoundry_train.jepa_objective import jepa_loss

    torch.manual_seed(0)
    enc = TinyVJEPA2Encoder(TINY_CFG)
    pred = TinyJEPAPredictor(TINY_CFG)
    dec = TinyPixelDecoder(TINY_CFG)
    target = TinyVJEPA2Encoder(TINY_CFG)
    target.load_state_dict(enc.state_dict())
    for p in target.parameters():
        p.requires_grad_(False)

    x = torch.rand(1, 4, 3, 32, 32)
    n_tokens = 4 * TINY_CFG.grid_h * TINY_CFG.grid_w
    mask = torch.zeros(1, n_tokens, dtype=torch.bool)
    mask[0, ::2] = True

    params = list(enc.parameters()) + list(pred.parameters()) + list(dec.parameters())
    opt = torch.optim.AdamW(params, lr=1e-2)

    def step() -> float:
        h = enc(x).last_hidden_state
        with torch.no_grad():
            h_t = target(x).last_hidden_state
        loss = jepa_loss(pred(h, mask), h_t[mask], loss_type="smooth_l1", beta=1.0)
        loss = loss + torch.nn.functional.l1_loss(dec(h, num_frames=4), x)
        opt.zero_grad()
        loss.backward()
        opt.step()
        return float(loss.detach())

    first = step()
    for _ in range(4):
        last = step()
    assert np.isfinite(first) and np.isfinite(last)
    assert last < first


# ---------------------------------------------------------------------------
# EMA + checkpoint
# ---------------------------------------------------------------------------


def test_tiny_ema_update_moves_target_toward_online() -> None:
    online = TinyVJEPA2Encoder(TINY_CFG)
    target = TinyVJEPA2Encoder(TINY_CFG)
    target.load_state_dict(online.state_dict())
    with torch.no_grad():
        for p in online.parameters():
            p.add_(1.0)

    before = next(target.parameters()).clone()
    TinyEMAUpdater(momentum=0.9).update(target, online)
    after = next(target.parameters())
    assert not torch.allclose(before, after)
    # target <- 0.9*target + 0.1*(target+1) == target + 0.1
    assert torch.allclose(after, before + 0.1, atol=1e-6)


def test_tiny_ema_rejects_bad_momentum() -> None:
    with pytest.raises(ValueError):
        TinyEMAUpdater(momentum=1.5)
    with pytest.raises(ValueError):
        TinyEMAUpdater(momentum=-0.1)


def test_tiny_checkpoint_round_trip(tmp_path: Path) -> None:
    enc = TinyVJEPA2Encoder(TINY_CFG)
    pred = TinyJEPAPredictor(TINY_CFG)
    dec = TinyPixelDecoder(TINY_CFG)
    opt = torch.optim.AdamW(enc.parameters(), lr=1e-3)

    path = save_tiny_checkpoint(
        tmp_path / "ck.pt", encoder=enc, predictor=pred, decoder=dec, optimizer=opt, step=7,
    )
    assert path.is_file()
    assert not (tmp_path / "ck.pt.tmp").exists()  # atomic write cleaned up

    bundle = load_tiny_checkpoint(path, device="cpu")
    assert bundle["step"] == 7
    assert bundle["encoder_config"]["embed_dim"] == TINY_CFG.embed_dim

    x = torch.rand(1, 4, 3, 32, 32)
    enc.eval()
    bundle["encoder"].eval()
    with torch.no_grad():
        assert torch.allclose(
            enc(x).last_hidden_state, bundle["encoder"](x).last_hidden_state, atol=1e-6,
        )


def test_tiny_checkpoint_load_raises_on_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_tiny_checkpoint(tmp_path / "nope.pt")


def test_tiny_checkpoint_load_raises_on_incomplete_payload(tmp_path: Path) -> None:
    path = tmp_path / "bad.pt"
    torch.save({"step": 1}, path)
    with pytest.raises(KeyError):
        load_tiny_checkpoint(path)


# ---------------------------------------------------------------------------
# Orchestrator (end-to-end, subprocess)
# ---------------------------------------------------------------------------


def test_orchestrator_runs_end_to_end(tmp_path: Path) -> None:
    """Render -> Train -> Inference must complete and emit a valid summary."""
    proc = subprocess.run(
        [
            sys.executable, str(REPO_ROOT / "scripts" / "render_train_inference_cpu.py"),
            "--out-dir", str(tmp_path),
            "--num-clips", "2",
            "--frames-per-clip", "4",
            "--height", "32", "--width", "32",
            "--num-steps", "4",
            "--seed", "42",
        ],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]

    summary = json.loads((tmp_path / "render_train_inference.json").read_text())
    assert summary["mode"] == "cpu"
    assert summary["train"]["n_train_steps"] == 4
    assert summary["train"]["n_encoder_params"] > 0
    assert np.isfinite(summary["train"]["final_total_loss"])
    assert -1.0 <= summary["inference"]["recon_cosine"] <= 1.0
    assert summary["inference"]["n_predicted_tokens"] > 0
    for phase in ("render", "train", "inference"):
        assert summary[phase]["elapsed_seconds"] >= 0.0

    assert (tmp_path / "manifest.csv").is_file()
    assert (tmp_path / "tiny_jepa.pt").is_file()
    assert (tmp_path / "render_vs_predicted.png").is_file()
    assert len(list((tmp_path / "videos").glob("clip_*.mp4"))) == 2


def test_orchestrator_skip_render_reuses_existing_manifest(tmp_path: Path) -> None:
    build_synthetic_manifest(SyntheticManifestSpec(
        out_dir=tmp_path, num_clips=2, frames_per_clip=4, height=32, width=32, seed=42,
    ))
    argv = [
        sys.executable, str(REPO_ROOT / "scripts" / "render_train_inference_cpu.py"),
        "--out-dir", str(tmp_path), "--num-clips", "2", "--frames-per-clip", "4",
        "--height", "32", "--width", "32", "--num-steps", "2", "--skip-render",
    ]
    proc = subprocess.run(argv, cwd=REPO_ROOT, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-3000:]
    summary = json.loads((tmp_path / "render_train_inference.json").read_text())
    assert summary["render"]["elapsed_seconds"] == 0.0

    # Missing manifest + --skip-render is a config error (rc=2), not a crash.
    empty = tmp_path / "empty"
    empty.mkdir()
    argv[argv.index("--out-dir") + 1] = str(empty)
    proc2 = subprocess.run(argv, cwd=REPO_ROOT, capture_output=True, text=True, timeout=600)
    assert proc2.returncode == 2
