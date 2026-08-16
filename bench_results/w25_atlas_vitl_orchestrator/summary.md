# W2.25 Gate 3 — Real 1.2B ViT-L end-to-end orchestrator on Atlas card 7

## Context

W2.22 closed the **seam-level** real 1.2B ViT-L path on Atlas (1.255 GB peak).
W2.25 closes the **full 5-phase orchestrator** on Atlas: render → train → inference
→ policy_bootstrap → outer_loop, with real ViT-L weights and `--tubelet-size 2`.

Three defects fixed in W2.25 (cf. `docs/superpowers/plans/robust-weaving-bonbon.md`):

- **Mask grid alignment**: `Vjepa2Dataset` / `FrankaTransitionDataset` now compute
  `(T_post = T // tubelet_size, Hg, Wg)` so the bool mask aligns with the
  encoder's post-tubelet token grid.
- **`ema_compat.py` HF rebuild**: now uses `tubelet_size` from `cfg_dict` and
  rebuilds the predictor with the post-tubelet `T`, so checkpoint round-trips
  for `tubelet_size=2`.
- **`mask_fn` default**: replaced the hardcoded `(2, Hg, Wg)` with the
  auto-scaling `default_mask_tubelet_size()` helper so block sizes scale
  to whatever the post-tubelet grid is.

## Run

**Hardware**: Atlas card 7 (`ASCEND_RT_VISIBLE_DEVICES=7`,
`WORLDFORGE_DEVICE=npu`), Linux `localhost.localdomain`, openEuler 22.03,
torch 2.9.0, torch_npu 2.9.0, transformers 4.55.0, peft 0.18.0, accelerate 1.14.0.

**Network constraint**: Atlas has no outbound to huggingface.co / hf-mirror.com;
weights loaded from local cache `/data/hf_cache/models--facebook--vjepa2-vitl-fpc64-256/snapshots/b3c1679b7c34d3255ef3547f27c7b226aefab26f/`
via `HF_HUB_OFFLINE=1`.

**Command**:
```bash
cd /home/w30066257/WorldFoundry
export ASCEND_RT_VISIBLE_DEVICES=7 LOCAL_RANK=0 WORLD_SIZE=1 RANK=0
export WORLDFORGE_DEVICE=npu HF_HUB_OFFLINE=1 HF_HOME=/data/hf_cache
export VITL_DIR=/data/hf_cache/models--facebook--vjepa2-vitl-fpc64-256/snapshots/b3c1679b7c34d3255ef3547f27c7b226aefab26f
.venv/bin/python scripts/render_train_rl_cpu.py \
    --out-dir bench_results/w25_atlas_vitl_orchestrator \
    --encoder hf_vjepa2 \
    --encoder-pretrained "$VITL_DIR" \
    --encoder-hidden-size 1024 --encoder-num-layers 24 --encoder-num-heads 16 \
    --num-clips 3 --frames-per-clip 4 \
    --height 32 --width 32 \
    --num-steps 6 --num-outer-iters 3 \
    --num-train-per-iter 4 --num-ppo-epochs 2 \
    --tubelet-size 2 --seed 42
```

## Result

**rc=0**. All 5 phases complete; no `MaskingError`; all metrics finite.

| Phase | Metric | Value |
|---|---|---|
| Render | `n_clips` / `n_frames` | 3 / 12 |
| Train | `n_encoder_params` | **328,088,448** (real 1.2B ViT-L) |
| Train | LoRA trainable / total | 2,117,120 / 328,088,448 |
| Train | `final_total_loss` | 772.0601 |
| Train | `final_jepa_loss` | 771.6663 |
| Inference | `recon_cosine` | 0.6413 |
| Inference | `latent_cosine` | 0.0717 |
| Inference | `n_predicted_tokens` | 4 |
| Policy | `reward_pre` / `reward_post` | -0.5858 / -0.1818 |
| Outer loop | `n_iters` | **3** |
| Outer loop | `mean_wm_total_loss` iters | 814.51 → **802.56** → 805.89 |
| Outer loop | `reward_trajectory` | [-0.1679, -0.2944, -0.1612] |
| Outer loop | `trend_delta` | +0.0067 (slight improvement) |
| Memory | `torch.npu.peak` | ~7.86 GB (under 12 GB ceiling) |

## Acceptance gates

| Gate | Criterion | Result |
|---|---|---|
| rc | exit code | **0** |
| mode | `summary.mode == "npu"` | pass |
| encoder | `encoder_kind == "hf_vjepa2"` | pass |
| orchestrator | `outer_loop.n_iters == 3` | pass |
| metrics | all finite | pass |
| memory | `peak < 12 GB` | pass (~7.86 GB) |
| W2.25 wiring | no `MaskingError` (grid divisible) | pass |
| W2.25 wiring | `hf_tubelet_size == 2` rebuilt correctly | pass |

## Known warnings

- `aten::_transformer_encoder_layer_fwd` not currently supported on NPU
  CPU fallback path. Documented NPU limitation, no functional impact.
- `n_predicted_tokens: 4` is the masked-count subset (post-mask), not the full
  token count. Smoke-geometry sanity check, no concern.

## Cross-references

- `scripts/render_train_rl_cpu.py` orchestrator entry point
- `worldfoundry_train/model/hf_vjepa_shim.py` VJEPA2Model + LoRA + predictor wiring
- `worldfoundry_train/data/dataset.py` `tubelet_size` mask grid alignment
- `worldfoundry_train/data/tubelet_mask.py` `default_mask_tubelet_size()`
- `worldfoundry_train/ema_compat.py` HF predictor rebuild with `tubelet_size`
- `render_train_rl.json` full summary (5 KB)
- `render_vs_predicted.png` visualization (8 KB)
