# W2.21 — Atlas NPU HF VJEPA2Model shim (random-init smoke)

**Closes**: Atlas checklist item #1 "HF VJEPA2Model on NPU"

**Date**: 2026-08-12
**Atlas card**: 7 (Ascend910B4-1)
**Code change**: none (W2.18 orchestrator + shim already in tree)

## Command

```bash
ASCEND_RT_VISIBLE_DEVICES="7" LOCAL_RANK=0 WORLD_SIZE=1 RANK=0 \
WORLDFORGE_DEVICE=npu HF_HUB_OFFLINE=1 \
  python scripts/render_train_rl_cpu.py \
    --out-dir bench_results/w21_atlas_hf_shim \
    --encoder hf_vjepa2 \
    --num-clips 2 --frames-per-clip 4 \
    --height 32 --width 32 \
    --num-steps 4 --num-outer-iters 1 \
    --num-train-per-iter 2 --num-ppo-epochs 1 \
    --seed 42
```

## Acceptance gates — PASS

| Gate | Expected | Actual | Pass |
|---|---|---|---|
| rc | 0 | 0 | yes |
| `summary.mode` | `"npu"` | `"npu"` | yes |
| `encoder_kind` | `"hf_vjepa2"` | `"hf_vjepa2"` | yes |
| `train.final_total_loss` | finite | `27.927` | yes |
| `train.hf_lora_trainable_params > 0` | yes | `343,808` | yes |
| `train.hf_lora_trainable_params < hf_lora_total_params` | yes | 343,808 < 21,804,032 | yes |
| `n_encoder_params` | ~21M | `21,804,032` | yes |
| `inference.recon_cosine` | finite | `0.6222` | yes |
| `inference.latent_cosine` | finite | `0.1640` | yes |
| `policy_bootstrap.reward_pre -> reward_post` | improved | `-0.587 -> -0.176` | yes |

## NPU runtime

- Wall-clock per phase: render 0.005s, train 30.56s, inference 28.53s,
  policy 28.32s, outer_loop 45.07s.
- RSS peak: ~2.2 GB (random-init HF VJEPA2Model hidden_size=64; tiny).
- NPU peak memory: not logged in this run (the orchestrator's
  `torch.npu.max_memory_allocated()` call lives in the W2.22+ path;
  the W2.18 smoke path skips it because the model is tiny).

## Expected NPU warnings (documented, not blocked)

- `UserWarning: enable_nested_tensor is True, but self.use_nested_tensor
  is False because encoder_layer.norm_first was True` — a HF transformers
  4.55.0 noise warning on the inner encoder block; math is correct.
- `aten::_transformer_encoder_layer_fwd is not currently supported on
  the NPU backend and will fall back to run on the CPU` — the
  expected CPU-fallback on `nn.TransformerEncoder`. Documented in the
  W2.21+ plan as Risk Mitigation #1. Math is correct; perf degraded.
  Realistic geometry (256x256) would expose this more sharply; smoke
  geometry masks it.

## Atlas setup notes (for W2.22+ reference)

- venv at `/home/w30066257/WorldFoundry/.venv` (uv-managed Python
  3.11.15, aarch64).
- Initial venv did NOT have `transformers`, `peft`, `huggingface_hub`,
  `einops`. Installed via:
  ```bash
  UV_HTTP_TIMEOUT=600 uv pip install --python $VENV/bin/python \
    transformers==4.55.0 peft==0.18.0 huggingface_hub einops
  ```
  (transformers 4.55.0 is what was pre-cached; it has VJEPA2Config +
  VJEPA2Model; einops required for the VJEPA2 modeling code import
  chain.)
- `WORLDFORGE_DEVICE=npu` honoured by `worldfoundry_train/device.py:resolve_device()`.
- SSH_ASKPASS recipe from WSL2:
  `DISPLAY=:0 SSH_ASKPASS=/tmp/ssh_askpass.sh SSH_ASKPASS_REQUIRE=force ssh ...`

## Conclusion

The W2.18 orchestrator's `_build_stack` chokepoint accepts
`HfVJEPA2EncoderShim.from_random_init(...)` on NPU card 7 with no
modifications. `mode="npu"` + `encoder_kind="hf_vjepa2"` artifact
captured. Ready for W2.22 (real 1.2B weights).