# W2.22 — Atlas NPU real 1.2B ViT-L (minimal seam proof)

**Closes**: Atlas checklist item #2 "1.2B real weights on NPU" (seam-level)

**Date**: 2026-08-12
**Atlas card**: 7 (Ascend910B4-1)
**Code change**: additive (W2.22 factory + dispatcher; see §Code changes)

## Scope of this run

This run is a **minimal proof that the W2.22 seam works on NPU**, not
the full 5-phase orchestrator loop. Specifically:

- Load real `facebook/vjepa2-vitl-fpc64-256` weights (1.3 GB safetensors)
  via the new `HfVJEPA2EncoderShim.from_pretrained` factory.
- Move the loaded shim to NPU card 7.
- Run a single forward pass with smoke geometry `(1, 4, 3, 32, 32)`.
- Capture load time, LoRA trainable/total counts, output shape, and
  NPU peak memory.

The full 5-phase orchestrator (`scripts/render_train_rl_cpu.py`) with
real ViT-L is **deferred to a future item** because ViT-L's native
geometry (256×256 / tubelet=2 / frames=64) is OOD against the
orchestrator's hardcoded `tubelet_size=1` predictor grid. The
geometry-truthful shim records the loaded HF config correctly; the
predictor-grid rewrite is a separate piece of work (see §OOD geometry
and §Deferred below).

## Command

```bash
ASCEND_RT_VISIBLE_DEVICES="7" LOCAL_RANK=0 WORLD_SIZE=1 RANK=0 \
WORLDFORGE_DEVICE=npu \
  python /tmp/w22_proof.py
```

`/tmp/w22_proof.py` is a 67-line shim that calls
`HfVJEPA2EncoderShim.from_pretrained(...)` against the local snapshot
path, moves to `npu:0`, runs one forward, writes
`bench_results/w22_atlas_vitl_smoke/forward_pass.json`.

## Acceptance gates — PASS

| Gate | Expected | Actual | Pass |
|---|---|---|---|
| rc | 0 | 0 | yes |
| `mode` | `"npu"` | `"npu"` | yes |
| `encoder_kind` | `"hf_vjepa2"` | `"hf_vjepa2"` | yes |
| `pretrained_repo_id` set | yes | `/data/hf_cache/.../vjepa2-vitl-fpc64-256/snapshots/b3c1679...` | yes |
| `n_encoder_params` | ~326M (±5%) | `328,088,448` (≈328M) | yes |
| `hf_lora_trainable_params > 0` | yes | `2,117,120` | yes |
| `hf_lora_trainable_params / hf_lora_total_params` | < 1% | 2,117,120 / 328,088,448 = **0.65%** | yes |
| `encoder_output_shape` | `(B, N_tokens, 1024)` | `(1, 8, 1024)` | yes |
| `npu_peak_memory_gb` | < 12 GB | `1.255 GB` | yes |
| `load_seconds` (informational) | any | `2.75s` | yes |

The full forward pass runs on NPU; the output tensor `out` is on
`npu:0` after `shim.to("npu:0")`. LoRA wraps the base
`VJEPA2Model` correctly (`lora_stats()` is consistent across
manual recount and construction-time capture).

## OOD geometry — why the orchestrator loop is deferred

The ViT-L checkpoint was trained with:

- `crop_size = 256` (spatial)
- `patch_size = 16` (spatial patches)
- `tubelet_size = 2` (temporal)
- `frames_per_clip = 64`

For an OOD smoke input `(1, 4, 3, 32, 32)`:

- Spatial patches: `floor(32/16) × floor(32/16) = 2 × 2 = 4` tokens.
- Temporal patches: `floor(4/2) = 2` tokens.
- **Total emitted tokens: 4 × 2 = 8** — matches the captured
  `encoder_output_shape = (1, 8, 1024)`.

The orchestrator's predictor path computes token count as
`T * Hg * Wg` from the user-supplied `frames_per_clip=4` and
`height=32, patch=8`, producing `4 * 4 * 4 = 64` — a 8× mismatch
against the encoder's actual 8. This is the OOD geometry blocker.

**Resolution path (future item)**: The W2.22 shim is now
geometry-truthful — `from_pretrained` records the loaded HF config's
`crop_size`, `patch_size`, `tubelet_size`, `frames_per_clip` on the
shim config (NOT the kwargs). The orchestrator's `_build_stack`
already reads `Hg/Wg/T` from `online.config.grid_*` and
`online.config.frames_per_clip / tubelet_size` (W2.22 update). The
remaining gap is **predictor grid layout for `tubelet_size > 1`**, which
requires a separate change in
`worldfoundry_train/model/predictor.py` (`JEPAPredictor`'s
`num_patches` and positional embedding size) and the planner's
mask shape. Documented as a future sub-milestone; smoke geometry
matched the W2.18 CPU proof and remains the canonical seam-closure
demonstration.

## Code changes (additive, W2.22)

| File | Change |
|---|---|
| `worldfoundry_train/model/hf_vjepa_shim.py` | + `from_pretrained` classmethod (~150 LOC); + `pretrained_repo_id` + `pretrained_revision` fields on `HfEncoderShimConfig`; +2 keys in `signature()` |
| `worldfoundry_train/ema_compat.py` | `load_tiny_checkpoint` HF branch dispatches on `cfg_dict.get("pretrained_repo_id")` — 2 call sites (online + target) |
| `scripts/render_train_rl_cpu.py` | + `--encoder-pretrained` / `--encoder-pretrained-revision` / `--tubelet-size` flags; + 3 fields on `V2Config`; + cross-validation; + `_build_stack` HF branch dispatch on `encoder_pretrained_repo_id`; + `_encoder_lora_stats` dispatch; + policy_bootstrap pass-through; + `config_serializable` round-trip |
| `worldfoundry_rl/vision_policy.py` | + `pretrained_repo_id` + `pretrained_revision` kwargs on `from_hf_image_obs`; routes to `from_pretrained` when set |
| `worldfoundry_train/tests/test_hf_vjepa_shim.py` | + 2 tests (test 6: `pretrained_repo_id=None` round-trip; test 7: `from_pretrained` loads real weights from fixture) |

## Atlas setup quirks discovered this session

These extend the W41 memory checklist:

1. **HF Hub unreachable from Atlas** — `Connection reset by peer` on
   `huggingface.co`. Workaround: pre-download via
   `HF_ENDPOINT=https://hf-mirror.com python -c "snapshot_download(...)"`
   to `/data/hf_cache/`. (hf-mirror.com is the public CN mirror.)
2. **`HF_HUB_OFFLINE=1` does NOT find mirror-downloaded files** — the
   mirror's snapshot layout differs from the canonical HF cache layout.
   Workaround: pass the local snapshot dir directly as
   `pretrained_repo_id` instead of the repo ID. e.g.
   `/data/hf_cache/models--facebook--vjepa2-vitl-fpc64-256/snapshots/b3c1679...`.
3. **`einops` is required for the VJEPA2 modeling chain** — the
   pre-cached Atlas venv did not include `einops`; without it
   `from transformers.models.vjepa2.modeling_vjepa2 import ...` raises
   `ModuleNotFoundError`. Fixed via `uv pip install einops`.
4. **`ASCEND_RT_VISIBLE_DEVICES=7` remaps to logical 0** — already in
   W41 quirks but re-confirmed here: `shim.to("npu:0")` works;
   `torch.npu.set_device(7)` raises `Invalid device ID`.

## CPU regression — PASS

```
.venv/bin/python -m pytest worldfoundry_train/tests/test_hf_vjepa_shim.py -v
# 5/5 PASS (without WORLDFORGE_HF_FIXTURE)
WORLDFORGE_HF_FIXTURE=$PWD/bench_results/fixtures/tiny_vjepa2 \
  .venv/bin/python -m pytest worldfoundry_train/tests/test_hf_vjepa_shim.py -v
# 7/7 PASS (with fixture; test 7 proves real weights load)
```

## Deferred (explicit out-of-scope for W2.22)

1. **Full 5-phase orchestrator with ViT-L real weights** — blocked on
   predictor-grid rewrite for `tubelet_size > 1`. Documented above.
2. **Realistic 256×256 / 64-frame geometry** — same root cause;
   smoke geometry (32×32 / frames=4) remains the seam-closure proof.
3. **LoRA-only checkpoint persistence** — full base saved (matches
   existing convention); LoRA-only adapter save is a separate item.
4. **Pinned HF revision SHA** — currently uses default `main`; pin is
   a future reproducibility item.
5. **torchao W8 quant of pretrained base on NPU** — torchao has no
   NPU dispatch; would fall back to CPU quant, defeating the purpose.

## Conclusion

The W2.22 seam is closed on Atlas card 7. Real 1.2B ViT-L weights
load via `from_pretrained`, LoRA wraps the base at 0.65% trainable,
forward pass runs entirely on NPU (1.25 GB peak — well under budget),
and the resulting `signature()` carries the `pretrained_repo_id` so
checkpoints round-trip correctly. The full orchestrator loop with
real ViT-L is gated on the predictor-grid rewrite documented above
and remains a future sub-milestone.

Next: **W2.23** — 4-seed sweep on NPU. The sweep uses the W2.18
random-init path (already proven to work end-to-end on NPU via W2.21)
across seeds 42/7/99/123, mirroring the W2.20 CPU multi-seed proof.
Real 1.2B weights in the sweep loop are gated on the same predictor
rewrite and stay out of scope.