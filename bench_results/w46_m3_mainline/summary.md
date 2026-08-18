# W46 — C-T8 M3 mainline smoke on Atlas card 7 (seam-level closed)

> Date: 2026-08-18 · Container: `worldfoundry-franka:npu-cann9`
> (`m1t4-rl`, privileged, `/workspace` = project mount) · NPU: card 7
> (`ASCEND_RT_VISIBLE_DEVICES=7` → container-local `npu:0`)

## Command

```
python scripts/render_train_rl_cpu.py \
  --inference-backend torch --rl-backend self_ppo \
  --frame-gen-backend mindie_sd \
  --num-outer-iters 1 \
  --out-dir bench_results/w46_m3_mainline
```

## Result

| Field | Value |
|---|---|
| rc | 0 (5-phase orchestrator completed) |
| `summary.mode` | `npu` |
| `frame_gen.backend` | `mindie_sd` |
| `frame_gen.dispatched` | `True` |
| `frame_gen.engine_type` | `CPUPixelDecoderPipeline` (v1 CPU pixel-decoder stub) |
| `frame_gen.n_rollout_frames` | 2 |
| `frame_gen.elapsed_seconds` | 191.8 (incl. ~190s HF CLIP-download retry timeout) |
| `frame_metrics.ssim` / `psnr_db` | 0.0467 / 9.4634 |
| `frame_metrics.clip_score` | null (huggingface.co unreachable in container — CLIP model download blocked) |
| `outer wm_loss` | 9.9753 |

## Honest note

M3-T8 seam is wired and the 5-phase orchestrator now invokes the
frame_gen phase with `mindie_sd` backend (`dispatched=True`). The
active engine is the v1 CPU pixel-decoder stub
(`CPUPixelDecoderPipeline`) — the real NPU diffusion path
(`NPUDiffusionPipeline`) still raises `PipelineNotAvailableError` until
M3-T4's compiled `.om` is loaded into
`worldfoundry_inf/diffusion/backends/npu.py`. `frame_metrics` were
computed by the M3-T6 eval module but against stub output, so SSIM/PSNR
are low and CLIP is null (HF unreachable); they are NOT production
quality numbers. M3 mainline = seam-level closed; real-NPU rollout is
the follow-on (roadmap §9.4 P3: M3-T4 `.om` → `NPUDiffusionPipeline`).

## C-T8 gate status

| Gate | Criterion | Result |
|---|---|---|
| 5-phase orchestrator on Atlas card 7 | rc=0, summary JSON in `bench_results/` | ✅ `w46_m3_mainline/render_train_rl.json` |
| frame_gen phase invoked | `--frame-gen-backend mindie_sd` dispatches | ✅ `dispatched=True` |
| frame_metrics recorded | SSIM/PSNR/CLIP eval module runs (M3-T6) | ✅ ssim/psnr computed; clip null (HF blocked) |
| Real NPU diffusion | `NPUDiffusionPipeline` loads `.om` | ⏸ follow-on after M3-T4 `.om` load |

## Cross-references

- `bench_results/w46_m3_t4_compile/summary.md` — M3-T4 real `.om` compile (closed 2026-08-18)
- `bench_results/w45_m3_mainline_summary.md` — previous seam-only status
- `docs/roadmap/2026-08-14-opensource-extensions-loop-todo.md` §9.4 — P3 (M3-T4..T8)
