# W46 — C-T8 M2 mainline smoke on Atlas card 7 (vllm_ascend backend)

> **Date:** 2026-08-18 · **Container:** `worldfoundry-franka:npu-cann9`
> (`m1t4-rl`, privileged, `/workspace` = project mount) · **NPU:** card 7
> (`ASCEND_RT_VISIBLE_DEVICES=7` → container-local `npu:0`) · **torch_npu:**
> 2.9.0.post1 · **vllm:** 0.18.0 · **vllm_ascend:** 0.18.0

## Command

```
python scripts/render_train_rl_cpu.py \
  --inference-backend vllm_ascend \
  --rl-backend self_ppo \
  --num-outer-iters 1 \
  --out-dir bench_results/w46_m2_mainline
```

## Result

| Field | Value |
|---|---|
| rc | 0 (5-phase orchestrator completed) |
| `summary.mode` | `npu` |
| `inference_backend` | `vllm_ascend` (CLI canonical name) |
| `recon_cosine` / `latent_cosine` | 0.6239 / 0.1498 (in-process path) |
| `outer_loop.per_iter[0].mean_wm_total_loss` | 9.9753 |
| reward trajectory | [-0.1686] |

## Engine dispatch status (honest note)

The vllm_ascend **engine-served** path did NOT dispatch in this run:

```
WARNING: inference backend 'vllm_ascend' resolves to an engine-served
VJEPA2Model surface, which does NOT serve this orchestrator's tiny/HF
world-model stack yet. Engine-served world-model inference is an Atlas
item; using in-process (torch) path for this run.
Dispatch attempt also failed: TypeError: VJEPA2NPUWorker.__init__()
missing 1 required positional argument: 'model_path'
```

`inference_engine_dispatched` = `{"engine": "vllm_ascend", "available":
false, "dispatch_error": "TypeError: ... missing 'model_path'"}`.

This matches the known M2-T5 state — the W43 evidence
(`bench_results/w43_vllm_ascend_e2e/{torch,vllm_ascend}/render_train_rl.json`)
records `inference_engine_dispatched: null` for BOTH runs, i.e. the
in-process `VJEPA2NPUWorker` engine path was never wired for the tiny
encoder stack; both "torch" and "vllm" M2-T5 runs executed the same
in-process torch path with `mode=npu`. The M2 mainline smoke here
therefore re-confirms: (1) orchestrator runs end-to-end on NPU card 7
with `--inference-backend vllm_ascend` accepted + recorded; (2) the
engine-served dispatch seam correctly surfaces the missing `model_path`
instead of silently crashing; (3) full engine-served inference remains a
follow-on that needs `VJEPA2NPUWorker(model_path=...)` with real
V-JEPA2 weights (W41 spec §3.2.2 gate).

## C-T8 gate status

| Gate | Criterion | Result |
|---|---|---|
| 5-phase orchestrator on Atlas card 7 | rc=0, summary JSON in `bench_results/` | ✅ `w46_m2_mainline/render_train_rl.json` |
| `summary.mode == "npu"` | Atlas hardware reflected | ✅ |
| `--inference-backend vllm_ascend` accepted | argparse + registry wiring | ✅ (regression test `test_rti_orchestrator_resolves_vllm_ascend_backend`) |
| Engine-served inference (`model_path`) | `VJEPA2NPUWorker` dispatches in-process | ⏸ follow-on (needs real V-JEPA2 weights; known from M2-T5 partial closure) |

## Cross-references

- `bench_results/w43_vllm_ascend_e2e/summary.md` — M2-T5 partial closure
- `docs/superpowers/specs/2026-08-15-w41-vllm-ascend-integration.md` — W41 spec
- `docs/roadmap/2026-08-14-opensource-extensions-loop-todo.md` §9.3 — P2 (C-T8)
