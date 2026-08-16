# WorldFoundry

Ascend NPU world model training + inference platform. **Phase-1** prototype ships
training + inference for Meta V-JEPA 2 (1.2B).

## Prerequisites

WorldFoundry installs **only inside the Ascend NPU base image**:

```dockerfile
FROM ascend-pytorch:2.9.0-cann8.5.1-ubuntu22.04
```

The image ships `torch==2.9.0+cpu` and `torch_npu==2.9.0` together. `pip install -e .[...]`
from a stock PyPI + Python 3.11 host fails at dependency resolution (no `torch==2.9.0`
wheel for that environment). For dev/test workflows, install the optional extras from
inside the base image:

```bash
pip install -e .[dev,video]   # pytest + av + decord for video-decode tests
```

Other extras: `[inf]` (vllm-ascend), `[quant]` (msmodelslim).

## Optional runtimes

The orchestrator (`scripts/render_train_rl_cpu.py`) has three pluggable
backend axes that select which runtime library actually drives each phase.
The **default** value of each flag is the reference path; the non-default
values are dispatchable end-to-end today and recorded in the JSON summary
under `summary["inference_engine_dispatched"]` /
`summary["policy_bootstrap"]["rlinf_*"]` so downstream consumers can audit
which engine actually served the run (see also
[§9](docs/roadmap/2026-08-14-opensource-extensions-loop-todo.md)).

| Flag | Choices (default **bold**) | Status |
|---|---|---|
| `--inference-backend` | **`torch`**, `vllm`, `sglang`, `vllm_omni` | ✅ shipped — `torch` is the reference path; `vllm`/`sglang` reach the vLLM-Ascend / SGLang worker on Atlas and record a structured dispatch fingerprint on host (Atlas §9.3 P2); `vllm_omni` is the unified vLLM-Omni stand-in (ORT encoder + ORT diffusion), host-reachable today via `pip install -e .[vllm_omni]` and Atlas-reachable via `onnxruntime-ascend`. |
| `--rl-backend` | **`self_ppo`**, `verl`, `rlinf` | ✅ shipped — `self_ppo` is the in-process PPO; `rlinf` runs through the RLinfRunner adapter (host smoke: mock factory; Atlas: real RLinf v0.3+); `verl` reserved hook (M1-T5 §9.3 P2). Today's surface: `--rl-backend rlinf --rlinf-runner-factory <callable>`. |
| `--frame-gen-backend` | **`none`**, `mindie_sd` | ✅ registered — `none` shipped; `mindie_sd` returns the CPU `LatentToFramePipeline` stub (random projection) so the orchestrator's diffusion seam still type-checks today; real `.om` MindIE-SD compile lands in M3-T8 (§9.4 P3, W44+). |

Examples:

```bash
# Default — runs anywhere (CPU / Atlas / container) today.
python scripts/render_train_rl_cpu.py

# Atlas / container — vLLM-Ascend worker (VJEPA2NPUWorker). On Atlas
# boxes with the vllm-ascend wheel installed the orchestrator records
# ``inference_engine_dispatched.available=true, engine_type=VJEPA2NPUWorker``;
# on host smoke the missing wheel surfaces as a structured ImportError.
python scripts/render_train_rl_cpu.py --inference-backend vllm

# Host-reachable vLLM-Omni stand-in (VLLMOmniPipeline = ORT encoder +
# ORT diffusion). Dispatchable today on any box with onnxruntime
# installed; on Atlas the providers auto-swap to AscendExecutionProvider.
# Summary key ``inference_engine_dispatched`` carries the fingerprint.
python scripts/render_train_rl_cpu.py --inference-backend vllm_omni

# RLinf via mock factory (host smoke, no Ray cluster needed).
# Summary keys ``policy_bootstrap.rlinf_framework=rlinf`` and
# ``policy_bootstrap.rlinf_status=completed`` prove the dispatch.
python scripts/render_train_rl_cpu.py \
    --rl-backend rlinf \
    --rlinf-runner-factory worldfoundry_rl.rlinf_mock_factory:build_runner

# Returns the CPU LatentToFramePipeline stub today; real MindIE-SD
# compile lands in M3-T8. The orchestrator's diffusion seam still
# type-checks so the swap-point is exercisable end-to-end.
python scripts/render_train_rl_cpu.py --frame-gen-backend mindie_sd
```

Recommended execution order is in
[§9.6](docs/roadmap/2026-08-14-opensource-extensions-loop-todo.md): ship
P0 (this week, host-only) → P1 (next week, container images) →
P2 (Atlas card 7) → P3 (MindIE-SD real link) → P4 (annual vllm-omni
re-review).

See:
- `docs/superpowers/specs/2026-07-27-ascend-world-model-design.md` — design spec
- `docs/superpowers/plans/2026-07-27-worldforge-phase1.md` — Phase-1 implementation plan
- `docs/user/quickstart.md` — end-user quickstart (added in Task 1)
- `docs/dev/environment.md` — dev environment setup (added in Task 1)
- `docs/dev/cpu-rti-quickstart.md` — **CPU render → train → inference → policy loop (no NPU, no HF 1.2B download)** — runs end-to-end in ~3.5 s on a single CPU thread

## CPU render-train-inference loop (RTI)

If you don't have an Atlas box handy, the **render → train → inference →
policy** loop runs end-to-end on a stock CPU in seconds, with no HF
V-JEPA 2 1.2B download and no mujoco/dm_control dependency. It's a
self-contained RLinf-shaped smoke that proves the loop works:

```bash
bash scripts/render_train_rl_cpu.sh
```

This runs 5 phases on a single CPU thread in ~3.5 s:

1. **Render** — cv2 → real `.mp4` clips + CSV manifest
2. **Train** — TinyViT encoder + action-conditioned predictor + pixel decoder
3. **Inference** — pixel + latent cosine probes
4. **PPO** — multi-epoch PPO update from imagined rollouts
5. **Outer loop** — world-model ↔ PPO alternation (RLinf pattern)

Output: `bench_results/render_train_rl/render_train_rl.json`. See
`docs/dev/cpu-rti-quickstart.md` for flags, expected metrics, and how
to interpret them.