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