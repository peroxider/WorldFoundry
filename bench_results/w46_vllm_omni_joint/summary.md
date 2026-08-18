# W46 — vLLM-Omni joint eval (M3-T7 + M2-T6) on Atlas card 7 (P4 CLOSED)

> Date: 2026-08-18 · Container: `worldfoundry-franka:npu-cann9`
> (`m1t4-rl`, privileged, `/workspace` = project mount) · NPU: card 7
> (`ASCEND_RT_VISIBLE_DEVICES=7` → container-local `npu:0`)
> · `HF_ENDPOINT=https://hf-mirror.com`

## What closed

**P4 completion criterion** ("vllm-omni 公开后一次性把 M3-T7 + M2-T6 跑完，
写入 `bench_results/wXX_vllm_omni_joint/`") — the public `vllm-omni`
wheel is now released (0.18.0 aligns with the container's vllm 0.18.0),
and the joint eval ran end-to-end on Atlas:

| Axis | Result |
|---|---|
| **M3-T7** (编排) | Omni output component resolved to **`RealVLLMOmniPipeline`** (kind=`real_vllm_omni`, `vllm_omni_version=0.18.0`) |
| **M2-T6** (adapter) | `vllm_version=0.18.0+empty`, `llm_class_import_ok=true`, `llm_class_path=vllm.entrypoints.llm.LLM` |
| **M3-T6** (metrics) | `ssim/psnr/clip: null` — RealVLLMOmniPipeline has no VAE reconstruct path (recorded honestly) |
| rollout smoke | `(1, 2, 3, 64, 64)` through the Omni component surface ✅ |

## Code landed (P4 close)

| File | Change |
|---|---|
| `worldfoundry_inf/omni/output.py` + `__init__.py` | **new** — `OmniOutputComponent` (LatentToFramePipeline surface + provider fingerprint) + `build_omni_output` priority dispatch: RealVLLMOmniPipeline → VAEDiffusionPipeline → ORT stand-in; `vllm_omni` version probe |
| `scripts/eval_vllm_omni_joint.py` | **new** — joint eval entry: component build → adapter evidence → frame metrics → rollout smoke → `joint_eval.json` + `summary.md` |
| `worldfoundry_inf/_vllm_ascend_plugin.py` | `register()` returns `None` — vLLM 0.18 raises `RuntimeError: Only one platform plugin can be activated` when the project plugin + vllm-ascend's own `ascend` plugin both return a class path; vllm-ascend now owns activation |
| `worldfoundry_inf/onnx_vllm_worker.py` | `encode()` runs ORT **per frame** (batch=1) — the exported ViT's `gemm_input_reshape` node is batch-1-only on torch 2.9 export; batching (B×T) crashed it on Atlas |
| tests | `test_omni_output.py` (7) + `test_eval_vllm_omni_joint.py` (4) — component dispatch, shapes, joint artifacts; plugin test updated to `register() is None` |

## Container setup

```
pip install --no-deps vllm-omni==0.18.0   # aligns with vllm 0.18.0 (NOT 0.26)
pip install --no-deps aenum prettytable janus   # iterative dep fill
```

`import vllm_omni` OK; platform plugin `ascend` activated (single).

## Result (`bench_results/w46_vllm_omni_joint/joint_eval.json`)

```json
{
  "m3_t7": {"pipeline_type": "RealVLLMOmniPipeline", "pipeline_kind": "real_vllm_omni",
            "vllm_omni_version": "0.18.0", "extras": {"dispatch": "real_vllm", "vllm_version": "0.18.0+empty"}},
  "m2_t6_adapter": {"vllm_version": "0.18.0+empty", "llm_class_import_ok": true,
                    "llm_class_path": "vllm.entrypoints.llm.LLM"},
  "frame_metrics": {"ssim": null, "psnr_db": null, "clip_score": null, "n_frames": 8,
                    "note": "pipeline has no VAE reconstruct path; metrics skipped"},
  "rollout_shape": [1, 2, 3, 64, 64]
}
```

## Honest note

- **frame_metrics null**: `RealVLLMOmniPipeline` (the resolved pipeline)
  has no VAE `reconstruct` path, so the M3-T6 frame-quality section
  records null (honest skip) instead of fabricating numbers. The M3-T6
  SSIM/CLIP gate itself was closed earlier via the trained-VAE path
  (`bench_results/w46_m3_t6_prod/`); this joint run's job is the
  **orchestration + adapter evidence** (M3-T7/M2-T6), which is complete.
- **`llm_class_import_ok=true`** but `vllm_version="0.18.0+empty"`: the
  real vllm package + `LLM` class are importable (the adapter seam is
  live); instantiating `LLM(model=...)` needs a model + accelerator
  (beyond the joint-eval smoke).
- The `_vllm_ascend_plugin` + `OnnxVJEPAWorker.encode` fixes are
  **real bug fixes** that were required for vllm-omni to load and run
  on Atlas (double platform activation; batch-1-only ONNX export).

## P4 gate status

| Gate | Criterion | Result |
|---|---|---|
| vllm-omni public wheel | released + installed (0.18.0, aligned) | ✅ |
| M3-T7 Omni output component | `build_omni_output` → RealVLLMOmniPipeline | ✅ |
| M2-T6 adapter evidence | vllm version + LLM import probe | ✅ |
| Joint artifacts in `bench_results/` | `w46_vllm_omni_joint/{joint_eval.json,summary.md}` | ✅ |
| M3-T6 frame-quality gate | (closed separately via trained VAE) | ⏸ not re-run here (no VAE path in resolved pipeline) |

## Cross-references

- `worldfoundry_inf/omni/output.py` — OmniOutputComponent (M3-T7)
- `scripts/eval_vllm_omni_joint.py` — joint eval entry (M2-T6)
- `worldfoundry_inf/_vllm_ascend_plugin.py` — platform double-activation fix
- `worldfoundry_inf/onnx_vllm_worker.py` — per-frame ORT encode fix
- `worldfoundry_inf/vllm_omni_real.py` — RealVLLMOmniPipeline (M2-T6 P1)
- `docs/roadmap/2026-08-14-opensource-extensions-loop-todo.md` §9.5 — P4
