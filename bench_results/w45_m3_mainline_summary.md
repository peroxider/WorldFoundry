# W45 — M3 mainline summary (seam-level closure)

## Context

C-T8 requires a per-mainline Atlas card 7 smoke and `bench_results/wXX_<topic>/summary.md`
for each of M1 / M2 / M3. On 2026-08-16 the status is:

- **M1 mainline** ✅ closed end-to-end on Atlas — `bench_results/w25_atlas_vitl_orchestrator/summary.md`
- **M2 mainline** ⏸ PARTIALLY closed (HTTP-OpenAI-API probe only) — `bench_results/w43_vllm_ascend_e2e/summary.md`
- **M3 mainline** ⏸ seam-level only — **this document**

The M3 mainline is the **latent-to-frame diffusion / DiT pipeline** that
sits between the world model and the policy / video collector in the
orchestrator. Roadmap §9.4 P3 owns its real-NPU rollout (M3-T4 +
M3-T1 subdirs + M3-T5 W8A8 quant + M3-T6 eval + M3-T8 5-stage
integration). On 2026-08-16 only the seam-level scaffolding is in place.

## What was delivered (commit `243b6d4`)

| Module | Purpose |
|---|---|
| `worldfoundry_inf/diffusion/pipeline.py` | `LatentToFramePipeline` ABC exposing `encode_latents / decode_latents / rollout` surface |
| `worldfoundry_inf/diffusion/backends/cpu.py` | `CPUPixelDecoderPipeline` — v1 stub: deterministic random projection from latent to `(H, W, 3)` uint8; always buildable on CPU-only hosts |
| `worldfoundry_inf/diffusion/backends/npu.py` | `NPUDiffusionPipeline` — v2 stub (M3-T4) |
| `worldfoundry_common/backends/inference.py` | `_mindie_sd_available()` + `_build_mindie_sd()` — `mindie_sd` registry entry returns `CPUPixelDecoderPipeline` for v1; `vllm_omni` returns `NotImplementedError` with the W44+ hint |
| `worldfoundry_inf/tests/test_diffusion_pipeline.py` | 16 unit tests covering `LatentToFramePipeline` ABC + `CPUPixelDecoderPipeline` determinism + `_mindie_sd_available()` / `_build_mindie_sd()` wiring |

All 16 unit tests pass in the host pytest run (commit `243b6d4`: 128 total
passing across adapters + backends + inf-parity/diffusion + train-shim/
mask/CLI).

## What was NOT delivered (P3 / roadmap §9.4)

- **M3-T4** real MindIE-SD compile — host-side code lands 2026-08-17 (this
  commit): `worldfoundry_inf/compile/mindie_sd.py` now dispatches
  `compile_to_om` to `mindie.sd.tools.compile(...)` (preferred) or `atc`
  CLI subprocess (fallback) or JSON stub (backward compat). 19 unit
  tests green. Atlas-side run blocked on the air-gap (no `mindie-sd`
  wheel in uv cache + no outbound DNS); runbook at
  `docs/dev/atlas-m3-t4-runbook.md` covers sideload path. **M3-T4
  partial closure status:** code ✅ + tests ✅ + registry probe ✅ +
  Atlas runbook ✅; Atlas end-to-end ⏸ blocked on wheel sideload.
- **M3-T1 subdirs** `worldfoundry_inf/diffusion/{vae,transformer,scheduler}`
  real implementations: blocked on M3-T4 Atlas run.
- **M3-T5** W8A8 quant + accuracy regression: blocked on M3-T4.
- **M3-T6** temporal rollout eval (CLIP / SSIM / PSNR): blocked on
  M3-T4.
- **M3-T8** 5-stage integration (`--frame-gen-backend mindie_sd`): blocked
  on M3-T4.
- **Atlas e2e**: blocked on M3-T4 (and on the same Atlas air-gap that
  blocks M2-T5 / M1-T4 / M1-T5 if compile requires external toolchain
  wheels).

## M3-T4 partial closure — host-side code lands (2026-08-17)

Commit `pending` lands the real `compile_to_om` implementation:

### What changed

- `worldfoundry_inf/compile/mindie_sd.py` (rewrite, ~250 LOC) —
  three-way dispatch in priority order: `mindie.sd.tools.compile` →
  `atc` CLI subprocess → JSON stub. New `prefer` arg pins the path.
- `worldfoundry_common/backends/inference.py` — adds
  `_mindie_sd_v2_available()` probe (returns True iff either
  `mindie_sd` wheel importable OR `atc` CLI on PATH); updates
  `mindie_sd` registry hint to reference `compile_to_om` real path.
- `worldfoundry_inf/tests/test_mindie_sd.py` — extended from 2 to 19
  tests: probe + dispatch + stub-payload schema + ATC subprocess
  (with mocked `shutil.which` + `subprocess.run`) + mindie_sd path +
  unsupported-hardware + provider_summary diagnostic.

### Acceptance gates

| Gate | Criterion | Result |
|---|---|---|
| Dual-track dispatch | `compile_path()` returns `mindie_sd` / `atc` / `stub` based on env | ✅ 3 probe tests + 3 explicit-prefer tests |
| ATC subprocess call | `atc --model ... --framework 5 --output ... --soc-version Ascend910B4` | ✅ subprocess test asserts full cmd shape |
| ATC failure surfaces | `OMCompileError` with stderr excerpt on non-zero rc | ✅ stderr-tail test |
| ATC timeout | `OMCompileError` on `subprocess.TimeoutExpired` | ✅ timeout test |
| mindie.sd.tools.compile path | Called with right kwargs + writes returned artifact | ✅ mocked-module test |
| mindie_sd internal failure | `OMCompileError` with exception repr | ✅ failure test |
| 910B vs 910C soc-version | `Ascend910B4` vs `Ascend910C` flag mapping | ✅ per-hardware test |
| Stub backward compat | `phase == "phase1-stub"` key preserved | ✅ schema-preserved test |
| Stub additive fields | New `compile_path` + `stub_reason` keys added | ✅ schema test |
| Stub checksum determinism | Same source content → same sha256 | ✅ determinism test |
| Provider diagnostic | `provider_summary()` returns shape-stable dict | ✅ diagnostic test |
| Registry wiring | `_mindie_sd_v2_available` exposed + hint updated | ✅ smoke-tested |
| **Atlas end-to-end** | Real `.om` artifact on Atlas card 7 | ⏸ blocked on Atlas air-gap + wheel sideload (runbook covers path) |

### What is intentionally NOT in this commit

- **NPU pipeline runtime** — `worldfoundry_inf/diffusion/backends/npu.py`
  still raises `PipelineNotAvailableError`. The next session replaces
  this with a real `acl.runtime` / `mslite` loader once an Atlas run
  produces a working `.om`.
- **W8A8 quant (M3-T5)** — requires `worldfoundry_inf/quantize_backends`
  W8A8 quantizer to land (separate work).
- **CLIP / SSIM / PSNR eval (M3-T6)** — requires a real decoder
  producing pixel frames on Atlas.
- **5-stage orchestrator integration (M3-T8)** — `--frame-gen-backend
  mindie_sd` wiring into `scripts/render_train_rl_cpu.py`.

## What the orchestrator currently does

`scripts/render_train_rl_cpu.py` accepts `--frame-gen-backend mindie_sd`
but does **not** invoke it yet in any phase (the v1 stub isn't wired into
the 5-phase pipeline — only the registry probe exists). W2.25 Gate 3 ran
without any diffusion-stage call. The seam-level wiring is in place so a
future M3-T4 closure can drop the real pipeline into
`scripts/render_train_rl_cpu.py` without touching the registry.

## Acceptance gate status

| Gate | Criterion | Result |
|---|---|---|
| `LatentToFramePipeline` ABC | `encode_latents / decode_latents / rollout` surface defined | ✅ in source (`243b6d4`) |
| `CPUPixelDecoderPipeline` v1 | Deterministic random-projection stub, always buildable | ✅ in source |
| `mindie_sd` registry entry | `_mindie_sd_available() == True`, `_build_mindie_sd()` returns `CPUPixelDecoderPipeline` | ✅ |
| `vllm_omni` registry entry | Stub raises `NotImplementedError` with W44+ hint | ✅ |
| Unit tests | 16 diffusion-pipeline tests pass | ✅ (pytest 128 total) |
| Real MindIE-SD compile (M3-T4) | `.om` produced; shape parity to stub | ❌ blocked on W44+ upstream chain |
| `--frame-gen-backend mindie_sd` 5-stage orchestrator | `summary.frame_gen.backend == "mindie_sd"`; diffusion-stage time < 40% of total | ❌ blocked on M3-T4 (P3) |
| Atlas NPU smoke | nonzero-pixel frames from real pipeline | ❌ blocked on M3-T4 (P3) |

## How to apply / next steps

- P3 work (M3-T4 → M3-T1 subdirs → M3-T5 → M3-T6 → M3-T8) is owned
  upstream on the Atlas MindIE-SD chain; no work can land in the
  WorldFoundry repo until `compile_to_om` is callable from a host with
  the chain installed.
- Until then, M3 mainline stays at "seam-level + 16 unit tests pass +
  registry entry returns CPU stub". The orchestrator's
  `--frame-gen-backend mindie_sd` flag is accepted but inert.
- Re-run the 16 unit tests in any session to confirm seam health:
  `uv run --python .venv/bin/python -m pytest
  worldfoundry_inf/tests/test_diffusion_pipeline.py -q`.
- The C-T8 per-mainline table is complete on this commit: M1 ✅ (W2.25),
  M2 ⏸ partial (W43), M3 ⏸ seam-only (W45 this doc).

## Cross-references

- `worldfoundry_inf/diffusion/pipeline.py` — `LatentToFramePipeline` ABC
- `worldfoundry_inf/diffusion/backends/cpu.py` — `CPUPixelDecoderPipeline` v1
- `worldfoundry_inf/diffusion/backends/npu.py` — `NPUDiffusionPipeline` v2 stub
- `worldfoundry_common/backends/inference.py` — `mindie_sd` + `vllm_omni` registry
- `worldfoundry_inf/tests/test_diffusion_pipeline.py` — 16 unit tests
- `docs/roadmap/2026-08-14-opensource-extensions-loop-todo.md` §9.4 — P3 list
- `docs/superpowers/specs/2026-08-06-w40-frame-gen-adr.md` — W40 ADR
- `bench_results/w25_atlas_vitl_orchestrator/summary.md` — M1 mainline (closed)
- `bench_results/w43_vllm_ascend_e2e/summary.md` — M2 mainline (partial)