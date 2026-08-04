# Runbook (Phase-1)

## TL;DR

```bash
# 1. Build & launch container (one-time)
docker build -t worldfoundry-base:phase1 -f docker/worldfoundry-base/Dockerfile .

# 2. Smoke test on CPU
docker run --rm -v $(pwd):/workspace worldfoundry-base:phase1 \
    bash scripts/phase1_smoke.sh --cpu

# 3. Smoke test on NPU (real hardware)
docker run --rm --device /dev/davinci0:/dev/davinci0 \
    -v $(pwd):/workspace -v /data:/data \
    worldfoundry-base:phase1 \
    bash scripts/phase1_smoke.sh

# 4. AC finetuning (W7-W8)
docker run --rm --device /dev/davinci0:/dev/davinci0 \
    -v $(pwd):/workspace -v /data:/data \
    -e WORLDFORGE_MODEL=models/vjepa2-vitl-1.2b \
    -e WORLDFORGE_DROID_MANIFEST=/data/droid/manifest.csv \
    worldfoundry-base:phase1 \
    bash scripts/phase1_ac_repro.sh
```

## Common pitfalls

### "LoRA matched 0 modules"
Per Global Constraint G4 + spec §3.2: `model.print_trainable_parameters()` returns 0.
Fix: inspect `base_model.named_modules()`; the V-JEPA 2 actual module names
may differ from the candidate list. Update `worldfoundry_train/model/lora_wrapper.py`
`DEFAULT_LORA_CANDIDATES`.

### "version mismatch" at startup
You installed a package version not on the locked matrix. Either:
- Reinstall with the pinned versions in `pyproject.toml`, or
- Pass `--strict-versions` to fail fast on the next run.

### "Adapter not registered"
`AdapterNotFoundError` from `worldfoundry_inf.scheduler.AdapterRouter.route`.
The inference service does not silently default to a fallback adapter
(per spec §4.7). Restart the server with the missing adapter loaded, or
fix the client to use a registered adapter name.

### DVPP decoder unavailable
`_dvpp_available()` returns False on non-Ascend hosts. The build_video_source
constructor will silently fall back to decord; this is expected on CPU.

## Acceptance criteria check

See spec §9. For Phase-1 (Tasks 1-21), the relevant ones are:
- #1 (W1-W2 train) — verified by `pytest worldfoundry_train/tests` + smoke
- #2 (W3-W4 inference) — verified by `pytest worldfoundry_inf/tests` + curl /v1/world/encode
- #3 (W5-W6 smoke) — verified by `bash scripts/phase1_smoke.sh`
- #4 (W7-W8 AC finetuning) — verified by `bash scripts/phase1_ac_repro.sh`

## Next phase handoff

Phase-1 deliverables are gated at the final commit of Task 21 on the
`phase1/implementation` branch. Phase-2 (MindSpeed, W4A8, RL loop, custom env)
is issued as a separate implementation plan after Phase-1 acceptance.