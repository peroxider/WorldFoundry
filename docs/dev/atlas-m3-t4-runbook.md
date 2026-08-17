# Atlas M3-T4 runbook — sideload mindie-sd + verify real compile_to_om

> **Status (2026-08-17):** kept **local-only** (uncommitted) per chad's
> narrow-scope Atlas-runbook preference. Sister file
> `atlas-booking-runbook-w35-p2.md` covers W35.1 + M2-T5.
> Re-evaluate commit scope after the M3-T4 Atlas run produces real
> evidence in `bench_results/`.

## 0. Why this runbook exists

M3-T4 (real MindIE-SD compile) closes when `compile_to_om()` runs on
Atlas and writes a real Ascend `.om` artifact (not the JSON stub).
Host-side code is in (`worldfoundry_inf/compile/mindie_sd.py`,
2026-08-17 commit, 19 unit tests green) with dual-track dispatch:

1. `mindie.sd.tools.compile(...)` — preferred path
2. `atc` CLI subprocess — fallback path
3. JSON stub — backward-compat fallback

Atlas side is blocked on the **Atlas air-gap** (zero outbound DNS —
NXDOMAIN for aliyun, tsinghua, hf-mirror, huggingface.co,
files.pythonhosted.org all tried on 2026-08-16; uv wheel cache holds
metadata-only `.msgpack` for vllm / vllm-ascend / verl / rlinf and
**no `mindie-sd` artifacts at all**). Two unlock paths exist:

- **Path A (this runbook):** sideload `mindie-sd-3.0.0-cp311-*.whl`
  tarball from a host with egress, copy into Atlas, `uv pip install
  --offline`. Then verify `atc` on PATH inside
  `worldfoundry-npu-vllm:phase1` (or a fresh `worldfoundry-npu-mindie`
  image).
- **Path B:** Atlas gets outbound DNS / mirror access. Outside this
  runbook's scope.

## 1. Prerequisites

### 1.1 Connectivity (read this first)

Per `project_atlas_npu_server.md` "WSL2 vs Windows host reachability":
**I (Claude Code in WSL2) cannot drive Atlas SSH directly.** Confirmed
2026-08-17 by three TCP probes (3-second timeout each):

| Path | Result |
|---|---|
| `127.0.0.1:2222` (mirrored-mode loopback) | Connection refused |
| `172.18.112.1:2222` (WSL2 NAT IP) | Timeout |
| `192.168.9.146:22` (direct Atlas) | Timeout |

**Only the Windows host terminal can SSH to Atlas.** chad's working
pattern: PowerShell / cmd → `ssh -p 2222 root@172.18.112.1`. I do not
paste credentials into commands — chad handles the password prompt.

### 1.2 Atlas multi-tenant guard (mandatory preflight)

Per memory "Atlas multi-tenant guard": Atlas is shared production.
**Always check `uptime` + `npu-smi info` before launching heavy
jobs.** Abort if `load > 10` or all NPU Hbm > 50%.

### 1.3 Card selection

Physical card 7 with `ASCEND_RT_VISIBLE_DEVICES=7` remaps to
container-local logical `npu:0`. Do not select index 7 inside Python.

### 1.4 UV environment (mandatory)

Use the project uv environment. Never `pip install` against host Python.

```bash
export PATH=/root/.local/bin:$PATH
cd /home/w30066257/WorldFoundry
source .venv/bin/activate   # OR: uv run --python .venv/bin/python …
```

## 2. Path A — sideload mindie-sd 3.0.0 wheel

### 2.1 Pre-stage the wheel on a host with egress

From any host with internet access (chad's Windows shell, an internal
mirror server, a CI runner):

```bash
mkdir -p /tmp/wheels && cd /tmp/wheels
uv pip download \
  --python 3.11 \
  --platform manylinux2014_aarch64 \
  --only-binary=:all: \
  --dest . \
  "mindie-sd==3.0.0" \
  "torch==2.9.0" \
  "torch_npu==7.3.0"
# Bundle into one tarball for transport
tar -czf mindie-sd-wheels.tar.gz .
```

Expected artifact: `mindie-sd-wheels.tar.gz` (~150–400 MB).

### 2.2 Transport into Atlas

From Windows terminal (chad pastes):

```powershell
scp -P 2222 C:\path\to\mindie-sd-wheels.tar.gz root@172.18.112.1:/tmp/
```

### 2.3 Extract + offline install (inside Atlas SSH session)

```bash
# In Windows terminal SSH session:
ssh -p 2222 root@172.18.112.1

# On Atlas:
mkdir -p /tmp/wheels && cd /tmp/wheels
tar -xzf /tmp/mindie-sd-wheels.tar.gz
ls -la | head -20  # sanity-check the wheels

export PATH=/root/.local/bin:$PATH
cd /home/w30066257/WorldFoundry

# Install into the project uv env -- DO NOT touch host Python
uv pip install --python .venv/bin/python --offline /tmp/wheels/*.whl 2>&1 | tee /tmp/mindie-sd-install.log
```

**Expected log tail:**

```
Resolved 14 packages in 12ms
Installed 14 packages in 45s
 + mindie-sd==3.0.0
 + torch==2.9.0
 + torch-npu==7.3.0
 + …
```

If `uv` reports "no compatible version" or wheel mismatch — **stop,
do not retry with different flags**. Capture `uv pip install --offline
--verbose /tmp/wheels/*.whl 2>&1 | tail -200`, paste back, and
diagnose root cause first.

### 2.4 Verify import

```bash
.venv/bin/python -c "
import mindie_sd
from mindie.sd.tools import compile as msd_compile
print('mindie_sd OK; compile =', msd_compile)
"
```

Expected: `mindie_sd OK; compile = <function compile at 0x…>`.

### 2.5 Verify `atc` on PATH

```bash
which atc && atc --version
```

If `atc` is missing inside the project uv environment, source the
Ascend toolkit:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
which atc   # expect /usr/local/Ascend/ascend-toolkit/latest/<…>/atc
```

## 3. Run compile_to_om smoke

### 3.1 Build a minimal quantized model directory

The compile path requires `config.json` + `quant_meta.json` +
`model.onnx` in the source directory. The repo already ships
`worldfoundry_inf/quantize_backends` — for the smoke we synthesize
the trio directly:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
import torch
from worldfoundry_train.model.tiny_jepa import TinyJEPAPredictor

src = Path("/tmp/w25_compile_src")
src.mkdir(parents=True, exist_ok=True)

# config.json -- minimal HF-style metadata
(src / "config.json").write_text('{"model_type":"tiny_jepa","hidden_size":64}')

# quant_meta.json -- W8A8 quant spec (presence-only check by compile_to_om)
(src / "quant_meta.json").write_text('{"quant_type":"w8a8","bits":8}')

# model.onnx -- export a tiny DiT-shaped decoder (mirrors onnx_diffusion.py)
from worldfoundry_inf.diffusion.backends.onnx_diffusion import (
    _TinyDiTDecoder, OnnxDiffusionConfig,
)
import io

cfg = OnnxDiffusionConfig()
decoder = _TinyDiTDecoder(cfg).eval()
dummy = torch.zeros(1, cfg.num_patches, cfg.latent_dim)
buf = io.BytesIO()
torch.onnx.export(decoder, (dummy,), buf,
    input_names=["latent"], output_names=["pixel_patches"],
    dynamic_axes={"latent": {0: "batch", 1: "tokens"},
                  "pixel_patches": {0: "batch", 1: "tokens"}},
    opset_version=17, do_constant_folding=True)
(src / "model.onnx").write_bytes(buf.getvalue())
print("wrote", src, "(", sum(p.stat().st_size for p in src.iterdir()), "bytes)")
PY
```

### 3.2 Run the compile (auto path — picks mindie_sd if wheel available)

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
from worldfoundry_inf.compile.mindie_sd import (
    compile_to_om, provider_summary,
)
import json

print("=== compile environment ===")
print(json.dumps(provider_summary(), indent=2))

out = Path("/tmp/w25_compile_out/model.om")
out.parent.mkdir(parents=True, exist_ok=True)
result = compile_to_om(Path("/tmp/w25_compile_src"), out)
print(f"compile_to_om returned: {result}")

# If result ends with .json it was the stub path -- paste provider_summary
# back to chad. If it ends with .om the real compile succeeded.
PY
```

Expected:

- **Success path:** `compile_to_om returned: /tmp/w25_compile_out/model.om`, file size ~kilobytes to megabytes depending on DiT depth.
- **Stub path (mindie_sd + atc both unavailable):** file is JSON, `phase: "phase1-stub"`. **Do not count as M3-T4 closure** — paste the `provider_summary` back; chad will need to diagnose which dep is missing.

### 3.3 Run compile via ATC explicitly (sanity-check the fallback)

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
from worldfoundry_inf.compile.mindie_sd import compile_to_om
out = Path("/tmp/w25_compile_atc/model.om")
out.parent.mkdir(parents=True, exist_ok=True)
result = compile_to_om(Path("/tmp/w25_compile_src"), out, prefer="atc")
print(f"atc compile: {result}")
PY
```

Expected: real `.om` binary if `atc` is on PATH and the DiT ONNX is
compatible. If ATC fails, capture stderr — common cause is
`soc-version` mismatch (`Ascend910B4` vs `Ascend910B3`).

## 4. Capture evidence

After a successful compile, persist a bench summary so the M3-T4
partial-closure status is recoverable for the next session:

```bash
mkdir -p /home/w30066257/WorldFoundry/bench_results/w46_m3_t4_compile

cat > /home/w30066257/WorldFoundry/bench_results/w46_m3_t4_compile/summary.md <<'MD'
# W46 — M3-T4 real compile smoke on Atlas card 7

## Inputs
- mindie-sd version: (run `pip show mindie-sd | grep Version`)
- atc version: (run `atc --version`)
- target_hardware: Atlas 800 A2 / 910B
- source dir: /tmp/w25_compile_src (config.json + quant_meta.json + model.onnx)
- output: /tmp/w25_compile_out/model.om (size = $(stat -c%s /tmp/w25_compile_out/model.om) bytes)

## Provider summary (paste verbatim)
\`\`\`json
$(.venv/bin/python -c "import json; from worldfoundry_inf.compile.mindie_sd import provider_summary; print(json.dumps(provider_summary(), indent=2))")
\`\`\`

## Compile result
- compile_path: (mindie_sd / atc / stub)
- output size: $(stat -c%s /tmp/w25_compile_out/model.om 2>/dev/null || echo MISSING) bytes
- rc: $(.venv/bin/python -c "from pathlib import Path; from worldfoundry_inf.compile.mindie_sd import compile_to_om; print(compile_to_om(Path('/tmp/w25_compile_src'), Path('/tmp/w25_compile_out/model.om')))")
MD

ls -la /home/w30066257/WorldFoundry/bench_results/w46_m3_t4_compile/
```

## 5. Cleanup

```bash
rm -rf /tmp/w25_compile_src /tmp/w25_compile_out /tmp/w25_compile_atc
rm -f /tmp/mindie-sd-wheels.tar.gz /tmp/mindie-sd-install.log
# Leave bench_results/w46_m3_t4_compile/ in place -- that's the evidence.
```

## 6. After the run

Paste back to the WSL2 Claude session:

1. The full stdout of step 3.2 (including the `provider_summary` JSON).
2. The contents of `bench_results/w46_m3_t4_compile/summary.md`.
3. The result of `npu-smi info` after the run (confirm no HBM leak).

If M3-T4 closes on Atlas, the next session updates
`worldfoundry_inf/diffusion/backends/npu.py` to actually load the
compiled `.om` (replacing the `PipelineNotAvailableError` stub) and
updates `bench_results/w45_m3_mainline_summary.md` to "✅ closed".

## 7. What is NOT in this runbook

- **W8A8 quant (M3-T5)**: separate task; depends on
  `worldfoundry_inf/quantize_backends` W8A8 quantizer landing. Run
  the cosine eval (`worldfoundry_inf/eval/quant_eval.py`) once both
  bf16 baseline + W8A8 quantized model are on disk.
- **CLIP / SSIM / PSNR eval (M3-T6)**: depends on M3-T4 + a real
  decoder producing pixel frames on Atlas.
- **5-stage orchestrator integration (M3-T8)**: depends on M3-T4 +
  M3-T6. Updates `--frame-gen-backend mindie_sd` to actually call
  the new pipeline.
- **W44+ upstream chain**: outside this runbook.

These land in the next session if M3-T4 closes here.
