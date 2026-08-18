# W46 — M3-T4 real compile smoke on Atlas card 7 (CLOSED 2026-08-18)

## What closed
compile_to_om() produced a real Ascend .om artifact (NOT the JSON stub)
on Atlas via the atc fallback path (mindie-sd wheel is not on aliyun/pypi).

## Provider summary


## Code fix landed (this closure)
worldfoundry_inf/compile/mindie_sd.py:
- _onnx_input_shape(): materialize ATC --input_shape from ONNX graph
  inputs (dynamic dims -> 1). torch.onnx.export with dynamic_axes makes
  atc fail with "Value NULL for parameter --input_shape".
- _compile_via_atc(): pass --input_shape; handle ATC appending .om to
  --output (model.om -> model.om.om) by renaming to the requested path.

## Compile result
- compile_path: atc (mindie_sd unavailable: no wheel on public index)
- atc binary: /usr/local/Ascend/ascend-toolkit/latest/bin/atc
- source dir: /tmp/w25_compile_src (config.json + quant_meta.json + model.onnx, 203631 bytes)
- output: /tmp/w25_compile_out/model.om (size = 225249 bytes, suffix .om)
- explicit prefer=atc: /tmp/w25_compile_atc/model.om (same size)
- soc-version: Ascend910B4

## Acceptance gate status
| Gate | Criterion | Result |
|---|---|---|
| Real .om artifact | compile_to_om returns path ending .om, non-JSON | ✅ 225249 bytes |
| atc fallback path | atc CLI compiles ONNX on Atlas | ✅ rc=0 |
| mindie_sd preferred path | mindie.sd.tools.compile | ⏸ wheel not on public index (aliyun/pypi); atc is the active path |
