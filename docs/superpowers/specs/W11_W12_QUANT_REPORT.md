# W11-W12 Quantization Report (template)

**Status**: Template - populated after real 8xNPU hardware run.

## Environment

- Hardware: Atlas 800 A2 / 910B x 8 (production target)
- Software: CANN 8.5.1, torch_npu 2.9.0, msmodelslim <version>
- Model: facebook/vjepa2-vitl-1.2b (bf16 baseline)
- Calibration: VideoMix22M 512-sample subset, seed=42
- Eval holdout: VideoMix22M 100-sample subset, seed=123

## Results

| Metric | Target | Measured |
|--------|--------|----------|
| OM size vs bf16 baseline | ~50% compression | [W17-MEASURED: OM size] |
| Downstream feature cosine >= 0.98 | yes | [W17-MEASURED: aggregate cosine] |
| Calibration wall time | <10 min on 8xNPU | TBD |
| Quantize wall time | <30 min on 8xNPU | [W17-MEASURED: quantize wall time] |

## Verdict

- **[W17-MEASURED: GO/NO-GO verdict]** (cosine ≥ 0.98 → GO; < 0.98 → NO-GO)
- **NO-GO** otherwise (deferred to future sub-project)

## Artifacts

- Quantized model dir: `models/vjepa2-vitl-1.2b-w8a8/`
- Eval report JSON: `bench_results/w11_w12_quant_eval.json`
- Logs: `bench_results/w11_w12_quant.log`
