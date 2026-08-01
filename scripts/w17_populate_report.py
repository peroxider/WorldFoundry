"""Populate docs/superpowers/specs/W11_W12_QUANT_REPORT.md from measured values.

Reads:
  - <output_dir>/quant_meta.json (msmodelslim_version, calibration_samples, etc.)
  - bench_results/w17_w8a8_eval.json (aggregate_cosine, per_sample_cosine, passed)

Substitutes bracketed field markers in the report template and writes the
final report. Determines verdict from cosine >= 0.98.

Usage:
    python scripts/w17_populate_report.py \\
        --quant-meta models/vjepa2-vitl-1.2b-w8a8/quant_meta.json \\
        --eval-report bench_results/w17_w8a8_eval.json \\
        --output docs/superpowers/specs/W11_W12_QUANT_REPORT.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPORT_TEMPLATE = Path("docs/superpowers/specs/W11_W12_QUANT_REPORT.md")
FIELD_OM_SIZE = "[W17-MEASURED: OM size]"
FIELD_QUANT_WALL = "[W17-MEASURED: quantize wall time]"
FIELD_COSINE = "[W17-MEASURED: aggregate cosine]"
FIELD_VERDICT = "[W17-MEASURED: GO/NO-GO verdict]"


def _load_json(path: Path) -> dict:
    if not path.is_file():
        print(f"ERROR: {path} missing", file=sys.stderr)
        sys.exit(2)
    return json.loads(path.read_text())


def _compute_om_size(meta_dir: Path) -> str:
    bf16_bytes = sum(p.stat().st_size for p in meta_dir.glob("*.safetensors"))
    bf16_bytes += sum(p.stat().st_size for p in meta_dir.glob("*.bin"))
    om_files = list(meta_dir.parent.glob("*.om")) if meta_dir.parent.exists() else []
    if not om_files:
        return "TBD (no .om file)"
    om_bytes = sum(p.stat().st_size for p in om_files)
    ratio = om_bytes / max(bf16_bytes, 1)
    return f"{om_bytes / 1e6:.1f}MB OM ({ratio:.1%} of {bf16_bytes / 1e6:.1f}MB bf16)"


def populate(quant_meta_path: Path, eval_report_path: Path,
             template_path: Path, output_path: Path) -> None:
    meta = _load_json(quant_meta_path)
    eval_report = _load_json(eval_report_path)
    template = template_path.read_text()

    om_size = _compute_om_size(quant_meta_path.parent)
    cosine = eval_report["aggregate_cosine"]
    verdict = "GO" if eval_report["passed"] else "NO-GO"

    populated = template
    populated = populated.replace(FIELD_OM_SIZE, om_size)
    populated = populated.replace(FIELD_QUANT_WALL,
                                   meta.get("quantize_wall_seconds", "TBD"))
    populated = populated.replace(FIELD_COSINE, f"{cosine:.4f}")
    populated = populated.replace(FIELD_VERDICT,
                                   f"**{verdict}** (cosine {cosine:.4f} "
                                   f"vs threshold {eval_report['threshold']:.2f})")

    output_path.write_text(populated)
    print(f"Wrote {output_path}: verdict={verdict}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--quant-meta", type=Path, required=True)
    p.add_argument("--eval-report", type=Path, required=True)
    p.add_argument("--template", type=Path, default=REPORT_TEMPLATE)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    populate(args.quant_meta, args.eval_report, args.template, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
