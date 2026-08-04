"""CLI: python -m worldfoundry_train.bench --profile <p> --steps N --warmup W --tag <t>"""
from __future__ import annotations

import argparse
from pathlib import Path

from worldfoundry_train.bench.runner import run_bench


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="WorldFoundry benchmark runner")
    p.add_argument("--profile", required=True, choices=["train", "train-mmspeed"])
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--tag", required=True)
    p.add_argument("--output-dir", type=Path, default=Path("bench_results"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out = run_bench(
        profile=args.profile,
        steps=args.steps,
        warmup=args.warmup,
        tag=args.tag,
        output_dir=args.output_dir,
    )
    print(f"[bench] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
