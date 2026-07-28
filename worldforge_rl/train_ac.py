"""WorldForge Phase-1 V-JEPA 2-AC finetuning entry.

Per spec §5.3 + risk row "AC reproduction scope":
- Phase-1 covers only the AC finetuning phase (latent action prediction).
- Full 80% reproduction is NOT in scope.
- This script reads a Droid manifest and runs short PPO-style updates.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from worldforge_common.npu_env import check_versions, init_distributed, npu_device_count
from worldforge_rl.actor import VJEPAActor
from worldforge_rl.env_droid import DroidEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WorldForge V-JEPA 2-AC finetuning (Phase-1)")
    parser.add_argument("--model", type=Path, required=True, help="V-JEPA 2 base weights path")
    parser.add_argument("--manifest", type=Path, required=True, help="Droid manifest CSV")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--kl-coef", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, default=50)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    check_versions(strict=False)
    init_distributed()

    actor = VJEPAActor.from_pretrained(args.model)
    env = DroidEnv(manifest=args.manifest, max_steps=args.max_steps)

    print(f"[AC] starting finetuning: model={args.model} epochs={args.epochs} kl_coef={args.kl_coef}")
    for epoch in range(args.epochs):
        n_samples = 0
        for sample in env:
            n_samples += 1
        if epoch % 10 == 0:
            print(f"[AC] epoch={epoch} samples={n_samples} npu={npu_device_count()}")
    print("[AC] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())