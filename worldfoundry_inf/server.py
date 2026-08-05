"""WorldFoundry inference FastAPI app."""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI

from worldfoundry_common.npu_env import is_npu_available
from worldfoundry_inf.api import encode, plan, predict
from worldfoundry_inf.scheduler import AdapterRouter
from worldfoundry_inf.security import HmacTokenAuth
import worldfoundry_inf.vjepa2_worker as vjepa2_worker


def create_app(
    model_path: Path | str,
    secret: str | None = None,
    predictor_ckpt: Path | str | None = None,
    past_frames: int = 1,
    future_frames: int = 1,
    patch_size: int = 16,
    image_size: int = 224,
) -> FastAPI:
    secret = secret or os.environ.get("WORLDFORGE_HMAC_SECRET", "change-me")
    app = FastAPI(title="WorldFoundry Inference", version="0.2.0-w24")

    app.state.auth = HmacTokenAuth(secret=secret)
    app.state.router = AdapterRouter()
    app.state.router.register(
        "default",
        vjepa2_worker.VJEPA2Worker(
            model_path,
            predictor_ckpt=predictor_ckpt,
            past_frames=past_frames,
            future_frames=future_frames,
            patch_size=patch_size,
            image_size=image_size,
        ),
    )

    for r in (encode.router, predict.router, plan.router):
        app.include_router(r, prefix="/v1/world")

    @app.get("/health")
    async def health() -> dict:
        return {
            "ok": True,
            "status": "live",
            "npu_available": is_npu_available(),
            "adapters": app.state.router.available_adapters(),
            "temporal_mode_enabled": predictor_ckpt is not None,
        }

    return app


def main() -> int:
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("models/vjepa2-vitl-1.2b"))
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument(
        "--predictor-ckpt",
        type=Path,
        default=None,
        help="W22 JEPA training checkpoint (.pt) — enables temporal mode "
        "on /v1/world/predict. When omitted, the worker is masked-mode only.",
    )
    parser.add_argument(
        "--past-frames", type=int, default=1,
        help="K (past-frame count) for the temporal predictor's input grid.",
    )
    parser.add_argument(
        "--future-frames", type=int, default=1,
        help="H (future-frame count) for the temporal predictor's output grid.",
    )
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()

    app = create_app(
        args.model,
        predictor_ckpt=args.predictor_ckpt,
        past_frames=args.past_frames,
        future_frames=args.future_frames,
        patch_size=args.patch_size,
        image_size=args.image_size,
    )
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
