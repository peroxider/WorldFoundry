"""WorldFoundry inference FastAPI app."""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI

from worldfoundry_common.npu_env import is_npu_available
from worldfoundry_inf.api import encode, plan, predict
from worldfoundry_inf.scheduler import AdapterRouter
import worldfoundry_inf.vjepa2_worker as vjepa2_worker
from worldfoundry_inf.security import HmacTokenAuth
import worldfoundry_inf.vjepa2_worker as vjepa2_worker


def create_app(model_path: Path | str, secret: str | None = None) -> FastAPI:
    secret = secret or os.environ.get("WORLDFORGE_HMAC_SECRET", "change-me")
    app = FastAPI(title="WorldFoundry Inference", version="0.1.0-phase1")

    app.state.auth = HmacTokenAuth(secret=secret)
    app.state.router = AdapterRouter()
    app.state.router.register("default", vjepa2_worker.VJEPA2Worker(model_path))

    for r in (encode.router, predict.router, plan.router):
        app.include_router(r, prefix="/v1/world")

    @app.get("/health")
    async def health() -> dict:
        return {
            "ok": True,
            "status": "live",
            "npu_available": is_npu_available(),
            "adapters": app.state.router.available_adapters(),
        }

    return app


def main() -> int:
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("models/vjepa2-vitl-1.2b"))
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    app = create_app(args.model)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
