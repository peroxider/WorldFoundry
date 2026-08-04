"""POST /v1/world/encode — return dense features from a video."""
from __future__ import annotations

import base64

import numpy as np
import torch
from fastapi import APIRouter, Depends, HTTPException, Request

from worldfoundry_common.exceptions import AdapterNotFoundError
from worldfoundry_inf.security import HmacTokenAuth

router = APIRouter()


@router.post("/encode")
async def encode(request: Request, auth: HmacTokenAuth = Depends(lambda: None)) -> dict:
    auth = request.app.state.auth
    body_bytes = await request.body()
    _check_auth(request, auth, body_bytes)
    payload = await request.json()
    if "video" not in payload:
        raise HTTPException(status_code=400, detail="missing 'video' field")
    try:
        video_bytes = base64.b64decode(payload["video"])
        video = np.frombuffer(video_bytes, dtype=np.uint8).reshape(-1, 64, 64, 3)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid video payload: {exc}") from exc
    adapter_name = payload.get("adapter", "default")
    router_obj = request.app.state.router
    try:
        worker = router_obj.route(adapter_name)
    except AdapterNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"adapter {adapter_name!r} not registered") from exc
    tensor = torch.from_numpy(video).float().permute(0, 3, 1, 2).unsqueeze(0)
    features = worker.encode(tensor)
    return {"features": features.tolist(), "shape": list(features.shape)}


def _check_auth(request: Request, auth: HmacTokenAuth, body_bytes: bytes) -> None:
    """Validate HMAC headers. Raises HTTPException 401 on failure."""
    hdr = request.headers.get("Authorization", "")
    ts = request.headers.get("X-WorldFoundry-Timestamp", "")
    if not hdr.startswith("HMAC "):
        raise HTTPException(status_code=401, detail="missing or malformed Authorization header")
    sig = hdr[len("HMAC "):]
    if not auth.verify(request.method, request.url.path, body_bytes, ts, sig):
        raise HTTPException(status_code=401, detail="HMAC verification failed")
