"""POST /v1/world/predict — predict masked tokens.

Per spec §4.3: returns {predicted_features, loss_score} given a video and
mask_ratio. Uses V-JEPA 2's native masked prediction objective.
"""
from __future__ import annotations

import base64
import binascii

import numpy as np
import torch
from fastapi import APIRouter, Depends, HTTPException, Request

from worldforge_common.exceptions import AdapterNotFoundError
from worldforge_inf.security import HmacTokenAuth

router = APIRouter()


@router.post("/predict")
async def predict(request: Request, auth: HmacTokenAuth = Depends(lambda: None)) -> dict:
    auth = request.app.state.auth
    body_bytes = await request.body()
    _check_auth(request, auth, body_bytes)
    payload = await request.json()
    if "video" not in payload or "mask_ratio" not in payload:
        raise HTTPException(status_code=400, detail="missing 'video' or 'mask_ratio'")
    mask_ratio = float(payload["mask_ratio"])
    if not 0.0 <= mask_ratio <= 1.0:
        raise HTTPException(status_code=400, detail="mask_ratio must be in [0,1]")

    try:
        video_bytes = base64.b64decode(payload["video"], validate=True)
        video = np.frombuffer(video_bytes, dtype=np.uint8).reshape(-1, 64, 64, 3)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"invalid video payload: {exc}") from exc
    tensor = torch.from_numpy(video).float().permute(0, 3, 1, 2).unsqueeze(0)
    mask = torch.zeros(1, video.shape[0], dtype=torch.bool)
    n_masked = int(mask_ratio * video.shape[0])
    mask[0, :n_masked] = True

    adapter_name = payload.get("adapter", "default")
    try:
        worker = request.app.state.router.route(adapter_name)
    except AdapterNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"adapter {adapter_name!r} not registered") from exc
    pred = worker.predict(tensor, mask)
    return {"predicted_features": pred.tolist(), "loss_score": float((pred ** 2).mean())}


def _check_auth(request: Request, auth: HmacTokenAuth, body_bytes: bytes) -> None:
    hdr = request.headers.get("Authorization", "")
    ts = request.headers.get("X-WorldForge-Timestamp", "")
    if not hdr.startswith("HMAC "):
        raise HTTPException(status_code=401, detail="missing or malformed Authorization header")
    sig = hdr[len("HMAC "):]
    if not auth.verify(request.method, request.url.path, body_bytes, ts, sig):
        raise HTTPException(status_code=401, detail="HMAC verification failed")
