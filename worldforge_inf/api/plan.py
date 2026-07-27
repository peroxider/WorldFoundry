"""POST /v1/world/plan — V-JEPA 2-AC planning.

Per spec §4.3: input (video, actions); output (next_features, planned_actions).
"""
from __future__ import annotations

import base64

import numpy as np
import torch
from fastapi import APIRouter, Depends, HTTPException, Request

from worldforge_common.exceptions import AdapterNotFoundError
from worldforge_inf.security import HmacTokenAuth

router = APIRouter()


@router.post("/plan")
async def plan(request: Request, auth: HmacTokenAuth = Depends(lambda: None)) -> dict:
    auth = request.app.state.auth
    body_bytes = await request.body()
    _check_auth(request, auth, body_bytes)
    payload = await request.json()
    if "video" not in payload or "actions" not in payload:
        raise HTTPException(status_code=400, detail="missing 'video' or 'actions'")
    actions = list(payload["actions"])
    if not all(isinstance(a, int) for a in actions):
        raise HTTPException(status_code=400, detail="'actions' must be a list of integers")

    video_bytes = base64.b64decode(payload["video"])
    video = np.frombuffer(video_bytes, dtype=np.uint8).reshape(-1, 64, 64, 3)
    tensor = torch.from_numpy(video).float().permute(0, 3, 1, 2).unsqueeze(0)

    adapter_name = payload.get("adapter", "default")
    try:
        worker = request.app.state.router.route(adapter_name)
    except AdapterNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"adapter {adapter_name!r} not registered") from exc
    next_feats, planned = worker.plan(tensor, actions)
    return {"next_features": next_feats.tolist(), "planned_actions": planned}


def _check_auth(request: Request, auth: HmacTokenAuth, body_bytes: bytes) -> None:
    hdr = request.headers.get("Authorization", "")
    ts = request.headers.get("X-WorldForge-Timestamp", "")
    if not hdr.startswith("HMAC "):
        raise HTTPException(status_code=401, detail="missing or malformed Authorization header")
    sig = hdr[len("HMAC "):]
    if not auth.verify(request.method, request.url.path, body_bytes, ts, sig):
        raise HTTPException(status_code=401, detail="HMAC verification failed")
