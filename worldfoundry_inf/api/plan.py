"""POST /v1/world/plan — V-JEPA 2-AC planning (W33).

Per spec §4.3 + W33: input ``(video, actions, past_frames, future_frames)``
where ``actions`` is a 2-D float list ``(past_frames, action_dim)``. The
worker encodes the past K frames plus the per-frame actions with the W27
v2 ``ActionConditionedPredictor`` (sister to ``mode="action"`` on
``/v1/world/predict``) and returns the predicted future H frame latents.

The legacy Phase-1 ``worker.plan(...)`` passthrough (integer actions,
``features, list(actions)``) is retained on the worker for backward
compatibility with older callers; new clients should call
``/v1/world/predict {mode: "action", ...}`` directly or use the W33
``/v1/world/plan`` shape documented here.
"""
from __future__ import annotations

import base64
import binascii

import numpy as np
import torch
from fastapi import APIRouter, Depends, HTTPException, Request

from worldfoundry_common.exceptions import AdapterNotFoundError
from worldfoundry_inf.security import HmacTokenAuth

router = APIRouter()


@router.post("/plan")
async def plan(request: Request, auth: HmacTokenAuth = Depends(lambda: None)) -> dict:
    auth = request.app.state.auth
    body_bytes = await request.body()
    _check_auth(request, auth, body_bytes)
    payload = await request.json()
    if "video" not in payload or "actions" not in payload:
        raise HTTPException(status_code=400, detail="missing 'video' or 'actions'")
    if not isinstance(payload["actions"], list):
        raise HTTPException(
            status_code=400,
            detail="'actions' must be a list of action vectors (one per past "
            "frame)",
        )

    try:
        video_bytes = base64.b64decode(payload["video"], validate=True)
        video = np.frombuffer(video_bytes, dtype=np.uint8).reshape(-1, 64, 64, 3)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"invalid video payload: {exc}") from exc

    adapter_name = payload.get("adapter", "default")
    try:
        worker = request.app.state.router.route(adapter_name)
    except AdapterNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"adapter {adapter_name!r} not registered") from exc

    if not hasattr(worker, "predict_action_conditioned"):
        # Pre-W33 worker: legacy Phase-1 plan() passthrough. Accept a flat
        # integer list and return the original (features, planned_actions)
        # shape for backward compatibility with callers that predate W33.
        actions_raw = payload["actions"]
        if not all(
            isinstance(a, int) and not isinstance(a, bool) for a in actions_raw
        ):
            raise HTTPException(
                status_code=400,
                detail="worker does not support action-conditioned prediction; "
                "'actions' must be a flat list of integers for legacy plan()",
            )
        tensor = torch.from_numpy(video).float().permute(0, 3, 1, 2).unsqueeze(0)
        next_feats, planned = worker.plan(tensor, actions_raw)
        return {"next_features": next_feats.tolist(), "planned_actions": planned}

    # W33: action-conditioned predictor (2-D action vectors).
    if "past_frames" not in payload or "future_frames" not in payload:
        raise HTTPException(
            status_code=400,
            detail="plan endpoint requires 'past_frames' and 'future_frames'",
        )
    past_frames = int(payload["past_frames"])
    future_frames = int(payload["future_frames"])
    if past_frames < 1 or future_frames < 1:
        raise HTTPException(
            status_code=400,
            detail=f"past_frames and future_frames must be >= 1, got "
            f"past_frames={past_frames}, future_frames={future_frames}",
        )
    if past_frames + future_frames > video.shape[0]:
        raise HTTPException(
            status_code=400,
            detail=f"past_frames + future_frames ({past_frames + future_frames}) "
            f"exceeds video length ({video.shape[0]} frames)",
        )

    actions_raw = payload["actions"]
    if len(actions_raw) != past_frames:
        raise HTTPException(
            status_code=400,
            detail=f"'actions' length {len(actions_raw)} != past_frames "
            f"{past_frames}",
        )
    worker_action_dim = int(getattr(worker, "action_dim", 0))
    for i, vec in enumerate(actions_raw):
        if not isinstance(vec, list):
            raise HTTPException(
                status_code=400,
                detail=f"actions[{i}] must be a list of floats, got "
                f"{type(vec).__name__}",
            )
        if len(vec) != worker_action_dim:
            raise HTTPException(
                status_code=400,
                detail=f"actions[{i}] length {len(vec)} != worker's "
                f"action_dim {worker_action_dim}",
            )
        for j, a in enumerate(vec):
            if isinstance(a, bool) or not isinstance(a, (int, float)):
                raise HTTPException(
                    status_code=400,
                    detail=f"actions[{i}][{j}] must be a number, got "
                    f"{type(a).__name__}",
                )

    tensor = torch.from_numpy(video).float().permute(0, 3, 1, 2).unsqueeze(0)
    actions_tensor = torch.tensor(actions_raw, dtype=torch.float32).unsqueeze(0)

    try:
        future_features, confidence = worker.predict_action_conditioned(
            tensor, actions_tensor, past_frames, future_frames,
        )
    except RuntimeError as exc:
        # Worker configured without action_predictor_ckpt, or K/H mismatch
        # against the worker's grid, or video too short.
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {
        "next_features": future_features.tolist(),
        "planned_actions": actions_raw,
        "confidence": float(confidence),
        "mode": "action",
    }


def _check_auth(request: Request, auth: HmacTokenAuth, body_bytes: bytes) -> None:
    hdr = request.headers.get("Authorization", "")
    ts = request.headers.get("X-WorldFoundry-Timestamp", "")
    if not hdr.startswith("HMAC "):
        raise HTTPException(status_code=401, detail="missing or malformed Authorization header")
    sig = hdr[len("HMAC "):]
    if not auth.verify(request.method, request.url.path, body_bytes, ts, sig):
        raise HTTPException(status_code=401, detail="HMAC verification failed")
