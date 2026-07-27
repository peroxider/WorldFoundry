"""POST /v1/world/plan — V-JEPA 2-AC planning. Phase-1 stub; full impl W4."""
from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.post("/plan")
async def plan_stub() -> dict:
    return {"status": "phase1-stub", "note": "Phase-2 W4: V-JEPA 2-AC planning"}
