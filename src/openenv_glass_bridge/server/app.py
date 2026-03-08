from __future__ import annotations

from fastapi import FastAPI, HTTPException

from openenv_glass_bridge.models import (
    CloseResponse,
    ResetRequest,
    ResetResponse,
    StepRequest,
    StepResponse,
)
from openenv_glass_bridge.server.glass_bridge_environment import GlassBridgeSessionManager

app = FastAPI(title="OpenEnv Glass Bridge")
manager = GlassBridgeSessionManager()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/reset", response_model=ResetResponse)
def reset_environment(request: ResetRequest) -> ResetResponse:
    try:
        print("New game started.", flush=True)
        return manager.reset(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/step", response_model=StepResponse)
def step_environment(request: StepRequest) -> StepResponse:
    try:
        return manager.step(request)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.delete("/close/{session_id}", response_model=CloseResponse)
def close_environment(session_id: str) -> CloseResponse:
    return manager.close(session_id)
