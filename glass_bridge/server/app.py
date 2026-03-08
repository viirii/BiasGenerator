# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from glass_bridge.models import (
    CloseResponse,
    ResetRequest,
    ResetResponse,
    StepRequest,
    StepResponse,
)
from glass_bridge.server.glass_bridge_environment import GlassBridgeSessionManager

app = FastAPI(title="Glass Bridge OpenEnv")
manager = GlassBridgeSessionManager()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/reset", response_model=ResetResponse)
def reset_environment(request: ResetRequest) -> ResetResponse:
    try:
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


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
