# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any

import requests

from .models import (
    CloseResponse,
    ResetRequest,
    ResetResponse,
    StepRequest,
    StepResponse,
)


class OpenEnvGlassBridgeClient:
    def __init__(self, base_url: str, timeout_s: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.session_id: str | None = None
        self._http = requests.Session()

    def reset(self, request: ResetRequest | None = None, **kwargs: Any) -> ResetResponse:
        payload = request or ResetRequest(**kwargs)
        response = self._http.post(
            f"{self.base_url}/reset",
            json=payload.model_dump(mode="json"),
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        parsed = ResetResponse.model_validate(response.json())
        self.session_id = parsed.session_id
        return parsed

    def step(self, action: StepRequest | dict[str, Any]) -> StepResponse:
        if self.session_id is None:
            raise RuntimeError("Call reset() before step()")

        if isinstance(action, StepRequest):
            request = action
        else:
            request = StepRequest(session_id=self.session_id, actions=action)

        response = self._http.post(
            f"{self.base_url}/step",
            json=request.model_dump(mode="json"),
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        return StepResponse.model_validate(response.json())

    def close(self) -> CloseResponse:
        if self.session_id is None:
            self._http.close()
            return CloseResponse(session_id="", closed=False)

        response = self._http.delete(
            f"{self.base_url}/close/{self.session_id}",
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        parsed = CloseResponse.model_validate(response.json())
        self.session_id = None
        self._http.close()
        return parsed


GlassBridgeEnv = OpenEnvGlassBridgeClient
