from __future__ import annotations

from threading import Lock
from uuid import uuid4

from starter_stack.envs.glass_bridge.glass_bridge_tournament_env import GlassBridgeTournamentEnv

from openenv_glass_bridge.models import (
    AgentAction,
    AgentObservation,
    CloseResponse,
    EnvironmentResult,
    ResetRequest,
    ResetResponse,
    StepRequest,
    StepResponse,
)


class GlassBridgeOpenEnvSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.env: GlassBridgeTournamentEnv | None = None

    def reset(self, request: ResetRequest) -> ResetResponse:
        seed = 0 if request.seed is None else int(request.seed)
        self.env = GlassBridgeTournamentEnv(
            seed=seed,
            max_rounds=int(request.max_rounds),
            initial_players=int(request.initial_players),
            first_round_num_steps=int(request.first_round_num_steps),
            strategy_profiles=self._normalize_strategy_profiles(request),
            share_rates=request.share_rates,
            truth_rates=request.truth_rates,
            llm_model_pool=request.llm_model_pool,
        )
        raw = self.env.reset(seed=seed)
        return ResetResponse(
            session_id=self.session_id,
            result=self._build_result(raw),
        )

    def step(self, request: StepRequest) -> StepResponse:
        env = self._require_env()
        raw_actions = {
            agent_name: action.to_env_action()
            for agent_name, action in request.actions.items()
        }
        raw = env.step(raw_actions)
        return StepResponse(
            session_id=self.session_id,
            result=self._build_result(raw),
        )

    def close(self) -> CloseResponse:
        self.env = None
        return CloseResponse(session_id=self.session_id, closed=True)

    def _require_env(self) -> GlassBridgeTournamentEnv:
        if self.env is None:
            raise ValueError("Environment session has not been reset yet")
        return self.env

    def _normalize_strategy_profiles(self, request: ResetRequest) -> dict[str, dict]:
        if request.strategy_profiles:
            return {
                agent_name: profile.model_dump(mode="python")
                for agent_name, profile in request.strategy_profiles.items()
            }
        return {}

    def _build_result(self, raw: dict) -> EnvironmentResult:
        return EnvironmentResult.model_validate(raw)


class GlassBridgeSessionManager:
    def __init__(self):
        self._sessions: dict[str, GlassBridgeOpenEnvSession] = {}
        self._lock = Lock()

    def reset(self, request: ResetRequest) -> ResetResponse:
        with self._lock:
            session_id = request.session_id or str(uuid4())
            session = self._sessions.get(session_id)
            if session is None:
                session = GlassBridgeOpenEnvSession(session_id=session_id)
                self._sessions[session_id] = session
        return session.reset(request)

    def step(self, request: StepRequest) -> StepResponse:
        return self._session(request.session_id).step(request)

    def close(self, session_id: str) -> CloseResponse:
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is None:
            return CloseResponse(session_id=session_id, closed=False)
        return session.close()

    def _session(self, session_id: str) -> GlassBridgeOpenEnvSession:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise ValueError(f"Unknown session_id: {session_id}")
        return session


def filtered_observations(result: EnvironmentResult) -> dict[str, AgentObservation]:
    return result.observations
