from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ReputationMemory(BaseModel):
    truth_count: int = 0
    lie_count: int = 0
    share_count: int = 0


class NegotiationClaim(BaseModel):
    step_idx: int
    claimed_side: int


class IncomingOffer(BaseModel):
    offer_id: int
    proposer: str
    claims: list[NegotiationClaim] = Field(default_factory=list)
    request_steps: list[int] = Field(default_factory=list)
    pair_rejections: int = 0


class NegotiationOfferProposal(BaseModel):
    recipient: str
    give_steps: list[int] = Field(default_factory=list)
    request_steps: list[int] = Field(default_factory=list)
    claim_mode: Literal["truth", "lie"] = "truth"


class AgentAction(BaseModel):
    action_type: Literal["NOOP", "LEFT", "RIGHT", "OFFERS", "RESPONSES"] = "NOOP"
    offers: list[NegotiationOfferProposal] = Field(default_factory=list)
    accept_offer_ids: list[int] = Field(default_factory=list)

    @classmethod
    def from_policy_output(cls, action: Any) -> "AgentAction":
        if isinstance(action, cls):
            return action
        if isinstance(action, str):
            if action in {"LEFT", "RIGHT"}:
                return cls(action_type=action)
            return cls(action_type="NOOP")
        if isinstance(action, dict):
            action_type = str(action.get("type", "NOOP")).upper()
            if action_type == "OFFERS":
                offers = [
                    NegotiationOfferProposal.model_validate(offer)
                    for offer in action.get("offers", [])
                ]
                return cls(action_type="OFFERS", offers=offers)
            if action_type == "RESPONSES":
                accept_offer_ids = [int(offer_id) for offer_id in action.get("accept_offer_ids", [])]
                return cls(action_type="RESPONSES", accept_offer_ids=accept_offer_ids)
            return cls(action_type="NOOP")
        raise TypeError(f"Unsupported action payload: {action!r}")

    def to_env_action(self) -> Any:
        if self.action_type in {"LEFT", "RIGHT"}:
            return self.action_type
        if self.action_type == "OFFERS":
            return {
                "type": "OFFERS",
                "offers": [offer.model_dump(mode="python") for offer in self.offers],
            }
        if self.action_type == "RESPONSES":
            return {
                "type": "RESPONSES",
                "accept_offer_ids": self.accept_offer_ids[:],
            }
        return {"type": "NOOP"}


class AgentObservation(BaseModel):
    model_config = ConfigDict(extra="allow")

    agent_name: str
    self_id: int
    phase: str
    round_idx: int
    initial_players: int
    round_num_steps: int
    active_agents: list[str] = Field(default_factory=list)
    current_order: list[str] = Field(default_factory=list)
    position_map: dict[str, int] = Field(default_factory=dict)
    assignment_by_agent: dict[str, list[int]] = Field(default_factory=dict)
    current_actor: str | None = None
    current_position: int | None = None
    round_alive: dict[str, bool] = Field(default_factory=dict)
    round_finished: dict[str, bool] = Field(default_factory=dict)
    round_progress: dict[str, int] = Field(default_factory=dict)
    verified_public: list[int] = Field(default_factory=list)
    private_known_steps: dict[int, int] = Field(default_factory=dict)
    owned_steps: list[int] = Field(default_factory=list)
    owned_sides: list[int] = Field(default_factory=list)
    current_step_idx: int | None = None
    reputation: dict[str, ReputationMemory] = Field(default_factory=dict)
    pair_rejections: dict[str, int] = Field(default_factory=dict)
    negotiable_partners: list[str] = Field(default_factory=list)
    incoming_offers: list[IncomingOffer] = Field(default_factory=list)
    strategy_profile: dict[str, Any] = Field(default_factory=dict)
    round_history: list[dict[str, Any]] = Field(default_factory=list)
    legal_actions: list[Any] = Field(default_factory=list)


class EnvironmentInfo(BaseModel):
    model_config = ConfigDict(extra="allow")

    phase: str
    round_idx: int
    initial_players: int
    round_num_steps: int
    active_agents: list[str] = Field(default_factory=list)
    current_order: list[str] = Field(default_factory=list)
    current_actor: str | None = None
    verified_public_count: int = 0
    pair_offer_counts: dict[str, int] = Field(default_factory=dict)
    pair_rejections: dict[str, int] = Field(default_factory=dict)
    round_history: list[dict[str, Any]] = Field(default_factory=list)
    strategy_profiles: dict[str, dict[str, Any]] = Field(default_factory=dict)
    cumulative_stats: dict[str, dict[str, int]] = Field(default_factory=dict)
    winner: str | None = None
    winner_strategy: dict[str, Any] = Field(default_factory=dict)
    events: list[dict[str, Any]] = Field(default_factory=list)


class EnvironmentResult(BaseModel):
    observations: dict[str, AgentObservation]
    rewards: dict[str, float]
    done: bool
    info: EnvironmentInfo


class StrategyProfile(BaseModel):
    model_config = ConfigDict(extra="allow")

    kind: str = "share_profile"
    model_name: str = "qwen3.5"
    share_rate: float = 0.5
    truth_rate: float = 0.5
    label: str = "model_qwen3.5_share_0.50_truth_0.50"


class ResetRequest(BaseModel):
    session_id: str | None = None
    seed: int | None = None
    max_rounds: int = 25
    initial_players: int = 16
    first_round_num_steps: int = 18
    share_rates: list[float] | None = None
    truth_rates: list[float] | None = None
    llm_model_pool: list[str] | None = None
    strategy_profiles: dict[str, StrategyProfile] | None = None


class ResetResponse(BaseModel):
    session_id: str
    result: EnvironmentResult


class StepRequest(BaseModel):
    session_id: str
    actions: dict[str, AgentAction] = Field(default_factory=dict)


class StepResponse(BaseModel):
    session_id: str
    result: EnvironmentResult


class CloseResponse(BaseModel):
    session_id: str
    closed: bool
