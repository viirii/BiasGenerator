from __future__ import annotations

import random
from typing import Any

from .tournament_env import GlassBridgeTournamentEnv


class TournamentGlassBridgePolicy:
    name = "tournament_strategy"

    def __init__(
        self,
        strategy_profile: dict[str, Any],
        seed: int = 0,
        adaptation_config: dict[str, Any] | None = None,
    ):
        self.strategy_profile = dict(strategy_profile)
        self._rng = random.Random(seed)
        self.adaptation = build_tournament_adaptation_strategy(adaptation_config or {})

    def select_action(self, observation: dict) -> Any:
        legal = observation.get("legal_actions", [])
        if not legal:
            raise RuntimeError("No legal actions available")

        phase = observation.get("phase")
        if phase == GlassBridgeTournamentEnv.PHASE_COMMUNICATION_OFFER:
            return self._offer_action(observation)
        if phase == GlassBridgeTournamentEnv.PHASE_COMMUNICATION_RESPONSE:
            return self._response_action(observation)
        return self._movement_action(observation, legal)

    def _offer_action(self, observation: dict) -> dict[str, Any]:
        share_rate = float(self.strategy_profile.get("share_rate", 0.0))
        partners = list(observation.get("negotiable_partners", []))
        if not partners:
            return {"type": "NOOP"}

        private_known = {
            int(step_idx): side
            for step_idx, side in observation.get("private_known_steps", {}).items()
        }
        if not private_known:
            return {"type": "NOOP"}

        scored_partners: list[tuple[float, str, list[int], list[int]]] = []
        for partner in partners:
            give_steps = [
                step_idx
                for step_idx in private_known.keys()
                if step_idx not in observation.get("assignment_by_agent", {}).get(partner, [])
            ]
            request_steps = [
                step_idx
                for step_idx in observation.get("assignment_by_agent", {}).get(partner, [])
                if int(step_idx) not in private_known
            ]
            if not give_steps or not request_steps:
                continue
            novelty_score = len(request_steps) / max(
                len(observation.get("assignment_by_agent", {}).get(partner, [])),
                1,
            )
            scored_partners.append(
                (
                    self.adaptation.partner_priority(
                        observation=observation,
                        partner=partner,
                        novelty_score=novelty_score,
                        policy=self,
                    ),
                    partner,
                    give_steps,
                    request_steps,
                )
            )

        if not scored_partners:
            return {"type": "NOOP"}

        scored_partners.sort(key=lambda item: item[0], reverse=True)
        offers = []
        for _, partner, give_steps, request_steps in scored_partners:
            effective_share_rate = self.adaptation.effective_share_rate(
                observation=observation,
                partner=partner,
                base_share_rate=share_rate,
                policy=self,
            )
            if self._rng.random() > effective_share_rate:
                continue

            claim_mode = self._sample_claim_mode_for_partner(
                observation=observation,
                partner=partner,
            )
            offers.append(
                {
                    "recipient": partner,
                    "give_steps": [self._rng.choice(give_steps)],
                    "request_steps": [int(self._rng.choice(request_steps))],
                    "claim_mode": claim_mode,
                }
            )

        if not offers:
            return {"type": "NOOP"}
        return {"type": "OFFERS", "offers": offers}

    def _response_action(self, observation: dict) -> dict[str, Any]:
        incoming_offers = list(observation.get("incoming_offers", []))
        if not incoming_offers:
            return {"type": "NOOP"}

        accepted_offer_ids: list[int] = []
        private_known = {
            int(step_idx): side
            for step_idx, side in observation.get("private_known_steps", {}).items()
        }
        for offer in incoming_offers:
            proposer = offer["proposer"]
            requested_steps = [int(step_idx) for step_idx in offer.get("request_steps", [])]
            offered_claims = offer.get("claims", [])
            if not requested_steps or not offered_claims:
                continue

            cost = sum(1 for step_idx in requested_steps if step_idx in private_known)
            benefit = sum(1 for claim in offered_claims if int(claim["step_idx"]) not in private_known)
            if benefit <= 0:
                continue
            if cost > benefit:
                continue
            if self._partner_reliability(observation, proposer) < self.adaptation.min_accept_reliability():
                continue
            accepted_offer_ids.append(int(offer["offer_id"]))

        if not accepted_offer_ids:
            return {"type": "NOOP"}
        return {"type": "RESPONSES", "accept_offer_ids": accepted_offer_ids}

    def _movement_action(self, observation: dict, legal: list[str]) -> str:
        if GlassBridgeTournamentEnv.ACTION_LEFT not in legal and GlassBridgeTournamentEnv.ACTION_RIGHT not in legal:
            return legal[0]

        current_step_idx = observation.get("current_step_idx")
        verified_public = observation.get("verified_public", [])
        if current_step_idx is not None and 0 <= current_step_idx < len(verified_public):
            public_side = verified_public[current_step_idx]
            if public_side == GlassBridgeTournamentEnv.LEFT:
                return GlassBridgeTournamentEnv.ACTION_LEFT
            if public_side == GlassBridgeTournamentEnv.RIGHT:
                return GlassBridgeTournamentEnv.ACTION_RIGHT

        private_known = observation.get("private_known_steps", {})
        if current_step_idx is not None and str(current_step_idx) in {str(step) for step in private_known.keys()}:
            private_side = private_known.get(current_step_idx)
            if private_side is None:
                private_side = private_known.get(str(current_step_idx))
            if private_side == GlassBridgeTournamentEnv.LEFT:
                return GlassBridgeTournamentEnv.ACTION_LEFT
            if private_side == GlassBridgeTournamentEnv.RIGHT:
                return GlassBridgeTournamentEnv.ACTION_RIGHT

        return self._rng.choice(
            [GlassBridgeTournamentEnv.ACTION_LEFT, GlassBridgeTournamentEnv.ACTION_RIGHT]
        )

    def _partner_reliability(self, observation: dict, partner: str) -> float:
        rep = observation.get("reputation", {}).get(partner, {})
        truth_count = int(rep.get("truth_count", 0))
        lie_count = int(rep.get("lie_count", 0))
        return (truth_count + 1.0) / (truth_count + lie_count + 2.0)

    @staticmethod
    def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
        return max(lower, min(upper, value))

    def _sample_claim_mode_for_partner(self, observation: dict, partner: str) -> str:
        effective_truth_rate = self.adaptation.effective_truth_rate(
            observation=observation,
            partner=partner,
            base_truth_rate=float(self.strategy_profile.get("truth_rate", 1.0)),
            policy=self,
        )
        return "truth" if self._rng.random() <= effective_truth_rate else "lie"


class TournamentAdaptationStrategy:
    kind = "base"

    def partner_priority(
        self,
        observation: dict,
        partner: str,
        novelty_score: float,
        policy: TournamentGlassBridgePolicy,
    ) -> float:
        return novelty_score

    def effective_share_rate(
        self,
        observation: dict,
        partner: str,
        base_share_rate: float,
        policy: TournamentGlassBridgePolicy,
    ) -> float:
        return policy._clamp(base_share_rate)

    def effective_truth_rate(
        self,
        observation: dict,
        partner: str,
        base_truth_rate: float,
        policy: TournamentGlassBridgePolicy,
    ) -> float:
        return policy._clamp(base_truth_rate)

    def min_accept_reliability(self) -> float:
        return 0.0


class NoReputationAdaptationStrategy(TournamentAdaptationStrategy):
    kind = "none"


class TruthScaledByReputationStrategy(TournamentAdaptationStrategy):
    kind = "truth_scaled_by_reputation"

    def effective_truth_rate(
        self,
        observation: dict,
        partner: str,
        base_truth_rate: float,
        policy: TournamentGlassBridgePolicy,
    ) -> float:
        return policy._clamp(base_truth_rate * policy._partner_reliability(observation, partner))


def build_tournament_adaptation_strategy(config: dict[str, Any]) -> TournamentAdaptationStrategy:
    kind = str(config.get("kind", "truth_scaled_by_reputation"))
    if kind == "none":
        return NoReputationAdaptationStrategy()
    if kind == "truth_scaled_by_reputation":
        return TruthScaledByReputationStrategy()
    raise ValueError(f"Unknown tournament adaptation strategy: {kind}")


def build_tournament_strategy_grid(
    share_rates: list[float],
    truth_rates: list[float],
) -> list[dict[str, Any]]:
    return [
        {
            "kind": "share_profile",
            "share_rate": float(share_rate),
            "truth_rate": float(truth_rate),
            "label": f"share_{float(share_rate):.2f}_truth_{float(truth_rate):.2f}",
        }
        for share_rate in share_rates
        for truth_rate in truth_rates
    ]


def assign_tournament_strategy_profiles(
    agent_names: list[str],
    seed: int,
    share_rates: list[float],
    truth_rates: list[float],
) -> dict[str, dict[str, Any]]:
    rng = random.Random(seed)
    grid = build_tournament_strategy_grid(share_rates=share_rates, truth_rates=truth_rates)
    return {agent_name: dict(rng.choice(grid)) for agent_name in agent_names}


def build_tournament_glass_bridge_population(
    strategy_profiles: dict[str, dict[str, Any]],
    seed: int,
    adaptation_config: dict[str, Any] | None = None,
) -> dict[str, TournamentGlassBridgePolicy]:
    population: dict[str, TournamentGlassBridgePolicy] = {}
    for offset, agent_name in enumerate(sorted(strategy_profiles.keys())):
        population[agent_name] = TournamentGlassBridgePolicy(
            strategy_profile=strategy_profiles[agent_name],
            seed=(seed * 1000) + 50_000 + offset,
            adaptation_config=adaptation_config,
        )
    return population
