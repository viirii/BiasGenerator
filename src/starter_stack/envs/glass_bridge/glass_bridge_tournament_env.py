from __future__ import annotations

import random
from typing import Any


class GlassBridgeTournamentEnv:
    DEFAULT_INITIAL_PLAYERS = 16
    DEFAULT_FIRST_ROUND_NUM_STEPS = 18
    UNKNOWN = -1
    LEFT = 0
    RIGHT = 1
    MAX_PAIR_PROPOSALS_PER_ROUND = 3
    MAX_PAIR_REJECTIONS_PER_ROUND = 3

    PHASE_COMMUNICATION_OFFER = "communication_offer"
    PHASE_COMMUNICATION_RESPONSE = "communication_response"
    PHASE_MOVEMENT = "movement"
    PHASE_TERMINAL = "terminal"

    ACTION_LEFT = "LEFT"
    ACTION_RIGHT = "RIGHT"
    ACTION_NOOP = {"type": "NOOP"}

    def __init__(
        self,
        seed: int = 0,
        max_rounds: int = 25,
        initial_players: int = DEFAULT_INITIAL_PLAYERS,
        first_round_num_steps: int = DEFAULT_FIRST_ROUND_NUM_STEPS,
        strategy_profiles: dict[str, dict[str, Any]] | None = None,
    ):
        self.rng = random.Random(seed)
        self.max_rounds = max_rounds
        self.initial_players = initial_players
        self.first_round_num_steps = first_round_num_steps
        self.strategy_profiles = strategy_profiles or {}

        self.all_agents = [self.agent_name(i) for i in range(self.initial_players)]
        self.phase = self.PHASE_TERMINAL
        self.round_idx = 0
        self.round_num_steps = 0
        self.active_agents: list[str] = []
        self.current_order: list[str] = []
        self.position_map: dict[str, int] = {}
        self.safe_sides: list[int] = []
        self.verified_public: list[int] = []
        self.round_alive: dict[str, bool] = {}
        self.round_finished: dict[str, bool] = {}
        self.round_progress: dict[str, int] = {}
        self.round_assignment: dict[str, list[int]] = {}
        self.round_private_knowledge: dict[str, dict[int, int]] = {}
        self.private_known_by_agent: dict[str, dict[int, int]] = {}
        self.current_actor: str | None = None
        self.reputation: dict[str, dict[str, dict[str, int]]] = {}
        self.pair_rejections: dict[str, int] = {}
        self.pair_offer_counts: dict[str, int] = {}
        self.locked_pairs: set[str] = set()
        self.pending_offers: list[dict[str, Any]] = []
        self.private_claims_by_step: dict[int, list[dict[str, Any]]] = {}
        self.round_trade_summary: dict[str, Any] = {}
        self.cumulative_stats: dict[str, dict[str, int]] = {}
        self.round_history: list[dict[str, Any]] = []
        self.winner: str | None = None
        self._offer_seq = 0

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            self.rng.seed(seed)

        self.phase = self.PHASE_COMMUNICATION_OFFER
        self.round_idx = 0
        self.active_agents = self.all_agents[:]
        self.current_order = []
        self.position_map = {}
        self.round_num_steps = 0
        self.safe_sides = []
        self.verified_public = []
        self.private_claims_by_step = {}
        self.pending_offers = []
        self.pair_rejections = {}
        self.pair_offer_counts = {}
        self.locked_pairs = set()
        self.round_trade_summary = {}
        self.round_alive = {}
        self.round_finished = {}
        self.round_progress = {}
        self.round_assignment = {}
        self.round_private_knowledge = {}
        self.private_known_by_agent = {}
        self.current_actor = None
        self.round_history = []
        self.winner = None
        self._offer_seq = 0
        self.reputation = {
            observer: {
                speaker: {"truth_count": 0, "lie_count": 0, "share_count": 0}
                for speaker in self.all_agents
                if speaker != observer
            }
            for observer in self.all_agents
        }
        self.cumulative_stats = {
            agent_name: {"rounds_played": 0, "rounds_survived": 0, "total_progress": 0}
            for agent_name in self.all_agents
        }

        events = self._start_new_round()
        return self._result(self._zero_rewards(), done=False, events=events)

    def step(self, action_dict: dict[Any, str]) -> dict[str, Any]:
        normalized_actions = self._normalize_action_dict(action_dict)

        if self.phase == self.PHASE_TERMINAL:
            return self._result(self._final_rewards(), done=True, events=[])

        if self.current_actor is None:
            self.phase = self.PHASE_TERMINAL
            return self._result(self._final_rewards(), done=True, events=[])

        if self.phase == self.PHASE_COMMUNICATION_OFFER:
            events = self._apply_offer_actions(normalized_actions)
            self.phase = self.PHASE_COMMUNICATION_RESPONSE
            return self._result(self._zero_rewards(), done=False, events=events)

        if self.phase == self.PHASE_COMMUNICATION_RESPONSE:
            events = self._apply_response_actions(normalized_actions)
            self.phase = self.PHASE_MOVEMENT
            return self._result(self._zero_rewards(), done=False, events=events)

        if self.phase == self.PHASE_MOVEMENT:
            action = normalized_actions.get(self.current_actor, self.ACTION_NOOP)
            events = self._apply_move_action(self.current_actor, action)
            done = False
            rewards = self._zero_rewards()

            if self.current_actor is None:
                round_events, done = self._complete_round()
                events.extend(round_events)
                rewards = self._final_rewards() if done else self._zero_rewards()

            self.phase = self.PHASE_TERMINAL if done else self.PHASE_MOVEMENT
            return self._result(rewards, done=done, events=events)

        raise RuntimeError(f"Unknown phase: {self.phase}")

    def get_observation(self, agent_id: int | str) -> dict[str, Any]:
        agent_name = self._coerce_agent_name(agent_id)
        current_step_idx = self._current_step_idx(self.current_actor) if self.current_actor is not None else None

        return {
            "agent_name": agent_name,
            "self_id": self._coerce_agent_index(agent_name),
            "phase": self.phase,
            "round_idx": self.round_idx,
            "initial_players": self.initial_players,
            "round_num_steps": self.round_num_steps,
            "active_agents": self.active_agents[:],
            "current_order": self.current_order[:],
            "position_map": self.position_map.copy(),
            "assignment_by_agent": {name: steps[:] for name, steps in self.round_assignment.items()},
            "current_actor": self.current_actor,
            "current_position": None if self.current_actor is None else self.position_map[self.current_actor],
            "round_alive": self.round_alive.copy(),
            "round_finished": self.round_finished.copy(),
            "round_progress": self.round_progress.copy(),
            "verified_public": self.verified_public[:],
            "private_known_steps": dict(sorted(self.private_known_by_agent.get(agent_name, {}).items())),
            "owned_steps": self.round_assignment.get(agent_name, [])[:],
            "owned_sides": [
                self.round_private_knowledge[agent_name][step_idx]
                for step_idx in self.round_assignment.get(agent_name, [])
            ],
            "current_step_idx": current_step_idx,
            "reputation": {
                speaker: counts.copy()
                for speaker, counts in self.reputation.get(agent_name, {}).items()
            },
            "pair_rejections": {
                partner: self._pair_rejection_count(agent_name, partner)
                for partner in self._negotiable_partners(agent_name)
            },
            "negotiable_partners": self._negotiable_partners(agent_name),
            "incoming_offers": self._incoming_offers_for(agent_name),
            "strategy_profile": dict(self.strategy_profiles.get(agent_name, {})),
            "round_history": self.round_history[:],
            "legal_actions": self.legal_actions(agent_name),
        }

    def legal_actions(self, agent_id: int | str) -> list[str]:
        agent_name = self._coerce_agent_name(agent_id)

        if self.phase == self.PHASE_TERMINAL or agent_name not in self.active_agents:
            return [{"type": "NOOP"}]

        if self.phase == self.PHASE_COMMUNICATION_OFFER:
            if not self.round_alive.get(agent_name, False):
                return [{"type": "NOOP"}]
            return [{"type": "NOOP"}, {"type": "OFFERS"}]

        if self.phase == self.PHASE_COMMUNICATION_RESPONSE:
            if not self.round_alive.get(agent_name, False):
                return [{"type": "NOOP"}]
            return [{"type": "NOOP"}, {"type": "RESPONSES"}]

        if self.phase == self.PHASE_MOVEMENT:
            if agent_name == self.current_actor:
                return [self.ACTION_LEFT, self.ACTION_RIGHT]
            return [{"type": "NOOP"}]

        return [{"type": "NOOP"}]

    def _start_new_round(self) -> list[dict[str, Any]]:
        self.round_idx += 1
        self.current_order = self.active_agents[:]
        self.rng.shuffle(self.current_order)
        self.position_map = {agent_name: idx for idx, agent_name in enumerate(self.current_order)}
        # Round 1 uses a fixed bridge length, then later rounds shorten to
        # active_players + 1 so more agents can survive and build reputation.
        if self.round_idx == 1:
            self.round_num_steps = self.first_round_num_steps
        else:
            self.round_num_steps = len(self.active_agents) + 1
        self.safe_sides = [self.rng.randint(self.LEFT, self.RIGHT) for _ in range(self.round_num_steps)]
        self.verified_public = [self.UNKNOWN] * self.round_num_steps
        self.private_claims_by_step = {step_idx: [] for step_idx in range(self.round_num_steps)}
        self.pending_offers = []
        self.pair_rejections = {}
        self.pair_offer_counts = {}
        self.locked_pairs = set()
        self.round_trade_summary = {
            "offers_made": 0,
            "offers_rejected": 0,
            "offers_accepted": 0,
            "accepted_trade_pairs": {},
            "pair_offer_counts": {},
            "pair_rejections": {},
        }
        self.round_alive = {agent_name: True for agent_name in self.active_agents}
        self.round_finished = {agent_name: False for agent_name in self.active_agents}
        self.round_progress = {agent_name: 0 for agent_name in self.active_agents}
        self.round_assignment = self._build_round_assignment(self.current_order)
        self.round_private_knowledge = {
            agent_name: {
                step_idx: self.safe_sides[step_idx]
                for step_idx in self.round_assignment[agent_name]
            }
            for agent_name in self.active_agents
        }
        self.private_known_by_agent = {
            agent_name: knowledge.copy()
            for agent_name, knowledge in self.round_private_knowledge.items()
        }
        self.current_actor = self.current_order[0] if self.current_order else None

        return [
            {
                "type": "round_start",
                "round_idx": self.round_idx,
                "round_num_steps": self.round_num_steps,
                "order": self.current_order[:],
                "assignment_by_agent": {name: steps[:] for name, steps in self.round_assignment.items()},
                "active_agents": self.active_agents[:],
            }
        ]

    def _build_round_assignment(self, order: list[str]) -> dict[str, list[int]]:
        if not order:
            return {}

        # Assign contiguous step chunks in reverse order so earlier bridge
        # positions still own later bridge knowledge. With round_num_steps equal
        # to the number of active players, this gives each player one private step.
        num_agents = len(order)
        base_chunk = self.round_num_steps // num_agents
        remainder = self.round_num_steps % num_agents
        cursor = self.round_num_steps
        assignment: dict[str, list[int]] = {}
        for position, agent_name in enumerate(order):
            chunk_size = base_chunk + (1 if position < remainder else 0)
            start = max(0, cursor - chunk_size)
            assignment[agent_name] = list(range(start, cursor))
            cursor = start
        return assignment

    def _apply_offer_actions(self, action_dict: dict[str, Any]) -> list[dict[str, Any]]:
        pending_offers: list[dict[str, Any]] = []

        for agent_name in self.current_order:
            if not self.round_alive.get(agent_name, False):
                continue

            action = action_dict.get(agent_name, {"type": "NOOP"})
            if not isinstance(action, dict):
                raise ValueError(f"Illegal offer action for {agent_name}: {action}")
            if action.get("type", "NOOP") == "NOOP":
                continue
            if action.get("type") != "OFFERS":
                raise ValueError(f"Illegal offer action type for {agent_name}: {action}")

            offers = list(action.get("offers", []))
            for offer in offers:
                recipient = self._coerce_agent_name(offer["recipient"])
                if not self._can_pair_negotiate(agent_name, recipient):
                    continue

                give_steps = self._normalize_offer_steps(
                    owner=agent_name,
                    step_values=offer.get("give_steps", []),
                    known_steps=self.private_known_by_agent[agent_name],
                )
                request_steps = self._normalize_offer_steps(
                    owner=recipient,
                    step_values=offer.get("request_steps", []),
                    known_steps=self.round_private_knowledge[recipient],
                )
                if not give_steps or not request_steps:
                    continue

                claim_mode = offer.get("claim_mode", "truth")
                if claim_mode not in ("truth", "lie"):
                    claim_mode = "truth"

                outgoing_claims = [
                    {
                        "step_idx": step_idx,
                        "claimed_side": self._claim_side(agent_name, step_idx, claim_mode),
                    }
                    for step_idx in give_steps
                ]
                pending_offers.append(
                    {
                        "offer_id": self._next_offer_id(),
                        "proposer": agent_name,
                        "recipient": recipient,
                        "give_steps": give_steps,
                        "request_steps": request_steps,
                        "claim_mode": claim_mode,
                        "claims": outgoing_claims,
                    }
                )
                self._increment_pair_offer_count(agent_name, recipient)
                self.round_trade_summary["offers_made"] += 1
                self.round_trade_summary["pair_offer_counts"] = self.pair_offer_counts.copy()

        self.pending_offers = pending_offers
        return [
            {
                "type": self.PHASE_COMMUNICATION_OFFER,
                "round_idx": self.round_idx,
                "actor": self.current_actor,
                "pending_offer_count": len(self.pending_offers),
                "private_offer_pairs": sorted({self._pair_key(offer["proposer"], offer["recipient"]) for offer in self.pending_offers}),
            }
        ]

    def _apply_response_actions(self, action_dict: dict[str, Any]) -> list[dict[str, Any]]:
        accepted_pairs: dict[str, int] = {}
        rejected_pairs: dict[str, int] = {}

        responses_by_agent: dict[str, set[int]] = {}
        for agent_name in self.current_order:
            if not self.round_alive.get(agent_name, False):
                continue

            action = action_dict.get(agent_name, {"type": "NOOP"})
            if not isinstance(action, dict):
                raise ValueError(f"Illegal response action for {agent_name}: {action}")
            if action.get("type", "NOOP") == "NOOP":
                responses_by_agent[agent_name] = set()
                continue
            if action.get("type") != "RESPONSES":
                raise ValueError(f"Illegal response action type for {agent_name}: {action}")

            accepted_ids = {
                int(offer_id)
                for offer_id in action.get("accept_offer_ids", [])
            }
            responses_by_agent[agent_name] = accepted_ids

        for offer in self.pending_offers:
            recipient = offer["recipient"]
            proposer = offer["proposer"]
            pair_key = self._pair_key(proposer, recipient)

            accepted = offer["offer_id"] in responses_by_agent.get(recipient, set())
            if not self._can_pair_negotiate(proposer, recipient):
                accepted = False

            if accepted:
                self._execute_trade(offer)
                accepted_pairs[pair_key] = accepted_pairs.get(pair_key, 0) + 1
                self.round_trade_summary["offers_accepted"] += 1
                summary_pairs = self.round_trade_summary.setdefault("accepted_trade_pairs", {})
                summary_pairs[pair_key] = summary_pairs.get(pair_key, 0) + 1
                self.locked_pairs.add(pair_key)
            else:
                self._increment_pair_rejection(proposer, recipient)
                rejected_pairs[pair_key] = rejected_pairs.get(pair_key, 0) + 1
                self.round_trade_summary["offers_rejected"] += 1

        self.pending_offers = []
        self.round_trade_summary["pair_offer_counts"] = self.pair_offer_counts.copy()
        self.round_trade_summary["pair_rejections"] = self.pair_rejections.copy()

        return [
            {
                "type": self.PHASE_COMMUNICATION_RESPONSE,
                "round_idx": self.round_idx,
                "actor": self.current_actor,
                "accepted_trade_pairs": accepted_pairs,
                "rejected_trade_pairs": rejected_pairs,
                "pair_rejections": self.pair_rejections.copy(),
            }
        ]

    def _apply_move_action(self, actor: str, action: Any) -> list[dict[str, Any]]:
        legal = self.legal_actions(actor)
        if action not in legal:
            raise ValueError(f"Illegal movement action for {actor}: {action}")

        step_idx = self._current_step_idx(actor)
        chosen_side = self.LEFT if action == self.ACTION_LEFT else self.RIGHT
        correct_side = self.safe_sides[step_idx]
        result = "success"

        if chosen_side == correct_side:
            self.verified_public[step_idx] = correct_side
            self._resolve_private_claims_for_step(step_idx, correct_side)
            self.round_progress[actor] += 1
            if self.round_progress[actor] == self.round_num_steps:
                self.round_finished[actor] = True
                result = "finished_round"
                self._advance_actor(self.position_map[actor] + 1)
        else:
            self.round_alive[actor] = False
            result = "death"
            self._advance_actor(self.position_map[actor] + 1)

        return [
            {
                "type": self.PHASE_MOVEMENT,
                "round_idx": self.round_idx,
                "actor": actor,
                "step_idx": step_idx,
                "chosen_side": chosen_side,
                "correct_side": correct_side,
                "result": result,
            }
        ]

    def _complete_round(self) -> tuple[list[dict[str, Any]], bool]:
        self._resolve_all_private_claims()
        round_survivors = [agent_name for agent_name in self.active_agents if self.round_finished[agent_name]]
        fallback_used = False

        if not round_survivors:
            fallback_used = True
            best_progress = max(self.round_progress.values()) if self.round_progress else 0
            round_survivors = [
                agent_name
                for agent_name in self.active_agents
                if self.round_progress[agent_name] == best_progress
            ]

        for agent_name in self.active_agents:
            stats = self.cumulative_stats[agent_name]
            stats["rounds_played"] += 1
            stats["total_progress"] += self.round_progress[agent_name]
            if agent_name in round_survivors:
                stats["rounds_survived"] += 1

        round_summary = {
            "round_idx": self.round_idx,
            "round_num_steps": self.round_num_steps,
            "order": self.current_order[:],
            "survivors": round_survivors[:],
            "eliminated": [agent_name for agent_name in self.active_agents if agent_name not in round_survivors],
            "progress": self.round_progress.copy(),
            "finished": self.round_finished.copy(),
            "fallback_used": fallback_used,
            "trade_summary": {
                "offers_made": self.round_trade_summary.get("offers_made", 0),
                "offers_accepted": self.round_trade_summary.get("offers_accepted", 0),
                "offers_rejected": self.round_trade_summary.get("offers_rejected", 0),
                "accepted_trade_pairs": dict(self.round_trade_summary.get("accepted_trade_pairs", {})),
                "pair_offer_counts": dict(self.round_trade_summary.get("pair_offer_counts", {})),
                "pair_rejections": dict(self.round_trade_summary.get("pair_rejections", {})),
            },
        }
        self.round_history.append(round_summary)

        events: list[dict[str, Any]] = [
            {
                "type": "round_end",
                **round_summary,
            }
        ]

        self.active_agents = round_survivors[:]

        if len(self.active_agents) == 1:
            self.winner = self.active_agents[0]
            events.append(self._game_end_event(reason="single_survivor"))
            return events, True

        if self.round_idx >= self.max_rounds:
            self.winner = self._choose_winner_at_cap()
            events.append(self._game_end_event(reason="max_rounds"))
            return events, True

        events.extend(self._start_new_round())
        return events, False

    def _resolve_private_claims_for_step(self, step_idx: int, actual_side: int) -> None:
        for claim in self.private_claims_by_step.get(step_idx, []):
            if claim["resolved"]:
                continue
            truthful = claim["claimed_side"] == actual_side
            speaker = claim["speaker"]
            observer = claim["recipient"]
            bucket = self.reputation[observer][speaker]
            bucket["share_count"] += 1
            if truthful:
                bucket["truth_count"] += 1
            else:
                bucket["lie_count"] += 1
            claim["resolved"] = True

    def _resolve_all_private_claims(self) -> None:
        for step_idx in range(self.round_num_steps):
            self._resolve_private_claims_for_step(step_idx, self.safe_sides[step_idx])

    def _advance_actor(self, start_position: int) -> None:
        self.current_actor = self._find_next_alive_unfinished_agent(start_position)

    def _find_next_alive_unfinished_agent(self, start_position: int) -> str | None:
        for position in range(start_position, len(self.current_order)):
            agent_name = self.current_order[position]
            if self.round_alive.get(agent_name, False) and not self.round_finished.get(agent_name, False):
                return agent_name
        return None

    def _choose_winner_at_cap(self) -> str:
        ranked = sorted(
            self.active_agents,
            key=lambda agent_name: (
                self.cumulative_stats[agent_name]["rounds_survived"],
                self.cumulative_stats[agent_name]["total_progress"],
            ),
            reverse=True,
        )
        best_tuple = (
            self.cumulative_stats[ranked[0]]["rounds_survived"],
            self.cumulative_stats[ranked[0]]["total_progress"],
        )
        finalists = [
            agent_name
            for agent_name in ranked
            if (
                self.cumulative_stats[agent_name]["rounds_survived"],
                self.cumulative_stats[agent_name]["total_progress"],
            )
            == best_tuple
        ]
        return self.rng.choice(finalists)

    def _game_end_event(self, reason: str) -> dict[str, Any]:
        return {
            "type": "game_end",
            "reason": reason,
            "winner": self.winner,
            "winner_strategy": dict(self.strategy_profiles.get(self.winner or "", {})),
            "active_agents": self.active_agents[:],
            "rounds_played": self.round_idx,
            "initial_players": self.initial_players,
            "round_num_steps": self.round_num_steps,
        }

    def _current_step_idx(self, actor: str | None) -> int | None:
        if actor is None:
            return None
        return self.round_progress[actor]

    def _obs_all(self) -> dict[str, dict[str, Any]]:
        return {agent_name: self.get_observation(agent_name) for agent_name in self.all_agents}

    def _info(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "round_idx": self.round_idx,
            "initial_players": self.initial_players,
            "round_num_steps": self.round_num_steps,
            "active_agents": self.active_agents[:],
            "current_order": self.current_order[:],
            "current_actor": self.current_actor,
            "verified_public_count": sum(1 for side in self.verified_public if side != self.UNKNOWN),
            "pair_offer_counts": self.pair_offer_counts.copy(),
            "pair_rejections": self.pair_rejections.copy(),
            "round_history": self.round_history[:],
            "strategy_profiles": {name: dict(profile) for name, profile in self.strategy_profiles.items()},
            "cumulative_stats": {name: stats.copy() for name, stats in self.cumulative_stats.items()},
            "winner": self.winner,
            "winner_strategy": dict(self.strategy_profiles.get(self.winner or "", {})),
        }

    def _result(
        self,
        rewards: dict[str, float],
        done: bool,
        events: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "observations": self._obs_all(),
            "rewards": rewards,
            "done": done,
            "info": {
                **self._info(),
                "events": events,
            },
        }

    def _normalize_action_dict(self, action_dict: dict[Any, str]) -> dict[str, str]:
        normalized: dict[str, Any] = {}
        for agent_id, action in action_dict.items():
            normalized[self._coerce_agent_name(agent_id)] = action
        return normalized

    def _final_rewards(self) -> dict[str, float]:
        rewards = {agent_name: 0.0 for agent_name in self.all_agents}
        if self.winner is not None:
            rewards[self.winner] = 1.0
        return rewards

    def _zero_rewards(self) -> dict[str, float]:
        return {agent_name: 0.0 for agent_name in self.all_agents}

    def _next_offer_id(self) -> int:
        self._offer_seq += 1
        return self._offer_seq

    def _incoming_offers_for(self, agent_name: str) -> list[dict[str, Any]]:
        incoming = []
        if self.phase != self.PHASE_COMMUNICATION_RESPONSE:
            return incoming
        for offer in self.pending_offers:
            if offer["recipient"] != agent_name:
                continue
            incoming.append(
                {
                    "offer_id": offer["offer_id"],
                    "proposer": offer["proposer"],
                    "claims": [claim.copy() for claim in offer["claims"]],
                    "request_steps": offer["request_steps"][:],
                    "pair_rejections": self._pair_rejection_count(agent_name, offer["proposer"]),
                }
            )
        return incoming

    def _negotiable_partners(self, agent_name: str) -> list[str]:
        if not self.round_alive.get(agent_name, False):
            return []
        partners = []
        for partner in self.active_agents:
            if partner == agent_name:
                continue
            if not self.round_alive.get(partner, False):
                continue
            if self._can_pair_negotiate(agent_name, partner):
                partners.append(partner)
        return partners

    def _pair_key(self, a: str, b: str) -> str:
        left, right = sorted((a, b))
        return f"{left}|{right}"

    def _pair_rejection_count(self, a: str, b: str) -> int:
        return self.pair_rejections.get(self._pair_key(a, b), 0)

    def _pair_offer_count(self, a: str, b: str) -> int:
        return self.pair_offer_counts.get(self._pair_key(a, b), 0)

    def _increment_pair_offer_count(self, a: str, b: str) -> None:
        key = self._pair_key(a, b)
        self.pair_offer_counts[key] = self.pair_offer_counts.get(key, 0) + 1

    def _increment_pair_rejection(self, a: str, b: str) -> None:
        key = self._pair_key(a, b)
        self.pair_rejections[key] = self.pair_rejections.get(key, 0) + 1

    def _can_pair_negotiate(self, a: str, b: str) -> bool:
        if a == b:
            return False
        if not self.round_alive.get(a, False) or not self.round_alive.get(b, False):
            return False
        key = self._pair_key(a, b)
        if key in self.locked_pairs:
            return False
        if self._pair_offer_count(a, b) >= self.MAX_PAIR_PROPOSALS_PER_ROUND:
            return False
        return self._pair_rejection_count(a, b) < self.MAX_PAIR_REJECTIONS_PER_ROUND

    def _normalize_offer_steps(
        self,
        owner: str,
        step_values: list[Any],
        known_steps: dict[int, int],
    ) -> list[int]:
        normalized: list[int] = []
        for step_idx in step_values:
            idx = int(step_idx)
            if idx in known_steps and idx not in normalized and self.verified_public[idx] == self.UNKNOWN:
                normalized.append(idx)
        return normalized

    def _claim_side(self, agent_name: str, step_idx: int, claim_mode: str) -> int:
        truthful_side = self.private_known_by_agent[agent_name][step_idx]
        if claim_mode == "lie":
            return self._flip_side(truthful_side)
        return truthful_side

    def _execute_trade(self, offer: dict[str, Any]) -> None:
        proposer = offer["proposer"]
        recipient = offer["recipient"]

        for claim in offer["claims"]:
            step_idx = claim["step_idx"]
            if self.verified_public[step_idx] == self.UNKNOWN:
                self.private_known_by_agent[recipient][step_idx] = claim["claimed_side"]
            self.private_claims_by_step[step_idx].append(
                {
                    "speaker": proposer,
                    "recipient": recipient,
                    "step_idx": step_idx,
                    "claimed_side": claim["claimed_side"],
                    "resolved": False,
                }
            )

        reciprocal_truth_mode = self._sample_truth_mode(recipient)
        for step_idx in offer["request_steps"]:
            claimed_side = self._claim_side(recipient, step_idx, reciprocal_truth_mode)
            if self.verified_public[step_idx] == self.UNKNOWN:
                self.private_known_by_agent[proposer][step_idx] = claimed_side
            self.private_claims_by_step[step_idx].append(
                {
                    "speaker": recipient,
                    "recipient": proposer,
                    "step_idx": step_idx,
                    "claimed_side": claimed_side,
                    "resolved": False,
                }
            )

    def _sample_truth_mode(self, agent_name: str) -> str:
        truth_rate = float(self.strategy_profiles.get(agent_name, {}).get("truth_rate", 1.0))
        return "truth" if self.rng.random() <= truth_rate else "lie"

    def _coerce_agent_name(self, agent_id: int | str) -> str:
        if isinstance(agent_id, str):
            if agent_id in self.all_agents:
                return agent_id
            if agent_id.startswith("p"):
                idx = int(agent_id[1:])
                return self.agent_name(idx)
        if isinstance(agent_id, int):
            return self.agent_name(agent_id)
        raise ValueError(f"Unsupported agent id: {agent_id}")

    @classmethod
    def _coerce_agent_index(cls, agent_name: str) -> int:
        if not agent_name.startswith("p"):
            raise ValueError(f"Unsupported agent name: {agent_name}")
        return int(agent_name[1:])

    @classmethod
    def agent_name(cls, agent_id: int) -> str:
        if agent_id < 0:
            raise ValueError(f"Agent index out of range: {agent_id}")
        return f"p{agent_id}"

    @staticmethod
    def _flip_side(side: int) -> int:
        return 1 - side

    @staticmethod
    def side_name(side: int) -> str:
        return "LEFT" if side == GlassBridgeTournamentEnv.LEFT else "RIGHT"
