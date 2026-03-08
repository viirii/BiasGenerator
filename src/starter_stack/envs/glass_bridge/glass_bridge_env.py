from __future__ import annotations

import random
from typing import Any


class GlassBridgeEnv:
    NUM_AGENTS = 10
    NUM_STEPS = 20
    UNKNOWN = -1
    LEFT = 0
    RIGHT = 1
    ALPHA = 0.05

    PHASE_COMMUNICATION = "communication"
    PHASE_MOVEMENT = "movement"
    PHASE_TERMINAL = "terminal"

    ACTION_SHARE_NONE = "SHARE_NONE"
    ACTION_SHARE_FIRST = "SHARE_FIRST"
    ACTION_SHARE_SECOND = "SHARE_SECOND"
    ACTION_SHARE_BOTH = "SHARE_BOTH"
    ACTION_LEFT = "LEFT"
    ACTION_RIGHT = "RIGHT"
    ACTION_NOOP = "NOOP"

    STEP_ASSIGNMENT = {
        0: [18, 19],
        1: [16, 17],
        2: [14, 15],
        3: [12, 13],
        4: [10, 11],
        5: [8, 9],
        6: [6, 7],
        7: [4, 5],
        8: [2, 3],
        9: [0, 1],
    }

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self.safe_sides: list[int] = []
        self.alive: list[bool] = []
        self.finished: list[bool] = []
        self.progress: list[int] = []
        self.public_known: list[int] = []
        self.private_knowledge: dict[int, dict[int, int]] = {}
        self.current_actor: int | None = None
        self.phase = self.PHASE_TERMINAL

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            self.rng.seed(seed)

        self.safe_sides = [self.rng.randint(self.LEFT, self.RIGHT) for _ in range(self.NUM_STEPS)]
        self.alive = [True] * self.NUM_AGENTS
        self.finished = [False] * self.NUM_AGENTS
        self.progress = [0] * self.NUM_AGENTS
        self.public_known = [self.UNKNOWN] * self.NUM_STEPS
        self.private_knowledge = {
            agent_id: {step_idx: self.safe_sides[step_idx] for step_idx in self._owned_steps(agent_id)}
            for agent_id in range(self.NUM_AGENTS)
        }
        self.current_actor = 0
        self.phase = self.PHASE_COMMUNICATION
        return self._result(self._zero_rewards(), done=False, events=[])

    def step(self, action_dict: dict[Any, str]) -> dict[str, Any]:
        normalized_actions = self._normalize_action_dict(action_dict)
        if self.phase == self.PHASE_TERMINAL:
            return self._result(self._final_rewards(), done=True, events=[])

        if self.current_actor is None:
            self.phase = self.PHASE_TERMINAL
            return self._result(self._final_rewards(), done=True, events=[])

        if self.phase == self.PHASE_COMMUNICATION:
            events = self._apply_comm_actions(normalized_actions)
            self.phase = self.PHASE_MOVEMENT
            return self._result(self._zero_rewards(), done=False, events=events)

        if self.phase == self.PHASE_MOVEMENT:
            actor_name = self.agent_name(self.current_actor)
            action = normalized_actions.get(actor_name, self.ACTION_NOOP)
            events = self._apply_move_action(self.current_actor, action)
            done = self.current_actor is None
            if done:
                self.phase = self.PHASE_TERMINAL
                rewards = self._final_rewards()
            else:
                self.phase = self.PHASE_COMMUNICATION
                rewards = self._zero_rewards()
            return self._result(rewards, done=done, events=events)

        raise RuntimeError(f"Unknown phase: {self.phase}")

    def get_observation(self, agent_id: int | str) -> dict[str, Any]:
        idx = self._coerce_agent_id(agent_id)
        owned_steps = self._owned_steps(idx)
        return {
            "self_id": idx,
            "agent_name": self.agent_name(idx),
            "current_actor": self.current_actor,
            "current_actor_name": None if self.current_actor is None else self.agent_name(self.current_actor),
            "phase": self.phase,
            "alive": [int(value) for value in self.alive],
            "finished": [int(value) for value in self.finished],
            "progress": self.progress[:],
            "public_known": self.public_known[:],
            "owned_steps": owned_steps[:],
            "owned_sides": [self.private_knowledge[idx][step_idx] for step_idx in owned_steps],
            "owned_is_public": [self.public_known[step_idx] != self.UNKNOWN for step_idx in owned_steps],
            "legal_actions": self.legal_actions(idx),
        }

    def legal_actions(self, agent_id: int | str) -> list[str]:
        idx = self._coerce_agent_id(agent_id)

        if self.phase == self.PHASE_TERMINAL:
            return [self.ACTION_NOOP]

        if self.phase == self.PHASE_COMMUNICATION:
            if not self.alive[idx]:
                return [self.ACTION_NOOP]
            if idx == self.current_actor:
                return [self.ACTION_NOOP]

            first_step, second_step = self._owned_steps(idx)
            first_public = self.public_known[first_step] != self.UNKNOWN
            second_public = self.public_known[second_step] != self.UNKNOWN
            legal = [self.ACTION_SHARE_NONE]
            if not first_public:
                legal.append(self.ACTION_SHARE_FIRST)
            if not second_public:
                legal.append(self.ACTION_SHARE_SECOND)
            if not (first_public and second_public):
                legal.append(self.ACTION_SHARE_BOTH)
            return legal

        if self.phase == self.PHASE_MOVEMENT:
            if idx == self.current_actor:
                return [self.ACTION_LEFT, self.ACTION_RIGHT]
            return [self.ACTION_NOOP]

        return [self.ACTION_NOOP]

    def _apply_comm_actions(self, action_dict: dict[str, str]) -> list[dict[str, Any]]:
        reveals: list[dict[str, Any]] = []
        for agent_id in range(self.NUM_AGENTS):
            if not self.alive[agent_id] or agent_id == self.current_actor:
                continue

            agent_name = self.agent_name(agent_id)
            legal = self.legal_actions(agent_id)
            action = action_dict.get(agent_name, self.ACTION_SHARE_NONE)
            if action not in legal:
                raise ValueError(f"Illegal communication action for {agent_name}: {action}")

            revealed_steps = self._reveal_steps_for_action(agent_id, action)
            if not revealed_steps:
                continue

            reveals.append(
                {
                    "sharer": agent_name,
                    "steps": revealed_steps,
                    "sides": [self.public_known[step_idx] for step_idx in revealed_steps],
                }
            )

        return [
            {
                "type": self.PHASE_COMMUNICATION,
                "actor": None if self.current_actor is None else self.agent_name(self.current_actor),
                "reveals": reveals,
            }
        ]

    def _apply_move_action(self, actor: int, action: str) -> list[dict[str, Any]]:
        legal = self.legal_actions(actor)
        if action not in legal:
            raise ValueError(f"Illegal movement action for {self.agent_name(actor)}: {action}")

        step_idx = self._current_step_idx(actor)
        chosen_side = self.LEFT if action == self.ACTION_LEFT else self.RIGHT
        correct_side = self.safe_sides[step_idx]
        result = "success"

        if chosen_side == correct_side:
            self.public_known[step_idx] = correct_side
            self.progress[actor] += 1
            if self.progress[actor] == self.NUM_STEPS:
                self.finished[actor] = True
                result = "finished"
                self._advance_actor(actor + 1)
        else:
            self.alive[actor] = False
            result = "death"
            self._advance_actor(actor + 1)

        return [
            {
                "type": self.PHASE_MOVEMENT,
                "actor": self.agent_name(actor),
                "step_idx": step_idx,
                "chosen_side": chosen_side,
                "correct_side": correct_side,
                "result": result,
            }
        ]

    def _advance_actor(self, start_idx: int) -> None:
        self.current_actor = self._find_next_alive_unfinished(start_idx)

    def _find_next_alive_unfinished(self, start_idx: int) -> int | None:
        for agent_id in range(start_idx, self.NUM_AGENTS):
            if self.alive[agent_id] and not self.finished[agent_id]:
                return agent_id
        return None

    def _final_rewards(self) -> dict[str, float]:
        rewards = {}
        for i in range(self.NUM_AGENTS):
            survived = 1.0 if self.finished[i] else 0.0
            survivors_ahead = sum(1 for j in range(i) if self.finished[j])
            rewards[self.agent_name(i)] = survived - self.ALPHA * survivors_ahead
        return rewards

    def _obs_all(self) -> dict[str, dict[str, Any]]:
        return {self.agent_name(i): self.get_observation(i) for i in range(self.NUM_AGENTS)}

    def _info(self) -> dict[str, Any]:
        info = {
            "phase": self.phase,
            "current_actor": self.current_actor,
            "current_actor_name": None if self.current_actor is None else self.agent_name(self.current_actor),
            "num_survivors": sum(self.finished),
            "finished": self.finished[:],
            "alive": self.alive[:],
            "progress": self.progress[:],
            "public_known_count": sum(1 for side in self.public_known if side != self.UNKNOWN),
        }
        if self.phase == self.PHASE_TERMINAL:
            info["survivor_ids"] = [self.agent_name(i) for i in range(self.NUM_AGENTS) if self.finished[i]]
        return info

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

    def _reveal_steps_for_action(self, agent_id: int, action: str) -> list[int]:
        if action == self.ACTION_SHARE_NONE:
            return []

        first_step, second_step = self._owned_steps(agent_id)
        requested_steps: list[int] = []
        if action in (self.ACTION_SHARE_FIRST, self.ACTION_SHARE_BOTH):
            requested_steps.append(first_step)
        if action in (self.ACTION_SHARE_SECOND, self.ACTION_SHARE_BOTH):
            requested_steps.append(second_step)

        revealed_steps: list[int] = []
        for step_idx in requested_steps:
            if self.public_known[step_idx] == self.UNKNOWN:
                self.public_known[step_idx] = self.private_knowledge[agent_id][step_idx]
                revealed_steps.append(step_idx)
        return revealed_steps

    def _owned_steps(self, agent_id: int) -> list[int]:
        return self.STEP_ASSIGNMENT[agent_id]

    def _current_step_idx(self, actor: int) -> int:
        return self.progress[actor]

    def _normalize_action_dict(self, action_dict: dict[Any, str]) -> dict[str, str]:
        normalized: dict[str, str] = {}
        for agent_id, action in action_dict.items():
            idx = self._coerce_agent_id(agent_id)
            normalized[self.agent_name(idx)] = action
        return normalized

    def _coerce_agent_id(self, agent_id: int | str) -> int:
        if isinstance(agent_id, int):
            if 0 <= agent_id < self.NUM_AGENTS:
                return agent_id
            raise ValueError(f"Agent index out of range: {agent_id}")
        if isinstance(agent_id, str) and agent_id.startswith("p"):
            idx = int(agent_id[1:])
            if 0 <= idx < self.NUM_AGENTS:
                return idx
        raise ValueError(f"Unsupported agent id: {agent_id}")

    def _zero_rewards(self) -> dict[str, float]:
        return {self.agent_name(i): 0.0 for i in range(self.NUM_AGENTS)}

    @staticmethod
    def agent_name(agent_id: int) -> str:
        return f"p{agent_id}"

    @staticmethod
    def side_name(side: int) -> str:
        return "LEFT" if side == GlassBridgeEnv.LEFT else "RIGHT"
