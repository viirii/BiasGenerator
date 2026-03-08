from __future__ import annotations

from typing import Any

from starter_stack.envs.glass_bridge.glass_bridge_env import GlassBridgeEnv


class GlassBridgeRolloutRunner:
    def __init__(
        self,
        env: GlassBridgeEnv,
        policies: dict[str, Any],
        max_turns: int = 512,
    ):
        self.env = env
        self.policies = policies
        self.max_turns = max_turns

    def run_episode(self, seed: int | None = None) -> dict[str, Any]:
        result = self.env.reset(seed=seed)
        trace: list[str] = []
        turn_idx = 0

        while not result["done"] and turn_idx < self.max_turns:
            actions = self._select_actions(result["observations"])
            result = self.env.step(actions)
            trace.extend(self._format_events(result["info"].get("events", [])))
            turn_idx += 1

        if not result["done"]:
            raise RuntimeError(f"Episode did not terminate within {self.max_turns} turns")

        return {
            "seed": seed,
            "turns": turn_idx,
            "trace": trace,
            "survivors": result["info"].get("survivor_ids", []),
            "rewards": result["rewards"],
            "progress": result["info"]["progress"],
            "public_known_count": result["info"]["public_known_count"],
            "finished": result["info"]["finished"],
            "alive": result["info"]["alive"],
        }

    def _select_actions(self, observations: dict[str, dict[str, Any]]) -> dict[str, str]:
        actions: dict[str, str] = {}
        for agent_name, observation in observations.items():
            policy = self.policies[agent_name]
            action = policy.select_action(observation)
            legal = observation.get("legal_actions", [])
            if action not in legal:
                raise RuntimeError(f"Policy selected illegal action for {agent_name}: {action} not in {legal}")
            actions[agent_name] = action
        return actions

    def _format_events(self, events: list[dict[str, Any]]) -> list[str]:
        lines: list[str] = []
        for event in events:
            event_type = event.get("type")
            if event_type == GlassBridgeEnv.PHASE_COMMUNICATION:
                actor = event.get("actor")
                reveals = event.get("reveals", [])
                if not reveals:
                    lines.append(f"[COMM] actor={actor} reveals: none")
                    continue

                fragments = []
                for reveal in reveals:
                    step_names = ",".join(f"s{step_idx}" for step_idx in reveal.get("steps", []))
                    fragments.append(f"{reveal['sharer']}->{step_names}")
                lines.append(f"[COMM] actor={actor} reveals: {', '.join(fragments)}")
                continue

            if event_type == GlassBridgeEnv.PHASE_MOVEMENT:
                actor = event["actor"]
                step_idx = event["step_idx"]
                chosen_side = GlassBridgeEnv.side_name(event["chosen_side"])
                correct_side = GlassBridgeEnv.side_name(event["correct_side"])
                result = event["result"]
                lines.append(
                    f"[MOVE] {actor} step=s{step_idx} chose={chosen_side} "
                    f"correct={correct_side} result={result}"
                )
        return lines
