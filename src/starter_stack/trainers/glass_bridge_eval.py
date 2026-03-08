from __future__ import annotations

from statistics import mean
from typing import Any

from starter_stack.config import ensure_run_dirs
from starter_stack.envs.glass_bridge.glass_bridge_env import GlassBridgeEnv
from starter_stack.logging_utils import append_jsonl
from starter_stack.policies.glass_bridge import build_glass_bridge_population
from starter_stack.trainers.glass_bridge_rollout import GlassBridgeRolloutRunner


class GlassBridgeEvaluator:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.paths = ensure_run_dirs(config)

    def run(self) -> dict[str, Any]:
        eval_cfg = self.config["evaluation"]
        rollout_cfg = self.config["rollout"]
        base_seed = int(self.config.get("seed", 0))
        episodes_per_scenario = int(eval_cfg["episodes_per_scenario"])
        max_turns = int(rollout_cfg["max_turns"])
        scenarios = list(eval_cfg["scenarios"])
        log_path = str(self.paths["logs"] / f"{self.config['run_name']}.jsonl")

        scenario_summaries = []
        for scenario_idx, scenario in enumerate(scenarios):
            survivor_counts: list[int] = []
            public_known_counts: list[int] = []
            turn_counts: list[int] = []

            for episode_idx in range(episodes_per_scenario):
                episode_seed = base_seed + (scenario_idx * 10_000) + episode_idx
                env = GlassBridgeEnv(seed=episode_seed)
                runner = GlassBridgeRolloutRunner(
                    env=env,
                    policies=build_glass_bridge_population(scenario, seed=episode_seed),
                    max_turns=max_turns,
                )
                result = runner.run_episode(seed=episode_seed)

                survivor_count = len(result["survivors"])
                survivor_counts.append(survivor_count)
                public_known_counts.append(int(result["public_known_count"]))
                turn_counts.append(int(result["turns"]))

                append_jsonl(
                    log_path,
                    {
                        "type": "glass_bridge_episode",
                        "run_name": self.config["run_name"],
                        "scenario": scenario,
                        "episode_idx": episode_idx,
                        "seed": episode_seed,
                        "survivor_count": survivor_count,
                        "survivors": result["survivors"],
                        "public_known_count": result["public_known_count"],
                        "turns": result["turns"],
                        "rewards": result["rewards"],
                        "progress": result["progress"],
                    },
                )

            summary = {
                "scenario": scenario,
                "episodes": episodes_per_scenario,
                "avg_survivors": mean(survivor_counts) if survivor_counts else 0.0,
                "max_survivors": max(survivor_counts) if survivor_counts else 0,
                "min_survivors": min(survivor_counts) if survivor_counts else 0,
                "avg_public_known_count": mean(public_known_counts) if public_known_counts else 0.0,
                "avg_turns": mean(turn_counts) if turn_counts else 0.0,
                "survivor_counts": survivor_counts,
            }
            scenario_summaries.append(summary)
            append_jsonl(log_path, {"type": "glass_bridge_scenario_summary", **summary})

        payload = {
            "run_name": self.config["run_name"],
            "episodes_per_scenario": episodes_per_scenario,
            "scenarios": scenario_summaries,
        }
        append_jsonl(log_path, {"type": "glass_bridge_run_summary", **payload})
        return payload
