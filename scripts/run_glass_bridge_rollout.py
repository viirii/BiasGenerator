from __future__ import annotations

import argparse
import json

from starter_stack.envs.glass_bridge.glass_bridge_env import GlassBridgeEnv
from starter_stack.policies.glass_bridge import (
    build_glass_bridge_population,
)
from starter_stack.trainers.glass_bridge_rollout import GlassBridgeRolloutRunner


def run_scenario(scenario: str, seed: int, max_turns: int) -> dict:
    env = GlassBridgeEnv(seed=seed)
    runner = GlassBridgeRolloutRunner(
        env=env,
        policies=build_glass_bridge_population(scenario, seed=seed),
        max_turns=max_turns,
    )
    result = runner.run_episode(seed=seed)
    payload = {
        "scenario": scenario,
        "seed": seed,
        "survivors": result["survivors"],
        "survivor_count": len(result["survivors"]),
        "rewards": result["rewards"],
        "progress": result["progress"],
        "public_known_count": result["public_known_count"],
        "turns": result["turns"],
    }
    return {
        "summary": payload,
        "trace": result["trace"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        choices=["all", "never_share", "always_share", "mixed"],
        default="all",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-turns", type=int, default=512)
    args = parser.parse_args()

    scenarios = (
        ["never_share", "always_share", "mixed"]
        if args.scenario == "all"
        else [args.scenario]
    )

    summaries = []
    for offset, scenario in enumerate(scenarios):
        run_seed = args.seed + offset
        result = run_scenario(scenario, seed=run_seed, max_turns=args.max_turns)
        print(f"=== scenario={scenario} seed={run_seed} ===")
        for line in result["trace"]:
            print(line)
        print(json.dumps(result["summary"], indent=2))
        summaries.append(result["summary"])

    if len(summaries) > 1:
        print("=== aggregate ===")
        print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
