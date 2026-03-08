from __future__ import annotations

import argparse
import json

from openenv_glass_bridge.client import OpenEnvGlassBridgeClient
from openenv_glass_bridge.models import AgentAction, ResetRequest, StepRequest, StrategyProfile
from starter_stack.envs.glass_bridge.glass_bridge_tournament_env import GlassBridgeTournamentEnv
from starter_stack.policies.glass_bridge import (
    assign_tournament_strategy_profiles,
    build_tournament_glass_bridge_population,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--initial-players", type=int, default=10)
    parser.add_argument("--first-round-steps", type=int, default=18)
    parser.add_argument("--max-rounds", type=int, default=25)
    parser.add_argument("--max-turns", type=int, default=9600)
    parser.add_argument(
        "--adaptation-kind",
        choices=["none", "truth_scaled_by_reputation"],
        default="truth_scaled_by_reputation",
    )
    args = parser.parse_args()

    agent_names = [GlassBridgeTournamentEnv.agent_name(i) for i in range(args.initial_players)]
    raw_profiles = assign_tournament_strategy_profiles(
        agent_names=agent_names,
        seed=args.seed,
        share_rates=[0.0, 0.25, 0.5, 0.75, 1.0],
        truth_rates=[0.0, 0.25, 0.5, 0.75, 1.0],
    )
    profiles = {
        agent_name: StrategyProfile.model_validate(profile)
        for agent_name, profile in raw_profiles.items()
    }
    policies = build_tournament_glass_bridge_population(
        raw_profiles,
        seed=args.seed,
        adaptation_config={"kind": args.adaptation_kind},
    )

    client = OpenEnvGlassBridgeClient(base_url=args.base_url)
    try:
        reset_response = client.reset(
            ResetRequest(
                seed=args.seed,
                initial_players=args.initial_players,
                first_round_num_steps=args.first_round_steps,
                max_rounds=args.max_rounds,
                strategy_profiles=profiles,
            )
        )
        result = reset_response.result
        turn_idx = 0

        while not result.done and turn_idx < args.max_turns:
            actions = {}
            for agent_name, observation in result.observations.items():
                policy_action = policies[agent_name].select_action(observation.model_dump(mode="python"))
                actions[agent_name] = AgentAction.from_policy_output(policy_action)

            step_response = client.step(StepRequest(session_id=reset_response.session_id, actions=actions))
            result = step_response.result
            turn_idx += 1

        if not result.done:
            raise RuntimeError(f"Environment did not terminate within {args.max_turns} turns")

        payload = {
            "session_id": reset_response.session_id,
            "turns": turn_idx,
            "winner": result.info.winner,
            "winner_strategy": result.info.winner_strategy,
            "rounds_played": result.info.round_idx,
            "events": result.info.events,
            "final_phase": result.info.phase,
            "active_agents": result.info.active_agents,
        }
        print(json.dumps(payload, indent=2))
    finally:
        client.close()


if __name__ == "__main__":
    main()
