from __future__ import annotations

import argparse
import json

from glass_bridge.client import OpenEnvGlassBridgeClient
from glass_bridge.models import AgentAction, ResetRequest, StepRequest
from glass_bridge.policies import build_tournament_glass_bridge_population


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--initial-players", type=int, default=16)
    parser.add_argument("--first-round-steps", type=int, default=18)
    parser.add_argument("--max-rounds", type=int, default=25)
    parser.add_argument("--max-turns", type=int, default=9600)
    parser.add_argument(
        "--adaptation-kind",
        choices=["none", "truth_scaled_by_reputation"],
        default="truth_scaled_by_reputation",
    )
    args = parser.parse_args()

    client = OpenEnvGlassBridgeClient(base_url=args.base_url)
    try:
        reset_response = client.reset(
            ResetRequest(
                seed=args.seed,
                initial_players=args.initial_players,
                first_round_num_steps=args.first_round_steps,
                max_rounds=args.max_rounds,
                share_rates=[0.0, 0.25, 0.5, 0.75, 1.0],
                truth_rates=[0.0, 0.25, 0.5, 0.75, 1.0],
                llm_model_pool=["qwen3.5"],
            )
        )
        result = reset_response.result
        policies = build_tournament_glass_bridge_population(
            result.info.strategy_profiles,
            seed=args.seed,
            adaptation_config={"kind": args.adaptation_kind},
        )
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
