from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from starter_stack.config import load_config
from starter_stack.trainers import GlassBridgeTournamentEvaluator


def _env_default(name: str, fallback: str) -> str:
    return os.environ.get(name, fallback)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=_env_default("GLASS_BRIDGE_CONFIG", "configs/glass_bridge_tournament_northflank.yaml"),
    )
    parser.add_argument(
        "--games",
        type=int,
        default=int(_env_default("GLASS_BRIDGE_GAMES", "1")),
    )
    parser.add_argument(
        "--learning-model",
        choices=["none", "truth_scaled_by_reputation"],
        default=_env_default("GLASS_BRIDGE_LEARNING_MODEL", "truth_scaled_by_reputation"),
    )
    parser.add_argument(
        "--run-name",
        default=_env_default("GLASS_BRIDGE_RUN_NAME", ""),
    )
    parser.add_argument(
        "--base-url",
        default=_env_default("GLASS_BRIDGE_BASE_URL", ""),
    )
    parser.add_argument(
        "--auto-start-server",
        choices=["true", "false"],
        default=_env_default("GLASS_BRIDGE_AUTO_START_SERVER", ""),
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=float(_env_default("GLASS_BRIDGE_TIMEOUT_S", "0")),
    )
    parser.add_argument(
        "--llm-model-pool",
        default=_env_default("GLASS_BRIDGE_LLM_MODEL_POOL", ""),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    config["env"] = dict(config["env"])
    config["evaluation"] = dict(config["evaluation"])
    config["strategy"] = dict(config["strategy"])
    config["adaptation"] = {"kind": args.learning_model}
    config["learning_model"] = args.learning_model
    config["evaluation"]["games"] = int(args.games)

    if args.base_url:
        config["env"]["base_url"] = args.base_url
    if args.auto_start_server:
        config["env"]["auto_start_server"] = args.auto_start_server.lower() == "true"
    if args.timeout_s > 0:
        config["env"]["timeout_s"] = float(args.timeout_s)
    if args.llm_model_pool:
        config["strategy"]["llm_model_pool"] = [
            model_name.strip()
            for model_name in args.llm_model_pool.split(",")
            if model_name.strip()
        ]

    if args.run_name:
        config["run_name"] = args.run_name
    else:
        config["run_name"] = (
            f"{Path(args.config).stem}_{args.learning_model}_{int(args.games)}games"
        )

    result = GlassBridgeTournamentEvaluator(config).run()
    print(
        json.dumps(
            {
                "run_name": result["run_name"],
                "games": result["games"],
                "learning_model": result["learning_model"],
                "winner_strategy_counts": result["winner_strategy_counts"],
                "winners_csv": result["winners_csv"],
                "winner_positions_csv": result["winner_positions_csv"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
