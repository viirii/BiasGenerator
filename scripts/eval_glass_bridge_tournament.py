from __future__ import annotations

import argparse
import json
from pathlib import Path

from starter_stack.config import load_config
from starter_stack.trainers import GlassBridgeTournamentEvaluator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--games", type=int)
    parser.add_argument(
        "--learning-model",
        choices=["none", "truth_scaled_by_reputation"],
    )
    parser.add_argument("--run-name")
    parser.add_argument("--base-url")
    parser.add_argument("--transport", choices=["inprocess", "openenv"])
    parser.add_argument("--auto-start-server", choices=["true", "false"])
    parser.add_argument("--timeout-s", type=float)
    parser.add_argument("--llm-model-pool")
    args = parser.parse_args()

    config = load_config(args.config)
    config["env"] = dict(config["env"])
    config["evaluation"] = dict(config["evaluation"])
    config["strategy"] = dict(config["strategy"])

    if args.games is not None:
        config["evaluation"]["games"] = int(args.games)

    if args.learning_model is not None:
        config["adaptation"] = {"kind": args.learning_model}
        config["learning_model"] = args.learning_model

    if args.base_url is not None:
        config["env"]["base_url"] = args.base_url

    if args.transport is not None:
        config["env"]["transport"] = args.transport

    if args.auto_start_server is not None:
        config["env"]["auto_start_server"] = args.auto_start_server.lower() == "true"

    if args.timeout_s is not None:
        config["env"]["timeout_s"] = float(args.timeout_s)

    if args.llm_model_pool is not None:
        config["strategy"]["llm_model_pool"] = [
            model_name.strip()
            for model_name in args.llm_model_pool.split(",")
            if model_name.strip()
        ]

    if args.run_name is not None:
        config["run_name"] = args.run_name
    elif args.games is not None or args.learning_model is not None:
        learning_model = str(config.get("learning_model", config.get("adaptation", {}).get("kind", "default")))
        config["run_name"] = f"{Path(args.config).stem}_{learning_model}_{int(config['evaluation']['games'])}games"

    result = GlassBridgeTournamentEvaluator(config).run()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
