from __future__ import annotations

import argparse
import json

from starter_stack.config import load_config
from starter_stack.trainers import GlassBridgeEvaluator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    result = GlassBridgeEvaluator(config).run()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
