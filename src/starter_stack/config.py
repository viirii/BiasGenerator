from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_run_dirs(config: dict[str, Any]) -> dict[str, Path]:
    artifacts_dir = Path(config.get("artifacts_dir", "artifacts"))
    checkpoints = Path(config["checkpoint"]["dir"])
    logs = Path(config["logging"]["dir"])

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    return {
        "artifacts_dir": artifacts_dir,
        "checkpoints": checkpoints,
        "logs": logs,
    }
