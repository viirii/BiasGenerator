from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from starter_stack.config import ensure_run_dirs, load_config
from starter_stack.device import pick_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    paths = ensure_run_dirs(config)
    device = pick_device()

    model = torch.nn.Sequential(
        torch.nn.Linear(16, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 2),
    ).to(device)

    x = torch.randn(256, 16, device=device)
    y = model(x)
    loss = (y ** 2).mean()
    loss.backward()

    ckpt = Path(config["checkpoint"]["dir"]) / f"gpu_smoke_{config['checkpoint']['name']}"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "device": device}, ckpt)

    print(
        json.dumps(
            {
                "device": device,
                "loss": float(loss.detach().cpu().item()),
                "checkpoint": str(ckpt),
                "artifacts_dir": str(paths["artifacts_dir"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
