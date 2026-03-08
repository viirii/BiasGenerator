from __future__ import annotations

import random


class RandomPolicy:
    name = "random"

    def __init__(self, seed: int = 0):
        self._rng = random.Random(seed)

    def select_action(self, observation: dict) -> int:
        legal = observation.get("legal_actions", [])
        if not legal:
            raise RuntimeError("No legal actions available")
        return self._rng.choice(legal)
