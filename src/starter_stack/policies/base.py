from __future__ import annotations

from typing import Protocol


class Policy(Protocol):
    name: str

    def select_action(self, observation: dict) -> int: ...
