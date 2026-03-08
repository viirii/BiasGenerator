from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Transition:
    episode_idx: int
    step_idx: int
    action_id: int
    action_name: str
    reward: float | None
    done: bool
    legal_actions: list[int] = field(default_factory=list)
    observation: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EpisodeSummary:
    episode_idx: int
    total_reward: float
    steps: int
    done: bool
    final_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
