from starter_stack.policies.glass_bridge import (
    AlwaysSharePolicy,
    NeverSharePolicy,
    RandomGlassBridgePolicy,
    TournamentGlassBridgePolicy,
    assign_tournament_strategy_profiles,
    build_tournament_glass_bridge_population,
    build_tournament_strategy_grid,
    build_glass_bridge_population,
    build_glass_bridge_policy,
)
from starter_stack.policies.random_policy import RandomPolicy


def build_policy(name: str, seed: int = 0):
    if name == "random":
        return RandomPolicy(seed=seed)
    raise ValueError(f"Unknown policy: {name}")


__all__ = [
    "AlwaysSharePolicy",
    "NeverSharePolicy",
    "RandomGlassBridgePolicy",
    "RandomPolicy",
    "TournamentGlassBridgePolicy",
    "assign_tournament_strategy_profiles",
    "build_glass_bridge_population",
    "build_glass_bridge_policy",
    "build_tournament_glass_bridge_population",
    "build_tournament_strategy_grid",
    "build_policy",
]
