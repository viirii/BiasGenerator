__all__ = [
    "GlassBridgeEvaluator",
    "GlassBridgeRolloutRunner",
    "GlassBridgeTournamentEvaluator",
]


def __getattr__(name: str):
    if name == "GlassBridgeRolloutRunner":
        from starter_stack.trainers.glass_bridge_rollout import GlassBridgeRolloutRunner

        return GlassBridgeRolloutRunner
    if name == "GlassBridgeEvaluator":
        from starter_stack.trainers.glass_bridge_eval import GlassBridgeEvaluator

        return GlassBridgeEvaluator
    if name == "GlassBridgeTournamentEvaluator":
        from starter_stack.trainers.glass_bridge_tournament_eval import GlassBridgeTournamentEvaluator

        return GlassBridgeTournamentEvaluator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
