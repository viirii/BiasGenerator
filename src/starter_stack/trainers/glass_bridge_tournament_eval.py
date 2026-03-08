from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from openenv_glass_bridge.client import OpenEnvGlassBridgeClient
from openenv_glass_bridge.models import AgentAction, ResetRequest, StepRequest
from starter_stack.config import ensure_run_dirs
from starter_stack.envs.glass_bridge.glass_bridge_tournament_env import GlassBridgeTournamentEnv
from starter_stack.logging_utils import append_jsonl
from starter_stack.policies.glass_bridge import (
    build_tournament_glass_bridge_population,
)


class GlassBridgeTournamentEvaluator:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.paths = ensure_run_dirs(config)

    def run(self) -> dict[str, Any]:
        env_cfg = self.config["env"]
        eval_cfg = self.config["evaluation"]
        strategy_cfg = self.config["strategy"]
        adaptation_cfg = dict(self.config.get("adaptation", {}))
        learning_model = str(self.config.get("learning_model", adaptation_cfg.get("kind", "unknown")))
        rollout_cfg = self.config.get("rollout", {})

        games = int(eval_cfg.get("games", 1))
        base_seed = int(self.config.get("seed", 0))
        initial_players = int(env_cfg.get("initial_players", GlassBridgeTournamentEnv.DEFAULT_INITIAL_PLAYERS))
        first_round_num_steps = int(
            env_cfg.get("first_round_num_steps", GlassBridgeTournamentEnv.DEFAULT_FIRST_ROUND_NUM_STEPS)
        )
        max_rounds = int(env_cfg["max_rounds"])
        max_turns = self._resolve_max_turns(
            rollout_cfg=rollout_cfg,
            initial_players=initial_players,
            max_rounds=max_rounds,
        )
        transport = str(env_cfg.get("transport", "inprocess"))
        log_path = str(self.paths["logs"] / f"{self.config['run_name']}.jsonl")
        winners_csv_path = self.paths["logs"] / f"{self.config['run_name']}_winners.csv"
        winner_positions_csv_path = self.paths["logs"] / f"{self.config['run_name']}_winner_positions.csv"

        game_summaries = []
        winner_strategy_counts: Counter[str] = Counter()
        share_rates = list(strategy_cfg["share_rates"])
        truth_rates = list(strategy_cfg["truth_rates"])
        llm_model_pool = [str(model_name) for model_name in strategy_cfg.get("llm_model_pool", ["qwen3.5"])]
        llm_model_paths = dict(strategy_cfg.get("llm_model_paths", {}))
        # For LLM agents, start everyone at 0.5/0.5; each agent can evolve over the game.
        if any(str(m).lower() not in ("none", "null", "") for m in llm_model_pool):
            share_rates = [0.5]
            truth_rates = [0.5]

        with self._maybe_run_openenv_server(env_cfg):
            for game_idx in range(games):
                print(f"Game {game_idx + 1}/{games} started.", flush=True)
                game_seed = base_seed + (game_idx * 10_000)
                if transport == "openenv":
                    result = self._run_game_openenv(
                        env_cfg=env_cfg,
                        seed=game_seed,
                        initial_players=initial_players,
                        first_round_num_steps=first_round_num_steps,
                        max_rounds=max_rounds,
                        max_turns=max_turns,
                        share_rates=share_rates,
                        truth_rates=truth_rates,
                        llm_model_pool=llm_model_pool,
                        adaptation_config=adaptation_cfg,
                        llm_model_paths=llm_model_paths,
                    )
                else:
                    env = GlassBridgeTournamentEnv(
                        seed=game_seed,
                        max_rounds=max_rounds,
                        initial_players=initial_players,
                        first_round_num_steps=first_round_num_steps,
                        share_rates=share_rates,
                        truth_rates=truth_rates,
                        llm_model_pool=llm_model_pool,
                    )
                    result = self._run_game(
                        env=env,
                        seed=game_seed,
                        max_turns=max_turns,
                        adaptation_config=adaptation_cfg,
                        llm_model_paths=llm_model_paths,
                    )
                winner_strategy = result["winner_strategy"].get("label", "unknown")
                winner_strategy_counts[winner_strategy] += 1

                strategy_results = self._summarize_strategy_results(
                    strategy_profiles=result["strategy_profiles"],
                    cumulative_stats=result["cumulative_stats"],
                    winner=result["winner"],
                )
                first_round_info = result["round_history"][0] if result["round_history"] else {}
                first_round_order = list(first_round_info.get("order", []))
                first_round_position_map = {
                    agent_name: position
                    for position, agent_name in enumerate(first_round_order)
                }
                game_summary = {
                    "game_idx": game_idx,
                    "seed": game_seed,
                    "learning_model": learning_model,
                    "winner": result["winner"],
                    "winner_strategy": result["winner_strategy"],
                    "winner_model_name": result["winner_strategy"].get("model_name"),
                    "winner_starting_policy": result["strategy_profiles"].get(result["winner"] or "", {}),
                    "winner_policy_history": result["policy_history"].get(result["winner"] or "", []),
                    "winner_first_round_position": first_round_position_map.get(result["winner"]),
                    "initial_players": initial_players,
                    "max_turns": max_turns,
                    "rounds_played": result["rounds_played"],
                    "first_round_order": first_round_order,
                    "first_round_position_map": first_round_position_map,
                    "round_num_steps": [round_info["round_num_steps"] for round_info in result["round_history"]],
                    "round_survivor_counts": [len(round_info["survivors"]) for round_info in result["round_history"]],
                    "round_trade_counts": [
                        {
                            "offers_made": round_info.get("trade_summary", {}).get("offers_made", 0),
                            "offers_accepted": round_info.get("trade_summary", {}).get("offers_accepted", 0),
                            "offers_rejected": round_info.get("trade_summary", {}).get("offers_rejected", 0),
                        }
                        for round_info in result["round_history"]
                    ],
                    "round_history": result["round_history"],
                    "strategy_profiles": result["strategy_profiles"],
                    "strategy_results": strategy_results,
                    "cumulative_stats": result["cumulative_stats"],
                }
                game_summaries.append(game_summary)
                append_jsonl(log_path, {"type": "glass_bridge_tournament_game", **game_summary})

        self._write_winners_csv(winners_csv_path, game_summaries)
        self._write_winner_positions_csv(
            path=winner_positions_csv_path,
            game_summaries=game_summaries,
            initial_players=initial_players,
        )
        payload = {
            "run_name": self.config["run_name"],
            "games": games,
            "learning_model": learning_model,
            "initial_players": initial_players,
            "max_turns": max_turns,
            "winner_strategy_counts": dict(winner_strategy_counts),
            "winners_csv": str(winners_csv_path),
            "winner_positions_csv": str(winner_positions_csv_path),
            "games_summary": game_summaries,
        }
        append_jsonl(log_path, {"type": "glass_bridge_tournament_run_summary", **payload})
        return payload

    def _resolve_max_turns(
        self,
        rollout_cfg: dict[str, Any],
        initial_players: int,
        max_rounds: int,
    ) -> int:
        configured = rollout_cfg.get("max_turns")
        if configured is not None and int(configured) > 0:
            return int(configured)

        # Each move attempt is paired with one communication phase. A generous cap
        # of 8 * max_rounds * initial_players^2 keeps configs simple while still
        # scaling automatically with tournament size.
        return 8 * max_rounds * initial_players * initial_players

    def _run_game(
        self,
        env: GlassBridgeTournamentEnv,
        seed: int,
        max_turns: int,
        adaptation_config: dict[str, Any] | None = None,
        llm_model_paths: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        result = env.reset(seed=seed)
        policies = build_tournament_glass_bridge_population(
            result["info"]["strategy_profiles"],
            seed=seed,
            adaptation_config=adaptation_config,
            llm_model_paths=llm_model_paths or {},
        )
        turn_idx = 0
        seen_rounds: set[int] = set()
        policy_history: dict[str, list[dict[str, Any]]] = {agent_name: [] for agent_name in policies}

        while not result["done"] and turn_idx < max_turns:
            self._record_policy_history(
                observations=result["observations"],
                policies=policies,
                seen_rounds=seen_rounds,
                policy_history=policy_history,
            )
            actions = self._select_actions(result["observations"], policies)
            result = env.step(actions)
            turn_idx += 1

        if not result["done"]:
            raise RuntimeError(f"Tournament game did not terminate within {max_turns} turns")

        info = result["info"]
        return {
            "winner": info["winner"],
            "winner_strategy": info["winner_strategy"],
            "rounds_played": info["round_idx"],
            "round_history": info["round_history"],
            "strategy_profiles": info["strategy_profiles"],
            "cumulative_stats": info["cumulative_stats"],
            "policy_history": policy_history,
            "turns": turn_idx,
        }

    def _run_game_openenv(
        self,
        env_cfg: dict[str, Any],
        seed: int,
        initial_players: int,
        first_round_num_steps: int,
        max_rounds: int,
        max_turns: int,
        share_rates: list[float],
        truth_rates: list[float],
        llm_model_pool: list[str],
        adaptation_config: dict[str, Any] | None = None,
        llm_model_paths: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        client = OpenEnvGlassBridgeClient(
            base_url=str(env_cfg["base_url"]),
            timeout_s=float(env_cfg.get("timeout_s", 30.0)),
        )
        seen_rounds: set[int] = set()
        try:
            reset_response = client.reset(
                ResetRequest(
                    seed=seed,
                    initial_players=initial_players,
                    first_round_num_steps=first_round_num_steps,
                    max_rounds=max_rounds,
                    share_rates=share_rates,
                    truth_rates=truth_rates,
                    llm_model_pool=llm_model_pool,
                )
            )
            result = reset_response.result
            info = result.info.model_dump(mode="python")
            policies = build_tournament_glass_bridge_population(
                info.get("strategy_profiles", {}),
                seed=seed,
                adaptation_config=adaptation_config,
                llm_model_paths=llm_model_paths or {},
            )
            policy_history: dict[str, list[dict[str, Any]]] = {agent_name: [] for agent_name in policies}
            turn_idx = 0

            while not result.done and turn_idx < max_turns:
                observations = {
                    agent_name: observation.model_dump(mode="python")
                    for agent_name, observation in result.observations.items()
                }
                self._record_policy_history(
                    observations=observations,
                    policies=policies,
                    seen_rounds=seen_rounds,
                    policy_history=policy_history,
                )
                raw_actions = self._select_actions(observations, policies)
                actions = {
                    agent_name: AgentAction.from_policy_output(action)
                    for agent_name, action in raw_actions.items()
                }
                step_response = client.step(
                    StepRequest(session_id=reset_response.session_id, actions=actions)
                )
                result = step_response.result
                turn_idx += 1

            if not result.done:
                raise RuntimeError(f"Tournament game did not terminate within {max_turns} turns")

            info = result.info.model_dump(mode="python")
            return {
                "winner": info["winner"],
                "winner_strategy": info["winner_strategy"],
                "rounds_played": info["round_idx"],
                "round_history": info["round_history"],
                "strategy_profiles": info["strategy_profiles"],
                "cumulative_stats": info["cumulative_stats"],
                "policy_history": policy_history,
                "turns": turn_idx,
            }
        finally:
            client.close()

    def _record_policy_history(
        self,
        observations: dict[str, dict[str, Any]],
        policies: dict[str, Any],
        seen_rounds: set[int],
        policy_history: dict[str, list[dict[str, Any]]],
    ) -> None:
        sample_observation = next(iter(observations.values()), None)
        if sample_observation is None:
            return
        if sample_observation.get("phase") != GlassBridgeTournamentEnv.PHASE_COMMUNICATION_OFFER:
            return

        round_idx = int(sample_observation.get("round_idx", 0))
        if round_idx in seen_rounds:
            return
        seen_rounds.add(round_idx)

        for agent_name, observation in observations.items():
            policy = policies[agent_name]
            snapshot_fn = getattr(policy, "adaptive_policy_snapshot", None)
            if callable(snapshot_fn):
                policy_history[agent_name].append(snapshot_fn(observation))

    @contextmanager
    def _maybe_run_openenv_server(self, env_cfg: dict[str, Any]):
        if str(env_cfg.get("transport", "inprocess")) != "openenv":
            yield
            return

        base_url = str(env_cfg["base_url"])
        auto_start = bool(env_cfg.get("auto_start_server", False))
        if not auto_start:
            self._wait_for_server(base_url)
            yield
            return

        if self._server_is_healthy(base_url):
            yield
            return

        parsed = urlparse(base_url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        repo_root = Path(__file__).resolve().parents[3]
        env = os.environ.copy()
        pythonpath = str(repo_root / "src")
        existing_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = pythonpath if not existing_pythonpath else f"{pythonpath}{os.pathsep}{existing_pythonpath}"

        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "openenv_glass_bridge.server.app:app",
                "--host",
                host,
                "--port",
                str(port),
            ],
            cwd=repo_root,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            self._wait_for_server(base_url)
            yield
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)

    def _wait_for_server(self, base_url: str, timeout_s: float = 10.0) -> None:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self._server_is_healthy(base_url):
                return
            time.sleep(0.2)
        raise RuntimeError(f"OpenEnv server did not become healthy at {base_url}")

    @staticmethod
    def _server_is_healthy(base_url: str) -> bool:
        try:
            response = requests.get(f"{base_url.rstrip('/')}/health", timeout=1.0)
            return response.ok
        except requests.RequestException:
            return False

    def _select_actions(
        self,
        observations: dict[str, dict[str, Any]],
        policies: dict[str, Any],
    ) -> dict[str, str]:
        actions: dict[str, str] = {}
        for agent_name, observation in observations.items():
            action = policies[agent_name].select_action(observation)
            legal = observation.get("legal_actions", [])
            if not legal:
                raise RuntimeError(f"No legal actions for {agent_name}")

            if isinstance(action, dict):
                action_type = action.get("type", "NOOP")
                if any(
                    leg.get("type") == action_type for leg in legal if isinstance(leg, dict)
                ):
                    actions[agent_name] = action
                else:
                    actions[agent_name] = legal[0]
                continue

            if action not in legal:
                actions[agent_name] = legal[0]
                continue
            actions[agent_name] = action
        return actions

    def _summarize_strategy_results(
        self,
        strategy_profiles: dict[str, dict[str, Any]],
        cumulative_stats: dict[str, dict[str, int]],
        winner: str | None,
    ) -> dict[str, dict[str, Any]]:
        results: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"agents": [], "rounds_survived": [], "total_progress": [], "wins": 0}
        )

        for agent_name, profile in strategy_profiles.items():
            label = profile.get("label", "unknown")
            results[label]["agents"].append(agent_name)
            results[label]["rounds_survived"].append(cumulative_stats[agent_name]["rounds_survived"])
            results[label]["total_progress"].append(cumulative_stats[agent_name]["total_progress"])
            if agent_name == winner:
                results[label]["wins"] += 1

        return dict(results)

    def _write_winners_csv(self, path: Path, game_summaries: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "game_idx",
                    "seed",
                    "learning_model",
                    "winner",
                    "winner_first_round_position",
                    "winner_label",
                    "winner_model_name",
                    "winner_share_rate",
                    "winner_truth_rate",
                    "winner_starting_policy",
                    "rounds_played",
                ],
            )
            writer.writeheader()
            for summary in game_summaries:
                winner_strategy = summary.get("winner_strategy", {})
                writer.writerow(
                    {
                        "game_idx": summary.get("game_idx"),
                        "seed": summary.get("seed"),
                        "learning_model": summary.get("learning_model"),
                        "winner": summary.get("winner"),
                        "winner_first_round_position": summary.get("winner_first_round_position"),
                        "winner_label": winner_strategy.get("label", "unknown"),
                        "winner_model_name": summary.get("winner_model_name"),
                        "winner_share_rate": winner_strategy.get("share_rate"),
                        "winner_truth_rate": winner_strategy.get("truth_rate"),
                        "winner_starting_policy": json.dumps(
                            summary.get("winner_starting_policy", {}),
                            sort_keys=True,
                        ),
                        "rounds_played": summary.get("rounds_played"),
                    }
                )

    def _write_winner_positions_csv(
        self,
        path: Path,
        game_summaries: list[dict[str, Any]],
        initial_players: int,
    ) -> None:
        counts = Counter(
            summary["winner_first_round_position"]
            for summary in game_summaries
            if summary.get("winner_first_round_position") is not None
        )
        total_games = len(game_summaries)

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "winner_first_round_position",
                    "wins",
                    "total_games",
                    "win_rate",
                ],
            )
            writer.writeheader()
            for position in range(initial_players):
                wins = counts.get(position, 0)
                writer.writerow(
                    {
                        "winner_first_round_position": position,
                        "wins": wins,
                        "total_games": total_games,
                        "win_rate": wins / total_games if total_games else 0.0,
                    }
                )
