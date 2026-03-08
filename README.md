# BiasGenerator

This repo is focused on Glass Bridge simulation and tournament evaluation.

Included:

- a single-round `GlassBridgeEnv`
- a multi-round `GlassBridgeTournamentEnv`
- an OpenEnv-style HTTP server/client wrapper for the tournament game
- scripted baseline policies
- config-driven local and remote evaluation scripts
- Dockerfiles for CPU and GPU-backed runs
- artifact and log output under `artifacts/`

## Local setup

Create a Python 3.11 virtual environment and install the local dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements/local.txt
pip install -e .
```

## Local runs

Single-round Glass Bridge evaluation:

```bash
python scripts/eval_glass_bridge.py --config configs/glass_bridge_local.yaml
```

Tournament evaluation:

```bash
python scripts/eval_glass_bridge_tournament.py --config configs/glass_bridge_tournament_local.yaml
```

The tournament configs now use the OpenEnv-style transport by default. For local runs, the evaluator can auto-start the local Glass Bridge server and talk to it over `http://127.0.0.1:8000`, so the same command shape can later be reused remotely.

## OpenEnv-style server mode

Start the local server:

```bash
PYTHONPATH=src uvicorn openenv_glass_bridge.server.app:app --host 127.0.0.1 --port 8000
```

Run the example client rollout against that server:

```bash
PYTHONPATH=src python -m openenv_glass_bridge.examples.example_usage --base-url http://127.0.0.1:8000
```

In this mode, all authoritative hidden state stays server-side. The client only sends actions and receives filtered per-agent observations.

## Standalone OpenEnv package

If you want to deploy through the `openenv init` style package layout, use the standalone package under `glass_bridge/`.

```bash
python -m pip install -e ./glass_bridge
openenv push --directory glass_bridge
```

Quick rollout trace:

```bash
python scripts/run_glass_bridge_rollout.py
```

Optional device smoke check:

```bash
python scripts/smoke_gpu.py --config configs/glass_bridge_tournament_remote.yaml
```

## Remote / Northflank

For fast Northflank runs, keep both the evaluator and the OpenEnv server on the same machine:

```bash
python scripts/run_northflank_glass_bridge_job.py
```

This config uses:
- `transport: openenv`
- `base_url: http://127.0.0.1:8000`
- `auto_start_server: true`

so the evaluator auto-starts the local OpenEnv server and talks to it over localhost.

If you want to validate the public Hugging Face deployment instead, use `configs/glass_bridge_tournament_remote.yaml` and point it at your HF Space URL.

`Dockerfile.gpu` is a good starting point for a Northflank Job. If you SSH into a Northflank GPU instance directly, run the same command there.

The Northflank job wrapper supports:

```bash
python scripts/run_northflank_glass_bridge_job.py --games 100 --learning-model none
python scripts/run_northflank_glass_bridge_job.py --games 100 --learning-model truth_scaled_by_reputation
```

## Useful commands

```bash
make eval-glass-bridge
make eval-glass-bridge-tournament
make smoke-gpu
```
