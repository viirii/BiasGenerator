# Northflank notes

Recommended fast path on a Northflank GPU instance:

```bash
python scripts/run_northflank_glass_bridge_job.py
```

This keeps both the evaluator and the OpenEnv server on the same machine. The evaluator auto-starts the local server on `127.0.0.1:8000`, so each game step stays local instead of going over the public internet to Hugging Face.

If you SSH into the Northflank instance directly, the same command works there too:

```bash
python scripts/run_northflank_glass_bridge_job.py
```

If you want a quick device check before the tournament run:

```bash
python scripts/smoke_gpu.py --config configs/glass_bridge_tournament_northflank.yaml
```

If you specifically want to test against the public Hugging Face Space instead of running locally on Northflank, use `configs/glass_bridge_tournament_remote.yaml` and point its `env.base_url` at the deployed HF Space.

Useful overrides for the job script:

```bash
python scripts/run_northflank_glass_bridge_job.py --games 100 --learning-model none
python scripts/run_northflank_glass_bridge_job.py --games 100 --learning-model truth_scaled_by_reputation
```

The same values can be provided through environment variables in the Northflank UI:

```bash
GLASS_BRIDGE_GAMES=100
GLASS_BRIDGE_LEARNING_MODEL=none
GLASS_BRIDGE_CONFIG=configs/glass_bridge_tournament_northflank.yaml
```
