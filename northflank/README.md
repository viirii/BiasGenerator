# Northflank notes

Suggested first remote command:

```bash
python scripts/eval_glass_bridge_tournament.py --config configs/glass_bridge_tournament_remote.yaml
```

This workload does not require an external game server. It can run directly as a batch job from the repo image.

If you want a quick device check before the tournament run:

```bash
python scripts/smoke_gpu.py --config configs/glass_bridge_tournament_remote.yaml
```
