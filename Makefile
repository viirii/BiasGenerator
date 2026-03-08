.PHONY: smoke-gpu eval-glass-bridge eval-glass-bridge-tournament serve-openenv-glass-bridge example-openenv-glass-bridge

smoke-gpu:
	python scripts/smoke_gpu.py --config configs/glass_bridge_tournament_northflank.yaml

eval-glass-bridge:
	python scripts/eval_glass_bridge.py --config configs/glass_bridge_local.yaml

eval-glass-bridge-tournament:
	python scripts/eval_glass_bridge_tournament.py --config configs/glass_bridge_tournament_local.yaml

serve-openenv-glass-bridge:
	PYTHONPATH=src uvicorn openenv_glass_bridge.server.app:app --host 127.0.0.1 --port 8000

example-openenv-glass-bridge:
	PYTHONPATH=src python -m openenv_glass_bridge.examples.example_usage --base-url http://127.0.0.1:8000
