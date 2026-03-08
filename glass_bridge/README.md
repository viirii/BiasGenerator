---
title: Glass Bridge
emoji: 🌉
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
---

# Glass Bridge OpenEnv Package

This directory is a standalone OpenEnv/Hugging Face deployable package for the multi-round Glass Bridge tournament.

It contains:

- the FastAPI server used by the environment runtime
- explicit action and observation models
- the tournament environment with server-side hidden state
- a lightweight scripted example client for smoke-testing the package by itself

## Local package setup

From the repo root:

```bash
python -m pip install -e ./glass_bridge
```

Or from inside this directory:

```bash
python -m pip install -e .
```

## Run locally

Start the server:

```bash
python -m glass_bridge.server.app --port 8000
```

Run the example client:

```bash
python -m glass_bridge.examples.example_usage --base-url http://127.0.0.1:8000
```

## Deploy with OpenEnv

From this directory, use the generated OpenEnv manifest:

```bash
openenv push
```

Or explicitly from the repo root:

```bash
openenv push --directory glass_bridge
```

## Notes

This package is intentionally self-contained so it can be uploaded independently of the larger research repo.
