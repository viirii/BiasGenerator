FROM python:3.11-slim

WORKDIR /app
COPY requirements/local.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY . /app
RUN pip install -e .
CMD ["python", "scripts/eval_glass_bridge_tournament.py", "--config", "configs/glass_bridge_tournament_local.yaml"]
