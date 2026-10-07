# Real-Time ML Serving

A **production-style inference service** for real-time ML predictions: FastAPI + Redis feature caching + model monitoring with drift detection. The reference implementation is a fraud-detection scorer, but the serving layer is model-agnostic — swap in any sklearn-compatible classifier.

## Architecture

```
Client ──▶ FastAPI (/predict)
              │
              ├─▶ Redis feature cache (skip recompute on cache hit)
              │
              ├─▶ Model inference (XGBoost / sklearn pipeline)
              │
              └─▶ Monitoring: log prediction, latency, feature stats
                       │
                       ▼
                 Drift detector (PSI) ──▶ alert when distribution shifts
```

## Quickstart

```bash
pip install -r requirements.txt
# train a demo model
python src/train.py
# serve it
uvicorn src.app:app --reload
# score a transaction
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' \
  -d '{"amount": 250.0, "merchant_category": "travel", "hour_of_day": 14,
       "days_since_last": 2, "avg_amount_30d": 180.0}'
```

With Docker:

```bash
docker build -t ml-serving .
docker run -p 8000:8000 ml-serving
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/predict` | Score one event → `{fraud_probability, latency_ms, cached}` |
| GET | `/health` | Liveness + model version |
| GET | `/metrics` | Prediction count, p50/p99 latency, drift status |

## Project layout

```
src/
  app.py         # FastAPI service: predict, health, metrics
  model.py       # Model wrapper: load, predict, version
  features.py    # Feature engineering + Redis cache layer
  monitoring.py  # Prediction logging, PSI drift detection
  train.py       # Trains a demo XGBoost fraud model on synthetic data
Dockerfile
```

## Key design decisions

- **Redis feature cache** — repeated entities (same card/user) skip feature recomputation; big win at high QPS.
- **Model versioning** — the model artifact carries a version string; `/health` exposes it so deploys are verifiable.
- **PSI drift detection** — Population Stability Index on incoming feature distributions vs training baseline; alerts before accuracy silently degrades.
- **Structured logging** — every prediction logs features, score, latency, and model version for offline analysis.

## Built with

Python · FastAPI · Redis · XGBoost · scikit-learn · Docker · Pydantic
