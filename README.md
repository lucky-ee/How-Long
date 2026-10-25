# How Long?

How long will a San Francisco 311 request take to close? Survival models trained on
[DataSF 311 Cases](https://data.sfgov.org/d/vw6y-z8j6), served with FastAPI behind a static web frontend.

## Setup

```sh
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
brew install libomp            # macOS only: xgboost needs OpenMP
```

## Pipeline

```sh
python -m src.ingest           # download raw cases into data/raw   (or: python -m src.synth for fake data)
python -m src.clean            # -> data/clean/cases.parquet
python -m src.train            # -> data/artifacts/{model.joblib,report.json,options.json}
```

## Run the app

```sh
uvicorn src.api:app --reload   # site at http://localhost:8000, API docs at /docs
```

| Endpoint | |
|---|---|
| `GET /api/health` | whether a trained model is loaded |
| `GET /api/options` | request types, sources, districts and current backlog for the form |
| `GET /api/report` | model metrics, effects and neighborhood comparison |
| `POST /api/predict` | survival curve and median wait for one request |

Environment: `HOWLONG_ARTIFACTS` (artifact directory), `HOWLONG_CORS_ORIGINS` (comma-separated, default `*`).

## Deploy

```sh
python -m src.train
docker build -t howlong . && docker run -p 8000:8000 howlong
```

The image honours `$PORT`, so it runs as-is on Render, Fly.io, Railway or Cloud Run.
The frontend can also be hosted on its own; see `frontend/README.md`.

## Tests

```sh
PYTHONPATH=. pytest -q
```
