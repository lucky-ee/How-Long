# How Long?

How long will a San Francisco 311 request take to close?

**How Long?** predicts how long a 311 service request will stay open, such as a pothole, graffiti, an encampment report or a streetlight outage. It's trained on the public
[DataSF 311 Cases](https://data.sfgov.org/d/vw6y-z8j6) dataset. You describe a request (what it is,
where, how you're filing it, whether you have a photo) and the site shows a full **survival curve**,
the chance the case is still open after an hour, a day or a week, along with a median wait.
It also shows which factors make requests close faster, and how neighborhoods compare.

Under the hood it's a set of survival models (Kaplan–Meier, penalized Cox and XGBoost AFT),
compared with time-aware cross-validation and served by a small FastAPI app with a static
frontend.

---

## Contents

- [Quick start](#quick-start)
- [How it works](#how-it-works)
- [Setup](#setup)
- [The pipeline](#the-pipeline)
  - [1. Ingest](#1-ingest)
  - [2. Clean](#2-clean)
  - [3. Train](#3-train)
  - [Synthetic data](#synthetic-data)
- [Running the app](#running-the-app)
- [API reference](#api-reference)
- [Modeling details](#modeling-details)
- [Project layout](#project-layout)
- [Testing](#testing)
- [Deployment](#deployment)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

---

## Quick start

To see the app running in a few minutes, use synthetic data:

```sh
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
brew install libomp              # macOS only: xgboost needs OpenMP

python -m src.synth              # 120,000 fake cases -> data/clean/cases.parquet
python -m src.train --fast       # quick, rough model
uvicorn src.api:app --reload     # open http://localhost:8000
```

For predictions from real SF data, see [The pipeline](#the-pipeline).

---

## How it works

```
DataSF SODA API ──► ingest ──► data/raw/YYYY-MM.parquet      (one file per month)
                                   │
                                 clean ──► data/clean/cases.parquet
                                   │
                                 train ──► data/artifacts/
                                   │         model.joblib   deployed model
                                   │         report.json    metrics, effects, hypothesis tests
                                   │         options.json   form choices + current backlog
                                   │         runs.db        history of every training run
                                   ▼
                       src/api.py (FastAPI) ──► /api/*  +  frontend/ at /
```

Every case is either **closed** (we know how long it took) or **still open** (we only know it has
taken *at least* this long so far). Survival models are built for exactly this kind of
partly observed data. Dropping open cases would bias predictions towards fast closures,
because the slowest cases are the ones most likely to still be open.

---

## Setup

Requirements:

- **Python 3.12**
- **macOS:** `brew install libomp` (XGBoost needs the OpenMP runtime)
- **Linux:** `libgomp1` (preinstalled on most distributions)
- A machine with plenty of RAM for a full real-data training run. A run on ~2.5 million cases used about 8.5 GB.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Dependencies in `requirements.txt` are **pinned on purpose**. `model.joblib` is a pickle, so the
environment that serves a model must match the one that trained it.

---

## The pipeline

Run all commands from the project root with the virtualenv active.

### 1. Ingest

```sh
python -m src.ingest                       # last ~3 years (default)
python -m src.ingest --since 2025-01-01    # custom start month
python -m src.ingest --force               # re-download every month, ignoring the cache
```

- Downloads cases from the DataSF SODA API **one month at a time** into `data/raw/YYYY-MM.parquet`.
- Months already on disk are **skipped**, except the two most recent months, which are always refreshed
  because their cases are still changing. If a download stops partway, run it again and
  it picks up where it left off.
- Each month is written atomically, so an interrupted run never leaves a half-written file.
- Records the fetch time in `data/raw/_manifest.json`. That moment is the cut-off for
  cases that are still open.
- Three years is roughly 2.5 million cases (about 55,000–78,000 per month).

**Get an app token.** DataSF rate-limits requests made without a token. Without one, a long
download can fail partway with `HTTP 428`. Register for a free token on the
[DataSF developer portal](https://data.sfgov.org/) and export it before ingesting:

```sh
export SODA_APP_TOKEN=your-token-here
python -m src.ingest
```

### 2. Clean

```sh
python -m src.clean
```

Turns the raw months into one tidy table, `data/clean/cases.parquet`, and writes a count of
everything it did to `data/clean/cleaning_report.json`. Cleaning:

- **Drops** test submissions, rows with no open time, duplicate case IDs, and close dates that
  are impossible (before the case opened, before July 2008, or after the fetch time).
- **Labels each case** as closed (`event = 1`) or still open (`event = 0`). Open cases get a
  duration up to the fetch time.
- **Normalizes labels**: lowercase, consistent spacing, and a few known aliases merged
  (e.g. `mobile/open311` → `mobile`).
- **Blanks** coordinates that fall outside San Francisco (rows are kept), and treats district `0`
  as unknown.
- **Flags** cases with a photo, and cases whose status notes mark them as duplicates. Duplicates
  are excluded during training, because they close almost instantly for administrative reasons.

From a real run on three years of data:

| Step | Rows |
|---|---:|
| Raw rows | 2,534,663 |
| Test submissions dropped | 1,046 |
| Invalid close dates dropped | 12,134 |
| Still-open cases kept | 96,158 |
| Duplicates flagged | 179,267 |
| **Clean rows** | **2,521,483** |

### 3. Train

```sh
python -m src.train          # full run
python -m src.train --fast   # fewer CV splits and a smaller grid, for a quick check
```

Training fits and compares every model, picks one to deploy, runs the final checks and writes
the artifacts. Each step prints a timestamped progress line. The steps are:

1. **Backlog feature.** For every case, the number of open cases of the same type in the same district at
   the moment it was filed (computed in DuckDB with an `ASOF` join).
2. **Holdout split.** The most recent 60 days are set aside and only used at the end.
3. **Proportional-hazards check.** Decides whether the Cox models should be stratified by
   category (see [Modeling details](#modeling-details)).
4. **Nested, time-ordered cross-validation.** Each model is tuned and scored across 3 outer splits,
   each with 3 inner splits for tuning.
5. **Model selection.** Picks the model to deploy (rules below).
6. **Final fit and holdout scoring.** Every model is retuned on all non-holdout data and scored once on
   the holdout.
7. **Confirmatory hypothesis tests** on the holdout.
8. **Serving model.** The chosen model is refit on all data and saved.

**How long does it take?**

| Data | Mode | Time (one laptop) |
|---|---|---|
| Synthetic, 120,000 cases | full | ~12 minutes |
| Real, ~2.5 million cases | full | several hours |
| Either | `--fast` | much shorter; use it to check that everything works |

The models cap how many rows they use (150,000 for the dense Cox model and 500,000 for the sparse
ones), but a full real-data run still repeats a lot of fitting. Use `--fast` while developing.

All artifacts are written atomically, so a running server never reads a half-written model. If
training is stopped partway, the previous model stays in place. Every completed run is also
logged to `data/artifacts/runs.db` (SQLite), with its settings, scores and the full report.

### Synthetic data

```sh
python -m src.synth
```

Generates 120,000 fake cases directly into `data/clean/cases.parquet`, **skipping ingest and
clean**. The cases come from a known model with planted effects (a photo speeds closure, Bayview is slower,
filing through the web has *no* effect, and so on), so you can check that the models recover the truth. This
is also what the test suite and the frontend's demo mode use.

> Running `python -m src.clean` after `src.synth` fails with "No raw data found". That's expected,
> because synthetic data doesn't need cleaning.

---

## Running the app

```sh
uvicorn src.api:app --reload
```

- Site: <http://localhost:8000>
- Interactive API docs: <http://localhost:8000/docs>

The app **starts even before any model exists**: `/api/health` reports what's missing and the other
endpoints return `503` until you train. It **picks up a new model automatically** when training
finishes, so there's no need to restart the server.

### The frontend

`frontend/` is a static site with no build step: plain HTML, CSS and ES modules. The API serves it
at `/` and points it at `/api` automatically. On the page you can:

- pick a request type, district, filing method, photo yes/no and filing time
- see the survival curve and median wait update as you change things
- share a prediction by URL
- browse the strongest effects and a neighborhood comparison from the training report

If the page can't reach an API, it switches to **demo mode** using the synthetic generating model,
and shows a badge saying so. See [`frontend/README.md`](frontend/README.md) to host the frontend on its own.

---

## API reference

All routes are under `/api`.

| Method | Route | Returns |
|---|---|---|
| `GET` | `/api/health` | `{"ok": true, "deployed": "linear"}`, or `{"ok": false, "detail": ...}` if not trained |
| `GET` | `/api/options` | request types, sources, districts, and the current backlog for the form |
| `GET` | `/api/report` | metrics, model selection, hypothesis tests, effects and neighborhoods |
| `POST` | `/api/predict` | the survival curve and median wait for one request |

### `POST /api/predict`

```sh
curl -X POST localhost:8000/api/predict \
  -H 'content-type: application/json' \
  -d '{
        "opened": "2026-10-23T14:00:00",
        "category": "street and sidewalk cleaning",
        "request_type": "bulky items",
        "source": "mobile",
        "supervisor_district": 9,
        "has_photo": true
      }'
```

| Field | Type | Notes |
|---|---|---|
| `opened` | datetime | Filing time. Times without a time zone are read as San Francisco local time. |
| `category` | string | e.g. `"graffiti"` |
| `request_type` | string | e.g. `"graffiti on pole"` |
| `request_details` | string or null | optional |
| `source` | string | e.g. `"mobile"`, `"phone"`, `"web"` |
| `supervisor_district` | int 1–11 or null | `null` for "not sure" |
| `has_photo` | bool | default `false` |
| `backlog_at_filing` | int ≥ 0 or null | open cases of this type in the district; available from `/api/options` |

Labels are normalized the same way as during cleaning, so capitalization and spacing don't matter.
Request types and sources the model has never seen still get a prediction, falling back to
an "other" group.

Response:

```json
{
  "median_hours": 14.9,
  "hours":    [0.05, "...", 17520],
  "survival": [1.0,  "...", 0.02]
}
```

`survival[i]` is the probability the case is **still open** after `hours[i]`. `median_hours` is
`null` if the curve never drops below 50% within two years.

---

## Modeling details

### Candidate models

| Name | Model | Notes |
|---|---|---|
| `km_baseline` | Kaplan–Meier curve per request type | The bar every other model has to beat |
| `linear` | Elastic-net Cox (scikit-survival Coxnet), or a custom FISTA solver when stratifying | Interpretable; its coefficients drive the "what makes a request close faster" section |
| `group_lasso` | Cox with a group-lasso penalty (custom FISTA solver, `src/fista_cox.py`) | Keeps or drops whole features (e.g. all districts at once) |
| `xgboost` | XGBoost accelerated failure time | Captures nonlinear effects and interactions |

**Features** (`src/features.py`): category, request type, request details, source,
supervisor district, day of week, hour and month (encoded as positions on a circle, so 11pm sits next to midnight), photo yes/no, and log
backlog at filing. Rare categories (fewer than 50 cases) are grouped together.

### Validation

- **Time-ordered splits only.** Each split trains on everything before a cut-off date and validates on
  the next 30 days, so no model ever learns from the future. Training labels are recomputed as they
  would have looked on the cut-off date (`censor_at`), so a case that closed *after* the cut-off counts as
  still open.
- **Nested CV.** Hyperparameters are tuned on inner splits and scored on outer splits.
- **Metrics** (`src/evaluate.py`): Uno's C-index (main metric), integrated Brier score, and
  time-dependent AUC at 24h, 72h and 168h.

### Proportional-hazards check

Cox models assume each factor speeds closure by the same proportion at all times.
`src/diagnostics.py` tests this on a sample of the data. If category fails the test,
the Cox models are **stratified by category**: each category gets its own baseline curve, with shared effects.

### Model selection

The linear model is deployed by default. XGBoost replaces it only if **both** of these hold:

- its average C-index gain across outer splits is at least `0.005`, and
- a corrected resampled t-test (Nadeau & Bengio) finds the gain significant at p < 0.05.

The corrected test accounts for CV splits sharing training data, which makes an ordinary t-test
overconfident.

### Confirmatory tests

Run once on the 60-day holdout, with Holm correction across all four:

| ID | Claim | Test |
|---|---|---|
| H1 | Mobile and phone filings close at different rates (same request type) | stratified log-rank |
| H2 | Cases with a photo close at a different rate (same request type) | stratified log-rank |
| H3 | Closure rates differ across supervisor districts (same request type) | stratified log-rank |
| H4 | The deployed model ranks cases better than the Kaplan–Meier baseline | paired bootstrap on C-index |

The report also includes an **observed vs. expected** closure ratio for each neighborhood, compared with
what the baseline predicts for its mix of request types.

---

## Project layout

```
.
├── src/
│   ├── config.py        paths, DataSF dataset ID, column list, label aliases
│   ├── ingest.py        month-by-month download from the SODA API
│   ├── clean.py         raw -> clean table + cleaning report
│   ├── synth.py         synthetic data with known, planted effects
│   ├── backlog.py       open-case backlog at filing time (DuckDB ASOF join)
│   ├── features.py      feature derivation + preprocessing pipeline
│   ├── survival.py      censoring, time-ordered splits, subsampling
│   ├── models.py        Kaplan-Meier, Coxnet, FISTA Cox, XGBoost AFT
│   ├── fista_cox.py     custom penalized/stratified Cox solver
│   ├── tuning.py        regularization-path and grid tuning
│   ├── evaluate.py      C-index, Brier score, time-dependent AUC
│   ├── diagnostics.py   proportional-hazards check
│   ├── confirm.py       log-rank tests, bootstrap, corrected t-test
│   ├── train.py         the full training run -> artifacts
│   └── api.py           FastAPI app serving the model and the frontend
├── frontend/            static site (HTML, CSS, ES modules; no build step)
├── tests/               pytest suite (pipeline, models, API)
├── data/                downloaded and generated data (git-ignored)
├── Dockerfile
├── requirements.txt
└── .github/workflows/ci.yml
```

`data/` is never committed. Anyone who clones the repo rebuilds it by running the pipeline.

---

## Testing

```sh
pip install pytest httpx
PYTHONPATH=. pytest -q
```

The suite uses synthetic data and needs no network or downloaded data. It covers:

- **Pipeline:** durations and censoring, invalid dates, label normalization, photo and duplicate flags,
  and backlog computation checked against a brute-force version.
- **Models:** the FISTA solver against scikit-survival, gradient checks, the group penalty, recovering
  the planted effects, the PH check, the time-ordered splits, and the log-rank and t-test statistics.
- **API:** prediction shape, planted effects showing up in predictions, unseen labels, input
  validation, the 503 before training, and reloading after a retrain.

CI (`.github/workflows/ci.yml`) runs the suite on Python 3.12 on every push and pull request.

---

## Deployment

The Docker image bundles the API, the frontend and your trained artifacts, so **train first**:

```sh
python -m src.train
docker build -t howlong .
docker run -p 8000:8000 howlong
```

The image honours `$PORT`, so it runs as-is on Render, Fly.io, Railway or Cloud Run. Raw and
cleaned data and `runs.db` are kept out of the image (see `.dockerignore`); only the three serving
artifacts go in.

To host the frontend separately (Netlify, Vercel, GitHub Pages, …), see
[`frontend/README.md`](frontend/README.md), and restrict cross-origin access with
`HOWLONG_CORS_ORIGINS`.

---

## Configuration

| Variable | Used by | Default | Purpose |
|---|---|---|---|
| `SODA_APP_TOKEN` | ingest | unset | DataSF app token; avoids rate limiting |
| `HOWLONG_ARTIFACTS` | API | `data/artifacts` | where to load the trained model from |
| `HOWLONG_CORS_ORIGINS` | API | `*` | comma-separated list of sites allowed to call the API |
| `PORT` | Docker | `8000` | port the container listens on |

Training settings (holdout length, number of splits, tuning grids, row caps, the XGBoost gain
threshold) are in the `Settings` class at the top of `src/train.py`.

---

## Troubleshooting

**`HTTPError: 428 Client Error` during ingest.** DataSF is rate-limiting you. Set `SODA_APP_TOKEN`
(see [Ingest](#1-ingest)), or just run `python -m src.ingest` again: finished months are cached, so
each run gets further.

**`No raw data found. Run python -m src.ingest first.`** `src.clean` needs downloaded data. If you
used `src.synth`, skip cleaning and go straight to training.

**`/api/health` returns `"ok": false`, or other endpoints return 503.** No model has been trained yet.
Run `python -m src.train` (or `--fast`).

**XGBoost fails to import on macOS (`libomp` / `libxgboost.dylib`).** Run `brew install libomp`.

**The site says "demo mode".** The page couldn't reach the API. Make sure `uvicorn` is running, and
open the site through it (<http://localhost:8000>), not from the file directly.

**Training is taking hours.** That's expected for a full run on real data. Use `--fast` for a quicker check;
the server keeps serving the previous model until the new one is ready.

**Errors loading `model.joblib` after upgrading packages.** The model is a pickle tied to the
package versions that trained it. Reinstall the pinned `requirements.txt` or retrain.
