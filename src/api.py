"""HTTP API for the deployed model, plus the static frontend.

    uvicorn src.api:app --reload        # then open http://localhost:8000

Serves the artifacts written by `python -m src.train` (model.joblib, report.json, options.json).
Set HOWLONG_ARTIFACTS to load them from somewhere else, and HOWLONG_CORS_ORIGINS
(comma-separated) to restrict which sites may call the API.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src import config
from src.clean import normalize_label
from src.features import INPUT_COLUMNS
from src.models import MEDIAN_GRID, _median_from_curves

ARTIFACT_DIR = Path(os.environ.get("HOWLONG_ARTIFACTS", config.ARTIFACT_DIR))
FRONTEND_DIR = config.ROOT / "frontend"
CORS_ORIGINS = [o.strip() for o in os.environ.get("HOWLONG_CORS_ORIGINS", "*").split(",")]
SF_TZ = ZoneInfo("America/Los_Angeles")

# report.json also holds full regularization paths and CV curves; the site needs none of that
REPORT_KEYS = ["run_id", "created_at", "data", "deployed", "holdout", "model_selection",
               "confirmatory", "neighborhoods"]


class Case(BaseModel):
    """One request as a caller describes it at filing time (features.INPUT_COLUMNS)."""
    opened: datetime = Field(description="Filing time; naive values are San Francisco local time")
    category: str = Field(min_length=1)
    request_type: str = Field(min_length=1)
    request_details: Optional[str] = None
    source: str = Field(min_length=1)
    supervisor_district: Optional[int] = Field(None, ge=1, le=11)
    has_photo: bool = False
    backlog_at_filing: Optional[int] = Field(None, ge=0, description="Open cases of this type in the district")


class Prediction(BaseModel):
    median_hours: Optional[float] = Field(description="null when the curve never drops to 50%")
    hours: List[float]
    survival: List[float] = Field(description="P(still open) at each of `hours`")


class Artifacts:
    """Loads the training outputs once, on first use, so the app starts even before training."""

    FILES = ("model.joblib", "report.json", "options.json")

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self._cache: Optional[dict] = None

    def get(self) -> dict:
        if self._cache is None:
            missing = [f for f in self.FILES if not (self.directory / f).exists()]
            if missing:
                raise HTTPException(503, f"No trained model yet (missing {', '.join(missing)}). "
                                         "Run `python -m src.train` first.")
            bundle = joblib.load(self.directory / "model.joblib")
            report = json.loads((self.directory / "report.json").read_text())
            summary = {k: report[k] for k in REPORT_KEYS if k in report}
            summary["linear_model"] = {"coefficients": report.get("linear_model", {}).get("coefficients", [])}
            self._cache = {
                "model": bundle["model"],
                "report": summary,
                "options": json.loads((self.directory / "options.json").read_text()),
            }
        return self._cache


def to_frame(case: Case) -> pd.DataFrame:
    """The request as a one-row frame, with labels normalized the way src.clean normalizes them."""
    opened = case.opened
    if opened.tzinfo is not None:
        opened = opened.astimezone(SF_TZ).replace(tzinfo=None)
    labels = normalize_label(pd.Series(
        [case.category, case.request_type, case.request_details, case.source], dtype="string"))
    row = pd.DataFrame([{
        "opened": pd.Timestamp(opened),
        "category": config.CATEGORY_ALIASES.get(labels[0], labels[0]),
        "request_type": labels[1],
        "request_details": labels[2],
        "source": config.SOURCE_ALIASES.get(labels[3], labels[3]),
        "supervisor_district": np.nan if case.supervisor_district is None else float(case.supervisor_district),
        "has_photo": case.has_photo,
        "backlog_at_filing": np.nan if case.backlog_at_filing is None else float(case.backlog_at_filing),
    }])
    return row[INPUT_COLUMNS]


def create_app(artifact_dir: Path = ARTIFACT_DIR, frontend_dir: Optional[Path] = FRONTEND_DIR) -> FastAPI:
    artifacts = Artifacts(artifact_dir)
    app = FastAPI(title="How Long?", description="How long will an SF 311 request take to close?")
    app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS,
                       allow_methods=["GET", "POST"], allow_headers=["Content-Type"])

    api = APIRouter(prefix="/api")

    @api.get("/health")
    def health():
        try:
            loaded = artifacts.get()
        except HTTPException as err:
            return {"ok": False, "detail": err.detail}
        return {"ok": True, "deployed": loaded["report"].get("deployed", {}).get("name")}

    @api.get("/options")
    def options():
        return artifacts.get()["options"]

    @api.get("/report")
    def report():
        return artifacts.get()["report"]

    @api.post("/predict", response_model=Prediction)
    def predict(case: Case):
        model = artifacts.get()["model"]
        surv = np.clip(model.survival(to_frame(case), MEDIAN_GRID), 0.0, 1.0)
        median = float(_median_from_curves(surv, MEDIAN_GRID)[0])
        return Prediction(
            median_hours=median if np.isfinite(median) else None,
            hours=MEDIAN_GRID.tolist(),
            survival=surv[0].tolist(),
        )

    app.include_router(api)

    if frontend_dir is not None and Path(frontend_dir).is_dir():
        @app.get("/config.js", include_in_schema=False)
        def frontend_config():
            # served from the same origin, so the page talks to this API rather than demo data
            return Response('window.HOWLONG_CONFIG = { apiBase: "/api" };\n', media_type="text/javascript")

        app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")

    return app


app = create_app()
