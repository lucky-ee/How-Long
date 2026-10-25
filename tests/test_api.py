import json
import warnings

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src import config
from src.api import create_app
from src.models import FistaModel
from src.synth import generate

warnings.filterwarnings("ignore")

CASE = {
    "opened": "2026-10-01T09:30:00",
    "category": "Graffiti",
    "request_type": "graffiti on pole",
    "source": "mobile",
    "supervisor_district": 10,
    "has_photo": False,
    "backlog_at_filing": 3,
}


@pytest.fixture(scope="module")
def artifact_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("artifacts")
    df = generate(5_000, seed=5, start="2026-05-01")
    df = df[~df["is_duplicate"]].reset_index(drop=True)
    model = FistaModel(0.5, strata_col="category", n_alphas=5).fit(df)
    model.set_alpha(min(model.alphas_))           # weakest penalty, so planted effects survive
    joblib.dump({"model": model, "baseline": None, "deployed": "linear"}, out / "model.joblib")
    (out / "report.json").write_text(json.dumps({
        "deployed": {"name": "linear"},
        "holdout": {"linear": {"c_index": 0.7}},
        "linear_model": {"coefficients": [{"feature": "num__has_photo", "coef": 0.3}], "path": {"alphas": [1]}},
    }))
    (out / "options.json").write_text(json.dumps({"as_of": "2026-10-23", "request_types": [], "sources": ["mobile"],
                                                  "districts": [], "backlog": {}}))
    return out


@pytest.fixture(scope="module")
def client(artifact_dir):
    return TestClient(create_app(artifact_dir, frontend_dir=None))


def predict(client, **changes):
    r = client.post("/api/predict", json={**CASE, **changes})
    assert r.status_code == 200, r.text
    return r.json()


def test_predict_returns_a_survival_curve_and_its_median(client):
    body = predict(client)
    hours, surv = np.array(body["hours"]), np.array(body["survival"])
    assert len(hours) == len(surv) == 400
    assert np.all(np.diff(surv) <= 1e-12) and surv[0] <= 1 and surv[-1] >= 0
    i = np.searchsorted(hours, body["median_hours"])
    assert surv[i] <= 0.5 < surv[i - 1]


def test_predictions_follow_the_planted_effects(client):
    base = predict(client)["median_hours"]
    assert predict(client, has_photo=True)["median_hours"] < base       # photos close faster
    assert predict(client, supervisor_district=2)["median_hours"] < base  # Marina faster than Bayview


def test_unseen_labels_and_unknown_district_still_predict(client):
    body = predict(client, category="Brand New Category", request_type="never seen", supervisor_district=None,
                   backlog_at_filing=None, opened="2026-10-01T09:30:00-07:00")
    assert body["median_hours"] is None or body["median_hours"] > 0


@pytest.mark.parametrize("bad", [{"supervisor_district": 99}, {"request_type": ""}, {"opened": "yesterday"},
                                 {"backlog_at_filing": -1}])
def test_rejects_invalid_input(client, bad):
    assert client.post("/api/predict", json={**CASE, **bad}).status_code == 422


def test_report_drops_the_heavy_parts(client):
    body = client.get("/api/report").json()
    assert body["linear_model"] == {"coefficients": [{"feature": "num__has_photo", "coef": 0.3}]}
    assert client.get("/api/options").json()["sources"] == ["mobile"]


def test_missing_artifacts_is_a_clear_503(tmp_path):
    c = TestClient(create_app(tmp_path, frontend_dir=None))
    assert c.get("/api/options").status_code == 503
    assert c.get("/api/health").json()["ok"] is False


def test_serves_frontend_pointed_at_this_api(artifact_dir):
    c = TestClient(create_app(artifact_dir, frontend_dir=config.ROOT / "frontend"))
    assert "/api" in c.get("/config.js").text
    assert "How Long" in c.get("/").text
    assert c.get("/api/health").json() == {"ok": True, "deployed": "linear"}
