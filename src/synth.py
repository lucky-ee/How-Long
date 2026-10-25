from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src import config

CATEGORIES = {
    "street and sidewalk cleaning": (["bulky items", "general cleaning", "human or animal waste"], 30.0, 1.3),
    "graffiti": (["graffiti on building commercial", "graffiti on pole"], 240.0, 0.8),
    "encampments": (["encampment reports"], 60.0, 1.0),
    "parking enforcement": (["blocking driveway cite only", "parking on sidewalk"], 6.0, 1.6),
    "streetlights": (["streetlight out"], 300.0, 1.1),
    "tree maintenance": (["tree trimming"], 700.0, 0.9),
}
CATEGORY_WEIGHTS = [0.40, 0.18, 0.12, 0.15, 0.07, 0.08]

DISTRICTS = {
    1: ("Outer Richmond", 37.778, -122.490), 2: ("Marina", 37.800, -122.437),
    3: ("North Beach", 37.800, -122.408), 4: ("Sunset/Parkside", 37.750, -122.495),
    5: ("Haight Ashbury", 37.770, -122.446), 6: ("South of Market", 37.778, -122.405),
    7: ("West of Twin Peaks", 37.738, -122.460), 8: ("Castro/Upper Market", 37.760, -122.435),
    9: ("Mission", 37.760, -122.415), 10: ("Bayview Hunters Point", 37.730, -122.390),
    11: ("Excelsior", 37.722, -122.430),
}

# True log-hazard effects. Positive = closes FASTER (higher hazard of closure).
TRUE_EFFECTS = {
    "has_photo": 0.30,
    "source=mobile": 0.20,      # relative to phone
    "source=web": 0.00,         # relative to phone: no effect, a planted null
    "district=10": -0.35,       # Bayview slower, relative to the average district
    "district=2": 0.25,         # Marina faster
    "night_filing": -0.25,      # filed between midnight and 6am
}
SOURCES = ["mobile", "phone", "web"]
SOURCE_WEIGHTS = [0.5, 0.3, 0.2]


def generate(
    n: int = 50_000,
    seed: int = 0,
    start: str = "2025-01-01",
    as_of: str = "2026-10-23 20:00",
    duplicate_rate: float = 0.03,
) -> pd.DataFrame:
    """Rows shaped exactly like the output of `src.clean.clean`."""
    rng = np.random.default_rng(seed)
    t0, t_end = pd.Timestamp(start), pd.Timestamp(as_of)
    span_h = (t_end - t0).total_seconds() / 3600

    cats = rng.choice(list(CATEGORIES), n, p=CATEGORY_WEIGHTS)
    req = np.array([rng.choice(CATEGORIES[c][0]) for c in cats])
    scale = np.array([CATEGORIES[c][1] for c in cats])
    shape = np.array([CATEGORIES[c][2] for c in cats])
    district = rng.integers(1, 12, n)
    source = rng.choice(SOURCES, n, p=SOURCE_WEIGHTS)
    has_photo = rng.random(n) < np.where(source == "mobile", 0.6, 0.15)

    # filing times: daytime-heavy, so night filings are rarer (as in reality)
    day = rng.integers(0, int(span_h // 24), n)
    hour = rng.choice(24, n, p=_hour_weights())
    opened = t0 + pd.to_timedelta(day * 24 + hour, unit="h") + pd.to_timedelta(rng.integers(0, 3600, n), unit="s")

    eta = (
        TRUE_EFFECTS["has_photo"] * has_photo
        + TRUE_EFFECTS["source=mobile"] * (source == "mobile")
        + TRUE_EFFECTS["source=web"] * (source == "web")
        + TRUE_EFFECTS["district=10"] * (district == 10)
        + TRUE_EFFECTS["district=2"] * (district == 2)
        + TRUE_EFFECTS["night_filing"] * (hour < 6)
    )
    # inverse-transform sampling from the Weibull PH model
    e = rng.exponential(1.0, n)
    t_hours = scale * (e / np.exp(eta)) ** (1.0 / shape)

    is_dup = rng.random(n) < duplicate_rate
    t_hours = np.where(is_dup, rng.uniform(0.01, 0.5, n), t_hours)

    closed = opened + pd.to_timedelta(t_hours, unit="h")
    event = (closed <= t_end).astype("int8")
    end = np.where(event == 1, closed, t_end)
    duration = (pd.DatetimeIndex(end) - opened).total_seconds() / 3600

    # DISTRICTS as a table indexed by district number, so one .loc gives every row its neighborhood, lat and lon
    geo = pd.DataFrame.from_dict(DISTRICTS, orient="index", columns=["neighborhood", "lat", "lon"]).loc[district]
    lat = geo["lat"].to_numpy() + rng.normal(0, 0.006, n)
    lon = geo["lon"].to_numpy() + rng.normal(0, 0.006, n)

    df = pd.DataFrame({
        "case_id": [f"S{i:07d}" for i in range(n)],
        "opened": opened,
        "closed": pd.Series(closed).where(event == 1),
        "event": event,
        "duration_hours": duration,
        "category": cats,
        "request_type": req,
        "request_details": pd.NA,
        "source": source,
        "agency": "synthetic agency",
        "supervisor_district": district.astype(float),
        "neighborhood": geo["neighborhood"].to_numpy(),
        "lat": lat,
        "lon": lon,
        "address": pd.NA,
        "has_photo": has_photo,
        "is_duplicate": is_dup,
    })
    for col in ["category", "request_type", "request_details", "source", "agency", "neighborhood", "address"]:
        df[col] = df[col].astype("string")
    return df.sort_values("opened").reset_index(drop=True)


def _hour_weights() -> np.ndarray:
    w = np.array([0.4] * 6 + [1.0] * 3 + [2.0] * 9 + [1.2] * 4 + [0.6] * 2)
    return w / w.sum()


def main() -> None:
    as_of = "2026-10-23 20:00"
    df = generate(n=120_000, seed=0, start="2024-10-01", as_of=as_of)
    config.CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.CLEAN_DIR / "cases.parquet", index=False)
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    config.MANIFEST.write_text(json.dumps({"fetched_at": as_of, "synthetic": True}, indent=2))
    print(f"wrote {len(df):,} synthetic cases to {config.CLEAN_DIR / 'cases.parquet'}")


if __name__ == "__main__":
    main()