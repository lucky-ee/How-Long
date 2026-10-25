from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
from statsmodels.stats.multitest import multipletests

CHECK_VARIABLES = ["category", "source", "has_photo", "night_filing"]


def _design(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    d = pd.DataFrame({"T": df["duration_hours"].clip(lower=1e-3).to_numpy(),
                      "E": df["event"].to_numpy()}, index=df.index)
    owners: dict[str, list[str]] = {}
    for var in ["category", "source"]:
        dummies = pd.get_dummies(df[var].astype(str), prefix=var, drop_first=False, dtype=float)
        dummies = dummies.drop(columns=dummies.sum().idxmax())     # most common level = reference
        d = d.join(dummies)
        owners[var] = list(dummies.columns)
    d["has_photo"] = df["has_photo"].astype(float)
    d["night_filing"] = (pd.to_datetime(df["opened"]).dt.hour < 6).astype(float)
    owners["has_photo"], owners["night_filing"] = ["has_photo"], ["night_filing"]
    return d, owners


def ph_check(df: pd.DataFrame, sample: int = 20_000, alpha: float = 0.01, seed: int = 0) -> dict:
    df = df.sample(min(sample, len(df)), random_state=seed)
    design, owners = _design(df)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cph = CoxPHFitter(penalizer=0.01).fit(design, "T", "E")
        test = proportional_hazard_test(cph, design, time_transform="rank")
    p = test.summary["p"]

    rows = []
    for var, cols in owners.items():
        rows.append({"variable": var, "p_raw": float(min(1.0, p[cols].min() * len(cols)))})
    reject, p_adj, _, _ = multipletests([r["p_raw"] for r in rows], alpha=alpha, method="holm")
    for r, rej, pa in zip(rows, reject, p_adj):
        r.update({"p_holm": float(pa), "violates_ph": bool(rej)})

    category_fails = next(r["violates_ph"] for r in rows if r["variable"] == "category")
    return {
        "sample_size": len(df),
        "alpha": alpha,
        "results": rows,
        "decision": "stratify_by_category" if category_fails else "no_stratification",
    }
