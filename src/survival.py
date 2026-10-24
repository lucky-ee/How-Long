from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sksurv.util import Surv


def surv_y(df: pd.DataFrame):
    """Structured array (event, duration) as scikit-survival expects."""
    return Surv.from_arrays(df["event"].astype(bool).to_numpy(), df["duration_hours"].to_numpy(float))


def censor_at(df: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Cases opened before `cutoff`, with labels as they would have looked at `cutoff`."""
    out = df[df["opened"] < cutoff].copy()
    known_closed = (out["event"] == 1) & (out["closed"] < cutoff)
    hours_to_cutoff = (cutoff - out["opened"]).dt.total_seconds() / 3600
    out["event"] = known_closed.astype("int8")
    out["duration_hours"] = np.where(known_closed, out["duration_hours"], hours_to_cutoff)
    out.loc[~known_closed, "closed"] = pd.NaT
    return out


@dataclass
class Split:
    cutoff: pd.Timestamp
    train: pd.DataFrame      # opened before cutoff, censored at cutoff
    valid: pd.DataFrame      # opened in [cutoff, cutoff + window), labels as observed


def forward_splits(df: pd.DataFrame, n_splits: int, valid_days: int,
                   end: pd.Timestamp | None = None) -> list[Split]:
    """Expanding-window splits: train on everything before each cutoff, validate on the
    next `valid_days`. Cutoffs step backwards from `end`, so the last split is the latest."""
    end = df["opened"].max() if end is None else end
    window = pd.to_timedelta(int(valid_days), unit="D")
    splits = []
    for k in range(n_splits, 0, -1):
        cutoff = end - k * window
        valid = df[(df["opened"] >= cutoff) & (df["opened"] < cutoff + window)]
        train = censor_at(df, cutoff)
        if len(train) and len(valid):
            splits.append(Split(cutoff, train, valid))
    return splits


def subsample(df: pd.DataFrame, max_rows: int | None, seed: int = 0) -> pd.DataFrame:
    if max_rows is None or len(df) <= max_rows:
        return df
    return df.sample(max_rows, random_state=seed).sort_values("opened")
