from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

CATEGORICAL = ["category", "request_type", "request_details", "source", "supervisor_district", "day_of_week"]
NUMERIC = ["hour_sin", "hour_cos", "month_sin", "month_cos", "has_photo", "log_backlog", "backlog_missing"]

# What a caller must supply to predict (the API validates against this).
INPUT_COLUMNS = ["opened", "category", "request_type", "request_details", "source",
                 "supervisor_district", "has_photo", "backlog_at_filing"]


def derive(df: pd.DataFrame) -> pd.DataFrame:
    """Raw columns -> model inputs. Pure function of each row, so it is safe at predict time."""
    out = pd.DataFrame(index=df.index)
    opened = pd.to_datetime(df["opened"])
    hour = opened.dt.hour + opened.dt.minute / 60
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    month = opened.dt.month - 1
    out["month_sin"] = np.sin(2 * np.pi * month / 12)
    out["month_cos"] = np.cos(2 * np.pi * month / 12)
    out["day_of_week"] = opened.dt.day_name().str.lower()

    for col in ["category", "request_type", "request_details", "source"]:
        out[col] = df[col].astype("object").where(df[col].notna(), "unknown").astype(str)
    district = pd.to_numeric(df["supervisor_district"], errors="coerce")
    out["supervisor_district"] = district.map(lambda d: "unknown" if pd.isna(d) else str(int(d)))

    out["has_photo"] = df["has_photo"].astype(float)
    backlog = pd.to_numeric(df.get("backlog_at_filing", pd.Series(np.nan, index=df.index)), errors="coerce")
    out["backlog_missing"] = backlog.isna().astype(float)
    out["log_backlog"] = np.log1p(backlog.fillna(0).clip(lower=0))
    return out


def make_preprocessor(min_frequency: int = 50, sparse: bool = False) -> Pipeline:
    onehot = OneHotEncoder(
        handle_unknown="infrequent_if_exist",
        min_frequency=min_frequency,
        drop="first",
        sparse_output=sparse,
    )
    columns = ColumnTransformer(
        [("cat", onehot, CATEGORICAL), ("num", StandardScaler(), NUMERIC)],
        sparse_threshold=1.0 if sparse else 0.0,
        verbose_feature_names_out=True,
    )
    return Pipeline([("derive", FunctionTransformer(derive)), ("columns", columns)])


def feature_names(pre: Pipeline) -> np.ndarray:
    return pre.named_steps["columns"].get_feature_names_out()


def feature_groups(pre: Pipeline) -> list[np.ndarray]:
    names = feature_names(pre)
    groups: dict[str, list[int]] = {}
    for i, name in enumerate(names):
        kind, rest = name.split("__", 1)
        key = next(c for c in CATEGORICAL if rest.startswith(c + "_")) if kind == "cat" else rest
        groups.setdefault(key, []).append(i)
    return [np.array(v) for v in groups.values()]


def group_labels(pre: Pipeline) -> list[str]:
    names = feature_names(pre)
    labels = []
    for idx in feature_groups(pre):
        kind, rest = names[idx[0]].split("__", 1)
        labels.append(next(c for c in CATEGORICAL if rest.startswith(c + "_")) if kind == "cat" else rest)
    return labels
