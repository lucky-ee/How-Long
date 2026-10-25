from __future__ import annotations

import json
import re

import pandas as pd

from src import config

KEEP = [
    "case_id", "opened", "closed", "event", "duration_hours",
    "category", "request_type", "request_details", "source", "agency",
    "supervisor_district", "neighborhood", "lat", "lon", "address",
    "has_photo", "is_duplicate",
]


def normalize_label(s: pd.Series) -> pd.Series:
    return (
        s.str.lower()
        .str.replace("_", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .replace("", pd.NA)
    )


def clean(raw: pd.DataFrame, as_of: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    df = raw.copy()
    report: dict[str, int] = {"raw_rows": len(df)}

    for col in ["requested_datetime", "closed_date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in ["lat", "long", "supervisor_district"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["source"] = normalize_label(df["source"]).replace(config.SOURCE_ALIASES)
    test_rows = df["source"].isin(config.DROP_SOURCES)
    report["dropped_test_source"] = int(test_rows.sum())
    df = df[~test_rows]

    no_open = df["requested_datetime"].isna()
    report["dropped_missing_opened"] = int(no_open.sum())
    df = df[~no_open]

    dup_id = df["service_request_id"].duplicated(keep="last")
    report["dropped_duplicate_ids"] = int(dup_id.sum())
    df = df[~dup_id]

    is_closed = df["status_description"].str.lower().eq("closed") & df["closed_date"].notna()
    bad_close = is_closed & (
        (df["closed_date"] < df["requested_datetime"])
        | (df["closed_date"] < pd.Timestamp(config.MIN_VALID_DATE))
        | (df["closed_date"] > as_of)
    )
    report["dropped_invalid_close_date"] = int(bad_close.sum())
    df = df[~bad_close]
    is_closed = is_closed[~bad_close]

    df["event"] = is_closed.astype("int8")
    end = df["closed_date"].where(is_closed, as_of)
    df["duration_hours"] = (end - df["requested_datetime"]).dt.total_seconds() / 3600
    report["censored_open_cases"] = int((df["event"] == 0).sum())

    bad_xy = ~(
        df["lat"].between(*config.SF_LAT) & df["long"].between(*config.SF_LON)
    )
    report["coords_set_missing"] = int((bad_xy & df["lat"].notna()).sum())
    df.loc[bad_xy, ["lat", "long"]] = pd.NA
    df.loc[df["supervisor_district"] == 0, "supervisor_district"] = pd.NA

    df["category"] = normalize_label(df["service_name"]).replace(config.CATEGORY_ALIASES)
    df["request_type"] = normalize_label(df["service_subtype"])
    df["request_details"] = normalize_label(df["service_details"])
    df["agency"] = normalize_label(df["agency_responsible"])

    df["has_photo"] = df["media_url"].notna() & df["media_url"].ne("")

    df["is_duplicate"] = df["status_notes"].str.contains("duplicate", case=False, na=False)
    report["duplicate_cases"] = int(df["is_duplicate"].sum())

    out = df.rename(columns={
        "service_request_id": "case_id",
        "requested_datetime": "opened",
        "closed_date": "closed",
        "analysis_neighborhood": "neighborhood",
        "long": "lon",
    })
    out.loc[out["event"] == 0, "closed"] = pd.NaT
    out = out[KEEP].sort_values("opened").reset_index(drop=True)
    report["clean_rows"] = len(out)
    return out, report


def main() -> None:
    files = sorted(config.RAW_DIR.glob("*.parquet"))
    if not files:
        raise SystemExit("No raw data found. Run `python -m src.ingest` first.")
    manifest = json.loads(config.MANIFEST.read_text())
    as_of = pd.Timestamp(manifest["fetched_at"])

    raw = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    cleaned, report = clean(raw, as_of)

    config.CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    cleaned.to_parquet(config.CLEAN_DIR / "cases.parquet", index=False)
    (config.CLEAN_DIR / "cleaning_report.json").write_text(json.dumps(report, indent=2))

    width = max(map(len, report))
    for k, v in report.items():
        print(f"{k:<{width}}  {v:>10,}")


if __name__ == "__main__":
    main()