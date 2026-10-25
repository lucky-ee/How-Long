from __future__ import annotations

import argparse
import json
import os
import time
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src import config


def _next_month(d: date) -> date:
    return date(d.year + (d.month == 12), d.month % 12 + 1, 1)


def month_starts(since: date, until: date) -> list[date]:
    """First day of every month from `since`'s month through `until`'s month."""
    months, cur = [], date(since.year, since.month, 1)
    while cur <= until:
        months.append(cur)
        cur = _next_month(cur)
    return months


def _session() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=5, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])
    s.mount("https://", HTTPAdapter(max_retries=retry))
    token = os.environ.get("SODA_APP_TOKEN")
    if token:
        s.headers["X-App-Token"] = token
    return s


def fetch_month(session: requests.Session, start: date) -> pd.DataFrame:
    """All cases whose requested_datetime falls in the month beginning at `start`."""
    end = _next_month(start)
    url = f"{config.SODA_BASE}/resource/{config.DATASET_ID}.json"
    where = (
        f"requested_datetime >= '{start.isoformat()}T00:00:00' "
        f"AND requested_datetime < '{end.isoformat()}T00:00:00'"
    )
    rows, offset = [], 0
    while True:
        params = {
            "$select": ",".join(config.API_COLUMNS),
            "$where": where,
            "$order": "service_request_id",  # stable order makes paging deterministic
            "$limit": config.PAGE_SIZE,
            "$offset": offset,
        }
        resp = session.get(url, params=params, timeout=120)
        resp.raise_for_status()
        page = resp.json()
        rows.extend(page)
        if len(page) < config.PAGE_SIZE:
            break
        offset += config.PAGE_SIZE
    df = pd.DataFrame(rows, columns=config.API_COLUMNS)
    # media_url arrives as {"url": ...}; flatten so Parquet gets a plain string column
    df["media_url"] = df["media_url"].map(
        lambda v: v.get("url") if isinstance(v, dict) else v
    )
    return df.astype("string")  # keep raw as text; typing happens in clean.py


def run(since: date, force: bool = False) -> None:
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    today = date.today()
    months = month_starts(since, today)
    always_refresh = set(months[-2:])
    session = _session()

    for m in months:
        out = config.RAW_DIR / f"{m:%Y-%m}.parquet"
        if out.exists() and not force and m not in always_refresh:
            print(f"{m:%Y-%m}: cached")
            continue
        t0 = time.time()
        df = fetch_month(session, m)
        tmp = out.with_suffix(".tmp")
        df.to_parquet(tmp, index=False)
        tmp.replace(out)  # atomic: never leave a half-written month
        print(f"{m:%Y-%m}: {len(df):,} rows ({time.time() - t0:.0f}s)")

    # The fetch time is the censoring time for cases that are still open.
    # DataSF timestamps are SF local time with no zone, so record local time too.
    config.MANIFEST.write_text(json.dumps({
        "fetched_at": datetime.now(ZoneInfo("America/Los_Angeles")).replace(tzinfo=None).isoformat(),
        "since": since.isoformat(),
    }, indent=2))


def main() -> None:
    default_since = date(date.today().year - 3, date.today().month, 1)
    p = argparse.ArgumentParser(description="Fetch monthly case data from the SODA API.")
    p.add_argument("--since", type=date.fromisoformat, default=default_since)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    run(args.since, args.force)


if __name__ == "__main__":
    main()