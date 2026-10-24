import numpy as np
import pandas as pd
import pytest

from sf311 import config
from sf311.backlog import add_backlog
from sf311.clean import clean
from sf311.ingest import month_starts

AS_OF = pd.Timestamp("2026-10-23 20:00")


def raw_row(**overrides):
    row = {
        "service_request_id": "1",
        "requested_datetime": "2026-10-01T09:00:00.000",
        "closed_date": "2026-10-02T09:00:00.000",
        "updated_datetime": "2026-10-02T09:00:00.000",
        "status_description": "Closed",
        "status_notes": "Case Resolved",
        "agency_responsible": "DPW Ops Queue",
        "service_name": "Street and Sidewalk Cleaning",
        "service_subtype": "Bulky Items",
        "service_details": "Mattress",
        "address": "100 MISSION ST",
        "supervisor_district": "6",
        "analysis_neighborhood": "South of Market",
        "lat": "37.78",
        "long": "-122.40",
        "source": "Mobile/Open311",
        "media_url": None,
    }
    row.update(overrides)
    return row


def raw_df(*rows):
    return pd.DataFrame(rows, columns=config.API_COLUMNS).astype("string")


def test_closed_case_duration_and_event():
    out, _ = clean(raw_df(raw_row()), AS_OF)
    assert out.loc[0, "event"] == 1
    assert out.loc[0, "duration_hours"] == pytest.approx(24.0)


def test_open_case_is_censored_at_fetch_time():
    out, rep = clean(raw_df(raw_row(status_description="Open", closed_date=None)), AS_OF)
    assert out.loc[0, "event"] == 0
    expected = (AS_OF - pd.Timestamp("2026-10-01 09:00")).total_seconds() / 3600
    assert out.loc[0, "duration_hours"] == pytest.approx(expected)
    assert pd.isna(out.loc[0, "closed"])
    assert rep["censored_open_cases"] == 1


@pytest.mark.parametrize("closed", [
    "1899-12-31T23:07:00.000",   # placeholder date seen in the real data
    "2026-09-30T09:00:00.000",   # before it opened
    "2026-12-22T00:00:00.000",   # in the future
])
def test_invalid_close_dates_are_dropped(closed):
    out, rep = clean(raw_df(raw_row(closed_date=closed)), AS_OF)
    assert len(out) == 0
    assert rep["dropped_invalid_close_date"] == 1


def test_test_source_dropped_and_mobile_merged():
    out, rep = clean(raw_df(
        raw_row(service_request_id="1", source="Test"),
        raw_row(service_request_id="2", source="Mobile/Open311"),
        raw_row(service_request_id="3", source="Mobile"),
    ), AS_OF)
    assert rep["dropped_test_source"] == 1
    assert set(out["source"]) == {"mobile"}


def test_zero_coordinates_become_missing_but_row_kept():
    out, rep = clean(raw_df(raw_row(lat="0.0", long="0.0")), AS_OF)
    assert len(out) == 1
    assert pd.isna(out.loc[0, "lat"]) and pd.isna(out.loc[0, "lon"])
    assert rep["coords_set_missing"] == 1


def test_labels_normalized_and_aliased():
    out, _ = clean(raw_df(
        raw_row(service_request_id="1", service_name="Encampment", service_subtype="garbage_and_debris"),
        raw_row(service_request_id="2", service_name="Encampments", service_subtype="Garbage and Debris "),
    ), AS_OF)
    assert out["category"].nunique() == 1
    assert out["request_type"].nunique() == 1


def test_duplicate_flag_and_photo_flag():
    out, _ = clean(raw_df(
        raw_row(service_request_id="1", status_notes="Case is a Duplicate", media_url="https://x/img.jpg"),
        raw_row(service_request_id="2"),
    ), AS_OF)
    out = out.set_index("case_id")
    assert out.loc["1", "is_duplicate"] and out.loc["1", "has_photo"]
    assert not out.loc["2", "is_duplicate"] and not out.loc["2", "has_photo"]


def test_month_starts_crosses_year_boundary():
    from datetime import date
    assert month_starts(date(2025, 11, 15), date(2026, 2, 1)) == [
        date(2025, 11, 1), date(2025, 12, 1), date(2026, 1, 1), date(2026, 2, 1),
    ]


# --- backlog: compare the O(n log n) sweep against an O(n^2) brute force ----------------

def brute_force_backlog(cases):
    out = []
    for _, a in cases.iterrows():
        if pd.isna(a.request_type) or pd.isna(a.supervisor_district):
            out.append(pd.NA)
            continue
        same = cases[(cases.request_type == a.request_type)
                     & (cases.supervisor_district == a.supervisor_district)]
        is_open = (same.opened < a.opened) & (same.closed.isna() | (same.closed >= a.opened))
        out.append(int(is_open.sum()))
    return pd.Series(out, index=cases.case_id, dtype="Int64")


def random_cases(n, seed):
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2026-01-01")
    # coarse hour grid on purpose, so simultaneous opens/closes (ties) happen often
    opened = base + pd.to_timedelta(rng.integers(0, 200, n), unit="h")
    dur = pd.to_timedelta(rng.integers(0, 60, n), unit="h")
    closed = pd.Series(opened + dur)
    closed[rng.random(n) < 0.2] = pd.NaT  # still open
    df = pd.DataFrame({
        "case_id": [str(i) for i in range(n)],
        "request_type": rng.choice(["bulky items", "graffiti", None], n, p=[.45, .45, .1]),
        "supervisor_district": rng.choice([1.0, 2.0, np.nan], n, p=[.45, .45, .1]),
        "opened": opened,
        "closed": closed,
    })
    return df


@pytest.mark.parametrize("seed", range(5))
def test_backlog_matches_brute_force(seed):
    cases = random_cases(300, seed)
    fast = add_backlog(cases).set_index("case_id")["backlog_at_filing"]
    slow = brute_force_backlog(cases)
    pd.testing.assert_series_equal(fast.sort_index(), slow.sort_index(), check_names=False)
