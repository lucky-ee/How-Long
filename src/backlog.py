from __future__ import annotations

import duckdb
import pandas as pd


BACKLOG_SQL = """
WITH events AS (
    SELECT request_type, supervisor_district, opened AS t, 1 AS delta FROM cases
    UNION ALL
    SELECT request_type, supervisor_district, closed AS t, -1 AS delta
    FROM cases WHERE closed IS NOT NULL
),
-- collapse simultaneous events first, so the ASOF match below is unambiguous
per_time AS (
    SELECT request_type, supervisor_district, t, SUM(delta) AS delta
    FROM events GROUP BY ALL
),
running AS (
    SELECT request_type, supervisor_district, t,
           SUM(delta) OVER (PARTITION BY request_type, supervisor_district ORDER BY t) AS open_count
    FROM per_time
)
SELECT c.case_id, COALESCE(r.open_count, 0)::INTEGER AS backlog_at_filing
FROM cases c
ASOF LEFT JOIN running r
  ON  c.request_type = r.request_type
  AND c.supervisor_district = r.supervisor_district
  AND c.opened > r.t
"""


def add_backlog(cases: pd.DataFrame) -> pd.DataFrame:
    """Return `cases` with a `backlog_at_filing` column (null if type/district unknown)."""
    con = duckdb.connect()
    con.register("cases", cases[["case_id", "request_type", "supervisor_district", "opened", "closed"]])
    backlog = con.execute(BACKLOG_SQL).df()
    out = cases.merge(backlog, on="case_id", how="left")
    out["backlog_at_filing"] = out["backlog_at_filing"].astype("Int64")
    unknown = out["request_type"].isna() | out["supervisor_district"].isna()
    out.loc[unknown, "backlog_at_filing"] = pd.NA
    return out