from __future__ import annotations

import numpy as np
import pandas as pd
from sksurv.metrics import (concordance_index_censored, concordance_index_ipcw,
                            cumulative_dynamic_auc, integrated_brier_score)

from howlong.survival import surv_y

AUC_HOURS = (24.0, 72.0, 168.0)


def _clip_to_train(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    limit = train["duration_hours"].max() * 0.999
    test = test.copy()
    over = test["duration_hours"] > limit
    test.loc[over, "event"] = 0
    test.loc[over, "duration_hours"] = limit
    return test


def eval_times(train: pd.DataFrame, test: pd.DataFrame, n: int = 40) -> np.ndarray:
    ev = test.loc[test["event"] == 1, "duration_hours"]
    lo, hi = np.percentile(ev, [5, 90])
    hi = min(hi, test["duration_hours"].max() * 0.99, train["duration_hours"].max() * 0.99)
    return np.unique(np.geomspace(max(lo, 1e-3), hi, n))


def c_index(train: pd.DataFrame, test: pd.DataFrame, risk: np.ndarray) -> float:
    test = _clip_to_train(train, test)
    tau = np.percentile(test["duration_hours"], 95)
    try:
        return float(concordance_index_ipcw(surv_y(train), surv_y(test), risk, tau=tau)[0])
    except ValueError:  # degenerate fold (e.g. no events before tau): fall back to Harrell
        return float(concordance_index_censored(test["event"].astype(bool), test["duration_hours"], risk)[0])


def evaluate(model, train: pd.DataFrame, test: pd.DataFrame) -> dict[str, float]:
    test = _clip_to_train(train, test)
    risk = model.risk(test)
    out = {"c_index": c_index(train, test, risk)}

    times = eval_times(train, test)
    surv = model.survival(test, times)
    try:
        out["ibs"] = float(integrated_brier_score(surv_y(train), surv_y(test), surv, times))
    except ValueError:
        out["ibs"] = float("nan")

    hours = [h for h in AUC_HOURS if times[0] < h < times[-1]]
    if hours:
        risk_at = 1 - model.survival(test, np.array(hours))
        auc, _ = cumulative_dynamic_auc(surv_y(train), surv_y(test), risk_at, hours)
        out.update({f"auc@{int(h)}h": float(a) for h, a in zip(hours, auc)})
    return out
