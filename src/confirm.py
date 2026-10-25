from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sksurv.metrics import concordance_index_censored
from statsmodels.stats.multitest import multipletests

ALPHA = 0.05


def stratified_logrank(df: pd.DataFrame, group: str, strata: str) -> dict:
    levels = sorted(df[group].dropna().unique().tolist())
    K = len(levels)
    code = df[group].map({l: i for i, l in enumerate(levels)}).to_numpy()
    U, V = np.zeros(K), np.zeros((K, K))
    O, E = np.zeros(K), np.zeros(K)

    for _, s in df.assign(_g=code).groupby(strata):
        t, e, g = s["duration_hours"].to_numpy(), s["event"].to_numpy(bool), s["_g"].to_numpy()
        u = np.unique(t[e])
        if len(u) == 0:
            continue
        n_k = np.zeros((len(u), K))
        d_k = np.zeros((len(u), K))
        for k in range(K):
            tk = np.sort(t[g == k])
            ek = np.sort(t[(g == k) & e])
            n_k[:, k] = len(tk) - np.searchsorted(tk, u, "left")
            d_k[:, k] = np.searchsorted(ek, u, "right") - np.searchsorted(ek, u, "left")
        n, d = n_k.sum(1), d_k.sum(1)
        keep = n > 1
        n_k, d_k, n, d = n_k[keep], d_k[keep], n[keep], d[keep]
        exp_k = d[:, None] * n_k / n[:, None]
        O += d_k.sum(0)
        E += exp_k.sum(0)
        U += (d_k - exp_k).sum(0)
        f = (d * (n - d) / (n - 1))[:, None, None]
        frac = n_k / n[:, None]
        V += (f * (frac[:, :, None] * (np.eye(K)[None] - frac[:, None, :]))).sum(0)

    u, v = U[:-1], V[:-1, :-1]
    chi2 = float(u @ np.linalg.solve(v, u))
    return {
        "test": "stratified log-rank", "group": group, "strata": strata,
        "chi2": chi2, "df": K - 1, "p": float(stats.chi2.sf(chi2, K - 1)),
        "observed_over_expected": {str(l): float(o / e) for l, o, e in zip(levels, O, E)},
    }


def paired_bootstrap_c(df: pd.DataFrame, risk_a: np.ndarray, risk_b: np.ndarray,
                       n_boot: int = 300, max_rows: int = 5000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), min(max_rows, len(df)), replace=False)
    e, t = df["event"].to_numpy(bool)[idx], df["duration_hours"].to_numpy()[idx]
    a, b = risk_a[idx], risk_b[idx]

    def c(r, ii):
        return concordance_index_censored(e[ii], t[ii], r[ii])[0]

    full = np.arange(len(idx))
    diff = c(a, full) - c(b, full)
    boots = np.array([c(a, ii) - c(b, ii) for ii in (rng.integers(0, len(idx), len(idx)) for _ in range(n_boot))])
    p = float((np.sum(boots <= 0) + 1) / (n_boot + 1))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"test": "paired bootstrap, Harrell C", "c_diff": float(diff),
            "ci95": [float(lo), float(hi)], "p": p, "n_rows": len(idx)}


def corrected_resampled_ttest(diffs: list[float], n_train: float, n_test: float) -> dict:
    """Nadeau & Bengio: CV folds share training data, so the naive t-test is overconfident.
    The variance is inflated by (1/J + n_test/n_train)."""
    d = np.asarray(diffs, float)
    J = len(d)
    if J < 2:
        return {"mean_diff": float(d.mean()), "t": float("nan"), "p": float("nan")}
    var = d.var(ddof=1) * (1 / J + n_test / n_train)
    t = d.mean() / np.sqrt(var) if var > 0 else np.inf
    return {"mean_diff": float(d.mean()), "t": float(t), "p": float(stats.t.sf(t, J - 1))}


def run_confirmatory(holdout: pd.DataFrame, deployed_risk: np.ndarray, baseline_risk: np.ndarray) -> dict:
    h = holdout.copy()
    h["supervisor_district"] = h["supervisor_district"].map(lambda d: "unknown" if pd.isna(d) else str(int(d)))
    h["has_photo"] = h["has_photo"].map({True: "photo", False: "no photo"})
    two_sources = h[h["source"].isin(["mobile", "phone"])]

    results = [
        {"id": "H1", "claim": "Mobile and phone filings close at different rates (same request type)",
         **stratified_logrank(two_sources, "source", "request_type")},
        {"id": "H2", "claim": "Cases with a photo close at a different rate (same request type)",
         **stratified_logrank(h, "has_photo", "request_type")},
        {"id": "H3", "claim": "Closure rates differ across supervisor districts (same request type)",
         **stratified_logrank(h[h["supervisor_district"] != "unknown"], "supervisor_district", "request_type")},
        {"id": "H4", "claim": "The deployed model ranks cases better than the Kaplan-Meier baseline",
         **paired_bootstrap_c(h, deployed_risk, baseline_risk)},
    ]
    reject, p_adj, _, _ = multipletests([r["p"] for r in results], alpha=ALPHA, method="holm")
    for r, rej, pa in zip(results, reject, p_adj):
        r["p_holm"], r["supported"] = float(pa), bool(rej)
    return {
        "holdout_start": str(holdout["opened"].min()),
        "holdout_end": str(holdout["opened"].max()),
        "n_cases": len(holdout),
        "alpha": ALPHA,
        "correction": "holm",
        "hypotheses": results,
    }
