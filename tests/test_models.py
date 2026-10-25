import warnings

import numpy as np
import pandas as pd
import pytest
from lifelines.statistics import multivariate_logrank_test
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.util import Surv

from src.confirm import corrected_resampled_ttest, stratified_logrank
from src.diagnostics import ph_check
from src.evaluate import evaluate
from src.fista_cox import FistaCox, _prepare, neg_loglik, neg_loglik_and_grad_eta, prox_group
from src.models import CoxnetModel, FistaModel, KMBaseline, XGBoostAFT
from src.survival import censor_at, forward_splits
from src.synth import generate

warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def synth():
    df = generate(12_000, seed=11, start="2025-09-01")
    return df[~df["is_duplicate"]].reset_index(drop=True)


def tied_cox_data(n=1500, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    t = np.round(rng.exponential(1 / np.exp(X @ np.array([0.8, -0.5, 0, 0, 0.3, 0]))), 1) + 0.1
    return X, t, rng.random(n) < 0.8


def test_gradient_matches_finite_differences():
    X, t, e = tied_cox_data()
    st = _prepare(t, e, None)
    eta = np.random.default_rng(0).normal(size=len(t)) * 0.3
    f0, g = neg_loglik_and_grad_eta(eta, st)
    assert f0 == pytest.approx(neg_loglik(eta, st))
    for i in [0, 17, 400]:
        bumped = eta.copy()
        bumped[i] += 1e-6
        assert g[i] == pytest.approx((neg_loglik(bumped, st) - f0) / 1e-6, rel=1e-3, abs=1e-6)


@pytest.mark.parametrize("l1_ratio", [1.0, 0.5])
def test_fista_matches_scikit_survival(l1_ratio):
    X, t, e = tied_cox_data()
    ref = CoxnetSurvivalAnalysis(alphas=[0.02], l1_ratio=l1_ratio, tol=1e-10).fit(X, Surv.from_arrays(e, t))
    ours = FistaCox(alpha=0.02, l1_ratio=l1_ratio, tol=1e-10, max_iter=5000).fit(X, t, e)
    np.testing.assert_allclose(ours.coef_, ref.coef_[:, 0], atol=1e-4)


def test_group_prox_zeroes_or_shrinks_whole_groups():
    v = np.array([0.1, 0.1, 3.0, 4.0])
    out = prox_group(v, 1.0, [np.array([0, 1]), np.array([2, 3])])
    assert np.all(out[:2] == 0)                                  # small group dropped entirely
    np.testing.assert_allclose(out[2:] / v[2:], 1 - np.sqrt(2) / 5)  # big group shrunk, direction kept


def test_stratification_recovers_effects_when_ph_fails():
    rng = np.random.default_rng(2)
    n = 20_000
    X = rng.normal(size=(n, 3))
    strata = rng.integers(0, 3, n)
    shape = np.array([0.6, 1.0, 2.0])[strata]          # different curve shapes per stratum
    t = (rng.exponential(1, n) / np.exp(X @ np.array([0.5, 0, -0.4]))) ** (1 / shape)
    e = np.ones(n, bool)
    strat = FistaCox(alpha=1e-5, l1_ratio=1).fit(X, t, e, strata=strata).coef_
    pooled = FistaCox(alpha=1e-5, l1_ratio=1).fit(X, t, e).coef_
    np.testing.assert_allclose(strat, [0.5, 0, -0.4], atol=0.04)
    assert abs(pooled[0]) < abs(strat[0]) - 0.03          # ignoring strata biases toward zero


def test_all_models_beat_random_and_baseline_is_beaten(synth):
    split = forward_splits(synth, 1, 45)[0]
    scores = {}
    for m in [KMBaseline(), CoxnetModel(0.5), FistaModel(0.5, strata_col="category", n_alphas=8),
              XGBoostAFT(max_rounds=100)]:
        m.fit(split.train)
        if hasattr(m, "alphas_"):
            m.set_alpha(m.alphas_[-2])
        res = evaluate(m, split.train, split.valid)
        scores[m.name] = res["c_index"]
        assert 0.0 <= res["ibs"] < 0.25
        surv = m.survival(split.valid.head(20), np.array([1.0, 10.0, 100.0]))
        assert np.all(np.diff(surv, axis=1) <= 1e-12)                 # curves never go up
    assert all(c > 0.6 for c in scores.values())
    assert scores["fista_elastic_net_by_category"] > scores["km_baseline"]


def test_stratified_model_recovers_planted_signs(synth):
    m = FistaModel(0.5, strata_col="category", n_alphas=10).fit_path(synth)
    m.set_alpha(m.alphas_[-1])                                        # least regularized
    coef = m.coef_table().set_index("feature")["coef"]
    assert coef["num__has_photo"] > 0                                 # photos close faster
    assert coef["cat__source_phone"] < 0                              # phone slower than mobile
    assert coef["cat__supervisor_district_2"] > coef["cat__supervisor_district_10"]


def test_ph_check_flags_category_only(synth):
    res = ph_check(synth, sample=8000)
    flags = {r["variable"]: r["violates_ph"] for r in res["results"]}
    assert flags["category"] and res["decision"] == "stratify_by_category"
    assert not flags["has_photo"]


def test_censor_at_hides_outcomes_after_cutoff(synth):
    cutoff = synth["opened"].quantile(0.7)
    tr = censor_at(synth, cutoff)
    assert (tr["opened"] < cutoff).all()
    end = tr["opened"] + pd.to_timedelta(tr["duration_hours"], unit="h")
    assert (end <= cutoff + pd.Timedelta(seconds=1)).all()
    assert (tr.loc[tr["event"] == 1, "closed"] < cutoff).all()


def test_forward_splits_are_ordered_in_time(synth):
    for s in forward_splits(synth, 3, 20):
        assert s.train["opened"].max() < s.cutoff <= s.valid["opened"].min()


def test_logrank_matches_lifelines_without_strata():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"duration_hours": np.round(rng.exponential(10, 2000), 1),
                       "event": rng.random(2000) < .8, "g": rng.choice(list("abc"), 2000), "s": "x"})
    ours = stratified_logrank(df, "g", "s")
    ref = multivariate_logrank_test(df["duration_hours"], df["g"], df["event"])
    assert ours["chi2"] == pytest.approx(ref.test_statistic, rel=1e-8)


def test_logrank_null_and_real_effect(synth):
    null = stratified_logrank(synth[synth["source"].isin(["phone", "web"])], "source", "request_type")
    real = stratified_logrank(synth[synth["source"].isin(["phone", "mobile"])], "source", "request_type")
    assert null["p"] > 0.01          # web vs phone: no planted effect
    assert real["p"] < 1e-6          # mobile vs phone: planted effect


def test_corrected_ttest_is_more_conservative_than_naive():
    diffs = [0.01, 0.012, 0.008]
    corrected = corrected_resampled_ttest(diffs, n_train=1000, n_test=500)
    naive = corrected_resampled_ttest(diffs, n_train=1e12, n_test=1)
    assert corrected["p"] > naive["p"]
