from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import norm
from sksurv.linear_model import CoxnetSurvivalAnalysis

from howlong.features import feature_groups, feature_names, group_labels, make_preprocessor
from howlong.fista_cox import FistaCox, neg_loglik_and_grad_eta, _prepare
from howlong.survival import subsample, surv_y

RISK_HORIZON_H = 72.0
MEDIAN_GRID = np.geomspace(0.05, 24 * 365 * 2, 400)


def _median_from_curves(surv: np.ndarray, grid: np.ndarray) -> np.ndarray:
    below = surv <= 0.5
    idx = below.argmax(axis=1)
    return np.where(below.any(axis=1), grid[idx], np.inf)


def _step_eval(x: np.ndarray, y: np.ndarray, t: np.ndarray, left: float) -> np.ndarray:
    i = np.searchsorted(x, t, side="right") - 1
    return np.where(i < 0, left, y[np.clip(i, 0, len(y) - 1)])


class BaseModel:
    name = "base"

    def median_hours(self, df: pd.DataFrame) -> np.ndarray:
        return _median_from_curves(self.survival(df, MEDIAN_GRID), MEDIAN_GRID)

    def risk(self, df: pd.DataFrame) -> np.ndarray:
        return 1.0 - self.survival(df, np.array([RISK_HORIZON_H]))[:, 0]


class KMBaseline(BaseModel):
    name = "km_baseline"

    def __init__(self, group: str = "request_type", min_count: int = 30):
        self.group, self.min_count = group, min_count

    @staticmethod
    def _km(t: np.ndarray, e: np.ndarray):
        order = np.argsort(t)
        t, e = t[order], e[order].astype(float)
        uniq, first = np.unique(t, return_index=True)
        deaths = np.add.reduceat(e, first)
        at_risk = len(t) - first
        return uniq, np.cumprod(1.0 - deaths / at_risk)

    def fit(self, train: pd.DataFrame):
        self.curves_ = {}
        for key, g in train.groupby(self.group, dropna=False):
            if len(g) >= self.min_count:
                self.curves_[key] = self._km(g["duration_hours"].to_numpy(), g["event"].to_numpy())
        self.global_ = self._km(train["duration_hours"].to_numpy(), train["event"].to_numpy())
        return self

    def survival(self, df, times):
        times = np.asarray(times, float)
        out = np.empty((len(df), len(times)))
        keys = df[self.group].to_numpy()
        cache = {}
        for i, k in enumerate(keys):
            if k not in cache:
                x, s = self.curves_.get(k, self.global_)
                cache[k] = _step_eval(x, s, times, left=1.0)
            out[i] = cache[k]
        return out

    def describe(self):
        return {"model": self.name, "group": self.group, "n_groups": len(self.curves_)}


class CoxnetModel(BaseModel):
    name = "coxnet"

    def __init__(self, l1_ratio: float = 0.5, alpha: float | None = None,
                 max_rows: int | None = 150_000, min_frequency: int = 50):
        self.l1_ratio, self.alpha = l1_ratio, alpha
        self.max_rows, self.min_frequency = max_rows, min_frequency

    def fit_path(self, train: pd.DataFrame, alphas=None):
        train = subsample(train, self.max_rows)
        self.pre_ = make_preprocessor(self.min_frequency, sparse=False)
        X = self.pre_.fit_transform(train)
        kw = {"alphas": alphas} if alphas is not None else {"n_alphas": 25, "alpha_min_ratio": 1e-3}
        self.est_ = CoxnetSurvivalAnalysis(l1_ratio=self.l1_ratio, fit_baseline_model=True,
                                           max_iter=200_000, **kw).fit(X, surv_y(train))
        self.alphas_ = self.est_.alphas_
        if self.alpha is None or alphas is None:
            self.alpha = float(self.alphas_[len(self.alphas_) // 2])
        return self

    fit = fit_path

    def set_alpha(self, alpha: float):
        self.alpha = float(alpha)
        return self

    def survival(self, df, times):
        X = self.pre_.transform(df)
        s = self.est_.predict_survival_function(X, alpha=self.alpha, return_array=True)
        return np.stack([_step_eval(self.est_.unique_times_, row, np.asarray(times, float), 1.0) for row in s]) \
            if len(s) < 50 else self._vector_step(s, times)

    def _vector_step(self, s, times):
        i = np.searchsorted(self.est_.unique_times_, np.asarray(times, float), side="right") - 1
        out = s[:, np.clip(i, 0, s.shape[1] - 1)]
        out[:, i < 0] = 1.0
        return out

    def coef_table(self) -> pd.DataFrame:
        names = feature_names(self.pre_)
        coef = self.est_.coef_[:, self._alpha_index()]
        return pd.DataFrame({"feature": names, "coef": coef})

    def _alpha_index(self) -> int:
        return int(np.argmin(np.abs(self.est_.alphas_ - self.alpha)))

    def path(self) -> dict:
        return {"alphas": self.est_.alphas_.tolist(), "features": feature_names(self.pre_).tolist(),
                "coefs": self.est_.coef_.tolist()}

    def describe(self):
        return {"model": self.name, "l1_ratio": self.l1_ratio, "alpha": self.alpha,
                "n_nonzero": int((self.coef_table()["coef"] != 0).sum())}


class FistaModel(BaseModel):

    def __init__(self, l1_ratio: float = 0.5, alpha: float | None = None, penalty: str = "elastic_net",
                 strata_col: str | None = None, max_rows: int | None = 1_000_000,
                 min_frequency: int = 50, n_alphas: int = 20, alpha_min_ratio: float = 1e-3):
        self.l1_ratio, self.alpha, self.penalty, self.strata_col = l1_ratio, alpha, penalty, strata_col
        self.max_rows, self.min_frequency = max_rows, min_frequency
        self.n_alphas, self.alpha_min_ratio = n_alphas, alpha_min_ratio

    @property
    def name(self):
        kind = "group_lasso" if self.penalty == "group" else "elastic_net"
        return f"fista_{kind}" + (f"_by_{self.strata_col}" if self.strata_col else "")

    def _strata(self, df):
        return None if self.strata_col is None else df[self.strata_col].astype(str).to_numpy()

    def _alpha_max(self, X, t, e, strata) -> float:
        _, g_eta = neg_loglik_and_grad_eta(np.zeros(X.shape[0]), _prepare(t, e, strata))
        grad = np.abs(np.asarray(X.T @ g_eta).ravel()) / X.shape[0]
        if self.penalty == "group":
            return max(np.linalg.norm(grad[g]) / np.sqrt(len(g)) for g in self.groups_) / self.l1_ratio
        return grad.max() / self.l1_ratio

    def fit_path(self, train: pd.DataFrame, alphas=None):
        train = subsample(train, self.max_rows)
        self.pre_ = make_preprocessor(self.min_frequency, sparse=True)
        X = self.pre_.fit_transform(train)
        self.groups_ = feature_groups(self.pre_)
        t = train["duration_hours"].to_numpy(float)
        e = train["event"].to_numpy(bool)
        strata = self._strata(train)
        if alphas is None:
            amax = self._alpha_max(X, t, e, strata)
            alphas = np.geomspace(amax, amax * self.alpha_min_ratio, self.n_alphas)
        self.alphas_ = np.sort(np.asarray(alphas, float))[::-1]

        solver = FistaCox(l1_ratio=self.l1_ratio, penalty=self.penalty, groups=self.groups_, tol=1e-6)
        prepared = _prepare(t, e, strata)
        self.path_ = {}
        beta = None
        for a in self.alphas_:
            solver.alpha = a
            solver.fit(X, t, e, strata=strata, beta0=beta, prepared=prepared)
            beta = solver.coef_.copy()
            self.path_[float(a)] = (beta, solver.baseline_, solver.info_.n_iter)
        self.solver_ = solver
        if self.alpha is None or float(self.alpha) not in self.path_:
            self.alpha = float(self.alphas_[len(self.alphas_) // 2])
        return self.set_alpha(self.alpha)

    fit = fit_path

    def set_alpha(self, alpha: float):
        self.alpha = min(self.path_, key=lambda a: abs(a - alpha))
        beta, baseline, _ = self.path_[self.alpha]
        self.solver_.coef_, self.solver_.baseline_ = beta, baseline
        return self

    def survival(self, df, times):
        X = self.pre_.transform(df)
        H = self.solver_.cumulative_hazard(X, times, strata=self._strata(df))
        return np.exp(-H)

    def coef_table(self) -> pd.DataFrame:
        return pd.DataFrame({"feature": feature_names(self.pre_), "coef": self.solver_.coef_})

    def path(self) -> dict:
        alphas = sorted(self.path_, reverse=True)
        return {"alphas": alphas, "features": feature_names(self.pre_).tolist(),
                "coefs": np.stack([self.path_[a][0] for a in alphas], axis=1).tolist(),
                "groups": group_labels(self.pre_)}

    def describe(self):
        return {"model": self.name, "l1_ratio": self.l1_ratio, "alpha": self.alpha,
                "penalty": self.penalty, "strata": self.strata_col,
                "n_nonzero": int((self.solver_.coef_ != 0).sum()),
                "iterations": int(self.path_[self.alpha][2])}


class XGBoostAFT(BaseModel):
    name = "xgboost_aft"

    def __init__(self, learning_rate: float = 0.1, max_depth: int = 6, sigma: float = 1.2,
                 max_rounds: int = 600, min_frequency: int = 50, max_rows: int | None = 1_000_000):
        self.learning_rate, self.max_depth, self.sigma = learning_rate, max_depth, sigma
        self.max_rounds, self.min_frequency, self.max_rows = max_rounds, min_frequency, max_rows

    def _dmatrix(self, X, df):
        t = np.maximum(df["duration_hours"].to_numpy(float), 1e-3)
        upper = np.where(df["event"].to_numpy(bool), t, np.inf)
        d = xgb.DMatrix(X)
        d.set_float_info("label_lower_bound", t)
        d.set_float_info("label_upper_bound", upper)
        return d

    def fit(self, train: pd.DataFrame):
        train = subsample(train, self.max_rows)
        self.pre_ = make_preprocessor(self.min_frequency, sparse=True)
        n_fit = int(len(train) * 0.85)               # latest 15% (by time) for early stopping
        fit_df, stop_df = train.iloc[:n_fit], train.iloc[n_fit:]
        X_fit = self.pre_.fit_transform(fit_df)
        params = {
            "objective": "survival:aft", "eval_metric": "aft-nloglik",
            "aft_loss_distribution": "normal", "aft_loss_distribution_scale": self.sigma,
            "learning_rate": self.learning_rate, "max_depth": self.max_depth,
            "min_child_weight": 20, "subsample": 0.8, "tree_method": "hist", "verbosity": 0,
        }
        self.booster_ = xgb.train(
            params, self._dmatrix(X_fit, fit_df), num_boost_round=self.max_rounds,
            evals=[(self._dmatrix(self.pre_.transform(stop_df), stop_df), "stop")],
            early_stopping_rounds=30, verbose_eval=False,
        )
        return self

    def _log_time(self, df):
        d = xgb.DMatrix(self.pre_.transform(df))
        pred = self.booster_.predict(d, iteration_range=(0, self.booster_.best_iteration + 1))
        return np.log(np.maximum(pred, 1e-6))

    def survival(self, df, times):
        mu = self._log_time(df)[:, None]
        return norm.sf((np.log(np.asarray(times, float))[None, :] - mu) / self.sigma)

    def describe(self):
        return {"model": self.name, "learning_rate": self.learning_rate, "max_depth": self.max_depth,
                "sigma": self.sigma, "rounds": int(self.booster_.best_iteration + 1)}
