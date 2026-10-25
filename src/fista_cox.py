from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp


@dataclass
class _Stratum:
    idx: np.ndarray        # row indices, sorted by ascending time
    event: np.ndarray      # event indicator in that order (float)
    first: np.ndarray      # position of the first row with the same time (tie group start)
    last: np.ndarray       # position of the last row with the same time (tie group end)


def _prepare(time: np.ndarray, event: np.ndarray, strata: np.ndarray | None) -> list[_Stratum]:
    if strata is None:
        strata = np.zeros(len(time), dtype=int)
    out = []
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        idx = idx[np.argsort(time[idx], kind="stable")]
        t = time[idx]
        n = len(t)
        starts = np.r_[0, np.flatnonzero(np.diff(t) != 0) + 1]
        ends = np.r_[starts[1:] - 1, n - 1]
        sizes = ends - starts + 1
        out.append(_Stratum(
            idx=idx,
            event=event[idx].astype(float),
            first=np.repeat(starts, sizes),
            last=np.repeat(ends, sizes),
        ))
    return out


def neg_loglik(eta: np.ndarray, strata: list[_Stratum]) -> float:
    loss = 0.0
    for st in strata:
        e = eta[st.idx]
        m = e.max()
        risk = np.cumsum(np.exp(e - m)[::-1])[::-1][st.first]
        loss -= np.sum(st.event * (e - m - np.log(risk)))
    return loss


def neg_loglik_and_grad_eta(eta: np.ndarray, strata: list[_Stratum]) -> tuple[float, np.ndarray]:
    loss, grad = 0.0, np.zeros_like(eta)
    for st in strata:
        e = eta[st.idx]
        m = e.max()
        w = np.exp(e - m)
        risk = np.cumsum(w[::-1])[::-1][st.first]          # sum of w over each risk set
        loss -= np.sum(st.event * (e - m - np.log(risk)))
        c = np.cumsum(st.event / risk)[st.last]            # sum of 1/risk over relevant events
        grad[st.idx] = w * c - st.event
    return loss, grad


def prox_l1(v: np.ndarray, thresh: float) -> np.ndarray:
    return np.sign(v) * np.maximum(np.abs(v) - thresh, 0.0)


def prox_group(v: np.ndarray, thresh: float, groups: list[np.ndarray]) -> np.ndarray:
    out = v.copy()
    for g in groups:
        norm = np.linalg.norm(v[g])
        t = thresh * np.sqrt(len(g))
        out[g] = 0.0 if norm <= t else (1.0 - t / norm) * v[g]
    return out

@dataclass
class FitInfo:
    n_iter: int
    converged: bool
    objective: list[float] = field(default_factory=list)
    step_size: list[float] = field(default_factory=list)


class FistaCox:

    def __init__(self, alpha: float = 0.01, l1_ratio: float = 0.5, penalty: str = "elastic_net",
                 groups: list[np.ndarray] | None = None, tol: float = 1e-7, max_iter: int = 2000):
        if penalty not in ("elastic_net", "group"):
            raise ValueError("penalty must be 'elastic_net' or 'group'")
        if penalty == "group" and groups is None:
            raise ValueError("group penalty needs `groups`")
        if not 0 < l1_ratio <= 1:
            raise ValueError("l1_ratio must be in (0, 1]")
        self.alpha, self.l1_ratio, self.penalty = alpha, l1_ratio, penalty
        self.groups, self.tol, self.max_iter = groups, tol, max_iter

    def _smooth(self, beta, X, XT, strata, n):
        nll, g_eta = neg_loglik_and_grad_eta(X @ beta, strata)
        ridge = self.alpha * (1 - self.l1_ratio)
        f = nll / n + 0.5 * ridge * beta @ beta
        grad = XT @ g_eta / n + ridge * beta
        return f, np.asarray(grad).ravel()

    def _smooth_value(self, beta, X, strata, n):
        ridge = self.alpha * (1 - self.l1_ratio)
        return neg_loglik(X @ beta, strata) / n + 0.5 * ridge * beta @ beta

    def _penalty(self, beta):
        a = self.alpha * self.l1_ratio
        if self.penalty == "elastic_net":
            return a * np.abs(beta).sum()
        return a * sum(np.sqrt(len(g)) * np.linalg.norm(beta[g]) for g in self.groups)

    def _prox(self, v, step):
        thresh = step * self.alpha * self.l1_ratio
        if self.penalty == "elastic_net":
            return prox_l1(v, thresh)
        return prox_group(v, thresh, self.groups)

    def _prox_step(self, y, X, XT, st, n, step):
        f_y, g_y = self._smooth(y, X, XT, st, n)
        while True:
            x_new = self._prox(y - step * g_y, step)
            d = x_new - y
            f_new = self._smooth_value(x_new, X, st, n)
            if f_new <= f_y + g_y @ d + (d @ d) / (2 * step) + 1e-12:
                return x_new, f_new + self._penalty(x_new), step
            step *= 0.5

    def fit(self, X, time, event, strata=None, beta0=None, step0: float = 1.0, prepared=None):
        """Fit the Cox model with FISTA. `X` can be dense or sparse; `time` and `event` are 1D arrays."""
        X = sp.csr_matrix(X) if sp.issparse(X) else np.asarray(X, dtype=float)
        XT = X.T.tocsr() if sp.issparse(X) else X.T
        time, event = np.asarray(time, float), np.asarray(event, bool)
        n, p = X.shape
        st = prepared or _prepare(time, event, None if strata is None else np.asarray(strata))

        x = np.zeros(p) if beta0 is None else beta0.astype(float).copy()
        y, theta, step = x.copy(), 1.0, step0
        f_x = self._smooth_value(x, X, st, n) + self._penalty(x)
        info = FitInfo(0, False)

        for k in range(1, self.max_iter + 1):
            x_new, f_new, step = self._prox_step(y, X, XT, st, n, step)
            if f_new > f_x:
                theta = 1.0
                x_new, f_new, step = self._prox_step(x, X, XT, st, n, step)
            info.objective.append(f_new)
            info.step_size.append(step)

            theta_new = (1 + np.sqrt(1 + 4 * theta**2)) / 2
            y = x_new + ((theta - 1) / theta_new) * (x_new - x)
            change = np.linalg.norm(x_new - x) / max(1.0, np.linalg.norm(x))
            x, f_x, theta = x_new, min(f_new, f_x), theta_new
            step *= 1.2                                      # let the step grow back
            if change < self.tol:
                info.converged = True
                break
        info.n_iter = k
        self.coef_, self.info_ = x, info
        self._fit_baseline(X, time, event, strata)
        return self

    def fit_path(self, X, time, event, alphas, strata=None):
        """Fit a sequence of models with decreasing `alphas`, warm-starting each from the previous."""
        betas, beta = [], None
        for a in sorted(alphas, reverse=True):
            self.alpha = a
            self.fit(X, time, event, strata=strata, beta0=beta)
            beta = self.coef_
            betas.append((a, beta.copy()))
        return betas

    def _fit_baseline(self, X, time, event, strata):
        """Compute the baseline cumulative hazard for each stratum, after fitting."""
        eta = X @ self.coef_
        keys = np.zeros(len(time), dtype=int) if strata is None else np.asarray(strata)
        self.baseline_ = {}
        for s in np.unique(keys):
            m = keys == s
            t, e, w = time[m], event[m], np.exp(eta[m])
            order = np.argsort(t)
            t, e, w = t[order], e[order], w[order]
            risk = np.cumsum(w[::-1])[::-1]
            uniq, first = np.unique(t, return_index=True)
            deaths = np.add.reduceat(e.astype(float), first)
            keep = deaths > 0
            self.baseline_[s] = (uniq[keep], np.cumsum(deaths[keep] / risk[first][keep]))

    def cumulative_hazard(self, X, times, strata=None) -> np.ndarray:
        """Cumulative hazard at `times` for each row in `X`, using the fitted baseline."""
        eta = np.asarray(X @ self.coef_).ravel()
        keys = np.zeros(len(eta), dtype=int) if strata is None else np.asarray(strata)
        times = np.asarray(times, float)
        out = np.zeros((len(eta), len(times)))
        fallback = self._pooled_baseline()
        for s in np.unique(keys):
            t_s, h_s = self.baseline_.get(s, fallback)
            h0 = np.interp(times, t_s, h_s, left=0.0, right=h_s[-1]) if len(t_s) else np.zeros(len(times))
            m = keys == s
            out[m] = np.exp(eta[m])[:, None] * h0[None, :]
        return out

    def _pooled_baseline(self):
        """Average baseline hazard across strata, for missing strata at predict time."""
        grid = np.unique(np.concatenate([t for t, _ in self.baseline_.values()]))
        hs = [np.interp(grid, t, h, left=0, right=h[-1]) for t, h in self.baseline_.values() if len(t)]
        return grid, np.mean(hs, axis=0)
