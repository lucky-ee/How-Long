from __future__ import annotations

import argparse
import json
import os
import sqlite3
import tempfile
import time
import uuid
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import config
from src.backlog import add_backlog
from src.confirm import corrected_resampled_ttest, run_confirmatory
from src.diagnostics import ph_check
from src.evaluate import evaluate
from src.models import CoxnetModel, FistaModel, KMBaseline, XGBoostAFT, _step_eval
from src.survival import censor_at, forward_splits
from src.tuning import tune_grid, tune_path

warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass
class Settings:
    holdout_days: int = 60
    valid_days: int = 30
    outer_splits: int = 3
    inner_splits: int = 3
    l1_ratios: tuple = (0.1, 0.5, 1.0)
    xgb_grid: tuple = (("learning_rate", (0.05, 0.2)), ("max_depth", (4, 6)))
    linear_max_rows: int = 150_000     # Coxnet needs dense input
    sparse_max_rows: int = 500_000     # FISTA and XGBoost take sparse input
    min_xgb_gain: float = 0.005        # C-index gain XGBoost must show to be deployed

    @classmethod
    def fast(cls):
        return cls(holdout_days=45, valid_days=30, outer_splits=2, inner_splits=2,
                   l1_ratios=(0.5, 1.0), xgb_grid=(("learning_rate", (0.1,)), ("max_depth", (4,))),
                   linear_max_rows=40_000, sparse_max_rows=60_000)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def candidates(stratify: bool, s: Settings) -> dict:
    strata = "category" if stratify else None

    if stratify:   # Coxnet can't stratify; our solver can
        def linear(l1, alpha=None):
            return FistaModel(l1_ratio=l1, alpha=alpha, strata_col=strata, max_rows=s.sparse_max_rows)
    else:
        def linear(l1, alpha=None):
            return CoxnetModel(l1_ratio=l1, alpha=alpha, max_rows=s.linear_max_rows)

    def group(l1, alpha=None):
        return FistaModel(l1_ratio=l1, alpha=alpha, penalty="group", strata_col=strata,
                          max_rows=s.sparse_max_rows)

    def xgbm(**p):
        return XGBoostAFT(max_rows=s.sparse_max_rows, **p)

    def path_entry(factory):
        return (lambda splits: tune_path(factory, list(s.l1_ratios), splits),
                lambda p: factory(p["l1_ratio"], p["alpha"]))

    return {
        "km_baseline": (lambda splits: {}, lambda p: KMBaseline()),
        "linear": path_entry(linear),
        "group_lasso": path_entry(group),
        "xgboost": (lambda splits: tune_grid(xgbm, dict(s.xgb_grid), splits),
                    lambda p: xgbm(**{k: p[k] for k, _ in s.xgb_grid})),
    }


def fit_tuned(build, params, train):
    model = build(params)
    if "curves" in params:
        grid = np.array(params["curves"][str(params["l1_ratio"])]["alphas"])
        model.fit_path(train, alphas=grid)
        return model.set_alpha(params["alpha"])
    return model.fit(train)


def strip(params: dict) -> dict:
    return {k: v for k, v in params.items() if k not in ("curves", "grid")}


def neighborhood_oe(holdout: pd.DataFrame, km: KMBaseline) -> list[dict]:
    h = holdout.dropna(subset=["neighborhood"])
    expected = np.empty(len(h))
    for key, idx in h.groupby("request_type").indices.items():
        x, surv = km.curves_.get(key, km.global_)
        s = _step_eval(x, surv, h["duration_hours"].to_numpy()[idx], left=1.0)
        expected[idx] = -np.log(np.clip(s, 1e-6, 1.0))
    h = h.assign(_exp=expected)
    out = h.groupby("neighborhood").agg(observed=("event", "sum"), expected=("_exp", "sum"), cases=("event", "size"))
    out = out[out["cases"] >= 30]
    out["oe"] = out["observed"] / out["expected"]
    out["median_hours"] = h.groupby("neighborhood")["duration_hours"].median()
    return [{"neighborhood": n, **{k: float(v) for k, v in r.items()}} for n, r in out.iterrows()]


def serving_options(df: pd.DataFrame, as_of: pd.Timestamp) -> dict:
    rt = df.dropna(subset=["request_type", "category"])
    counts = rt.groupby(["request_type", "category"]).size().reset_index(name="n")
    top = counts.sort_values("n", ascending=False).drop_duplicates("request_type")
    open_now = df[df["event"] == 0].dropna(subset=["request_type", "supervisor_district"])
    backlog = open_now.groupby(["request_type", "supervisor_district"]).size()
    districts = (df.dropna(subset=["supervisor_district", "neighborhood"])
                 .groupby("supervisor_district")["neighborhood"].agg(lambda s: s.value_counts().index[0]))
    return {
        "as_of": str(as_of),
        "request_types": [{"request_type": r, "category": c, "cases": int(n)}
                          for r, c, n in top[["request_type", "category", "n"]].itertuples(index=False)
                          if n >= 50],
        "sources": sorted(df["source"].dropna().unique().tolist()),
        "districts": [{"district": int(d), "neighborhood": n} for d, n in districts.items()],
        "backlog": {f"{r}|{int(d)}": int(v) for (r, d), v in backlog.items()},
    }


def json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def atomic_write(path: Path, write) -> None:
    """Calls write(tmp) on a temp file beside `path`, then swaps it in, so the API (which reloads
    whenever these files change) never reads a half-written one."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    tmp = Path(tmp)
    try:
        write(tmp)
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(tmp, 0o666 & ~umask)  # mkstemp makes it owner-only; match a normal write
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def save_run(report: dict) -> None:
    con = sqlite3.connect(config.RUNS_DB)
    con.execute("""CREATE TABLE IF NOT EXISTS runs (
        run_id TEXT PRIMARY KEY, created_at TEXT, deployed TEXT,
        holdout_c_index REAL, settings TEXT, report TEXT)""")
    dep = report["deployed"]["name"]
    con.execute("INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?)", (
        report["run_id"], report["created_at"], dep,
        report["holdout"][dep]["c_index"], json.dumps(report["settings"]), json.dumps(report)))
    con.commit()
    con.close()


def run(cases: pd.DataFrame, as_of: pd.Timestamp, s: Settings, out_dir=config.ARTIFACT_DIR) -> dict:
    t_start = time.time()
    log(f"{len(cases):,} cases; computing backlog")
    df = add_backlog(cases)
    df = df[~df["is_duplicate"]].sort_values("opened").reset_index(drop=True)

    holdout_start = df["opened"].max() - pd.to_timedelta(int(s.holdout_days), unit="D")
    dev = censor_at(df, holdout_start)
    holdout = df[df["opened"] >= holdout_start]
    log(f"dev {len(dev):,} cases, holdout {len(holdout):,} cases from {holdout_start:%Y-%m-%d}")

    ph = ph_check(dev)
    stratify = ph["decision"] == "stratify_by_category"
    log(f"proportional hazards: {ph['decision']}")
    cands = candidates(stratify, s)

    outer = forward_splits(dev, s.outer_splits, s.valid_days)
    cv = {name: [] for name in cands}
    for k, split in enumerate(outer, 1):
        inner = forward_splits(split.train, s.inner_splits, s.valid_days)
        for name, (tune, build) in cands.items():
            params = tune(inner)
            model = fit_tuned(build, params, split.train)
            scores = evaluate(model, split.train, split.valid)
            cv[name].append({"cutoff": str(split.cutoff), "params": strip(params), **scores})
            log(f"outer {k}/{len(outer)} {name:12s} C={scores['c_index']:.3f}")

    n_train = np.mean([len(sp.train) for sp in outer])
    n_valid = np.mean([len(sp.valid) for sp in outer])
    diffs = [x["c_index"] - l["c_index"] for x, l in zip(cv["xgboost"], cv["linear"])]
    xgb_test = corrected_resampled_ttest(diffs, n_train, n_valid)
    deploy_xgb = (xgb_test["p"] < 0.05) and (xgb_test["mean_diff"] >= s.min_xgb_gain)
    deployed_name = "xgboost" if deploy_xgb else "linear"
    log(f"xgboost vs linear: diff={xgb_test['mean_diff']:+.4f}, p={xgb_test['p']:.3g} -> deploy {deployed_name}")

    final_inner = forward_splits(dev, s.inner_splits, s.valid_days)
    fitted, final_params, holdout_scores = {}, {}, {}
    for name, (tune, build) in cands.items():
        params = tune(final_inner)
        fitted[name] = fit_tuned(build, params, dev)
        final_params[name] = params
        holdout_scores[name] = {"params": strip(params), **evaluate(fitted[name], dev, holdout)}
        log(f"holdout     {name:12s} C={holdout_scores[name]['c_index']:.3f}")

    confirm = run_confirmatory(holdout, fitted[deployed_name].risk(holdout), fitted["km_baseline"].risk(holdout))
    for h in confirm["hypotheses"]:
        log(f"{h['id']} p_holm={h['p_holm']:.3g} supported={h['supported']}")

    serving = fit_tuned(cands[deployed_name][1], final_params[deployed_name], df)

    linear = fitted["linear"]
    coefs = linear.coef_table().assign(hazard_ratio=lambda d: np.exp(d["coef"]))
    report = {
        "run_id": uuid.uuid4().hex[:12],
        "created_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "data": {"as_of": str(as_of), "cases": len(df), "dev_cases": len(dev), "holdout_cases": len(holdout),
                 "holdout_start": str(holdout_start), "share_open": float(1 - df["event"].mean())},
        "settings": asdict(s),
        "ph_check": ph,
        "cv": cv,
        "model_selection": {"xgboost_vs_linear": xgb_test, "rule": (
            f"deploy xgboost only if mean C-index gain >= {s.min_xgb_gain} and corrected t-test p < 0.05")},
        "deployed": {"name": deployed_name, **serving.describe()},
        "holdout": holdout_scores,
        "confirmatory": confirm,
        "linear_model": {
            "describe": linear.describe(),
            "coefficients": coefs[coefs["coef"] != 0].sort_values("coef").to_dict("records"),
            "path": linear.path(),
            "cv_curves": final_params["linear"]["curves"],
        },
        "group_lasso": {"describe": fitted["group_lasso"].describe(), "path": fitted["group_lasso"].path()},
        "neighborhoods": neighborhood_oe(holdout, fitted["km_baseline"]),
        "seconds": round(time.time() - t_start, 1),
    }
    report = json_safe(report)

    out_dir.mkdir(parents=True, exist_ok=True)
    report_text = json.dumps(report, indent=2, allow_nan=False)
    options_text = json.dumps(serving_options(df, as_of), indent=2)
    bundle = {"model": serving, "deployed": deployed_name}
    atomic_write(out_dir / "model.joblib", lambda tmp: joblib.dump(bundle, tmp))
    atomic_write(out_dir / "report.json", lambda tmp: tmp.write_text(report_text))
    atomic_write(out_dir / "options.json", lambda tmp: tmp.write_text(options_text))
    if out_dir == config.ARTIFACT_DIR:
        save_run(report)
    log(f"done in {report['seconds']}s; artifacts in {out_dir}")
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="Train and evaluate all models.")
    p.add_argument("--fast", action="store_true", help="smaller CV for a quick check")
    args = p.parse_args()
    cases = pd.read_parquet(config.CLEAN_DIR / "cases.parquet")
    as_of = pd.Timestamp(json.loads(config.MANIFEST.read_text())["fetched_at"])
    run(cases, as_of, Settings.fast() if args.fast else Settings())


if __name__ == "__main__":
    main()