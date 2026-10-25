from __future__ import annotations

import itertools

import numpy as np

from src.evaluate import c_index
from src.survival import Split


def _check(splits: list[Split]) -> None:
    if not splits:
        raise ValueError("no usable CV splits; use more data or fewer/shorter validation windows")


def tune_path(factory, l1_ratios: list[float], splits: list[Split]) -> dict:
    """Pick (l1_ratio, alpha) by mean validation C-index along each regularization path.
    The alpha grid comes from the latest split (the most training data) and is reused on the
    others, so their curves line up and can be averaged."""
    _check(splits)
    curves, best = {}, None
    for l1 in l1_ratios:
        first = factory(l1).fit_path(splits[-1].train)
        grid = np.asarray(first.alphas_, float)
        scores = np.empty((len(splits), len(grid)))
        for i, split in enumerate(splits):
            model = first if i == len(splits) - 1 else factory(l1).fit_path(split.train, alphas=grid)
            for j, a in enumerate(grid):
                scores[i, j] = c_index(split.train, split.valid, model.set_alpha(a).risk(split.valid))
        mean = scores.mean(axis=0)
        curves[str(l1)] = {"alphas": grid.tolist(), "c_index": mean.tolist(),
                           "c_index_sd": scores.std(axis=0).tolist()}
        j = int(np.nanargmax(mean))
        if best is None or mean[j] > best["c_index"]:
            best = {"l1_ratio": l1, "alpha": float(grid[j]), "c_index": float(mean[j])}
    return {**best, "curves": curves}


def tune_grid(factory, grid: dict[str, tuple], splits: list[Split]) -> dict:
    """Exhaustive search over `grid`, scored by mean validation C-index."""
    _check(splits)
    results = []
    for values in itertools.product(*grid.values()):
        params = dict(zip(grid, values))
        scores = [c_index(s.train, s.valid, factory(**params).fit(s.train).risk(s.valid)) for s in splits]
        results.append({"params": params, "c_index": float(np.mean(scores))})
    best = max(results, key=lambda r: r["c_index"])
    return {**best["params"], "c_index": best["c_index"], "grid": results}
