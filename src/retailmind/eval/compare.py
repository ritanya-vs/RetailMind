"""Paired comparison of two runs on the same queries."""

from __future__ import annotations

import numpy as np


def paired_comparison(
    scores_a: dict[str, float],
    scores_b: dict[str, float],
    *,
    n_resamples: int = 5000,
    seed: int = 0,
) -> dict[str, float]:
    """Compare per-query scores of run B against run A.

    Returns the mean difference (B - A), a 95% bootstrap interval for it, and how many queries
    B wins and loses. If the interval excludes 0 the difference is unlikely to be chance.
    """
    if set(scores_a) != set(scores_b):
        raise ValueError("both runs must be scored on the same queries")
    if not scores_a:
        raise ValueError("no queries to compare")
    ids = sorted(scores_a)
    diffs = np.array([scores_b[i] - scores_a[i] for i in ids], dtype=np.float64)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(diffs), size=(n_resamples, len(diffs)))
    means = diffs[picks].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {
        "mean_diff": float(diffs.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "wins": int((diffs > 1e-9).sum()),
        "losses": int((diffs < -1e-9).sum()),
        "n": len(diffs),
    }
