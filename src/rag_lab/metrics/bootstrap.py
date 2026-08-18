"""Bootstrap confidence intervals (plan Phase 6, Step 6.5).

With ~200 queries a bootstrap CI on recall@k is roughly +/-5 points, meaning
two configurations within 5 points of each other are not distinguishable.
Every reported comparison in ``experiment report``/``compare`` carries one of
these -- without it, noise reads as signal.
"""

from __future__ import annotations

import numpy as np


def bootstrap_ci(
    values: list[float],
    n_iterations: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile-method bootstrap CI on the mean of ``values`` (resampling
    *queries* with replacement -- the query is the unit of measurement every
    per-query metric is computed over).

    Seeded so report output is reproducible across identical
    ``experiment report`` invocations -- the CI band is itself a statistic
    with its own sampling noise, and an unseeded RNG would make it wobble for
    no reason between two reads of the same run.
    """
    if not values:
        return (0.0, 0.0)
    arr = np.asarray(values, dtype=float)
    if len(arr) == 1:
        return (float(arr[0]), float(arr[0]))

    rng = np.random.default_rng(seed)
    n = len(arr)
    means = np.empty(n_iterations)
    for i in range(n_iterations):
        means[i] = rng.choice(arr, size=n, replace=True).mean()

    lo = float(np.percentile(means, 100 * (alpha / 2)))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return (lo, hi)


__all__ = ["bootstrap_ci"]
