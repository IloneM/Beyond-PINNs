"""Optimizer-diagnostic monitoring utilities.

Provenance
----------
The historical Part B script's `has_passed_minimum` function carried the
comment "(verbatim from pinn.trainer64)" -- a small heuristic copied from the
external `pinn` package (github.com/IloneM/physics-informed-neural-networks,
commit 8b30ec3, no LICENSE -- see docs/PROVENANCE.md) into the MIT-licensed
research script. Since `pinn` has no license, this release does not carry
that copied source: `detect_trend_reversal` below is an INDEPENDENT
reimplementation from the mathematical specification (a sliding-window
log-linear-regression slope/correlation check), using different variable
names and structure -- no `pinn` source text or comments are reused. The
release's `beyond_pinns.legacy.part_b_general` no longer contains any
copied third-party source; see docs/PROVENANCE.md for the historical-vs-
release accounting.

Mathematical specification (reproduced exactly, verified by regression
tests in tests/test_monitoring.py against the historical behavior on
monotone-decreasing, post-minimum, flat, noisy, and short-window sequences):

  1. If fewer than `window` values have been recorded, return False (not
     enough history to judge a trend).
  2. Take the last `window` values, floor them at the smallest representable
     positive float64 (avoids log(0)/log(negative) for a sequence that may
     legitimately reach exactly 0 or dip non-positive due to floating point).
  3. Fit a degree-1 least-squares polynomial (ordinary linear regression) of
     log10(value) against the window's own index (0..window-1); take its
     slope.
  4. Compute the Pearson correlation coefficient between the index sequence
     and log10(value) over the same window.
  5. Return True iff the slope exceeds `min_slope` (the log-value trend is
     rising, i.e. the historically-monitored quantity -- e.g. a
     trust-region regularization parameter -- has stopped decreasing and
     started climbing) AND the correlation exceeds `min_correlation` (the
     rising trend is not just noise). A NaN correlation (e.g. a perfectly
     flat window, zero variance) is treated as "no trend detected".
"""
from __future__ import annotations

import numpy as np


def detect_trend_reversal(values, window_size: int = 10, min_slope: float = 1e-4,
                           min_correlation: float = 0.1) -> bool:
    """True iff the last `window_size` values of `values` show a
    statistically meaningful upward trend in log10-space (a "bottomed out
    and started rising" detector). See module docstring for the exact
    specification. (`window_size` keyword name matches the historical call
    sites in `beyond_pinns.legacy.part_b_general` exactly, for a drop-in,
    zero-diff substitution at every call site.)
    """
    if len(values) < window_size:
        return False
    recent = np.asarray(values[-window_size:], dtype=float)
    recent = np.maximum(recent, np.finfo(float).tiny)
    log_recent = np.log10(recent)
    index = np.arange(window_size)
    slope, _intercept = np.polyfit(index, log_recent, 1)
    correlation_matrix = np.corrcoef(index, log_recent)
    correlation = correlation_matrix[0, 1]
    if np.isnan(correlation):
        return False
    return bool(slope > min_slope and correlation > min_correlation)
