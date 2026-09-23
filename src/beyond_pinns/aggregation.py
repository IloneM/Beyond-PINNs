"""Cross-seed aggregation for Part A convergence curves (median + IQR band).

Two axes need different treatment because only one of them is a shared,
data-independent grid:

* **Iteration-indexed curves** (``xkey == "iteration"``): the iteration grid
  is fixed and shared across seeds (checkpoints are recorded on the same
  cadence for every seed). Last-observation-carried-forward (LOCF) is
  applied to each seed's recorded quantity: a seed that stops at its own
  iteration ``T_s`` (e.g. because DSGNAR's adaptive trust-region radius
  collapsed below its stopping threshold -- the returned parameters are then
  fixed) has its plotted value held constant at its own true final value for
  every later common iteration, rather than being truncated out of the
  aggregate. No sample ever shrinks with iteration; nothing is fabricated,
  since the held-constant value is the seed's own true, known-final state.

* **Wallclock-indexed curves** (any other ``xkey``, e.g.
  ``"wallclock_optim"``): the x-axis itself is elapsed time, which is *not*
  shared across seeds -- each seed advances its own clock only while it is
  actually training. Reusing the iteration-indexed convention here (holding
  a per-seed "index" fixed and taking the median of each seed's wallclock
  *at that index*) can distort the x-axis once seeds run for very different
  numbers of iterations: seeds that finished early keep contributing a
  frozen, small elapsed-time value at every later index, pulling the median
  x-position left even though the still-running seeds have genuinely used
  much more wall-clock time to reach that later data point.
  ``aggregate_wallclock_curve`` instead resamples onto a *shared time grid*
  spanning ``[0, max over seeds of that seed's own true final wallclock]``
  and evaluates each seed as a right-continuous step function of elapsed
  time, held flat (LOCF in the time domain) after its own true last
  recorded timestamp. A finished seed's clock is therefore never advanced
  past the time it actually stopped, and no seed's elapsed time is ever
  invented -- only its own previously-recorded value is repeated.
"""
from __future__ import annotations

import numpy as np


def _locf_extend(values, n_max):
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == n_max:
        return values
    out = np.empty(n_max, dtype=float)
    out[:n] = values
    out[n:] = values[-1]
    return out


def aggregate_iteration_curve(histories, ykey, iteration_key="iteration"):
    """LOCF median/Q1/Q3 of `ykey` across seeds, indexed by the shared
    iteration grid, extended to the longest-running seed."""
    hs = list(histories.values())
    n_max = max(len(h[ykey]) for h in hs)
    Y = np.stack([_locf_extend(h[ykey], n_max) for h in hs])
    x = max(hs, key=lambda h: len(h[ykey]))[iteration_key][:n_max]
    return x, np.median(Y, axis=0), np.percentile(Y, 25, axis=0), np.percentile(Y, 75, axis=0)


def aggregate_wallclock_curve(histories, ykey, xkey="wallclock_optim", n_grid=None):
    """Median/Q1/Q3 of `ykey` across seeds on a shared wall-clock time grid.

    Each seed contributes a right-continuous step function of its own
    recorded (`xkey`, `ykey`) pairs, held flat after its own last recorded
    timestamp (LOCF in time, not in iteration index). The grid spans
    `[0, max over seeds of that seed's own final `xkey`]`; no seed's elapsed
    time is ever extrapolated beyond its own true last recorded value.
    """
    hs = list(histories.values())
    if n_grid is None:
        n_grid = max(len(h[xkey]) for h in hs)
    t_max = max(float(h[xkey][-1]) for h in hs)
    t_grid = np.linspace(0.0, t_max, n_grid)
    Y = np.empty((len(hs), n_grid))
    for i, h in enumerate(hs):
        t = np.asarray(h[xkey], dtype=float)
        y = np.asarray(h[ykey], dtype=float)
        idx = np.searchsorted(t, t_grid, side="right") - 1
        idx = np.clip(idx, 0, len(t) - 1)
        Y[i] = y[idx]
    return t_grid, np.median(Y, axis=0), np.percentile(Y, 25, axis=0), np.percentile(Y, 75, axis=0)


def aggregate_curve(histories, xkey, ykey, iteration_key="iteration", n_grid=None):
    """Dispatches to `aggregate_iteration_curve` when `xkey == "iteration"`,
    else to `aggregate_wallclock_curve`. Drop-in replacement for the
    historical `median_iqr_curve(histories, xkey, ykey)` call signature."""
    if xkey == iteration_key:
        return aggregate_iteration_curve(histories, ykey, iteration_key=iteration_key)
    return aggregate_wallclock_curve(histories, ykey, xkey=xkey, n_grid=n_grid)
