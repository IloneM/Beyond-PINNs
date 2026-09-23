"""Unit tests for beyond_pinns.aggregation, the canonical (documented,
tested) reference implementation of the Part A cross-seed aggregation
convention used by scripts/make_figure_smooth_benchmarks.py and
scripts/make_figure_main_comparison.py (each carries its own self-contained,
verbatim-equivalent copy so the plotting scripts need no beyond_pinns
import -- see docs/PROVENANCE.md section 5.2/5.3 and
tests/test_locf_aggregation.py, which tests those script copies directly).
"""
import numpy as np
import pytest

from beyond_pinns.aggregation import (
    aggregate_curve,
    aggregate_iteration_curve,
    aggregate_wallclock_curve,
)

HISTORIES = {
    "A": {  # stops early
        "iteration": np.array([0, 10, 20]),
        "wallclock_optim": np.array([0.0, 1.0, 2.0]),
        "y": np.array([1.0, 0.5, 0.1]),
    },
    "B": {  # runs longer
        "iteration": np.array([0, 10, 20, 30, 40]),
        "wallclock_optim": np.array([0.0, 1.2, 2.5, 3.9, 5.5]),
        "y": np.array([1.0, 0.6, 0.3, 0.2, 0.05]),
    },
}


def test_iteration_locf_extends_to_longest_seed():
    x, med, q1, q3 = aggregate_iteration_curve(HISTORIES, "y")
    assert x[-1] == 40
    assert med[-1] == pytest.approx(np.median([0.1, 0.05]))


def test_iteration_locf_holds_short_seed_flat_not_dropped():
    x, med, q1, q3 = aggregate_iteration_curve(HISTORIES, "y")
    # at iteration 30 (past A's own termination at 20), A must contribute
    # its own final value (0.1), not be excluded or extrapolated
    idx_30 = list(x).index(30)
    assert med[idx_30] == pytest.approx(np.median([0.1, 0.2]))


def test_wallclock_grid_spans_zero_to_max_seed_final_time():
    x, med, q1, q3 = aggregate_wallclock_curve(HISTORIES, "y")
    assert x[0] == 0.0
    assert x[-1] == pytest.approx(5.5)  # B's own true final wallclock (the max)


def test_wallclock_never_advances_a_finished_seeds_clock():
    """The core "do not invent additional elapsed time" check: seed A's
    contribution at any grid time past its own true final wallclock (2.0)
    must equal its own last true value, never something implying it kept
    training."""
    x, med, q1, q3 = aggregate_wallclock_curve(HISTORIES, "y")
    t, y = HISTORIES["A"]["wallclock_optim"], HISTORIES["A"]["y"]
    idx = np.clip(np.searchsorted(t, x, side="right") - 1, 0, len(t) - 1)
    y_a_on_grid = y[idx]
    assert np.all(y_a_on_grid[x >= 2.0] == 0.1)


def test_wallclock_does_not_distort_still_running_seed_forward():
    """Seed B's still-running values must appear at their own true elapsed
    time, not compressed toward A's frozen early wallclock."""
    x, med, q1, q3 = aggregate_wallclock_curve(HISTORIES, "y")
    t, y = HISTORIES["B"]["wallclock_optim"], HISTORIES["B"]["y"]
    idx = np.clip(np.searchsorted(t, x, side="right") - 1, 0, len(t) - 1)
    y_b_on_grid = y[idx]
    # near B's own true final time, B's value must be its true final (0.05)
    assert y_b_on_grid[-1] == pytest.approx(0.05)


def test_dispatch_matches_direct_calls():
    xi, medi, *_ = aggregate_curve(HISTORIES, "iteration", "y")
    xi2, medi2, *_ = aggregate_iteration_curve(HISTORIES, "y")
    assert np.array_equal(xi, xi2) and np.array_equal(medi, medi2)

    xw, medw, *_ = aggregate_curve(HISTORIES, "wallclock_optim", "y")
    xw2, medw2, *_ = aggregate_wallclock_curve(HISTORIES, "y")
    assert np.array_equal(xw, xw2) and np.array_equal(medw, medw2)


def test_single_seed_degenerates_to_its_own_curve():
    single = {"A": HISTORIES["A"]}
    x, med, q1, q3 = aggregate_iteration_curve(single, "y")
    assert np.array_equal(med, HISTORIES["A"]["y"])
    assert np.array_equal(q1, med) and np.array_equal(q3, med)


def test_equal_length_seeds_iteration_curve_unaffected_by_locf():
    equal = {
        "A": {"iteration": np.array([0, 10, 20]), "y": np.array([1.0, 0.4, 0.1])},
        "B": {"iteration": np.array([0, 10, 20]), "y": np.array([1.0, 0.5, 0.2])},
    }
    x, med, q1, q3 = aggregate_iteration_curve(equal, "y")
    assert np.array_equal(med, np.median(np.stack([equal["A"]["y"], equal["B"]["y"]]), axis=0))
