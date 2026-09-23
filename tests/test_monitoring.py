"""Regression tests for beyond_pinns.monitoring.detect_trend_reversal against
the exact historical formula it replaces (the `has_passed_minimum` function
formerly copied into src/beyond_pinns/legacy/part_b_general.py from the
external `pinn` package -- see docs/PROVENANCE.md). The historical formula is
reproduced here ONLY for this regression comparison -- it is not shipped
anywhere in the release's production code (part_b_general.py now imports
detect_trend_reversal instead).
"""
import numpy as np
import pytest

from beyond_pinns.monitoring import detect_trend_reversal


def _historical_formula(sequence, window_size=10, min_slope=1e-4, min_correlation=0.1):
    """Exact historical algorithm, for comparison only -- not part of the release."""
    if len(sequence) < window_size:
        return False
    window = np.array(sequence[-window_size:], dtype=float)
    window = np.maximum(window, np.finfo(float).tiny)
    log_window = np.log10(window)
    x = np.arange(window_size)
    slope, _ = np.polyfit(x, log_window, 1)
    r_matrix = np.corrcoef(x, log_window)
    r_value = r_matrix[0, 1]
    if np.isnan(r_value):
        return False
    return slope > min_slope and r_value > min_correlation


CASES = {
    "monotone_decreasing": [10.0 ** (-k) for k in range(20)],
    "monotone_decreasing_short": [10.0 ** (-k) for k in range(5)],
    "after_minimum_rising": [10.0 ** (-min(k, 10)) if k <= 10 else 10.0 ** (-10 + (k - 10) * 0.3)
                              for k in range(25)],
    "flat": [1e-3] * 15,
    "flat_then_rising": [1e-6] * 12 + [1e-6 * 1.5 ** k for k in range(1, 9)],
    "noisy_no_trend": [1e-4 * (1 + 0.3 * np.sin(k)) for k in range(20)],
    "noisy_rising": [1e-6 * (1.4 ** k) * (1 + 0.1 * np.sin(3 * k)) for k in range(15)],
    "very_short": [1.0, 0.5, 0.3],
    "exactly_window_size_decreasing": [10.0 ** (-k) for k in range(10)],
    "exactly_window_size_flat": [1e-2] * 10,
    "zeros_and_positive": [0.0, 0.0, 0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
}


@pytest.mark.parametrize("name", list(CASES.keys()))
def test_matches_historical_formula(name):
    seq = CASES[name]
    assert detect_trend_reversal(seq) == _historical_formula(seq), name


def test_short_sequence_returns_false():
    assert detect_trend_reversal([1.0, 0.5]) is False


def test_monotone_decreasing_never_triggers():
    seq = [10.0 ** (-k) for k in range(30)]
    assert detect_trend_reversal(seq) is False


def test_clear_rise_after_plateau_triggers():
    seq = [1e-8] * 10 + [1e-8 * 2.0 ** k for k in range(1, 12)]
    assert detect_trend_reversal(seq) is True


def test_flat_sequence_does_not_trigger():
    assert detect_trend_reversal([1e-5] * 20) is False


@pytest.mark.parametrize("seed", range(5))
def test_random_sequences_match_historical_formula(seed):
    rng = np.random.RandomState(seed)
    seq = list(np.abs(rng.normal(1e-3, 1e-3, size=25)))
    assert detect_trend_reversal(seq) == _historical_formula(seq)
