"""Standalone unit test of the last-observation-carried-forward (LOCF)
aggregation logic used in scripts/make_figure_random_hats.py and
scripts/make_figure_main_comparison.py (both derived from the historical
`generate_pure_vs_projected_random_hats_panels.py` /
`generate_petrov_figures.py`, fixed during this release's Phase A audit --
see docs/PROVENANCE.md section 5.2).

This directly tests the specific bug that was found and fixed: the OLD
behavior truncated every seed's trajectory to the length of the
shortest-running seed before aggregating, which could make the plotted
curve's endpoint reflect an early, unconverged value rather than each seed's
true final (post-termination-constant) error. The function under test here
is a verbatim transcription (numpy, not jax) of the fixed
`median_iqr_curve`'s core LOCF logic.
"""
import numpy as np
import pytest


def median_iqr_curve_locf(histories):
    """`histories`: dict[seed] -> 1D array of recorded values (already-final
    value implicitly repeats for k beyond that seed's own length). Mirrors
    the fixed median_iqr_curve(histories, ykey) exactly, specialized to a
    single array per seed rather than a dict-of-arrays-per-seed."""
    hs = list(histories.values())
    n_max = max(len(h) for h in hs)
    Y = np.empty((len(hs), n_max))
    for i, h in enumerate(hs):
        n = len(h)
        Y[i, :n] = h
        if n < n_max:
            Y[i, n:] = h[-1]
    med = np.median(Y, axis=0)
    q1 = np.percentile(Y, 25, axis=0)
    q3 = np.percentile(Y, 75, axis=0)
    return Y, med, q1, q3


SEEDS = {
    "A": np.array([1.0, 0.5, 0.1]),                      # stops early (n=3)
    "B": np.array([1.0, 0.6, 0.3, 0.2, 0.05]),           # n=5
    "C": np.array([1.0, 0.7, 0.4, 0.2, 0.02]),           # n=5
    "D": np.array([1.0, 0.55, 0.25]),                    # stops early (n=3)
    "E": np.array([1.0, 0.65, 0.35, 0.18, 0.03]),        # n=5
}
N_SEEDS = len(SEEDS)
N_MAX = max(len(v) for v in SEEDS.values())


def test_extended_length_is_max_across_seeds():
    Y, med, q1, q3 = median_iqr_curve_locf(SEEDS)
    assert Y.shape == (N_SEEDS, N_MAX)
    assert med.shape == (N_MAX,)


def test_values_past_own_length_equal_own_last_value_not_another_seeds():
    Y, *_ = median_iqr_curve_locf(SEEDS)
    seed_names = list(SEEDS.keys())
    for i, name in enumerate(seed_names):
        own_last = SEEDS[name][-1]
        own_len = len(SEEDS[name])
        for k in range(own_len, N_MAX):
            assert Y[i, k] == own_last, (name, k)
        # and definitely not equal to some OTHER seed's raw recorded value at
        # that iteration unless coincidentally identical -- sanity spot check
        # against a seed we know differs numerically:
    assert Y[0, -1] == SEEDS["A"][-1] == 0.1  # seed A held constant at 0.1 to the end
    assert Y[0, -1] != SEEDS["B"][-1]  # not silently substituted with seed B's value


def test_median_at_final_iteration_equals_median_of_true_finals():
    _, med, _, _ = median_iqr_curve_locf(SEEDS)
    true_finals = np.array([v[-1] for v in SEEDS.values()])
    assert med[-1] == pytest.approx(np.median(true_finals))


def test_no_nan_and_full_sample_size_at_every_iteration():
    Y, med, q1, q3 = median_iqr_curve_locf(SEEDS)
    assert not np.any(np.isnan(Y))
    assert not np.any(np.isnan(med))
    # every column (iteration) is computed over all N_SEEDS values, never fewer
    for k in range(N_MAX):
        assert np.sum(~np.isnan(Y[:, k])) == N_SEEDS


def test_old_truncate_to_shortest_behavior_would_have_dropped_information():
    """Documents WHY the fix matters: the old (buggy) truncate-to-shortest
    convention discards exactly the information LOCF preserves."""
    shortest = min(len(v) for v in SEEDS.values())
    old_truncated_median_at_end = np.median([v[:shortest][-1] for v in SEEDS.values()])
    _, med, _, _ = median_iqr_curve_locf(SEEDS)
    locf_median_at_end = med[-1]
    # The old truncated endpoint reflects iteration `shortest`, not each
    # seed's true final -- for this synthetic example they differ:
    assert old_truncated_median_at_end != pytest.approx(locf_median_at_end)
