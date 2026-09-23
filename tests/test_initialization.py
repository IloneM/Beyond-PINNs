"""Deterministic checks for beyond_pinns.initialization -- no GNDRM required.

Golden values were captured from a live GNDRM checkout
(github.com/Jinxl-pp/GaussNewtonDRM, commit 9dd1ee8) and are also checked
against directly in test_gndrm_equivalence.py's golden-fixture path.
"""
import json
from pathlib import Path

import jax.numpy as jnp
from jax import random

from beyond_pinns.initialization import init_shallow_mlp_params

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "gndrm_golden.json").read_text())


def test_deterministic_for_fixed_key():
    key = random.PRNGKey(0)
    p1 = init_shallow_mlp_params([1, 32, 1], key)
    p2 = init_shallow_mlp_params([1, 32, 1], key)
    (w1a, b1a), w2a = p1
    (w1b, b1b), w2b = p2
    assert jnp.array_equal(w1a, w1b)
    assert jnp.array_equal(b1a, b1b)
    assert jnp.array_equal(w2a, w2b)


def test_different_seeds_differ():
    p0 = init_shallow_mlp_params([1, 32, 1], random.PRNGKey(0))
    p1 = init_shallow_mlp_params([1, 32, 1], random.PRNGKey(1))
    assert not jnp.array_equal(p0[0][0], p1[0][0])


def test_shapes():
    (w1, b1), w2 = init_shallow_mlp_params([1, 64, 1], random.PRNGKey(3))
    assert w1.shape == (64, 1)
    assert b1.shape == (64,)
    assert w2.shape == (1, 64)


def test_rejects_non_shallow_sizes():
    import pytest
    with pytest.raises(ValueError):
        init_shallow_mlp_params([1, 32, 32, 1], random.PRNGKey(0))


def test_matches_golden_gndrm_fixture():
    """Golden regression check against a real GNDRM run captured earlier --
    does not require a live GNDRM checkout (see test_gndrm_equivalence.py for
    the live comparison, which is skipped when GNDRM isn't installed).

    Tolerance is 1e-12, not bit-exact: the fixture was captured on a
    conda-installed jax build, and this test may run against a
    pip-installed jax build of the same version -- `jax.random`'s
    counter-based PRNG stream itself is portable/bit-exact, but the libm
    calls converting raw bits to normal-distributed floats can differ by a
    few ULPs across build toolchains (observed: ~1.4e-17, i.e. a couple of
    ULPs at float64 -- see docs/PROVENANCE.md's note on not claiming
    cross-environment bitwise determinism)."""
    g = FIXTURE["init_seed0_sizes_1_32_1"]
    (w1, b1), w2 = init_shallow_mlp_params(g["sizes"], random.PRNGKey(g["seed"]))
    assert jnp.allclose(w1, jnp.asarray(g["w1"]), atol=1e-12, rtol=0.0)
    assert jnp.allclose(b1, jnp.asarray(g["b1"]), atol=1e-12, rtol=0.0)
    assert jnp.allclose(w2, jnp.asarray(g["w2"]), atol=1e-12, rtol=0.0)
