"""Standalone mathematical checks for beyond_pinns.quadrature -- exact
integration of polynomials up to the guaranteed Gauss-Legendre degree
(2*npts - 1), independent of any GNDRM checkout.

Note on the weight/area convention: an n-point-per-cell rule here stores
REFERENCE weights that sum to 1 over [-1, 1] (not the textbook 2 -- see
beyond_pinns/quadrature.py's module docstring for why, and
test_gndrm_equivalence.py for the empirical confirmation against real
GNDRM), and `interval_quadpts`'s `integrator` applies a global `area = h`
factor (not `h/2`). The two conventions are complementary: for a single
subinterval of width h, `area * sum(weight_k * g(ref_node_k))
= h * (1/2) * integral_{-1}^{1} g(t) dt = integral_{x_l}^{x_r} f(x) dx`
via the standard affine substitution -- i.e. this construction computes the
mathematically correct integral, just factored unusually. These tests check
exactly that end-to-end correctness, not the intermediate factoring.
"""
import json
from pathlib import Path

import jax.numpy as jnp
import pytest

from beyond_pinns.quadrature import PiecewiseGaussLegendre1D

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "gndrm_golden.json").read_text())


def _integral_single_cell(npts, a, b, f):
    q = PiecewiseGaussLegendre1D(npts=npts)
    integ = q.interval_quadpts(jnp.array([[a, b]]), jnp.array([b - a]))
    return float(integ(lambda x: f(x.flatten())))


@pytest.mark.parametrize("npts", [1, 2, 3, 4, 8])
def test_integrates_constant_exactly(npts):
    got = _integral_single_cell(npts, 0.0, 1.0, lambda x: jnp.ones_like(x))
    assert abs(got - 1.0) < 1e-12


@pytest.mark.parametrize("npts", [2, 3, 4, 8])
def test_integrates_low_degree_polynomials_exactly(npts):
    # npts-point Gauss-Legendre is exact for polynomials up to degree 2*npts-1.
    max_degree = 2 * npts - 1
    a, b = -0.3, 1.7
    for degree in range(0, max_degree + 1):
        got = _integral_single_cell(npts, a, b, lambda x, d=degree: x ** d)
        expected = (b ** (degree + 1) - a ** (degree + 1)) / (degree + 1)
        assert abs(got - expected) < 1e-10, (npts, degree, got, expected)


def test_multi_cell_matches_single_cell_sum():
    # Two adjacent cells [0,0.5] and [0.5,1] should sum to the same result
    # as one direct integral over [0,1], for a function exact for the rule.
    npts = 4
    whole = _integral_single_cell(npts, 0.0, 1.0, lambda x: x ** 3)
    q = PiecewiseGaussLegendre1D(npts=npts)
    integ = q.interval_quadpts(jnp.array([[0.0, 1.0]]), jnp.array([0.5]))  # 2 cells
    got = float(integ(lambda x: x.flatten() ** 3))
    assert abs(got - whole) < 1e-10


def test_rejects_K_other_than_1():
    q = PiecewiseGaussLegendre1D(npts=4)
    with pytest.raises(NotImplementedError):
        q.interval_quadpts(jnp.array([[0.0, 1.0]]), jnp.array([0.5]), K=2)


def test_matches_golden_gndrm_fixture():
    """Golden regression check against a real GNDRM run captured earlier --
    does not require a live GNDRM checkout."""
    g = FIXTURE["quadrature_npts4"]
    q = PiecewiseGaussLegendre1D(npts=4)
    # GNDRM's eigendecomposition produces nodes/weights in a different (but
    # equally valid) order than leggauss's naturally-sorted output -- same
    # values, e.g. [-.861,-.340,.861,.340] vs [-.861,-.340,.340,.861] for
    # npts=4. Sorted comparison is the correct check (matching
    # test_gndrm_equivalence.py's live comparison); the unordered-but-correct
    # construction is exercised end-to-end by the exact-integration tests
    # above, which do not depend on node order at all.
    assert jnp.allclose(jnp.sort(q.quadpts.flatten()), jnp.sort(jnp.asarray(g["nodes"])), atol=1e-14)
    assert jnp.allclose(jnp.sort(q.weights.flatten()), jnp.sort(jnp.asarray(g["weights"])), atol=1e-14)

    g2 = FIXTURE["quadrature_npts4_interval_m1_1_h0125"]
    integ = q.interval_quadpts(jnp.array([[-1.0, 1.0]]), jnp.array([0.125]))
    const_got = float(integ(lambda x: jnp.ones((x.shape[0],))))
    poly_got = float(integ(lambda x: (x.flatten()) ** 3 - 2 * x.flatten()))
    assert abs(const_got - g2["const_integral"]) < 1e-12
    assert abs(poly_got - g2["poly_deg3_integral"]) < 1e-12


def test_reference_weights_sum_to_one_not_two():
    """Documents the (historically-matched, non-textbook) weight convention
    explicitly -- see module docstring. A silent switch back to the
    textbook sum-to-2 convention would break integration correctness given
    the `area = h` (not `h/2`) scaling used in `interval_quadpts`."""
    q = PiecewiseGaussLegendre1D(npts=4)
    assert abs(float(jnp.sum(q.weights)) - 1.0) < 1e-12
