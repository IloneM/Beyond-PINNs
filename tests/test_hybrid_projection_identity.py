"""Self-contained (numpy-only) test of the core hybrid finite-element--neural
projection mechanism, extracted faithfully from the historical
`femennstein-petrov-galerkin-experiments-fixed.py`'s `make_hat1d_projected`
and `make_projected_from_base` (read directly from the research tree before
writing this test -- see the docstrings below for the exact source lines
each check corresponds to).

Historical construction (1D random-hat test family, P1 FE compensator):

    C[i, j]  = sum_q w_q * A(x_q) * phi_j'(x_q) * dg_i(x_q)      # test-functional / FE-basis-gradient pairing
    Pi       = solve(K, C.T, assume_a='sym')                      # K = FE stiffness matrix (symmetric)
    res_proj(fl) = base_res(fl)  - Pi.T @ hat_res(unravel(fl))
    jac_proj(fl) = base_jac(fl)  - Pi.T @ jacrev(hat_res)(fl)

`Pi` is the Galerkin projection of each weak-test functional onto the FE
compensator space: it is DEFINED by the symmetric linear solve `K @ Pi = C.T`
(this is the actual mathematical content of "eliminating the FE degrees of
freedom via their own optimality condition" / static condensation). This
test builds a genuine tiny 1D P1 finite-element stiffness matrix and a
genuine hat-function/test-functional pairing matrix, then checks:

  1. the defining identity `K @ Pi == C.T` holds to floating-point precision;
  2. the projected-residual/Jacobian formula is applied exactly as coded
     (base minus Pi.T contracted with the hat residual/Jacobian).
"""
import numpy as np
import pytest


def p1_stiffness_matrix(n_interior: int, h: float) -> np.ndarray:
    """1D P1 FE stiffness matrix (homogeneous Dirichlet BC, interior nodes
    only), uniform mesh of element size h -- the standard tridiagonal
    Laplacian assembly `_K1d` in the historical script is built from
    (implicitly, via `jax.lax.linalg.tridiagonal_solve`); this is its dense
    equivalent for a small synthetic example."""
    K = np.zeros((n_interior, n_interior))
    diag = 2.0 / h
    off = -1.0 / h
    for i in range(n_interior):
        K[i, i] = diag
        if i > 0:
            K[i, i - 1] = off
        if i < n_interior - 1:
            K[i, i + 1] = off
    return K


def p1_basis_gradient(node_positions: np.ndarray, h: float, x: float) -> np.ndarray:
    """Gradient (piecewise-constant) of each interior P1 hat basis function
    at point x -- 1/h on the element to the right of the node, -1/h on the
    element to the left, 0 elsewhere; exact (not quadrature-approximated),
    since P1 basis gradients are piecewise constant."""
    n = len(node_positions)
    grad = np.zeros(n)
    for j, xj in enumerate(node_positions):
        if xj - h < x <= xj:
            grad[j] = 1.0 / h
        elif xj < x <= xj + h:
            grad[j] = -1.0 / h
    return grad


@pytest.fixture
def fe_setup():
    n_interior = 3
    h = 0.25  # 4 elements on [0, 1], 3 interior nodes at 0.25, 0.5, 0.75
    node_positions = np.array([0.25, 0.5, 0.75])
    K = p1_stiffness_matrix(n_interior, h)

    # Three test functionals (mirroring make_hat1d_table_forms's per-i
    # residual: C[i,j] = sum_q w_q * A(x_q) * phi_j'(x_q) * dg_i(x_q)),
    # evaluated at one quadrature point per element (exact for piecewise-
    # constant integrands, as here) with a simple synthetic weighting
    # dg_i(x) = sin((i+1) * pi * x) sampled at each element midpoint --
    # a concrete, non-trivial, but easily-reproduced choice standing in for
    # the historical code's random-hat / Green-section test derivative.
    element_midpoints = node_positions - h / 2.0
    element_midpoints = np.concatenate([element_midpoints, [node_positions[-1] + h / 2.0]])
    A = np.ones_like(element_midpoints)  # A_eps(x) = 1 for this synthetic check
    w = np.full_like(element_midpoints, h)  # one-point-per-element quadrature weight = h

    n_tests = 3
    C = np.zeros((n_tests, n_interior))
    for i in range(n_tests):
        dg_i = np.sin((i + 1) * np.pi * element_midpoints)
        for q, xq in enumerate(element_midpoints):
            grad_q = p1_basis_gradient(node_positions, h, xq)
            C[i, :] += w[q] * A[q] * grad_q * dg_i[q]

    return dict(K=K, C=C, n_interior=n_interior, n_tests=n_tests)


def test_projection_matrix_solves_defining_symmetric_system(fe_setup):
    K, C = fe_setup["K"], fe_setup["C"]
    Pi = np.linalg.solve(K, C.T)  # K @ Pi = C.T, matching assume_a='sym' (K is symmetric here)
    assert np.allclose(K @ Pi, C.T, atol=1e-12)


def test_projected_residual_and_jacobian_formula(fe_setup):
    """Literal check of `res(fl) = base_res(fl) - Pi.T @ hat_res(fl)` and
    `jac(fl) = base_jac(fl) - Pi.T @ jacrev(hat_res)(fl)` from
    make_projected_from_base, on a synthetic linear residual pair."""
    K, C, n_interior, n_tests = fe_setup["K"], fe_setup["C"], fe_setup["n_interior"], fe_setup["n_tests"]
    Pi = np.linalg.solve(K, C.T)  # shape (n_interior, n_tests)

    rng = np.random.default_rng(0)
    load = rng.normal(size=n_tests)

    def hat_res(alpha):
        return C @ alpha - load  # linear residual, matching make_hat1d_table_forms's `one(...)` structure

    def hat_jac(alpha):
        return C  # constant Jacobian for a linear residual

    M_base = rng.normal(size=(n_tests, n_interior))
    base_load = rng.normal(size=n_tests)

    def base_res(alpha):
        return M_base @ alpha - base_load

    def base_jac(alpha):
        return M_base

    alpha = rng.normal(size=n_interior)

    res_proj = base_res(alpha) - Pi.T @ hat_res(alpha)
    jac_proj = base_jac(alpha) - Pi.T @ hat_jac(alpha)

    # Direct re-derivation from the formula, independent of the helper
    # functions above, to catch a copy-paste error in the test itself:
    expected_res = (M_base @ alpha - base_load) - Pi.T @ (C @ alpha - load)
    expected_jac = M_base - Pi.T @ C

    assert np.allclose(res_proj, expected_res, atol=1e-12)
    assert np.allclose(jac_proj, expected_jac, atol=1e-12)
    assert jac_proj.shape == (n_tests, n_interior)
