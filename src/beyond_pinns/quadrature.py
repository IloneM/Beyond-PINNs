"""Piecewise 1D Gauss-Legendre quadrature used by Part A's own methods to
build training/test/evaluation integrators on a uniformly-meshed interval.

Provenance
----------
The historical Part A script sourced this from the external GaussNewtonDRM
package (github.com/Jinxl-pp/GaussNewtonDRM, commit 9dd1ee8, no LICENSE --
see docs/PROVENANCE.md), via its `GaussLegendrePiecewise` class. Only the 1D
`interval_quadpts` path is used anywhere in Part A (the 2D
`rectangle_quadpts` method and the Quasi-Monte-Carlo/Monte-Carlo classes in
the same upstream file are never called). This module reimplements exactly
that 1D path from its mathematical definition -- no GNDRM source text,
comments, or internal helper names are reused.

Reference nodes/weights: rather than reproducing the historical
Golub-Welsch/Jacobi-eigendecomposition construction, this uses
`numpy.polynomial.legendre.leggauss` -- the standard library primitive for
exactly this purpose, and already used elsewhere in this codebase's own Part
B script. An n-point Gauss-Legendre rule is the unique rule of that order (by
the theory of orthogonal polynomials), so both constructions produce the same
NODES to essentially machine precision (confirmed empirically, max diff
~4.4e-16). The historical WEIGHTS, however, are exactly `leggauss`'s weights
divided by 2 (confirmed empirically: `GNDRM.weights == leggauss(npts)[1] / 2`
elementwise, to roundoff) -- i.e. they sum to 1 over [-1, 1], not the
"textbook" 2. This halving is preserved deliberately below (see
`interval_quadpts`'s docstring note on `area`), since it is compensated by a
matching convention in how the global `h`-scaling is applied, and the calling
Part A code was written/tuned against GNDRM's actual (not the textbook)
output. Empirical agreement (nodes, weights, and end-to-end `integrator`
results) is checked in tests/test_gndrm_equivalence.py when a GNDRM checkout
is available; exact-integration-of-polynomials checks against the closed-form
Gauss-Legendre error bound (not requiring GNDRM) live in
tests/test_quadrature.py.

Characterization of the historical `interval_quadpts(interval, h, K=1)`
behavior, preserved deliberately here:

  * `interval = [[a, b]]`, `h = [h_val]`: the interval [a, b] is covered by
    N = round((b - a) / h_val) + 1 uniformly spaced breakpoints (so
    N - 1 subintervals of width h_val -- requires (b - a) to be a multiple
    of h_val, exactly as historically required and never validated);
  * on each subinterval, the reference n-point rule on [-1, 1] is mapped
    affinely: physical_node = subinterval_midpoint + ref_node * h_val / 2;
  * the returned per-node weight is the RAW reference weight (which sums to
    1 over [-1, 1] -- see the halving note above), tiled once per
    subinterval -- NOT pre-scaled by the subinterval half-width. The
    physical h-scaling is instead applied once,
    globally, inside the returned `integrator(f)` closure, as
    `f(nodes) * weights * area` with `area = prod(h)` (i.e. `h_val` in 1D).
    This mirrors the historical construction exactly, including its
    convention of applying a single global `h_val` factor (not `h_val / 2`)
    -- callers historically compensate for this in how they define their
    own energy/residual expressions, so it is preserved as-is rather than
    "corrected";
  * nodes/weights are flattened in row-major (node-fastest) order across
    subintervals: all n reference nodes of subinterval 0, then all n of
    subinterval 1, etc.;
  * the `K` segmentation parameter of the historical class (splitting the
    flattened node list into K chunks summed separately, presumably for
    memory chunking) is never invoked with K != 1 anywhere in Part A, so it
    is not reimplemented; `interval_quadpts` here only supports K=1 and
    raises if anything else is requested.

Downstream callers pass a vectorized function `f(nodes) -> array` (nodes has
shape (n_total, 1)) to the returned `integrator`, exactly as historically.
"""
from __future__ import annotations

import jax.numpy as jnp
from numpy.polynomial.legendre import leggauss


class PiecewiseGaussLegendre1D:
    """n-point-per-cell Gauss-Legendre quadrature on a uniformly meshed 1D interval."""

    def __init__(self, npts: int):
        # `leggauss` returns the textbook n-point rule on [-1, 1] with weights
        # summing to 2 (the measure of [-1, 1]). The historical GNDRM
        # construction (a Golub-Welsch Jacobi-eigendecomposition, computing
        # weight_i = 2*v_i[0]**2 and then an extra "/2") produces weights
        # summing to 1 instead -- confirmed empirically (see
        # tests/test_gndrm_equivalence.py) to be uniformly leggauss's weights
        # divided by 2, not a different quadrature rule. This halving is
        # compensated by `interval_quadpts`'s own `area = prod(h)` convention
        # below (rather than the "textbook" `h/2`), so the two together still
        # integrate correctly end-to-end; preserved exactly rather than
        # "corrected" in isolation, per the module's provenance policy.
        ref_nodes, ref_weights = leggauss(int(npts))  # nodes/weights on [-1, 1]
        self.quadpts = jnp.asarray(ref_nodes, dtype=jnp.float64).reshape(-1, 1)
        self.weights = jnp.asarray(ref_weights, dtype=jnp.float64).reshape(-1, 1) / 2.0

    def interval_quadpts(self, interval, h, K: int = 1):
        if K != 1:
            raise NotImplementedError(
                "PiecewiseGaussLegendre1D only supports K=1 (no Part A experiment "
                "requests K != 1 from the historical quadrature either)."
            )
        a, b = interval[0][0], interval[0][1]
        h_val = h[0]
        n_breakpoints = int((b - a) / h_val + 1)
        breakpoints = jnp.linspace(a, b, n_breakpoints).reshape(1, n_breakpoints)
        left = breakpoints[0][:-1].reshape(1, n_breakpoints - 1)
        right = breakpoints[0][1:].reshape(1, n_breakpoints - 1)

        nodes = (self.quadpts * h_val + left + right) / 2          # (npts, n_cells)
        weights = jnp.tile(self.weights, nodes.shape[1])            # (npts, n_cells), raw reference weights
        nodes = nodes.flatten().reshape(-1, 1)
        weights = weights.flatten().reshape(-1, 1)

        def integrator(f):
            area = jnp.prod(h)
            f_val = f(nodes)
            broadcast_shape = [weights.shape[0]] + [1] * (len(f_val.shape) - 1)
            weighted = f_val * jnp.reshape(weights, broadcast_shape) * area
            return jnp.sum(weighted, axis=0)

        return integrator
