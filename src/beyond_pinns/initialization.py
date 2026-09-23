"""Gaussian parameter initializer for the shallow one-hidden-layer network
used throughout Part A's own methods (plain_gn_*, amstramgram_*,
dsgnar_*).

Provenance
----------
The historical Part A script sourced this from the external GaussNewtonDRM
package (github.com/Jinxl-pp/GaussNewtonDRM, commit 9dd1ee8, no LICENSE --
see docs/PROVENANCE.md). What it computes is a completely generic,
textbook Gaussian weight initializer with no algorithmic content specific to
that repository's own contribution (which lies entirely in its Gauss-Newton
optimizer machinery, not its initializer). This module is written from the
mathematical specification, independently -- no GNDRM source text, comments,
or internal helper names are reused.

To remain a drop-in numerical replacement for every historically-seeded
experiment, the RNG semantics below are pinned exactly (verified in
tests/test_initialization.py, plus an optional bit-for-bit comparison against
a live GNDRM checkout in tests/test_gndrm_equivalence.py):

  * one `jax.random.split(key, n_layers)` call producing exactly one subkey
    per network layer, consumed in layer order;
  * per layer, exactly one further `jax.random.split` into a weight-subkey
    and a bias-subkey, drawn in that order;
  * both weight and bias entries sampled from `jax.random.normal`, then
    scaled by 0.1 and shifted by 0.0;
  * weight matrix shape (out_features, in_features); bias shape
    (out_features,);
  * float64 throughout (the caller enables `jax.config.update
    ("jax_enable_x64", True)` globally, as in the historical script).

Architecture note: the returned pytree assumes exactly one hidden layer
(`sizes` has 3 entries: [in_dim, hidden_width, out_dim]) and omits a bias on
the final/output layer, matching `beyond_pinns.models.shallow_mlp` exactly --
this is not a general N-layer initializer.
"""
from __future__ import annotations

from jax import random


def _sample_affine_layer(fan_in: int, fan_out: int, key, scale: float = 0.1, shift: float = 0.0):
    """One (weight, bias) pair, weight shape (fan_out, fan_in), bias shape (fan_out,)."""
    weight_key, bias_key = random.split(key)
    weight = scale * random.normal(weight_key, (fan_out, fan_in)) + shift
    bias = scale * random.normal(bias_key, (fan_out,)) + shift
    return weight, bias


def init_shallow_mlp_params(sizes, key):
    """Initialize parameters for `beyond_pinns.models.shallow_mlp(sizes, ...)`.

    `sizes` = [in_dim, hidden_width, out_dim] (exactly 3 entries). Returns
    `[(W_hidden, b_hidden), W_out]` -- a hidden affine layer with bias, and a
    bias-free output layer, matching the historical pytree shape exactly.
    """
    if len(sizes) != 3:
        raise ValueError(
            f"init_shallow_mlp_params expects exactly 3 sizes (in, hidden, out), got {sizes!r}. "
            "This mirrors the historical initializer's own (undocumented) restriction to "
            "one-hidden-layer networks -- see module docstring."
        )
    layer_keys = random.split(key, len(sizes))
    hidden_weight, hidden_bias = _sample_affine_layer(sizes[0], sizes[1], layer_keys[0])
    output_weight, _unused_output_bias = _sample_affine_layer(sizes[1], sizes[2], layer_keys[1])
    return [(hidden_weight, hidden_bias), output_weight]
