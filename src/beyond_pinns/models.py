"""Neural-network forward models: `shallow_mlp` (one-hidden-layer, used by
Part A's own methods -- plain_gn_*, amstramgram_*, dsgnar_*) and
`deep_mlp` (general N-hidden-layer, used by Part B -- MS/JF/LS/RC).

Provenance
----------
The historical Part A script sourced this from the external GaussNewtonDRM
package (github.com/Jinxl-pp/GaussNewtonDRM, commit 9dd1ee8, no LICENSE --
see docs/PROVENANCE.md). The architecture itself is a completely standard,
textbook one-hidden-layer MLP with no output bias -- there is nothing
Gauss-Newton- or Hao/Jin-specific about the forward pass itself (their actual
contribution is the optimizer, in `jacobian_matrix`/`gn_direction`/
`grid_line_search`, which stay external and optional). This module is
written from the mathematical specification, independently -- no GNDRM
source text, comments, or internal helper names are reused.

Exact historical architecture (preserved deliberately, not "improved"):

    x -> affine hidden layer (weight W_hidden, bias b_hidden) -> activation
      -> affine output layer (weight W_out, NO bias) -> scalar

Characterization required for numerical equivalence (see module docstring
in beyond_pinns/initialization.py for the matching parameter pytree, and
tests/test_models.py / tests/test_gndrm_equivalence.py for the checks):

  * parameter pytree: `[(W_hidden, b_hidden), W_out]` -- a 2-element list,
    first entry a (weight, bias) tuple, second entry a bare weight array;
  * W_hidden shape (hidden_width, in_dim), b_hidden shape (hidden_width,);
  * W_out shape (out_dim, hidden_width), NO bias;
  * input `x` is a single point, shape (in_dim,) -- NOT batched; batching is
    done by the caller via `jax.vmap(model, (None, 0))`, exactly as in the
    historical script;
  * activation is applied elementwise to the hidden pre-activation only;
  * output is reshaped to a 0-d scalar array (`jnp.reshape(..., ())`), not a
    length-1 vector -- callers rely on this for `jax.grad`/`jacfwd` to return
    naturally-shaped derivatives;
  * dtype follows the input/parameter dtype (float64 throughout, since the
    caller sets `jax.config.update("jax_enable_x64", True)` globally);
  * fully compatible with `jax.jit`, `jax.vmap`, `jax.grad`, `jax.jacfwd`,
    `jax.jacrev` on both the input `x` and the parameter pytree, since it is
    built entirely from `jnp.dot`/`jnp.reshape` and a user-supplied
    elementwise activation.
"""
from __future__ import annotations

import jax.numpy as jnp


def shallow_mlp(activation):
    """Return a `model(params, x) -> scalar` closure for a one-hidden-layer
    network with the given elementwise activation and no output bias.

    `params` must be the `[(W_hidden, b_hidden), W_out]` pytree produced by
    `beyond_pinns.initialization.init_shallow_mlp_params`. `x` is a single
    input point of shape (in_dim,).
    """
    def model(params, x):
        (hidden_weight, hidden_bias), output_weight = params
        pre_activation = jnp.dot(hidden_weight, x) + hidden_bias
        hidden = activation(pre_activation)
        output = jnp.dot(output_weight, hidden)
        return jnp.reshape(output, ())

    return model


def deep_mlp(activation):
    """General N-hidden-layer MLP forward pass (N >= 1), WITH an output bias
    (unlike `shallow_mlp` above) -- used by Part B (MS/JF/LS/RC), whose own
    architectures are 2-hidden-layer (e.g. layer_sizes [n_in, 64, 64, 1]).

    Provenance: the historical Part B script (`femennstein-petrov-galerkin-
    experiments-fixed.py`) imports this forward pass from the external
    `ngrad` package (`from ngrad.models import mlp`,
    github.com/IloneM/Natural-Gradient-PINNs-ICML23, commit 0a0055c -- see
    docs/PROVENANCE.md). The architecture itself is a completely standard,
    textbook N-layer MLP with no algorithmic content specific to natural
    gradients or that repository -- `ngrad`'s own source file header
    explicitly comments its neighboring init helpers as "Code from Jax
    documentation", underscoring the genericness. This module is written
    from the mathematical specification, independently -- no `ngrad` source
    text, comments, or internal helper/variable names are reused.

    Exact historical architecture (preserved deliberately, not "improved"):

        x -> N affine hidden layers, each followed by `activation`
          -> final affine output layer, WITH bias -> scalar

    Characterization required for numerical equivalence (see
    tests/test_models.py / tests/test_ngrad_equivalence.py):

      * parameter pytree: a flat list of `(W_i, b_i)` tuples, one per layer
        (hidden layers AND the output layer -- every layer has a bias,
        unlike `shallow_mlp`'s bias-free output layer);
      * `W_i` shape (out_features_i, in_features_i), `b_i` shape (out_features_i,);
      * input `x` is a single point, shape (in_dim,) -- not batched; batching
        is the caller's responsibility via `jax.vmap`, exactly as historically;
      * activation applied to every layer's pre-activation EXCEPT the last
        (output) layer, which stays purely affine;
      * output reshaped to a 0-d scalar array (`jnp.reshape(..., ())`);
      * dtype follows the input/parameter dtype (float64 throughout, given
        `jax.config.update("jax_enable_x64", True)`);
      * fully compatible with `jax.jit`/`jax.vmap`/`jax.grad`/`jax.jacfwd`/
        `jax.jacrev` on both `x` and the parameter pytree.
    """
    def model(params, x):
        hidden = x
        for weight, bias in params[:-1]:
            pre_activation = jnp.dot(weight, hidden) + bias
            hidden = activation(pre_activation)
        output_weight, output_bias = params[-1]
        output = jnp.dot(output_weight, hidden) + output_bias
        return jnp.reshape(output, ())

    return model
