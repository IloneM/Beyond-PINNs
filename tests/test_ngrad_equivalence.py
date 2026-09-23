"""OPTIONAL compatibility test: compares beyond_pinns.models.deep_mlp against
a LIVE Natural-Gradient-PINNs-ICML23 (`ngrad`) checkout, when one happens to
be discoverable on sys.path (as an `ngrad` package importable directly, or a
`Natural-Gradient-PINNs-ICML23`/`ngrad-src` sibling directory containing
`ngrad/models.py`). Entirely skipped otherwise -- this file, and therefore
CI, never requires `ngrad`.

To run these locally:
    git clone https://github.com/IloneM/Natural-Gradient-PINNs-ICML23 ngrad-src
    cd ngrad-src && git checkout 0a0055cb648bc41b9773e6855f6df8c8748cebc5 && cd -
    pytest tests/test_ngrad_equivalence.py -v
"""
import sys
from pathlib import Path

import jax
import jax.flatten_util
import jax.numpy as jnp
import pytest
from jax import random, grad, jacfwd, vmap

from beyond_pinns.models import deep_mlp


def _find_ngrad():
    for candidate in ("ngrad-src", "Natural-Gradient-PINNs-ICML23", "../ngrad-src",
                       "../Natural-Gradient-PINNs-ICML23"):
        p = (Path.cwd() / candidate).resolve()
        if (p / "ngrad" / "models.py").exists():
            return p
    return None


_NGRAD_ROOT = _find_ngrad()
if _NGRAD_ROOT is not None and str(_NGRAD_ROOT) not in sys.path:
    sys.path.insert(0, str(_NGRAD_ROOT))

pytestmark = pytest.mark.skipif(
    _NGRAD_ROOT is None,
    reason="No local ngrad checkout found (optional -- see module docstring for how to add one)",
)


def _glorot_like_init(sizes, key):
    params = []
    for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
        key, wk, bk = random.split(key, 3)
        w = random.normal(wk, (fan_out, fan_in)) * jnp.sqrt(2.0 / (fan_in + fan_out))
        b = random.normal(bk, (fan_out,)) * 0.1
        params.append((w, b))
    return params


def test_deep_mlp_matches_ngrad_value_derivatives_and_jacobian():
    from ngrad.models import mlp as ngrad_mlp

    for sizes in ([1, 32, 32, 1], [4, 64, 64, 1], [2, 16, 1]):
        for act_name, act in (("tanh", jnp.tanh), ("relu3", lambda z: jnp.maximum(z, 0.0) ** 3)):
            params = _glorot_like_init(sizes, random.PRNGKey(hash((tuple(sizes), act_name)) % (2**31)))
            model_g = ngrad_mlp(act)
            model_o = deep_mlp(act)
            for trial in range(3):
                x = random.normal(random.PRNGKey(trial), (sizes[0],))
                assert jnp.allclose(model_g(params, x), model_o(params, x), atol=1e-13)
                dg = grad(lambda xx: model_g(params, xx))(x)
                do = grad(lambda xx: model_o(params, xx))(x)
                assert jnp.allclose(dg, do, atol=1e-13)
                jg = jax.flatten_util.ravel_pytree(jacfwd(lambda p: model_g(p, x))(params))[0]
                jo = jax.flatten_util.ravel_pytree(jacfwd(lambda p: model_o(p, x))(params))[0]
                assert jnp.allclose(jg, jo, atol=1e-13)
            xb = random.normal(random.PRNGKey(99), (7, sizes[0]))
            assert jnp.allclose(vmap(model_g, (None, 0))(params, xb),
                                 vmap(model_o, (None, 0))(params, xb), atol=1e-13)
