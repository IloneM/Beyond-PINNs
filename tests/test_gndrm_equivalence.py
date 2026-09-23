"""OPTIONAL compatibility test: compares beyond_pinns.{initialization,models,
quadrature} against a LIVE GaussNewtonDRM checkout, when one happens to be
discoverable on sys.path (as a `GNDRM`/`GaussNewtonDRM` sibling directory, or
already importable). Entirely skipped otherwise -- this file, and therefore
CI, never requires GNDRM.

To run these locally:
    git clone https://github.com/Jinxl-pp/GaussNewtonDRM ../GNDRM
    cd ../GNDRM && git checkout 9dd1ee8df8aca41c5f85157c2d7bf2afa12035ce && cd -
    pytest tests/test_gndrm_equivalence.py -v
"""
import sys
from pathlib import Path

import jax
import jax.flatten_util
import jax.numpy as jnp
import pytest
from jax import random, grad, jacfwd, vmap

from beyond_pinns.initialization import init_shallow_mlp_params
from beyond_pinns.models import shallow_mlp
from beyond_pinns.quadrature import PiecewiseGaussLegendre1D


def _find_gndrm():
    for candidate in ("GNDRM", "GaussNewtonDRM", "../GNDRM", "../GaussNewtonDRM"):
        p = (Path.cwd() / candidate).resolve()
        if (p / "tool" / "model.py").exists():
            return p
    return None


_GNDRM_ROOT = _find_gndrm()
if _GNDRM_ROOT is not None and str(_GNDRM_ROOT) not in sys.path:
    sys.path.insert(0, str(_GNDRM_ROOT))

pytestmark = pytest.mark.skipif(
    _GNDRM_ROOT is None,
    reason="No local GNDRM checkout found (optional -- see module docstring for how to add one)",
)


def test_initializer_bit_identical_to_gndrm():
    from tool.model import normal_init as gndrm_normal_init
    for seed in (0, 1, 2, 7, 42):
        for sizes in ([1, 32, 1], [1, 64, 1], [1, 16, 1]):
            key = random.PRNGKey(seed)
            (w1g, b1g), w2g = gndrm_normal_init(sizes, key)
            (w1o, b1o), w2o = init_shallow_mlp_params(sizes, key)
            assert jnp.array_equal(w1g, w1o)
            assert jnp.array_equal(b1g, b1o)
            assert jnp.array_equal(w2g, w2o)


def test_model_value_and_derivatives_match_gndrm():
    from tool.model import normal_init as gndrm_normal_init, shallow_network as gndrm_shallow_network
    for act_name, act in (("tanh", jnp.tanh), ("relu3", lambda z: jnp.maximum(z, 0.0) ** 3)):
        model_g = gndrm_shallow_network(act)
        model_o = shallow_mlp(act)
        for seed in (0, 3, 11):
            params = gndrm_normal_init([1, 32, 1], random.PRNGKey(seed))
            for x0 in (-0.7, 0.0, 0.3, 0.95):
                x = jnp.array([x0])
                assert jnp.allclose(model_g(params, x), model_o(params, x), atol=1e-13)
                dg = grad(lambda xx: model_g(params, xx))(x)
                do = grad(lambda xx: model_o(params, xx))(x)
                assert jnp.allclose(dg, do, atol=1e-13)
                jg = jax.flatten_util.ravel_pytree(jacfwd(lambda p: model_g(p, x))(params))[0]
                jo = jax.flatten_util.ravel_pytree(jacfwd(lambda p: model_o(p, x))(params))[0]
                assert jnp.allclose(jg, jo, atol=1e-13)
            xb = jnp.linspace(-1, 1, 50).reshape(-1, 1)
            assert jnp.allclose(vmap(model_g, (None, 0))(params, xb),
                                 vmap(model_o, (None, 0))(params, xb), atol=1e-13)


def test_quadrature_nodes_weights_and_integrals_match_gndrm():
    from tool.quadrature import GaussLegendrePiecewise as GNDRM_Quad
    for npts in (2, 4, 8):
        qg = GNDRM_Quad(npts=npts)
        qo = PiecewiseGaussLegendre1D(npts=npts)
        assert jnp.allclose(jnp.sort(qg.quadpts.flatten()), jnp.sort(qo.quadpts.flatten()), atol=1e-14)
        assert jnp.allclose(jnp.sort(qg.weights.flatten()), jnp.sort(qo.weights.flatten()), atol=1e-14)
        for a, b, h in ((-1.0, 1.0, 1.0 / 8), (0.0, 1.0, 1.0 / 16)):
            interval, hh = jnp.array([[a, b]]), jnp.array([h])
            ig = qg.interval_quadpts(interval, hh)
            io = qo.interval_quadpts(interval, hh)
            for f in (
                lambda x: jnp.ones((x.shape[0],)),
                lambda x: x.flatten(),
                lambda x: x.flatten() ** 3 - 2 * x.flatten(),
                lambda x: jnp.cos(3.0 * x.flatten()),
            ):
                assert abs(float(ig(f)) - float(io(f))) < 1e-10
