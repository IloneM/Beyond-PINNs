"""Deterministic checks for beyond_pinns.models -- no GNDRM required."""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
from jax import random, grad, vmap

from beyond_pinns.initialization import init_shallow_mlp_params
from beyond_pinns.models import shallow_mlp, deep_mlp

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "gndrm_golden.json").read_text())
NGRAD_FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "ngrad_golden.json").read_text())


def test_output_is_scalar():
    model = shallow_mlp(jnp.tanh)
    params = init_shallow_mlp_params([1, 32, 1], random.PRNGKey(0))
    out = model(params, jnp.array([0.5]))
    assert out.shape == ()


def test_no_output_bias_by_construction():
    """params only ever supplies [(W_hidden, b_hidden), W_out] -- there is no
    slot for an output bias, so a zero-weight output layer must give exactly
    zero regardless of the hidden activation."""
    model = shallow_mlp(jnp.tanh)
    (w1, b1), _ = init_shallow_mlp_params([1, 32, 1], random.PRNGKey(0))
    zero_w2 = jnp.zeros((1, 32))
    out = model(((w1, b1), zero_w2), jnp.array([0.37]))
    assert float(out) == 0.0


def test_batched_matches_scalar_calls():
    model = shallow_mlp(jnp.tanh)
    params = init_shallow_mlp_params([1, 32, 1], random.PRNGKey(2))
    xs = jnp.linspace(-1, 1, 11).reshape(-1, 1)
    batched = vmap(model, (None, 0))(params, xs)
    individual = jnp.array([model(params, x) for x in xs])
    assert jnp.allclose(batched, individual)


def test_jit_vmap_grad_jacfwd_all_work():
    model = shallow_mlp(jnp.tanh)
    params = init_shallow_mlp_params([1, 16, 1], random.PRNGKey(5))
    x = jnp.array([0.2])
    jax.jit(model)(params, x)
    grad(lambda xx: model(params, xx))(x)
    jax.jacfwd(lambda p: model(p, x))(params)


def test_matches_golden_gndrm_fixture():
    g = FIXTURE["model_tanh_seed0_sizes_1_32_1"]
    params = init_shallow_mlp_params([1, 32, 1], random.PRNGKey(0))
    model = shallow_mlp(jnp.tanh)
    for x, expected in zip(g["xs"], g["values"]):
        got = float(model(params, jnp.array([x])))
        assert abs(got - expected) < 1e-12, (x, got, expected)


# ---------------- deep_mlp (Part B's general N-hidden-layer model) ----------------

def _params_from_fixture(fixture):
    return [(jnp.asarray(w), jnp.asarray(b)) for w, b in fixture["params"]]


def test_deep_mlp_output_is_scalar():
    model = deep_mlp(jnp.tanh)
    params = _params_from_fixture(NGRAD_FIXTURE)
    out = model(params, jnp.array([0.5]))
    assert out.shape == ()


def test_deep_mlp_has_output_bias_unlike_shallow_mlp():
    """Unlike shallow_mlp (no output bias, matching GNDRM), deep_mlp's final
    layer DOES have a bias, matching ngrad.models.mlp exactly -- a zero-weight
    output layer with nonzero bias must return exactly that bias."""
    model = deep_mlp(jnp.tanh)
    params = _params_from_fixture(NGRAD_FIXTURE)
    (w_last, b_last) = params[-1]
    params_zeroed = params[:-1] + [(jnp.zeros_like(w_last), b_last)]
    out = model(params_zeroed, jnp.array([0.37]))
    assert jnp.allclose(out, b_last[0])


def test_deep_mlp_batched_matches_scalar_calls():
    model = deep_mlp(jnp.tanh)
    params = _params_from_fixture(NGRAD_FIXTURE)
    xs = jnp.linspace(-1, 1, 11).reshape(-1, 1)
    batched = vmap(model, (None, 0))(params, xs)
    individual = jnp.array([model(params, x) for x in xs])
    assert jnp.allclose(batched, individual)


def test_deep_mlp_jit_grad_jacfwd_all_work():
    model = deep_mlp(jnp.tanh)
    params = _params_from_fixture(NGRAD_FIXTURE)
    x = jnp.array([0.2])
    jax.jit(model)(params, x)
    grad(lambda xx: model(params, xx))(x)
    jax.jacfwd(lambda p: model(p, x))(params)


def test_deep_mlp_matches_golden_ngrad_fixture():
    """Golden regression check against a real ngrad.models.mlp run captured
    earlier -- does not require a live ngrad checkout (see
    test_ngrad_equivalence.py for the live comparison)."""
    params = _params_from_fixture(NGRAD_FIXTURE)
    model = deep_mlp(jnp.tanh)
    for x, expected in zip(NGRAD_FIXTURE["xs"], NGRAD_FIXTURE["values"]):
        got = float(model(params, jnp.array([x])))
        assert abs(got - expected) < 1e-12, (x, got, expected)
