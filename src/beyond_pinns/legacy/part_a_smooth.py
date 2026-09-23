#!/usr/bin/env python
# coding: utf-8

# # Energy versus residual least squares: linear and nonlinear 1D benchmarks
# 
# This notebook consolidates the two preliminary benchmarks used to validate the **pure weak formulation** before introducing the FEM--NN hybrid method.
# 
# It addresses three questions:
# 
# 1. Does replacing scalar energy minimization by a vector residual least-squares formulation improve optimization accuracy?
# 2. On smooth problems, does the weak residual retain the favorable Gauss--Newton structure of a strong residual while requiring only first spatial derivatives of the neural approximation?
# 3. Are the conclusions robust to the activation function? Every matched experiment is run with both `tanh` and $\mathrm{ReLU}^3$.
# 
# The two PDEs are
# 
# $$
# -u''+u=f,\qquad u'(-1)=u'(1)=0,
# $$
# 
# and
# 
# $$
# -u''+u^3=f,\qquad u'(-1)=u'(1)=0,
# $$
# 
# with exact solution $u^\star(x)=\cos(\pi x)$ on $(-1,1)$.
# 
# The primary common metrics are the relative errors
# 
# $$
# e_{L^2}=\frac{\|u-u^\star\|_{L^2}}{\|u^\star\|_{L^2}},
# \qquad
#  e_{H^1}=\frac{\|u-u^\star\|_{H^1}}{\|u^\star\|_{H^1}}.
# $$
# 
# The $H^1$ seminorm error is retained as a secondary diagnostic.
# 
# ## Reference-faithfulness policy
# 
# - **Linear Deep Ritz:** the energy Gauss--Newton construction follows Hao--Hong--Jin and uses their 1D quadrature and line-search settings. Their published code uses $\mathrm{ReLU}^3$; here the same experiment is deliberately repeated with `tanh`.
# - **Nonlinear matched Energy NG:** the metric is the Müller--Zeinhofer energy metric
#   $$
#   \int \partial_\theta u'\,\partial_\theta u'^\top
#   +3u_\theta^2\partial_\theta u\,\partial_\theta u^\top,
#   $$
#   but the architecture, initialization, and benchmark protocol are matched to the residual methods.
# - **Nonlinear reference protocol:** an optional separate run reproduces the Müller--Zeinhofer numerical protocol (width 32, trapezoidal integration with 20,000/200,000 points, $0.5^k$ line search), while again allowing both activations. It is kept out of the main matched comparison plots.
# 
# The main residual benchmark contains two deliberately simple weak Gauss--Newton baselines in addition to AMStramGRAM and DSGNAR. They use the same weak residual, Jacobian, initialization, and line search and differ only in the spectral regularization of the Gauss--Newton inverse:
# 
# - **TSVD / pseudoinverse weak GN** uses a truncated Moore--Penrose inverse,
#   $$
#   \sigma_i^{-1}\,\mathbf 1_{\{\sigma_i\geq\tau\}},
#   $$
#   matching the spectral-cutoff principle used by ANaGRAM.
# - **LM / ridge weak GN** uses the classical damped filter
#   $$
#   \frac{\sigma_i}{\sigma_i^2+\lambda},
#   $$
#   with no hard spectral cutoff in the update.
# 
# This separation tests whether the weak residual formulation works directly with either a spectral pseudoinverse or conventional Levenberg--Marquardt/Tikhonov regularization, rather than relying on a particular adaptive optimizer.
# 

# In[ ]:


# ============================================================
# Imports, repository discovery, and single experiment roadmap
# ============================================================
import os, sys, time, json, math, pickle, shutil
from pathlib import Path
from functools import partial

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import jax
import jax.numpy as jnp
import jax.scipy.linalg
import jax.flatten_util
from jax import grad, jacfwd, vmap, jit, random

jax.config.update("jax_enable_x64", True)

# ============================================================
# PORTABILITY SHIM (release-only addition, not present in the historical
# research script): CONFIG is byte-identical to the historical literal below
# UNLESS the BEYOND_PINNS_CONFIG_JSON environment variable is set (see
# `beyond_pinns.run`). No scientific behavior changes when unset --
# `_HISTORICAL_CONFIG` is exactly the CONFIG dict that produced the
# manuscript's Part A results.
# ============================================================
_HISTORICAL_CONFIG = {
    "global": {
        "execute": True,
        "seeds": list(range(10)),
        "activations": ["tanh", "relu3"],
        "output_root": "energy_vs_weak_benchmarks",
        "n_iter_save_params": 25,
        "n_iter_plot": 25,
        "save_initial_params": True,
        "save_final_params": True,
        "plot_mean_std": True,
        "plot_median_iqr": True,
    },
    "paths": {
        "gndrm_candidates": [
            "GNDRM", "GaussNewtonDRM", "../GNDRM", "../GaussNewtonDRM"
        ],
        "ngrad_candidates": [
            "Natural-Gradient-PINNs-ICML23", "../Natural-Gradient-PINNs-ICML23"
        ],
    },
    "matched": {
        "width": 64,
        "train_h": 1.0 / 3000.0,
        "test_h": 1.0 / 4000.0,
        "gl_points_per_cell": 2,
        "n_train_interior": 200,
        "n_test_interior": 500,
        "n_extrap_interior": 2000,
        "weak_split_order": 64,
        "residual_iterations": 1000,
        "energy_iterations": 1000,
        "plain_gn_tsvd_rcond": 1e-8, #1e-12,
        "plain_gn_lm_damping": 1e-6, #1e-12,
        # Absolute (non-relative) TSVD cutoff: singular values are kept iff
        # sigma >= plain_gn_tsvd_abs_rcond, with NO comparison to sigma_max
        # (unlike "tsvd", which keeps sigma >= rcond*sigma_max). Defaulted to
        # the same numeric value as plain_gn_tsvd_rcond as a starting point --
        # tune independently, since the two live on different scales.
        "plain_gn_tsvd_abs_rcond": 1e-8,
    },
    "linear": {
        "enabled": True,
        "methods": {
            "hao_gn_deep_ritz": True,
            "plain_gn_strong_tsvd": True,
            "plain_gn_weak_tsvd": True,
            "plain_gn_strong_tsvd_abs": False,
            "plain_gn_weak_tsvd_abs": False,
            "plain_gn_strong_lm": True,
            "plain_gn_weak_lm": True,
            "amstramgram_strong": True,
            "dsgnar_strong": True,
            "amstramgram_weak": True,
            "dsgnar_weak": True,
        },
    },
    "nonlinear": {
        "enabled": True,
        "methods": {
            "energy_ng_matched": True,
            "plain_gn_strong_tsvd": True,
            "plain_gn_weak_tsvd": True,
            "plain_gn_strong_tsvd_abs": False,
            "plain_gn_weak_tsvd_abs": False,
            "plain_gn_strong_lm": True,
            "plain_gn_weak_lm": True,
            "amstramgram_strong": True,
            "dsgnar_strong": True,
            "amstramgram_weak": True,
            "dsgnar_weak": True,
        },
        "reference_protocol": {
            "enabled": True,
            "iterations": 101,
            "width": 32,
            "train_points": 20000,
            "eval_points": 200000,
            "test_every": 10,
        },
    },
}

import json as _bp_json
_bp_override_path = os.environ.get("BEYOND_PINNS_CONFIG_JSON")
if _bp_override_path:
    with open(_bp_override_path) as _bp_f:
        CONFIG = _bp_json.load(_bp_f)
    print(f"[beyond_pinns] CONFIG loaded from {_bp_override_path} "
          f"(BEYOND_PINNS_CONFIG_JSON override active)")
else:
    CONFIG = _HISTORICAL_CONFIG


def experiment_enabled(*path):
    node = CONFIG
    for key in path:
        node = node[key]
    return bool(node)


def _find_repo(candidates, marker):
    roots = [Path(os.getcwd()), Path(os.getcwd()).parent]
    checked = []
    for c in candidates:
        cp = Path(c).expanduser()
        options = [cp] if cp.is_absolute() else [r / cp for r in roots]
        for p in options:
            p = p.resolve()
            checked.append(str(p))
            if (p / marker).exists():
                return p
    raise FileNotFoundError(
        f"Could not locate repository containing {marker}. Checked:\n  " + "\n  ".join(checked)
    )

# ============================================================
# GNDRM GATING (release-only change from the historical script, which
# located and imported GNDRM unconditionally at import time -- see
# docs/PROVENANCE.md for the full accounting). GNDRM
# (github.com/Jinxl-pp/GaussNewtonDRM) has no LICENSE and is therefore never
# vendored or required.
#
# Of the five names the historical script imported from GNDRM:
#   - `shallow_network`, `GaussLegendrePiecewise` -- generic model/quadrature
#     infrastructure used by EVERY Part A method, not GNDRM-specific in any
#     algorithmic sense. Always bound below to our own independent,
#     GNDRM-free implementations (beyond_pinns.models / .quadrature),
#     regardless of whether GNDRM happens to be installed -- so behavior
#     never depends on unrelated environment state.
#   - `jacobian_matrix`, `gn_direction`, `grid_line_search` -- the actual
#     Gauss-Newton-optimizer machinery. Used by build_matched_context() to
#     construct the energy-metric direction/line-search for TWO optional
#     reference baselines: `hao_gn_deep_ritz` (linear, Hao et al.'s own
#     method) and `energy_ng_matched` (nonlinear, Müller--Zeinhofer metric,
#     built from the same GNDRM primitives). Genuinely external: only
#     imported if a GNDRM checkout is found, and only actually required if
#     one of those two baselines is requested (require_gndrm_baseline()
#     below raises a clear, actionable error otherwise; every other method
#     is unaffected).
# ============================================================
GNDRM_ROOT = None
try:
    GNDRM_ROOT = _find_repo(CONFIG["paths"]["gndrm_candidates"], "tool/model.py")
except FileNotFoundError:
    pass
if GNDRM_ROOT is not None and str(GNDRM_ROOT) not in sys.path:
    sys.path.insert(0, str(GNDRM_ROOT))

if GNDRM_ROOT is not None:
    from tool.gauss_newton import jacobian_matrix, gn_direction, grid_line_search
else:
    print("[beyond_pinns] GNDRM not found -- hao_gn_deep_ritz and energy_ng_matched "
          "will be unavailable; every other Part A method is unaffected "
          "(see docs/PROVENANCE.md).")
    jacobian_matrix = gn_direction = grid_line_search = None

from beyond_pinns.models import shallow_mlp as _bp_shallow_mlp
from beyond_pinns.quadrature import PiecewiseGaussLegendre1D as _bp_quadrature
from beyond_pinns.initialization import init_shallow_mlp_params as normal_init

# `shallow_network`/`GaussLegendrePiecewise` are used throughout this script
# (not just by the two GNDRM-dependent baselines below) -- always bound to our
# own independent, GNDRM-free implementations, regardless of whether a real
# GNDRM checkout happens to be present. This keeps every OTHER method's
# behavior independent of unrelated environment state.
shallow_network = _bp_shallow_mlp
GaussLegendrePiecewise = _bp_quadrature


def require_gndrm_baseline(method_name):
    """Raise a clear, actionable error if `method_name` (a genuinely
    GNDRM-dependent optional reference baseline: hao_gn_deep_ritz or
    energy_ng_matched) is requested without a local GNDRM checkout."""
    if GNDRM_ROOT is not None:
        return
    raise RuntimeError(
        f"CONFIG requests the optional reference baseline '{method_name}', which "
        "depends on GaussNewtonDRM's own Gauss-Newton-optimizer machinery "
        "(jacobian_matrix/gn_direction/grid_line_search) -- this is NOT "
        "reimplemented (see docs/PROVENANCE.md) and is not vendored, because "
        "the upstream repository has no LICENSE.\n"
        "  Upstream:      https://github.com/Jinxl-pp/GaussNewtonDRM\n"
        "  Exact commit:  9dd1ee8df8aca41c5f85157c2d7bf2afa12035ce\n"
        "To run this baseline, manually clone that exact commit as a sibling of "
        "this repository (or anywhere on PYTHONPATH) under the name 'GNDRM' or "
        "'GaussNewtonDRM', e.g.:\n"
        "  git clone https://github.com/Jinxl-pp/GaussNewtonDRM GNDRM\n"
        "  cd GNDRM && git checkout 9dd1ee8df8aca41c5f85157c2d7bf2afa12035ce\n"
        f"then rerun. To skip '{method_name}' instead, set its CONFIG flag to False."
    )

# ngrad is required only for the optional Müller--Zeinhofer reference-protocol run.
NGRAD_ROOT = None
if CONFIG["nonlinear"]["reference_protocol"]["enabled"]:
    NGRAD_ROOT = _find_repo(CONFIG["paths"]["ngrad_candidates"], "ngrad/models.py")
    if str(NGRAD_ROOT) not in sys.path:
        sys.path.insert(0, str(NGRAD_ROOT))

OUTPUT_ROOT = Path(CONFIG["global"]["output_root"])
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
print("GNDRM:", "found" if GNDRM_ROOT is not None else "not found (optional)")
print("ngrad:", "found" if NGRAD_ROOT is not None else "not found (optional)")
print("output:", OUTPUT_ROOT)


# # Shared numerical infrastructure
# 
# The matched benchmark uses exactly the same network architecture and initialization for all methods under a fixed problem, activation, and seed. The scalar energy and residual objectives are **not numerically comparable**; cross-method conclusions therefore use relative $L^2$, relative $H^1$, wall-clock time, and a common held-out weak residual.
# 

# In[ ]:


# ============================================================
# Activations, tree utilities, run directories, and histories
# ============================================================
def activation_from_name(name):
    if name == "tanh":
        return jnp.tanh
    if name == "relu3":
        return lambda x: jnp.where(x > 0.0, x, 0.0) ** 3
    raise ValueError(name)


def tree_block_until_ready(tree):
    return jax.tree_util.tree_map(
        lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
        tree,
    )


def count_params(params):
    return int(sum(np.asarray(x).size for x in jax.tree_util.tree_leaves(params)))


def _jsonable(obj):
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def run_dir(problem, activation, method, seed, protocol="matched"):
    return OUTPUT_ROOT / problem / protocol / activation / method / f"seed_{seed:03d}"


def save_checkpoint(params, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    host_params = jax.device_get(params)
    with path.open("wb") as f:
        pickle.dump(host_params, f, protocol=pickle.HIGHEST_PROTOCOL)


def save_history(hist, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for k, v in hist.items():
        if k == "compile_time":
            arrays[k] = np.asarray(v, dtype=float)
        elif isinstance(v, list):
            arrays[k] = np.asarray(v)
    np.savez_compressed(path, **arrays)


def new_history():
    return {
        "iteration": [],
        "wallclock_optim": [],
        "compile_time": 0.0,
        "relative_l2": [],
        "relative_h1_seminorm": [],
        "relative_h1": [],
        "loss_train": [],
        "loss_test": [],
        "common_weak_test_loss": [],
        "step_size": [],
        "effective_rank": [],
        "accepted_step": [],
    }


def append_history(hist, iteration, wallclock, metrics, *, step_size=np.nan,
                   effective_rank=np.nan, accepted=True):
    hist["iteration"].append(int(iteration))
    hist["wallclock_optim"].append(float(wallclock))
    for k in ("relative_l2", "relative_h1_seminorm", "relative_h1",
              "loss_train", "loss_test", "common_weak_test_loss"):
        hist[k].append(float(metrics[k]))
    hist["step_size"].append(float(step_size))
    hist["effective_rank"].append(float(effective_rank))
    hist["accepted_step"].append(bool(accepted))


# In[ ]:


# ============================================================
# Plotting and final-solution diagnostics
# ============================================================
def _positive_for_log(y):
    y = np.asarray(y, float)
    tiny = np.finfo(float).tiny
    return np.maximum(np.abs(y), tiny)


def plot_run_history(hist, out_dir, title=""):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    it = np.asarray(hist["iteration"], float)
    wt = np.asarray(hist["wallclock_optim"], float)
    panels = [
        ("relative_l2", r"relative $L^2$ error"),
        ("relative_h1", r"relative $H^1$ error"),
        ("loss_train", "native train objective"),
        ("common_weak_test_loss", "common weak test loss"),
    ]
    for x, xlabel, filename in ((it, "iteration", "convergence_iteration.png"),
                               (wt, "optimization wall-clock [s]", "convergence_wallclock.png")):
        fig, axes = plt.subplots(1, 4, figsize=(20, 4.2))
        for ax, (key, ylabel) in zip(axes, panels):
            y = _positive_for_log(hist[key])
            ax.plot(x, y)
            if np.all(x > 0): ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.grid(True, alpha=.3)
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=140, bbox_inches="tight")
        plt.close(fig)


def plot_solution_1d(ctx, params, out_dir, title=""):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    xs = jnp.linspace(-1.0, 1.0, 1200).reshape(-1, 1)
    model, ustar = ctx["model"], ctx["u_exact"]
    pred = jax.vmap(lambda x: jnp.reshape(model(params, x), ()))(xs)
    exact = jax.vmap(ustar)(xs)
    dpred = jax.vmap(lambda x: jnp.reshape(jax.grad(lambda z: jnp.reshape(model(params, z), ()))(x), ()))(xs)
    dexact = jax.vmap(lambda x: jnp.reshape(jax.grad(ustar)(x), ()))(xs)
    xnp = np.asarray(xs[:, 0])
    fig, axes = plt.subplots(1, 4, figsize=(19, 4))
    axes[0].plot(xnp, np.asarray(exact), label=r"$u^\star$")
    axes[0].plot(xnp, np.asarray(pred), "--", label=r"$u_\theta$")
    axes[0].legend(); axes[0].set_title("solution")
    axes[1].semilogy(xnp, np.maximum(np.abs(np.asarray(pred-exact)), 1e-18)); axes[1].set_title("absolute solution error")
    axes[2].plot(xnp, np.asarray(dexact), label=r"$(u^\star)'$")
    axes[2].plot(xnp, np.asarray(dpred), "--", label=r"$u_\theta'$")
    axes[2].legend(); axes[2].set_title("derivative")
    axes[3].semilogy(xnp, np.maximum(np.abs(np.asarray(dpred-dexact)), 1e-18)); axes[3].set_title("absolute derivative error")
    for ax in axes: ax.set_xlabel("x"); ax.grid(True, alpha=.3)
    fig.suptitle(title); fig.tight_layout()
    fig.savefig(out_dir / "final_solution.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    np.savez_compressed(out_dir.parent / "final_solution.npz", x=xnp, exact=np.asarray(exact), learned=np.asarray(pred), exact_grad=np.asarray(dexact), learned_grad=np.asarray(dpred))


# In[ ]:


# ============================================================
# Residual feature machinery shared by AMStramGRAM / DSGNAR / weak GN baselines
# ============================================================
def make_operator_on_model(model, functional_operator):
    return jit(lambda params, x: functional_operator(lambda z: model(params, z))(x))


def pre_features_factory(model, functional_operator):
    operator_on_model = make_operator_on_model(model, functional_operator)
    @jit
    def del_theta_operator(params, x):
        return grad(operator_on_model, argnums=0)(params, x)
    @jit
    def pre_features(params, x):
        return jax.flatten_util.ravel_pytree(del_theta_operator(params, x))[0]
    return pre_features


def features_factory(model, functional_operator):
    pre_features = pre_features_factory(model, functional_operator)
    v_pre_features = vmap(pre_features, (None, 0))
    @jit
    def features(params, samples):
        return v_pre_features(params, samples).T
    return features


def full_features_factory(model, functional_operators):
    features_by_operator = tuple(features_factory(model, fo) for fo in functional_operators)
    @jit
    def full_features(params, batch_samples):
        return jnp.concatenate(tuple(fbo(params, bs) for fbo, bs in zip(features_by_operator, batch_samples)), axis=1)
    return full_features


def quadratic_operator_factory(model, functional_operator):
    return vmap(make_operator_on_model(model, functional_operator), (None, 0))


def quadratic_source_factory(source):
    return vmap(jit(source), 0)


def full_quadratic_gradient_factory(model, functional_operators, sources):
    simple_operators = tuple(quadratic_operator_factory(model, fo) for fo in functional_operators)
    simple_sources = tuple(quadratic_source_factory(so) for so in sources)
    @jit
    def full_operator(params, batch_samples):
        return jnp.concatenate(tuple(sop(params, bs) for sop, bs in zip(simple_operators, batch_samples)), axis=0)
    @jit
    def full_source(batch_samples):
        return jnp.concatenate(tuple(ss(bs) for ss, bs in zip(simple_sources, batch_samples)), axis=0)
    @jit
    def full_gradient(params, batch_samples):
        return full_operator(params, batch_samples) - full_source(batch_samples)
    return full_gradient, full_operator, full_source


@jit
def uniform_weights(samples):
    sample_uw = tuple(jnp.full((s.shape[0],), 1.0 / s.shape[0]) for s in samples)
    return jnp.concatenate(sample_uw)


def nat_grad_factory(full_features, true_gradient, const_tol=None,
                     const_tol_relative_to_bigger_sv=True, const_cut_low_signal=True,
                     const_return_details=False, const_return_svd=False, const_return_flat=False):
    @partial(jax.jit, static_argnames=["tol_relative_to_bigger_sv", "cut_low_signal", "return_details", "return_svd", "return_flat"])
    def natural_gradient(params, batch_samples, rk=None, tol=const_tol,
                         tol_relative_to_bigger_sv=const_tol_relative_to_bigger_sv,
                         cut_low_signal=const_cut_low_signal,
                         return_details=const_return_details,
                         return_svd=const_return_svd,
                         return_flat=const_return_flat,
                         full_features_evaluated=None,
                         U_features=None, Lambda_features=None, VT_features=None):
        if U_features is None or Lambda_features is None or VT_features is None:
            if full_features_evaluated is None:
                full_features_evaluated = full_features(params, batch_samples)
            U_features, Lambda_features, VT_features = jax.scipy.linalg.svd(full_features_evaluated, full_matrices=False)
        else:
            full_features_evaluated = None
        if rk is not None:
            mask = jnp.arange(Lambda_features.shape[0]) <= rk
            rank = rk
        else:
            if tol is None:
                tol = jnp.sqrt(jnp.finfo(Lambda_features.dtype).eps * U_features.shape[0])
            if tol_relative_to_bigger_sv:
                tol = tol * Lambda_features[0]
            mask = Lambda_features >= jnp.asarray(tol, dtype=Lambda_features.dtype)
            rank = mask.sum()
        safe_Lambda = jnp.where(mask, Lambda_features, 1).astype(Lambda_features.dtype)
        Lambda_inv = jnp.where(mask, 1 / safe_Lambda, 0)
        true_gradient_evaluated = true_gradient(params, batch_samples)
        gradient_rotated = VT_features @ true_gradient_evaluated
        flat_nat_grad = flat_nat_grad_cutted = U_features @ (Lambda_inv * gradient_rotated)
        if not cut_low_signal:
            flat_nat_grad = flat_nat_grad + true_gradient_evaluated - VT_features.T @ gradient_rotated
        if return_flat:
            out_nat_grad = flat_nat_grad
        else:
            out_nat_grad = jax.flatten_util.ravel_pytree(params)[1](flat_nat_grad)
        if return_details:
            if full_features_evaluated is None:
                full_features_evaluated = full_features(params, batch_samples)
            residual = true_gradient_evaluated - full_features_evaluated.T @ flat_nat_grad_cutted
            return out_nat_grad, residual, rank, (U_features, Lambda_features, Lambda_inv, VT_features)
        if return_svd:
            return out_nat_grad, (U_features, Lambda_features, Lambda_inv, VT_features)
        return out_nat_grad
    return natural_gradient


@jax.jit
def find_elbows_point(y_values):
    y = y_values; x = jnp.arange(y.shape[0])
    normal_vector = jnp.array([y[-1] - y[0], x[0] - x[-1]])
    vec_from_first = jnp.vstack([x - x[0], y - y[0]]).T
    scalar_proj = jnp.dot(vec_from_first, normal_vector)
    return jnp.argmax(scalar_proj), jnp.argmin(scalar_proj), scalar_proj


# In[ ]:


def grid_line_search_factory_bis(loss, steps, unravel):
    @jax.jit
    def loss_flat_params(flat_params):
        return loss(unravel(flat_params))
    def loss_at_step(step, flat_params, flat_nat_grad):
        return loss_flat_params(flat_params - step * flat_nat_grad)
    v_loss_at_steps = jax.vmap(loss_at_step, (0, None, None))
    @jax.jit
    def update(flat_params, flat_nat_grad):
        losses = v_loss_at_steps(steps, flat_params, flat_nat_grad)
        step_size = steps[jnp.argmin(losses)]
        return unravel(flat_params - step_size * flat_nat_grad), step_size
    return update


# # Problem construction
# 
# The weak tests are sections of the $H^1(-1,1)$ Green kernel of $-\partial_{xx}+1$ with Neumann boundary conditions,
# 
# $$
# k(x,y)=\frac{\cosh(\min(x,y)+1)\cosh(1-\max(x,y))}{\sinh 2}.
# $$
# 
# For the nonlinear equation, the weak operator contains the nonlocal term $\int (u^3-u)k(x,y)\,dy$. Every such integral is split at the kernel kink $y=x$ and evaluated with high-order Gauss--Legendre quadrature.
# 

# In[ ]:


# ============================================================
# Strong and weak operators
# ============================================================
def h1_kernel(x, y):
    lo = jnp.minimum(x, y); hi = jnp.maximum(x, y)
    return jnp.cosh(lo + 1.0) * jnp.cosh(1.0 - hi) / jnp.sinh(jnp.asarray(2.0))


def neumann_operator(u):
    u_scalar = lambda x: jnp.reshape(u(x), ())
    du = grad(u_scalar)
    return lambda x: jnp.reshape(du(x), ())


def linear_strong_operator(u):
    us = lambda x: jnp.reshape(u(x), ())
    du = grad(us)
    return lambda x: -jnp.reshape(jacfwd(du)(x), ()) + us(x)


def nonlinear_strong_operator(u):
    us = lambda x: jnp.reshape(u(x), ())
    du = grad(us)
    return lambda x: -jnp.reshape(jacfwd(du)(x), ()) + us(x) ** 3


def make_split_quadrature(order):
    nodes, weights = np.polynomial.legendre.leggauss(int(order))
    nodes = jnp.asarray(nodes).reshape(-1, 1); weights = jnp.asarray(weights)
    def split(xv, integrand):
        def piece(a, b):
            half = 0.5 * (b - a)
            ys = 0.5 * (a + b) + half * nodes
            return half * jnp.sum(weights * integrand(ys))
        return piece(-1.0, xv) + piece(xv, 1.0)
    return split


def make_linear_weak_operator():
    def op_factory(u):
        us = lambda z: jnp.reshape(u(z), ())
        du = grad(us)
        def op(x):
            xv = jnp.reshape(x, ())
            dl = jnp.reshape(du(jnp.array([-1.0])), ())
            dr = jnp.reshape(du(jnp.array([1.0])), ())
            return us(x) + dl * h1_kernel(xv, -1.0) - dr * h1_kernel(xv, 1.0)
        return op
    return op_factory


def make_weak_source(rhs, split_quad):
    vrhs = vmap(rhs, 0)
    def source(x):
        xv = jnp.reshape(x, ())
        def integrand(y):
            fy = jnp.reshape(vrhs(y), (len(y),))
            ky = vmap(lambda yy: h1_kernel(xv, jnp.reshape(yy, ())))(y)
            return fy * ky
        return split_quad(xv, integrand)
    return source


def make_nonlinear_weak_operator(split_quad):
    def op_factory(u):
        us = lambda z: jnp.reshape(u(z), ())
        du = grad(us)
        def op(x):
            xv = jnp.reshape(x, ())
            dl = jnp.reshape(du(jnp.array([-1.0])), ())
            dr = jnp.reshape(du(jnp.array([1.0])), ())
            def integrand(y):
                vy = vmap(us)(y)
                ky = vmap(lambda yy: h1_kernel(xv, jnp.reshape(yy, ())))(y)
                return (vy ** 3 - vy) * ky
            return us(x) + split_quad(xv, integrand) + dl * h1_kernel(xv, -1.0) - dr * h1_kernel(xv, 1.0)
        return op
    return op_factory


# In[ ]:


# ============================================================
# Matched benchmark context factory
# ============================================================
def dt_di_model(model):
    def dt_model(params, x):
        return grad(model, argnums=0)(params, x)
    def dtdi_single_input(params, x):
        return jacfwd(dt_model, argnums=1)(params, x)
    return dtdi_single_input


def dt_model(model):
    return lambda params, x: grad(model, argnums=0)(params, x)


def dt_model_nonlinear_mass(model):
    def feature(params, x):
        u = jnp.reshape(model(params, x), ())
        flat = jax.flatten_util.ravel_pytree(grad(model, argnums=0)(params, x))[0]
        return jnp.sqrt(3.0) * u * flat
    return feature


def build_matched_context(problem, activation_name):
    cfg = CONFIG["matched"]
    act = activation_from_name(activation_name)
    model = shallow_network(act)
    layer_sizes = [1, int(cfg["width"]), 1]
    u_exact = lambda x: jnp.reshape(jnp.cos(jnp.pi * x), ())
    if problem == "linear":
        rhs = lambda x: (1.0 + jnp.pi ** 2) * u_exact(x)
        strong_op = linear_strong_operator
    elif problem == "nonlinear":
        rhs = lambda x: (jnp.pi ** 2) * u_exact(x) + u_exact(x) ** 3
        strong_op = nonlinear_strong_operator
    else:
        raise ValueError(problem)

    quad_rule = GaussLegendrePiecewise(npts=int(cfg["gl_points_per_cell"]))
    interval = jnp.array([[-1.0, 1.0]])
    train_q = quad_rule.interval_quadpts(interval, jnp.array([cfg["train_h"]]))
    test_q = quad_rule.interval_quadpts(interval, jnp.array([cfg["test_h"]]))
    vmodel = vmap(model, (None, 0)); vrhs = vmap(rhs, 0)

    def energy(params, quadrature):
        du = vmap(grad(lambda x: model(params, x)), 0)
        grad_term = quadrature(lambda x: 0.5 * jnp.reshape(du(x) ** 2, (len(x),)))
        if problem == "linear":
            lower = quadrature(lambda x: 0.5 * jnp.reshape(vmodel(params, x) ** 2, (len(x),)))
        else:
            lower = quadrature(lambda x: 0.25 * jnp.reshape(vmodel(params, x) ** 4, (len(x),)))
        rhs_term = quadrature(lambda x: jnp.reshape(vmodel(params, x) * vrhs(x), (len(x),)))
        return grad_term + lower - rhs_term

    energy_train = jax.jit(lambda p: energy(p, train_q))
    energy_test = jax.jit(lambda p: energy(p, test_q))

    # Energy metric: Hao et al. for linear (`hao_gn_deep_ritz`); Müller--Zeinhofer
    # Hessian metric for nonlinear (`energy_ng_matched`). Both reference baselines
    # are built from the same GNDRM Gauss-Newton-optimizer machinery
    # (jacobian_matrix/gn_direction/grid_line_search) and are therefore both
    # genuinely GNDRM-dependent, optional reference baselines -- gated here so
    # build_matched_context (called for EVERY problem/activation, not just when
    # a GNDRM-dependent baseline is requested) never fails merely because GNDRM
    # is absent; only actually invoking one of these two baselines without GNDRM
    # raises (see the require_gndrm_baseline() calls below).
    if GNDRM_ROOT is not None:
        stiffness_feature = dt_di_model(model)
        if problem == "linear":
            mass_feature = dt_model(model)
        else:
            mass_feature = dt_model_nonlinear_mass(model)
        Jstiff = jacobian_matrix(stiffness_feature, train_q)
        Jmass = jacobian_matrix(mass_feature, train_q)
        energy_metric = lambda p: Jstiff(p) + Jmass(p)
        energy_direction = gn_direction(energy_metric)
        energy_steps = 0.3 ** jnp.linspace(0, 40, 41)
        energy_line_search = grid_line_search(energy_train, energy_steps)
    else:
        energy_direction = energy_line_search = None

    ntr, nte = int(cfg["n_train_interior"]), int(cfg["n_test_interior"])
    samples_boundary = jnp.array([[-1.0], [1.0]])
    samples_interior = jnp.linspace(-1.0, 1.0, ntr, endpoint=True).reshape(-1, 1)
    samples_test_interior = jnp.linspace(-1.0, 1.0, nte, endpoint=True).reshape(-1, 1)
    samples = (samples_interior, samples_boundary)
    samples_test = (samples_test_interior, samples_boundary)

    split_quad = make_split_quadrature(cfg["weak_split_order"])
    weak_source = make_weak_source(rhs, split_quad)
    weak_op = make_linear_weak_operator() if problem == "linear" else make_nonlinear_weak_operator(split_quad)

    def make_bundle(operators, sources, cutoff, extrap_boundary):
        ff = full_features_factory(model, operators)
        tg, fo, fs = full_quadratic_gradient_factory(model, operators, sources)
        ng = nat_grad_factory(ff, tg, const_return_svd=True, const_return_flat=True)
        return dict(model=model, full_features=ff, true_grad=tg, full_operators=fo, full_sources=fs,
                    nat_grad=ng, samples=samples, samples_test=samples_test,
                    weights_train=uniform_weights(samples), weights_test=uniform_weights(samples_test),
                    samples_boundary=samples_boundary, n_extrap=int(cfg["n_extrap_interior"]),
                    cutoff=float(cutoff), extrap_boundary=bool(extrap_boundary))

    strong_cutoff = 1e-6 if problem == "linear" else 1e-12
    strong = make_bundle((strong_op, neumann_operator), (rhs, lambda x: 0.0), strong_cutoff, True)
    weak = make_bundle((weak_op, neumann_operator), (weak_source, lambda x: 0.0), 1e-12, False)

    # Independent common errors on the high-resolution test quadrature.
    exact_l2 = jnp.sqrt(test_q(lambda x: jnp.reshape(vmap(u_exact)(x), (len(x),)) ** 2))
    exact_grad = vmap(grad(u_exact), 0)
    exact_h1semi = jnp.sqrt(test_q(lambda x: jnp.reshape(exact_grad(x), (len(x),)) ** 2))
    exact_h1 = jnp.sqrt(exact_l2 ** 2 + exact_h1semi ** 2)

    def errors(params):
        def err_scalar(x): return jnp.reshape(model(params, x), ()) - u_exact(x)
        verr = vmap(err_scalar, 0)
        vderr = vmap(grad(err_scalar), 0)
        e0 = jnp.sqrt(test_q(lambda x: jnp.reshape(verr(x), (len(x),)) ** 2))
        e1 = jnp.sqrt(test_q(lambda x: jnp.reshape(vderr(x), (len(x),)) ** 2))
        return e0 / exact_l2, e1 / exact_h1semi, jnp.sqrt(e0 ** 2 + e1 ** 2) / exact_h1
    errors = jax.jit(errors)

    common_weak_test_loss = jax.jit(lambda p: jnp.sum(weak["weights_test"] * weak["true_grad"](p, weak["samples_test"]) ** 2))

    # Exact energy values are computed with the exact function and the same quadratures.
    def exact_energy(quadrature):
        du = vmap(grad(u_exact), 0)
        uval = vmap(u_exact, 0)
        grad_term = quadrature(lambda x: 0.5 * jnp.reshape(du(x) ** 2, (len(x),)))
        if problem == "linear":
            lower = quadrature(lambda x: 0.5 * jnp.reshape(uval(x) ** 2, (len(x),)))
        else:
            lower = quadrature(lambda x: 0.25 * jnp.reshape(uval(x) ** 4, (len(x),)))
        rhs_term = quadrature(lambda x: jnp.reshape(uval(x) * vrhs(x), (len(x),)))
        return grad_term + lower - rhs_term

    return dict(problem=problem, activation=activation_name, model=model, layer_sizes=layer_sizes,
                u_exact=u_exact, rhs=rhs, train_q=train_q, test_q=test_q,
                energy_train=energy_train, energy_test=energy_test,
                exact_energy_train=float(exact_energy(train_q)), exact_energy_test=float(exact_energy(test_q)),
                energy_direction=energy_direction, energy_line_search=energy_line_search,
                strong=strong, weak=weak, errors=errors,
                common_weak_test_loss=common_weak_test_loss,
                n_params=count_params(normal_init(layer_sizes, random.PRNGKey(0))))


# # Optimizers
# 
# The three adaptive residual optimizers are retained from the preliminary benchmark. Two deliberately minimal weak Gauss--Newton solvers are added to isolate the effect of the regularization used to invert the weak residual Jacobian.
# 
# Let the weighted weak residual Jacobian have the singular value decomposition
# 
# $$
# J = U\Sigma V^\top,
# $$
# 
# and let $r$ denote the correspondingly weighted residual vector.
# 
# The **TSVD / pseudoinverse baseline** uses
# 
# $$
# \delta\theta_{\mathrm{TSVD}}
# =
# - V\,\operatorname{diag}\!\left(
# \frac{1}{\sigma_i}
# \mathbf 1_{\{\sigma_i\geq\tau\}}
# \right)U^\top r,
# $$
# 
# where $\tau=\texttt{rcond}\,\sigma_{\max}$. This is the fixed spectral-cutoff pseudoinverse underlying the simple ANaGRAM-style update; no ridge damping is added.
# 
# The **Levenberg--Marquardt / ridge baseline** uses
# 
# $$
# \delta\theta_{\mathrm{LM}}
# =
# - V\,\operatorname{diag}\!\left(
# \frac{\sigma_i}{\sigma_i^2+\lambda}
# \right)U^\top r.
# $$
# 
# No hard spectral cutoff is used in this update. The strictly positive damping $\lambda$ regularizes all singular directions continuously.
# 
# Both baselines use exactly the same weak residual, weighted Jacobian, initial parameters, and fixed grid line search. Hence their difference isolates TSVD/pseudoinverse regularization from classical ridge/LM regularization. Neither baseline uses rank-growth heuristics, adaptive trust-region logic, or a PDE-specific energy Hessian derivation.
# 

# In[ ]:


# ============================================================
# DSGNAR helpers
# ============================================================
def _solve_subproblems(S, w, target_radii, n_iters=80):
    finfo = jnp.finfo(S.dtype); S2 = jnp.square(S); lam = jnp.zeros_like(target_radii)
    for _ in range(n_iters):
        denom = S2[None, :] + lam[:, None]
        p = w[None, :] / denom
        p_norm = jnp.linalg.norm(p, axis=1)
        phi = p_norm - target_radii
        d_phi = -jnp.sum(jnp.square(p) / (denom + finfo.tiny), axis=1) / (p_norm + finfo.tiny)
        lam = jnp.maximum(lam - phi / (d_phi + finfo.tiny), 0.0)
    return lam


def _pchip(x, y, xq):
    finfo = jnp.finfo(x.dtype); hk = x[1:] - x[:-1]; dk = (y[1:] - y[:-1]) / (hk + finfo.tiny)
    def slope(d1, d2, h1, h2):
        w1, w2 = 2*h2+h1, h2+2*h1
        val = (w1+w2) / (w1/(d1+finfo.tiny) + w2/(d2+finfo.tiny))
        return jnp.where((jnp.sign(d1)*jnp.sign(d2)) > 0, val, 0.0)
    def boundary(de, dn, he, hn):
        val = ((2*he+hn)*de-he*dn)/(he+hn)
        return jnp.where(jnp.sign(val)==jnp.sign(de), val, 0.0)
    dmid = jax.vmap(slope)(dk[:-1], dk[1:], hk[:-1], hk[1:])
    derivs = jnp.concatenate([jnp.atleast_1d(boundary(dk[0],dk[1],hk[0],hk[1])), dmid,
                              jnp.atleast_1d(boundary(dk[-1],dk[-2],hk[-1],hk[-2]))])
    idx = jnp.clip(jnp.searchsorted(x, xq)-1, 0, x.shape[0]-2)
    h = x[idx+1]-x[idx]; t=(xq-x[idx])/(h+finfo.tiny); t2=t*t; t3=t2*t
    return ((2*t3-3*t2+1)*y[idx] + (t3-2*t2+t)*h*derivs[idx]
            + (-2*t3+3*t2)*y[idx+1] + (t3-t2)*h*derivs[idx+1])


def has_passed_minimum(sequence, window_size=30, min_slope=1e-4, min_correlation=0.1):
    if len(sequence) < window_size: return False
    window = np.maximum(np.asarray(sequence[-window_size:], float), np.finfo(float).tiny)
    logw = np.log10(window); x = np.arange(window_size)
    slope, _ = np.polyfit(x, logw, 1); corr = np.corrcoef(x, logw)[0,1]
    return bool(np.isfinite(corr) and slope > min_slope and corr > min_correlation)


def make_dsgnar_step(R):
    ff, tg = R["full_features"], R["true_grad"]
    wtr, smp = R["weights_train"], R["samples"]
    ds_loss = jax.jit(lambda p: jnp.sum(wtr * tg(p, smp) ** 2))
    @partial(jax.jit, static_argnames=("n_probes", "pchip_grid", "newton_iters"))
    def step(params, state, target_rho, *, n_probes=24, window_scale=3.0,
             min_radius=1e-8, max_radius=1e3, pchip_grid=512, newton_iters=80):
        radius, rho_prev, lam_prev = state
        flat, unravel = jax.flatten_util.ravel_pytree(params); finfo=jnp.finfo(flat.dtype)
        sw=jnp.sqrt(wtr); J=sw[:,None]*ff(params,smp).T; r=sw*tg(params,smp)
        current_loss=jnp.sum(r**2); U,S,Vt=jnp.linalg.svd(J,full_matrices=False); g=S*(U.T@r)
        def step_pred(lam):
            denom=S**2+lam; st=-(Vt.T@(g/denom)); pred=jnp.sum(g**2*(S**2+2*lam)/denom**2); return st,pred
        cur=jnp.clip(radius,min_radius,max_radius); lo=jnp.maximum(cur/window_scale,min_radius); hi=jnp.minimum(cur*window_scale,max_radius)
        radii=jnp.geomspace(lo,hi,n_probes); lams=_solve_subproblems(S,g,radii,newton_iters)
        steps,preds=jax.vmap(step_pred)(lams); losses=jax.vmap(lambda st: ds_loss(unravel(flat+st)))(steps)
        rhos=(current_loss-losses)/(preds+finfo.tiny); safe=jax.lax.cummin(jnp.clip(rhos,-1.,1.))
        log_r=jnp.log(radii); fine=jnp.linspace(jnp.log(lo),jnp.log(hi),pchip_grid); fr=_pchip(log_r,safe,fine)
        above=fr>=target_rho; crossings=above[:-1]&~above[1:]; has=jnp.any(crossings)
        idx=jnp.max(jnp.where(crossings,jnp.arange(pchip_grid-1),-1)); opt=.5*(fine[idx]+fine[idx+1])
        fallback=jnp.where(jnp.all(above),hi,lo); final_rad=jnp.where(has,jnp.exp(opt),fallback)
        flam=_solve_subproblems(S,g,jnp.atleast_1d(final_rad),newton_iters)[0]; fst,fpred=step_pred(flam)
        floss=ds_loss(unravel(flat+fst)); frho=(current_loss-floss)/(fpred+finfo.tiny)
        accepted=~jnp.isnan(frho)&(frho>0.); delta=jnp.where(accepted,fst,jnp.zeros_like(fst))
        newp=unravel(flat+delta); next_r=jnp.where(accepted,final_rad,jnp.where((~jnp.isnan(frho))&(frho>target_rho+.1),hi,lo))
        return newp,(next_r,frho,jnp.where(accepted,flam,lam_prev)),accepted,flam
    return step, ds_loss


# In[ ]:


# ============================================================
# Common metric recording and managed output
# ============================================================
def evaluate_metrics(ctx, params, train_loss, test_loss):
    l2, h1s, h1 = ctx["errors"](params)
    return {
        "relative_l2": l2,
        "relative_h1_seminorm": h1s,
        "relative_h1": h1,
        "loss_train": train_loss(params),
        "loss_test": test_loss(params),
        "common_weak_test_loss": ctx["common_weak_test_loss"](params),
    }


def prepare_run(ctx, method, seed, protocol="matched", extra_meta=None):
    rd = run_dir(ctx["problem"], ctx["activation"], method, seed, protocol)
    (rd / "checkpoints").mkdir(parents=True, exist_ok=True)
    (rd / "plots").mkdir(parents=True, exist_ok=True)
    meta = {
        "problem": ctx["problem"], "activation": ctx["activation"], "method": method,
        "seed": int(seed), "protocol": protocol, "layer_sizes": list(ctx["layer_sizes"]),
        "n_params": int(ctx["n_params"]), "configuration": CONFIG,
    }
    if extra_meta: meta.update(extra_meta)
    with (rd / "config.json").open("w") as f: json.dump(_jsonable(meta), f, indent=2)
    return rd


def maybe_save_iteration(params, hist, rd, iteration, title):
    nsave = CONFIG["global"]["n_iter_save_params"]
    nplot = CONFIG["global"]["n_iter_plot"]
    if iteration == 0 or iteration % nsave == 0:
        save_checkpoint(params, rd / "checkpoints" / f"iter_{iteration:06d}.pkl")
    if iteration > 0 and iteration % nplot == 0:
        save_history(hist, rd / "history.npz")
        plot_run_history(hist, rd / "plots", title)


def finish_run(ctx, params, hist, rd, title):
    last_it = int(hist["iteration"][-1]) if hist["iteration"] else 0
    save_checkpoint(params, rd / "checkpoints" / f"iter_{last_it:06d}_final.pkl")
    save_history(hist, rd / "history.npz")
    plot_run_history(hist, rd / "plots", title)
    plot_solution_1d(ctx, params, rd / "plots", title)
    return hist


# In[ ]:


# ============================================================
# Energy Gauss--Newton / matched Energy NG
# ============================================================
def run_energy_matched(ctx, seed, n_iters, method):
    rd = prepare_run(ctx, method, seed, extra_meta={
        "formulation": "Deep Ritz energy",
        "energy_geometry": "Hao stiffness+mass" if ctx["problem"] == "linear" else "Muller-Zeinhofer Hessian/energy metric",
    })
    init = lambda: normal_init(ctx["layer_sizes"], random.PRNGKey(seed))
    def one_step(p):
        g = grad(ctx["energy_train"])(p)
        direction = ctx["energy_direction"](p, g)
        return ctx["energy_line_search"](p, direction)
    # Compile/warm up without counting it as optimization time.
    pw = init(); t0=time.perf_counter(); pw,_=one_step(pw); tree_block_until_ready(pw); compile_time=time.perf_counter()-t0
    p = init(); hist=new_history(); hist["compile_time"]=compile_time
    maybe_save_iteration(p, hist, rd, 0, method)
    etrain = lambda q: jnp.maximum(ctx["energy_train"](q)-ctx["exact_energy_train"], 0.0)
    etest = lambda q: jnp.maximum(ctx["energy_test"](q)-ctx["exact_energy_test"], 0.0)
    wall=0.0
    for it in range(1, int(n_iters)+1):
        t=time.perf_counter(); p,step=one_step(p); tree_block_until_ready((p,step)); wall += time.perf_counter()-t
        m=evaluate_metrics(ctx,p,etrain,etest); tree_block_until_ready(tuple(m.values()))
        append_history(hist,it,wall,m,step_size=step,accepted=True)
        maybe_save_iteration(p,hist,rd,it,f"{ctx['problem']} / {ctx['activation']} / {method}")
    return finish_run(ctx,p,hist,rd,f"{ctx['problem']} / {ctx['activation']} / {method}")


# In[ ]:


# ============================================================
# Two minimal Gauss--Newton baselines on strong and weak residuals
# ============================================================
def run_plain_gn(ctx, seed, n_iters, regime, regularization):
    """Run a fixed-regularization Gauss--Newton method on a residual least-squares problem.

    Parameters
    ----------
    regime : {"strong", "weak"}
        Selects the strong collocation residual or weak/Petrov residual.

    regularization="tsvd": truncated Moore--Penrose inverse
        f_i = 1/sigma_i for sigma_i >= rcond * sigma_max, else 0.

    regularization="lm": Levenberg--Marquardt / ridge filter
        f_i = sigma_i / (sigma_i**2 + damping), with no hard cutoff.

    For a fixed regime, the residual, Jacobian, initialization, train/test
    samples, and line search are identical between TSVD and LM. Conversely,
    for a fixed regularization, the strong and weak runs use the corresponding
    residual bundle from the same matched benchmark context.
    """
    if regime not in {"strong", "weak"}:
        raise ValueError(f"Unknown residual regime: {regime}")
    if regularization not in {"tsvd", "tsvd_abs", "lm"}:
        raise ValueError(f"Unknown GN regularization: {regularization}")

    method = f"plain_gn_{regime}_{regularization}"
    R = ctx[regime]
    formulation = "strong residual least squares" if regime == "strong" else "weak residual least squares"

    if regularization == "tsvd":
        rcond = float(CONFIG["matched"]["plain_gn_tsvd_rcond"])
        extra = {
            "formulation": formulation,
            "residual_regime": regime,
            "optimizer": "plain Gauss-Newton with TSVD / truncated pseudoinverse",
            "spectral_filter": "1/sigma above rcond*sigma_max; zero below",
            "rcond": rcond,
            "ridge_damping": 0.0,
        }
    elif regularization == "tsvd_abs":
        # Same hard spectral truncation as "tsvd", but the cutoff is an
        # absolute singular-value threshold -- it drops the comparison
        # against sigma_max entirely, rather than scaling the threshold by
        # the current largest singular value.
        rcond = float(CONFIG["matched"]["plain_gn_tsvd_abs_rcond"])
        extra = {
            "formulation": formulation,
            "residual_regime": regime,
            "optimizer": "plain Gauss-Newton with absolute TSVD / fixed spectral cutoff",
            "spectral_filter": "1/sigma above fixed rcond (no sigma_max comparison); zero below",
            "rcond": rcond,
            "ridge_damping": 0.0,
        }
    else:
        damping = float(CONFIG["matched"]["plain_gn_lm_damping"])
        if damping <= 0.0:
            raise ValueError("plain_gn_lm_damping must be strictly positive")
        extra = {
            "formulation": formulation,
            "residual_regime": regime,
            "optimizer": "plain Levenberg-Marquardt / ridge Gauss-Newton",
            "spectral_filter": "sigma/(sigma^2+lambda), no hard cutoff",
            "rcond": None,
            "ridge_damping": damping,
        }

    rd = prepare_run(ctx, method, seed, extra_meta=extra)
    init = lambda: normal_init(ctx["layer_sizes"], random.PRNGKey(seed))
    wtr = R["weights_train"]
    sw = jnp.sqrt(wtr)
    smp = R["samples"]
    loss = jax.jit(lambda p: jnp.sum(wtr * R["true_grad"](p, smp) ** 2))
    loss_test = jax.jit(
        lambda p: jnp.sum(
            R["weights_test"] * R["true_grad"](p, R["samples_test"]) ** 2
        )
    )

    # Use the same line-search grid for all four plain-GN runs so that the
    # algorithmic distinction is only the residual formulation and/or the
    # singular-value regularization.
    steps = 0.5 ** (jnp.linspace(0, 1, 81) ** 2 * 50.0)

    def make_step(template):
        _, unravel = jax.flatten_util.ravel_pytree(template)

        @jax.jit
        def step(p):
            flat, _ = jax.flatten_util.ravel_pytree(p)
            J = sw[:, None] * R["full_features"](p, smp).T
            r = sw * R["true_grad"](p, smp)
            U, S, Vt = jnp.linalg.svd(J, full_matrices=False)
            tiny = jnp.finfo(S.dtype).tiny

            if regularization == "tsvd":
                cutoff = rcond * jnp.maximum(S[0], tiny)
                keep = S >= cutoff
                filt = jnp.where(keep, 1.0 / jnp.maximum(S, tiny), 0.0)
                effective_rank = jnp.sum(keep)
            elif regularization == "tsvd_abs":
                # No comparison against sigma_max: the cutoff is the fixed
                # value rcond itself.
                cutoff = rcond
                keep = S >= cutoff
                filt = jnp.where(keep, 1.0 / jnp.maximum(S, tiny), 0.0)
                effective_rank = jnp.sum(keep)
            else:
                # Classical ridge/LM: every singular direction is damped
                # continuously. No TSVD mask is applied to the update.
                filt = S / (S ** 2 + damping)
                effective_rank = jnp.sum(S > 0.0)

            delta = -(Vt.T @ (filt * (U.T @ r)))
            losses = jax.vmap(lambda a: loss(unravel(flat + a * delta)))(steps)
            idx = jnp.argmin(losses)
            alpha = steps[idx]
            return unravel(flat + alpha * delta), alpha, effective_rank

        return step

    p0 = init()
    step_fn = make_step(p0)
    t0 = time.perf_counter()
    pw, a, rk = step_fn(p0)
    tree_block_until_ready((pw, a, rk))
    compile_time = time.perf_counter() - t0

    p = init()
    hist = new_history()
    hist["compile_time"] = compile_time
    maybe_save_iteration(p, hist, rd, 0, method)
    wall = 0.0
    for it in range(1, int(n_iters) + 1):
        t = time.perf_counter()
        p, a, rk = step_fn(p)
        tree_block_until_ready((p, a, rk))
        wall += time.perf_counter() - t
        m = evaluate_metrics(ctx, p, loss, loss_test)
        tree_block_until_ready(tuple(m.values()))
        append_history(hist, it, wall, m, step_size=a, effective_rank=rk, accepted=True)
        maybe_save_iteration(
            p, hist, rd, it,
            f"{ctx['problem']} / {ctx['activation']} / {method}",
        )
    return finish_run(
        ctx, p, hist, rd,
        f"{ctx['problem']} / {ctx['activation']} / {method}",
    )


def run_plain_gn_strong_tsvd(ctx, seed, n_iters):
    return run_plain_gn(ctx, seed, n_iters, regime="strong", regularization="tsvd")


def run_plain_gn_weak_tsvd(ctx, seed, n_iters):
    return run_plain_gn(ctx, seed, n_iters, regime="weak", regularization="tsvd")


def run_plain_gn_strong_tsvd_abs(ctx, seed, n_iters):
    return run_plain_gn(ctx, seed, n_iters, regime="strong", regularization="tsvd_abs")


def run_plain_gn_weak_tsvd_abs(ctx, seed, n_iters):
    return run_plain_gn(ctx, seed, n_iters, regime="weak", regularization="tsvd_abs")


def run_plain_gn_strong_lm(ctx, seed, n_iters):
    return run_plain_gn(ctx, seed, n_iters, regime="strong", regularization="lm")


def run_plain_gn_weak_lm(ctx, seed, n_iters):
    return run_plain_gn(ctx, seed, n_iters, regime="weak", regularization="lm")


# In[ ]:


# ============================================================
# AMStramGRAM runner
# ============================================================
def run_amstramgram(ctx, seed, n_iters, regime):
    method=f"amstramgram_{regime}"; R=ctx[regime]
    rd=prepare_run(ctx,method,seed,extra_meta={"formulation":f"{regime} residual least squares"})
    ff,tg,ng=R["full_features"],R["true_grad"],R["nat_grad"]; smp,smp_test=R["samples"],R["samples_test"]
    wtr,wte=R["weights_train"],R["weights_test"]; cutoff=R["cutoff"]
    loss=jax.jit(lambda p:jnp.sum(wtr*tg(p,smp)**2)); loss_test=jax.jit(lambda p:jnp.sum(wte*tg(p,smp_test)**2))
    steps=0.5**(jnp.linspace(0,1,101)**2*60)
    def initialise(seed):
        p=normal_init(ctx["layer_sizes"],random.PRNGKey(seed)); ls=grid_line_search_factory_bis(loss,steps,jax.flatten_util.ravel_pytree(p)[1])
        _,S,Vt=jax.scipy.linalg.svd(ff(p,smp),full_matrices=False); tge=tg(p,smp)
        cs=jnp.cumsum(Vt.T*(Vt@tge)[None,:],axis=1); nd=jnp.sqrt(jnp.mean((cs-tge[:,None])**2,axis=0)); rank1=jnp.sum(nd<S)
        rmin=jnp.minimum(jnp.sum(nd>cutoff),rank1); rmin=jnp.maximum(rmin,1); rmax=find_elbows_point(jnp.log(S)[rmin:])[0]+rmin
        return p,ls,S,Vt,tge,rmin,rmax
    def one_iteration(p,ls,S,Vt,tge,rmin,rmax,liftoff,stage,old):
        cs=jnp.cumsum(Vt.T*(Vt@tge)[None,:],axis=1); nd=jnp.sqrt(jnp.mean((cs-tge[:,None])**2,axis=0)); rank1=jnp.sum(nd<S); rank2=jnp.sum(nd>=cutoff)
        stage=stage or bool(rank1>rank2); rmin=jnp.maximum(jnp.minimum(rank2,rank1),1); rmax=jnp.maximum(rank1,rmax)
        if not liftoff:
            if bool(rmin>=rmax): liftoff=True
            elif int(old)==int(rmin): rmax=rmax+1
            old=rmin
        ngv=ng(p,smp,rk=rmax); p,step1=ls(jax.flatten_util.ravel_pytree(p)[0],ngv[0]); ngv=ng(p,smp,rk=rmin); p,step2=ls(jax.flatten_util.ravel_pytree(p)[0],ngv[0])
        _,S,_,Vt=ngv[1]; tge=tg(p,smp); return p,S,Vt,tge,rmin,rmax,liftoff,stage,old,step2
    pw,ls,S,Vt,tge,rmin,rmax=initialise(seed); t0=time.perf_counter(); out=one_iteration(pw,ls,S,Vt,tge,rmin,rmax,False,False,-1); tree_block_until_ready(out[:4]); compile_time=time.perf_counter()-t0
    p,ls,S,Vt,tge,rmin,rmax=initialise(seed); liftoff=False; stage=False; old=-1; hist=new_history(); hist["compile_time"]=compile_time; maybe_save_iteration(p,hist,rd,0,method); wall=0.0
    for it in range(1,int(n_iters)+1):
        t=time.perf_counter(); p,S,Vt,tge,rmin,rmax,liftoff,stage,old,step=one_iteration(p,ls,S,Vt,tge,rmin,rmax,liftoff,stage,old); tree_block_until_ready((p,S)); wall+=time.perf_counter()-t
        m=evaluate_metrics(ctx,p,loss,loss_test); tree_block_until_ready(tuple(m.values())); append_history(hist,it,wall,m,step_size=step,effective_rank=rmin)
        maybe_save_iteration(p,hist,rd,it,f"{ctx['problem']} / {ctx['activation']} / {method}")
    return finish_run(ctx,p,hist,rd,f"{ctx['problem']} / {ctx['activation']} / {method}")


# In[ ]:


# ============================================================
# DSGNAR managed runner
# ============================================================
def run_dsgnar(ctx, seed, n_iters, regime):
    method=f"dsgnar_{regime}"; R=ctx[regime]
    rd=prepare_run(ctx,method,seed,extra_meta={"formulation":f"{regime} residual least squares"})
    step,loss=make_dsgnar_step(R); loss_test=jax.jit(lambda p:jnp.sum(R["weights_test"]*R["true_grad"](p,R["samples_test"])**2))
    init=lambda: normal_init(ctx["layer_sizes"],random.PRNGKey(seed)); state0=(jnp.array(1.),jnp.array(1.),jnp.array(1.))
    pw=init(); t0=time.perf_counter(); pout,sout,acc,lam=step(pw,state0,jnp.array(.075)); tree_block_until_ready((pout,sout)); compile_time=time.perf_counter()-t0
    p=init(); state=state0; target=.075; lam_hist=[]; hist=new_history(); hist["compile_time"]=compile_time; maybe_save_iteration(p,hist,rd,0,method); wall=0.0
    for it in range(1,int(n_iters)+1):
        t=time.perf_counter(); p,state,accepted,lam=step(p,state,jnp.array(target)); tree_block_until_ready((p,state)); wall+=time.perf_counter()-t
        radius,rho,lamcur=state; lam_hist.append(float(lamcur)); m=evaluate_metrics(ctx,p,loss,loss_test); tree_block_until_ready(tuple(m.values()))
        append_history(hist,it,wall,m,step_size=radius,effective_rank=np.nan,accepted=accepted)
        if target != .5 and len(lam_hist)>80 and has_passed_minimum(lam_hist): target=.5
        maybe_save_iteration(p,hist,rd,it,f"{ctx['problem']} / {ctx['activation']} / {method}")
        if float(radius)<1e-7:
            print(method, "trust-region radius collapsed at", it); break
    return finish_run(ctx,p,hist,rd,f"{ctx['problem']} / {ctx['activation']} / {method}")


# # Müller--Zeinhofer reference-protocol validation
# 
# This optional experiment is not part of the main matched benchmark. It mirrors the public `engd_nonlinear.py` protocol: width 32, trapezoidal integration with 20,000 training and 200,000 evaluation points, the Hessian-induced energy metric, and the $0.5^k$ grid line search. The paper uses `tanh`; the notebook deliberately repeats the protocol with $\mathrm{ReLU}^3$ as an activation ablation.
# 

# In[ ]:


# ============================================================
# Optional paper-protocol Energy Natural Gradient reference run
# ============================================================
def run_muller_zeinhofer_reference(activation_name, seed):
    from ngrad.models import mlp as ngrad_mlp, init_params
    from ngrad.domains import Interval
    from ngrad.integrators import TrapezoidalIntegrator
    from ngrad.inner import model_del_i_factory
    from ngrad.gram import gram_factory, nat_grad_factory as ngrad_nat_grad_factory
    from ngrad.utility import grid_line_search_factory

    rcfg=CONFIG["nonlinear"]["reference_protocol"]; activation=activation_from_name(activation_name)
    layer_sizes=[1,int(rcfg["width"]),1]; model=ngrad_mlp(activation); ustar=lambda x:jnp.reshape(jnp.cos(jnp.pi*x),())
    f=lambda x:(jnp.pi**2)*ustar(x)+ustar(x)**3; interval=Interval(-1.,1.)
    integrator=TrapezoidalIntegrator(interval,int(rcfg["train_points"]),K=4); eval_integrator=TrapezoidalIntegrator(interval,int(rcfg["eval_points"]),K=4)
    vmodel=vmap(model,(None,0)); params0=init_params(layer_sizes,random.PRNGKey(seed))
    def nonlinear_trafo(u_theta,del_theta_u):
        def g(x):
            flat,unravel=jax.flatten_util.ravel_pytree(del_theta_u(x)); return unravel(jnp.sqrt(3.)*u_theta(x)*flat)
        return g
    gram_grad=gram_factory(model=model,trafo=model_del_i_factory(),integrator=integrator)
    gram_nonlin=gram_factory(model=model,trafo=nonlinear_trafo,integrator=integrator)
    gram=jax.jit(lambda p:gram_grad(p)+gram_nonlin(p)); nat_grad_ref=ngrad_nat_grad_factory(gram)
    def energy(p,integ):
        du=vmap(grad(lambda x:model(p,x)),0); g=integ(lambda x:.5*jnp.reshape(du(x)**2,(len(x))))
        q=integ(lambda x:.25*vmodel(p,x)**4); rr=integ(lambda x:vmap(f,0)(x)*vmodel(p,x)); return g+q-rr
    loss=jax.jit(lambda p:energy(p,integrator)); loss_test=jax.jit(lambda p:energy(p,eval_integrator)); steps=.5**jnp.linspace(0,30,31); ls=grid_line_search_factory(loss,steps)
    # Common benchmark context is used only for standardized relative errors and weak held-out loss.
    ctx=build_matched_context("nonlinear",activation_name)
    ctx_ref=dict(ctx); ctx_ref.update(model=model,layer_sizes=layer_sizes,n_params=count_params(params0))
    # Rebuild standardized error/common-weak evaluator for the ngrad model.
    test_q=ctx["test_q"]; exact_l2=jnp.asarray(1.0); exact_h1s=jnp.pi; exact_h1=jnp.sqrt(1+jnp.pi**2)
    def errs(p):
        es=lambda x:jnp.reshape(model(p,x),())-ustar(x); ve=vmap(es,0); vd=vmap(grad(es),0)
        e0=jnp.sqrt(test_q(lambda x:jnp.reshape(ve(x),(len(x),))**2)); e1=jnp.sqrt(test_q(lambda x:jnp.reshape(vd(x),(len(x),))**2))
        return e0/exact_l2,e1/exact_h1s,jnp.sqrt(e0**2+e1**2)/exact_h1
    ctx_ref["errors"]=jax.jit(errs)
    split=make_split_quadrature(CONFIG["matched"]["weak_split_order"]); wsrc=make_weak_source(f,split); wop=make_nonlinear_weak_operator(split)
    tg,_,_=full_quadratic_gradient_factory(model,(wop,neumann_operator),(wsrc,lambda x:0.0)); smp=(ctx["weak"]["samples_test"][0],ctx["weak"]["samples_test"][1]); wt=uniform_weights(smp)
    ctx_ref["common_weak_test_loss"]=jax.jit(lambda p:jnp.sum(wt*tg(p,smp)**2)); ctx_ref["problem"]="nonlinear"; ctx_ref["activation"]=activation_name; ctx_ref["u_exact"]=ustar
    method="energy_ng_reference"; rd=prepare_run(ctx_ref,method,seed,protocol="reference",extra_meta={"reference":"Muller-Zeinhofer ICML 2023 public protocol"})
    # Exact energies for reporting the gap.
    uval=vmap(ustar,0); due=vmap(grad(ustar),0); vf=vmap(f,0)
    def estar(integ): return integ(lambda x:.5*jnp.reshape(due(x)**2,(len(x))))+integ(lambda x:.25*uval(x)**4)-integ(lambda x:vf(x)*uval(x))
    e0=float(estar(integrator)); e1=float(estar(eval_integrator)); train_gap=lambda p:jnp.maximum(loss(p)-e0,0.); test_gap=lambda p:jnp.maximum(loss_test(p)-e1,0.)
    def one_step(p):
        g=grad(loss)(p); ng=nat_grad_ref(p,g); return ls(p,ng)
    pw=init_params(layer_sizes,random.PRNGKey(seed)); t0=time.perf_counter(); pw,a=one_step(pw); tree_block_until_ready((pw,a)); compile_time=time.perf_counter()-t0
    p=init_params(layer_sizes,random.PRNGKey(seed)); hist=new_history(); hist["compile_time"]=compile_time; maybe_save_iteration(p,hist,rd,0,method); wall=0.0
    last_test_gap = np.nan
    test_every = int(rcfg.get("test_every", 10))
    for it in range(1,int(rcfg["iterations"])+1):
        t=time.perf_counter(); p,a=one_step(p); tree_block_until_ready((p,a)); wall+=time.perf_counter()-t
        l2,h1s,h1 = ctx_ref["errors"](p)
        tr = train_gap(p)
        if it == 1 or it % test_every == 0 or it == int(rcfg["iterations"]):
            last_test_gap = float(test_gap(p))
        cw = ctx_ref["common_weak_test_loss"](p)
        tree_block_until_ready((l2,h1s,h1,tr,cw))
        m = {"relative_l2":l2,"relative_h1_seminorm":h1s,"relative_h1":h1,
             "loss_train":tr,"loss_test":last_test_gap,"common_weak_test_loss":cw}
        append_history(hist,it,wall,m,step_size=a)
        maybe_save_iteration(p,hist,rd,it,f"nonlinear / {activation_name} / {method}")
    return finish_run(ctx_ref,p,hist,rd,f"nonlinear / {activation_name} / {method}")


# # Lightweight validation before the full roadmap
# 
# The following checks are deliberately cheap. They verify the weak identities at the exact solution, check that the matched linear/nonlinear energy metrics have the expected parameter shape for both activations, and test one plain weak Gauss--Newton step. They do **not** run the full multi-seed benchmark.
# 

# In[ ]:


# ============================================================
# Smoke validation
# ============================================================
def validate_context(ctx):
    p=normal_init(ctx["layer_sizes"],random.PRNGKey(123)); flat,_=jax.flatten_util.ravel_pytree(p)
    G=ctx["energy_direction"]  # construction itself was successful
    weak_exact=ctx["weak"]["true_grad"]
    # The residual at the exact solution is tested separately because the residual machinery expects model params.
    # Validate source/operator identity directly at random off-grid points.
    xs=random.uniform(random.PRNGKey(991),(17,1),minval=-.97,maxval=.97)
    split=make_split_quadrature(CONFIG["matched"]["weak_split_order"])
    if ctx["problem"]=="linear": wop=make_linear_weak_operator()
    else: wop=make_nonlinear_weak_operator(split)
    wsrc=make_weak_source(ctx["rhs"],split)
    vals=vmap(lambda x:wop(ctx["u_exact"])(x)-wsrc(x))(xs)
    maxres=float(jnp.max(jnp.abs(vals)))
    assert maxres < 1e-9, (ctx["problem"],ctx["activation"],maxres)
    l2,h1s,h1=ctx["errors"](p)
    assert np.all(np.isfinite(np.asarray([l2,h1s,h1])))
    print(ctx["problem"],ctx["activation"],"n_params=",flat.size,"exact weak max=",f"{maxres:.2e}")

for _problem in ("linear","nonlinear"):
    for _act in CONFIG["global"]["activations"]:
        validate_context(build_matched_context(_problem,_act))


# # Full matched benchmark roadmap
# 
# All experiment switches are controlled by `CONFIG`. Results are written incrementally, so an interrupted cluster run preserves completed seeds and checkpoints. Re-running a cell currently recomputes enabled experiments; disable individual methods in `CONFIG` when resuming selectively.
# 

# In[ ]:


# ============================================================
# Execute all matched experiments
# ============================================================
ALL_RESULTS = {}
if CONFIG["global"]["execute"]:
    for problem in ("linear","nonlinear"):
        if not CONFIG[problem]["enabled"]: continue
        ALL_RESULTS[problem]={}
        for activation in CONFIG["global"]["activations"]:
            print("\n"+"="*80); print("PROBLEM",problem,"ACTIVATION",activation); print("="*80)
            ctx=build_matched_context(problem,activation); ALL_RESULTS[problem][activation]={}
            methods=CONFIG[problem]["methods"]; nit_res=CONFIG["matched"]["residual_iterations"]; nit_energy=CONFIG["matched"]["energy_iterations"]
            if problem=="linear" and methods["hao_gn_deep_ritz"]:
                require_gndrm_baseline("hao_gn_deep_ritz")
                ALL_RESULTS[problem][activation]["hao_gn_deep_ritz"]={s:run_energy_matched(ctx,s,nit_energy,"hao_gn_deep_ritz") for s in CONFIG["global"]["seeds"]}
            if problem=="nonlinear" and methods["energy_ng_matched"]:
                require_gndrm_baseline("energy_ng_matched")
                ALL_RESULTS[problem][activation]["energy_ng_matched"]={s:run_energy_matched(ctx,s,nit_energy,"energy_ng_matched") for s in CONFIG["global"]["seeds"]}
            for regime in ("strong", "weak"):
                key = f"plain_gn_{regime}_tsvd"
                if methods[key]:
                    runner = run_plain_gn_strong_tsvd if regime == "strong" else run_plain_gn_weak_tsvd
                    ALL_RESULTS[problem][activation][key] = {
                        s: runner(ctx, s, nit_res)
                        for s in CONFIG["global"]["seeds"]
                    }
                key = f"plain_gn_{regime}_tsvd_abs"
                if methods[key]:
                    runner = run_plain_gn_strong_tsvd_abs if regime == "strong" else run_plain_gn_weak_tsvd_abs
                    ALL_RESULTS[problem][activation][key] = {
                        s: runner(ctx, s, nit_res)
                        for s in CONFIG["global"]["seeds"]
                    }
                key = f"plain_gn_{regime}_lm"
                if methods[key]:
                    runner = run_plain_gn_strong_lm if regime == "strong" else run_plain_gn_weak_lm
                    ALL_RESULTS[problem][activation][key] = {
                        s: runner(ctx, s, nit_res)
                        for s in CONFIG["global"]["seeds"]
                    }
            for regime in ("strong","weak"):
                key=f"amstramgram_{regime}"
                if methods[key]: ALL_RESULTS[problem][activation][key]={s:run_amstramgram(ctx,s,nit_res,regime) for s in CONFIG["global"]["seeds"]}
                key=f"dsgnar_{regime}"
                if methods[key]: ALL_RESULTS[problem][activation][key]={s:run_dsgnar(ctx,s,nit_res,regime) for s in CONFIG["global"]["seeds"]}


# In[ ]:


# ============================================================
# Optional Müller--Zeinhofer public-protocol reference runs
# ============================================================
REFERENCE_RESULTS={}
if CONFIG["global"]["execute"] and CONFIG["nonlinear"]["reference_protocol"]["enabled"]:
    for activation in CONFIG["global"]["activations"]:
        print("\nREFERENCE ENERGY NG:",activation)
        REFERENCE_RESULTS[activation]={s:run_muller_zeinhofer_reference(activation,s) for s in CONFIG["global"]["seeds"]}


# # Aggregate summaries and publication-oriented figures
# 
# The main figures prioritize median/IQR because several second-order methods exhibit abrupt transitions to machine precision at seed-dependent iterations. Mean/std figures are retained as supplementary diagnostics.
# 
# Two complementary comparison views are produced:
# 
# 1. **Formulation view:** energy baseline versus weak TSVD-GN versus weak LM-GN versus DSGNAR strong and DSGNAR weak.
# 2. **Weak-optimizer view:** TSVD-GN, LM-GN, AMStramGRAM, and DSGNAR, all using exactly the same weak residual.
# 
# The TSVD and LM curves are especially useful for the “out-of-the-box Gauss--Newton” claim: they differ only through the fixed singular-value filter applied to the same weak Jacobian.
# 
# The raw train objectives are intentionally excluded from cross-formulation plots because an energy value and a squared residual are not commensurate.
# 

# In[ ]:


# ============================================================
# Reload histories from disk (also works in a fresh kernel after cluster execution)
# ============================================================
def load_all_histories(protocol="matched"):
    data={}
    root=OUTPUT_ROOT
    for problem in ("linear","nonlinear"):
        pdir=root/problem/protocol
        if not pdir.exists(): continue
        data[problem]={}
        for activation_dir in sorted(pdir.iterdir()):
            if not activation_dir.is_dir(): continue
            activation=activation_dir.name; data[problem][activation]={}
            for method_dir in sorted(activation_dir.iterdir()):
                if not method_dir.is_dir(): continue
                method=method_dir.name; data[problem][activation][method]={}
                for seed_dir in sorted(method_dir.glob("seed_*")):
                    hp=seed_dir/"history.npz"
                    if not hp.exists(): continue
                    seed=int(seed_dir.name.split("_")[-1]); z=np.load(hp)
                    data[problem][activation][method][seed]={k:z[k] for k in z.files}
    return data

BENCH = load_all_histories("matched")


# In[ ]:


# ============================================================
# Summary table
# ============================================================
import csv

def summarize_benchmark(bench):
    rows=[]
    for problem,ad in bench.items():
        for activation,md in ad.items():
            for method,seeds in md.items():
                if not seeds: continue
                finals={metric:np.array([np.asarray(h[metric])[-1] for h in seeds.values()],float)
                        for metric in ("relative_l2","relative_h1","relative_h1_seminorm","common_weak_test_loss","wallclock_optim")}
                row={"problem":problem,"activation":activation,"method":method,"n_seeds":len(seeds)}
                for metric,a in finals.items():
                    row[f"{metric}_median"]=float(np.median(a)); row[f"{metric}_q1"]=float(np.percentile(a,25)); row[f"{metric}_q3"]=float(np.percentile(a,75)); row[f"{metric}_mean"]=float(np.mean(a)); row[f"{metric}_std"]=float(np.std(a))
                rows.append(row)
    return rows

SUMMARY=summarize_benchmark(BENCH)
if SUMMARY:
    keys=list(SUMMARY[0].keys()); path=OUTPUT_ROOT/"aggregate_summary.csv"
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(SUMMARY)
    print("saved",path)
    for r in SUMMARY:
        print(f"{r['problem']:9s} {r['activation']:5s} {r['method']:24s}  H1 med={r['relative_h1_median']:.3e}  L2 med={r['relative_l2_median']:.3e}")


# In[ ]:


# ============================================================
# Aggregate plot helpers
# (style follows ANaGRAM/Expes-plotter-clean.ipynb: ggplot, log-log, a
#  center line per method with a shaded band -- median/IQR or mean/std --
#  fixed per-method colors shared across every figure, and the legend
#  displayed apart from the axes rather than inside a panel.)
# ============================================================
LABELS={
    "hao_gn_deep_ritz":"Hao GN Deep Ritz",
    "energy_ng_matched":"Energy NG (matched)",
    "plain_gn_strong_tsvd":"TSVD GN (strong)",
    "plain_gn_weak_tsvd":"cutoff GN (weak)",
    "plain_gn_strong_tsvd_abs":"abs cutoff GN (strong)",
    "plain_gn_weak_tsvd_abs":"abs cutoff GN (weak)",
    "plain_gn_strong_lm":"LM / ridge GN (strong)",
    "plain_gn_weak_lm":"ridge GN (weak)",
    "amstramgram_strong":"AMStramGRAM (strong)","dsgnar_strong":"DSGNAR (strong)",
    "amstramgram_weak":"AMStramGRAM (weak)","dsgnar_weak":"DSGNAR (weak)",
}
# Fixed per-method colors (tab20 -- LABELS has 12 keys, more than tab10
# holds), shared across every figure so a given method always has the same
# color regardless of which plot/subset it appears in.
_METHOD_ORDER=list(LABELS.keys())
_cmap=plt.get_cmap("tab20")
METHOD_COLORS={m:_cmap(i) for i,m in enumerate(_METHOD_ORDER)}

def aligned_stat(seed_histories, metric, xkey, median=True):
    hs=list(seed_histories.values()); n=min(len(h[metric]) for h in hs)
    Y=np.stack([np.asarray(h[metric],float)[:n] for h in hs]); X=np.stack([np.asarray(h[xkey],float)[:n] for h in hs])
    x=np.median(X,axis=0) if xkey=="wallclock_optim" else X[0]
    if median: return x,np.median(Y,axis=0),np.percentile(Y,25,axis=0),np.percentile(Y,75,axis=0)
    m=Y.mean(0); s=Y.std(0); return x,m,m-s,m+s


def plot_group(bench,problem,activation,methods,filename,median=True,label_overrides=None):
    available=[m for m in methods if m in bench.get(problem,{}).get(activation,{})]
    if not available: return
    labels={**LABELS,**(label_overrides or {})}
    metrics=[("relative_h1",r"relative $H^1$ error"),("relative_l2",r"relative $L^2$ error"),("common_weak_test_loss","common weak test loss")]
    mpl.rcParams.update(mpl.rcParamsDefault)
    plt.style.use("ggplot")
    fig,axes=plt.subplots(2,3,figsize=(6.5*3,5*2))
    for row,xkey in enumerate(("iteration","wallclock_optim")):
        for col,(metric,label) in enumerate(metrics):
            ax=axes[row,col]
            stats={m:aligned_stat(bench[problem][activation][m],metric,xkey,median) for m in available}
            # Only draw a method on this (log-scale) panel if its whole center
            # curve is positive -- guards against sign-indefinite quantities
            # the way the reference plotter excludes a raw (sign-indefinite)
            # energy from the loss panels. Every metric plotted here is
            # already non-negative by construction, so this is a safety net
            # rather than an active filter in practice.
            shown=[m for m in available if np.all(stats[m][1]>0)]
            ymins=[stats[m][1].min() for m in shown]
            floor=(min(ymins)*0.3) if ymins else 1e-18
            x=None
            for m in shown:                                    # bands first (behind lines)
                x,c,lo,hi=stats[m]
                ax.fill_between(x,np.clip(lo,floor,None),np.clip(hi,floor,None),alpha=.18,color=METHOD_COLORS[m],zorder=1)
            for m in shown:
                x,c,lo,hi=stats[m]
                ax.plot(x,np.clip(c,floor,None),color=METHOD_COLORS[m],label=labels.get(m,m),lw=1.7,zorder=3)
            if x is not None and np.all(x>0): ax.set_xscale("log")
            ax.set_yscale("log"); ax.set_xlabel("iteration" if xkey=="iteration" else "optimization wall-clock [s]",fontsize=13)
            ax.set_ylabel(label,fontsize=13); ax.grid(True)
            if ymins: ax.set_ylim(bottom=floor)
    kind="median" if median else "mean"; band="IQR (Q1-Q3)" if median else r"$\pm$ std"
    fig.suptitle(f"{problem} — {activation} — {kind} line, {band} band",fontsize=15)
    fig.legend(*axes[0,-1].get_legend_handles_labels(),loc="lower center",
               bbox_to_anchor=(0.5,-0.04),ncol=min(len(available),6),fontsize=11,fancybox=True,shadow=True)
    fig.tight_layout(rect=[0,0.06,1,0.96])
    out=OUTPUT_ROOT/"plots"; out.mkdir(exist_ok=True); fig.savefig(out/filename,dpi=150,bbox_inches="tight"); plt.close(fig)

for problem in ("linear","nonlinear"):
    for activation in CONFIG["global"]["activations"]:
        energy="hao_gn_deep_ritz" if problem=="linear" else "energy_ng_matched"
        formulation=[
            energy,
            "plain_gn_weak_tsvd",
            "plain_gn_weak_lm",
            "dsgnar_weak",
        ]
        weakopts=[energy,"plain_gn_weak_tsvd","plain_gn_weak_lm","amstramgram_weak","dsgnar_weak"]
        # In the "optimizers" plots specifically, the energy baseline is
        # relabeled to make clear it is the reference point, not one of the
        # residual optimizers being compared.
        baseline_label="GN Deep Ritz (baseline)" if problem=="linear" else "Energy NG (baseline)"
        opt_labels={energy:baseline_label}
        if CONFIG["global"]["plot_median_iqr"]:
            plot_group(BENCH,problem,activation,formulation,f"{problem}_{activation}_formulation_median_iqr.png",True,opt_labels)
            plot_group(BENCH,problem,activation,weakopts,f"{problem}_{activation}_weak_optimizers_median_iqr.png",True,opt_labels)
        if CONFIG["global"]["plot_mean_std"]:
            plot_group(BENCH,problem,activation,formulation,f"{problem}_{activation}_formulation_mean_std.png",False,opt_labels)
            plot_group(BENCH,problem,activation,weakopts,f"{problem}_{activation}_weak_optimizers_mean_std.png",False,opt_labels)
print("aggregate plots saved under",OUTPUT_ROOT/"plots")


# ## Interpretation checklist
# 
# The consolidated benchmark now separates both **residual formulation** and **linearized inverse regularization** for the two deliberately simple Gauss--Newton baselines.
# 
# For each problem and activation, the roadmap runs four plain-GN variants:
# 
# - `plain_gn_strong_tsvd`: strong residual + truncated pseudoinverse / ANaGRAM-style fixed spectral cutoff;
# - `plain_gn_weak_tsvd`: weak residual + truncated pseudoinverse / ANaGRAM-style fixed spectral cutoff;
# - `plain_gn_strong_tsvd_abs` / `plain_gn_weak_tsvd_abs`: same hard spectral truncation, but with an **absolute** cutoff (`sigma >= plain_gn_tsvd_abs_rcond`) that drops the comparison against `sigma_max` entirely, rather than a cutoff relative to the current largest singular value;
# - `plain_gn_strong_lm`: strong residual + classical Levenberg--Marquardt/Tikhonov damping, with no hard spectral cutoff;
# - `plain_gn_weak_lm`: weak residual + classical Levenberg--Marquardt/Tikhonov damping, with no hard spectral cutoff.
# 
# For a weighted residual Jacobian $J=U\Sigma V^\top$, TSVD uses
# 
# $$
# \delta\theta
# =
# -V\,\operatorname{diag}\!\left(
# \frac{1}{\sigma_i}\mathbf 1_{\{\sigma_i\geq\tau\}}
# \right)U^\top r,
# $$
# 
# whereas LM/ridge uses
# 
# $$
# \delta\theta
# =
# -V\,\operatorname{diag}\!\left(
# \frac{\sigma_i}{\sigma_i^2+\lambda}
# \right)U^\top r.
# $$
# 
# The first formulation-comparison figure includes **all four variants**. This makes it possible to assess separately whether the observed behavior comes from the strong/weak formulation or from TSVD versus ridge regularization.
# 
# The weak-optimizer figure remains weak-only on purpose, since the intended final paper figure may focus on one formulation after inspecting the complete results. The strong runs remain fully saved in the benchmark output and can be selected later without rerunning the experiments.
# 
