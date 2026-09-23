#!/usr/bin/env python
"""
[Copied from the research tree with minimal path/import adaptation only -- see docs/PROVENANCE.md.]
Single fresh-process FE timing shot. Launched as a brand-new interpreter
for every repetition (by the driver shell loop), so this captures a genuine
T_FE_cold sample: no JAX/XLA in-memory state, no warm Python import cache,
can persist across repetitions -- only whatever the NVIDIA driver-level
PTX/SASS ComputeCache (~/.nv/ComputeCache, outside JAX's own control) or an
explicit JAX_COMPILATION_CACHE_DIR (checked and reported, not set here) might
still carry over between processes.

Prints one JSON line to stdout: {benchmark, T_FE_total, n_dofs, rel_l2, rel_h1}.
Usage: python run_fe_single_cold.py MS|JF|LS|RC
"""
import json, sys, time
from pathlib import Path
import numpy as np
import petrov_fem_utils as F

REF_DIR = Path("fe_refs")

def ms_solve():
    return F.solve_lagrange_1d(96, 4, F.A_eps_1d, f_func=lambda x: 1.0, n_quad=20)

def jf_solve():
    nodes, tris, bm = F.make_unit_square_mesh(12, 8)
    return F.solve_pk_dirichlet(nodes, tris, bm, F.f22_single, degree=4,
                                 quadrature_order=10, A_func=F.A_one)

def ls_solve():
    nodes, tris, bm = F.make_unit_square_mesh(14, 12)
    return F.solve_pk_dirichlet(nodes, tris, bm, F.f24_reg_single, degree=3, A_func=F.A_one,
                                 line_source=(F.kink_line_points, F.kink_line_weighted_density))

def rc_solve():
    nodes, tris, bm = F.make_lshape_corner_mesh(n_radial=16, n_angular=40, beta=2.5, grading_mode="power")
    return F.solve_pk_dirichlet(nodes, tris, bm, F.fL_single, degree=2, A_func=None)

BENCH = {
    "MS": dict(solve=ms_solve, is_1d=True, ref="ms_ref.npz"),
    "JF": dict(solve=jf_solve, is_1d=False, ref="jf_ref.npz"),
    "LS": dict(solve=ls_solve, is_1d=False, ref="ls_ref.npz"),
    "RC": dict(solve=rc_solve, is_1d=False, ref="rc_ref.npz"),
}

def relerr(u_num, u_exact, g_num=None, g_exact=None):
    l2_num = float(np.sum((u_num - u_exact) ** 2)); l2_den = float(np.sum(u_exact ** 2))
    rel_l2 = float(np.sqrt(l2_num / l2_den))
    if g_num is None:
        return rel_l2, None
    g_num_sq = float(np.sum((g_num - g_exact) ** 2)); g_den_sq = float(np.sum(g_exact ** 2))
    rel_h1 = float(np.sqrt((l2_num + g_num_sq) / (l2_den + g_den_sq)))
    return rel_l2, rel_h1

tag = sys.argv[1]
cfg = BENCH[tag]

t0 = time.perf_counter()
bundle = cfg["solve"]()
t1 = time.perf_counter()

ref = np.load(REF_DIR / cfg["ref"])
if cfg["is_1d"]:
    u_eval = F.eval_lagrange_1d(bundle, ref["x"], derivative=False)
    g_eval = F.eval_lagrange_1d(bundle, ref["x"], derivative=True)
else:
    u_eval, g_eval = F.eval_pk_solution(bundle, ref["points"], return_gradient=True)
rel_l2, rel_h1 = relerr(u_eval, ref["u_exact"], g_eval, ref["grad_exact"])

print(json.dumps(dict(
    benchmark=tag, T_FE_total=t1 - t0, n_dofs=int(bundle["n_dofs"]),
    rel_l2=rel_l2, rel_h1=rel_h1,
)))
