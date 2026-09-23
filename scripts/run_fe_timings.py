#!/usr/bin/env python
"""
[Copied from the research tree with minimal path/import adaptation only -- see docs/PROVENANCE.md.]
Controlled, common end-to-end FE timing benchmark for MS/JF/LS/RC.

Uses petrov_fem_utils.py verbatim (the already-established, dependency-light
re-implementation of the FE assembly/solve routines used to build
petrov_standalone_fem_cache/) -- no changes to its solver math. Adds ONE new
common timer wrapping mesh construction + basis/quadrature setup + BC
bookkeeping + assembly + sparse direct solve, run fresh (no reuse) 5 times per
candidate, matching fem_baselines/selected.json -> best_fem_h1_under_hybrid_budget
for each section exactly (same degree/mesh/coefficient/RHS as
compute_standalone_fem.py). Error evaluation happens strictly OUTSIDE the
timed region, as a separate validation step against the already-recorded
selected.json rel_l2/rel_h1.

Writes machine_metadata.json and per-benchmark timing.json + one aggregate
CSV under the given output directory.
"""
import json, platform, subprocess, sys, time, csv
from pathlib import Path
from datetime import datetime, timezone
import numpy as np

import petrov_fem_utils as F

OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("controlled_timings_fe")
OUT_DIR.mkdir(parents=True, exist_ok=True)
REF_DIR = Path("results/reference/fem")  # standalone FE cache from the reference-results archive
if not REF_DIR.exists():
    raise SystemExit(
        "Reference results are not installed.\n\n"
        "Download/extract the Beyond PINNs reference-results archive into:\n"
        "    results/reference/\n\n"
        f"(expected to find data at {REF_DIR.resolve()})\n"
        "See docs/REPRODUCIBILITY.md or run scripts/download_reference_results.sh."
    )

# ------------------------------------------------------------------
# Machine metadata
# ------------------------------------------------------------------
def _cmd(args):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception as e:
        return f"<unavailable: {e}>"

meta = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "os": platform.platform(),
    "python_version": sys.version,
    "cpu_model": _cmd(["bash", "-c", "lscpu | grep 'Model name' | head -1"]) or _cmd(["bash", "-c", "cat /proc/cpuinfo | grep 'model name' | head -1"]),
    "cpu_count_logical": _cmd(["bash", "-c", "nproc"]),
    "cpu_count_physical_info": _cmd(["bash", "-c", "lscpu | grep -E '^Socket|^Core\\(s\\) per socket|^Thread\\(s\\) per core'"]),
    "gpu_info": _cmd(["bash", "-c", "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo none"]),
    "numpy_version": np.__version__,
    "scipy_version": __import__("scipy").__version__,
    "dtype": "float64 (numpy/scipy default; this FE benchmark does not require jax_enable_x64)",
    "OMP_NUM_THREADS": __import__("os").environ.get("OMP_NUM_THREADS"),
    "MKL_NUM_THREADS": __import__("os").environ.get("MKL_NUM_THREADS"),
    "OPENBLAS_NUM_THREADS": __import__("os").environ.get("OPENBLAS_NUM_THREADS"),
    "NUMEXPR_NUM_THREADS": __import__("os").environ.get("NUMEXPR_NUM_THREADS"),
    "notes": "This process does not import jax for the timed FE region beyond "
             "petrov_fem_utils' own jax.vmap(A_func)/jax.vmap(f_func) coefficient "
             "evaluations, which are materialized via np.asarray(...) inside the "
             "already-existing solver code (a genuine synchronization point). "
             "jax device/version recorded separately if jax is importable.",
}
try:
    import jax
    meta["jax_version"] = jax.__version__
    meta["jax_devices"] = [str(d) for d in jax.devices()]
    meta["jax_default_backend"] = jax.default_backend()
except Exception as e:
    meta["jax_import_error"] = repr(e)

with open(OUT_DIR / "machine_metadata.json", "w") as fh:
    json.dump(meta, fh, indent=2)
print("machine metadata:")
print(json.dumps(meta, indent=2))

# ------------------------------------------------------------------
# Candidate definitions -- verbatim from compute_standalone_fem.py, each
# wrapped as a zero-arg callable that does mesh construction THROUGH solve
# fresh on every call (no reuse across repetitions).
# ------------------------------------------------------------------
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

BENCHMARKS = {
    "MS": dict(solve=ms_solve, is_1d=True, ref="section_1.npz", expected_dofs=383, expected_degree=4,
               expected_desc="96 uniform elements", selected_json="results/reference/general/section_1/fem_baselines/selected.json"),
    "JF": dict(solve=jf_solve, is_1d=False, ref="section_2.npz", expected_dofs=1457, expected_degree=4,
               expected_desc="12x8 interface P4", selected_json="results/reference/general/section_2/fem_baselines/selected.json"),
    "LS": dict(solve=ls_solve, is_1d=False, ref="section_3.npz", expected_dofs=1435, expected_degree=3,
               expected_desc="14x12 interface P3", selected_json="results/reference/general/section_3/fem_baselines/selected.json"),
    "RC": dict(solve=rc_solve, is_1d=False, ref="section_4.npz", expected_dofs=2409, expected_degree=2,
               expected_desc="power P2 nr=16 na=40 beta=2.5", selected_json="results/reference/general/section_4/fem_baselines/selected.json"),
}

def relerr(u_num, u_exact, g_num=None, g_exact=None):
    l2_num = float(np.sum((u_num - u_exact) ** 2)); l2_den = float(np.sum(u_exact ** 2))
    rel_l2 = float(np.sqrt(l2_num / l2_den))
    if g_num is None:
        return rel_l2, None
    g_num_sq = float(np.sum((g_num - g_exact) ** 2)); g_den_sq = float(np.sum(g_exact ** 2))
    rel_h1 = float(np.sqrt((l2_num + g_num_sq) / (l2_den + g_den_sq)))
    return rel_l2, rel_h1

REPS = 5
rows = []
for tag, cfg in BENCHMARKS.items():
    print(f"\n=== {tag} ===")
    sel = json.load(open(cfg["selected_json"]))["best_fem_h1_under_hybrid_budget"]
    assert sel["description"] == cfg["expected_desc"] and sel["degree"] == cfg["expected_degree"], \
        f"{tag}: selected.json candidate does not match expected -- STOPPING rather than silently choosing"

    times = []
    bundle = None
    for rep in range(REPS):
        t0 = time.perf_counter()
        bundle = cfg["solve"]()   # mesh + basis/quadrature + BC bookkeeping + assembly + direct solve, fresh
        t1 = time.perf_counter()
        times.append(t1 - t0)
        print(f"  rep {rep}: T_FE_total = {t1-t0:.4f} s  (n_dofs={bundle['n_dofs']})")

    times = np.array(times)
    median_t = float(np.median(times))
    q1, q3 = float(np.percentile(times, 25)), float(np.percentile(times, 75))

    # Accuracy validation, OUTSIDE the timed region, against a saved reference grid.
    ref_path = REF_DIR / cfg["ref"]
    ref = np.load(ref_path)
    if cfg["is_1d"]:
        u_eval = F.eval_lagrange_1d(bundle, ref["x"], derivative=False)
        g_eval = F.eval_lagrange_1d(bundle, ref["x"], derivative=True)
    else:
        u_eval, g_eval = F.eval_pk_solution(bundle, ref["points"], return_gradient=True)
    rel_l2, rel_h1 = relerr(u_eval, ref["u"], g_eval, ref["grad"])

    n_dofs = int(bundle["n_dofs"])
    dofs_match = (n_dofs == cfg["expected_dofs"])
    l2_close = abs(rel_l2 - sel["rel_l2"]) < 1e-6 * max(1.0, abs(sel["rel_l2"]))
    h1_close = abs(rel_h1 - sel["rel_h1"]) < 1e-6 * max(1.0, abs(sel["rel_h1"]))
    print(f"  n_dofs={n_dofs} (selected.json {cfg['expected_dofs']}, match={dofs_match})")
    print(f"  rel_l2={rel_l2:.6e} (selected.json {sel['rel_l2']:.6e}, close={l2_close})")
    print(f"  rel_h1={rel_h1:.6e} (selected.json {sel['rel_h1']:.6e}, close={h1_close})")
    print(f"  T_FE_total median={median_t:.4f}s  IQR=[{q1:.4f}, {q3:.4f}]")

    result = dict(
        benchmark=tag, degree=cfg["expected_degree"], mesh_description=cfg["expected_desc"],
        n_dofs=n_dofs, n_dofs_matches_selected=dofs_match,
        repetitions=REPS, T_FE_total_per_rep=times.tolist(),
        T_FE_total_median=median_t, T_FE_total_q1=q1, T_FE_total_q3=q3,
        final_rel_l2=rel_l2, final_rel_h1=rel_h1,
        selected_json_rel_l2=sel["rel_l2"], selected_json_rel_h1=sel["rel_h1"],
        accuracy_matches_selected_json=bool(l2_close and h1_close),
        note="T_FE_total = one common timer around mesh/enriched-mesh construction, "
             "basis/quadrature setup, BC bookkeeping, sparse assembly, and direct "
             "sparse solve, run fresh (no factorization/matrix reuse) each repetition. "
             "Error evaluation is strictly outside the timed region.",
    )
    with open(OUT_DIR / f"timing_FE_{tag}.json", "w") as fh:
        json.dump(result, fh, indent=2)
    rows.append(result)

with open(OUT_DIR / "aggregate_FE_timings.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["benchmark", "degree", "mesh_description", "n_dofs",
                                        "repetitions", "T_FE_total_median", "T_FE_total_q1", "T_FE_total_q3",
                                        "final_rel_l2", "final_rel_h1", "accuracy_matches_selected_json"])
    w.writeheader()
    for r in rows:
        w.writerow({k: r[k] for k in w.fieldnames})

print(f"\nSaved machine_metadata.json, per-benchmark timing_FE_*.json, and aggregate_FE_timings.csv under {OUT_DIR}")
