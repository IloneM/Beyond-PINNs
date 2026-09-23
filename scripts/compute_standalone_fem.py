#!/usr/bin/env python
"""
[Copied from the research tree with minimal path/import adaptation only -- see docs/PROVENANCE.md.]
Reconstruct the standalone near-budget FEM reference solution field for
each section, at the EXACT candidate described by
fem_baselines/selected.json -> best_fem_h1_under_hybrid_budget. Deterministic
FEM assembly/solve only -- no neural-network optimization.

Caches results to petrov_standalone_fem_cache/section_N.npz and prints a
self-check against the rel_l2/rel_h1 already recorded in selected.json.
"""
import json
from pathlib import Path
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from beyond_pinns import fem_utils as F

ROOT = Path("results/reference/general")
CACHE = Path("results/reference/fem")
if not ROOT.exists():
    raise SystemExit(
        "Reference results are not installed.\n\n"
        "Download/extract the Beyond PINNs reference-results archive into:\n"
        "    results/reference/\n\n"
        f"(expected to find data at {ROOT.resolve()})\n"
        "See docs/REPRODUCIBILITY.md or run scripts/download_reference_results.sh."
    )
CACHE.mkdir(exist_ok=True, parents=True)


def relerr(u_num, u_exact, g_num=None, g_exact=None):
    l2_num = float(np.sum((u_num - u_exact) ** 2)); l2_den = float(np.sum(u_exact ** 2))
    rel_l2 = float(np.sqrt(l2_num / l2_den))
    if g_num is None:
        return rel_l2, None
    g_num_sq = float(np.sum((g_num - g_exact) ** 2)); g_den_sq = float(np.sum(g_exact ** 2))
    rel_h1 = float(np.sqrt((l2_num + g_num_sq) / (l2_den + g_den_sq)))
    return rel_l2, rel_h1


# ---------------- Section 1: 1D, degree=4, 96 elements ----------------
print("=== Section 1 ===")
sel1 = json.load(open(ROOT / "section_1" / "fem_baselines" / "selected.json"))["best_fem_h1_under_hybrid_budget"]
assert sel1["description"] == "96 uniform elements" and sel1["degree"] == 4
bundle1 = F.solve_lagrange_1d(96, 4, F.A_eps_1d, f_func=lambda x: 1.0, n_quad=20)
ref1 = np.load(next((ROOT / "section_1" / "s1_proj_h01_green_quadrature").glob("seed_*/final_solution.npz")))
x1 = ref1["x"]
u1 = F.eval_lagrange_1d(bundle1, x1, derivative=False)
du1 = F.eval_lagrange_1d(bundle1, x1, derivative=True)
rel_l2, rel_h1 = relerr(u1, ref1["u_exact"], du1, ref1["grad_exact"])
print(f"  n_dofs={bundle1['n_dofs']} (expect {sel1['n_dofs']})  rel_l2={rel_l2:.4e} (selected.json {sel1['rel_l2']:.4e})"
      f"  rel_h1={rel_h1:.4e} (selected.json {sel1['rel_h1']:.4e})")
np.savez(CACHE / "section_1.npz", x=x1, u=u1, grad=du1, degree=4, n_dofs=bundle1["n_dofs"],
         description=sel1["description"])

# ---------------- Section 2: 2D, degree=4, 12x8 interface P4 ----------------
print("=== Section 2 ===")
sel2 = json.load(open(ROOT / "section_2" / "fem_baselines" / "selected.json"))["best_fem_h1_under_hybrid_budget"]
assert sel2["description"] == "12x8 interface P4" and sel2["degree"] == 4
nodes2, tris2, bm2 = F.make_unit_square_mesh(12, 8)
bundle2 = F.solve_pk_dirichlet(nodes2, tris2, bm2, F.f22_single, degree=4,
                                quadrature_order=10, A_func=F.A_one)
ref2 = np.load(next((ROOT / "section_2" / "s2_proj_h01_tensor_quadrature").glob("seed_*/final_solution.npz")))
pts2 = ref2["points"]
u2, g2 = F.eval_pk_solution(bundle2, pts2, return_gradient=True)
rel_l2, rel_h1 = relerr(u2, ref2["u_exact"], g2, ref2["grad_exact"])
print(f"  n_dofs={bundle2['n_dofs']} (expect {sel2['n_dofs']})  rel_l2={rel_l2:.4e} (selected.json {sel2['rel_l2']:.4e})"
      f"  rel_h1={rel_h1:.4e} (selected.json {sel2['rel_h1']:.4e})")
np.savez(CACHE / "section_2.npz", points=pts2, u=u2, grad=g2, degree=4, n_dofs=bundle2["n_dofs"],
         description=sel2["description"])

# ---------------- Section 3: 2D, degree=3, 14x12 interface P3, line source ----------------
print("=== Section 3 ===")
sel3 = json.load(open(ROOT / "section_3" / "fem_baselines" / "selected.json"))["best_fem_h1_under_hybrid_budget"]
assert sel3["description"] == "14x12 interface P3" and sel3["degree"] == 3
nodes3, tris3, bm3 = F.make_unit_square_mesh(14, 12)
bundle3 = F.solve_pk_dirichlet(nodes3, tris3, bm3, F.f24_reg_single, degree=3, A_func=F.A_one,
                                line_source=(F.kink_line_points, F.kink_line_weighted_density))
ref3 = np.load(next((ROOT / "section_3" / "s3_proj_eigen_green_quadrature").glob("seed_*/final_solution.npz")))
pts3 = ref3["points"]
u3, g3 = F.eval_pk_solution(bundle3, pts3, return_gradient=True)
rel_l2, rel_h1 = relerr(u3, ref3["u_exact"], g3, ref3["grad_exact"])
print(f"  n_dofs={bundle3['n_dofs']} (expect {sel3['n_dofs']})  rel_l2={rel_l2:.4e} (selected.json {sel3['rel_l2']:.4e})"
      f"  rel_h1={rel_h1:.4e} (selected.json {sel3['rel_h1']:.4e})")
np.savez(CACHE / "section_3.npz", points=pts3, u=u3, grad=g3, degree=3, n_dofs=bundle3["n_dofs"],
         description=sel3["description"])

# ---------------- Section 4: 2D, degree=2, power P2 nr=16 na=40 beta=2.5 ----------------
print("=== Section 4 ===")
sel4 = json.load(open(ROOT / "section_4" / "fem_baselines" / "selected.json"))["best_fem_h1_under_hybrid_budget"]
assert sel4["description"] == "power P2 nr=16 na=40 beta=2.5" and sel4["degree"] == 2
nodes4, tris4, bm4 = F.make_lshape_corner_mesh(n_radial=16, n_angular=40, beta=2.5, grading_mode="power")
bundle4 = F.solve_pk_dirichlet(nodes4, tris4, bm4, F.fL_single, degree=2, A_func=None)
ref4 = np.load(next((ROOT / "section_4" / "s4_proj_random_hats").glob("seed_*/final_solution.npz")))
pts4 = ref4["points"]
u4, g4 = F.eval_pk_solution(bundle4, pts4, return_gradient=True)
rel_l2, rel_h1 = relerr(u4, ref4["u_exact"], g4, ref4["grad_exact"])
print(f"  n_dofs={bundle4['n_dofs']} (expect {sel4['n_dofs']})  rel_l2={rel_l2:.4e} (selected.json {sel4['rel_l2']:.4e})"
      f"  rel_h1={rel_h1:.4e} (selected.json {sel4['rel_h1']:.4e})")
np.savez(CACHE / "section_4.npz", points=pts4, u=u4, grad=g4, degree=2, n_dofs=bundle4["n_dofs"],
         description=sel4["description"])

print("\nDONE -- standalone FEM fields cached under", CACHE.resolve())
