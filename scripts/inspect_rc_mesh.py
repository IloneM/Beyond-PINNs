#!/usr/bin/env python
"""
[Copied from the research tree with minimal path/import adaptation only -- see docs/PROVENANCE.md.]
Read-only inspection of the RC hybrid FE-compensator mesh (NOT the
standalone-FE-reference mesh, which is a different, independent construction).

Exact source: femennstein-petrov-galerkin-experiments-fixed.py, lines
~3951-3962 ("polar compensator (Ex14/18 config, 3-pt rule)"):

    _ndP, _elP, _bmP = make_lshape_corner_mesh(n_radial=14, n_angular=32, beta=2.0)
    freeP = jnp.where(~jnp.array(_bmP))[0]   # interior (free) DOFs
    # degree = 1 (P1), grading_mode="power" (function default), q=0.7 (unused for power)

make_lshape_corner_mesh itself: petrov_fem_utils.py lines 109-142 (verbatim copy of
the same function used in the main script).

No training, no GPU, no plotting redesign -- pure numpy mesh-geometry inspection.
"""
import numpy as np
import os
os.environ.pop("MPLBACKEND", None)  # must happen BEFORE importing matplotlib -- see other scripts' comment
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

def make_lshape_corner_mesh(n_radial=16, n_angular=48, beta=2.0, R_out=1.0,
                             grading_mode="power", q=0.7):
    seg = [0.0, np.pi / 4, 3 * np.pi / 4, 5 * np.pi / 4, 1.5 * np.pi]
    lens = np.diff(seg)
    counts = np.maximum(1, np.round(n_angular * lens / lens.sum()).astype(int))
    thetas = np.concatenate([np.linspace(seg[k], seg[k + 1], counts[k] + 1)[:-1]
                             for k in range(4)] + [np.array([1.5 * np.pi])])
    Nth, M = thetas.size, n_radial
    j = np.arange(0, M + 1)
    if grading_mode == "power":
        t = (j / M) ** beta
    elif grading_mode == "geometric":
        q = float(q)
        t = (q ** (M - j) - q ** M) / (1.0 - q ** M)
    else:
        raise ValueError(grading_mode)
    c, s = np.cos(thetas), np.sin(thetas)
    rho = R_out / np.maximum(np.abs(c), np.abs(s))
    ring = np.stack([t[1:, None] * rho[None, :] * c[None, :],
                     t[1:, None] * rho[None, :] * s[None, :]], axis=-1)
    nodes = np.concatenate([np.array([[0.0, 0.0]]), ring.reshape(-1, 2)], axis=0)
    def idx(i, j): return 1 + i * Nth + j
    tris = [[0, idx(0, j), idx(0, j + 1)] for j in range(Nth - 1)]
    for i in range(M - 1):
        for j in range(Nth - 1):
            a, b, cc, d = idx(i, j), idx(i, j + 1), idx(i + 1, j + 1), idx(i + 1, j)
            tris += [[a, b, cc], [a, cc, d]]
    elements = np.array(tris)
    bmask = np.zeros(nodes.shape[0], dtype=bool); bmask[0] = True
    for i in range(M):
        for j in range(Nth):
            if i == M - 1 or j == 0 or j == Nth - 1:
                bmask[idx(i, j)] = True
    return nodes, elements, bmask, thetas, t

# ==================================================================
# Section 1: exact HYBRID compensator mesh (n_radial=14, n_angular=32, beta=2.0)
# ==================================================================
N_RADIAL, N_ANGULAR, BETA = 14, 32, 2.0
nodes, elements, bmask, thetas, t = make_lshape_corner_mesh(N_RADIAL, N_ANGULAR, BETA)
free = np.where(~bmask)[0]

print("=== A. Exact mesh parameters (hybrid FE compensator, section_4/RC) ===")
print("source: femennstein-petrov-galerkin-experiments-fixed.py lines 3955-3962")
print("        make_lshape_corner_mesh() defined at line 3819 (identical copy in petrov_fem_utils.py:109)")
print(f"call: make_lshape_corner_mesh(n_radial={N_RADIAL}, n_angular={N_ANGULAR}, beta={BETA})")
print("grading_mode='power' (function default, not overridden), q=0.7 (unused for power mode)")
print("polynomial degree: P1 (piecewise-linear), per explicit code comment")
print(f"total nodes: {nodes.shape[0]}, total elements: {elements.shape[0]}")
print(f"free (interior, non-Dirichlet) DOFs: {free.shape[0]}  <- must match FE_dof_count=403")
print(f"angular samples Nth = thetas.size = {thetas.size}")

print("\n=== B. Radial layers along the positive x-axis (theta=0, rho=1) ===")
print("formula: r_j = (j/n_radial)^beta * rho(theta), rho(0)=1")
for j in range(0, min(N_RADIAL, 14) + 1):
    r_formula = (j / N_RADIAL) ** BETA
    print(f"  r_{j:2d} = ({j}/{N_RADIAL})^{BETA} = {r_formula:.6f}")

# cross-check against the actual generated node coordinates along theta=0
theta0_idx = np.argmin(np.abs(thetas - 0.0))
print(f"\ncross-check from actual generated nodes at theta index {theta0_idx} (theta={thetas[theta0_idx]:.4f}):")
def idx(i, jx, Nth=thetas.size): return 1 + i * Nth + jx
for i in range(0, min(N_RADIAL, 10)):
    node = nodes[idx(i, theta0_idx)]
    r_actual = np.hypot(*node)
    print(f"  ring i={i:2d}: node={node}, r={r_actual:.6f}")

print("\n=== C. Local element size estimate near specific radii (along theta=0 direction) ===")
for r_target in (0.35, 0.12, 0.05):
    # find the ring whose radius (theta=0) is closest to r_target
    ring_r = np.array([(i / N_RADIAL) ** BETA for i in range(N_RADIAL + 1)])
    i_ring = int(np.argmin(np.abs(ring_r - r_target)))
    dr_in = ring_r[i_ring] - ring_r[max(i_ring - 1, 0)]
    dr_out = ring_r[min(i_ring + 1, N_RADIAL)] - ring_r[i_ring]
    # angular arc length at that radius: r * dtheta, dtheta ~ average angular spacing
    dtheta = (1.5 * np.pi) / N_ANGULAR
    arc = ring_r[i_ring] * dtheta
    print(f"  near r~{r_target}: closest ring i={i_ring} (r={ring_r[i_ring]:.4f}), "
          f"radial step in={dr_in:.5f} out={dr_out:.5f}, angular arc~{arc:.5f}")

# smallest element adjacent to the corner: ring 0 (the corner) to ring 1
r1 = (1 / N_RADIAL) ** BETA
dtheta = (1.5 * np.pi) / N_ANGULAR
print(f"\nsmallest elements (corner triangle fan, ring 0->1): radial extent 0 -> {r1:.6f}, "
      f"angular arc at r1 ~ {r1*dtheta:.6f}")

print("\n=== D. Node/element counts inside candidate zoom windows ===")
for hw in (0.35, 0.12, 0.05):
    in_win = (np.abs(nodes[:, 0]) <= hw) & (np.abs(nodes[:, 1]) <= hw)
    n_nodes_in = int(in_win.sum())
    # elements intersecting: any of its 3 vertices inside window (approximate but reasonable)
    elem_any_in = in_win[elements].any(axis=1)
    n_elem_in = int(elem_any_in.sum())
    # distinct radial rings represented (ring index 0..N_RADIAL, node 0 is the corner itself)
    r_of_node = np.hypot(nodes[:, 0], nodes[:, 1])
    # assign each node to nearest ring index via r_formula
    ring_r = np.array([(i / N_RADIAL) ** BETA for i in range(N_RADIAL + 1)])
    # only consider interior nodes actually in the window (excluding corner node 0 duplicate count)
    rings_in_window = set()
    for ridx, rr in enumerate(r_of_node[in_win]):
        # nearest ring by radius alone (rough; true ring also depends on angle-dependent rho)
        pass
    # More directly: recover ring index i from node ordering (node k>0 -> ring i = (k-1)//Nth)
    Nth = thetas.size
    ring_idx_of_node = np.full(nodes.shape[0], -1, dtype=int)
    ring_idx_of_node[0] = 0
    for k in range(1, nodes.shape[0]):
        ring_idx_of_node[k] = 1 + (k - 1) // Nth
    distinct_rings = sorted(set(ring_idx_of_node[in_win].tolist()))
    n_free_in = int(np.isin(np.arange(nodes.shape[0]), free)[in_win].sum())
    print(f"  window +/-{hw}: nodes={n_nodes_in} (of which free/basis-supporting={n_free_in}), "
          f"elements(any vertex inside)={n_elem_in}, distinct radial rings={distinct_rings}")

# ==================================================================
# E. Diagnostic plot
# ==================================================================
OUT_DIR = Path("figures/petrov/mesh_diagnostic")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def plot_mesh(ax, half_window=None):
    tris = elements
    for tri in tris:
        pts = nodes[tri]
        if half_window is not None:
            if not np.any((np.abs(pts[:, 0]) <= half_window) & (np.abs(pts[:, 1]) <= half_window)):
                continue
        loop = np.vstack([pts, pts[:1]])
        ax.plot(loop[:, 0], loop[:, 1], color="0.35", lw=0.4)
    ax.plot(0, 0, marker="*", color="red", ms=10, mec="white", mew=0.5, zorder=5)
    ax.set_aspect("equal")
    if half_window is not None:
        ax.set_xlim(-half_window, half_window); ax.set_ylim(-half_window, half_window)
    else:
        ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05)
    ax.add_patch(plt.Rectangle((0, -1.05), 1.1, 1.05, color="0.85", zorder=0))

fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
plot_mesh(axes[0], None)
axes[0].set_title(f"Full RC compensator mesh\n(n_radial={N_RADIAL}, n_angular={N_ANGULAR}, beta={BETA}, P1)")
plot_mesh(axes[1], 0.15)
axes[1].set_title("Zoom $\\pm$0.15")
plot_mesh(axes[2], 0.05)
axes[2].set_title("Zoom $\\pm$0.05")
fig.tight_layout()
out = OUT_DIR / "rc_hybrid_compensator_mesh_diagnostic"
fig.savefig(f"{out}.png", dpi=180, bbox_inches="tight")
print(f"\nsaved {out}.png")
