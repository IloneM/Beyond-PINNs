#!/usr/bin/env python
"""
[Copied from the research tree with minimal path/import adaptation only -- see docs/PROVENANCE.md.]
Splits the combined figures/petrov/appendix/meshes.pdf (4 horizontal panels:
MS/JF/LS/RC hybrid FE compensation spaces) into four separate publication
panels for LaTeX `subcaption` assembly.

Presentation-only: mesh-generation code and drawing logic are copied VERBATIM
from generate_petrov_figures.py (np_make_unit_square_mesh,
np_make_lshape_corner_mesh, and the four per-section drawing blocks) -- same
DOF counts, same colors, same highlighted lines/markers, same epsilon
annotation. Only removed: the internal "S1:/S2:/S3:/S4:" titles (replaced by
LaTeX subcaptions) and the small per-panel legends (per request, since the
subcaptions will explain the highlighted structures in prose). No scientific
content, mesh parameters, or data changed. Does not touch the standalone
FE-reference meshes, and does not regenerate/overwrite the old combined
meshes.pdf.

Output: figures/petrov/section_{1,2,3,4}_{ms,jf,ls,rc}_compensation_mesh.pdf/.png
"""
from pathlib import Path
import numpy as np
import os
os.environ.pop("MPLBACKEND", None)  # must happen BEFORE importing matplotlib: an inherited
# invalid value (e.g. Jupyter's 'module://matplotlib_inline.backend_inline') crashes
# matplotlib's own __init__.py at import time, before any mpl.use() call could help.
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = Path("figures/petrov")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OTHER_COLOR = {
    "deep_ritz": "#1b9e77",
    "deep_ritz_l2_metric": "#a6d854",
    "strong_pinn": "#e6ab02",
    "exact_regression": "#666666",
}

mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": "0.3",
    "axes.grid": True, "grid.color": "0.85", "grid.linewidth": 0.6,
    "font.size": 10.5, "axes.labelsize": 11.5, "legend.fontsize": 9.5,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.6,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

# ------------------------------------------------------------------
# Mesh generators -- verbatim from generate_petrov_figures.py
# ------------------------------------------------------------------
def np_make_unit_square_mesh(nx, ny):
    xs = np.linspace(0.0, 1.0, nx + 1)
    ys = np.linspace(0.0, 1.0, ny + 1)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    nodes = np.stack([X.ravel(), Y.ravel()], 1)
    Npy = ny + 1
    ix, iy = np.mgrid[0:nx, 0:ny]; ix, iy = ix.ravel(), iy.ravel()
    n00 = ix * Npy + iy; n10 = (ix + 1) * Npy + iy
    n11 = (ix + 1) * Npy + (iy + 1); n01 = ix * Npy + (iy + 1)
    even = ((ix + iy) % 2 == 0)
    t1 = np.where(even[:, None], np.stack([n00, n10, n11], 1), np.stack([n00, n10, n01], 1))
    t2 = np.where(even[:, None], np.stack([n00, n11, n01], 1), np.stack([n10, n11, n01], 1))
    tris = np.concatenate([t1, t2], 0)
    return nodes, tris

def np_make_lshape_corner_mesh(n_radial=14, n_angular=32, beta=2.0, R_out=1.0):
    seg = [0.0, np.pi / 4, 3 * np.pi / 4, 5 * np.pi / 4, 1.5 * np.pi]
    lens = np.diff(seg)
    counts = np.maximum(1, np.round(n_angular * lens / lens.sum()).astype(int))
    thetas = np.concatenate([np.linspace(seg[k], seg[k + 1], counts[k] + 1)[:-1]
                             for k in range(4)] + [np.array([1.5 * np.pi])])
    Nth, M = thetas.size, n_radial
    j = np.arange(0, M + 1)
    t = (j / M) ** beta
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
    return nodes, np.array(tris)

def save(fig, stem):
    out = OUT_DIR / stem
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    fig.savefig(f"{out}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}.pdf and .png")

# ==================================================================
# MS (section_1): 16-element P1 mesh on [0,1]
# ==================================================================
fig, ax = plt.subplots(figsize=(4.3, 2.1))
n_elem = 16
nodes1d = np.linspace(0.0, 1.0, n_elem + 1)
ax.plot(nodes1d, np.zeros_like(nodes1d), color="0.5", lw=1.2, zorder=1)
ax.scatter(nodes1d[[0, -1]], [0, 0], color="black", marker="s", s=45, zorder=3)
ax.scatter(nodes1d[1:-1], np.zeros(n_elem - 1), color=OTHER_COLOR["deep_ritz"], marker="o", s=45, zorder=3)
eps = 1.0 / 16.0
ax.annotate("", xy=(eps, 0.15), xytext=(0.0, 0.15), arrowprops=dict(arrowstyle="<->"))
ax.text(eps / 2, 0.19, r"$\epsilon=1/16$", ha="center", fontsize=10)
ax.set_ylim(-0.3, 0.4); ax.set_xlim(-0.03, 1.03)
ax.set_yticks([]); ax.set_xlabel("$x$")
fig.tight_layout()
save(fig, "section_1_ms_compensation_mesh")

# ==================================================================
# JF (section_2): 16x12 triangular mesh, highlight interface x=1/2
# ==================================================================
fig, ax = plt.subplots(figsize=(4.2, 4.2))
nodes22, tris22 = np_make_unit_square_mesh(16, 12)
ax.triplot(nodes22[:, 0], nodes22[:, 1], tris22, color="0.5", lw=0.6)
ax.axvline(0.5, color=OTHER_COLOR["deep_ritz"], lw=2.2)
ax.set_aspect("equal"); ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
fig.tight_layout()
save(fig, "section_2_jf_compensation_mesh")

# ==================================================================
# LS (section_3): 16x12 triangular mesh, highlight loaded line x=1/2
# ==================================================================
fig, ax = plt.subplots(figsize=(4.2, 4.2))
nodes24, tris24 = np_make_unit_square_mesh(16, 12)
ax.triplot(nodes24[:, 0], nodes24[:, 1], tris24, color="0.5", lw=0.6)
ax.axvline(0.5, color="#d95f02", lw=2.6)
ax.set_aspect("equal"); ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
fig.tight_layout()
save(fig, "section_3_ls_compensation_mesh")

# ==================================================================
# RC (section_4): corner-graded polar mesh, n_radial=14, n_angular=32, beta=2
# ==================================================================
fig, ax = plt.subplots(figsize=(4.2, 4.2))
nodesP, trisP = np_make_lshape_corner_mesh(n_radial=14, n_angular=32, beta=2.0)
ax.triplot(nodesP[:, 0], nodesP[:, 1], trisP, color="0.5", lw=0.5)
ax.scatter([0.0], [0.0], color="#d95f02", marker="*", s=160, zorder=5)
ax.add_patch(plt.Rectangle((0, -1.05), 1.05, 1.05, color="0.9", zorder=0))
ax.set_aspect("equal"); ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05)
fig.tight_layout()
save(fig, "section_4_rc_compensation_mesh")

print("\ndone -- old figures/petrov/appendix/meshes.pdf left untouched.")
