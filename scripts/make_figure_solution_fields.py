#!/usr/bin/env python
"""
[Copied from the research tree with minimal path/import adaptation only -- see docs/PROVENANCE.md.]
Appendix solution/gradient/cross-section figures for the Petrov-Galerkin
suite. READS the already-computed final_solution.npz per representative
seed, plus the standalone-FEM cache built by compute_standalone_fem.py
(deterministic FEM solve, no NN training). No neural optimization here.
"""
import json, glob
from pathlib import Path
import numpy as np
import os
os.environ.pop("MPLBACKEND", None)  # must happen BEFORE importing matplotlib: an inherited
# invalid value (e.g. Jupyter's 'module://matplotlib_inline.backend_inline') crashes
# matplotlib's own __init__.py at import time, before any mpl.use() call could help.
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

ROOT = Path("results/reference/general")
CACHE = Path("results/reference/fem")
if not ROOT.exists() or not CACHE.exists():
    raise SystemExit(
        "Reference results are not installed.\n\n"
        "Download/extract the Beyond PINNs reference-results archive into:\n"
        "    results/reference/\n\n"
        f"(expected to find data at {ROOT.resolve()} and {CACHE.resolve()})\n"
        "See docs/REPRODUCIBILITY.md or run scripts/download_reference_results.sh."
    )
APP_DIR = Path("figures/petrov/appendix")
APP_DIR.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": "0.3",
    "axes.grid": True, "grid.color": "0.85", "grid.linewidth": 0.5,
    "font.size": 10, "axes.labelsize": 10.5, "legend.fontsize": 9,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.4,
})

PRINCIPAL = {
    "section_1": "s1_proj_h01_green_quadrature",
    "section_2": "s2_proj_h01_tensor_quadrature",
    "section_3": "s3_proj_eigen_green_quadrature",
    "section_4": "s4_proj_random_hats",
}

def representative_seed(section, run):
    seeds = {}
    for p in sorted(glob.glob(str(ROOT / section / run / "seed_*" / "history.npz"))):
        seed = int(Path(p).parent.name.split("_")[1])
        seeds[seed] = float(np.load(p)["relative_h1"][-1])
    med = np.median(list(seeds.values()))
    return min(seeds, key=lambda s: abs(seeds[s] - med)), seeds

REP_SEED = {}
for section, run in PRINCIPAL.items():
    seed, seeds = representative_seed(section, run)
    REP_SEED[section] = seed
    print(f"{section}: representative seed = {seed} (median H1 over {sorted(seeds)} = {np.median(list(seeds.values())):.3e})")

FLOOR_REL = 1e-12  # multiplied by max|u_exact| per instructions

def load_final(section, run, seed):
    return np.load(ROOT / section / run / f"seed_{seed:03d}" / "final_solution.npz")

# ==================================================================
# Section 1: solution + gradient component figures (1D)
# ==================================================================
run1 = PRINCIPAL["section_1"]; seed1 = REP_SEED["section_1"]
d1 = load_final("section_1", run1, seed1)
fem1 = np.load(CACHE / "section_1.npz")
x1 = d1["x"]
assert np.allclose(x1, fem1["x"])

rows = [
    ("Hybrid total", d1["u_total"], d1["grad_total"]),
    ("NN component alone", d1["u_nn"], d1["grad_nn"]),
    ("FEM compensator alone", d1["u_fem"], d1["grad_fem"]),
    ("Standalone near-budget FEM", fem1["u"], fem1["grad"]),
]
floor1 = max(1e-16, 1e-12 * np.max(np.abs(d1["u_exact"])))

fig, axes = plt.subplots(4, 2, figsize=(11, 13))
for i, (label, u, _) in enumerate(rows):
    axL, axR = axes[i, 0], axes[i, 1]
    axL.plot(x1, d1["u_exact"], color="black", lw=1.3, label="exact" if i == 0 else None)
    axL.plot(x1, u, color="#d95f02", lw=1.3, label=label if i == 0 else None)
    axL.set_ylabel(label, fontsize=9.5)
    err = np.abs(u - d1["u_exact"])
    axR.plot(x1, np.clip(err, floor1, None), color="#d95f02", lw=1.1)
    axR.set_yscale("log")
    if i == 0:
        axL.legend(fontsize=8, loc="best")
        axL.set_title("value", fontsize=10.5)
        axR.set_title(r"absolute error $|u_{\rm approx}-u_{\rm exact}|$", fontsize=10.5)
    if i == 3:
        axL.set_xlabel("$x$"); axR.set_xlabel("$x$")
fig.suptitle("")
fig.tight_layout()
fig.savefig(APP_DIR / "section_1_solution_components.pdf", bbox_inches="tight")
fig.savefig(APP_DIR / "section_1_solution_components.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print("saved", APP_DIR / "section_1_solution_components.pdf", "and .png")

fig, axes = plt.subplots(4, 2, figsize=(11, 13))
gfloor1 = max(1e-16, 1e-12 * np.max(np.abs(d1["grad_exact"])))
for i, (label, _, g) in enumerate(rows):
    axL, axR = axes[i, 0], axes[i, 1]
    axL.plot(x1, d1["grad_exact"], color="black", lw=1.3, label="exact" if i == 0 else None)
    axL.plot(x1, g, color="#d95f02", lw=1.3, label=label if i == 0 else None)
    axL.set_ylabel(label, fontsize=9.5)
    gerr = np.abs(g - d1["grad_exact"])
    axR.plot(x1, np.clip(gerr, gfloor1, None), color="#d95f02", lw=1.1)
    axR.set_yscale("log")
    if i == 0:
        axL.legend(fontsize=8, loc="best")
        axL.set_title("derivative", fontsize=10.5)
        axR.set_title(r"absolute error $|u'_{\rm approx}-u'_{\rm exact}|$", fontsize=10.5)
    if i == 3:
        axL.set_xlabel("$x$"); axR.set_xlabel("$x$")
fig.tight_layout()
fig.savefig(APP_DIR / "section_1_gradient_components.pdf", bbox_inches="tight")
fig.savefig(APP_DIR / "section_1_gradient_components.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print("saved", APP_DIR / "section_1_gradient_components.pdf", "and .png")

# ==================================================================
# Sections 2-4: 2D solution-field figures (4 rows x 3 cols)
# ==================================================================
def to_grid(points, values, xs, ys):
    """Map scattered (points,values) onto a regular xs x ys grid, NaN where missing."""
    nx, ny = len(xs), len(ys)
    ix = np.searchsorted(xs, points[:, 0])
    iy = np.searchsorted(ys, points[:, 1])
    ix = np.clip(ix, 0, nx - 1); iy = np.clip(iy, 0, ny - 1)
    grid = np.full((ny, nx), np.nan)
    grid[iy, ix] = values
    return grid

SECTION2D = {
    "section_2": dict(run=PRINCIPAL["section_2"], seed=REP_SEED["section_2"],
                       fem_cache="section_2.npz", mark=lambda ax: ax.axvline(0.5, color="white", lw=1.0, ls="--")),
    "section_3": dict(run=PRINCIPAL["section_3"], seed=REP_SEED["section_3"],
                       fem_cache="section_3.npz", mark=lambda ax: ax.axvline(0.5, color="white", lw=1.2, ls="--")),
    "section_4": dict(run=PRINCIPAL["section_4"], seed=REP_SEED["section_4"],
                       fem_cache="section_4.npz", mark=lambda ax: ax.plot(0, 0, marker="*", color="red", ms=10)),
}

ROW_LABELS = ["Hybrid total", "NN component alone", "FEM compensator alone", "Standalone near-budget FEM"]

for section, cfg in SECTION2D.items():
    d = load_final(section, cfg["run"], cfg["seed"])
    fem = np.load(CACHE / cfg["fem_cache"])
    pts = d["points"]
    xs = np.unique(pts[:, 0]); ys = np.unique(pts[:, 1])
    u_exact_g = to_grid(pts, d["u_exact"], xs, ys)
    rows_u = [d["u_total"], d["u_nn"], d["u_fem"], fem["u"]]
    floor = max(1e-16, 1e-12 * np.nanmax(np.abs(u_exact_g)))
    vmin_all, vmax_all = np.nanmin(u_exact_g), np.nanmax(u_exact_g)
    for u in rows_u:
        g = to_grid(pts, u, xs, ys)
        vmin_all = min(vmin_all, np.nanmin(g)); vmax_all = max(vmax_all, np.nanmax(g))

    fig, axes = plt.subplots(4, 3, figsize=(12.5, 15.5))
    cmap = plt.get_cmap("viridis").copy(); cmap.set_bad("0.92")
    ecmap = plt.get_cmap("inferno").copy(); ecmap.set_bad("0.92")
    extent = [xs.min(), xs.max(), ys.min(), ys.max()]
    im_val = None
    for i, (label, u) in enumerate(zip(ROW_LABELS, rows_u)):
        g = to_grid(pts, u, xs, ys)
        err = np.abs(g - u_exact_g)
        axE, axA, axR = axes[i, 0], axes[i, 1], axes[i, 2]
        if i == 0:
            im_val = axE.imshow(u_exact_g, origin="lower", extent=extent, cmap=cmap, vmin=vmin_all, vmax=vmax_all)
        else:
            axE.imshow(u_exact_g, origin="lower", extent=extent, cmap=cmap, vmin=vmin_all, vmax=vmax_all)
        axA.imshow(g, origin="lower", extent=extent, cmap=cmap, vmin=vmin_all, vmax=vmax_all)
        # Each row gets its OWN error colorbar: rows span many orders of
        # magnitude (hybrid vs. FEM-alone vs. standalone FEM), so a single
        # shared error scale would mislabel every row but the last.
        row_vmax = max(floor * 10, np.nanmax(err))
        im_err_row = axR.imshow(np.clip(err, floor, None), origin="lower", extent=extent, cmap=ecmap,
                                 norm=LogNorm(vmin=floor, vmax=row_vmax))
        fig.colorbar(im_err_row, ax=axR, shrink=0.85, pad=0.02, label="abs. error" if i == 0 else None)
        for ax in (axE, axA, axR):
            ax.set_aspect("equal"); cfg["mark"](ax)
            ax.set_xticks([]); ax.set_yticks([])
        axE.set_ylabel(label, fontsize=10)
        if i == 0:
            axE.set_title("exact", fontsize=10.5); axA.set_title("approximation", fontsize=10.5)
            axR.set_title(r"$|u_{\rm approx}-u_{\rm exact}|$ (log, per-row scale)", fontsize=10.5)
    fig.colorbar(im_val, ax=axes[:, :2].ravel().tolist(), shrink=0.6, label="$u$")
    out = APP_DIR / f"{section}_solution_components"
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    fig.savefig(f"{out}.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("saved", f"{out}.pdf", "and .png")

# ==================================================================
# Cross-section diagnostics: Section 2 (y=1/2), Section 3 (y=1/2, kink)
# ==================================================================
CROSS_COMPONENTS = [("exact", "u_exact", "grad_exact", "black", "-"),
                     ("hybrid total", "u_total", "grad_total", "#d95f02", "-"),
                     ("NN component", "u_nn", "grad_nn", "#7570b3", "--"),
                     ("FEM compensator", "u_fem", "grad_fem", "#1b9e77", ":"),
                     ("standalone FEM", None, None, "#e7298a", "-.")]

for section, out_name, kink_note in (("section_2", "section_2_cross_sections", False),
                                      ("section_3", "section_3_cross_sections", True)):
    cfg = SECTION2D[section]
    d = load_final(section, cfg["run"], cfg["seed"])
    fem = np.load(CACHE / cfg["fem_cache"])
    pts = d["points"]
    xs = np.unique(pts[:, 0]); ys = np.unique(pts[:, 1])
    iy_half = np.argmin(np.abs(ys - 0.5))

    def row_of(values):
        g = to_grid(pts, values, xs, ys)
        return g[iy_half, :]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axU, axDU, axUerr, axDUerr = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    u_exact_row = row_of(d["u_exact"]); gx_exact_row = row_of(d["grad_exact"][:, 0])
    for label, ukey, gkey, color, ls in CROSS_COMPONENTS:
        if ukey is None:
            u_row = row_of(fem["u"]); gx_row = row_of(fem["grad"][:, 0])
        else:
            u_row = row_of(d[ukey]); gx_row = row_of(d[gkey][:, 0])
        axU.plot(xs, u_row, color=color, ls=ls, lw=1.4, label=label)
        axDU.plot(xs, gx_row, color=color, ls=ls, lw=1.4, label=label)
        if label != "exact":
            floor = max(1e-16, 1e-12 * np.nanmax(np.abs(u_exact_row)))
            gfloor = max(1e-16, 1e-12 * np.nanmax(np.abs(gx_exact_row)))
            axUerr.plot(xs, np.clip(np.abs(u_row - u_exact_row), floor, None), color=color, ls=ls, lw=1.2)
            axDUerr.plot(xs, np.clip(np.abs(gx_row - gx_exact_row), gfloor, None), color=color, ls=ls, lw=1.2)
    for ax in (axUerr, axDUerr):
        ax.set_yscale("log")
    if kink_note:
        for ax in axes.ravel():
            ax.axvline(0.5, color="0.5", lw=1.0, ls=":", zorder=0)
        axDU.text(0.5, 0.02, "derivative jump at $x=1/2$ (kink, not smoothed)",
                   transform=axDU.transAxes, ha="center", fontsize=8, style="italic")
    axU.set_ylabel("$u(x, y{=}1/2)$"); axDU.set_ylabel(r"$\partial u/\partial x\ (x, y{=}1/2)$")
    axUerr.set_ylabel("abs. error in $u$"); axDUerr.set_ylabel(r"abs. error in $\partial u/\partial x$")
    for ax in axes[1, :]:
        ax.set_xlabel("$x$")
    axU.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(APP_DIR / f"{out_name}.pdf", bbox_inches="tight")
    fig.savefig(APP_DIR / f"{out_name}.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("saved", APP_DIR / f"{out_name}.pdf", "and .png")

# ==================================================================
# Section 4: corner radial profiles (never interpolate through the
# removed quadrant -- rays are chosen strictly within the admissible
# 270-degree sector)
# ==================================================================
from scipy.interpolate import griddata

cfg = SECTION2D["section_4"]
d = load_final("section_4", cfg["run"], cfg["seed"])
fem = np.load(CACHE / cfg["fem_cache"])
pts = d["points"]
radii = np.linspace(0.03, 0.95, 60)
ANGLES_DEG = [45, 135, 225]  # admissible rays strictly inside the 270-degree sector
fig, axes = plt.subplots(2, len(ANGLES_DEG), figsize=(5 * len(ANGLES_DEG), 8))
for col, ang in enumerate(ANGLES_DEG):
    th = np.deg2rad(ang)
    ray = np.stack([radii * np.cos(th), radii * np.sin(th)], axis=1)
    def interp(values):
        return griddata(pts, values, ray, method="linear")
    u_exact_r = interp(d["u_exact"])
    g_exact_r = interp(d["grad_exact"])
    gmag_exact_r = np.linalg.norm(g_exact_r, axis=1)
    axU, axG = axes[0, col], axes[1, col]
    for label, ukey, gkey, color, ls in CROSS_COMPONENTS:
        if ukey is None:
            u_r = interp(fem["u"]); g_r = interp(fem["grad"])
        else:
            u_r = interp(d[ukey]); g_r = interp(d[gkey])
        gmag_r = np.linalg.norm(g_r, axis=1)
        axU.plot(radii, u_r, color=color, ls=ls, lw=1.3, label=label if col == 0 else None)
        axG.plot(radii, gmag_r, color=color, ls=ls, lw=1.3, label=label if col == 0 else None)
    axU.set_title(rf"$\theta={ang}^\circ$", fontsize=10.5)
    axU.set_xlabel("$r$"); axG.set_xlabel("$r$")
    if col == 0:
        axU.set_ylabel("$u(r,\\theta)$"); axG.set_ylabel(r"$|\nabla u|(r,\theta)$")
        axU.legend(fontsize=7.5, loc="best")
fig.suptitle("Corner radial profiles (never crossing the removed quadrant)", fontsize=11, y=1.01)
fig.tight_layout()
fig.savefig(APP_DIR / "section_4_corner_profiles.pdf", bbox_inches="tight")
fig.savefig(APP_DIR / "section_4_corner_profiles.png", dpi=170, bbox_inches="tight")
plt.close(fig)
print("saved", APP_DIR / "section_4_corner_profiles.pdf", "and .png")
