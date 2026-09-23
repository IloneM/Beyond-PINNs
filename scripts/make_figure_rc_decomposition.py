#!/usr/bin/env python
"""
[Copied from the research tree with minimal path/import adaptation only -- see docs/PROVENANCE.md.]
New, simplified RC (section_4) structural decomposition figure.

READS-ONLY from petrov_galerkin_results/section_4/ (final_solution.npz for the
representative seed). No neural training, no FEM sweep, no seed selection
beyond the exact pre-existing representative_seed() logic (copied verbatim
from generate_petrov_solution_figures.py) applied to the unchanged saved
history.npz files. Does not touch the standalone FE cache, the MS figures, or
the old 4x3 figures/petrov/appendix/section_4_solution_components.pdf.

Replaces the "too many panels" old RC figure, for the main text, with ONLY the
three structural fields:
    exact solution u
    hybrid approximation v_{theta,n}   (u_total)
    FE compensator w_n                  (u_fem)

as THREE SEPARATE panel files (for LaTeX subcaption), plus a combined preview.

Also runs a short, lightweight diagnostic (no new heavy computation) on
whether |w_n| is concentrated near the reentrant corner (at (0,0), matching
the marker convention already used for section_4 in
generate_petrov_solution_figures.py's SECTION2D["section_4"]["mark"]).

Output (figures/petrov/appendix/):
    section_4_rc_exact.pdf/.png
    section_4_rc_hybrid.pdf/.png
    section_4_rc_fe_compensator.pdf/.png
    section_4_rc_decomposition_preview.png   (combined 1x3, quick inspection only)

Run from the directory containing petrov_galerkin_results/:
    python generate_section4_rc_decomposition.py
"""
import glob
from pathlib import Path
import numpy as np
import os
os.environ.pop("MPLBACKEND", None)  # must happen BEFORE importing matplotlib: an inherited
# invalid value (e.g. Jupyter's 'module://matplotlib_inline.backend_inline') crashes
# matplotlib's own __init__.py at import time, before any mpl.use() call could help.
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("results/reference/general")
APP_DIR = Path("figures/petrov/appendix")
if not ROOT.exists():
    raise SystemExit(
        "Reference results are not installed.\n\n"
        "Download/extract the Beyond PINNs reference-results archive into:\n"
        "    results/reference/\n\n"
        f"(expected to find data at {ROOT.resolve()})\n"
        "See docs/REPRODUCIBILITY.md or run scripts/download_reference_results.sh."
    )
APP_DIR.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": "0.3",
    "font.size": 10, "axes.labelsize": 10.5, "legend.fontsize": 9,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

SECTION = "section_4"
RUN = "s4_proj_random_hats"  # principal RC hybrid method, unchanged (matches
                              # both PRINCIPAL["section_4"] in
                              # generate_petrov_solution_figures.py and the
                              # random-hats pair used in the manuscript's main
                              # pure-vs-hybrid accuracy table)

# ------------------------------------------------------------------
# Representative-seed selection -- copied verbatim, unmodified.
# ------------------------------------------------------------------
def representative_seed(section, run):
    seeds = {}
    for p in sorted(glob.glob(str(ROOT / section / run / "seed_*" / "history.npz"))):
        seed = int(Path(p).parent.name.split("_")[1])
        seeds[seed] = float(np.load(p)["relative_h1"][-1])
    med = np.median(list(seeds.values()))
    return min(seeds, key=lambda s: abs(seeds[s] - med)), seeds

seed, seeds = representative_seed(SECTION, RUN)
print(f"{SECTION}/{RUN}: representative seed = {seed}")
print(f"  final relative_h1 per seed: {dict(sorted(seeds.items()))}")
print(f"  median relative_h1 over {len(seeds)} seeds = {np.median(list(seeds.values())):.6e}")
print(f"  seed {seed} relative_h1 = {seeds[seed]:.6e}  (closest to median)")

final_path = ROOT / SECTION / RUN / f"seed_{seed:03d}" / "final_solution.npz"
if not final_path.exists():
    raise SystemExit(f"Expected file missing, stopping rather than improvising: {final_path}")
d = np.load(final_path)

pts = d["points"]
print(f"\ndomain bounding box: x in [{pts[:,0].min():.4f}, {pts[:,0].max():.4f}], "
      f"y in [{pts[:,1].min():.4f}, {pts[:,1].max():.4f}], n_points={len(pts)}")

def to_grid(points, values, xs, ys):
    nx, ny = len(xs), len(ys)
    ix = np.clip(np.searchsorted(xs, points[:, 0]), 0, nx - 1)
    iy = np.clip(np.searchsorted(ys, points[:, 1]), 0, ny - 1)
    grid = np.full((ny, nx), np.nan)
    grid[iy, ix] = values
    return grid

xs = np.unique(pts[:, 0]); ys = np.unique(pts[:, 1])
extent = [xs.min(), xs.max(), ys.min(), ys.max()]

u_exact_g = to_grid(pts, d["u_exact"], xs, ys)
u_total_g = to_grid(pts, d["u_total"], xs, ys)
w_n_g = to_grid(pts, d["u_fem"], xs, ys)

# ==================================================================
# C. Lightweight FE-signal diagnostic (no new heavy computation)
# ==================================================================
r = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
w_n = d["u_fem"]
abs_w = np.abs(w_n)
i_max = int(np.argmax(abs_w))
r_near = 0.15 * (extent[1] - extent[0])  # "near the corner" radius, a modest
                                          # fraction of the domain size, used
                                          # only for this diagnostic
near_mask = r <= r_near
far_mask = ~near_mask

print("\n--- FE-compensator (w_n) localization diagnostic ---")
print(f"  max|w_n| = {abs_w.max():.6e} at point ({pts[i_max,0]:.4f}, {pts[i_max,1]:.4f}), "
      f"distance from corner (0,0) = {r[i_max]:.4f}")
print(f"  mean|w_n| overall           = {abs_w.mean():.6e}")
print(f"  mean|w_n| within r<={r_near:.4f} of corner (n={near_mask.sum()}) = {abs_w[near_mask].mean():.6e}")
print(f"  mean|w_n| for r>{r_near:.4f}  (n={far_mask.sum()})              = {abs_w[far_mask].mean():.6e}")
print(f"  max|w_n|  within r<={r_near:.4f} of corner                      = {abs_w[near_mask].max():.6e}")
print(f"  max|w_n|  for r>{r_near:.4f}                                    = {abs_w[far_mask].max() if far_mask.any() else float('nan'):.6e}")
ratio_mean = abs_w[near_mask].mean() / abs_w[far_mask].mean() if far_mask.any() and abs_w[far_mask].mean() > 0 else float("nan")
print(f"  near/far mean|w_n| ratio    = {ratio_mean:.3f}")
max_u_total = np.max(np.abs(d["u_total"]))
print(f"  max|w_n| / max|v_theta,n|   = {abs_w.max()/max_u_total:.6e}  (max|v_theta,n|={max_u_total:.6e})")

# ==================================================================
# B. Three separate structural panels + combined preview
# ==================================================================
cmap_u = plt.get_cmap("turbo").copy(); cmap_u.set_bad("0.92")
# FE compensator stays on a diverging map (turbo is not diverging and would
# obscure the sign change in w_n); kept as RdBu_r, unchanged.
cmap_w = plt.get_cmap("RdBu_r").copy(); cmap_w.set_bad("0.92")

vmin_u = min(np.nanmin(u_exact_g), np.nanmin(u_total_g))
vmax_u = max(np.nanmax(u_exact_g), np.nanmax(u_total_g))
w_abs_max = np.nanmax(np.abs(w_n_g))

def mark_corner(ax):
    ax.plot(0, 0, marker="*", color="red", ms=11, mec="white", mew=0.6, zorder=5)

def make_panel(grid, cmap, vmin, vmax, title, cbar_label, filename, symmetric_cbar=False):
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    im = ax.imshow(grid, origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_aspect("equal")
    # Corner star intentionally removed from these three per-panel figures
    # (section_4_rc_exact / _hybrid / _fe_compensator) per request. Left
    # untouched below for the combined preview, which isn't one of them.
    # Coordinate ticks/labels kept visible (previously suppressed here).
    ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
    ax.set_title(title, fontsize=11)
    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.03)
    cb.set_label(cbar_label, fontsize=9.5)
    fig.tight_layout()
    out = APP_DIR / filename
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    fig.savefig(f"{out}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}.pdf and .png")
    return im

PANELS = [
    dict(grid=u_exact_g, cmap=cmap_u, vmin=vmin_u, vmax=vmax_u,
         title=r"exact solution $u$", cbar_label="$u$",
         filename="section_4_rc_exact"),
    dict(grid=u_total_g, cmap=cmap_u, vmin=vmin_u, vmax=vmax_u,
         title=r"hybrid approximation $v_{\theta,n}$", cbar_label="$v_{\\theta,n}$",
         filename="section_4_rc_hybrid"),
    # FE compensator: own (symmetric, diverging) colorbar -- its magnitude is
    # much smaller than u/v_{theta,n} (see diagnostic above), so sharing the
    # same color scale would render it visually blank. Explicitly noted here
    # and in the report.
    dict(grid=w_n_g, cmap=cmap_w, vmin=-w_abs_max, vmax=w_abs_max,
         title=r"FE compensator $w_n$", cbar_label="$w_n$",
         filename="section_4_rc_fe_compensator"),
]

for p in PANELS:
    make_panel(p["grid"], p["cmap"], p["vmin"], p["vmax"], p["title"], p["cbar_label"], p["filename"])

# Combined preview (quick inspection only, not a primary deliverable)
fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.0))
for ax, p in zip(axes, PANELS):
    im = ax.imshow(p["grid"], origin="lower", extent=extent, cmap=p["cmap"], vmin=p["vmin"], vmax=p["vmax"])
    ax.set_aspect("equal"); mark_corner(ax)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(p["title"], fontsize=11)
    cb = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.03)
    cb.set_label(p["cbar_label"], fontsize=9)
fig.tight_layout()
preview = APP_DIR / "section_4_rc_decomposition_preview.png"
fig.savefig(str(preview), dpi=170, bbox_inches="tight")
plt.close(fig)
print(f"saved {preview} (combined preview, not a primary deliverable)")
