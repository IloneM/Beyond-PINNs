#!/usr/bin/env python
"""
[Copied from the research tree with minimal path/import adaptation only -- see docs/PROVENANCE.md.]
Redesigned MS (section_1) hybrid-decomposition appendix figure.

READS-ONLY from petrov_galerkin_results/section_1/ (final_solution.npz for the
representative seed) and petrov_standalone_fem_cache/section_1.npz. No neural
training, no FEM sweep, no seed selection beyond the exact pre-existing
representative_seed() logic (copied verbatim from generate_petrov_solution_figures.py)
applied to the unchanged saved history.npz files.

Splits the old 4-row figures/petrov/appendix/section_1_solution_components.pdf
("Hybrid total" / "NN component alone" / "FEM compensator alone" / "Standalone
near-budget FEM") into two deliverables:

  1. figures/petrov/appendix/section_1_solution_components.pdf
     -- 3 rows x 2 cols: Hybrid approximation / Neural component / FE
        compensator. No legend box anywhere -- self-contained via
        mathematical left/right panel titles and row labels instead (a
        standalone legend was generated in an earlier revision of this
        figure; that call was removed once the titles made it redundant --
        see "no legend" below. Any previously-generated
        section_1_solution_components_legend.pdf/.png is left on disk
        untouched but is no longer produced or used).

  2. figures/petrov/appendix/section_1_fe_reference.pdf
     -- 1x2: exact u vs. the standalone FE reference (left), |u_h-u| (right).
     + figures/petrov/appendix/section_1_fe_reference_legend.pdf
     (unchanged by the no-legend revision above; only the main decomposition
     figure dropped its legend.)

Notation (matches the code's own variable names, carried into the labels):
  u             exact solution                  (final_solution.npz: u_exact)
  v_{theta,n}   hybrid total                     (u_total)
  v_theta       neural component                 (u_nn)
  w_n           FE compensator                   (u_fem)
  u_h           standalone FE reference           (petrov_standalone_fem_cache/section_1.npz: u)

Run from the directory containing petrov_galerkin_results/:
    python generate_section1_hybrid_decomposition.py
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
from matplotlib.lines import Line2D

ROOT = Path("results/reference/general")
CACHE = Path("results/reference/fem")
APP_DIR = Path("figures/petrov/appendix")
if not ROOT.exists() or not CACHE.exists():
    raise SystemExit(
        "Reference results are not installed.\n\n"
        "Download/extract the Beyond PINNs reference-results archive into:\n"
        "    results/reference/\n\n"
        f"(expected to find data at {ROOT.resolve()} and {CACHE.resolve()})\n"
        "See docs/REPRODUCIBILITY.md or run scripts/download_reference_results.sh."
    )
APP_DIR.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": "0.3",
    "axes.grid": True, "grid.color": "0.85", "grid.linewidth": 0.5,
    "font.size": 10, "axes.labelsize": 10.5, "legend.fontsize": 9,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.4,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

SECTION = "section_1"
RUN = "s1_proj_h01_green_quadrature"  # principal hybrid method for MS, unchanged

# ------------------------------------------------------------------
# Representative-seed selection -- copied verbatim from
# generate_petrov_solution_figures.py::representative_seed, unmodified. Applied
# to the same saved history.npz files, so it reproduces the same seed as the
# existing figure (no new selection is performed).
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

d = np.load(ROOT / SECTION / RUN / f"seed_{seed:03d}" / "final_solution.npz")
fem = np.load(CACHE / f"{SECTION}.npz")
x = d["x"]
assert np.allclose(x, fem["x"]), "sample grid mismatch between hybrid run and standalone FE cache"

print(f"\nstandalone FE reference cache: degree={int(fem['degree'])}, n_dofs={int(fem['n_dofs'])}, "
      f"description={str(fem['description'])}")

# ------------------------------------------------------------------
# Fixed style -- reuses the color/linestyle convention already established
# for these exact semantic roles elsewhere in this figure suite
# (generate_petrov_solution_figures.py CROSS_COMPONENTS), so this figure reads
# consistently with the rest of the appendix.
# ------------------------------------------------------------------
STYLE = {
    "exact":    dict(color="black",    ls="-",  lw=1.3),
    "hybrid":   dict(color="#d95f02",  ls="-",  lw=1.3),   # v_{theta,n}
    "neural":   dict(color="#7570b3",  ls="--", lw=1.3),   # v_theta
    # Solid rather than dotted: with no legend to disambiguate by linestyle,
    # the bottom-left panel plots w_n alone (no second curve to distinguish
    # it from), so a clean solid line reads better than a dotted one.
    "fe_comp":  dict(color="#1b9e77",  ls="-",  lw=1.3),   # w_n
    "fe_ref":   dict(color="#e7298a",  ls="-.", lw=1.3),   # u_h (standalone)
}

FLOOR_REL = 1e-16  # main decomposition figure only (see below): a
                    # double-precision-appropriate floor, lowered from an
                    # earlier 1e-12 that was clipping the |v_{theta,n}-u| and
                    # |v_theta-u| panels to a flat line (their true pointwise
                    # error sits well below the old 1e-12*max|u_exact|~2.5e-13
                    # floor). With max|u_exact|~0.25 and max|w_n|~1.5e-13, the
                    # relative term is negligible for both, so in practice
                    # this floor is just the absolute 1e-16 machine-precision
                    # guard below, applied uniformly -- exact zeros still need
                    # *some* positive floor to be representable on a log axis.
FE_REF_FLOOR_REL = 1e-12  # standalone FE-reference figure only, kept at its
                    # original value so that block's rendering is untouched
                    # by this revision (its errors sit well above either
                    # floor anyway, so this only matters for a few dip points).

def clip_floor(y, ref, floor_rel=FLOOR_REL):
    floor = max(1e-16, floor_rel * np.max(np.abs(ref)))
    return np.clip(np.abs(y), floor, None)

# ==================================================================
# 1. Main hybrid-decomposition figure: 3 rows x 2 cols
# ==================================================================
ROWS = [
    # row_label,             left curves,                                        right quantity,               right title,              left title
    ("Hybrid approximation", [("exact", d["u_exact"]), ("hybrid", d["u_total"])], d["u_total"] - d["u_exact"],  r"$|v_{\theta,n}-u|$",    r"$u$ and $v_{\theta,n}$"),
    ("Neural component",     [("exact", d["u_exact"]), ("neural", d["u_nn"])],    d["u_nn"] - d["u_exact"],     r"$|v_\theta-u|$",        r"$u$ and $v_\theta$"),
    ("FE compensator",       [("fe_comp", d["u_fem"])],                          d["u_fem"],                    r"$|w_n|$",               r"$w_n$"),
]

fig, axes = plt.subplots(3, 2, figsize=(11, 9.7))
err_axes_machine_precision = []   # rows 0,1 (|v_{theta,n}-u|, |v_theta-u|): candidates for a shared y-scale
err_vals_machine_precision = []
for i, (row_label, left_curves, right_vals, right_title, left_title) in enumerate(ROWS):
    axL, axR = axes[i, 0], axes[i, 1]
    for role, vals in left_curves:
        axL.plot(x, vals, **STYLE[role])
    axL.set_ylabel(row_label, fontsize=9.5)
    axL.set_title(left_title, fontsize=10.5)

    # Row-specific floor reference: hybrid/neural rows are errors against u
    # (floor relative to |u|); the compensator row is a magnitude in its own
    # right (floor relative to its own scale), not an error against u.
    ref = d["u_exact"] if row_label != "FE compensator" else d["u_fem"]
    plotted = clip_floor(right_vals, ref)
    color = STYLE["hybrid" if i == 0 else ("neural" if i == 1 else "fe_comp")]["color"]
    axR.plot(x, plotted, color=color, lw=1.2)
    axR.set_yscale("log")
    axR.set_title(right_title, fontsize=10.5)
    if i in (0, 1):
        err_axes_machine_precision.append(axR)
        err_vals_machine_precision.append(plotted)
    if i == len(ROWS) - 1:
        axL.set_xlabel("$x$"); axR.set_xlabel("$x$")

# Shared y-limits for the two machine-precision error panels (|v_{theta,n}-u|,
# |v_theta-u|), so their common scale is directly comparable at a glance.
# |w_n| (row 2) is a magnitude, not an approximation error, and keeps its own
# autoscale, per instructions.
_combined = np.concatenate(err_vals_machine_precision)
_ymin, _ymax = float(_combined.min()), float(_combined.max())
if _ymax > _ymin > 0:
    _pad = (_ymax / _ymin) ** 0.15
    _shared_ylim = (_ymin / _pad, _ymax * _pad)
    for ax in err_axes_machine_precision:
        ax.set_ylim(*_shared_ylim)
    print(f"shared y-limits applied to the two machine-precision error panels: {_shared_ylim}")
else:
    print("machine-precision error panels left on independent autoscale "
          "(degenerate combined range, shared limits would not be meaningful)")

fig.tight_layout()
out = APP_DIR / "section_1_solution_components"
fig.savefig(f"{out}.pdf", bbox_inches="tight")
fig.savefig(f"{out}.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"saved {out}.pdf and .png")

# ==================================================================
# 2. Standalone FE-reference figure: 1x2
# ==================================================================
fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.4, 3.7))
axL.plot(x, d["u_exact"], **STYLE["exact"])
axL.plot(fem["x"], fem["u"], **STYLE["fe_ref"])
axL.set_title("value", fontsize=10.5)
axL.set_xlabel("$x$"); axL.set_ylabel("Standalone FE reference", fontsize=9.5)

err_ref = np.abs(fem["u"] - d["u_exact"])
axR.plot(fem["x"], clip_floor(err_ref, d["u_exact"], floor_rel=FE_REF_FLOOR_REL), color=STYLE["fe_ref"]["color"], lw=1.2)
axR.set_yscale("log")
axR.set_title(r"$|u_h-u|$", fontsize=10.5)
axR.set_xlabel("$x$")

fig.tight_layout()
out = APP_DIR / "section_1_fe_reference"
fig.savefig(f"{out}.pdf", bbox_inches="tight")
fig.savefig(f"{out}.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"saved {out}.pdf and .png")

handles = [Line2D([0], [0], **STYLE["exact"]), Line2D([0], [0], **STYLE["fe_ref"])]
labels = [r"exact solution $u$", r"standalone FE reference $u_h$"]
fig_leg = plt.figure(figsize=(4.2, 0.42))
fig_leg.legend(handles, labels, loc="center", ncol=2, frameon=True, fancybox=False,
               borderaxespad=0.2, columnspacing=1.4, handlelength=2.4)
leg_out = APP_DIR / "section_1_fe_reference_legend"
fig_leg.savefig(f"{leg_out}.pdf", bbox_inches="tight")
fig_leg.savefig(f"{leg_out}.png", dpi=200, bbox_inches="tight")
plt.close(fig_leg)
print(f"saved {leg_out}.pdf and .png")
