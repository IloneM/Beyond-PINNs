#!/usr/bin/env python
"""
[Copied from the research tree with minimal path/import adaptation only -- see docs/PROVENANCE.md.]
MS (section_1) diagnostic-only panels for LaTeX subcaption assembly.

READS-ONLY from petrov_galerkin_results/section_1/ (final_solution.npz for the
representative seed) and petrov_standalone_fem_cache/section_1.npz. No neural
training, no FEM sweep, no seed selection beyond the exact pre-existing
representative_seed() logic (copied verbatim from
generate_petrov_solution_figures.py / generate_section1_hybrid_decomposition.py)
applied to the unchanged saved history.npz files. The standalone FE reference
is the same cached candidate used by every earlier revision of these figures
(selected.json -> best_fem_h1_under_hybrid_budget, degree 4, 96 uniform
elements, 383 DOFs, ~0.98x the hybrid budget) -- read verbatim, not reselected.

Presentation-only refactor: drops the solution/value panels entirely and
produces FOUR separate single-plot diagnostic panels (log-scale error/
magnitude curves only, no legend, no value plots), for LaTeX subcaption
assembly, under figures/petrov/ (NOT figures/petrov/appendix/, as requested):

  figures/petrov/section_1_ms_hybrid_error.pdf/.png       |v_{theta,n}-u|
  figures/petrov/section_1_ms_neural_error.pdf/.png       |v_theta-u|
  figures/petrov/section_1_ms_fe_compensator.pdf/.png     |w_n|
  figures/petrov/section_1_ms_fe_reference_error.pdf/.png |u_h-u|

Plus an optional combined 2x2 preview PNG (quick visual inspection only, not
a primary deliverable): figures/petrov/section_1_ms_diagnostics_preview.png

Does not touch figures/petrov/appendix/section_1_solution_components.pdf,
figures/petrov/appendix/section_1_fe_reference.pdf, or any other existing
figure -- this script only adds the four new files (+ optional preview).

Run from the directory containing petrov_galerkin_results/:
    python generate_section1_ms_diagnostics.py
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
CACHE = Path("results/reference/fem")
OUT_DIR = Path("figures/petrov")
if not ROOT.exists() or not CACHE.exists():
    raise SystemExit(
        "Reference results are not installed.\n\n"
        "Download/extract the Beyond PINNs reference-results archive into:\n"
        "    results/reference/\n\n"
        f"(expected to find data at {ROOT.resolve()} and {CACHE.resolve()})\n"
        "See docs/REPRODUCIBILITY.md or run scripts/download_reference_results.sh."
    )
OUT_DIR.mkdir(parents=True, exist_ok=True)

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
# Representative-seed selection -- copied verbatim, unmodified. Applied to
# the same saved history.npz files, so it reproduces the same seed as every
# earlier revision of this figure (no new selection is performed).
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
cache_path = CACHE / f"{SECTION}.npz"
if not final_path.exists():
    raise SystemExit(f"Expected file missing, stopping rather than improvising: {final_path}")
if not cache_path.exists():
    raise SystemExit(f"Expected file missing, stopping rather than improvising: {cache_path}")

d = np.load(final_path)
fem = np.load(cache_path)
x = d["x"]
if not np.allclose(x, fem["x"]):
    raise SystemExit("Sample grid mismatch between hybrid run and standalone FE cache -- stopping.")

print(f"\nstandalone FE reference cache (unchanged): degree={int(fem['degree'])}, "
      f"n_dofs={int(fem['n_dofs'])}, description={str(fem['description'])}")

# ------------------------------------------------------------------
# Same color convention as the rest of this figure suite; floors match the
# already-corrected values from the last revision (1e-16-relative for the
# three hybrid-construction quantities, 1e-12-relative -- unchanged -- for
# the standalone FE reference).
# ------------------------------------------------------------------
COLOR = {
    "hybrid_err":  "#d95f02",  # |v_{theta,n}-u|
    "neural_err":  "#7570b3",  # |v_theta-u|
    "fe_comp":     "#1b9e77",  # |w_n|
    "fe_ref_err":  "#e7298a",  # |u_h-u|
}

def clip_floor(y, ref, floor_rel):
    floor = max(1e-16, floor_rel * np.max(np.abs(ref)))
    return np.clip(np.abs(y), floor, None)

PANELS = [
    # filename stem,               values,                             ref for floor,   floor_rel, color,               title
    ("section_1_ms_hybrid_error",       d["u_total"] - d["u_exact"], d["u_exact"], 1e-16, COLOR["hybrid_err"], r"$|v_{\theta,n}-u|$"),
    ("section_1_ms_neural_error",       d["u_nn"] - d["u_exact"],    d["u_exact"], 1e-16, COLOR["neural_err"], r"$|v_\theta-u|$"),
    ("section_1_ms_fe_compensator",     d["u_fem"],                  d["u_fem"],   1e-16, COLOR["fe_comp"],    r"$|w_n|$"),
    ("section_1_ms_fe_reference_error", fem["u"] - d["u_exact"],     d["u_exact"], 1e-12, COLOR["fe_ref_err"], r"$|u_h-u|$"),
]

PANEL_FIGSIZE = (3.6, 3.0)  # single clean panel, subcaption-friendly

generated = []
for stem, raw_vals, ref, floor_rel, color, title in PANELS:
    xg = fem["x"] if stem == "section_1_ms_fe_reference_error" else x
    yg = clip_floor(raw_vals, ref, floor_rel)
    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)
    ax.plot(xg, yg, color=color, lw=1.3)
    ax.set_yscale("log")
    ax.set_title(title, fontsize=10.5)
    ax.set_xlabel("$x$")
    fig.tight_layout()
    out = OUT_DIR / stem
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    fig.savefig(f"{out}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    generated.append(str(out))
    print(f"saved {out}.pdf and .png")

# ------------------------------------------------------------------
# Optional combined 2x2 preview (quick visual inspection only)
# ------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(9.4, 7.6))
for ax, (stem, raw_vals, ref, floor_rel, color, title) in zip(axes.ravel(), PANELS):
    xg = fem["x"] if stem == "section_1_ms_fe_reference_error" else x
    yg = clip_floor(raw_vals, ref, floor_rel)
    ax.plot(xg, yg, color=color, lw=1.3)
    ax.set_yscale("log")
    ax.set_title(title, fontsize=10.5)
    ax.set_xlabel("$x$")
fig.tight_layout()
preview = OUT_DIR / "section_1_ms_diagnostics_preview.png"
fig.savefig(str(preview), dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"saved {preview} (combined preview, not a primary deliverable)")

print("\nGenerated primary files:")
for g in generated:
    print(f"  {g}.pdf")
    print(f"  {g}.png")
