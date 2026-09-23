#!/usr/bin/env python
"""
[Copied from the research tree with minimal path/import adaptation only -- see docs/PROVENANCE.md.]
Publication-quality figure generation for the Petrov-Galerkin / Deep Ritz /
projected FEM-NN hybrid experiment suite. READS-ONLY from the completed
petrov_galerkin_results/ directory -- does not rerun neural-network
optimization, retune DSGNAR, or alter any history/solution file.

Run from /workspace:  python generate_petrov_figures.py
"""
import json, glob, csv
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
if not ROOT.exists():
    raise SystemExit(
        "Reference results are not installed.\n\n"
        "Download/extract the Beyond PINNs reference-results archive into:\n"
        "    results/reference/\n\n"
        f"(expected to find data at {ROOT.resolve()})\n"
        "See docs/REPRODUCIBILITY.md or run scripts/download_reference_results.sh."
    )
print("Using experiment root:", ROOT.resolve())

OUT_DIR = Path("figures/petrov")
APP_DIR = OUT_DIR / "appendix"
OUT_DIR.mkdir(parents=True, exist_ok=True)
APP_DIR.mkdir(parents=True, exist_ok=True)

SECTIONS = ["section_1", "section_2", "section_3", "section_4"]
SECTION_TITLES = {
    "section_1": "S1: multiscale $A$-Laplacian",
    "section_2": "S2: discontinuous forcing",
    "section_3": "S3: line source / kink",
    "section_4": "S4: L-shaped reentrant corner",
}

# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------
def seed_dirs(section, run_name):
    pat = str(ROOT / section / run_name / "seed_*" / "history.npz")
    return sorted(glob.glob(pat))

def load_histories(section, run_name):
    out = {}
    for p in seed_dirs(section, run_name):
        seed = int(Path(p).parent.name.split("_")[1])
        d = np.load(p)
        keys = ("iteration", "relative_l2", "relative_h1", "wallclock_optim")
        out[seed] = {k: np.asarray(d[k]) for k in keys}
    return out

def median_iqr_curve(histories, xkey, ykey):
    # Iteration axis: last-observation-carried-forward (LOCF) -- each seed s
    # stops at its own iteration T_s with a fixed returned solution, so its
    # error is held constant for k > T_s rather than dropped. All seeds
    # therefore contribute at every displayed iteration, out to the
    # longest-running seed -- no truncation to the shortest seed, no
    # shrinking sample size. (Same convention as
    # generate_pure_vs_projected_random_hats_panels.py.)
    #
    # Wall-clock axis: the x-axis itself is elapsed time and is not shared
    # across seeds, so reusing the index-based LOCF above (median of each
    # seed's own wallclock AT INDEX k) would distort it once seeds run very
    # different iteration counts -- a seed that finished early keeps
    # contributing a frozen, small elapsed-time value at every later index,
    # pulling the aggregate x-position left even though still-running seeds
    # used much more real time to reach that later data point. Instead,
    # resample onto a shared time grid spanning [0, max over seeds of that
    # seed's own true final wallclock], evaluating each seed as a
    # right-continuous step function of elapsed time, held flat after its
    # own last recorded timestamp -- no seed's clock is ever advanced past
    # the time it actually stopped, and no elapsed time is invented. See
    # docs/PROVENANCE.md section 5.2/5.3 / src/beyond_pinns/aggregation.py.
    hs = list(histories.values())
    if xkey == "iteration":
        n_max = max(len(h[ykey]) for h in hs)
        Y = np.empty((len(hs), n_max))
        for i, h in enumerate(hs):
            y = h[ykey]; n = len(y)
            Y[i, :n] = y
            if n < n_max:
                Y[i, n:] = y[-1]
        x = max(hs, key=lambda h: len(h[ykey]))["iteration"][:n_max]
    else:
        n_grid = max(len(h[xkey]) for h in hs)
        t_max = max(float(h[xkey][-1]) for h in hs)
        x = np.linspace(0.0, t_max, n_grid)
        Y = np.empty((len(hs), n_grid))
        for i, h in enumerate(hs):
            t = np.asarray(h[xkey], dtype=float)
            y = np.asarray(h[ykey], dtype=float)
            idx = np.searchsorted(t, x, side="right") - 1
            idx = np.clip(idx, 0, len(t) - 1)
            Y[i] = y[idx]
    med = np.median(Y, axis=0)
    q1 = np.percentile(Y, 25, axis=0)
    q3 = np.percentile(Y, 75, axis=0)
    return x, med, q1, q3

def final_stats(histories):
    l2 = np.array([h["relative_l2"][-1] for h in histories.values()])
    h1 = np.array([h["relative_h1"][-1] for h in histories.values()])
    wc = np.array([h["wallclock_optim"][-1] for h in histories.values()])
    return l2, h1, wc

# ------------------------------------------------------------------
# Fixed style
# ------------------------------------------------------------------
# Two-channel scheme, kept identical across every figure in this suite:
#   color     = TEST FAMILY (operator-adapted Green / non-operator kernel / random hats),
#               or a fixed color for the non-Petrov-Galerkin baselines;
#   linestyle = HYBRID TYPE (pure / projected / lagged / alternating).
# This is what lets "Projected, operator Green" and "Projected, random hats"
# (both hybrid_type=projected) stay visually distinct, and what lets Appendix A
# (projected vs lagged vs alternating, fixed test family) read cleanly off
# linestyle alone.
TEST_FAMILY_COLOR = {
    "green": "#377eb8",   # operator-adapted Green (a_green_*, eigen_green_*)
    "kernel": "#4daf4a",  # non-operator Sobolev/kernel (h01_green, h01_tensor, shifted_h1)
    "hats": "#984ea3",    # compact random hats
}
OTHER_COLOR = {
    "deep_ritz": "#1b9e77",
    "deep_ritz_l2_metric": "#a6d854",
    "strong_pinn": "#e6ab02",
    "exact_regression": "#666666",
}
HYBRID_LINESTYLE = {"pure": ":", "projected": "-", "lagged": "--", "alt": "-."}

def test_family_of(run_name):
    if "a_green" in run_name or "eigen_green" in run_name:
        return "green"
    if "h01_green" in run_name or "h01_tensor" in run_name or "shifted_h1" in run_name:
        return "kernel"
    if "random_hats" in run_name:
        return "hats"
    return None

def hybrid_type_of(run_name):
    if "_pure_" in run_name:
        return "pure"
    if "_proj_" in run_name:
        return "projected"
    if "_lagged_" in run_name:
        return "lagged"
    if "_alt_" in run_name:
        return "alt"
    return None

def family_of(run_name):
    """Coarse category, only used for Appendix D grouping."""
    if "exact_regression" in run_name:
        return "exact_regression"
    if "deep_ritz_l2_metric" in run_name:
        return "deep_ritz_l2_metric"
    if "deep_ritz" in run_name:
        return "deep_ritz"
    if "strong_pinn" in run_name:
        return "strong_pinn"
    hyb = hybrid_type_of(run_name)
    return hyb if hyb else "other"

def style_for(run_name):
    if "exact_regression" in run_name:
        return OTHER_COLOR["exact_regression"], "-"
    if "deep_ritz_l2_metric" in run_name:
        return OTHER_COLOR["deep_ritz_l2_metric"], "-"
    if "deep_ritz" in run_name:
        return OTHER_COLOR["deep_ritz"], "-"
    if "strong_pinn" in run_name:
        ls = "--" if ("missing_line" in run_name or "negative_control" in run_name) else "-"
        return OTHER_COLOR["strong_pinn"], ls
    fam = test_family_of(run_name)
    hyb = hybrid_type_of(run_name)
    return TEST_FAMILY_COLOR.get(fam, "0.4"), HYBRID_LINESTYLE.get(hyb, "-")

FLOOR = 1e-16

mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": "0.3",
    "axes.grid": True, "grid.color": "0.85", "grid.linewidth": 0.6,
    "font.size": 10.5, "axes.labelsize": 11.5, "legend.fontsize": 9.5,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.6,
})

def clip_for_log(y):
    return np.clip(y, FLOOR, None)

def finish_figure(fig, handles_ax, out_path_noext, ncol, rect=(0.02, 0.08, 1, 0.93)):
    handles, labels = handles_ax.get_legend_handles_labels()
    # de-duplicate while preserving order
    seen = set(); H = []; L = []
    for h, l in zip(handles, labels):
        if l not in seen:
            H.append(h); L.append(l); seen.add(l)
    fig.legend(H, L, loc="lower center", bbox_to_anchor=(0.5, -0.05),
               ncol=min(len(L), ncol), frameon=True, fancybox=False)
    fig.tight_layout(rect=list(rect))
    fig.savefig(f"{out_path_noext}.pdf", bbox_inches="tight")
    fig.savefig(f"{out_path_noext}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", f"{out_path_noext}.pdf", "and .png")

# ==================================================================
# Verification table (before plotting)
# ==================================================================
FIG1_METHODS = {
    "section_1": [("s1_deep_ritz", "Deep Ritz"), ("s1_strong_pinn", "Strong PINN"),
                  ("s1_pure_a_green_quadrature", "Pure PG, operator Green"),
                  ("s1_proj_a_green_quadrature", "Projected, operator Green"),
                  ("s1_proj_random_hats", "Projected, random hats")],
    "section_2": [("s2_deep_ritz", "Deep Ritz"), ("s2_strong_pinn", "Strong PINN"),
                  ("s2_pure_eigen_green_quadrature", "Pure PG, operator Green"),
                  ("s2_proj_eigen_green_quadrature", "Projected, operator Green"),
                  ("s2_proj_random_hats", "Projected, random hats")],
    "section_3": [("s3_deep_ritz", "Deep Ritz"),
                  ("s3_strong_pinn_missing_line_negative_control", "Strong PINN, missing line (wrong PDE)"),
                  ("s3_pure_eigen_green_quadrature", "Pure PG, operator Green"),
                  ("s3_proj_eigen_green_quadrature", "Projected, operator Green"),
                  ("s3_proj_random_hats", "Projected, random hats")],
    "section_4": [("s4_deep_ritz", "Deep Ritz"), ("s4_strong_pinn", "Strong PINN"),
                  ("s4_pure_eigen_green_quadrature", "Pure PG, operator Green"),
                  ("s4_proj_eigen_green_quadrature", "Projected, operator Green"),
                  ("s4_proj_random_hats", "Projected, random hats")],
}

# Full manifest of every run referenced by ANY figure in this script, so the
# verification table (and the two CSV exports) cover the whole suite.
ALL_RUN_NAMES = {}
for section in SECTIONS:
    names = sorted(Path(p).name for p in glob.glob(str(ROOT / section / "s*")) if Path(p).is_dir())
    ALL_RUN_NAMES[section] = names

print("\n" + "=" * 110)
print(f"{'section':10s} {'run_name':46s} {'n_seeds':7s} {'med L2':>10s} {'med H1':>10s} {'H1 IQR':>10s} {'med wall[s]':>12s}")
print("-" * 110)
VERIFY_ROWS = {}
SEED_ROWS = []  # for the CSV export
for section in SECTIONS:
    for run in ALL_RUN_NAMES[section]:
        hs = load_histories(section, run)
        if not hs:
            print(f"{section:10s} {run:46s}  MISSING"); continue
        l2, h1, wc = final_stats(hs)
        med_h1 = np.median(h1); q1, q3 = np.percentile(h1, 25), np.percentile(h1, 75)
        VERIFY_ROWS[(section, run)] = dict(n=len(hs), med_l2=np.median(l2), med_h1=med_h1,
                                            h1_iqr=q3 - q1, med_wc=np.median(wc))
        print(f"{section:10s} {run:46s} {len(hs):7d} {np.median(l2):10.3e} {med_h1:10.3e} {q3-q1:10.3e} {np.median(wc):12.2f}")
        for seed, h in hs.items():
            SEED_ROWS.append(dict(section=section, run_name=run, seed=seed,
                                   final_relative_l2=float(h["relative_l2"][-1]),
                                   final_relative_h1=float(h["relative_h1"][-1]),
                                   final_wallclock_optim=float(h["wallclock_optim"][-1]),
                                   n_recorded_iters=int(len(h["iteration"]))))
print("=" * 110 + "\n")

# ==================================================================
# Main Figure 1: formulation and hybrid overview (2x4)
# ==================================================================
fig, axes = plt.subplots(2, 4, figsize=(20, 8.6))
handles_ax = None
for col, section in enumerate(SECTIONS):
    for row, xkey in enumerate(("iteration", "wallclock_optim")):
        ax = axes[row, col]
        for run, label in FIG1_METHODS[section]:
            hs = load_histories(section, run)
            if not hs:
                continue
            x, med, q1, q3 = median_iqr_curve(hs, xkey, "relative_h1")
            color, ls = style_for(run)
            ax.fill_between(x, clip_for_log(q1), clip_for_log(q3), color=color, alpha=0.15, lw=0)
            ax.plot(x, clip_for_log(med), color=color, linestyle=ls, label=label)
        ax.set_yscale("log")
        ax.set_xlabel("optimization wall-clock [s]" if row == 1 else "iteration")
        if col == 0:
            ax.set_ylabel(r"relative $H^1$ error")
        handles_ax = ax
    axes[0, col].text(0.5, 1.08, SECTION_TITLES[section], transform=axes[0, col].transAxes,
                       ha="center", va="bottom", fontsize=11.5, fontweight="bold")
axes[0, 2].text(0.5, 1.22, "dashed = negative control (wrong PDE, not a valid competitor)",
                transform=axes[0, 2].transAxes, ha="center", va="bottom", fontsize=9, style="italic")
finish_figure(fig, handles_ax, str(OUT_DIR / "main_h1_comparison"), ncol=5)

# ==================================================================
# Main Figure 2: projected test-family comparison (1x4)
# ==================================================================
FIG2_METHODS = {
    "section_1": [("s1_proj_a_green_quadrature", "A-Green"), ("s1_proj_h01_green_quadrature", "H01 kernel"),
                  ("s1_proj_random_hats", "Random hats")],
    "section_2": [("s2_proj_eigen_green_quadrature", "Eigen-Green"), ("s2_proj_h01_tensor_quadrature", "Tensor H01"),
                  ("s2_proj_random_hats", "Random hats")],
    "section_3": [("s3_proj_eigen_green_quadrature", "Eigen-Green"), ("s3_proj_h01_tensor_quadrature", "Tensor H01"),
                  ("s3_proj_random_hats", "Random hats")],
    "section_4": [("s4_proj_eigen_green_quadrature", "Eigen-Green"), ("s4_proj_shifted_h1_quadrature", "Shifted Sobolev"),
                  ("s4_proj_random_hats", "Random hats")],
}
fig, axes = plt.subplots(1, 4, figsize=(20, 4.6))
handles_ax = None
for col, section in enumerate(SECTIONS):
    ax = axes[col]
    for run, label in FIG2_METHODS[section]:
        hs = load_histories(section, run)
        if not hs:
            continue
        x, med, q1, q3 = median_iqr_curve(hs, "iteration", "relative_h1")
        color, ls = style_for(run)
        ax.fill_between(x, clip_for_log(q1), clip_for_log(q3), color=color, alpha=0.15, lw=0)
        ax.plot(x, clip_for_log(med), color=color, linestyle=ls, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    if col == 0:
        ax.set_ylabel(r"relative $H^1$ error")
    ax.text(0.5, 1.08, SECTION_TITLES[section], transform=ax.transAxes,
             ha="center", va="bottom", fontsize=11.5, fontweight="bold")
    handles_ax = ax
finish_figure(fig, handles_ax, str(OUT_DIR / "projected_test_families_h1"), ncol=3, rect=(0.02, 0.1, 1, 0.9))

# ==================================================================
# Main/secondary Figure 3: pure vs projected random hats (2x4)
# ==================================================================
fig, axes = plt.subplots(2, 4, figsize=(20, 8.6))
handles_ax = None
for col, section in enumerate(SECTIONS):
    prefix = section.replace("section_", "s")
    runs = [(f"{prefix}_pure_random_hats", "Pure PG, random hats"),
            (f"{prefix}_proj_random_hats", "Projected, random hats")]
    for row, xkey in enumerate(("iteration", "wallclock_optim")):
        ax = axes[row, col]
        for run, label in runs:
            hs = load_histories(section, run)
            if not hs:
                continue
            x, med, q1, q3 = median_iqr_curve(hs, xkey, "relative_h1")
            color, ls = style_for(run)
            ax.fill_between(x, clip_for_log(q1), clip_for_log(q3), color=color, alpha=0.15, lw=0)
            ax.plot(x, clip_for_log(med), color=color, linestyle=ls, label=label)
        ax.set_yscale("log")
        ax.set_xlabel("optimization wall-clock [s]" if row == 1 else "iteration")
        if col == 0:
            ax.set_ylabel(r"relative $H^1$ error")
        handles_ax = ax
    axes[0, col].text(0.5, 1.08, SECTION_TITLES[section], transform=axes[0, col].transAxes,
                       ha="center", va="bottom", fontsize=11.5, fontweight="bold")
finish_figure(fig, handles_ax, str(OUT_DIR / "pure_vs_projected_random_hats_h1"), ncol=2)

# ==================================================================
# Appendix Figure A: projected / lagged / alternating ablation (2x4)
# ==================================================================
ABLATION_TEST = {"section_1": "a_green_quadrature", "section_2": "eigen_green_quadrature",
                  "section_3": "eigen_green_quadrature", "section_4": "eigen_green_quadrature"}
fig, axes = plt.subplots(2, 4, figsize=(20, 8.6))
handles_ax = None
for col, section in enumerate(SECTIONS):
    prefix = section.replace("section_", "s")
    test = ABLATION_TEST[section]
    rows = [("green-family test", test), ("random hats", "random_hats")]
    for row, (row_label, test_suffix) in enumerate(rows):
        ax = axes[row, col]
        for hyb, hyb_label in (("proj", "Projected"), ("lagged", "Lagged-Jacobian"), ("alt", "Alternating")):
            run = f"{prefix}_{hyb}_{test_suffix}"
            hs = load_histories(section, run)
            if not hs:
                continue
            x, med, q1, q3 = median_iqr_curve(hs, "iteration", "relative_h1")
            color, ls = style_for(run)
            ax.fill_between(x, clip_for_log(q1), clip_for_log(q3), color=color, alpha=0.15, lw=0)
            ax.plot(x, clip_for_log(med), color=color, linestyle=ls, label=hyb_label)
        ax.set_yscale("log")
        ax.set_xlabel("iteration")
        if col == 0:
            ax.set_ylabel(r"relative $H^1$ error" + f"\n({row_label})")
        handles_ax = ax
    axes[0, col].text(0.5, 1.08, SECTION_TITLES[section], transform=axes[0, col].transAxes,
                       ha="center", va="bottom", fontsize=11.5, fontweight="bold")
finish_figure(fig, handles_ax, str(APP_DIR / "hybrid_update_ablation_h1"), ncol=3)

# ==================================================================
# Appendix Figure D: all principal ablations, categorical (1x4)
# ==================================================================
CATEGORY_ORDER = ["exact_regression", "deep_ritz", "deep_ritz_l2_metric", "strong_pinn",
                  "pure", "projected", "lagged", "alt"]
CATEGORY_LABEL = {"exact_regression": "exact\nregression", "deep_ritz": "Deep\nRitz",
                   "deep_ritz_l2_metric": "Deep Ritz\nL2 metric", "strong_pinn": "strong\nPINN",
                   "pure": "pure\nPG", "projected": "projected\nhybrid",
                   "lagged": "lagged\nhybrid", "alt": "alternating\nhybrid"}
MARKER = {"pure": "o", "projected": "s", "lagged": "^", "alt": "D"}
OTHER_MARKER = {"deep_ritz": "P", "deep_ritz_l2_metric": "X", "strong_pinn": "*", "exact_regression": "v"}

def marker_for(run_name):
    if "exact_regression" in run_name:
        return OTHER_MARKER["exact_regression"]
    if "deep_ritz_l2_metric" in run_name:
        return OTHER_MARKER["deep_ritz_l2_metric"]
    if "deep_ritz" in run_name:
        return OTHER_MARKER["deep_ritz"]
    if "strong_pinn" in run_name:
        return OTHER_MARKER["strong_pinn"]
    return MARKER.get(hybrid_type_of(run_name), "o")

fig, axes = plt.subplots(1, 4, figsize=(20, 5.4))
rng = np.random.RandomState(0)
legend_handles = {}
for col, section in enumerate(SECTIONS):
    ax = axes[col]
    summary = json.load(open(ROOT / section / "section_summary.json"))
    for r in summary:
        run = r["run_name"]
        cat = family_of(run)
        if cat not in CATEGORY_ORDER:
            continue
        xpos = CATEGORY_ORDER.index(cat) + rng.uniform(-0.18, 0.18)
        color, _ = style_for(run)
        mk = marker_for(run)
        h = ax.scatter(xpos, max(r["median_h1"], FLOOR), color=color, marker=mk, s=55,
                        edgecolor="black", linewidth=0.4, zorder=3)
        legend_handles.setdefault(cat, (h, CATEGORY_LABEL[cat].replace("\n", " ")))
    ax.set_yscale("log")
    ax.set_xticks(range(len(CATEGORY_ORDER)))
    ax.set_xticklabels([CATEGORY_LABEL[c] for c in CATEGORY_ORDER], fontsize=8.5)
    ax.set_xlim(-0.6, len(CATEGORY_ORDER) - 0.4)
    if col == 0:
        ax.set_ylabel(r"median final relative $H^1$ error")
    ax.text(0.5, 1.05, SECTION_TITLES[section], transform=ax.transAxes,
             ha="center", va="bottom", fontsize=11.5, fontweight="bold")
H = [v[0] for v in legend_handles.values()]; L = [v[1] for v in legend_handles.values()]
fig.legend(H, L, loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=8, frameon=True, fontsize=9.5)
fig.tight_layout(rect=[0.02, 0.1, 1, 0.93])
fig.savefig(APP_DIR / "all_method_medians_h1.pdf", bbox_inches="tight")
fig.savefig(APP_DIR / "all_method_medians_h1.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("saved", APP_DIR / "all_method_medians_h1.pdf", "and .png")

# ==================================================================
# Appendix Figure C: standalone FEM comparison (1x4 bars)
# ==================================================================
PRINCIPAL_PROJECTED = {"section_1": "s1_proj_a_green_quadrature", "section_2": "s2_proj_eigen_green_quadrature",
                        "section_3": "s3_proj_eigen_green_quadrature", "section_4": "s4_proj_eigen_green_quadrature"}
fig, axes = plt.subplots(1, 4, figsize=(20, 5.0))
bar_colors = ["#d95f02", "#377eb8", "#377eb8"]
bar_alphas = [1.0, 1.0, 0.55]
for col, section in enumerate(SECTIONS):
    ax = axes[col]
    run = PRINCIPAL_PROJECTED[section]
    hyb_h1 = VERIFY_ROWS[(section, run)]["med_h1"]
    sel = json.load(open(ROOT / section / "fem_baselines" / "selected.json"))
    fem_hybrid_budget = sel["best_fem_h1_under_hybrid_budget"]
    fem_nn_budget = sel["best_fem_h1_under_nn_budget"]
    vals = [hyb_h1, fem_hybrid_budget["rel_h1"], fem_nn_budget["rel_h1"]]
    labels = ["Projected\nhybrid", f"FEM @ hybrid\nbudget ({fem_hybrid_budget['n_dofs']} dof)",
              f"FEM @ NN\nbudget ({fem_nn_budget['n_dofs']} dof)"]
    bars = ax.bar(range(3), np.clip(vals, FLOOR, None), color=bar_colors,
                  edgecolor="black", linewidth=0.6)
    for b, a in zip(bars, bar_alphas):
        b.set_alpha(a)
    ax.set_yscale("log")
    ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=8.5)
    if col == 0:
        ax.set_ylabel(r"median/selected final relative $H^1$ error")
    ax.text(0.5, 1.05, SECTION_TITLES[section], transform=ax.transAxes,
             ha="center", va="bottom", fontsize=11.5, fontweight="bold")
fig.suptitle("", fontsize=1)  # no in-panel titles; caption (outside this script) carries the note below
fig.tight_layout(rect=[0.02, 0.02, 1, 0.93])
fig.savefig(APP_DIR / "hybrid_vs_fem_h1.pdf", bbox_inches="tight")
fig.savefig(APP_DIR / "hybrid_vs_fem_h1.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("saved", APP_DIR / "hybrid_vs_fem_h1.pdf", "and .png")
print("NOTE for caption: FEM candidates are selected within ~+/-20% of the nominal DOF/parameter")
print("budget (see budget_ratio in fem_baselines/selected.json) -- NOT strictly under budget in every case.")
print("NOTE for caption (Section 2 only): the selected interface-fitted P4 FEM space CONTAINS the")
print("manufactured piecewise-polynomial exact solution, so it reaches near machine precision. This is a")
print("structure-exploiting reference, not a generic FEM-vs-neural claim.")

# ==================================================================
# Appendix Figure B: hybrid compensator meshes (1x4)
# ==================================================================
# Reimplemented in plain NumPy from femennstein-petrov-galerkin-experiments-fixed.py
# (make_unit_square_mesh, make_lshape_corner_mesh) -- geometry/connectivity only,
# no PDE solve, so no need to import the full JAX notebook module.

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

fig, axes = plt.subplots(1, 4, figsize=(20, 5.4))

# Section 1: 16-element P1 mesh on [0,1]
ax = axes[0]
n_elem = 16
nodes1d = np.linspace(0.0, 1.0, n_elem + 1)
ax.plot(nodes1d, np.zeros_like(nodes1d), color="0.5", lw=1.2, zorder=1)
ax.scatter(nodes1d[[0, -1]], [0, 0], color="black", marker="s", s=45, zorder=3, label="boundary DOF (Dirichlet)")
ax.scatter(nodes1d[1:-1], np.zeros(n_elem - 1), color=OTHER_COLOR["deep_ritz"], marker="o", s=45, zorder=3,
           label="interior FEM DOF")
eps = 1.0 / 16.0
ax.annotate("", xy=(eps, 0.15), xytext=(0.0, 0.15), arrowprops=dict(arrowstyle="<->"))
ax.text(eps / 2, 0.19, r"$\epsilon=1/16$", ha="center", fontsize=10)
ax.set_ylim(-0.3, 0.4); ax.set_xlim(-0.03, 1.03)
ax.set_yticks([]); ax.set_xlabel("$x$")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.42), ncol=1, fontsize=8.5, frameon=True)
ax.text(0.5, 1.08, SECTION_TITLES["section_1"], transform=ax.transAxes, ha="center", va="bottom",
         fontsize=11.5, fontweight="bold")

# Section 2: 16x12 interface-fitted triangular mesh, highlight x=1/2
ax = axes[1]
nodes22, tris22 = np_make_unit_square_mesh(16, 12)
ax.triplot(nodes22[:, 0], nodes22[:, 1], tris22, color="0.5", lw=0.6)
ax.axvline(0.5, color=OTHER_COLOR["deep_ritz"], lw=2.2, label="$x=1/2$ (interface)")
ax.set_aspect("equal"); ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=1, fontsize=8.5, frameon=True)
ax.text(0.5, 1.08, SECTION_TITLES["section_2"], transform=ax.transAxes, ha="center", va="bottom",
         fontsize=11.5, fontweight="bold")

# Section 3: 16x12 interface-fitted triangular mesh, highlight loaded line x=1/2
ax = axes[2]
nodes24, tris24 = np_make_unit_square_mesh(16, 12)
ax.triplot(nodes24[:, 0], nodes24[:, 1], tris24, color="0.5", lw=0.6)
ax.axvline(0.5, color="#d95f02", lw=2.6, label="$x=1/2$ (Dirac line load)")
ax.set_aspect("equal"); ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=1, fontsize=8.5, frameon=True)
ax.text(0.5, 1.08, SECTION_TITLES["section_3"], transform=ax.transAxes, ha="center", va="bottom",
         fontsize=11.5, fontweight="bold")

# Section 4: corner-graded polar mesh, n_radial=14, n_angular=32, beta=2
ax = axes[3]
nodesP, trisP = np_make_lshape_corner_mesh(n_radial=14, n_angular=32, beta=2.0)
ax.triplot(nodesP[:, 0], nodesP[:, 1], trisP, color="0.5", lw=0.5)
ax.scatter([0.0], [0.0], color="#d95f02", marker="*", s=160, zorder=5, label="reentrant corner")
# shade the removed quadrant (x>0, y<0) to make the L-shape explicit
ax.add_patch(plt.Rectangle((0, -1.05), 1.05, 1.05, color="0.9", zorder=0))
ax.set_aspect("equal"); ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=1, fontsize=8.5, frameon=True)
ax.text(0.5, 1.08, SECTION_TITLES["section_4"], transform=ax.transAxes, ha="center", va="bottom",
         fontsize=11.5, fontweight="bold")

fig.suptitle("")
fig.tight_layout(rect=[0.01, 0.14, 1, 0.9])
fig.savefig(APP_DIR / "meshes.pdf", bbox_inches="tight")
fig.savefig(APP_DIR / "meshes.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("saved", APP_DIR / "meshes.pdf", "and .png")
print("(these are the HYBRID COMPENSATOR meshes, not the larger standalone FEM reference candidates)")

# ==================================================================
# Final numerical report + CSV exports
# ==================================================================
with open(OUT_DIR / "petrov_seed_level_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(SEED_ROWS[0].keys()))
    w.writeheader(); w.writerows(SEED_ROWS)
print("saved", OUT_DIR / "petrov_seed_level_summary.csv")

METHOD_ROWS = []
for (section, run), row in sorted(VERIFY_ROWS.items()):
    METHOD_ROWS.append(dict(section=section, run_name=run, n_seeds=row["n"],
                             median_relative_l2=row["med_l2"], median_relative_h1=row["med_h1"],
                             h1_iqr=row["h1_iqr"], median_wallclock_optim=row["med_wc"],
                             test_family=test_family_of(run), hybrid_type=hybrid_type_of(run)))
with open(OUT_DIR / "petrov_method_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(METHOD_ROWS[0].keys()))
    w.writeheader(); w.writerows(METHOD_ROWS)
print("saved", OUT_DIR / "petrov_method_summary.csv")

REFERENCE = {
    ("section_1", "s1_pure_h01_green_quadrature"): 1.1e-13,
    ("section_1", "s1_proj_h01_green_quadrature"): 1.1e-12,
    ("section_1", "s1_proj_random_hats"): 2.2e-11,
    ("section_2", "s2_proj_h01_tensor_quadrature"): 4.3e-4,
    ("section_2", "s2_proj_random_hats"): 1.3e-3,
    ("section_2", "s2_proj_eigen_green_quadrature"): 2.0e-3,
    ("section_2", "s2_deep_ritz"): 2.5e-2,
    ("section_3", "s3_pure_eigen_green_quadrature"): 5.6e-3,
    ("section_3", "s3_proj_eigen_green_quadrature"): 6.7e-3,
    ("section_3", "s3_proj_random_hats"): 3.3e-2,
    ("section_4", "s4_proj_random_hats"): 1.2e-2,
    ("section_4", "s4_proj_eigen_green_quadrature"): 1.4e-1,
    ("section_4", "s4_proj_shifted_h1_quadrature"): 1.5e-1,
    ("section_4", "s4_deep_ritz"): 1.15,
}
print("\n" + "#" * 100)
print("FINAL NUMERICAL REPORT")
print("#" * 100)
print("Input experiment directory:", ROOT.resolve())
n_fail = 0
print("\nChecks against the given reference numbers (approximate, order-of-magnitude expected):")
for (section, run), ref in REFERENCE.items():
    got = VERIFY_ROWS.get((section, run))
    if got is None:
        print(f"  {section:10s} {run:40s}  MISSING"); n_fail += 1; continue
    rel_diff = abs(got["med_h1"] - ref) / ref
    status = "OK" if rel_diff < 0.25 else "DISCREPANCY"
    if status != "OK":
        n_fail += 1
    print(f"  {section:10s} {run:40s} got={got['med_h1']:.3e}  ref~={ref:.2e}  {status}")
print("\nSection 3 negative control:",
      "s3_strong_pinn_missing_line_negative_control H1 =",
      f"{VERIFY_ROWS[('section_3','s3_strong_pinn_missing_line_negative_control')]['med_h1']:.3e}",
      "-- NOT a solution of the target PDE (omits the line measure), reported for completeness only, not as a competitor.")

print(f"\n{'No discrepancies.' if n_fail == 0 else f'*** {n_fail} discrepancy/discrepancies found -- see above. ***'}")

print("\nSeeds found per run (all should be 5/5):")
short = [(s, r, v['n']) for (s, r), v in sorted(VERIFY_ROWS.items()) if v['n'] != 5]
print("  all runs have 5/5 seeds" if not short else short)

print("\nGenerated main + appendix files so far:")
for f_ in sorted(OUT_DIR.rglob("*")):
    if f_.is_file():
        print(" ", f_.resolve())
