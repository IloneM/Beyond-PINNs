#!/usr/bin/env python
"""
[Copied from the research tree with minimal path/import adaptation only -- see docs/PROVENANCE.md.]
Pure-neural vs projected FE-neural comparison, fixed random-hat weak test
family, one clean panel per benchmark (MS/JF/LS/RC), for `subcaption`
assembly in the manuscript.

READS-ONLY from petrov_galerkin_results/ -- does not rerun optimization or
touch any saved history/config/solution file. Replaces the old 2x4
figures/petrov/pure_vs_projected_random_hats_h1.pdf (iteration+wallclock rows,
H1-only, S1-S4 titles) with four independent single-plot PDFs (iteration only,
L2+H1 together, no embedded titles) plus a standalone legend PDF.

Run from the directory containing petrov_galerkin_results/:
    python generate_pure_vs_projected_random_hats_panels.py
Add --legend-only to only rebuild the standalone legend PDF/PNG (no data
loading, no panel regeneration) -- used e.g. for pure label-wording updates.
"""
import argparse
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

parser = argparse.ArgumentParser()
parser.add_argument("--legend-only", action="store_true",
                     help="Only rebuild the standalone legend PDF/PNG: no ROOT check, "
                          "no data loading, no manuscript-value verification, no panel "
                          "regeneration. Use this for label-wording-only updates.")
args = parser.parse_args()

OUT_DIR = Path("figures/petrov")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# section_1..4 -> (short tag used only for filenames/printing, subcaption text)
SECTIONS = [
    ("section_1", "ms", "MS: multiscale diffusion"),
    ("section_2", "jf", "JF: jump forcing"),
    ("section_3", "ls", "LS: line source"),
    ("section_4", "rc", "RC: reentrant corner"),
]

# ------------------------------------------------------------------
# Data loading (same convention as generate_petrov_figures.py)
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

def median_iqr_curve(histories, ykey):
    # Last-observation-carried-forward (LOCF) aggregation: each seed s stops
    # at its own iteration T_s with a fixed returned solution, so its error
    # is held constant (E_s(k) = E_s(T_s) for k > T_s) rather than dropped.
    # All 5 seeds therefore contribute to the median/IQR at every displayed
    # iteration, out to the longest-running seed -- no truncation to the
    # shortest seed, and no shrinking sample size with iteration.
    hs = list(histories.values())
    n_max = max(len(h[ykey]) for h in hs)
    Y = np.empty((len(hs), n_max))
    for i, h in enumerate(hs):
        y = h[ykey]
        n = len(y)
        Y[i, :n] = y
        if n < n_max:
            Y[i, n:] = y[-1]  # carry the final (post-termination) error forward
    x = max(hs, key=lambda h: len(h[ykey]))["iteration"][:n_max]
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
# Verification against the manuscript table (before plotting anything)
# ------------------------------------------------------------------
MANUSCRIPT = {
    # tag: (l2_pure, l2_proj, h1_pure, h1_proj)
    "ms": (2.19e-13, 9.14e-13, 8.26e-12, 2.16e-11),
    "jf": (2.05,     1.71e-4, 4.43,     1.34e-3),
    "ls": (9.93e-1,  4.07e-3, 9.89e-1,  3.32e-2),
    "rc": (1.052,    5.04e-4, 1.40,     1.20e-2),
}

DATA = {}  # tag -> {"pure": histories, "proj": histories}
if not args.legend_only:
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

    print("\n" + "=" * 118)
    print(f"{'tag':4s} {'run_name':22s} {'n_seeds':7s} {'n_iters(min seed)':18s} "
          f"{'med L2':>12s} {'manuscript L2':>14s} {'med H1':>12s} {'manuscript H1':>14s}")
    print("-" * 118)
    for section, tag, _ in SECTIONS:
        prefix = section.replace("section_", "s")
        DATA[tag] = {}
        l2_pure_ms, l2_proj_ms, h1_pure_ms, h1_proj_ms = MANUSCRIPT[tag]
        for model, ms_l2, ms_h1 in (("pure", l2_pure_ms, h1_pure_ms), ("proj", l2_proj_ms, h1_proj_ms)):
            run_name = f"{prefix}_{model}_random_hats"
            hs = load_histories(section, run_name)
            if not hs:
                print(f"{tag:4s} {run_name:22s}  MISSING")
                continue
            if len(hs) != 5:
                print(f"{tag:4s} {run_name:22s}  WARNING: {len(hs)} seeds found, expected 5")
            l2, h1, wc = final_stats(hs)
            med_l2, med_h1 = np.median(l2), np.median(h1)
            n_iters = min(len(h["iteration"]) for h in hs.values())
            rel_l2 = abs(med_l2 - ms_l2) / abs(ms_l2)
            rel_h1 = abs(med_h1 - ms_h1) / abs(ms_h1)
            flag = "" if (rel_l2 < 0.02 and rel_h1 < 0.02) else "  <-- MISMATCH"
            print(f"{tag:4s} {run_name:22s} {len(hs):7d} {n_iters:18d} "
                  f"{med_l2:12.4e} {ms_l2:14.4e} {med_h1:12.4e} {ms_h1:14.4e}{flag}")
            DATA[tag][model] = hs
    print("=" * 118 + "\n")
else:
    print("--legend-only: skipping ROOT check, data loading, verification, and panel regeneration.")

# ------------------------------------------------------------------
# Fixed style: linestyle = model, color = metric.
# Kept deliberately distinct from the color roles used elsewhere in the
# petrov figure suite (TEST_FAMILY_COLOR / OTHER_COLOR in
# generate_petrov_figures.py), since here color no longer encodes test
# family (fixed to random hats throughout) but instead the error metric.
# ------------------------------------------------------------------
METRIC_COLOR = {"l2": "#e41a1c", "h1": "#377eb8"}   # red = L2, blue = H1
MODEL_LINESTYLE = {"pure": ":", "proj": "-"}         # dotted = pure, solid = projected/hybrid
MODEL_LABEL = {"pure": "Pure neural", "proj": "Hybrid FE–neural"}
METRIC_LABEL = {"l2": "$L^2$", "h1": "$H^1$"}
YKEY = {"l2": "relative_l2", "h1": "relative_h1"}

FLOOR = 1e-16
def clip_for_log(y):
    return np.clip(y, FLOOR, None)

mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": "0.3",
    "axes.grid": True, "grid.color": "0.85", "grid.linewidth": 0.6,
    "font.size": 9.5, "axes.labelsize": 10.5, "legend.fontsize": 8.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.6,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

PANEL_FIGSIZE = (3.4, 2.9)  # mildly landscape; ~native size at 0.48\textwidth

def draw_panel(ax, tag):
    for model in ("pure", "proj"):
        hs = DATA[tag].get(model)
        if not hs:
            continue
        for metric in ("l2", "h1"):
            x, med, q1, q3 = median_iqr_curve(hs, YKEY[metric])
            color = METRIC_COLOR[metric]
            ls = MODEL_LINESTYLE[model]
            ax.fill_between(x, clip_for_log(q1), clip_for_log(q3), color=color, alpha=0.14, lw=0)
            ax.plot(x, clip_for_log(med), color=color, linestyle=ls, lw=1.6,
                     label=f"{MODEL_LABEL[model]} – {METRIC_LABEL[metric]}")
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("relative error")
    ax.margins(x=0.02)

# ------------------------------------------------------------------
# Four independent single-plot PDFs, no titles, no legends.
# ------------------------------------------------------------------
if not args.legend_only:
    for section, tag, subcap in SECTIONS:
        fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)
        draw_panel(ax, tag)
        fig.tight_layout(pad=0.3)
        out = OUT_DIR / f"pure_vs_projected_random_hats_{tag}"
        fig.savefig(f"{out}.pdf", bbox_inches="tight")
        fig.savefig(f"{out}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {out}.pdf and .png  ({subcap})")

# ------------------------------------------------------------------
# Standalone legend PDF (4 entries, one row), reusing the exact handles/
# styles drawn above so it is guaranteed consistent with the panels.
# ------------------------------------------------------------------
proxy_handles = [
    Line2D([0], [0], color=METRIC_COLOR["l2"], linestyle=MODEL_LINESTYLE["pure"], lw=1.6),
    Line2D([0], [0], color=METRIC_COLOR["h1"], linestyle=MODEL_LINESTYLE["pure"], lw=1.6),
    Line2D([0], [0], color=METRIC_COLOR["l2"], linestyle=MODEL_LINESTYLE["proj"], lw=1.6),
    Line2D([0], [0], color=METRIC_COLOR["h1"], linestyle=MODEL_LINESTYLE["proj"], lw=1.6),
]
proxy_labels = [
    f"{MODEL_LABEL['pure']} – {METRIC_LABEL['l2']}",
    f"{MODEL_LABEL['pure']} – {METRIC_LABEL['h1']}",
    f"{MODEL_LABEL['proj']} – {METRIC_LABEL['l2']}",
    f"{MODEL_LABEL['proj']} – {METRIC_LABEL['h1']}",
]
fig_leg = plt.figure(figsize=(6.4, 0.42))
fig_leg.legend(proxy_handles, proxy_labels, loc="center", ncol=4, frameon=True,
               fancybox=False, borderaxespad=0.2, columnspacing=1.4, handlelength=2.2)
fig_leg.savefig(str(OUT_DIR / "pure_vs_projected_random_hats_legend.pdf"), bbox_inches="tight")
fig_leg.savefig(str(OUT_DIR / "pure_vs_projected_random_hats_legend.png"), dpi=220, bbox_inches="tight")
plt.close(fig_leg)
print(f"saved {OUT_DIR / 'pure_vs_projected_random_hats_legend.pdf'} and .png")
