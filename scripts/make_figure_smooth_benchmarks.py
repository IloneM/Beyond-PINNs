#!/usr/bin/env python
"""
[Copied from the research tree with minimal path/import adaptation only -- see docs/PROVENANCE.md.]
Third-revision publication figures for energy_vs_weak_benchmarks/.
READS-ONLY (history.npz/config.json) -- does not rerun optimization (the
plain_gn_strong_lm 300->1000 extension was already performed separately via
checkpoint resume; this script only re-aggregates and re-plots).

Changes vs the previous revision:
  * figures 3-6 (tanh formulation + weak-optimizer) drop Cutoff GN entirely;
  * figures 5-6 (weak-optimizer) now also include the energy baseline(s);
  * figures 3-4 use a 3-column semantic legend layout:
      col1: baseline (row2 empty placeholder)
      col2: Ridge GN strong / weak
      col3: DSGNAR strong / weak
  * ReLU^3 figures are intentionally NOT regenerated here (unaffected).
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
import matplotlib.lines as mlines

ROOT = Path("results/reference/smooth")
if not ROOT.exists():
    raise SystemExit(
        "Reference results are not installed.\n\n"
        "Download/extract the Beyond PINNs reference-results archive into:\n"
        "    results/reference/\n\n"
        f"(expected to find data at {ROOT.resolve()})\n"
        "See docs/REPRODUCIBILITY.md or run scripts/download_reference_results.sh."
    )
print("Using experiment root:", ROOT.resolve())

OUT_DIR = Path("figures/energy_weak")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROTOCOL_OVERRIDE = {"energy_ng_reference": "reference"}

def seed_dirs(problem, activation, method):
    protocol = PROTOCOL_OVERRIDE.get(method, "matched")
    pat = str(ROOT / problem / protocol / activation / method / "seed_*" / "history.npz")
    return sorted(glob.glob(pat))

def load_histories(problem, activation, method):
    out = {}
    for p in seed_dirs(problem, activation, method):
        seed = int(Path(p).parent.name.split("_")[1])
        d = np.load(p)
        out[seed] = {k: np.asarray(d[k]) for k in ("iteration", "relative_l2", "relative_h1", "wallclock_optim")}
    return out

def median_iqr_curve(histories, xkey, ykey):
    """Cross-seed median/Q1/Q3 aggregation.

    Iteration axis (xkey == "iteration"): last-observation-carried-forward
    (LOCF) -- each seed's value is held constant past its own termination
    iteration, extended out to the longest-running seed. No truncation to
    the shortest seed, no shrinking sample size.

    Wall-clock axis (any other xkey, e.g. "wallclock_optim"): the x-axis
    itself is elapsed time and is not shared across seeds, so index-based
    LOCF would distort it (a seed that finished early would keep
    contributing a frozen, small elapsed-time value at every later index,
    pulling the aggregate x-position left even though still-running seeds
    used much more real time to reach that later data point). Instead,
    resample onto a shared time grid spanning [0, max over seeds of that
    seed's own true final wallclock], evaluating each seed as a
    right-continuous step function of elapsed time, held flat after its own
    last recorded timestamp -- no seed's clock is ever advanced past the
    time it actually stopped, and no elapsed time is invented. See
    docs/PROVENANCE.md section 5.2/5.3 / src/beyond_pinns/aggregation.py.
    """
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
    return x, np.median(Y, axis=0), np.percentile(Y, 25, axis=0), np.percentile(Y, 75, axis=0)

def final_stats(histories):
    l2 = np.array([h["relative_l2"][-1] for h in histories.values()])
    h1 = np.array([h["relative_h1"][-1] for h in histories.values()])
    wc = np.array([h["wallclock_optim"][-1] for h in histories.values()])
    it = np.array([h["iteration"][-1] for h in histories.values()])
    return l2, h1, wc, it

COLOR = {
    "hao_gn_deep_ritz": "#1b9e77",
    "energy_ng_reference": "#1b9e77",
    "energy_ng_matched": "#66c2a5",
    "plain_gn_lm": "#d95f02",
    "amstramgram": "#66a61e",
    "dsgnar": "#e7298a",
}
def style_for(method):
    if method in ("hao_gn_deep_ritz", "energy_ng_reference", "energy_ng_matched"):
        return COLOR[method], "-"
    regime = "strong" if "strong" in method else ("weak" if "weak" in method else None)
    ls = "--" if regime == "strong" else "-"
    if "plain_gn_" in method and "lm" in method:
        return COLOR["plain_gn_lm"], ls
    if "amstramgram" in method:
        return COLOR["amstramgram"], ls
    if "dsgnar" in method:
        return COLOR["dsgnar"], ls
    return "0.4", "-"

FLOOR = 1e-16

mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": "0.3",
    "axes.grid": True, "grid.color": "0.85", "grid.linewidth": 0.6,
    "font.size": 11, "axes.labelsize": 12, "legend.fontsize": 10,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.7,
})

def clip(y):
    return np.clip(y, FLOOR, None)

def plot_panels(axes, problem, activation, methods, label_map):
    """methods: list of method keys; draws widest-IQR-last so no band gets
    visually buried (matches the earlier fix for the bimodal Energy-NG case)."""
    panels = [("relative_l2", "iteration", axes[0, 0], r"relative $L^2$ error", "iteration"),
              ("relative_h1", "iteration", axes[0, 1], r"relative $H^1$ error", "iteration"),
              ("relative_l2", "wallclock_optim", axes[1, 0], r"relative $L^2$ error", "optimization wall-clock [s]"),
              ("relative_h1", "wallclock_optim", axes[1, 1], r"relative $H^1$ error", "optimization wall-clock [s]")]
    for ykey, xkey, ax, ylabel, xlabel in panels:
        curves = {}
        for m in methods:
            hs = load_histories(problem, activation, m)
            if hs:
                curves[m] = median_iqr_curve(hs, xkey, ykey)
        fill_order = sorted(curves, key=lambda m: np.max(clip(curves[m][3])) - np.min(clip(curves[m][2])))
        for m in fill_order:
            x, med, q1, q3 = curves[m]
            color, ls = style_for(m)
            ax.fill_between(x, clip(q1), clip(q3), color=color, alpha=0.18, lw=0)
        for m in methods:
            if m not in curves:
                continue
            x, med, q1, q3 = curves[m]
            color, ls = style_for(m)
            ax.plot(x, clip(med), color=color, linestyle=ls, label=label_map[m])
        ax.set_yscale("log"); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)

def save(fig, out_stem):
    fig.savefig(OUT_DIR / f"{out_stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{out_stem}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", OUT_DIR / f"{out_stem}.pdf", "and .png")

def build_3col_legend_figure(problem, activation, baseline_methods, ridge_methods, dsgnar_methods,
                              label_map, out_stem):
    """baseline_methods: list of 1 or 2 method keys shown in column 1 (energy baseline(s));
    ridge_methods: [strong, weak]; dsgnar_methods: [strong, weak]."""
    methods = list(baseline_methods) + list(ridge_methods) + list(dsgnar_methods)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    plot_panels(axes, problem, activation, methods, label_map)

    # ---- 3-semantic-column legend, column-major fill ----
    handles_by_method = {}
    for line in axes[0, 1].get_lines():
        handles_by_method[line.get_label()] = line
    col1 = [label_map[m] for m in baseline_methods]
    while len(col1) < 2:
        col1.append(None)  # empty placeholder row
    col2 = [label_map[m] for m in ridge_methods]
    col3 = [label_map[m] for m in dsgnar_methods]
    ordered_labels = col1 + col2 + col3
    handles, labels = [], []
    blank = mlines.Line2D([], [], color="none")
    for lbl in ordered_labels:
        if lbl is None:
            handles.append(blank); labels.append("")
        else:
            handles.append(handles_by_method[lbl]); labels.append(lbl)
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=3, frameon=True)
    fig.tight_layout(rect=[0.02, 0.1, 1, 0.94])
    save(fig, out_stem)
    return methods

def build_simple_legend_figure(problem, activation, methods, label_map, out_stem, ncol=5):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    plot_panels(axes, problem, activation, methods, label_map)
    handles, labels = axes[0, 1].get_legend_handles_labels()
    seen = set(); H = []; L = []
    for h, l in zip(handles, labels):
        if l not in seen:
            H.append(h); L.append(l); seen.add(l)
    fig.legend(H, L, loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=min(len(L), ncol), frameon=True)
    fig.tight_layout(rect=[0.02, 0.09, 1, 0.94])
    save(fig, out_stem)


def report_group(name, problem, activation, methods, label_map):
    print(f"\n--- {name}: {problem}/{activation} ---")
    print(f"{'method':32s} {'n':3s} {'iters':12s} {'medL2':>10s} {'Q1L2':>10s} {'Q3L2':>10s} "
          f"{'medH1':>10s} {'Q1H1':>10s} {'Q3H1':>10s} {'medWC':>10s}")
    rows = {}
    for m in methods:
        hs = load_histories(problem, activation, m)
        if not hs:
            print(f"{label_map.get(m,m):32s}  MISSING"); continue
        l2, h1, wc, it = final_stats(hs)
        row = dict(n=len(hs), it_min=int(it.min()), it_max=int(it.max()),
                   medl2=np.median(l2), q1l2=np.percentile(l2, 25), q3l2=np.percentile(l2, 75),
                   medh1=np.median(h1), q1h1=np.percentile(h1, 25), q3h1=np.percentile(h1, 75), medwc=np.median(wc))
        rows[m] = row
        it_str = f"{row['it_min']}" if row['it_min'] == row['it_max'] else f"{row['it_min']}-{row['it_max']}"
        print(f"{label_map.get(m,m):32s} {row['n']:3d} {it_str:12s} {row['medl2']:10.3e} {row['q1l2']:10.3e} {row['q3l2']:10.3e} "
              f"{row['medh1']:10.3e} {row['q1h1']:10.3e} {row['q3h1']:10.3e} {row['medwc']:10.2f}")
    return rows

ALL_ROWS = {}

# ==================================================================
# Figure 3: linear tanh formulation (3-column legend)
# ==================================================================
labels3 = {
    "hao_gn_deep_ritz": "GN Deep Ritz (baseline)",
    "plain_gn_strong_lm": "Ridge GN (strong)",
    "plain_gn_weak_lm": "Ridge GN (weak)",
    "dsgnar_strong": "DSGNAR (strong)",
    "dsgnar_weak": "DSGNAR (weak)",
}
methods3 = build_3col_legend_figure(
    "linear", "tanh",
    baseline_methods=["hao_gn_deep_ritz"],
    ridge_methods=["plain_gn_strong_lm", "plain_gn_weak_lm"],
    dsgnar_methods=["dsgnar_strong", "dsgnar_weak"],
    label_map=labels3, out_stem="linear_tanh_formulation_errors")
ALL_ROWS["3 linear/tanh formulation"] = report_group("3 linear/tanh formulation", "linear", "tanh", methods3, labels3)

# ==================================================================
# Figure 4: nonlinear tanh formulation (3-column legend, 2 baselines)
# ==================================================================
labels4 = {
    "energy_ng_reference": "Energy NG (baseline)",
    "energy_ng_matched": "Energy NG (matched)",
    "plain_gn_strong_lm": "Ridge GN (strong)",
    "plain_gn_weak_lm": "Ridge GN (weak)",
    "dsgnar_strong": "DSGNAR (strong)",
    "dsgnar_weak": "DSGNAR (weak)",
}
methods4 = build_3col_legend_figure(
    "nonlinear", "tanh",
    baseline_methods=["energy_ng_reference", "energy_ng_matched"],
    ridge_methods=["plain_gn_strong_lm", "plain_gn_weak_lm"],
    dsgnar_methods=["dsgnar_strong", "dsgnar_weak"],
    label_map=labels4, out_stem="nonlinear_tanh_formulation_errors")
ALL_ROWS["4 nonlinear/tanh formulation"] = report_group("4 nonlinear/tanh formulation", "nonlinear", "tanh", methods4, labels4)

# ==================================================================
# Figure 5: linear tanh weak-optimizer (baseline + Ridge GN + AMStramGRAM + DSGNAR)
# ==================================================================
labels5 = {
    "hao_gn_deep_ritz": "GN Deep Ritz (baseline)",
    "plain_gn_weak_lm": "Ridge GN",
    "amstramgram_weak": "AMStramGRAM",
    "dsgnar_weak": "DSGNAR",
}
methods5 = list(labels5.keys())
build_simple_legend_figure("linear", "tanh", methods5, labels5, "linear_tanh_weak_optimizers_errors", ncol=4)
ALL_ROWS["5 linear/tanh weak optimizers"] = report_group("5 linear/tanh weak optimizers", "linear", "tanh", methods5, labels5)

# ==================================================================
# Figure 6: nonlinear tanh weak-optimizer (2 baselines + Ridge GN + AMStramGRAM + DSGNAR)
# ==================================================================
labels6 = {
    "energy_ng_reference": "Energy NG (baseline)",
    "energy_ng_matched": "Energy NG (matched)",
    "plain_gn_weak_lm": "Ridge GN",
    "amstramgram_weak": "AMStramGRAM",
    "dsgnar_weak": "DSGNAR",
}
methods6 = list(labels6.keys())
build_simple_legend_figure("nonlinear", "tanh", methods6, labels6, "nonlinear_tanh_weak_optimizers_errors", ncol=5)
ALL_ROWS["6 nonlinear/tanh weak optimizers"] = report_group("6 nonlinear/tanh weak optimizers", "nonlinear", "tanh", methods6, labels6)

# ==================================================================
# Old-vs-new medians for the extended/restarted methods
# ==================================================================
import json as _json
OLD_VALUES = {
    # (problem, method): (old_medL2, old_medH1, old_medWC, old_n_iters)
    ("linear", "plain_gn_strong_lm"): (2.736e-7, 2.696e-7, 29.01, 300),
    ("nonlinear", "plain_gn_strong_lm"): (1.093e-7, 1.462e-7, 28.68, 300),
}
print("\n" + "=" * 100)
print("OLD (300 iters) vs NEW (1000 iters, checkpoint-resumed) medians -- plain_gn_strong_lm")
print("=" * 100)
latex_rows = []
for problem in ("linear", "nonlinear"):
    key = ("3 linear/tanh formulation" if problem == "linear" else "4 nonlinear/tanh formulation")
    new_row = ALL_ROWS[key]["plain_gn_strong_lm"]
    old_l2, old_h1, old_wc, old_n = OLD_VALUES[(problem, "plain_gn_strong_lm")]
    rel_change_h1 = (new_row["medh1"] - old_h1) / old_h1 * 100
    rel_change_l2 = (new_row["medl2"] - old_l2) / old_l2 * 100
    print(f"{problem:9s} plain_gn_strong_lm: "
          f"L2 {old_l2:.3e} -> {new_row['medl2']:.3e} ({rel_change_l2:+.1f}%)   "
          f"H1 {old_h1:.3e} -> {new_row['medh1']:.3e} ({rel_change_h1:+.1f}%)   "
          f"wc {old_wc:.2f}s -> {new_row['medwc']:.2f}s   n_iters {old_n} -> {new_row['it_max']}")
    latex_rows.append((problem, old_l2, new_row["medl2"], old_h1, new_row["medh1"], old_wc, new_row["medwc"]))

print("\nLaTeX table (corrected medians for plain_gn_strong_lm):")
print(r"\begin{tabular}{lrrrrrr}")
print(r"\toprule")
print(r"Problem & \multicolumn{2}{c}{rel. $L^2$} & \multicolumn{2}{c}{rel. $H^1$} & \multicolumn{2}{c}{wallclock [s]} \\")
print(r" & old (300it) & new (1000it) & old (300it) & new (1000it) & old & new \\")
print(r"\midrule")
for problem, ol2, nl2, oh1, nh1, owc, nwc in latex_rows:
    print(f"{problem} & {ol2:.3e} & {nl2:.3e} & {oh1:.3e} & {nh1:.3e} & {owc:.1f} & {nwc:.1f} \\\\")
print(r"\bottomrule")
print(r"\end{tabular}")

# ==================================================================
# Final answers to the 5 audit questions
# ==================================================================
print("\n" + "#" * 100)
print("FINAL AUDIT ANSWERS")
print("#" * 100)
print("""
1. Why did some existing experiments stop at 300 iterations?
   Their OWN config.json (configuration.matched.residual_iterations) was
   literally set to 300 at the time those runs were launched -- this is the
   value stored inside each run's saved config snapshot, not a runtime
   early-stop. hao_gn_deep_ritz / energy_ng_matched use a SEPARATE config key
   (matched.energy_iterations = 1000) and were unaffected. DSGNAR's variable,
   seed-dependent stop points (163-293) are a genuinely different mechanism:
   its own adaptive trust-region collapse, not a fixed iteration budget.

2. Was this intentional or a configuration inconsistency?
   A configuration inconsistency. residual_iterations=300 was the ORIGINAL
   project default (later changed to 1000 by the user mid-project), and
   plain_gn_strong_lm / amstramgram_weak were never rerun under the corrected
   default -- unlike plain_gn_weak_lm / plain_gn_weak_tsvd, which WERE
   rerun at 1000 iterations during the earlier regularization-tuning phase of
   this project (see prior conversation). This was a genuine gap, not a
   deliberate choice.

3. Which runs, if any, were extended to 1000?
   plain_gn_strong_lm, both linear and nonlinear tanh, all 10 seeds each
   (20 runs total). amstramgram_weak was deliberately NOT extended, per
   explicit user instruction: its trajectory visibly converges well before
   300 iterations in the existing plots, so its 300-iteration stop is being
   treated as a legitimate practical convergence point rather than a
   configuration bug, and its original 300-iteration data is kept as-is.
   dsgnar_strong/weak were left untouched (adaptive early-stopping is a
   genuine algorithmic behavior, not something to "fix" by forcing more
   iterations). energy_ng_reference (101 iterations) is the literal
   Muller-Zeinhofer public reference protocol and was correctly left
   untouched.

4. Were they resumed or restarted?
   RESUMED from the iter_000300 checkpoint, not restarted. This is safe here
   specifically because the plain Levenberg-Marquardt/ridge GN step is a
   PURE FUNCTION of the current parameters (a fresh SVD of J(p) every call,
   no persistent optimizer state) -- so continuing from checkpoint is
   bit-identical to an uninterrupted 1000-iteration run. This was verified
   empirically for all 20 runs: recomputing relative_h1 at the checkpointed
   params reproduced the originally-recorded iteration-300 value with
   rel_diff=0.00e+00 (bit-exact) in every case before any new iterations
   were appended.

5. Did extending them materially change any conclusion or numerical value?
   Yes, quantitatively (H1 improved by roughly one order of magnitude in
   both problems -- see the old-vs-new table above), but NOT qualitatively:
   plain_gn_strong_lm's ranking relative to the other methods in these
   figures (DSGNAR, energy baselines, Ridge GN weak) is unchanged -- it
   remains a mid-accuracy method, well behind DSGNAR and the energy
   baselines, and the fixed-regularization Ridge GN sensitivity story from
   earlier in this project is unaffected (only the STRONG-residual LM run
   was under-trained; nothing about its regularization value changed).
   Manuscript medians quoted for plain_gn_strong_lm should be updated to the
   NEW values above; all other quoted medians (DSGNAR, energy baselines,
   Ridge GN weak) are unaffected since those runs were untouched.
""")
