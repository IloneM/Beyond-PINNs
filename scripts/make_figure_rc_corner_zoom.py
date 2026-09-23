#!/usr/bin/env python
"""RC (section_4) multilevel corner-zoom diagnostics.

READS ONLY from:
    petrov_galerkin_results/section_4/s4_proj_random_hats/seed_001/

No training and no new seed selection are performed.  The representative-seed
logic is re-run only as a consistency check.

Final outputs:
    figures/petrov/section_4_rc_hybrid_difference_corner.pdf/.png
    figures/petrov/section_4_rc_fe_compensator_corner.pdf/.png

The final figures use a callout chain

    full domain -> zoom 1 -> zoom 2 -> zoom 3

with two connector branches at every level:
    upper-left crop corner -> upper-left destination corner
    lower-left crop corner -> lower-left destination corner.

The zoom panels use linear interpolation of the already-saved field values on
a finer display grid.  This improves presentation only; it is NOT a new PDE or
model evaluation and therefore does not create additional scientific
resolution.

A third zoom can be enabled simply by adding another window to FINAL_WINDOWS,
but the default stops at two zoom levels because increasingly tight views
would otherwise magnify interpolated data rather than genuinely new model
information.

Provenance note -- zoom/callout technique (Rectangle + ConnectionPatch
coupling a full view to a zoom panel): this pattern is ADAPTED from
Andrea Combette's SIREN-Init repository (github.com/AndreaCombette/SIREN-Init,
file src/image_fitting.ipynb, commit 95d25fd108), accompanying the paper
Combette, Venaille & Pustelnik, "A new initialisation to Control Gradients in
Sinusoidal Neural network" (arXiv:2512.06427). It is an adaptation of the
general plotting pattern for a different domain (a continuous PDE field with
an explicit physical zoom window, not a discrete pixel image with a
pixel-fraction crop heuristic), not a copy of SIREN-Init's source text. An
earlier draft of this figure mistakenly cited this material's origin as
"NG-INR"; that attribution was incorrect and is corrected here.

Redistribution status: PERMISSION GRANTED -- Andrea Combette gave explicit
written permission (2026-09-22) for this release to include this adapted
material. See docs/PROVENANCE.md section 4 for the full record.

"""

import glob
import json
from pathlib import Path

import os
os.environ.pop("MPLBACKEND", None)  # must happen BEFORE importing matplotlib: an inherited
# invalid value (e.g. Jupyter's 'module://matplotlib_inline.backend_inline') crashes
# matplotlib's own __init__.py at import time, before any mpl.use() call could help.
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import ConnectionPatch, Rectangle
import numpy as np
from scipy.interpolate import LinearNDInterpolator


ROOT = Path("results/reference/general")
OUT_DIR = Path("figures/petrov")
PREVIEW_DIR = OUT_DIR / "rc_corner_zoom_previews"

if not ROOT.exists():
    raise SystemExit(
        "Reference results are not installed.\n\n"
        "Download/extract the Beyond PINNs reference-results archive into:\n"
        "    results/reference/\n\n"
        f"(expected to find data at {ROOT.resolve()})\n"
        "See docs/REPRODUCIBILITY.md or run scripts/download_reference_results.sh."
    )

OUT_DIR.mkdir(parents=True, exist_ok=True)
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 10,
        "axes.labelsize": 9.5,
        "axes.titlesize": 10.5,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

SECTION = "section_4"
RUN = "s4_proj_random_hats"


def representative_seed(section, run):
    seeds = {}
    pattern = ROOT / section / run / "seed_*" / "history.npz"
    for p in sorted(glob.glob(str(pattern))):
        seed = int(Path(p).parent.name.split("_")[1])
        seeds[seed] = float(np.load(p)["relative_h1"][-1])
    med = np.median(list(seeds.values()))
    return min(seeds, key=lambda s: abs(seeds[s] - med)), seeds


seed, seeds = representative_seed(SECTION, RUN)
print(f"{SECTION}/{RUN}: representative seed = {seed} (expect 1)")
assert seed == 1, f"Representative seed changed unexpectedly: {seed}"

d = np.load(ROOT / SECTION / RUN / f"seed_{seed:03d}" / "final_solution.npz")
pts = d["points"]
u_total = d["u_total"]
u_exact = d["u_exact"]
w_n = d["u_fem"]
hybrid_diff = u_total - u_exact


def to_grid(points, values, xs, ys):
    nx, ny = len(xs), len(ys)
    ix = np.clip(np.searchsorted(xs, points[:, 0]), 0, nx - 1)
    iy = np.clip(np.searchsorted(ys, points[:, 1]), 0, ny - 1)
    grid = np.full((ny, nx), np.nan)
    grid[iy, ix] = values
    return grid


xs = np.unique(pts[:, 0])
ys = np.unique(pts[:, 1])
extent = [xs.min(), xs.max(), ys.min(), ys.max()]

diff_grid = to_grid(pts, hybrid_diff, xs, ys)
w_grid = to_grid(pts, w_n, xs, ys)

# Interpolation is only for display inside magnified panels.
diff_interp = LinearNDInterpolator(pts, hybrid_diff, fill_value=np.nan)
w_interp = LinearNDInterpolator(pts, w_n, fill_value=np.nan)


def make_fine_zoom(interpolator, zoom_window, n=500):
    """Interpolate saved values onto a fine visualization grid.

    The excised lower-right quadrant of the L-shaped domain is explicitly
    re-masked after interpolation.
    """
    (zx0, zx1), (zy0, zy1) = zoom_window
    xq = np.linspace(zx0, zx1, n)
    yq = np.linspace(zy0, zy1, n)
    Xq, Yq = np.meshgrid(xq, yq)
    Zq = np.asarray(interpolator(Xq, Yq), dtype=float)

    # Omega_L = (-1,1)^2 \ ([0,1) x (-1,0]).
    hole = (Xq >= 0.0) & (Yq < 0.0)
    Zq[hole] = np.nan
    return Zq, [zx0, zx1, zy0, zy1]



def nice_ticks(lo, hi, is_full=False):
    """Small, explicit coordinate graduations for the full and zoom panels."""
    if is_full:
        # For the present RC domain this gives -1, -0.5, 0, 0.5, 1.
        return np.linspace(lo, hi, 5)

    # Symmetric zooms around the reentrant corner: show both boundaries and 0.
    if lo < 0.0 < hi and np.isclose(abs(lo), abs(hi)):
        return np.array([lo, 0.0, hi])

    return np.linspace(lo, hi, 3)


def format_axis_coordinates(ax, xlim, ylim, *, is_full=False):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks(nice_ticks(*xlim, is_full=is_full))
    ax.set_yticks(nice_ticks(*ylim, is_full=is_full))
    ax.set_xlabel(r"$x$", labelpad=2)
    ax.set_ylabel(r"$y$", labelpad=2)
    ax.tick_params(direction="out", length=2.5, pad=2)


def add_crop_rectangle(ax, window, *, color="red", linewidth=1.5):
    (zx0, zx1), (zy0, zy1) = window
    rect = Rectangle(
        (zx0, zy0),
        zx1 - zx0,
        zy1 - zy0,
        fill=False,
        edgecolor=color,
        linewidth=linewidth,
        zorder=12,
    )
    ax.add_patch(rect)
    return rect


def connect_crop_to_zoom(
    fig,
    src_ax,
    dst_ax,
    crop_window,
    *,
    color="red",
    linewidth=1.25,
):
    """Connect both left crop corners to both left destination corners."""
    (zx0, _zx1), (zy0, zy1) = crop_window

    # Upper-left crop corner -> upper-left zoom-panel corner.
    fig.add_artist(
        ConnectionPatch(
            xyA=(zx0, zy1),
            coordsA=src_ax.transData,
            xyB=(0.0, 1.0),
            coordsB=dst_ax.transAxes,
            color=color,
            linewidth=linewidth,
            alpha=0.95,
            clip_on=False,
            zorder=15,
        )
    )

    # Lower-left crop corner -> lower-left zoom-panel corner.
    fig.add_artist(
        ConnectionPatch(
            xyA=(zx0, zy0),
            coordsA=src_ax.transData,
            xyB=(0.0, 0.0),
            coordsB=dst_ax.transAxes,
            color=color,
            linewidth=linewidth,
            alpha=0.95,
            clip_on=False,
            zorder=15,
        )
    )


def render_multilevel_corner_zoom(
    grid,
    interpolator,
    cmap,
    norm,
    title_quantity,
    zoom_windows,
    *,
    out_path=None,
    cbar_label="",
    zoom_resolution=650,
    full_resolution=850,
    colorbar_ticks=None,
):
    """Render full domain followed by one or more nested corner zooms.

    Each zoom level is selected by a red rectangle in the preceding panel.
    Two red connector branches link the rectangle's upper-left/lower-left
    corners to the corresponding upper-left/lower-left corners of the next
    zoom panel.
    """
    n_zoom = len(zoom_windows)
    n_panels = 1 + n_zoom

    # Last narrow column is reserved exclusively for the colorbar.
    fig_width = 3.55 * n_panels + 0.75
    fig = plt.figure(figsize=(fig_width, 4.2), constrained_layout=True)
    width_ratios = [1.0] * n_panels + [0.045]
    gs = fig.add_gridspec(
        1,
        n_panels + 1,
        width_ratios=width_ratios,
        wspace=0.06,
    )
    axes = [fig.add_subplot(gs[0, i]) for i in range(n_panels)]
    cax = fig.add_subplot(gs[0, -1])

    # -----------------------
    # Full-domain panel
    # -----------------------
    ax0 = axes[0]
    full_window = ((extent[0], extent[1]), (extent[2], extent[3]))
    full_grid, full_extent = make_fine_zoom(
        interpolator, full_window, n=full_resolution
    )
    ax0.imshow(
        full_grid,
        origin="lower",
        extent=full_extent,
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )
    ax0.set_aspect("equal")
    format_axis_coordinates(
        ax0,
        (extent[0], extent[1]),
        (extent[2], extent[3]),
        is_full=True,
    )
    ax0.set_title(rf"{title_quantity} on $[-1,1]^2$", fontsize=13.5)

    # -----------------------
    # Zoom chain
    # -----------------------
    source_ax = ax0
    final_im = None

    for level, (ax, window) in enumerate(zip(axes[1:], zoom_windows), start=1):
        # Crop box lives in the PREVIOUS panel.
        add_crop_rectangle(source_ax, window)

        fine_grid, fine_extent = make_fine_zoom(
            interpolator,
            window,
            n=zoom_resolution,
        )

        final_im = ax.imshow(
            fine_grid,
            origin="lower",
            extent=fine_extent,
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
        )
        ax.set_aspect("equal")

        (zx0, zx1), (zy0, zy1) = window
        format_axis_coordinates(ax, (zx0, zx1), (zy0, zy1), is_full=False)

        half_width = max(abs(zx0), abs(zx1), abs(zy0), abs(zy1))
        ax.set_title(
            rf"{title_quantity} on $[-{half_width:g},{half_width:g}]^2$",
            fontsize=13.5,
        )

        # Red frame around each enlarged panel.
        for spine in ax.spines.values():
            spine.set_edgecolor("red")
            spine.set_linewidth(1.5)
            spine.set_visible(True)

        connect_crop_to_zoom(fig, source_ax, ax, window)

        # Do not place the star in the zoomed panels: it obscures precisely
        # the local field structure we want to inspect.
        source_ax = ax

    # A dedicated colorbar axis gives consistent placement across quantities.
    if final_im is None:
        raise RuntimeError("At least one zoom window is required.")

    cb = fig.colorbar(final_im, cax=cax, ticks=colorbar_ticks)
    cb.set_label(cbar_label, rotation=90, labelpad=8)
    cb.ax.tick_params(labelsize=8.5)

    if out_path is not None:
        fig.savefig(f"{out_path}.pdf", bbox_inches="tight")
        fig.savefig(f"{out_path}.png", dpi=240, bbox_inches="tight")
        print(f"saved {out_path}.pdf and .png")

    return fig


# ======================================================================
# Spatial diagnostics retained from the previous script.
# ======================================================================
r = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
i_diff_max = int(np.argmax(np.abs(hybrid_diff)))
i_w_max = int(np.argmax(np.abs(w_n)))

diag = dict(
    max_abs_hybrid_diff=float(np.abs(hybrid_diff)[i_diff_max]),
    max_abs_hybrid_diff_loc=[
        float(pts[i_diff_max, 0]),
        float(pts[i_diff_max, 1]),
    ],
    max_abs_hybrid_diff_r=float(r[i_diff_max]),
    max_abs_w_n=float(np.abs(w_n)[i_w_max]),
    max_abs_w_n_loc=[
        float(pts[i_w_max, 0]),
        float(pts[i_w_max, 1]),
    ],
    max_abs_w_n_r=float(r[i_w_max]),
)

for rad in (0.1, 0.2, 0.3):
    mask = r <= rad
    diag[f"mean_abs_hybrid_diff_r<={rad}"] = float(np.abs(hybrid_diff[mask]).mean())
    diag[f"mean_abs_w_n_r<={rad}"] = float(np.abs(w_n)[mask].mean())

mask_far = r > 0.3
diag["mean_abs_hybrid_diff_r>0.3"] = float(np.abs(hybrid_diff[mask_far]).mean())
diag["mean_abs_w_n_r>0.3"] = float(np.abs(w_n)[mask_far].mean())

near = r <= 0.3
if near.sum() > 2:
    diag["pearson_corr_absdiff_vs_absW_r<=0.3"] = float(
        np.corrcoef(np.abs(hybrid_diff[near]), np.abs(w_n)[near])[0, 1]
    )

with open(PREVIEW_DIR / "rc_corner_diagnostics.json", "w") as fh:
    json.dump(diag, fh, indent=2)


# ======================================================================
# Common signed colormap / normalization
# ======================================================================
# Both quantities are signed and must use the same visual scale.
cmap_common = plt.get_cmap("RdBu_r").copy()
cmap_common.set_bad("0.92")

raw_common_max = max(
    float(np.nanmax(np.abs(hybrid_diff))),
    float(np.nanmax(np.abs(w_n))),
)

# Round upward to one significant decimal decade:
# e.g. 7.85e-3 -> 8e-3.
_exp = np.floor(np.log10(raw_common_max))
_step = 10.0 ** _exp
COMMON_MAX = float(np.ceil(raw_common_max / _step) * _step)

common_norm = TwoSlopeNorm(
    vcenter=0.0,
    vmin=-COMMON_MAX,
    vmax=COMMON_MAX,
)
common_ticks = np.linspace(-COMMON_MAX, COMMON_MAX, 5)

print(
    f"Common signed color scale: raw max={raw_common_max:.6e}, "
    f"rounded max={COMMON_MAX:.6e}"
)


# ======================================================================
# Final multilevel zoom design
# ======================================================================
# The RC mesh audit confirmed that +/-0.05 is well resolved:
# 4 graded radial layers, about 100 vertices, and 224 intersecting elements.
FINAL_WINDOWS = [
    ((-0.35, 0.35), (-0.35, 0.35)),
    ((-0.12, 0.12), (-0.12, 0.12)),
    ((-0.05, 0.05), (-0.05, 0.05)),
]

render_multilevel_corner_zoom(
    diff_grid,
    diff_interp,
    cmap_common,
    common_norm,
    r"$v_{\theta,n}-u$",
    FINAL_WINDOWS,
    out_path=OUT_DIR / "section_4_rc_hybrid_difference_corner",
    cbar_label=r"$v_{\theta,n}-u$",
    zoom_resolution=650,
    full_resolution=850,
    colorbar_ticks=common_ticks,
)

render_multilevel_corner_zoom(
    w_grid,
    w_interp,
    cmap_common,
    common_norm,
    r"$w_n$",
    FINAL_WINDOWS,
    out_path=OUT_DIR / "section_4_rc_fe_compensator_corner",
    cbar_label=r"$w_n$",
    zoom_resolution=650,
    full_resolution=850,
    colorbar_ticks=common_ticks,
)

print("\nFinal signed multilevel RC corner figures written to", OUT_DIR.resolve())
