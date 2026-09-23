# Provenance

This document accounts for every piece of code in this repository that did
not originate as original work by the paper's authors for this repository:
external dependencies, code adapted from other projects, and corrections
made to the historical research code during the release process. It is the
authoritative record referenced by `LICENSE` and by the Scientific Fidelity
Audit in the Phase B report.

## 1. License

The repository-level `LICENSE` (MIT) applies to code written by the paper's
authors (Nilo Schwencke, Roland Maier) in this repository. It does **not**
relicense any third-party component described below, each of which retains
its own terms.

No file in the historical research tree carried a license header, and the
research tree itself had no LICENSE file. This release adds MIT for the
authors' own code; nothing here silently reinterprets pre-existing
third-party code as MIT.

## 2. Summary: what's what

This release's non-original code falls into exactly four categories. Every
component in sections 2-3 below is tagged with one or more of these; none of
this implies public GitHub availability constitutes an open-source license
(see section 1) -- categories (a)/(c)/(d) name dependencies whose own
license status is independently tracked per-component below, most of which
have **no license at all**.

| # | Category | Meaning |
|---|---|---|
| (a) | Historical dependency | Code the paper's own experiments actually ran against, whether or not any of it survives into this release |
| (b) | Independently reimplemented | Rewritten from a mathematical/behavioral specification for this release, with no source text or comments carried over, and verified equivalent by regression tests |
| (c) | Genuinely external, optional | Not needed to reproduce the paper's core manuscript results; only for an optional reference baseline/protocol |
| (d) | Pinned external dependency, still required | Kept external (not vendored, not reimplemented) because it supplies substantial scientific functionality, not a generic helper; installed at an exact pinned commit |

| Component | Category | Where |
|---|---|---|
| `pinn.optimiser.Optimiser`/`VectorOptimiser`&`EnergyOptimiser`'s internals/`operators.laplacian` | (a) + (d) | section 3.1 |
| `has_passed_minimum` (historically copied from `pinn.trainer64`) | (a) + (b) -> `detect_trend_reversal` | section 3.1.1 |
| `ngrad.models.mlp` | (a) + (b) -> `deep_mlp` | section 3.2 |
| `Natural-Gradient-PINNs-ICML23` full repo (Muller-Zeinhofer reference protocol only) | (c) | section 3.2 |
| GNDRM generic infra (`normal_init`/`shallow_network`/`GaussLegendrePiecewise`) | (a) + (b) -> `beyond_pinns.initialization`/`models`/`quadrature` | section 3.3 |
| GNDRM optimizer machinery (`jacobian_matrix`/`gn_direction`/`grid_line_search`) | (a) + (c) | section 3.3 |
| SIREN-Init zoom-callout pattern | (a), permission granted 2026-09-22 | section 4 |

## 3. External dependencies (not vendored)

### 3.1 `pinn` (DSGNAR optimizer) -- REQUIRED for Part B core reproduction

- Upstream: `github.com/IloneM/physics-informed-neural-networks`
- Commit used (full SHA): `8b30ec3961de76ed414dd0f918646dd6358d8fcf` (2026-07-05T11:07:46Z)
- License: **none found** (no LICENSE file, no license mention in the README)
- Role: supplies `pinn.optimiser.Optimiser` (DSGNAR) and `pinn.operators`,
  imported **unconditionally** at the top of
  `src/beyond_pinns/legacy/part_b_general.py` (`import pinn`,
  `from pinn.optimiser import Optimiser`, `from pinn import operators as op`,
  plus a second `from pinn.optimiser import (...)` deeper in the file for
  the sketched-Jacobian primitives). This makes `pinn` **required, not
  optional**, for every Part B benchmark (MS/JF/LS/RC) -- i.e. required for
  the paper's core manuscript reproduction, not merely an optional baseline.
  Part A does not import `pinn` at all.
- Not vendored, not a PyPI dependency: declared in `pyproject.toml` as a
  pinned direct Git dependency (PEP 508 VCS URL, exact commit above), so a
  normal `uv sync`/`pip install -e .` clones and builds it automatically --
  see `docs/REPRODUCIBILITY.md`. This is a reproducibility/install
  mechanism, equivalent in kind to a pinned manual clone or a Git submodule
  (same exact commit, same upstream URL, nothing vendored into this
  repository). **It is not a license**: it does not change `pinn`'s own
  no-license status (below), and this repository's MIT license does not
  extend to it.

**Exact symbol-level dependency map.** Before deciding whether any part of
`pinn` could be independently reimplemented (as was done for `ngrad` and for
`has_passed_minimum`, below), every `pinn.*` symbol actually used by
`src/beyond_pinns/legacy/part_b_general.py` was enumerated directly from the
source (`grep` for `pinn\.`/`from pinn`):

| Symbol | Import site | Used for | Classification |
|---|---|---|---|
| `pinn.optimiser.Optimiser` | `from pinn.optimiser import Optimiser` (top of file) | The top-level scalar-residual DSGNAR trust-region solver -- instantiated and run as-is in `run_dsgnar()` (`solver = Optimiser(**solver_kwargs)`) | Substantial -- the paper's own DSGNAR optimizer *is* this class |
| `pinn.optimiser._make_srct` | second `from pinn.optimiser import (...)`, deeper in the file | Builds the random parameter-sketch transform, reused by the locally-defined `VectorOptimiser`/`EnergyOptimiser` | Substantial -- core sketching primitive |
| `pinn.optimiser._apply_srct` | same | Applies that sketch to the Jacobian | Substantial |
| `pinn.optimiser._count_sketch` | same | Randomized count-sketch of the srct-sketched Jacobian/residual pair | Substantial |
| `pinn.optimiser._lift_update` | same | Lifts a sketched-space Newton step back to full parameter space | Substantial |
| `pinn.optimiser._solve_subproblems` | same | Per-probe-radius regularized Newton solve for the trust-region subproblem | Substantial |
| `pinn.optimiser._pchip` | same | PCHIP interpolation locating the target-rho crossing that selects the trust-region radius | Substantial |
| `pinn.optimiser.SolverState` | same | The optimizer's state namedtuple (radius/rho/lam/key) | Substantial -- structural, tied 1:1 to the above |
| `pinn.operators.laplacian` (`op.laplacian`) | `from pinn import operators as op` | The Laplacian differential operator building the strong-form PDE residual for the section_1/section_2 problems | Substantial -- a PDE-specific differential-operator implementation, not a generic numerical helper |

Every symbol used from `pinn` is either the paper's own optimizer
(`Optimiser`, used directly) or an internal building block of that *same*
sketched trust-region Gauss-Newton algorithm -- reused, by directly
importing pinn's private (`_`-prefixed) internals, to build two further
solver variants (`VectorOptimiser`, `EnergyOptimiser`, both defined locally
in `part_b_general.py`, per the file's own comment "Reuses pinn.optimiser's
building blocks verbatim; only the Jacobian ACQUISITION changes"). None of
this is a small, generic, drop-in-replaceable helper comparable to
`has_passed_minimum` (below) -- it *is* DSGNAR's actual sketching machinery,
and `op.laplacian` is likewise a PDE-specific operator, not a generic
utility.

**Conclusion: `pinn` stays external and pinned exactly; no part of it is
reimplemented for this release.** Independently reimplementing DSGNAR's own
sketched-trust-region internals would mean rewriting the paper's own
load-bearing optimizer from a black-box behavioral specification -- exactly
the "replace a scientific algorithm merely to remove the dependency" outcome
the release's validation policy warns against (see the `energy_ng_matched`
discussion in section 3.3). Unlike GNDRM, `pinn` is under the same authors'
own control, so pinning and documenting it (as already done above) fully
satisfies the release's reproducibility requirement without that risk.

#### 3.1.1 `has_passed_minimum` -> independently reimplemented as `detect_trend_reversal`

`src/beyond_pinns/legacy/part_b_general.py`'s `has_passed_minimum` function
carried the historical comment "(verbatim from pinn.trainer64)" -- a
~15-line heuristic (log-linear-regression slope/correlation check on a
loss-history window) copied from the `pinn` package into the Part B script
itself. Since `pinn` has no LICENSE, this release does not carry that copied
source.

**Status: independently reimplemented, not merely flagged.** Unlike the
symbols in section 3.1 above, this heuristic is small, generic, and fully
specified by its own mathematical behavior (a sliding-window log-linear
regression slope/correlation test), so it was rewritten from that
specification -- different variable names and structure throughout, no
`pinn` source text or comments reused -- as
`beyond_pinns.monitoring.detect_trend_reversal` (see that module's
docstring for the exact specification). `part_b_general.py` now imports
`detect_trend_reversal as has_passed_minimum` at every call site, so **no
copied `pinn` source remains anywhere in this release.**

Regression-tested in `tests/test_monitoring.py` against a locally
reproduced transcription of the historical formula (kept only for this
comparison, never shipped) across monotone-decreasing, post-minimum-rising,
flat, noisy, and short-window (`len(values) < window_size`) sequences, plus
5 seeds of random sequences and several additional edge cases (exactly at
`window_size`, zeros mixed with positive values) -- 21 cases, all matching
bit-for-bit (boolean equality, not a tolerance).

### 3.2 `ngrad` (Natural-Gradient-PINNs-ICML23) -- historical dependency, independently reimplemented; genuinely optional external repo remains for one reference protocol

- Upstream used historically: `github.com/IloneM/Natural-Gradient-PINNs-ICML23`
- Commit used (full SHA): `0a0055cb648bc41b9773e6855f6df8c8748cebc5` (2024-06-06T15:55:19Z)
- License: **none found**
- **Fork provenance, verified.** This is a fork of
  `github.com/MariusZeinhofer/Natural-Gradient-PINNs-ICML23` (the original
  ICML'23 "energy natural gradients" paper codebase). Checked via the GitHub
  API compare endpoint before treating the two as interchangeable: the
  `IloneM` fork has **3 own commits** not present in the `MariusZeinhofer`
  original (a JAX-version compatibility fix, a genuine bug fix, and a
  feature addition) -- the two repositories are **not equivalent**, and the
  historically-used `IloneM` fork at the pinned commit above is preserved
  rather than silently swapped for the "canonical" upstream.
- **Historical Part B role: `ngrad.models.mlp` was REQUIRED, not optional**
  (this corrected an earlier, wrong Part-A-only characterization of this
  dependency, found during final validation).
  `src/beyond_pinns/legacy/part_b_general.py` unconditionally imported
  `from ngrad.models import mlp` and used it directly as the core
  neural-network forward pass for the whole benchmark suite:
  `pre_model_1d = mlp(jnp.tanh)` and `pre_model_2d = mlp(jnp.tanh)`.
- **Status: independently reimplemented -- `ngrad` is no longer required for
  Part B.** `ngrad.models.mlp` is, on inspection, a completely generic,
  textbook N-hidden-layer MLP forward pass with an output bias (unlike
  GNDRM's 1-hidden-layer `shallow_network`, which has none). Its own file
  header explicitly comments "------Code from Jax documentation-----" for
  the accompanying `random_layer_params`/`init_params` helpers (not used by
  Part B, which already used this release's own `beyond_pinns.initialization`
  instead) -- i.e. even *ngrad's own authors* flag this specific code as
  copied from JAX's public documentation, not original/distinctive content.
  It was rewritten from that specification as `beyond_pinns.models.deep_mlp`
  -- different names/structure throughout (`hidden`/`pre_activation`/
  `output_weight`/`output_bias`, not `ngrad`'s own names), no `ngrad` source
  text or comments reused. `part_b_general.py` now imports
  `deep_mlp as mlp` in place of `from ngrad.models import mlp`, and the
  `sys.path.append(.../Natural-Gradient-PINNs-ICML23)` line the historical
  script used to make `ngrad` importable has been removed entirely.
  **The historical experiments used `ngrad.models.mlp` from
  Natural-Gradient-PINNs-ICML23. The public release contains an
  independently implemented equivalent. This release does not vendor or
  redistribute `ngrad` code.**
- **Equivalence verification** (same protocol as GNDRM's reimplementation,
  section 3.3.1): golden-fixture regression test (`tests/test_models.py`,
  fixture captured from a real `ngrad.models.mlp` run, no live `ngrad`
  needed) plus an optional live-comparison test
  (`tests/test_ngrad_equivalence.py`, skipped unless a local `ngrad`
  checkout is present) checking value, `d/dx`, and the full parameter
  Jacobian (`jacfwd`), scalar and batched (`vmap`), for 3 architectures x 2
  activations (tanh, relu^3) x 3 trial points + one batched call. Also run
  live against the real `ngrad.models.mlp` on a reference compute node
  (commit `0a0055c`, matching the pin above): value/`d/dx`/parameter-Jacobian
  agreement to `atol=1e-13` (float64) in every case. **End-to-end
  integration check**: an MS-hybrid smoke-training run of
  `part_b_general.py` with the substituted `deep_mlp` (and
  `detect_trend_reversal`, section 3.1.1) completed successfully on GPU with
  no errors, producing full results (`history.npz`, checkpoints, plots,
  section summaries) -- not just a unit-level match. No manuscript values
  were recomputed from this run; it verifies the substitution's fidelity,
  not a new result.
- **Part A role: still genuinely optional, external, and unchanged** -- only
  needed for the "Muller-Zeinhofer reference protocol"
  (`CONFIG["nonlinear"]["reference_protocol"]`), a stricter apples-to-apples
  comparison against the published `ngrad` implementation directly, distinct
  from `energy_ng_matched` (see section 3.3 below: `energy_ng_matched` is
  GNDRM-dependent, not `ngrad`-dependent, and stays optional for a different
  reason). Keeping the full `Natural-Gradient-PINNs-ICML23` repository as a
  pinned external dependency remains acceptable for this optional protocol
  specifically -- clone the pinned commit above (the verified `IloneM` fork,
  not `MariusZeinhofer`'s original) and add it to `PYTHONPATH` if this
  protocol is needed.

### 3.3 GaussNewtonDRM (GNDRM) -- Hao/Jin Deep-Ritz reference baseline

- Upstream: `github.com/Jinxl-pp/GaussNewtonDRM`
- Author: Xianlin Jin
- Exact commit: `9dd1ee8df8aca41c5f85157c2d7bf2afa12035ce` (2026-01-21, repo HEAD)
- License: **none** (no LICENSE file; GitHub's license-detection API returns
  404; confirmed by direct download of `tool/model.py`, `tool/quadrature.py`,
  `tool/gauss_newton.py` at that commit)
- Local modifications relative to upstream: **none** -- all three files used
  were byte-identical to that commit (verified by diff before any release
  work began).

**Historical usage** (in `energy_vs_weak_benchmarks_gn_strong_weak.py`):
imported unconditionally at module load time,

```python
from tool.model import shallow_network, normal_init
from tool.quadrature import GaussLegendrePiecewise
from tool.gauss_newton import jacobian_matrix, gn_direction, grid_line_search
```

Of these five names, two different roles were identified during the Phase A/B
audit:

- **Generic infrastructure** (`normal_init`, `shallow_network`,
  `GaussLegendrePiecewise`): a standard Gaussian weight initializer, a
  textbook one-hidden-layer MLP forward pass, and a standard piecewise
  Gauss-Legendre quadrature construction. Used not only by the Hao/Jin
  baseline but by **every** method in Part A (`plain_gn_*`,
  `amstramgram_*`, `dsgnar_*`) -- i.e. load-bearing for the paper's own
  methods, not GNDRM-specific in any algorithmic sense.
- **GNDRM's actual optimizer machinery** (`jacobian_matrix`, `gn_direction`,
  `grid_line_search`): the genuine Gauss-Newton-metric/line-search
  implementation. Used to build the energy-metric direction and line search
  for exactly two optional reference baselines: `hao_gn_deep_ritz` (linear
  problem, Hao et al.'s own method) and `energy_ng_matched` (nonlinear
  problem, Muller-Zeinhofer Hessian metric, built from the same GNDRM
  primitives).

**Release decision**: since GNDRM has no license, none of it is vendored.
The generic infrastructure is **independently reimplemented** (from the
mathematical specification, not from GNDRM's source text) in:

- `src/beyond_pinns/initialization.py` -- `init_shallow_mlp_params`
- `src/beyond_pinns/models.py` -- `shallow_mlp`
- `src/beyond_pinns/quadrature.py` -- `PiecewiseGaussLegendre1D`

so that every one of Part A's own methods (`plain_gn_*`,
`amstramgram_*`, `dsgnar_*`) runs with **zero GNDRM dependency**. The
optimizer machinery (`jacobian_matrix`/`gn_direction`/`grid_line_search`)
remains genuinely external: `hao_gn_deep_ritz` and `energy_ng_matched` are
gated behind `require_gndrm_baseline()` in
`src/beyond_pinns/legacy/part_a_smooth.py`, which raises a clear error naming
the exact upstream URL and commit if either is requested without a local
GNDRM checkout. No top-level import of `beyond_pinns` or of the Part A
legacy script fails because GNDRM is absent.

**Do not imply GNDRM's scientific optimization method was reimplemented**:
it was not. Only the three generic, non-algorithmic utilities above were
independently rewritten; the actual Hao/Jin Gauss-Newton-Deep-Ritz method
stays entirely external and optional.

**`energy_ng_matched` re-investigated (final validation pass)**: re-traced
specifically to confirm whether it depends only on the already-reimplemented
generic pieces or genuinely needs GNDRM's optimizer machinery. Confirmed via
direct code trace: `build_matched_context()`'s "Energy metric: Hao et al.
for linear; Muller-Zeinhofer Hessian metric for nonlinear" block computes
`energy_direction`/`energy_line_search` for **both** `hao_gn_deep_ritz` and
`energy_ng_matched` using the same `jacobian_matrix`/`gn_direction`/
`grid_line_search` calls -- there is no code path where `energy_ng_matched`
avoids these three. Per the stated policy (reimplement only what's
confirmed generic; leave genuinely GNDRM-dependent code external), it
remains gated behind `require_gndrm_baseline("energy_ng_matched")`, **no
code change made**.

One related observation, noted but explicitly not acted on: `jacobian_matrix`
(outer-product metric via quadrature), `gn_direction` (a plain
`jax.numpy.linalg.lstsq` solve), and `grid_line_search` (a fixed-grid
backtracking search hardcoded to the 2-layer parameter pytree shape) are
themselves, on inspection, fairly generic linear-algebra utilities -- no more
algorithmically distinctive than `normal_init`/`shallow_network`/
`GaussLegendrePiecewise` were. The genuinely Hao/Jin-specific content is the
*choice* of which functional derivatives compose the energy metric
(stiffness + mass, via `dt_di_model`/`dt_model`/`dt_model_nonlinear_mass`),
which is **already local/own code**, not from GNDRM. This suggests
`energy_ng_matched` (and, by the same reasoning, `hao_gn_deep_ritz`) *could*
plausibly be made GNDRM-free by extending the reimplementation to these
three functions too -- but this was flagged, not acted on: extending the
reimplementation boundary a second time without being asked risks exactly
the "replace a scientific algorithm merely to remove the dependency"
outcome the validation instructions explicitly warned against. Left for an
explicit future decision.

#### 3.3.1 Equivalence verification (golden fidelity checks)

Each reimplementation was checked against a live GNDRM checkout (commit
`9dd1ee8`) on a reference compute node before being adopted as the default:

| component | check | max abs. discrepancy |
|---|---|---|
| `init_shallow_mlp_params` | 15 (seed, architecture) combinations, full parameter pytree | **0.0** (bit-identical) |
| `shallow_mlp` | forward value, `d/dx`, parameter Jacobian, batched (`vmap`), tanh + relu³, 3 seeds x 4 points + 1 batch | **0.0** (bit-identical) |
| `PiecewiseGaussLegendre1D` nodes | npts in {2,4,8}, sorted comparison (node *order* legitimately differs -- GNDRM's eigendecomposition vs. `leggauss`'s sorted output; same values) | 4.44e-16 (machine epsilon) |
| `PiecewiseGaussLegendre1D` weights | same | 7.08e-16 (machine epsilon) |
| `PiecewiseGaussLegendre1D` end-to-end `integrator(f)` | const/linear/cubic/cosine test functions, 2 intervals, npts in {2,4,8} | 1.11e-15 (machine epsilon) |

One substantive finding during this verification: GNDRM's stored quadrature
weights sum to **1** over `[-1,1]`, not the textbook 2 (confirmed empirically:
`GNDRM.weights == leggauss(npts).weights / 2`, elementwise, to roundoff).
This is compensated by a matching convention in `interval_quadpts`'s global
`area = h` scaling (not `h/2`) -- the two together are mathematically
correct (verified independently by exact-integration-of-low-degree-
polynomial tests that do not require GNDRM at all, see
`tests/test_quadrature.py`), just factored unusually. This is preserved
exactly rather than "corrected", per the release's scientific-fidelity
policy. See `src/beyond_pinns/quadrature.py`'s module docstring for the full
characterization.

No manuscript outputs changed as a result of this reimplementation: since
every check above is either bit-identical or at machine-precision, the
independent implementations are numerically indistinguishable from the
historical GNDRM-backed computation for every fixed seed/architecture/
configuration exercised.

**To install GNDRM** (only needed for the two optional reference baselines):

```bash
git clone https://github.com/Jinxl-pp/GaussNewtonDRM GNDRM
cd GNDRM && git checkout 9dd1ee8df8aca41c5f85157c2d7bf2afa12035ce
```

Place it as a sibling of this repository (or anywhere on `PYTHONPATH`) under
the name `GNDRM` or `GaussNewtonDRM`.

## 4. SIREN-Init (Rectangle+ConnectionPatch zoom-callout technique)

- Upstream: `github.com/AndreaCombette/SIREN-Init`
- Author: Andrea Combette (paper co-authored with Antoine Venaille, Nelly
  Pustelnik)
- File: `src/image_fitting.ipynb` (cells defining `_zoom_bounds`,
  `_connect_main_to_inset`, `visualize_experiment_results`)
- Commit: `95d25fd108` (2026-08-04) -- the repository's only commit to that
  file after 2026-04-20 and before this release's own corner-zoom script was
  written (2026-09-15), so it is the only plausible source version; not
  independently confirmed against an older snapshot.
- License: **none in the upstream repository** (same situation as GNDRM: no
  LICENSE file, GitHub license-detection API returns nothing). Unlike GNDRM,
  however, the copyright holder (Andrea Combette) has separately given this
  release **explicit written permission** to include the specific adapted
  material identified below (see "Redistribution status" further down) --
  that direct author permission, not a repository LICENSE file, is what
  authorizes its inclusion here.
- Paper: Combette, Venaille, Pustelnik, "A new initialisation to Control
  Gradients in Sinusoidal Neural network", arXiv:2512.06427,
  https://proceedings.iclr.cc/paper_files/paper/2026/hash/f1fe380100b2e17f5779cd3456873028-Abstract-Conference.html

  ```bibtex
  @article{combette2025new,
    title        = {A new initialisation to Control Gradients in Sinusoidal Neural network},
    author       = {Combette, Andrea and Venaille, Antoine and Pustelnik, Nelly},
    journal      = {arXiv preprint arXiv:2512.06427},
    year         = {2025},
    doi          = {10.48550/arXiv.2512.06427}
  }
  ```

**What is adapted, precisely**: `scripts/make_figure_rc_corner_zoom.py`
(from the historical `generate_section4_rc_corner_zoom_final.py`) implements
a "callout" plot -- a full-domain view with a red `Rectangle` marking a zoom
region, connected via a `ConnectionPatch` to a separate zoomed axes. This
visual pattern -- Rectangle + ConnectionPatch coupling a full view to a zoom
panel -- is the same core idea used in SIREN-Init's `image_fitting.ipynb`
(there implemented as an `inset_axes` inset with `_connect_main_to_inset`,
for comparing image-reconstruction quality; here as a separate zoom subplot,
for a continuous PDE domain, with the zoom window defined by an explicit
physical coordinate range rather than SIREN-Init's `_zoom_bounds` bottom-
right pixel-fraction heuristic). This is an **adaptation of the general
plotting pattern**, not a copy of SIREN-Init's source text -- variable names,
axes layout, and the zoom-window definition all differ, driven by the
different domain (continuous PDE field vs. discrete pixel image). The
historical research-tree comment referring to this pattern's origin as
"NG-INR" was inaccurate; it has been corrected in the release copy to
reference SIREN-Init directly (see the script's own header comment).

**Redistribution status: PERMISSION GRANTED.** Andrea Combette gave explicit
written permission on **2026-09-22** for this release to include the
SIREN-Init-derived material identified above (the Rectangle+ConnectionPatch
zoom-callout adaptation in `scripts/make_figure_rc_corner_zoom.py`). This
permission covers specifically that adapted material and its attribution as
described in this section; it is not a blanket license to any other part of
the SIREN-Init repository. The private correspondence itself is not
reproduced here; this note is the record of the permission, its date, and
its scope.

**Independent-of-SIREN-Init note** (kept for clarity, not because it is
needed now that permission is granted): the RC decomposition figures
(`scripts/make_figure_rc_decomposition.py`, producing the exact-solution/
hybrid-approximation/FE-compensator panels) are entirely independent of this
adaptation and regenerate fully from stored results with no SIREN-Init
material at all. Only the specific corner-zoom callout figure
(`scripts/make_figure_rc_corner_zoom.py`) uses the adapted material.

## 5. Bug fixes carried from the research tree into this release

Both corrections below were found during the Phase A audit of the research
tree (not introduced by this release) and are kept in the release. Neither
is an "invisible" difference: both are documented here, and both are
isolated, minimal diffs against the historical files (available for direct
inspection by diffing `src/beyond_pinns/legacy/*.py` against the
corresponding file in the historical research tree).

### 5.1 MS-skip configuration bug (Part B entry point)

- **Affected file**: `femennstein-petrov-galerkin-experiments-fixed.py`
  (research tree) / `src/beyond_pinns/legacy/part_b_general.py` (release)
- **Original behavior**: `CONFIG["section_1"]`'s individual run-type flags
  (`exact_regression`, `deep_ritz`, `deep_ritz_l2_metric`, `strong_pinn`, and
  every family under `pure_petrov`/`alternating_hybrid`/`lagged_hybrid`/
  `projected_hybrid`) were hardcoded `False`, with a comment explaining this
  was a mid-run resume state: section_1 (MS) had already completed on disk
  before a crash at section_2's first run, so section_1's training was
  skipped to avoid redoing it -- a snapshot of a specific resumption, left
  in the "canonical" `-fixed.py` file.
- **Corrected behavior**: all section_1 flags restored to `True`, matching
  the pre-crash original file and matching the config that actually
  generated the archived, already-verified `petrov_galerkin_results/
  section_1/` results on disk.
- **Exact reason it was a bug**: a human-readable entry point (the one file
  new users/reviewers would run) that, run fresh from a clean checkout, would
  silently skip the entire MS benchmark rather than reproducing it.
- **Did manuscript numerical values change?** No. The restored flags exactly
  match the configuration that produced the archived MS results already used
  throughout the manuscript; nothing was recomputed or altered.
- **Execution-only or visualization-only?** Execution/configuration only --
  no plotting code touched.
- **Did any figure change?** No.
- **Diff**: 15-line change, restoring `False -> True` for 4 scalar flags and
  16 nested family flags in `CONFIG["section_1"]`, and removing the
  now-inapplicable resume comment. See the earlier diff already applied in
  the research tree (this release's copy already reflects it, not a further
  in-flight change).

### 5.2 Figure-1 truncation-artifact bug (main-text aggregation)

- **Affected files**: `generate_petrov_figures.py` /
  `generate_pure_vs_projected_random_hats_panels.py` (research tree) ->
  `scripts/make_figure_main_comparison.py` /
  `scripts/make_figure_random_hats.py` (release, see docs/RESULTS.md for
  exact mapping)
- **Original behavior**: `median_iqr_curve()` truncated every seed's
  recorded trajectory to the length of the *shortest*-running seed before
  computing the per-iteration median/IQR curve. Since DSGNAR's adaptive
  trust-region stopping rule causes different seeds to stop at very
  different iteration counts -- and, critically, the shortest-running seed
  is sometimes one that stopped early at comparatively poor accuracy -- the
  plotted curve's right-hand edge could sit many orders of magnitude above
  the true converged value the other (longer-running) seeds actually reach.
  For MS specifically, this made the plotted main-text Figure 1 curves for
  the projected/hybrid methods flatline around 1e-2 after ~90 iterations,
  even though the manuscript table (and the archived per-seed final values)
  correctly report ~1e-9 to ~1e-13.
- **Corrected behavior**: last-observation-carried-forward (LOCF).
  Each seed's displayed error after its own termination iteration T_s is
  held constant at its own final recorded value (`E_s(k) = E_s(T_s)` for
  k > T_s), and the median/IQR is computed over all 5 seeds at every
  iteration out to the longest-running seed -- no truncation to the
  shortest seed, no shrinking sample size.
- **Exact reason it was a bug**: once DSGNAR's trust radius collapses below
  its stopping threshold, the returned parameters are fixed, so the seed's
  true error is genuinely constant thereafter -- truncating to the shortest
  seed instead discards this correct, known-constant continuation and
  substitutes an arbitrary earlier (often much worse) value for every other
  seed at that same iteration.
- **Did manuscript numerical values change?** No. The manuscript's
  *reported table values* were always computed from each seed's own true
  final value (`final_stats()`/the verification table), never from the
  plotted curve's truncated endpoint -- so the manuscript's numbers were
  always correct. Only the **plotted curve's visual right-hand endpoint**
  was wrong; it visually understated convergence for MS (and, less severely,
  for the JF hybrid run) by up to ~10 orders of magnitude on the plotted
  line, without misrepresenting any number in a table.
- **Execution-only or visualization-only?** Visualization/aggregation only.
- **Did any figure change?** Yes: `main_h1_comparison.pdf` (main-text
  Figure 1), `pure_vs_projected_random_hats_h1.pdf` and `appendix/
  hybrid_update_ablation_h1.pdf` (both from `generate_petrov_figures.py`),
  and the four `pure_vs_projected_random_hats_{ms,jf,ls,rc}.pdf` panels
  (from `generate_pure_vs_projected_random_hats_panels.py`) were all
  regenerated with the fix and visually verified (MS's projected/hybrid
  curves now correctly descend to ~1e-9/1e-13 instead of flatlining at
  ~1e-2). All are included in `figures/reference/`.
- **Diff**: `median_iqr_curve()`'s truncation (`n = min(len(h[ykey]) for h in
  hs)`) replaced by an LOCF extension to `n_max = max(...)`, holding each
  seed's value constant past its own length; applied identically in both
  affected scripts. See `docs/RESULTS.md` for the before/after numerical
  comparison at the plotted endpoint for all 8 (benchmark x model)
  combinations.

### 5.3 Wall-clock aggregation: index-based median replaced with time-grid resampling (extends 5.2)

Two further scripts aggregate a wall-clock-indexed axis that section 5.2's
fix did not originally reach:

- `generate_energy_weak_v3_figures.py` -> `scripts/make_figure_smooth_benchmarks.py`
  (Part A "energy vs weak" figures 3-6, `energy_vs_weak_benchmarks/`) had
  **not been fixed at all** -- still truncate-to-shortest-seed on both axes.
- `generate_petrov_figures.py` -> `scripts/make_figure_main_comparison.py`
  (main-text Figure 1 and its wall-clock row) had section 5.2's
  iteration-style LOCF applied, but its wall-clock panels still used an
  index-based convention -- `x = median(X, axis=0)` where `X` is each
  seed's own (LOCF-extended) wallclock array *at shared index k*.
- `generate_pure_vs_projected_random_hats_panels.py` ->
  `scripts/make_figure_random_hats.py` were checked and confirmed to have
  **no wall-clock-indexed panels at all** (iteration axis only) -- out of
  scope, unchanged.

**Why the index-based wall-clock convention needed replacing, not just
LOCF-extending**: unlike the iteration axis, wall-clock elapsed time is not
a shared, data-independent grid -- it is computed *from* each seed's own
data. Once seeds run for very different numbers of iterations (as DSGNAR's
adaptive trust-region stopping guarantees), a seed that finishes early keeps
contributing a frozen, small elapsed-time value at every later shared
index, pulling the aggregate x-position left even though the still-running
seeds used substantially more real time to reach that later data point --
i.e. "index k" stops meaning "the same point in time" across seeds. This was
reported to you, with two candidate conventions, before any change was made
(per explicit instruction not to change wall-clock aggregation
unilaterally):

1. **Time-grid resampling (selected)**: build a shared wall-clock grid
   spanning `[0, max over seeds of that seed's own true final wallclock]`;
   evaluate each seed as a right-continuous step function of elapsed time,
   held flat (LOCF, in the time domain) after its own last recorded
   timestamp; aggregate median/Q1/Q3 across seeds at each grid time. No
   seed's elapsed time is ever invented -- a finished seed's clock is never
   advanced past the time it actually stopped; only its own previously
   recorded value is repeated.
2. Symmetric-LOCF fallback keeping the index-based convention (not
   selected) -- simpler, but keeps the index/time conflation above.

**Implementation**: the canonical, tested reference is
`src/beyond_pinns/aggregation.py` (`aggregate_iteration_curve`,
`aggregate_wallclock_curve`, `aggregate_curve`; tested directly in
`tests/test_aggregation.py`, 8 tests, including an explicit check that a
finished seed's contribution past its own true final wallclock always
equals its own last true value, never an extrapolated or borrowed one).
Each plotting script keeps its own self-contained, verbatim-equivalent copy
of the same two-branch logic (matching this release's existing convention
of script self-containment -- see section 5.2), in
`generate_energy_weak_v3_figures.py`/`scripts/make_figure_smooth_benchmarks.py`
and `generate_petrov_figures.py`/`scripts/make_figure_main_comparison.py`.

**Regenerated and compared**:

- **Figures 3-6** (`linear_tanh_formulation_errors`,
  `nonlinear_tanh_formulation_errors`, `linear_tanh_weak_optimizers_errors`,
  `nonlinear_tanh_weak_optimizers_errors`, all 4 panels each): regenerated
  and pixel-compared against the pre-fix versions. **The curves change
  materially** -- roughly 12-16% of pixels differ beyond a small tolerance
  in every figure, on both the iteration and wall-clock rows. Concretely,
  in `linear_tanh_formulation_errors`: DSGNAR's seeds stop between
  163-293 iterations depending on run/seed; under the old truncate-to-
  shortest convention its plotted curve stopped around iteration ~180-220
  (~20-24s) at roughly relative L2 ~1e-13; under the new LOCF + time-grid
  convention it correctly extends to ~300 iterations (~30s), and the
  now-fully-represented longer-running seeds pull the aggregate down to its
  true converged plateau (~1e-14). This is the same class of effect as
  section 5.2's MS truncation bug (a genuinely converged, longer-running
  seed's information was being discarded by the old convention), now fixed
  for the wall-clock-indexed panels of these figures specifically.
  Regenerated PDFs/PNGs are in `figures/energy_weak/` (research tree) and
  should be copied into `beyond-pinns-code/figures/reference/` for the
  release, replacing the pre-fix versions.
- **`main_h1_comparison.pdf`** (main-text Figure 1, S1-S4): regenerated
  cleanly (exit 0, no errors/NaNs); all iteration-row curves match section
  5.2's already-verified behavior unchanged (that branch was not touched);
  the wall-clock row now correctly stretches each method's curve out to its
  own true elapsed time (e.g. S1's "Projected, operator Green" run visibly
  extends to ~450s on the corrected axis, tracking the shape of its own
  iteration-row curve) rather than being distorted toward the index-based
  median. Not pixel-diffed against the pre-fix version in this pass (the
  underlying mechanism is identical to figures 3-6, already demonstrated to
  be material above); a full diff should be done before treating this as
  final if the manuscript quotes specific wall-clock values read off this
  figure.
- **No manuscript *table* values are affected** by this change -- exactly as
  in section 5.2, tables are computed from each seed's own true final value
  (`final_stats()`), never from a plotted curve; only the **plotted curves'
  shape and visual endpoints** on the wall-clock axis change.

**Tests**: `tests/test_aggregation.py` (new, 8 tests) and the existing
`tests/test_locf_aggregation.py`/`tests/test_monitoring.py` (33 tests total
across all three files) pass locally.
