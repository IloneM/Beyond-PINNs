# Results traceability

This is the authoritative map from each manuscript item to its exact
configuration, historical campaign location, results-archive path,
aggregation convention, generation script, and reported value. Every script
listed here reads stored results and never reruns optimization. See
`docs/PROVENANCE.md` for the bug fixes referenced below (MS-skip config bug;
LOCF + time-grid aggregation bug).

## 0. Path mapping: historical campaign -> results archive -> `results/reference/`

Every "raw result" path below is given relative to `results/reference/`
(this repository's documented extraction point, see
`docs/REPRODUCIBILITY.md`). That directory's contents come, unmodified,
from the separate `beyond-pinns-results` archive, which in turn was copied
unmodified from the authors' research tree. The three locations share
identical relative paths under a fixed prefix substitution:

| Historical campaign (research tree) | Results archive (`beyond-pinns-results/`) | This repository (`results/reference/`, once extracted) |
|---|---|---|
| `petrov_galerkin_results/section_N/...` | `general/section_N/...` | `general/section_N/...` |
| `petrov_galerkin_results/controlled_timings/...` (excluding an abandoned partial dry-run and unused generator scripts, see the archive's own `MANIFEST.json`) | `timings/...` | `timings/...` |
| `petrov_standalone_fem_cache/section_N.npz` | `fem/section_N.npz` | `fem/section_N.npz` |
| `energy_vs_weak_benchmarks/{problem}/{protocol}/{activation}/{method}/seed_NNN/...` | `smooth/{problem}/{protocol}/{activation}/{method}/seed_NNN/...` | `smooth/{problem}/{protocol}/{activation}/{method}/seed_NNN/...` |

So e.g. a "raw result" of `results/reference/general/section_1/s1_proj_h01_green_quadrature/seed_000/history.npz`
below is, unmodified, `general/section_1/s1_proj_h01_green_quadrature/seed_000/history.npz`
in the archive, which is `petrov_galerkin_results/section_1/s1_proj_h01_green_quadrature/seed_000/history.npz`
in the historical research tree.

## 1. General/hybrid benchmark summary table (MS/JF/LS/RC, Part B)

| | |
|---|---|
| Manuscript item | Main accuracy table, all 4 benchmarks, all methods (pure Petrov-Galerkin / Deep Ritz / strong PINN / projected-lagged-alternating hybrid) |
| Config | `configs/general/{ms,jf,ls,rc}_{pure,hybrid}.yaml` for the manuscript-selected methods (`configs/method_mapping.yaml`); every other method/test-family combination is reproducible the same way by editing `benchmark`/`model_class`/`family` in a copy of these YAML files |
| Raw result | `results/reference/general/section_N/s{N}_{pure,proj,lagged,alt}_{family}/seed_{000..004}/history.npz` (per-seed convergence: `iteration`, `relative_l2`, `relative_h1`, `wallclock_optim`) |
| Aggregation | `median_iqr_curve()` / `final_stats()` in `scripts/make_figure_main_comparison.py` (copy of the research tree's `generate_petrov_figures.py`, LOCF-fixed -- see below); `final_stats()` uses each seed's own true final recorded value, never a truncated/aggregated one |
| Plotting/table script | `scripts/make_figure_main_comparison.py` -- writes `figures/reference/main_h1_comparison.{pdf,png}` (main-text Figure, 2x4 grid) plus `figures/reference/petrov_seed_level_summary.csv` (per-seed rows) and `petrov_method_summary.csv` (per-method medians/IQR/wallclock) |
| Reported value | Built-in verification table (printed at script start, before plotting) cross-checks 14 (section, run) pairs against a hardcoded `REFERENCE` dict at 25% relative tolerance; all 14 pass with zero discrepancies as of the last run in this repository's construction |

## 2. Selected method mapping (pure/hybrid test family per benchmark)

| | |
|---|---|
| Manuscript item | Which weak test family is used for the "pure" and "hybrid" columns of the main table, per benchmark |
| Config | `configs/method_mapping.yaml` (machine-readable, exact internal identifiers) |
| Raw result | `results/reference/method_mapping.json` (verbatim source this YAML was transcribed from) |
| Aggregation | none needed -- this is itself already the audit output of an earlier computational-cost/method-mapping study (5-seed median L2/H1 per candidate family, selecting the best) |
| Selected mapping | MS pure/hybrid: `h01_green_quadrature` / `h01_green_quadrature`. JF pure/hybrid: `eigen_green_quadrature` / `h01_tensor_quadrature`. LS pure/hybrid: `eigen_green_quadrature` / `eigen_green_quadrature`. RC pure/hybrid: `random_hats` / `random_hats`. |
| Verified by | `tests/test_config_equivalence.py::test_method_mapping_yaml_matches_manuscript_selection` and `::test_method_mapping_families_exist_in_historical_config` |

## 3. Random-hat comparison (pure vs. hybrid, all 4 benchmarks)

| | |
|---|---|
| Manuscript item | Pure-vs-projected convergence comparison under the generic `random_hats` test family, all 4 benchmarks |
| Config | `configs/general/{ms,jf,ls,rc}_{pure,hybrid}.yaml` with `family: random_hats` (already the selection for RC; for MS/JF/LS this is a secondary/ablation family, not the manuscript's *selected* one -- see item 2) |
| Raw result | `results/reference/general/section_N/s{N}_{pure,proj}_random_hats/seed_{000..004}/history.npz` |
| Aggregation | **Last-observation-carried-forward (LOCF)**, not truncate-to-shortest-seed. Each seed's displayed error is held constant at its own final recorded value past its own stopping iteration; median/IQR computed over all 5 seeds at every iteration out to the longest-running seed. This is a corrected bug (see `docs/PROVENANCE.md` section 5.2; wall-clock-axis panels additionally use time-grid resampling, section 5.3) -- **do not reintroduce truncate-to-shortest-seed aggregation.** |
| Plotting script | `scripts/make_figure_random_hats.py` (copy of `generate_pure_vs_projected_random_hats_panels.py`) -- writes `figures/reference/pure_vs_projected_random_hats_{ms,jf,ls,rc}.{pdf,png}` and a standalone legend |
| Reported value (final plotted median, confirmed = manuscript table) | MS pure L2/H1 = 2.1861e-13/8.2626e-12; MS hybrid = 9.1437e-13/2.1553e-11; JF pure = 2.0496/4.4259; JF hybrid = 1.7128e-04/1.3419e-03; LS pure = 9.9271e-01/9.8891e-01; LS hybrid = 4.0687e-03/3.3209e-02; RC pure = 1.0520/1.3974; RC hybrid = 5.0353e-04/1.2029e-02 |
| Regression test | `tests/test_locf_aggregation.py` -- unit-tests the LOCF mechanism itself on synthetic data (length extension, per-seed value-holding, final-median equivalence, no dropped seeds) |

## 4. Standalone FE reference comparison

| | |
|---|---|
| Manuscript item | Near-hybrid-budget standalone FE reference L2/H1, all 4 benchmarks (`tab:exp-general-summary`'s FE row) |
| Config | Benchmark-fixed FE discretization, not a YAML config (FE mesh/degree are hardcoded per-benchmark constants, not swept per run) -- see `scripts/compute_standalone_fem.py` |
| Raw result | `results/reference/general/section_N/fem_baselines/selected.json` -- the `best_fem_h1_under_hybrid_budget` entry, containing BOTH `rel_l2` and `rel_h1` from the same original FE sweep (not a separate reconstruction) |
| Selected configuration | MS: P4, 96 uniform elements, 383 DOFs. JF: P4, 12x8 interface-fitted mesh, 1457 DOFs. LS: P3, 14x12 interface-fitted mesh, 1435 DOFs. RC: P2, power-graded mesh (n_radial=16, n_angular=40, beta=2.5), 2409 DOFs. |
| Reported value | rel_l2/rel_h1: MS 2.186e-07/7.307e-05; JF 2.950e-14/5.439e-14; LS 1.359e-03/1.480e-02; RC 1.235e-03/1.596e-02. All four H1 values match the manuscript's previously-reported H1-only row exactly; L2 values are directly stored (not reconstructed) from the same FE solve, confirmed on the SAME quadrature convention used for the neural/hybrid errors elsewhere in the table (`reevaluate_fem_records_2d`/`finalise_fem_comparison` in the legacy script re-evaluates FE candidates on the shared `ERR_X{N}`/`ERR_W{N}` error-quadrature grid before selection). |
| Verified by | `tests/test_fe_selected_configurations.py` (exact degree/description/n_dofs assertions) |
| Caveat | A separate FIELD-reconstruction path (`scripts/compute_standalone_fem.py`, needed only for the solution-field appendix figures) evaluates on a different point set and shows real but small discrepancies vs. the directly-stored values (worst case RC: L2 off by ~2x, H1 by ~70%, due to evaluation-point differences near the reentrant-corner singularity) -- **do not use the field-reconstruction values for the summary table**, only the directly-stored `selected.json` values above. |

## 5. Timing table

| | |
|---|---|
| Manuscript item | Controlled FE-vs-neural/hybrid timing comparison |
| Config | FE: `scripts/run_fe_timings.py` / `scripts/run_fe_single_cold.py` (cold vs. warm distinction). Neural/hybrid: per-benchmark instrumented scripts isolating exactly one (section, family, model_class) target, same isolation convention as `beyond_pinns.config.build_full_config_general` |
| Raw result | FE: `results/reference/timings/timing_FE_{MS,JF,LS,RC}.json` (5 repetitions each, warm median reported). Neural/hybrid: `results/reference/timings/neural_timings_reduced_2seed/timing_neural_{MS,JF,LS,RC}_{pure,hybrid}.json` |
| Aggregation | `scripts/build_fe_timing_summary.py` for the FE side |
| **Reduced-cost status** | The neural/hybrid timing numbers currently in this repository are a **REDUCED-COST CHECK (2 of 5 seeds)**, not the full manuscript campaign -- see the README in `results/reference/timings/neural_timings_reduced_2seed/`. Median T_optim per seed: MS pure 299.5s / MS hybrid 26.4s / JF pure 75.7s / JF hybrid 36.2s / LS pure 134.5s / LS hybrid 22.5s / RC pure 9.7s / RC hybrid 36.2s, vs. FE warm-median 0.040s (MS) / 0.309s (JF) / 0.345s (LS) / 2.172s (RC). The full 5-seed campaign has not yet been run in this repository's construction; extrapolated additional cost from these measured per-seed numbers is roughly 35-45 more minutes of sequential GPU time for the remaining 3 seeds x 8 methods (see `docs/HARDWARE.md`/the neural timing status note for how to resume). |
| Timing methodology | T_optim excludes one-time JIT compilation (T_setup_compile, measured separately); no old uncontrolled/JIT-contaminated timings are mixed in -- see `docs/EXPERIMENTS.md` for the compile-then-time protocol. |

## 6. Smooth benchmark summary table and figures (Part A)

Curated into `results/reference/smooth/` (3.6MB, `history.npz` only -- no
`final_solution.npz`/checkpoints/plots -- mirroring the exact
`{problem}/{matched,reference}/{activation}/{method}/seed_NNN/history.npz`
layout `generate_energy_weak_v3_figures.py`/`scripts/make_figure_smooth_benchmarks.py`
expects via `seed_dirs()`/`load_histories()`): every method actually plotted
by that script's current figures (3-6), **tanh only** (the script's own
header comment: "ReLU^3 figures are intentionally NOT regenerated here --
unaffected" -- so the curated set deliberately excludes relu3, matching what
the current figure pipeline itself uses), all 10 seeds each:

- `linear/matched/tanh/{hao_gn_deep_ritz,plain_gn_strong_lm,plain_gn_weak_lm,dsgnar_strong,dsgnar_weak,amstramgram_weak}/seed_{000..009}/history.npz`
- `nonlinear/matched/tanh/{energy_ng_matched,plain_gn_strong_lm,plain_gn_weak_lm,dsgnar_strong,dsgnar_weak,amstramgram_weak}/seed_{000..009}/history.npz`
- `nonlinear/reference/tanh/energy_ng_reference/seed_{000..009}/history.npz` (note the `reference` protocol directory, not `matched` -- this is the literal Muller-Zeinhofer public-protocol reproduction, `PROTOCOL_OVERRIDE` in the generator script)
- `results/reference/smooth/aggregate_summary.csv` -- the historical script's own auto-generated summary (written as an import-time side effect of `energy_vs_weak_benchmarks_gn_strong_weak.py`; confirmed read by **no** downstream script -- `generate_energy_weak_v3_figures.py` independently re-reads `history.npz` directly, so this CSV is included for reference/regression-diffing only, not because anything consumes it)

| Figure | Methods | Problem | Activation | Output |
|---|---|---|---|---|
| 3 (formulation) | `hao_gn_deep_ritz`, `plain_gn_strong_lm`, `plain_gn_weak_lm`, `dsgnar_strong`, `dsgnar_weak` | linear | tanh | `figures/energy_weak/linear_tanh_formulation_errors.{pdf,png}` |
| 4 (formulation) | `energy_ng_reference`, `energy_ng_matched`, `plain_gn_strong_lm`, `plain_gn_weak_lm`, `dsgnar_strong`, `dsgnar_weak` | nonlinear | tanh | `figures/energy_weak/nonlinear_tanh_formulation_errors.{pdf,png}` |
| 5 (weak optimizers) | `hao_gn_deep_ritz`, `plain_gn_weak_lm`, `amstramgram_weak`, `dsgnar_weak` | linear | tanh | `figures/energy_weak/linear_tanh_weak_optimizers_errors.{pdf,png}` |
| 6 (weak optimizers) | `energy_ng_reference`, `energy_ng_matched`, `plain_gn_weak_lm`, `amstramgram_weak`, `dsgnar_weak` | nonlinear | tanh | `figures/energy_weak/nonlinear_tanh_weak_optimizers_errors.{pdf,png}` |

Now present in `figures/reference/` (copied in during the LOCF/time-grid
aggregation fix below -- previously only Part B figures were there).
Regenerate via `scripts/make_figure_smooth_benchmarks.py` (see
`scripts/reproduce_figures.sh`) if `energy_vs_weak_benchmarks/` changes.

**Config**: `configs/smooth/*.yaml` (4 shipped: `linear_dsgnar_weak`,
`linear_plain_gn_weak_tsvd`, `nonlinear_dsgnar_weak`,
`linear_hao_gn_deep_ritz_OPTIONAL`) target individual methods via
`beyond_pinns.run`; every other `CONFIG[problem]["methods"]` key (including
all methods in the table above) is reachable the same way by adding a YAML
file with the corresponding `method:` value -- no code change needed.

**Regression-testing final L2/H1**: `tests/test_config_equivalence.py`
verifies config resolution; there is not yet a dedicated Part A
numerical-regression test analogous to `tests/test_fe_selected_configurations.py`
for Part B/FE -- noted as a remaining gap (the shipped `history.npz` files
above make one straightforward to add: assert each seed's final
`relative_l2`/`relative_h1` against the values recorded in
`aggregate_summary.csv`).

**Caveat** (unchanged from the historical logs, preserved as scientific
data, not a bug): many individual (method, seed) combinations in the
historical 460-run campaign did not converge (stuck near relative-L2 ~ 1) --
most notably several `plain_gn_*_tsvd` seeds and almost every
strong-residual sophisticated optimizer under `relu3`, especially combined
with the nonlinear PDE. See `docs/EXPERIMENTS.md` and
`running_fixes_weak-nrg_benchmark.md` in the research tree.

**Formerly a known live issue -- now fixed.** `generate_energy_weak_v3_figures.py`'s
`median_iqr_curve()` used the same truncate-to-shortest-seed convention that
was the LOCF bug fixed in Part B's `generate_petrov_figures.py` and
`generate_pure_vs_projected_random_hats_panels.py`, and additionally (on the
wall-clock axis) an index-based median-of-wallclock convention that distorts
once seed run-lengths diverge. Since Part A's own `dsgnar_strong`/
`dsgnar_weak` CAN stop early (trust-region collapse, unlike AMStramGRAM
which runs the full fixed iteration budget), Figures 3-6 above --
all of which include a DSGNAR curve -- had the same visual-truncation
distortion already diagnosed for Part B's MS panel. **Fixed**: LOCF for the
iteration axis, time-grid resampling for the wall-clock axis (see
`docs/PROVENANCE.md` section 5.3 for the full before/after comparison and
rationale). Figures 3-6 were regenerated with the fix; the curves change
materially (DSGNAR's true converged plateau is now correctly reached instead
of being cut off at its shortest seed's stopping point).

## 7. MS diagnostic figure

| | |
|---|---|
| Manuscript item | MS (multiscale diffusion) diagnostic figure |
| Config | Reads `results/reference/general/section_1/...` (representative-seed selection, see below) |
| Script | `scripts/make_figure_ms_diagnostics.py` (copy of `generate_section1_ms_diagnostics.py`) and `scripts/make_figure_ms_decomposition.py` (copy of `generate_section1_hybrid_decomposition.py`) |
| Representative-seed rule | The seed whose final relative H1 error is closest to the 5-seed median -- **not** the best-performing seed. Implemented in `representative_seed()`, copied verbatim/unmodified from `generate_petrov_solution_figures.py`'s original selection logic. |

## 8. RC decomposition / overview

| | |
|---|---|
| Manuscript item | RC (reentrant corner) structural decomposition: exact solution, hybrid approximation, FE compensator |
| Script | `scripts/make_figure_rc_decomposition.py` (copy of `generate_section4_rc_decomposition.py`) |
| Raw result | `results/reference/general/section_4/s4_proj_random_hats/seed_001/final_solution.npz` (seed 1 is the representative seed by the same median-H1 rule as item 7) |
| Provenance | Fully independent of any external adapted material -- regenerates entirely from stored results and original plotting code. |

## 9. RC zoom figure

| | |
|---|---|
| Manuscript item | RC reentrant-corner zoom/callout figure (multilevel corner-zoom diagnostics) |
| Script | `scripts/make_figure_rc_corner_zoom.py` (copy of `generate_section4_rc_corner_zoom_final.py`, the latest/canonical of four historical drafts) |
| Provenance | The Rectangle+ConnectionPatch zoom-callout mechanism is **adapted from SIREN-Init** (github.com/AndreaCombette/SIREN-Init, `src/image_fitting.ipynb`, commit `95d25fd108`), with attribution to Andrea Combette -- see `docs/PROVENANCE.md` section 4 for the full accounting. **Redistribution status: PERMISSION GRANTED** (2026-09-22, written permission from Andrea Combette for this specific adapted material). |
| Independent alternative | Item 8 (RC decomposition) does not depend on this adaptation at all and can be published/regenerated independently if permission is not granted for the zoom figure specifically. |
