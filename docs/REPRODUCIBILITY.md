# Reproducibility

Exact workflow from a fresh clone to manuscript outputs, in FOUR levels of
increasing cost: fast smoke test, integration smoke test, reduced-cost
scientific run, full reproduction.

## Two-layer release

This project ships as two separate artifacts:

- **CODE**: this repository (`beyond-pinns-code`, GitHub). Source, configs,
  tests, docs, figure/table generation scripts, and a handful of tiny
  synthetic test fixtures. **Contains no manuscript experiment
  histories/results/logs.**
- **RESULTS**: `beyond-pinns-results`, a separate versioned research-data
  archive on Zenodo. The complete trusted numerical outputs (per-seed
  histories, final solution fields, FE caches, timing measurements)
  underlying every current manuscript table and figure. See that archive's
  own `README.md`/`MANIFEST.json` for its exact contents and structure.

  **Dataset DOI: 10.5281/zenodo.22904196 --
  <https://doi.org/10.5281/zenodo.22904196>.** This is the results dataset's
  own DOI -- distinct from the paper's DOI (10.48550/arXiv.2609.20641, see
  the repository root `README.md`) and not a software DOI.

This repository expects the results archive, if used at all, extracted into
`results/reference/` (see `scripts/download_reference_results.sh`); that
directory is absent/empty (apart from its own short `README.md`) in a fresh
clone, and nothing here implies otherwise.

### Three workflows

**A. Install and test the code** (no results archive needed):

```bash
git clone https://github.com/IloneM/Beyond-PINNs beyond-pinns-code
cd beyond-pinns-code
uv sync --extra test
pytest
./scripts/reproduce_smoke.sh
```

**B. Run new experiments from scratch** (no results archive needed -- this
*produces* results, it does not consume the archive):

```bash
python -m beyond_pinns.run --config configs/general/ms_hybrid_smoke.yaml   # reduced-cost check
python -m beyond_pinns.run --config configs/general/ms_hybrid.yaml         # full manuscript config
```

Output goes to that config's own `output_root` (default: top-level
`petrov_galerkin_results/` or `energy_vs_weak_benchmarks/`, matching the
historical convention -- gitignored, distinct from `results/reference/`).

**C. Reproduce the published figures/tables from stored results** (needs the
results archive):

```bash
./scripts/download_reference_results.sh   # or extract the archive into results/reference/ manually
./scripts/reproduce_tables.sh
./scripts/reproduce_figures.sh
```

If `results/reference/` is not populated, these scripts fail cleanly with an
actionable message telling you to run `download_reference_results.sh` --
never a bare `FileNotFoundError`.

## 0. Install

```bash
git clone https://github.com/IloneM/Beyond-PINNs beyond-pinns-code
cd beyond-pinns-code
uv sync --extra test            # or: pip install -e ".[test]"
```

This installs the PyPI-only core dependencies (numpy, scipy, jax, equinox,
jaxtyping, matplotlib, pyyaml, pytest) plus `pinn` (the DSGNAR optimizer,
required for every Part B run -- it supplies internals reused by
`VectorOptimiser`/`EnergyOptimiser`, see `docs/PROVENANCE.md` section 3.1
for the exact symbol-level map). `pinn` is not on PyPI, but `pyproject.toml`
pins it as a direct Git dependency at the exact historical commit (see
`docs/PROVENANCE.md` section 3.1), so `uv sync`/`pip install -e .` clone and
build it automatically -- no manual clone or `PYTHONPATH` step needed.

It does **not** install GaussNewtonDRM (GNDRM), which remains a manual,
genuinely optional dependency (see below).

**`ngrad` is no longer needed for Part B at all.** The historical
`ngrad.models.mlp` forward pass has been independently reimplemented as
`beyond_pinns.models.deep_mlp` (see `docs/PROVENANCE.md` section 3.2) --
every Part B config under `configs/general/` runs with the installed `pinn`
package and nothing else external. `ngrad` remains a genuinely optional
external dependency only for Part A's Muller-Zeinhofer reference protocol;
GNDRM remains optional only for the `hao_gn_deep_ritz`/`energy_ng_matched`
baselines (see `docs/PROVENANCE.md` for exact commits and manual
clone+`PYTHONPATH` install commands, since neither is a pinned pyproject.toml
dependency). Every Part A config under `configs/smooth/` except
`linear_hao_gn_deep_ritz_OPTIONAL.yaml` runs with neither `ngrad` nor GNDRM
installed.

GPU: install the `cuda` extra (`uv sync --extra cuda`) for CUDA-enabled jax;
CPU-only jax is sufficient for FE reference computations and for running the
test suite.

## 1. Fast smoke test (<= 2 minutes, target)

```bash
./scripts/reproduce_smoke.sh
```

Run this FIRST on any fresh install / in CI. Deliberately avoids the
historical legacy scripts entirely -- no GPU, no `pinn`/`ngrad`/GNDRM needed.
Checks: package import, config loading (`--print-config-only`, no training),
the independent initializer/shallow-model/quadrature (`beyond_pinns.
{initialization,models,quadrature}`), a tiny 4-element 1D FE solve (not the
manuscript's 96-element MS configuration -- just confirms `fem_utils.py`
works), the hybrid-projection identity (`tests/test_hybrid_projection_identity.py`),
and one tiny plotting/output path. See the script for the exact steps.

## 1b. Integration smoke test (~15 minutes)

```bash
./scripts/reproduce_integration_smoke.sh
```

A real, minimal end-to-end pass through the trusted HISTORICAL Part B
script: one hybrid solve, one FE solve, one figure generation. Does not
verify manuscript-level accuracy (see Level 2/3 for that) -- verifies the
actual historical pipeline runs, not just this release's own new code.

**Timing note**: measured end-to-end at ~853s (~14 minutes) on the reference
GPU compute node (see `docs/HARDWARE.md`), not "a few minutes." The `part_b_general.py`
legacy script's setup is shared top-to-bottom across all four benchmarks
(MS/JF/LS/RC): even when only MS is requested, the script still assembles a
small "smoke-scale" FE candidate mesh for JF/LS/RC as part of its
unconditional shared setup (this release's config isolation already reduces
that from the *full* per-benchmark candidate sweep, ~20 mesh candidates
each, down to the historical script's own smaller "smoke" candidate list,
~5 each -- confirmed by `configs/general/ms_hybrid_smoke.yaml` resolving
`fem_baseline`/`fem_sweep` to `False` for every section, see
`tests/test_config_equivalence.py`). Eliminating this remaining ~14-minute
floor entirely would require restructuring the legacy script's own control
flow to skip unrelated benchmarks' setup outright -- exactly the kind of
"substantial refactor of trusted scientific code" this release deliberately
avoids (see docs/PROVENANCE.md). Treat ~15 minutes, not "a few minutes," as
the honest expectation for this integration smoke test -- the fast smoke
test above (Level 1) is the one intended to actually be fast.

## 2. Reduced-cost reproduction

Any `configs/general/*_smoke.yaml` (currently `ms_hybrid_smoke.yaml`) or a
config invoked with a seed/iteration override is a REDUCED-COST CHECK.
These are useful for verifying the pipeline behaves sensibly on your
hardware, but **must never be reported as manuscript numbers** -- their
whole purpose is to be cheap, not accurate. For example, this session's own
neural-timing measurements in `docs/HARDWARE.md`/`docs/EXPERIMENTS.md` used
a 2-of-5-seed reduced sweep and are labeled as such throughout.

To run a reduced version of any principal config yourself without editing
the YAML file, use `beyond_pinns.config.build_full_config` directly with a
`seeds`/`n_iterations` override, or copy a `configs/general/*.yaml` file and
add `seeds: [0]` / `n_iterations: 20` -- keep the copy's `description` field
prefixed `REDUCED-COST:` so it's unambiguous in any generated output.

## 3. Full manuscript reproduction

**This is expensive.** `scripts/reproduce_all.sh` prints the expected
computational requirements and asks for deliberate confirmation before
running anything -- it never launches an expensive campaign silently.

Individual principal experiments (the manuscript's actual selected
pure/hybrid method per benchmark, `configs/method_mapping.yaml`), each with
the full historical 5 seeds and 300 iterations (Part B) or 10 seeds (Part A):

```bash
# Part B (general/hybrid), one command per (benchmark, model):
python -m beyond_pinns.run --config configs/general/ms_pure.yaml
python -m beyond_pinns.run --config configs/general/ms_hybrid.yaml
python -m beyond_pinns.run --config configs/general/jf_pure.yaml
python -m beyond_pinns.run --config configs/general/jf_hybrid.yaml
python -m beyond_pinns.run --config configs/general/ls_pure.yaml
python -m beyond_pinns.run --config configs/general/ls_hybrid.yaml
python -m beyond_pinns.run --config configs/general/rc_pure.yaml
python -m beyond_pinns.run --config configs/general/rc_hybrid.yaml

# Part A (smooth), representative methods (add more configs/smooth/*.yaml
# following the same pattern for the remaining methods in docs/EXPERIMENTS.md):
python -m beyond_pinns.run --config configs/smooth/linear_dsgnar_weak.yaml
python -m beyond_pinns.run --config configs/smooth/linear_plain_gn_weak_tsvd.yaml
python -m beyond_pinns.run --config configs/smooth/nonlinear_dsgnar_weak.yaml

# Optional, requires external GNDRM (see docs/PROVENANCE.md):
python -m beyond_pinns.run --config configs/smooth/linear_hao_gn_deep_ritz_OPTIONAL.yaml
```

Add `--dry-run` to any of these to print the fully-resolved historical
`CONFIG` dict (and which legacy script would run) without executing
anything; add `--print-config-only` for just the JSON, useful for scripting
or diffing against `results/reference/method_mapping.json`.

## How configuration works (for anyone extending this)

`python -m beyond_pinns.run --config <yaml>` reads the YAML spec,
reconstructs the exact historical `CONFIG` dict for that one experiment
(`beyond_pinns.config.build_full_config`, which parses the `_HISTORICAL_
CONFIG` literal directly out of the legacy script's source via `ast.
literal_eval` -- no execution, no drift risk from re-transcription) with
every flag except the one requested target zeroed out (same isolation
convention already used by the paper's own controlled-timing study), then
runs the corresponding legacy script (`src/beyond_pinns/legacy/part_{a,b}_
*.py`) as a subprocess, passing the resolved config via the
`BEYOND_PINNS_CONFIG_JSON` environment variable. The legacy scripts
themselves are run exactly as they always were (`python script.py`) -- this
release does not wrap 4000+ lines of notebook-style trusted scientific code
into an importable function merely for configurability. See
`tests/test_config_equivalence.py` for the regression check that this
reconstruction is faithful to the historical CONFIG structure.

## Randomness and determinism

- **Only `jax.random.PRNGKey`-based randomness is used anywhere in this
  codebase** -- no `numpy.random` or Python stdlib `random` seeding exists
  in either legacy script (confirmed by audit).
- **Part B (general/hybrid) seeds**: `[0, 1, 2, 3, 4]` -- 5 seeds.
- **Part A (smooth) seeds**: `list(range(10))` = `[0, ..., 9]` -- **10
  seeds**, not 5. Do not assume a uniform seed count across the two parts.
- The full-reproduction configs above use exactly these seed sets (the
  historical `CONFIG["global"]["seeds"]`); only `_smoke`/reduced-cost
  configs override them.
- **No bitwise-determinism claim across environments.** JAX/XLA/CUDA do not
  guarantee bit-identical floating-point results across different GPU
  models, CUDA/driver versions, or jax/jaxlib versions -- only
  same-environment (same GPU, same software stack) runs are expected to
  reproduce exactly. Cross-environment reproduction should match to within
  ordinary floating-point/algorithmic tolerance, not bit-for-bit. This is
  documented, not something this release can control.
