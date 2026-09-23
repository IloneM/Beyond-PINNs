# Beyond PINNs

Reproducibility code for the numerical experiments in **"Beyond PINNs: A
Unified Gauss-Newton and Petrov-Galerkin Framework for Neural and Hybrid PDE
Solvers"** (Nilo Schwencke and Roland Maier).

## 1. Overview

This repository reproduces the paper's two experiment suites:

- **Part A ("smooth")**: linear and nonlinear 1D elliptic benchmarks
  comparing strong-residual and weak-residual formulations under ridge
  Gauss-Newton, DSGNAR, and related adaptive optimizers.
- **Part B ("general"/hybrid)**: four 2D/1D benchmarks -- MS (multiscale
  diffusion), JF (discontinuous forcing), LS (distributional line source),
  RC (reentrant corner) -- each solved by a *pure* Petrov-Galerkin neural
  method and by a *hybrid finite element--neural* method, across several
  weak test-function families.

It is a release build derived from the authors' own research tree, verified
for scientific fidelity against the archived manuscript results (see
`docs/PROVENANCE.md` for exactly what changed and why).

## 2. Paper / associated publication

> **Beyond PINNs: A Unified Gauss-Newton and Petrov-Galerkin Framework for
> Neural and Hybrid PDE Solvers**
> Nilo Schwencke and Roland Maier
>
> arXiv:2609.20641 -- <https://arxiv.org/abs/2609.20641>
> DOI: [10.48550/arXiv.2609.20641](https://doi.org/10.48550/arXiv.2609.20641)

Repository: <https://github.com/IloneM/Beyond-PINNs>

See `CITATION.cff` for the machine-readable citation (software +
preferred-citation for the preprint above).

## 3. Two-layer release: code vs. results

This project is released as **two separate artifacts**:

- **CODE** (this repository, `beyond-pinns-code`, on GitHub): source,
  configs, tests, docs, figure/table generation scripts, and a handful of
  tiny synthetic test fixtures. **It does not contain manuscript experiment
  histories, results, or logs.**
- **RESULTS** (`beyond-pinns-results`, a separate versioned archive
  deposited on Zenodo): the complete trusted numerical outputs (per-seed
  histories, final solution fields, FE caches, timing measurements)
  underlying every current manuscript table and figure. The v1.0
  reference-results archive is available at
  <https://doi.org/10.5281/zenodo.22904196>.

  This is the DOI for the *results dataset*, not for the paper or the
  software; see §2 above for the paper's own DOI. See §14 below for the
  download workflow.

You only need the results archive if you want to *reproduce the published
figures/tables from stored data* rather than *run experiments yourself*.
See `docs/REPRODUCIBILITY.md` for the three concrete workflows (install/test
the code, run new experiments from scratch, reproduce published
figures/tables) and exactly which of the two artifacts each one needs.

## 4. Installation

```bash
git clone https://github.com/IloneM/Beyond-PINNs beyond-pinns-code
cd beyond-pinns-code
uv sync --extra test        # or: pip install -e ".[test]"
```

Optional extras: `cuda` (CUDA-enabled jax, needed for GPU training),
`plotting` (matplotlib, already a core dependency but isolable), `test`
(pytest).

`pinn` (the DSGNAR optimizer) is **required** for every Part B run. It is
not on PyPI and not vendored here, but `pyproject.toml` pins it as a direct
Git dependency at the exact historical commit, so the `uv sync`/`pip
install` command above installs it automatically -- see
`docs/REPRODUCIBILITY.md` and `docs/PROVENANCE.md` section 3.1 for exactly
which commit and why. `ngrad` and GaussNewtonDRM (GNDRM) are genuinely
**optional** external dependencies (needed only for specific reference
baselines, manual clone + `PYTHONPATH` install) -- see `docs/PROVENANCE.md`.

## 5. Quick start

Four reproduction levels, in order of cost -- see `docs/REPRODUCIBILITY.md`
for full detail:

```bash
# LEVEL 1 -- fast smoke test (<=2 min, no GPU/pinn/ngrad/GNDRM needed):
./scripts/reproduce_smoke.sh

# LEVEL 1b -- integration smoke test (~15 min, one real hybrid solve through
# the trusted historical Part B script; see its own header for why it isn't faster):
./scripts/reproduce_integration_smoke.sh

# LEVEL 2 -- reduced-cost scientific run (fewer seeds/iterations than the
# manuscript -- e.g. configs/general/ms_hybrid_smoke.yaml is one such config;
# NEVER presented as the manuscript numbers):
python -m beyond_pinns.run --config configs/general/ms_hybrid_smoke.yaml

# LEVEL 3 -- full manuscript reproduction (exact seeds/iterations; requires
# `pinn`, installed automatically, see Installation; expensive, see
# reproduce_general.sh/reproduce_smooth.sh/reproduce_all.sh for cost warnings):
python -m beyond_pinns.run --config configs/general/ms_hybrid.yaml

# Test suite (no GNDRM/pinn/ngrad required):
pytest
```

## 6. Repository structure

```
beyond-pinns-code/
├── configs/            # YAML experiment specs (general/, smooth/, method_mapping.yaml)
├── src/beyond_pinns/    # config loader, CLI, independent init/model/quadrature
│   └── legacy/          # minimally-adapted copies of the trusted research scripts
├── scripts/             # figure/table generation, FE reference computation, reproduce_*.sh
├── results/reference/   # EMPTY in a fresh clone -- extraction point for beyond-pinns-results
├── figures/reference/   # small set of already-regenerated manuscript figures
├── tests/                # pytest suite (see docs/REPRODUCIBILITY.md)
└── docs/                 # REPRODUCIBILITY, EXPERIMENTS, RESULTS, HARDWARE, PROVENANCE
```

## 7. Reproducing the smooth experiments

See `configs/smooth/*.yaml` and `docs/EXPERIMENTS.md`'s smooth-linear/
smooth-nonlinear sections. Quick example:

```bash
python -m beyond_pinns.run --config configs/smooth/linear_dsgnar_weak.yaml
```

## 8. Reproducing the general/hybrid experiments

See `configs/general/*.yaml`. `configs/method_mapping.yaml` is the
machine-readable record of the manuscript's selected pure/hybrid method per
benchmark (test family, run name, historical median errors) -- the 8
`configs/general/{ms,jf,ls,rc}_{pure,hybrid}.yaml` files correspond to it
exactly.

```bash
python -m beyond_pinns.run --config configs/general/rc_hybrid.yaml
```

## 9. Finite element references

The standalone FE reference computations (`scripts/compute_standalone_fem.py`)
use these selected configurations, verified against the original trusted
result files (see `docs/RESULTS.md`):

| benchmark | degree | mesh | DOFs |
|---|---|---|---|
| MS | P4 | 96 uniform elements | 383 |
| JF | P4 | 12x8 interface-fitted mesh | 1457 |
| LS | P3 | 14x12 interface-fitted mesh | 1435 |
| RC | P2 | power-graded, n_radial=16, n_angular=40, beta=2.5 | 2409 |

## 10. Figures and tables

Every manuscript figure generated from code has a corresponding script under
`scripts/make_figure_*.py`, reading stored results only (no retraining). See
`docs/RESULTS.md` for the full manuscript-item -> config -> result ->
script -> figure/table traceability table.

## 11. Timing experiments

Kept separate from accuracy experiments (`scripts/run_fe_timings.py`,
`scripts/run_fe_single_cold.py`, `scripts/build_fe_timing_summary.py`).
The FE timing baselines are the full, complete manuscript measurements. The
neural/hybrid timing sweep in `results/reference/timings/
neural_timings_reduced_2seed/` is an explicitly reduced-cost (2 of 5 seeds)
measurement, not the full manuscript campaign -- see `docs/HARDWARE.md`.

## 12. Computational requirements

GPU environment historically used for neural/hybrid training: jax 0.10.2
(CUDA build) on an NVIDIA RTX PRO 6000 Blackwell node. FE-only work used a
separate CPU-only environment (jax 0.5.0, numpy 2.1.3, scipy 1.15.2, Python
3.12.9) -- both are documented exactly (nothing beyond what was recorded) in
`docs/HARDWARE.md`, including known per-benchmark timing measurements.

## 13. Optional external baselines

- **GaussNewtonDRM (GNDRM)**: required only for the `hao_gn_deep_ritz` and
  `energy_ng_matched` reference baselines (Part A). No LICENSE upstream, so
  not vendored; see `docs/PROVENANCE.md` for the exact pinned commit and
  install steps. Every other method in this repository runs without it.
- **`ngrad`**: required only for the optional Muller-Zeinhofer reference
  protocol. See `docs/PROVENANCE.md`.

## 14. Reference results / Data

`results/reference/` is intentionally **empty** in this repository (see
section 3 above) -- it is the extraction point for the separate
`beyond-pinns-results` archive, deposited on Zenodo:

- **Dataset DOI: 10.5281/zenodo.22904196**
- **Link: <https://doi.org/10.5281/zenodo.22904196>**

Download and extract it with `scripts/download_reference_results.sh`, or
manually via the DOI link above, into `results/reference/`.

`figures/reference/` does ship a small set of already-regenerated
(bug-fixed, see `docs/PROVENANCE.md` section 5) manuscript figures directly
in this repository, for browsing without running anything. Full
traceability table: `docs/RESULTS.md`.

## 15. Citation

If you use this code, please cite the paper:

```bibtex
@article{schwencke2026beyondpinns,
  title   = {Beyond PINNs: A Unified Gauss-Newton and Petrov-Galerkin Framework for Neural and Hybrid PDE Solvers},
  author  = {Schwencke, Nilo and Maier, Roland},
  year    = {2026},
  eprint  = {2609.20641},
  archivePrefix = {arXiv},
  doi     = {10.48550/arXiv.2609.20641},
  url     = {https://arxiv.org/abs/2609.20641}
}
```

For the software itself specifically (e.g. a particular release version):

```bibtex
@software{schwencke2026beyondpinnscode,
  title  = {Beyond PINNs: A Unified Gauss-Newton and Petrov-Galerkin Framework for Neural and Hybrid PDE Solvers (code)},
  author = {Schwencke, Nilo and Maier, Roland},
  year   = {2026},
  url    = {https://github.com/IloneM/Beyond-PINNs},
  note   = {Software, Version 1.0.0}
}
```

See `CITATION.cff` for the machine-readable form (software metadata plus a
`preferred-citation` entry for the paper above).

## 16. License and provenance

The authors' own code in this repository is MIT-licensed (see `LICENSE`).
This does **not** extend to third-party components -- `pinn`, `ngrad`,
GaussNewtonDRM, and code adapted from SIREN-Init all retain their own terms
(none of them carry an upstream license). None of them are vendored, except
the SIREN-Init-derived plotting adaptation
(`scripts/make_figure_rc_corner_zoom.py`), for which Andrea Combette gave
explicit written permission (2026-09-22) to include this specific adapted
material with attribution -- see `docs/PROVENANCE.md` section 4.

Full accounting of every external dependency, adaptation, and the bug fixes
carried over from the research tree: `docs/PROVENANCE.md`.
