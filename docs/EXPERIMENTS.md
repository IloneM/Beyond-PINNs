# Experiments

One subsection per benchmark. "Seeds" and "optimizer" names use the exact
identifiers from the historical `CONFIG` dicts (see
`src/beyond_pinns/legacy/part_{a,b}_*.py` and `src/beyond_pinns/config.py`).

## MS -- multiscale diffusion (Part B, section 1, 1D)

- **PDE**: 1D elliptic problem with a rapidly oscillating (multiscale)
  diffusion coefficient `A_eps`, period `eps = 1/16`.
- **Model**: pure Petrov-Galerkin and hybrid finite-element--neural
  (`pure_petrov` / `projected_hybrid`, plus `lagged_hybrid` / `alternating_hybrid`
  ablations).
- **Manuscript-selected test family**: `h01_green_quadrature` for both pure
  and hybrid (`configs/method_mapping.yaml`).
- **FE reference / compensator**: P4, 96 uniform elements, 383 DOFs.
- **Seeds**: `[0, 1, 2, 3, 4]` (`CONFIG["global"]["seeds"]`, Part B).
- **Expected output**: `petrov_galerkin_results/section_1/s1_{pure,proj}_h01_green_quadrature/seed_NNN/`.
- **Measured cost** (reduced-cost 2-seed sweep, not the full 5-seed campaign
  -- see `docs/HARDWARE.md`): median `T_optim` 299.5 s (pure) / 26.4 s
  (hybrid) per seed.

## JF -- discontinuous/jump forcing (Part B, section 2, 2D)

- **PDE**: 2D elliptic problem with a piecewise-constant (jump-discontinuous)
  forcing term across an interface.
- **Model**: pure / hybrid, as above.
- **Manuscript-selected test family**: `eigen_green_quadrature` (pure),
  `h01_tensor_quadrature` (hybrid) -- note these DIFFER between pure and
  hybrid; this is the manuscript's actual selection, not a typo.
- **FE reference / compensator**: P4, 12x8 interface-fitted mesh, 1457 DOFs.
- **Seeds**: `[0, 1, 2, 3, 4]`.
- **Expected output**: `petrov_galerkin_results/section_2/s2_pure_eigen_green_quadrature/` and `s2_proj_h01_tensor_quadrature/`.
- **Measured cost**: median `T_optim` 75.7 s (pure) / 36.2 s (hybrid).

## LS -- distributional line source (Part B, section 3, 2D)

- **PDE**: 2D elliptic problem with a line-source forcing term.
- **Scientific-integrity note**: the line source is implemented as a **true
  distributional functional** (a Dirac measure supported on the line, via
  dedicated line-source quadrature in the FE assembly and in the neural
  residual's weak-form evaluation) -- **not** a mollified Gaussian, a
  volumetric approximation, or a regularized narrow strip. This is preserved
  exactly as in the historical implementation; do not substitute a smoothed
  source when reproducing this benchmark unless explicitly running it as a
  separate, clearly-labeled ablation.
- **Model**: pure / hybrid.
- **Manuscript-selected test family**: `eigen_green_quadrature` for both pure
  and hybrid.
- **FE reference / compensator**: P3, 14x12 interface-fitted mesh, 1435 DOFs.
- **Seeds**: `[0, 1, 2, 3, 4]`.
- **Expected output**: `petrov_galerkin_results/section_3/s3_{pure,proj}_eigen_green_quadrature/`.
- **Measured cost**: median `T_optim` 134.5 s (pure) / 22.5 s (hybrid).

## RC -- reentrant corner (Part B, section 4, 2D)

- **PDE**: 2D Laplace/Poisson problem on an L-shaped domain with a reentrant
  corner singularity at the origin.
- **Model**: pure / hybrid.
- **Manuscript-selected test family**: `random_hats` for both pure and
  hybrid (compact, randomly-placed local finite-element hat test functions).
- **FE reference / compensator**: P2, power-graded mesh, `n_radial=16`,
  `n_angular=40`, `beta=2.5`, 2409 DOFs.
- **Seeds**: `[0, 1, 2, 3, 4]`.
- **Expected output**: `petrov_galerkin_results/section_4/s4_{pure,proj}_random_hats/`.
- **Measured cost**: median `T_optim` 9.7 s (pure) / 36.2 s (hybrid).
- **Aggregation note**: this benchmark's random-hat convergence figure uses
  last-observation-carried-forward for stopped seeds (see
  `docs/PROVENANCE.md` section 5.2/5.3 and `docs/RESULTS.md`).

## Smooth-linear (Part A, `CONFIG["linear"]`)

- **PDE**: `-u'' + u = f`, `u'(-1) = u'(1) = 0`, exact solution
  `u*(x) = cos(pi x)` on `(-1, 1)`.
- **Methods** (`CONFIG["linear"]["methods"]`, all "our own" and GNDRM-free
  except the one marked): `plain_gn_strong_tsvd`, `plain_gn_weak_tsvd`,
  `plain_gn_strong_tsvd_abs`, `plain_gn_weak_tsvd_abs`, `plain_gn_strong_lm`,
  `plain_gn_weak_lm` (ridge Gauss-Newton, strong/weak residual, TSVD or
  Levenberg-Marquardt regularization), `amstramgram_strong`,
  `amstramgram_weak`, `dsgnar_strong`, `dsgnar_weak` (two adaptive residual
  optimizers, strong/weak residual formulations), and `hao_gn_deep_ritz`
  (**OPTIONAL**, requires external GNDRM -- see
  `docs/PROVENANCE.md`).
- **Activations**: both `tanh` and `relu3` are run for every method
  (`CONFIG["global"]["activations"]`).
- **Seeds**: `list(range(10))` = `[0, 1, ..., 9]` -- note this is **10
  seeds**, not 5 (Part A and Part B use different seed counts).
- **Expected output**: `energy_vs_weak_benchmarks/linear/matched/{tanh,relu3}/<method>/seed_NNN/`.

## Smooth-nonlinear (Part A, `CONFIG["nonlinear"]`)

- **PDE**: `-u'' + u^3 = f`, same boundary conditions and exact solution as
  the linear case.
- **Methods** (`CONFIG["nonlinear"]["methods"]`): the same
  `plain_gn_*`/`amstramgram_*`/`dsgnar_*` family as above, plus
  `energy_ng_matched` (Muller-Zeinhofer Hessian metric, **OPTIONAL**,
  requires external GNDRM -- see `docs/PROVENANCE.md`; distinct from the
  separate, also-optional `reference_protocol`, which requires `ngrad`
  instead for a stricter apples-to-apples comparison against the published
  Muller-Zeinhofer implementation).
- **Activations**: both `tanh` and `relu3`.
- **Seeds**: `list(range(10))`.
- **Expected output**: `energy_vs_weak_benchmarks/nonlinear/matched/{tanh,relu3}/<method>/seed_NNN/`.
