# Hardware

This document records only what is actually preserved in the historical
research logs. Where a fact was not recorded, it says so explicitly rather
than guessing (per the release's "do not invent missing hardware
information" policy).

## FE-reference / standalone-FEM timing environment

Recorded directly in `petrov_galerkin_results/controlled_timings/
machine_metadata.json` in the historical research tree (captured
2026-09-14T21:50:18Z):

| field | value |
|---|---|
| OS | Linux 6.12.101+deb13-amd64, glibc 2.41 |
| CPU | AMD Ryzen 9 9950X, 16-Core Processor, 1 socket, 2 threads/core |
| GPU | NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 97887 MiB, driver 590.48.01 |
| Python | 3.12.9 (Anaconda) |
| NumPy | 2.1.3 |
| SciPy | 1.15.2 |
| JAX (this run) | 0.5.0, CPU backend (`TFRT_CPU_0`) -- the FE-only timing region does not require `jax_enable_x64` or a GPU |
| Thread caps | `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1` |

This is the environment that produced `results/reference/timings/
timing_FE_{MS,JF,LS,RC}.json` (the FE assembly/solve timing baselines).

## Neural / hybrid (DSGNAR) GPU environment

Used for all Part B neural and hybrid training runs, and for the reduced-cost
2-seed neural timing sweep referenced in `docs/EXPERIMENTS.md`:

| field | value |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Workstation Edition (same card) |
| JAX | 0.10.2, CUDA-enabled build (`.venv-cuda-x86_64` environment) |
| Precision | float64 (`jax.config.update("jax_enable_x64", True)`, set globally in both legacy scripts) |

CPU model, exact thread counts, CUDA toolkit/driver version pairing, and
Python/numpy/scipy versions **specific to this GPU environment** are not
separately recorded in the historical logs (only the CPU/FE-only environment
above has a full captured snapshot); if precise reproduction of this specific
environment matters, treat the FE-timing snapshot's CPU/OS fields as
representative of the same physical node, but do not assume the GPU
environment's Python/package versions matched the CPU one -- they are known
to differ (jax 0.5.0 vs. 0.10.2).

## Standalone-FE computation cost (for planning full reproduction)

From `results/reference/timings/timing_FE_*.json` (warm/repeated-run median,
5 repetitions each):

| benchmark | FE config | warm median total time |
|---|---|---|
| MS | P4, 96 elements, 383 DOFs | 0.040 s |
| JF | P4, 12x8 interface mesh, 1457 DOFs | 0.309 s |
| LS | P3, 14x12 interface mesh, 1435 DOFs | 0.345 s |
| RC | P2, power-graded mesh, 2409 DOFs | 2.172 s |

## Neural/hybrid optimization cost (reduced-cost measurement)

From this release's reduced-cost 2-seed neural timing sweep (not the full
5-seed manuscript campaign -- see `docs/EXPERIMENTS.md` and
`results/reference/timings/neural_timings_reduced_2seed/`), median
`T_optim` (GPU wall-clock, excluding one-time JIT compilation) per seed:

| benchmark | model | median T_optim |
|---|---|---|
| MS | pure | 299.5 s |
| MS | hybrid | 26.4 s |
| JF | pure | 75.7 s |
| JF | hybrid | 36.2 s |
| LS | pure | 134.5 s |
| LS | hybrid | 22.5 s |
| RC | pure | 9.7 s |
| RC | hybrid | 36.2 s |

These are real measurements from 2 of the manuscript's 5 seeds per method,
not extrapolations and not extrapolated to 5 seeds here -- treat as an
order-of-magnitude planning guide for the full 8-method x 5-seed campaign
(roughly 5/2 x the sum above as a rough upper bound, ignoring the one-time
per-process JIT compilation cost already excluded from `T_optim`).

Smooth-benchmark (Part A) timing: the original 460-run Part A campaign
(reported in the historical research tree's `running_fixes_weak-nrg_
benchmark.md`) completed in 12h52m on one NVIDIA L40S GPU (46GB), peak
process memory ~9.5GB -- a **different GPU model** than the RTX PRO 6000
Blackwell used for Part B and the FE timings above; full detail (CPU model,
driver/CUDA versions) for that specific run was not separately captured.
