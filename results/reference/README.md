# Reference results (not shipped in this repository)

This directory is the documented extraction point for the **Beyond PINNs
reference-results archive** (`beyond-pinns-results`, deposited on Zenodo:
<https://doi.org/10.5281/zenodo.22904196>). It is intentionally empty in a
fresh clone -- manuscript experiment histories/results are data, not code,
and are versioned separately.

To reproduce the manuscript's tables/figures, obtain the archive and extract
it here (so that `smooth/`, `general/`, `fem/`, and `timings/` end up as
direct subdirectories of this one), either via:

```bash
./scripts/download_reference_results.sh
```

or manually -- see `docs/REPRODUCIBILITY.md` for the exact archive layout
and extraction command.

Running new experiments from scratch (`beyond_pinns.run`) does not require
anything in this directory; it is only needed to *reproduce the published
figures/tables* via `scripts/reproduce_tables.sh` / `scripts/reproduce_figures.sh`.
