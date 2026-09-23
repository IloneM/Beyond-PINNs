#!/usr/bin/env bash
# Regenerate the manuscript's numerical summary tables (per-seed and
# per-method CSVs) -- a side effect of the main comparison figure script,
# which reads stored results only (no retraining).
set -euo pipefail
cd "$(dirname "$0")/.."

python scripts/make_figure_main_comparison.py

echo
echo "Tables written: figures/petrov/petrov_seed_level_summary.csv, figures/petrov/petrov_method_summary.csv"
echo "(see docs/RESULTS.md for the manuscript-item -> table traceability chain)."
