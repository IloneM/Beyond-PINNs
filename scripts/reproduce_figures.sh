#!/usr/bin/env bash
# Regenerate every manuscript figure from stored results (no retraining).
# Requires the relevant result trees (results/reference/ for a quick check,
# or the full petrov_galerkin_results/ / energy_vs_weak_benchmarks/ trees
# from scripts/reproduce_general.sh / reproduce_smooth.sh for the complete
# figure set).
set -euo pipefail
cd "$(dirname "$0")/.."

for script in scripts/make_figure_*.py; do
  echo "=== $script ==="
  python "$script"
  echo
done

echo "Figures written under figures/ (see docs/RESULTS.md for the manuscript-item -> figure traceability table)."
