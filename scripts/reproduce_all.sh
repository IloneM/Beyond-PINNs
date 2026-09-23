#!/usr/bin/env bash
# LEVEL 3 -- full manuscript reproduction, EVERYTHING.
#
# This chains reproduce_general.sh (Part B: 8 methods x 5 seeds x up to 300
# DSGNAR iterations, ~1 hour of GPU time by rough extrapolation from this
# session's measured per-method timings) + reproduce_smooth.sh (Part A:
# historically ~12h52m wall-clock for the full 460-run campaign on one
# NVIDIA L40S -- see running_fixes_weak-nrg_benchmark.md in the research
# tree; the configs shipped here are a representative subset, not the full
# campaign, but each is still a genuine 10-seed training run) +
# reproduce_fem.sh (seconds) + reproduce_figures.sh + reproduce_tables.sh
# (fast, read stored results only).
#
# TOTAL ESTIMATED COST: several hours of GPU time, dominated by Part A.
# This does NOT run silently -- it requires an explicit flag.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=================================================================="
echo " FULL MANUSCRIPT REPRODUCTION -- reproduce_all.sh"
echo "=================================================================="
echo
echo "This will run, in order:"
echo "  1. reproduce_general.sh  -- Part B, 8 methods x 5 seeds       (~1 hour, rough estimate)"
echo "  2. reproduce_smooth.sh   -- Part A, 10 seeds per method       (several hours, historical"
echo "                                                                  reference: ~12h52m for the"
echo "                                                                  full 460-run campaign)"
echo "  3. reproduce_fem.sh      -- standalone FE references + timing (seconds)"
echo "  4. reproduce_figures.sh  -- regenerate every figure           (fast, reads stored results)"
echo "  5. reproduce_tables.sh   -- regenerate summary tables         (fast, reads stored results)"
echo
echo "TOTAL ESTIMATED COST: several hours of GPU time. This is a genuine, expensive"
echo "training campaign, not a quick check. Use scripts/reproduce_smoke.sh first if you"
echo "have not already verified your installation works."
echo

FLAG_PRESENT=false
for arg in "$@"; do
  [ "$arg" = "--yes-i-understand-the-cost" ] && FLAG_PRESENT=true
done

if [ "$FLAG_PRESENT" != "true" ]; then
  echo "Refusing to proceed without explicit confirmation."
  echo "Re-run as: $0 --yes-i-understand-the-cost"
  exit 1
fi

./scripts/reproduce_general.sh --yes
./scripts/reproduce_smooth.sh --yes
./scripts/reproduce_fem.sh
./scripts/reproduce_figures.sh
./scripts/reproduce_tables.sh

echo "Full reproduction complete."
