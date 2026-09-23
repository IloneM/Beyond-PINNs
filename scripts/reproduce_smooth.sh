#!/usr/bin/env bash
# LEVEL 3 -- full manuscript reproduction of Part A (smooth linear/nonlinear
# benchmarks), using the historical 10-seed budget from configs/smooth/*.yaml.
#
# By default this excludes *_OPTIONAL.yaml configs (the Hao/Jin GN-Deep-Ritz
# and Muller-Zeinhofer-matched reference baselines, which require an external
# GNDRM checkout -- see docs/PROVENANCE.md). Pass --include-optional to also
# run those (after confirming GNDRM is installed).
#
# The historical full campaign (460 runs, both problems, both activations,
# all methods) took ~12h52m wall-clock on a single NVIDIA L40S (46 GB) --
# see running_fixes_weak-nrg_benchmark.md in the research tree. Running only
# the configs shipped here (a handful of representative methods, not the
# full campaign) will be substantially less, but each config still trains
# 10 seeds x up to ~1000 iterations.
set -euo pipefail
cd "$(dirname "$0")/.."

INCLUDE_OPTIONAL=false
CONFIRM=false
for arg in "$@"; do
  [ "$arg" = "--include-optional" ] && INCLUDE_OPTIONAL=true
  [ "$arg" = "--yes" ] && CONFIRM=true
done

CONFIGS=()
for f in configs/smooth/*.yaml; do
  if [[ "$f" == *_OPTIONAL.yaml ]]; then
    [ "$INCLUDE_OPTIONAL" = "true" ] && CONFIGS+=("$f")
  else
    CONFIGS+=("$f")
  fi
done

echo "This will run ${#CONFIGS[@]} full manuscript Part A experiments (10 seeds each):"
for c in "${CONFIGS[@]}"; do echo "  - $c"; done
echo
if [ "$INCLUDE_OPTIONAL" = "true" ]; then
  echo "NOTE: --include-optional was passed. GNDRM-dependent baselines will raise a clear"
  echo "error naming the exact required commit if GNDRM is not installed -- see"
  echo "docs/PROVENANCE.md section 2.3 for the clone/checkout command."
  echo
fi
echo "Historical reference: the full 460-run campaign took ~12h52m on one NVIDIA L40S."
echo "This subset (${#CONFIGS[@]} configs) will be substantially less but is still a"
echo "genuine 10-seed training campaign per config, not a quick check."
echo

if [ "$CONFIRM" != "true" ]; then
  read -r -p "Proceed? [y/N] " reply
  case "$reply" in
    [yY]|[yY][eE][sS]) ;;
    *) echo "Aborted."; exit 1 ;;
  esac
fi

for c in "${CONFIGS[@]}"; do
  echo "=== Running $c ==="
  python -m beyond_pinns.run --config "$c"
done

echo "Done. Results under the output_root recorded in each config (default: energy_vs_weak_benchmarks/)."
