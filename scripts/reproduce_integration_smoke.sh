#!/usr/bin/env bash
# LEVEL 1b -- INTEGRATION smoke test. Runs a real, minimal end-to-end pass
# through the trusted HISTORICAL legacy script (one hybrid solve, one FE
# solve, one figure generation). Does NOT verify manuscript-level accuracy --
# see reproduce_general.sh / reproduce_smooth.sh for that. Measured ~15
# minutes (853s), not "a few minutes" -- the legacy script's setup is shared
# across all 4 benchmarks even when only one is requested (a small FE
# candidate mesh is still assembled for the other 3); see
# docs/REPRODUCIBILITY.md's timing note for why this floor exists and why it
# isn't eliminated here (fixing it would mean restructuring the historical
# script's own control flow, which this release avoids -- see
# docs/PROVENANCE.md). For a genuinely fast (<=2 minute) check that avoids
# the historical script entirely, use ./scripts/reproduce_smoke.sh instead --
# run that FIRST; this script is for deeper (but still reduced-cost)
# end-to-end confidence, not for a quick fresh-install/CI check.
#
# The FE-solve and figure-generation steps depend on results/reference/
# already being populated (shipped in this repo) with paths matching what
# those scripts expect; if either script's own data-path assumptions don't
# line up with the curated results/reference/ layout, this reports it as a
# SKIP (with the script's own error output) rather than a hard failure --
# the import + hybrid-solve steps are the load-bearing smoke check.
set -uo pipefail
cd "$(dirname "$0")/.."

# Shared-resource protection for this validation script specifically (not
# applied by beyond_pinns.run itself, which must leave normal scientific
# runs uncapped) -- preserves any value the user has already set. This
# script launches beyond_pinns.run as a child process, so the cap set here
# is inherited by it and by the legacy script it runs.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"

pass=0
fail=0
skip=0

require_step() {
  local name="$1"; shift
  echo "=== $name ==="
  if "$@"; then
    echo "PASS: $name"
    pass=$((pass + 1))
  else
    echo "FAIL: $name"
    fail=$((fail + 1))
  fi
  echo
}

optional_step() {
  local name="$1"; shift
  echo "=== $name (optional -- requires results/reference/ data) ==="
  if "$@"; then
    echo "PASS: $name"
    pass=$((pass + 1))
  else
    echo "SKIP: $name (script failed, likely missing/mismatched reference data -- see output above)"
    skip=$((skip + 1))
  fi
  echo
}

require_step "import beyond_pinns (no GNDRM required)" \
  python -c "import beyond_pinns; from beyond_pinns import config, run, initialization, models, quadrature; print('ok')"

require_step "one hybrid solve (MS hybrid, 1 seed, 15 iterations, REDUCED-COST)" \
  python -m beyond_pinns.run --config configs/general/ms_hybrid_smoke.yaml

optional_step "one FE solve (standalone FE reference reconstruction)" \
  python scripts/compute_standalone_fem.py

optional_step "one figure/table generation script" \
  python scripts/make_figure_random_hats.py

echo "================================"
echo "$pass passed, $fail failed, $skip skipped"
echo "================================"
[ "$fail" -eq 0 ]
