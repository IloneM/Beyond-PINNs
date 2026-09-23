#!/usr/bin/env bash
# LEVEL 3 -- full manuscript reproduction of Part B (MS/JF/LS/RC, pure +
# hybrid), using the exact historical seeds [0,1,2,3,4] and iteration budget
# (300) baked into every configs/general/*.yaml (excluding *_smoke.yaml).
#
# This is NOT quick: 8 methods x 5 seeds x up to 300 DSGNAR iterations each.
# Measured (this session's reduced 2-seed sweep) per-seed median optimization
# time: MS pure ~300s, MS hybrid ~26s, JF pure ~76s, JF hybrid ~36s,
# LS pure ~135s, LS hybrid ~23s, RC pure ~10s, RC hybrid ~36s -- extrapolated
# to 5 seeds each, expect on the order of an hour of GPU time total (varies
# significantly by hardware; DSGNAR's adaptive stopping rule means seeds can
# also finish much earlier or run longer than this estimate).
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIGS=(configs/general/ms_pure.yaml configs/general/ms_hybrid.yaml
         configs/general/jf_pure.yaml configs/general/jf_hybrid.yaml
         configs/general/ls_pure.yaml configs/general/ls_hybrid.yaml
         configs/general/rc_pure.yaml configs/general/rc_hybrid.yaml)

CONFIRM=false
for arg in "$@"; do
  [ "$arg" = "--yes" ] && CONFIRM=true
done

echo "This will run ${#CONFIGS[@]} full manuscript experiments (5 seeds, 300 iterations each):"
for c in "${CONFIGS[@]}"; do echo "  - $c"; done
echo
echo "Estimated cost: on the order of an hour of GPU time (see script header for the"
echo "measured per-method timings this estimate is based on). This is a real training"
echo "campaign, not a quick check -- use scripts/reproduce_smoke.sh for a fast sanity check."
echo

if [ "$CONFIRM" != "true" ]; then
  read -r -p "Proceed with the full Part B reproduction? [y/N] " reply
  case "$reply" in
    [yY]|[yY][eE][sS]) ;;
    *) echo "Aborted."; exit 1 ;;
  esac
fi

for c in "${CONFIGS[@]}"; do
  echo "=== Running $c ==="
  python -m beyond_pinns.run --config "$c"
done

echo "Done. Results under the output_root recorded in each config (default: petrov_galerkin_results/)."
