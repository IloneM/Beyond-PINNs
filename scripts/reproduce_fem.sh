#!/usr/bin/env bash
# Standalone finite-element reference reconstruction + FE timing benchmarks.
# Pure deterministic linear algebra -- no neural-network training, no GPU
# required. Should complete in well under a minute.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Standalone FE reference reconstruction (MS/JF/LS/RC) ==="
python scripts/compute_standalone_fem.py

echo
echo "=== FE timing benchmarks ==="
python scripts/run_fe_timings.py
