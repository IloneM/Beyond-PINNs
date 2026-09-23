#!/usr/bin/env bash
# LEVEL 1 -- FAST smoke test. Target: <= 2 minutes. Intended for fresh-install
# checks and CI. Deliberately avoids the historical legacy scripts entirely
# (no jax GPU, no pinn/ngrad/GNDRM needed) -- it exercises only this
# release's own new/independent code: package import, config loading, the
# independent initializer/model/quadrature, a tiny FE solve, the hybrid
# projection identity, and one tiny plotting/output path.
#
# For a real (but still reduced-cost) end-to-end pass through the trusted
# HISTORICAL Part B script (~15 minutes), see
# ./scripts/reproduce_integration_smoke.sh -- run this fast one first.
set -uo pipefail
cd "$(dirname "$0")/.."

# Shared-resource protection for this validation script specifically (not
# applied by beyond_pinns.run itself, which must leave normal scientific
# runs uncapped) -- preserves any value the user has already set.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"

t0=$(date +%s)

pass=0
fail=0

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

require_step "package import (no GNDRM required)" \
  python -c "import beyond_pinns; from beyond_pinns import config, run, initialization, models, quadrature; print('ok')"

require_step "config loading (--dry-run, no training)" \
  python -m beyond_pinns.run --config configs/general/ms_hybrid_smoke.yaml --print-config-only

require_step "independent initializer + shallow model + quadrature" \
  python -c "
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from jax import random
from beyond_pinns.initialization import init_shallow_mlp_params
from beyond_pinns.models import shallow_mlp
from beyond_pinns.quadrature import PiecewiseGaussLegendre1D

params = init_shallow_mlp_params([1, 8, 1], random.PRNGKey(0))
assert not any(jnp.any(jnp.isnan(p)) for p in (params[0][0], params[0][1], params[1]))
model = shallow_mlp(jnp.tanh)
val = model(params, jnp.array([0.3]))
assert jnp.isfinite(val)
q = PiecewiseGaussLegendre1D(npts=4)
integ = q.interval_quadpts(jnp.array([[0.0, 1.0]]), jnp.array([0.25]))
const_integral = float(integ(lambda x: jnp.ones((x.shape[0],))))
assert abs(const_integral - 1.0) < 1e-10, const_integral
print('init/model/quadrature ok')
"

require_step "tiny FE solve (4-element 1D Lagrange, not the manuscript's 96-element MS config)" \
  python -c "
import numpy as np
from beyond_pinns.fem_utils import solve_lagrange_1d, eval_lagrange_1d
bundle = solve_lagrange_1d(4, degree=1, A_func=lambda x: 1.0, f_func=lambda x: 1.0, n_quad=4)
xs = np.linspace(0.0, 1.0, 5)
vals = eval_lagrange_1d(bundle, xs)
assert np.all(np.isfinite(vals)), vals
print('tiny FE solve ok, n_dofs=', bundle['n_dofs'])
"

require_step "hybrid projection identity (tests/test_hybrid_projection_identity.py)" \
  python -m pytest tests/test_hybrid_projection_identity.py -q

require_step "one tiny plotting/output path" \
  env -u MPLBACKEND python -c "
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tempfile, os
fig, ax = plt.subplots()
ax.plot([0, 1, 2], [0, 1, 4])
out = os.path.join(tempfile.gettempdir(), 'beyond_pinns_smoke_check.png')
fig.savefig(out)
plt.close(fig)
assert os.path.exists(out) and os.path.getsize(out) > 0
os.remove(out)
print('plotting toolchain ok')
"

elapsed=$(( $(date +%s) - t0 ))
echo "================================"
echo "$pass passed, $fail failed -- ${elapsed}s elapsed (target <= 120s)"
echo "================================"
[ "$fail" -eq 0 ]
