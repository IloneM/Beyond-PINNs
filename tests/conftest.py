import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# All manuscript experiments use float64 throughout (see docs/PROVENANCE.md /
# docs/EXPERIMENTS.md, section 11 "Precision"); the legacy scripts enable
# this globally themselves, but standalone test/comparison code (this test
# suite, and beyond_pinns.{initialization,models,quadrature} when used
# outside those scripts) must do it explicitly, or JAX silently defaults to
# float32 and every fixed-key/golden-fixture comparison below would fail
# for the wrong reason (a precision mismatch, not a real inequivalence).
import jax
jax.config.update("jax_enable_x64", True)
