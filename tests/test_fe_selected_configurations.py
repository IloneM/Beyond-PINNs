"""Verify the standalone FE reference configurations selected for the
manuscript's hybrid-budget comparison match the trusted values established
during the Phase A audit. Reads results/reference/general/section_N/
fem_baselines/selected.json directly (no jax needed); skips gracefully if
not yet present.
"""
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE = REPO_ROOT / "results" / "reference" / "general"

EXPECTED = {
    "section_1": dict(benchmark="MS", degree=4, description="96 uniform elements", n_dofs=383),
    "section_2": dict(benchmark="JF", degree=4, description="12x8 interface P4", n_dofs=1457),
    "section_3": dict(benchmark="LS", degree=3, description="14x12 interface P3", n_dofs=1435),
    "section_4": dict(benchmark="RC", degree=2, description="power P2 nr=16 na=40 beta=2.5", n_dofs=2409),
}


@pytest.mark.parametrize("section,expected", sorted(EXPECTED.items()))
def test_selected_fe_configuration_matches_manuscript(section, expected):
    path = REFERENCE / section / "fem_baselines" / "selected.json"
    if not path.exists():
        pytest.skip(f"{path} not present (results/reference/ not yet populated)")
    data = json.loads(path.read_text())
    sel = data["best_fem_h1_under_hybrid_budget"]
    assert sel["degree"] == expected["degree"], (section, "degree")
    assert sel["description"] == expected["description"], (section, "description")
    assert sel["n_dofs"] == expected["n_dofs"], (section, "n_dofs")


def test_all_four_sections_present_or_all_skipped():
    """Sanity check: don't silently pass with zero sections checked -- if
    results/reference exists at all, every section must be there."""
    if not REFERENCE.exists():
        pytest.skip("results/reference/ not yet populated")
    missing = [s for s in EXPECTED if not (REFERENCE / s / "fem_baselines" / "selected.json").exists()]
    assert not missing, f"results/reference/ exists but is missing: {missing}"
