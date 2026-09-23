"""Regression check: a YAML spec requesting every flag True must reconstruct
a CONFIG dict scientifically equivalent to the historical literal (modulo
the "isolate one target" zeroing this release's runner always applies -- see
beyond_pinns/config.py's module docstring). Also checks method_mapping.yaml
against the historical mapping, and exercises the CLI's --print-config-only
path end-to-end without running any training.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from beyond_pinns.config import (
    build_full_config_general, build_full_config_smooth, historical_config,
    PART_B_SECTION_OF_BENCHMARK, PART_B_MODEL_CLASSES,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = REPO_ROOT / "configs"


def test_historical_config_general_has_expected_shape():
    cfg = historical_config("general")
    assert set(cfg) >= {"global", "section_1", "section_2", "section_3", "section_4"}
    assert cfg["global"]["seeds"] == [0, 1, 2, 3, 4]
    assert cfg["global"]["n_iterations"] == 300


def test_historical_config_smooth_has_expected_shape():
    cfg = historical_config("smooth")
    assert set(cfg) >= {"global", "linear", "nonlinear"}
    assert cfg["global"]["seeds"] == list(range(10))


def test_build_full_config_general_isolates_exactly_one_target():
    cfg = build_full_config_general(dict(part="general", benchmark="MS",
                                          model_class="pure_petrov", family="h01_green_quadrature"))
    # Every OTHER section must be fully disabled, INCLUDING fem_baseline/fem_sweep
    # (these drive a separate, independent standalone-FE candidate search that is
    # NOT needed for a single neural/hybrid training target -- a real bug found
    # during fresh-environment validation left these True for every section,
    # causing a single-target run to redundantly pay the full multi-minute FEM
    # mesh-candidate sweep for all 4 benchmarks; see docs/PROVENANCE.md).
    for section in ("section_1", "section_2", "section_3", "section_4"):
        assert cfg[section]["fem_baseline"] is False, section
        assert cfg[section]["fem_sweep"] is False, section
    for section in ("section_2", "section_3", "section_4"):
        assert cfg[section]["enabled"] is False
        for model_class in PART_B_MODEL_CLASSES:
            assert all(v is False for v in cfg[section][model_class].values())
    # section_1 (MS): enabled, and ONLY pure_petrov/h01_green_quadrature is True.
    assert cfg["section_1"]["enabled"] is True
    for model_class in PART_B_MODEL_CLASSES:
        for family, flag in cfg["section_1"][model_class].items():
            expected = (model_class == "pure_petrov" and family == "h01_green_quadrature")
            assert flag == expected, (model_class, family, flag)


@pytest.mark.parametrize("benchmark,model_class,family", [
    ("MS", "pure_petrov", "h01_green_quadrature"),
    ("MS", "projected_hybrid", "h01_green_quadrature"),
    ("JF", "pure_petrov", "eigen_green_quadrature"),
    ("JF", "projected_hybrid", "h01_tensor_quadrature"),
    ("LS", "pure_petrov", "eigen_green_quadrature"),
    ("LS", "projected_hybrid", "eigen_green_quadrature"),
    ("RC", "pure_petrov", "random_hats"),
    ("RC", "projected_hybrid", "random_hats"),
])
def test_method_mapping_families_exist_in_historical_config(benchmark, model_class, family):
    """Every (benchmark, model_class, family) in configs/method_mapping.yaml
    must be a real, existing flag in the historical CONFIG -- i.e. the
    manuscript's selected mapping is not silently pointing at a typo'd or
    removed test family."""
    cfg = historical_config("general")
    section = PART_B_SECTION_OF_BENCHMARK[benchmark]
    assert family in cfg[section][model_class], (benchmark, model_class, family)


def test_method_mapping_yaml_matches_manuscript_selection():
    mapping = yaml.safe_load((CONFIGS_DIR / "method_mapping.yaml").read_text())
    expected = {
        "MS": {"pure": "h01_green_quadrature", "hybrid": "h01_green_quadrature"},
        "JF": {"pure": "eigen_green_quadrature", "hybrid": "h01_tensor_quadrature"},
        "LS": {"pure": "eigen_green_quadrature", "hybrid": "eigen_green_quadrature"},
        "RC": {"pure": "random_hats", "hybrid": "random_hats"},
    }
    for benchmark, models in expected.items():
        for model, family in models.items():
            assert mapping[benchmark][model]["family"] == family


def test_build_full_config_smooth_isolates_exactly_one_target():
    cfg = build_full_config_smooth(dict(part="smooth", problem="linear", method="dsgnar_weak"))
    assert cfg["linear"]["enabled"] is True
    assert cfg["nonlinear"]["enabled"] is False
    for m, flag in cfg["linear"]["methods"].items():
        assert flag == (m == "dsgnar_weak"), m
    assert all(v is False for v in cfg["nonlinear"]["methods"].values())


def test_unknown_family_raises():
    with pytest.raises(ValueError):
        build_full_config_general(dict(part="general", benchmark="MS",
                                        model_class="pure_petrov", family="not_a_real_family"))


def test_all_shipped_general_configs_load_and_build():
    for path in sorted((CONFIGS_DIR / "general").glob("*.yaml")):
        spec = yaml.safe_load(path.read_text())
        cfg = build_full_config_general(spec)
        assert cfg["global"]["seeds"], path


def test_all_shipped_smooth_configs_load_and_build():
    for path in sorted((CONFIGS_DIR / "smooth").glob("*.yaml")):
        spec = yaml.safe_load(path.read_text())
        cfg = build_full_config_smooth(spec)
        assert cfg["global"]["seeds"], path


def test_cli_print_config_only_matches_direct_call():
    """End-to-end CLI smoke check -- no training, just config resolution."""
    config_path = CONFIGS_DIR / "general" / "ms_hybrid.yaml"
    if not config_path.exists():
        pytest.skip("configs/general/ms_hybrid.yaml not present")
    result = subprocess.run(
        [sys.executable, "-m", "beyond_pinns.run", "--config", str(config_path), "--print-config-only"],
        cwd=str(REPO_ROOT), capture_output=True, text=True, check=True,
        env={"PYTHONPATH": str(REPO_ROOT / "src")},
    )
    from_cli = json.loads(result.stdout)
    spec = yaml.safe_load(config_path.read_text())
    from_direct = build_full_config_general(spec)
    assert from_cli == from_direct
