"""Lightweight smoke coverage: every shipped YAML config under configs/general/
and configs/smooth/ must load and resolve into a structurally valid,
single-target CONFIG dict via beyond_pinns.config -- i.e. the "table" of
experiment specs the CLI/scripts consume is internally consistent. This is
deliberately redundant with tests/test_config_equivalence.py's own
`test_all_shipped_*_configs_load_and_build` (some overlap in smoke coverage
is fine); this file additionally checks EXACTLY one flag is True per config
and that every config is individually distinguishable (no two general
configs resolve to the identical target).
"""
from pathlib import Path

import pytest
import yaml

from beyond_pinns.config import (
    build_full_config_general, build_full_config_smooth,
    PART_B_MODEL_CLASSES, PART_B_SECTION_OF_BENCHMARK,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = REPO_ROOT / "configs"


def _general_paths():
    d = CONFIGS_DIR / "general"
    return sorted(d.glob("*.yaml")) if d.exists() else []


def _smooth_paths():
    d = CONFIGS_DIR / "smooth"
    return sorted(d.glob("*.yaml")) if d.exists() else []


def _count_true_flags_general(cfg):
    n = 0
    for section in ("section_1", "section_2", "section_3", "section_4"):
        for model_class in PART_B_MODEL_CLASSES:
            if model_class in cfg[section]:
                n += sum(1 for v in cfg[section][model_class].values() if v is True)
    return n


def _count_true_flags_smooth(cfg):
    n = 0
    for problem in ("linear", "nonlinear"):
        n += sum(1 for v in cfg[problem]["methods"].values() if v is True)
    return n


@pytest.mark.parametrize("path", _general_paths(), ids=lambda p: p.name)
def test_general_config_resolves_to_exactly_one_target(path):
    spec = yaml.safe_load(path.read_text())
    cfg = build_full_config_general(spec)
    assert _count_true_flags_general(cfg) == 1, path


@pytest.mark.parametrize("path", _smooth_paths(), ids=lambda p: p.name)
def test_smooth_config_resolves_to_exactly_one_target(path):
    spec = yaml.safe_load(path.read_text())
    cfg = build_full_config_smooth(spec)
    assert _count_true_flags_smooth(cfg) == 1, path


def test_no_two_general_configs_target_the_same_experiment():
    seen = {}
    for path in _general_paths():
        spec = yaml.safe_load(path.read_text())
        key = (spec.get("benchmark"), spec.get("model_class"), spec.get("family"),
               tuple(spec.get("seeds", [])), spec.get("n_iterations"))
        if key in seen:
            pytest.fail(f"{path.name} and {seen[key]} target the identical experiment {key}")
        seen[key] = path.name


def test_at_least_one_config_present_for_each_shipped_directory():
    if CONFIGS_DIR.exists():
        assert _general_paths(), "no configs/general/*.yaml found"
        assert _smooth_paths(), "no configs/smooth/*.yaml found"
