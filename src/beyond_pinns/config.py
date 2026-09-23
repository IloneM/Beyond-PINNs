"""Thin YAML -> historical-CONFIG-dict bridge.

Design principle (see docs/REPRODUCIBILITY.md and docs/PROVENANCE.md): the
historical `femennstein-petrov-galerkin-experiments-fixed.py` (Part B) and
`energy_vs_weak_benchmarks_gn_strong_weak.py` (Part A) CONFIG dicts encode
ONLY which experiment(s) run (a tree of boolean flags per benchmark/model
class/test family) plus a handful of global settings (seeds, iteration
budget, output directory). Architecture, optimizer hyperparameters,
quadrature order, and FE discretization are NOT threaded through CONFIG at
all in the historical code -- they are fixed constants defined elsewhere in
each script, identical for every run of a given benchmark. This module does
not fabricate configurability that was never there: YAML configs expose
exactly what CONFIG historically exposed, plus read-only/informational
fields (architecture, quadrature, fe_mesh, ...) recorded for documentation
and traceability (see docs/EXPERIMENTS.md), not wired to anything.

Rather than re-transcribing the historical CONFIG structure a second time
(risking silent drift from the source of truth), `historical_config()` reads
the exact `_HISTORICAL_CONFIG` dict literal straight out of the legacy
script's source via `ast.literal_eval` -- no import, no execution, no jax
dependency. `build_full_config()` takes a deep copy of that literal and
applies the same "zero every flag except the one target" isolation already
established (and used for the paper's own controlled-timing study) before
this release existed, then overlays the requested global settings (seeds,
n_iterations, output_root). See tests/test_config_equivalence.py for the
regression check that this reconstruction is byte-for-byte faithful to the
historical literal whenever every flag would be requested True.
"""
from __future__ import annotations

import ast
import copy
from pathlib import Path
from typing import Any

LEGACY_DIR = Path(__file__).resolve().parent / "legacy"

_HISTORICAL_VAR_NAME = "_HISTORICAL_CONFIG"

# Part B (general/hybrid): section -> test-family group name -> tuple of
# family identifiers. Mirrors the structure of _HISTORICAL_CONFIG exactly;
# used only to validate a YAML spec's requested family against what actually
# exists for that section (a typo'd family name should fail loudly, not
# silently produce an all-False CONFIG that trains nothing).
PART_B_MODEL_CLASSES = ("pure_petrov", "alternating_hybrid", "lagged_hybrid", "projected_hybrid")
PART_B_SECTION_OF_BENCHMARK = {"MS": "section_1", "JF": "section_2", "LS": "section_3", "RC": "section_4"}
PART_B_SCALAR_METHODS = ("exact_regression", "deep_ritz", "deep_ritz_l2_metric", "strong_pinn",
                          "strong_pinn_missing_line_negative_control",
                          # fem_baseline/fem_sweep drive a SEPARATE, independent standalone-FE
                          # candidate search (see docs/RESULTS.md item 4 / scripts/
                          # compute_standalone_fem.py) -- not needed for, and not part of, a
                          # single neural/hybrid training target. Zeroed for every section
                          # (including the target one), exactly matching the already-proven
                          # controlled-timing isolation convention (make_timed_script.py in the
                          # research tree): without this, a single-target run redundantly pays
                          # the full multi-minute FEM mesh-candidate sweep for all 4 benchmarks.
                          "fem_baseline", "fem_sweep")


def _legacy_script_path(part: str) -> Path:
    if part == "general":
        return LEGACY_DIR / "part_b_general.py"
    if part == "smooth":
        return LEGACY_DIR / "part_a_smooth.py"
    raise ValueError(f"Unknown part {part!r}; expected 'general' (Part B) or 'smooth' (Part A).")


_SAFE_CONFIG_EVAL_NAMES = {"list": list, "range": range}


def _safe_eval_config_node(node: ast.AST):
    """Evaluate a CONFIG dict AST node without executing the surrounding
    script. Tries `ast.literal_eval` first (handles every plain literal);
    falls back to `eval` under a minimal, explicit namespace (`list`,
    `range` only, no builtins) for the few non-literal expressions the
    historical CONFIG dicts actually contain (e.g. Part A's
    `"seeds": list(range(10))`). This only ever evaluates the legacy
    script's OWN trusted source (never external/user input), so a scoped
    eval is appropriate here in place of hand-rolling an AST transformer for
    every expression shape that might appear in a config literal."""
    try:
        return ast.literal_eval(node)
    except ValueError:
        code = compile(ast.Expression(body=node), "<historical CONFIG literal>", "eval")
        return eval(code, {"__builtins__": {}}, dict(_SAFE_CONFIG_EVAL_NAMES))  # noqa: S307


def historical_config(part: str) -> dict:
    """Return the exact historical `_HISTORICAL_CONFIG` dict literal for
    `part`, parsed directly from the legacy script's source (no execution
    of the surrounding script -- see _safe_eval_config_node)."""
    source = _legacy_script_path(part).read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id == _HISTORICAL_VAR_NAME:
                return _safe_eval_config_node(node.value)
    raise RuntimeError(f"Could not find `{_HISTORICAL_VAR_NAME} = {{...}}` in {_legacy_script_path(part)}")


def _zero_all_part_b_flags(cfg: dict) -> None:
    for section in ("section_1", "section_2", "section_3", "section_4"):
        for scalar in PART_B_SCALAR_METHODS:
            if scalar in cfg[section]:
                cfg[section][scalar] = False
        for model_class in PART_B_MODEL_CLASSES:
            if model_class in cfg[section]:
                for family in cfg[section][model_class]:
                    cfg[section][model_class][family] = False
        cfg[section]["enabled"] = False


def build_full_config_general(spec: dict) -> dict:
    """Build a full Part B CONFIG dict from a `part: general` YAML spec.

    Required spec keys: `benchmark` (MS/JF/LS/RC), `model_class`
    (pure_petrov/projected_hybrid/lagged_hybrid/alternating_hybrid), `family`
    (the exact internal test-family identifier, e.g. h01_green_quadrature).
    Optional: `seeds`, `n_iterations`, `output_root` (override the
    corresponding `global` entries).
    """
    cfg = copy.deepcopy(historical_config("general"))
    benchmark = spec["benchmark"]
    if benchmark not in PART_B_SECTION_OF_BENCHMARK:
        raise ValueError(f"Unknown benchmark {benchmark!r}; expected one of {list(PART_B_SECTION_OF_BENCHMARK)}")
    section = PART_B_SECTION_OF_BENCHMARK[benchmark]
    model_class = spec["model_class"]
    family = spec["family"]
    if model_class not in cfg[section] or model_class not in PART_B_MODEL_CLASSES:
        raise ValueError(f"Unknown model_class {model_class!r} for benchmark {benchmark!r}")
    if family not in cfg[section][model_class]:
        raise ValueError(
            f"Unknown test family {family!r} for benchmark {benchmark!r}/{model_class!r}; "
            f"available: {sorted(cfg[section][model_class])}"
        )

    _zero_all_part_b_flags(cfg)
    cfg[section]["enabled"] = True
    cfg[section][model_class][family] = True

    if "seeds" in spec:
        cfg["global"]["seeds"] = list(spec["seeds"])
    if "n_iterations" in spec:
        cfg["global"]["n_iterations"] = int(spec["n_iterations"])
    if "output_root" in spec:
        cfg["global"]["output_root"] = str(spec["output_root"])
    return cfg


def build_full_config_smooth(spec: dict) -> dict:
    """Build a full Part A CONFIG dict from a `part: smooth` YAML spec.

    Required spec keys: `problem` (linear/nonlinear), `method` (the exact
    key under CONFIG[problem]["methods"], e.g. dsgnar_weak). Optional:
    `seeds`, `activations`, `output_root`.
    """
    cfg = copy.deepcopy(historical_config("smooth"))
    problem = spec["problem"]
    if problem not in ("linear", "nonlinear"):
        raise ValueError(f"Unknown problem {problem!r}; expected 'linear' or 'nonlinear'")
    method = spec["method"]
    if method not in cfg[problem]["methods"]:
        raise ValueError(f"Unknown method {method!r} for problem {problem!r}; "
                          f"available: {sorted(cfg[problem]['methods'])}")

    for p in ("linear", "nonlinear"):
        cfg[p]["enabled"] = (p == problem)
        for m in cfg[p]["methods"]:
            cfg[p]["methods"][m] = False
    cfg[problem]["methods"][method] = True

    if "seeds" in spec:
        cfg["global"]["seeds"] = list(spec["seeds"])
    if "activations" in spec:
        cfg["global"]["activations"] = list(spec["activations"])
    if "output_root" in spec:
        cfg["global"]["output_root"] = str(spec["output_root"])
    return cfg


def build_full_config(spec: dict) -> dict:
    """Dispatch to build_full_config_{general,smooth} based on spec['part']."""
    part = spec["part"]
    if part == "general":
        return build_full_config_general(spec)
    if part == "smooth":
        return build_full_config_smooth(spec)
    raise ValueError(f"spec['part'] must be 'general' or 'smooth', got {part!r}")


def legacy_script_for(part: str) -> Path:
    return _legacy_script_path(part)
