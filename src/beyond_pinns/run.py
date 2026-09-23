"""Thin CLI entry point around the trusted legacy experiment scripts.

    python -m beyond_pinns.run --config configs/general/ms_hybrid.yaml

Reads a YAML spec, reconstructs the exact historical CONFIG dict for the
single requested experiment (see beyond_pinns.config), and runs the
corresponding legacy script (src/beyond_pinns/legacy/part_{a,b}_*.py)
UNMODIFIED, passing the reconstructed CONFIG in via the
BEYOND_PINNS_CONFIG_JSON environment variable (see that script's own
"PORTABILITY SHIM" comment). The legacy script is run as a subprocess (not
imported), exactly reproducing how it has always been run (`python
script.py`) -- this deliberately avoids wrapping 4000+ lines of
notebook-style top-level code in an importable function, which would be
exactly the kind of "substantial refactor of trusted scientific code" this
release avoids.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

from beyond_pinns.config import build_full_config, legacy_script_for

REPO_ROOT = Path(__file__).resolve().parents[2]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m beyond_pinns.run",
        description="Run a single named Beyond PINNs experiment from a YAML config.",
    )
    parser.add_argument("--config", required=True, type=Path,
                         help="Path to a YAML experiment spec, e.g. configs/general/ms_hybrid.yaml")
    parser.add_argument("--dry-run", action="store_true",
                         help="Print the reconstructed CONFIG dict and the command that would run, then exit.")
    parser.add_argument("--print-config-only", action="store_true",
                         help="Like --dry-run, but print ONLY the JSON-serialized CONFIG (for scripting/diffing).")
    args = parser.parse_args(argv)

    spec = yaml.safe_load(args.config.read_text())
    full_config = build_full_config(spec)
    script = legacy_script_for(spec["part"])

    if args.print_config_only:
        print(json.dumps(full_config, indent=2))
        return 0

    if args.dry_run:
        print(f"# Resolved from {args.config}")
        print(f"# Legacy script: {script.relative_to(REPO_ROOT)}")
        print(f"# Working directory: repository root")
        print(json.dumps(full_config, indent=2))
        return 0

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, dir=tempfile.gettempdir()) as f:
        json.dump(full_config, f)
        config_json_path = f.name

    env = os.environ.copy()
    env["BEYOND_PINNS_CONFIG_JSON"] = config_json_path
    # Drop an inherited MPLBACKEND (e.g. Jupyter's 'module://matplotlib_inline.
    # backend_inline', set when this CLI itself runs inside a notebook kernel):
    # the legacy script imports matplotlib.pyplot for its own plotting and
    # that backend value is invalid outside an actual notebook, crashing the
    # import. The legacy script never needs an interactive backend.
    env.pop("MPLBACKEND", None)
    # No default CPU-thread cap here: this is the general experiment runner,
    # used for real scientific runs as well as quick checks, and it must
    # inherit the user's own environment (including any thread-count
    # variables they've set) unchanged. Shared-resource protection for
    # validation/CI-style runs belongs in the validation scripts themselves
    # (scripts/reproduce_smoke.sh, scripts/reproduce_integration_smoke.sh),
    # not silently applied to every run through this CLI.
    print(f"[beyond_pinns.run] {spec.get('description', args.config.name)}")
    print(f"[beyond_pinns.run] running {script.relative_to(REPO_ROOT)} "
          f"(cwd=repository root, CONFIG from {config_json_path})")
    try:
        result = subprocess.run([sys.executable, str(script)], cwd=str(REPO_ROOT), env=env)
    finally:
        try:
            os.unlink(config_json_path)
        except OSError:
            pass
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
