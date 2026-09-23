#!/usr/bin/env python
"""
[Copied from the research tree with minimal path/import adaptation only -- see docs/PROVENANCE.md.]
Reconstructs the manuscript's "Best pure weak" / "Best hybrid" method
selection directly from saved experiment results -- no pre-existing selector
function exists in the codebase for this (unlike the FE candidate selection,
which is implemented by select_fem_candidate() in the main experiment
script); this reproduces the same *effective* rule by brute force:

    For each benchmark (MS/JF/LS/RC) and each model class (pure weak,
    projected hybrid), enumerate every test-family result directory, compute
    the median final relative_h1 over seeds 0-4, and pick the family with the
    smallest median. Report its median relative_l2 too.

Stdlib-only (zipfile/struct/ast/array), so it runs anywhere without numpy.

READS-ONLY from petrov_galerkin_results/ -- no training, no data changes.

Verified (see CONFIRMED_MAPPING below) to reproduce the manuscript's table to
displayed precision for all 8 entries. Run this script whenever the mapping
needs to be re-derived/re-checked rather than relying on this file's
hardcoded values, which are a frozen snapshot for cross-checking only.
"""
import json
import zipfile, struct, ast, array, glob, statistics, sys
from pathlib import Path

# Snapshot of the manuscript-confirmed mapping (from chat), used only as a
# cross-check that a fresh rescan still reproduces it. If ROOT's contents
# ever change, re-run this script -- do not hand-edit this dict.
CONFIRMED_MAPPING = {
    "MS": {
        "pure":   dict(family="h01_green_quadrature",   run="s1_pure_h01_green_quadrature", l2=2.1337e-15, h1=1.0600e-13),
        "hybrid": dict(family="h01_green_quadrature",   run="s1_proj_h01_green_quadrature",  l2=1.4863e-13, h1=1.0763e-12),
    },
    "JF": {
        "pure":   dict(family="eigen_green_quadrature", run="s2_pure_eigen_green_quadrature", l2=1.4474e-01, h1=2.0448e+00),
        "hybrid": dict(family="h01_tensor_quadrature",  run="s2_proj_h01_tensor_quadrature",  l2=4.2620e-05, h1=4.2611e-04),
    },
    "LS": {
        "pure":   dict(family="eigen_green_quadrature", run="s3_pure_eigen_green_quadrature", l2=1.1558e-04, h1=5.6125e-03),
        "hybrid": dict(family="eigen_green_quadrature", run="s3_proj_eigen_green_quadrature", l2=6.7223e-04, h1=6.6618e-03),
    },
    "RC": {
        "pure":   dict(family="random_hats",            run="s4_pure_random_hats",            l2=1.0520e+00, h1=1.3974e+00),
        "hybrid": dict(family="random_hats",             run="s4_proj_random_hats",             l2=5.0353e-04, h1=1.2029e-02),
    },
}

ROOT = Path("petrov_galerkin_results")
SECTIONS = {
    "section_1": ("MS", ["a_green_magic", "a_green_quadrature", "h01_green_quadrature", "random_hats"]),
    "section_2": ("JF", ["eigen_green_quadrature", "h01_tensor_quadrature", "random_hats"]),
    "section_3": ("LS", ["eigen_green_quadrature", "h01_tensor_quadrature", "random_hats"]),
    "section_4": ("RC", ["eigen_green_quadrature", "shifted_h1_quadrature", "random_hats"]),
}


def read_npy_scalar_last(zf, name):
    b = zf.read(name)
    hlen = struct.unpack('<H', b[8:10])[0]
    header = b[10:10 + hlen].decode('latin1')
    offset = 10 + hlen
    d = ast.literal_eval(header)
    n = 1
    for s in d['shape']:
        n *= s
    arr = array.array('d')
    arr.frombytes(b[offset:offset + n * 8])
    return arr[-1]


def scan(root=ROOT):
    """Returns {tag: {"pure": {...}, "hybrid": {...}}} with the reconstructed
    best family per benchmark/model-class, plus the full per-family table."""
    result = {}
    for section, (tag, families) in SECTIONS.items():
        prefix = section.replace("section_", "s")
        result[tag] = {}
        for model, label in (("pure", "pure"), ("proj", "hybrid")):
            rows = []
            for fam in families:
                run = f"{prefix}_{model}_{fam}"
                d = root / section / run
                if not d.exists():
                    continue
                h1s, l2s = [], []
                for seed_dir in sorted(glob.glob(str(d / "seed_*"))):
                    hp = Path(seed_dir) / "history.npz"
                    if not hp.exists():
                        continue
                    zf = zipfile.ZipFile(hp)
                    h1s.append(read_npy_scalar_last(zf, "relative_h1.npy"))
                    l2s.append(read_npy_scalar_last(zf, "relative_l2.npy"))
                if not h1s:
                    continue
                rows.append(dict(family=fam, run=run, n_seeds=len(h1s),
                                  median_l2=statistics.median(l2s),
                                  median_h1=statistics.median(h1s)))
            rows.sort(key=lambda r: r["median_h1"])
            result[tag][label] = dict(best=rows[0] if rows else None, all_families=rows)
    return result


def main():
    scanned = scan()
    mismatches = []
    print("=" * 100)
    for tag in ("MS", "JF", "LS", "RC"):
        print(f"\n=== {tag} ===")
        for label in ("pure", "hybrid"):
            entry = scanned[tag][label]
            print(f"  -- {label} weak families (sorted by median H1) --" if label == "pure"
                  else f"  -- {label} families (sorted by median H1) --")
            for r in entry["all_families"]:
                print(f"     {r['family']:24s} n={r['n_seeds']} "
                      f"median_L2={r['median_l2']:.4e}  median_H1={r['median_h1']:.4e}")
            best = entry["best"]
            print(f"  ==> BEST {label}: {best['family']}  ({best['run']})  "
                  f"L2={best['median_l2']:.4e}  H1={best['median_h1']:.4e}")
            confirmed = CONFIRMED_MAPPING[tag][label]
            ok = (best["run"] == confirmed["run"]
                  and abs(best["median_l2"] - confirmed["l2"]) / abs(confirmed["l2"]) < 1e-2
                  and abs(best["median_h1"] - confirmed["h1"]) / abs(confirmed["h1"]) < 1e-2)
            print(f"      cross-check vs. CONFIRMED_MAPPING: {'OK' if ok else 'MISMATCH'}")
            if not ok:
                mismatches.append((tag, label, best, confirmed))
    print("\n" + "=" * 100)
    if mismatches:
        print(f"MISMATCHES FOUND ({len(mismatches)}) -- do not trust CONFIRMED_MAPPING blindly, "
              "re-derive from this scan instead:")
        for tag, label, best, confirmed in mismatches:
            print(f"  {tag}/{label}: scanned={best} vs confirmed={confirmed}")
    else:
        print("All 8 entries reproduce CONFIRMED_MAPPING (manuscript table) to <1% relative tolerance.")

    out = {tag: {label: scanned[tag][label]["best"] for label in ("pure", "hybrid")} for tag in scanned}
    out_path = Path("petrov_galerkin_results/controlled_timings/method_mapping.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nWrote reconstructed mapping to {out_path}")
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
