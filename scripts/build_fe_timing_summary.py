# [Copied from the research tree with minimal path/import adaptation only -- see docs/PROVENANCE.md.]
import json, statistics
from pathlib import Path

# Raw fresh-process cold samples, captured from the 20 independent
# `python run_fe_single_cold.py <TAG>` invocations (5 per benchmark), each a
# brand-new interpreter (see report for the exact remote session transcript).
COLD_RAW = {
    "MS": [0.45144297694787383, 0.11096957605332136, 0.11333881295286119, 0.11309885513037443, 0.11013971595093608],
    "JF": [0.5055521151516587, 0.4717476109508425, 0.4681918181013316, 0.47480324003845453, 0.4664729868527502],
    "LS": [0.439204263035208, 0.4387104611378163, 0.44194099493324757, 0.4433344090357423, 0.4368564870674163],
    "RC": [2.3289272121619433, 2.3290317880455405, 2.343744465149939, 2.3405815670266747, 2.3236274768132716],
}
COLD_ACCURACY = {
    "MS": dict(n_dofs=383, rel_l2=2.2311183630653737e-07, rel_h1=7.859153077858996e-05),
    "JF": dict(n_dofs=1457, rel_l2=2.9633436091752755e-14, rel_h1=5.330785172798486e-14),
    "LS": dict(n_dofs=1435, rel_l2=0.0014001855136551984, rel_h1=0.016975878248022613),
    "RC": dict(n_dofs=2409, rel_l2=0.0006433252645026393, rel_h1=0.02724668875578721),
}

OUT = Path("petrov_galerkin_results/controlled_timings")

for tag in ("MS", "JF", "LS", "RC"):
    existing = json.load(open(OUT / f"timing_FE_{tag}.json"))
    warm_reps = existing["T_FE_total_per_rep"][1:]  # drop within-process rep 0
    cold = COLD_RAW[tag]

    def stats(xs):
        xs = sorted(xs)
        return dict(median=statistics.median(xs),
                    q1=statistics.quantiles(xs, n=4, method="inclusive")[0] if len(xs) >= 2 else xs[0],
                    q3=statistics.quantiles(xs, n=4, method="inclusive")[2] if len(xs) >= 2 else xs[0],
                    samples=xs)

    updated = dict(existing)
    updated["T_FE_cold"] = dict(
        definition="Each of the 5 samples is a SEPARATE fresh Python process "
                    "(run_fe_single_cold.py invoked independently 5 times): fresh "
                    "interpreter, fresh JAX/XLA state, no in-memory compilation reuse. "
                    "No persistent JAX compilation cache is configured "
                    "(JAX_COMPILATION_CACHE_DIR unset; no jax cache dir found) -- the only "
                    "thing that CAN persist across these processes is the NVIDIA "
                    "driver-level PTX/SASS ComputeCache (~/.nv/ComputeCache), which is "
                    "irrelevant here since JAX_PLATFORMS=cpu was set (no GPU/PTX kernels "
                    "involved in this FE benchmark).",
        **stats(cold),
        **COLD_ACCURACY[tag],
    )
    updated["T_FE_warm"] = dict(
        definition="Median/IQR of repetitions 2-5 (index 1..4) of the ORIGINAL "
                    "5-repetition single-process run; repetition 0 of that run is "
                    "excluded here since it is itself a within-process cold sample, "
                    "now superseded by T_FE_cold above.",
        **stats(warm_reps),
    )
    updated["cold_warm_ratio_median"] = updated["T_FE_cold"]["median"] / updated["T_FE_warm"]["median"]
    with open(OUT / f"timing_FE_{tag}.json", "w") as fh:
        json.dump(updated, fh, indent=2)
    print(f"{tag}: cold median={updated['T_FE_cold']['median']:.4f}s  "
          f"warm median={updated['T_FE_warm']['median']:.4f}s  "
          f"ratio={updated['cold_warm_ratio_median']:.2f}x")
    print(f"     cold accuracy: rel_l2={COLD_ACCURACY[tag]['rel_l2']:.4e} rel_h1={COLD_ACCURACY[tag]['rel_h1']:.4e}")
