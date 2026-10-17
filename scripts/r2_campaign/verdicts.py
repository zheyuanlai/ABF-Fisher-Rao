"""Frozen verdict rules of the R2 campaign (docs/r2_campaign/PREREGISTRATION.md section 4), applied to the output of
the frozen scripts/adaptive_fr/analyze_online.py (results/r2_campaign/analysis/<block>.json).  No statistic is
recomputed from run files: this script only re-reads the per-seed S_geo values and the secondary CIs that
analyze_online.py wrote, and applies the preregistered decision rules.

  superiority (positive cells): per-seed log S_geo -> median, percentile bootstrap (10 000 resamples, seed 2026, the
      campaign's atlas_lib convention): 95 % CI, and the Bonferroni CI for the 8 cells (1 - 0.05/8 = 99.375 %).
      SUPPORTED if the CI is > 1, CONTRADICTED if < 1, else UNRESOLVED.
  equivalence (ERI cells, TOST at alpha 0.05 = the 90 % CI inside the margin [1/1.25, 1.25]):
      EQUIVALENT if the 90 % CI lies inside the margin; HARMFUL if the 95 % CI is < 1; BENEFICIAL if the 95 % CI is > 1
      (if also inside the margin: EQUIVALENT takes precedence); otherwise UNRESOLVED.
  endpoint flag (both): e_F(T) relative change, 95 % CI: 'penalty' if the lower bound > 0; 'gain' if the upper
      bound < 0; 'removed' if the upper bound < +10 %; else 'unresolved'.

    python scripts/r2_campaign/verdicts.py            -> results/r2_campaign/analysis/verdicts.json + stdout table
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "convergence_fr"))
import atlas_lib as L  # noqa: E402

AN = os.path.join(ROOT, "results", "r2_campaign", "analysis")
MARGIN = 1.25
N_CELLS = 8
CELLS = [  # block, role, test
    ("R2C1", "ITE 300 K K1024 T50 (in-sample)", "superiority"),
    ("R2C2", "ITE 150 K K256 T100 (in-sample)", "superiority"),
    ("R2C5", "ERI 600 K K1024 T40 (in-sample; expected-negative)", "equivalence"),
    ("R2C3", "LTA 150 K K1024 T60 (held out; positive control)", "superiority"),
    ("R2C4", "LTA 300 K K1024 T60 (held out)", "superiority"),
    ("R2H80", "LTA 80 K K1024 T60 (held out; new strong-positive candidate)", "superiority"),
    ("R2H350", "LTA 350 K K1024 T60 (held out; new borderline candidate)", "superiority"),
    ("R2N256", "ERI 600 K K256 T40 (held-out allocation; expected-negative)", "equivalence"),
]


def cis(logS, n=10000, seed=L.BOOT_SEED):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(logS), size=(n, len(logS)))
    m = np.median(logS[idx], axis=1)
    q = lambda a: [float(np.exp(np.quantile(m, a / 2))), float(np.exp(np.quantile(m, 1 - a / 2)))]
    return dict(median=float(np.exp(np.median(logS))), ci95=q(0.05), ci90=q(0.10), ci_bonf=q(0.05 / N_CELLS),
                bonf_level=1 - 0.05 / N_CELLS)


def superiority(ci):
    return "SUPPORTED" if ci[0] > 1 else "CONTRADICTED" if ci[1] < 1 else "UNRESOLVED"


def equivalence(c):
    inside = c["ci90"][0] > 1 / MARGIN and c["ci90"][1] < MARGIN
    if inside:
        return "EQUIVALENT"
    if c["ci95"][1] < 1:
        return "HARMFUL"
    if c["ci95"][0] > 1:
        return "BENEFICIAL"
    return "UNRESOLVED"


def endpoint_flag(ci):
    if ci[0] > 0:
        return "penalty"
    if ci[1] < 0:
        return "gain"
    if ci[1] < 0.10:
        return "removed"
    return "unresolved"


def main():
    out = {}
    lines = ["| cell | role | test | stop t.u. median [IQR] (range) | never | S_geo [95 %] | 90 % | Bonferroni-8 | verdict | e_F(T) | endpoint |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for block, role, test in CELLS:
        p = os.path.join(AN, f"{block}.json")
        if not os.path.exists(p):
            out[block] = dict(role=role, test=test, status="NOT RUN")
            lines.append(f"| {block} | {role} | {test} | not run | | | | | | | |")
            continue
        d = json.load(open(p))
        S = np.array(d["primary"]["per_seed"], float)
        assert len(S) == 16 == d["n_seeds"], (block, len(S))
        c = cis(np.log(S))
        assert abs(c["median"] - d["primary"]["median"]) < 1e-9 and np.allclose(c["ci95"], d["primary"]["ci95"]), block
        sch = d["fr_schedule"]
        assert sch["events_after_stop_total"] == 0
        eft = d["secondary"]["eF_T"]
        v = dict(role=role, test=test, n_seeds=d["n_seeds"], **c,
                 verdict_95=superiority(c["ci95"]), verdict_bonf=superiority(c["ci_bonf"]),
                 tost=equivalence(c) if test == "equivalence" else None,
                 fr_faster=d["primary"]["fr_faster_seeds"],
                 stop_median=sch.get("stop_t_median"), stop_iqr=sch.get("stop_t_iqr"),
                 stop_range=[sch.get("stop_t_min"), sch.get("stop_t_max")], never_stopped=sch["never_stopped"],
                 reasons=sorted(set(sch.get("reason", []))), events_after_stop=sch["events_after_stop_total"],
                 eF_T=dict(median=eft["median_rel"], ci95=eft["ci95"], wins=eft["wins_fr"]), endpoint=endpoint_flag(eft["ci95"]),
                 establishment=d["establishment"], paired=all(d["paired_bitwise_before_fr_start"]))
        v["verdict"] = v["tost"] if test == "equivalence" else v["verdict_bonf"]
        out[block] = v
        lines.append(f"| {block} | {role} | {test} | {v['stop_median']:.2f} [{v['stop_iqr'][0]:.2f}, {v['stop_iqr'][1]:.2f}] "
                     f"({v['stop_range'][0]:.1f}-{v['stop_range'][1]:.1f}) | {v['never_stopped']} | **{c['median']:.2f} [{c['ci95'][0]:.2f}, {c['ci95'][1]:.2f}]** | "
                     f"[{c['ci90'][0]:.2f}, {c['ci90'][1]:.2f}] | [{c['ci_bonf'][0]:.2f}, {c['ci_bonf'][1]:.2f}] | "
                     f"**{v['verdict']}** (95 %: {v['verdict_95']}; FR faster {v['fr_faster']}/16) | "
                     f"{100 * eft['median_rel']:+.0f} % [{100 * eft['ci95'][0]:+.0f}, {100 * eft['ci95'][1]:+.0f}] ({eft['wins_fr']}/16) | {v['endpoint']} |")
    # rule-level questions (section 4.3 of the preregistration)
    have = lambda b: out.get(b, {}).get("verdict")
    q = {}
    q["preserves_acceleration"] = (have("R2C3") == "SUPPORTED" and have("R2H80") == "SUPPORTED"
                                   and all(have(b) in ("SUPPORTED", "UNRESOLVED") for b in ("R2C1", "R2C2", "R2C4", "R2H350"))
                                   and all(have(b) is not None for b in ("R2C1", "R2C2", "R2C3", "R2C4", "R2H80", "R2H350")))
    q["in_sample_reproduced"] = all(have(b) == "SUPPORTED" for b in ("R2C1", "R2C2"))
    q["avoids_eri_harm"] = all(have(b) in ("EQUIVALENT", "BENEFICIAL") for b in ("R2C5", "R2N256"))
    q["eri_harm_declared"] = any(have(b) == "HARMFUL" for b in ("R2C5", "R2N256"))
    q["generalises"] = bool(q["preserves_acceleration"] and q["avoids_eri_harm"])
    out["_questions"] = q
    out["_rules"] = dict(margin=MARGIN, n_cells=N_CELLS, bonferroni_level=1 - 0.05 / N_CELLS, tost_ci=0.90, boot=dict(n=10000, seed=L.BOOT_SEED))
    os.makedirs(AN, exist_ok=True)
    json.dump(out, open(os.path.join(AN, "verdicts.json"), "w"), indent=1)
    print("\n".join(lines))
    print(json.dumps(q, indent=1))


if __name__ == "__main__":
    main()
