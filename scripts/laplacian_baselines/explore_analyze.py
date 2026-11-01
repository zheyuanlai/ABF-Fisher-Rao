#!/usr/bin/env python3
"""Analysis of the EXPLORATORY wider-bandwidth Laplacian runs (scripts/laplacian_baselines/make_explore.py).

Per cell (fresh seeds, three arms + Laplacian variants at identical seeds / budgets / save grids): eqb_metrics paired
contrasts of every Laplacian (bandwidth, scale) variant against ABF and against FR, selection activity, and the
marginal-flatness / death-location diagnostic of score_noise_diagnostic.py.  Writes
results/laplacian_baselines/analysis/explore_bw.{json,md}.  Secondary to analyze.py; nothing here changes the primary
tables (SUMMARY.md sections 1-6).

  python scripts/laplacian_baselines/explore_analyze.py [--cells R2C3 R2H80] [--n-boot 10000]
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze as A  # noqa: E402
import score_noise_diagnostic as D  # noqa: E402

M = A.M
ROOT = A.ROOT
EXPLORE = {"R2C3": ("R2C3_lta150_K1024_bw", 1024, "LTA 150 K K1024"), "R2H80": ("R2H80_lta80_K1024_bw", 1024, "LTA 80 K K1024")}


def _g(c):
    return f"{100 * c['G_median']:+.1f} % [{100 * c['G_ci95'][0]:+.1f}, {100 * c['G_ci95'][1]:+.1f}] ({c['wins']}/{c['n']})"


def analyze_cell(case, nboot):
    tag, n, label = EXPLORE[case]
    d = ROOT / "results/laplacian_baselines/explore" / tag
    system, scorer = A._scorer(case)
    seeds = sorted(int(re.match(rf"abf_N{n}_s(\d+)\.npz", p.name).group(1)) for p in d.glob(f"abf_N{n}_s*.npz"))
    abf = {s: A._read(case, s, "abf", d / f"abf_N{n}_s{s}.npz", system, scorer) for s in seeds}
    fr = {s: A._read(case, s, "fr", d / f"fr_N{n}_s{s}.npz", system, scorer) for s in seeds}
    variants = {}
    for p in sorted(d.glob(f"laplacian_bw*_sc*_N{n}_s*.npz")):
        m = re.match(rf"laplacian_bw([0-9.]+)_sc([0-9.]+)_N{n}_s(\d+)\.npz", p.name)
        variants.setdefault((float(m.group(1)), float(m.group(2))), {})[int(m.group(3))] = p
    out = {"case": case, "label": label, "n_seeds": len(seeds), "seeds": seeds,
           "note": "EXPLORATORY: fresh seeds, Laplacian KDE bandwidth and scale varied; not part of the predeclared protocol",
           "abf": _summary(abf), "fr": _summary(fr), "fr_vs_abf": {m_: M.paired_contrast({s: abf[s][m_] for s in seeds}, {s: fr[s][m_] for s in seeds}, n_boot=nboot)
                                                                for m_ in ("I_F", "e_F", "I_Fp", "e_Fp")},
           "variants": {}}
    out["fr"]["diag"] = D.arm_stats([str(d / f"fr_N{n}_s{s}.npz") for s in seeds])
    out["abf"]["diag"] = D.arm_stats([str(d / f"abf_N{n}_s{s}.npz") for s in seeds])
    for (bw, sc), paths in sorted(variants.items()):
        have = sorted(s for s in paths if s in abf)
        lap = {s: A._read(case, s, "laplacian", paths[s], system, scorer) for s in have}
        for s in have:
            if not np.array_equal(abf[s]["step"], lap[s]["step"]) or abf[s]["nfe"] != lap[s]["nfe"]:
                raise ValueError(f"{case} seed {s} bw {bw} sc {sc}: save grid / force evaluations differ from ABF")
        row = _summary(lap)
        row["n"] = len(have)
        row["vs_abf"] = {m_: M.paired_contrast({s: abf[s][m_] for s in have}, {s: lap[s][m_] for s in have}, n_boot=nboot) for m_ in ("I_F", "e_F", "I_Fp", "e_Fp")}
        row["vs_fr"] = {m_: M.paired_contrast({s: fr[s][m_] for s in have}, {s: lap[s][m_] for s in have}, n_boot=nboot) for m_ in ("I_F", "e_F", "I_Fp", "e_Fp")}
        row["diag"] = D.arm_stats([str(paths[s]) for s in have])
        out["variants"][f"bw{bw:g}_sc{sc:g}"] = dict(bw=bw, scale=sc, **row)
    return out


def _summary(rows):
    vals = list(rows.values())
    d = {m_: float(np.median([r[m_] for r in vals])) for m_ in ("I_F", "e_F", "I_Fp", "e_Fp")}
    d["repl_median"] = float(np.median([r["repl"] for r in vals]))
    d["essw_min_median"] = float(np.median([r["essw_min"] for r in vals])) if vals[0]["essw_min"] is not None else None
    d["maxfam_max_median"] = float(np.median([r["maxfam_max"] for r in vals])) if vals[0]["maxfam_max"] is not None else None
    d["n_safety_stops"] = sum(1 for r in vals if r["stop_reason"] and "safety" in r["stop_reason"].lower())
    d["per_seed"] = {int(s): {k: r[k] for k in ("I_F", "e_F", "I_Fp", "e_Fp", "repl", "essw_min", "maxfam_max", "stop_reason")} for s, r in rows.items()}
    return d


def markdown(results):
    L = ["# EXPLORATORY: Laplacian selection with a wider KDE bandwidth (LTA 150 K and 80 K, fresh seeds)", "",
         "Not part of the predeclared protocol (docs/LRS_LAPLACIAN_BASELINE.md).  Motivation: score_noise.md shows that at the "
         "0.1 rad bandwidth shared with FR the Laplacian score is ~99 % finite-K noise.  Here ABF, FR (unchanged, bandwidth 0.1) "
         "and Laplacian variants with kde_bandwidth {0.3, 0.5} rad x scale {0.1, 0.3, 1, 3} run on the same 8 fresh seeds.  "
         "G = (second - first) / first, median with seed-bootstrap 95 % CI, (wins / n).  Negative favours the second arm.  "
         "TV(time-avg) = total variation of the time-averaged post-fr_start marginal from uniform; 'density at deaths' = "
         "marginal density relative to uniform where the arm killed (a flattening rule kills at > 1).", ""]
    for case, r in results.items():
        L += [f"## {r['label']} ({r['n_seeds']} fresh seeds)", "",
              "| arm | bw | scale | replacements | min ESS/K | safety stops | TV(time-avg) | density at deaths | e_F(T) median | I_F median | G e_F: vs ABF | G I_F: vs ABF | G e_F: vs FR | G I_F: vs FR |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|"]
        a, f = r["abf"], r["fr"]
        L.append(f"| ABF | | | 0 | | | {a['diag']['tv_timeavg']:.3f} | | {a['e_F']:.4g} | {a['I_F']:.4g} | | | | |")
        L.append(f"| FR | 0.1 | | {f['repl_median']:.0f} | {f['essw_min_median']:.3f} | {f['n_safety_stops']} | {f['diag']['tv_timeavg']:.3f} | {f['diag']['dens_at_death']:.2f} | "
                 f"{f['e_F']:.4g} | {f['I_F']:.4g} | {_g(r['fr_vs_abf']['e_F'])} | {_g(r['fr_vs_abf']['I_F'])} | | |")
        for key, v in r["variants"].items():
            dd = v["diag"]
            dens = "" if dd["dens_at_death"] is None else f"{dd['dens_at_death']:.2f}"
            L.append(f"| Laplacian | {v['bw']:g} | {v['scale']:g} | {v['repl_median']:.0f} | {v['essw_min_median']:.3f} | {v['n_safety_stops']} | {dd['tv_timeavg']:.3f} | {dens} | "
                     f"{v['e_F']:.4g} | {v['I_F']:.4g} | {_g(v['vs_abf']['e_F'])} | {_g(v['vs_abf']['I_F'])} | {_g(v['vs_fr']['e_F'])} | {_g(v['vs_fr']['I_F'])} |")
        L.append("")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cells", nargs="*", default=list(EXPLORE))
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--out", default="results/laplacian_baselines/analysis")
    ap.add_argument("--lta-ref-dir", default=None)
    a = ap.parse_args(argv)
    A.LTA_REF_DIR = a.lta_ref_dir
    results = {}
    for case in a.cells:
        try:
            results[case] = analyze_cell(case, a.n_boot)
        except Exception as e:
            print(f"{case}: SKIPPED ({type(e).__name__}: {e})")
            continue
        r = results[case]
        print(f"{case}: ABF e_F {r['abf']['e_F']:.4g} FR e_F {r['fr']['e_F']:.4g} (G vs ABF {_g(r['fr_vs_abf']['e_F'])})")
        for key, v in r["variants"].items():
            print(f"  {key:12s} repl {v['repl_median']:6.0f} ess {v['essw_min_median']:.2f} TV {v['diag']['tv_timeavg']:.3f} dens@death {v['diag']['dens_at_death']}"
                  f" e_F {v['e_F']:.4g}  G vs ABF {_g(v['vs_abf']['e_F'])}  G vs FR {_g(v['vs_fr']['e_F'])}")
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "explore_bw.json").write_text(json.dumps(results, indent=1, default=float) + "\n")
    (out / "explore_bw.md").write_text(markdown(results))
    print(f"wrote {out / 'explore_bw.md'}")


if __name__ == "__main__":
    main()
