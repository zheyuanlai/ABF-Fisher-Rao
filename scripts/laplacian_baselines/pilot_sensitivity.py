#!/usr/bin/env python3
"""Post-hoc scale sensitivity of the LRS Laplacian arm on the fresh-seed PILOT runs (SECONDARY analysis).

The pilot ladder (docs/LRS_LAPLACIAN_BASELINE.md, "NSCC execution") ran FR and Laplacian x {0.003, 0.01, 0.03, 0.1, 0.3}
on 8 fresh seeds per cell with the production budget / save grid, and the scale was frozen WITHOUT looking at
free-energy errors.  Once frozen, these runs can be scored like any other run: this script tabulates, per cell and scale,
the eqb_metrics errors of the Laplacian arm against the FR arm on the same fresh seeds (paired, same force evaluations).
There is no ABF arm in the pilot, so only Laplacian-vs-FR contrasts are available; the PRIMARY comparison is analyze.py
on the production seeds at the frozen scale.  Output: results/laplacian_baselines/analysis/pilot_sensitivity.{json,md}.

  python scripts/laplacian_baselines/pilot_sensitivity.py [--cells R2C3 gateway512 ...] [--n-boot 10000]
      [--ite300-refbase ... --ite150-refbase ... --lta-ref-dir ...]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze as A  # noqa: E402  (same scorers, readers and case tables as the primary analysis)

M = A.M
ROOT = A.ROOT
PILOT = ROOT / "results/laplacian_baselines/pilot"
SCALES = [0.003, 0.01, 0.03, 0.1, 0.3]
LABEL = {"gateway512": "gateway K512", "gateway2048": "gateway K2048", "R2C3": "LTA 150 K K1024", "R2C4": "LTA 300 K K1024",
         "R2H80": "LTA 80 K K1024", "R2H350": "LTA 350 K K1024", "R2C1": "ITE 300 K K1024", "R2C2": "ITE 150 K K256"}


def pilot_paths(case):
    """{seed: {"fr": path, "laplacian": {scale: path}}} from the pilot directory of `case`."""
    runs = {}
    if case in A.GATEWAY:
        n = A.GATEWAY[case]
        d = PILOT / "gateway" / f"N{n}"
        for p in sorted(d.glob("s*_fr.npz")):
            seed = int(re.match(r"s(\d+)_fr\.npz", p.name).group(1))
            runs[seed] = {"fr": p, "laplacian": {}}
        for p in sorted(d.glob("s*_laplacian_sc*.npz")):
            m = re.match(r"s(\d+)_laplacian_sc([0-9.eE+-]+)\.npz", p.name)
            seed, sc = int(m.group(1)), float(m.group(2))
            runs.setdefault(seed, {"fr": None, "laplacian": {}})["laplacian"][sc] = p
        return runs
    tag = (A.ITE[case][2] if case in A.ITE else A.LTA[case][2])
    n = (A.ITE[case][1] if case in A.ITE else A.LTA[case][1])
    d = PILOT / tag
    for p in sorted(d.glob(f"fr_N{n}_s*.npz")):
        seed = int(re.match(rf"fr_N{n}_s(\d+)\.npz", p.name).group(1))
        runs[seed] = {"fr": p, "laplacian": {}}
    for p in sorted(d.glob(f"laplacian_sc*_N{n}_s*.npz")):
        m = re.match(rf"laplacian_sc([0-9.eE+-]+)_N{n}_s(\d+)\.npz", p.name)
        sc, seed = float(m.group(1)), int(m.group(2))
        runs.setdefault(seed, {"fr": None, "laplacian": {}})["laplacian"][sc] = p
    return runs


def analyze_cell(case, nboot):
    system, scorer = A._scorer(case)
    runs = pilot_paths(case)
    seeds = sorted(s for s, r in runs.items() if r["fr"] is not None)
    fr = {s: A._read(case, s, "fr", runs[s]["fr"], system, scorer) for s in seeds}
    out = {"case": case, "label": LABEL[case], "n_seeds": len(seeds), "seeds": seeds,
           "note": "fresh pilot seeds; no ABF arm; Laplacian vs FR paired on seed; secondary to analyze.py at the frozen scale",
           "fr": _arm_summary(fr), "laplacian": {}}
    for sc in SCALES:
        have = [s for s in seeds if sc in runs[s]["laplacian"]]
        if not have:
            continue
        lap = {s: A._read(case, s, "laplacian", runs[s]["laplacian"][sc], system, scorer) for s in have}
        for s in have:
            if not np.array_equal(fr[s]["step"], lap[s]["step"]):
                raise ValueError(f"{case} seed {s} scale {sc}: save grid differs from the FR pilot run")
            if fr[s]["nfe"] != lap[s]["nfe"]:
                raise ValueError(f"{case} seed {s} scale {sc}: unequal force evaluations")
        row = _arm_summary(lap)
        row["n"] = len(have)
        row["vs_fr"] = {metric: M.paired_contrast({s: fr[s][metric] for s in have}, {s: lap[s][metric] for s in have}, n_boot=nboot)
                        for metric in ("I_F", "e_F", "I_Fp", "e_Fp")}
        out["laplacian"][f"{sc:g}"] = row
    return out


def _arm_summary(rows):
    vals = list(rows.values())
    d = {metric: float(np.median([r[metric] for r in vals])) for metric in ("I_F", "e_F", "I_Fp", "e_Fp")}
    d["repl_median"] = float(np.median([r["repl"] for r in vals]))
    d["essw_min_median"] = float(np.median([r["essw_min"] for r in vals])) if vals[0]["essw_min"] is not None else None
    d["maxfam_max_median"] = float(np.median([r["maxfam_max"] for r in vals])) if vals[0]["maxfam_max"] is not None else None
    d["n_safety_stops"] = sum(1 for r in vals if r["stop_reason"] and "safety" in r["stop_reason"].lower())
    d["per_seed"] = {int(s): {k: r[k] for k in ("I_F", "e_F", "I_Fp", "e_Fp", "repl", "essw_min", "maxfam_max", "stop_reason")} for s, r in rows.items()}
    return d


def _g(c):
    return f"{100 * c['G_median']:+.1f} % [{100 * c['G_ci95'][0]:+.1f}, {100 * c['G_ci95'][1]:+.1f}] ({c['wins']}/{c['n']})"


def markdown(results, decision):
    L = ["# Laplacian scale sensitivity on the fresh-seed pilot (secondary analysis)", "",
         "Pilot runs (8 fresh seeds per cell, production budget and save grid) scored after the scale was frozen. "
         "Contrast G = (Laplacian - FR) / FR on paired seeds, median with seed-bootstrap 95 % CI, (wins / n) = seeds where "
         "the Laplacian arm has the lower error.  No ABF arm exists in the pilot.  The bold scale is the one frozen for "
         "production by the predeclared activity-matching rule.", ""]
    for case, r in results.items():
        frozen = decision.get(case, {}).get("scale")
        L += [f"## {r['label']} ({r['n_seeds']} fresh seeds)", "",
              "| arm | scale | replacements | min ESS/K | I_F median | e_F(T) median | G I_F: Lap vs FR | G e_F(T): Lap vs FR | G I_F': Lap vs FR | G e_F'(T): Lap vs FR |",
              "|---|---:|---:|---:|---:|---:|---|---|---|---|"]
        f = r["fr"]
        L.append(f"| FR | | {f['repl_median']:.0f} | {f['essw_min_median']:.3f} | {f['I_F']:.4g} | {f['e_F']:.4g} | | | | |")
        for sc, a in r["laplacian"].items():
            b = "**" if frozen is not None and abs(float(sc) - float(frozen)) < 1e-12 else ""
            stops = f" ({a['n_safety_stops']} safety stops)" if a["n_safety_stops"] else ""
            L.append(f"| Laplacian | {b}{sc}{b} | {a['repl_median']:.0f}{stops} | {a['essw_min_median']:.3f} | {a['I_F']:.4g} | {a['e_F']:.4g} | "
                     f"{_g(a['vs_fr']['I_F'])} | {_g(a['vs_fr']['e_F'])} | {_g(a['vs_fr']['I_Fp'])} | {_g(a['vs_fr']['e_Fp'])} |")
        L.append("")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cells", nargs="*", default=[*A.GATEWAY, *A.LTA, *A.ITE])
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--out", default="results/laplacian_baselines/analysis")
    ap.add_argument("--ite300-refbase", default=None)
    ap.add_argument("--ite150-refbase", default=None)
    ap.add_argument("--lta-ref-dir", default=None)
    args = ap.parse_args(argv)
    A.LTA_REF_DIR = args.lta_ref_dir
    if args.ite300_refbase:
        A.ITE_REFBASE["R2C1"] = args.ite300_refbase
    if args.ite150_refbase:
        A.ITE_REFBASE["R2C2"] = args.ite150_refbase
    dec = {}
    try:
        dec = json.load(open(ROOT / "results/laplacian_baselines/pilot/decision.json")).get("cells", {})
    except FileNotFoundError:
        pass
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for case in args.cells:
        try:
            results[case] = analyze_cell(case, args.n_boot)
        except Exception as e:  # keep the other cells
            print(f"{case}: SKIPPED ({type(e).__name__}: {e})")
            continue
        r = results[case]
        print(f"{case}: FR repl {r['fr']['repl_median']:.0f} e_F {r['fr']['e_F']:.4g} I_F {r['fr']['I_F']:.4g}")
        for sc, a in r["laplacian"].items():
            print(f"  lap {sc:>5s}: repl {a['repl_median']:6.0f} e_F {a['e_F']:.4g} I_F {a['I_F']:.4g}  G e_F {_g(a['vs_fr']['e_F'])}  G I_F {_g(a['vs_fr']['I_F'])}")
    (out / "pilot_sensitivity.json").write_text(json.dumps(results, indent=1, default=float) + "\n")
    (out / "pilot_sensitivity.md").write_text(markdown(results, dec))
    print(f"wrote {out / 'pilot_sensitivity.md'}")


if __name__ == "__main__":
    main()
