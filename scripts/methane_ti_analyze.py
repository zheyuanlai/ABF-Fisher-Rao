"""Turn the constrained-TI cells into ``F_ref``/``W_ref``, and evaluate Gate 0.

Consumes ``build{b}_r{r}.npz`` written by ``methane_ti_reference.py`` and produces

    fbar(r)   pooled conditional mean force, per build and consensus
    F_ref(r)  = C + integral fbar ds                       (cumulative-trapezoid)
    W_ref(r)  = F_ref(r) + 2 beta^-1 log r + C'            (SPEC §2.3)

and the two acceptance statistics the campaign requires:

* **Reference acceptance (Amendment 12.2 / §4.5)** --
  ``ratio = max pairwise L2 between builds / (0.10 * consensus F span)``, accepted at
  ``<= 0.5``.  Runs on whatever builds exist, and says how many.

* **Gate 0 (Amendment 10, evaluated first)** -- the cross-family ``|wet - dry|`` spread of the
  conditional mean force, relative to ``|F'_ref|``.  This is the controlled experiment
  Amendment 9 identifies as *the* instrument for a conditional-equilibration question:
  independently prepared solvent families at the same ``r`` either agree, or the difference is
  the signal.  Calibration ladder, from the campaign:

      WCA     0.040 global                          -> passes
      gateway 0.036 global / 0.189 in the constriction -> passes, marginal
      deca    0.61                                  -> fails
      R15 b=2 0.564 / 0.593                         -> fails

  **No numerical threshold is set**, deliberately: Amendment 9 refused to set one after seeing
  R15's number, and that refusal is binding here.  The verdict is argued against the ladder with
  the numbers in hand, and the region that carries the free-energy error is reported separately
  from the global figure -- the gateway passed globally at 0.036 while sitting at 0.189 exactly
  where its mechanism operated.

Usage:
    python scripts/methane_ti_analyze.py --ti results/methane/ti --out results/methane/ti
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from methane import system as msys                               # noqa: E402


def load_cells(ti_dir):
    cells = {}
    for path in sorted(glob.glob(os.path.join(ti_dir, "build*_r*.npz"))):
        d = np.load(path)
        cells.setdefault(int(d["build"]), {})[float(d["r_nm"])] = dict(
            family=d["family"], fbar=d["fbar"], fsem=d["fsem"], ngap=d["ngap"])
    return cells


def integrate(r, f):
    """Cumulative trapezoid of ``f`` over ``r``, mean-centred (the additive constant is free)."""
    F = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(r))])
    return F - F.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ti", default="results/methane/ti")
    ap.add_argument("--out", default="results/methane/ti")
    args = ap.parse_args()

    beta = msys.beta_per_kJ()
    kT = msys.kT_kJ()
    cells = load_cells(args.ti)
    if not cells:
        raise SystemExit(f"no TI cells found in {args.ti}")
    builds = sorted(cells)
    r_common = sorted(set.intersection(*[set(cells[b]) for b in builds]))
    if len(r_common) < 3:
        raise SystemExit(f"only {len(r_common)} r-points complete in every build; too few")
    r = np.asarray(r_common)
    print(f"[load] builds {builds}, {len(r)} r-points complete in all of them "
          f"({r.min():.3f} .. {r.max():.3f} nm)")

    per_build_f, per_build_F = [], []
    dry_m, wet_m, spread, ngap_m, sem_m = [], [], [], [], []
    for b in builds:
        fb = np.asarray([cells[b][x]["fbar"].mean() for x in r])
        per_build_f.append(fb)
        per_build_F.append(integrate(r, fb))
    per_build_f = np.asarray(per_build_f)
    per_build_F = np.asarray(per_build_F)

    for x in r:
        fam = np.concatenate([cells[b][x]["family"] for b in builds])
        fb = np.concatenate([cells[b][x]["fbar"] for b in builds])
        sem = np.concatenate([cells[b][x]["fsem"] for b in builds])
        ng = np.concatenate([cells[b][x]["ngap"] for b in builds])
        dry_m.append(fb[fam == 0].mean())
        wet_m.append(fb[fam == 1].mean())
        spread.append(abs(fb[fam == 1].mean() - fb[fam == 0].mean()))
        ngap_m.append(ng.mean())
        sem_m.append(np.sqrt((sem ** 2).sum()) / len(sem))
    dry_m, wet_m = np.asarray(dry_m), np.asarray(wet_m)
    spread, ngap_m, sem_m = np.asarray(spread), np.asarray(ngap_m), np.asarray(sem_m)

    f_cons = per_build_f.mean(axis=0)
    F_cons = integrate(r, f_cons)
    W_cons = F_cons + (2.0 / beta) * np.log(r)
    W_cons = W_cons - W_cons[-1]                     # anchor W -> 0 at the largest r

    # ---- reference acceptance (Amendment 12.2 / §4.5) ------------------------------------
    span = float(F_cons.max() - F_cons.min())
    if len(builds) >= 2:
        pair = [float(np.sqrt(np.mean((per_build_F[i] - per_build_F[j]) ** 2)))
                for i in range(len(builds)) for j in range(i + 1, len(builds))]
        ratio = max(pair) / max(0.10 * span, 1e-12)
    else:
        pair, ratio = [], float("nan")

    # ---- Gate 0: cross-family spread relative to |F'| -------------------------------------
    denom = np.mean(np.abs(f_cons))
    rel_global = float(np.mean(spread) / max(denom, 1e-12))
    # the region that carries the structure: around the desolvation barrier / first minima
    core = (r >= 0.42) & (r <= 0.70)
    rel_core = float(np.mean(spread[core]) / max(np.mean(np.abs(f_cons[core])), 1e-12))
    worst_i = int(np.argmax(spread / max(denom, 1e-12)))

    print(f"\n{'r (nm)':>8} {'fbar':>10} {'dry':>10} {'wet':>10} {'|w-d|':>8} "
          f"{'sem':>7} {'n_gap':>6} {'F (kT)':>8} {'W (kT)':>8}")
    for i, x in enumerate(r):
        print(f"{x:8.3f} {f_cons[i]:10.2f} {dry_m[i]:10.2f} {wet_m[i]:10.2f} {spread[i]:8.2f} "
              f"{sem_m[i]:7.2f} {ngap_m[i]:6.2f} {F_cons[i]/kT:8.2f} {W_cons[i]/kT:8.2f}")

    print(f"\n[reference] builds = {len(builds)}   F span = {span/kT:.2f} kT")
    if pair:
        print(f"[reference] max pairwise L2 between builds = {max(pair):.3f} kJ/mol")
        print(f"[reference] ratio = {ratio:.4f}   (accept <= 0.5)   "
              f"{'ACCEPTED' if ratio <= 0.5 else 'NOT ACCEPTED'}")
    else:
        print("[reference] only one build complete -- acceptance not evaluable yet")

    print(f"\n[gate 0] mean |wet-dry| / mean|F'| : global {rel_global:.3f}   "
          f"core 0.42-0.70 nm {rel_core:.3f}")
    print(f"[gate 0] worst r = {r[worst_i]:.3f} nm, spread {spread[worst_i]:.2f} kJ/mol/nm "
          f"({spread[worst_i]/max(denom,1e-12):.3f} of mean|F'|)")
    print("[gate 0] calibration: WCA 0.040 pass | gateway 0.036 global, 0.189 constriction "
          "pass(marginal) | deca 0.61 fail | R15 b=2 0.564/0.593 fail")
    print("[gate 0] NO numerical threshold is set (Amendment 9); the verdict is argued "
          "against the ladder in RESULT.md.")

    out = dict(r_nm=r.tolist(), fbar=f_cons.tolist(), fbar_per_build=per_build_f.tolist(),
               F_kJ=F_cons.tolist(), W_kJ=W_cons.tolist(),
               F_kT=(F_cons / kT).tolist(), W_kT=(W_cons / kT).tolist(),
               dry_mean=dry_m.tolist(), wet_mean=wet_m.tolist(),
               wet_dry_spread=spread.tolist(), sem=sem_m.tolist(), ngap=ngap_m.tolist(),
               builds=builds, n_builds=len(builds), F_span_kJ=span, F_span_kT=span / kT,
               pairwise_L2=pair, acceptance_ratio=ratio,
               accepted=bool(len(builds) >= 2 and ratio <= 0.5),
               gate0_rel_global=rel_global, gate0_rel_core=rel_core,
               gate0_worst_r_nm=float(r[worst_i]),
               gate0_worst_rel=float(spread[worst_i] / max(denom, 1e-12)),
               calibration=dict(wca=0.040, gateway_global=0.036, gateway_constriction=0.189,
                                deca=0.61, r15_b2=[0.564, 0.593]),
               beta_per_kJ=beta, kT_kJ=kT)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "reference.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    np.savez_compressed(os.path.join(args.out, "reference.npz"),
                        r_nm=r, fbar=f_cons, F_kJ=F_cons, W_kJ=W_cons,
                        fbar_per_build=per_build_f, wet_dry_spread=spread,
                        dry_mean=dry_m, wet_mean=wet_m, ngap=ngap_m)
    print(f"\n[done] -> {args.out}/reference.json")


if __name__ == "__main__":
    main()
