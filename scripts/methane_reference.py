"""Build ``F_ref``/``W_ref`` from the batched constrained-TI run, and evaluate Gate 0.

Consumes ``ti_final.npz`` from ``methane_ti_torch.py`` and produces the two statistics the
campaign requires before anything else may be interpreted:

* **Reference acceptance** (Amendment 12.2 / §4.5) -- build-to-build agreement, as
  ``ratio = max pairwise L2 between builds / (0.10 * consensus F span)``, accepted at ``<= 0.5``.

* **Gate 0** (Amendment 10, evaluated first) -- the cross-family ``|wet - dry|`` spread of the
  conditional mean force relative to ``mean|F'|``.  Independently prepared solvent families at the
  same ``r`` either agree, or the difference is the conditional-equilibration signal.  Amendment 9
  identifies this controlled experiment as *the* instrument and forbids substituting a screen
  statistic for it.

  **No numerical threshold is set.**  Amendment 9 refused to fix one after seeing R15's number and
  that refusal binds here; the verdict is argued against the campaign's calibration ladder, with
  the region carrying the free-energy error reported separately from the global figure -- the
  gateway passed globally at 0.036 while sitting at 0.189 exactly where its mechanism operated.

Basins come from ``W_ref`` by the Amendment 3 rule (2 kT merge), never from the literature.

Usage:
    python scripts/methane_reference.py --ti results/methane/ti_torch --out results/methane/ref
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from methane import system as msys                               # noqa: E402

CALIBRATION = dict(wca=0.040, gateway_global=0.036, gateway_constriction=0.189,
                   deca=0.61, r15_beta2=[0.564, 0.593])


def integrate(r, f):
    """Cumulative trapezoid; the additive constant is free so the result is mean-centred."""
    F = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(r))])
    return F - F.mean()


def find_basins(W, r, kT, merge_kT=2.0):
    """Amendment 3: local minima of the reference, merged while the separating barrier < 2 kT."""
    mins = [i for i in range(1, len(W) - 1) if W[i] <= W[i - 1] and W[i] <= W[i + 1]]
    if not mins:
        return [], []
    changed = True
    while changed and len(mins) > 1:
        changed = False
        for a in range(len(mins) - 1):
            i, j = mins[a], mins[a + 1]
            barrier = W[i:j + 1].max() - max(W[i], W[j])
            if barrier < merge_kT * kT:
                drop = a if W[i] > W[j] else a + 1
                mins.pop(drop)
                changed = True
                break
    maxima = []
    for a in range(len(mins) - 1):
        i, j = mins[a], mins[a + 1]
        maxima.append(i + int(np.argmax(W[i:j + 1])))
    return mins, maxima


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ti", default="results/methane/ti_torch")
    ap.add_argument("--out", default="results/methane/ref")
    ap.add_argument("--core-lo", type=float, default=0.42)
    ap.add_argument("--core-hi", type=float, default=0.70)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    d = np.load(os.path.join(args.ti, "ti_final.npz"))
    recs, fbar, fcnt = d["recs"], d["fbar"], d["fcnt"]
    retired_at = d["retired_at"]
    ngsum = d["ngsum"]
    beta, kT = msys.beta_per_kJ(), msys.kT_kJ()

    r = np.unique(recs[:, 0])
    builds = sorted(set(recs[:, 1].astype(int)))
    per_build = np.zeros((len(builds), len(r)))
    dry = np.zeros(len(r)); wet = np.zeros(len(r)); ng = np.zeros(len(r)); samp = np.zeros(len(r))
    for k, x in enumerate(r):
        m = recs[:, 0] == x
        for bi, b in enumerate(builds):
            per_build[bi, k] = fbar[m & (recs[:, 1] == b)].mean()
        dry[k] = fbar[m & (recs[:, 2] == 0)].mean()
        wet[k] = fbar[m & (recs[:, 2] == 1)].mean()
        # ``ngsum`` accumulates one n_gap sample per *production block*, and the retirement rule
        # gives different r-points different numbers of blocks -- so it must be divided by the
        # block count, not reported raw.  Reporting the raw sum makes n_gap look non-monotonic in
        # r purely because retired points ran fewer blocks.
        n_blocks = {50.0: 1, 100.0: 2, 200.0: 3}.get(float(retired_at[k]), 3)
        ng[k] = float(ngsum[m].mean() / n_blocks)
        samp[k] = fcnt[m].mean()

    f_cons = per_build.mean(axis=0)
    F = integrate(r, f_cons)
    W = F + (2.0 / beta) * np.log(r)
    W = W - W[-1]
    F = F - F.min()
    per_F = np.asarray([integrate(r, per_build[i]) for i in range(len(builds))])

    span = float(F.max() - F.min())
    pair = [float(np.sqrt(np.mean((per_F[i] - per_F[j]) ** 2)))
            for i in range(len(builds)) for j in range(i + 1, len(builds))]
    ratio = max(pair) / max(0.10 * span, 1e-12)

    spread = np.abs(wet - dry)
    denom = float(np.mean(np.abs(f_cons)))
    core = (r >= args.core_lo) & (r <= args.core_hi)
    rel_global = float(np.mean(spread) / denom)
    rel_core = float(np.mean(spread[core]) / float(np.mean(np.abs(f_cons[core]))))
    wi = int(np.argmax(spread))

    print(f"{'r(nm)':>7} {'fbar':>9} {'dry':>9} {'wet':>9} {'|w-d|':>7} {'n_gap':>6} "
          f"{'F(kT)':>7} {'W(kT)':>7} {'retire':>7}")
    for k, x in enumerate(r):
        ridx = int(np.argmin(np.abs(np.arange(len(r)) - k)))
        ret = retired_at[ridx] if ridx < len(retired_at) else np.nan
        print(f"{x:7.3f} {f_cons[k]:9.2f} {dry[k]:9.2f} {wet[k]:9.2f} {spread[k]:7.2f} "
              f"{ng[k]:6.2f} {F[k]/kT:7.2f} {W[k]/kT:7.2f} "
              f"{('%.0f' % ret) if np.isfinite(ret) else '200+':>7}")

    mins, maxima = find_basins(W, r, kT)
    print(f"\n[basins] minima at r = {[round(float(r[i]),3) for i in mins]} nm; "
          f"barriers at {[round(float(r[i]),3) for i in maxima]} nm")
    for i in mins:
        print(f"   minimum r = {r[i]:.3f} nm   W = {W[i]/kT:+.2f} kT")
    for i in maxima:
        print(f"   barrier r = {r[i]:.3f} nm   W = {W[i]/kT:+.2f} kT")

    print(f"\n[reference] builds = {len(builds)}   F span = {span/kT:.2f} kT")
    print(f"[reference] max pairwise L2 between builds = {max(pair):.3f} kJ/mol")
    print(f"[reference] ratio = {ratio:.4f}  (accept <= 0.5)  "
          f"{'ACCEPTED' if ratio <= 0.5 else 'NOT ACCEPTED'}")

    print(f"\n[gate 0] mean|wet-dry| / mean|F'|:  global {rel_global:.3f}   "
          f"core {args.core_lo}-{args.core_hi} nm {rel_core:.3f}")
    print(f"[gate 0] worst r = {r[wi]:.3f} nm, spread {spread[wi]:.2f} kJ/mol/nm "
          f"({spread[wi]/denom:.3f} of mean|F'|)")
    print(f"[gate 0] calibration: WCA {CALIBRATION['wca']} PASS | gateway "
          f"{CALIBRATION['gateway_global']} global / {CALIBRATION['gateway_constriction']} "
          f"constriction PASS(marginal) | deca {CALIBRATION['deca']} FAIL | "
          f"R15 b=2 {CALIBRATION['r15_beta2']} FAIL")
    print("[gate 0] NO numerical threshold is set (Amendment 9). The verdict is argued against "
          "this ladder in RESULT.md.")

    out = dict(r_nm=r.tolist(), fbar=f_cons.tolist(), fbar_per_build=per_build.tolist(),
               F_kJ=F.tolist(), W_kJ=W.tolist(), F_kT=(F / kT).tolist(),
               W_kT=(W / kT).tolist(), dry=dry.tolist(), wet=wet.tolist(),
               wet_dry_spread=spread.tolist(), n_gap=ng.tolist(),
               samples_per_replica=samp.tolist(), retired_at_ps=retired_at.tolist(),
               builds=len(builds), F_span_kJ=span, F_span_kT=span / kT,
               pairwise_L2=pair, acceptance_ratio=ratio, accepted=bool(ratio <= 0.5),
               gate0_rel_global=rel_global, gate0_rel_core=rel_core,
               gate0_worst_r_nm=float(r[wi]), gate0_worst_rel=float(spread[wi] / denom),
               basin_minima_nm=[float(r[i]) for i in mins],
               basin_barriers_nm=[float(r[i]) for i in maxima],
               basin_minima_W_kT=[float(W[i] / kT) for i in mins],
               basin_barriers_W_kT=[float(W[i] / kT) for i in maxima],
               calibration=CALIBRATION, beta_per_kJ=beta, kT_kJ=kT,
               git_commit=subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                         text=True).stdout.strip())
    with open(os.path.join(args.out, "reference.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    np.savez_compressed(os.path.join(args.out, "reference.npz"), r_nm=r, fbar=f_cons,
                        F_kJ=F, W_kJ=W, per_build=per_build, wet=wet, dry=dry,
                        spread=spread, n_gap=ng)
    print(f"\n[done] -> {args.out}/reference.json")


if __name__ == "__main__":
    main()
