#!/usr/bin/env python
"""Is ZIF-8's mFR bias caused by OVERWEIGHTING REPLICATED LINEAGES?

The ordinary ABF estimate weights each lineage by its descendant count:
    m_g     = sum_a n_ag m_ag / sum_a n_ag
The lineage-BALANCED diagnostic weights each ancestral discovery once:
    m_g^lin = (1/A_g) sum_{a: n_ag>0} m_ag
If cloning biases the conditional average by replicating particular microstates,
the balanced estimator must be closer to the reference -- and the gap must be
LARGER in the FR arm than in the ABF arm.

Predictions frozen in configs/information_campaign/lineage_mechanism_prereg.md.

    python scripts/analyze_lineage_mechanism.py
"""
from __future__ import annotations
import json, os, sys
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
D = os.path.join(ROOT, "results/information_campaign/lineage")
MIN_PER_LINEAGE = 50          # frozen; sensitivity reported at 20 and 100


def estimates(fs, cs, min_n):
    """(ordinary, balanced, n_lineages) per bin, from (R, N_anc, G) accumulators."""
    tot_f, tot_c = fs.sum(1), cs.sum(1)
    ordinary = np.where(tot_c > 0, tot_f / np.maximum(tot_c, 1e-12), np.nan)
    ok = cs >= min_n
    m_ag = np.where(ok, fs / np.maximum(cs, 1e-12), np.nan)
    with np.errstate(invalid="ignore"):
        balanced = np.nanmean(np.where(ok, m_ag, np.nan), axis=1)
    return ordinary, balanced, ok.sum(1)


def main():
    ref = np.load(os.path.join(ROOT, "results/uniform_campaign/zif8/reference/"
                                     "reference_T300.npz"), allow_pickle=True)
    F = np.asarray(ref["F"], float); xi = np.asarray(ref["xi_grid"], float)
    G = F.size; dphi = 2 * np.pi / G
    Fp = np.gradient(F, dphi, edge_order=2)          # reference mean force
    arms = {}
    for m in ("abf", "fr_uniform"):
        p = os.path.join(D, f"{m}.npz")
        if not os.path.exists(p):
            print(f"missing {p}"); return 1
        arms[m] = np.load(p, allow_pickle=True)

    win = np.abs(xi) < 1.5
    print("=== P2: is the LINEAGE-BALANCED estimator less biased? (barrier bins) ===")
    print(f"{'arm':12s} {'ordinary':>10} {'balanced':>10} {'gap':>9} {'n_lineage':>10}")
    res = {}
    for m, z in arms.items():
        fs = np.asarray(z["lineage_fsum"], float)
        cs = np.asarray(z["lineage_csum"], float)
        o, b, A = estimates(fs, cs, MIN_PER_LINEAGE)
        eo = np.sqrt(np.nanmean((o - Fp[None]) ** 2, axis=0))
        eb = np.sqrt(np.nanmean((b - Fp[None]) ** 2, axis=0))
        res[m] = dict(eo=float(np.nanmean(eo[win])), eb=float(np.nanmean(eb[win])),
                      A=float(np.nanmean(A[:, win])))
        r = res[m]
        print(f"{m:12s} {r['eo']:10.4f} {r['eb']:10.4f} {r['eo']-r['eb']:+9.4f} "
              f"{r['A']:10.1f}")
    gap_fr = res["fr_uniform"]["eo"] - res["fr_uniform"]["eb"]
    gap_ab = res["abf"]["eo"] - res["abf"]["eb"]
    print(f"\n  PREDICTION 2: gap(FR) > gap(ABF).  gap_FR {gap_fr:+.4f} vs "
          f"gap_ABF {gap_ab:+.4f}  -> {'SUPPORTED' if gap_fr > gap_ab else 'NOT supported'}")

    print("\n=== P1: per-bin ancestor ESS/N_g, FR vs ABF ===")
    for lab, sl in (("barrier |xi|<1.5", win), ("cages |xi|>5", np.abs(xi) > 5.0)):
        vals = {}
        for m, z in arms.items():
            e = np.asarray(z["ess_anc_bin"], float)
            vals[m] = float(np.nanmean(e[e.shape[0] // 2:, :, sl]))
        print(f"  {lab:18s} abf {vals['abf']:8.2f}   fr {vals['fr_uniform']:8.2f}"
              f"   -> {'FR LOWER (predicted)' if vals['fr_uniform'] < vals['abf'] else 'FR not lower'}")

    print("\n=== P3: force residual by CLONE AGE (barrier bins) ===")
    edges = np.asarray(arms["fr_uniform"]["clone_age_edges"], float)
    labs = [f"<{edges[0]:g} ps"] + [f"{edges[i]:g}-{edges[i+1]:g} ps"
                                    for i in range(len(edges) - 1)] + [f">{edges[-1]:g} ps"]
    for m, z in arms.items():
        fa = np.asarray(z["cloneage_fsum"], float).sum(0)
        ca = np.asarray(z["cloneage_csum"], float).sum(0)
        print(f"  {m}:")
        for k in range(fa.shape[0]):
            c = ca[k][win].sum()
            if c < 100:
                print(f"    {labs[k]:12s} n={c:9.0f}  (too few)"); continue
            r = (fa[k][win].sum() / c) - np.average(Fp[win], weights=ca[k][win])
            print(f"    {labs[k]:12s} n={c:9.0f}  mean residual {r:+8.3f} kJ/mol/rad")

    print("\n=== sensitivity of P2 to the min-samples-per-lineage cut ===")
    for mn in (20, 50, 100):
        g = {}
        for m, z in arms.items():
            o, b, _ = estimates(np.asarray(z["lineage_fsum"], float),
                                np.asarray(z["lineage_csum"], float), mn)
            eo = np.sqrt(np.nanmean((o - Fp[None]) ** 2, axis=0))
            eb = np.sqrt(np.nanmean((b - Fp[None]) ** 2, axis=0))
            g[m] = float(np.nanmean(eo[win]) - np.nanmean(eb[win]))
        print(f"  min_n={mn:3d}: gap_FR {g['fr_uniform']:+.4f}  gap_ABF {g['abf']:+.4f}"
              f"  -> {'supported' if g['fr_uniform']>g['abf'] else 'NOT supported'}")
    os.makedirs(os.path.join(ROOT, "results/information_campaign"), exist_ok=True)
    with open(os.path.join(ROOT, "results/information_campaign/lineage_mechanism.json"),
              "w") as fh:
        json.dump(dict(barrier=res, gap_fr=gap_fr, gap_abf=gap_ab,
                       min_per_lineage=MIN_PER_LINEAGE), fh, indent=2, default=float)
    return 0


if __name__ == "__main__":
    sys.exit(main())
