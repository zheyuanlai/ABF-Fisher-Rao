#!/usr/bin/env python
"""Retrospective test: does a mean-force INFORMATION-adequacy clock explain the
campaign's closed positive/negative split, where the marginal clock did not?

EXPLORATORY by construction -- every cell here is closed and its Delta I_F is
already known, so this can only fail to falsify the new predictor.  The
definition and the predicted ordering were frozen first, in
configs/information_campaign/prediction_v1.md.

    python scripts/audit_information_clock.py
"""
from __future__ import annotations

import json, os, sys
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from infoadq.information_adequacy import (block_accumulators,            # noqa: E402
                                          relative_clock,
                                          uncertainty_across_seeds,
                                          uncertainty_block_bootstrap)

OUT = os.path.join(ROOT, "results/information_campaign")

# cell -> (abf npz, periodic?, min_count, warmup steps, fr_start steps, known dI_F %)
CELLS = {}


def add(tag, path, periodic, min_count, warm, frstart, dI, note):
    p = os.path.join(ROOT, path)
    if os.path.exists(p):
        CELLS[tag] = dict(path=p, periodic=periodic, min_count=min_count,
                          warm=warm, fr_start=frstart, dI=dI, note=note)


# ZIF-8: the harmful cell (periodic CV, prereg sampler block)
add("zif8_T300", "results/uniform_campaign/zif8/production_T300/abf.npz",
    True, 20.0, 60000, 60000, +3.67, "HARMFUL")
# LTA T-sweep: the campaign's strongest positives (v2, fr_start = warmup)
for T, dI in ((80, -35.14), (150, -31.92), (225, -21.28), (300, -14.84)):
    add(f"lta_T{T}", f"results/uniform_campaign/lta/production_T{T}/abf.npz",
        True, 0.0, 20000, 20000, dI, "strong positive")
# LTA v1: same system, FR started LATE -- the campaign's own "FR lateness" null
add("lta_v1_late", "results/uniform_campaign/lta/production/abf.npz",
    True, 0.0, 20000, 40000, -0.21, "null (FR started late)")
# CHA olefins: small sub-threshold positives
for g, T, dI in (("ethene", 450, -5.96), ("propene", 450, -5.72),
                 ("propene", 600, -5.96)):
    add(f"cha_{g}_{T}", f"results/uniform_campaign/cha/production_{g}_{T:g}/abf.npz",
        False, 20.0, 25000, 25000, dI, "small positive")


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    print(f"{'cell':16s} {'dI_F%':>7} {'T':>7} {'t_FR':>6} {'T_info':>8} {'H':>7} "
          f"{'T_marg':>8} {'boot/seed':>10} {'status':>15}")
    print("-" * 96)
    for tag, c in CELLS.items():
        z = np.load(c["path"], allow_pickle=True)
        mf = np.asarray(z["mean_force"], float)      # (T,R,G)
        ec = np.asarray(z["eff_counts"], float)
        st = np.asarray(z["steps"]); tt = np.asarray(z["times"], float)
        G = mf.shape[-1]
        dz = (2 * np.pi / G) if c["periodic"] else float(z["dz"]) if "dz" in z else 1.0
        i_burn = int(np.argmax(st >= c["warm"]))
        dW, dY = block_accumulators(ec, mf, i_burn, c["min_count"])
        dt_save = float(tt[1] - tt[0])

        # --- instrument check: single-run block bootstrap vs across-seed spread
        u_seed = uncertainty_across_seeds(np.asarray(z["pmf"], float))[i_burn:]
        ks, ub = [], []
        for r in range(mf.shape[1]):
            k, u = uncertainty_block_bootstrap(dW[:, r, :], dY[:, r, :], dz,
                                               c["min_count"], c["periodic"],
                                               block=5, n_boot=150, seed=r)
            if k.size:
                ks, ub_r = k, u
                ub.append(u)
        if not ub:
            print(f"{tag:16s}  too few blocks"); continue
        ub = np.median(np.stack(ub), axis=0)
        t_boot = tt[i_burn] + np.asarray(ks) * dt_save
        us_at = np.interp(t_boot, tt[i_burn:], u_seed)
        ratio = float(np.median(ub / np.maximum(us_at, 1e-12)))

        # --- the frozen clock, on the deployable single-run estimate
        t_fr = c["fr_start"] * (tt[1] - tt[0]) / (st[1] - st[0])
        T_info, U0, Uinf, thr, stt = relative_clock(t_boot, ub, t0=t_fr)
        T = float(tt[-1]); H = (T_info - t_fr) / T if np.isfinite(T_info) else np.inf
        # T_marg for comparison, from the same run
        pk = "kl_uniform" if "kl_uniform" in z else None
        T_marg = np.nan
        if pk:
            kl = np.asarray(z[pk], float).mean(1)
            T_marg = relative_clock(tt, kl, t0=t_fr)[0]
        rows.append(dict(cell=tag, dI=c["dI"], note=c["note"], T=T, t_fr=t_fr,
                         T_info=T_info, H=H, T_marg=T_marg,
                         boot_over_seed=ratio, status=stt,
                         U0=U0, Uinf=Uinf))
        print(f"{tag:16s} {c['dI']:+7.2f} {T:7.0f} {t_fr:6.1f} {T_info:8.1f} "
              f"{H:+7.3f} {T_marg:8.1f} {ratio:10.2f} {stt:>15s}")

    with open(os.path.join(OUT, "information_clock_audit.json"), "w") as fh:
        json.dump(rows, fh, indent=2, default=float)
    print(f"\nwrote {OUT}/information_clock_audit.json")

    # --- the decisive relation
    ok = [r for r in rows if np.isfinite(r["H"])]
    if len(ok) >= 3:
        H = np.array([r["H"] for r in ok]); d = np.array([r["dI"] for r in ok])
        from scipy.stats import spearmanr
        rho, p = spearmanr(H, d)
        print(f"\nSpearman( headroom H , Delta I_F ) = {rho:+.3f}  (p={p:.3f}, n={len(ok)})")
        print("  hypothesis predicts NEGATIVE: more headroom -> more benefit (dI_F more negative)")
    else:
        print(f"\nonly {len(ok)} cells with a finite clock -- relation not assessable")


if __name__ == "__main__":
    main()
