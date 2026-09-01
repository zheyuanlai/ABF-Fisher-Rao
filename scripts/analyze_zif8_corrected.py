#!/usr/bin/env python
"""Corrected-baseline verdict: ABF vs ABF+uniform mFR at h_bias=0.10, h_read=0.05.

Every arm is re-scored from its RAW accumulators at the frozen h_read, so the
comparison is not contaminated by the online bandwidth.  The verdict must hold
against the full umbrella reference AND both of its split halves; a verdict
that flips between them is reported as unstable, not as a verdict.

    python scripts/analyze_zif8_corrected.py
"""
from __future__ import annotations
import json, os, sys
import numpy as np, torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from alkanes import periodic as per                                    # noqa: E402
from zif8.core_zif8 import mean_force_regularized                      # noqa: E402

O = os.path.join(ROOT, "results/information_campaign/corrected")
PRE = json.load(open(os.path.join(
    ROOT, "configs/information_campaign/corrected_baseline_prereg.json")))
K_PHI = 0.42701
N_BOOT, BOOT_SEED = 10000, 20260829


def boot_ci(x, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(N_BOOT, len(x)))
    m = np.median(np.asarray(x)[idx], axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def pmf_series(z, h_read, min_count=20.0):
    """(T, R, G) PMF re-derived at h_read from the saved raw accumulators."""
    fs = np.asarray(z["raw_fsum_t"], float); cs = np.asarray(z["raw_csum_t"], float)
    G = fs.shape[-1]
    grid, dphi = per.periodic_grid(G, dtype=torch.float64)
    K = per.wrapped_gaussian_kernel_matrix(grid, h_read * K_PHI)
    T, R, _ = fs.shape
    mf = mean_force_regularized(torch.as_tensor(fs.reshape(-1, G)),
                                torch.as_tensor(cs.reshape(-1, G)), K, min_count)
    return per.free_energy_from_mean_force(mf, grid, dphi).numpy().reshape(T, R, G)


def eF(pmf, F_ref):
    d = pmf - F_ref[None, None, :]
    d = d - d.mean(-1, keepdims=True)
    return np.sqrt((d * d).mean(-1))


def main():
    ref = np.load(os.path.join(ROOT, "results/uniform_campaign/zif8/reference/"
                                     "reference_T300.npz"), allow_pickle=True)
    refs = {"full": np.asarray(ref["F"], float),
            "split_A": np.asarray(ref["F_split"], float)[0],
            "split_B": np.asarray(ref["F_split"], float)[1]}
    h_read = PRE["corrected_baseline"]["h_read_A"]
    arms = {}
    for m in ("abf", "fr_uniform"):
        p = os.path.join(O, f"{m}.npz")
        if not os.path.exists(p):
            print(f"missing {p}"); return 1
        arms[m] = np.load(p, allow_pickle=True)
    meta = json.loads(str(arms["fr_uniform"]["meta"]))
    t = np.asarray(arms["abf"]["times"], float)
    st = np.asarray(arms["abf"]["steps"])
    P = {m: pmf_series(arms[m], h_read) for m in arms}

    print(f"CORRECTED BASELINE  h_bias={meta['h_bias_A']} A  h_read={h_read} A  "
          f"fr_rate={meta['fr_rate']}  seeds {meta['seeds'][0]}-{meta['seeds'][-1]}")
    pre = st < meta["fr_start_steps"]
    same = np.array_equal(P["abf"][pre], P["fr_uniform"][pre])
    print(f"  pairing: arms bit-identical before the first FR event: {same}"
          f"  {'PASS' if same else 'FAIL'}")
    ev = int(np.asarray(arms["fr_uniform"]["total_replacement_events"]).sum())
    N = meta["n_replicas"]; R = P["abf"].shape[1]
    print(f"  FR acted: {ev} events, {ev/(R*N):.2f} per replica")

    print(f"\n{'reference':>10} {'median dI_F':>12} {'CI95':>20} {'worse':>7} "
          f"{'dI_F postFR':>12} {'d e_F(T)':>10}")
    verdicts, rows = {}, {}
    for lab, F in refs.items():
        e = {m: eF(P[m], F) for m in P}
        I = {m: np.trapezoid(e[m], t, axis=0) for m in e}
        d = 100 * (I["fr_uniform"] - I["abf"]) / I["abf"]
        lo, hi = boot_ci(d)
        post = t >= meta["fr_start_steps"] * meta["dt"]
        Ip = {m: np.trapezoid(e[m][post], t[post], axis=0) for m in e}
        dp = 100 * (Ip["fr_uniform"] - Ip["abf"]) / Ip["abf"]
        dfin = 100 * (e["fr_uniform"][-1] - e["abf"][-1]) / e["abf"][-1]
        rows[lab] = dict(median=float(np.median(d)), ci=[lo, hi],
                         worse=int((d > 0).sum()), n=len(d),
                         median_post=float(np.median(dp)),
                         median_final=float(np.median(dfin)))
        if lo > 0:
            v = "R1_HARMFUL"
        elif np.median(d) <= -10.0 and hi < 0:
            v = "R3_ACCELERATOR"
        elif lo > -5.0 and hi < 5.0:
            v = "R2_NEUTRAL"
        else:
            v = "R4_INCONCLUSIVE"
        verdicts[lab] = v
        print(f"{lab:>10} {np.median(d):+12.2f} [{lo:+7.2f},{hi:+7.2f}] "
              f"{rows[lab]['worse']:>4}/{len(d)} {np.median(dp):+12.2f} "
              f"{np.median(dfin):+10.2f}   {v}")

    act = st >= meta["fr_start_steps"]
    ess = np.asarray(arms["fr_uniform"]["ancestor_ess"], float)[act] / N
    wmax = np.asarray(arms["fr_uniform"]["max_ancestor_frac"], float)[act]
    ess_med = float(np.median(np.nanmin(ess, axis=0)))
    wmax_med = float(np.median(np.nanmax(wmax, axis=0)))
    health = ess_med >= 0.30 and wmax_med <= 0.05
    print(f"\n  genealogy: median min ESS/N {ess_med:.3f}, median max lineage "
          f"{wmax_med:.4f}  -> {'OK' if health else 'FLOOR VIOLATED'}")
    print(f"  transits: abf {int(np.asarray(arms['abf']['cross_gate_samples']).size)}"
          f"  fr {int(np.asarray(arms['fr_uniform']['cross_gate_samples']).size)}")

    stable = len(set(verdicts.values())) == 1
    final = verdicts["full"] if stable else "UNSTABLE_ACROSS_REFERENCES"
    print(f"\n  verdict per reference: {verdicts}")
    print(f"  VERDICT: {final}" + ("" if stable else
          "  <- the three references disagree; not reported as a verdict"))
    if final == "R2_NEUTRAL" and not health:
        print("  (note: genealogy floor violated -- neutrality is on accuracy only)")
    json.dump(dict(h_bias=meta["h_bias_A"], h_read=h_read, fr_rate=meta["fr_rate"],
                   rows=rows, verdicts=verdicts, verdict=final, paired=bool(same),
                   events=ev, ess_min_median=ess_med, wmax_median=wmax_med,
                   genealogy_ok=bool(health)),
              open(os.path.join(O, "corrected_summary.json"), "w"), indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
