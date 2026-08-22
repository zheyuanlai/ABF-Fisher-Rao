"""Analysis for the two follow-up sweeps.

E4b (burn-in): does "transport, freeze, equilibrate, then estimate" escape the
    trade-off?  At fixed n_cond = 20 the transport rate per epoch is fixed and
    only the fraction of steps that DEPOSIT changes, so raising n_eq buys less
    conditional lag and pays with proportionally fewer samples.  If the two
    cancel, the curve is flat and Version I is a re-parameterization, not an
    escape.

E8 (estimator reset): the self-built lift is worst early and the accumulator
    keeps every deposit.  Discarding the warm-up should recover part of it -- and
    the control is whether the same reset helps the cartesian arm, which has no
    warm-up to discard.
"""
import glob, json, re, sys
import numpy as np

ARMS = ["ti_cold", "wfr_cart", "wfr_minnorm", "wfr_fit_decay", "wfr_adiab"]


def med(v):
    return float(np.median(np.asarray(v, float)))


def burnin():
    files = sorted(glob.glob("results/manifold/arms/*_nc20_neq*.json"))
    files = [f for f in files if "_reset" not in f]
    if not files:
        print("(no burn-in sweep yet)"); return
    rows = {}
    floor = None
    for f in files:
        neq = int(re.search(r"_neq(\d+)", f).group(1))
        d = json.load(open(f)); floor = d["floor"]
        rows[neq] = {a: med(v["final"]) for a, v in d["arms"].items()}
    neqs = sorted(rows)
    arms = [a for a in ARMS if a in rows[neqs[0]]]
    print(f"E4b BURN-IN SWEEP (n_cond = 20, floor {floor:.5f})")
    print("  deposits keep only the last (20 - n_eq) of every 20 fiber steps")
    print(f"  {'n_eq':>5}{'deposit frac':>14}" + "".join(f"{a:>16}" for a in arms))
    for n in neqs:
        print(f"  {n:5d}{(20-n)/20:14.2f}" + "".join(f"{rows[n][a]:16.5f}" for a in arms))
    print("\n  relative to n_eq = 0:")
    for n in neqs[1:]:
        print(f"  {n:5d}{'':>14}" + "".join(
            f"{100*(rows[n][a]-rows[0][a])/rows[0][a]:+15.1f}%" for a in arms))
    print("\n  best n_eq per arm:", {a: min(neqs, key=lambda n: rows[n][a]) for a in arms})


def reset():
    files = sorted(glob.glob("results/manifold/arms/*_reset*.json"))
    if not files:
        print("\n(no reset sweep yet)"); return
    base_f = "results/manifold/arms/CHANNEL_a0.6_k1.4_nc5_neq0_fit.json"
    base = json.load(open(base_f))["arms"]
    print("\nE8 ESTIMATOR RESET (discard the warm-up deposits)")
    print(f"  {'reset at':>9}" + "".join(f"{a:>16}" for a in
                                         ("wfr_cart", "wfr_fit_decay", "wfr_adiab")))
    print(f"  {'none':>9}" + "".join(
        f"{med(base[a]['final']):16.5f}" if a in base else f"{'--':>16}"
        for a in ("wfr_cart", "wfr_fit_decay", "wfr_adiab")))
    for f in files:
        r = re.search(r"_reset([\d.]+)\.json", f).group(1)
        d = json.load(open(f))["arms"]
        print(f"  {r:>9}" + "".join(
            f"{med(d[a]['final']):16.5f}" if a in d else f"{'--':>16}"
            for a in ("wfr_cart", "wfr_fit_decay", "wfr_adiab")))
    print("\n  gain from the reset, relative to no reset:")
    for f in files:
        r = re.search(r"_reset([\d.]+)\.json", f).group(1)
        d = json.load(open(f))["arms"]
        print(f"  {r:>9}" + "".join(
            f"{100*(med(d[a]['final'])-med(base[a]['final']))/med(base[a]['final']):+15.1f}%"
            if a in d and a in base else f"{'--':>16}"
            for a in ("wfr_cart", "wfr_fit_decay", "wfr_adiab")))


def kappa():
    files = sorted(glob.glob("results/manifold/arms/*_kap*.json"),
                   key=lambda f: float(re.search(r"_kap([\d.]+)\.json", f).group(1)))
    if not files:
        print("\n(no kappa sweep yet)"); return
    print("\nE9 TRANSPORT-RATE SWEEP")
    print("  kappa sets how far z moves per epoch, i.e. how little fiber relaxation")
    print("  happens per unit transport.  A lift with a bias term should trace a U;")
    print("  a lift without one should not.")
    arms = ("wfr_cart", "wfr_minnorm", "wfr_adiab")
    floor = None
    print(f"  {'kappa':>7}" + "".join(f"{a:>16}" for a in arms))
    best = {a: (None, 1e9) for a in arms}
    for f in files:
        k = float(re.search(r"_kap([\d.]+)\.json", f).group(1))
        d = json.load(open(f)); floor = d["floor"]
        vals = {a: med(d["arms"][a]["final"]) for a in arms if a in d["arms"]}
        print(f"  {k:7.2f}" + "".join(
            f"{vals[a]:16.5f}" if a in vals else f"{'--':>16}" for a in arms))
        for a, v in vals.items():
            if v < best[a][1]:
                best[a] = (k, v)
    print(f"\n  estimator floor {floor:.5f}")
    for a in arms:
        k, v = best[a]
        if k is not None:
            print(f"  best {a:14s} kappa = {k:5.2f}  e_F = {v:.5f}  "
                  f"({v/floor:.1f}x floor)")


if __name__ == "__main__":
    burnin()
    reset()
    kappa()
