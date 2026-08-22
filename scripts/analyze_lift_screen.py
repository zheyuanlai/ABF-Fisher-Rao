"""E10/E11 analysis: the learned lift's knobs, and the secondary-CV design rule."""
import glob, json, re, sys
import numpy as np


def med(v):
    return float(np.median(np.asarray(v, float)))


def screen():
    fs = glob.glob("results/manifold/arms/*_scr_d*_b*.json")
    if not fs:
        print("E10 (decay x bandwidth screen): not yet\n"); return
    rows = {}
    floor = None
    for f in fs:
        m = re.search(r"_scr_d([\d.]+)_b([\d.]+)\.json", f)
        d, b = float(m.group(1)), float(m.group(2))
        j = json.load(open(f)); floor = j["floor"]
        rows[(d, b)] = med(j["arms"]["wfr_fit_decay"]["final"])
    ds = sorted({k[0] for k in rows}, reverse=True)
    bs = sorted({k[1] for k in rows})
    print(f"E10 LEARNED-LIFT SCREEN (reset at 0.5, floor {floor:.5f})")
    print("  rows: forgetting factor per epoch (1.0 = no forgetting)")
    print("  cols: z-bandwidth of the conditional estimate")
    print(f"  {'decay':>8}" + "".join(f"{b:>12}" for b in bs))
    best = (None, 1e9)
    for d in ds:
        cells = []
        for b in bs:
            v = rows.get((d, b))
            cells.append(f"{v:12.5f}" if v is not None else f"{'--':>12}")
            if v is not None and v < best[1]:
                best = ((d, b), v)
        print(f"  {d:8.4f}" + "".join(cells))
    if best[0]:
        print(f"\n  best: decay {best[0][0]}, bw_z {best[0][1]}  ->  e_F {best[1]:.5f} "
              f"({best[1]/floor:.1f}x floor)")
        print(f"  reference at the same budget: cartesian 0.66254 (reset 0.5), "
              f"exact lift 0.00885")
        print(f"  => learned lift is {100*(best[1]-0.66254)/0.66254:+.1f}% vs cartesian, "
              f"and {best[1]/0.00885:.0f}x the exact lift")
    print()


def secondary():
    try:
        d = json.load(open("results/manifold/secondary_cv.json"))
    except FileNotFoundError:
        print("E11 (secondary-CV sweep): not yet"); return
    print("E11 SECONDARY-CV LIFT vs SPECTATOR TIMESCALE")
    print("  wfr_promote is EXACT on the promoted mode and NAIVE on the spectators;")
    print("  wfr_both is exact on both.  The design rule says they should coincide")
    print("  while the spectators relax fast and separate as they slow down.")
    arms = ["ti_cold", "wfr_naive", "wfr_learned", "wfr_promote", "wfr_both",
            "wfr_oracle"]
    hdr = f"  {'omega_s':>8}{'tau_spec':>10}" + "".join(f"{a:>13}" for a in arms)
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for om in sorted(d, key=float):
        r = d[om]
        cells = []
        for a in arms:
            cells.append(f"{med(r['arms'][a]['final']):13.5f}" if a in r["arms"]
                         else f"{'--':>13}")
        print(f"  {float(om):8.2f}{r['tau_spec']:10.2f}" + "".join(cells))
    print("\n  promote / both  (1.0 = naive spectator lift costs nothing):")
    for om in sorted(d, key=float):
        r = d[om]["arms"]
        if "wfr_promote" in r and "wfr_both" in r:
            ratio = med(r["wfr_promote"]["final"]) / med(r["wfr_both"]["final"])
            print(f"  omega_s={float(om):5.2f} tau={d[om]['tau_spec']:6.2f}  "
                  f"ratio {ratio:6.3f}   Dz(S) promote {r['wfr_promote']['Dz_S']:.4f} "
                  f"vs both {r['wfr_both']['Dz_S']:.4f}")


if __name__ == "__main__":
    screen()
    secondary()
