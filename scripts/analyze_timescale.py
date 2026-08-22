"""E3 analysis: where the linear-response lift-lag law holds, and where it breaks.

    D_cond(steady) = C_eff(z, lift) v^2 / 2      (linear response)

Linear response requires D_cond << 1, so the law is tested in the well region and
its breakdown at the barrier top -- where the conditional is multimodal and C_eff
diverges because moving mass across a near-empty valley needs unbounded velocity
-- is reported rather than averaged away.
"""
import json, sys
import numpy as np

d = json.load(open("results/manifold/timescale.json"))
rows = [r for r in d["rows"] if r["v"] > 0]
MODES = ("cartesian", "minnorm", "adiabatic")

print("floor (v = 0 control):")
for r in json.load(open("results/manifold/timescale.json"))["rows"]:
    if r["v"] == 0:
        print(f"   om={r['omega']:.2f}  " + "  ".join(
            f"{m[:4]}={r['D_'+m]:+.5f}" for m in MODES))
        break

WELL = lambda r: -1.15 < r["z"] < -0.55
print("\n--- LINEAR-RESPONSE REGIME (left well, predicted D < 0.1) ---")
print(f"{'om':>5}{'v':>7}{'lift':>11}{'measured':>11}{'predicted':>11}{'ratio':>8}")
pairs = {m: [] for m in MODES}
for om in sorted(set(r["omega"] for r in rows)):
    for v in sorted(set(r["v"] for r in rows)):
        sub = [r for r in rows if r["omega"] == om and r["v"] == v and WELL(r)]
        if not sub:
            continue
        for m in ("cartesian", "minnorm"):
            meas = float(np.mean([r["D_" + m] for r in sub]))
            pred = float(np.mean([r["pred_" + m] for r in sub]))
            if pred > 0.1 or pred < 3e-4:
                continue
            pairs[m].append((pred, meas))
            print(f"{om:5.2f}{v:7.3f}{m:>11}{meas:11.5f}{pred:11.5f}{meas/pred:8.2f}")

print("\nratio measured/predicted, pooled:")
for m in ("cartesian", "minnorm"):
    if pairs[m]:
        rr = np.array([b / a for a, b in pairs[m]])
        print(f"   {m:10s} n={len(rr):2d}  median {np.median(rr):.2f}  "
              f"[{rr.min():.2f}, {rr.max():.2f}]")

print("\n--- v-scaling exponent (should be 2) ---")
for om in sorted(set(r["omega"] for r in rows)):
    for m in ("cartesian", "minnorm"):
        vs, ds = [], []
        for v in sorted(set(r["v"] for r in rows)):
            sub = [r for r in rows if r["omega"] == om and r["v"] == v and WELL(r)]
            if not sub:
                continue
            dd = float(np.mean([r["D_" + m] for r in sub]))
            if 1e-3 < dd < 0.1:
                vs.append(v); ds.append(dd)
        if len(vs) >= 3:
            p = np.polyfit(np.log(vs), np.log(ds), 1)[0]
            print(f"   om={om:4.2f} {m:10s} n={len(vs)}  exponent {p:.2f}")

print("\n--- where the law FAILS: barrier top ---")
BAR = lambda r: abs(r["z"]) < 0.15
for om in sorted(set(r["omega"] for r in rows))[:2]:
    for v in sorted(set(r["v"] for r in rows)):
        sub = [r for r in rows if r["omega"] == om and r["v"] == v and BAR(r)]
        if not sub:
            continue
        meas = float(np.mean([r["D_cartesian"] for r in sub]))
        pred = float(np.mean([r["pred_cartesian"] for r in sub]))
        print(f"   om={om:4.2f} v={v:5.3f}  measured {meas:8.4f}  predicted {pred:10.2f}"
              f"  (over-prediction {pred/max(meas,1e-9):8.1f}x)")

print("\n--- adiabatic lift under transport ---")
ad = [r["D_adiabatic"] for r in rows]
print(f"   D_cond over ALL (z, v, omega): median {np.median(ad):.6f}, "
      f"max {max(ad):.6f}   (cartesian max {max(r['D_cartesian'] for r in rows):.4f})")
