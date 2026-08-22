"""E3: the condition "tau_mix << tau_WFR", turned into a parameter-free prediction.

An ensemble sits on ONE fiber label z(t) = z_0 + v t (every particle shares z, so
a single PIT histogram measures the conditional lag D_cond exactly at that z).
The fiber relaxes while the label moves.  Linear response gives, with no fitted
constant,

    D_cond(z)  =  C_eff(z, lift) * v^2 / 2,
    C_eff      =  beta^2 Var_nu( integral (w - w*) ),

and the ratio tau_eff = sqrt(C_eff / C) is the timescale that actually controls
the trade-off -- NOT the fiber's spectral gap 1/omega^2, because only the part of
the lift error that lands on slow fiber modes survives.

v = 0 is run as a control: it measures the discretization + histogram floor that
every other number has to be read against.

usage: python scripts/exp_timescale.py [--omegas ...] [--vs ...]
"""
import sys, json, argparse
sys.path.insert(0, "src")
import numpy as np
import torch

from rcwfr.grid import DEVICE, DTYPE
from rcwfr.systems.graph import build_mfib, lag_coefficients

MODES = ("cartesian", "minnorm", "adiabatic")


def kl_hist(u, nbins=64):
    h = torch.histc(u, bins=nbins, min=0.0, max=1.0)
    n = float(h.sum())
    p = h / n
    kl = float((p * torch.log(torch.clamp(p * nbins, min=1e-30))).sum())
    return kl - (nbins - 1) / (2.0 * n)


def sweep(s, v, M, dt, z0, z1, n_report=26):
    """All three lifts advanced together as rows of one tensor."""
    gen = torch.Generator(device=DEVICE); gen.manual_seed(int(1000 * v) + 5)
    R = len(MODES)
    z = torch.full((R, M), z0 if v > 0 else -0.9, device=DEVICE, dtype=DTYPE)
    y = s.sample_fiber(z, gen)
    n_steps = max(1, int(round((z1 - z0) / (v * dt)))) if v > 0 else 20_000
    every = max(1, n_steps // n_report)
    rec = []
    # z is deterministic and shared by every particle, so it is tracked as a PYTHON
    # float alongside the tensor: reading it back off the device each step would
    # synchronize and cost more than the dynamics.
    zc = z0 if v > 0 else -0.9
    for t in range(n_steps):
        y = s.step_fiber_z(z, y, dt, gen)
        if v > 0:
            dz = v * dt
            y_new = torch.empty_like(y)
            for r, m in enumerate(MODES):
                if m == "adiabatic":
                    y_new[r] = s.lift_cdf_scalar(zc, y[r], zc + dz)
                else:
                    y_new[r] = y[r] + dz * s.fiber_velocity(z[r], y[r], m)
            y = torch.clamp(y_new, -s.p.y_max, s.p.y_max)
            z = z + dz
            zc = zc + dz
        if t % every == 0 and t > 0:
            rec.append(dict(z=zc, step=t,
                            **{f"D_{m}": kl_hist(s.pit_scalar(zc, y[r]))
                               for r, m in enumerate(MODES)}))
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=float, default=0.6)
    ap.add_argument("--k", type=float, default=0.7)
    ap.add_argument("--omegas", type=float, nargs="+", default=[1.0, 1.4, 2.0, 2.8])
    ap.add_argument("--vs", type=float, nargs="+", default=[0.0, 0.05, 0.1, 0.2, 0.4])
    ap.add_argument("--M", type=int, default=100_000)
    ap.add_argument("--dt", type=float, default=1e-3)
    ap.add_argument("--out", default="results/manifold/timescale.json")
    args = ap.parse_args()

    all_rows = []
    for om in args.omegas:
        s = build_mfib(omega=om, a=args.a, k=args.k)
        coef = {}
        for zc in np.linspace(-1.2, 1.2, 25):
            coef[round(float(zc), 4)] = {m: lag_coefficients(s, float(zc), m)
                                         for m in MODES}
        for v in args.vs:
            rec = sweep(s, v, args.M, args.dt, -1.3, 1.3)
            for r in rec:
                zc = min(coef, key=lambda c: abs(c - r["z"]))
                for m in MODES:
                    r[f"pred_{m}"] = coef[zc][m]["C_eff"] * v * v / 2.0
                    r[f"Ceff_{m}"] = coef[zc][m]["C_eff"]
                    r[f"tau_{m}"] = coef[zc][m]["tau_eff"]
                r.update(omega=om, v=v, tau_gap=1.0 / om ** 2)
            all_rows += rec
            mid = [r for r in rec if abs(r["z"]) < 0.7] or rec
            print(f"om={om:4.2f} v={v:5.3f}  " + "  ".join(
                f"{m[:4]}: D={np.mean([r['D_'+m] for r in mid]):.5f} "
                f"pred={np.mean([r['pred_'+m] for r in mid]):.5f}" for m in MODES),
                flush=True)
        del s; torch.cuda.empty_cache()

    with open(args.out, "w") as fh:
        json.dump(dict(a=args.a, k=args.k, M=args.M, dt=args.dt, rows=all_rows), fh)
    print("wrote", args.out)
