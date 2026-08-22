"""E2: which lift actually keeps the fiber conditional, and does 'geometrically
natural' mean 'statistically correct'?

Every lift is a choice of fiber velocity w = dy/dz; the ambient displacement
dq = dz (1 - c w, w) satisfies grad xi . dq = dz for ANY w, so the lift is not
determined by the constraint.  Three choices:

    cartesian  w = 0            move the ambient coordinate x only
    minnorm    w = c / G        minimum EUCLIDEAN norm; what a moving hard
                                constraint produces (constraint force along grad xi)
    adiabatic  w = w*(y, z)     solves  d_z nu + d_y (nu w) = 0

E2a  PURE TRANSPORT, frozen fiber (tau_mix = infinity).  The lift is then the only
     thing acting, and the resulting conditional lag is computable EXACTLY by
     pushing nu(.|z0) through the lift map and comparing with nu(.|z1) --
     no Monte Carlo noise at all.  Cross-checked against a 2e5-sample estimate.

E2b  TRANSPORT + RELAXATION at finite speed.  Gives D_cond(v, tau_mix) in
     quasi-steady state, which is the timescale condition tau_mix << tau_WFR
     turned into a measured curve.

usage:  python scripts/exp_lift.py a|b [--system EB] [--a 0.6] [--k 1.4]
"""
import sys, json, argparse, math
sys.path.insert(0, "src")
import numpy as np
import torch

from rcwfr.grid import DEVICE, DTYPE
from rcwfr.systems.graph import build_graph, build_mfib

MODES = ("cartesian", "minnorm", "adiabatic")


# ---------------------------------------------------------------------------
def transport_map(s, z0, z1, mode, n_y=20001, n_sub=400):
    """The lift's map on the fiber; return (y0, y1 = T(y0), J = dT/dy0).

    'adiabatic' uses the EXACT flow of w*, i.e. the monotone CDF-matching map,
    because w* itself diverges in any low-density valley of the conditional and
    no ODE integrator can be trusted there.
    """
    y0 = torch.linspace(-s.p.y_max, s.p.y_max, n_y, device=DEVICE, dtype=DTYPE)
    if mode == "adiabatic":
        y1 = s.lift_cdf(torch.full_like(y0, z0), y0, torch.full_like(y0, z1))
        return y0, y1, torch.gradient(y1, spacing=(y0,))[0]
    y = y0.clone()
    h = (z1 - z0) / n_sub
    z = torch.full_like(y, z0)
    for _ in range(n_sub):                      # RK2 midpoint
        k1 = s.fiber_velocity(z, y, mode)
        k2 = s.fiber_velocity(z + 0.5 * h, y + 0.5 * h * k1, mode)
        y = y + h * k2
        z = z + h
    J = torch.gradient(y, spacing=(y0,))[0]
    return y0, y, J


def kl_pushforward(s, z0, z1, mode):
    """KL( T_# nu(.|z0) || nu(.|z1) ) by quadrature -- exact, no sampling."""
    y0, y1, J = transport_map(s, z0, z1, mode)
    lp0 = _log_nu(s, z0, y0)
    lp1 = _log_nu(s, z1, y1)
    mu = torch.exp(lp0)
    integrand = mu * (lp0 - torch.log(torch.clamp(J, min=1e-12)) - lp1)
    ok = mu > 1e-12
    return float(torch.trapezoid(torch.where(ok, integrand, torch.zeros_like(integrand)),
                                 x=y0))


def _log_nu(s, z_scalar, y):
    from rcwfr.systems.graph import log_nu_exact
    return log_nu_exact(s, z_scalar, y)


def kl_hist(u, nbins=64):
    """Plug-in KL of a PIT sample from Uniform[0,1], Miller-Madow debiased."""
    h = torch.histc(u, bins=nbins, min=0.0, max=1.0)
    n = float(h.sum())
    p = h / n
    kl = float((p * torch.log(torch.clamp(p * nbins, min=1e-30))).sum())
    return kl - (nbins - 1) / (2.0 * n)          # Miller-Madow


# ---------------------------------------------------------------------------
def run_a(s, args):
    """Exact pushforward lag + the leading-order law  D_cond = C dz^2 / 2."""
    from rcwfr.systems.graph import lag_coefficient
    gen = torch.Generator(device=DEVICE); gen.manual_seed(11)
    rows = []
    for z0 in (-1.2, -0.9, -0.6, -0.3, 0.0, 0.3, 0.6, 0.9):
        C = {m: lag_coefficient(s, z0, m) for m in MODES}
        for dz in (0.0125, 0.025, 0.05, 0.1, 0.2, 0.4):
            if z0 + dz > s.grid.eval_hi:
                continue
            rec = dict(z0=z0, dz=dz, C=C)
            for mode in MODES:
                kl = kl_pushforward(s, z0, z0 + dz, mode)
                rec[f"kl_{mode}"] = kl
                rec[f"pred_{mode}"] = C[mode] * dz * dz / 2.0
                # Monte-Carlo cross-check; a 64-bin PIT histogram saturates at
                # log 64 = 4.16, so it is only quoted where it is well below that
                M = 400_000
                zz = torch.full((M,), z0, device=DEVICE, dtype=DTYPE)
                y = s.sample_fiber(zz, gen)
                yn = s.lift_fiber(zz, y, dz, mode, n_sub=64)
                rec[f"klmc_{mode}"] = kl_hist(s.pit(torch.full_like(yn, z0 + dz), yn))
            rows.append(rec)
            print(f"  z0={z0:+.2f} dz={dz:.4f}  " + "  ".join(
                f"{m[:4]}={rec['kl_'+m]:9.5f}|mc {rec['klmc_'+m]:8.5f}|pred {rec['pred_'+m]:8.5f}"
                for m in MODES), flush=True)
    return rows


def run_b(s, args):
    gen = torch.Generator(device=DEVICE); gen.manual_seed(13)
    M, dt = 300_000, 5e-4
    z_lo, z_hi = -1.2, 1.2
    rows = []
    for v in (0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2):
        n_steps = int((z_hi - z_lo) / (v * dt))
        for mode in MODES:
            zz = torch.full((M,), z_lo, device=DEVICE, dtype=DTYPE)
            y = s.sample_fiber(zz, gen)
            kls, n_meas = [], 0
            meas_every = max(1, n_steps // 200)
            for t in range(n_steps):
                y = s.step_fiber_z(zz, y, dt, gen)
                y = y + (v * dt) * s.fiber_velocity(zz, y, mode)
                zz = zz + v * dt
                y = torch.clamp(y, -s.p.y_max, s.p.y_max)
                if t % meas_every == 0 and t > n_steps // 2:
                    kls.append(kl_hist(s.pit(zz, y)))
            kl = float(np.mean(kls))
            tau = float(1.0 / s.omega(torch.tensor(0.9, device=DEVICE, dtype=DTYPE)) ** 2)
            rows.append(dict(v=v, mode=mode, kl=kl, n_steps=n_steps,
                             eps=v * tau, tau_mix_out=tau))
            print(f"  v={v:5.2f} {mode:10s} n_steps={n_steps:7d} D_cond={kl:.5f}", flush=True)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["a", "b"])
    ap.add_argument("--system", default="EB")
    ap.add_argument("--a", type=float, default=0.6)
    ap.add_argument("--k", type=float, default=1.4)
    ap.add_argument("--omega_out", type=float, default=None)
    ap.add_argument("--omega", type=float, default=1.0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if args.system == "MFIB":
        s = build_mfib(omega=args.omega, a=args.a, k=args.k)
        kw = dict(omega=args.omega)
    else:
        kw = {} if args.omega_out is None else dict(omega_out=args.omega_out)
        s = build_graph(args.system, a=args.a, k=args.k, **kw)
    print(f"[E2{args.which}] {args.system} a={args.a} k={args.k} {kw}", flush=True)
    rows = run_a(s, args) if args.which == "a" else run_b(s, args)
    tag = args.tag or (f"{args.system}_a{args.a}_k{args.k}"
                       + (f"_om{args.omega}" if args.system == "MFIB" else "")
                       + (f"_w{args.omega_out}" if args.omega_out else ""))
    path = f"results/manifold/lift_{args.which}_{tag}.json"
    with open(path, "w") as fh:
        json.dump(dict(system=args.system, a=args.a, k=args.k,
                       omega_out=args.omega_out, rows=rows), fh, indent=2)
    print("wrote", path)
