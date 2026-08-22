"""E4: does lift CORRECTNESS convert the RC-WFR advantage into an unbiased one?

The frozen campaign's win on CHANNEL came from transporting hidden-channel
content across the reaction coordinate; its bias came from transporting the
WRONG channel content.  §3.2 of docs/MANIFOLD_FORMULATION.md argues these are
the same quantity with opposite sign, so "freeze and equilibrate" cannot
separate them -- only a conditionally correct lift can.  This tests that on a
NONLINEAR reaction coordinate, where the three lifts genuinely differ.

Arms, all at matched force evaluations (N * n_steps fiber steps each):

    ti_cold        fixed uniform windows, cold fiber start   (the mandatory baseline)
    ti_warm        fixed uniform windows, equilibrium start  (upper bound for static)
    wfr_cart       flow-W + FR, cartesian lift               (the naive lift)
    wfr_minnorm    flow-W + FR, minimum-norm lift            (what the proposal recommends)
    wfr_adiab      flow-W + FR, exact conditional lift       (what §3.1 says is correct)
    wfr_oracle     flow-W + FR, resample y ~ nu(.|z)         (unreachable upper bound)
    fr_only        FR only, no transport                     (the birth-death half alone)

Hyper-parameters are the frozen Stage-1 winners from docs/RESULTS_LOG.md
(kappa = 2.0 probability flow, theta = 0.3, n_cond = 5, jitter = 0.01) -- not
retuned here, and identical across every WFR arm so the lift is the only
difference.
"""
import sys, os, json, argparse, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import torch

from rcwfr.estimators import MeanForceAccumulator, gauge_l2
from rcwfr.fisher_rao import theta_backoff
from rcwfr.gmm import GMM1D
from rcwfr.grid import DEVICE, DTYPE, EPS
from rcwfr.resampling import systematic_resample
from rcwfr.adaptive_lift import AdaptiveFiberCDF
from rcwfr.systems.graph import build_graph

LIFTS = dict(wfr_cart="cartesian", wfr_minnorm="minnorm", wfr_adiab="adiabatic",
             wfr_oracle="oracle", fr_only="cartesian",
             wfr_fit="fitted", wfr_fit_decay="fitted")


def _kl(u, nbins):
    h = torch.histc(u, bins=nbins, min=0.0, max=1.0)
    n = float(h.sum())
    if n < 4 * nbins:
        return None
    p = h / n
    return float((p * torch.log(torch.clamp(p * nbins, min=1e-30))).sum()) \
        - (nbins - 1) / (2.0 * n)


def kl_pit(s, z, y, nbins=64):
    """POOLED conditional lag: one PIT histogram over the whole ensemble."""
    return _kl(s.pit(z, y).reshape(-1), nbins) or 0.0


class ZResolvedPIT:
    """Time-accumulated, z-RESOLVED conditional lag.

        D_z = int KL[ rho(.|z) || nu(.|z) ] p(z) dz

    A single-snapshot estimate has too few particles per z-bin to resolve this, so
    the (z, u) histogram is accumulated over the production half of the run.  The
    samples are time-correlated, which inflates the plug-in bias, so the same
    accumulation is applied to every arm and the numbers are only compared with
    each other and with the ORACLE arm's value, which measures the floor.
    """

    def __init__(self, grid, device, dtype, n_z=12, n_u=16):
        self.grid, self.n_z, self.n_u = grid, n_z, n_u
        self.H = torch.zeros((n_z, n_u), device=device, dtype=dtype)
        self.edges = torch.linspace(grid.eval_lo, grid.eval_hi, n_z + 1,
                                    device=device, dtype=dtype)

    def add(self, s, z, y):
        u = s.pit(z, y).reshape(-1)
        zz = z.reshape(-1)
        iz = torch.clamp(torch.bucketize(zz, self.edges) - 1, 0, self.n_z - 1)
        iu = torch.clamp((u * self.n_u).long(), 0, self.n_u - 1)
        inside = (zz >= self.edges[0]) & (zz <= self.edges[-1])
        flat = (iz * self.n_u + iu)[inside]
        self.H.view(-1).scatter_add_(0, flat, torch.ones_like(flat, dtype=self.H.dtype))

    def value(self):
        n = self.H.sum(-1)
        ok = n >= 8 * self.n_u
        if not bool(ok.any()):
            return 0.0
        p = self.H[ok] / n[ok].unsqueeze(-1)
        kl = (p * torch.log(torch.clamp(p * self.n_u, min=1e-30))).sum(-1)
        kl = kl - (self.n_u - 1) / (2.0 * n[ok])
        w = n[ok] / n[ok].sum()
        return float((w * torch.clamp(kl, min=0.0)).sum())


def kl_pit_z(s, z, y, n_bins_z=12, nbins=16):
    """z-RESOLVED conditional lag, which is the quantity the error functional
    actually contains:  int KL[ rho(.|z) || nu(.|z) ] p(z) dz.

    The pooled version can read near zero while every individual fiber is wrong,
    because errors at different z cancel in one histogram.  A lift that is graded
    on the pooled number can therefore look like it is helping while it is not.
    """
    g = s.grid
    u = s.pit(z, y).reshape(-1)
    zz = z.reshape(-1)
    edges = torch.linspace(g.eval_lo, g.eval_hi, n_bins_z + 1,
                           device=z.device, dtype=z.dtype)
    tot, wsum = 0.0, 0.0
    for b in range(n_bins_z):
        m = (zz >= edges[b]) & (zz < edges[b + 1])
        c = int(m.sum())
        k = _kl(u[m], nbins) if c else None
        if k is not None:
            tot += c * max(k, 0.0); wsum += c
    return tot / wsum if wsum else 0.0


def estimator_floor(s, cfg, n_samples, rows=4, seed=99):
    g, dev, dt = s.grid, s.device, s.dtype
    gen = torch.Generator(device=dev); gen.manual_seed(seed)
    mask = g.eval_mask(dev, dt)
    acc = MeanForceAccumulator(rows, g, cfg["bw_mf"], cfg["n_min"], dev, dt)
    done, chunk = 0, 131072
    while done < n_samples:
        b = min(chunk, n_samples - done)
        Z = (torch.rand((rows, b), device=dev, dtype=dt, generator=gen)
             * (g.eval_hi - g.eval_lo) + g.eval_lo)
        Y = s.sample_fiber(Z, gen)
        acc.deposit(Z, s.mean_force_z(Z, Y))
        done += b
    return float(gauge_l2(acc.free_energy(mask), s.F_ref, mask).mean())


def run(s, arm, cfg, rows, seed):
    g, dev, dt = s.grid, s.device, s.dtype
    gen = torch.Generator(device=dev); gen.manual_seed(seed)
    N, n_cond = cfg["N"], cfg["n_cond"]
    mask = g.eval_mask(dev, dt)

    # --- initial condition --------------------------------------------------
    if arm.startswith("ti_"):
        Z = torch.linspace(g.eval_lo, g.eval_hi, N, device=dev, dtype=dt)
        Z = Z.unsqueeze(0).expand(rows, N).contiguous()
    else:
        Z = torch.full((rows, N), cfg["z0"], device=dev, dtype=dt)
        Z = g.enforce(Z + cfg["z0_jitter"] * torch.randn(Z.shape, device=dev,
                                                         dtype=dt, generator=gen))
    if arm == "ti_warm":
        Y = s.sample_fiber(Z, gen)
    else:                                   # COLD: everything in one channel
        Y = torch.full((rows, N), cfg["y0"], device=dev, dtype=dt)

    acc = MeanForceAccumulator(rows, g, cfg["bw_mf"], cfg["n_min"], dev, dt)
    afc = None
    if LIFTS.get(arm) == "fitted":
        afc = AdaptiveFiberCDF(rows, g, s.p.y_max, dev, dt, n_y=cfg["fit_n_y"],
                               bw_z=cfg["fit_bw_z"], bw_y=cfg["fit_bw_y"],
                               n_min=cfg["fit_n_min"],
                               decay=cfg["fit_decay"] if arm.endswith("decay") else 1.0)
    gmm = None
    if not arm.startswith("ti_"):
        gmm = GMM1D(rows, g, cfg["gmm_K"], dev, dt, eps_bg=1e-3)
        gmm.fit(Z, n_em=10)
    zpit = ZResolvedPIT(g, dev, dt)
    n_outer = cfg["n_steps"] // n_cond
    reset_it = (None if not cfg.get("acc_reset_at")
                else int(cfg["acc_reset_at"] * n_outer))
    dtau = n_cond * cfg["dt"]
    save_every = max(1, n_outer // cfg["n_saves"])
    curve_fe, curve_e, curve_d, curve_dz, curve_c = [], [], [], [], []
    log_eta = float(np.log(1.0 / g.volume))

    for it in range(n_outer):
        if reset_it is not None and it == reset_it:
            # a self-built lift is worst early, and the mean-force accumulator keeps
            # every deposit forever; discarding the warm-up is the obvious pairing
            acc.S0.zero_(); acc.S1.zero_()
        st = cfg.get("dep_stride", 1)
        for k in range(n_cond):
            Y = s.step_fiber_z(Z, Y, cfg["dt"], gen)
            # the fiber autocorrelation time is O(1) against dt = 1e-3, so
            # consecutive deposits are ~1000x redundant: striding costs no
            # effective sample size and the inner loop is launch-bound
            if k >= cfg["n_eq"] and k % st == 0:
                acc.deposit(Z, s.mean_force_z(Z, Y))
            if afc is not None and k % st == 0:
                afc.deposit(Z, Y)
        if gmm is not None:
            gmm.fit(Z, n_em=3)
            # --- Wasserstein probability flow toward uniform -----------------
            if arm != "fr_only":
                sc = torch.clamp(gmm.score(Z), -cfg["w_clip"], cfg["w_clip"])
                Zn = g.enforce(Z - cfg["kappa"] * dtau * sc)
                Y = lift(s, Z, Y, Zn, LIFTS[arm], gen, afc)
                Z = Zn
            # --- Fisher-Rao selection ---------------------------------------
            lr = log_eta - gmm.log_prob(Z)
            th = torch.full((rows,), cfg["theta"], device=dev, dtype=dt)
            w_, _, _ = theta_backoff(lr, th, cfg["alpha_ess"])
            sel = systematic_resample(w_, gen)
            Z, Y = torch.gather(Z, 1, sel), torch.gather(Y, 1, sel)
            if cfg["jitter"] > 0:
                Zn = g.enforce(Z + cfg["jitter"] * torch.randn(
                    Z.shape, device=dev, dtype=dt, generator=gen))
                Y = lift(s, Z, Y, Zn, LIFTS[arm], gen, afc)
                Z = Zn
        if it >= n_outer // 2:
            zpit.add(s, Z, Y)
        if (it + 1) % save_every == 0:
            curve_fe.append(float((it + 1) * n_cond * N))
            curve_e.append(gauge_l2(acc.free_energy(mask), s.F_ref, mask).cpu().numpy())
            curve_d.append(kl_pit(s, Z, Y))
            curve_dz.append(zpit.value())
            curve_c.append(afc.coverage() if afc is not None else 1.0)
    return dict(fe=np.array(curve_fe), err=np.stack(curve_e, 0), dcond=np.array(curve_d),
                dcond_z=np.array(curve_dz), cover=np.array(curve_c), Z=Z, Y=Y)


def lift(s, Z, Y, Zn, mode, gen, afc=None):
    if mode == "fitted":
        return afc.lift(Z, Y, Zn)
    if mode == "oracle":
        return s.sample_fiber(Zn, gen)
    if mode == "adiabatic":
        return s.lift_cdf(Z, Y, Zn)
    if mode == "cartesian":
        return Y
    if mode == "minnorm":
        c = s.cv.c(Y)
        return torch.clamp(Y + (Zn - Z) * c / (1.0 + c * c), -s.p.y_max, s.p.y_max)
    raise ValueError(mode)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="CHANNEL")
    ap.add_argument("--a", type=float, default=0.6)
    ap.add_argument("--k", type=float, default=1.4)
    ap.add_argument("--N", type=int, default=256)
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--n_eq", type=int, default=0)
    ap.add_argument("--n_cond", type=int, default=5)
    ap.add_argument("--arms", nargs="+", default=None)
    ap.add_argument("--out", default="results/manifold/arms")
    ap.add_argument("--tag", default="")
    ap.add_argument("--acc_reset_at", type=float, default=None)
    ap.add_argument("--kappa", type=float, default=2.0)
    ap.add_argument("--fit_decay", type=float, default=0.999)
    ap.add_argument("--fit_bw_z", type=float, default=0.03)
    ap.add_argument("--fit_bw_y", type=float, default=0.05)
    args = ap.parse_args()

    s = build_graph(args.system, a=args.a, k=args.k)
    cfg = dict(N=args.N, n_steps=args.steps, n_cond=args.n_cond, n_eq=args.n_eq, dt=1e-3,
               bw_mf=0.02, n_min=1.0, kappa=args.kappa, theta=0.3, jitter=0.01,
               w_clip=50.0, gmm_K=24, alpha_ess=0.5, z0=-1.0, z0_jitter=0.05,
               y0=-1.5, n_saves=100,
               # fitted-lift settings: bw_z is the sensitive one (the conditional
               # changes fast with z), and the pair has its OWN systematic floor,
               # measured at D_cond ~ 0.003 -- see docs/RESULTS_LOG.md
               fit_n_y=321, fit_bw_z=args.fit_bw_z, fit_bw_y=args.fit_bw_y,
               fit_n_min=200.0, fit_decay=args.fit_decay,
               acc_reset_at=args.acc_reset_at)
    floor = estimator_floor(s, cfg, 2 ** 23)
    print(f"=== E4 {args.system} a={args.a} k={args.k}  N={args.N} steps={args.steps} "
          f"seeds={args.seeds} n_eq={args.n_eq}  floor={floor:.5f} ===", flush=True)

    out = {}
    arms = args.arms or ["ti_cold", "ti_warm", "fr_only", "wfr_cart",
                         "wfr_minnorm", "wfr_fit", "wfr_fit_decay",
                         "wfr_adiab", "wfr_oracle"]
    for arm in arms:
        t0 = time.time()
        r = run(s, arm, cfg, args.seeds, 9000)
        fin = r["err"][-1]
        out[arm] = dict(fe=r["fe"].tolist(), err=r["err"].tolist(),
                        dcond=r["dcond"].tolist(), dcond_z=r["dcond_z"].tolist(),
                        cover=r["cover"].tolist(),
                        final=fin.tolist(),
                        final_median=float(np.median(fin)), dcond_final=float(r["dcond"][-1]))
        print(f"  {arm:12s} eF={np.median(fin):.5f} "
              f"[{np.percentile(fin,25):.5f},{np.percentile(fin,75):.5f}] "
              f"/floor={np.median(fin)/floor:5.1f}  D_pool={r['dcond'][-1]:.4f} "
              f"D_z={r['dcond_z'][-1]:.4f} cover={r['cover'][-1]:.2f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    os.makedirs(args.out, exist_ok=True)
    path = (f"{args.out}/{args.system}_a{args.a}_k{args.k}"
            f"_nc{args.n_cond}_neq{args.n_eq}{args.tag}.json")
    with open(path, "w") as fh:
        json.dump(dict(cfg=cfg, floor=floor, a=args.a, k=args.k, arms=out), fh)
    print("wrote", path)
