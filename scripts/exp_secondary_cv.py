"""E11: does promoting the slow fiber mode buy the oracle's benefit at 1-D cost?

The audit's design rule says a lift only has to be conditionally correct on fiber
modes slower than the transport, and that such a mode should have been a
collective variable.  Branch B of the plan turns that into an algorithm: instead
of learning nu^xi(dq | z) in a high-dimensional fiber, promote the one slow
direction y_1 and learn only p(y_1 | z).

Here the fiber is (1 + m)-dimensional: y_1 enters xi and is the promotable mode;
m spectators enter V but not xi, with an x-dependent conditional width, so their
conditional genuinely changes along z and lifting them naively is genuinely wrong.
Their overall stiffness omega_s is a clean fiber-timescale knob (tau ~ 1/omega_s^2),
and the ratio omega_in/omega_out is held fixed so the SHAPE of the change is the
same at every stiffness.

Arms (the lift is the only difference between the WFR arms):

  ti_cold        fixed windows, cold fiber                    classical baseline
  wfr_naive      cartesian on y_1 and on S                    RC-WFR + naive lift
  wfr_promote    EXACT on y_1 (CDF map), naive on S           the secondary-CV arm
  wfr_learned    LEARNED on y_1, naive on S                   the deployable version
  wfr_both       exact on y_1, exact (width-rescaled) on S     both blocks correct
  wfr_oracle     redraw the whole fiber                        unreachable ceiling

Prediction: wfr_promote tracks wfr_both while the spectators relax fast, and falls
away from it as they slow down.  wfr_learned says how much of that survives when
p(y_1|z) is estimated from the run rather than known.
"""
import sys, os, json, argparse, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import torch

from rcwfr.adaptive_lift import AdaptiveFiberCDF
from rcwfr.estimators import MeanForceAccumulator, gauge_l2
from rcwfr.fisher_rao import theta_backoff
from rcwfr.gmm import GMM1D
from rcwfr.grid import DEVICE, DTYPE
from rcwfr.resampling import systematic_resample
from rcwfr.systems.graph import build_graph_nd

# (y_1 lift, spectator lift)
ARMS = {
    "ti_cold":     (None, None),
    "wfr_naive":   ("cartesian", "cartesian"),
    "wfr_promote": ("exact", "cartesian"),
    "wfr_learned": ("learned", "cartesian"),
    "wfr_both":    ("exact", "scaled"),
    "wfr_oracle":  ("oracle", "oracle"),
}


class ZPIT:
    """z-resolved conditional lag, accumulated over the production half."""

    def __init__(self, grid, device, dtype, n_z=12, n_u=16):
        self.grid, self.n_z, self.n_u = grid, n_z, n_u
        self.H = torch.zeros((n_z, n_u), device=device, dtype=dtype)
        self.edges = torch.linspace(grid.eval_lo, grid.eval_hi, n_z + 1,
                                    device=device, dtype=dtype)

    def add(self, z, u):
        z, u = z.reshape(-1), u.reshape(-1)
        iz = torch.clamp(torch.bucketize(z, self.edges) - 1, 0, self.n_z - 1)
        iu = torch.clamp((u * self.n_u).long(), 0, self.n_u - 1)
        m = (z >= self.edges[0]) & (z <= self.edges[-1])
        f = (iz * self.n_u + iu)[m]
        self.H.view(-1).scatter_add_(0, f, torch.ones_like(f, dtype=self.H.dtype))

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


def estimator_floor(s, cfg, n_samples, rows=4, seed=99):
    g, dev, dt = s.grid, s.device, s.dtype
    gen = torch.Generator(device=dev); gen.manual_seed(seed)
    acc = MeanForceAccumulator(rows, g, cfg["bw_mf"], cfg["n_min"], dev, dt)
    done = 0
    while done < n_samples:
        b = min(131072, n_samples - done)
        Z = (torch.rand((rows, b), device=dev, dtype=dt, generator=gen)
             * (g.eval_hi - g.eval_lo) + g.eval_lo)
        Y1 = s.sample_fiber(Z, gen)
        S = s.sample_spectators(Z, Y1, gen)
        acc.deposit(Z, s.mean_force_nd(Z, Y1, S))
        done += b
    return float(gauge_l2(acc.free_energy(g.eval_mask(dev, dt)), s.F_ref,
                          g.eval_mask(dev, dt)).mean())


def lift(s, Z, Y1, S, Zn, y_mode, s_mode, gen, afc):
    if y_mode == "cartesian":
        Y1n = Y1
    elif y_mode == "exact":
        Y1n = s.lift_cdf(Z, Y1, Zn)
    elif y_mode == "learned":
        Y1n = afc.lift(Z, Y1, Zn)
    elif y_mode == "oracle":
        Y1n = s.sample_fiber(Zn, gen)
    else:
        raise ValueError(y_mode)
    Sn = s.lift_spectators(Z, Y1, Zn, Y1n, S, s_mode, gen)
    return Y1n, Sn


def run_all(s, arms, cfg, seeds, seed):
    """Every arm at once, as blocks of the ROW axis.

    The inner loop is launch-latency-bound, not bandwidth-bound: 4096 and 40960
    particles cost the same 2.7 ms per step.  Running the arms sequentially
    therefore wasted a factor len(arms) of wall clock for nothing.  Each arm owns
    rows [i*seeds, (i+1)*seeds); the lift is applied per block (cheap, it is a
    handful of kernels on a slice) and everything else is one batched call.
    """
    g, dev, dt = s.grid, s.device, s.dtype
    gen = torch.Generator(device=dev); gen.manual_seed(seed)
    N, n_cond = cfg["N"], cfg["n_cond"]
    mask = g.eval_mask(dev, dt)
    A = len(arms)
    rows = A * seeds
    blk = {a: slice(i * seeds, (i + 1) * seeds) for i, a in enumerate(arms)}
    is_ti = torch.zeros(rows, dtype=torch.bool, device=dev)

    Z = torch.empty((rows, N), device=dev, dtype=dt)
    grid_z = torch.linspace(g.eval_lo, g.eval_hi, N, device=dev, dtype=dt)
    for a in arms:
        if a.startswith("ti_"):
            Z[blk[a]] = grid_z
            is_ti[blk[a]] = True
        else:
            Z[blk[a]] = cfg["z0"]
    jit = cfg["z0_jitter"] * torch.randn(Z.shape, device=dev, dtype=dt, generator=gen)
    Z = torch.where(is_ti.unsqueeze(1), Z, g.enforce(Z + jit))
    Y1 = torch.full((rows, N), cfg["y0"], device=dev, dtype=dt)
    S = torch.zeros((rows, N, s.p.m_spec), device=dev, dtype=dt)

    acc = MeanForceAccumulator(rows, g, cfg["bw_mf"], cfg["n_min"], dev, dt)
    afc = None
    if any(ARMS[a][0] == "learned" for a in arms):
        afc = AdaptiveFiberCDF(rows, g, s.p.y_max, dev, dt, n_y=cfg["fit_n_y"],
                               bw_z=cfg["fit_bw_z"], bw_y=cfg["fit_bw_y"],
                               n_min=200.0, decay=cfg["fit_decay"])
    gmm = GMM1D(rows, g, cfg["gmm_K"], dev, dt, eps_bg=1e-3)
    gmm.fit(Z, n_em=10)
    n_outer = cfg["n_steps"] // n_cond
    reset_it = None if not cfg["acc_reset_at"] else int(cfg["acc_reset_at"] * n_outer)
    dtau = n_cond * cfg["dt"]
    log_eta = float(np.log(1.0 / g.volume))
    stride = cfg.get("dep_stride", 1)
    zy = {a: ZPIT(g, dev, dt) for a in arms}
    zs = {a: ZPIT(g, dev, dt) for a in arms}

    def do_lift(Zn):
        nonlocal Y1, S
        Y1n, Sn = Y1.clone(), S.clone()
        for a in arms:
            if a.startswith("ti_"):
                continue
            b = blk[a]
            ym, sm = ARMS[a]
            Y1n[b], Sn[b] = lift(s, Z[b], Y1[b], S[b], Zn[b], ym, sm, gen,
                                 _Slice(afc, b) if afc is not None else None)
        Y1, S = Y1n, Sn

    for it in range(n_outer):
        if reset_it is not None and it == reset_it:
            acc.S0.zero_(); acc.S1.zero_()
        for j in range(n_cond):
            Y1, S = s.step_fiber_nd(Z, Y1, S, cfg["dt"], gen)
            if j % stride == 0:
                acc.deposit(Z, s.mean_force_nd(Z, Y1, S))
            if afc is not None and j % stride == 0:
                afc.deposit(Z, Y1)
        gmm.fit(Z, n_em=3)
        sc = torch.clamp(gmm.score(Z), -cfg["w_clip"], cfg["w_clip"])
        Zn = torch.where(is_ti.unsqueeze(1), Z,
                         g.enforce(Z - cfg["kappa"] * dtau * sc))
        do_lift(Zn)
        Z = Zn
        lr = log_eta - gmm.log_prob(Z)
        th = torch.full((rows,), cfg["theta"], device=dev, dtype=dt)
        w_, _, _ = theta_backoff(lr, th, cfg["alpha_ess"])
        sel = systematic_resample(w_, gen)
        keep = torch.arange(N, device=dev).unsqueeze(0).expand(rows, N)
        sel = torch.where(is_ti.unsqueeze(1), keep, sel)
        Z, Y1 = torch.gather(Z, 1, sel), torch.gather(Y1, 1, sel)
        S = torch.gather(S, 1, sel.unsqueeze(-1).expand(-1, -1, s.p.m_spec))
        if cfg["jitter"] > 0:
            Zn = torch.where(is_ti.unsqueeze(1), Z, g.enforce(Z + cfg["jitter"]
                 * torch.randn(Z.shape, device=dev, dtype=dt, generator=gen)))
            do_lift(Zn)
            Z = Zn
        if it >= n_outer // 2 and it % 4 == 0:
            uy, us = s.pit(Z, Y1), s.pit_spectators(Z, Y1, S)
            for a in arms:
                b = blk[a]
                zy[a].add(Z[b], uy[b])
                zs[a].add(Z[b].unsqueeze(-1).expand_as(S[b]), us[b])
    err = gauge_l2(acc.free_energy(mask), s.F_ref, mask).cpu().numpy()
    return {a: dict(final=err[blk[a]].tolist(), Dz_y=zy[a].value(),
                    Dz_S=zs[a].value()) for a in arms}


class _Slice:
    """Row-slice view of an AdaptiveFiberCDF, so one estimator serves all arms."""

    def __init__(self, afc, b):
        self.afc, self.b = afc, b

    def lift(self, Z, Y, Zn):
        if self.afc._cdf is None:
            self.afc._build()
        full_Z = torch.zeros((self.afc.rows, Z.shape[1]), device=Z.device,
                             dtype=Z.dtype)
        full_Y = torch.zeros_like(full_Z)
        full_Zn = torch.zeros_like(full_Z)
        full_Z[self.b], full_Y[self.b], full_Zn[self.b] = Z, Y, Zn
        return self.afc.lift(full_Z, full_Y, full_Zn)[self.b]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="CHANNEL")
    ap.add_argument("--a", type=float, default=0.6)
    ap.add_argument("--k", type=float, default=1.4)
    ap.add_argument("--m_spec", type=int, default=4)
    ap.add_argument("--oms_out", type=float, nargs="+",
                    default=[0.25, 0.5, 1.0, 2.0])
    # A = spectator shift in CONDITIONAL WIDTHS per unit z.  mu_amp = A / omega_s
    # holds the block's lift-lag coefficient fixed at C_S = m beta A^2 while the
    # relaxation time 1/omega_s^2 sweeps freely -- the two knobs the design rule
    # needs to be varied independently.  A = 0 reproduces the width-only block,
    # whose C_S is 0.1% of the y_1 block's and which therefore tests nothing.
    ap.add_argument("--A", type=float, default=0.0)
    ap.add_argument("--N", type=int, default=256)
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--out", default="results/manifold/secondary_cv.json")
    args = ap.parse_args()

    out = {}
    for om in args.oms_out:
        s = build_graph_nd(args.system, a=args.a, k=args.k, m_spec=args.m_spec,
                           oms_out=om, oms_ratio=4.0,
                           mu_amp=(args.A / om) if args.A else 0.0)
        cfg = dict(N=args.N, n_steps=args.steps, n_cond=5, dt=1e-3, bw_mf=0.02,
                   n_min=1.0, kappa=2.0, theta=0.3, jitter=0.01, w_clip=50.0,
                   gmm_K=24, alpha_ess=0.5, z0=-1.0, z0_jitter=0.05, y0=-1.5,
                   fit_n_y=321, fit_bw_z=0.06, fit_bw_y=0.05, fit_decay=0.999,
                   acc_reset_at=0.5,
                   # the fiber autocorrelation time is ~1/omega^2 = O(1) against
                   # dt = 1e-3, so consecutive deposits are ~1000x redundant;
                   # depositing every 5th step costs no effective sample size
                   dep_stride=5)
        fl = estimator_floor(s, cfg, 2 ** 23)
        tau = 1.0 / om ** 2
        C_S = args.m_spec * s.p.beta * args.A ** 2
        print(f"=== omega_s = {om}  (tau_spec ~ {tau:.3f})  m_spec={args.m_spec}  "
              f"A={args.A} (C_S ~ {C_S:.1f} vs C_y1 ~ 35)  floor {fl:.5f} ===",
              flush=True)
        rec = {"floor": fl, "tau_spec": tau, "A": args.A, "C_S": C_S, "arms": {}}
        t0 = time.time()
        res = run_all(s, args.arms, cfg, args.seeds, 9000)
        for arm in args.arms:
            r = res[arm]
            rec["arms"][arm] = r
            m = float(np.median(r["final"]))
            print(f"  {arm:12s} eF={m:.5f} /floor={m/fl:6.1f}  "
                  f"Dz(y1)={r['Dz_y']:.4f}  Dz(S)={r['Dz_S']:.4f}", flush=True)
        print(f"  ({time.time()-t0:.0f}s for all {len(args.arms)} arms)", flush=True)
        out[str(om)] = rec
        del s
        torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"))
    print("wrote", args.out)
