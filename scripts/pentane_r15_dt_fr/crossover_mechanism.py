#!/usr/bin/env python
"""Addendum to docs/PENTANE_R15_DT_FR.md: why does ABF at the TG/TT crossover (R ~ 3.03-3.14) get WORSE when
the timestep is refined?  ABF only.  Frozen design: configs/pentane_r15_dt_fr/addendum_crossover_mechanism.json.

Two run modes, one per-walker tracker:

* ``adaptive``: the production ABF sampler (``core_dist.run_sampler_dist``, unchanged) with a ``force_fn`` hook whose
  arithmetic is bitwise the production force (``run_arm.ClipStatsForces``).  The hook sees every configuration
  before each update, so it records per-walker torsional state, R, the f_R sample and clip activity without
  touching the dynamics, and it re-accumulates the ABF sums exactly as the engine does, which gives the APPLIED
  bias at any step.
* ``frozen``: the production update (EM, clip 200, bias +A'(R) grad R clipped to +-60, soft walls, COM removal) with
  a FIXED bias profile taken from an ``adaptive`` run, no estimator, all-trans start.  Options: no physical clip;
  ``--shadow-dt0``: integrate the first-order EM shadow drift b + (dt0/2) J_b b (J_b b by central differences of
  the full drift), i.e. the invariant measure of EM at dt0 with accurate kinetics at --dt.

    python scripts/pentane_r15_dt_fr/crossover_mechanism.py adaptive --dt 5e-4 --rng-seed 20261201
    python scripts/pentane_r15_dt_fr/crossover_mechanism.py frozen --dt 2.5e-4 --bias early5 --T 12 --rng-seed 20261301
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import platform
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)
torch.set_default_dtype(torch.float64)
from alkanes import potentials as pot, core_dist as cd, geometry as geom, interval as iv   # noqa: E402
from alkanes.distance_cv import DistanceCV, dist_bias_force                                 # noqa: E402
from run_arm import CELL, build as build_arm, git_commit                                   # noqa: E402

OUT = os.path.join(ROOT, "results", "pentane_r15_dt_fr", "crossover")
GA = math.radians(116.57)           # gauche core centre (as torsion_rates.py)
WCORE = math.radians(30.0)          # committed-state cores +-30 deg
BAR = math.radians(61.6)            # engine's basin boundary (DistSimConfig.basin_barrier)
FINE_EDGES = np.round(np.arange(2.5, 3.6 + 1e-9, 0.025), 6)          # 44 bins
COND_EDGES = np.linspace(1.4, 3.7, 13)                                # engine's 12 conditional bins
EXC_IN, EXC_OUT = 3.10, 3.15        # TT excursion below the crossover: enter R < 3.10, leave R > 3.15 (hysteresis)
DUR_EDGES = np.logspace(-3, 2, 51)
FAM = ["TT", "TG", "GG same", "GG opp"]
# name -> (source adaptive run dt, its T, bias time, ramp factor the engine applied at that time).  early5 / mid5 come from a
# T = 4 run, bitwise identical to the T = 40 run up to t = 4.  mid5 (exploratory follow-up): t = 2, inside the 2.5 warm-up ramp
BIASES = {"early5": ("5e-4", 4.0, 4.0, 1.0), "mid5": ("5e-4", 4.0, 2.0, 0.8), "final5": ("5e-4", 40.0, 40.0, 1.0),
          "final25": ("2.5e-4", 40.0, 40.0, 1.0)}


def _fam(b1, b2):
    ng = (b1 > 0).long() + (b2 > 0).long()
    f = torch.where(ng == 0, torch.zeros_like(b1), torch.ones_like(b1))
    f = torch.where((ng == 2) & (b1 == b2), 2 * torch.ones_like(b1), f)
    return torch.where((ng == 2) & (b1 != b2), 3 * torch.ones_like(b1), f)


def _basin(phi):
    b = torch.zeros_like(phi, dtype=torch.long)
    b = torch.where(phi >= BAR, torch.ones_like(b), b)
    return torch.where(phi <= -BAR, 2 * torch.ones_like(b), b)


def _core(phi):
    c = torch.full_like(phi, -1, dtype=torch.long)
    c = torch.where(phi.abs() < WCORE, torch.zeros_like(c), c)
    c = torch.where((phi - GA).abs() < WCORE, torch.ones_like(c), c)
    return torch.where((phi + GA).abs() < WCORE, 2 * torch.ones_like(c), c)


class Tracker:
    """Per-walker torsional state, crossover residence, per-family deposits and clip activity (no effect on dynamics)."""

    def __init__(self, R, N, dt, beta, prod_step=0, ck_every=1, bias_every=None, abf_sums=False):
        self.R, self.N, self.dt, self.beta = R, N, float(dt), beta
        self.prod_step, self.ck_every, self.bias_every = int(prod_step), int(ck_every), bias_every
        self.cv = DistanceCV(0, 4)
        self.fe = torch.as_tensor(FINE_EDGES); self.ce = torch.as_tensor(COND_EDGES); self.nb = len(FINE_EDGES) - 1
        self.seed_of = torch.arange(R).repeat_interleave(N)
        self.step = 0
        self.t_origin, self.step_origin = 0.0, 0                        # physical time = t_origin + (step - step_origin) * dt
        self.cstate = torch.zeros(R * N, 2, dtype=torch.long)          # all-trans start
        self.cfam_prev = torch.zeros(R * N, dtype=torch.long)
        z = lambda *s: torch.zeros(*s)                                  # noqa: E731
        nb = self.nb
        self.t_cfam = z(R, nb, 4); self.ev = z(R, nb, 4, 4)
        self.dep_n = z(R, nb, 4); self.dep_f = z(R, nb, 4); self.dep_f2 = z(R, nb, 4)
        self.clip_n = z(R, nb, 4); self.clip_x = z(R, nb, 4)
        self.cond_n = z(R, 12, 4); self.cond_f = z(R, 12, 4)
        self.clip_walker_steps = z(R); self.clip_atom_evals = z(R); self.walker_steps = z(R); self.wall_steps = z(R)
        self.exc_in = torch.zeros(R * N, dtype=torch.bool); self.exc_start = torch.zeros(R * N)          # physical start time
        self.exc_n = z(R, 2); self.exc_t = z(R, 2); self.exc_hist = z(R, 2, len(DUR_EDGES) - 1)   # [conversion, exit]
        self.bond = z(R, 3); self.bend = z(R, 3)                        # committed-TT walkers: n, sum, sum sq
        self.w8_cum = z(R, 4)
        self.ts = {k: [] for k in ("t", "cfam", "bfam", "n_exc", "w8", "ev_cum", "R_TT_mean", "n_TT_below")}
        self.abf_sums = abf_sums
        if abf_sums:
            self.grid, _ = iv.interval_grid(256, 1.4, 3.7)
            self.K = iv.gaussian_kernel_matrix(self.grid, 0.04)
            self.fsum = z(R, 256); self.csum = z(R, 256); self.fsum_prod = z(R, 256); self.csum_prod = z(R, 256)
            self.bias_t = []; self.bias = []

    def now(self):
        return self.t_origin + (self.step - self.step_origin) * self.dt

    def set_dt(self, dt):
        """Timestep switch (exploratory dt-switch runs): later walker-times, durations and checkpoints use the new dt."""
        self.t_origin, self.step_origin = self.now(), self.step
        self.dt = float(dt); self.ck_every = round(0.25 / self.dt)

    def _acc(self, target, seed, idx, valid, w, shape):
        lin = (seed * int(np.prod(shape[1:])) + idx)[valid]
        target += torch.bincount(lin, weights=w[valid], minlength=int(np.prod(shape))).reshape(shape)

    def observe(self, q, f, norm):
        R, N, nb, dt = self.R, self.N, self.nb, self.dt
        f_loc, Rv, _ = self.cv.local_mean_force(q, f, self.beta)
        f_loc = torch.clamp(f_loc, -CELL["abf_force_clip"] * 8, CELL["abf_force_clip"] * 8)
        if self.abf_sums:                                               # exactly the engine's deposit (deposit_stride 1)
            Rr, fr = Rv.reshape(R, N), f_loc.reshape(R, N)
            self.fsum += iv.bin_sum(Rr, fr, 256, 1.4, 3.7); self.csum += iv.bin_counts(Rr, 256, 1.4, 3.7)
            if self.step >= self.prod_step:
                self.fsum_prod += iv.bin_sum(Rr, fr, 256, 1.4, 3.7); self.csum_prod += iv.bin_counts(Rr, 256, 1.4, 3.7)
            if self.bias_every and self.step % self.bias_every == 0:
                self.bias_t.append(self.step * dt); self.bias.append(iv.mean_force_profile(self.fsum, self.csum, self.K).numpy().copy())
        phi1 = geom.signed_dihedral(q, 0, 1, 2, 3); phi2 = geom.signed_dihedral(q, 1, 2, 3, 4)
        bfam = _fam(_basin(phi1), _basin(phi2))
        c = torch.stack([_core(phi1), _core(phi2)], 1)
        self.cstate = torch.where(c >= 0, c, self.cstate)
        cfam = _fam(self.cstate[:, 0], self.cstate[:, 1])
        s = self.seed_of
        fb = torch.bucketize(Rv, self.fe) - 1; fv = (fb >= 0) & (fb < nb); fb = fb.clamp(0, nb - 1)
        one = torch.ones_like(Rv)
        self._acc(self.t_cfam, s, fb * 4 + cfam, fv, one * dt, (R, nb, 4))
        ch = cfam != self.cfam_prev
        self._acc(self.ev, s, fb * 16 + self.cfam_prev * 4 + cfam, fv & ch, one, (R, nb, 4, 4))
        ncl = (norm > CELL["force_clip"]).sum(-1).to(Rv.dtype); clipped = (ncl > 0).to(Rv.dtype)
        excess = (norm - CELL["force_clip"]).clamp_min(0.0).sum(-1)
        self._acc(self.clip_n, s, fb * 4 + bfam, fv, clipped, (R, nb, 4)); self._acc(self.clip_x, s, fb * 4 + bfam, fv, excess, (R, nb, 4))
        allv = torch.ones_like(fv)
        self.clip_walker_steps += torch.bincount(s, weights=clipped, minlength=R); self.clip_atom_evals += torch.bincount(s, weights=ncl, minlength=R)
        self.walker_steps += torch.bincount(s, minlength=R).to(Rv.dtype)
        self.wall_steps += torch.bincount(s, weights=((Rv < CELL["wall_lo"]) | (Rv > CELL["wall_hi"])).to(Rv.dtype), minlength=R)
        if self.step >= self.prod_step:
            self._acc(self.dep_n, s, fb * 4 + bfam, fv, one, (R, nb, 4)); self._acc(self.dep_f, s, fb * 4 + bfam, fv, f_loc, (R, nb, 4))
            self._acc(self.dep_f2, s, fb * 4 + bfam, fv, f_loc * f_loc, (R, nb, 4))
            cb = (torch.bucketize(Rv, self.ce) - 1).clamp(0, 11)
            self._acc(self.cond_n, s, cb * 4 + bfam, allv, one, (R, 12, 4)); self._acc(self.cond_f, s, cb * 4 + bfam, allv, f_loc, (R, 12, 4))
        # TT excursions below the crossover
        tt = cfam == 0
        end_conv = self.exc_in & ~tt
        end_exit = self.exc_in & tt & (Rv > EXC_OUT)
        tnow = self.now()
        for j, m in enumerate((end_conv, end_exit)):
            if m.any():
                dur = tnow - self.exc_start[m]
                sj = s[m]
                self.exc_n[:, j] += torch.bincount(sj, minlength=R).to(Rv.dtype); self.exc_t[:, j] += torch.bincount(sj, weights=dur, minlength=R)
                hb = (torch.bucketize(dur, torch.as_tensor(DUR_EDGES)) - 1).clamp(0, len(DUR_EDGES) - 2)
                self.exc_hist[:, j] += torch.bincount(sj * (len(DUR_EDGES) - 1) + hb, minlength=R * (len(DUR_EDGES) - 1)).reshape(R, -1).to(Rv.dtype)
        self.exc_in = self.exc_in & ~(end_conv | end_exit)
        start = ~self.exc_in & tt & (Rv < EXC_IN)
        self.exc_start = torch.where(start, torch.full_like(self.exc_start, tnow), self.exc_start)
        self.exc_in = self.exc_in | start
        # stiff-mode statistics of committed-TT walkers
        if tt.any():
            qt = q[tt]; st = s[tt]
            bl = torch.linalg.norm(qt[:, 1:] - qt[:, :-1], dim=-1)                       # (n, 4)
            b1 = qt[:, 1:-1] - qt[:, :-2]; b2 = qt[:, 2:] - qt[:, 1:-1]
            ct = ((b1 * b2).sum(-1) / (torch.linalg.norm(b1, dim=-1) * torch.linalg.norm(b2, dim=-1))).clamp(-1 + 1e-12, 1 - 1e-12)
            th = torch.arccos(ct)                                                          # (n, 3) bend angles
            for arr, x in ((self.bond, bl), (self.bend, th)):
                arr[:, 0] += torch.bincount(st, minlength=R).to(x.dtype) * x.shape[1]
                arr[:, 1] += torch.bincount(st, weights=x.sum(-1), minlength=R); arr[:, 2] += torch.bincount(st, weights=(x * x).sum(-1), minlength=R)
        self.cfam_prev = cfam
        # basin-family walker-time in conditional bin 8 (2.933-3.125), cumulative over all steps
        in8 = ((Rv >= COND_EDGES[8]) & (Rv < COND_EDGES[9])).to(Rv.dtype)
        self.w8_cum += torch.bincount(s * 4 + bfam, weights=in8 * dt, minlength=R * 4).reshape(R, 4)
        if (self.step - self.step_origin) % self.ck_every == 0:
            oh = lambda fm: torch.nn.functional.one_hot(fm, 4).to(Rv.dtype)                 # noqa: E731
            self.ts["t"].append(tnow)
            self.ts["cfam"].append(torch.zeros(R, 4).index_add_(0, s, oh(cfam)).numpy())
            self.ts["bfam"].append(torch.zeros(R, 4).index_add_(0, s, oh(bfam)).numpy())
            self.ts["n_exc"].append(torch.bincount(s, weights=self.exc_in.to(Rv.dtype), minlength=R).numpy())
            self.ts["w8"].append(self.w8_cum.numpy().copy())
            self.ts["ev_cum"].append(self.ev.sum(1).numpy().copy())
            ttd = tt.to(Rv.dtype)
            self.ts["R_TT_mean"].append((torch.bincount(s, weights=Rv * ttd, minlength=R) / torch.bincount(s, weights=ttd, minlength=R).clamp_min(1)).numpy())
            self.ts["n_TT_below"].append(torch.bincount(s, weights=(tt & (Rv < 3.125)).to(Rv.dtype), minlength=R).numpy())
        self.step += 1

    def arrays(self):
        out = dict(t_cfam=self.t_cfam, ev=self.ev, dep_n=self.dep_n, dep_f=self.dep_f, dep_f2=self.dep_f2, clip_n=self.clip_n,
                   clip_x=self.clip_x, cond_n=self.cond_n, cond_f=self.cond_f, clip_walker_steps=self.clip_walker_steps,
                   clip_atom_evals=self.clip_atom_evals, walker_steps=self.walker_steps, wall_steps=self.wall_steps,
                   exc_n=self.exc_n, exc_t=self.exc_t, exc_hist=self.exc_hist, bond=self.bond, bend=self.bend, w8_final=self.w8_cum)
        out = {k: v.numpy() for k, v in out.items()}
        out["exc_open"] = torch.bincount(self.seed_of, weights=self.exc_in.to(torch.float64), minlength=self.R).numpy()
        for k, v in self.ts.items():
            out["ts_" + k] = np.asarray(v)
        if self.abf_sums:
            out.update(fsum=self.fsum.numpy(), csum=self.csum.numpy(), fsum_prod_hook=self.fsum_prod.numpy(), csum_prod_hook=self.csum_prod.numpy(),
                       bias_t=np.asarray(self.bias_t), bias=np.asarray(self.bias))
        out.update(fine_edges=FINE_EDGES, cond_edges=COND_EDGES, dur_edges=DUR_EDGES)
        return out


class HookForces:
    """Production force (bitwise ``run_arm.ClipStatsForces`` arithmetic) that hands every configuration to a Tracker."""

    def __init__(self, tracker):
        self.tracker = tracker

    def __call__(self, q, p):
        q = q.detach().requires_grad_(True)
        e = pot.total_energy(q, p).sum()
        (g,) = torch.autograd.grad(e, q, create_graph=False)
        f = -g
        norm = torch.linalg.norm(f, dim=-1, keepdim=True)
        if p.force_clip and p.force_clip > 0:
            scale = torch.clamp(p.force_clip / norm.clamp_min(pot.EPS), max=1.0)
            f = f * scale
        f = f.detach()
        self.tracker.observe(q.detach(), f, norm[..., 0].detach())
        return f


def _save(path, arrays, meta):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez(tmp, meta_json=json.dumps(meta), **arrays); os.replace(tmp, path)
    with open(path + ".sha256", "w") as fh:
        fh.write(hashlib.sha256(open(path, "rb").read()).hexdigest() + "\n")


def _meta(args, extra):
    m = dict(vars(args)); m.update(extra)
    m.update(git_commit=git_commit(), torch=torch.__version__, numpy=np.__version__, python=platform.python_version(), host=platform.node(),
             threads=torch.get_num_threads(), device="cpu", cell=CELL)
    return m


def cmd_adaptive(args):
    a = SimpleNamespace(arm="abf", fr_rate=0.0, dt=args.dt, T=args.T, deposit_interval=None, rng_seed=args.rng_seed, n_seeds=args.n_seeds)
    p, sim, method, sch = build_arm(a)
    path = os.path.join(OUT, "raw", f"adaptive__dt{args.dt:g}__T{args.T:g}__ns{args.n_seeds}__rng{args.rng_seed}.npz")
    if os.path.exists(path):
        print("exists", path); return
    tr = Tracker(args.n_seeds, sim.n_replicas, args.dt, p.beta, prod_step=sim.estimator_burn_in_steps,
                 ck_every=round(0.25 / args.dt), bias_every=round(1.0 / args.dt), abf_sums=True)
    t0 = time.time()
    out = cd.run_sampler_dist(method, p, sim, list(range(args.n_seeds)), DistanceCV(0, 4), "cpu", initial_dihedrals=[0.0, 0.0],
                              oracle_free_energy=None, collect_conditional=True, verbose=True, force_fn=HookForces(tr))
    wall = time.time() - t0
    arr = tr.arrays()
    arr.update(pmf=out["pmf"], mean_force=out["mean_force"], times=out["times"], fsum_prod=out["fsum_prod"], csum_prod=out["csum_prod"],
               cond_basin=out["cond_basin"], fam_count=out["fam_count"], grid=out["grid"])
    _save(path, arr, _meta(args, dict(mode="adaptive", schedule=sch, wall_seconds=wall, n_steps=sch["n_steps"])))
    print(f"wrote {path} wall {wall / 60:.1f} min", flush=True)


def load_bias(name):
    dts, Ts, tb, _ = BIASES[name]
    fs = sorted(glob.glob(os.path.join(OUT, "raw", f"adaptive__dt{float(dts):g}__T{Ts:g}__ns4__rng20261201.npz")))
    assert len(fs) == 1, fs
    d = np.load(fs[0])
    k = int(np.argmin(np.abs(d["bias_t"] - tb))); assert abs(d["bias_t"][k] - tb) < 1e-9
    return d["bias"][k].mean(0), fs[0], float(d["bias_t"][k])           # seed-mean applied (unclipped) mean-force profile


def cmd_frozen(args):
    torch.set_num_threads(1)
    tag = f"frozen__{args.bias}__dt{args.dt:g}" + ("__noclip" if args.no_clip else "") + (f"__shadow{args.shadow_dt0:g}c{args.shadow_cap:g}" if args.shadow_dt0 else "")
    path = os.path.join(OUT, "raw", f"{tag}__T{args.T:g}__ns{args.n_seeds}__rng{args.rng_seed}.npz")
    if os.path.exists(path):
        print("exists", path); return
    prof, src, tb = load_bias(args.bias)
    scale = BIASES[args.bias][3]
    R, N, A = args.n_seeds, CELL["n_replicas"], 5
    p = pot.AlkaneParams(n_atoms=5, beta=CELL["beta"], sigma=CELL["sigma"], epsilon=CELL["epsilon"], decouple=False,
                         force_clip=0.0 if args.no_clip else CELL["force_clip"])
    sim = SimpleNamespace(wall_lo=CELL["wall_lo"], wall_hi=CELL["wall_hi"], k_wall=CELL["k_wall"])
    cv = DistanceCV(0, 4)
    grid, _ = iv.interval_grid(256, 1.4, 3.7)
    mf = torch.as_tensor(prof)[None, :].expand(R, -1).contiguous()
    gen = torch.Generator().manual_seed(int(args.rng_seed))
    init = torch.zeros(R * N, 2)
    q = geom.place_chain(init, A, d0=p.d0, theta0=p.theta0).reshape(R, N, A, 3)
    q = geom.remove_com(q + 1e-3 * torch.randn(q.shape, generator=gen))
    n_steps = int(round(args.T / args.dt)); assert abs(n_steps * args.dt - args.T) < 1e-9
    noise_scale = math.sqrt(2.0 * args.dt / p.beta)
    tr = Tracker(R, N, args.dt, p.beta, prod_step=0, ck_every=round(0.25 / args.dt))
    hook = HookForces(tr)

    def bias_wall(qf, F):
        Rv, grad_full, _ = cv.geometry(qf)
        Rr = Rv.reshape(R, N)
        bias_at = scale * iv.interval_interp(mf, grid, Rr).clamp(-CELL["abf_force_clip"], CELL["abf_force_clip"])
        return F + dist_bias_force(grad_full, (bias_at + cd._wall_gforce(Rr, sim)).reshape(R * N))

    def drift_plain(qf):
        return bias_wall(qf, pot.forces(qf, p))

    corr_max = 0.0; corr_rel_big = 0; t0 = time.time()
    for step in range(n_steps + 1):
        qf = q.reshape(R * N, A, 3)
        b = bias_wall(qf, hook(qf, p))
        if args.shadow_dt0:
            bn = torch.linalg.norm(b.reshape(R * N, -1), dim=-1).clamp_min(1.0)
            eps = (args.fd_h / bn)[:, None, None]
            Jb = (drift_plain(qf + eps * b) - drift_plain(qf - eps * b)) / (2.0 * eps)
            corr = 0.5 * args.shadow_dt0 * Jb
            cn = torch.linalg.norm(corr.reshape(R * N, -1), dim=-1)
            corr_max = max(corr_max, float(cn.max())); corr_rel_big += int((cn > args.shadow_cap * bn).sum())
            corr = corr * torch.clamp(args.shadow_cap * bn / cn.clamp_min(1e-300), max=1.0)[:, None, None]
            b = b + corr
        if step == n_steps:
            break
        noise = torch.randn(q.shape, generator=gen)
        q = geom.remove_com(q + args.dt * b.reshape(R, N, A, 3) + noise_scale * noise)
        if step % 20000 == 0:
            print(f"  step {step}/{n_steps} {time.time() - t0:.0f}s TT {float((tr.cfam_prev == 0).double().mean()):.3f}", flush=True)
    wall = time.time() - t0
    nonfinite = int((~torch.isfinite(q).reshape(R * N, -1).all(-1)).sum())
    _save(path, tr.arrays(), _meta(args, dict(mode="frozen", bias_source=src, bias_t=tb, bias_scale=scale, bias_profile=list(map(float, prof)), wall_seconds=wall,
                                               n_steps=n_steps, shadow_corr_max=corr_max, shadow_corr_capped=corr_rel_big, nonfinite_walkers=nonfinite)))
    print(f"wrote {path} wall {wall / 60:.1f} min nonfinite {nonfinite} corr_max {corr_max:.3g} corr>drift {corr_rel_big}", flush=True)


def cmd_adaptive2(args):
    """Exploratory follow-up: the production ABF update written out (bitwise run_sampler_dist with --shadow-dt0 0 and the
    clip on; checked on t <= 4 against the confirmatory file), so that the DYNAMICS can integrate the EM shadow drift of
    dt0 while the estimator keeps depositing the physical f_R sample."""
    torch.set_num_threads(1)
    a = SimpleNamespace(arm="abf", fr_rate=0.0, dt=args.dt, T=args.T, deposit_interval=None, rng_seed=args.rng_seed, n_seeds=args.n_seeds)
    p0, sim, _, sch = build_arm(a)
    sw = args.dt2 > 0
    tag = f"adaptive2__dt{args.dt:g}" + (f"__to{args.dt2:g}at{args.t_switch:g}" if sw else "") + ("__noclip" if args.no_clip else "") + (f"__shadow{args.shadow_dt0:g}c{args.shadow_cap:g}" if args.shadow_dt0 else "")
    path = os.path.join(OUT, "raw", f"{tag}__T{args.T:g}__ns{args.n_seeds}__rng{args.rng_seed}.npz")
    if os.path.exists(path):
        print("exists", path); return
    p = pot.AlkaneParams(n_atoms=5, beta=CELL["beta"], sigma=CELL["sigma"], epsilon=CELL["epsilon"], decouple=False,
                         force_clip=0.0 if args.no_clip else CELL["force_clip"])
    R, N, A, beta = args.n_seeds, sim.n_replicas, 5, p.beta
    cv = DistanceCV(0, 4)
    grid, dz = iv.interval_grid(sim.n_grid, sim.R_lo, sim.R_hi)
    K = iv.gaussian_kernel_matrix(grid, sim.abf_bandwidth)
    gen = torch.Generator().manual_seed(int(sim.rng_seed))
    init = torch.zeros(R, N, 2)
    q = geom.place_chain(init.reshape(R * N, 2), A, d0=p.d0, theta0=p.theta0).reshape(R, N, A, 3)
    q = geom.remove_com(q + 1e-3 * torch.randn(q.shape, generator=gen))
    noise_scale = math.sqrt(2.0 * sim.dt / beta)
    z = lambda: torch.zeros(R, sim.n_grid)                                           # noqa: E731
    fsum, csum, fsum_prod, csum_prod = z(), z(), z(), z()
    tr = Tracker(R, N, args.dt, beta, prod_step=sim.estimator_burn_in_steps, ck_every=round(0.25 / args.dt))
    hook = HookForces(tr)
    diag = {k: [] for k in ("times", "mean_force", "pmf", "bias_t", "bias")}

    def drift(qf, F, mf_profile, abf_scale):
        Rv, grad_full, _ = cv.geometry(qf)
        Rr = Rv.reshape(R, N)
        mf_at = iv.interval_interp(mf_profile, grid, Rr).clamp(-sim.abf_force_clip, sim.abf_force_clip)
        applied = abf_scale * mf_at + cd._wall_gforce(Rr, sim)
        return F.reshape(R, N, A, 3) + dist_bias_force(grad_full, applied.reshape(R * N)).reshape(R, N, A, 3)

    n_eval, dt_cur, w, n1 = sim.n_steps, sim.dt, 1.0, None
    if sw:   # exploratory dt switch at t_switch (>= the burn-in, integer): deposits weighted dt / dt1 so each unit of time weighs the same
        assert args.t_switch >= sim.estimator_burn_in_steps * sim.dt - 1e-12 and abs(args.t_switch - round(args.t_switch)) < 1e-12
        n1 = int(round(args.t_switch / sim.dt)); n_eval = n1 + int(round((args.T - args.t_switch) / args.dt2)); save2 = int(round(1.0 / args.dt2))
    corr_max = 0.0; corr_rel_big = 0; t0 = time.time()
    for step in range(n_eval + 1):
        if sw and step == n1:
            dt_cur = args.dt2; noise_scale = math.sqrt(2.0 * dt_cur / beta); tr.set_dt(dt_cur); w = dt_cur / sim.dt
        qf = q.reshape(R * N, A, 3)
        F = hook(qf, p)
        f_loc, R_f, _ = cv.local_mean_force(qf, F, beta)
        Rv = R_f.reshape(R, N)
        f_loc = torch.clamp(f_loc, -sim.abf_force_clip * 8, sim.abf_force_clip * 8).reshape(R, N)
        if w == 1.0:
            df, dc = iv.bin_sum(Rv, f_loc, sim.n_grid, sim.R_lo, sim.R_hi), iv.bin_counts(Rv, sim.n_grid, sim.R_lo, sim.R_hi)
        else:
            df, dc = iv.bin_sum(Rv, w * f_loc, sim.n_grid, sim.R_lo, sim.R_hi), iv.bin_sum(Rv, torch.full_like(Rv, w), sim.n_grid, sim.R_lo, sim.R_hi)
        fsum += df; csum += dc
        if step >= sim.estimator_burn_in_steps:
            fsum_prod += df; csum_prod += dc
        mf_profile = iv.mean_force_profile(fsum, csum, K)
        abf_scale = sim.abf_bias_scale * min(1.0, step / max(sim.abf_warmup_steps, 1))
        b = drift(qf, F, mf_profile, abf_scale)
        if args.shadow_dt0:
            bf = b.reshape(R * N, A, 3)
            bn = torch.linalg.norm(bf.reshape(R * N, -1), dim=-1).clamp_min(1.0)
            eps = (args.fd_h / bn)[:, None, None]
            qp, qm = qf + eps * bf, qf - eps * bf
            Jb = (drift(qp, pot.forces(qp, p), mf_profile, abf_scale) - drift(qm, pot.forces(qm, p), mf_profile, abf_scale)).reshape(R * N, A, 3) / (2.0 * eps)
            corr = 0.5 * args.shadow_dt0 * Jb
            cn = torch.linalg.norm(corr.reshape(R * N, -1), dim=-1)
            corr_max = max(corr_max, float(cn.max())); corr_rel_big += int((cn > args.shadow_cap * bn).sum())
            corr = corr * torch.clamp(args.shadow_cap * bn / cn.clamp_min(1e-300), max=1.0)[:, None, None]
            b = b + corr.reshape(R, N, A, 3)
        phase2 = sw and step >= n1
        if (((step - n1) % save2 == 0) if phase2 else (step % sim.save_every == 0)) or step == n_eval:
            tnow = (args.t_switch + (step - n1) * args.dt2) if phase2 else step * sim.dt
            rf, rc = (fsum_prod, csum_prod) if csum_prod.sum() > 0 else (fsum, csum)
            mf_rep = iv.mean_force_profile(rf, rc, K)
            diag["times"].append(tnow); diag["mean_force"].append(mf_rep.numpy().copy())
            diag["pmf"].append(iv.free_energy_from_mean_force(mf_rep, grid, dz).numpy().copy())
            diag["bias_t"].append(tnow); diag["bias"].append(mf_profile.numpy().copy())
        if step == n_eval:
            break
        noise = torch.randn(q.shape, generator=gen)
        q = geom.remove_com(q + dt_cur * b + noise_scale * noise)
        if step % 40000 == 0:
            print(f"  step {step}/{n_eval} {time.time() - t0:.0f}s TT {float((tr.cfam_prev == 0).double().mean()):.3f}", flush=True)
    wall = time.time() - t0
    arr = tr.arrays()
    arr.update({k: np.asarray(v) for k, v in diag.items()})
    arr.update(fsum_prod=fsum_prod.numpy(), csum_prod=csum_prod.numpy(), fsum=fsum.numpy(), csum=csum.numpy(), grid=grid.numpy())
    nonfinite = int((~torch.isfinite(q).reshape(R * N, -1).all(-1)).sum())
    _save(path, arr, _meta(args, dict(mode="adaptive2", schedule=sch, wall_seconds=wall, n_steps=sch["n_steps"], shadow_corr_max=corr_max,
                                       shadow_corr_gt_drift=corr_rel_big, nonfinite_walkers=nonfinite)))
    print(f"wrote {path} wall {wall / 60:.1f} min nonfinite {nonfinite} corr_max {corr_max:.3g} corr>drift {corr_rel_big}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("adaptive"); a.add_argument("--dt", type=float, required=True); a.add_argument("--T", type=float, default=40.0)
    a.add_argument("--rng-seed", type=int, default=20261201); a.add_argument("--n-seeds", type=int, default=4)
    f = sp.add_parser("frozen"); f.add_argument("--dt", type=float, required=True); f.add_argument("--bias", required=True, choices=sorted(BIASES))
    f.add_argument("--T", type=float, required=True); f.add_argument("--rng-seed", type=int, required=True); f.add_argument("--n-seeds", type=int, default=4)
    f.add_argument("--no-clip", action="store_true"); f.add_argument("--shadow-dt0", type=float, default=0.0); f.add_argument("--fd-h", type=float, default=1e-5)
    f.add_argument("--shadow-cap", type=float, default=0.9, help="cap |shadow correction| <= cap * |drift| (the uncapped first-order shadow potential is unbounded below where the local curvature exceeds 2/dt0)")
    b = sp.add_parser("adaptive2"); b.add_argument("--dt", type=float, required=True); b.add_argument("--T", type=float, default=40.0)
    b.add_argument("--rng-seed", type=int, default=20261201); b.add_argument("--n-seeds", type=int, default=4)
    b.add_argument("--no-clip", action="store_true"); b.add_argument("--shadow-dt0", type=float, default=0.0); b.add_argument("--fd-h", type=float, default=1e-5)
    b.add_argument("--shadow-cap", type=float, default=0.9)
    b.add_argument("--dt2", type=float, default=0.0, help="exploratory: switch to this timestep at --t-switch"); b.add_argument("--t-switch", type=float, default=0.0)
    args = ap.parse_args()
    torch.set_num_threads(1)
    {"adaptive": cmd_adaptive, "frozen": cmd_frozen, "adaptive2": cmd_adaptive2}[args.cmd](args)


if __name__ == "__main__":
    main()
