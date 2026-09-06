#!/usr/bin/env python
"""Pentane R15 OT + repair, stages P1 / P2 / tau_perp (docs/PENTANE_R15_OT_REPAIR.md).

No sampler, no ABF estimator, no free-energy error -- the operator and the mechanism, measured
directly against the joint importance-sampling reference:

  0. EXACT conditional samples: the reference's own internal-coordinate importance sampler
     (v4 proposal, weights exp(-beta V_nb)) is re-run and, for every thermal R bin, n_rep
     configurations are drawn by weighted resampling -> exact samples of p(q | R in bin)
     (weights bounded, ESS reported).  Also the identity check <f_R>_ref,bin = <F'_ref>_bin.
  1. P1 stationarity: each replica's R is FIXED at its own value and the projected constrained
     scheme runs n_eq + n_rec steps.  Recorded per bin: b_inf = <f_R> - <F'_ref(R_i)> (the
     operator's mean-force offset), the pooled (phi1,phi2) histogram vs p_ref(phi1,phi2 | bin)
     (TV, against the finite-sample floor TV(0)), and the 9-basin probabilities over time
     (family-mixture drift).  Because |grad R|^2 = 2 the constrained measure IS the conditional.
  2. tau_perp: single-family ensembles (T_T, T_G+, G+_G+, ... wherever the reference gives the
     family >= 5 %) at fixed R; the 9-basin vector p_t(. | R, a) is followed for n_tau steps;
     tau_perp = first t with TV[p_t, p_ref(. | R)] <= 0.2; also the survival in the start family.
  3. P2 single event: from the P1-equilibrated ensembles, LIFT by dR in +-{1/4,1/2,1,2,4,8} bins
     and repair at fixed R' for M steps recording b(m) = <f_R(m) - F'_ref(R'_i)>; then a long run
     for b_inf(R'); conditional TV at m in {0, 5, 20, M} against the destination bin's reference.
     Lift audit: energy jump, max force, max bond strain.

    CUDA_VISIBLE_DEVICES=1 python -u scripts/pentane_r15_ot_p1p2.py [--quick]
Outputs: results/ot_repair_campaign/pentane_r15/P1P2/{p1p2.json, p1p2.npz, figures/}
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
torch.set_default_dtype(torch.float64)
from alkanes import potentials as pot, geometry as geom, interval as iv          # noqa: E402
from alkanes.reference import sample_bond_lengths, sample_bond_angles           # noqa: E402
from alkanes.reference_cv import sample_dihedral_v4                              # noqa: E402
from alkanes.distance_cv import DistanceCV                                       # noqa: E402
from alkanes.ot_repair_dist import lift_to_R, lj_forbidden_radius, compiled_forces, eager_forces   # noqa: E402

OUT = os.path.join(ROOT, "results", "ot_repair_campaign", "pentane_r15", "P1P2")
REF = os.path.join(ROOT, "cache", "alkanes_cv", "ref_pentane_b2_R15_v2_meanforce.npz")   # corrected mean-force-route reference
PI = math.pi
BARRIER = math.radians(61.6)
BASINS = ["T_T", "T_Gp", "T_Gm", "Gp_T", "Gp_Gp", "Gp_Gm", "Gm_T", "Gm_Gp", "Gm_Gm"]
# frozen production cell (configs/alkanes_cv_extension/r15_methods.yaml, stage production)
CELL = dict(beta=2.0, sigma=2.3, epsilon=1.0, force_clip=200.0, dt=5.0e-4, R_lo=1.4, R_hi=3.7,
            n_grid=256, n_rbins=12, n_grid2=48, abf_force_clip=60.0, thermal_delta=10.0)
DZ = (CELL["R_hi"] - CELL["R_lo"]) / CELL["n_grid"]
DR_BINS = (-8, -4, -2, -1, -0.5, -0.25, 0.25, 0.5, 1, 2, 4, 8)


def params():
    return pot.AlkaneParams(n_atoms=5, beta=CELL["beta"], sigma=CELL["sigma"], epsilon=CELL["epsilon"],
                            decouple=False, force_clip=CELL["force_clip"])


def basin_index(phi1, phi2):
    def b(phi):
        return torch.where(phi >= BARRIER, torch.ones_like(phi, dtype=torch.long),
                           torch.where(phi <= -BARRIER, 2 * torch.ones_like(phi, dtype=torch.long),
                                       torch.zeros_like(phi, dtype=torch.long)))
    return b(phi1) * 3 + b(phi2)


def torsions(q):
    return geom.signed_dihedral(q, 0, 1, 2, 3), geom.signed_dihedral(q, 1, 2, 3, 4)


def hist2(phi1, phi2, group, n_groups, n2, dphi, weights=None):
    """(n_groups, n2, n2) torsion histogram, ``group`` (B,) long."""
    i1 = torch.floor((phi1 + PI) / dphi).long().clamp(0, n2 - 1)
    i2 = torch.floor((phi2 + PI) / dphi).long().clamp(0, n2 - 1)
    lin = group * (n2 * n2) + i1 * n2 + i2
    out = torch.zeros(n_groups * n2 * n2, device=phi1.device, dtype=phi1.dtype)
    out.scatter_add_(0, lin, torch.ones_like(phi1) if weights is None else weights)
    return out.reshape(n_groups, n2, n2)


def basin_probs(phi1, phi2, group, n_groups, weights=None):
    b = basin_index(phi1, phi2)
    out = torch.zeros(n_groups * 9, device=phi1.device, dtype=phi1.dtype)
    out.scatter_add_(0, group * 9 + b, torch.ones_like(phi1) if weights is None else weights)
    out = out.reshape(n_groups, 9)
    return out / out.sum(-1, keepdim=True).clamp_min(1e-12)


def tv_hist(h, ref_dens, dphi):
    """TV between a count histogram (n2,n2) and a reference density (n2,n2)."""
    hs = h.sum()
    if hs <= 0:
        return float("nan")
    p = h / (hs * dphi * dphi)
    r = ref_dens / max(float((ref_dens.sum() * dphi * dphi)), 1e-300)
    return float(0.5 * np.abs(p - r).sum() * dphi * dphi)


def fmean(q, F, cv, beta):
    """The estimator's local-mean-force sample (clipped like the sampler)."""
    f, _, _ = cv.local_mean_force(q, F, beta)
    return torch.clamp(f, -CELL["abf_force_clip"] * 8, CELL["abf_force_clip"] * 8)


@torch.no_grad()
def exact_samples(p, ref, n_total, chunk, n_rep, dev, gen, thermal_bins, cv, force_fn, Fp_interp):
    """Weighted-resampled exact conditional samples per thermal bin + the <f_R> identity check."""
    A = 5
    edges = torch.as_tensor(ref["cond_edges"], device=dev)
    nb = int(ref["cond_hist"].shape[0])
    cap = 40000                                            # uniformly thinned reservoir per bin (weights kept)
    res_q = {k: [] for k in thermal_bins}; res_w = {k: [] for k in thermal_bins}; res_f = {k: [] for k in thermal_bins}
    res_R = {k: [] for k in thermal_bins}
    raw_count = torch.zeros(nb, device=dev); raw_w = torch.zeros(nb, device=dev); raw_w2 = torch.zeros(nb, device=dev)
    fw_sum = torch.zeros(nb, device=dev); fpw_sum = torch.zeros(nb, device=dev)
    thin = None
    done = 0
    while done < n_total:
        m = min(chunk, n_total - done)
        bonds = torch.stack([sample_bond_lengths(m, p, gen, dev) for _ in range(A - 1)], 1)
        angles = torch.stack([sample_bond_angles(m, p, gen, dev) for _ in range(A - 2)], 1)
        dih = torch.stack([sample_dihedral_v4(m, p, gen, dev) for _ in range(A - 3)], 1)
        q = geom.place_chain_internal(bonds, angles, dih, A, device=dev)
        R = cv.value(q)
        w = torch.exp(-p.beta * pot.nonbonded_energy(q, p))
        F = force_fn(q, p)
        f = fmean(q, F, cv, p.beta)
        bin_id = (torch.bucketize(R, edges) - 1).clamp(0, nb - 1)
        raw_count.scatter_add_(0, bin_id, torch.ones_like(w)); raw_w.scatter_add_(0, bin_id, w); raw_w2.scatter_add_(0, bin_id, w * w)
        fw_sum.scatter_add_(0, bin_id, w * f); fpw_sum.scatter_add_(0, bin_id, w * Fp_interp(R))
        if thin is None:                                   # thinning probabilities from the first chunk
            est = raw_count * (n_total / m)
            thin = torch.clamp(cap / est.clamp_min(1.0), max=1.0)
        keep = torch.rand(m, device=dev, generator=gen) < thin[bin_id]
        for k in thermal_bins:
            sel = torch.nonzero((bin_id == k) & keep).flatten()
            if sel.numel():
                res_q[k].append(q[sel]); res_w[k].append(w[sel]); res_f[k].append(f[sel]); res_R[k].append(R[sel])
        done += m
    out = {}
    for k in thermal_bins:
        Q = torch.cat(res_q[k]); W = torch.cat(res_w[k]); Fk = torch.cat(res_f[k]); Rk = torch.cat(res_R[k])
        idx = torch.multinomial(W, n_rep, replacement=True, generator=gen)
        ess = float(W.sum() ** 2 / (W * W).sum())
        out[k] = dict(q=Q[idx].clone(), R=Rk[idx].clone(), f_exact=Fk[idx].clone(), n_raw=int(Q.shape[0]), ess=ess,
                      basin_probs_exact=basin_probs(*torsions(Q), torch.zeros(Q.shape[0], dtype=torch.long, device=dev), 1, W)[0].cpu().numpy())
    ident = dict(bin=list(range(nb)), n_raw=raw_count.cpu().numpy().tolist(), ess_raw=(raw_w ** 2 / raw_w2.clamp_min(1e-300)).cpu().numpy().tolist(),
                 f_mean_ref=(fw_sum / raw_w.clamp_min(1e-300)).cpu().numpy().tolist(),
                 Fp_mean_ref=(fpw_sum / raw_w.clamp_min(1e-300)).cpu().numpy().tolist())
    return out, ident


@torch.no_grad()
def constrained_run(q, R_fixed, n_steps, p, gen, force_fn, cv, group=None, n_groups=None, n2=48, dphi=None,
                    record=False, snap_every=0, ref_basins=None):
    """Projected constrained EM at fixed R for every walker.  Optionally records per-walker f_R
    time means, the pooled torsion histogram per group and 9-basin snapshots."""
    dt = p.dt if hasattr(p, "dt") else CELL["dt"]
    noise_scale = math.sqrt(2.0 * dt / p.beta)
    f_sum = torch.zeros(q.shape[0], device=q.device) if record else None
    f_sq = torch.zeros(q.shape[0], device=q.device) if record else None
    hist = torch.zeros(n_groups, n2, n2, device=q.device) if record else None
    snaps = []
    for s in range(n_steps):
        F = force_fn(q, p)
        if record:
            f = fmean(q, F, cv, p.beta); f_sum += f; f_sq += f * f
            ph1, ph2 = torsions(q)
            hist += hist2(ph1, ph2, group, n_groups, n2, dphi)
        if snap_every and (s % snap_every == 0):
            ph1, ph2 = torsions(q)
            snaps.append(basin_probs(ph1, ph2, group, n_groups).cpu().numpy())
        noise = torch.randn(q.shape, generator=gen, device=q.device, dtype=q.dtype)
        q = geom.remove_com(lift_to_R(q + dt * F + noise_scale * noise, R_fixed, cv.i, cv.j))
    if snap_every:
        ph1, ph2 = torsions(q)
        snaps.append(basin_probs(ph1, ph2, group, n_groups).cpu().numpy())
    return q, (f_sum / n_steps if record else None), (f_sq / n_steps if record else None), hist, (np.stack(snaps) if snaps else None)


@torch.no_grad()
def lift_audit(p, q_before, q_after):
    V0 = pot.total_energy(q_before, p); V1 = pot.total_energy(q_after, p)
    F1 = eager_forces(q_after, p)
    fmax = torch.linalg.norm(F1, dim=-1).amax(1)
    bonds = torch.stack([torch.linalg.norm(q_after[:, a + 1] - q_after[:, a], dim=-1) for a in range(4)], 1)
    return dict(dV_mean=float((V1 - V0).mean()), dV_max=float((V1 - V0).max()), fmax_median=float(fmax.median()),
                fmax_max=float(fmax.max()), bond_strain_max=float((bonds - p.d0).abs().max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--n-rep", type=int, default=1024)
    ap.add_argument("--n-eq", type=int, default=4000)
    ap.add_argument("--n-rec", type=int, default=4000)
    ap.add_argument("--n-tau", type=int, default=40000)
    ap.add_argument("--m-repair", type=int, default=60)
    ap.add_argument("--n-samples", type=int, default=8_000_000)
    a = ap.parse_args()
    if a.quick:
        a.n_rep, a.n_eq, a.n_rec, a.n_tau, a.m_repair, a.n_samples = 128, 200, 200, 1000, 20, 400_000
    os.makedirs(os.path.join(a.out, "figures"), exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    p = params(); cv = DistanceCV(0, 4)
    force_fn = compiled_forces() if dev == "cuda" else eager_forces
    ref = np.load(REF, allow_pickle=True)
    grid = np.asarray(ref["grid"]); F_ref = np.asarray(ref["F"]); Fp_ref = np.asarray(ref["Fprime"])
    cond_dens = np.asarray(ref["cond_dens"]); edges = np.asarray(ref["cond_edges"]); dphi = float(ref["cond_dphi"])
    ref_basins = np.asarray(ref["cond_basin_probs"]); nb = cond_dens.shape[0]; n2 = cond_dens.shape[1]
    centres = 0.5 * (edges[1:] + edges[:-1])
    F_c = np.interp(centres, grid, F_ref)
    w_lo, w_hi = float(ref["window_lo"]), float(ref["window_hi"])          # v2 evaluation window (well-determined bins)
    thermal_bins = [k for k in range(nb) if w_lo <= centres[k] <= w_hi]
    grid_t = torch.as_tensor(grid, device=dev); Fp_t = torch.as_tensor(Fp_ref, device=dev)
    Fp_interp = lambda R: iv.interval_interp(Fp_t[None, :], grid_t, R[None, :])[0]      # noqa: E731
    gen = torch.Generator(device=dev).manual_seed(20260906)
    t_start = time.time()
    print(f"device {dev}; thermal bins {thermal_bins} (centres {np.round(centres[thermal_bins], 3).tolist()}); "
          f"OT domain lower edge (LJ rule, delta {CELL['thermal_delta']}) = {lj_forbidden_radius(p, CELL['thermal_delta']):.4f}", flush=True)

    # ---------------- 0. exact conditional samples ----------------
    t0 = time.time()
    ex, ident = exact_samples(p, ref, a.n_samples, 500_000, a.n_rep, dev, gen, thermal_bins, cv, force_fn, Fp_interp)
    print(f"exact samples: {a.n_samples} proposals in {time.time() - t0:.0f}s", flush=True)
    for k in thermal_bins:
        print(f"  bin {k:2d} R~{centres[k]:.3f}: raw {ex[k]['n_raw']:7d} ESS {ex[k]['ess']:9.0f}  <f_R>_ref {ident['f_mean_ref'][k]:+7.3f} "
              f"vs <F'_ref> {ident['Fp_mean_ref'][k]:+7.3f}  basins {np.round(ex[k]['basin_probs_exact'], 3).tolist()}", flush=True)

    # ---------------- 1. P1 stationarity at fixed R ----------------
    G = len(thermal_bins)
    q0 = torch.cat([ex[k]["q"] for k in thermal_bins]); R0 = torch.cat([ex[k]["R"] for k in thermal_bins])
    group = torch.arange(G, device=dev).repeat_interleave(a.n_rep)
    Fp_at = Fp_interp(R0)
    ph1, ph2 = torsions(q0)
    hist0 = hist2(ph1, ph2, group, G, n2, dphi).cpu().numpy()
    tv0 = [tv_hist(hist0[g], cond_dens[thermal_bins[g]], dphi) for g in range(G)]
    t0 = time.time()
    snap_every = max(a.n_eq // 8, 1)
    q_eq, _, _, _, snaps_eq = constrained_run(q0, R0, a.n_eq, p, gen, force_fn, cv, group, G, n2, dphi, snap_every=snap_every)
    q_eq, fbar, fsq, hist1, snaps_rec = constrained_run(q_eq, R0, a.n_rec, p, gen, force_fn, cv, group, G, n2, dphi,
                                                        record=True, snap_every=snap_every)
    hist1 = hist1.cpu().numpy()
    p1 = []
    for g, k in enumerate(thermal_bins):
        sel = group == g
        d = (fbar[sel] - Fp_at[sel])
        b_inf = float(d.mean()); se = float(d.std() / math.sqrt(int(sel.sum())))
        tv1 = tv_hist(hist1[g], cond_dens[k], dphi)
        bp0 = ex[k]["basin_probs_exact"]; bp1 = snaps_rec[-1][g]
        drift = float(0.5 * np.abs(bp1 - bp0).sum()); tv_ref_basin = float(0.5 * np.abs(bp1 - ref_basins[k]).sum())
        p1.append(dict(bin=k, R_centre=float(centres[k]), F_ref=float(F_c[k]), Fp_mean=float(Fp_at[sel].mean()),
                       b_inf=b_inf, se=se, f_std=float(torch.sqrt((fsq[sel] - fbar[sel] ** 2).clamp_min(0)).mean()),
                       tv0=tv0[g], tv_rec=tv1, basin_drift=drift, basin_tv_ref=tv_ref_basin,
                       basins_exact=bp0.tolist(), basins_end=bp1.tolist(), n_raw=ex[k]["n_raw"], ess_exact=ex[k]["ess"]))
        print(f"P1 bin {k:2d} R~{centres[k]:.3f}: b_inf {b_inf:+.3f} +- {se:.3f} (F' {float(Fp_at[sel].mean()):+.2f}); "
              f"TV(phi|R) exact-sample floor {tv0[g]:.3f} -> after {a.n_eq}+{a.n_rec} steps {tv1:.3f}; basin drift {drift:.3f} (vs ref {tv_ref_basin:.3f})", flush=True)
    print(f"P1 done {time.time() - t0:.0f}s", flush=True)

    # ---------------- 2. tau_perp: single-family starts at fixed R ----------------
    t0 = time.time()
    fam_groups = []
    for k in thermal_bins:
        feasible = [b for b in range(9) if ref_basins[k, b] >= 0.05]
        if len(feasible) < 2:
            continue
        Q = ex[k]["q"]; ph1, ph2 = torsions(Q); bi = basin_index(ph1, ph2)
        for b in feasible:
            sel = torch.nonzero(bi == b).flatten()
            if sel.numel() < 20:
                continue
            idx = sel[torch.randint(0, sel.numel(), (a.n_rep,), generator=gen, device=dev)]
            fam_groups.append((k, b, Q[idx].clone(), ex[k]["R"][idx].clone()))
    tau = []
    if fam_groups:
        qf = torch.cat([g[2] for g in fam_groups]); Rf = torch.cat([g[3] for g in fam_groups])
        grp = torch.arange(len(fam_groups), device=dev).repeat_interleave(a.n_rep)
        snap_tau = max(a.n_tau // 200, 1)
        _, _, _, _, snaps = constrained_run(qf, Rf, a.n_tau, p, gen, force_fn, cv, grp, len(fam_groups), n2, dphi, snap_every=snap_tau)
        t_ax = np.arange(snaps.shape[0]) * snap_tau
        for gi, (k, b, _, _) in enumerate(fam_groups):
            tv_t = 0.5 * np.abs(snaps[:, gi, :] - ref_basins[k][None, :]).sum(-1)
            surv = snaps[:, gi, b]
            hit = np.nonzero(tv_t <= 0.2)[0]
            k_esc = float(-math.log(max(surv[-1], 1e-6)) / max(t_ax[-1], 1)) if surv[-1] < 1 else 0.0
            tau.append(dict(bin=k, R_centre=float(centres[k]), family=BASINS[b], ref_prob=float(ref_basins[k, b]),
                            tau_perp_steps=(int(t_ax[hit[0]]) if hit.size else None), tv_start=float(tv_t[0]), tv_end=float(tv_t[-1]),
                            survival_end=float(surv[-1]), escape_rate_per_step=k_esc,
                            tau_escape_steps=(1.0 / k_esc if k_esc > 0 else float("inf"))))
            print(f"tau_perp bin {k:2d} R~{centres[k]:.3f} start {BASINS[b]:6s} (ref {ref_basins[k, b]:.2f}): TV {tv_t[0]:.2f} -> {tv_t[-1]:.2f} "
                  f"after {a.n_tau} steps; survival {surv[-1]:.3f}; tau_perp {tau[-1]['tau_perp_steps']}; tau_escape {tau[-1]['tau_escape_steps']:.0f} steps", flush=True)
        tau_arrays = dict(tau_t=t_ax, tau_snaps=snaps)
    else:
        tau_arrays = {}
    print(f"tau_perp done {time.time() - t0:.0f}s", flush=True)

    # ---------------- 3. P2 single event: lift + repair ----------------
    t0 = time.time()
    n_dr = len(DR_BINS)
    q_src = q_eq.repeat(n_dr, 1, 1)                                    # (n_dr * G * n_rep, 5, 3), dR-major
    R_src = R0.repeat(n_dr)
    dR = torch.as_tensor([d * DZ for d in DR_BINS], device=dev).repeat_interleave(G * a.n_rep)
    R_dst = (R_src + dR).clamp(CELL["R_lo"] + 1e-6, CELL["R_hi"] - 1e-6)
    ggrp = (torch.arange(n_dr, device=dev).repeat_interleave(G * a.n_rep) * G + group.repeat(n_dr))   # (dR, site) group
    NG = n_dr * G
    q_lift = lift_to_R(q_src, R_dst, cv.i, cv.j)
    audit_rows = []
    for gi in range(NG):
        sel = ggrp == gi
        audit_rows.append(lift_audit(p, q_src[sel], q_lift[sel]))
    Fp_dst = Fp_interp(R_dst)
    dest_bin = (torch.bucketize(R_dst, torch.as_tensor(edges, device=dev)) - 1).clamp(0, nb - 1)
    # conditional at destination bins: pooled per (group, dest bin)
    def cond_tv_by_group(q_state):
        ph1, ph2 = torsions(q_state)
        h = hist2(ph1, ph2, ggrp * nb + dest_bin, NG * nb, n2, dphi).cpu().numpy().reshape(NG, nb, n2, n2)
        out = np.full(NG, np.nan)
        for gi in range(NG):
            tvs, ws = [], []
            for k in range(nb):
                c = h[gi, k].sum()
                if c >= 50:
                    tvs.append(tv_hist(h[gi, k], cond_dens[k], dphi)); ws.append(c)
            if ws:
                out[gi] = float(np.average(tvs, weights=ws))
        return out
    tv_src = cond_tv_by_group(q_src)                                   # before the lift (floor at these R')
    noise_scale = math.sqrt(2.0 * CELL["dt"] / p.beta)
    M = a.m_repair
    b_m = np.zeros((M + 1, NG)); se_m = np.zeros((M + 1, NG))
    tv_m = {}
    q = q_lift
    with torch.no_grad():
        for m in range(M + 1):
            F = force_fn(q, p)
            f = fmean(q, F, cv, p.beta) - Fp_dst
            for gi in range(NG):
                sel = ggrp == gi
                b_m[m, gi] = float(f[sel].mean()); se_m[m, gi] = float(f[sel].std() / math.sqrt(int(sel.sum())))
            if m in (0, 5, 20, M):
                tv_m[m] = cond_tv_by_group(q)
            if m == M:
                break
            noise = torch.randn(q.shape, generator=gen, device=q.device, dtype=q.dtype)
            q = geom.remove_com(lift_to_R(q + CELL["dt"] * F + noise_scale * noise, R_dst, cv.i, cv.j))
        q, _, _, _, _ = constrained_run(q, R_dst, max(a.n_eq // 2, 10), p, gen, force_fn, cv, ggrp, NG, n2, dphi)
        q, fbar2, _, _, _ = constrained_run(q, R_dst, max(a.n_rec // 4, 10), p, gen, force_fn, cv, ggrp, NG, n2, dphi, record=True)
    tv_inf = cond_tv_by_group(q)
    p2 = []
    for gi in range(NG):
        di, g = divmod(gi, G); k = thermal_bins[g]; sel = ggrp == gi
        d = fbar2[sel] - Fp_dst[sel]
        b_inf = float(d.mean()); se_inf = float(d.std() / math.sqrt(int(sel.sum())))
        b0 = b_m[0, gi]
        frac = (b_m[:, gi] - b_inf) / (b0 - b_inf) if abs(b0 - b_inf) > 1e-9 else np.full(M + 1, np.nan)
        def first_below(th):
            ok = np.nonzero(np.abs(frac) <= th)[0]
            return int(ok[0]) if ok.size else None
        p2.append(dict(bin=k, R_centre=float(centres[k]), dR_bins=DR_BINS[di], dR=float(DR_BINS[di] * DZ), Fp_ref=float(Fp_dst[sel].mean()),
                       b0=float(b0), se0=float(se_m[0, gi]), b_inf=b_inf, se_inf=se_inf,
                       b_at={str(m): float(b_m[m, gi]) for m in (0, 1, 2, 3, 5, 10, 20, 40, M) if m <= M},
                       frac_at={str(m): float(frac[m]) for m in (1, 2, 3, 5, 10, 20, 40, M) if m <= M},
                       m_20pct=first_below(0.2), m_10pct=first_below(0.1),
                       tv_src=float(tv_src[gi]), tv_m={str(m): float(v[gi]) for m, v in tv_m.items()}, tv_inf=float(tv_inf[gi]),
                       **audit_rows[gi]))
    for k in thermal_bins:
        rows = [r for r in p2 if r["bin"] == k]
        print(f"P2 bin {k:2d} R~{centres[k]:.3f}: " + "  ".join(f"dR{r['dR_bins']:+g}b: b0 {r['b0']:+6.2f}->b5 {r['b_at']['5']:+5.2f} (inf {r['b_inf']:+5.2f})" for r in rows if abs(r["dR_bins"]) in (1, 2, 8)), flush=True)
    print(f"P2 done {time.time() - t0:.0f}s", flush=True)

    # ---------------- operator gate (frozen in the prereg) ----------------
    gate = dict(max_abs_b_inf=float(max(abs(r["b_inf"]) for r in p1)),
                max_basin_drift=float(max(r["basin_drift"] for r in p1)),
                max_tv_increase=float(max(r["tv_rec"] - r["tv0"] for r in p1)))
    gate["pass_meanforce"] = gate["max_abs_b_inf"] <= 1.0
    gate["pass_conditional"] = gate["max_basin_drift"] <= 0.05 and gate["max_tv_increase"] <= 0.05
    gate["hard_stop"] = gate["max_basin_drift"] > 0.10
    meta = dict(n_rep=a.n_rep, n_eq=a.n_eq, n_rec=a.n_rec, n_tau=a.n_tau, m_repair=a.m_repair, n_samples=a.n_samples, dz=DZ,
                dR_bins=list(DR_BINS), cell=CELL, thermal_bins=thermal_bins, reference=os.path.relpath(REF, ROOT),
                lj_domain_lo=lj_forbidden_radius(p, CELL["thermal_delta"]), wall_seconds=time.time() - t_start, quick=bool(a.quick),
                device=dev, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    json.dump(dict(meta=meta, identity=ident, p1=p1, tau_perp=tau, p2=p2, gate=gate), open(os.path.join(a.out, "p1p2.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(a.out, "p1p2.npz"), b_m=b_m, se_m=se_m, snaps_eq=snaps_eq, snaps_rec=snaps_rec, hist_p1=hist1, hist0=hist0, **tau_arrays)
    print(f"GATE: {gate}", flush=True)
    try:
        plot(p1, tau, p2, b_m, tau_arrays, meta, os.path.join(a.out, "figures"), thermal_bins, centres)
    except Exception as exc:                                              # plots must never lose the data
        print(f"plotting failed: {exc!r}")
    print(f"wrote {a.out} ({meta['wall_seconds'] / 60:.1f} min)")


def plot(p1, tau, p2, b_m, tau_arrays, meta, fig_dir, thermal_bins, centres):
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib.pyplot as plt
    G = len(thermal_bins)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), layout="constrained")
    ax = axes[0]
    ax.errorbar([r["R_centre"] for r in p1], [r["b_inf"] for r in p1], yerr=[2 * r["se"] for r in p1], fmt="o", ms=4, color="C0")
    ax.axhline(0, color="k", lw=0.7); ax.set_xlabel("R"); ax.set_ylabel("<f_R> - <F'_ref>  (fixed R)"); ax.set_title("P1: operator mean-force offset", fontsize=9)
    ax = axes[1]
    ax.plot([r["R_centre"] for r in p1], [r["tv0"] for r in p1], "s--", ms=4, label="exact samples (floor)")
    ax.plot([r["R_centre"] for r in p1], [r["tv_rec"] for r in p1], "o-", ms=4, label="after constrained run")
    ax.plot([r["R_centre"] for r in p1], [r["basin_drift"] for r in p1], "^:", ms=4, label="9-basin drift")
    ax.set_xlabel("R"); ax.set_ylabel("TV"); ax.set_title("P1: p(phi1,phi2 | R) vs reference", fontsize=9); ax.legend(fontsize=7, frameon=False)
    ax = axes[2]
    if tau and "tau_t" in tau_arrays:
        snaps = tau_arrays["tau_snaps"]; t_ax = tau_arrays["tau_t"]
        import numpy as _np
        for gi, r in enumerate(tau):
            ax.plot(t_ax, snaps[:, gi, BASINS.index(r["family"])], lw=1, label=f"R~{r['R_centre']:.2f} {r['family']}")
        ax.set_xlabel("constrained steps"); ax.set_ylabel("survival in start family"); ax.set_ylim(0, 1.02)
        ax.set_title("tau_perp: family survival at fixed R", fontsize=9); ax.legend(fontsize=5.5, frameon=False, ncol=2)
    fig.savefig(os.path.join(fig_dir, "p1_tau.png"), dpi=160); plt.close(fig)
    fig, axes = plt.subplots(2, (G + 1) // 2, figsize=(2.6 * ((G + 1) // 2), 5.2), layout="constrained")
    axes = np.atleast_1d(axes).ravel(); cmap = plt.get_cmap("coolwarm"); mx = max(abs(d) for d in DR_BINS)
    for g, k in enumerate(thermal_bins):
        ax = axes[g]
        for r in p2:
            if r["bin"] != k:
                continue
            gi = DR_BINS.index(r["dR_bins"]) * G + g
            ax.plot(np.arange(b_m.shape[0]), b_m[:, gi] - r["b_inf"], color=cmap(0.5 + 0.5 * r["dR_bins"] / mx), lw=1, label=f"{r['dR_bins']:+g} bins")
        ax.axhline(0, color="k", lw=0.7); ax.set_xscale("symlog", linthresh=5); ax.set_title(f"R~{centres[k]:.2f}", fontsize=8)
        ax.set_xlabel("repair steps m"); ax.set_ylabel("b(m) - b_inf")
        if g == 0:
            ax.legend(fontsize=5, frameon=False, ncol=2)
    for ax in axes[G:]:
        ax.axis("off")
    fig.suptitle("P2: injected mean-force bias after an OT lift and its decay under projected repair", fontsize=9)
    fig.savefig(os.path.join(fig_dir, "p2_repair_decay.png"), dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.0), layout="constrained")
    ax = axes[0]
    for g, k in enumerate(thermal_bins):
        rows = [r for r in p2 if r["bin"] == k]
        ax.plot([abs(r["dR"]) for r in rows if r["dR"] > 0], [abs(r["b0"] - r["b_inf"]) for r in rows if r["dR"] > 0], "^-", ms=3, color=plt.get_cmap("viridis")(g / max(G - 1, 1)), label=f"R~{centres[k]:.2f}")
        ax.plot([abs(r["dR"]) for r in rows if r["dR"] < 0], [abs(r["b0"] - r["b_inf"]) for r in rows if r["dR"] < 0], "v--", ms=3, color=plt.get_cmap("viridis")(g / max(G - 1, 1)))
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("|dR|"); ax.set_ylabel("|injected bias|"); ax.legend(fontsize=5.5, frameon=False); ax.set_title("injection vs move (up stretch, down compress)", fontsize=8)
    ax = axes[1]
    for r in p2:
        if abs(r["b0"] - r["b_inf"]) > 0.3:
            gi = DR_BINS.index(r["dR_bins"]) * G + thermal_bins.index(r["bin"])
            ax.plot(np.arange(b_m.shape[0]), (b_m[:, gi] - r["b_inf"]) / (r["b0"] - r["b_inf"]), color="gray", alpha=0.4, lw=0.7)
    ax.axhline(0.2, color="k", ls=":", lw=0.8); ax.axhline(0, color="k", lw=0.7); ax.set_xlim(0, 40); ax.set_ylim(-0.3, 1.1)
    ax.set_xlabel("repair steps m"); ax.set_ylabel("fraction of injected bias remaining"); ax.set_title("repair curves (|b0| > 0.3)", fontsize=8)
    fig.savefig(os.path.join(fig_dir, "p2_injection_fraction.png"), dpi=160); plt.close(fig)


if __name__ == "__main__":
    main()
