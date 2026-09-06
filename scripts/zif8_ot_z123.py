#!/usr/bin/env python
"""Ethane / flexible ZIF-8: OT lift + constrained gate repair, stages Z1 / Z2 / Z3
(docs/ZIF8_OT_REPAIR.md).  No ABF, no FR, no free-energy error -- the operator and the mechanism
against the accepted umbrella/WHAM reference (F(xi) and p_ref(A_gate | xi sub-bin), |xi| < 1 A).

  Z1  fixed-xi constrained BAOAB at six sites chosen mechanically from the reference F(xi):
      cage minimum, left half-height, left band sub-bin, window plane (two central sub-bins),
      peak sub-bin, right half-height.  Each replica keeps its own xi' drawn uniformly in the
      site interval (so band sites compare like-for-like with the reference sub-bin).  Init:
      pool configurations nearest in circular xi, lattice-shifted, PULLED to the site over
      n_pull steps, equilibrated n_eq, recorded n_rec.  Outputs: b_inf = <f_xi> - <F'_ref(xi')>,
      first/second-half drift, A_gate / theta_gate statistics, TV to the reference gate
      conditional (band sites), integrated autocorrelation time of A_gate at fixed xi.
  Z2  single OT event: lift by dxi in +-{1/2, 1, 2, 4, 8} grid bins from the equilibrated
      ensembles; record b(m) and <A_gate>(m) for m = 0..M repair steps, then the stationary
      values at xi' (b_inf, A_inf) after a long tail; lift audit (dU, max guest force, min
      host-guest distance); coarse-bin D_gate(m) vs the stationary law at xi'.
  Z3  tau_gate(xi): from the Z1 autocorrelation and from the Z2 decay of <A_gate>(m) - A_inf and
      b(m) - b_inf (time to 20 % / 10 % remaining, exponential fit over the first e-fold).

    CUDA_VISIBLE_DEVICES=1 python -u scripts/zif8_ot_z123.py [--quick] [--deterministic]
-> results/ot_repair_campaign/zif8/Z123/{z123.json, z123.npz, figures/}
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
PREREG = os.path.join(ROOT, "configs/uniform_campaign/zif8_prereg.json")
REF = os.path.join(ROOT, "results/uniform_campaign/zif8/reference/reference_T300.npz")
POOL = os.path.join(ROOT, "cache/zif8/init_pool_T300.npz")
OUT = os.path.join(ROOT, "results/ot_repair_campaign/zif8/Z123")
DR_BINS = (-8, -4, -2, -1, -0.5, 0.5, 1, 2, 4, 8)
COARSE_EDGES = np.arange(2.2, 4.6 + 1e-9, 0.1)          # 24 bins of 0.1 A for D_gate(m) (256-sample floor ~0.15)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--deterministic", action="store_true", help="keep the sampler's determinism flags (1.75x slower; not needed here)")
    ap.add_argument("--n-rep", type=int, default=256)
    ap.add_argument("--n-pull", type=int, default=4000)
    ap.add_argument("--n-eq", type=int, default=40000)
    ap.add_argument("--n-rec", type=int, default=40000)
    ap.add_argument("--m-repair", type=int, default=400)
    ap.add_argument("--n-tail", type=int, default=3600)
    ap.add_argument("--n-rec2", type=int, default=2000)
    ap.add_argument("--gate-stride", type=int, default=5)
    ap.add_argument("--acf-window", type=int, default=8000)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    if a.quick:
        a.n_rep, a.n_pull, a.n_eq, a.n_rec, a.m_repair, a.n_tail, a.n_rec2, a.acf_window = 16, 200, 400, 400, 40, 100, 100, 200
    os.makedirs(os.path.join(a.out, "figures"), exist_ok=True)
    if not a.deterministic:                                  # pairing is irrelevant for a mechanism study
        torch.use_deterministic_algorithms(False)
        try:
            torch._inductor.config.deterministic = False
        except Exception:
            pass
    from zif8.core_zif8 import ZIF8SimConfig, ZIF8System, engine_kwargs
    from zif8.ot_repair_zif8 import (ConstrainedBAOAB, lift_guest, local_mean_force_xi, reference_mean_force,
                                     gate_pdf, tv, integrated_autocorr)
    pre = json.load(open(PREREG))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    system = ZIF8System(300.0, dev, root=ROOT, **engine_kwargs(pre))
    s_cfg = {k: v for k, v in pre["sampler"].items() if not k.startswith("_")}
    sim = ZIF8SimConfig(**s_cfg)
    clip_A = sim.abf_force_clip_A * 8
    ref = np.load(REF, allow_pickle=True)
    xi_g, Fp_g, Fp_interp = reference_mean_force(ref)
    F_ref = np.asarray(ref["F"]); L = float(ref["period"]); dbin = L / sim.n_grid
    gate_edges = np.asarray(ref["gate_edges"]); gh_xi = np.asarray(ref["gate_hist_window_xi"]); gx = np.asarray(ref["gate_xi_edges"])
    kT = float(ref["kT"])
    gen = torch.Generator(device=dev).manual_seed(20260906)
    t_start = time.time()

    # ---------------- sites (mechanical, from the reference F) ----------------
    i_min, i_max = int(np.nanargmin(F_ref)), int(np.nanargmax(F_ref))
    half = F_ref[i_min] + 0.5 * (F_ref[i_max] - F_ref[i_min])
    def crossing(side):
        idx = [i for i in range(len(xi_g) - 1) if (xi_g[i] < 0) == (side == "left") and (F_ref[i] - half) * (F_ref[i + 1] - half) < 0]
        i = idx[0] if side == "left" else idx[-1]
        t = (half - F_ref[i]) / (F_ref[i + 1] - F_ref[i]); return float(xi_g[i] + t * (xi_g[i + 1] - xi_g[i]))
    xl, xr = crossing("left"), crossing("right")
    sites = [dict(name="cage_min", lo=float(xi_g[i_min]) - dbin / 2, hi=float(xi_g[i_min]) + dbin / 2, ref_bins=None),
             dict(name="left_half", lo=xl - dbin / 2, hi=xl + dbin / 2, ref_bins=None),
             dict(name="left_band", lo=float(gx[2]), hi=float(gx[3]), ref_bins=[2]),
             dict(name="window_plane", lo=float(gx[3]), hi=float(gx[5]), ref_bins=[3, 4]),
             dict(name="peak_band", lo=float(gx[7]), hi=float(gx[8]), ref_bins=[7]),
             dict(name="right_half", lo=xr - dbin / 2, hi=xr + dbin / 2, ref_bins=None)]
    for st in sites:
        st["centre"] = 0.5 * (st["lo"] + st["hi"]); st["F_ref"] = float(np.interp(st["centre"], xi_g, F_ref)); st["Fp_ref"] = float(Fp_interp(st["centre"]))
    S, n = len(sites), a.n_rep
    print(f"device {dev}; L {L:.4f} A, bin {dbin:.4f} A, barrier {F_ref[i_max] - F_ref[i_min]:.2f} kJ/mol; sites: " +
          ", ".join(f"{st['name']} [{st['lo']:+.3f},{st['hi']:+.3f}] F' {st['Fp_ref']:+.1f}" for st in sites), flush=True)

    # ---------------- initial configurations: nearest pool configs (circular), lattice-shifted ----------------
    pool = torch.as_tensor(np.load(POOL)["q"], device=dev, dtype=system.dtype)
    xi_pool = system.xi_value(pool)
    q0, xi_tgt, group = [], [], []
    for si, st in enumerate(sites):
        tgt = st["lo"] + (st["hi"] - st["lo"]) * torch.rand(n, generator=gen, device=dev, dtype=system.dtype)
        d = xi_pool[None, :] - tgt[:, None]; d = d - L * torch.round(d / L)              # circular distance, per replica
        order = torch.argsort(d.abs(), dim=1)[:, :8]                                      # 8 nearest per replica, then one at random
        pick = order[torch.arange(n, device=dev), torch.randint(0, 8, (n,), generator=gen, device=dev)]
        qq = pool[pick].clone()
        shift = torch.round((tgt - xi_pool[pick]) / L) * L                              # lattice translation along n (exact symmetry)
        qq[:, system.n_frame:] += (shift[:, None] * system.normal[None, :])[:, None, :]
        q0.append(qq); xi_tgt.append(tgt); group.append(torch.full((n,), si, device=dev, dtype=torch.long))
    q = torch.cat(q0); xi_tgt = torch.cat(xi_tgt); group = torch.cat(group); B = q.shape[0]
    xi_start = system.xi_value(q)
    v = system.pin_frame_com(system.maxwell_velocities((B,), gen))
    dyn = ConstrainedBAOAB(system, sim, gen)
    print(f"pull distances per site (A): " + ", ".join(f"{float((xi_tgt - xi_start)[group == si].abs().max()):.2f}" for si in range(S)), flush=True)

    # ---------------- Z1: pull -> equilibrate -> record ----------------
    t0 = time.time()
    sched = lambda k: xi_start + (xi_tgt - xi_start) * (k + 1) / a.n_pull              # noqa: E731
    q, v, F = dyn.run(q, v, xi_tgt, a.n_pull, xi_schedule=sched)
    assert float((system.xi_value(q) - xi_tgt).abs().max()) < 1e-8
    q, v, F = dyn.run(q, v, xi_tgt, a.n_eq, F=F)
    print(f"Z1 pull+eq done {time.time() - t0:.0f}s", flush=True)
    fsum = torch.zeros(B, device=dev, dtype=system.dtype); fsq = torch.zeros_like(fsum); fsum_h = [torch.zeros_like(fsum), torch.zeros_like(fsum)]
    gsum = torch.zeros_like(fsum); gsq = torch.zeros_like(fsum); tsum = torch.zeros_like(fsum); ng = [0]; nf = [0]
    ghist = np.zeros((2, S, len(gate_edges) - 1)); acf_series = []
    Fp_at = torch.as_tensor(Fp_interp(xi_tgt.cpu().numpy()), device=dev, dtype=system.dtype)
    half_step = a.n_rec // 2

    def rec1(k, qq, vv, FF):
        f = local_mean_force_xi(system, qq, FF, clip_A)
        fsum.add_(f); fsq.add_(f * f); fsum_h[0 if k < half_step else 1].add_(f); nf[0] += 1
        if k % a.gate_stride == 0:
            ag, th = system.gate_observables(qq)
            gsum.add_(ag); gsq.add_(ag * ag); tsum.add_(th); ng[0] += 1
            agn = ag.cpu().numpy(); hsel = 0 if k < half_step else 1
            for si in range(S):
                h, _ = np.histogram(agn[(group == si).cpu().numpy()], bins=gate_edges); ghist[hsel, si] += h
        if k >= a.n_rec - a.acf_window and k % 2 == 0:
            acf_series.append(system.gate_observables(qq)[0].cpu().numpy() if k % a.gate_stride else ag.cpu().numpy())
    t0 = time.time()
    q, v, F = dyn.run(q, v, xi_tgt, a.n_rec, F=F, record=rec1)
    print(f"Z1 record done {time.time() - t0:.0f}s", flush=True)
    fbar = (fsum / nf[0]); fvar = (fsq / nf[0] - fbar ** 2).clamp_min(0)
    fh = [(fsum_h[0] / half_step), (fsum_h[1] / (a.n_rec - half_step))]
    abar = gsum / ng[0]; asd = (gsq / ng[0] - abar ** 2).clamp_min(0).sqrt(); tbar = tsum / ng[0]
    acf_arr = np.stack(acf_series) if acf_series else None
    z1 = []
    for si, st in enumerate(sites):
        sel = group == si; d = fbar[sel] - Fp_at[sel]
        b_inf = float(d.mean()); se = float(d.std() / math.sqrt(n))
        drift_f = float((fh[1][sel] - fh[0][sel]).mean())
        p_rec = (ghist[0, si] + ghist[1, si]); p_rec = p_rec / max(p_rec.sum(), 1)
        p_a = ghist[0, si] / max(ghist[0, si].sum(), 1); p_b = ghist[1, si] / max(ghist[1, si].sum(), 1)
        tv_half = tv(p_a, p_b)
        tv_ref = float("nan")
        if st["ref_bins"] is not None:
            pr = gh_xi[st["ref_bins"]].sum(0); pr = pr / pr.sum(); tv_ref = tv(p_rec, pr)
            st["A_ref_mean"] = float(np.sum(pr * 0.5 * (gate_edges[1:] + gate_edges[:-1])))
        tau_A, _ = (integrated_autocorr(acf_arr[:, sel.cpu().numpy()], 2 * sim.dt) if acf_arr is not None else (float("nan"), None))
        z1.append(dict(site=st["name"], lo=st["lo"], hi=st["hi"], centre=st["centre"], F_ref=st["F_ref"], Fp_ref_mean=float(Fp_at[sel].mean()),
                       f_mean=float(fbar[sel].mean()), b_inf=b_inf, se=se, f_sd=float(fvar[sel].sqrt().mean()), drift_halves=drift_f,
                       A_mean=float(abar[sel].mean()), A_sd_within=float(asd[sel].mean()), A_sd_total=float(np.sqrt(np.sum(p_rec * (0.5 * (gate_edges[1:] + gate_edges[:-1])) ** 2) - np.sum(p_rec * 0.5 * (gate_edges[1:] + gate_edges[:-1])) ** 2)),
                       theta_mean=float(tbar[sel].mean()), tv_halves=tv_half, tv_ref=tv_ref, A_ref_mean=st.get("A_ref_mean", float("nan")),
                       tau_A_ps=tau_A, tau_A_steps=(tau_A / sim.dt if np.isfinite(tau_A) else float("nan")), n_gate_samples=int(ghist[:, si].sum())))
        print(f"Z1 {st['name']:13s} xi~{st['centre']:+.3f}: <f> {z1[-1]['f_mean']:+7.2f} vs F'_ref {z1[-1]['Fp_ref_mean']:+7.2f} -> b_inf {b_inf:+.3f} +- {se:.3f} (halves drift {drift_f:+.3f}); "
              f"A_gate {z1[-1]['A_mean']:.4f} +- {z1[-1]['A_sd_total']:.4f}" + (f" (ref {z1[-1]['A_ref_mean']:.4f}, TV {tv_ref:.3f})" if st["ref_bins"] else "") +
              f"; TV halves {tv_half:.3f}; theta {z1[-1]['theta_mean']:.1f}; tau_A {1000 * tau_A:.1f} fs = {z1[-1]['tau_A_steps']:.0f} steps", flush=True)

    # ---------------- Z2: single OT event ----------------
    t0 = time.time()
    n_dr = len(DR_BINS)
    q_src = q.repeat(n_dr, 1, 1); v_src = v.repeat(n_dr, 1, 1); F_src = F.repeat(n_dr, 1, 1)
    xi_src = xi_tgt.repeat(n_dr); grp = group.repeat(n_dr) + S * torch.arange(n_dr, device=dev).repeat_interleave(B)
    dxi = torch.as_tensor([d * dbin for d in DR_BINS], device=dev, dtype=system.dtype).repeat_interleave(B)
    xi_dst = xi_src + dxi
    NG = n_dr * S
    q_lift = lift_guest(system, q_src, xi_dst)
    # lift audit
    U0 = system.potential_energy(q_src); U1 = system.potential_energy(q_lift); F1 = system.forces(q_lift)
    fg = F1[:, system.n_frame:].norm(dim=-1).amax(1)
    dv = q_lift[:, system.n_frame:, None, :] - q_lift[:, None, :system.n_frame, :]
    dv = dv - system.box * torch.round(dv / system.box); dmin = dv.norm(dim=-1).amin(dim=(1, 2))
    Fp_dst = torch.as_tensor(Fp_interp(xi_dst.cpu().numpy()), device=dev, dtype=system.dtype)
    M = a.m_repair
    b_m = np.zeros((M + 1, NG)); se_m = np.zeros((M + 1, NG)); A_m = np.zeros((M + 1, NG)); Asd_m = np.zeros((M + 1, NG))
    gcoarse = {}
    gidx = [(grp == g).cpu().numpy() for g in range(NG)]

    def rec2(k, qq, vv, FF):
        f = (local_mean_force_xi(system, qq, FF, clip_A) - Fp_dst).cpu().numpy()
        ag = system.gate_observables(qq)[0].cpu().numpy()
        for g in range(NG):
            fg_ = f[gidx[g]]; b_m[k, g] = fg_.mean(); se_m[k, g] = fg_.std() / math.sqrt(n)
            A_m[k, g] = ag[gidx[g]].mean(); Asd_m[k, g] = ag[gidx[g]].std()
        if k in (0, 5, 20, 100, M):
            gcoarse[k] = np.stack([gate_pdf(ag[gidx[g]], COARSE_EDGES)[0] for g in range(NG)])
    qq, vv, FF = q_lift, v_src, F1
    for k in range(M + 1):
        rec2(k, qq, vv, FF)
        if k == M:
            break
        qq, vv, FF = dyn.step(qq, vv, FF, xi_dst)
    print(f"Z2 repair curves done {time.time() - t0:.0f}s", flush=True)
    # stationary law at xi'
    qq, vv, FF = dyn.run(qq, vv, xi_dst, a.n_tail, F=FF)
    fs2 = torch.zeros(NG * 0 + qq.shape[0], device=dev, dtype=system.dtype); as2 = torch.zeros_like(fs2); as2q = torch.zeros_like(fs2); n2 = [0]
    gpool = [[] for _ in range(NG)]

    def rec3(k, q_, v_, F_):
        fs2.add_(local_mean_force_xi(system, q_, F_, clip_A) - Fp_dst); n2[0] += 1
        if k % a.gate_stride == 0:
            ag = system.gate_observables(q_)[0]; as2.add_(ag); as2q.add_(ag * ag)
            agn = ag.cpu().numpy()
            for g in range(NG):
                gpool[g].append(agn[gidx[g]])
    qq, vv, FF = dyn.run(qq, vv, xi_dst, a.n_rec2, F=FF, record=rec3)
    ngs = max(a.n_rec2 // a.gate_stride, 1)
    z2 = []
    for g in range(NG):
        di, si = divmod(g, S); st = sites[si]; sel = gidx[g]
        d_inf = (fs2 / n2[0])[torch.as_tensor(sel, device=dev)]
        b_inf = float(d_inf.mean()); se_inf = float(d_inf.std() / math.sqrt(n))
        A_inf = float((as2 / ngs)[torch.as_tensor(sel, device=dev)].mean()); A_inf_sd = float(np.concatenate(gpool[g]).std())
        p_inf = gate_pdf(np.concatenate(gpool[g]), COARSE_EDGES)[0]
        p_floor = gate_pdf(gpool[g][-1], COARSE_EDGES)[0]                       # one 256-sample snapshot from the stationary law
        b0 = b_m[0, g]; A0 = A_m[0, g]
        fr_b = (b_m[:, g] - b_inf) / (b0 - b_inf) if abs(b0 - b_inf) > 1e-9 else np.full(M + 1, np.nan)
        fr_A = (A_m[:, g] - A_inf) / (A0 - A_inf) if abs(A0 - A_inf) > 1e-9 else np.full(M + 1, np.nan)
        def first_below(fr, th):
            ok = np.nonzero(np.abs(fr) <= th)[0]; return int(ok[0]) if ok.size else None
        def efold(fr):
            ok = np.nonzero(fr <= math.exp(-1))[0]; return int(ok[0]) if ok.size else None
        z2.append(dict(site=st["name"], centre=st["centre"], dxi_bins=DR_BINS[di], dxi=float(DR_BINS[di] * dbin), xi_dst_mean=float(xi_dst[torch.as_tensor(sel, device=dev)].mean()),
                       Fp_ref_dst=float(Fp_dst[torch.as_tensor(sel, device=dev)].mean()), b0=float(b0), se0=float(se_m[0, g]), b_inf=b_inf, se_inf=se_inf,
                       b_at={str(m): float(b_m[m, g]) for m in (0, 1, 2, 5, 10, 20, 50, 100, 200, M) if m <= M},
                       frac_b_at={str(m): float(fr_b[m]) for m in (5, 10, 20, 50, 100, 200, M) if m <= M},
                       m_b_20pct=first_below(fr_b, 0.2), m_b_10pct=first_below(fr_b, 0.1), m_b_efold=efold(fr_b),
                       A0=float(A0), A_inf=A_inf, A_inf_sd=A_inf_sd, A_shift0_sd=float((A0 - A_inf) / max(A_inf_sd, 1e-9)),
                       frac_A_at={str(m): float(fr_A[m]) for m in (5, 10, 20, 50, 100, 200, M) if m <= M},
                       m_A_20pct=first_below(fr_A, 0.2), m_A_efold=efold(fr_A),
                       D_gate_at={str(m): tv(gcoarse[m][g], p_inf) for m in gcoarse}, D_gate_floor=tv(p_floor, p_inf),
                       dU_mean=float((U1 - U0)[torch.as_tensor(sel, device=dev)].mean()), dU_max=float((U1 - U0)[torch.as_tensor(sel, device=dev)].max()),
                       fguest_max_median=float(fg[torch.as_tensor(sel, device=dev)].median()), dmin_hostguest_min=float(dmin[torch.as_tensor(sel, device=dev)].min())))
    for si, st in enumerate(sites):
        rows = [r for r in z2 if r["site"] == st["name"]]
        print(f"Z2 {st['name']:13s}: " + "  ".join(f"d{r['dxi_bins']:+g}b: b0 {r['b0']:+6.2f}->b5 {r['b_at']['5']:+5.2f}->b{M} {r['b_at'][str(M)]:+5.2f} (inf {r['b_inf']:+5.2f}) dA0 {r['A_shift0_sd']:+.2f}sd tauA20 {r['m_A_20pct']}"
                                                  for r in rows if abs(r["dxi_bins"]) in (1, 2, 8)), flush=True)
    print(f"Z2 done {time.time() - t0:.0f}s", flush=True)

    # ---------------- Z3 summary + gate ----------------
    z3 = dict(tau_A_fixed_xi_steps={r["site"]: r["tau_A_steps"] for r in z1},
              tau_A_relax_20pct_steps_2bin={r["site"] + f"_{r['dxi_bins']:+g}": r["m_A_20pct"] for r in z2 if abs(r["dxi_bins"]) == 2},
              tau_b_relax_20pct_steps_2bin={r["site"] + f"_{r['dxi_bins']:+g}": r["m_b_20pct"] for r in z2 if abs(r["dxi_bins"]) == 2})
    slopes = [abs(r["b0"] - r["b_inf"]) / abs(r["dxi"]) for r in z2 if abs(r["dxi_bins"]) in (1, 2, 4)]
    gate = dict(max_abs_b_inf=float(max(abs(r["b_inf"]) for r in z1)), max_b_inf_over_se=float(max(abs(r["b_inf"]) / max(r["se"], 1e-9) for r in z1)),
                max_A_drift_halves=float(max(abs(r["drift_halves"]) for r in z1)),
                max_tv_ref_minus_halves=float(np.nanmax([r["tv_ref"] - r["tv_halves"] for r in z1 if np.isfinite(r["tv_ref"])])),
                median_injection_slope_kJ_per_mol_A2=float(np.median(slopes)))
    gate["pass_meanforce"] = gate["max_abs_b_inf"] <= 1.0 or gate["max_b_inf_over_se"] <= 3.0
    gate["pass_conditional"] = gate["max_tv_ref_minus_halves"] <= 0.03
    meta = dict(n_rep=n, n_pull=a.n_pull, n_eq=a.n_eq, n_rec=a.n_rec, m_repair=M, n_tail=a.n_tail, n_rec2=a.n_rec2, gate_stride=a.gate_stride, dt_ps=sim.dt,
                bin_A=dbin, dR_bins=list(DR_BINS), sites=sites, reference=os.path.relpath(REF, ROOT), pool=os.path.relpath(POOL, ROOT),
                deterministic=a.deterministic, quick=bool(a.quick), device=str(dev), cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                wall_seconds=time.time() - t_start, kT=kT)
    json.dump(dict(meta=meta, z1=z1, z2=z2, z3=z3, gate=gate), open(os.path.join(a.out, "z123.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(a.out, "z123.npz"), b_m=b_m, se_m=se_m, A_m=A_m, Asd_m=Asd_m, ghist_z1=ghist, gate_edges=gate_edges,
                        coarse_edges=COARSE_EDGES, **{f"gcoarse_{k}": v for k, v in gcoarse.items()}, acf_series=(acf_arr if acf_arr is not None else np.zeros(0)))
    print(f"GATE: {gate}", flush=True)
    try:
        plot(z1, z2, b_m, A_m, meta, sites, os.path.join(a.out, "figures"))
    except Exception as exc:
        print(f"plotting failed: {exc!r}")
    print(f"wrote {a.out} ({meta['wall_seconds'] / 60:.1f} min)")


def plot(z1, z2, b_m, A_m, meta, sites, fig_dir):
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib.pyplot as plt
    S = len(sites); n_dr = len(meta["dR_bins"]); M = b_m.shape[0] - 1; dt_fs = 1000 * meta["dt_ps"]
    fig, axes = plt.subplots(2, S, figsize=(2.7 * S, 5.6), layout="constrained")
    cmap = plt.get_cmap("coolwarm"); mx = max(abs(d) for d in meta["dR_bins"])
    for si, st in enumerate(sites):
        for r in z2:
            if r["site"] != st["name"]:
                continue
            g = meta["dR_bins"].index(r["dxi_bins"]) * S + si; c = cmap(0.5 + 0.5 * r["dxi_bins"] / mx)
            axes[0, si].plot(np.arange(M + 1) * dt_fs, b_m[:, g] - r["b_inf"], color=c, lw=1, label=f"{r['dxi_bins']:+g} bins")
            axes[1, si].plot(np.arange(M + 1) * dt_fs, (A_m[:, g] - r["A_inf"]) / max(r["A_inf_sd"], 1e-9), color=c, lw=1)
        for ax in axes[:, si]:
            ax.axhline(0, color="k", lw=0.7); ax.set_xscale("symlog", linthresh=10); ax.set_xlabel("repair time (fs)")
        axes[0, si].set_title(f"{st['name']}  xi~{st['centre']:+.2f}", fontsize=8.5); axes[0, si].set_ylabel("b(m) - b_inf  (kJ/mol/A)"); axes[1, si].set_ylabel("(<A_gate>(m) - A_inf)/sd")
        if si == 0:
            axes[0, si].legend(fontsize=5.5, frameon=False, ncol=2)
    fig.suptitle("ZIF-8 single OT event: injected mean-force bias (top) and gate-aperture lag (bottom) under constrained BAOAB repair", fontsize=9.5)
    fig.savefig(os.path.join(fig_dir, "z2_repair_decay.png"), dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), layout="constrained")
    ax = axes[0]; ax.errorbar([r["centre"] for r in z1], [r["b_inf"] for r in z1], yerr=[2 * r["se"] for r in z1], fmt="o", ms=4)
    ax.axhline(0, color="k", lw=0.7); ax.set_xlabel("xi (A)"); ax.set_ylabel("<f_xi> - F'_ref  (kJ/mol/A)"); ax.set_title("Z1: operator mean-force offset", fontsize=9)
    ax = axes[1]; ax.plot([r["centre"] for r in z1], [r["A_mean"] for r in z1], "o-", ms=4, label="constrained ensemble")
    rr = [r for r in z1 if np.isfinite(r["A_ref_mean"])]; ax.plot([r["centre"] for r in rr], [r["A_ref_mean"] for r in rr], "s", ms=5, label="umbrella reference")
    ax.set_xlabel("xi (A)"); ax.set_ylabel("<A_gate> (A)"); ax.set_title("Z1: gate aperture vs guest position", fontsize=9); ax.legend(fontsize=7, frameon=False)
    ax = axes[2]
    for si, st in enumerate(sites):
        pts = [(abs(r["dxi"]), abs(r["b0"] - r["b_inf"])) for r in z2 if r["site"] == st["name"] and r["dxi_bins"] > 0]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "^-", ms=3, color=plt.get_cmap("viridis")(si / max(S - 1, 1)), label=st["name"])
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("|dxi| (A)"); ax.set_ylabel("|injected bias| (kJ/mol/A)"); ax.set_title("Z2: injection vs move (stretch)", fontsize=9); ax.legend(fontsize=6, frameon=False)
    fig.savefig(os.path.join(fig_dir, "z1_z2_summary.png"), dpi=160); plt.close(fig)


if __name__ == "__main__":
    main()
