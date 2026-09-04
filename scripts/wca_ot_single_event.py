#!/usr/bin/env python
"""WCA OT + repair, stage M1: single-event lift-and-repair on conditionally equilibrated fibres.

The mechanism the OT+repair design rests on, measured directly and cheaply, with no
sampler, no ABF estimator and no free-energy error:

  1. take the pooled final configurations of the W0-A plain-ABF runs, pick the replicas
     nearest a source site z_src, project the dimer to z_src exactly and equilibrate the
     fibre with the reference-consistent PROJECTED constrained scheme (all particles move,
     dimer re-projected to z each step; the TI reference's own operator);
  2. LIFT: move the dimer to z_dst = z_src + dz with ``project_dimer_to_z`` (midpoint and
     direction kept, bath untouched) -- exactly the OT lift the sampler would apply;
  3. measure the local mean force immediately (m = 0) and then after every projected
     repair step m = 1..M at fixed z_dst, against the TI reference F'_ref(z_dst):

         b(m) = < f_loc(q_m) | z_dst > - F'_ref(z_dst)

     b(0) is the conditional error the lift INJECTS (what an unrepaired OT arm would
     deposit into ABF); b(m) / b(0) is the fraction REMAINING after m repair steps.
     The stationary value b_inf (a long projected run at z_dst) is the operator's own
     offset from the reference and is subtracted before the fraction is formed.

Also recorded per lift (safety audit): potential-energy jump, maximum particle force after
the lift, fraction of replicas with a bath overlap (any pair below params.min_r * sigma).

    CUDA_VISIBLE_DEVICES=3 python -u scripts/wca_ot_single_event.py [--quick]
Outputs: results/ot_repair_campaign/wca/M1/{single_event.json, single_event.npz, figures/}
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import wca_abffr_core as core      # noqa: E402
import wca_phase_jobs as jobs      # noqa: E402
from run_wca_targeted_relax import CAMPAIGN as TR_CAMPAIGN, REFERENCE_NPZ, base_config, make_spec  # noqa: E402

OUT = os.path.join(ROOT, "results", "ot_repair_campaign", "wca", "M1")
SITES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
DZ = (-0.2, -0.1, -0.05, -0.02, -0.01, 0.01, 0.02, 0.05, 0.1, 0.2)
Z_LO, Z_HI = -0.15, 1.15          # destinations must stay inside the walled domain
N_REP = 1024
N_EQ, N_REC = 3000, 1000          # fibre equilibration (>> the 0.3-0.7 t.u. ACF tail); stationary read-out
M_REPAIR = 80                     # repair steps recorded after the lift (tau_f ~ 3-6 steps)


def fmean_sample(q, f_raw, params, sim):
    """The estimator's own local-mean-force sample (clipped force, clipped sample)."""
    f_phys = core.clip_forces(f_raw, params.force_clip)
    fl = core.local_mean_force(q, f_phys if sim.use_clipped_force_for_mean_force else f_raw, params)
    return torch.clamp(fl, -sim.mean_force_sample_clip, sim.mean_force_sample_clip)


@torch.inference_mode()
def projected_run(engine, params, sim, q, z_fixed, n_steps, gen, record=False):
    """The TI reference's constrained scheme for ``n_steps`` on every replica.  If ``record``,
    the estimator's f_loc sample at the state BEFORE each step is returned as (n_steps, B):
    row 0 is the state handed in (for a lifted state, the m = 0 injected sample)."""
    noise_scale = math.sqrt(2.0 * sim.dt / params.beta)
    rec = []
    for _ in range(n_steps):
        f_raw = engine.force(q, compute_energy=False)
        if record:
            rec.append(fmean_sample(q, f_raw, params, sim).to(torch.float64))
        f_phys = core.clip_forces(f_raw, params.force_clip)
        noise = torch.randn(q.shape, device=q.device, dtype=q.dtype, generator=gen)
        q = core.project_dimer_to_z(core.wrap_positions(q + sim.dt * f_phys + noise_scale * noise, params.box_length),
                                    z_fixed, params)
    return q, (torch.stack(rec, 0) if record else None)


@torch.inference_mode()
def lift_audit(engine, params, q_before, q_after):
    """Energy jump, max particle force after the lift, overlap fraction."""
    V0, _ = engine.force(q_before, compute_energy=True)
    V1, f1 = engine.force(q_after, compute_energy=True)
    fmax = torch.linalg.norm(f1, dim=-1).amax(dim=1)
    # overlap: any pair (excluding the bonded dimer pair) closer than min_r * sigma
    qi = q_after.index_select(1, engine.pair_i)
    qj = q_after.index_select(1, engine.pair_j)
    r = torch.linalg.norm(core.minimum_image(qi - qj, engine.L), dim=-1)
    overlap = (r < engine.min_r * engine.sigma).any(dim=1)
    return dict(dV_mean=float((V1 - V0).mean()), dV_max=float((V1 - V0).max()),
                fmax_median=float(fmax.median()), fmax_max=float(fmax.max()),
                overlap_frac=float(overlap.to(torch.float64).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="2 sites x 4 dz, short runs (smoke)")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    sites, dzs, n_rep, n_eq, n_rec, m_rep = SITES, DZ, N_REP, N_EQ, N_REC, M_REPAIR
    if a.quick:
        sites, dzs, n_rep, n_eq, n_rec, m_rep = (0.2, 0.8), (-0.05, -0.01, 0.01, 0.05), 32, 400, 100, 20
    os.makedirs(os.path.join(a.out, "figures"), exist_ok=True)

    base = base_config()
    spec = make_spec("M1", "abf", "abf", 0)
    params = jobs.build_params(spec)
    sim = jobs.build_sim(spec, base)
    engine = core.WCADimerEngine(params, core.DEVICE, core.DTYPE)
    ref = np.load(REFERENCE_NPZ, allow_pickle=True)
    ref_grid, ref_mf = np.asarray(ref["grid"], float), np.asarray(ref["mean_force"], float)
    Fp_ref = lambda z: float(np.interp(z, ref_grid, ref_mf))            # noqa: E731

    files = sorted(glob.glob(os.path.join(TR_CAMPAIGN, "W0", "raw", "W0A__abf__*.npz")))
    assert len(files) >= 8, f"need the W0-A pools, found {len(files)}"
    Q = torch.as_tensor(np.concatenate([np.asarray(jobs.load_run(f)["final_q"], np.float32) for f in files]),
                        device=engine.device, dtype=engine.dtype)
    Zq = core.reaction_coordinate(Q, params)
    print(f"pool: {Q.shape[0]} configurations from {len(files)} W0-A runs; z in [{float(Zq.min()):.3f}, {float(Zq.max()):.3f}]; "
          f"dt {sim.dt}, beta {params.beta}, force_clip {params.force_clip}, device {engine.device}", flush=True)

    results, arrays = [], {}
    t_start = time.time()
    for si, z_src in enumerate(sites):
        gen = torch.Generator(device=engine.device); gen.manual_seed(20260904 + 1000 * si)
        order = torch.argsort((Zq - float(z_src)).abs())[:n_rep]
        q0 = core.project_dimer_to_z(Q.index_select(0, order), torch.full((n_rep,), float(z_src), device=engine.device, dtype=engine.dtype), params)
        z_src_t = torch.full((n_rep,), float(z_src), device=engine.device, dtype=engine.dtype)
        t0 = time.time()
        q_eq, _ = projected_run(engine, params, sim, q0, z_src_t, n_eq, gen)
        q_eq, f_src = projected_run(engine, params, sim, q_eq, z_src_t, n_rec, gen, record=True)
        rep_means = f_src.mean(0)                                        # per-replica time mean (B,)
        b_src, se_src = float(rep_means.mean()) - Fp_ref(z_src), float(rep_means.std() / math.sqrt(n_rep))
        print(f"site z={z_src:+.2f}: equilibrated {n_rep} replicas ({n_eq}+{n_rec} steps, {time.time() - t0:.0f}s); "
              f"stationary bias vs reference {b_src:+.3f} +- {se_src:.3f}  (F'_ref {Fp_ref(z_src):+.2f})", flush=True)
        for dz in dzs:
            z_dst = z_src + dz
            if not (Z_LO <= z_dst <= Z_HI):
                continue
            z_dst_t = torch.full((n_rep,), float(z_dst), device=engine.device, dtype=engine.dtype)
            q_lift = core.project_dimer_to_z(q_eq, z_dst_t, params)
            audit = lift_audit(engine, params, q_eq, q_lift)
            resid = float((core.reaction_coordinate(q_lift, params) - z_dst_t).abs().max())
            t0 = time.time()
            q_rep, f_rep = projected_run(engine, params, sim, q_lift, z_dst_t, m_rep + 1, gen, record=True)   # rows 0..M
            q_rep, _ = projected_run(engine, params, sim, q_rep, z_dst_t, n_eq, gen)
            _, f_inf = projected_run(engine, params, sim, q_rep, z_dst_t, n_rec, gen, record=True)
            fp = Fp_ref(z_dst)
            b_m = (f_rep.mean(1) - fp).cpu().numpy()                     # (M+1,) bias after m repair steps
            se_m = (f_rep.std(1) / math.sqrt(n_rep)).cpu().numpy()
            inf_means = f_inf.mean(0)
            b_inf, se_inf = float(inf_means.mean()) - fp, float(inf_means.std() / math.sqrt(n_rep))
            b0 = float(b_m[0])
            frac = (b_m - b_inf) / (b0 - b_inf) if abs(b0 - b_inf) > 1e-9 else np.full_like(b_m, np.nan)
            def first_below(th):
                ok = np.nonzero(np.abs(frac) <= th)[0]
                return int(ok[0]) if ok.size else None
            rec = dict(z_src=float(z_src), dz=float(dz), z_dst=float(z_dst), Fp_ref=fp, b_src=b_src, se_src=se_src,
                       b0=b0, se0=float(se_m[0]), b_inf=b_inf, se_inf=se_inf,
                       m_20pct=first_below(0.20), m_10pct=first_below(0.10), m_05pct=first_below(0.05),
                       b_at=dict((str(m), float(b_m[m])) for m in (0, 1, 2, 3, 5, 10, 20, 40, m_rep) if m <= m_rep),
                       lift_residual=resid, **audit, wall_s=time.time() - t0)
            results.append(rec)
            key = f"s{si}_dz{dz:+.2f}"
            arrays[f"b_{key}"] = b_m.astype(np.float64); arrays[f"se_{key}"] = se_m.astype(np.float64)
            print(f"   dz={dz:+.2f} -> z={z_dst:+.2f}: injected b0 {b0:+7.2f} (ref F' {fp:+6.2f}) -> b(5) {b_m[min(5, m_rep)]:+6.2f} "
                  f"b(20) {b_m[min(20, m_rep)]:+6.2f} b_inf {b_inf:+5.2f}+-{se_inf:.2f}; 20%/10% at m={rec['m_20pct']}/{rec['m_10pct']}; "
                  f"dV {audit['dV_mean']:+.1f} fmax {audit['fmax_median']:.0f} overlap {audit['overlap_frac']:.2f} ({rec['wall_s']:.0f}s)", flush=True)

    meta = dict(sites=list(sites), dz=list(dzs), n_rep=n_rep, n_eq=n_eq, n_rec=n_rec, m_repair=m_rep, dt=sim.dt,
                beta=params.beta, scheme="projected (TI-reference scheme)", reference=os.path.relpath(REFERENCE_NPZ, ROOT),
                pool=[os.path.relpath(f, ROOT) for f in files], wall_seconds=time.time() - t_start, quick=bool(a.quick))
    json.dump(dict(meta=meta, results=results), open(os.path.join(a.out, "single_event.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(a.out, "single_event.npz"), **arrays)
    plot(results, arrays, meta, os.path.join(a.out, "figures"))
    print(f"wrote {a.out} ({meta['wall_seconds'] / 60:.1f} min)")


def plot(results, arrays, meta, fig_dir):
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib.pyplot as plt
    sites = meta["sites"]; dzs = meta["dz"]
    fig, axes = plt.subplots(2, max(1, (len(sites) + 1) // 2), figsize=(3.2 * max(1, (len(sites) + 1) // 2), 5.6), layout="constrained")
    axes = np.atleast_1d(axes).ravel()
    cmap = plt.get_cmap("coolwarm")
    for si, z_src in enumerate(sites):
        ax = axes[si]
        for r in results:
            if r["z_src"] != z_src:
                continue
            key = f"s{si}_dz{r['dz']:+.2f}"
            b = arrays[f"b_{key}"]
            c = cmap(0.5 + 0.5 * r["dz"] / max(abs(d) for d in dzs))
            ax.plot(np.arange(len(b)), b - r["b_inf"], color=c, lw=1.2, label=f"dz {r['dz']:+.2f}")
        ax.axhline(0, color="k", lw=0.7)
        ax.set_title(f"z_src = {z_src:+.1f}", fontsize=9); ax.set_xlabel("repair steps m"); ax.set_ylabel("b(m) - b_inf")
        ax.set_xscale("symlog", linthresh=5)
        if si == 0:
            ax.legend(fontsize=6, frameon=False, ncol=2)
    for ax in axes[len(sites):]:
        ax.axis("off")
    fig.suptitle("WCA single OT event: injected mean-force bias and its decay under projected repair", fontsize=9.5)
    fig.savefig(os.path.join(fig_dir, "repair_decay.png"), dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.0), layout="constrained")
    ax = axes[0]
    for si, z_src in enumerate(sites):
        pts = [(abs(r["dz"]), abs(r["b0"] - r["b_inf"]), r["dz"] > 0) for r in results if r["z_src"] == z_src]
        for sign, mk in ((True, "^"), (False, "v")):
            xs = [p[0] for p in pts if p[2] == sign]; ys = [p[1] for p in pts if p[2] == sign]
            ax.plot(xs, ys, mk, ms=4, color=plt.get_cmap("viridis")(si / max(len(sites) - 1, 1)), label=f"z_src {z_src:+.1f}" if sign else None)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("|dz|"); ax.set_ylabel("|injected bias b0 - b_inf|")
    ax.set_title("injection vs move size (up = stretch, down = compress)", fontsize=8.5); ax.legend(fontsize=6, frameon=False)
    ax = axes[1]
    for r in results:
        key = f"s{sites.index(r['z_src'])}_dz{r['dz']:+.2f}"
        b = arrays[f"b_{key}"]
        if abs(r["b0"] - r["b_inf"]) > 0.5:
            ax.plot(np.arange(len(b)), (b - r["b_inf"]) / (r["b0"] - r["b_inf"]), color="gray", alpha=0.5, lw=0.8)
    ax.axhline(0.2, color="k", ls=":", lw=0.8); ax.axhline(0, color="k", lw=0.7)
    ax.set_xlim(0, 40); ax.set_ylim(-0.3, 1.1); ax.set_xlabel("repair steps m"); ax.set_ylabel("fraction of injected bias remaining")
    ax.set_title("repair curves, all (site, dz) with |b0| > 0.5", fontsize=8.5)
    fig.savefig(os.path.join(fig_dir, "injection_and_fraction.png"), dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
