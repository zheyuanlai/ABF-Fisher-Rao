#!/usr/bin/env python
"""Information-conversion audit runner.

Frozen protocol: ``docs/INFORMATION_CONVERSION_AUDIT_PREREGISTRATION.md``.

Subcommands (run in this order):

  reference   Stage 0A: the validated long-run A2 difficulty reference for
              K2/K3 (V_j = sigma^2 tau, tau_j), and the horizon H per cell.
  pilot       Stage 0B-D + Stage 1 frontier on the pilot seeds.  Runs the
              oracle opportunity gate BEFORE any FR; if it stops, no FR runs.
  confirm     The selected dose + plain ABF on the confirmation seeds; saves
              end states for the optional continuation.
  continue    Long continuation of the saved confirmation states to T = 100.
              Only valid after a mechanism PASS; no second pulse exists here.

Dose selection lives in ``scripts/analyze_info_conversion.py`` and is
structurally blind to FEC outcomes.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))

from abffr import info_conversion as ic          # noqa: E402
from abffr import metrics, reference, simulation_torch, torch_utils as tu  # noqa: E402
from abffr.io_utils import RunSpec               # noqa: E402


# --------------------------------------------------------------------------- #
# Shared construction
# --------------------------------------------------------------------------- #
def load_cfg(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def build_reference(cfg):
    dom = cfg["domain"]
    x = np.linspace(dom["x_min"], dom["x_max"], int(dom["nx_profile"]))
    y = np.linspace(dom["y_min"], dom["y_max"], int(dom["ny_ref"]))
    ref = reference.compute_reference(
        x, y, beta=float(cfg["simulation"]["beta"]),
        x_tilt=float(cfg["potential"]["x_tilt"]))
    margin = float(cfg["allocation"]["eval_margin"])
    mask = (x >= dom["x_min"] + margin) & (x <= dom["x_max"] - margin)
    return x, ref, mask


def receipt(cfg, extra=None):
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    r = dict(commit=commit, torch=torch.__version__,
             cuda=torch.version.cuda, when=time.strftime("%Y-%m-%d %H:%M:%S"),
             config=cfg)
    if extra:
        r.update(extra)
    return r


def out_dir(cfg, *parts):
    d = os.path.join(cfg["output_root"], *parts)
    os.makedirs(d, exist_ok=True)
    return d


# --------------------------------------------------------------------------- #
# Stage 0A: reference difficulty (the validated long-run A2 machinery)
# --------------------------------------------------------------------------- #
def a2_reference_cfg(cfg, cell, n_steps, history_capacity):
    """Verbatim shape of validate_qr_gamma.base_cfg(arm='A2') long runs."""
    return {
        "simulation": {"beta": cfg["simulation"]["beta"],
                       "dt": cfg["simulation"]["dt"], "n_steps": int(n_steps),
                       "n_particles": cfg["simulation"]["n_particles"],
                       "eval_every": 500,
                       "x_init_mode": "uniform", "y_init_mode": "uniform"},
        "domain": dict(cfg["domain"]),
        "potential": {"x_tilt": cfg["potential"]["x_tilt"]},
        "kappa": {"cell": cell},
        "abf": {"estimator": "binned_smooth",
                "observation_order": "post_propagation",
                "h": cfg["abf"]["h"], "update_every": cfg["abf"]["update_every"],
                "min_count": cfg["abf"]["min_count"]},
        "fr": {"enabled": False, "noise_chunk_steps": 256},
        "qr": dict(enabled=True, arm="A2", n_cells=cfg["allocation"]["n_cells"],
                   opportunity_every=500, burnin_fraction=0.20,
                   stop_fraction=0.80, history_capacity=int(history_capacity)),
    }


def cmd_reference(cfg, device):
    x, ref, mask = build_reference(cfg)
    geom = ic.build_cells(x, mask, cfg["allocation"]["n_cells"],
                          ref["Fprime_ref"])
    ev = metrics.EvalConfig.from_domain(cfg["domain"])
    rcfg_meta = cfg["reference"]
    d = out_dir(cfg, "reference")
    dt = float(cfg["simulation"]["dt"])
    K = int(cfg["simulation"]["n_particles"])
    out = {}
    for cell in cfg["cells"]:
        rcfg = a2_reference_cfg(cfg, cell, rcfg_meta["n_steps"],
                                rcfg_meta["history_capacity"])
        specs = [RunSpec(method="abf_only", target_type="none", seed=int(s),
                         gamma=0.0, eta=0.10, burnin_fraction=0.0, fr_every=1,
                         stop_fraction=1.0) for s in rcfg_meta["seeds"]]
        t0 = time.time()
        res = simulation_torch.run_batch(
            specs, cfg=rcfg, x_grid=x, F_ref=ref["F_ref"],
            Fprime_ref=ref["Fprime_ref"], ev=ev, device=device,
            dtype=torch.float64, estimator="binned_smooth", base_seed=0)
        gam = np.array([dd["qr_gamma_final"] for dd in res.diags])
        tau = np.array([dd["qr_tau_final"] for dd in res.diags])
        V = np.nanmedian(gam, axis=0)
        tau_med = np.nanmedian(tau, axis=0)
        eval_cells = geom.a_cell > 0
        tau_eval = tau_med[eval_cells]
        if not np.isfinite(tau_eval).any():
            raise RuntimeError(f"{cell}: no finite reference tau on eval cells")
        tau_max = float(np.nanmax(tau_eval))
        H = int(np.ceil(tau_max / dt))
        out[cell] = dict(
            V=V.tolist(), tau=tau_med.tolist(),
            per_seed_gamma=gam.tolist(), per_seed_tau=tau.tolist(),
            a_cell=geom.a_cell.tolist(), f_ref_cell=geom.f_ref_cell.tolist(),
            tau_max_eval=tau_max, H=H, M=K * H,
            n_nan_tau_eval=int(np.sum(~np.isfinite(tau_eval))),
            runtime_s=time.time() - t0, seeds=rcfg_meta["seeds"],
            n_steps=rcfg_meta["n_steps"])
        print(f"{cell}: tau_max(eval)={tau_max:.3f}  H={H} steps "
              f"({H * dt:.1f} time)  M={K * H}  "
              f"V spread={np.nanmax(V[eval_cells]) / np.nanmin(V[eval_cells]):.1f}x "
              f"[{time.time() - t0:.0f}s]", flush=True)
    with open(os.path.join(d, "reference_difficulty.json"), "w") as fh:
        json.dump(dict(receipt=receipt(cfg), cells=out), fh, indent=1)
    print(f"reference written to {d}")


# --------------------------------------------------------------------------- #
# Stage 0B-D + Stage 1: checkpoint, target, gate, frontier
# --------------------------------------------------------------------------- #
def make_profile_cb(rows_out, x, ref, mask, dt, eval_every, seeds, arms, cell):
    def cb(st):
        F = st.F_hat.detach().cpu().numpy()
        Fp = st.Fprime_hat.detach().cpu().numpy()
        t = st.step * dt
        for b in range(F.shape[0]):
            rows_out.append(dict(
                cell=cell, seed=seeds[b], arm=arms[b], t=t,
                e_F=metrics.l2_error_F(F[b], ref["F_ref"], x, mask),
                e_Fprime=metrics.l2_error_Fprime(Fp[b], ref["Fprime_ref"], x,
                                                 mask)))
    return cb


def burnin_and_stage0(cfg, cell, seeds, ref_cell, device, profile_rows):
    """Plain ABF to the checkpoint; solve the target; report the gate inputs."""
    x, ref, mask = build_reference(cfg)
    geom = ic.build_cells(x, mask, cfg["allocation"]["n_cells"],
                          ref["Fprime_ref"])
    engine = ic.InfoConversionEngine(cfg, cell, x, ref["F_ref"],
                                     ref["Fprime_ref"], geom, device)
    state = engine.init_state(seeds)
    dt = float(cfg["simulation"]["dt"])
    cb = make_profile_cb(profile_rows, x, ref, mask, dt,
                         cfg["simulation"]["eval_every"], seeds,
                         ["abf"] * len(seeds), cell)
    t0 = time.time()
    engine.run(state, int(cfg["simulation"]["burnin_steps"]), profile_cb=cb,
               eval_every=int(cfg["simulation"]["eval_every"]))
    K = int(cfg["simulation"]["n_particles"])
    J = geom.n_cells
    V = np.asarray(ref_cell["V"], dtype=float)
    H, M = int(ref_cell["H"]), float(ref_cell["M"])
    av = geom.a_cell * V

    C = state.cell_cnt.detach().cpu().numpy()          # (S, J) raw deposits
    occ0 = np.stack([np.bincount(
        ic.cell_index_torch(state.X[b], engine.edges_t).cpu().numpy(),
        minlength=J) for b in range(len(seeds))]) / K

    stage0_rows, cell_rows, sols = [], [], []
    r_asym = np.sqrt(np.maximum(av, 0.0))
    r_asym = r_asym / r_asym.sum()
    live = av > 0
    R_asym_opt = float(np.sum(av[live] / r_asym[live]))
    R_asym_unif = float(np.sum(av[live] * J))
    for i, s in enumerate(seeds):
        sol = ic.solve_finite_horizon_target(av, C[i], M, K)
        R_opt = sol["risk"]
        R_unif = ic.predicted_finite_risk(av, C[i], M, np.full(J, 1.0 / J))
        G_ideal = 1.0 - R_opt / R_unif
        sols.append(sol)
        stage0_rows.append(dict(
            cell=cell, seed=s, H=H, M=M, R_opt=R_opt, R_unif=R_unif,
            G_ideal=G_ideal, lam=sol["lam"],
            n_floor_bound=int(sol["floor_bound"].sum()),
            R_asym_ratio=R_asym_opt / R_asym_unif,
            C_total=float(C[i].sum())))
        for j in range(J):
            cell_rows.append(dict(
                cell=cell, seed=s, j=j, C=C[i, j], a=geom.a_cell[j], V=V[j],
                pi_star=sol["pi"][j], occ0=occ0[i, j],
                f_ref=geom.f_ref_cell[j]))
    print(f"{cell}: burn-in {len(seeds)} seeds done in {time.time() - t0:.0f}s; "
          f"median G_ideal = {np.median([r['G_ideal'] for r in stage0_rows]):.4f}",
          flush=True)
    return dict(engine=engine, state=state, geom=geom, x=x, ref=ref, mask=mask,
                sols=sols, occ0=occ0, stage0_rows=stage0_rows,
                cell_rows=cell_rows, H=H, M=M, seeds=list(seeds))


def frontier(cfg, cell, ctx, doses, device, stage_dir, save_state=False):
    """Fork the checkpoint into plain ABF + one-pulse arms; run the cooldown."""
    engine, state, geom = ctx["engine"], ctx["state"], ctx["geom"]
    x, ref, mask = ctx["x"], ctx["ref"], ctx["mask"]
    seeds, sols, occ0 = ctx["seeds"], ctx["sols"], ctx["occ0"]
    H = ctx["H"]
    dt = float(cfg["simulation"]["dt"])
    K = int(cfg["simulation"]["n_particles"])
    J = geom.n_cells
    arms = ["abf"] + [f"p{d:g}" for d in doses]
    A = len(arms)
    S = len(seeds)

    st = engine.fork(state, A)
    arm_of_row = [arms[r % A] for r in range(S * A)]
    seed_of_row = [seeds[r // A] for r in range(S * A)]

    q_np = np.stack([ic.target_density_grid(sols[i]["pi"], geom, x)
                     for i in range(S)])
    q_rows = torch.as_tensor(np.repeat(q_np, A, axis=0), device=device,
                             dtype=engine.dtype)
    q_rows = q_rows / tu.trapezoid(q_rows, engine.dx).clamp_min(1e-300).unsqueeze(1)

    p90_by_row = {}
    gens = {}
    for r in range(S * A):
        a = r % A
        if a == 0:
            continue
        p90_by_row[r] = float(doses[a - 1])
        gens[r] = tu.make_generator(
            tu.stable_seed("fr-pulse", 0, seed_of_row[r]), device)

    cnt_at_fork = st.cell_cnt.clone()
    pulse_rows = engine.pulse(st, q_rows, p90_by_row, gens)

    # local genealogy in target-gaining cells + sibling pair trackers
    pair_trackers = {}
    pulse_out = []
    for pr in pulse_rows:
        r = pr["row"]
        i = r // A
        rec = {k: pr[k] for k in
               ("p90", "s90", "dtau", "n_events", "n_replacements",
                "degenerate", "kl_pre", "kl_post", "tv_pre", "tv_post",
                "ess_anc", "wmax_family")}
        rec.update(cell=cell, seed=seed_of_row[r], arm=arm_of_row[r])
        if not pr["degenerate"]:
            gaining = sols[i]["pi"] > occ0[i]
            anc = st.ancestors[r].detach().cpu().numpy()
            cells_now = ic.cell_index_torch(
                st.X[r], engine.edges_t).detach().cpu().numpy()
            rec.update(ic.local_ancestor_ess(anc, cells_now, gaining, K))
            ca, cb_ = ic.clone_pairs(pr["src"])
            if ca.size:
                pair_trackers[r] = ic.PairTracker(
                    clone_idx=torch.as_tensor(ca, device=device),
                    cont_idx=torch.as_tensor(cb_, device=device))
        pulse_out.append(rec)

    profile_rows = []
    cb = make_profile_cb(profile_rows, x, ref, mask, dt,
                         cfg["simulation"]["eval_every"], seed_of_row,
                         arm_of_row, cell)
    t0 = time.time()
    engine.run(st, H, profile_cb=cb,
               eval_every=int(cfg["simulation"]["eval_every"]),
               pair_trackers=pair_trackers, pulse_step=state.step)
    print(f"{cell}: frontier cooldown {H} steps x {S * A} rows in "
          f"{time.time() - t0:.0f}s", flush=True)

    # endpoint: realized information risk from the cumulative hard-cell estimator
    cnt = st.cell_cnt.detach().cpu().numpy()
    fsum = st.cell_sum.detach().cpu().numpy()
    n_fut = cnt - cnt_at_fork.detach().cpu().numpy()
    run_rows, cellrun_rows = [], []
    pulse_by = {(p["seed"], p["arm"]): p for p in pulse_out}
    for r in range(S * A):
        i = r // A
        s, a = seed_of_row[r], arm_of_row[r]
        pi = sols[i]["pi"]
        with np.errstate(invalid="ignore", divide="ignore"):
            fhat = np.where(cnt[r] > 0, fsum[r] / np.maximum(cnt[r], 1e-300),
                            np.nan)
        live = geom.a_cell > 0
        empty_live = int(np.sum(live & ~(cnt[r] > 0)))
        err = np.where(np.isfinite(fhat), fhat - geom.f_ref_cell, 0.0)
        R_s = float(np.sum(geom.a_cell[live] * err[live] ** 2))
        r_fut = n_fut[r] / max(n_fut[r].sum(), 1.0)
        tv_fut = 0.5 * float(np.abs(r_fut - pi).sum())
        row = dict(cell=cell, seed=s, arm=a, R_s=R_s,
                   tv_future=tv_fut, n_future_total=float(n_fut[r].sum()),
                   empty_live_cells=empty_live,
                   ess_anc_final=ic._anc_stats_row(st.ancestors[r], K)[0])
        row.update({k: pulse_by.get((s, a), {}).get(k, np.nan) for k in
                    ("p90", "s90", "dtau", "n_events", "n_replacements",
                     "kl_pre", "kl_post", "tv_pre", "tv_post", "ess_anc",
                     "wmax_family", "local_ess_min", "local_ess_median",
                     "n_gaining_occupied")})
        run_rows.append(row)
        for j in range(J):
            cellrun_rows.append(dict(
                cell=cell, seed=s, arm=a, j=j, n_future=n_fut[r, j],
                fhat=fhat[j], f_ref=geom.f_ref_cell[j], a=geom.a_cell[j],
                pi_star=pi[j]))

    sib_rows = []
    for r, trk in pair_trackers.items():
        for rec in trk.sums:
            sib_rows.append(dict(cell=cell, seed=seed_of_row[r],
                                 arm=arm_of_row[r], step_since=rec[0],
                                 n_pairs=rec[1], sum_fa=rec[2], sum_fb=rec[3],
                                 sum_fafb=rec[4], sum_fa2=rec[5],
                                 sum_fb2=rec[6]))

    pd.DataFrame(run_rows).to_csv(
        os.path.join(stage_dir, f"{cell}_runs.csv"), index=False)
    pd.DataFrame(cellrun_rows).to_csv(
        os.path.join(stage_dir, f"{cell}_cellruns.csv"), index=False)
    pd.DataFrame(profile_rows).to_csv(
        os.path.join(stage_dir, f"{cell}_cooldown_profiles.csv"), index=False)
    if sib_rows:
        pd.DataFrame(sib_rows).to_csv(
            os.path.join(stage_dir, f"{cell}_siblings.csv"), index=False)

    if save_state:
        torch.save(dict(
            X=st.X.cpu(), Y=st.Y.cpu(), C_acc=st.C_acc.cpu(),
            S_acc=st.S_acc.cpu(), cell_cnt=st.cell_cnt.cpu(),
            cell_sum=st.cell_sum.cpu(), ancestors=st.ancestors.cpu(),
            step=st.step, seeds=seed_of_row, arms=arm_of_row,
            n_pulses=st.n_pulses),
            os.path.join(stage_dir, f"{cell}_state.pt"))


def load_reference(cfg):
    p = os.path.join(cfg["output_root"], "reference",
                     "reference_difficulty.json")
    with open(p) as fh:
        return json.load(fh)["cells"]


def cmd_stage(cfg, device, stage: str):
    """pilot or confirm."""
    ref_cells = load_reference(cfg)
    if stage == "pilot":
        seeds = [int(s) for s in cfg["seeds"]["pilot"]]
        doses = [float(d) for d in cfg["pulse"]["doses_p90"]]
        save_state = False
    else:
        s0 = int(cfg["seeds"]["confirmation_start"])
        seeds = list(range(s0, s0 + int(cfg["seeds"]["confirmation_count"])))
        vp = os.path.join(cfg["output_root"], "pilot", "pilot_verdict.json")
        with open(vp) as fh:
            verdict = json.load(fh)
        if verdict.get("selected_p90") is None:
            raise SystemExit(f"no selected dose in {vp}; confirmation refused")
        doses = [float(verdict["selected_p90"])]
        save_state = True
    stage_dir = out_dir(cfg, stage)

    burnin_profiles = []
    ctx = {}
    for cell in cfg["cells"]:
        ctx[cell] = burnin_and_stage0(cfg, cell, seeds, ref_cells[cell],
                                      device, burnin_profiles)
        pd.DataFrame(ctx[cell]["stage0_rows"]).to_csv(
            os.path.join(stage_dir, f"{cell}_stage0.csv"), index=False)
        pd.DataFrame(ctx[cell]["cell_rows"]).to_csv(
            os.path.join(stage_dir, f"{cell}_stage0_cells.csv"), index=False)

    # ---- Stage 0D gate: BEFORE any FR --------------------------------------
    medians = {cell: float(np.median([r["G_ideal"]
                                      for r in ctx[cell]["stage0_rows"]]))
               for cell in cfg["cells"]}
    gate_min = float(cfg["gate_0d"]["g_ideal_min"])
    with open(os.path.join(stage_dir, "stage0_gate.json"), "w") as fh:
        json.dump(dict(median_G_ideal=medians, g_ideal_min=gate_min,
                       receipt=receipt(cfg)), fh, indent=1)
    if stage == "pilot" and all(m < gate_min for m in medians.values()):
        verdict = dict(verdict="NO_FINITE_HORIZON_ALLOCATION_OPPORTUNITY",
                       median_G_ideal=medians, g_ideal_min=gate_min,
                       selected_p90=None, receipt=receipt(cfg))
        with open(os.path.join(stage_dir, "pilot_verdict.json"), "w") as fh:
            json.dump(verdict, fh, indent=1)
        print("STOP: NO_FINITE_HORIZON_ALLOCATION_OPPORTUNITY "
              f"(median G_ideal {medians}); no FR was run.")
        pd.DataFrame(burnin_profiles).to_csv(
            os.path.join(stage_dir, "burnin_profiles.csv"), index=False)
        return
    print(f"Stage 0D gate: median G_ideal {medians} vs {gate_min} -> continue",
          flush=True)

    for cell in cfg["cells"]:
        frontier(cfg, cell, ctx[cell], doses, device, stage_dir,
                 save_state=save_state)
    pd.DataFrame(burnin_profiles).to_csv(
        os.path.join(stage_dir, "burnin_profiles.csv"), index=False)
    with open(os.path.join(stage_dir, "receipt.json"), "w") as fh:
        json.dump(receipt(cfg, dict(stage=stage, seeds=seeds, doses=doses)),
                  fh, indent=1)
    print(f"{stage} runs complete -> {stage_dir}")


# --------------------------------------------------------------------------- #
# Long continuation (secondary; only after mechanism PASS)
# --------------------------------------------------------------------------- #
def cmd_continue(cfg, device):
    vp = os.path.join(cfg["output_root"], "confirm", "confirm_verdict.json")
    with open(vp) as fh:
        verdict = json.load(fh)
    if not verdict.get("mechanism_pass_both_cells", False):
        raise SystemExit("continuation refused: no mechanism PASS in "
                         f"{vp}")
    x = None
    for cell in cfg["cells"]:
        sp = os.path.join(cfg["output_root"], "confirm", f"{cell}_state.pt")
        blob = torch.load(sp, weights_only=False)
        x, ref, mask = build_reference(cfg)
        geom = ic.build_cells(x, mask, cfg["allocation"]["n_cells"],
                              ref["Fprime_ref"])
        engine = ic.InfoConversionEngine(cfg, cell, x, ref["F_ref"],
                                         ref["Fprime_ref"], geom, device)
        seeds, arms = list(blob["seeds"]), list(blob["arms"])
        st = ic.EngineState(
            X=blob["X"].to(device), Y=blob["Y"].to(device),
            C_acc=blob["C_acc"].to(device), S_acc=blob["S_acc"].to(device),
            Fprime_hat=torch.zeros_like(blob["C_acc"]).to(device),
            F_hat=torch.zeros_like(blob["C_acc"]).to(device),
            ancestors=blob["ancestors"].to(device),
            cell_cnt=blob["cell_cnt"].to(device),
            cell_sum=blob["cell_sum"].to(device),
            step=int(blob["step"]), seeds=seeds,
            n_pulses=np.asarray(blob["n_pulses"]))
        engine.recompute_grid(st)
        engine.noise = ic.ChunkKeyedNoise(
            seeds, engine.N, device, engine.dtype, engine.base_seed,
            engine.chunk_steps)
        remaining = int(cfg["simulation"]["full_horizon_steps"]) - st.step
        if remaining <= 0:
            raise SystemExit(f"{cell}: state already past the full horizon")
        profile_rows = []
        dt = float(cfg["simulation"]["dt"])
        cb = make_profile_cb(profile_rows, x, ref, mask, dt,
                             cfg["simulation"]["eval_every"], seeds, arms,
                             cell)
        t0 = time.time()
        engine.run(st, remaining, profile_cb=cb,
                   eval_every=int(cfg["simulation"]["eval_every"]))
        print(f"{cell}: continuation {remaining} steps x {len(seeds)} rows in "
              f"{time.time() - t0:.0f}s", flush=True)
        d = out_dir(cfg, "continuation")
        pd.DataFrame(profile_rows).to_csv(
            os.path.join(d, f"{cell}_profiles.csv"), index=False)
    with open(os.path.join(cfg["output_root"], "continuation",
                           "receipt.json"), "w") as fh:
        json.dump(receipt(cfg, dict(stage="continuation")), fh, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command",
                    choices=["reference", "pilot", "confirm", "continue"])
    ap.add_argument("--config", default="configs/information_conversion/frozen.yaml")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    device = torch.device(args.device or
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.set_grad_enabled(False)
    if args.command == "reference":
        cmd_reference(cfg, device)
    elif args.command in ("pilot", "confirm"):
        cmd_stage(cfg, device, args.command)
    else:
        cmd_continue(cfg, device)


if __name__ == "__main__":
    main()
