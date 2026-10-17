"""DEVELOPMENT ONLY (R2, docs/r2_campaign/DEVELOPMENT.md): replay candidate occupancy-triggered stop rules on EXISTING
trajectories.  No new simulation.  The controller code (src/eri/fr_controller.OnlineStop) is fed the saved
cumulative deposit accumulators at the save times, with check_every = the file's save interval (0.1-0.5 t.u.;
the production grid is 0.25), so a replayed t_est is the save-grid value and the production value is the first
0.25-grid check at or after it.

Development set (choices are made on these only): every ITE FR arm and the ERI K1024 FR arms.
Read-only check (no choice is made on them): LTA C3 / C4 FR arms; ERI K256 is NOT loaded (held out).
ABF arms are replayed too, for the ABF-arm establishment time under the same statistic (context only).

    ABFFR_DATA_ROOT=/home/zheyuanlai/ABF-Fisher-Rao python scripts/r2_campaign/dev_replay_r2.py
-> results/r2_campaign/dev/replay_r2.json, replay_r2_runs.csv
"""
import glob
import itertools
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DATA = os.environ.get("ABFFR_DATA_ROOT", ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "convergence_fr"))
from eri.fr_controller import R2, OnlineStop, window_mask  # noqa: E402
import atlas_lib as L  # noqa: E402

TWO_PI = 2.0 * np.pi
LTA_A = 11.919

# block -> (engine, directory, role, note)
BLOCKS = {
    # ---- development: ITE ----
    "C1_ite300_K1024_R1": ("eri", "results/adaptive_fr_campaign/C1_ite300_K1024_R1", "dev ITE 300 K K1024 (R1 stop)", ""),
    "C2_ite150_K256_R1": ("eri", "results/adaptive_fr_campaign/C2_ite150_K256_R1", "dev ITE 150 K K256 (R1 stop)", ""),
    "EA_ite300_K256_T200_R1": ("eri", "results/adaptive_fr_campaign/EA_ite300_K256_T200_R1", "dev ITE 300 K K256 T200 (R1 stop; saves 0.5)", "fr-only"),
    "A_ite300_K256_T200_stop10": ("eri", "results/adaptive_fr_campaign/A_ite300_K256_T200_stop10", "dev ITE 300 K K256 T200 (stop 10; saves 0.5)", "fr-only"),
    "E1_ite300_stop10": ("eri", "results/convergence_fr_campaign/E1_ite300_stop10", "dev ITE 300 K K1024 (stop 10)", ""),
    "E2_ite150_stop15": ("eri", "results/convergence_fr_campaign/E2_ite150_stop15", "dev ITE 150 K K256 (stop 15)", ""),
    "E1x_ite300_ladder": ("eri", "results/convergence_fr_campaign/E1x_ite300_ladder", "dev ITE 300 K K1024 (stop 5 / 20 ladder)", "fr-glob:fr_stop*_N1024_s*.npz"),
    "X1_ite300_K1024": ("eri", "results/overnight/ite/prod_confirm", "dev ITE 300 K K1024 (continuous FR)", ""),
    "X2_ite150_K256": ("eri", "results/overnight/ite150/prod_confirm", "dev ITE 150 K K256 (continuous FR; 360 bins; saves 0.5)", ""),
    "M300_256_rate1": ("eri", "results/convergence_fr_campaign/M300_256_rate1", "dev ITE 300 K K256 (continuous FR, rate 1)", ""),
    "M150_256_rate1": ("eri", "results/convergence_fr_campaign/M150_256_rate1", "dev ITE 150 K K256 (continuous FR, rate 1)", ""),
    "dev_monitor": ("dev", "results/adaptive_fr_campaign/dev", "dev ITE monitor runs (continuous FR; 0.25 grid exact)", ""),
    # ---- development: ERI K1024 only ----
    "C5_eri600_K1024_R1": ("eri", "results/adaptive_fr_campaign/C5_eri600_K1024_R1", "dev ERI 600 K K1024 (R1 stop)", ""),
    "ERI_K1024": ("eri", "results/convergence_fr_campaign/ERI_K1024", "dev ERI 600 K K1024 (continuous FR)", ""),
    # ---- read-only check: LTA ----
    "C3_lta150_K1024_R1": ("lta", "results/adaptive_fr_campaign/C3_lta150_K1024_R1", "CHECK LTA 150 K K1024 (R1 stop)", ""),
    "C4_lta300_K1024_R1": ("lta", "results/adaptive_fr_campaign/C4_lta300_K1024_R1", "CHECK LTA 300 K K1024 (R1 stop)", ""),
}
DEV_SYSTEMS = {"ITE", "ERI"}

LEVELS = (0.5, 0.6, 0.7, 0.75, 0.8, 0.9)
FACTORS = (1.5, 2.0, 2.5, 3.0)
GUARDS = {"none": dict(support_count=0.0, stability_gate=False),
          "support": dict(stability_gate=False),                 # = the frozen R2 (stability logged, not gating)
          "support+stability": dict(stability_gate=True)}        # the brief's primary candidate as first stated
FROZEN = "W1_L0.75_support_p1"
PERSIST = (1, 2, 4)
WINDOWS = (0.5, 1.0)


def _files(rel_dir, pattern):
    found = {}
    for base in (DATA, ROOT):
        for f in glob.glob(os.path.join(base, rel_dir, pattern)):
            if not f.endswith((".ckpt.npz", ".tmp.npz")):
                found[os.path.basename(f)] = f
    return [found[k] for k in sorted(found)]


def load_eri(path):
    z = np.load(path, allow_pickle=False)
    c = json.loads(str(z["cfg"]))
    stop = int(z["fr_stop_effective_steps"]) if "fr_stop_effective_steps" in z.files else int(c.get("fr_stop_steps", -1))
    return dict(system="ERI" if c["variant"] == "gard_tait" else "ITE", T_K=float(c["T_K"]), K=int(c["N"]),
                t=np.asarray(z["times"], float), C=np.asarray(z["sv_C"], float), M=np.asarray(z["sv_M"], float),
                win=z["sv_i"][:, 3] / float(c["N"]), essw=np.asarray(z["sv_u"][:, 2], float),
                maxw=np.asarray(z["sv_u"][:, 3], float), mask=window_mask(int(c["n_grid"]), float(c["window_phi"])),
                share_inst=2.0 * float(c["window_phi"]) / TWO_PI, t_start=float(c["fr_start_steps"]) * float(c["dt"]),
                t_ramp=float(c["warmup_steps"]) * float(c["dt"]), stop_t=(stop * float(c["dt"]) if stop >= 0 else None),
                method=str(z["method"]), seed=int(c["seed"]), rate=float(c["fr_rate"]))


def load_lta(path):
    import lta_ladder_numba as LE
    r = LE.load_result(path)
    c = json.loads(str(r["cfg_json"]))
    a = float(json.loads(str(r["meta_json"])).get("a_pseudo", LTA_A))
    stop = int(r["fr_stop_effective_steps"]) if "fr_stop_effective_steps" in r else int(c.get("fr_stop_steps", -1))
    ns = len(r["save_t"])
    return dict(system="LTA", T_K=float(c["T_K"]), K=int(c["N"]), t=np.asarray(r["save_t"], float),
                C=np.asarray(r["C_all"], float), M=np.asarray(r["M_all"], float),
                win=np.asarray(r["region_frac"], float)[:, 2],
                essw=np.asarray(r.get("gen_ess_win", np.ones(ns)), float),
                maxw=np.asarray(r.get("gen_max_frac_win", np.zeros(ns)), float),
                mask=window_mask(int(c["n_grid"]), float(c["window_half"]) * TWO_PI / a),
                share_inst=2.0 * float(c["window_half"]) / a, t_start=float(c["fr_start_steps"]) * float(c["h"]),
                t_ramp=float(c["warmup_steps"]) * float(c["h"]), stop_t=(stop * float(c["h"]) if stop >= 0 else None),
                method=("fr" if int(np.asarray(r.get("cum_deaths", [0])).ravel()[-1]) > 0 or "fr_controller_t1" in r else "abf"),
                seed=int(c["seed"]), rate=float(c["fr_rate"]))


def load_dev(path):
    z = np.load(path, allow_pickle=False)
    c = json.loads(str(z["cfg"]))
    occ = np.asarray(z["occ"], float)
    mask = window_mask(int(c["n_grid"]), float(c["window_phi"]))
    return dict(system="ITE", T_K=float(c["T_K"]), K=int(c["N"]), t=np.asarray(z["t"], float),
                C=np.asarray(z["C"], float), M=np.asarray(z["M"], float), win=occ[:, mask].sum(1) / occ.sum(1),
                essw=np.asarray(z["essw"], float), maxw=np.asarray(z["maxfw"], float), mask=mask,
                share_inst=float(mask.mean()), t_start=float(c["fr_start_steps"]) * float(c["dt"]),
                t_ramp=float(c["warmup_steps"]) * float(c["dt"]), stop_t=None, method="fr", seed=int(c["seed"]),
                rate=float(c["fr_rate"]))


def replay(run, spec):
    """Feed the real controller at the save times (check_every = save interval).  Returns the controller."""
    t = run["t"]
    ds = float(np.median(np.diff(t)))
    sp = dict(spec, check_every=ds)
    if abs(round(sp["W"] / ds) * ds - sp["W"]) > 1e-9:
        return None
    ctl = OnlineStop(sp, run["K"], run["t_start"], window_mask=run["mask"])
    zero = np.zeros(run["C"].shape[1])
    for i, ti in enumerate(t):
        if ti < run["t_start"] - 1e-9:
            continue
        if abs((ti - run["t_start"]) / ds - round((ti - run["t_start"]) / ds)) > 1e-6:
            continue
        ts = ctl.observe(ti, zero, zero, run["C"][i], run["M"][i], run["essw"][i], run["maxw"][i])
        if ts is not None:
            break
    return ctl


def main():
    out_dir = os.path.join(ROOT, "results", "r2_campaign", "dev")
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    per_block = {}
    for block, (engine, rel, role, note) in BLOCKS.items():
        if engine == "dev":
            files = _files(rel, "dev_*.npz")
            runs = [load_dev(f) for f in files]
        else:
            pat = note.split("fr-glob:")[1] if note.startswith("fr-glob:") else "*_N*_s*.npz"
            files = [f for f in _files(rel, pat) if not os.path.basename(f).startswith("frcal")]
            runs = [(load_eri if engine == "eri" else load_lta)(f) for f in files]
            for f, r in zip(files, runs):          # the arm is the file-name prefix (abf_ / fr_ / fr_stop*)
                r["method"] = "abf" if os.path.basename(f).startswith("abf") else "fr"
        if not runs:
            print("MISSING", block, rel)
            continue
        per_block[block] = dict(role=role, engine=engine, n_runs=len(runs), files=[os.path.basename(f) for f in files],
                                system=runs[0]["system"], T_K=runs[0]["T_K"], K=runs[0]["K"],
                                save_interval=float(np.median(np.diff(runs[0]["t"]))), t_ramp=runs[0]["t_ramp"],
                                t_start=runs[0]["t_start"], share_mask=float(runs[0]["mask"].mean()),
                                share_inst=runs[0]["share_inst"], candidates={})
        # offline establishment (instantaneous window fraction at saves, 1 t.u. trailing mean, level 0.75)
        for arm in ("abf", "fr"):
            sel = [r for r in runs if r["method"] == arm]
            if sel:
                est = [float(L.establishment_time(r["win"][:, None], r["t"], r["share_inst"])[0]) for r in sel]
                per_block[block][f"offline_est_{arm}"] = dict(median=float(np.median(est)), values=est)
        # candidate rules on the FR arms (and the ABF arms, context)
        for W, level, gname, persist in itertools.product(WINDOWS, LEVELS, GUARDS, PERSIST):
            key = f"W{W:g}_L{level:g}_{gname}_p{persist}"
            spec = dict(R2, W=W, level=level, persist=persist, **GUARDS[gname])
            res = {"abf": [], "fr": []}
            for r in runs:
                ctl = replay(r, spec)
                if ctl is None:
                    continue
                res[r["method"]].append(dict(seed=r["seed"], t_est=ctl.t1, t_decision=ctl.t_decision, reason=ctl.reason,
                                             actual_stop=r["stop_t"],
                                             valid=(r["stop_t"] is None or ctl.t1 is None or ctl.t1 < r["stop_t"] - 1e-9)))
                if r["method"] == "fr" and key == FROZEN:
                    # the full log of the primary candidate (occupancy, guards) for the record
                    rows.append(dict(block=block, seed=r["seed"], rate=r["rate"], actual_stop=r["stop_t"],
                                     t_est=ctl.t1, t_decision=ctl.t_decision, reason=ctl.reason,
                                     occ_first_cross=next((x["t"] for x in ctl.log if x.get("occ", 0) >= 0.75 * ctl.share), None),
                                     support_first=next((x["t"] for x in ctl.log if x.get("support")), None),
                                     stability_first=next((x["t"] for x in ctl.log if x.get("stability", 9) <= 0.05), None),
                                     occ_at_t_est=next((x["occ"] / x["share"] for x in ctl.log if ctl.t1 is not None and x["t"] == ctl.t1), None)))
            summ = {}
            for arm, lst in res.items():
                if not lst:
                    continue
                te = np.array([np.inf if x["t_est"] is None else x["t_est"] for x in lst])
                td = np.array([np.inf if x["t_decision"] is None else x["t_decision"] for x in lst])
                fin = np.isfinite(te)
                s = dict(n=len(lst), n_never=int((~fin).sum()), n_safety=int(sum(x["reason"] == "safety" for x in lst)),
                         n_invalid=int(sum(not x["valid"] for x in lst)),
                         t_est=[None if not np.isfinite(v) else float(v) for v in te],
                         t_est_median=float(np.median(te[fin])) if fin.any() else None,
                         t_est_min=float(te[fin].min()) if fin.any() else None,
                         t_est_max=float(te[fin].max()) if fin.any() else None)
                for f in FACTORS:
                    ts = np.maximum(f * te, td)
                    s[f"stop_f{f:g}"] = dict(median=float(np.median(ts[fin])) if fin.any() else None,
                                             q25=float(np.quantile(ts[fin], 0.25)) if fin.any() else None,
                                             q75=float(np.quantile(ts[fin], 0.75)) if fin.any() else None,
                                             min=float(ts[fin].min()) if fin.any() else None,
                                             max=float(ts[fin].max()) if fin.any() else None)
                summ[arm] = s
            per_block[block]["candidates"][key] = summ
        b = per_block[block]
        c = b["candidates"][FROZEN]
        print(f"{block:28s} n={b['n_runs']:2d} save {b['save_interval']:.3f}  offline est ABF/FR "
              f"{b.get('offline_est_abf', {}).get('median')}/{b.get('offline_est_fr', {}).get('median')}  "
              f"R2 primary t_est FR median {c.get('fr', {}).get('t_est_median')} [{c.get('fr', {}).get('t_est_min')}, "
              f"{c.get('fr', {}).get('t_est_max')}] stop x2 median {c.get('fr', {}).get('stop_f2', {}).get('median')} "
              f"never {c.get('fr', {}).get('n_never')} invalid {c.get('fr', {}).get('n_invalid')}")
    json.dump(dict(levels=LEVELS, factors=FACTORS, guards=list(GUARDS), persist=PERSIST, windows=WINDOWS, frozen=FROZEN,
                   blocks=per_block), open(os.path.join(out_dir, "replay_r2.json"), "w"), indent=1, default=float)
    import csv
    with open(os.path.join(out_dir, "replay_r2_runs.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out_dir)


if __name__ == "__main__":
    main()
