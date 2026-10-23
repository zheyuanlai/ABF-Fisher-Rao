#!/usr/bin/env python
"""Automated validation of the WCA timing campaign (run after analyze_campaign.py, export_figures.py, plot_figures.py).

Checks (section 14 of the task):
  completeness (every frozen seed x arm present), one commit / engine / frozen config for every run, the validated
  physics (cfg = the Phase-V cfg with the arm's FR start), equal K, dt, T, n_steps (=> equal force evaluations),
  FR events only inside each arm's schedule, no event after a stop, z_hist rows sum to K, finite errors and states,
  pre-FR identity of every FR arm with ABF (shared Langevin noise), the bitwise regressions against the stored
  Phase-V trajectories, the engine test-suite logs, statistics recomputed independently from data/per_seed, the
  three-panel CSV schema and grids, and the presence of every figure.  Writes VALIDATION_REPORT.md (appending
  scripts/validation_notes.md if present) and data/summary/validation_results.json.  Exit 1 on any FAIL.
"""
import csv
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common_wca as CW  # noqa: E402
import export_figures as EX  # noqa: E402

SUM = os.path.join(CW.PKG, "data", "summary")
REG = os.path.join(SUM, "regression")
ROWS = []


def chk(ok, name, detail=""):
    ROWS.append(dict(check=name, status="PASS" if ok else "FAIL", detail=str(detail)))


def info(name, detail):
    ROWS.append(dict(check=name, status="INFO", detail=str(detail)))


def main():
    fz = CW.frozen()
    A = json.load(open(os.path.join(SUM, "analysis.json")))
    arms = [a["name"] for a in fz["arms"] if a["run"]]
    spec = {a["name"]: a for a in fz["arms"]}
    seeds = [int(s) for s in fz["seeds"]]
    rc = A["run_checks"]
    chk(not A["missing_runs"] and len(rc) == len(arms) * len(seeds), "every frozen (seed, arm) run is present and scored",
        f"{len(rc)} runs, missing {A['missing_runs']}")
    chk(len({r["commit"] for r in rc}) == 1, "one commit for every run", {r["commit"] for r in rc})
    chk(len({r["engine_sha256"] for r in rc}) == 1, "one engine (sha256) for every run", {r["engine_sha256"][:12] for r in rc})
    chk(len({r["frozen_config_sha256"] for r in rc}) == 1 and rc[0]["frozen_config_sha256"] == CW.sha256_file(CW.FROZEN),
        "every run used the committed frozen_config.json")
    chk(all(r["cfg_matches_frozen"] for r in rc), "every run's engine cfg = the validated Phase-V cfg with its FR start")
    chk(all(r["N"] == CW.N and r["n_steps"] == CW.s(CW.T) for r in rc), "equal K = 256 and n_steps = 1 920 000 (equal force evaluations)")
    chk(all(r["fr_events_only_in_schedule"] for r in rc), "FR events only inside each arm's [start, stop) (none in ABF)")
    for arm in arms:
        st = spec[arm]
        if st["fr"]:
            exp_stop = CW.s(st["fr_stop_t"]) if st["fr_stop_t"] is not None else -1
            ok = all(r["fr_stop_steps"] == exp_stop and r["fr_stream_ids"] == [st["fr_stream_id"]] for r in rc if r["arm"] == arm)
            chk(ok, f"{arm}: fr_stop_steps = {exp_stop}, FR stream id {st['fr_stream_id']}")
    chk(all(r["z_hist_sums_to_N"] for r in rc), "z_hist rows sum to K at every checkpoint")
    chk(all(r["finite_errors"] and r["finite_state"] for r in rc), "finite e_F, e_F' and final states in every run")
    info("ABF bias-clip fraction (median over runs, per arm)",
         {arm: float(np.median([r["bias_clip_fraction"] for r in rc if r["arm"] == arm])) for arm in arms})
    # pre-FR identity with ABF (separate processes, shared noise): identical per-seed e_F before each start
    t, Eabf, _ = EX.read("abf", "eF")
    for arm in arms:
        st = spec[arm]
        if not st["fr"] or st["fr_start_t"] == 0:
            continue
        _, E, _ = EX.read(arm, "eF")
        pre = t < st["fr_start_t"] - 1e-9
        chk(np.array_equal(E[pre], Eabf[pre]), f"{arm} = ABF bitwise before its FR start t = {st['fr_start_t']:g}",
            f"{int(pre.sum())} checkpoints")
    for arm in ("fr_start0_stop20", "fr_start0_stop40"):
        _, R, _ = EX.read(arm, "repl_cum")
        tstop = spec[arm]["fr_stop_t"]
        post = t >= tstop
        chk(np.all(R[post] == R[post][0:1]), f"{arm}: cumulative replacements constant for t >= {tstop:g}",
            f"median replacements at stop {np.median(R[post][0]):.0f}")
    # regressions against the stored Phase-V trajectories
    for f in sorted(glob.glob(os.path.join(REG, "*.json"))):
        r = json.load(open(f))
        chk(r["all_bitwise"], f"regression {r['arm']}: extended engine = stored Phase-V seed 5100 bitwise through t = {r['t_end']:g}",
            f"{len(r['saves_compared'])} saves, replacements {r['replacements_at_t_end']}")
    for f in sorted(glob.glob(os.path.join(SUM, "tests", "*.txt"))):
        txt = open(f).read()
        last = [l for l in txt.splitlines() if " passed" in l or " failed" in l or " error" in l]
        chk(bool(last) and "failed" not in last[-1] and "error" not in last[-1], f"test log {os.path.basename(f)}", last[-1] if last else "no summary line")
    # statistics recomputed independently from the per-seed files
    for key in ("P1", "P2"):
        c = A["primary"][key]
        m = c["metric"]
        ta, Xa, _ = EX.read(c["arm"], "eF")
        _, Xb, _ = EX.read(c["base"], "eF")
        if m == "e_F(240)":
            va, vb = Xa[-1], Xb[-1]
        else:
            lo, hi = [float(x) for x in m[m.index("[") + 1:m.index("]")].split(",")]
            sel = (ta >= lo - 1e-9) & (ta <= hi + 1e-9)
            va, vb = np.trapezoid(Xa[sel], ta[sel], axis=0), np.trapezoid(Xb[sel], ta[sel], axis=0)
        med = float(np.median(va / vb - 1))
        chk(abs(med - c["median"]) < 1e-8, f"{key} median recomputed from data/per_seed (10-digit CSV precision)",
            f"{100 * med:+.3f} % (diff {abs(med - c['median']):.1e})")
    sc = list(csv.DictReader(open(os.path.join(CW.PKG, "data", "per_seed", "scalars.csv"))))
    chk(len(sc) == len(arms) * len(seeds), "scalars.csv has one row per (arm, seed)", len(sc))
    # three-panel CSVs
    for pre, arm in EX.PREFIX.items():
        if arm not in arms:
            continue
        grids = []
        for panel in ("eF", "eFp", "exi"):
            p = os.path.join(CW.PKG, "data", "threepanel", f"{pre}_{panel}.csv")
            if not os.path.exists(p):
                chk(False, f"{pre}_{panel}.csv present")
                continue
            rows = list(csv.reader(open(p)))
            head, a = rows[0], np.array([[float(v) for v in r] for r in rows[1:]])
            ok = (tuple(head) == EX.COLS and a.shape[1] == 7 and np.all(np.isfinite(a)) and np.all(a[:, 1:] > 0)
                  and np.all(np.diff(a[:, 0]) > 0) and abs(a[-1, 0] - 240) < 1e-9
                  and np.all(a[:, 2] <= a[:, 1]) and np.all(a[:, 1] <= a[:, 3]) and np.all(a[:, 5] <= a[:, 4]) and np.all(a[:, 4] <= a[:, 6]))
            chk(ok, f"{pre}_{panel}.csv schema, finite > 0, increasing t to 240, q25 <= median <= q75", f"{a.shape[0]} rows")
            grids.append(a[:, 0])
        chk(len(grids) == 3 and all(np.array_equal(g, CW.save_grid() * CW.DT) for g in grids),
            f"{pre}: the three panels use the 133 saved checkpoints (no interpolation)")
    for name in ("main_threepanel", "timing_comparison", "marginal_vs_free_energy", "supp_all_arms", "supp_genealogy"):
        chk(os.path.exists(os.path.join(CW.PKG, "figures", f"{name}.pdf")), f"figure {name}.pdf present")
    chk(A["reference_sha"] == fz["reference"]["array_sha256"], "scored against the frozen exact-MC reference", A["reference"])
    n_fail = sum(r["status"] == "FAIL" for r in ROWS)
    n_pass = sum(r["status"] == "PASS" for r in ROWS)
    json.dump(dict(n_pass=n_pass, n_fail=n_fail, checks=ROWS), open(os.path.join(SUM, "validation_results.json"), "w"), indent=1)
    L = ["# Validation report (generated by `scripts/validate_campaign.py`)", "", f"**{n_pass} checks passed, {n_fail} failed.**", "",
         "| check | status | detail |", "|---|---|---|"] + [f"| {r['check']} | {r['status']} | {r['detail']} |" for r in ROWS]
    notes = os.path.join(HERE, "validation_notes.md")
    if os.path.exists(notes):
        L += ["", open(notes).read()]
    open(os.path.join(CW.PKG, "VALIDATION_REPORT.md"), "w").write("\n".join(L) + "\n")
    print(f"{n_pass} PASS, {n_fail} FAIL")
    for r in ROWS:
        if r["status"] == "FAIL":
            print("  FAIL", r["check"], r["detail"])
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
