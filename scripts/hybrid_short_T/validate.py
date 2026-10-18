#!/usr/bin/env python
"""Stage I numerical validation of the hybrid short-T study (prereg section 3, gates V1-V9).

    python scripts/hybrid_short_T/validate.py run [--only V2 V3 ...] [--max-ccds 8]   # every engine check, each in
                                                                                     # its own pinned process
    python scripts/hybrid_short_T/validate.py one --name <check> --out <json>         # one check (internal)
    python scripts/hybrid_short_T/validate.py summary                                 # print the verdict table

Every engine check runs in a fresh process pinned (taskset) to a CCD set of this campaign's lock registry (the same
allocator as production; NOT timing runs, so no telemetry or contention rule), with OMP / NUMBA thread counts equal
to the largest thread count the check uses.  Outputs results/hybrid_short_T/validation/<check>.json and
validation.json (all checks + verdicts).  A FAIL of V2-V6 blocks the affected backend for the affected system
(prereg section 3); this script only reports, the routing step (benchmark.py freeze) reads the verdicts.

Checks
  V1  budget / grid arithmetic (no engine)
  V2  truncation identity: S1 through this harness (expB_common.run_serial: SaveClock hooks, diagnostics off) at the
      first production seed equals the stored equal-budget production run bitwise at every common save (M_all, C_all,
      FR death counters; LTA also M_prod, C_prod); full short budget at N0 (both arms), a window at small N
  V3  parallel replay identity: P_t (noise 'replay') == S1 bitwise on the science keys at threads {1, 4, 16}
  V4  counter-mode thread invariance: P_t (noise 'counter') bitwise identical over threads {1, 8, 16}
  V5  arm pairing: ABF and FR of one (N, seed) bitwise identical at every save before FR's first realised death,
      for S1 and for P_t counter mode; the FR arm must realise deaths later in the window (else inconclusive)
  V6  timing hooks inert: S1 via the SaveClock hooks == plain run_arm; P_t timers on == timers off
  V7  scorer: this harness's e_F on stored production files == the equal-budget metric cache at every save
  V8  schedules and caps: LTA no FR opportunity before step fr_start, first at fr_start; gateway fr_every 160,
      ramp 160000 steps; caps of S1 and P_t identical at N 2, 4, 16, 64, N0
  V9  timing instrumentation: per-save stamps finite and non-decreasing; engine wall_total_s <= the caller's
      perf_counter span of the call; the parallel engine's in-kernel clock span agrees with the caller's within
      1 % + 0.05 s (after the warm-up it reports)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

import hst_common as H  # noqa: E402
import expB_common as C  # noqa: E402

VAL_DIR = os.path.join(H.OUT_ROOT, "validation")


# ================================================================================================ helpers
def _bits_equal(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and np.ascontiguousarray(a).tobytes() == \
        np.ascontiguousarray(b).tobytes()


def _diff(k, a, b):
    a, b = np.asarray(a), np.asarray(b)
    rec = dict(key=k, equal=_bits_equal(a, b), shape_a=list(a.shape), shape_b=list(b.shape))
    if not rec["equal"] and a.shape == b.shape and a.size and a.dtype.kind in "fiu":
        d = np.abs(a.astype(float) - b.astype(float))
        rec["max_abs_diff"] = float(np.nanmax(d))
        rec["n_diff"] = int(np.count_nonzero(a != b))
        if a.ndim >= 1:
            rows = np.nonzero((a != b).reshape(a.shape[0], -1).any(axis=1))[0]
            rec["first_diff_row"] = int(rows[0]) if rows.size else None
    return rec


def science_keys(kind):
    if kind == "gateway":
        return ("save_step", "M_all", "C_all", "fr_deaths_cum", "fr_kd_cum", "fr_kc_cum", "fr_opp_cum",
                "fr_opp_event_cum", "X_final", "Y_final", "max_abs_bias", "n_force_evals")
    return ("save_step", "M_all", "C_all", "M_prod", "C_prod", "cum_deaths", "cum_opps", "cum_opps_with_event",
            "events_per_opp_hist", "q_final", "anc_final", "M_all_final", "C_all_final", "M_prod_final",
            "C_prod_final", "birth_hist", "death_hist", "total_replacement_events", "n_fr_opps", "fr_score_std",
            "fr_score_absmax", "n_force_evals")


def compare(ref, res, keys):
    recs = []
    for k in keys:
        if k not in ref or k not in res:
            recs.append(dict(key=k, equal=False, missing=[x for x, d in (("ref", ref), ("res", res)) if k not in d]))
            continue
        recs.append(_diff(k, ref[k], res[k]))
    return all(r["equal"] for r in recs), recs


def _meta_of(res):
    m = res.get("meta")
    if m is None and "meta_json" in res:
        m = str(res["meta_json"])
    if isinstance(m, (str, np.ndarray)):
        try:
            m = json.loads(str(m))
        except ValueError:
            m = {}
    return m or {}


def deaths_key(kind):
    return "fr_deaths_cum" if kind == "gateway" else "cum_deaths"


def score_keys(kind):
    return C.SCORE_KEYS[kind]


def _timing(res, span):
    w = np.asarray(res["wall_s_at_save"], float)
    out = dict(stamps_finite=bool(np.all(np.isfinite(w))), stamps_monotone=bool(np.all(np.diff(w) >= 0)),
               wall_total_s=float(res["wall_total_s"]), warmup_s=float(res.get("warmup_s", 0.0)),
               caller_span_s=float(span))
    out["wall_total_le_span"] = bool(out["wall_total_s"] <= span + 1e-6)
    return out


def run_serial_timed(spec, method, N, seed, n_steps, saves):
    t = time.perf_counter()
    res = C.run_serial(spec, method, int(N), int(seed), int(n_steps), np.asarray(saves, np.int64))
    return res, time.perf_counter() - t


def run_parallel_timed(spec, method, N, seed, n_steps, saves, threads, noise, timers=True):
    E = C.parallel_engine(spec["system"], spec["engine_module"])
    cfg = C.engine_cfg_for(spec, N, seed, n_steps)
    t = time.perf_counter()
    res = E.run_arm_parallel(cfg, method, int(N), int(seed), int(n_steps), np.asarray(saves, np.int64),
                             threads=int(threads), noise=noise, timers=timers)
    return dict(res), time.perf_counter() - t


def window_saves(spec, N, n_w, n=20):
    st = np.unique(np.clip(np.round(np.arange(1, n + 1) / n * n_w).astype(np.int64), 1, n_w))
    if spec["kind"] == "lta" and 0 < spec["fr_start"] < n_w:
        st = np.unique(np.concatenate([st, [spec["fr_start"] - 1, spec["fr_start"], spec["fr_start"] + 5]]))
    return st


# ================================================================================================ checks (engine)
def check_truncation(system, N, method, window=None, dense=False):
    """V2 (dense = V2d: the run saves on the full A2.1 grid; compared at the production saves it contains)."""
    import eqb_metrics as M
    spec = H.system_spec(system)
    seed = spec["production_reference_seeds"][0]
    path = os.path.join(spec["eqb_dir"], f"N{N}", f"s{seed}_{method}.npz")
    prod = M.engine_module(spec["kind"]).load_result(path)
    n_short = H.n_steps_for(spec, N)
    n = n_short if window is None else int(window)
    pst = np.asarray(prod["save_step"], np.int64)
    saves = pst[(pst >= 1) & (pst <= n)]
    if window is None and saves[-1] != n_short:
        raise H.HSTError("n_short is not a production save")
    run_saves = H.save_grid(spec, N)[0] if dense else saves
    res, span = run_serial_timed(spec, method, N, seed, n, run_saves)
    if dense:
        rs = np.asarray(res["save_step"], np.int64)
        common = np.intersect1d(rs, saves)
        sel = np.searchsorted(rs, common)
        res = {k: (np.asarray(v)[sel] if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == rs.size else v)
               for k, v in res.items()}
        saves = common
    idx = np.searchsorted(pst, saves)
    keys = list(score_keys(spec["kind"])) + [deaths_key(spec["kind"])]
    recs = []
    for k in keys:
        if k not in prod:
            recs.append(dict(key=k, equal=None, note="not stored in the production file"))
            continue
        recs.append(_diff(k, np.asarray(prod[k])[idx], np.asarray(res[k])))
    ok = all(r["equal"] for r in recs if r["equal"] is not None)
    return dict(check="V2d" if dense else "V2", system=system, N=N, method=method, seed=seed, n_steps=int(n),
                full_short_budget=window is None, n_run_saves=int(len(run_saves)),
                n_common_saves=int(saves.size), production_file=os.path.relpath(path, H.ROOT),
                production_diagnostics=_meta_of(prod).get("diagnostics"),
                fr_deaths_end=int(np.asarray(res[deaths_key(spec["kind"])])[-1]), keys=recs,
                verdict="PASS" if ok else "FAIL", timing=_timing(res, span))


def check_replay(system, N, method, threads, n_w, seed=None):
    """V3."""
    spec = H.system_spec(system)
    seed = seed if seed is not None else spec["validation_seeds"][0]
    saves = window_saves(spec, N, n_w)
    ref, span = run_serial_timed(spec, method, N, seed, n_w, saves)
    rows = []
    ok_all = True
    for t in threads:
        par, sp = run_parallel_timed(spec, method, N, seed, n_w, saves, t, "replay")
        ok, recs = compare(ref, par, science_keys(spec["kind"]))
        ok_all &= ok
        rows.append(dict(threads=t, equal=ok, differing=[r for r in recs if not r["equal"]],
                         timing=_timing(par, sp)))
    return dict(check="V3", system=system, N=N, method=method, seed=seed, n_steps=int(n_w), threads=list(threads),
                fr_deaths_end=int(np.asarray(ref[deaths_key(spec["kind"])])[-1]), rows=rows,
                serial_timing=_timing(ref, span), verdict="PASS" if ok_all else "FAIL")


def check_counter_invariance(system, N, method, threads, n_w, seed=None):
    """V4 (+ V9 in-kernel clock vs the caller's span)."""
    spec = H.system_spec(system)
    seed = seed if seed is not None else spec["validation_seeds"][1]
    saves = window_saves(spec, N, n_w)
    base, sp0 = run_parallel_timed(spec, method, N, seed, n_w, saves, threads[0], "counter")
    rows = [dict(threads=threads[0], equal=True, timing=_timing(base, sp0))]
    ok_all = True
    for t in threads[1:]:
        par, sp = run_parallel_timed(spec, method, N, seed, n_w, saves, t, "counter")
        ok, recs = compare(base, par, science_keys(spec["kind"]))
        ok_all &= ok
        rows.append(dict(threads=t, equal=ok, differing=[r for r in recs if not r["equal"]], timing=_timing(par, sp)))
    clock = []
    for r in rows:
        tm = r["timing"]
        inner = tm["wall_total_s"] + tm["warmup_s"]
        clock.append(dict(threads=r["threads"], inner_s=inner, span_s=tm["caller_span_s"],
                          untimed_remainder_s=tm["caller_span_s"] - inner,
                          within_span=bool(inner <= tm["caller_span_s"] + 1e-3),
                          stamps_ok=bool(tm["stamps_finite"] and tm["stamps_monotone"])))
    clock_ok = all(c["within_span"] and c["stamps_ok"] for c in clock)
    return dict(check="V4", system=system, N=N, method=method, seed=seed, n_steps=int(n_w), threads=list(threads),
                fr_deaths_end=int(np.asarray(base[deaths_key(spec["kind"])])[-1]), rows=rows, clock=clock,
                V9_clock_ok=clock_ok, perf_counter_clock=str(time.get_clock_info("perf_counter")),
                verdict="PASS" if (ok_all and clock_ok) else "FAIL")


def check_pairing(system, N, impl, threads, n_w, seed=None):
    """V5: ABF vs FR bitwise before FR's first realised death."""
    spec = H.system_spec(system)
    seed = seed if seed is not None else spec["validation_seeds"][2]
    saves = window_saves(spec, N, n_w, n=40)
    if impl == "serial":
        a, _ = run_serial_timed(spec, "abf", N, seed, n_w, saves)
        f, _ = run_serial_timed(spec, "fr", N, seed, n_w, saves)
    else:
        a, _ = run_parallel_timed(spec, "abf", N, seed, n_w, saves, threads, "counter")
        f, _ = run_parallel_timed(spec, "fr", N, seed, n_w, saves, threads, "counter")
    d = np.asarray(f[deaths_key(spec["kind"])])
    pre = np.flatnonzero(d == 0)
    keys = score_keys(spec["kind"])
    same = [all(_bits_equal(np.asarray(a[k])[i], np.asarray(f[k])[i]) for k in keys) for i in range(len(saves))]
    first_diff = next((i for i, s in enumerate(same) if not s), None)
    ok_pre = all(same[i] for i in pre)
    inconclusive = int(d[-1]) == 0
    verdict = "PASS" if ok_pre and not inconclusive else ("INCONCLUSIVE" if ok_pre else "FAIL")
    return dict(check="V5", system=system, N=N, impl=impl, threads=threads, seed=seed, n_steps=int(n_w),
                n_saves=int(len(saves)), n_saves_before_first_death=int(pre.size),
                first_save_with_death=(int(np.flatnonzero(d > 0)[0]) if np.any(d > 0) else None),
                first_save_differing=first_diff, fr_deaths_end=int(d[-1]),
                identical_at_every_save_before_first_death=bool(ok_pre),
                differ_after_first_death=(first_diff is not None), verdict=verdict)


def check_hooks(system, N, method, n_w, threads=4, seed=None):
    """V6: SaveClock hooks inert (S1) and parallel timers inert."""
    spec = H.system_spec(system)
    seed = seed if seed is not None else spec["validation_seeds"][3]
    saves = window_saves(spec, N, n_w)
    hooked, _ = run_serial_timed(spec, method, N, seed, n_w, saves)
    E = __import__(spec["validated_module"])
    if spec["kind"] == "gateway":
        plain = E.run_arm(dict(spec["engine_cfg"]), method, N, seed, n_w, saves, diagnostics=False)
    else:
        c = E.make_cfg(float(spec["T_K"]), N, seed, **dict(spec["engine_cfg"], diagnostics=False, n_steps=n_w))
        plain = E.run_arm(c, method, saves)
    ok1, r1 = compare(plain, hooked, science_keys(spec["kind"]))
    on, _ = run_parallel_timed(spec, method, N, seed, n_w, saves, threads, "counter", timers=True)
    off, _ = run_parallel_timed(spec, method, N, seed, n_w, saves, threads, "counter", timers=False)
    ok2, r2 = compare(on, off, science_keys(spec["kind"]))
    return dict(check="V6", system=system, N=N, method=method, seed=seed, n_steps=int(n_w), threads=threads,
                serial_hooks_inert=ok1, serial_differing=[r for r in r1 if not r["equal"]],
                parallel_timers_inert=ok2, parallel_differing=[r for r in r2 if not r["equal"]],
                verdict="PASS" if (ok1 and ok2) else "FAIL")


# ================================================================================================ checks (no engine)
def check_grid():
    """V1."""
    S = C.load_json(os.path.join(H.ROOT, "results", "equal_budget_v2", "short_T", "summary.json"))
    eq = dict(gateway="gateway", lta300="lta_T300", lta150="lta_T150")
    rows, ok = [], True
    for s in H.SYSTEMS:
        spec = H.system_spec(s)
        sT = {int(r["N"]): float(r["T_N"]) for r in S[eq[s]]["rows"]}
        if abs(S[eq[s]]["B_new"] - spec["B"]) > 0.5 or abs(S[eq[s]]["T_star"] - spec["T_star"]) > 1e-9:
            ok = False
            rows.append(dict(system=s, problem="B or T* differs from the short-T summary"))
        for N in spec["N_ladder"]:
            n = H.n_steps_for(spec, N)
            st, bm, um = H.save_grid(spec, N)
            E = H._engine(spec["kind"])
            orig, _ = E.budget_save_grid(spec["B_orig"] // N)
            h = spec["h"]
            phys = [int(round(t / h)) for t in spec["P"]["physical_checkpoints_t"]
                    if 0 < round(t / h) <= spec["B_orig"] // N]
            orig_all = np.unique(np.concatenate([orig, np.asarray(phys, np.int64)]))
            sub = bool(np.all(np.isin(st, orig_all)))
            r = dict(system=s, N=N, n_steps=n, integer=(n * N == spec["B"]), T_N=n * h,
                     T_N_matches_short_T=abs(n * h - sT[N]) <= 1e-9 * max(1.0, sT[N]), last_save_is_n=bool(
                         st[-1] == n), n_saves=int(st.size), n_budget=int(bm.sum()), n_uniform=int(um.sum()),
                     subset_of_production_grid=sub)
            ok &= r["integer"] and r["T_N_matches_short_T"] and r["last_save_is_n"] and sub
            rows.append(r)
    return dict(check="V1", rows=rows, verdict="PASS" if ok else "FAIL")


def check_scorer(n_files=3):
    """V7."""
    import eqb_metrics as M
    rows, ok = [], True
    for s in H.SYSTEMS:
        spec = H.system_spec(s)
        sc = H.Scorer(spec)
        E = M.engine_module(spec["kind"])
        for N, method in ((spec["N0"], "fr"), (16, "abf"), (2, "fr"))[:n_files]:
            seed = spec["production_reference_seeds"][1]
            src = os.path.join(spec["eqb_dir"], f"N{N}", f"s{seed}_{method}.npz")
            cache = os.path.join(spec["eqb_dir"], "analysis", "run_metrics", f"N{N}", f"s{seed}_{method}.npz")
            res = E.load_result(src)
            e = sc.e_F(res)
            z = np.load(cache)
            ec = np.asarray(z["e_F"], float)
            same_grid = bool(np.array_equal(np.asarray(z["save_t"], float), np.asarray(res["save_t"], float)))
            mad = float(np.nanmax(np.abs(e - ec))) if e.shape == ec.shape else math.inf
            r = dict(system=s, N=N, method=method, seed=seed, n_saves=int(e.size), max_abs_diff=mad,
                     same_grid=same_grid, bitwise=bool(e.shape == ec.shape and _bits_equal(e, ec)))
            ok &= mad <= 1e-12 and same_grid
            rows.append(r)
    return dict(check="V7", rows=rows, verdict="PASS" if ok else "FAIL")


def check_schedules():
    """V8 (static part: derived knobs and caps of S1 vs P_t)."""
    rows, ok = [], True
    import gateway_ladder_numba as G
    import lta_ladder_numba as L
    for s in H.SYSTEMS:
        spec = H.system_spec(s)
        for N in sorted({2, 4, 16, 64, spec["N0"]}):
            if spec["kind"] == "gateway":
                kn = G.derived_knobs(G.full_cfg(dict(spec["engine_cfg"])), N)
                r = dict(system=s, N=N, cap_S1=kn["cap"], fr_every=kn["fr_every"], ramp_steps=kn["ramp_steps"],
                         cap_expected=max(1, int(math.floor(0.08 * N))))
                r["ok"] = (kn["fr_every"] == 160 and kn["ramp_steps"] == 160000.0 and kn["cap"] == r["cap_expected"])
            else:
                c = L.make_cfg(float(spec["T_K"]), N, 1, **dict(spec["engine_cfg"], diagnostics=False,
                                                                 n_steps=H.n_steps_for(spec, N)))
                import lta_parallel as LP
                cp = LP.resolve(dict(spec["engine_cfg"], T_K=spec["T_K"]), N, 1, H.n_steps_for(spec, N))
                r = dict(system=s, N=N, cap_S1=c["cap"], cap_P=cp["cap"], fr_start=c["fr_start_steps"],
                         fr_every=c["fr_every"], cap_expected=max(1, int(0.02 * N)),
                         T_K_S1=c["T_K"], T_K_P=cp["T_K"])
                r["ok"] = (c["cap"] == cp["cap"] == r["cap_expected"] and c["fr_start_steps"] == 20000 and
                           c["fr_every"] == 5 and cp["fr_start_steps"] == 20000 and cp["fr_every"] == 5 and
                           c["T_K"] == cp["T_K"] == spec["T_K"])
            ok &= r["ok"]
            rows.append(r)
    return dict(check="V8", rows=rows, verdict="PASS" if ok else "FAIL")


def check_lta_fr_start(system, N=16, n_w=30000):
    """V8 (dynamic part): no FR opportunity before fr_start, the first one at nxt = fr_start; both backends."""
    spec = H.system_spec(system)
    seed = spec["validation_seeds"][4]
    fs = spec["fr_start"]
    saves = np.array([fs - 2, fs - 1, fs, fs + 1, fs + 5, n_w], np.int64)
    a, _ = run_serial_timed(spec, "fr", N, seed, n_w, saves)
    b, _ = run_parallel_timed(spec, "fr", N, seed, n_w, saves, 4, "counter")
    ops_a = np.asarray(a["cum_opps"]).tolist()
    ops_b = np.asarray(b["cum_opps"]).tolist()
    ok = ops_a == ops_b and ops_a[0] == 0 and ops_a[1] == 0
    return dict(check="V8", system=system, N=N, seed=seed, saves=saves.tolist(), cum_opps_S1=ops_a,
                cum_opps_P=ops_b, verdict="PASS" if ok else "FAIL",
                note="cum_opps at the saves (save k is taken during evaluated step k, before its move / FR): 0 at "
                     "fr_start - 2 and fr_start - 1 means no opportunity before fr_start")


# ================================================================================================ plan
def plan():
    """[(name, threads, func name, kwargs)]."""
    P = []
    win_small = {"gateway": (16, 2_000_000), "lta300": (4, 200_000), "lta150": (4, 200_000)}
    for s, N0 in (("gateway", 2048), ("lta300", 1024), ("lta150", 1024)):
        for m in ("abf", "fr"):
            P.append((f"V2_{s}_N{N0}_{m}_full", 1, "check_truncation", dict(system=s, N=N0, method=m)))
            Ns, nw = win_small[s]
            P.append((f"V2_{s}_N{Ns}_{m}_window", 1, "check_truncation", dict(system=s, N=Ns, method=m, window=nw)))
    v3 = [("gateway", 2048, 20000), ("gateway", 64, 200000), ("lta300", 1024, 25000), ("lta300", 256, 40000),
          ("lta300", 16, 120000), ("lta150", 1024, 25000)]
    for s, N, nw in v3:
        for m in ("abf", "fr"):
            P.append((f"V3_{s}_N{N}_{m}", 16, "check_replay", dict(system=s, N=N, method=m,
                                                                   threads=[t for t in (1, 4, 16) if t <= N],
                                                                   n_w=nw)))
    for s, N, nw in (("lta300", 256, 40000), ("gateway", 2048, 20000)):
        for m in ("abf", "fr"):
            P.append((f"V4_{s}_N{N}_{m}", 16, "check_counter_invariance",
                      dict(system=s, N=N, method=m, threads=[1, 8, 16], n_w=nw)))
    for s, N, nw in (("gateway", 64, 200000), ("lta300", 1024, 25000), ("lta150", 1024, 25000)):
        P.append((f"V5_{s}_N{N}_S1", 1, "check_pairing", dict(system=s, N=N, impl="serial", threads=1, n_w=nw)))
        P.append((f"V5_{s}_N{N}_P8", 8, "check_pairing", dict(system=s, N=N, impl="parallel", threads=8, n_w=nw)))
    for s, N, nw in (("gateway", 64, 100000), ("lta300", 64, 30000)):
        P.append((f"V6_{s}_N{N}_fr", 4, "check_hooks", dict(system=s, N=N, method="fr", n_w=nw)))
    for s in ("lta300", "lta150"):
        P.append((f"V8_{s}_frstart", 4, "check_lta_fr_start", dict(system=s)))
    # Amendment A2.9
    P.append(("V2d_gateway_N2048_fr_full", 1, "check_truncation", dict(system="gateway", N=2048, method="fr",
                                                                       dense=True)))
    P.append(("V2d_lta300_N1024_fr_full", 1, "check_truncation", dict(system="lta300", N=1024, method="fr",
                                                                      dense=True)))
    for s, N, nw in (("lta300", 16, 120000), ("lta300", 64, 60000), ("gateway", 16, 400000)):
        for m in ("abf", "fr"):
            P.append((f"V4b_{s}_N{N}_{m}", 4, "check_counter_invariance",
                      dict(system=s, N=N, method=m, threads=[1, 2, 4], n_w=nw)))
    return P


# ================================================================================================ driver
def run_all(only=None, max_ccds=8, log=print):
    import expB_alloc as A
    import hst_sched as S
    os.makedirs(VAL_DIR, exist_ok=True)
    out = {}
    for name, fn in (("V1_grid", check_grid), ("V7_scorer", check_scorer), ("V8_static", check_schedules)):
        if only and not any(name.startswith(o) for o in only):
            continue
        r = fn()
        C.write_json(os.path.join(VAL_DIR, f"{name}.json"), r)
        out[name] = r
        log(f"{name}: {r['verdict']}")
    todo = [p for p in plan() if not only or any(p[0].startswith(o) for o in only)]
    todo = [p for p in todo if not _done(p[0])]
    reg = A.Registry(H.REGISTRY_DIR)
    alloc = S.HSTAllocator(reg, log=log, reserved=S.RESERVED_CCDS, threshold=S.QUALIFY_BUSY, sample_s=1.0)
    active = {}
    t_start = time.time()
    core = None
    allowed = [c for i in S.RESERVED_CCDS for c in A.ccd_cores(i)]
    try:
        while todo or active:
            core = A.pin_outside(reg, allowed, current=core)
            for name in list(active):
                p, a, t0 = active[name]
                if p.poll() is None:
                    continue
                wall = time.time() - t0
                alloc.release(a)
                del active[name]
                H.ledger_append("valid", name, a["threads"], wall, note=f"rc {p.returncode}; ccds {a['ccds']}")
                log(f"  {name}: rc {p.returncode} ({wall:.0f} s) -> {_verdict(name)}")
            if not todo:
                time.sleep(1.0)
                continue
            name, t, fn, kw = todo[0]
            n_ccd = A.n_ccds_for(t)
            held = [i for i in reg.held() if i not in S.RESERVED_CCDS]
            if len(held) + n_ccd > max_ccds:
                time.sleep(1.0)
                continue
            a = alloc.fallback(t, f"hst-valid-{name}-{os.getpid()}",
                               dict(kind="valid", job_key=f"hst|valid|{name}",
                                    driver=dict(pid=os.getpid(), start=A.proc_start_time(os.getpid()))))
            if a is None:
                time.sleep(2.0)
                continue
            todo.pop(0)
            cmd = [sys.executable, os.path.abspath(__file__), "one", "--name", name, "--func", fn, "--kwargs",
                   json.dumps(kw), "--out", os.path.join(VAL_DIR, f"{name}.json")]
            env = dict(NUMBA_CACHE_DIR=os.environ.get("NUMBA_CACHE_DIR", os.path.expanduser("~/.cache/numba_hst")))
            p = A.launch_pinned(cmd, a["cores"], log_path=os.path.join(VAL_DIR, f"{name}.log"), extra_env=env)
            reg.add_pid(a["ccds"], a["token"], p.pid)
            active[name] = (p, a, time.time())
            log(f"  launched {name} on cores {A.cpu_list_str(a['cores'])}")
    finally:
        for name, (p, a, t0) in active.items():
            A.kill_process_group(p)
            alloc.release(a)
    log(f"engine checks done in {time.time() - t_start:.0f} s")
    return summarize()


def _done(name):
    p = os.path.join(VAL_DIR, f"{name}.json")
    if not os.path.exists(p):
        return False
    try:
        return C.read_json(p).get("verdict") in ("PASS", "FAIL", "INCONCLUSIVE")
    except (OSError, ValueError):
        return False


def _verdict(name):
    p = os.path.join(VAL_DIR, f"{name}.json")
    try:
        return C.read_json(p).get("verdict")
    except (OSError, ValueError):
        return "NO RECORD"


def summarize():
    rows = {}
    for f in sorted(os.listdir(VAL_DIR)):
        if f.endswith(".json") and f != "validation.json":
            r = C.read_json(os.path.join(VAL_DIR, f))
            rows[f[:-5]] = r.get("verdict")
    names = [p[0] for p in plan()] + ["V1_grid", "V7_scorer", "V8_static"]
    missing = [n for n in names if n not in rows]
    blocked = {}
    for n, v in rows.items():
        if v != "PASS" and n.split("_")[0] in ("V2", "V3", "V4", "V5", "V6"):
            sysn = n.split("_")[1]
            blocked.setdefault(sysn, []).append(f"{n}: {v}")
    out = dict(verdicts=rows, missing=missing, n_pass=sum(v == "PASS" for v in rows.values()), n=len(rows),
               blocked=blocked, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    C.write_json(os.path.join(VAL_DIR, "validation.json"), out)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--only", nargs="*", default=None)
    r.add_argument("--max-ccds", type=int, default=8)
    o = sub.add_parser("one")
    o.add_argument("--name", required=True)
    o.add_argument("--func", required=True)
    o.add_argument("--kwargs", required=True)
    o.add_argument("--out", required=True)
    sub.add_parser("summary")
    a = ap.parse_args(argv)
    if a.cmd == "one":
        t0 = time.time()
        # scripts/mechanism/parallel_benchmark (imported by expB_common.run_serial) sets NUMBA_NUM_THREADS /
        # OMP_NUM_THREADS = 1 in os.environ at import; numba re-reads its config at the next compilation, so a check
        # that runs S1 before P_t would get a one-thread pool.  Import it first and restore the launch environment.
        keep = {k: os.environ.get(k) for k in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS")}
        import parallel_benchmark  # noqa: F401
        for k, v in keep.items():
            if v is not None:
                os.environ[k] = v
        try:
            res = globals()[a.func](**json.loads(a.kwargs))
        except Exception as e:  # noqa: BLE001
            import traceback
            res = dict(check=a.name, verdict="ERROR", error=f"{type(e).__name__}: {e}",
                       traceback=traceback.format_exc())
        res.update(name=a.name, wall_s=time.time() - t0, cpus=sorted(os.sched_getaffinity(0)),
                   utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        C.write_json(a.out, res)
        return 0 if res.get("verdict") in ("PASS", "INCONCLUSIVE") else 1
    if a.cmd == "run":
        out = run_all(a.only, a.max_ccds)
    else:
        out = summarize()
    for k, v in sorted(out["verdicts"].items()):
        print(f"{k:40s} {v}")
    print(f"{out['n_pass']}/{out['n']} PASS; missing {out['missing']}; blocked {out['blocked']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
