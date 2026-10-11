#!/usr/bin/env python
"""Check 6 of Experiment A (docs/final_campaign/PREREGISTRATION.md section 1.3): ABF / FR pairing before the first
birth-death.

    python scripts/final_campaign/expA_check_pairing.py --smoke [--dry-run]        # check-9 smoke runs (gate)
    python scripts/final_campaign/expA_check_pairing.py --production [--dry-run]   # all production runs (audit)
    python scripts/final_campaign/expA_check_pairing.py --root DIR [--out FILE]    # any <DIR>/<cell>/N<N>/s<seed>_*.npz

For every (cell, N, seed) with both s<seed>_abf.npz and s<seed>_fr.npz under the root:

1. Precondition: the two run signatures (meta.run_signature) agree on every field except 'method' (same cfg / model,
   N, seed, n_steps, noise seed, FR entropy, initial-state sha256, save / trace / snapshot grids, diagnostics, engine).
2. Compared saves (prereg rule): the leading saves with fr_deaths_cum == 0 in the FR file, i.e. every save before the
   first realised FR death.  On them every SHARED saved array must be bitwise equal (dtype, shape, values, NaN = NaN):
     per save   save_step / save_t / save_u, M_all, C_all, hist_inst, region_frac, events_cum, lineage_visited,
                gen_nuniq_run, gen_nuniq_win, gen_win_age_steps, fr_deaths_cum, fr_kd_cum, fr_kc_cum,
                fr_opp_event_cum, fr_opp_death_cum, fr_death_pos_hist_cum, fr_birth_pos_hist_cum;
     per trace  rows with traces_step <= the last compared save step: traces, traces_y, traces_anc, traces_rebirth
                (traces_step / traces_t compared whole);
     per snap   rows with snap_step <= the last compared save step: dep_age_C, dep_age_M, dep_age_M2, dep_cross_C
                (snap_step / snap_t / snap_u compared whole);
     run        n_force_evals; first_arrival_step / first_arrival_t whenever either arm's first arrival is <= the last
                compared save step; and, if the FR arm had NO FR event in the whole run, every array incl. X_final,
                Y_final and max_abs_bias.
   NOT compared (different by construction, not pairing evidence): fr_opp_cum, fr_events_hist(_cum),
   fr_deaths_hist(_cum), n_fr_steps (the ABF arm has no FR opportunities), gen_ess_* and gen_maxfam_* (the ABF arm
   writes NaN, FR its family statistics), and the session bookkeeping (cfg_json, meta_json, wall_s, peak_rss_mb,
   n_checkpoints_written, resumed_from).  The ABF arm's FR counters must all be zero.
3. Permutation-only FR events.  The engine changes the state at the first opportunity with kd + kc > 0 (an 'event'),
   not at the first realised death: an event with kd = 0, kc >= 1 whose dropped pool entries are exactly the clone
   copies (probability ~2/(N + 1) for kc = 1) is a pure slot PERMUTATION -- zero realised deaths, yet slot order (deposit
   summation order, noise-to-walker assignment) changes and the arms legitimately diverge.  Such a pair FAILS the
   frozen prereg rule (compare up to the last save before the first realised death) whenever a save falls between
   that event and the first death.  If the prereg prefix differs ONLY on saves with fr_opp_event_cum > 0 (which, with
   fr_deaths_cum == 0 there, proves such an event happened), the tool also evaluates the 'event rule' (compare the
   saves before the first event, and run the replay (4)) and labels the pair PASS_EVENT_RULE -- NOT a PASS of the frozen
   rule.  The event rule is not preregistered: a check containing any PASS_EVENT_RULE pair has verdict
   AMENDMENT_REQUIRED (exit 4) unless --event-rule-amendment REF is given, where REF must occur in the dated-amendment
   section of docs/final_campaign/PREREGISTRATION.md (expA_run_common.amendment_ref_found; refused otherwise); REF is
   written into the record (event_rule_amendment) and into its input signature.
4. Replay, for a pair with 0 compared saves (and for every event-rule pair): both arms re-run from the stored cfg /
   meta (the stored run signature is first reproduced exactly at HEAD, so the replay IS the stored run) up to the
   first FR opportunity t with a non-zero event (found by an FR replay saving after every opportunity; the replay's
   M_all / C_all / FR counters must equal the stored FR file at every stored save <= the first event save): at
   completed step t the full state (X, Y, Mh, Ch, max|bias|) of ABF and FR is bitwise equal; at t + 1 Mh, Ch are equal
   (the step's deposits precede the move and FR) and every FR walker (x, y) is one of the ABF walkers' (x, y) (FR only
   gathers the common post-move state); the ABF replay equals the stored ABF file at every stored save <= t + 1.
   Diagnostics are off in the replay (inert: tests/test_gateway_family_numba.py F8).
Pair status PASS (the frozen rule) iff 1 holds, the death-rule prefix is bitwise equal, and (>= 1 compared save, or
no FR event in the whole run, or the replay passes); PASS_EVENT_RULE iff 1 holds, only the event-rule prefix is equal
and the replay passes; FAIL otherwise.  With --smoke / --production the expected pairs (4 cells x N {512, 128} x the
smoke / production seeds) must all be present and every file must be a complete result of its planned run (run
signature).  Check verdict: PASS iff every expected pair is PASS (or PASS_EVENT_RULE under a licensed
--event-rule-amendment); AMENDMENT_REQUIRED (exit 4) iff no pair FAILS but >= 1 is PASS_EVENT_RULE without one; FAIL
(exit 1) otherwise.  Output JSON (default results/final_campaign/expA/checks/check6_pairing_<smoke|production>.json)
with pass counts and per-pair records (number of compared saves, first death / event save steps, mismatches, replay).
Idempotent: an existing record with the same input signature is kept ('up to date'); a record with a different input
signature is never overwritten without --force (exit 2 if the verdict is PASS; a FAIL / AMENDMENT_REQUIRED verdict keeps
its exit code 1 / 4 and the refusal is printed).  The input signature (record_signature) covers the sha256 of every
result file, of the frozen inputs the verdict depends on (configs/final_campaign/expA.json, the engine, the four cell
configs and their reference files, the analysis modules eqb_family.py / eqb_metrics.py), of the check tools
(expA_check_pairing.py, expA_check_smoke.py, expA_run_common.py) and the options (mode, smoke_T, replay, pipeline,
event-rule amendment); ``record_is_current`` recomputes it (the production gate of run_cells.py uses it).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import expA_run_common as C  # noqa: E402

ROOT = C.ROOT
SAVE_KEYS = ("save_step", "save_t", "save_u", "M_all", "C_all", "hist_inst", "region_frac", "events_cum",
             "lineage_visited", "gen_nuniq_run", "gen_nuniq_win", "gen_win_age_steps", "fr_deaths_cum", "fr_kd_cum",
             "fr_kc_cum", "fr_opp_event_cum", "fr_opp_death_cum", "fr_death_pos_hist_cum", "fr_birth_pos_hist_cum")
TRACE_KEYS = ("traces", "traces_y", "traces_anc", "traces_rebirth")
SNAP_KEYS = ("dep_age_C", "dep_age_M", "dep_age_M2", "dep_cross_C")
GRID_KEYS = ("save_step", "save_t", "save_u", "traces_step", "traces_t", "snap_step", "snap_t", "snap_u",
             "gen_win_age_steps")
RUN_KEYS = ("n_force_evals",)
FINAL_KEYS = ("X_final", "Y_final", "max_abs_bias")
ARM_SPECIFIC = ("fr_opp_cum", "fr_events_hist", "fr_events_hist_cum", "fr_deaths_hist", "fr_deaths_hist_cum",
                "n_fr_steps", "gen_ess_run", "gen_maxfam_run", "gen_ess_win", "gen_maxfam_win")
BOOKKEEPING = ("cfg_json", "meta_json", "wall_s", "peak_rss_mb", "n_checkpoints_written", "resumed_from")
ABF_ZERO_KEYS = ("fr_deaths_cum", "fr_kd_cum", "fr_kc_cum", "fr_opp_cum", "fr_opp_event_cum", "fr_opp_death_cum",
                 "fr_death_pos_hist_cum", "fr_birth_pos_hist_cum", "n_fr_steps")
FILE_RE = re.compile(r"^s(\d+)_(abf|fr)\.npz$")


def sha256_file(path):
    return C.sha256_file(path)


def load(path):
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    d["meta"] = json.loads(str(d["meta_json"]))
    d["cfg"] = json.loads(str(d["cfg_json"]))
    return d


def bitwise(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    return bool(np.array_equal(a, b, equal_nan=a.dtype.kind in "fc"))


def first_diff_row(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.ndim == 0:
        return None
    for i in range(a.shape[0]):
        if not bitwise(a[i], b[i]):
            return i
    return None


def leading_zero_count(v):
    v = np.asarray(v)
    nz = np.flatnonzero(v != 0)
    return int(nz[0]) if nz.size else int(v.shape[0])


def signature_agreement(a, f):
    sa, sf = a["meta"].get("run_signature"), f["meta"].get("run_signature")
    if not isinstance(sa, dict) or not isinstance(sf, dict):
        return ["run_signature missing"]
    bad = sorted(k for k in set(sa) | set(sf) if k != "method" and sa.get(k) != sf.get(k))
    if sa.get("method") != "abf" or sf.get("method") != "fr":
        bad.append(f"methods {sa.get('method')!r} / {sf.get('method')!r}")
    return bad


def compare_prefix(a, f, n):
    """Bitwise comparison of the shared arrays on the first n saves (and the trace / snapshot rows up to the n-th save
    step).  Returns (mismatch dict key -> first differing row or 'whole', list of compared keys)."""
    mism, keys = {}, []
    step = np.asarray(f["save_step"])
    last = int(step[n - 1]) if n >= 1 else -1
    for k in GRID_KEYS + RUN_KEYS:
        if k in a and k in f:
            keys.append(k)
            if not bitwise(a[k], f[k]):
                mism[k] = "whole"
    if n >= 1:
        for k in SAVE_KEYS:
            if k in a and k in f and k not in GRID_KEYS:
                keys.append(k)
                if not bitwise(a[k][:n], f[k][:n]):
                    mism[k] = first_diff_row(a[k][:n], f[k][:n])
        for kk, stepk in ((TRACE_KEYS, "traces_step"), (SNAP_KEYS, "snap_step")):
            if stepk not in a or stepk not in f:
                continue
            m = int(np.searchsorted(np.asarray(f[stepk]), last, side="right"))
            for k in kk:
                if k in a and k in f and m > 0:
                    keys.append(f"{k}[:{m}]")
                    if not bitwise(a[k][:m], f[k][:m]):
                        mism[k] = first_diff_row(a[k][:m], f[k][:m])
        fa, ff = int(np.asarray(a.get("first_arrival_step", [-1])).ravel()[0]), \
            int(np.asarray(f.get("first_arrival_step", [-1])).ravel()[0])
        if (0 <= fa <= last) or (0 <= ff <= last):
            for k in ("first_arrival_step", "first_arrival_t"):
                if k in a and k in f:
                    keys.append(k)
                    if not bitwise(a[k], f[k]):
                        mism[k] = "whole"
    return mism, keys


def compare_pair(a, f):
    """The pairing record of one (abf, fr) result pair (dicts as returned by load); no engine run."""
    rec = dict(signature_problems=signature_agreement(a, f), abf_fr_counters_nonzero=[])
    for k in ABF_ZERO_KEYS:
        if k in a and np.any(np.asarray(a[k]) != 0):
            rec["abf_fr_counters_nonzero"].append(k)
    S = int(np.asarray(f["save_step"]).shape[0])
    d, e = np.asarray(f["fr_deaths_cum"]), np.asarray(f["fr_opp_event_cum"])
    if np.any(np.diff(d) < 0) or np.any(np.diff(e) < 0):
        rec["signature_problems"].append("FR cumulative counters not monotone")
    n_death, n_event = leading_zero_count(d), leading_zero_count(e)
    step = np.asarray(f["save_step"])
    rec.update(n_saves=S, n_saves_before_first_death=n_death, n_saves_before_first_event=n_event,
               first_death_save_step=(int(step[n_death]) if n_death < S else None),
               first_event_save_step=(int(step[n_event]) if n_event < S else None),
               fr_deaths_total=int(d[-1]), fr_events_total=int(e[-1]))
    mism, keys = compare_prefix(a, f, n_death)
    rule, n_cmp, flag = "death", n_death, None
    if mism and n_event < n_death:
        mism_e, keys_e = compare_prefix(a, f, n_event)
        if not mism_e:
            rule, n_cmp, flag = "event", n_event, ("permutation-only FR event (kd + kc > 0, zero realised deaths) "
                                                   "before the first death: the prereg prefix differs only after it")
            rec["death_rule_mismatches"] = {k: v for k, v in mism.items()}
            mism, keys = mism_e, keys_e
    if not mism and int(e[-1]) == 0 and n_death == S:
        for k in FINAL_KEYS:
            if k in a and k in f:
                keys.append(k)
                if not bitwise(a[k], f[k]):
                    mism[k] = "whole"
        rec["whole_run_without_fr_event"] = True
    rec.update(rule=rule, n_compared_saves=n_cmp, permutation_flag=flag, compared_keys=keys,
               mismatches={k: (v if v is None or isinstance(v, str) else int(v)) for k, v in mism.items()})
    rec["prefix_equal"] = not mism
    return rec


# ------------------------------------------------------------------------------------------------- replay
def _rows_equal(rep, stored, keys, steps):
    """Compare replay rows with stored rows at the given save steps (both must hold them)."""
    out = {}
    ri = {int(s): i for i, s in enumerate(np.asarray(rep["save_step"]))}
    si = {int(s): i for i, s in enumerate(np.asarray(stored["save_step"]))}
    for k in keys:
        bad = [int(s) for s in steps if not bitwise(np.asarray(rep[k])[ri[int(s)]], np.asarray(stored[k])[si[int(s)]])]
        if bad:
            out[k] = bad[:5]
    return out


def replay_pair(a, f):
    """Dedicated short replay of both arms up to the first FR opportunity with a non-zero event (see docstring)."""
    import gateway_family_numba as E
    rec = dict(problems=[])
    cfg, meta = f["cfg"], f["meta"]
    N, seed, n_steps = int(meta["N"]), int(meta["seed"]), int(meta["n_steps"])
    for arm, d in (("abf", a), ("fr", f)):
        want = E.run_signature(d["cfg"], arm, N, seed, n_steps, d["save_step"], trace_steps=d["traces_step"],
                               snapshot_steps=d["snap_step"], diagnostics=bool(d["meta"].get("diagnostics", True)))
        mm = E.signature_mismatch(d["meta"].get("run_signature"), want)
        if mm:
            rec["problems"].append(f"{arm}: stored run signature not reproduced at HEAD: {mm}")
    if rec["problems"]:
        rec["passed"] = False
        return rec
    e = np.asarray(f["fr_opp_event_cum"])
    if not np.any(e > 0):
        rec.update(passed=True, note="no FR event in the whole run: nothing to replay (whole run compared)")
        return rec
    s_ev = int(np.flatnonzero(e > 0)[0])
    n_h = int(np.asarray(f["save_step"])[s_ev])
    fe = int(E.derived_knobs(E.full_cfg(cfg), N)["fr_every"])
    opp = [t + 1 for t in range(0, n_h, fe)]
    stored_fr = [int(s) for s in np.asarray(f["save_step"]) if 1 <= s <= n_h]
    grid = np.unique(np.asarray(opp + stored_fr, dtype=np.int64))
    t0 = time.time()
    scan = E.run_arm(cfg, "fr", N, seed, n_h, grid, trace_steps=[0], snapshot_steps=[n_h], diagnostics=False)
    bad = _rows_equal(scan, f, ("M_all", "C_all", "fr_deaths_cum", "fr_opp_event_cum", "fr_kd_cum", "fr_kc_cum"),
                      stored_fr)
    if bad:
        rec["problems"].append(f"FR replay differs from the stored FR file at stored saves: {bad}")
    ev = np.asarray(scan["fr_opp_event_cum"])
    gi = {int(s): i for i, s in enumerate(np.asarray(scan["save_step"]))}
    t_first = None
    for t1 in opp:
        if ev[gi[t1]] > 0:
            t_first = t1 - 1
            break
    if t_first is None:
        rec["problems"].append(f"no FR event found up to step {n_h} although the stored file has one by then")
        rec["passed"] = False
        return rec
    rec.update(first_event_opportunity_step=t_first, first_event_opportunity_t=t_first * float(cfg["h"]),
               first_event_save_step=n_h, fr_every=fe)
    sa, sf = a["meta"]["run_signature"], f["meta"]["run_signature"]
    if t_first == 0:
        ok0 = sa["x0_sha"] == sf["x0_sha"] and sa["y0_sha"] == sf["y0_sha"]
        rec["state_at_t_equal"] = bool(ok0)
        rec["state_at_t_note"] = "t = 0: the initial state (x0 / y0 sha256 of both signatures)"
        if not ok0:
            rec["problems"].append("initial states differ")
    else:
        ra = E.run_arm(cfg, "abf", N, seed, t_first, [t_first], trace_steps=[0], snapshot_steps=[t_first],
                       diagnostics=False)
        rf = E.run_arm(cfg, "fr", N, seed, t_first, [t_first], trace_steps=[0], snapshot_steps=[t_first],
                       diagnostics=False)
        eq = {k: bitwise(ra[k], rf[k]) for k in ("X_final", "Y_final", "M_all", "C_all", "max_abs_bias")}
        rec["state_at_t_equal"] = bool(all(eq.values()))
        rec["state_at_t_keys"] = eq
        if int(np.asarray(rf["fr_opp_event_cum"])[-1]) != 0:
            rec["problems"].append("FR replay to t already has an event")
        if not rec["state_at_t_equal"]:
            rec["problems"].append(f"full state at t = {t_first} differs: {eq}")
    stored_abf = [int(s) for s in np.asarray(a["save_step"]) if 1 <= s <= t_first + 1]
    g1 = np.unique(np.asarray(stored_abf + [t_first + 1], dtype=np.int64))
    ra1 = E.run_arm(cfg, "abf", N, seed, t_first + 1, g1, trace_steps=[0], snapshot_steps=[t_first + 1],
                    diagnostics=False)
    rf1 = E.run_arm(cfg, "fr", N, seed, t_first + 1, [t_first + 1], trace_steps=[0], snapshot_steps=[t_first + 1],
                    diagnostics=False)
    bad = _rows_equal(ra1, a, ("M_all", "C_all"), stored_abf)
    if bad:
        rec["problems"].append(f"ABF replay differs from the stored ABF file at stored saves: {bad}")
    acc = bitwise(ra1["M_all"][-1], rf1["M_all"][-1]) and bitwise(ra1["C_all"][-1], rf1["C_all"][-1])
    pa = set(zip(np.asarray(ra1["X_final"]).tolist(), np.asarray(ra1["Y_final"]).tolist()))
    pf = list(zip(np.asarray(rf1["X_final"]).tolist(), np.asarray(rf1["Y_final"]).tolist()))
    gathered = all(p in pa for p in pf)
    n_ev = int(np.asarray(rf1["fr_opp_event_cum"])[-1])
    rec.update(accumulators_at_t1_equal=bool(acc), fr_state_is_gather_of_abf_state=bool(gathered),
               fr_events_at_t1=n_ev, fr_deaths_at_t1=int(np.asarray(rf1["fr_deaths_cum"])[-1]),
               n_stored_fr_saves_checked=len(stored_fr), n_stored_abf_saves_checked=len(stored_abf),
               replay_wall_s=time.time() - t0)
    if not acc:
        rec["problems"].append(f"Mh / Ch differ at t + 1 = {t_first + 1}")
    if not gathered:
        rec["problems"].append(f"an FR walker at t + 1 = {t_first + 1} is not one of the ABF walkers")
    if n_ev != 1:
        rec["problems"].append(f"FR replay to t + 1 has {n_ev} events (expected exactly 1)")
    rec["passed"] = not rec["problems"]
    return rec


# ------------------------------------------------------------------------------------------------- roots
def discover(root, cells=None):
    """{(cell, N, seed): {'abf': path, 'fr': path}} under root/<cell>/N<N>/."""
    out = {}
    if not os.path.isdir(root):
        return out
    for cell in sorted(os.listdir(root)):
        if cells is not None and cell not in cells:
            continue
        cd = os.path.join(root, cell)
        if not os.path.isdir(cd):
            continue
        for nd in sorted(os.listdir(cd)):
            m = re.match(r"^N(\d+)$", nd)
            if not m or not os.path.isdir(os.path.join(cd, nd)):
                continue
            for fn in sorted(os.listdir(os.path.join(cd, nd))):
                mm = FILE_RE.match(fn)
                if mm:
                    out.setdefault((cell, int(m.group(1)), int(mm.group(1))), {})[mm.group(2)] = \
                        os.path.join(cd, nd, fn)
    return out


def expected_jobs(mode, smoke_T=4.0):
    """{(cell, N, seed, method): job} of the expA production or smoke plan."""
    P = C.load_config()
    J = C.expA_jobs(P, smoke=(mode == "smoke"), smoke_T=smoke_T)
    return {(j["variant"], int(j["N"]), int(j["seed"]), j["method"]): j for j in J}


TOOL_FILES = [os.path.abspath(__file__), os.path.join(HERE, "expA_check_smoke.py"), os.path.join(HERE,
                                                                                         "expA_run_common.py")]
CELLS_DIR = os.path.join(ROOT, "configs", "final_campaign", "cells", "expA")
EQB_MODULES = [os.path.join(ROOT, "scripts", "equal_budget", "eqb_family.py"),
               os.path.join(ROOT, "scripts", "equal_budget", "eqb_metrics.py")]


def frozen_inputs():
    """The frozen files a check-6 / check-9 verdict depends on besides the result files: the expA config, the engine,
    the four cell configs, their reference files and the analysis modules (sorted, absolute)."""
    out = [C.CONFIG, C.ENGINE_FILE] + list(EQB_MODULES)
    if os.path.isdir(CELLS_DIR):
        for fn in sorted(os.listdir(CELLS_DIR)):
            if fn.endswith(".json"):
                p = os.path.join(CELLS_DIR, fn)
                out.append(p)
                try:
                    ref = json.load(open(p)).get("reference_file")
                except Exception:  # noqa: BLE001
                    ref = None
                if ref:
                    out.append(ref if os.path.isabs(ref) else os.path.join(ROOT, ref))
    return sorted(set(os.path.abspath(p) for p in out))


def _rel(p):
    return os.path.relpath(p, ROOT) if os.path.abspath(p).startswith(ROOT + os.sep) else os.path.abspath(p)


def input_signature(paths, tool_files=None, options=None, extra=None):
    """sha256 over (relative path, file sha256) of every input result file, of the frozen inputs (``extra``, default
    frozen_inputs()), of the tool files, and the canonical JSON of ``options``.  A missing file hashes as 'absent'."""
    h = hashlib.sha256()
    for group, ps in (("inputs", sorted(os.path.abspath(p) for p in paths)),
                      ("frozen", frozen_inputs() if extra is None else sorted(extra)),
                      ("tools", list(TOOL_FILES if tool_files is None else tool_files))):
        h.update(group.encode())
        for p in ps:
            h.update(_rel(p).encode())
            h.update((sha256_file(p) if os.path.exists(p) else "absent").encode())
    h.update(json.dumps(options or {}, sort_keys=True).encode())
    return h.hexdigest()


def record_signature(paths, options=None):
    """The input signature of a check-6 / check-9 record (fixed tool list TOOL_FILES, frozen_inputs())."""
    return input_signature(paths, TOOL_FILES, options)


def record_is_current(rec, paths):
    """(bool, reason): does the stored record's input signature equal the one recomputed now over ``paths`` with the
    record's own signature_options?"""
    if not isinstance(rec, dict) or "input_signature" not in rec:
        return False, "record has no input signature"
    if "signature_options" not in rec:
        return False, "record has no signature_options (written by an older tool version)"
    sig = record_signature(paths, rec["signature_options"])
    if sig != rec["input_signature"]:
        return False, "input signature differs from the current result files / frozen inputs / tools"
    return True, "current"


def check_root(root, cells=None, expected=None, replay=True, verbose=True, event_rule_amendment=None):
    """Check 6 on every pair under root.  ``expected``: {(cell, N, seed, method): job} or None (generic root).
    ``event_rule_amendment``: the (already validated) amendment reference licensing the event rule, or None."""
    pairs = discover(root, cells)
    recs, n_pass, n_event_pass, n_fail, n_replay = [], 0, 0, 0, 0
    missing = []
    if expected is not None:
        want = sorted({(c, N, s) for (c, N, s, m) in expected})
        for key in want:
            have = pairs.get(key, {})
            for m in ("abf", "fr"):
                if m not in have:
                    missing.append(f"{key[0]}/N{key[1]}/s{key[2]}_{m}.npz")
    else:
        want = sorted(pairs)
    import gateway_family_numba as E
    for key in sorted(set(want) | set(pairs)):
        have = pairs.get(key, {})
        r = dict(cell=key[0], N=key[1], seed=key[2])
        if "abf" not in have or "fr" not in have:
            r.update(status="INCOMPLETE", present=sorted(have))
            recs.append(r)
            n_fail += 1
            continue
        a, f = load(have["abf"]), load(have["fr"])
        r.update(compare_pair(a, f))
        plan_probs = []
        if expected is not None:
            for m, d, p in (("abf", a, have["abf"]), ("fr", f, have["fr"])):
                job = expected.get((key[0], key[1], key[2], m))
                if job is None:
                    plan_probs.append(f"{m}: not a planned job")
                    continue
                mm = E.signature_mismatch(d["meta"].get("run_signature"), C.job_signature(job))
                if mm:
                    plan_probs.append(f"{m}: not the planned run (signature fields {mm})")
                if d["meta"].get("status") != "complete":
                    plan_probs.append(f"{m}: status not complete")
        r["plan_problems"] = plan_probs
        need_replay = r["prefix_equal"] and (r["n_compared_saves"] == 0 or r["rule"] == "event") \
            and r["fr_events_total"] > 0
        if need_replay:
            if replay:
                n_replay += 1
                r["replay"] = replay_pair(a, f)
            else:
                r["replay"] = dict(passed=False, problems=["replay needed but disabled (--no-replay)"])
        base_ok = (not r["signature_problems"] and not r["abf_fr_counters_nonzero"] and not plan_probs
                   and r["prefix_equal"])
        rep_ok = r.get("replay", {}).get("passed") is True
        ok_frozen = (base_ok and r["rule"] == "death"
                     and (r["n_compared_saves"] >= 1 or r.get("whole_run_without_fr_event") or rep_ok))
        ok_event = base_ok and r["rule"] == "event" and rep_ok
        r["frozen_rule"] = "PASS" if ok_frozen else "FAIL"
        r["event_rule"] = ("PASS" if ok_event else "FAIL") if r["rule"] == "event" else "not applied"
        r["status"] = "PASS" if ok_frozen else ("PASS_EVENT_RULE" if ok_event else "FAIL")
        ok = ok_frozen or ok_event
        n_pass += ok_frozen
        n_event_pass += ok_event
        n_fail += not ok
        recs.append(r)
        if verbose:
            print(f"{key[0]} N{key[1]} s{key[2]}: {r['status']} rule {r['rule']}, {r['n_compared_saves']} compared "
                  f"saves of {r['n_saves']} (first death save step {r['first_death_save_step']}, first event save "
                  f"step {r['first_event_save_step']})"
                  + (f", replay {'PASS' if r['replay'].get('passed') else 'FAIL'}" if "replay" in r else "")
                  + ("" if ok else f"  mismatches {r['mismatches']} {r['signature_problems']} {plan_probs} "
                                   f"{r.get('replay', {}).get('problems', '')}"), flush=True)
    n_expected = len(want)
    complete = n_fail == 0 and not missing and n_pass + n_event_pass == n_expected and n_expected > 0
    if complete and n_event_pass == 0:
        verdict, note = "PASS", "every pair passes the frozen prereg rule (first realised death)"
    elif complete and event_rule_amendment:
        verdict = "PASS"
        note = (f"{n_pass} pair(s) pass the frozen rule and {n_event_pass} pass only the event rule, licensed by the "
                f"dated amendment {event_rule_amendment!r}")
    elif complete:
        verdict = "AMENDMENT_REQUIRED"
        note = (f"{n_event_pass} pair(s) FAIL the frozen prereg rule (a permutation-only FR event before the first "
                f"realised death, with a save in between) and pass only the event rule + replay; the event rule is not "
                f"preregistered: write a dated amendment, then re-run with --event-rule-amendment <ref>")
    else:
        verdict, note = "FAIL", f"{n_fail} pair(s) FAIL, {len(missing)} file(s) missing"
    return dict(check="6", root=root, n_expected_pairs=n_expected, n_pairs_found=len(pairs), n_pass=n_pass,
                n_pass_event_rule_only=n_event_pass, n_fail=n_fail, n_replays=n_replay, missing_files=missing,
                min_compared_saves=min([r.get("n_compared_saves", 0) for r in recs if "n_compared_saves" in r],
                                       default=None),
                n_event_rule=sum(r.get("rule") == "event" for r in recs),
                event_rule_amendment=event_rule_amendment, verdict=verdict, verdict_note=note, pairs=recs)


def write_record(rec, out, inputs, force=False, dry_run=False, options=None, tool=None):
    """Idempotent write: same input signature -> keep ('up to date'); different -> refuse unless force.  The signature
    is record_signature(inputs, options); options and the input file list are stored in the record."""
    options = dict(options or {})
    sig = record_signature(inputs, options)
    rec = dict(rec, input_signature=sig, signature_options=options,
               signature_inputs=sorted(_rel(p) for p in inputs), signature_frozen_inputs=[_rel(p) for p in
                                                                                       frozen_inputs()],
               signature_tools=[_rel(p) for p in TOOL_FILES], generated_utc=C.utc(),
               tool=tool or os.path.relpath(__file__, ROOT), provenance=C.provenance())
    if dry_run:
        return "dry-run", rec
    if os.path.exists(out):
        try:
            old = json.load(open(out))
        except Exception:  # noqa: BLE001
            old = {}
        if old.get("input_signature") == sig and not force:
            return "up to date", old
        if old.get("input_signature") != sig and not force:
            return "REFUSED", rec
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    tmp = out + f".tmp{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(rec, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    os.replace(tmp, out)
    return "written", rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--smoke", action="store_true", help="results/final_campaign/expA/smoke/T<smoke_T> (gate)")
    g.add_argument("--production", action="store_true", help="results/final_campaign/expA (audit, 256 pairs)")
    g.add_argument("--root", default=None, help="any root holding <cell>/N<N>/s<seed>_<abf|fr>.npz (no plan check)")
    ap.add_argument("--smoke-T", type=float, default=4.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-replay", action="store_true")
    ap.add_argument("--force", action="store_true", help="overwrite a record with a different input signature")
    ap.add_argument("--dry-run", action="store_true", help="list the pairs found and the plan, run nothing")
    ap.add_argument("--event-rule-amendment", default=None, metavar="REF",
                    help="license the (non-preregistered) event rule for permutation-only first FR events; REF must "
                         "occur in the dated-amendment section of docs/final_campaign/PREREGISTRATION.md")
    ap.add_argument("--out-root", default=None, help=argparse.SUPPRESS)      # tests only (driver's --out-root)
    ap.add_argument("--prereg", default=None, help=argparse.SUPPRESS)        # tests only (amendment lookup)
    a = ap.parse_args(argv)
    C.check_engine()
    if a.event_rule_amendment is not None and not C.amendment_ref_found(a.event_rule_amendment, a.prereg):
        print(f"REFUSED: --event-rule-amendment {a.event_rule_amendment!r} does not occur in the dated-amendment section "
              f"of {a.prereg or C.PREREG}; write the amendment first", flush=True)
        return 2
    if a.smoke:
        root, mode = C.smoke_root(a.out_root, a.smoke_T), "smoke"
    elif a.production:
        root, mode = C.prod_root(a.out_root), "production"
    else:
        root, mode = os.path.abspath(a.root), None
    out = a.out or os.path.join(C.prod_root(a.out_root), "checks", f"check6_pairing_{mode or 'custom'}.json")
    P = C.load_config()
    cells = [v["name"] for v in P["variants"]] if mode else None
    expected = expected_jobs(mode, a.smoke_T) if mode else None
    pairs = discover(root, cells)
    if a.dry_run:
        n_exp = len({(c, N, s) for (c, N, s, m) in expected}) if expected else None
        print(f"check 6 dry run: root {root}; {len(pairs)} pairs found"
              + (f"; {n_exp} expected" if n_exp is not None else "") + f"; output {out}")
        for k, v in list(sorted(pairs.items()))[:10]:
            print(f"  {k}: {sorted(v)}")
        return 0
    rec = check_root(root, cells, expected, replay=not a.no_replay, event_rule_amendment=a.event_rule_amendment)
    inputs = [p for v in pairs.values() for p in v.values()]
    st, rec2 = write_record(rec, out, inputs, force=a.force,
                            options=c6_options(mode, a.smoke_T if mode == "smoke" else None, not a.no_replay,
                                               a.event_rule_amendment))
    print(f"check 6: {rec['verdict']} -- {rec['n_pass']}/{rec['n_expected_pairs']} pairs PASS the frozen rule, "
          f"{rec['n_pass_event_rule_only']} PASS only the event rule, {rec['n_fail']} FAIL ({rec['n_replays']} replays, "
          f"min compared saves {rec['min_compared_saves']}); record {st}: {out}")
    print(f"  {rec['verdict_note']}")
    if st == "REFUSED":
        print("  an existing record with a DIFFERENT input signature is kept (use --force to replace it)")
    return record_exit(rec["verdict"], st)


def c6_options(mode, smoke_T, replay, amendment):
    """Signature options of a check-6 record (identical whichever tool writes it)."""
    return dict(check="6", mode=mode, smoke_T=(float(smoke_T) if smoke_T is not None else None), replay=bool(replay),
                event_rule_amendment=amendment)


def verdict_exit(verdict):
    """Exit code of a check verdict: PASS 0, AMENDMENT_REQUIRED 4, anything else 1."""
    return 0 if verdict == "PASS" else (4 if verdict == "AMENDMENT_REQUIRED" else 1)


def record_exit(verdict, *write_states):
    """Exit code of a check run: a non-PASS verdict wins (FAIL 1, AMENDMENT_REQUIRED 4) even when the record write was
    refused; a PASS whose record was REFUSED (a different record kept on disk) exits 2; otherwise 0."""
    if verdict != "PASS":
        return verdict_exit(verdict)
    return 2 if "REFUSED" in write_states else 0


if __name__ == "__main__":
    sys.exit(main())
