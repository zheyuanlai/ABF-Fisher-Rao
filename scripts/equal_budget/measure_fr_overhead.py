#!/usr/bin/env python
"""Computational overhead of uniform FR on the two equal-budget engines, broken down into its components.

    python scripts/equal_budget/measure_fr_overhead.py [--cpus 1,2,8,9,18,25] [--reps 5] [--target-s 12]
    python scripts/equal_budget/measure_fr_overhead.py --summarize      # re-merge existing parts, print tables

Driver (default): picks idle logical CPUs on distinct physical cores (or takes --cpus), launches ONE worker per
(system, N) case pinned with ``taskset -c <cpu>`` to a single logical CPU (6 cases -> 6 CPUs, never more than 8),
merges the worker parts into results/equal_budget_v2/fr_overhead/fr_overhead.json together with the FR/ABF wall
ratios of the production ledgers (read only), and prints the summary tables used in docs/equal_budget/FR_OVERHEAD.md.

Worker (--worker --system {gateway,lta300} --N N --cpu C --out PART), all on its one pinned CPU:
 (1) FULL SHORT RUNS of the engine's own run_arm with the production engine_cfg of
     configs/equal_budget_v2/{gateway,lta_300K}_production.json, the production save-grid rule, seed = the first
     production seed, and n_steps shortened so that one ABF run takes ~target_s seconds (calibrated on the spot).
     Four variants -- ABF / ABF+FR x diagnostics off / on -- are each run `reps` times in an interleaved order that is
     rotated every repetition (drift cancels in the within-repetition ratios).  Every run is deterministic (same seed,
     same n_steps), so the reps repeat identical work and their spread is pure timing noise.  Per run: wall
     (perf_counter), thread CPU time, the engine's own wall_s, 1-min load average before/after, busy fraction of the
     pinned CPU and of its SMT sibling during the run (/proc/stat), nonvoluntary context switches, FR opportunity and
     event counts.  LTA only: fr_start_steps = 5 (production 20000 = 4 t.u.) so that FR is active from the first
     opportunity of the short run, as it is in 93-100 % of the steps of a production run; everything else is the
     production engine_cfg.
 (2) MICRO-BENCHMARKS of the FR components per FR opportunity, as njit loops that call the engines' own njit helpers
     (no Python call overhead per iteration), on representative population states: the final states of the
     production FR arm at the same N (first 4 seeds, read only) and the end state of this worker's own short FR run.
       gateway:  KDE = gateway_ladder_numba.kde_density; score = uniform_scores; selection = fr_resample (engine
                 function, with the engine's uniform-block layout and fully ramped rate gamma); gather = an op-for-op
                 copy of the slot gather in _advance (diagnostics off; it is inline there), charged per opportunity
                 with >= 1 candidate; RNG = the Python-side PCG64 draw of the FR uniform blocks (fr_rng.random(n_opp L)
                 per chunk of 2^23 // N steps, as run_arm does).
       LTA:      post-move CV = op-for-op copy of the CV/binning loop in _run_chunk step (7) (inline there, uses the
                 engine's _wrap/_bin); KDE and score = an op-for-op split of the engine's _fr_score into its density
                 part (histogram, ascending touched-bin sort, unnormalised wrapped-kernel product, clamp, normalise)
                 and its score part (KL, periodic interpolation, centring, 3 x clip-recentre, std/absmax) -- the split
                 is asserted BITWISE equal to _fr_score on every state, and the engine's whole _fr_score is timed too;
                 selection = fr_select_native (engine function); copy = op-for-op copy of the event-application loop
                 (histograms, whole-molecule copy, ancestor labels, walker labels), charged per realised event;
                 RNG = gen_fr.random((n_opp, N + 2 cap)) per chunk of 1e6 // N steps, as run_arm does; diagnostics-only
                 FR work = the engine's _true_event on the post-move positions at every opportunity.
     Predicted FR overhead per opportunity = sum of the components; it is compared with the measured
     (FR - ABF, diagnostics off) wall difference divided by the number of opportunities of the short run.

Noise.  The host (2 x EPYC 9554, 256 logical CPUs, SMT-2) is shared with other users' jobs, so a pinned CPU's SMT
sibling may be busy; the busy fraction of the pinned CPU and of its sibling, the 1-min load average and the
nonvoluntary context switches are recorded per run.  (In the 2026-10-10 measurement the confirmation production run
had finished, the load average was 19-23 and every sibling was <= 2 % busy; production ran under ~110-120 concurrent
single-threaded processes.)  The within-repetition paired ratios and their min/max over reps are reported;
differences below ~0.3 % per case are not resolved.

Environment (as the production runs): CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1
NUMBA_CACHE_DIR=~/.cache/numba_eqb.  Writes only under results/equal_budget_v2/fr_overhead/.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import subprocess
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _k in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = "1"
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.expanduser("~/.cache/numba_eqb"))

import numpy as np  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
OUT_DIR = os.path.join(ROOT, "results", "equal_budget_v2", "fr_overhead")
OUT_JSON = os.path.join(OUT_DIR, "fr_overhead.json")
PART_DIR = os.path.join(OUT_DIR, "parts")
CFG_FILES = {"gateway": "gateway_production.json", "lta300": "lta_300K_production.json"}
PROD_DIRS = {"gateway": "gateway", "lta300": "lta_T300", "lta150": "lta_T150"}
CASES = [("gateway", 2), ("gateway", 16), ("gateway", 2048), ("lta300", 2), ("lta300", 16), ("lta300", 1024)]
VARIANTS = [("abf", False), ("fr", False), ("abf", True), ("fr", True), ("frsched", False)]
# 'frsched' = the FR arm with fr_every = n_steps (one FR opportunity in the whole run, at its first scheduled step):
# it pays the per-step FR scheduling check and nothing else, so frsched - abf isolates that per-step cost (a lower
# bound: the divisor is larger than in production, which can shorten the integer division)
LTA_FR_START_TIMING = 5
N_PROD_STATES = 4
MAX_CPUS = 8


# =====================================================================================================================
# host measurements
# =====================================================================================================================
def cpu_times():
    """{cpu: (total jiffies, idle jiffies)} from /proc/stat."""
    d = {}
    with open("/proc/stat") as f:
        for line in f:
            if line.startswith("cpu") and line[3].isdigit():
                p = line.split()
                v = [int(x) for x in p[1:]]
                d[int(p[0][3:])] = (sum(v), v[3] + v[4])
    return d


def busy_between(a, b, cpu):
    tot = b[cpu][0] - a[cpu][0]
    return float("nan") if tot <= 0 else 1.0 - (b[cpu][1] - a[cpu][1]) / tot


def sibling(cpu):
    try:
        with open(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list") as f:
            s = f.read().strip()
        ids = []
        for part in s.split(","):
            if "-" in part:
                lo, hi = part.split("-")
                ids += list(range(int(lo), int(hi) + 1))
            else:
                ids.append(int(part))
        others = [i for i in ids if i != cpu]
        return others[0] if others else None
    except OSError:
        return None


def nonvol_ctxt():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("nonvoluntary_ctxt_switches"):
                return int(line.split()[1])
    return -1


def pick_idle_cpus(n, sample_s=4.0):
    """n idle logical CPUs (busy < 5 % over sample_s) on distinct physical cores, least-busy SMT sibling first."""
    a = cpu_times()
    time.sleep(sample_s)
    b = cpu_times()
    busy = {c: busy_between(a, b, c) for c in a}
    cand = []
    seen_core = set()
    for c in sorted(busy, key=lambda c: (busy[c], busy.get(sibling(c), 1.0) if sibling(c) is not None else 0.0)):
        if busy[c] >= 0.05 or c == 0 or sibling(c) == 0:      # CPU 0 (and its sibling) services interrupts
            continue
        sib = sibling(c)
        core = tuple(sorted([c] + ([sib] if sib is not None else [])))
        if core in seen_core:
            continue
        seen_core.add(core)
        cand.append((c, busy[c], busy.get(sib, float("nan")) if sib is not None else float("nan")))
    cand.sort(key=lambda t: t[2])
    n_idle = sum(v < 0.05 for v in busy.values())
    n_busy = sum(v > 0.5 for v in busy.values())
    return [c for c, _, _ in cand[:n]], dict(n_logical=len(busy), n_idle_logical=n_idle, n_busy_logical=n_busy,
                                             picked=[dict(cpu=c, busy=round(x, 3), sibling_busy=round(y, 3))
                                                     for c, x, y in cand[:n]])


def git_commit():
    try:
        return subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=20).stdout.strip()
    except Exception:
        return "unknown"


def stats(v):
    v = np.asarray(v, dtype=np.float64)
    if v.size == 0:
        return dict(n=0)
    return dict(n=int(v.size), median=float(np.median(v)), min=float(v.min()), max=float(v.max()),
                q25=float(np.percentile(v, 25)), q75=float(np.percentile(v, 75)), mean=float(v.mean()),
                rel_spread=float((v.max() - v.min()) / np.median(v)) if np.median(v) != 0 else float("nan"))


def atomic_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + f".tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_prod_cfg(system):
    return json.load(open(os.path.join(ROOT, "configs", "equal_budget_v2", CFG_FILES[system])))


def save_grid(E, P, n_steps):
    """scripts/equal_budget/run_ladder.save_grid (the production save grid): budget grid (200 uniform + 24 log-spaced
    u) union the frozen physical-time checkpoints t <= T_N at the nearest integration step."""
    steps, _ = E.budget_save_grid(n_steps)
    h = float(P["engine_cfg"]["h"])
    extra = [int(round(t / h)) for t in P["physical_checkpoints_t"] if 0 < round(t / h) <= n_steps]
    return np.unique(np.concatenate([steps, np.asarray(extra, dtype=np.int64)]))


# =====================================================================================================================
# micro-benchmark kernels (built lazily inside the worker so the driver never imports numba)
# =====================================================================================================================
def build_gateway_kernels():
    from numba import njit
    import gateway_ladder_numba as G

    @njit
    def kde_loop(X, n, kern, r, dx, p, hist, K):
        acc = 0.0
        for _ in range(K):
            acc += G.kde_density(X, n, kern, r, dx, p, hist)
        return acc

    @njit
    def score_loop(X, n, p, dx, clip, S, K):
        acc = 0.0
        for _ in range(K):
            acc += G.uniform_scores(X, n, p, dx, clip, S)
        return acc

    @njit
    def select_loop(S, n, g, dt_fr, cap, sel, die_c, clone_c, pool, ubuf, L, nblk, K):
        n_ev = 0
        n_kd = 0
        n_kc = 0
        for k in range(K):
            kd, kc, nu = G.fr_resample(S, n, g, dt_fr, cap, sel, die_c, clone_c, pool, ubuf, (k % nblk) * L, False)
            if kd + kc > 0:
                n_ev += 1
            n_kd += kd
            n_kc += kc
        return n_ev, n_kd, n_kc

    @njit
    def first_event_sel(S, n, g, dt_fr, cap, sel, die_c, clone_c, pool, ubuf, L, nblk):
        for k in range(nblk):
            kd, kc, nu = G.fr_resample(S, n, g, dt_fr, cap, sel, die_c, clone_c, pool, ubuf, k * L, False)
            if kd + kc > 0:
                return k
        return -1

    @njit
    def gather_loop(X, Y, sel, n, tmpx, tmpy, Xs, Ys, ocnt, K):
        # op-for-op copy of the gather of gateway_ladder_numba._advance (FR opportunity with >= 1 candidate,
        # diagnostics off), writing into the scratch Xs / Ys so that the representative state stays fixed
        tot = 0
        for _ in range(K):
            for i in range(n):
                tmpx[i] = X[sel[i]]
                tmpy[i] = Y[sel[i]]
            for i in range(n):
                Xs[i] = tmpx[i]
                Ys[i] = tmpy[i]
            for i in range(n):
                ocnt[i] = 0
            for i in range(n):
                ocnt[sel[i]] += 1
            nde = 0
            for i in range(n):
                if ocnt[i] == 0:
                    nde += 1
            tot += nde
        return tot

    return dict(kde_loop=kde_loop, score_loop=score_loop, select_loop=select_loop, first_event_sel=first_event_sel,
                gather_loop=gather_loop)


def build_lta_kernels():
    from numba import njit
    import lta_ladder_numba as E
    EPS = E.EPS
    NW = E.NW
    A_BH, A_DH = E.A_BH, E.A_DH

    @njit(error_model="numpy")
    def cv_loop(q, N, cphi, dphi, ng, xn, phin, jn, K):
        # op-for-op copy of the post-move CV / binning loop of lta_ladder_numba._run_chunk step (7)
        s = 0
        for _ in range(K):
            for i in range(N):
                x = (q[i, 0, 0] + q[i, 1, 0]) * 0.5
                xn[i] = x
                ph = E._wrap(cphi * x)
                phin[i] = ph
                jn[i] = E._bin(ph, dphi, ng)
            s += jn[0]
        return s

    @njit(error_model="numpy")
    def kde_part(jn, N, ng, dphi, Kk, cnt, pk, touched):
        # density part of lta_ladder_numba._fr_score, op for op
        nt = 0
        for i in range(N):
            j = jn[i]
            if cnt[j] == 0.0:
                touched[nt] = j
                nt += 1
            cnt[j] += 1.0
        for t in range(1, nt):
            v = touched[t]
            u = t - 1
            while u >= 0 and touched[u] > v:
                touched[u + 1] = touched[u]
                u -= 1
            touched[u + 1] = v
        for k in range(ng):
            pk[k] = 0.0
        for t in range(nt):
            j = touched[t]
            c = cnt[j]
            for k in range(ng):
                pk[k] += c * Kk[j, k]
            cnt[j] = 0.0
        tot = 0.0
        for k in range(ng):
            v = pk[k]
            if v < 0.0:
                v = 0.0
            pk[k] = v
            tot += v
        mass = tot * dphi
        if mass < EPS:
            mass = EPS
        for k in range(ng):
            pk[k] = pk[k] / mass
        return nt

    @njit(error_model="numpy")
    def score_part(phin, N, ng, dphi, pk, grid0, qv, lq, clip, raw, sc):
        # score part of lta_ladder_numba._fr_score, op for op
        kl = 0.0
        for k in range(ng):
            v = pk[k]
            kl += v * (math.log(v if v > EPS else EPS) - lq)
        kl = kl * dphi
        for i in range(N):
            xx = (phin[i] - grid0) / dphi
            f0 = np.floor(xx)
            fr = xx - f0
            i0 = np.int64(f0)
            p0 = pk[i0 % ng]
            p1 = pk[(i0 + 1) % ng]
            pat = (1.0 - fr) * p0 + fr * p1
            qat = (1.0 - fr) * qv + fr * qv
            raw[i] = (math.log(pat if pat > EPS else EPS) - math.log(qat if qat > EPS else EPS)) - kl
        m = 0.0
        for i in range(N):
            m += raw[i]
        m = m / N
        for i in range(N):
            sc[i] = raw[i] - m
        for _ in range(3):
            m = 0.0
            for i in range(N):
                v = sc[i]
                if v < -clip:
                    v = -clip
                elif v > clip:
                    v = clip
                sc[i] = v
                m += v
            m = m / N
            for i in range(N):
                sc[i] = sc[i] - m
        m = 0.0
        amax = 0.0
        for i in range(N):
            m += sc[i]
            if abs(sc[i]) > amax:
                amax = abs(sc[i])
        m = m / N
        v2 = 0.0
        for i in range(N):
            d = sc[i] - m
            v2 += d * d
        return math.sqrt(v2 / N), amax

    @njit(error_model="numpy")
    def kde_loop(jn, N, ng, dphi, Kk, cnt, pk, touched, K):
        s = 0
        for _ in range(K):
            s += kde_part(jn, N, ng, dphi, Kk, cnt, pk, touched)
        return s

    @njit(error_model="numpy")
    def score_loop(phin, N, ng, dphi, pk, grid0, qv, lq, clip, raw, sc, K):
        s = 0.0
        for _ in range(K):
            a, b = score_part(phin, N, ng, dphi, pk, grid0, qv, lq, clip, raw, sc)
            s += a
        return s

    @njit(error_model="numpy")
    def frscore_loop(phin, jn, N, ng, dphi, Kk, grid0, qv, lq, clip, cnt, pk, raw, sc, touched, K):
        s = 0.0
        for _ in range(K):
            a, b = E._fr_score(phin, jn, N, ng, dphi, Kk, grid0, qv, lq, clip, cnt, pk, raw, sc, touched)
            s += a
        return s

    @njit(error_model="numpy")
    def select_loop(sc, N, cap, rate, dt_eff, frb, nblk, cand, dsel, ssel, dp, cdf, K):
        n_ev = 0
        n_tot = 0
        for k in range(K):
            n = E.fr_select_native(sc, N, cap, rate, dt_eff, frb[k % nblk], cand, dsel, ssel, dp, cdf)
            if n > 0:
                n_ev += 1
            n_tot += n
        return n_ev, n_tot

    @njit(error_model="numpy")
    def copy_loop(nev, dsel, ssel, jn, acc, q, anc, ancw, wst, K):
        # op-for-op copy of the event-application loop of _run_chunk step (7) (record_events off), on scratch state
        s = 0
        for _ in range(K):
            for t in range(nev):
                d = dsel[t]
                s_ = ssel[t]
                acc[A_BH, jn[s_]] += 1.0
                acc[A_DH, jn[d]] += 1.0
                for b in range(2):
                    for c3 in range(3):
                        q[d, b, c3] = q[s_, b, c3]
                anc[d] = anc[s_]
                ancw[d] = ancw[s_]
                for w in range(NW):
                    wst[w, d] = wst[w, s_]
            s += nev
        return s

    @njit(error_model="numpy")
    def true_event_loop(nxt, N, xn, phin, a, wh, cm, wst, ist, K):
        # diagnostics-only FR work of _run_chunk step (7): _true_event on the post-move positions of every walker
        for _ in range(K):
            for i in range(N):
                E._true_event(nxt, i, xn[i], E._region(phin[i], a, wh, cm), a, wst, ist)
        return ist[0]

    return dict(cv_loop=cv_loop, kde_part=kde_part, score_part=score_part, kde_loop=kde_loop, score_loop=score_loop,
                frscore_loop=frscore_loop, select_loop=select_loop, copy_loop=copy_loop,
                true_event_loop=true_event_loop)


def time_loop(fn, args, target_s=0.25, reps=7, k0=16):
    """Median / min / max seconds per iteration of fn(*args, K) over `reps` timings of ~target_s each."""
    fn(*args, 1)                                # compile / warm
    K = k0
    while True:
        t0 = time.perf_counter()
        fn(*args, K)
        dt = time.perf_counter() - t0
        if dt >= 0.02 or K >= 1 << 30:
            break
        K *= 4
    K = max(1, int(K * target_s / max(dt, 1e-9)))
    per = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(*args, K)
        per.append((time.perf_counter() - t0) / K)
    s = stats(per)
    s["K"] = K
    return s


# =====================================================================================================================
# worker: full short runs
# =====================================================================================================================
def run_variant(system, E, P, N, n_steps, method, diag, seed):
    eng_method = "abf" if method == "abf" else "fr"
    if system == "gateway":
        save = save_grid(E, P, n_steps)
        kw = dict(diagnostics=diag)
        gcfg = dict(P["engine_cfg"])
        if method == "frsched":
            gcfg["fr_interval_t"] = n_steps * float(gcfg["h"])        # fr_every = n_steps: one opportunity (step 0)
        call = lambda: E.run_arm(gcfg, eng_method, N, seed, n_steps, save, **kw)  # noqa: E731
    else:
        ov = dict(P["engine_cfg"])
        ov.update(n_steps=int(n_steps), fr_start_steps=LTA_FR_START_TIMING, diagnostics=bool(diag))
        if method == "frsched":
            ov["fr_every"] = int(n_steps)                              # one opportunity (nxt = fr_start)
        cfg = E.make_cfg(P["T_K"], N, seed, **ov)
        save = save_grid(E, P, n_steps)
        call = lambda: E.run_arm(cfg, eng_method, save)  # noqa: E731
    cpu = os.sched_getaffinity(0)
    my = next(iter(cpu)) if len(cpu) == 1 else None
    sib = sibling(my) if my is not None else None
    la0 = os.getloadavg()[0]
    nv0 = nonvol_ctxt()
    ct0 = cpu_times()
    c0 = time.thread_time()
    t0 = time.perf_counter()
    res = call()
    wall = time.perf_counter() - t0
    cpu_s = time.thread_time() - c0
    ct1 = cpu_times()
    nv1 = nonvol_ctxt()
    la1 = os.getloadavg()[0]
    rec = dict(method=method, diag=bool(diag), N=int(N), n_steps=int(n_steps), wall_s=wall, thread_cpu_s=cpu_s,
               engine_wall_s=float(res["wall_s"]), ns_per_walker_step=wall / (N * n_steps) * 1e9,
               loadavg1_before=la0, loadavg1_after=la1, nonvoluntary_ctxt_switches=nv1 - nv0,
               cpu=my, cpu_busy=busy_between(ct0, ct1, my) if my is not None else None,
               sibling_cpu=sib, sibling_busy=busy_between(ct0, ct1, sib) if sib is not None else None,
               started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - wall)))
    if system == "gateway":
        rec.update(n_fr_opps=int(res["n_fr_steps"]), fr_deaths=int(res["fr_deaths_cum"][-1]),
                   fr_opps_with_candidate=int(res["fr_opp_event_cum"][-1]),
                   fr_candidates=int(res["fr_kd_cum"][-1] + res["fr_kc_cum"][-1]))
        state = dict(X=np.asarray(res["X_final"]).copy(), Y=np.asarray(res["Y_final"]).copy())
    else:
        rec.update(n_fr_opps=int(res["n_fr_opps"]), fr_deaths=int(res["total_replacement_events"]),
                   fr_opps_with_event=int(res["cum_opps_with_event"][-1]) if len(res["cum_opps_with_event"]) else 0)
        state = dict(q=np.asarray(res["q_final"]).copy())
    return rec, state


def full_runs(system, N, reps, target_s, log):
    P = load_prod_cfg(system)
    seed = int(P["seeds"][0])
    if system == "gateway":
        import gateway_ladder_numba as E
        fr_every = int(round(P["engine_cfg"]["fr_interval_t"] / P["engine_cfg"]["h"]))
        cal_ws = 1.5e7
    else:
        import lta_ladder_numba as E
        fr_every = int(P["engine_cfg"]["fr_every"])
        cal_ws = 1.5e6
    # calibration (also the warm-up of each variant: numba cache load, first-call dispatch)
    n_cal = max(10 * fr_every, int(round(cal_ws / N / fr_every)) * fr_every)
    cal = []
    for method, diag in VARIANTS:
        r, _ = run_variant(system, E, P, N, n_cal, method, diag, seed)
        cal.append(r)
        log(f"  warm-up {method} diag={diag}: {r['wall_s']:.2f} s  ({r['ns_per_walker_step']:.1f} ns/ws)")
    r, _ = run_variant(system, E, P, N, n_cal, "abf", False, seed)     # calibration proper (after the warm-ups)
    cal.append(r)
    ns_ws = r["ns_per_walker_step"]
    n_steps = max(20 * fr_every, int(round(target_s / (ns_ws * 1e-9) / N / fr_every)) * fr_every)
    log(f"  calibrated n_steps = {n_steps} (N {N}, target {target_s} s at {ns_ws:.1f} ns/ws)")
    runs = []
    end_state = None
    for rep in range(reps):
        order = VARIANTS[rep % len(VARIANTS):] + VARIANTS[:rep % len(VARIANTS)]
        for pos, (method, diag) in enumerate(order):
            r, st = run_variant(system, E, P, N, n_steps, method, diag, seed)
            r.update(rep=rep, position=pos)
            runs.append(r)
            if method == "fr" and not diag:
                end_state = st
            log(f"  rep {rep} {method:3s} diag={int(diag)}: {r['wall_s']:.3f} s  {r['ns_per_walker_step']:.2f} ns/ws  "
                f"load {r['loadavg1_before']:.0f}  sib {r['sibling_busy']:.2f}  nvcs {r['nonvoluntary_ctxt_switches']}")
    return dict(seed=seed, fr_every=fr_every, n_steps=n_steps, n_cal=n_cal, calibration=cal, runs=runs,
                timing_overrides=({} if system == "gateway" else dict(fr_start_steps=LTA_FR_START_TIMING))), end_state


def summarize_runs(fr):
    runs = fr["runs"]
    reps = sorted({r["rep"] for r in runs})
    by = {(r["rep"], r["method"], r["diag"]): r for r in runs}
    out = {}
    for method, diag in VARIANTS:
        key = f"{method}_diag_{'on' if diag else 'off'}"
        w = [by[(k, method, diag)]["wall_s"] for k in reps]
        c = [by[(k, method, diag)]["thread_cpu_s"] for k in reps]
        ns = [by[(k, method, diag)]["ns_per_walker_step"] for k in reps]
        if method == "frsched":
            key = "fr_sched_only_diag_off"
        out[key] = dict(wall_s=stats(w), thread_cpu_s=stats(c), ns_per_walker_step=stats(ns))

    def paired(num, den, field="wall_s"):
        return stats([by[(k,) + num][field] / by[(k,) + den][field] - 1.0 for k in reps])

    out["ratios_minus_1"] = dict(
        diag_cost_abf=paired(("abf", True), ("abf", False)),
        diag_cost_fr=paired(("fr", True), ("fr", False)),
        fr_overhead_diag_off=paired(("fr", False), ("abf", False)),
        fr_overhead_diag_on=paired(("fr", True), ("abf", True)),
        diag_cost_abf_cpu=paired(("abf", True), ("abf", False), "thread_cpu_s"),
        fr_overhead_diag_off_cpu=paired(("fr", False), ("abf", False), "thread_cpu_s"),
        fr_overhead_diag_on_cpu=paired(("fr", True), ("abf", True), "thread_cpu_s"),
        fr_sched_only=paired(("frsched", False), ("abf", False)),
    )
    ex = [(by[(k, "fr", False)]["wall_s"] - by[(k, "frsched", False)]["wall_s"]) / by[(k, "abf", False)]["wall_s"]
          for k in reps]
    out["ratios_minus_1"]["fr_execution_only"] = stats(ex)        # (fr - frsched) / abf
    d = [by[(k, "fr", False)]["wall_s"] - by[(k, "abf", False)]["wall_s"] for k in reps]
    n_opp = by[(reps[0], "fr", False)]["n_fr_opps"]
    out["fr_minus_abf_diag_off_s"] = stats(d)
    out["n_fr_opps_per_run"] = int(n_opp)
    out["measured_fr_overhead_us_per_opp"] = stats(np.asarray(d) / max(n_opp, 1) * 1e6)
    n_opp_s = by[(reps[0], "frsched", False)]["n_fr_opps"]
    de = [by[(k, "fr", False)]["wall_s"] - by[(k, "frsched", False)]["wall_s"] for k in reps]
    ds = [by[(k, "frsched", False)]["wall_s"] - by[(k, "abf", False)]["wall_s"] for k in reps]
    out["n_fr_opps_frsched"] = int(n_opp_s)
    out["measured_fr_execution_us_per_opp"] = stats(np.asarray(de) / max(n_opp - n_opp_s, 1) * 1e6)
    out["measured_sched_ns_per_step"] = stats(np.asarray(ds) / by[(reps[0], "abf", False)]["n_steps"] * 1e9)
    dd = [(by[(k, "fr", True)]["wall_s"] - by[(k, "fr", False)]["wall_s"])
          - (by[(k, "abf", True)]["wall_s"] - by[(k, "abf", False)]["wall_s"]) for k in reps]
    out["fr_only_diag_extra_s"] = stats(dd)
    out["loadavg1"] = stats([r["loadavg1_before"] for r in runs] + [r["loadavg1_after"] for r in runs])
    out["sibling_busy"] = stats([r["sibling_busy"] for r in runs if r["sibling_busy"] is not None])
    out["cpu_over_wall"] = stats([r["thread_cpu_s"] / r["wall_s"] for r in runs])
    out["nonvoluntary_ctxt_switches"] = stats([r["nonvoluntary_ctxt_switches"] for r in runs])
    return out


# =====================================================================================================================
# worker: micro-benchmarks
# =====================================================================================================================
def micro_gateway(N, P, states, n_steps_run, log):
    import gateway_ladder_numba as G
    Kn = build_gateway_kernels()
    c = G.full_cfg(P["engine_cfg"])
    kn = G.derived_knobs(c, N)
    dx = G.grid_dx()
    kern, r = G.gaussian_kernel_np(float(c["eta"]), dx)
    cap, L, fr_every, dt_fr = kn["cap"], kn["fr_block"], kn["fr_every"], kn["dt_fr"]
    g = float(c["gamma"])                                    # fully ramped rate
    clip = float(c["score_clip"])
    nblk = 4096
    ubuf = np.random.Generator(np.random.PCG64(12345)).random(nblk * L)
    chunk_steps = max(1, (1 << 23) // N)
    n_opp_chunk = max(1, -(-chunk_steps // fr_every))
    out = dict(fr_every=fr_every, cap=cap, fr_block_L=L, rate_g=g, dt_fr=dt_fr, nblk_uniforms=nblk,
               chunk_steps=chunk_steps, n_opp_per_chunk=n_opp_chunk, states={})
    # RNG: Python-side uniform blocks, one call per chunk as in run_arm
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([1, N, 1])))
    per = []
    for _ in range(9):
        reps = max(1, int(0.05 / max(1e-7, n_opp_chunk * L * 4e-9)))
        t0 = time.perf_counter()
        for _ in range(reps):
            rng.random(n_opp_chunk * L)
        per.append((time.perf_counter() - t0) / reps / n_opp_chunk)
    out["rng_s_per_opp"] = stats(per)
    for name, st in states.items():
        X = np.ascontiguousarray(st["X"], dtype=np.float64)
        Y = np.ascontiguousarray(st["Y"], dtype=np.float64)
        assert X.shape == (N,)
        p = np.zeros(G.N_GRID)
        hist = np.zeros(G.N_GRID)
        S = np.empty(N)
        G.kde_density(X, N, kern, r, dx, p, hist)
        kl = G.uniform_scores(X, N, p, dx, clip, S)
        occ = int((hist > 0).sum())
        sel = np.empty(N, dtype=np.int64)
        die_c = np.empty(N, dtype=np.int64)
        clone_c = np.empty(N, dtype=np.int64)
        pool = np.empty(2 * N, dtype=np.int64)
        t_kde = time_loop(Kn["kde_loop"], (X, N, kern, r, dx, np.zeros(G.N_GRID), np.zeros(G.N_GRID)))
        t_sc = time_loop(Kn["score_loop"], (X, N, p, dx, clip, np.empty(N)))
        t_sel = time_loop(Kn["select_loop"], (S, N, g, dt_fr, cap, sel, die_c, clone_c, pool, ubuf, L, nblk))
        n_ev, n_kd, n_kc = Kn["select_loop"](S, N, g, dt_fr, cap, sel, die_c, clone_c, pool, ubuf, L, nblk, nblk)
        frac_ev = n_ev / nblk
        k_ev = Kn["first_event_sel"](S, N, g, dt_fr, cap, sel, die_c, clone_c, pool, ubuf, L, nblk)
        if k_ev < 0:                                             # no candidate in nblk blocks: time a pure permutation
            sel[:] = np.arange(N)[::-1]
        t_ga = time_loop(Kn["gather_loop"], (X, Y, sel.copy(), N, np.empty(N), np.empty(N), np.empty(N),
                                             np.empty(N), np.zeros(N, dtype=np.int64)))
        comp = dict(kde=t_kde["median"], score=t_sc["median"], selection=t_sel["median"],
                    gather=t_ga["median"] * frac_ev, rng=out["rng_s_per_opp"]["median"])
        comp["resampling"] = comp["selection"] + comp["gather"] + comp["rng"]
        comp["total"] = comp["kde"] + comp["score"] + comp["resampling"]
        out["states"][name] = dict(occupied_grid_cells=occ, kl=float(kl), score_abs_mean=float(np.abs(S).mean()),
                                   frac_opps_with_candidate=frac_ev, candidates_per_opp=(n_kd + n_kc) / nblk,
                                   timings_s_per_call=dict(kde=t_kde, score=t_sc, selection=t_sel,
                                                           gather_per_event_opp=t_ga),
                                   components_s_per_opp=comp)
        log(f"  micro[{name}]: kde {comp['kde'] * 1e6:.3f} us, score {comp['score'] * 1e6:.3f} us, sel "
            f"{comp['selection'] * 1e6:.3f} us, gather {comp['gather'] * 1e6:.3f} us, rng {comp['rng'] * 1e6:.3f} us"
            f" per opp (occ {occ}, frac_ev {frac_ev:.3f})")
    return out


def micro_lta(N, P, states, log):
    import lta_ladder_numba as E
    Kn = build_lta_kernels()
    c = E.make_cfg(P["T_K"], N, int(P["seeds"][0]), **P["engine_cfg"])
    O, a, Lbox = E.framework(c["framework_npz"])
    fp, ip, beta, fr_on = E._params(c, "fr", O, a, Lbox, 0)
    Kk, _, dphi = E.kde_matrix(int(c["n_grid"]), float(c["kde_bandwidth"]))
    ng = int(c["n_grid"])
    cap = int(c["cap"])
    fr_every = int(c["fr_every"])
    rate, dt_eff = float(fp[E.F_RATE]), float(fp[E.F_DTEFF])
    grid0, qv, lq, clip = float(fp[E.F_GRID0]), float(fp[E.F_QV]), float(fp[E.F_LQ]), float(fp[E.F_SCLIP])
    cphi, wh, cm = float(fp[E.F_CPHI]), float(fp[E.F_WH]), float(fp[E.F_CM])
    nblk = 2048
    frb = np.random.Generator(np.random.PCG64(12345)).random((nblk, N + 2 * cap))
    chunk_steps = max(1, 1_000_000 // N)
    n_opp_chunk = max(1, chunk_steps // fr_every)
    out = dict(fr_every=fr_every, cap=cap, rate=rate, dt_eff=dt_eff, nblk_uniforms=nblk, chunk_steps=chunk_steps,
               n_opp_per_chunk=n_opp_chunk, states={})
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([1, N, 300])))
    per = []
    for _ in range(9):
        reps = max(1, int(0.05 / max(1e-7, n_opp_chunk * (N + 2 * cap) * 4e-9)))
        t0 = time.perf_counter()
        for _ in range(reps):
            rng.random((n_opp_chunk, N + 2 * cap))
        per.append((time.perf_counter() - t0) / reps / n_opp_chunk)
    out["rng_s_per_opp"] = stats(per)
    for name, st in states.items():
        q = np.ascontiguousarray(st["q"], dtype=np.float64)
        assert q.shape == (N, 2, 3)
        xn = np.empty(N)
        phin = np.empty(N)
        jn = np.empty(N, dtype=np.int64)
        Kn["cv_loop"](q, N, cphi, dphi, ng, xn, phin, jn, 1)
        # bitwise check of the KDE/score split against the engine's _fr_score
        cnt = np.zeros(ng)
        touched = np.empty(ng, dtype=np.int64)
        pk_e, raw_e, sc_e = np.empty(ng), np.empty(N), np.empty(N)
        sstd_e, sabs_e = E._fr_score(phin, jn, N, ng, dphi, Kk, grid0, qv, lq, clip, cnt, pk_e, raw_e, sc_e, touched)
        pk_r, raw_r, sc_r = np.empty(ng), np.empty(N), np.empty(N)
        nt = Kn["kde_part"](jn, N, ng, dphi, Kk, cnt, pk_r, touched)
        sstd_r, sabs_r = Kn["score_part"](phin, N, ng, dphi, pk_r, grid0, qv, lq, clip, raw_r, sc_r)
        bitwise = bool(np.array_equal(pk_e, pk_r) and np.array_equal(sc_e, sc_r) and sstd_e == sstd_r
                       and sabs_e == sabs_r and not cnt.any())
        if not bitwise:
            raise RuntimeError(f"KDE/score split differs from lta_ladder_numba._fr_score on state {name}")
        sc = sc_e
        t_cv = time_loop(Kn["cv_loop"], (q, N, cphi, dphi, ng, np.empty(N), np.empty(N), np.empty(N, dtype=np.int64)))
        t_kde = time_loop(Kn["kde_loop"], (jn, N, ng, dphi, Kk, np.zeros(ng), np.empty(ng),
                                           np.empty(ng, dtype=np.int64)))
        t_sc = time_loop(Kn["score_loop"], (phin, N, ng, dphi, pk_e, grid0, qv, lq, clip, np.empty(N), np.empty(N)))
        t_whole = time_loop(Kn["frscore_loop"], (phin, jn, N, ng, dphi, Kk, grid0, qv, lq, clip, np.zeros(ng),
                                                 np.empty(ng), np.empty(N), np.empty(N), np.empty(ng, dtype=np.int64)))
        cand = np.empty(N, dtype=np.int64)
        dsel = np.empty(N, dtype=np.int64)
        ssel = np.empty(N, dtype=np.int64)
        dp = np.empty(N)
        cdf = np.empty(N)
        t_sel = time_loop(Kn["select_loop"], (sc, N, cap, rate, dt_eff, frb, nblk, cand, dsel, ssel, dp, cdf))
        n_ev, n_tot = Kn["select_loop"](sc, N, cap, rate, dt_eff, frb, nblk, cand, dsel, ssel, dp, cdf, nblk)
        # a representative event set for the copy loop (the first opportunity with events, else one synthetic event)
        nev_rep = 0
        for k in range(nblk):
            nev_rep = E.fr_select_native(sc, N, cap, rate, dt_eff, frb[k], cand, dsel, ssel, dp, cdf)
            if nev_rep > 0:
                break
        if nev_rep == 0:
            dsel[0], ssel[0], nev_rep = 0, N - 1, 1
        wst = np.zeros((E.NW, N), dtype=np.int64)
        E._init_labels(q, fp, wst)
        t_cp = time_loop(Kn["copy_loop"], (nev_rep, dsel.copy(), ssel.copy(), jn, np.zeros((E.NA, ng)), q.copy(),
                                           np.arange(N, dtype=np.int64), np.arange(N, dtype=np.int64), wst.copy()))
        copy_per_event = t_cp["median"] / nev_rep
        ist = np.zeros(E.NIST, dtype=np.int64)
        ist[E.S_FIRSTWIN] = -1
        ist[E.S_FIRSTOPP] = -1
        t_te = time_loop(Kn["true_event_loop"], (100, N, xn, phin, a, wh, cm, wst.copy(), ist))
        comp = dict(cv=t_cv["median"], kde=t_kde["median"], score=t_sc["median"], selection=t_sel["median"],
                    copy=copy_per_event * n_tot / nblk, rng=out["rng_s_per_opp"]["median"])
        comp["kde_incl_cv"] = comp["cv"] + comp["kde"]
        comp["resampling"] = comp["selection"] + comp["copy"] + comp["rng"]
        comp["total"] = comp["cv"] + comp["kde"] + comp["score"] + comp["resampling"]
        comp["diag_only_true_event"] = t_te["median"]
        out["states"][name] = dict(touched_bins=int(nt), score_std=float(sstd_e), score_absmax=float(sabs_e),
                                   frac_opps_with_event=n_ev / nblk, events_per_opp=n_tot / nblk,
                                   split_bitwise_equal_to_engine_fr_score=bitwise,
                                   fr_score_whole_s=t_whole["median"],
                                   kde_plus_score_over_whole=(t_kde["median"] + t_sc["median"]) / t_whole["median"],
                                   timings_s_per_call=dict(cv=t_cv, kde=t_kde, score=t_sc, fr_score_whole=t_whole,
                                                           selection=t_sel, copy_per_call=t_cp, true_event=t_te),
                                   copy_events_in_timed_call=int(nev_rep), components_s_per_opp=comp)
        log(f"  micro[{name}]: cv {comp['cv'] * 1e6:.3f} kde {comp['kde'] * 1e6:.3f} score {comp['score'] * 1e6:.3f} "
            f"(whole {t_whole['median'] * 1e6:.3f}) sel {comp['selection'] * 1e6:.3f} copy {comp['copy'] * 1e6:.3f} "
            f"rng {comp['rng'] * 1e6:.3f} us/opp (bins {nt}, ev/opp {n_tot / nblk:.3f})")
    return out


def prod_states(system, N):
    d = os.path.join(ROOT, "results", "equal_budget_v2", PROD_DIRS[system], f"N{N}")
    P = load_prod_cfg(system)
    out = {}
    for s in P["seeds"][:N_PROD_STATES]:
        p = os.path.join(d, f"s{s}_fr.npz")
        if not os.path.exists(p):
            continue
        with np.load(p, allow_pickle=False) as z:
            if system == "gateway":
                out[f"prod_s{s}_final"] = dict(X=z["X_final"].copy(), Y=z["Y_final"].copy())
            else:
                out[f"prod_s{s}_final"] = dict(q=z["q_final"].copy())
    return out


def worker(a):
    t_start = time.time()
    lines = []

    def log(msg):
        lines.append(msg)
        print(f"[{a.system} N{a.N} cpu{a.cpu}] {msg}", flush=True)

    aff = sorted(os.sched_getaffinity(0))
    log(f"affinity {aff}")
    if len(aff) != 1:
        raise SystemExit("worker must be pinned to exactly one CPU (taskset -c <cpu>)")
    fr, end_state = full_runs(a.system, a.N, a.reps, a.target_s, log)
    fr["summary"] = summarize_runs(fr)
    P = load_prod_cfg(a.system)
    states = prod_states(a.system, a.N)
    states["own_short_run_end"] = end_state
    if a.system == "gateway":
        micro = micro_gateway(a.N, P, states, fr["n_steps"], log)
    else:
        micro = micro_lta(a.N, P, states, log)
    # predicted vs measured
    s = fr["summary"]
    abf_off = s["abf_diag_off"]["wall_s"]["median"]
    abf_on_ws = s["abf_diag_on"]["ns_per_walker_step"]["median"] * 1e-9
    n_opp = s["n_fr_opps_per_run"]
    # production run of this N: n_steps = B / N; FR opportunities of the production schedule
    n_prod = int(P["B"]) // a.N
    fe = fr["fr_every"]
    if a.system == "gateway":
        n_opp_prod = -(-n_prod // fe)                                     # steps 0, fe, 2 fe, ... < n_prod
    else:
        fs = int(P["engine_cfg"]["fr_start_steps"])
        n_opp_prod = (n_prod - fs) // fe + 1 if n_prod >= fs else 0       # nxt = fs, fs + fe, ... <= n_prod
    pred = {}
    for name, st in micro["states"].items():
        comp = st["components_s_per_opp"]
        fr_total = comp["total"]
        pred[name] = dict(predicted_us_per_opp=fr_total * 1e6,
                          predicted_overhead_frac=fr_total * n_opp / abf_off,
                          components_frac_of_abf_diag_off={k: v * n_opp / abf_off for k, v in comp.items()},
                          predicted_overhead_frac_production_schedule=fr_total * n_opp_prod / (abf_on_ws * a.N * n_prod))
    fr["production_schedule"] = dict(n_steps=n_prod, n_fr_opps=n_opp_prod,
                                     note="production run of this N: n_steps = B / N; LTA FR from step 20000; the "
                                          "production-schedule prediction divides by the short-run ABF diag-ON cost "
                                          "per walker-step (production ran with diagnostics on)")
    meas = s["measured_fr_overhead_us_per_opp"]
    own = pred["own_short_run_end"]["predicted_us_per_opp"]
    mex = s["measured_fr_execution_us_per_opp"]
    sched_us_per_opp = s["measured_sched_ns_per_step"]["median"] * 1e-3 * fe
    cross = dict(measured_us_per_opp=meas, predicted_us_per_opp_own_state=own,
                 predicted_over_measured=own / meas["median"] if meas["median"] else float("nan"),
                 measured_execution_us_per_opp=mex,
                 predicted_over_measured_execution=own / mex["median"] if mex["median"] else float("nan"),
                 measured_sched_us_per_opp=sched_us_per_opp,
                 components_plus_sched_over_measured=((own + sched_us_per_opp) / meas["median"]
                                                      if meas["median"] else float("nan")),
                 measured_overhead_frac=s["ratios_minus_1"]["fr_overhead_diag_off"],
                 predicted_overhead_frac_own_state=pred["own_short_run_end"]["predicted_overhead_frac"],
                 note=("measured = (FR - ABF) wall / FR opportunities (diag off); execution = (FR - FRsched) / "
                       "(opportunities - 1); sched = (FRsched - ABF) per step x fr_every; predicted = sum of the "
                       "micro-benchmarked components at the end state of the worker's own short FR run"))
    part = dict(system=a.system, N=a.N, cpu=a.cpu, sibling_cpu=sibling(a.cpu), host=platform.node(),
                started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t_start)),
                finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), reps=a.reps, target_s=a.target_s,
                full_runs=fr, micro=micro, predicted=pred, cross_check=cross, log=lines)
    atomic_json(a.out, part)
    log(f"done in {time.time() - t_start:.0f} s -> {a.out}")


# =====================================================================================================================
# driver
# =====================================================================================================================
def production_ledger_ratios():
    out = {}
    for sysk, d in PROD_DIRS.items():
        p = os.path.join(ROOT, "results", "equal_budget_v2", d, "ledger.csv")
        if not os.path.exists(p):
            continue
        last = {}
        for r in csv.DictReader(open(p)):
            last[(int(r["N"]), r["seed"], r["method"])] = r
        P = json.load(open(os.path.join(ROOT, "configs", "equal_budget_v2",
                                        {"gateway": "gateway_production.json", "lta300": "lta_300K_production.json",
                                         "lta150": "lta_150K_production.json"}[sysk])))
        B = int(P["B"])
        res = {}
        for N in sorted({k[0] for k in last}):
            seeds = sorted({k[1] for k in last if k[0] == N})
            ok = lambda k: k in last and last[k]["status"].startswith("complete")  # noqa: E731
            pairs = [(float(last[(N, s, "abf")]["wall_s"]), float(last[(N, s, "fr")]["wall_s"]))
                     for s in seeds if ok((N, s, "abf")) and ok((N, s, "fr"))]
            wa = [float(last[(N, s, "abf")]["wall_s"]) for s in seeds if ok((N, s, "abf"))]
            row = dict(abf_wall_s=stats(wa), abf_ns_per_walker_step_median=float(np.median(wa)) / B * 1e9 if wa else None)
            if pairs:
                rat = [f / a_ - 1.0 for a_, f in pairs]
                row.update(n_pairs=len(pairs), fr_over_abf_minus_1_paired=stats(rat),
                           fr_over_abf_minus_1_ratio_of_sums=sum(f for _, f in pairs) / sum(a_ for a_, _ in pairs) - 1,
                           fr_over_abf_minus_1_ratio_of_medians=float(np.median([f for _, f in pairs])
                                                                      / np.median([a_ for a_, _ in pairs]) - 1))
            res[str(N)] = row
        out[sysk] = res
    return out


def merge(parts_dir, out_json, meta_extra=None):
    parts = {}
    for sysk, N in CASES:
        p = os.path.join(parts_dir, f"{sysk}_N{N}.json")
        if os.path.exists(p):
            parts[f"{sysk}_N{N}"] = json.load(open(p))
    meta = dict(created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), host=platform.node(),
                git_commit=git_commit(), script="scripts/equal_budget/measure_fr_overhead.py",
                cpu_model=_cpu_model(), nproc=os.cpu_count(),
                env={k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS",
                                                    "NUMBA_CACHE_DIR")},
                definitions=dict(
                    ns_per_walker_step="wall seconds of one run_arm call / (N n_steps) x 1e9",
                    ratios_minus_1="within-repetition paired ratio - 1 (num / den - 1), stats over reps",
                    fr_overhead_diag_off="(ABF+FR, diagnostics off) / (ABF, diagnostics off) - 1",
                    fr_overhead_diag_on="(ABF+FR, diagnostics on) / (ABF, diagnostics on) - 1 (the production "
                                        "configuration; compare with the production ledger)",
                    diag_cost_abf="(ABF, diagnostics on) / (ABF, diagnostics off) - 1",
                    components_frac_of_abf_diag_off="component seconds per FR opportunity x opportunities per short "
                                                    "run / median ABF diagnostics-off wall of the short run",
                    production_ledger="results/equal_budget_v2/<dir>/ledger.csv, last row per (N, seed, method); "
                                      "FR and ABF of one seed paired; 110-120 concurrent single-threaded processes"))
    try:
        import numba
        meta.update(numba=numba.__version__, numpy=np.__version__, python=platform.python_version())
    except Exception:
        pass
    if meta_extra is None and os.path.exists(out_json):        # --summarize: keep the launch-time records
        try:
            old = json.load(open(out_json))["meta"]
            meta_extra = {k: old[k] for k in ("cpu_pick", "loadavg_at_merge") if k in old}
        except Exception:
            meta_extra = None
    if meta_extra:
        meta.update(meta_extra)
    obj = dict(meta=meta, cases=parts, production_ledger=production_ledger_ratios())
    atomic_json(out_json, obj)
    return obj


def _cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def fmt(x, nd=1):
    return "--" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{nd}f}"


def print_tables(obj):
    led = obj["production_ledger"]
    print("\n| system | N | ABF ns/ws diag off | ABF ns/ws diag on | diagnostics % (ABF) | FR total overhead % "
          "(diag off) | KDE % | score % | resampling % | sum of components (excl. per-step check) % | "
          "FR % diag on (short runs) | "
          "production ledger FR/ABF % |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for key, part in obj["cases"].items():
        s = part["full_runs"]["summary"]
        r = s["ratios_minus_1"]
        own = part["predicted"]["own_short_run_end"]["components_frac_of_abf_diag_off"]
        kde = own.get("kde_incl_cv", own["kde"])
        lsys = "lta300" if part["system"] == "lta300" else "gateway"
        L = led.get(lsys, {}).get(str(part["N"]), {}).get("fr_over_abf_minus_1_paired", {})
        print(f"| {part['system']} | {part['N']} | {fmt(s['abf_diag_off']['ns_per_walker_step']['median'])} | "
              f"{fmt(s['abf_diag_on']['ns_per_walker_step']['median'])} | "
              f"{fmt(100 * r['diag_cost_abf']['median'], 2)} [{fmt(100 * r['diag_cost_abf']['min'], 2)}, "
              f"{fmt(100 * r['diag_cost_abf']['max'], 2)}] | "
              f"{fmt(100 * r['fr_overhead_diag_off']['median'], 2)} [{fmt(100 * r['fr_overhead_diag_off']['min'], 2)}, "
              f"{fmt(100 * r['fr_overhead_diag_off']['max'], 2)}] | {fmt(100 * kde, 2)} | {fmt(100 * own['score'], 2)} | "
              f"{fmt(100 * own['resampling'], 2)} | {fmt(100 * own['total'], 2)} | "
              f"{fmt(100 * r['fr_overhead_diag_on']['median'], 2)} | "
              f"{fmt(100 * L.get('median', float('nan')), 2)} [{fmt(100 * L.get('q25', float('nan')), 2)}, "
              f"{fmt(100 * L.get('q75', float('nan')), 2)}] |")
    print("\nCross-check (diag off): measured FR-ABF us per opportunity vs predicted (own short-run end state) and "
          "production-state predictions")
    print("| system | N | opps/run | measured FR-ABF us/opp median [min, max] | measured FR-FRsched us/opp | "
          "sched us/opp (FRsched-ABF) | predicted components us/opp (own state) | pred/meas(total) | "
          "(pred+sched)/meas(total) | pred/meas(execution) | predicted us/opp (production final states) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for key, part in obj["cases"].items():
        c = part["cross_check"]
        m = c["measured_us_per_opp"]
        me = c["measured_execution_us_per_opp"]
        prod = [v["predicted_us_per_opp"] for k, v in part["predicted"].items() if k.startswith("prod_")]
        print(f"| {part['system']} | {part['N']} | {part['full_runs']['summary']['n_fr_opps_per_run']} | "
              f"{fmt(m['median'], 3)} [{fmt(m['min'], 3)}, {fmt(m['max'], 3)}] | "
              f"{fmt(me['median'], 3)} [{fmt(me['min'], 3)}, {fmt(me['max'], 3)}] | {fmt(c['measured_sched_us_per_opp'], 3)} | "
              f"{fmt(c['predicted_us_per_opp_own_state'], 3)} | {fmt(c['predicted_over_measured'], 2)} | "
              f"{fmt(c['components_plus_sched_over_measured'], 2)} | {fmt(c['predicted_over_measured_execution'], 2)} | "
              f"{fmt(min(prod), 3) if prod else '--'} .. {fmt(max(prod), 3) if prod else '--'} |")
    print("\nRuns: load average / SMT-sibling busy / cpu-over-wall / per-variant spread")
    for key, part in obj["cases"].items():
        s = part["full_runs"]["summary"]
        print(f"  {key}: n_steps {part['full_runs']['n_steps']}, load {fmt(s['loadavg1']['min'], 0)}-"
              f"{fmt(s['loadavg1']['max'], 0)}, sibling busy median {fmt(s['sibling_busy']['median'], 2)} max "
              f"{fmt(s['sibling_busy']['max'], 2)}, cpu/wall min {fmt(s['cpu_over_wall']['min'], 4)}, spreads "
              + ", ".join(f"{k} {fmt(100 * v['wall_s']['rel_spread'], 2)}%" for k, v in s.items()
                          if isinstance(v, dict) and "wall_s" in v)
              + f"; sched-only {fmt(100 * s['ratios_minus_1']['fr_sched_only']['median'], 2)}% "
                f"({fmt(s['measured_sched_ns_per_step']['median'], 2)} ns/step), FR-only diag extra "
                f"{fmt(100 * s['fr_only_diag_extra_s']['median'] / s['abf_diag_off']['wall_s']['median'], 2)}%")


def driver(a):
    os.makedirs(PART_DIR, exist_ok=True)
    if a.summarize:
        obj = merge(PART_DIR, OUT_JSON)
        print_tables(obj)
        return
    cases = CASES if not a.cases else [(c.split(":")[0], int(c.split(":")[1])) for c in a.cases]
    if a.cpus:
        cpus = [int(x) for x in a.cpus.split(",")]
        pick_info = dict(picked=[dict(cpu=c) for c in cpus], source="--cpus")
    else:
        cpus, pick_info = pick_idle_cpus(len(cases))
        pick_info["source"] = "auto (idle over 4 s, distinct physical cores)"
    if len(cpus) < len(cases):
        raise SystemExit(f"need {len(cases)} CPUs, have {cpus}")
    if len(cpus) > MAX_CPUS:
        raise SystemExit(f"at most {MAX_CPUS} CPUs")
    print(f"CPUs: {cpus[:len(cases)]}  ({pick_info})  load {os.getloadavg()}", flush=True)
    procs = []
    for (sysk, N), cpu in zip(cases, cpus):
        part = os.path.join(PART_DIR, f"{sysk}_N{N}.json")
        logf = open(os.path.join(PART_DIR, f"{sysk}_N{N}.log"), "w")
        cmd = ["taskset", "-c", str(cpu), sys.executable, os.path.abspath(__file__), "--worker", "--system", sysk,
               "--N", str(N), "--cpu", str(cpu), "--reps", str(a.reps), "--target-s", str(a.target_s), "--out", part]
        procs.append((subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=ROOT), logf, sysk, N))
    rc = 0
    for p, logf, sysk, N in procs:
        p.wait()
        logf.close()
        print(f"{sysk} N{N}: exit {p.returncode}", flush=True)
        rc |= p.returncode
    obj = merge(PART_DIR, OUT_JSON, dict(cpu_pick=pick_info, loadavg_at_merge=os.getloadavg()))
    print_tables(obj)
    if rc:
        raise SystemExit("a worker failed (see parts/*.log)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--system", choices=list(CFG_FILES))
    ap.add_argument("--N", type=int)
    ap.add_argument("--cpu", type=int)
    ap.add_argument("--out")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--target-s", type=float, default=12.0)
    ap.add_argument("--cpus", default=None, help="comma-separated logical CPUs (one per case)")
    ap.add_argument("--cases", nargs="*", default=None, help="subset, e.g. gateway:2 lta300:16")
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    if a.reps < 3:
        raise SystemExit("reps must be >= 3")
    if a.worker:
        worker(a)
    else:
        driver(a)


if __name__ == "__main__":
    main()
