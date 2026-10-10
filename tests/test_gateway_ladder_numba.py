"""Equivalence, checkpoint and accounting tests of src/gateway_ladder_numba.py (equal-budget replica ladder).

Q0  constants, init_left, save grid, kernel, knob conversion and the numba MT struct layout;
Q1  the ABF arm is BITWISE equal to gateway_numba.run_ladder_point with a single ABF arm (h 4e-4 and 2.5e-5,
    N 1, 16, 2048; M, C at every save, final X, Y), with odd chunk sizes;
Q2  FR: (a) KDE and uniform score bitwise equal to gateway_numba's; (b) fr_resample fed the recorded uniform
    sequence reproduces gateway_numba.fr_resample bitwise (kd, kc, sel, number of uniforms consumed) in every
    regime (cap binding, P >= N, P < N, no event); (c) with the test-only internal-RNG switch a whole FR-arm
    run is bitwise gateway_numba's single FR arm (M, C, X, Y, windowed ESS/wmax, die, clone, regions);
    (d) the production FR arm (separate PCG64 stream, fixed blocks) equals an independent pure-Python
    reference bitwise, including every diagnostic;
Q3  bitwise checkpoint/resume (interrupt after k chunks, at an FR step, at a genealogy-window reset, inside
    the FR ramp, different chunk sizes on resume, scrambled RNG in between, in a fresh process), both arms;
Q4  diagnostics inert (on/off, different save and trace grids); FR with gamma 0 == ABF bitwise (also in
    separate processes); FR == ABF at every save before the first FR event;
Q5  force-evaluation accounting, N = 1 (ABF only), FR bookkeeping and genealogy invariants (per-save candidate and
    realised-death histograms, window age), result I/O; traces follow walkers (persistent ids: a trace jumps only
    at a death/rebirth); run_job refuses another job's result and defaults to the preregistered save grid
    (== scripts/equal_budget/run_ladder.save_grid); grid / chunk validation; diagnostics-off omits the diagnostic
    keys; the common-name view has lta_ladder_numba's key names, dtypes and ranks;
Q6  speed benchmark printout (ns per walker-step).

CPU only:  CUDA_VISIBLE_DEVICES="" NUMBA_CACHE_DIR=... python -m pytest tests/test_gateway_ladder_numba.py -q -s
"""
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections import Counter

import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
import gateway_numba as gn            # noqa: E402
import gateway_ladder_numba as gl     # noqa: E402

slow = pytest.mark.skipif(os.environ.get("RUN_SLOW") != "1", reason="slow; set RUN_SLOW=1")

CELL = dict(beta=16.0, H=0.5, omega_out=1.0, omega_in=32.0, s=0.1)
EASY = dict(beta=1.0, H=0.1, omega_out=1.0, omega_in=4.0, s=0.1)    # frequent crossings and wall reflections


def gn_cell(h, cell=CELL):
    return dict(cell, dt=h)


def gn_fr(cfg, N):
    c = gl.full_cfg(cfg)
    kn = gl.derived_knobs(c, N)
    return dict(gamma=c["gamma"], eta=c["eta"], fr_every=kn["fr_every"], score_clip=c["score_clip"],
                max_event_fraction=c["max_event_fraction"], ramp_steps=kn["ramp_steps"], cap_min=c["cap_min"],
                min_count=c["min_count"])


DYN_KEYS = ["M_all", "C_all", "X_final", "Y_final", "fr_kd_cum", "fr_kc_cum", "fr_deaths_cum", "fr_opp_cum",
            "fr_opp_event_cum", "fr_opp_death_cum", "fr_events_hist", "fr_events_hist_cum", "fr_deaths_hist",
            "fr_deaths_hist_cum", "max_abs_bias", "n_force_evals", "save_step"]
DIAG_KEYS = ["hist_inst", "region_frac", "events_cum", "lineage_visited", "first_arrival_step",
             "gen_nuniq_run", "gen_ess_run", "gen_maxfam_run", "gen_nuniq_win", "gen_ess_win", "gen_maxfam_win",
             "gen_win_age_steps", "traces", "traces_anc", "traces_rebirth", "traces_step"]


def assert_same(a, b, keys):
    for k in keys:
        assert np.array_equal(np.asarray(a[k]), np.asarray(b[k]), equal_nan=True), k


# ------------------------------------------------------------------------------------------------- Q0
def test_q0_constants_init_grids_knobs():
    assert (gl.XMIN, gl.XMAX, gl.N_GRID, gl.EPS, gl.X_BASIN) == (gn.XMIN, gn.XMAX, gn.N_GRID, gn.EPS, gn.X_BASIN)
    assert gl.grid_dx() == gn.grid_dx()
    k1, r1 = gl.gaussian_kernel_np(0.1, gl.grid_dx()); k2, r2 = gn.gaussian_kernel_np(0.1, gn.grid_dx())
    assert r1 == r2 == 20 and np.array_equal(k1, k2)
    for seed, N in ((7100, 1), (7100, 2048), (5, 37)):
        x, y = gl.init_left(seed, N)
        xr, yr = gn.init_left(seed, N, 16.0, 1.0, 32.0, 0.1)
        assert np.array_equal(x, xr) and np.array_equal(y, yr)
    for n in (1000, 100000, 1_600_000, 3_300_000_000):
        a, ua = gl.budget_save_grid(n); b, ub = gn.budget_save_grid(n)
        assert np.array_equal(a, b) and np.array_equal(ua, ub)
    assert gl.default_noise_seed(7100, 16) == int(1_000_003 * 7100 + 7919 * 16) % (2 ** 31 - 1)
    kn = gl.derived_knobs(gl.full_cfg(), 2048)
    assert kn["fr_every"] == 160 and kn["ramp_steps"] == 160000.0 and kn["window_steps"] == 64000
    caps = {N: gl.derived_knobs(gl.full_cfg(), N)["cap"] for N in (2, 4, 16, 24, 25, 32, 64, 128, 1024, 2048)}
    assert caps == {2: 1, 4: 1, 16: 1, 24: 1, 25: 2, 32: 2, 64: 5, 128: 10, 1024: 81, 2048: 163}
    kn4 = gl.derived_knobs(gl.full_cfg(dict(h=4e-4)), 16)
    assert kn4["fr_every"] == 10 and kn4["ramp_steps"] == 10000.0 and kn4["window_steps"] == 4000
    with pytest.raises(ValueError):
        gl.derived_knobs(gl.full_cfg(dict(h=3e-5)), 16)        # 0.004 / 3e-5 is not an integer
    assert gl.check_mt_layout()
    # trace grid: physical cadence, bounded K even at N = 1 (3.3e9 steps)
    st, ldt = gl.default_trace_grid(3_300_000_000, 2.5e-5)
    assert len(st) < 5000 and st[0] == 0 and st[1] == 4000 and st[400] == 1_600_000 and ldt == 20.0
    st, ldt = gl.default_trace_grid(1_600_000, 2.5e-5)
    assert st[-1] == 1_600_000 and len(st) == 401 and ldt == 10.0
    st, ldt = gl.default_trace_grid(6_400_000, 2.5e-5)   # T = 160: 400 early + 12 late (every 10 t.u.)
    assert len(st) == 401 + 12 and np.all(np.diff(st[400:]) == 400_000)


def test_q0_mt_state_roundtrip_and_even_normals():
    gl._mt_seed(424242)
    gl._mt_normals(10)
    assert gl.mt_has_gauss() == 0
    idx, key = gl.mt_get()
    a = gl._mt_normals(64)
    gl._mt_seed(1)                       # scramble
    gl.mt_set(idx, key)
    assert np.array_equal(gl._mt_normals(64), a)
    gl._mt_normals(1)
    with pytest.raises(RuntimeError):    # an odd normal count would leave a cached gaussian: refuse
        gl.mt_get()
    gl._mt_normals(1)


# ------------------------------------------------------------------------------------------------- Q1
@pytest.mark.parametrize("h", [4e-4, 2.5e-5])
@pytest.mark.parametrize("N,n_steps", [(1, 20000), (16, 6000), (2048, 3000)])
def test_q1_abf_bitwise_vs_gateway_numba(h, N, n_steps):
    cfg = dict(h=h)
    ref = gn.run_ladder_point(7100, N, n_steps, [("abf_h180", False, 180)], gn_cell(h), gn_fr(cfg, N))
    res = gl.run_arm(cfg, "abf", N, 7100, n_steps, ref["save_at"], chunk_steps=777 if N < 2048 else 211)
    assert np.array_equal(res["save_step"], ref["save_at"])
    assert np.array_equal(res["M_all"], ref["M"][0]) and np.array_equal(res["C_all"], ref["C"][0])
    assert np.array_equal(res["X_final"], ref["X_final"][0]) and np.array_equal(res["Y_final"], ref["Y_final"][0])
    assert res["max_abs_bias"] == ref["maxbias"][0]
    # regions agree with gateway_numba's (it sums 1/N increments, so compare as counts)
    assert np.array_equal(np.rint(res["region_frac"] * N), np.rint(ref["P"][0] * N))


# ------------------------------------------------------------------------------------------------- Q2
@pytest.mark.parametrize("N", [1, 3, 37, 2048])
def test_q2a_kde_and_score_bitwise(N):
    rng = np.random.default_rng(N)
    X = np.concatenate([rng.uniform(gl.XMIN, gl.XMAX, max(N - 3, 0)), [gl.XMIN, gl.XMAX, -1.79]])[:N]
    dx = gl.grid_dx(); kern, r = gl.gaussian_kernel_np(0.1, dx)
    p1, h1, p2, h2 = (np.zeros(gl.N_GRID) for _ in range(4))
    m1 = gl.kde_density(X, N, kern, r, dx, p1, h1); m2 = gn.kde_density(X, N, kern, r, dx, p2, h2)
    assert m1 == m2 and np.array_equal(p1, p2)
    S1, S2 = np.empty(N), np.empty(N)
    assert gl.uniform_scores(X, N, p1, dx, 3.0, S1) == gn.uniform_scores(X, N, p2, dx, 3.0, S2)
    assert np.array_equal(S1, S2)


def _replay_case(S, cap, g, dt_fr, seed):
    """gateway_numba.fr_resample on numba's stream vs ours fed the recorded stream."""
    N = len(S)
    L = N + cap + max(N, cap)
    gn._seed_for_tests(seed)
    U = gl._mt_uniforms(L + 1)
    gn._seed_for_tests(seed)
    sel_r, dc, cc, pool = (np.empty(N, np.int64), np.empty(N, np.int64), np.empty(N, np.int64), np.empty(2 * N, np.int64))
    kd_r, kc_r = gn.fr_resample(S, N, g, dt_fr, cap, sel_r, dc, cc, pool)
    nxt = gl._mt_uniforms(1)[0]          # the next uniform gateway_numba did NOT consume
    sel, dc2, cc2, pool2 = (np.empty(N, np.int64), np.empty(N, np.int64), np.empty(N, np.int64), np.empty(2 * N, np.int64))
    kd, kc, nu = gl.fr_resample(S, N, g, dt_fr, cap, sel, dc2, cc2, pool2, U, 0, False)
    assert (kd, kc) == (kd_r, kc_r)
    assert nu <= L and U[nu] == nxt, "number of uniforms consumed differs"
    if kd + kc > 0:
        assert np.array_equal(sel, sel_r)
        assert np.bincount(sel, minlength=N).sum() == N
        deaths = N - len(set(sel.tolist()))
        assert deaths <= max(kd, kc) <= cap            # the realised-death histogram has cap + 1 bins
        if N - kd + kc < N:
            assert deaths == kd                        # P < N: every survivor keeps >= 1 copy
    return kd, kc, N - kd + kc


def test_q2b_fr_law_replay_bitwise():
    rng = np.random.default_rng(3)
    seen = Counter()
    seed = 1000
    for N in (2, 3, 8, 16, 64, 2048):
        cap = max(1, int(math.floor(0.08 * N)))
        for g in (0.0, 1.5, 40.0, 400.0):
            for rep in range(40 if N < 2048 else 6):
                S = np.clip(rng.normal(0.0, 2.0, N), -3.0, 3.0)
                if rep % 5 == 0:
                    S[rng.integers(0, N, max(1, N // 4))] = 0.0          # S == 0: never an event
                for cp in sorted({cap, 1, 3, N - 1} & set(range(1, N))):   # cap < N: see test_q2b_cap_must_be_below_N
                    seed += 1
                    kd, kc, P = _replay_case(S, cp, g, 0.004, seed)
                    seen["none" if kd + kc == 0 else ("P>=N" if P >= N else "P<N")] += 1
                    if kd + kc == cp and kd + kc > 0:
                        seen["cap_full"] += 1
    assert min(seen[k] for k in ("none", "P>=N", "P<N", "cap_full")) >= 20, seen


def test_q2b_cap_must_be_below_N():
    """cap >= N would let every walker die (kd = N, no survivor): gateway_numba.fr_resample then reads
    pool[n_surv - 1] = pool[-1], an uninitialised slot.  Unreachable with the production cap
    max(1, floor(0.08 N)) < N for N >= 2; this engine refuses such a cap instead of reproducing the defect."""
    for N in range(2, 4097):
        assert gl.derived_knobs(gl.full_cfg(), N)["cap"] < N
    with pytest.raises(ValueError):
        gl.run_arm(dict(h=4e-4, cap_min=2), "fr", 2, 7100, 10, [10])
    gl.run_arm(dict(h=4e-4, cap_min=2), "abf", 2, 7100, 10, [10])        # ABF has no cap


@pytest.mark.parametrize("h,N,n_steps,ramp_t,gamma", [(4e-4, 16, 6000, 4.0, 1.5), (4e-4, 64, 6000, 0.4, 15.0),
                                                      (4e-4, 2048, 800, 0.4, 15.0), (2.5e-5, 16, 40000, 0.05, 15.0),
                                                      (2.5e-5, 256, 8000, 0.05, 40.0)])
def test_q2c_fr_arm_bitwise_vs_gateway_numba_internal_rng(h, N, n_steps, ramp_t, gamma):
    cfg = dict(h=h, ramp_t=ramp_t, gamma=gamma)
    kn = gl.derived_knobs(gl.full_cfg(cfg), N)
    ref = gn.run_ladder_point(7100, N, n_steps, [("fr_h180", True, 180)], gn_cell(h), gn_fr(cfg, N),
                              ess_window=kn["window_steps"])
    res = gl.run_arm(cfg, "fr", N, 7100, n_steps, ref["save_at"], chunk_steps=333, _fr_internal_rng=True)
    assert ref["die"][0, -1] + ref["clone"][0, -1] > 0, "test must exercise FR events"
    assert np.array_equal(res["M_all"], ref["M"][0]) and np.array_equal(res["C_all"], ref["C"][0])
    assert np.array_equal(res["X_final"], ref["X_final"][0]) and np.array_equal(res["Y_final"], ref["Y_final"][0])
    assert np.array_equal(res["gen_ess_win"], ref["ess"][0]) and np.array_equal(res["gen_maxfam_win"], ref["wmax"][0])
    assert np.array_equal(res["fr_kd_cum"], ref["die"][0]) and np.array_equal(res["fr_kc_cum"], ref["clone"][0])
    assert np.array_equal(np.rint(res["region_frac"] * N), np.rint(ref["P"][0] * N))
    assert res["n_fr_steps"] == ref["n_fr"]


# ---- Q2d: independent pure-Python reference of the PRODUCTION FR arm (separate PCG64 stream, fixed blocks)
def _py_reflect(q, lo, hi):
    span = hi - lo
    qm = (q - lo) % (2.0 * span)
    if qm > span:
        qm = 2.0 * span - qm
    return qm + lo


def _py_resample(S, N, g, dt_fr, cap, U):
    it = iter(U)
    die, clo = [], []
    for i in range(N):
        u = next(it)
        if S[i] > 0.0:
            if u < 1.0 - math.exp(-g * S[i] * dt_fr):
                die.append(i)
        elif S[i] < 0.0:
            if u < 1.0 - math.exp(g * S[i] * dt_fr):
                clo.append(i)
    nd, nc = len(die), len(clo)
    if nd + nc > cap:
        kd = min(int(round(cap * nd / max(nd + nc, 1))), nd)
        kc = min(cap - kd, nc)
    else:
        kd, kc = nd, nc
    if kd + kc == 0:
        return 0, 0, None
    for lst, k in ((die, kd), (clo, kc)):
        n = len(lst)
        for t in range(k):
            j = min(t + int(next(it) * (n - t)), n - 1)
            lst[t], lst[j] = lst[j], lst[t]
    dead = set(die[:kd])
    pool = [i for i in range(N) if i not in dead]
    n_surv = len(pool)
    pool += clo[:kc]
    P = len(pool)
    if P >= N:
        for t in range(N):
            j = min(t + int(next(it) * (P - t)), P - 1)
            pool[t], pool[j] = pool[j], pool[t]
        sel = pool[:N]
    else:
        sel = pool + [pool[min(int(next(it) * n_surv), n_surv - 1)] for _ in range(N - P)]
    return kd, kc, sel


def py_reference(cfg, method, N, seed, n_steps, save_at, trace_at):
    c = gl.full_cfg(cfg); kn = gl.derived_knobs(c, N)
    h, beta, H, oout, oin, s = (float(c[k]) for k in ("h", "beta", "H", "omega_out", "omega_in", "s"))
    nb, mc = int(c["nb"]), float(c["min_count"]); delta = (gl.XMAX - gl.XMIN) / nb
    fe, W, cap, L, ramp = kn["fr_every"], kn["window_steps"], kn["cap"], kn["fr_block"], kn["ramp_steps"]
    dt_fr = h * fe
    ntr = min(N, 32)
    x0, y0 = gn.init_left(seed, N, beta, oout, oin, s)
    x, y = list(x0), list(y0)
    gn._seed_for_tests(gl.default_noise_seed(seed, N))
    Z = gl._mt_normals(2 * N * n_steps)
    U = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, N, 1]))).random(-(-n_steps // fe) * L)
    dx = gn.grid_dx(); kern, r = gn.gaussian_kernel_np(c["eta"], dx)
    M, C = [0.0] * nb, [0.0] * nb
    lab = [-1 if v < -0.5 else (1 if v > 0.5 else 0) for v in x]
    vis = [1 if v > 0.5 else 0 for v in x]
    anc_r, anc_w = list(range(N)), list(range(N))
    wid, reborn = list(range(N)), [0] * N          # persistent walker id per slot; rebirths per id
    lr = rl = deaths = kdc = kcc = nopp = nopp_ev = nopp_d = 0
    first = -1
    evh, dth = [0] * (cap + 1), [0] * (cap + 1)
    age = 0                                         # steps since the last windowed-label reset
    amp = math.sqrt(2.0 * h / beta)
    out = {k: [] for k in ("M", "C", "hist", "reg", "ev", "vis", "gen", "fr", "evh", "dth", "age")}
    trs, tra, trb = [], [], []

    def fam(lbl):
        cnt = Counter(lbl); ss = sum(float(v) * float(v) for v in cnt.values())
        return [len(cnt), (N * N / ss) / N, max(cnt.values()) / N]

    def save():
        out["M"].append(list(M)); out["C"].append(list(C))
        hh = [0] * nb
        for v in x:
            hh[min(max(int(math.floor((v - gl.XMIN) / delta + 1e-9)), 0), nb - 1)] += 1
        out["hist"].append(hh)
        nl = sum(v < -0.5 for v in x); nr = sum(v > 0.5 for v in x)
        out["reg"].append([nl / N, (N - nl - nr) / N, nr / N]); out["ev"].append([lr, rl]); out["vis"].append([sum(vis)])
        out["gen"].append(fam(anc_r) + fam(anc_w) if method == "fr" else [N, np.nan, np.nan, N, np.nan, np.nan])
        out["fr"].append([deaths, kdc, kcc, nopp, nopp_ev, nopp_d])
        out["evh"].append(list(evh)); out["dth"].append(list(dth)); out["age"].append(age)

    def trace():
        slot = {w: i for i, w in enumerate(wid)}
        trs.append([x[slot[j]] for j in range(ntr)]); tra.append([anc_r[slot[j]] for j in range(ntr)])
        trb.append([reborn[j] for j in range(ntr)])

    si = ti = 0
    if si < len(save_at) and save_at[si] == 0:
        save(); si += 1
    if ti < len(trace_at) and trace_at[ti] == 0:
        trace(); ti += 1
    kopp = 0
    for step in range(n_steps):
        if step % W == 0:
            age = 0
            if method == "fr":
                anc_w = list(range(N))
        age += 1
        zx = Z[2 * N * step: 2 * N * step + N]; zy = Z[2 * N * step + N: 2 * N * (step + 1)]
        fx, fy, jb = [0.0] * N, [0.0] * N, [0] * N
        for i in range(N):
            xi, yi = x[i], y[i]
            e = math.exp(-xi * xi / (2.0 * s * s))
            om = oout + (oin - oout) * e
            dom = -(oin - oout) * (xi / (s * s)) * e
            fx[i] = 4.0 * H * xi * (xi * xi - 1.0) + om * dom * yi * yi
            fy[i] = om * om * yi
            j = min(max(int(math.floor((xi - gl.XMIN) / delta + 1e-9)), 0), nb - 1)
            jb[i] = j; C[j] += 1.0; M[j] += fx[i]
        for i in range(N):
            j = jb[i]
            bias = M[j] / (C[j] + mc)
            x[i] = _py_reflect(x[i] + (-fx[i] + bias) * h + amp * zx[i], gl.XMIN, gl.XMAX)
            y[i] = y[i] - fy[i] * h + amp * zy[i]
            if x[i] > 0.5:
                lr += lab[i] == -1; lab[i] = 1; vis[i] = 1
                if first < 0:
                    first = step + 1
            elif x[i] < -0.5:
                rl += lab[i] == 1; lab[i] = -1
        if method == "fr" and step % fe == 0:
            g = c["gamma"] * (1.0 - math.exp(-step / ramp)) if ramp > 0 else c["gamma"]
            p, hb, Sx = np.zeros(gl.N_GRID), np.zeros(gl.N_GRID), np.empty(N)
            xa = np.array(x)
            gn.kde_density(xa, N, kern, r, dx, p, hb); gn.uniform_scores(xa, N, p, dx, c["score_clip"], Sx)
            kd, kc, sel = _py_resample(list(Sx), N, g, dt_fr, cap, U[kopp * L:(kopp + 1) * L])
            kopp += 1; nopp += 1; evh[kd + kc] += 1
            d = 0
            if kd + kc > 0:
                nopp_ev += 1; kdc += kd; kcc += kc
                d = N - len(set(sel)); deaths += d; nopp_d += d > 0
                x, y, anc_r, anc_w, lab, vis = ([a[k] for k in sel] for a in (x, y, anc_r, anc_w, lab, vis))
                # ids: a walker's first copy keeps its id; the dead walkers' ids go to the other copies in order
                taken, new = set(), []
                for k in sel:
                    new.append(wid[k] if k not in taken else None)
                    taken.add(k)
                free = iter([wid[k] for k in range(N) if k not in taken])
                for i, w in enumerate(new):
                    if w is None:
                        new[i] = next(free); reborn[new[i]] += 1
                wid = new
            dth[d] += 1
        while ti < len(trace_at) and trace_at[ti] == step + 1:
            trace(); ti += 1
        while si < len(save_at) and save_at[si] == step + 1:
            save(); si += 1
    return dict(M_all=np.array(out["M"]), C_all=np.array(out["C"]), hist_inst=np.array(out["hist"]),
                region_frac=np.array(out["reg"]), events_cum=np.array(out["ev"]), lineage_visited=np.array(out["vis"]),
                gen=np.array(out["gen"]), fr=np.array(out["fr"]), fr_events_hist=np.array(evh),
                fr_deaths_hist=np.array(dth), fr_events_hist_cum=np.array(out["evh"]),
                fr_deaths_hist_cum=np.array(out["dth"]), gen_win_age_steps=np.array(out["age"]),
                first_arrival_step=np.array([first]), X_final=np.array(x), Y_final=np.array(y),
                traces=np.array(trs), traces_anc=np.array(tra), traces_rebirth=np.array(trb))


@pytest.mark.parametrize("method,N,cell,gamma,ramp_t", [("fr", 40, EASY, 15.0, 0.4), ("fr", 8, EASY, 40.0, 0.4),
                                                        ("fr", 24, CELL, 15.0, 0.4), ("abf", 5, EASY, 1.5, 4.0)])
def test_q2d_production_arm_vs_python_reference(method, N, cell, gamma, ramp_t):
    h, n_steps = 4e-4, 6000
    cfg = dict(cell, h=h, gamma=gamma, ramp_t=ramp_t, genealogy_window_t=0.4)   # window 1000 steps
    save_at = np.unique(np.r_[0, 1, 9, 10, 11, 999, 1000, 1001, np.arange(97, n_steps + 1, 97), n_steps])
    trace_at = np.arange(0, n_steps + 1, 50)
    ref = py_reference(cfg, method, N, 7100, n_steps, save_at, trace_at)
    res = gl.run_arm(cfg, method, N, 7100, n_steps, save_at, trace_steps=trace_at, chunk_steps=123)
    for k in ("M_all", "C_all", "hist_inst", "region_frac", "events_cum", "lineage_visited", "fr_events_hist",
              "fr_deaths_hist", "fr_events_hist_cum", "fr_deaths_hist_cum", "gen_win_age_steps",
              "first_arrival_step", "X_final", "Y_final", "traces", "traces_anc", "traces_rebirth"):
        assert np.array_equal(res[k], ref[k], equal_nan=True), k
    gen = np.stack([res[f"gen_{a}_{b}"] for b in ("run", "win") for a in ("nuniq", "ess", "maxfam")], 1)
    assert np.array_equal(gen, ref["gen"], equal_nan=True)
    fr = np.stack([res[k] for k in ("fr_deaths_cum", "fr_kd_cum", "fr_kc_cum", "fr_opp_cum", "fr_opp_event_cum",
                                    "fr_opp_death_cum")], 1)
    assert np.array_equal(fr, ref["fr"])
    if method == "fr":
        assert res["fr_kd_cum"][-1] > 0 and res["fr_kc_cum"][-1] > 0
        assert res["traces_rebirth"][-1].sum() > 0, "test must exercise deaths of traced walkers"
    if cell is EASY:
        assert res["events_cum"][-1].min() > 0, "test must exercise transitions both ways"


# ------------------------------------------------------------------------------------------------- Q3
def _run(cfg, method, N, n_steps, save_at, trace_at, **kw):
    return gl.run_arm(cfg, method, N, 7100, n_steps, save_at, trace_steps=trace_at, **kw)


@pytest.mark.parametrize("method", ["abf", "fr"])
def test_q3_checkpoint_resume_bitwise(method, tmp_path):
    h, N, n_steps = 4e-4, 16, 9000          # fr_every 10, window 4000, ramp 10000 steps
    cfg = dict(EASY, h=h, gamma=15.0)
    save_at, _ = gl.budget_save_grid(n_steps)
    trace_at = np.arange(0, n_steps + 1, 50)
    full = _run(cfg, method, N, n_steps, save_at, trace_at)
    ck = str(tmp_path / "ck.npz")
    # chunks of 160 steps: boundaries at FR steps; stop after 25 chunks = step 4000 = a window reset + FR step
    with pytest.raises(gl.RunInterrupted):
        _run(cfg, method, N, n_steps, save_at, trace_at, chunk_steps=160, checkpoint_path=ck,
             checkpoint_every_s=0.0, stop_after_chunks=25)
    with np.load(ck) as z:
        assert int(z["ictr"][gl.I_STEP]) == 4000
    gl._mt_seed(987654)                                       # scramble numba's stream: the restore must matter
    stale = str(tmp_path / "stale.npz"); shutil.copy(ck, stale)
    # resume with a DIFFERENT, odd chunk size (boundaries off the FR grid), interrupt again, resume to the end
    with pytest.raises(gl.RunInterrupted):
        _run(cfg, method, N, n_steps, save_at, trace_at, chunk_steps=37, checkpoint_path=ck,
             checkpoint_every_s=0.0, stop_after_chunks=3)
    gl._mt_seed(1)
    res = _run(cfg, method, N, n_steps, save_at, trace_at, chunk_steps=1001, checkpoint_path=ck, checkpoint_every_s=0.0)
    assert_same(res, full, DYN_KEYS + DIAG_KEYS)
    assert list(res["resumed_from"]) == [4000, 4111] and res["meta"]["status"] == "complete"
    # cost / provenance ledger: one durable session per process lifetime, each with its own engine sha256
    ss = res["meta"]["sessions"]
    assert [x["start_step"] for x in ss] == [0, 4000, 4111] and ss[-1]["end_step"] == n_steps
    assert [x["last_ckpt_step"] for x in ss[:2]] == [4000, 4111]
    assert all(x["engine_sha256"] == res["meta"]["engine_sha256"] == gl._PROVENANCE["engine_sha256"] for x in ss)
    assert res["meta"]["wall_lost_upper_s"] >= 0.0 and res["peak_rss_mb"] > 0.0
    assert res["meta"]["git_dirty"] in (True, False, None)
    # a killed run that had advanced past its last checkpoint resumes from that (stale) checkpoint, bitwise
    res2 = _run(cfg, method, N, n_steps, save_at, trace_at, chunk_steps=500, checkpoint_path=stale)
    assert_same(res2, full, DYN_KEYS + DIAG_KEYS)
    assert [x["start_step"] for x in res2["meta"]["sessions"]] == [0, 4000]
    # a checkpoint never resumes a different run
    with pytest.raises(ValueError):
        _run(dict(cfg, gamma=1.0), method, N, n_steps, save_at, trace_at, checkpoint_path=stale)


@pytest.mark.parametrize("method", ["abf", "fr"])
def test_q3_resume_production_h_inside_ramp_at_window(method, tmp_path):
    """h = 2.5e-5: fr_every 160, window 64000, ramp 160000 steps; interrupt at step 64000 (FR step AND window
    reset, inside the ramp) and at 64161 (one step after an FR step)."""
    h, N, n_steps = 2.5e-5, 4, 70000
    cfg = dict(EASY, h=h, gamma=40.0)
    save_at = np.unique(np.r_[np.arange(0, n_steps + 1, 3200), 63999, 64000, 64001, 64160, 64161])
    trace_at = np.arange(0, n_steps + 1, 1000)
    full = _run(cfg, method, N, n_steps, save_at, trace_at)
    if method == "fr":
        assert full["fr_kd_cum"][-1] + full["fr_kc_cum"][-1] > 0
    ck = str(tmp_path / "ck.npz")
    with pytest.raises(gl.RunInterrupted):
        _run(cfg, method, N, n_steps, save_at, trace_at, chunk_steps=8000, checkpoint_path=ck,
             checkpoint_every_s=0.0, stop_after_chunks=8)
    with pytest.raises(gl.RunInterrupted):
        _run(cfg, method, N, n_steps, save_at, trace_at, chunk_steps=161, checkpoint_path=ck,
             checkpoint_every_s=0.0, stop_after_chunks=1)
    gl._mt_seed(5)
    res = _run(cfg, method, N, n_steps, save_at, trace_at, checkpoint_path=ck)
    assert list(res["resumed_from"]) == [64000, 64161]
    assert_same(res, full, DYN_KEYS + DIAG_KEYS)


_SUB = r"""
import sys, numpy as np
sys.path.insert(0, {src!r})
import gateway_ladder_numba as gl
cfg = dict(beta=1.0, H=0.1, omega_out=1.0, omega_in=4.0, s=0.1, h=4e-4, gamma={gamma})
save_at = np.arange(0, 3001, 100)
try:
    res = gl.run_arm(cfg, {method!r}, 16, 7100, 3000, save_at, trace_steps=np.arange(0, 3001, 25), chunk_steps=160,
                     checkpoint_path={ck!r}, checkpoint_every_s=0.0, stop_after_chunks={stop})
except gl.RunInterrupted:
    sys.exit(3)
gl.save_result({out!r}, res)
"""


def _subrun(method, gamma, ck, out, stop):
    code = _SUB.format(src=os.path.join(ROOT, "src"), method=method, gamma=gamma, ck=ck, out=out, stop=stop)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", NUMBA_NUM_THREADS="1", MKL_NUM_THREADS="1")
    return subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=600)


def test_q3_q4_separate_processes(tmp_path):
    """Fresh processes: (i) an FR run interrupted in one process and resumed in another equals the in-process
    uninterrupted run; (ii) ABF and FR(gamma 0) run in different processes see the same noise, bitwise."""
    cfg = dict(EASY, h=4e-4, gamma=15.0)
    save_at = np.arange(0, 3001, 100); trace_at = np.arange(0, 3001, 25)
    full = gl.run_arm(cfg, "fr", 16, 7100, 3000, save_at, trace_steps=trace_at)
    ck, out = str(tmp_path / "fr.ck.npz"), str(tmp_path / "fr.npz")
    p1 = _subrun("fr", 15.0, ck, out, 7)
    assert p1.returncode == 3, p1.stderr
    assert not os.path.exists(out)
    p2 = _subrun("fr", 15.0, ck, out, None)
    assert p2.returncode == 0, p2.stderr
    res = gl.load_result(out)
    assert_same(res, full, DYN_KEYS + DIAG_KEYS)
    assert list(res["resumed_from"]) == [1120] and res["meta"]["status"] == "complete"
    pa = _subrun("abf", 15.0, str(tmp_path / "a.ck.npz"), str(tmp_path / "abf.npz"), None)
    pf = _subrun("fr", 0.0, str(tmp_path / "f.ck.npz"), str(tmp_path / "fr0.npz"), None)
    assert pa.returncode == 0 and pf.returncode == 0, (pa.stderr, pf.stderr)
    a, f = gl.load_result(str(tmp_path / "abf.npz")), gl.load_result(str(tmp_path / "fr0.npz"))
    assert_same(a, f, ["M_all", "C_all", "X_final", "Y_final", "hist_inst", "region_frac", "events_cum",
                       "lineage_visited", "first_arrival_step", "traces", "traces_anc", "n_force_evals"])
    assert f["fr_opp_cum"][-1] == 300 and f["fr_kd_cum"][-1] == 0


# ------------------------------------------------------------------------------------------------- Q4
def test_q4_diagnostics_inert_and_fr_gamma0_equals_abf():
    h, N, n_steps = 4e-4, 32, 6000
    cfg = dict(EASY, h=h, gamma=15.0)
    gA, _ = gl.budget_save_grid(n_steps)
    gB = np.unique(np.r_[np.arange(37, n_steps + 1, 37), gA[::3], n_steps])
    tA, tB = np.arange(0, n_steps + 1, 40), np.arange(0, n_steps + 1, 7)
    for method in ("abf", "fr"):
        on = gl.run_arm(cfg, method, N, 7100, n_steps, gA, trace_steps=tA)
        off = gl.run_arm(cfg, method, N, 7100, n_steps, gA, trace_steps=tA, diagnostics=False)
        assert_same(on, off, DYN_KEYS)
        # diagnostics off: the diagnostic keys are ABSENT (no initial value that would read as 'never happened')
        assert not any(k in off for k in gl.DIAG_ONLY_KEYS) and all(k in on for k in gl.DIAG_ONLY_KEYS)
        assert off["meta"]["omitted_keys"] == list(gl.DIAG_ONLY_KEYS) and on["meta"]["omitted_keys"] == []
        other = gl.run_arm(cfg, method, N, 7100, n_steps, gB, trace_steps=tB, chunk_steps=999)
        assert_same(on, other, ["X_final", "Y_final", "max_abs_bias", "n_force_evals", "fr_events_hist"])
        common, ia, ib = np.intersect1d(gA, gB, return_indices=True)
        assert len(common) > 20
        for k in ("M_all", "C_all", "hist_inst", "region_frac", "events_cum", "lineage_visited", "gen_ess_run",
                  "gen_nuniq_win", "gen_win_age_steps", "fr_kd_cum", "fr_deaths_cum", "fr_opp_death_cum",
                  "fr_events_hist_cum", "fr_deaths_hist_cum"):
            assert np.array_equal(on[k][ia], other[k][ib], equal_nan=True), k
        ct, ja, jb = np.intersect1d(tA, tB, return_indices=True)
        for k in ("traces", "traces_anc", "traces_rebirth"):
            assert np.array_equal(on[k][ja], other[k][jb]), k
    abf = gl.run_arm(cfg, "abf", N, 7100, n_steps, gA, trace_steps=tA)
    fr0 = gl.run_arm(dict(cfg, gamma=0.0), "fr", N, 7100, n_steps, gA, trace_steps=tA)
    assert_same(abf, fr0, ["M_all", "C_all", "X_final", "Y_final", "max_abs_bias", "n_force_evals", "hist_inst",
                           "region_frac", "events_cum", "lineage_visited", "first_arrival_step", "traces", "traces_anc",
                           "traces_rebirth"])
    assert fr0["traces_rebirth"].max() == 0 and fr0["fr_deaths_hist"][0] == 600
    assert fr0["fr_opp_cum"][-1] == 600 and fr0["fr_events_hist"][0] == 600 and fr0["gen_ess_run"][-1] == 1.0
    # gamma > 0: identical to ABF at every save before the first FR event, different after
    fr = gl.run_arm(cfg, "fr", N, 7100, n_steps, gA, trace_steps=tA)
    pre = fr["fr_opp_event_cum"] == 0
    assert pre.sum() >= 3 and (~pre).sum() >= 3
    assert np.array_equal(fr["M_all"][pre], abf["M_all"][pre]) and np.array_equal(fr["C_all"][pre], abf["C_all"][pre])
    assert not np.array_equal(fr["M_all"][~pre], abf["M_all"][~pre])


# ------------------------------------------------------------------------------------------------- Q5
def test_q5_force_accounting_n1_bookkeeping_io(tmp_path):
    h = 2.5e-5
    for method, N, n_steps in (("abf", 1, 50000), ("abf", 7, 20000), ("fr", 7, 20000), ("fr", 2, 30000)):
        cfg = dict(EASY, h=h, gamma=40.0, ramp_t=0.05)
        save_at = np.unique(np.r_[0, gl.budget_save_grid(n_steps)[0]])
        res = gl.run_arm(cfg, method, N, 7100, n_steps, save_at)
        assert res["n_force_evals"] == N * n_steps == res["C_all"][-1].sum()
        assert np.array_equal(res["C_all"].sum(1), N * save_at)          # one deposit per force evaluation
        assert res["C_all"][0].sum() == 0 and np.array_equal(res["save_u"], save_at / n_steps)
        assert res["meta"]["B"] == N * n_steps and res["meta"]["status"] == "complete"
        kn = gl.derived_knobs(gl.full_cfg(cfg), N)
        if method == "fr":
            n_opp = -(-n_steps // kn["fr_every"])
            assert res["fr_opp_cum"][-1] == n_opp == res["fr_events_hist"].sum() == res["n_fr_steps"]
            assert res["fr_events_hist"][0] == n_opp - res["fr_opp_event_cum"][-1]
            assert np.all(np.diff(res["fr_deaths_cum"]) >= 0) and res["fr_deaths_cum"][-1] > 0
            assert len(res["fr_events_hist"]) == kn["cap"] + 1
            ev, dh, kk = res["fr_events_hist_cum"], res["fr_deaths_hist_cum"], np.arange(kn["cap"] + 1)
            assert ev.shape == dh.shape == (len(save_at), kn["cap"] + 1)
            assert np.array_equal(ev[-1], res["fr_events_hist"]) and np.array_equal(dh[-1], res["fr_deaths_hist"])
            assert np.array_equal(ev.sum(1), res["fr_opp_cum"]) and np.array_equal(dh.sum(1), res["fr_opp_cum"])
            assert np.array_equal(ev[:, 1:].sum(1), res["fr_opp_event_cum"])
            assert np.array_equal(dh[:, 1:].sum(1), res["fr_opp_death_cum"])
            assert np.array_equal(ev @ kk, res["fr_kd_cum"] + res["fr_kc_cum"])
            assert np.array_equal(dh @ kk, res["fr_deaths_cum"])        # realised deaths, opportunity by opportunity
            assert np.all(np.diff(ev, axis=0) >= 0) and np.all(np.diff(dh, axis=0) >= 0)
            assert np.all(res["fr_opp_death_cum"] <= res["fr_opp_event_cum"])
            assert np.all(res["gen_nuniq_run"] <= res["gen_nuniq_win"])
            assert np.all(res["gen_ess_run"] <= res["gen_ess_win"] + 1e-15)
            assert np.all(res["gen_maxfam_run"] >= res["gen_maxfam_win"] - 1e-15)
        else:
            assert res["fr_opp_cum"][-1] == 0 and np.all(np.isnan(res["gen_ess_run"]))
            assert np.all(res["gen_nuniq_run"] == N)
            assert res["fr_events_hist_cum"].max() == 0 and res["fr_deaths_hist_cum"].max() == 0
        W = kn["window_steps"]
        age = res["gen_win_age_steps"]
        assert np.array_equal(age, [0 if k == 0 else k - W * ((k - 1) // W) for k in save_at])
        assert np.all((age >= 1) & (age <= W) | (save_at == 0))
        assert np.allclose(res["region_frac"].sum(1), 1.0) and np.all(res["hist_inst"].sum(1) == N)
        assert res["traces"].shape[1] == min(N, 32) and len(res["traces_step"]) < 5000
        p = str(tmp_path / f"{method}_{N}.npz")
        gl.save_result(p, res)
        back = gl.load_result(p)
        assert_same(back, res, DYN_KEYS + DIAG_KEYS + ["save_t", "save_u", "resumed_from", "traces_t"])
        assert back["meta"] == json.loads(json.dumps(res["meta"])) and back["cfg"]["h"] == h
    with pytest.raises(ValueError):
        gl.run_arm(dict(h=h), "fr", 1, 7100, 100, [100])
    bad = dict(res); bad["meta"] = dict(res["meta"], status="partial")
    with pytest.raises(ValueError):
        gl.save_result(str(tmp_path / "partial.npz"), bad)


def test_q5_run_job_signature_and_prereg_grid(tmp_path):
    """run_job returns an existing result only if it is the SAME run; its default grid is the preregistered one."""
    h = 2.5e-5
    out = str(tmp_path / "job.npz")
    r1 = gl.run_job(dict(h=h), "abf", 4, 7100, 4000, out)
    assert gl.is_complete(out) and not os.path.exists(out + ".ckpt.npz")
    assert np.array_equal(r1["save_step"], gl.prereg_save_grid(4000, h))
    r2 = gl.run_job(dict(h=h), "abf", 4, 7100, 4000, out)
    assert np.array_equal(r1["M_all"], r2["M_all"])
    sig = gl.run_signature(dict(h=h), "abf", 4, 7100, 4000, gl.prereg_save_grid(4000, h))
    assert gl.is_complete(out, sig) and r2["meta"]["run_signature"] == sig
    assert not gl.is_complete(out, gl.run_signature(dict(h=h), "abf", 4, 7101, 4000, gl.prereg_save_grid(4000, h)))
    base = dict(cfg=dict(h=h), method="abf", N=4, seed=7100, n_steps=4000)
    for bad in (dict(cfg=dict(h=h, gamma=0.7)), dict(method="fr"), dict(N=8), dict(seed=7101), dict(n_steps=8000)):
        a = dict(base, **bad)
        with pytest.raises(ValueError, match="DIFFERENT run"):
            gl.run_job(a["cfg"], a["method"], a["N"], a["seed"], a["n_steps"], out)
    with pytest.raises(ValueError, match="DIFFERENT run"):
        gl.run_job(dict(h=h), "abf", 4, 7100, 4000, out, save_steps=[2000, 4000])
    with pytest.raises(ValueError, match="DIFFERENT run"):
        gl.run_job(dict(h=h), "abf", 4, 7100, 4000, out, diagnostics=False)
    # an engine change is refused unless explicitly allowed; a result without a signature is refused
    t = gl.load_result(out)
    t["meta"]["run_signature"]["engine_sha"] = "0" * 64
    gl.save_result(out, t)
    with pytest.raises(ValueError, match="engine_sha"):
        gl.run_job(dict(h=h), "abf", 4, 7100, 4000, out)
    assert np.array_equal(gl.run_job(dict(h=h), "abf", 4, 7100, 4000, out, allow_engine_change=True)["M_all"],
                          r1["M_all"])
    del t["meta"]["run_signature"]
    gl.save_result(out, t)
    with pytest.raises(ValueError, match="run_signature missing"):
        gl.run_job(dict(h=h), "abf", 4, 7100, 4000, out)
    # the default grid carries the physical-time checkpoints t <= T (here T = 2.5: t 0.5, 1, 2)
    r = gl.run_job(dict(h=h), "abf", 1, 7100, 100_000, str(tmp_path / "n1.npz"))
    assert {20_000, 40_000, 80_000} <= set(r["save_step"].tolist())


def test_q5_prereg_grid_equals_production_driver():
    """prereg_save_grid == scripts/equal_budget/run_ladder.save_grid on every production rung, and it contains every
    preregistered physical checkpoint t <= T_N and profile snapshot."""
    spec = importlib.util.spec_from_file_location("eqb_run_ladder", os.path.join(ROOT, "scripts", "equal_budget",
                                                                                 "run_ladder.py"))
    rl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rl)
    P = rl.load_cfg("gateway")
    P["system_key"] = "gateway"
    h = float(P["engine_cfg"]["h"])
    assert tuple(P["physical_checkpoints_t"]) == gl.GATEWAY_PHYSICAL_CHECKPOINTS_T
    assert tuple(P["profile_snapshot_u"]) == gl.PROFILE_SNAPSHOT_U
    for N in P["N_ladder"]:
        n = int(P["B"]) // N
        g = gl.prereg_save_grid(n, h)
        assert np.array_equal(g, rl.save_grid(P, n)), N
        assert np.array_equal(g[np.isin(g, gl.budget_save_grid(n)[0])], gl.budget_save_grid(n)[0])
        for t in P["physical_checkpoints_t"]:
            assert (round(t / h) in g) == (t <= n * h), (N, t)
        for u in P["profile_snapshot_u"]:
            assert round(u * n) in g


def test_q5_peak_rss_is_this_run():
    """peak_rss_mb is the peak of THIS run: VmHWM is reset on entry, so earlier work of the process (a reused pool
    worker) does not leak into the ledger."""
    big = np.ones(50_000_000)
    big[::512] = 2.0
    del big                                                   # a 400 MB transient before the run
    hwm = gl._proc_status_mb("VmHWM")
    if hwm is None:                                           # pragma: no cover
        pytest.skip("no /proc/self/status")
    r = gl.run_arm(dict(h=2.5e-5), "abf", 2, 7100, 100, [100])
    s = r["meta"]["sessions"][0]
    if not s["peak_reset"]:                                   # pragma: no cover
        pytest.skip("/proc/self/clear_refs not writable")
    assert s["vmhwm_before_mb"] >= hwm - 1.0 and r["peak_rss_mb"] < hwm - 300.0
    assert r["meta"]["peak_rss_source"].startswith("max over sessions of VmHWM, reset")
    assert r["peak_rss_mb"] >= s["vmrss_start_mb"] - 1.0


def test_q5_grid_and_chunk_validation():
    h = 2.5e-5
    with pytest.raises(ValueError, match="integer step counts"):
        gl.run_arm(dict(h=h), "abf", 2, 7100, 20000, [0.3 / h, 20000])       # 11999.999...: never truncated
    with pytest.raises(ValueError, match="integer step counts"):
        gl.run_arm(dict(h=h), "abf", 2, 7100, 20000, [20000], trace_steps=[0, 0.5])
    r = gl.run_arm(dict(h=h), "abf", 2, 7100, 20000, [np.rint(0.3 / h), 20000.0], trace_steps=[0])
    assert list(r["save_step"]) == [12000, 20000]
    for bad in (0, -5, 2.5):
        with pytest.raises(ValueError, match="chunk_steps"):
            gl.run_arm(dict(h=h), "abf", 2, 7100, 100, [100], chunk_steps=bad)
    with pytest.raises(ValueError):
        gl.run_arm(dict(h=h), "abf", 2, 7100, 0, [0])
    with pytest.raises(ValueError):
        gl.run_arm(dict(h=h), "abf", 2, 7100, 100, [50, 50, 100])


def test_q5_fr_traces_follow_walkers():
    """Traces follow persistent walker ids: between consecutive traces a column moves by one EM step at most and
    keeps its ancestor label unless its walker died (traces_rebirth increments).  The pool law's slot permutations
    (cap 1 at N 24, so every clone-only opportunity permutes all slots) must not show."""
    h, N, n = 4e-4, 24, 6000
    tr = np.arange(0, n + 1)
    res = gl.run_arm(dict(EASY, h=h, gamma=40.0, ramp_t=0.4), "fr", N, 7100, n, [n], trace_steps=tr)
    x, anc, rb = res["traces"], res["traces_anc"], res["traces_rebirth"]
    assert x.shape == (n + 1, N) and not np.isnan(x).any()
    d_rb = np.diff(rb, axis=0)
    born = d_rb != 0
    assert np.all(d_rb >= 0) and born.sum() > 50
    assert rb[-1].sum() == res["fr_deaths_cum"][-1]               # n_tr = N: one rebirth per realised death
    assert not np.any((np.diff(anc, axis=0) != 0) & ~born)        # labels change only at a death/rebirth
    assert np.abs(np.diff(x, axis=0))[~born].max() < 0.25         # no teleport without a death (ABF max: 0.155)
    assert res["fr_opp_event_cum"][-1] > res["fr_opp_death_cum"][-1] > 0
    abf = gl.run_arm(dict(EASY, h=h), "abf", N, 7100, n, [n], trace_steps=tr)
    assert abf["traces_rebirth"].max() == 0 and np.abs(np.diff(abf["traces"], axis=0)).max() < 0.25


def test_q5_common_view_matches_lta_contract():
    """common_view(res) carries lta_ladder_numba's names, dtypes and ranks for the shared quantities, and the
    mapped death quantities obey LTA's relations (hist total = opportunities, sum k hist_k = deaths)."""
    try:
        import lta_ladder_numba as lt
        save = np.array([0, 1500, 3000])
        L = lt.run_arm(lt.make_cfg(300, 8, 0, n_steps=3000, warmup_steps=10, burn_in_steps=10, fr_start_steps=10,
                                   fr_rate=50.0), "fr", save)
    except (ImportError, FileNotFoundError, OSError) as e:      # pragma: no cover
        pytest.skip(f"lta_ladder_numba not runnable here: {e}")
    G = gl.common_view(gl.run_arm(dict(EASY, h=4e-4, gamma=40.0, ramp_t=0.4), "fr", 8, 7100, 3000, save))
    assert set(G) == set(gl.COMMON_KEYS) and set(G) <= set(L)
    for k in gl.COMMON_KEYS:
        a, b = np.asarray(G[k]), np.asarray(L[k])
        assert a.dtype == b.dtype and a.ndim == b.ndim, (k, a.dtype, b.dtype, a.shape, b.shape)
        if k.startswith(("gen_", "cum_", "save_", "M_", "C_", "hist_", "region_", "events_")):
            assert a.shape[0] == b.shape[0] == len(save), k
    for R in (G, L):
        eh, kk = R["events_per_opp_hist"], np.arange(R["events_per_opp_hist"].shape[1])
        assert np.array_equal(eh.sum(1), R["cum_opps"]) and np.array_equal(eh @ kk, R["cum_deaths"])
        assert np.array_equal(eh[:, 1:].sum(1), R["cum_opps_with_event"]) and R["cum_deaths"][-1] > 0
    assert gl.common_view(gl.run_arm(dict(h=4e-4), "abf", 4, 7100, 100, [100], diagnostics=False)).keys() \
        == {k for k, v in gl.COMMON_KEYS.items() if v not in gl.DIAG_ONLY_KEYS}


# ------------------------------------------------------------------------------------------------- Q6
def test_q6_speed_benchmark():
    h = 2.5e-5
    rows = []
    for N, n_steps in ((1, 2_000_000), (16, 200_000), (2048, 2_000)):
        for method in ("abf", "fr"):
            if method == "fr" and N == 1:
                continue
            save_at, _ = gl.budget_save_grid(n_steps)
            gl.run_arm(dict(h=h), method, N, 1, min(n_steps, 1000), [min(n_steps, 1000)])   # warm (cache) load
            t0 = time.perf_counter()
            res = gl.run_arm(dict(h=h), method, N, 7100, n_steps, save_at)
            dt = time.perf_counter() - t0
            ns = 1e9 * dt / (N * n_steps)
            t0 = time.perf_counter()
            gn.run_ladder_point(7100, N, n_steps, [(method, method == "fr", 180)], gn_cell(h), gn_fr(dict(h=h), N),
                                ess_window=64000)
            ns_ref = 1e9 * (time.perf_counter() - t0) / (N * n_steps)
            rows.append((N, method, ns, ns_ref))
            assert res["n_force_evals"] == N * n_steps
    print("\n  N     arm   ns/walker-step (this engine, diagnostics on)   gateway_numba single arm")
    for N, m, ns, ref in rows:
        print(f"  {N:<5d} {m:<4s}  {ns:8.1f}                                        {ref:8.1f}")
    for N, m, ns, ref in rows:
        assert ns < 400.0, (N, m, ns)


@slow
def test_slow_fr_internal_bitwise_long_production_h():
    """Production h with the production ramp, long enough for many FR events (N 64, 4e5 steps)."""
    h, N, n_steps = 2.5e-5, 64, 400_000
    cfg = dict(h=h)
    ref = gn.run_ladder_point(7100, N, n_steps, [("fr_h180", True, 180)], gn_cell(h), gn_fr(cfg, N), ess_window=64000)
    res = gl.run_arm(cfg, "fr", N, 7100, n_steps, ref["save_at"], _fr_internal_rng=True)
    assert ref["die"][0, -1] > 0
    assert np.array_equal(res["M_all"], ref["M"][0]) and np.array_equal(res["X_final"], ref["X_final"][0])
    assert np.array_equal(res["gen_ess_win"], ref["ess"][0]) and np.array_equal(res["fr_kd_cum"], ref["die"][0])
