"""Tests of the numba ethane/LTA ladder engine ``src/lta_ladder_numba.py``.

E1  replay equivalence with the production torch engine ``core_lta.run_sampler`` (CPU float64): the
    torch draws are recorded (monkeypatched torch.randn/rand/randperm/multinomial, the IC captured
    from LTASystem.initial_conditions) and replayed; T 300 and 150 K, N 64 (cap 1, randperm path),
    3000 steps, small warmup / burn-in / fr_start, fr_rate 40; a start with molecules in the window
    region (A3, cage-crossing branch); and N 256 (cap 5: several events per opportunity, multinomial
    with n >= 2).  Discrete outputs identical, floats within the AUDIT_0A_lta.md tolerances.
E2  law of the engine's own FR randomness (firing frequency, sources ~ S-, uniform cap subset) with
    powered chi-square tests and power checks against plausible wrong laws.
E3  bitwise checkpoint/resume (interrupts inside the burn-in, right after an FR step, and later); at
    N = 1, 2, 3 with the chunk size changed at every segment and a zero-move final chunk at n_steps;
    the resume signature (engine sha / ENGINE, numpy / numba versions, compile target) refuses a
    mismatch unless allowed; peak_rss_mb is per run (VmHWM reset), not the worker's lifetime peak.
E4  the FR arm with fr_rate 0 (and before fr_start) equals the ABF arm bitwise (shared noise); the
    FR arm is refused at N = 1, the ABF arm runs.
E5  diagnostics are inert: dynamics bitwise identical with diagnostics on/off and for two save grids;
    with diagnostics off the diagnostic-only keys are omitted (meta.omitted_keys).
E6  force-evaluation accounting n_force_evals = N (n_steps + 1).
E7  speed benchmark (prints microseconds per molecule-step; not pass/fail).
X*  true-event recount from dense traces, result round trip, trace-grid bound, cap rule, IC law,
    cross-process determinism of both arms (+ ABF/FR pairing across processes), deposit-clip rule and
    small-N FR flags (X8), FR-arm true events / lineage flags / genealogy VALUES under cloning from
    hand-scripted trajectories and an independent recount of a free FR run (X9).

Run on CPU:
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 NUMBA_CACHE_DIR=... \
      python -m pytest tests/test_lta_ladder_numba.py -q -s
"""
import json
import os
import sys
import time

import numpy as np
import pytest
from scipy import stats

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
import lta_ladder_numba as E  # noqa: E402

RUN_SLOW = os.environ.get("RUN_SLOW") == "1"


def small_cfg(T=300.0, N=16, seed=7, n_steps=1200, **kw):
    base = dict(n_steps=n_steps, warmup_steps=200, burn_in_steps=200, fr_start_steps=300, fr_rate=40.0,
                genealogy_window_tu=0.1)
    base.update(kw)
    return E.make_cfg(T, N, seed, **base)


def same_bits(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    return x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes()


def assert_results_bitwise(a, b, skip=E.COST_KEYS):
    assert set(a) == set(b)
    for k in a:
        if k in skip:
            continue
        if isinstance(a[k], str):
            assert a[k] == b[k], k
        else:
            assert same_bits(a[k], b[k]), k


DYNAMIC_KEYS = ("M_all", "C_all", "M_prod", "C_prod", "q_final", "M_all_final", "C_all_final", "M_prod_final",
                "C_prod_final")


# =====================================================================================
# E1  replay equivalence with core_lta.run_sampler
# =====================================================================================
TORCH_NAMES = ("randn", "rand", "randint", "randperm", "multinomial")


def record_torch(T, N, n_steps, save_every, warm, burn, fr_start, fr_rate, method, rng_seed, q0=None):
    torch = pytest.importorskip("torch")
    from lta.core_lta import LTAParams, LTASimConfig, LTASystem, run_sampler
    torch.set_num_threads(1)
    S = LTASystem(LTAParams(temperature=T), torch.device("cpu"), torch.float64, root=ROOT)
    sim = LTASimConfig(n_steps=n_steps, n_replicas=N, save_every=save_every, abf_warmup_steps=warm,
                       estimator_burn_in_steps=burn, fr_start_steps=fr_start, fr_every=5, fr_rate=fr_rate,
                       score_clip=2.0, kde_bandwidth=0.10, max_event_fraction=0.02, abf_force_clip=60.0,
                       abf_estimator="histogram", rng_seed=rng_seed, store_snapshots=save_every)
    log = {n: [] for n in TORCH_NAMES}
    cap = {}
    orig_ic = S.initial_conditions

    def ic(R, N_, gen):
        out = (torch.as_tensor(q0, dtype=torch.float64).reshape(R, N_, 2, 3).clone() if q0 is not None
               else orig_ic(R, N_, gen))
        cap["q0"] = out.clone().numpy().reshape(N_, 2, 3)
        cap["ic_done"] = True
        return out

    orig_f = S.forces

    def forces(q):
        cap["q_last"] = q.detach().clone()
        return orig_f(q)

    S.initial_conditions = ic
    S.forces = forces
    orig = {n: getattr(torch, n) for n in TORCH_NAMES}

    def wrap(n):
        f = orig[n]

        def g(*a, **k):
            out = f(*a, **k)
            if cap.get("ic_done"):
                log[n].append(out.detach().clone().numpy())
            return out
        return g

    for n in TORCH_NAMES:
        setattr(torch, n, wrap(n))
    try:
        out = run_sampler(method, S, sim, seeds=[0], verbose=False)
    finally:
        for n, f in orig.items():
            setattr(torch, n, f)
    assert not log["randint"]
    rep = dict(q0=cap["q0"], noise=np.stack([x.reshape(N, 2, 3) for x in log["randn"]]),
               fr_u=(np.stack([x.reshape(N) for x in log["rand"]]) if log["rand"] else np.zeros((0, N))),
               fr_perm=(np.concatenate(log["randperm"]) if log["randperm"] else np.zeros(0, np.int64)),
               fr_src=(np.concatenate(log["multinomial"]) if log["multinomial"] else np.zeros(0, np.int64)))
    return out, rep, cap["q_last"].numpy().reshape(N, 2, 3)


def near_window_q0(N, seed, frac=0.75, zlo=1.0, zhi=1.45):
    """A3: a fraction ``frac`` of the molecules with the COM at a distance in [zlo, zhi] A from a window
    plane (x = k a), on either side, axis along x; the rest at cage centres (the production law).
    The default puts them inside the window region |z| < 1.5, so torch's has_left_cage flag is set at
    step 0 and arrivals in a cage exercise the cage-crossing branch."""
    rng = np.random.default_rng(seed)
    O, a, L = E.framework()
    q0 = E.initial_conditions(rng, N, a, L)
    for i in range(int(frac * N)):
        sgn = 1.0 if rng.random() < 0.5 else -1.0
        com = np.array([a * rng.integers(0, 2) + sgn * rng.uniform(zlo, zhi),
                        a * (rng.integers(0, 2) + 0.5) + rng.normal(0, 0.2),
                        a * (rng.integers(0, 2) + 0.5) + rng.normal(0, 0.2)])
        u = np.array([1.0, 0.0, 0.0]) + rng.normal(0, 0.1, 3)
        u /= np.linalg.norm(u)
        q0[i, 0] = com + 0.77 * u
        q0[i, 1] = com - 0.77 * u
    return q0


E1_CASES = [(300.0, "cage", "abf", 64, 3000), (300.0, "cage", "fr", 64, 3000), (150.0, "cage", "abf", 64, 3000),
            (150.0, "cage", "fr", 64, 3000), (300.0, "window", "abf", 64, 3000), (300.0, "window", "fr", 64, 3000),
            (300.0, "cage", "fr", 256, 1500)]          # N 256: cap 5, several events per opportunity


@pytest.mark.parametrize("T,start,method,N,n_steps", E1_CASES)
def test_e1_replay_equivalence_with_torch(T, start, method, N, n_steps):
    save_every, warm, burn, fr_start, fr_rate = 500, 200, 200, 300, 40.0
    q0 = None
    if start == "window":
        q0 = near_window_q0(N, 5)
        warm = 2000          # a slower ramp lets the window starters reach the cages within 3000 steps
    ref, rep, q_last = record_torch(T, N, n_steps, save_every, warm, burn, fr_start, fr_rate,
                                    "fr_uniform" if method == "fr" else "abf", 20261010 + int(T), q0)
    cfg = E.make_cfg(T, N, 0, n_steps=n_steps, warmup_steps=warm, burn_in_steps=burn, fr_start_steps=fr_start,
                     fr_rate=fr_rate, cap_min=0, accumulate_u=True, record_events=True)
    assert cfg["cap"] == int(0.02 * N)
    steps = np.asarray(ref["steps"], np.int64)
    r = E.run_arm(cfg, method, steps, trace_steps=np.asarray(ref["snap_steps"], np.int64), replay=rep)
    # every recorded draw consumed
    assert r["replay_consumed"].tolist() == [rep["fr_u"].shape[0], rep["fr_perm"].size, rep["fr_src"].size]
    # ---- floats (audit tolerances) ----
    use_prod = r["C_prod"].sum(-1) > 0
    mf = np.where(use_prod[:, None], E.mean_force(r["M_prod"], r["C_prod"]), E.mean_force(r["M_all"], r["C_all"]))
    pmf = E.histogram_pmf(mf, E.TWO_PI / 180)
    Kk, _, dphi = E.kde_matrix(180, 0.10)
    ph = np.maximum(r["hist_inst"] @ Kk.T, 0.0)
    ph = ph / np.maximum(ph.sum(-1, keepdims=True) * dphi, 1e-12)
    d = dict(q=np.abs(r["q_final"] - q_last).max(),
             mean_force=np.abs(mf - ref["mean_force"][:, 0]).max(),
             pmf=np.abs(pmf - ref["pmf"][:, 0]).max(),
             p_hat=np.abs(ph - ref["p_hat"][:, 0]).max(),
             fsum_prod_rel=np.abs(r["M_prod_final"] - ref["fsum_prod"][0]).max() / np.abs(ref["fsum_prod"][0]).max(),
             fsum_rel=np.abs(r["M_all"][-1] - r["M_all_final"]).max(),
             u_of_z_rel=np.abs(r["u_of_z"] - ref["u_of_z"][0]).max() / np.abs(ref["u_of_z"][0]).max())
    print(f"\nE1 T {T} {start} {method}: " + ", ".join(f"{k} {v:.2e}" for k, v in d.items())
          + f"; events {int(r['total_replacement_events'])}, transitions {int(r['n_transitions'][-1])}, "
            f"cage crossings {int(r['n_cage_crossings_lineage'][-1])}, true window crossings "
            f"{int(r['events_window_crossings'][-1])}")
    assert d["q"] <= 1e-9
    assert d["pmf"] <= 1e-10
    assert d["mean_force"] <= 1e-9
    assert d["fsum_prod_rel"] <= 1e-12
    assert d["fsum_rel"] == 0.0
    assert d["p_hat"] <= 1e-13
    assert d["u_of_z_rel"] <= 1e-12
    # ---- discrete outputs: identical ----
    assert np.array_equal(r["C_all_final"], ref["final_eff_counts"][0])
    assert np.array_equal(r["C_all"][-1], ref["eff_counts"][-1, 0])      # torch CPU aliasing: final only
    assert np.array_equal(r["C_prod_final"], ref["u_counts"][0])
    assert int(r["n_transitions"][-1]) == int(ref["n_transitions"][0])
    assert int(r["n_cage_crossings_lineage"][-1]) == int(ref["n_cage_crossings"][0])
    for k, nm in enumerate(("cage", "neck", "window")):
        assert np.array_equal(r["region_frac"][:, k].astype(np.float32), ref[f"frac_{nm}"][:, 0]), nm
    assert np.array_equal(r["birth_hist"], ref["birth_hist"][0])
    assert np.array_equal(r["death_hist"], ref["death_hist"][0])
    assert int(r["total_replacement_events"]) == int(ref["total_replacement_events"][0])
    assert np.array_equal(r["cum_deaths"], ref["repl_cumulative"][:, 0])
    n_tr = r["traces_anc"].shape[1]
    assert np.array_equal(r["traces_anc"], ref["snap_anc"][:, :n_tr])
    assert np.abs(r["traces"] - ref["snap_phi"][:, :n_tr]).max() < 1e-5
    if method == "fr":
        assert np.array_equal(r["event_counts"], ref["event_counts"][:, 0])
        assert np.array_equal(r["event_log"][:, 1:], ref["snap_ev_slots"])          # (dying, source) in order
        assert np.array_equal(r["gen_n_unique"], ref["n_unique_ancestor"][:, 0])
        assert np.abs(r["gen_ess"] * N - ref["ancestor_ess"][:, 0]).max() <= 1e-12 * N
        assert np.array_equal(r["gen_max_frac"], ref["max_ancestor_frac"][:, 0])
        assert abs(r["fr_score_std"] - ref["fr_score_std"][0]) <= 1e-12
        assert abs(r["fr_score_absmax"] - ref["fr_score_absmax"][0]) <= 1e-12
        assert int(r["total_replacement_events"]) > 20                            # events exercised
        if N == 64:
            assert rep["fr_perm"].size > 0                                         # cap path exercised
        if N == 256:                                                               # n <= cap, n >= 2 path
            assert r["event_counts"].max() >= 2 and np.any((r["event_counts"] >= 2) & (r["event_counts"] < 5))
    else:
        assert r["event_counts"].size == 0 and int(r["total_replacement_events"]) == 0
    if start == "window":
        assert int(r["n_cage_crossings_lineage"][-1]) >= 1                         # crossing branch exercised
        assert int(r["events_window_crossings"][-1]) > 0


# =====================================================================================
# E2  law of the engine's own FR randomness
# =====================================================================================
def _work(N):
    return (np.empty(N, np.int64), np.empty(N, np.int64), np.empty(N, np.int64), np.empty(N), np.empty(N))


def _run_trials(sc, cap, rate, dt_eff, M, seed, collect_subsets=False):
    N = sc.size
    U = np.random.default_rng(seed).random((M, N + 2 * cap))
    cand, dsel, ssel, dp, cdf = _work(N)
    deaths = np.zeros(N, np.int64)
    srcs = np.zeros(N, np.int64)
    subsets, firsts, n_list = [], [], []
    for m in range(M):
        n = E.fr_select_native(sc, N, cap, rate, dt_eff, U[m], cand, dsel, ssel, dp, cdf)
        np.add.at(deaths, dsel[:n], 1)
        np.add.at(srcs, ssel[:n], 1)
        n_list.append(n)
        if collect_subsets and n:
            subsets.append(tuple(sorted(dsel[:n].tolist())))
            firsts.append(int(dsel[0]))
    return deaths, srcs, np.array(n_list), subsets, firsts


def test_e2_firing_frequency_and_source_law():
    N = 30
    sc = np.linspace(-1.5, 1.8, N)
    sc = sc - sc.mean()
    rate, dt_eff, M = 150.0, 1e-3, 40000
    deaths, srcs, nl, _, _ = _run_trials(sc, N, rate, dt_eff, M, 11)          # cap = N: no cap binding
    dw = np.maximum(sc, 0.0)
    bw = np.maximum(-sc, 0.0)
    p = np.where(dw > 0, 1.0 - np.exp(-rate * dw * dt_eff), 0.0)
    pos = p > 0
    assert np.all(deaths[~pos] == 0)
    x2 = (((deaths[pos] - M * p[pos]) ** 2) / (M * p[pos] * (1 - p[pos]))).sum()
    pval = stats.chi2.sf(x2, pos.sum())
    # power: the same statistic rejects a 5 % error in the rate law
    p_alt = 1.0 - np.exp(-1.05 * rate * dw * dt_eff)
    x2_alt = (((deaths[pos] - M * p_alt[pos]) ** 2) / (M * p_alt[pos] * (1 - p_alt[pos]))).sum()
    print(f"\nE2 firing: chi2 {x2:.1f} / {pos.sum()} dof p {pval:.3f}; vs 1.05 x rate chi2 {x2_alt:.0f} "
          f"p {stats.chi2.sf(x2_alt, pos.sum()):.1e}")
    assert pval > 1e-3
    assert stats.chi2.sf(x2_alt, pos.sum()) < 1e-6
    # sources: iid categorical proportional to S- (only S < 0 walkers)
    neg = bw > 0
    assert np.all(srcs[~neg] == 0)
    n_src = srcs.sum()
    assert n_src == deaths.sum() and n_src > 50000
    e = n_src * bw[neg] / bw.sum()
    x2s = ((srcs[neg] - e) ** 2 / e).sum()
    ps = stats.chi2.sf(x2s, neg.sum() - 1)
    e_alt = np.full(neg.sum(), n_src / neg.sum())                                # uniform among S < 0
    x2s_alt = ((srcs[neg] - e_alt) ** 2 / e_alt).sum()
    print(f"E2 sources: chi2 {x2s:.1f} / {neg.sum() - 1} dof p {ps:.3f}; vs uniform chi2 {x2s_alt:.0f}")
    assert ps > 1e-3
    assert stats.chi2.sf(x2s_alt, neg.sum() - 1) < 1e-6


def test_e2_cap_subset_is_uniform():
    # 10 positive-score walkers that always fire (rate so large that dp == 1.0), cap 3
    sc = np.array([-1.0, 0.3, -0.5, 0.8, 0.1, -2.0, 0.6, 0.2, 0.9, -0.7, 0.4, 0.5, 0.05, -0.3, 0.7])
    sc = sc - sc.mean()
    pos = np.nonzero(sc > 0)[0]
    n, cap, M = pos.size, 3, 30000
    assert n == 10
    deaths, srcs, nl, subsets, firsts = _run_trials(sc, cap, 1e9, 1e-3, M, 12, collect_subsets=True)
    assert np.all(nl == cap)
    # every 3-subset of the 10 candidates equally likely: multinomial GOF over C(10,3) = 120 cells
    from itertools import combinations
    cells = {s: 0 for s in combinations(pos.tolist(), cap)}
    for s in subsets:
        cells[s] += 1
    obs = np.array(list(cells.values()))
    x2 = ((obs - M / obs.size) ** 2 / (M / obs.size)).sum()
    pval = stats.chi2.sf(x2, obs.size - 1)
    # first selected element uniform among the candidates (ordered subset)
    fo = np.array([firsts.count(i) for i in pos])
    x2f = ((fo - M / n) ** 2 / (M / n)).sum()
    pf = stats.chi2.sf(x2f, n - 1)
    # power: a classic off-by-one partial Fisher-Yates (j in [t, n-2]) is rejected by the same test
    rng = np.random.default_rng(13)
    bad = {s: 0 for s in combinations(range(n), cap)}
    for _ in range(M):
        c = list(range(n))
        for t in range(cap):
            j = t + int(rng.random() * (n - t - 1))
            c[t], c[j] = c[j], c[t]
        bad[tuple(sorted(c[:cap]))] += 1
    ob = np.array(list(bad.values()))
    x2b = ((ob - M / ob.size) ** 2 / (M / ob.size)).sum()
    print(f"\nE2 cap subset: chi2 {x2:.1f} / 119 p {pval:.3f}; first {x2f:.1f} / 9 p {pf:.3f}; "
          f"off-by-one chi2 {x2b:.0f}")
    assert pval > 1e-3 and pf > 1e-3
    assert stats.chi2.sf(x2b, ob.size - 1) < 1e-6
    # sources still proportional to S- under the cap
    bw = np.maximum(-sc, 0.0)
    neg = bw > 0
    e = srcs.sum() * bw[neg] / bw.sum()
    assert stats.chi2.sf(((srcs[neg] - e) ** 2 / e).sum(), neg.sum() - 1) > 1e-3


def test_e2_guards_and_dt_eff():
    N = 8
    cand, dsel, ssel, dp, cdf = _work(N)
    u = np.zeros(N + 2)
    assert E.fr_select_native(np.zeros(N), N, 1, 1e9, 1e-3, u, cand, dsel, ssel, dp, cdf) == 0
    s = np.full(N, 0.5)                                                         # no S < 0: no event
    assert E.fr_select_native(s, N, 1, 1e9, 1e-3, u, cand, dsel, ssel, dp, cdf) == 0
    c = E.make_cfg(300.0, 1024, 0)
    O, a, L = E.framework()
    fp, ip, beta, fr_on = E._params(c, "fr", O, a, L, 0)
    assert fp[E.F_DTEFF] == 2e-4 * 5 and fr_on and ip[E.I_CAP] == 20
    fp, ip, beta, fr_on = E._params(c, "abf", O, a, L, 0)
    assert not fr_on


# =====================================================================================
# E3  bitwise checkpoint / resume
# =====================================================================================
@pytest.mark.parametrize("method", ["fr", "abf"])
def test_e3_checkpoint_resume_bitwise(tmp_path, method):
    c = small_cfg(N=16, n_steps=1200)
    save = np.unique(np.concatenate([np.arange(0, 1201, 50), [1, 199, 200, 299, 300, 301, 777, 1199]]))
    ref = E.run_arm(c, method, save, chunk_steps=1000)                 # uninterrupted, different chunking
    ck = str(tmp_path / f"ck_{method}.npz")
    # chunk boundaries at 150 (inside burn-in 200 and warmup), 300 (right after the FR at nxt = 300),
    # 450 (after an FR step), 600, ...
    with pytest.raises(E.InterruptedRun):
        E.run_arm(c, method, save, checkpoint_path=ck, checkpoint_every_s=0.0, chunk_steps=150,
                  _stop_after_chunks=1)
    st, extra = E._read_checkpoint(ck)
    assert int(st["ist"][E.S_STEP]) == 150 and extra["status"] == "partial"
    with pytest.raises(E.InterruptedRun):
        E.run_arm(c, method, save, checkpoint_path=ck, checkpoint_every_s=0.0, chunk_steps=150,
                  _stop_after_chunks=1)
    st, extra = E._read_checkpoint(ck)
    assert int(st["ist"][E.S_STEP]) == 300
    if method == "fr":
        assert int(st["ist"][E.S_NOPP]) == 1                            # the FR at 300 was applied
    with pytest.raises(E.InterruptedRun):
        E.run_arm(c, method, save, checkpoint_path=ck, checkpoint_every_s=0.0, chunk_steps=150,
                  _stop_after_chunks=2)
    res = E.run_arm(c, method, save, checkpoint_path=ck, checkpoint_every_s=0.0, chunk_steps=150)
    assert res["resumed_from"].tolist() == [150, 300, 600]
    assert json.loads(res["meta_json"])["status"] == "complete"
    assert_results_bitwise(ref, res)
    if method == "fr":
        assert int(res["total_replacement_events"]) > 0
    # a complete checkpoint returns the same result again; a different job refuses it
    res2 = E.run_arm(c, method, save, checkpoint_path=ck, checkpoint_every_s=0.0, chunk_steps=150)
    assert_results_bitwise(ref, res2)
    with pytest.raises(ValueError):
        E.run_arm(dict(c, fr_rate=1.0), method, save, checkpoint_path=ck, chunk_steps=150)


def test_e3_chunking_does_not_change_results():
    c = small_cfg(N=8, n_steps=700)
    save = np.arange(0, 701, 70)
    a = E.run_arm(c, "fr", save, chunk_steps=7)
    b = E.run_arm(c, "fr", save, chunk_steps=701)
    assert_results_bitwise(a, b)


@pytest.mark.parametrize("N,method,rate", [(1, "abf", 40.0), (2, "fr", 40000.0), (3, "fr", 4000.0)])
def test_e3_small_n_resume_changed_chunks_and_zero_move_final_chunk(tmp_path, N, method, rate):
    """Small-N resume: the chunk size changes at every segment, cuts inside the burn-in (150), just after the
    first FR opportunity (301), and at n_steps (600: the last segment is a single evaluation, zero moves)."""
    c = small_cfg(N=N, n_steps=600, fr_rate=rate)
    save = np.unique(np.concatenate([np.arange(0, 601, 60), [1, 150, 299, 300, 301, 599, 600]]))
    ref = E.run_arm(c, method, save, chunk_steps=1000)
    ck = str(tmp_path / f"ck_{N}.npz")
    for cs, stop_at in ((150, 150), (151, 301), (299, 600)):
        with pytest.raises(E.InterruptedRun):
            E.run_arm(c, method, save, checkpoint_path=ck, checkpoint_every_s=0.0, chunk_steps=cs,
                      _stop_after_chunks=1)
        st, extra = E._read_checkpoint(ck)
        assert int(st["ist"][E.S_STEP]) == stop_at and extra["status"] == "partial"
    res = E.run_arm(c, method, save, checkpoint_path=ck, checkpoint_every_s=0.0, chunk_steps=7)
    assert res["resumed_from"].tolist() == [150, 301, 600]
    assert_results_bitwise(ref, res)
    assert int(res["n_force_evals"]) == N * 601
    sess = json.loads(res["sessions_json"])
    assert [s["start_step"] for s in sess] == [0, 150, 301, 600] and sess[-1]["end_step"] == 601
    if method == "fr":
        assert int(res["total_replacement_events"]) > 0                   # the FR stream position was exercised


def _tamper_checkpoint(path, **sig_changes):
    st, extra = E._read_checkpoint(path)
    if sig_changes.pop("drop_signature", False):
        extra.pop("signature")
    else:
        extra["signature"] = dict(extra["signature"], **sig_changes)
    E._write_checkpoint(path, st, extra)


def test_e3_resume_signature_engine_and_platform(tmp_path):
    """A checkpoint written by other engine code, numpy / numba versions or compile target is refused unless the
    matching allow_* flag is given (then recorded in the sessions); one without a signature is refused."""
    c = small_cfg(N=8, n_steps=500)
    save = np.arange(0, 501, 50)
    ref = E.run_arm(c, "fr", save)
    sig = E.run_signature(c, "fr", save, E.trace_grid(c)[0])
    assert set(sig) == {"job", "engine", "engine_sha256", "numpy", "numba", "compile_target"}
    assert sig["compile_target"]["cpu_name"] == E.compile_target()["cpu_name"]
    meta = json.loads(ref["meta_json"])
    assert meta["run_signature"] == sig and meta["engine_sha256"] == sig["engine_sha256"]
    assert meta["git_dirty"] in (True, False, None) and meta["compile_target"]["triple"]
    cases = [(dict(engine_sha256="0" * 64), "allow_engine_change"),
             (dict(engine="lta_ladder_numba/0"), "allow_engine_change"),
             (dict(numpy="1.0.0"), "allow_platform_change"),
             (dict(numba="0.1.0"), "allow_platform_change"),
             (dict(compile_target=dict(sig["compile_target"], cpu_name="skylake-avx512")), "allow_platform_change")]
    for k, (change, flag) in enumerate(cases):
        ck = str(tmp_path / f"ck{k}.npz")
        with pytest.raises(E.InterruptedRun):
            E.run_arm(c, "fr", save, checkpoint_path=ck, checkpoint_every_s=0.0, chunk_steps=170, _stop_after_chunks=2)
        _tamper_checkpoint(ck, **change)
        with pytest.raises(ValueError, match="bitwise continuous"):
            E.run_arm(c, "fr", save, checkpoint_path=ck, chunk_steps=170)
        other = "allow_platform_change" if flag == "allow_engine_change" else "allow_engine_change"
        with pytest.raises(ValueError):
            E.run_arm(c, "fr", save, checkpoint_path=ck, chunk_steps=170, **{other: True})
        res = E.run_arm(c, "fr", save, checkpoint_path=ck, chunk_steps=170, **{flag: True})
        assert_results_bitwise(ref, res)                       # same code really ran: still bitwise
        assert json.loads(res["sessions_json"])[-1]["tolerated_mismatch"] == sorted(change)
    ck = str(tmp_path / "ck_nosig.npz")
    with pytest.raises(E.InterruptedRun):
        E.run_arm(c, "fr", save, checkpoint_path=ck, checkpoint_every_s=0.0, chunk_steps=170, _stop_after_chunks=1)
    _tamper_checkpoint(ck, drop_signature=True)
    with pytest.raises(ValueError, match="no run signature"):
        E.run_arm(c, "fr", save, checkpoint_path=ck, allow_engine_change=True, allow_platform_change=True)
    ck2 = str(tmp_path / "ck_job.npz")
    with pytest.raises(E.InterruptedRun):
        E.run_arm(c, "fr", save, checkpoint_path=ck2, checkpoint_every_s=0.0, chunk_steps=170, _stop_after_chunks=1)
    with pytest.raises(ValueError, match="different job"):          # job identity is never negotiable
        E.run_arm(dict(c, seed=8), "fr", save, checkpoint_path=ck2, allow_engine_change=True,
                  allow_platform_change=True)


def test_e3_peak_rss_is_per_run_not_worker_lifetime():
    """peak_rss_mb is VmHWM reset at run start: a large allocation freed BEFORE the run does not leak into it."""
    if E._proc_status_mb("VmHWM") is None or not E._reset_peak_rss():
        pytest.skip("needs Linux /proc/self/clear_refs")
    c = small_cfg(N=8, n_steps=200)
    r1 = E.run_arm(c, "fr", np.array([0, 200]))
    big = np.ones(int(1.2e9 / 8))                              # 1.2 GB touched, then freed
    big_mb = E._proc_status_mb("VmRSS")
    del big
    r2 = E.run_arm(c, "fr", np.array([0, 200]))
    meta = json.loads(r2["meta_json"])
    print(f"\nE3 RSS: run 1 {float(r1['peak_rss_mb']):.0f} MB, with the 1.2 GB array {big_mb:.0f} MB, "
          f"run 2 {float(r2['peak_rss_mb']):.0f} MB")
    assert meta["peak_rss_method"].startswith("VmHWM")
    assert float(r2["peak_rss_mb"]) < big_mb - 800.0
    s = json.loads(r2["sessions_json"])
    assert len(s) == 1 and s[0]["peak_reset"] and float(r2["peak_rss_mb"]) == s[0]["peak_rss_mb"]


# =====================================================================================
# E4  FR arm == ABF arm with fr_rate 0 / before fr_start; N = 1
# =====================================================================================
def test_e4_fr_rate_zero_equals_abf_bitwise():
    save = np.arange(0, 1201, 100)
    abf = E.run_arm(small_cfg(N=16, n_steps=1200), "abf", save)
    fr0 = E.run_arm(small_cfg(N=16, n_steps=1200, fr_rate=0.0), "fr", save)
    for k in DYNAMIC_KEYS + ("hist_inst", "region_frac", "traces", "events_window_crossings",
                             "events_translocations", "n_transitions", "n_cage_crossings_lineage",
                             "first_window_step", "first_opposite_step"):
        assert same_bits(abf[k], fr0[k]), k
    assert int(fr0["total_replacement_events"]) == 0 and int(fr0["cum_opps"][-1]) > 0
    # with FR on, everything saved BEFORE fr_start is identical; the IC is shared
    fr = E.run_arm(small_cfg(N=16, n_steps=1200), "fr", save)
    pre = save < 300
    for k in ("M_all", "C_all", "M_prod", "C_prod", "hist_inst", "region_frac"):
        assert same_bits(abf[k][pre], fr[k][pre]), k
    assert same_bits(abf["traces"][0], fr["traces"][0])
    assert int(fr["total_replacement_events"]) > 0
    assert not np.array_equal(abf["q_final"], fr["q_final"])


def test_e4_n1_abf_only():
    c = E.make_cfg(300.0, 1, 3, n_steps=3000, warmup_steps=200, burn_in_steps=200, fr_start_steps=300)
    with pytest.raises(ValueError):
        E.run_arm(c, "fr", np.array([0, 3000]))
    r = E.run_arm(c, "abf", np.array([0, 1500, 3000]))
    assert int(r["n_force_evals"]) == 3001
    assert np.all(r["gen_n_unique"] == 1) and np.all(np.isnan(r["gen_ess"]))
    assert r["traces"].shape[1] == 1 and r["C_all"][-1].sum() == 3001


# =====================================================================================
# E5  diagnostics inert
# =====================================================================================
def test_e5_diagnostics_and_save_grid_inert():
    c_on = small_cfg(N=16, n_steps=1200)
    c_off = small_cfg(N=16, n_steps=1200, diagnostics=False)
    g1 = np.arange(0, 1201, 100)
    g2 = np.unique(np.concatenate([[0, 3, 17, 300, 301, 650, 1199, 1200], np.arange(100, 1201, 300)]))
    runs = {(d, gi): E.run_arm(cc, "fr", g, chunk_steps=cs)
            for d, cc in (("on", c_on), ("off", c_off)) for gi, g, cs in (("g1", g1, 400), ("g2", g2, 333))}
    ref = runs[("on", "g1")]
    for key, r in runs.items():
        for k in ("q_final", "anc_final", "M_all_final", "C_all_final", "M_prod_final", "C_prod_final",
                  "birth_hist", "death_hist", "total_replacement_events", "fr_score_std", "n_force_evals"):
            assert same_bits(ref[k], r[k]), (key, k)
    for gi, g in (("g1", g1), ("g2", g2)):
        a, b = runs[("on", gi)], runs[("off", gi)]
        for k in ("M_all", "C_all", "M_prod", "C_prod", "cum_deaths", "cum_opps", "events_per_opp_hist"):
            assert same_bits(a[k], b[k]), (gi, k)
        # diagnostics off: the diagnostic-only keys are OMITTED (never sentinels that read as 'never' / 'none')
        assert not (set(E.DIAG_ONLY_KEYS) & set(b)) and set(E.DIAG_ONLY_KEYS) <= set(a)
        assert json.loads(b["meta_json"])["omitted_keys"] == list(E.DIAG_ONLY_KEYS)
        assert json.loads(a["meta_json"])["omitted_keys"] == []
    # the common save steps of the two grids carry identical accumulators
    common = np.intersect1d(g1, g2)
    i1 = np.searchsorted(g1, common)
    i2 = np.searchsorted(g2, common)
    for k in ("M_all", "C_all", "M_prod", "C_prod", "hist_inst"):
        assert same_bits(runs[("on", "g1")][k][i1], runs[("on", "g2")][k][i2]), k
    assert int(ref["total_replacement_events"]) > 0


# =====================================================================================
# E6  force-evaluation accounting
# =====================================================================================
@pytest.mark.parametrize("N,method", [(1, "abf"), (4, "fr"), (64, "abf"), (64, "fr")])
def test_e6_force_eval_accounting(N, method):
    n_steps = 400
    c = small_cfg(N=N, n_steps=n_steps)
    r = E.run_arm(c, method, np.array([0, n_steps]), chunk_steps=97)
    meta = json.loads(r["meta_json"])
    assert int(r["n_force_evals"]) == N * (n_steps + 1)       # N per evaluated step, steps 0..n_steps
    assert r["C_all_final"].sum() == N * (n_steps + 1)        # one deposit per force evaluation
    assert meta["B"] == N * n_steps and np.allclose(r["save_u"], [0.0, 1.0])
    assert r["C_prod_final"].sum() == N * (n_steps + 1 - 200)


# =====================================================================================
# E7  speed benchmark (prints; not pass/fail)
# =====================================================================================
def test_e7_speed_benchmark():
    out = []
    for N, n_steps in ((1, 200000), (16, 20000), (1024, 400)):
        for method in (("abf",) if N == 1 else ("abf", "fr")):
            c = E.make_cfg(300.0, N, 1, n_steps=n_steps, warmup_steps=100, burn_in_steps=100,
                           fr_start_steps=100, diagnostics=True)
            E.run_arm(dict(c, n_steps=200), method, np.array([0, 200]))           # warm (compile/cache)
            t = time.perf_counter()
            r = E.run_arm(c, method, np.array([0, n_steps]))
            dt = time.perf_counter() - t
            out.append(f"N {N:5d} {method:3s}: {dt / (N * (n_steps + 1)) * 1e6:.3f} us per molecule-step "
                       f"({dt:.2f} s, events {int(r['total_replacement_events'])})")
    print("\nE7 " + "\nE7 ".join(out))


# =====================================================================================
# X  further checks
# =====================================================================================
def test_x1_true_events_recount_from_dense_traces():
    """ABF (no clones): window-plane crossings and translocations recounted from the phi trace at
    every step agree with the engine's counters; first arrivals too."""
    O, a, L = E.framework()
    N, n = 16, 150000
    q0 = near_window_q0(N, 21, frac=1.0, zlo=0.0, zhi=3.8)
    noise = np.random.default_rng(22).standard_normal((n, N, 2, 3))
    c = E.make_cfg(300.0, N, 0, n_steps=n, warmup_steps=5000, burn_in_steps=5000)
    r = E.run_arm(c, "abf", np.array([0, n // 2, n]), trace_steps=np.arange(n + 1),
                  replay=dict(q0=q0, noise=noise))
    ph = r["traces"]                                            # (n+1, N)
    x0 = 0.5 * (q0[:, 0, 0] + q0[:, 1, 0])
    cell = np.floor(x0 / a).astype(np.int64)
    zcage = 4.0 * E.TWO_PI / a
    zwin = 1.5 * E.TWO_PI / a
    last_cage = np.where(np.abs(ph[0]) > zcage, cell, E.SENT)
    wc = 0
    tl = 0
    first_win = 0 if np.any(np.abs(ph[0]) < zwin) else -1
    first_opp = -1
    for t in range(1, n + 1):
        prev, cur = ph[t - 1], ph[t]
        cross = (np.sign(prev) != np.sign(cur)) & (np.abs(prev) < np.pi / 2) & (np.abs(cur) < np.pi / 2)
        cell = cell + np.where(cross, np.where(cur > prev, 1, -1), 0)
        wc += int(cross.sum())
        if first_win < 0 and np.any(np.abs(cur) < zwin):
            first_win = t
        inc = np.abs(cur) > zcage
        new = inc & (last_cage == E.SENT)
        last_cage = np.where(new, cell, last_cage)
        tr = inc & (last_cage != E.SENT) & (cell != last_cage)
        if tr.any():
            tl += int(tr.sum())
            last_cage = np.where(tr, cell, last_cage)
            if first_opp < 0:
                first_opp = t
    print(f"\nX1 recount: window crossings {wc} (engine {int(r['events_window_crossings'][-1])}), "
          f"translocations {tl} (engine {int(r['events_translocations'][-1])})")
    assert wc == int(r["events_window_crossings"][-1]) and wc > 100
    assert tl == int(r["events_translocations"][-1]) and tl >= 1
    assert first_win == int(r["first_window_step"]) and first_opp == int(r["first_opposite_step"])
    assert int(r["lineage_ever_opposite"][-1]) >= 1
    assert np.all(np.diff(r["events_window_crossings"]) >= 0)


@pytest.mark.parametrize("method", ["abf", "fr"])
def test_x5_cross_process_determinism(tmp_path, method):
    """An arm run in a fresh process (different chunk size) equals the in-process run bitwise (same seed, N, T);
    the FR arm from the fresh process is bitwise paired with the in-process ABF arm before fr_start."""
    import subprocess
    out = str(tmp_path / f"sub_{method}.npz")
    code = ("import sys, numpy as np; sys.path.insert(0, %r); sys.path.insert(0, %r); "
            "import lta_ladder_numba as E; from test_lta_ladder_numba import small_cfg; "
            "E.save_result(%r, E.run_arm(small_cfg(N=16, n_steps=800), %r, np.arange(0, 801, 50), chunk_steps=37))"
            % (os.path.join(ROOT, "src"), os.path.dirname(os.path.abspath(__file__)), out, method))
    subprocess.run([sys.executable, "-c", code], check=True, env=dict(os.environ), timeout=600)
    sub = E.load_result(out)
    here = E.run_arm(small_cfg(N=16, n_steps=800), method, np.arange(0, 801, 50))
    keys = DYNAMIC_KEYS + ("hist_inst", "region_frac", "traces", "traces_anc", "events_window_crossings",
                           "events_translocations", "n_transitions", "n_cage_crossings_lineage", "gen_n_unique",
                           "gen_ess", "gen_n_unique_win", "cum_deaths", "events_per_opp_hist", "birth_hist",
                           "death_hist", "anc_final", "total_replacement_events")
    for k in keys:
        assert same_bits(here[k], sub[k]), k
    assert json.loads(sub["meta_json"])["run_signature"] == json.loads(here["meta_json"])["run_signature"]
    if method == "fr":
        assert int(sub["total_replacement_events"]) > 0
        abf = E.run_arm(small_cfg(N=16, n_steps=800), "abf", np.arange(0, 801, 50))
        pre = abf["save_step"] < 300                                             # fr_start 300
        for k in ("M_all", "C_all", "M_prod", "C_prod", "hist_inst", "region_frac"):
            assert same_bits(abf[k][pre], sub[k][pre]), k
        assert not np.array_equal(abf["q_final"], sub["q_final"])


def test_x6_genealogy_window_reset():
    """Windowed labels reset after the save of every window multiple (here 500 steps); the windowed
    partition refines the whole-run one, so its n_unique and ESS are never smaller."""
    c = small_cfg(N=32, n_steps=1200, fr_rate=200.0)
    assert c["genealogy_window_steps"] == 500
    save = np.array([0, 300, 499, 500, 501, 900, 1000, 1001, 1200])
    r = E.run_arm(c, "fr", save)
    i501, i1001 = 4, 7
    assert r["gen_n_unique_win"][i501] == 32 and r["gen_ess_win"][i501] == 1.0
    assert r["gen_n_unique_win"][i1001] == 32 and r["gen_max_frac_win"][i1001] == 1.0 / 32
    assert np.all(r["gen_n_unique_win"] >= r["gen_n_unique"])
    assert np.all(r["gen_ess_win"] >= r["gen_ess"] - 1e-15)
    assert r["gen_n_unique"][-1] < 32 and r["gen_n_unique_win"][3] < 32         # events happened
    assert np.all(np.diff(r["gen_n_unique"]) <= 0)                               # whole-run never resets
    # bookkeeping: cumulative deaths = sum of k * hist[k]; opportunities with events = sum of hist[1:]
    h = r["events_per_opp_hist"]
    assert np.array_equal(r["cum_deaths"], (h * np.arange(h.shape[1])).sum(1))
    assert np.array_equal(r["cum_opps_with_event"], h[:, 1:].sum(1))
    assert np.array_equal(r["cum_opps"], h.sum(1))


def test_x7_forces_far_from_the_origin_match_torch():
    """Unwrapped coordinates drift over 3e8 steps: forces, CV, bins at |x| up to 1e3 A equal torch's."""
    torch = pytest.importorskip("torch")
    from lta.core_lta import LTAParams, LTASystem, bin_index
    S = LTASystem(LTAParams(temperature=300.0), torch.device("cpu"), torch.float64, root=ROOT)
    O, a, L = E.framework()
    Ox, Oy, Oz = (np.ascontiguousarray(O[:, k]) for k in range(3))
    fp, _, _, _ = E._params(E.make_cfg(300.0, 64, 0), "abf", O, a, L, 0)
    rng = np.random.default_rng(3)
    for scale in (12.0, 1e2, 1e3):
        q = rng.uniform(-scale, scale, (256, 1, 3)) + E.initial_conditions(rng, 256, a, L) - a
        Ft = S.forces(torch.as_tensor(q)).numpy()
        Fn = np.zeros_like(q)
        for i in range(q.shape[0]):
            E._mol_force(q, i, Ox, Oy, Oz, fp, Fn)
        ok = np.abs(Ft).max(axis=(1, 2)) < 1e6                                   # skip hard-core clashes
        rel = np.abs(Fn - Ft)[ok].max() / np.abs(Ft[ok]).max()
        phit = S.cv_value(torch.as_tensor(q)).numpy()
        phin = np.array([E._wrap(fp[E.F_CPHI] * (0.5 * (q[i, 0, 0] + q[i, 1, 0]))) for i in range(q.shape[0])])
        jt = bin_index(torch.as_tensor(phit), 180).numpy()
        jn = np.array([E._bin(v, fp[E.F_DPHI], 180) for v in phin])
        assert rel < 1e-12, (scale, rel)
        assert np.array_equal(phit, phin) and np.array_equal(jt, jn)


def test_x8_deposit_clip_rule_and_small_n_flags():
    """deposit_clip follows core_lta (8 x abf_force_clip) unless overridden explicitly; small-N FR cells are
    flagged in meta (N = 2: score zero by symmetry up to interpolation)."""
    assert E.make_cfg(300.0, 16, 0)["deposit_clip"] == 480.0
    assert E.make_cfg(300.0, 16, 0, abf_force_clip=30.0)["deposit_clip"] == 240.0
    with pytest.raises(ValueError, match="deposit_clip"):
        E.make_cfg(300.0, 16, 0, abf_force_clip=30.0, deposit_clip=480.0)
    c = E.make_cfg(300.0, 16, 0)
    with pytest.raises(ValueError, match="deposit_clip"):                       # a resolved cfg re-edited
        E.resolve_cfg(dict(c, abf_force_clip=30.0))
    c_ov = E.make_cfg(300.0, 16, 0, abf_force_clip=30.0, deposit_clip=480.0, deposit_clip_override=True)
    assert c_ov["deposit_clip"] == 480.0
    O, a, L = E.framework()
    assert E._params(E.make_cfg(300.0, 16, 0, abf_force_clip=30.0), "abf", O, a, L, 0)[0][E.F_DCLIP] == 240.0
    # the frozen production configs resolve to the torch rule
    for T in (300, 150):
        P = json.load(open(os.path.join(ROOT, "configs", "equal_budget_v2", f"lta_{T}K_production.json")))
        cc = E.make_cfg(float(T), 1024, 1, **P["engine_cfg"])
        assert cc["deposit_clip"] == 8.0 * cc["abf_force_clip"]
    r = E.run_arm(small_cfg(N=8, n_steps=60, abf_force_clip=2.0), "abf", np.array([0, 60]))
    m = json.loads(r["meta_json"])
    assert m["deposit_clip"] == 16.0 and m["deposit_clip_rule"].startswith("8 x")
    r = E.run_arm(dict(c_ov, n_steps=60, N=8), "abf", np.array([0, 60]))
    assert json.loads(r["meta_json"])["deposit_clip_rule"].startswith("EXPLICIT OVERRIDE")
    flags = {}
    for N, method in ((2, "fr"), (3, "fr"), (16, "fr"), (2, "abf")):
        m = json.loads(E.run_arm(small_cfg(N=N, n_steps=60), method, np.array([0, 60]))["meta_json"])
        flags[(N, method)] = (m["fr_score_degenerate"], m["fr_small_n_note"] is not None)
    assert flags == {(2, "fr"): (True, True), (3, "fr"): (False, True), (16, "fr"): (False, False),
                     (2, "abf"): (False, False)}


# =====================================================================================
# X9  FR-arm TRUE-event / lineage / genealogy VALUES (hand-scripted trajectories and a free recount)
# =====================================================================================
_O, _A, _L = E.framework()
_OX, _OY, _OZ = (np.ascontiguousarray(_O[:, k]) for k in range(3))


def _mol(xc, y=None, z=None):
    """Molecule (2, 3) with the COM at (xc, y, z), bond 1.54 along x."""
    y = 0.5 * _A + 0.21 if y is None else y
    z = 0.5 * _A - 0.13 if z is None else z
    com = np.array([xc, y, z])
    u = np.array([1.0, 0.0, 0.0])
    return np.stack([com + 0.77 * u, com - 0.77 * u])


def _call_chunk(fp, ip, Kk, save, trace, noise, frb, st, N):
    return E._run_chunk(1, fp, ip, _OX, _OY, _OZ, Kk, save, trace, np.ascontiguousarray(noise), frb,
                        np.zeros((0, N)), np.zeros(0, np.int64), np.zeros(0, np.int64), st["q"], st["anc"],
                        st["ancw"], st["wst"], st["acc"], st["ist"], st["fst"], st["sv_acc"], st["sv_reg"], st["sv_i"],
                        st["sv_f"], st["sv_evh"], st["traces"], st["traces_anc"], st["evh"], st["ev_counts"],
                        st["ev_log"])


class _Driver:
    """Drives the engine's own kernel one evaluated step at a time.  With abf_bias_scale 0 the move is
    q + dt F + ns xi, so xi = (target - (q + dt F)) / ns puts every walker (to rounding) on a scripted target;
    FR deaths are steered through the engine's own uniforms (u = 0 fires a positive-score walker, u = 1 never)."""

    def __init__(self, cfg, method, save, trace, q0):
        self.c = c = E.resolve_cfg(cfg)
        self.method, self.N, self.n_steps = method, c["N"], c["n_steps"]
        self.save, self.trace = np.asarray(save, np.int64), np.asarray(trace, np.int64)
        self.fp, self.ip, self.beta, self.fr_on = E._params(c, method, _O, _A, _L, 0)
        self.Kk = E.kde_matrix(int(c["n_grid"]), float(c["kde_bandwidth"]))[0]
        self.cap, self.fr_start, self.fr_every = int(c["cap"]), int(c["fr_start_steps"]), int(c["fr_every"])
        n_opps = E._n_opps_in(1, self.n_steps, self.fr_start, self.fr_every) if method == "fr" else 0
        self.st = E._new_state(c, q0, self.save.size, self.trace.size, n_opps)
        E._init_labels(self.st["q"], self.fp, self.st["wst"])
        self.q_eval = []

    def step(self, target=None, fr_u=None):
        st, k = self.st, int(self.st["ist"][E.S_STEP])
        self.q_eval.append(st["q"].copy())
        if k >= self.n_steps:
            noise = np.zeros((0, self.N, 2, 3))
        else:
            target = st["q"].copy() if target is None else target
            F = np.zeros_like(st["q"])
            for i in range(self.N):
                E._mol_force(st["q"], i, _OX, _OY, _OZ, self.fp, F)
            noise = ((target - (st["q"] + self.fp[E.F_DT] * F)) / self.fp[E.F_NS])[None]
        nxt = k + 1
        opp = (self.method == "fr" and self.fr_on and k < self.n_steps and nxt >= self.fr_start
               and (nxt - self.fr_start) % self.fr_every == 0)
        row = np.ones((1 if opp else 0, self.N + 2 * self.cap))
        if opp and fr_u is not None:
            row[0, :] = fr_u
        _, iopp = _call_chunk(self.fp, self.ip, self.Kk, self.save, self.trace, noise, row, st, self.N)
        assert iopp == (1 if opp else 0)

    def result(self):
        info = E.seed_streams(self.c["seed"], self.N, self.c["T_K"])[1]
        return E._build_result(self.c, self.method, self.st, self.save, self.trace, None, info, self.fr_on, self.beta,
                               self.fp[E.F_DT], _A, _L, 0.0, 0, [], False)


def _zdist(x):
    f = x / _A - np.floor(x / _A)
    return np.minimum(f, 1.0 - f) * _A


def _region_x(x):
    z = _zdist(x)
    return 0 if z > 4.0 else (2 if z < 1.5 else 1)


def _phi_x(x):
    return 2 * np.pi * ((x / _A + 0.5) % 1.0) - np.pi


def _bin_x(x, ng=180):
    return int(np.floor(((x / _A + 0.5) % 1.0) * ng)) % ng


_X0_PATH = [0.5 * _A + 0.31] * 5 + [
    _A - 2.5, _A - 1.0,              # 5 neck; 6 window (first window 6)
    _A + 1.0, _A - 0.5, _A + 0.5,    # 7, 8, 9 crossings 1-3
    _A + 2.0, _A + 5.0,              # 10 neck; 11 cage cell 1: translocation 1, first opposite 11
    1.5 * _A + 0.31, _A - 1.04,      # 12 cage cell 1; 13 window cell 0: crossing 4
    _A - 5.0,                        # 14 cage cell 0: translocation 2
    2.5 * _A + 0.31,                 # 15 cage cell 2: crossings 5, 6; translocation 3
    -0.5 * _A + 0.31,                # 16 cage cell -1: crossings 7-9; translocation 4
    -0.5 * _A + 0.61, -1.2, -0.3,    # 17 cage -1; 18, 19 window cell -1
    0.3, 4.5] + [4.5] * 9            # 20 crossing 10; 21 cage cell 0: translocation 5; 22..30
_X1 = 0.5 * _A + 0.17


def test_x9a_abf_scripted_events_regions_hist_traces():
    n = len(_X0_PATH) - 1
    c = E.make_cfg(300.0, 2, 0, n_steps=n, warmup_steps=5, burn_in_steps=3, fr_start_steps=10, abf_bias_scale=0.0,
                   genealogy_window_tu=0.002)
    y1 = 1.5 * _A + 0.21
    d = _Driver(c, "abf", np.arange(n + 1), np.arange(n + 1), np.stack([_mol(_X0_PATH[0]), _mol(_X1, y=y1)]))
    for k in range(n + 1):
        d.step(None if k == n else np.stack([_mol(_X0_PATH[k + 1]), _mol(_X1, y=y1)]))
    r = d.result()
    xs = np.array([0.5 * (q[0, 0, 0] + q[0, 1, 0]) for q in d.q_eval])
    assert np.abs(xs - np.array(_X0_PATH)).max() < 1e-9                         # the script was followed
    exp_wc, exp_tl = np.zeros(n + 1, int), np.zeros(n + 1, int)
    for k, inc in {7: 1, 8: 1, 9: 1, 13: 1, 15: 2, 16: 3, 20: 1}.items():
        exp_wc[k:] += inc
    for k in (11, 14, 15, 16, 21):
        exp_tl[k:] += 1
    assert r["events_window_crossings"].tolist() == exp_wc.tolist()
    assert r["events_translocations"].tolist() == exp_tl.tolist()
    assert int(r["first_window_step"]) == 6 and int(r["first_opposite_step"]) == 11
    assert r["lineage_ever_window"].tolist() == [0] * 6 + [1] * (n - 5)
    assert r["lineage_ever_opposite"].tolist() == [0] * 11 + [1] * (n - 10)
    C = np.zeros(180)
    for k in range(n + 1):
        regs = [_region_x(_X0_PATH[k]), _region_x(_X1)]
        assert np.array_equal(r["region_frac"][k], np.array([regs.count(j) for j in range(3)]) / 2.0), k
        h = np.zeros(180)
        h[_bin_x(_X0_PATH[k])] += 1
        h[_bin_x(_X1)] += 1
        C += h
        assert np.array_equal(r["hist_inst"][k], h) and np.array_equal(r["C_all"][k], C), k
        assert abs(r["traces"][k, 0] - _phi_x(_X0_PATH[k])) < 1e-9 and abs(r["traces"][k, 1] - _phi_x(_X1)) < 1e-9
    assert np.all(r["gen_n_unique"] == 2) and np.all(np.isnan(r["gen_ess"])) and np.all(r["cum_opps"] == 0)


_PLAN_B = {1: {0: _A - 2.5, 1: _A - 2.4, 2: _A - 2.3},          # neck at 2
           2: {0: _A - 1.0, 1: _A - 0.9, 2: _A - 0.8},          # window at 3: first window 3
           3: {0: _A - 0.30, 1: _A + 0.06, 2: _A - 0.06},       # slot 1 crosses at 4
           13: {3: 4.0 * _A + 0.23},                            # slot 3 crosses at 14
           14: {3: 4.5 * _A + 0.31},                            # slot 3 cage cell 4: translocation at 15
           17: {2: _A + 2.0},                                   # slot 2 crosses at 18
           18: {2: 1.5 * _A + 0.31},                            # slot 2 cage cell 1: translocation at 19
           19: {0: 3.5 * _A + 0.37},                            # slot 0 (clone of 3) jiggles in cage 3: nothing
           22: {2: 3.5 * _A + 0.43}}                            # slot 2 cage 1 -> 3: 2 crossings + transloc at 23
_DEATHS_B = {10: 0, 30: 3}                                       # nxt -> dying slot (source: lowest S < 0 slot)


def _scenario_b(dying_crosses=True):
    n = 40
    c = E.make_cfg(300.0, 4, 0, n_steps=n, warmup_steps=5, burn_in_steps=3, fr_start_steps=10, fr_every=5,
                   fr_rate=1e9, abf_bias_scale=0.0, genealogy_window_tu=0.005, record_events=True)
    assert c["cap"] == 1 and c["genealogy_window_steps"] == 25
    x = {0: 0.5 * _A + 0.31, 1: 0.5 * _A + 0.37, 2: 0.5 * _A + 0.43, 3: 3.5 * _A + 0.31}
    d = _Driver(c, "fr", np.arange(n + 1), np.arange(n + 1), np.stack([_mol(x[i]) for i in range(4)]))
    plan = dict(_PLAN_B)
    plan[9] = {0: (_A + 0.02) if dying_crosses else (_A - 0.02)}  # slot 0 crosses (or not) just before it dies
    for k in range(n + 1):
        tgt = None
        if k < n:
            com = d.st["q"].mean(1)
            pos = {i: com[i, 0] for i in range(4)}
            pos.update(plan.get(k, {}))
            tgt = np.stack([_mol(pos[i], y=com[i, 1], z=com[i, 2]) for i in range(4)])
        fr_u = None
        if k + 1 in _DEATHS_B:
            fr_u = np.ones(6)
            fr_u[_DEATHS_B[k + 1]] = 0.0
            fr_u[4:] = 0.0
        d.step(tgt, fr_u)
    return d, d.result()


def test_x9b_fr_scripted_clones_lineage_genealogy():
    d, r = _scenario_b()
    assert r["event_log"].tolist() == [[10, 0, 3], [30, 3, 1]]
    exp_wc, exp_tl = np.zeros(41, int), np.zeros(41, int)
    for k, inc in {4: 1, 10: 1, 14: 1, 18: 1, 23: 2}.items():
        exp_wc[k:] += inc
    for k in (15, 19, 23):
        exp_tl[k:] += 1
    assert r["events_window_crossings"].tolist() == exp_wc.tolist()          # clones create no events
    assert r["events_translocations"].tolist() == exp_tl.tolist()
    assert int(r["first_window_step"]) == 3 and int(r["first_opposite_step"]) == 15
    ew = np.zeros(41, int); ew[3:] = 3; ew[10:] = 2; ew[14:] = 3
    eo = np.zeros(41, int); eo[15:] = 1; eo[19:] = 2; eo[30:] = 1
    assert r["lineage_ever_window"].tolist() == ew.tolist()                    # flags inherited by clones
    assert r["lineage_ever_opposite"].tolist() == eo.tolist()
    assert r["gen_n_unique"].tolist() == [4] * 10 + [3] * 31
    assert np.allclose(r["gen_ess"][:10], 1.0) and np.allclose(r["gen_ess"][10:], 16 / 6 / 4)
    assert np.allclose(r["gen_max_frac"][:10], 0.25) and np.allclose(r["gen_max_frac"][10:], 0.5)
    assert r["gen_n_unique_win"].tolist() == [4] * 10 + [3] * 16 + [4] * 4 + [3] * 11   # reset after save 25
    assert np.allclose(r["gen_ess_win"][26:30], 1.0) and np.allclose(r["gen_max_frac_win"][30:], 0.5)
    assert r["cum_deaths"].tolist() == [0] * 10 + [1] * 20 + [2] * 11
    assert r["cum_opps"].tolist() == [sum(1 for o in (10, 15, 20, 25, 30, 35, 40) if o <= k) for k in range(41)]
    assert r["events_per_opp_hist"][-1].tolist() == [5, 2] and int(r["cum_opps_with_event"][-1]) == 2
    ta = r["traces_anc"]
    assert ta[9].tolist() == [0, 1, 2, 3] and ta[10].tolist() == [3, 1, 2, 3] and ta[30].tolist() == [3, 1, 2, 1]
    com = np.array([q.mean(1)[:, 0] for q in d.q_eval])
    assert np.abs(r["traces"] - _phi_x(com)).max() < 1e-9


def test_x9c_dying_walker_last_move_counted_once():
    _, r_cross = _scenario_b(dying_crosses=True)
    _, r_stay = _scenario_b(dying_crosses=False)
    assert r_cross["event_log"].tolist() == r_stay["event_log"].tolist()
    diff = r_cross["events_window_crossings"] - r_stay["events_window_crossings"]
    assert diff[:10].tolist() == [0] * 10 and np.all(diff[10:] == 1)
    assert np.array_equal(r_cross["events_translocations"], r_stay["events_translocations"])


@pytest.mark.parametrize("T,n,seed", [(300.0, 40000, 4), (900.0, 20000, 5)])
def test_x9d_free_fr_run_independent_recount(T, n, seed):
    """Real dynamics and real FR draws: the true events, lineage flags and genealogy (whole-run and windowed) at
    every save recounted from x directly, from the per-step positions, the post-move PRE-copy positions at every
    FR opportunity (each such step re-run on a copy of the state with the copies off) and the event log."""
    N, gwin_tu = 24, 0.2
    c = E.make_cfg(T, N, seed, n_steps=n, warmup_steps=300, burn_in_steps=300, fr_start_steps=400, fr_every=5,
                   fr_rate=40.0, genealogy_window_tu=gwin_tu, record_events=True)
    fp, ip, beta, fr_on = E._params(c, "fr", None, _A, _L, 0)
    ip_off = ip.copy()
    ip_off[E.I_FRON] = 0
    Kk = E.kde_matrix(180, 0.10)[0]
    rng = np.random.default_rng(seed)
    q0 = E.initial_conditions(rng, N, _A, _L)
    for i in range(N // 2):                                                   # near-window starts
        com = np.array([_A * rng.integers(-1, 3) + rng.uniform(-3.0, 3.0), _A * (rng.integers(0, 2) + 0.5),
                        _A * (rng.integers(0, 2) + 0.5)])
        q0[i, 0], q0[i, 1] = com + [0.77, 0, 0], com - [0.77, 0, 0]
    save, trace = np.arange(0, n + 1, 50), np.arange(n + 1)
    st = E._new_state(c, q0, save.size, trace.size, E._n_opps_in(1, n, 400, 5))
    E._init_labels(st["q"], fp, st["wst"])
    q_eval, q_pre = [], {}
    gen_fr = np.random.default_rng(seed + 1)
    for k in range(n + 1):
        q_eval.append(st["q"].copy())
        noise = rng.standard_normal((1 if k < n else 0, N, 2, 3))
        opp = k < n and k + 1 >= 400 and (k + 1 - 400) % 5 == 0
        frb = gen_fr.random((1 if opp else 0, N + 2 * c["cap"]))
        if opp:
            st2 = {kk: st[kk].copy() for kk in E._STATE_KEYS}
            _call_chunk(fp, ip_off, Kk, save, trace, noise, frb, st2, N)
            q_pre[k + 1] = st2["q"].copy()
        _call_chunk(fp, ip, Kk, save, trace, noise, frb, st, N)
    r = E._build_result(c, "fr", st, save, trace, None, E.seed_streams(seed, N, T)[1], fr_on, beta, c["h"], _A, _L,
                        0.0, 0, [], False)
    ev = {}
    for s_, d_, src in r["event_log"].tolist():
        ev.setdefault(s_, []).append((d_, src))
    x0 = 0.5 * (q_eval[0][:, 0, 0] + q_eval[0][:, 1, 0])
    cellv = np.floor(x0 / _A).astype(np.int64)
    lastcage = [int(cellv[i]) if _zdist(x0)[i] > 4.0 else None for i in range(N)]
    evwin, evopp, anc, ancw = np.zeros(N, int), np.zeros(N, int), np.arange(N), np.arange(N)
    cnt = dict(wc=0, tl=0, fw=-1, fo=-1)
    out = {kk: [] for kk in ("wc", "tl", "ew", "eo", "nu", "ess", "mf", "nuw", "essw", "mfw")}

    def inspect(step, x):
        cl, z = np.floor(x / _A).astype(np.int64), _zdist(x)
        for i in range(N):
            cnt["wc"] += abs(int(cl[i]) - int(cellv[i]))
            cellv[i] = cl[i]
            if z[i] < 1.5:
                evwin[i] = 1
                cnt["fw"] = step if cnt["fw"] < 0 else cnt["fw"]
            elif z[i] > 4.0:
                if lastcage[i] is None:
                    lastcage[i] = int(cl[i])
                elif int(cl[i]) != lastcage[i]:
                    cnt["tl"] += 1
                    lastcage[i] = int(cl[i])
                    evopp[i] = 1
                    cnt["fo"] = step if cnt["fo"] < 0 else cnt["fo"]

    def fam(lab):
        cc = np.bincount(lab, minlength=N)
        cc = cc[cc > 0]
        return cc.size, (cc.sum() ** 2 / (cc ** 2).sum()) / N, cc.max() / N

    saves = set(save.tolist())
    gwin = c["genealogy_window_steps"]
    for k in range(n + 1):
        inspect(k, 0.5 * (q_eval[k][:, 0, 0] + q_eval[k][:, 1, 0]))
        if k in saves:
            a_, b_ = fam(anc), fam(ancw)
            for kk, v in zip(out, (cnt["wc"], cnt["tl"], evwin.sum(), evopp.sum()) + a_ + b_):
                out[kk].append(v)
        if k > 0 and k % gwin == 0:
            ancw = np.arange(N)
        if k == n:
            break
        if k + 1 in q_pre:
            qq = q_pre[k + 1].copy()
            inspect(k + 1, 0.5 * (qq[:, 0, 0] + qq[:, 1, 0]))
            for d_, src in ev.get(k + 1, []):
                qq[d_] = qq[src]
                cellv[d_], lastcage[d_], evwin[d_], evopp[d_] = cellv[src], lastcage[src], evwin[src], evopp[src]
                anc[d_], ancw[d_] = anc[src], ancw[src]
            assert np.array_equal(qq, q_eval[k + 1]), k + 1                     # the copies the engine applied
            assert not ({d for d, _ in ev.get(k + 1, [])} & {s for _, s in ev.get(k + 1, [])})
    print(f"\nX9d T {T}: events {len(r['event_log'])}, window crossings {cnt['wc']}, translocations {cnt['tl']}")
    assert len(r["event_log"]) > 50 and cnt["wc"] > 50 and cnt["tl"] > 0
    assert r["events_window_crossings"].tolist() == out["wc"]
    assert r["events_translocations"].tolist() == out["tl"]
    assert int(r["first_window_step"]) == cnt["fw"] and int(r["first_opposite_step"]) == cnt["fo"]
    assert r["lineage_ever_window"].tolist() == out["ew"] and r["lineage_ever_opposite"].tolist() == out["eo"]
    assert r["gen_n_unique"].tolist() == out["nu"] and r["gen_n_unique_win"].tolist() == out["nuw"]
    assert np.abs(r["gen_ess"] - out["ess"]).max() <= 1e-14 and np.abs(r["gen_ess_win"] - out["essw"]).max() <= 1e-14
    assert np.array_equal(r["gen_max_frac"], out["mf"]) and np.array_equal(r["gen_max_frac_win"], out["mfw"])
    for si, k in enumerate(save):
        z = _zdist(0.5 * (q_eval[k][:, 0, 0] + q_eval[k][:, 1, 0]))
        rf = np.array([(z > 4).mean(), ((z >= 1.5) & (z <= 4)).mean(), (z < 1.5).mean()])
        assert np.array_equal(r["region_frac"][si], rf), k


def test_x2_result_round_trip(tmp_path):
    c = small_cfg(N=8, n_steps=500)
    r = E.run_arm(c, "fr", np.arange(0, 501, 100))
    p = str(tmp_path / "res.npz")
    E.save_result(p, r)
    back = E.load_result(p)
    assert set(back) == set(r)
    for k, v in r.items():
        if isinstance(v, str):
            assert back[k] == v
        else:
            assert np.array_equal(np.asarray(back[k]), np.asarray(v), equal_nan=True), k
    meta = json.loads(back["meta_json"])
    assert meta["status"] == "complete" and meta["cap"] == 1 and meta["N"] == 8
    assert meta["knobs"]["fr_start"] == {"steps": 300, "t": 300 * 2e-4}


def test_x3_ladder_grids_caps_and_ic():
    for N in E.LADDER_N:
        c = E.make_cfg(300.0, N, 0)
        assert c["n_steps"] * N == E.BUDGET
        ts, m = E.trace_grid(c)
        assert ts.size < 5000 and ts[0] == 0 and ts[-1] <= c["n_steps"]
        assert np.all(np.diff(ts[ts <= 300000]) == 250)                     # 0.05 t.u. up to 60 t.u.
        assert c["cap"] == max(1, int(0.02 * N))
        assert E.make_cfg(300.0, N, 0, cap_min=0)["cap"] == int(0.02 * N)
        steps, u = E.budget_save_grid(c["n_steps"])
        assert steps[-1] == c["n_steps"]
    assert E.make_cfg(300.0, 1, 0)["n_steps"] == 307_200_000
    # IC: identical for both arms (same seed, N, T), different across seeds; bond 1.54; COM near a cage
    O, a, L = E.framework()
    (g0, _, _), _ = E.seed_streams(5, 64, 300.0)
    (g1, _, _), _ = E.seed_streams(5, 64, 300.0)
    (g2, _, _), _ = E.seed_streams(6, 64, 300.0)
    q_a = E.initial_conditions(g0, 64, a, L)
    q_b = E.initial_conditions(g1, 64, a, L)
    q_c = E.initial_conditions(g2, 64, a, L)
    assert same_bits(q_a, q_b) and not np.array_equal(q_a, q_c)
    assert np.allclose(np.linalg.norm(q_a[:, 0] - q_a[:, 1], axis=-1), 1.54, atol=1e-12)
    com = q_a.mean(1)
    frac = com / a - 0.5
    assert np.all(np.abs(frac - np.round(frac)) * a < 3.0)                  # 0.5 A jitter: < 6 sd
    # the two arms of a run start from the same configuration
    c = small_cfg(N=64, n_steps=10)
    ra = E.run_arm(c, "abf", np.array([0]))
    rf = E.run_arm(c, "fr", np.array([0]))
    assert same_bits(ra["hist_inst"], rf["hist_inst"]) and same_bits(ra["M_all"], rf["M_all"])


@pytest.mark.skipif(not RUN_SLOW, reason="slow: set RUN_SLOW=1")
def test_slow_x4_ladder_knobs_short_full_config():
    """Frozen ladder knobs end to end at N = 1024 for 40000 steps (FR live after 20000)."""
    c = E.make_cfg(300.0, 1024, 0, n_steps=40000)
    r = E.run_arm(c, "fr", E.budget_save_grid(40000, n_lin=20, n_log=4)[0])
    assert int(r["total_replacement_events"]) > 0
    meta = json.loads(r["meta_json"])
    assert meta["cap"] == 20 and meta["fr_rate"] == 0.2
