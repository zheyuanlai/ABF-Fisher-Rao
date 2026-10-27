"""Stage I gates of the LTA A100 backend (called by validate_gpu.py --system lta150 | lta300).

  RNG1   C++ PCG64 == numpy PCG64 child 2 (the FR uniform stream), bitwise
  RNG2   GPU counter normals vs parallel.philox.lta_noise_block (CPU counter mode)
  FORCE  GPU forces / CV / local force / bin vs the validated numba _mol_force, _wrap, _bin on states of a CPU run
  FR     GPU FR (score + selection) vs the validated numba _fr_score + fr_select_native on identical inputs
  TRAJ   GPU vs CPU walker-parallel counter mode (same IC, Philox key, FR stream), production knobs, FR active
  INV    bitwise invariance across WB (walkers per block) at fixed G; DET two runs bitwise
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


TRAJ_TOL = 1e-6          # relative accumulator tolerance of the TRAJ gate (campaign.json stage1_gates)


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def lta_gates(system, out, quick=False):
    import lta_ladder_numba as E
    import lta_gpu as LG
    import hst_common as H
    import expB_common as C
    import parallel.lta_parallel as LP
    from parallel import philox as PX
    spec = H.system_spec(system)
    T = float(spec["T_K"])
    seed = spec["validation_seeds"][0]
    rep = {}

    # RNG1
    c = C.engine_cfg_for(spec, 64, seed, 1000)
    s = LG.setup(c, "fr")
    gens, _ = E.seed_streams(seed, 64, T)
    n = 1 << 20
    import gateway_gpu as GG
    a = GG.pcg64_doubles(s["pcg"], n)
    b = gens[2].random(n)
    rep["RNG1"] = dict(n=n, bitwise=bool(np.array_equal(a, b)), passed=bool(np.array_equal(a, b)))
    print("RNG1", rep["RNG1"], flush=True)

    # RNG2
    k0, k1 = PX.lta_key(seed, 64, T)
    zc = PX.lta_noise_block(k0, k1, 20000, 100, 64)
    zg = LG.counter_normals(s["key"], 20000, 100, 64)
    d = np.abs(zc - zg)
    rep["RNG2"] = dict(n=int(zc.size), frac_bitwise=float(np.mean(zc == zg)), max_abs=float(d.max()),
                       key_equal=bool(int(k0) == int(s["key"][0]) and int(k1) == int(s["key"][1])),
                       passed=bool(d.max() < 1e-12 and int(k0) == int(s["key"][0])))
    print("RNG2", rep["RNG2"], flush=True)

    # states from a CPU run (for FORCE and FR inputs)
    N = 256
    nst = 2000 if quick else 6000
    cN = C.engine_cfg_for(spec, N, seed, nst)
    rc = LP.run_arm_parallel(cN, "abf", N, seed, nst, np.array([nst]), threads=16, noise="counter")
    q = np.asarray(rc["q_final"])
    rng = np.random.default_rng(20261026)
    qs = np.concatenate([q, q + rng.normal(0, 0.3, q.shape), s["q0"][:64]], axis=0)
    cR = LG.resolve(cN, N, seed, nst)
    O, aa, L = E.framework(cR["framework_npz"])
    fp, ipn, beta, fr_on = E._params(cR, "abf", O, aa, L, 0)
    Ox, Oy, Oz = (np.ascontiguousarray(O[:, k]) for k in range(3))
    Fc = np.zeros((qs.shape[0], 2, 3))
    ph_c = np.zeros(qs.shape[0])
    fl_c = np.zeros(qs.shape[0])
    jb_c = np.zeros(qs.shape[0], dtype=np.int64)
    for i in range(qs.shape[0]):
        E._mol_force(qs, i, Ox, Oy, Oz, fp, Fc)
        x = (qs[i, 0, 0] + qs[i, 1, 0]) * 0.5
        ph_c[i] = E._wrap(fp[E.F_CPHI] * x)
        f = fp[E.F_CF] * (Fc[i, 0, 0] + Fc[i, 1, 0])
        fl_c[i] = min(max(f, -fp[E.F_DCLIP]), fp[E.F_DCLIP])
        jb_c[i] = E._bin(ph_c[i], fp[E.F_DPHI], int(cR["n_grid"]))
    frows = []
    for G in (8, 32, 128, 256):
        Fg, ph_g, fl_g, jb_g = LG.force_test(cR, qs, G=G)
        rel = np.max(np.abs(Fg - Fc)) / max(np.max(np.abs(Fc)), 1e-300)
        frows.append(dict(G=G, max_rel_F=float(rel), max_abs_F=float(np.max(np.abs(Fg - Fc))),
                          phi_identical=bool(np.array_equal(ph_g, ph_c)), bins_identical=bool(np.array_equal(jb_g, jb_c)),
                          max_abs_fl=float(np.max(np.abs(fl_g - fl_c)))))
    rep["FORCE"] = dict(rows=frows, n=int(qs.shape[0]),
                        passed=bool(all(r["max_rel_F"] < 1e-12 and r["bins_identical"] for r in frows)))
    print("FORCE", rep["FORCE"], flush=True)

    # FR selection on identical inputs: perturbed CPU states (clustered) at several N
    n_mis = 0
    n_ev = 0
    score_dev = 0.0
    cases = []
    for NN in (2, 16, 64, 256, 1024):
        cc = LG.resolve(C.engine_cfg_for(spec, NN, seed, 1000), NN, seed, 1000)
        fpp, _, _, _ = E._params(cc, "fr", O, aa, L, 0)
        Kk, _, _ = E.kde_matrix(int(cc["n_grid"]), float(cc["kde_bandwidth"]))
        cap = int(cc["cap"])
        ng = int(cc["n_grid"])
        n_opp = 40 if quick else 120
        for t in range(n_opp):
            idx = rng.integers(0, q.shape[0], NN)
            qq = q[idx] + rng.normal(0, rng.choice([0.05, 0.5, 2.0]), (NN, 2, 3))
            x = (qq[:, 0, 0] + qq[:, 1, 0]) * 0.5
            ph = np.array([E._wrap(fpp[E.F_CPHI] * v) for v in x])
            jn = np.array([E._bin(p, fpp[E.F_DPHI], ng) for p in ph], dtype=np.int64)
            ublk = rng.random(NN + 2 * cap)
            rate = float(fpp[E.F_RATE]) * rng.choice([1.0, 30.0, 1000.0])
            cnt = np.zeros(ng); pk = np.zeros(ng); raw = np.zeros(NN); sc = np.zeros(NN)
            touched = np.zeros(ng, dtype=np.int64)
            E._fr_score(ph, jn, NN, ng, fpp[E.F_DPHI], Kk, fpp[E.F_GRID0], fpp[E.F_QV], fpp[E.F_LQ],
                        fpp[E.F_SCLIP], cnt, pk, raw, sc, touched)
            cand = np.zeros(NN, dtype=np.int64); dsel = np.zeros(NN, dtype=np.int64)
            ssel = np.zeros(NN, dtype=np.int64); dp = np.zeros(NN); cdf = np.zeros(NN)
            nc = E.fr_select_native(sc, NN, cap, rate, fpp[E.F_DTEFF], ublk, cand, dsel, ssel, dp, cdf)
            cg = dict(cc, fr_rate=rate)
            ng_, dg, sg, scg, _, _ = LG.fr_test(cg, ph, jn, ublk, threads=min(1024, max(32, -(-NN // 32) * 32)))
            ok = (nc == ng_ and np.array_equal(dsel[:nc], dg) and np.array_equal(ssel[:nc], sg))
            n_mis += (not ok)
            n_ev += (nc > 0)
            score_dev = max(score_dev, float(np.max(np.abs(scg - sc))))
        cases.append(dict(N=NN, n_opp=n_opp))
    rep["FR"] = dict(cases=cases, mismatches=int(n_mis), opportunities_with_events=int(n_ev),
                     max_abs_score_dev=score_dev, passed=bool(n_mis == 0 and score_dev < 1e-9))
    print("FR", rep["FR"], flush=True)

    # TRAJ: GPU vs CPU parallel counter mode on the production knobs (FR from step 20000)
    rows = []
    for NN, method, nst in ((16, "fr", 40000), (64, "fr", 30000), (256, "abf", 24000), (1024, "fr", 23000)):
        if quick:
            nst = min(nst, 21000)
        cfg = C.engine_cfg_for(spec, NN, seed, nst)
        save = np.unique(np.concatenate([np.arange(0, 5001, 50), np.linspace(0, nst, 30).astype(np.int64),
                                         [20000, 20001]]))
        save = save[save <= nst]
        rcpu = LP.run_arm_parallel(cfg, method, NN, seed, nst, save, threads=min(16, NN), noise="counter")
        rg = LG.run_arm_gpu(cfg, method, NN, seed, nst, save)
        C.check_contract(rg, "lta", NN, nst, save)
        Cc, Cg = np.asarray(rcpu["C_all"]), np.asarray(rg["C_all"])
        Mc, Mg = np.asarray(rcpu["M_all"]), np.asarray(rg["M_all"])
        same_C = np.all(Cc == Cg, axis=1)
        dM = np.max(np.abs(Mc - Mg) / np.maximum(np.abs(Mc), 1.0), axis=1)
        dq = float(np.max(np.abs(np.asarray(rcpu["q_final"]) - np.asarray(rg["q_final"]))))
        row = dict(N=NN, method=method, n_steps=nst, C_identical_all=bool(same_C.all()),
                   first_save_C_differs=(int(save[np.argmin(same_C)]) if not same_C.all() else None),
                   max_rel_dM_while_C_equal=(float(dM[same_C].max()) if same_C.any() else None),
                   deaths_cpu=int(np.asarray(rcpu["cum_deaths"])[-1]), deaths_gpu=int(np.asarray(rg["cum_deaths"])[-1]),
                   opps_cpu=int(np.asarray(rcpu["cum_opps"])[-1]), opps_gpu=int(np.asarray(rg["cum_opps"])[-1]),
                   max_abs_dq_final=dq, wall_cpu=float(rcpu["wall_total_s"]), wall_gpu=float(rg["wall_total_s"]))
        # Gate (both systems): identical integer bin counts at every save of the first quarter of the run, identical
        # FR opportunity counts, accumulators within TRAJ_TOL relative there; horizon and whole-run numbers reported
        horizon = int(save[np.argmin(same_C)]) if not same_C.all() else nst
        row["coupling_horizon_steps"] = horizon
        qmask = save <= max(1, nst // 4)
        row["max_rel_dM_first_quarter"] = float(dM[qmask].max())
        row["passed"] = bool(same_C[qmask].all() and dM[qmask].max() < TRAJ_TOL
                             and row["opps_cpu"] == row["opps_gpu"])
        rows.append(row)
        print("TRAJ", row, flush=True)
    rep["TRAJ"] = dict(rows=rows, passed=bool(all(r["passed"] for r in rows)))

    # INV (WB at fixed G) + DET
    NN, nst = 256, (21000 if quick else 24000)
    cfg = C.engine_cfg_for(spec, NN, seed + 1, nst)
    save = np.unique(np.linspace(0, nst, 25).astype(np.int64))
    keys = ("M_all", "C_all", "M_prod", "C_prod", "cum_deaths", "cum_opps", "q_final")
    shas = {}
    for launch in ((32, 4), (32, 8), (32, 10), (32, 15)):
        r = LG.run_arm_gpu(cfg, "fr", NN, seed + 1, nst, save, launch=launch)
        shas[str(launch)] = {k: sha(np.asarray(r[k])) for k in keys}
    r2 = LG.run_arm_gpu(cfg, "fr", NN, seed + 1, nst, save, launch=(32, 4))
    vals = list(shas.values())
    rep["INV"] = dict(configs=list(shas), identical=bool(all(v == vals[0] for v in vals)),
                      passed=bool(all(v == vals[0] for v in vals)))
    d2 = {k: sha(np.asarray(r2[k])) for k in keys}
    rep["DET"] = dict(identical=bool(d2 == vals[0]), passed=bool(d2 == vals[0]))
    print("INV", rep["INV"], flush=True)
    print("DET", rep["DET"], flush=True)
    rep["CTR"] = dict(passed=True, note="check_contract passed on every TRAJ GPU run")
    return rep
