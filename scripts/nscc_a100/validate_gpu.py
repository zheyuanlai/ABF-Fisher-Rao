"""Stage I correctness gates of the A100 backends against the validated CPU engines (docs/nscc_a100/PREREGISTRATION.md).

Run on an NSCC A100 node (one GPU visible, the job's CPU cores for the CPU oracle):

    python scripts/nscc_a100/validate_gpu.py --system gateway --out results/nscc_a100/validation
    python scripts/nscc_a100/validate_gpu.py --system lta150 --out results/nscc_a100/validation

Gates (each writes its numbers into <out>/<system>_validation.json; the last line printed is the verdict):
  RNG1   C++ PCG64 == numpy PCG64 (the FR uniform stream), bitwise
  RNG2   GPU counter-mode normals vs the CPU counter mode: identical Philox integers by construction; reports the
         fraction of bitwise-equal normals and the max |dz| (CUDA vs glibc libm)
  FR     the GPU FR selection == the validated numba FR on identical post-move states and uniform blocks
         (event counts, cap split and the full gather index), many random opportunities
  TRAJ   GPU vs CPU walker-parallel counter mode (same IC, Philox key and FR stream): saves agree to ulp-level
         tolerance while the trajectories are pathwise coupled; reports the first save where integer counts differ
  INV    launch-configuration invariance: different (blocks, threads, wpt) give bitwise-identical science arrays
  DET    two runs of one job give bitwise-identical science arrays
  CTR    the result passes scripts/final_campaign/expB_common.check_contract
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for _p in (os.path.join(ROOT, "src"), os.path.join(ROOT, "src", "gpu"), os.path.join(ROOT, "scripts"),
           os.path.join(ROOT, "scripts", "final_campaign"), os.path.join(ROOT, "scripts", "hybrid_short_T"),
           os.path.join(ROOT, "scripts", "equal_budget"), os.path.join(ROOT, "scripts", "mechanism")):
    if _p not in sys.path:
        sys.path.append(_p)


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def science_sha(res, kind):
    keys = ("M_all", "C_all", "fr_deaths_cum", "fr_opp_cum", "X_final", "Y_final") if kind == "gateway" else \
        ("M_all", "C_all", "M_prod", "C_prod", "cum_deaths", "cum_opps", "q_final")
    return {k: sha(np.asarray(res[k])) for k in keys}


# ================================================================================================ gateway
def gateway_gates(out, quick=False):
    import gateway_ladder_numba as gl
    import gateway_gpu as GG
    import hst_common as H
    import expB_common as C
    spec = H.system_spec("gateway")
    cfg = dict(spec["engine_cfg"])
    rep = {}

    # RNG1
    words = GG.pcg64_state_words(9695, 512)
    n = 1 << 20
    a = GG.pcg64_doubles(words, n)
    b = np.random.Generator(np.random.PCG64(np.random.SeedSequence(gl.fr_seed_entropy(9695, 512)))).random(n)
    rep["RNG1"] = dict(n=n, bitwise=bool(np.array_equal(a, b)), passed=bool(np.array_equal(a, b)))
    print("RNG1", rep["RNG1"], flush=True)

    # RNG2 (CPU counter normals from the parallel engine's C Philox)
    import parallel.gateway_parallel as GP
    key = GP.counter_key(9695, 512)
    zc = GP.counter_normals(key, 12345, 200, 512, nthreads=4)
    zg = GG.counter_normals(key, 12345, 200, 512)
    d = np.abs(zc - zg)
    rel = d / np.maximum(np.abs(zc), 1e-300)
    rep["RNG2"] = dict(n=int(zc.size), frac_bitwise=float(np.mean(zc == zg)), max_abs=float(d.max()),
                       max_rel=float(rel.max()), passed=bool(d.max() < 1e-12))
    print("RNG2", rep["RNG2"], flush=True)

    # FR: states from a short CPU run (realistic positions), many opportunities, a range of g
    rng = np.random.default_rng(20261026)
    res_fr = []
    n_mis = 0
    n_ev = 0
    for N in (16, 64, 512, 2048):
        c = gl.full_cfg(cfg)
        kn = gl.derived_knobs(c, N)
        dx = gl.grid_dx()
        kern, r = gl.gaussian_kernel_np(float(c["eta"]), dx)
        L = int(kn["fr_block"])
        n_opp = 40 if quick else 150
        for t in range(n_opp):
            # mixture: left basin / gateway / right basin, plus reflections at the walls
            w = rng.dirichlet([2, 1, 2])
            comp = rng.choice(3, size=N, p=w)
            X = np.where(comp == 0, rng.normal(-1.0, 0.15, N), np.where(comp == 1, rng.normal(0.0, 0.08, N),
                                                                       rng.normal(1.0, 0.15, N)))
            X = np.clip(X, -1.8, 1.8)
            u = rng.random(L)
            g = float(c["gamma"]) * rng.choice([0.0, 0.05, 0.5, 1.0, 20.0])
            p = np.zeros(gl.N_GRID)
            hist = np.zeros(gl.N_GRID)
            Sc = np.zeros(N)
            gl.kde_density(X, N, kern, r, dx, p, hist)
            gl.uniform_scores(X, N, p, dx, float(c["score_clip"]), Sc)
            sel = np.zeros(N, dtype=np.int64)
            die_c = np.zeros(N, dtype=np.int64)
            clone_c = np.zeros(N, dtype=np.int64)
            pool = np.zeros(2 * N, dtype=np.int64)
            kd, kc, _ = gl.fr_resample(Sc, N, g, float(kn["dt_fr"]), int(kn["cap"]), sel, die_c, clone_c, pool,
                                       u, 0, False)
            if kd + kc == 0:
                sel = np.arange(N)
            sg, kdg, kcg, _, _, _ = GG.fr_test(N, X, u, g, cfg, threads=min(1024, max(32, -(-N // 32) * 32)))
            ok = (kd == kdg and kc == kcg and np.array_equal(sel, sg))
            n_mis += (not ok)
            n_ev += (kd + kc > 0)
        res_fr.append(dict(N=N, n_opp=n_opp))
    rep["FR"] = dict(cases=res_fr, mismatches=int(n_mis), opportunities_with_events=int(n_ev),
                     passed=bool(n_mis == 0))
    print("FR", rep["FR"], flush=True)

    # TRAJ: GPU vs CPU parallel counter mode
    traj = []
    for N, method, n_steps in ((64, "abf", 40000), (64, "fr", 40000), (512, "fr", 20000), (2048, "fr", 8000)):
        if quick:
            n_steps //= 4
        save = np.unique(np.linspace(1, n_steps, 40).astype(np.int64))
        rc = GP.run_arm_parallel(cfg, method, N, 9695, n_steps, save, threads=min(16, N), noise="counter")
        rg = GG.run_arm_gpu(cfg, method, N, 9695, n_steps, save)
        C.check_contract(rg, "gateway", N, n_steps, save)
        Mc, Mg = np.asarray(rc["M_all"]), np.asarray(rg["M_all"])
        Cc, Cg = np.asarray(rc["C_all"]), np.asarray(rg["C_all"])
        scale = np.maximum(np.abs(Mc), 1.0)
        dM = np.max(np.abs(Mc - Mg) / scale, axis=1)
        same_C = np.all(Cc == Cg, axis=1)
        same_fr = np.all(np.asarray(rc["fr_deaths_cum"]) == np.asarray(rg["fr_deaths_cum"])) if method == "fr" \
            else True
        first_diff = int(save[np.argmin(same_C)]) if not same_C.all() else None
        dX = float(np.max(np.abs(np.asarray(rc["X_final"]) - np.asarray(rg["X_final"]))))
        row = dict(N=N, method=method, n_steps=n_steps, saves=int(save.size),
                   max_rel_dM_while_C_equal=float(dM[same_C].max()) if same_C.any() else None,
                   first_save_C_differs=first_diff, C_identical_all=bool(same_C.all()),
                   fr_deaths_identical=bool(same_fr), max_abs_dX_final=dX,
                   deaths_cpu=int(np.asarray(rc["fr_deaths_cum"])[-1]),
                   deaths_gpu=int(np.asarray(rg["fr_deaths_cum"])[-1]),
                   wall_cpu=float(rc["wall_total_s"]), wall_gpu=float(rg["wall_total_s"]))
        # the gate (both systems): identical counts over the first quarter of the run, accumulators within 1e-6
        q = save <= n_steps // 4
        row["max_rel_dM_first_quarter"] = float(dM[q].max())
        row["passed"] = bool(same_C[q].all() and (dM[q].max() < 1e-6))
        traj.append(row)
        print("TRAJ", row, flush=True)
    rep["TRAJ"] = dict(rows=traj, passed=bool(all(r["passed"] for r in traj)))

    # INV + DET
    N, n_steps = 512, (6000 if quick else 20000)
    save = np.unique(np.linspace(1, n_steps, 30).astype(np.int64))
    shas = {}
    for launch in ((1, 512, 1, 0), (2, 256, 1, 0), (4, 128, 1, 0), (1, 256, 2, 0), (8, 64, 1, 0), (2, 512, 1, 1),
                   (4, 256, 1, 1), (2, 256, 2, 1)):
        r = GG.run_arm_gpu(cfg, "fr", N, 9696, n_steps, save, launch=launch)
        shas[str(launch)] = science_sha(r, "gateway")
    r2 = GG.run_arm_gpu(cfg, "fr", N, 9696, n_steps, save, launch=(1, 512, 1, 0))
    vals = list(shas.values())
    rep["INV"] = dict(configs=list(shas), identical=bool(all(v == vals[0] for v in vals)),
                      passed=bool(all(v == vals[0] for v in vals)))
    rep["DET"] = dict(identical=bool(science_sha(r2, "gateway") == vals[0]),
                      passed=bool(science_sha(r2, "gateway") == vals[0]))
    print("INV", rep["INV"], flush=True)
    print("DET", rep["DET"], flush=True)
    rep["CTR"] = dict(passed=True, note="check_contract passed on every TRAJ GPU run")
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, choices=("gateway", "lta150", "lta300"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "nscc_a100", "validation"))
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args(argv)
    t = time.time()
    if a.system == "gateway":
        rep = gateway_gates(a.out, a.quick)
    else:
        import validate_gpu_lta as VL
        rep = VL.lta_gates(a.system, a.out, a.quick)
    rep["wall_s"] = time.time() - t
    rep["host"] = os.uname().nodename
    rep["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"{a.system}_validation{'_quick' if a.quick else ''}.json")
    with open(path, "w") as f:
        json.dump(rep, f, indent=1, default=str)
    gates = {k: v.get("passed") for k, v in rep.items() if isinstance(v, dict) and "passed" in v}
    verdict = "PASS" if all(gates.values()) else "FAIL"
    print(f"VERDICT {a.system} {verdict} {gates} -> {path}", flush=True)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
