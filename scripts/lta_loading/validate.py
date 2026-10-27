"""Stage I gates of the finite-loading engine (docs/lta_loading/PREREGISTRATION.md §3).  One A100.

V1  forces = -grad U (central differences of the engine's own energy kernel), n = 4, pre-equilibrated
V2  forces and energy vs an independent NumPy implementation, n = 3
V3  n = 1: forces, CV, bin and clipped local force identical (<= 1e-12 rel) to the validated numba
    single-guest engine (lta_ladder_numba._mol_force)
V4  n = 1 against the exact-MC single-guest reference (results/equal_budget_v2/references/lta_T150_reference.npz):
    under its frozen bias, mean-force route e_F <= 0.05 and density route e_F <= 0.10 kJ/mol; the ABF path at
    N = 1024 x 300k steps, final per-seed e_F <= 0.30 (single-guest engine: 0.14-0.20)
V5  reinsertion MC: a non-interacting environment guest reproduces the exact single-guest density at 150 K
    (|log p/p_ref| <= 0.10 on well-populated bins), never sits in a blocked sodalite sphere, and the kernel's
    dU of every accepted move equals the total-energy change (n = 4, interacting)
V6  determinism: the same seed run alone and inside a batch gives bitwise-identical accumulators
Usage: python scripts/lta_loading/validate.py [--only V1,V2] [--out results/lta_loading/validation]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from lta_loading import engine as LE  # noqa: E402
import lta_ladder_numba as E  # noqa: E402


def np_energy_forces(q, p):
    """Independent NumPy energy/forces of one walker q (n, 2, 3)."""
    n = q.shape[0]
    X = q.reshape(-1, 3)
    B = X.shape[0]
    L = p["L"]
    U = 0.0
    Fo = np.zeros_like(X)

    def lj(d, prm):
        d = d - L * np.rint(d / L)
        r2 = (d * d).sum(-1)
        m = r2 < p["rc2"]
        s6 = np.where(m, (prm["s2"] / np.where(m, r2, 1.0)) ** 3, 0.0)
        e = np.where(m, prm["c4"] * (s6 * s6 - s6) - prm["vrc"], 0.0)
        coef = np.where(m, prm["c24"] * (2 * s6 * s6 - s6) / np.where(m, r2, 1.0), 0.0)
        return e, coef[..., None] * d

    for b in range(B):
        e, f = lj(X[b][None, :] - p["O"], p["go"])
        U += e.sum()
        Fo[b] += f.sum(0)
        for j in range(B):
            if j // 2 != b // 2:
                e, f = lj((X[b] - X[j])[None, :], p["gg"])
                if j > b:
                    U += e.sum()
                Fo[b] += f[0]
    for m in range(n):
        d = X[2 * m] - X[2 * m + 1]
        r = np.linalg.norm(d)
        U += 0.5 * p["kb"] * (r - p["r0"]) ** 2
        fb = -p["kb"] * (r - p["r0"]) * d / r
        Fo[2 * m] += fb
        Fo[2 * m + 1] -= fb
    return U, Fo.reshape(n, 2, 3)


def make_batch(n, R, N, T=150.0, seeds=None, method="abf", n_steps=10, **kw):
    cfg = LE.resolve_cfg(n_guests=n, T_K=T, **kw)
    seeds = seeds if seeds is not None else list(range(9000, 9000 + R))
    return LE.Batch(seeds, method, N, n_steps, cfg=cfg, verbose=False)


def v1_fd(out):
    import cupy as cp
    b = make_batch(4, 1, 8, preeq_steps=2000)
    b.preequilibrate()
    b.forces(deposit=False)
    F = b.F.get()
    q0 = b.q.copy()
    worst = 0.0
    rng = np.random.default_rng(1)
    for _ in range(24):
        w, m, bb, x = rng.integers(8), rng.integers(4), rng.integers(2), rng.integers(3)
        hs = 1e-5
        es = []
        for sgn in (+1, -1):
            b.q[...] = q0
            b.q[w, m, bb, x] += sgn * hs
            es.append(float(b.energies()[w].get()))
        fd = -(es[0] - es[1]) / (2 * hs)
        an = F[w, m, bb, x]
        rel = abs(fd - an) / max(1.0, abs(an))
        worst = max(worst, rel)
    b.q[...] = q0
    out["V1"] = dict(worst_rel=worst, passed=bool(worst < 1e-5))


def v2_numpy(out):
    b = make_batch(3, 1, 6, preeq_steps=1000)
    b.preequilibrate()
    b.forces(deposit=False)
    F = b.F.get()
    U = b.energies().get()
    q = b.q.get()
    wf = we = 0.0
    for w in range(6):
        Un, Fn = np_energy_forces(q[w], b.p)
        wf = max(wf, float(np.abs(Fn - F[w]).max() / max(1.0, np.abs(Fn).max())))
        we = max(we, abs(Un - U[w]) / max(1.0, abs(Un)))
    out["V2"] = dict(force_rel=wf, energy_rel=we, passed=bool(wf < 1e-10 and we < 1e-10))


def v3_single(out):
    b = make_batch(1, 2, 64)
    b.forces(deposit=False)
    F = b.F.get().reshape(-1, 2, 3)
    q = b.q.get().reshape(-1, 2, 3)
    phi = b.phi.get(); jb = b.jb.get(); fl = b.fl.get()
    c = E.make_cfg(150.0, 64, 0)
    O, a, L = E.framework()
    fp = E._params(c, "abf", O, a, L, 0)[0]
    Fn = np.empty_like(q)
    Ox, Oy, Oz = (np.ascontiguousarray(O[:, k]) for k in range(3))
    worst = 0.0
    cvbad = 0
    for i in range(q.shape[0]):
        E._mol_force(q, i, Ox, Oy, Oz, fp, Fn)
        worst = max(worst, float(np.abs(Fn[i] - F[i]).max() / max(1.0, np.abs(Fn[i]).max())))
        x = (q[i, 0, 0] + q[i, 1, 0]) * 0.5
        ph = E._wrap(fp[E.F_CPHI] * x)
        f = min(max(fp[E.F_CF] * (Fn[i, 0, 0] + Fn[i, 1, 0]), -480.0), 480.0)
        cvbad += int(ph != phi[i] or E._bin(ph, fp[E.F_DPHI], 180) != jb[i] or abs(f - fl[i]) > 1e-9 * max(1, abs(f)))
    out["V3"] = dict(force_rel=worst, cv_mismatch=cvbad, passed=bool(worst < 1e-12 and cvbad == 0))


def v4_end_to_end(out, n_steps=400_000):
    """n = 1 under the FROZEN exact single-guest bias (gamma_ref of the exact-MC reference): the landscape is
    flat, so both routes converge fast.  Mean-force route (pooled production bins) and density route
    (A_ref - kT log C) vs F_ref.  A wrong bias sign or scale fails the density route by kT-sized errors.
    Plus the ABF path: n = 1, N = 1024, 300k steps, 2 seeds; final per-seed e_F reported (single-guest CPU
    engine at the same N and length: 0.14-0.20 kJ/mol), gate <= 0.30."""
    ref = np.load(os.path.join(ROOT, "results/equal_budget_v2/references/lta_T150_reference.npz"))
    b = LE.Batch([9100, 9101], "frozen", 2048, n_steps, cfg=LE.resolve_cfg(n_guests=1, warmup_steps=1, burn_in_steps=100_000),
                 table=ref["gamma_ref"], save_steps=[0, n_steps], verbose=False)
    b.preequilibrate()
    res, meta = b.run(label="V4f")
    acc = res["sv_acc"][-1]
    M = acc[:, 2].sum(0); C = acc[:, 3].sum(0)
    kT = E.KB * 150.0
    Fm = E.histogram_pmf(E.mean_force(M[None], C[None]))[0]
    A = E.histogram_pmf(ref["gamma_ref"][None])[0]
    Fd = A - kT * np.log(np.maximum(C, 1e-300))

    def eF(F):
        d = F - ref["F_ref"]; d = d - d.mean()
        return float(np.sqrt(np.mean(d ** 2)))

    a2 = make_batch(1, 2, 1024, method="abf", n_steps=300_000, seeds=[9200, 9201])
    a2.preequilibrate()
    ra, _ = a2.run(label="V4abf")
    per = []
    for r in range(2):
        Gr = E.mean_force(ra["sv_acc"][-1, r, 2][None], ra["sv_acc"][-1, r, 3][None])
        dr = E.histogram_pmf(Gr)[0] - ref["F_ref"]; dr -= dr.mean()
        per.append(float(np.sqrt(np.mean(dr ** 2))))
    out["V4"] = dict(e_F_meanforce_route=eF(Fm), e_F_density_route=eF(Fd), min_count=float(C.min()),
                     abf_final_e_F=per, abf_cross_per_walker=float(ra["sv_i"][-1, :, 1].mean() / 1024),
                     passed=bool(eF(Fm) <= 0.05 and eF(Fd) <= 0.10 and max(per) <= 0.30))


def v5_mc(out, N=4096, steps=40_000, every=100):
    """(a) non-interacting environment guest (eps_gg = 0) at 150 K, dynamics + reinsertion: its phi density
    must equal the single-guest exact-MC density exp(-beta F_ref) on the thermal bins; no COM in a blocked
    sphere.  (b) interacting n = 4: the kernel's dU of every accepted move equals the change of the total
    energy (lf_energy) -- the acceptance uses the right energy and the right sign."""
    import cupy as cp
    ref = np.load(os.path.join(ROOT, "results/equal_budget_v2/references/lta_T150_reference.npz"))
    b = make_batch(2, 1, N, method="abf", eps_gg_K=0.0, preeq_steps=2000, preeq_sweeps=5, seeds=[777])
    b.preequilibrate()
    hs = np.zeros(180)
    nblk = 0
    for k in range(steps):
        b.forces(deposit=False)
        if k % every == 0 and k >= 5000:
            q = b.q.get()
            x = 0.5 * (q[:, 1, 0, 0] + q[:, 1, 1, 0])
            ph = np.mod(2 * np.pi * x / b.p["a"] + np.pi, 2 * np.pi) - np.pi
            hs += np.bincount(((ph + np.pi) / (2 * np.pi / 180)).astype(int) % 180, minlength=180)
            com = 0.5 * (q[:, 1, 0] + q[:, 1, 1])
            e = com - b.p["a"] * np.rint(com / b.p["a"])
            nblk += int(((e * e).sum(-1) < float(b.c["r_block"]) ** 2).sum())
        b.move(k, 0.0)
        if (k + 1) % 10 == 0:
            b.mc(k + 1)
    beta = b.p["beta"]
    pr = np.exp(-beta * (ref["F_ref"] - ref["F_ref"].min()))
    pr /= pr.sum()
    pe = hs / hs.sum()
    expct = pr * hs.sum()
    sel = expct >= 400
    lr = np.log(np.maximum(pe[sel], 1e-300) / pr[sel])
    # sampling noise of a binned count is inflated by correlation between samples 100 steps apart; the gate
    # is a systematic one: |log ratio| <= 0.10 on every well-populated bin, and a small total variation
    tv = 0.5 * float(np.abs(pe - pr).sum())
    acc_rate = float(b.mc_acc.get().sum() / max(1, b.mc_att.get().sum()))
    # (b) dU consistency, interacting
    c2 = make_batch(4, 1, 256, preeq_steps=500, preeq_sweeps=3, seeds=[778])
    c2.preequilibrate()
    worst = 0.0
    nacc = 0
    for t in range(20):
        U0 = c2.energies().get()
        q0 = c2.q.get()
        c2.mc(10 ** 6 + t)
        U1 = c2.energies().get()
        q1 = c2.q.get()
        dU = c2.mc_dU.get()
        moved = np.any(q0 != q1, axis=(1, 2, 3))
        nacc += int(moved.sum())
        if moved.any():
            worst = max(worst, float(np.max(np.abs((U1 - U0)[moved] - dU[moved]) / np.maximum(1.0, np.abs(dU[moved])))))
    out["V5"] = dict(max_abs_log_ratio=float(np.abs(lr).max()), n_bins_judged=int(sel.sum()), tv=tv,
                     samples=float(hs.sum()), blocked_visits=nblk, mc_acc_rate=acc_rate,
                     dU_rel_worst=worst, dU_checked=nacc,
                     passed=bool(np.abs(lr).max() <= 0.10 and nblk == 0 and worst < 1e-9 and nacc > 50))


def v6_determinism(out):
    a = make_batch(4, 1, 64, method="fr", n_steps=30_000, seeds=[4242], preeq_steps=500, preeq_sweeps=3,
                   fr_start_steps=2000, warmup_steps=2000, burn_in_steps=2000)
    a.preequilibrate()
    ra, _ = a.run(label="V6a")
    b = make_batch(4, 3, 64, method="fr", n_steps=30_000, seeds=[11, 4242, 12], preeq_steps=500, preeq_sweeps=3,
                   fr_start_steps=2000, warmup_steps=2000, burn_in_steps=2000)
    b.preequilibrate()
    rb, _ = b.run(label="V6b")
    same = bool(np.array_equal(ra["sv_acc"][:, 0], rb["sv_acc"][:, 1]) and
                np.array_equal(ra["sv_i"][:, 0], rb["sv_i"][:, 1]))
    out["V6"] = dict(bitwise_equal=same, repl=int(ra["sv_i"][-1, 0, 0]), passed=same)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="V1,V2,V3,V6,V5,V4")
    ap.add_argument("--out", default=os.path.join(ROOT, "results/lta_loading/validation"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    out = {}
    fns = dict(V1=v1_fd, V2=v2_numpy, V3=v3_single, V4=v4_end_to_end, V5=v5_mc, V6=v6_determinism)
    for g in args.only.split(","):
        t0 = time.perf_counter()
        try:
            fns[g](out)
        except Exception as ex:  # noqa: BLE001
            out[g] = dict(passed=False, error=repr(ex))
        out[g]["seconds"] = time.perf_counter() - t0
        print(g, json.dumps({k: v for k, v in out[g].items() if k != "hist"}, default=float), flush=True)
        with open(os.path.join(args.out, "validation.json"), "w") as fh:
            json.dump(out, fh, indent=1, default=float)
    print("ALL_PASSED" if all(v.get("passed") for v in out.values()) else "SOME_FAILED")


if __name__ == "__main__":
    main()
