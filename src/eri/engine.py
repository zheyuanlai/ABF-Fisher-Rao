"""ERI campaign ABF sampler: histogram (P0) ABF with and without uniform-target marginal Fisher-Rao birth-death.

Two arms only (user instruction 2026-10-16): ``abf`` and ``fr``.  CPU numba, one process per run, deterministic.

Physics  : src/eri/model.py (PNAS-2005-type ERI surrogate), overdamped Euler-Maruyama with unit mobility,
           q <- q + dt (F + bias) + sqrt(2 dt kT) xi.  Each replica is a full configuration of M ethane molecules
           (M = 1: infinite dilution); molecule 0 is the TAGGED molecule whose phi is the CV.
CV       : phi = wrap(2 pi (X_com,0 - a/4) / (a/2)) (windows at 0, cage rows at +-pi), linear, no geometric term;
           local mean force f = -(P / 2 pi)(F_00x + F_01x).
Estimator: the textbook online histogram (P0) estimator on n_grid circular bins, accumulated over ALL steps and ALL
           replicas from step 0 (C_j(n), M_j(n) of the campaign prompt), deposit BEFORE read, own-bin force
           Gamma_j = M_j / C_j, bias = ramp(t) * min(1, C_j / trust_count) * Gamma_j applied as +bias * grad phi to the
           TAGGED molecule only; ramp = min(1, step / warmup_steps); empty bins give 0.  No deposit clip; an optional
           bias read clip ``bias_clip`` (default None = off) whose activations are counted.  Reported free energy =
           exact integral of the piecewise-constant Gamma with the circular mean subtracted (histogram_pmf).
FR       : the validated LTA-ladder machinery IMPORTED unchanged (lta_ladder_numba._fr_score / fr_select_native):
           KDE (unnormalised wrapped Gaussian, kde_bandwidth) of the post-move phi, uniform target, score
           log p - log q - KL recentred and 3 x clipped (+-score_clip); death prob 1 - exp(-rate S+ dt fr_every);
           cap = max(cap_min, floor(max_event_fraction N)) per opportunity; sources ~ S-; the WHOLE replica
           (all M molecules) is copied, with its labels; descendants get independent noise because noise is drawn
           per slot.
RNG      : three PCG64 streams from SeedSequence([seed, N, M, round(T_K), variant_id]).spawn(3): 0 initial
           conditions, 1 Langevin noise (N*M*6 normals per step in slot order), 2 FR uniforms (N + 2 cap per
           opportunity).  The abf and fr arms of one (seed, N, M, T) share streams 0 and 1 slot by slot, so they are
           BITWISE identical until the first FR opportunity (tested).
Events   : TRUE events only (computed from a slot's own move; clones inherit labels, a copy is never an event):
           window crossings = changes of the tagged row index floor((X - a/4)/P); translocations = arrival in the
           cage region |phi| >= cage_phi in a row different from the slot's last cage row.
Saves    : at completed-step counts ``save_steps`` (the state at the START of that step, accumulators INCLUDING
           its deposit): raw C, M; instantaneous phi histogram; event counters; genealogy (whole-run and windowed
           ancestor ESS / max family share); per coarse bin instantaneous hidden-variable sums (eta, u_x^2).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time

import numpy as np
from numba import njit

from . import model as mdl
from .model import config_energy_force

import sys as _sys
_sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
from lta_ladder_numba import _fr_score, fr_select_native, kde_matrix  # noqa: E402  (validated FR machinery)

ENGINE = "eri_engine/1"
METHODS = ("abf", "fr")
TWO_PI = 2.0 * math.pi
EPS = 1.0e-12
SENT = -(2 ** 62)
VARIANT_ID = {"gard_tait": 1, "schlenker": 2}
N_COARSE = 18

DEFAULT_CFG = dict(
    variant="gard_tait", T_K=600.0, N=1024, M=1, seed=0,
    dt=1.0e-4, n_steps=300_000, save_every=1000,
    n_grid=180, warmup_steps=10_000, trust_count=0.0, bias_clip=None,
    fr_start_steps=10_000, fr_every=10, fr_rate=0.2, score_clip=2.0, kde_bandwidth=0.10,
    max_event_fraction=0.02, cap_min=1,
    window_phi=0.5, cage_phi=2.0,          # region bookkeeping (rad): |phi| < window_phi window, >= cage_phi cage
    genealogy_window_steps=20_000,
    init="cage",                           # "cage": tagged molecule (and guests) at random ERI-cage centres
    equil_steps=0,                         # unbiased EM pre-equilibration of the starting configurations (stream 0)
)

# integer state layout
(S_WCROSS, S_TRANSLOC, S_REPL, S_NOPP, S_NOPPEV, S_FIRSTWIN, S_FIRSTTRANS, S_CLIPHITS, S_NFE) = range(9)
NIST = 9
# per-save int columns
(V_WCROSS, V_TRANSLOC, V_REPL, V_NWIN, V_NCAGE, V_NU, V_NUW, V_NOPP, V_NOPPEV) = range(9)
NV = 9
# per-save float columns
(U_ESS, U_MAXF, U_ESSW, U_MAXFW, U_SSTD, U_GMAX) = range(6)
NU = 6


def _engine_sha256():
    with open(os.path.abspath(__file__), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


@njit(cache=True, error_model="numpy")
def _wrap(p):
    return (p + math.pi) - TWO_PI * math.floor((p + math.pi) / TWO_PI) - math.pi


@njit(cache=True, error_model="numpy")
def _family_stats(lab, N, fam):
    for i in range(N):
        fam[lab[i]] += 1
    nu = 0
    ssq = 0.0
    mx = 0
    for i in range(N):
        c = fam[i]
        if c > 0:
            nu += 1
            ssq += float(c) * float(c)
            if c > mx:
                mx = c
            fam[i] = 0
    return nu, float(N) * float(N) / ssq / float(N), float(mx) / float(N)


@njit(cache=True, error_model="numpy")
def _labels(i, step, xt, P, xw, ph, cage_phi, window_phi, row, lastcage, ist, evwin):
    """Update slot i's labels from its own current position; count TRUE events."""
    r = np.int64(math.floor((xt - xw) / P))
    if r != row[i]:
        ist[S_WCROSS] += abs(r - row[i])
        row[i] = r
    if abs(ph) < window_phi:
        evwin[i] = 1
        if ist[S_FIRSTWIN] < 0:
            ist[S_FIRSTWIN] = step
    if abs(ph) >= cage_phi:
        if lastcage[i] == SENT:
            lastcage[i] = r
        elif lastcage[i] != r:
            ist[S_TRANSLOC] += 1
            lastcage[i] = r
            if ist[S_FIRSTTRANS] < 0:
                ist[S_FIRSTTRANS] = step


@njit(cache=True, error_model="numpy")
def _run_chunk(k0, k1, q, F, Cacc, Macc, Seta, Sux2, BH, DH, anc, ancw, row, lastcage, evwin, ist, fst,
               fp, ip, box, cell, nvox, s0, x0, y0, z0, s1, x1, y1, z1, pp, rc2, kb, r0, Kk,
               rng_noise, rng_fr, save_steps, isave, sv_C, sv_M, sv_h, sv_i, sv_u, sv_cond):
    """Steps k0 <= k < k1 (k = completed-step count at the START of the step).  Returns the next save index."""
    N = q.shape[0]
    M = q.shape[1]
    dt = fp[0]
    ns = fp[1]
    P = fp[2]
    xw = fp[3]
    rate = fp[4]
    dteff = fp[5]
    sclip = fp[6]
    trust = fp[7]
    bclip = fp[8]
    window_phi = fp[9]
    cage_phi = fp[10]
    grid0 = fp[11]
    qv = fp[12]
    lq = fp[13]
    c_len = fp[14]
    ng = ip[0]
    nsteps = ip[1]
    warm = ip[2]
    frstart = ip[3]
    frevery = ip[4]
    cap = ip[5]
    is_fr = ip[6] == 1
    gwin = ip[7]
    dphi = TWO_PI / ng
    gx = math.pi / P
    cf = -(P / TWO_PI)
    phi = np.zeros(N)
    jb = np.zeros(N, dtype=np.int64)
    fl = np.zeros(N)
    Fi = np.zeros((M, 2, 3))
    # FR work arrays
    phin = np.zeros(N)
    jn = np.zeros(N, dtype=np.int64)
    cnt = np.zeros(ng)
    pk = np.zeros(ng)
    raw = np.zeros(N)
    sc = np.zeros(N)
    touched = np.zeros(ng, dtype=np.int64)
    cand = np.zeros(N, dtype=np.int64)
    dsel = np.zeros(max(cap, 1), dtype=np.int64)
    ssel = np.zeros(max(cap, 1), dtype=np.int64)
    dp = np.zeros(N)
    cdf = np.zeros(N)
    ublk = np.zeros(N + 2 * max(cap, 1))
    fam = np.zeros(N, dtype=np.int64)
    dzw = 2.0 * TWO_PI / c_len
    for k in range(k0, k1):
        # (1) forces / CV / labels at the start-of-step positions
        for i in range(N):
            config_energy_force(q[i], box, cell, nvox, s0, x0, y0, z0, s1, x1, y1, z1, pp, rc2, kb, r0, F[i])
            xt = 0.5 * (q[i, 0, 0, 0] + q[i, 0, 1, 0])
            ph = _wrap(TWO_PI * (xt - xw) / P)
            phi[i] = ph
            j = np.int64(math.floor((ph + math.pi) / dphi)) % ng
            jb[i] = j
            fl[i] = cf * (F[i, 0, 0, 0] + F[i, 0, 1, 0])
            _labels(i, k, xt, P, xw, ph, cage_phi, window_phi, row, lastcage, ist, evwin)
        ist[S_NFE] += N
        # (2) deposits of ALL walkers (sequential slot order), hidden-variable conditional sums
        for i in range(N):
            j = jb[i]
            Cacc[j] += 1.0
            Macc[j] += fl[i]
            dx = q[i, 0, 0, 0] - q[i, 0, 1, 0]
            dy = q[i, 0, 0, 1] - q[i, 0, 1, 1]
            dz = q[i, 0, 0, 2] - q[i, 0, 1, 2]
            rb2 = dx * dx + dy * dy + dz * dz
            Sux2[j] += dx * dx / rb2
            Seta[j] += math.cos(dzw * 0.5 * (q[i, 0, 0, 2] + q[i, 0, 1, 2]))
        # (3) save (state at the start of step k, accumulators including its deposit)
        if isave < save_steps.shape[0] and save_steps[isave] == k:
            for j in range(ng):
                sv_C[isave, j] = Cacc[j]
                sv_M[isave, j] = Macc[j]
                sv_h[isave, j] = 0.0
            for c in range(N_COARSE):
                sv_cond[isave, c, 0] = 0.0
                sv_cond[isave, c, 1] = 0.0
                sv_cond[isave, c, 2] = 0.0
            nwin = 0
            ncage = 0
            for i in range(N):
                sv_h[isave, jb[i]] += 1.0
                if abs(phi[i]) < window_phi:
                    nwin += 1
                if abs(phi[i]) >= cage_phi:
                    ncage += 1
                cbin = (jb[i] * N_COARSE) // ng
                dx = q[i, 0, 0, 0] - q[i, 0, 1, 0]
                dy = q[i, 0, 0, 1] - q[i, 0, 1, 1]
                dz = q[i, 0, 0, 2] - q[i, 0, 1, 2]
                rb2 = dx * dx + dy * dy + dz * dz
                sv_cond[isave, cbin, 0] += 1.0
                sv_cond[isave, cbin, 1] += math.cos(dzw * 0.5 * (q[i, 0, 0, 2] + q[i, 0, 1, 2]))
                sv_cond[isave, cbin, 2] += dx * dx / rb2
            sv_i[isave, V_WCROSS] = ist[S_WCROSS]
            sv_i[isave, V_TRANSLOC] = ist[S_TRANSLOC]
            sv_i[isave, V_REPL] = ist[S_REPL]
            sv_i[isave, V_NWIN] = nwin
            sv_i[isave, V_NCAGE] = ncage
            sv_i[isave, V_NOPP] = ist[S_NOPP]
            sv_i[isave, V_NOPPEV] = ist[S_NOPPEV]
            nu, ess, mxf = _family_stats(anc, N, fam)
            nuw, essw, mxfw = _family_stats(ancw, N, fam)
            sv_i[isave, V_NU] = nu
            sv_i[isave, V_NUW] = nuw
            sv_u[isave, U_ESS] = ess
            sv_u[isave, U_MAXF] = mxf
            sv_u[isave, U_ESSW] = essw
            sv_u[isave, U_MAXFW] = mxfw
            sv_u[isave, U_SSTD] = fst[0] / max(ist[S_NOPP], 1)
            sv_u[isave, U_GMAX] = fst[1]
            isave += 1
        if gwin > 0 and k > 0 and k % gwin == 0:
            for i in range(N):
                ancw[i] = i
        if k == nsteps:
            break
        # (4) bias from the accumulators including this step's deposit, then the EM move
        ramp = 1.0 if warm <= 0 else min(1.0, k / warm)
        for i in range(N):
            j = jb[i]
            c = Cacc[j]
            g = Macc[j] / c if c > 0.0 else 0.0
            w = 1.0
            if trust > 0.0:
                w = c / trust if c < trust else 1.0
            b = ramp * w * g
            if abs(b) > fst[1]:
                fst[1] = abs(b)
            if bclip > 0.0 and abs(b) > bclip:
                b = bclip if b > 0.0 else -bclip
                ist[S_CLIPHITS] += 1
            bf = b * gx
            for m in range(M):
                for bb in range(2):
                    for d in range(3):
                        dr = F[i, m, bb, d] * dt
                        if m == 0 and d == 0:
                            dr += bf * dt
                        q[i, m, bb, d] = q[i, m, bb, d] + dr + ns * rng_noise.standard_normal()
        nxt = k + 1
        # (5) Fisher-Rao at the POST-move positions
        if is_fr and nxt >= frstart and (nxt - frstart) % frevery == 0:
            for i in range(N):
                xt = 0.5 * (q[i, 0, 0, 0] + q[i, 0, 1, 0])
                ph = _wrap(TWO_PI * (xt - xw) / P)
                phin[i] = ph
                jn[i] = np.int64(math.floor((ph + math.pi) / dphi)) % ng
                _labels(i, nxt, xt, P, xw, ph, cage_phi, window_phi, row, lastcage, ist, evwin)
            ist[S_NOPP] += 1
            sstd, sabs = _fr_score(phin, jn, N, ng, dphi, Kk, grid0, qv, lq, sclip, cnt, pk, raw, sc, touched)
            fst[0] += sstd
            for u in range(N + 2 * cap):
                ublk[u] = rng_fr.random()
            nev = 0
            if cap >= 1 and rate > 0.0:
                nev = fr_select_native(sc, N, cap, rate, dteff, ublk, cand, dsel, ssel, dp, cdf)
            if nev > 0:
                ist[S_NOPPEV] += 1
            for t in range(nev):
                d_ = dsel[t]
                s_ = ssel[t]
                BH[jn[s_]] += 1.0
                DH[jn[d_]] += 1.0
                for m in range(M):
                    for bb in range(2):
                        for d in range(3):
                            q[d_, m, bb, d] = q[s_, m, bb, d]
                anc[d_] = anc[s_]
                ancw[d_] = ancw[s_]
                row[d_] = row[s_]
                lastcage[d_] = lastcage[s_]
                evwin[d_] = evwin[s_]
            ist[S_REPL] += nev
    return isave


# ----------------------------------------------------------------------------------------------------------------
# host side
# ----------------------------------------------------------------------------------------------------------------
def make_cfg(**over):
    c = dict(DEFAULT_CFG)
    for k in over:
        if k not in c:
            raise KeyError(f"unknown config key {k!r}")
    c.update(over)
    c["cap"] = max(int(c["cap_min"]), int(math.floor(c["max_event_fraction"] * c["N"])))
    return c


def config_hash(c, method):
    d = {k: c[k] for k in sorted(c)}
    d["method"] = method
    d["engine"] = ENGINE
    return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:16]


def seed_streams(c):
    ss = np.random.SeedSequence([int(c["seed"]), int(c["N"]), int(c["M"]), int(round(c["T_K"])),
                                 VARIANT_ID[c["variant"]]])
    return [np.random.Generator(np.random.PCG64(s)) for s in ss.spawn(3)]


def initial_conditions(gen, model, N, M, r0=1.54):
    """Every molecule at a random ERI-cage centre + 0.3 N(0,1) A jitter, uniform orientation; guests of one replica
    in DISTINCT cages chosen uniformly (M <= 72); then 2000 steps of steepest-descent-free relaxation is NOT applied
    (EM from the start; the cage centres are far from the walls)."""
    cages = model.fw.cages
    q = np.zeros((N, M, 2, 3))
    for i in range(N):
        pick = gen.permutation(len(cages))[:M]
        for m in range(M):
            com = cages[pick[m]] + 0.3 * gen.standard_normal(3)
            u = gen.standard_normal(3)
            u /= np.linalg.norm(u)
            q[i, m, 0] = com + 0.5 * r0 * u
            q[i, m, 1] = com - 0.5 * r0 * u
    return q


@njit(cache=True, error_model="numpy")
def _equilibrate(q, n_steps, dt, ns, rng, box, cell, nvox, s0, x0, y0, z0, s1, x1, y1, z1, pp, rc2, kb, r0):
    """Unbiased overdamped EM for every replica (no deposits, no bias, no FR): the starting configurations relax in
    their cages.  Uses the IC stream, so both arms of a (seed, N, M, T) start from the same equilibrated state."""
    N = q.shape[0]
    M = q.shape[1]
    F = np.zeros((M, 2, 3))
    for i in range(N):
        for k in range(n_steps):
            config_energy_force(q[i], box, cell, nvox, s0, x0, y0, z0, s1, x1, y1, z1, pp, rc2, kb, r0, F)
            for m in range(M):
                for b in range(2):
                    for d in range(3):
                        q[i, m, b, d] = q[i, m, b, d] + dt * F[m, b, d] + ns * rng.standard_normal()


def save_grid(c):
    s = np.arange(0, c["n_steps"] + 1, int(c["save_every"]), dtype=np.int64)
    if s[-1] != c["n_steps"]:
        s = np.append(s, c["n_steps"])
    return s


class Run:
    """One (cfg, method) run with chunked execution and checkpoint/resume."""

    def __init__(self, c, method, model=None):
        if method not in METHODS:
            raise ValueError(method)
        if method == "fr" and c["N"] < 2:
            raise ValueError("no FR at N = 1")
        self.c = c
        self.method = method
        self.model = model if model is not None else mdl.Model(c["variant"], T_K=c["T_K"])
        m = self.model
        g_ic, self.g_noise, self.g_fr = seed_streams(c)
        N, M = int(c["N"]), int(c["M"])
        ng = int(c["n_grid"])
        self.q = initial_conditions(g_ic, m, N, M, m.r0)
        if int(c["equil_steps"]) > 0:
            _equilibrate(self.q, int(c["equil_steps"]), float(c["dt"]), math.sqrt(2.0 * c["dt"] * m.kT), g_ic,
                         m.fw.box, *m.fw.packed(), m.pp, m.rc2, m.kb, m.r0)
        self.F = np.zeros_like(self.q)
        self.C = np.zeros(ng)
        self.Msum = np.zeros(ng)
        self.Seta = np.zeros(ng)
        self.Sux2 = np.zeros(ng)
        self.BH = np.zeros(ng)
        self.DH = np.zeros(ng)
        self.anc = np.arange(N, dtype=np.int64)
        self.ancw = np.arange(N, dtype=np.int64)
        P, xw = m.fw.P, m.fw.x_window
        X = 0.5 * (self.q[:, 0, 0, 0] + self.q[:, 0, 1, 0])
        self.row = np.floor((X - xw) / P).astype(np.int64)
        self.lastcage = np.full(N, SENT, dtype=np.int64)
        self.evwin = np.zeros(N, dtype=np.int64)
        self.ist = np.zeros(NIST, dtype=np.int64)
        self.ist[S_FIRSTWIN] = -1
        self.ist[S_FIRSTTRANS] = -1
        self.fst = np.zeros(2)
        self.save_steps = save_grid(c)
        ns_ = len(self.save_steps)
        self.sv_C = np.zeros((ns_, ng))
        self.sv_M = np.zeros((ns_, ng))
        self.sv_h = np.zeros((ns_, ng))
        self.sv_i = np.zeros((ns_, NV), dtype=np.int64)
        self.sv_u = np.zeros((ns_, NU))
        self.sv_cond = np.zeros((ns_, N_COARSE, 3))
        self.isave = 0
        self.k = 0
        self.Kk, _, _ = kde_matrix(ng, c["kde_bandwidth"])
        dphi = TWO_PI / ng
        self.fp = np.array([c["dt"], math.sqrt(2.0 * c["dt"] * m.kT), P, xw, c["fr_rate"],
                            c["dt"] * max(int(c["fr_every"]), 1), c["score_clip"], float(c["trust_count"]),
                            float(c["bias_clip"]) if c["bias_clip"] else -1.0, c["window_phi"], c["cage_phi"],
                            -math.pi + 0.5 * dphi, 1.0 / (ng * dphi), math.log(1.0 / (ng * dphi)), m.fw.c])
        self.ip = np.array([ng, int(c["n_steps"]), int(c["warmup_steps"]), int(c["fr_start_steps"]),
                            int(c["fr_every"]), int(c["cap"]), 1 if method == "fr" else 0,
                            int(c["genealogy_window_steps"])], dtype=np.int64)
        self.sessions = []

    def advance(self, k_target):
        m = self.model
        k_target = min(int(k_target), int(self.c["n_steps"]) + 1)
        if k_target <= self.k:
            return
        self.isave = _run_chunk(self.k, k_target, self.q, self.F, self.C, self.Msum, self.Seta, self.Sux2,
                                self.BH, self.DH, self.anc, self.ancw, self.row, self.lastcage, self.evwin,
                                self.ist, self.fst, self.fp, self.ip, m.fw.box, *m.fw.packed(), m.pp, m.rc2, m.kb,
                                m.r0, self.Kk, self.g_noise, self.g_fr, self.save_steps, self.isave, self.sv_C,
                                self.sv_M, self.sv_h, self.sv_i, self.sv_u, self.sv_cond)
        self.k = k_target

    @property
    def done(self):
        return self.k > int(self.c["n_steps"])

    _STATE = ("q", "C", "Msum", "Seta", "Sux2", "BH", "DH", "anc", "ancw", "row", "lastcage", "evwin", "ist", "fst",
              "sv_C", "sv_M", "sv_h", "sv_i", "sv_u", "sv_cond")

    def checkpoint(self, path):
        tmp = path + ".tmp.npz"
        np.savez(tmp, **{k: getattr(self, k) for k in self._STATE}, k=self.k, isave=self.isave,
                 rng=json.dumps([self.g_noise.bit_generator.state, self.g_fr.bit_generator.state], default=int),
                 sig=config_hash(self.c, self.method), engine_sha256=_engine_sha256())
        os.replace(tmp, path)

    def restore(self, path):
        z = np.load(path, allow_pickle=False)
        if str(z["sig"]) != config_hash(self.c, self.method):
            raise RuntimeError("checkpoint signature mismatch")
        if str(z["engine_sha256"]) != _engine_sha256():
            raise RuntimeError("engine changed since the checkpoint")
        for k in self._STATE:
            setattr(self, k, np.array(z[k]))
        self.k = int(z["k"])
        self.isave = int(z["isave"])
        st = json.loads(str(z["rng"]))
        self.g_noise.bit_generator.state = st[0]
        self.g_fr.bit_generator.state = st[1]

    def result(self):
        c = self.c
        return dict(cfg=json.dumps(c, default=str), method=self.method, config_hash=config_hash(c, self.method),
                    engine=ENGINE, engine_sha256=_engine_sha256(), save_steps=self.save_steps,
                    times=self.save_steps * c["dt"], sv_C=self.sv_C, sv_M=self.sv_M, sv_h=self.sv_h, sv_i=self.sv_i,
                    sv_u=self.sv_u, sv_cond=self.sv_cond, Seta=self.Seta, Sux2=self.Sux2, BH=self.BH, DH=self.DH,
                    ist=self.ist, fst=self.fst, q_final=self.q, n_force_evals=int(self.ist[S_NFE]),
                    n_equil_force_evals=int(c["N"]) * int(c["equil_steps"]),
                    P=self.model.fw.P, x_window=self.model.fw.x_window, a=self.model.fw.a, c_len=self.model.fw.c)


def run(c, method, out_path=None, checkpoint_path=None, checkpoint_every_s=600.0, chunk_steps=2000, model=None,
        verbose=False):
    """Run to completion with periodic checkpoints; resume from ``checkpoint_path`` if it exists."""
    r = Run(c, method, model=model)
    if checkpoint_path and os.path.exists(checkpoint_path):
        r.restore(checkpoint_path)
    t0 = time.time()
    tlast = t0
    while not r.done:
        r.advance(r.k + chunk_steps)
        if checkpoint_path and time.time() - tlast > checkpoint_every_s and not r.done:
            r.checkpoint(checkpoint_path)
            tlast = time.time()
            if verbose:
                print(f"  ckpt k={r.k} {time.time() - t0:.0f}s", flush=True)
    res = r.result()
    res["wall_seconds_session"] = time.time() - t0
    if out_path:
        tmp = out_path + ".tmp.npz"
        np.savez(tmp, **res)
        os.replace(tmp, out_path)
        if checkpoint_path and os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
    return res


# ----------------------------------------------------------------------------------------------------------------
# read-outs
# ----------------------------------------------------------------------------------------------------------------
def mean_force(Mv, Cv):
    return np.where(Cv > 0, Mv / np.maximum(Cv, 1.0), 0.0)


def histogram_pmf(G):
    dphi = TWO_PI / G.shape[-1]
    g0 = G - G.mean(-1, keepdims=True)
    Fr = np.cumsum(g0 * dphi, axis=-1)
    Fc = Fr - 0.5 * g0 * dphi
    return Fc - Fc.mean(-1, keepdims=True)
