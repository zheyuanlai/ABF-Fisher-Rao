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
Fibre     : whole-run per-bin deposit sums of four fibre coordinates of the tagged molecule: eta = cos(4 pi Z_com / c)
           (Seta), u_x^2 and u_z^2 (Sux2, Suz2; u = unit bond vector) and the bond length (Sbond).
Selection : ``diag_sel`` = 1 (default 0) turns on a READ-ONLY, bitwise-inert record of every FR birth / death (step,
diagnostic parent and killed slot, their CV bins, their ancestor tags, and the fibre state of both at the event), an
           ancestor tag per walker (= index of the most recent birth event that created the walker's slot content,
           -1 for founders), and per-event deposit accumulators: C, M per fine bin split at ``diag_sel_split_step``
           (default n_steps // 2: "early" k <= split, "late" k > split) and, per coarse bin and time-since-birth band
           (``diag_sel_band_steps``), the count and the sums of f, eta, u_x^2, bond, u_z^2.  Founders' deposits go to
           separate arrays.  Both arms also histogram the four fibre coordinates per fine bin and half (``sel_fh``),
           which gives the ABF arm's conditional law for class definitions.  No RNG draw, no write to any
           trajectory or estimator state (tested: tests/test_diag_sel.py).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time

import numpy as np
from numba import njit, prange

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
    fr_stop_steps=-1,                      # early-stopping FR: events only at steps fr_start <= k < fr_stop (-1: never stop)
    fr_controller=None,                    # online stop rule spec (src/eri/fr_controller.py), host-side, reference-free
    diag_age_steps=0,                      # diagnostic only: split deposits of walkers cloned < this many steps ago
    diag_sel=0,                            # diagnostic only (read-only): birth/death record + ancestor-tagged deposits
    diag_sel_band_steps=None,              # time-since-birth band edges (steps); None: 0.1, 0.3, 1, 3, 10 t.u.
    diag_sel_split_step=-1,                # deposits at k > split are "late" (-1: n_steps // 2)
    window_phi=0.5, cage_phi=2.0,          # region bookkeeping (rad): |phi| < window_phi window, >= cage_phi cage
    genealogy_window_steps=20_000,
    init="cage",                           # "cage": tagged molecule (and guests) at random ERI-cage centres
    equil_steps=0,                         # unbiased EM pre-equilibration of the starting configurations (stream 0)
    ff=None,                               # force-field parameter dict (None: model.PARAMS, the ERI/Schuering set)
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


# selection diagnostic layout
NFH = 40                                   # histogram bins per fibre coordinate
BOND_HALF = 0.4                            # bond histogram range r0 +- BOND_HALF (A); eta in [-1, 1], u^2 in [0, 1]
NEVI = 7                                   # event ints: step, parent, killed, parent bin, killed bin, parent tag, killed tag
NEVF = 8                                   # event floats: parent (eta, ux2, bond, uz2), killed (eta, ux2, bond, uz2)
NSB = 6                                    # per (half, band, coarse bin): count, f, eta, ux2, bond, uz2
SEL_BANDS_TU = (0.1, 0.3, 1.0, 3.0, 10.0)  # default band edges in t.u. (6 bands)

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROV_FILES = (os.path.join(_SRC_DIR, "engine.py"), os.path.join(_SRC_DIR, "model.py"),
               os.path.join(_SRC_DIR, "framework.py"), os.path.join(_SRC_DIR, "fr_controller.py"),
               os.path.normpath(os.path.join(_SRC_DIR, "..", "lta_ladder_numba.py")))


def _sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# provenance taken ONCE at import, i.e. of the code that runs (audit E2); covers every source the kernels compile from
_PROVENANCE = {os.path.basename(p): _sha(p) for p in _PROV_FILES}
_ENGINE_SHA = hashlib.sha256(json.dumps(_PROVENANCE, sort_keys=True).encode()).hexdigest()


def _engine_sha256():
    """Combined sha of engine.py, model.py, framework.py and lta_ladder_numba.py as imported (not as on disk now)."""
    return _ENGINE_SHA


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
def _hb(x, lo, hi):
    """Histogram bin of x on [lo, hi) with NFH bins, clipped to the edge bins."""
    b = np.int64(math.floor((x - lo) / (hi - lo) * NFH))
    if b < 0:
        b = 0
    if b > NFH - 1:
        b = NFH - 1
    return b


@njit(cache=True, error_model="numpy")
def _band(age, bands):
    b = 0
    while b < bands.shape[0] and age >= bands[b]:
        b += 1
    return b


@njit(cache=False, parallel=True, error_model="numpy")
def _forces_all(q, F, phi, jb, fl, box, cell, nvox, s0, x0, y0, z0, s1, x1, y1, z1, pp, rc2, kb, r0, P, xw, dphi, ng,
                cf):
    for i in prange(q.shape[0]):
        config_energy_force(q[i], box, cell, nvox, s0, x0, y0, z0, s1, x1, y1, z1, pp, rc2, kb, r0, F[i])
        xt = 0.5 * (q[i, 0, 0, 0] + q[i, 0, 1, 0])
        ph = _wrap(TWO_PI * (xt - xw) / P)
        phi[i] = ph
        jb[i] = np.int64(math.floor((ph + math.pi) / dphi)) % ng
        fl[i] = cf * (F[i, 0, 0, 0] + F[i, 0, 1, 0])


@njit(cache=False, error_model="numpy")
def _run_chunk(k0, k1, q, F, Cacc, Macc, Seta, Sux2, Sbond, Suz2, BH, DH, anc, ancw, row, lastcage, evwin, ist, fst,
               fp, ip, box, cell, nvox, s0, x0, y0, z0, s1, x1, y1, z1, pp, rc2, kb, r0, Kk,
               rng_noise, rng_fr, save_steps, isave, sv_C, sv_M, sv_h, sv_i, sv_u, sv_cond, born, dg, sv_dg,
               tag, ev_i, ev_f, evC, evM, evB, fdC, fdM, fdB, fh, dst, bands):
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
    frstop = ip[8]
    agew = ip[9]
    selon = ip[10] == 1
    khalf = ip[11]
    blo = r0 - BOND_HALF
    bhi = r0 + BOND_HALF
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
        # (1) forces / CV at the start-of-step positions: replicas are independent, so a parallel loop gives
        #     bitwise the serial result (each iteration writes only its own slot); labels then in slot order
        _forces_all(q, F, phi, jb, fl, box, cell, nvox, s0, x0, y0, z0, s1, x1, y1, z1, pp, rc2, kb, r0, P, xw, dphi,
                    ng, cf)
        for i in range(N):
            xt = 0.5 * (q[i, 0, 0, 0] + q[i, 0, 1, 0])
            _labels(i, k, xt, P, xw, phi[i], cage_phi, window_phi, row, lastcage, ist, evwin)
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
            ux2 = dx * dx / rb2
            eta = math.cos(dzw * 0.5 * (q[i, 0, 0, 2] + q[i, 0, 1, 2]))
            rb = math.sqrt(rb2)
            uz2 = dz * dz / rb2
            Sux2[j] += ux2
            Seta[j] += eta
            Sbond[j] += rb
            Suz2[j] += uz2
            if agew > 0 and k - born[i] < agew:          # diagnostic split: deposit of a recently cloned walker
                dg[0, j] += 1.0
                dg[1, j] += fl[i]
                dg[2, j] += eta
                dg[3, j] += ux2
            if selon:                                    # selection diagnostic: ancestor-tagged deposit (read-only)
                h = 1 if k > khalf else 0
                cb = (j * N_COARSE) // ng
                e = tag[i]
                if e == -1:
                    fdC[h, j] += 1.0
                    fdM[h, j] += fl[i]
                    fdB[h, cb, 0] += 1.0
                    fdB[h, cb, 1] += fl[i]
                    fdB[h, cb, 2] += eta
                    fdB[h, cb, 3] += ux2
                    fdB[h, cb, 4] += rb
                    fdB[h, cb, 5] += uz2
                elif e >= 0:
                    evC[e, h, j] += 1.0
                    evM[e, h, j] += fl[i]
                    bnd = _band(k - born[i], bands)
                    evB[e, h, bnd, cb, 0] += 1.0
                    evB[e, h, bnd, cb, 1] += fl[i]
                    evB[e, h, bnd, cb, 2] += eta
                    evB[e, h, bnd, cb, 3] += ux2
                    evB[e, h, bnd, cb, 4] += rb
                    evB[e, h, bnd, cb, 5] += uz2
                else:
                    dst[2] += 1                          # untracked (event table overflowed): counted, not binned
                fh[h, 0, j, _hb(eta, -1.0, 1.0)] += 1.0
                fh[h, 1, j, _hb(ux2, 0.0, 1.0)] += 1.0
                fh[h, 2, j, _hb(rb, blo, bhi)] += 1.0
                fh[h, 3, j, _hb(uz2, 0.0, 1.0)] += 1.0
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
            if agew > 0:
                for j in range(ng):
                    sv_dg[isave, 0, j] = dg[0, j]
                    sv_dg[isave, 1, j] = dg[1, j]
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
        # (5) Fisher-Rao at the POST-move positions; after fr_stop no FR event and no FR-stream draw (ABF unchanged)
        if is_fr and nxt >= frstart and (frstop < 0 or nxt < frstop) and (nxt - frstart) % frevery == 0:
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
                if selon:                                # birth / death record (reads only; before the copy)
                    e = dst[0]
                    if e < ev_i.shape[0]:
                        ev_i[e, 0] = nxt
                        ev_i[e, 1] = s_
                        ev_i[e, 2] = d_
                        ev_i[e, 3] = jn[s_]
                        ev_i[e, 4] = jn[d_]
                        ev_i[e, 5] = tag[s_]
                        ev_i[e, 6] = tag[d_]
                        for w_ in range(2):
                            ii = s_ if w_ == 0 else d_
                            dx = q[ii, 0, 0, 0] - q[ii, 0, 1, 0]
                            dy = q[ii, 0, 0, 1] - q[ii, 0, 1, 1]
                            dz = q[ii, 0, 0, 2] - q[ii, 0, 1, 2]
                            rb2 = dx * dx + dy * dy + dz * dz
                            ev_f[e, 4 * w_ + 0] = math.cos(dzw * 0.5 * (q[ii, 0, 0, 2] + q[ii, 0, 1, 2]))
                            ev_f[e, 4 * w_ + 1] = dx * dx / rb2
                            ev_f[e, 4 * w_ + 2] = math.sqrt(rb2)
                            ev_f[e, 4 * w_ + 3] = dz * dz / rb2
                        tag[d_] = e
                        dst[0] = e + 1
                    else:
                        dst[1] += 1
                        tag[d_] = -2
                for m in range(M):
                    for bb in range(2):
                        for d in range(3):
                            q[d_, m, bb, d] = q[s_, m, bb, d]
                anc[d_] = anc[s_]
                ancw[d_] = ancw[s_]
                born[d_] = nxt
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
    import zlib
    vid = VARIANT_ID.get(c["variant"], zlib.crc32(c["variant"].encode()) % 1000003)
    ss = np.random.SeedSequence([int(c["seed"]), int(c["N"]), int(c["M"]), int(round(c["T_K"])), vid])
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


@njit(cache=False, error_model="numpy")
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
        self.model = model if model is not None else mdl.Model(c["variant"], T_K=c["T_K"], params=c["ff"])
        ffp = dict(mdl.PARAMS if c["ff"] is None else c["ff"], T_K=c["T_K"])
        if model is not None and (model.fw.variant != c["variant"] or model.p != ffp):
            raise ValueError("model does not match the config (variant / T_K / force field)")
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
        self.Sbond = np.zeros(ng)
        self.Suz2 = np.zeros(ng)
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
                            int(c["genealogy_window_steps"]), int(c["fr_stop_steps"]), int(c["diag_age_steps"]),
                            1 if int(c["diag_sel"]) == 1 else 0,
                            int(c["diag_sel_split_step"]) if int(c["diag_sel_split_step"]) >= 0 else int(c["n_steps"]) // 2],
                           dtype=np.int64)
        self.born = np.full(N, -(2 ** 40), dtype=np.int64)
        self.dg = np.zeros((4, ng))
        self.sv_dg = np.zeros((ns_ if int(c["diag_age_steps"]) > 0 else 1, 2, ng))
        # selection diagnostic (read-only); dummies of the same ndim when off
        on = int(c["diag_sel"]) == 1
        if c["diag_sel_band_steps"] is None:
            self.bands = np.array([int(round(b / c["dt"])) for b in SEL_BANDS_TU], dtype=np.int64)
        else:
            self.bands = np.array(sorted(int(b) for b in c["diag_sel_band_steps"]), dtype=np.int64)
        nb = len(self.bands) + 1
        self.tag = np.full(N if on else 1, -1, dtype=np.int64)
        e0 = 2048 if on else 1
        self.ev_i = np.zeros((e0, NEVI), dtype=np.int64)
        self.ev_f = np.zeros((e0, NEVF))
        self.evC = np.zeros((e0, 2, ng if on else 1))
        self.evM = np.zeros((e0, 2, ng if on else 1))
        self.evB = np.zeros((e0, 2, nb, N_COARSE, NSB) if on else (1, 2, 1, 1, NSB))
        self.fdC = np.zeros((2, ng if on else 1))
        self.fdM = np.zeros((2, ng if on else 1))
        self.fdB = np.zeros((2, N_COARSE, NSB) if on else (2, 1, NSB))
        self.fh = np.zeros((2, 4, ng, NFH) if on else (2, 4, 1, NFH))
        self.dst = np.zeros(4, dtype=np.int64)         # n_events, overflowed events, untracked deposits, spare
        self.sessions = []
        self.ctl = None
        if c.get("fr_controller") and method == "fr":
            from .fr_controller import OnlineStop, check_grid
            if int(c["fr_stop_steps"]) >= 0:
                raise ValueError("fr_controller and a fixed fr_stop_steps are exclusive")
            if int(c["fr_start_steps"]) < 1:
                raise ValueError("the online controller needs fr_start_steps >= 1")
            self.ctl = OnlineStop(c["fr_controller"], N, c["fr_start_steps"] * c["dt"])
            self.ctl_every = check_grid(c["fr_controller"], c["dt"])

    def advance(self, k_target, _internal=False):
        if self.ctl is not None and not _internal:
            raise RuntimeError("a run with an online FR controller must be advanced with advance_controlled()")
        m = self.model
        k_target = min(int(k_target), int(self.c["n_steps"]) + 1)
        if k_target <= self.k:
            return
        if self.method == "fr" and self.ip[10] == 1:
            # the event table can never overflow: room for every event the chunk could produce (cap per opportunity)
            self._grow_events(((k_target - self.k) // max(int(self.c["fr_every"]), 1) + 2) * int(self.c["cap"]))
        self.isave = _run_chunk(self.k, k_target, self.q, self.F, self.C, self.Msum, self.Seta, self.Sux2, self.Sbond,
                                self.Suz2, self.BH, self.DH, self.anc, self.ancw, self.row, self.lastcage, self.evwin,
                                self.ist, self.fst, self.fp, self.ip, m.fw.box, *m.fw.packed(), m.pp, m.rc2, m.kb,
                                m.r0, self.Kk, self.g_noise, self.g_fr, self.save_steps, self.isave, self.sv_C,
                                self.sv_M, self.sv_h, self.sv_i, self.sv_u, self.sv_cond, self.born, self.dg, self.sv_dg,
                                self.tag, self.ev_i, self.ev_f, self.evC, self.evM, self.evB, self.fdC, self.fdM,
                                self.fdB, self.fh, self.dst, self.bands)
        self.k = k_target

    def _grow_events(self, need):
        """Ensure at least ``need`` free rows in the per-event diagnostic arrays (host side, between chunks)."""
        nev = int(self.dst[0])
        cap = self.ev_i.shape[0]
        if cap - nev >= need:
            return
        new = max(2 * cap, nev + need)
        for k in ("ev_i", "ev_f", "evC", "evM", "evB"):
            a = getattr(self, k)
            b = np.zeros((new,) + a.shape[1:], dtype=a.dtype)
            b[:cap] = a
            setattr(self, k, b)

    def _check(self):
        """Online controller check at the current step (state at the START of step self.k)."""
        lab = np.bincount(self.ancw, minlength=self.ancw.size).astype(float)
        essw = float(lab.sum() ** 2 / (lab ** 2).sum() / lab.size)
        maxlin = float(lab.max() / lab.size)
        ts = self.ctl.observe(self.k * self.c["dt"], self.BH, self.DH, self.C, self.Msum, essw, maxlin)
        if ts is not None:
            # no FR opportunity at nxt >= frstop; a stop is only ever brought earlier (safety), never postponed
            new = max(self.k + 1, int(round(ts / self.c["dt"])))
            if self.ip[8] < 0 or new < self.ip[8]:
                self.ip[8] = new

    def advance_controlled(self, k_target):
        """advance() with the online controller checked at every step k = fr_start + n * ctl_every (n >= 0),
        independent of how the caller chunks the run."""
        if self.ctl is None:
            return self.advance(k_target, _internal=True)
        k_target = min(int(k_target), int(self.c["n_steps"]) + 1)
        k0 = int(self.c["fr_start_steps"])
        while self.k < k_target:
            n = max(0, -(-(self.k + 1 - k0) // self.ctl_every))
            kc = k0 + n * self.ctl_every
            nxt = min(k_target, kc)
            self.advance(nxt, _internal=True)
            if self.k == kc and self.k <= int(self.c["n_steps"]) and (self.ip[8] < 0 or self.k < self.ip[8]):
                self._check()

    @property
    def done(self):
        return self.k > int(self.c["n_steps"])

    _STATE = ("q", "C", "Msum", "Seta", "Sux2", "Sbond", "Suz2", "BH", "DH", "anc", "ancw", "row", "lastcage", "evwin",
              "ist", "fst", "born", "dg", "sv_dg",
              "tag", "ev_i", "ev_f", "evC", "evM", "evB", "fdC", "fdM", "fdB", "fh", "dst",
              "sv_C", "sv_M", "sv_h", "sv_i", "sv_u", "sv_cond")

    def checkpoint(self, path):
        tmp = path + ".tmp.npz"
        extra = {}
        if self.ctl is not None:
            extra = dict(ip=self.ip, ctl=json.dumps(self.ctl.state()))
        np.savez(tmp, **{k: getattr(self, k) for k in self._STATE}, k=self.k, isave=self.isave, **extra,
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
        if self.ctl is not None:
            from .fr_controller import OnlineStop
            self.ip = np.array(z["ip"])
            self.ctl = OnlineStop.from_state(json.loads(str(z["ctl"])))

    def result(self):
        c = self.c
        return dict(cfg=json.dumps(c, default=str), method=self.method, config_hash=config_hash(c, self.method),
                    engine=ENGINE, engine_sha256=_engine_sha256(), provenance=json.dumps(_PROVENANCE),
                    save_steps=self.save_steps,
                    times=self.save_steps * c["dt"], sv_C=self.sv_C, sv_M=self.sv_M, sv_h=self.sv_h, sv_i=self.sv_i,
                    sv_u=self.sv_u, sv_cond=self.sv_cond, Seta=self.Seta, Sux2=self.Sux2, Sbond=self.Sbond,
                    Suz2=self.Suz2, BH=self.BH, DH=self.DH,
                    **({} if int(c["diag_age_steps"]) <= 0 else dict(dg_young=self.dg, sv_dg_young=self.sv_dg)),
                    **({} if int(c["diag_sel"]) != 1 else self.selection_result()),
                    ist=self.ist, fst=self.fst, q_final=self.q, n_force_evals=int(self.ist[S_NFE]),
                    n_equil_force_evals=int(c["N"]) * int(c["equil_steps"]),
                    P=self.model.fw.P, x_window=self.model.fw.x_window, a=self.model.fw.a, c_len=self.model.fw.c,
                    **({} if self.ctl is None else dict(
                        fr_stop_effective_steps=int(self.ip[8]), fr_controller_t1=-1.0 if self.ctl.t1 is None else self.ctl.t1,
                        fr_controller_t_stop=-1.0 if self.ctl.t_stop is None else self.ctl.t_stop,
                        fr_controller_reason=str(self.ctl.reason), fr_controller_log=json.dumps(self.ctl.log))))


    def selection_result(self):
        """The selection diagnostic's arrays, event rows trimmed to the recorded events (key prefix ``sel_``)."""
        n = int(self.dst[0])
        r0 = self.model.r0
        return dict(sel_n_events=n, sel_overflow_events=int(self.dst[1]), sel_untracked_deposits=int(self.dst[2]),
                    sel_split_step=int(self.ip[11]), sel_band_steps=self.bands.copy(),
                    sel_ev_i=self.ev_i[:n].copy(), sel_ev_f=self.ev_f[:n].copy(), sel_evC=self.evC[:n].copy(),
                    sel_evM=self.evM[:n].copy(), sel_evB=self.evB[:n].copy(), sel_fdC=self.fdC, sel_fdM=self.fdM,
                    sel_fdB=self.fdB, sel_fh=self.fh, sel_tag=self.tag, sel_born=self.born,
                    sel_fh_ranges=np.array([[-1.0, 1.0], [0.0, 1.0], [r0 - BOND_HALF, r0 + BOND_HALF], [0.0, 1.0]]),
                    sel_ev_i_cols="step,parent,killed,parent_bin,killed_bin,parent_tag,killed_tag",
                    sel_ev_f_cols="p_eta,p_ux2,p_bond,p_uz2,k_eta,k_ux2,k_bond,k_uz2",
                    sel_B_cols="count,f,eta,ux2,bond,uz2")


def run(c, method, out_path=None, checkpoint_path=None, checkpoint_every_s=600.0, chunk_steps=2000, model=None,
        verbose=False):
    """Run to completion with periodic checkpoints; resume from ``checkpoint_path`` if it exists."""
    r = Run(c, method, model=model)
    if checkpoint_path and os.path.exists(checkpoint_path):
        r.restore(checkpoint_path)
    t0 = time.time()
    tlast = t0
    while not r.done:
        r.advance_controlled(r.k + chunk_steps)
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
