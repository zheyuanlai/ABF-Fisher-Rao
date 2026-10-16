"""Independent equilibrium reference for the ERI campaign: umbrella sampling along phi with exact Metropolis Monte
Carlo (the Gibbs arbiter) and, for the Gate 2 timestep ladder, with the overdamped Langevin schemes
(Euler-Maruyama, Leimkuhler-Matthews) under the SAME umbrella.  Never uses any ABF or FR output.

One molecule (infinite dilution).  Window w samples exp(-beta [U(q) + (kappa/2) wrap(phi(q) - c_w)^2]).

Moves (MC; all symmetric proposals, so the Metropolis rule is exact):
    single bead   uniform cube +-d_bead around the bead (random bead);
    translation   uniform cube +-d_trans for the rigid molecule;
    rotation      rigid rotation about the COM by an angle uniform in [-d_rot, d_rot] about a uniform random axis.
Langevin (unit mobility, kT = k_B T):
    EM  q' = q + dt (F + F_umb) + sqrt(2 dt kT) xi_n
    LM  q' = q + dt (F + F_umb) + sqrt(2 dt kT) (xi_n + xi_{n+1}) / 2         (second order for the invariant law)

Read-outs per window (accumulated over the window's molecules, every ``sample_every`` steps after burn-in):
    C_fine          phi histogram on NFINE bins (WHAM input)
    per NB coarse bin: counts, sums of the local mean force f and f^2, the potential U and U^2, the bond length,
                    the orientation u_x^2 and u_z^2 (u = unit bond vector), eta = cos(4 pi (Z_com - z_w) / c)
                    (+1 at the window heights, -1 at the cage middles)
    Hs [NB, NS]     joint histogram of (phi bin, |s|) for BOTH beads, s = the bead's xy offset from the NEAREST
                    8-ring window stack projected on that window's centre-to-centre line (the literature single-bead
                    coordinate: window at s = 0, cage axes at |s| = a / (2 sqrt 3))
    Hx [NB, NXB]    joint histogram of (phi bin, bead X phase) for both beads (single-bead X projection)
    ts              time series (phi, eta, u_x^2) of the FIRST molecule at every sample (autocorrelation)
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit

from .model import bead_fw, config_energy_force, cv_phi

EM, LM, MC = 0, 1, 2
SCHEMES = {"EM": EM, "LM": LM, "MC": MC}
NFINE = 1800
NB = 180
NS = 64
NXB = 90
S_MAX_FACTOR = 1.05            # |s| histogram range [0, 1.05 * a / (2 sqrt 3)]
TWO_PI = 2.0 * math.pi
ACC_KEYS = ("C_fine", "C", "Mf", "Mf2", "MU", "MU2", "Mbond", "Mbond2", "Mux2", "Muz2", "Meta", "Hs", "Hx")


def window_stacks(fw):
    """xy positions (in the orthorhombic unit cell, with the 3 x 3 neighbouring cell images) and horizontal unit
    normals of the 8-ring window stacks.  Returns (W (K, 2), N (K, 2), cell (2,))."""
    cell = fw.box[:2] / np.array([3.0, 2.0])
    win = fw.windows
    cages = fw.cages
    stacks, normals = [], []
    for w in win:
        d = cages - w
        d -= fw.box * np.round(d / fw.box)
        r = np.sqrt((d * d).sum(1))
        i0, i1 = np.argsort(r)[:2]
        n = (d[i0] - d[i1])[:2]
        n = n / np.linalg.norm(n)
        if n[0] < -1e-12 or (abs(n[0]) < 1e-12 and n[1] < 0):
            n = -n
        p = w[:2] % cell
        if not any(np.max(np.abs(((p - s) + cell / 2) % cell - cell / 2)) < 1e-6 for s in stacks):
            stacks.append(p)
            normals.append(n)
    stacks, normals = np.array(stacks), np.array(normals)
    assert len(stacks) == 6, len(stacks)
    W, N = [], []
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            W.append(stacks + cell * np.array([i, j]))
            N.append(normals)
    return np.ascontiguousarray(np.concatenate(W)), np.ascontiguousarray(np.concatenate(N)), np.ascontiguousarray(cell)


@njit(cache=True, error_model="numpy")
def _abs_s(x, y, WS, WN, cell2):
    xr = x - cell2[0] * math.floor(x / cell2[0])
    yr = y - cell2[1] * math.floor(y / cell2[1])
    best = 1e300
    sb = 0.0
    for k in range(WS.shape[0]):
        dx = xr - WS[k, 0]
        dy = yr - WS[k, 1]
        r2 = dx * dx + dy * dy
        if r2 < best:
            best = r2
            sb = abs(dx * WN[k, 0] + dy * WN[k, 1])
    return sb


@njit(cache=True, error_model="numpy")
def _umb(ph, centre, kappa):
    d = ph - centre
    d = d - TWO_PI * math.floor((d + math.pi) / TWO_PI)
    return 0.5 * kappa * d * d, -kappa * d          # W, -dW/dphi


@njit(cache=True, error_model="numpy")
def _mol_energy(q, box, cell, nvox, s0, x0, y0, z0, s1, x1, y1, z1, pp, rc2, kb, r0, eb):
    """Per-bead framework energies into eb (2,), returns (total U, bond U)."""
    for b in range(2):
        e, fx, fy, fz = bead_fw(q[b, 0], q[b, 1], q[b, 2], cell, nvox, s0, x0, y0, z0, s1, x1, y1, z1, pp, rc2)
        eb[b] = e
    dx = q[0, 0] - q[1, 0]
    dy = q[0, 1] - q[1, 1]
    dz = q[0, 2] - q[1, 2]
    r = math.sqrt(dx * dx + dy * dy + dz * dz)
    ub = 0.5 * kb * (r - r0) * (r - r0)
    return eb[0] + eb[1] + ub, ub


@njit(cache=True, error_model="numpy")
def _bond_u(q, kb, r0):
    dx = q[0, 0] - q[1, 0]
    dy = q[0, 1] - q[1, 1]
    dz = q[0, 2] - q[1, 2]
    r = math.sqrt(dx * dx + dy * dy + dz * dz)
    return 0.5 * kb * (r - r0) * (r - r0)


@njit(cache=True, error_model="numpy")
def run_window(scheme, q0s, box, cell, nvox, s0, x0, y0, z0, s1, x1, y1, z1, pp, rc2, kb, r0, kT, P, xw, c_len,
               centre, kappa, dt, n_steps, burn_in, n_pre_mc, sample_every, d_bead, d_trans, d_rot, rng,
               WS, WN, cell2, s_max, C_fine, C, Mf, Mf2, MU, MU2, Mbond, Mbond2, Mux2, Muz2, Meta, Hs, Hx, ts, cnt,
               C_fine_h1, C_h1, Mf_h1):
    """All molecules q0s (n_mol, 2, 3) of one window, one after the other.  cnt (10,): [bead acc, bead prop,
    trans acc, trans prop, rot acc, rot prop, row crossings, max |step component|, n samples, n ts].
    C_fine_h1, C_h1, Mf_h1: the same accumulators restricted to the FIRST HALF of each molecule's production steps
    (st < n_steps // 2), for the first-half / second-half equilibration check."""
    q = np.zeros((1, 2, 3))
    qn = np.zeros((1, 2, 3))
    F = np.zeros((1, 2, 3))
    R = np.zeros((2, 3))
    eb = np.zeros(2)
    ebn = np.zeros(2)
    ns = math.sqrt(2.0 * dt * kT)
    gx = math.pi / P
    dzw = 2.0 * TWO_PI / c_len
    n_ts = 0
    for m in range(q0s.shape[0]):
        for b in range(2):
            for k in range(3):
                q[0, b, k] = q0s[m, b, k]
        U, ub = _mol_energy(q[0], box, cell, nvox, s0, x0, y0, z0, s1, x1, y1, z1, pp, rc2, kb, r0, eb)
        ph = cv_phi(q, P, xw)
        W, dW = _umb(ph, centre, kappa)
        for b in range(2):
            for k in range(3):
                R[b, k] = rng.standard_normal()
        if scheme != MC:
            config_energy_force(q, box, cell, nvox, s0, x0, y0, z0, s1, x1, y1, z1, pp, rc2, kb, r0, F)
        row = math.floor((0.5 * (q[0, 0, 0] + q[0, 1, 0]) - xw) / P)
        total = n_pre_mc + burn_in + n_steps
        for step in range(total):
            sch = MC if step < n_pre_mc else scheme
            if step == n_pre_mc and scheme != MC:
                U = config_energy_force(q, box, cell, nvox, s0, x0, y0, z0, s1, x1, y1, z1, pp, rc2, kb, r0, F)[0]
                ph = cv_phi(q, P, xw)
                W, dW = _umb(ph, centre, kappa)
            st = step - n_pre_mc - burn_in
            if st >= 0 and st % sample_every == 0:
                if sch == MC:
                    Ut = config_energy_force(q, box, cell, nvox, s0, x0, y0, z0, s1, x1, y1, z1, pp, rc2, kb, r0,
                                             F)[0]
                else:
                    Ut = U
                jf = int(math.floor((ph + math.pi) / TWO_PI * C_fine.shape[0]))
                if jf < 0:
                    jf = 0
                if jf > C_fine.shape[0] - 1:
                    jf = C_fine.shape[0] - 1
                C_fine[jf] += 1.0
                if st < n_steps // 2:
                    C_fine_h1[jf] += 1.0
                j = int(math.floor((ph + math.pi) / TWO_PI * C.shape[0]))
                if j < 0:
                    j = 0
                if j > C.shape[0] - 1:
                    j = C.shape[0] - 1
                f = -(P / TWO_PI) * (F[0, 0, 0] + F[0, 1, 0])
                dx = q[0, 0, 0] - q[0, 1, 0]
                dy = q[0, 0, 1] - q[0, 1, 1]
                dz = q[0, 0, 2] - q[0, 1, 2]
                rb = math.sqrt(dx * dx + dy * dy + dz * dz)
                ux2 = dx * dx / (rb * rb)
                uz2 = dz * dz / (rb * rb)
                zc = 0.5 * (q[0, 0, 2] + q[0, 1, 2])
                eta = math.cos(dzw * zc)
                C[j] += 1.0
                Mf[j] += f
                if st < n_steps // 2:
                    C_h1[j] += 1.0
                    Mf_h1[j] += f
                Mf2[j] += f * f
                MU[j] += Ut
                MU2[j] += Ut * Ut
                Mbond[j] += rb
                Mbond2[j] += rb * rb
                Mux2[j] += ux2
                Muz2[j] += uz2
                Meta[j] += eta
                for b in range(2):
                    sv = _abs_s(q[0, b, 0], q[0, b, 1], WS, WN, cell2)
                    js = int(sv / s_max * Hs.shape[1])
                    if js < Hs.shape[1]:
                        Hs[j, js] += 1.0
                    xp = (q[0, b, 0] - xw) / P
                    xp = xp - math.floor(xp)
                    jx = int(xp * Hx.shape[1])
                    if jx > Hx.shape[1] - 1:
                        jx = Hx.shape[1] - 1
                    Hx[j, jx] += 1.0
                cnt[8] += 1.0
                if m == 0 and n_ts < ts.shape[0]:
                    ts[n_ts, 0] = ph
                    ts[n_ts, 1] = eta
                    ts[n_ts, 2] = ux2
                    n_ts += 1
            if sch == EM or sch == LM:
                for b in range(2):
                    for k in range(3):
                        dr = F[0, b, k] * dt
                        if k == 0:
                            dr += dW * gx * dt
                        if sch == EM:
                            dr += ns * rng.standard_normal()
                        else:
                            rn = rng.standard_normal()
                            dr += ns * 0.5 * (R[b, k] + rn)
                            R[b, k] = rn
                        if abs(dr) > cnt[7]:
                            cnt[7] = abs(dr)
                        q[0, b, k] += dr
                U = config_energy_force(q, box, cell, nvox, s0, x0, y0, z0, s1, x1, y1, z1, pp, rc2, kb, r0, F)[0]
                ph = cv_phi(q, P, xw)
                W, dW = _umb(ph, centre, kappa)
            else:
                for move in range(3):
                    for b in range(2):
                        for k in range(3):
                            qn[0, b, k] = q[0, b, k]
                    if move == 0:
                        bb = 0 if rng.random() < 0.5 else 1
                        for k in range(3):
                            qn[0, bb, k] = q[0, bb, k] + d_bead * (2.0 * rng.random() - 1.0)
                        e, fx, fy, fz = bead_fw(qn[0, bb, 0], qn[0, bb, 1], qn[0, bb, 2], cell, nvox, s0, x0, y0,
                                                z0, s1, x1, y1, z1, pp, rc2)
                        ebn[0] = eb[0]
                        ebn[1] = eb[1]
                        ebn[bb] = e
                        ubn = _bond_u(qn[0], kb, r0)
                        Un = ebn[0] + ebn[1] + ubn
                    else:
                        if move == 1:
                            for k in range(3):
                                t = d_trans * (2.0 * rng.random() - 1.0)
                                qn[0, 0, k] = q[0, 0, k] + t
                                qn[0, 1, k] = q[0, 1, k] + t
                        else:
                            # uniform random axis, angle uniform in [-d_rot, d_rot] (Rodrigues)
                            ax = rng.standard_normal()
                            ay = rng.standard_normal()
                            az = rng.standard_normal()
                            an = math.sqrt(ax * ax + ay * ay + az * az)
                            ax /= an
                            ay /= an
                            az /= an
                            th = d_rot * (2.0 * rng.random() - 1.0)
                            ct = math.cos(th)
                            stt = math.sin(th)
                            cx = 0.5 * (q[0, 0, 0] + q[0, 1, 0])
                            cy = 0.5 * (q[0, 0, 1] + q[0, 1, 1])
                            cz = 0.5 * (q[0, 0, 2] + q[0, 1, 2])
                            for b in range(2):
                                vx = q[0, b, 0] - cx
                                vy = q[0, b, 1] - cy
                                vz = q[0, b, 2] - cz
                                dot = ax * vx + ay * vy + az * vz
                                crx = ay * vz - az * vy
                                cry = az * vx - ax * vz
                                crz = ax * vy - ay * vx
                                qn[0, b, 0] = cx + vx * ct + crx * stt + ax * dot * (1.0 - ct)
                                qn[0, b, 1] = cy + vy * ct + cry * stt + ay * dot * (1.0 - ct)
                                qn[0, b, 2] = cz + vz * ct + crz * stt + az * dot * (1.0 - ct)
                        Un, ubn = _mol_energy(qn[0], box, cell, nvox, s0, x0, y0, z0, s1, x1, y1, z1, pp, rc2, kb,
                                              r0, ebn)
                    phn = cv_phi(qn, P, xw)
                    Wn, dWn = _umb(phn, centre, kappa)
                    la = -((Un + Wn) - (U + W)) / kT
                    cnt[2 * move + 1] += 1.0
                    if la >= 0.0 or rng.random() < math.exp(la):
                        cnt[2 * move] += 1.0
                        for b in range(2):
                            for k in range(3):
                                q[0, b, k] = qn[0, b, k]
                        eb[0] = ebn[0]
                        eb[1] = ebn[1]
                        U = Un
                        ph = phn
                        W = Wn
                        dW = dWn
            if st >= 0:
                row2 = math.floor((0.5 * (q[0, 0, 0] + q[0, 1, 0]) - xw) / P)
                if row2 != row:
                    cnt[6] += 1.0
                row = row2
            else:
                row = math.floor((0.5 * (q[0, 0, 0] + q[0, 1, 0]) - xw) / P)
    cnt[9] = n_ts


def window_starts(fw, n, seed, centre, r0=1.54):
    """n molecules with COM on the horizontal line from a random between-row window (phi = 0 family) toward the
    adjacent cage axis, at phi = centre, random orientation (exact MC relaxation follows)."""
    g = np.random.default_rng(int(seed))
    P, xw = fw.P, fw.x_window
    wins = []
    for w in fw.windows:
        ph = TWO_PI * (w[0] - xw) / P
        ph = (ph + math.pi) % TWO_PI - math.pi
        if abs(ph) < 1e-6:
            wins.append(w)
    wins = np.array(wins)
    out = np.zeros((n, 2, 3))
    for i in range(n):
        w = wins[g.integers(len(wins))]
        d = fw.cages - w
        d -= fw.box * np.round(d / fw.box)
        r = np.sqrt((d * d).sum(1))
        i0, i1 = np.argsort(r)[:2]
        nv = np.array([*(d[i0] - d[i1])[:2], 0.0])
        nv /= np.linalg.norm(nv)
        if nv[0] < 0:
            nv = -nv
        s = (centre / TWO_PI) * P / nv[0]
        com = w + s * nv
        u = g.standard_normal(3)
        u /= np.linalg.norm(u)
        out[i] = [com + 0.5 * r0 * u, com - 0.5 * r0 * u]
    return out


def run(model, scheme, centre, kappa, n_mol, n_steps, burn_in, seed, dt=1e-4, n_pre_mc=5000, sample_every=5,
        d_bead=0.06, d_trans=0.3, d_rot=0.3, n_ts=4000):
    """One (window, group) job.  Returns a dict of accumulators + counters."""
    fw = model.fw
    WS, WN, cell2 = window_stacks(fw)
    s_max = S_MAX_FACTOR * fw.a / (2.0 * math.sqrt(3.0))
    q0 = window_starts(fw, n_mol, seed, centre, model.r0)
    rng = np.random.Generator(np.random.PCG64(int(seed) + 7919))
    acc = {"C_fine": np.zeros(NFINE)}
    for k in ACC_KEYS[1:11]:
        acc[k] = np.zeros(NB)
    acc["Hs"] = np.zeros((NB, NS))
    acc["Hx"] = np.zeros((NB, NXB))
    acc["C_fine_h1"] = np.zeros(NFINE)
    acc["C_h1"] = np.zeros(NB)
    acc["Mf_h1"] = np.zeros(NB)
    ts = np.zeros((n_ts, 3))
    cnt = np.zeros(10)
    run_window(SCHEMES[scheme], q0, model.fw.box, *model.fw.packed(), model.pp, model.rc2, model.kb, model.r0,
               model.kT, fw.P, fw.x_window, fw.c, float(centre), float(kappa), float(dt), int(n_steps), int(burn_in),
               int(n_pre_mc), int(sample_every), float(d_bead), float(d_trans), float(d_rot), rng, WS, WN, cell2,
               float(s_max), acc["C_fine"], acc["C"], acc["Mf"], acc["Mf2"], acc["MU"], acc["MU2"], acc["Mbond"],
               acc["Mbond2"], acc["Mux2"], acc["Muz2"], acc["Meta"], acc["Hs"], acc["Hx"], ts, cnt, acc["C_fine_h1"],
               acc["C_h1"], acc["Mf_h1"])
    acc["ts"] = ts[: int(cnt[9])]
    acc["cnt"] = cnt
    acc["s_max"] = s_max
    return acc
