"""Counter-based Langevin noise for the Experiment B walker-parallel backends: Philox4x64-10 + Box-Muller (numba).

Final campaign, Experiment B (docs/final_campaign/PREREGISTRATION.md section 2.2, EXPB_BACKEND_DESIGN.md section 3).

Generator
---------
Philox4x64-10 with the Random123 constants (Salmon et al. 2011; identical to numpy's ``np.random.Philox``):
  M0 = 0xD2E7470EE14C6C93, M1 = 0xCA5A826395121157 (round multipliers), W0 = 0x9E3779B97F4A7C15,
  W1 = 0xBB67AE8584CAA73B (Weyl key bumps); round(ctr, key) = (hi(M1 c2) ^ c1 ^ k0, lo(M1 c2), hi(M0 c0) ^ c3 ^ k1,
  lo(M0 c0)); 10 rounds with the key bumped between rounds (9 bumps).  The 64 x 64 -> 128 multiply is done with
  32-bit limbs held in uint64 (no 128-bit type in numba): every constant and shift count below is a np.uint64 so
  numba never promotes a uint64 / int64 mix to float64.

Key and counter (frozen)
------------------------
  key     = SeedSequence(entropy).generate_state(2, np.uint64): entropy [seed, N, 2] (gateway) or
            [seed, N, round(T_K), 2] = [seed, N, 150, 2] (LTA 150 K)
  counter = (c0, c1, c2, c3) = (global step index of the move, walker slot, word, 0)
  outputs o0..o3 of one counter -> uniforms u_k = (o_k >> 11) 2^-53 in [0, 1 - 2^-53]
  Box-Muller on (u0, u1) and (u2, u3): r = sqrt(-2 log(1 - u_a)), theta = 2 pi u_b, z = (r cos theta, r sin theta)
  gateway: word 0 -> (zx, zy) = Box-Muller(u0, u1)
  LTA:     words 0 and 1 -> 8 normals (z0..z3 from word 0, z4..z7 from word 1), the first 6 used in [bead][xyz]
           order: (bead 0: x y z, bead 1: x y z) = (z0, z1, z2, z3, z4, z5); z6, z7 are not computed.
A cloned walker continues with its slot's stream (the counter names the slot, not the lineage).

The map (step, slot, word) -> (step, slot, word, 0) is the coordinate-wise embedding of three integers < 2^64 into
the 256-bit counter, hence injective whenever every coordinate is < 2^64 (``check_counter_injective``); Philox is a
bijection of the counter space for a fixed key, so distinct counters give distinct output blocks.

numpy's ``Philox(key=k, counter=c)`` increments the counter BEFORE its first block, so its first four
``random_raw`` outputs are Philox(c + 1); ``numpy_raw`` passes c - 1 (256-bit borrow) to obtain Philox(c).  This
convention is verified in tests/test_parallel_lta.py (V-B3 i) against an independent pure-Python implementation.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

M0 = np.uint64(0xD2E7470EE14C6C93)
M1 = np.uint64(0xCA5A826395121157)
W0 = np.uint64(0x9E3779B97F4A7C15)
W1 = np.uint64(0xBB67AE8584CAA73B)
_MASK32 = np.uint64(0xFFFFFFFF)
_S32 = np.uint64(32)
_S11 = np.uint64(11)
_ZERO = np.uint64(0)
_ONE = np.uint64(1)
INV2_53 = 1.0 / 9007199254740992.0          # 2^-53
TWO_PI = 2.0 * math.pi
N_ROUNDS = 10
LTA_WORDS = 2                               # counter words per LTA walker-step
GATEWAY_WORDS = 1                           # counter words per gateway walker-step


# =====================================================================================
# numba primitives
# =====================================================================================
@njit(cache=True, inline="always")
def mulhilo64(a, b):
    """(hi, lo) of the 128-bit product of two uint64 (32-bit limbs in uint64 arithmetic)."""
    lo = a * b
    a0 = a & _MASK32
    a1 = a >> _S32
    b0 = b & _MASK32
    b1 = b >> _S32
    p00 = a0 * b0
    p01 = a0 * b1
    p10 = a1 * b0
    p11 = a1 * b1
    mid = (p00 >> _S32) + (p01 & _MASK32) + (p10 & _MASK32)
    hi = p11 + (p01 >> _S32) + (p10 >> _S32) + (mid >> _S32)
    return hi, lo


@njit(cache=True, inline="always")
def _round(c0, c1, c2, c3, k0, k1):
    hi0, lo0 = mulhilo64(M0, c0)
    hi1, lo1 = mulhilo64(M1, c2)
    return (hi1 ^ c1) ^ k0, lo1, (hi0 ^ c3) ^ k1, lo0


@njit(cache=True)
def philox4x64_10(c0, c1, c2, c3, k0, k1):
    """Philox4x64-10 of the counter (c0, c1, c2, c3) under the key (k0, k1); all uint64; returns 4 uint64."""
    c0, c1, c2, c3 = _round(c0, c1, c2, c3, k0, k1)
    for _ in range(N_ROUNDS - 1):
        k0 = k0 + W0
        k1 = k1 + W1
        c0, c1, c2, c3 = _round(c0, c1, c2, c3, k0, k1)
    return c0, c1, c2, c3


@njit(cache=True, inline="always")
def u01(o):
    """Uniform in [0, 1 - 2^-53]: the top 53 bits of a uint64 output times 2^-53 (exact)."""
    return np.float64(o >> _S11) * INV2_53


@njit(cache=True, inline="always")
def box_muller(ua, ub):
    """(r cos theta, r sin theta) with r = sqrt(-2 log(1 - ua)), theta = 2 pi ub (1 - ua in (0, 1])."""
    r = math.sqrt(-2.0 * math.log(1.0 - ua))
    th = TWO_PI * ub
    return r * math.cos(th), r * math.sin(th)


@njit(cache=True)
def gateway_normals2(k0, k1, step, slot):
    """Gateway counter-mode noise of one walker-step: (zx, zy) from word 0, (u0, u1)."""
    o0, o1, o2, o3 = philox4x64_10(np.uint64(step), np.uint64(slot), _ZERO, _ZERO, k0, k1)
    return box_muller(u01(o0), u01(o1))


@njit(cache=True)
def lta_normals6(k0, k1, step, slot):
    """LTA counter-mode noise of one walker-step: the 6 normals (bead 0 x, y, z, bead 1 x, y, z) from words 0
    (z0..z3) and 1 (z4, z5)."""
    st = np.uint64(step)
    sl = np.uint64(slot)
    a0, a1, a2, a3 = philox4x64_10(st, sl, _ZERO, _ZERO, k0, k1)
    b0, b1, b2, b3 = philox4x64_10(st, sl, _ONE, _ZERO, k0, k1)
    z0, z1 = box_muller(u01(a0), u01(a1))
    z2, z3 = box_muller(u01(a2), u01(a3))
    z4, z5 = box_muller(u01(b0), u01(b1))
    return z0, z1, z2, z3, z4, z5


# =====================================================================================
# array helpers (tests, RNG battery, replay-injection oracle)
# =====================================================================================
@njit(cache=True)
def raw_block(k0, k1, ctr):
    """Philox outputs (n, 4) uint64 of the counters ctr (n, 4) uint64 under one key."""
    n = ctr.shape[0]
    out = np.empty((n, 4), np.uint64)
    for i in range(n):
        o0, o1, o2, o3 = philox4x64_10(ctr[i, 0], ctr[i, 1], ctr[i, 2], ctr[i, 3], k0, k1)
        out[i, 0] = o0
        out[i, 1] = o1
        out[i, 2] = o2
        out[i, 3] = o3
    return out


@njit(cache=True, parallel=True)
def lta_noise_block(k0, k1, step0, n_steps, N):
    """The LTA counter-mode noise of moves step0 .. step0 + n_steps - 1, layout (n_steps, N, 2, 3) =
    [step][slot][bead][xyz] (the replay layout of lta_ladder_numba)."""
    out = np.empty((n_steps, N, 2, 3))
    for s in prange(n_steps):
        st = step0 + s
        for i in range(N):
            z0, z1, z2, z3, z4, z5 = lta_normals6(k0, k1, st, i)
            out[s, i, 0, 0] = z0
            out[s, i, 0, 1] = z1
            out[s, i, 0, 2] = z2
            out[s, i, 1, 0] = z3
            out[s, i, 1, 1] = z4
            out[s, i, 1, 2] = z5
    return out


@njit(cache=True, parallel=True)
def lta_battery_block(k0, k1, step0, n_steps, N):
    """(n_steps, N, 8) normals: the 8 Box-Muller normals of words 0 and 1 (z0..z7) of every walker-step (the
    RNG battery reads the 6 used ones and the word structure)."""
    out = np.empty((n_steps, N, 8))
    for s in prange(n_steps):
        st = np.uint64(step0 + s)
        for i in range(N):
            sl = np.uint64(i)
            a0, a1, a2, a3 = philox4x64_10(st, sl, _ZERO, _ZERO, k0, k1)
            b0, b1, b2, b3 = philox4x64_10(st, sl, _ONE, _ZERO, k0, k1)
            out[s, i, 0], out[s, i, 1] = box_muller(u01(a0), u01(a1))
            out[s, i, 2], out[s, i, 3] = box_muller(u01(a2), u01(a3))
            out[s, i, 4], out[s, i, 5] = box_muller(u01(b0), u01(b1))
            out[s, i, 6], out[s, i, 7] = box_muller(u01(b2), u01(b3))
    return out


@njit(cache=True, parallel=True)
def gateway_noise_block(k0, k1, step0, n_steps, N):
    """Gateway counter-mode noise (n_steps, 2, N): [step][zx | zy][slot]."""
    out = np.empty((n_steps, 2, N))
    for s in prange(n_steps):
        for i in range(N):
            zx, zy = gateway_normals2(k0, k1, step0 + s, i)
            out[s, 0, i] = zx
            out[s, 1, i] = zy
    return out


# =====================================================================================
# Python helpers
# =====================================================================================
def key_from_entropy(entropy):
    """The two uint64 key words SeedSequence(entropy).generate_state(2, np.uint64)."""
    k = np.random.SeedSequence([int(e) for e in entropy]).generate_state(2, np.uint64)
    return np.uint64(k[0]), np.uint64(k[1])


def gateway_key(seed, N):
    return key_from_entropy([seed, N, 2])


def lta_key(seed, N, T_K=150.0):
    return key_from_entropy([seed, N, int(round(float(T_K))), 2])


def ctr_to_int(c):
    """256-bit integer of a counter (c0 least significant), numpy's Philox counter convention."""
    return sum(int(c[i]) << (64 * i) for i in range(4))


def int_to_ctr(v):
    v %= 1 << 256
    return np.array([(v >> (64 * i)) & ((1 << 64) - 1) for i in range(4)], dtype=np.uint64)


def numpy_raw(k0, k1, ctr):
    """Oracle: numpy's Philox4x64-10 block for the counter ``ctr`` (4 words), i.e. np.random.Philox started at
    ctr - 1 (numpy increments before generating), first four random_raw outputs."""
    bg = np.random.Philox(key=np.array([int(k0), int(k1)], dtype=np.uint64), counter=int_to_ctr(ctr_to_int(ctr) - 1))
    return bg.random_raw(4)


def python_philox(ctr, key):
    """Independent pure-Python Philox4x64-10 (Python integers); ctr 4 ints, key 2 ints -> 4 ints."""
    m64 = (1 << 64) - 1
    c = [int(x) for x in ctr]
    k = [int(x) for x in key]
    for r in range(N_ROUNDS):
        if r:
            k = [(k[0] + int(W0)) & m64, (k[1] + int(W1)) & m64]
        p0 = int(M0) * c[0]
        p1 = int(M1) * c[2]
        c = [((p1 >> 64) ^ c[1] ^ k[0]) & m64, p1 & m64, ((p0 >> 64) ^ c[3] ^ k[1]) & m64, p0 & m64]
    return c


def check_counter_injective(N, n_steps, words):
    """Analytic injectivity proof obligation of the counter map (step, slot, word) -> (step, slot, word, 0) for one
    production configuration: the moves use steps 0 .. n_steps - 1, slots 0 .. N - 1 and words 0 .. words - 1; the
    map is the coordinate-wise embedding, injective iff no coordinate wraps modulo 2^64 (every coordinate < 2^64,
    all >= 0) and the fourth word is the constant 0.  Also verifies the actual encoding at the corner points
    (the numba noise functions build the counter with np.uint64 of the same Python ints).  Returns a dict."""
    N = int(N)
    n_steps = int(n_steps)
    words = int(words)
    lim = 1 << 64
    if N < 1 or n_steps < 1 or words < 1:
        raise ValueError("N, n_steps and words must be >= 1")
    maxc = (n_steps - 1, N - 1, words - 1)
    ok = all(0 <= m < lim for m in maxc)
    corners = []
    if ok:                                  # a coordinate >= 2^64 cannot even be encoded (np.uint64 overflows)
        for st in (0, n_steps - 1):
            for sl in (0, N - 1):
                for w in (0, words - 1):
                    enc = (int(np.uint64(st)), int(np.uint64(sl)), int(np.uint64(w)), 0)
                    corners.append(enc == (st, sl, w, 0))
    n_counters = n_steps * N * words
    return dict(N=N, n_steps=n_steps, words=words, max_coords=list(maxc), n_counters=n_counters,
                coords_below_2_64=bool(ok), corners_exact=bool(ok and all(corners)),
                injective=bool(ok and all(corners)))
