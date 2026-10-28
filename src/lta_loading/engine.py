"""A100 engine of histogram ABF (+ uniform-target marginal FR) for ethane/LTA at finite loading.

Physics (docs/lta_loading/PREREGISTRATION.md §2)
------------------------------------------------
* The single-guest model of ``src/lta_ladder_numba.py`` unchanged: 384 rigid framework O, CH3-O
  shifted LJ (93 K, 3.48 A, rc 10 A), harmonic bond (k 400, r0 1.54), overdamped Euler-Maruyama
  q <- (q + h (F + bias)) + sqrt(2 h kT) xi, coordinates never wrapped, minimum image d - L rint(d/L).
* ``n`` ethanes per 2x2x2 box (8 alpha cages).  Guest-guest CH3-CH3 shifted LJ between beads of
  different molecules (Dubbeldam et al. 2004 united atom: 108 K, 3.76 A, rc 10 A); no intramolecular LJ.
* Molecule 0 is TAGGED: the CV is its phi = wrap(2 pi x_COM / a), exactly the single-guest CV; ABF
  biases molecule 0 only, so the bias depends on xi alone and the conditional law of the other n - 1
  guests (the environment) is the unbiased one.
* Environment reinsertion MC every ``mc_every`` steps: one environment guest per walker, new COM
  uniform in the box, new orientation uniform, bond length kept; Metropolis on the intermolecular
  energy; COMs within ``r_block`` of a sodalite centre (the lattice corners) rejected (sodalite cages
  are inaccessible through the 6-rings, the standard blocking).  The proposal is symmetric and
  independent of the bias, so the kernel leaves the biased measure invariant.  Without it the
  environment would relax only through its own unbiased 13.5 kT window hops at 150 K.

Algorithm per evaluated step k (lta_ladder_numba._run_chunk, diagnostics reduced)
--------------------------------------------------------------------------------
forces at the pre-move positions -> deposit of the tagged local force of ALL walkers (fixed point
2^-40, integer atomics: exact, order-free) -> fold into the float64 all-steps / production
accumulators -> save -> bias = ramp clip(M_j / max(C_j, 1), +-60) pi / a on the tagged beads ->
Euler-Maruyama move (counter Philox4x64-10 noise keyed by seed, counter (k, slot, word, molecule))
-> FR at the post-move CV when nxt = k + 1 >= fr_start and (nxt - fr_start) % fr_every == 0 (score
and selection are the numba engine's own ``_fr_score`` / ``fr_select_native`` with its per-run
PCG64 FR stream; whole-walker copies: all n guests) -> environment MC when nxt % mc_every == 0.

``R`` seeds of one (method, N) run batched (one ABF accumulator set per seed): a seed's trajectory
depends only on its own keys and streams, never on the batch it runs in.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))

import lta_ladder_numba as E  # noqa: E402  (validated FR score / selection, IC law, grids, PMF)

KB = E.KB
PI = math.pi
TWO_PI = 2.0 * math.pi
FSCALE = float(2 ** 40)          # fixed-point deposit grid; |f| <= 480 and N <= 8192 keep int64 exact
PRE_MC_BASE = 1 << 60            # Philox step counters of the pre-equilibration (disjoint from production)
PRE_MD_BASE = 1 << 61
ENGINE = "lta_loading/1"

DEFAULT_CFG = dict(
    T_K=150.0, h=2.0e-4, n_grid=180,
    warmup_steps=20000, burn_in_steps=20000,
    fr_start_steps=20000, fr_every=5, fr_rate=0.2, score_clip=2.0, kde_bandwidth=0.10,
    max_event_fraction=0.02, cap_min=1,
    abf_force_clip=60.0, deposit_clip=480.0, abf_bias_scale=1.0,
    eps_go_K=93.0, sigma_go=3.48, eps_gg_K=108.0, sigma_gg=3.76, rc=10.0, r0_bond=1.54, k_bond=400.0,
    n_guests=16, mc_every=1, r_block=3.5, place_sweeps=100, preeq_sweeps=2000, preeq_steps=20000,
    framework_npz="cache/lta/framework.npz",
    tagged_ic="cage",                  # "cage" (single-guest law) | "uniform_x" (diagnostics: x uniform on the axis)
)

CUDA_SRC = r"""
#define PHILOX_M0 0xD2E7470EE14C6C93ULL
#define PHILOX_M1 0xCA5A826395121157ULL
#define PHILOX_W0 0x9E3779B97F4A7C15ULL
#define PHILOX_W1 0xBB67AE8584CAA73BULL
#define LF_PI 3.141592653589793
#define LF_TWO_PI 6.283185307179586
#define LF_EPS 1.0e-12
#define LF_U53 1.1102230246251565e-16

__device__ __forceinline__ void philox(unsigned long long c0, unsigned long long c1, unsigned long long c2,
                                       unsigned long long c3, unsigned long long k0, unsigned long long k1,
                                       unsigned long long o[4])
{
#pragma unroll
    for (int r = 0; r < 10; r++) {
        if (r > 0) { k0 += PHILOX_W0; k1 += PHILOX_W1; }
        const unsigned long long hi0 = __umul64hi(PHILOX_M0, c0), lo0 = PHILOX_M0 * c0;
        const unsigned long long hi1 = __umul64hi(PHILOX_M1, c2), lo1 = PHILOX_M1 * c2;
        c0 = hi1 ^ c1 ^ k0; c1 = lo1; c2 = hi0 ^ c3 ^ k1; c3 = lo0;
    }
    o[0] = c0; o[1] = c1; o[2] = c2; o[3] = c3;
}

__device__ __forceinline__ double u01(unsigned long long o) { return (double)(o >> 11) * LF_U53; }

__device__ __forceinline__ void box_muller(double ua, double ub, double &z0, double &z1)
{
    const double r = sqrt(-2.0 * log(1.0 - ua));
    const double th = (2.0 * LF_PI) * ub;
    z0 = r * cos(th);
    z1 = r * sin(th);
}

// the 6 normals of molecule m of walker-step (step, slot), [bead][xyz]: words 0 and 1, c3 = m
__device__ __forceinline__ void noise6(unsigned long long step, unsigned long long slot, unsigned long long m,
                                       unsigned long long k0, unsigned long long k1, double *z)
{
    unsigned long long o0[4], o1[4];
    philox(step, slot, 0ULL, m, k0, k1, o0);
    philox(step, slot, 1ULL, m, k0, k1, o1);
    box_muller(u01(o0[0]), u01(o0[1]), z[0], z[1]);
    box_muller(u01(o0[2]), u01(o0[3]), z[2], z[3]);
    box_muller(u01(o1[0]), u01(o1[1]), z[4], z[5]);
}

__device__ __forceinline__ double lf_wrap(double p)
{
    const double t = p + LF_PI;
    double m = fmod(t, LF_TWO_PI);
    if (m != 0.0 && m < 0.0) m += LF_TWO_PI;
    return m - LF_PI;
}

__device__ __forceinline__ int lf_bin(double ph, double dphi, int ng)
{
    int j = (int)floor((ph + LF_PI) / dphi);
    j %= ng;
    if (j < 0) j += ng;
    return j;
}

__device__ __forceinline__ void lj_force(double x, double y, double z, double ox, double oy, double oz, double L,
                                         double invL, double rc2, double s2, double c24,
                                         double &fx, double &fy, double &fz)
{
    double dx = x - ox; dx = dx - L * rint(dx * invL);
    double dy = y - oy; dy = dy - L * rint(dy * invL);
    double dz = z - oz; dz = dz - L * rint(dz * invL);
    const double r2 = (dx * dx + dy * dy) + dz * dz;
    if (r2 < rc2) {
        const double inv = 1.0 / (r2 > LF_EPS ? r2 : LF_EPS);
        const double s = s2 * inv;
        const double sr6 = (s * s) * s;
        const double coef = (c24 * ((2.0 * sr6) * sr6 - sr6)) * inv;
        fx += coef * dx; fy += coef * dy; fz += coef * dz;
    }
}

__device__ __forceinline__ double lj_energy(double x, double y, double z, double ox, double oy, double oz, double L,
                                            double invL, double rc2, double s2, double c4, double vrc)
{
    double dx = x - ox; dx = dx - L * rint(dx * invL);
    double dy = y - oy; dy = dy - L * rint(dy * invL);
    double dz = z - oz; dz = dz - L * rint(dz * invL);
    const double r2 = (dx * dx + dy * dy) + dz * dz;
    if (r2 < rc2) {
        const double s = s2 / (r2 > LF_EPS ? r2 : LF_EPS);
        const double sr6 = (s * s) * s;
        return c4 * (sr6 * sr6 - sr6) - vrc;
    }
    return 0.0;
}

// deterministic block sum (fixed shuffle tree, warp totals in warp order); blockDim a multiple of 32; sh >= 32
__device__ __forceinline__ double block_sum(double v, double *sh)
{
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    if (lane == 0) sh[warp] = v;
    __syncthreads();
    double t = 0.0;
    if (warp == 0) {
        t = (lane < nw) ? sh[lane] : 0.0;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) t += __shfl_down_sync(0xffffffffu, t, o);
        if (lane == 0) sh[0] = t;
    }
    __syncthreads();
    t = sh[0];
    __syncthreads();
    return t;
}

// one block per walker; thread (bead b, lane s) of a G-lane group; blockDim = 32 * ceil(2 n G / 32)
extern "C" __global__ void lf_forces(const double *__restrict__ q, double *__restrict__ F,
                                     const double *__restrict__ O, double *__restrict__ phi, int *__restrict__ jb,
                                     double *__restrict__ fl, unsigned long long *__restrict__ stM,
                                     unsigned long long *__restrict__ stC, int N, int n, int nO, int G, double L,
                                     double invL, double rc2, double s2o, double c24o, double s2g, double c24g,
                                     double kb, double r0, double cphi, double cf, double dphi, int ng, double dclip,
                                     double fscale, int deposit)
{
    extern __shared__ double sh[];
    double *sO = sh;
    double *sq = sO + 3 * nO;
    double *sF = sq + 6 * n;
    const int w = blockIdx.x;
    const int B = 2 * n;
    const double *qw = q + (size_t)w * 6 * n;
    for (int i = threadIdx.x; i < 3 * nO; i += blockDim.x) sO[i] = O[i];
    for (int i = threadIdx.x; i < 6 * n; i += blockDim.x) sq[i] = qw[i];
    __syncthreads();
    const int b = threadIdx.x / G, s = threadIdx.x % G;
    double fx = 0.0, fy = 0.0, fz = 0.0;
    if (b < B) {
        const double x = sq[3 * b], y = sq[3 * b + 1], z = sq[3 * b + 2];
        for (int k = s; k < nO; k += G)
            lj_force(x, y, z, sO[3 * k], sO[3 * k + 1], sO[3 * k + 2], L, invL, rc2, s2o, c24o, fx, fy, fz);
        const int mb = b >> 1;
        for (int j = s; j < B; j += G)
            if ((j >> 1) != mb)
                lj_force(x, y, z, sq[3 * j], sq[3 * j + 1], sq[3 * j + 2], L, invL, rc2, s2g, c24g, fx, fy, fz);
    }
    for (int o = G >> 1; o > 0; o >>= 1) {
        fx += __shfl_down_sync(0xffffffffu, fx, o, G);
        fy += __shfl_down_sync(0xffffffffu, fy, o, G);
        fz += __shfl_down_sync(0xffffffffu, fz, o, G);
    }
    if (s == 0 && b < B) { sF[3 * b] = fx; sF[3 * b + 1] = fy; sF[3 * b + 2] = fz; }
    __syncthreads();
    if (threadIdx.x < n) {
        const int m = threadIdx.x;
        const double *p0 = sq + 6 * m, *p1 = sq + 6 * m + 3;
        const double dx = p0[0] - p1[0], dy = p0[1] - p1[1], dz = p0[2] - p1[2];
        double r = sqrt((dx * dx + dy * dy) + dz * dz);
        if (r < LF_EPS) r = LF_EPS;
        const double c = (-kb) * (r - r0);
        const double fbx = (c * dx) / r, fby = (c * dy) / r, fbz = (c * dz) / r;
        sF[6 * m] = fbx + sF[6 * m]; sF[6 * m + 1] = fby + sF[6 * m + 1]; sF[6 * m + 2] = fbz + sF[6 * m + 2];
        sF[6 * m + 3] = (-fbx) + sF[6 * m + 3]; sF[6 * m + 4] = (-fby) + sF[6 * m + 4];
        sF[6 * m + 5] = (-fbz) + sF[6 * m + 5];
    }
    __syncthreads();
    double *Fw = F + (size_t)w * 6 * n;
    for (int i = threadIdx.x; i < 6 * n; i += blockDim.x) Fw[i] = sF[i];
    if (threadIdx.x == 0) {
        const double x = (sq[0] + sq[3]) * 0.5;
        const double ph = lf_wrap(cphi * x);
        const int j = lf_bin(ph, dphi, ng);
        double f = cf * (sF[0] + sF[3]);
        if (f < -dclip) f = -dclip; else if (f > dclip) f = dclip;
        phi[w] = ph; jb[w] = j; fl[w] = f;
        if (deposit) {
            const int r = w / N;
            const long long iv = llrint(f * fscale);
            atomicAdd(&stM[(size_t)r * ng + j], (unsigned long long)iv);
            atomicAdd(&stC[(size_t)r * ng + j], 1ULL);
        }
    }
}

// per-step bin sums (exact integers) into the float64 accumulators acc (R, 4, ng): FS, CS, FSP, CSP
extern "C" __global__ void lf_fold(unsigned long long *stM, unsigned long long *stC, double *acc, int R, int ng,
                                   double inv_fscale, int prod)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= R * ng) return;
    const unsigned long long c = stC[i];
    if (c) {
        const int r = i / ng, j = i % ng;
        const double M = (double)(long long)stM[i] * inv_fscale;
        const double C = (double)c;
        double *a = acc + (size_t)r * 4 * ng;
        a[j] += M;
        a[ng + j] += C;
        if (prod) { a[2 * ng + j] += M; a[3 * ng + j] += C; }
        stM[i] = 0ULL;
        stC[i] = 0ULL;
    }
}

// Euler-Maruyama move; one thread per (walker, molecule); ABF bias on the tagged molecule only
extern "C" __global__ void lf_move(double *q, const double *F, const int *jb, const double *acc, const double *table,
                                   const unsigned long long *key0, const unsigned long long *key1, long long *cell,
                                   int *cross, int W, int N, int n, int ng, unsigned long long step, double dt,
                                   double ns, double scl, double fclip, double gpx, double a, int use_table,
                                   int freeze_tagged)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= W * n) return;
    const int w = t / n, m = t % n, r = w / N;
    const unsigned long long slot = (unsigned long long)(w % N);
    if (m == 0 && freeze_tagged) return;
    double bx = 0.0;
    if (m == 0) {
        const int j = jb[w];
        double G;
        if (use_table) {
            G = table[j];
        } else {
            const double *A = acc + (size_t)r * 4 * ng;
            const double c = A[ng + j];
            G = c > 0.0 ? A[j] / (c > 1.0 ? c : 1.0) : 0.0;
        }
        if (G < -fclip) G = -fclip; else if (G > fclip) G = fclip;
        bx = (scl * G) * gpx;
    }
    double z[6];
    noise6(step, slot, (unsigned long long)m, key0[r], key1[r], z);
    double *qm = q + (size_t)w * 6 * n + 6 * m;
    const double *Fm = F + (size_t)w * 6 * n + 6 * m;
    for (int b = 0; b < 2; b++) {
        qm[3 * b] = (qm[3 * b] + dt * (Fm[3 * b] + bx)) + ns * z[3 * b];
        qm[3 * b + 1] = (qm[3 * b + 1] + dt * Fm[3 * b + 1]) + ns * z[3 * b + 1];
        qm[3 * b + 2] = (qm[3 * b + 2] + dt * Fm[3 * b + 2]) + ns * z[3 * b + 2];
    }
    if (m == 0) {
        const double x = (qm[0] + qm[3]) * 0.5;
        const long long c = (long long)floor(x / a);
        const long long lc = cell[w];
        if (c != lc) {
            const long long d = c - lc;
            atomicAdd(&cross[r], (int)(d > 0 ? d : -d));
            cell[w] = c;
        }
    }
}

extern "C" __global__ void lf_cv(const double *q, double *phin, int *jn, int W, int n, double cphi, double dphi,
                                 int ng)
{
    const int w = blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= W) return;
    const double *qw = q + (size_t)w * 6 * n;
    const double ph = lf_wrap(cphi * ((qw[0] + qw[3]) * 0.5));
    phin[w] = ph;
    jn[w] = lf_bin(ph, dphi, ng);
}

// environment reinsertion: one block (64 threads) per walker
extern "C" __global__ void lf_mc(double *q, const double *O, const unsigned long long *key0,
                                 const unsigned long long *key1, int *nacc, int *natt, double *dUo, int N, int n, int nO,
                                 unsigned long long step, int m_fixed, double L, double invL, double rc2, double s2o,
                                 double c4o, double vrco, double s2g, double c4g, double vrcg, double beta, double a,
                                 double rblock2, int place, double ucap)
{
    __shared__ double red[32];
    __shared__ double prop[6];
    __shared__ int sm, sblocked;
    __shared__ double suacc;
    const int w = blockIdx.x, r = w / N;
    const unsigned long long slot = (unsigned long long)(w % N);
    double *qw = q + (size_t)w * 6 * n;
    if (threadIdx.x == 0) {
        unsigned long long o8[4], o9[4], o10[4];
        philox(step, slot, 8ULL, 0ULL, key0[r], key1[r], o8);
        philox(step, slot, 9ULL, 0ULL, key0[r], key1[r], o9);
        philox(step, slot, 10ULL, 0ULL, key0[r], key1[r], o10);
        int m = m_fixed >= 1 ? m_fixed : 1 + (int)(u01(o8[0]) * (double)(n - 1));
        if (m > n - 1) m = n - 1;
        const double cx = u01(o8[2]) * L, cy = u01(o8[3]) * L, cz = u01(o9[0]) * L;
        const double ex = cx - a * rint(cx / a), ey = cy - a * rint(cy / a), ez = cz - a * rint(cz / a);
        sblocked = ((ex * ex + ey * ey) + ez * ez) < rblock2;
        double n0, n1, n2, n3;
        box_muller(u01(o9[1]), u01(o9[2]), n0, n1);
        box_muller(u01(o9[3]), u01(o10[0]), n2, n3);
        double nn = sqrt((n0 * n0 + n1 * n1) + n2 * n2);
        if (nn < LF_EPS) nn = LF_EPS;
        const double *p0 = qw + 6 * m, *p1 = qw + 6 * m + 3;
        const double dx = p0[0] - p1[0], dy = p0[1] - p1[1], dz = p0[2] - p1[2];
        const double hr = 0.5 * sqrt((dx * dx + dy * dy) + dz * dz) / nn;
        prop[0] = cx + hr * n0; prop[1] = cy + hr * n1; prop[2] = cz + hr * n2;
        prop[3] = cx - hr * n0; prop[4] = cy - hr * n1; prop[5] = cz - hr * n2;
        sm = m;
        suacc = u01(o8[1]);
    }
    __syncthreads();
    const int m = sm;
    const double *po = qw + 6 * m;
    double eo = 0.0, en = 0.0;
    const int nitem = nO + 2 * n;
    for (int k = threadIdx.x; k < nitem; k += blockDim.x) {
        if (k < nO) {
            const double ox = O[3 * k], oy = O[3 * k + 1], oz = O[3 * k + 2];
            for (int bb = 0; bb < 2; bb++) {
                eo += lj_energy(po[3 * bb], po[3 * bb + 1], po[3 * bb + 2], ox, oy, oz, L, invL, rc2, s2o, c4o, vrco);
                en += lj_energy(prop[3 * bb], prop[3 * bb + 1], prop[3 * bb + 2], ox, oy, oz, L, invL, rc2, s2o, c4o,
                                vrco);
            }
        } else {
            const int jbd = k - nO;
            if ((jbd >> 1) != m) {
                const double ox = qw[3 * jbd], oy = qw[3 * jbd + 1], oz = qw[3 * jbd + 2];
                for (int bb = 0; bb < 2; bb++) {
                    eo += lj_energy(po[3 * bb], po[3 * bb + 1], po[3 * bb + 2], ox, oy, oz, L, invL, rc2, s2g, c4g,
                                    vrcg);
                    en += lj_energy(prop[3 * bb], prop[3 * bb + 1], prop[3 * bb + 2], ox, oy, oz, L, invL, rc2, s2g,
                                    c4g, vrcg);
                }
            }
        }
    }
    eo = block_sum(eo, red);
    en = block_sum(en, red);
    if (threadIdx.x == 0) {
        atomicAdd(&natt[r], 1);
        dUo[w] = sblocked ? nan("") : en - eo;
        if (!sblocked) {
            const double dU = en - eo;
            // place = 1: initial placement only (not detailed balance) -- any non-overlapping site is taken
            if ((place && en <= ucap) || dU <= 0.0 || suacc < exp(-beta * dU)) {
                for (int i = 0; i < 6; i++) qw[6 * m + i] = prop[i];
                atomicAdd(&nacc[r], 1);
            }
        }
    }
}

// total potential energy per walker (validation): framework LJ + guest-guest LJ (each pair once) + bonds
extern "C" __global__ void lf_energy(const double *q, const double *O, double *U, int n, int nO, double L,
                                     double invL, double rc2, double s2o, double c4o, double vrco, double s2g,
                                     double c4g, double vrcg, double kb, double r0)
{
    __shared__ double red[32];
    const int w = blockIdx.x, B = 2 * n;
    const double *qw = q + (size_t)w * 6 * n;
    double e = 0.0;
    for (int k = threadIdx.x; k < B * nO; k += blockDim.x) {
        const int b = k / nO, o = k % nO;
        e += lj_energy(qw[3 * b], qw[3 * b + 1], qw[3 * b + 2], O[3 * o], O[3 * o + 1], O[3 * o + 2], L, invL, rc2,
                       s2o, c4o, vrco);
    }
    for (int k = threadIdx.x; k < B * B; k += blockDim.x) {
        const int i = k / B, j = k % B;
        if (j > i && (i >> 1) != (j >> 1))
            e += lj_energy(qw[3 * i], qw[3 * i + 1], qw[3 * i + 2], qw[3 * j], qw[3 * j + 1], qw[3 * j + 2], L, invL,
                           rc2, s2g, c4g, vrcg);
    }
    for (int m = threadIdx.x; m < n; m += blockDim.x) {
        const double dx = qw[6 * m] - qw[6 * m + 3], dy = qw[6 * m + 1] - qw[6 * m + 4], dz = qw[6 * m + 2] - qw[6 * m + 5];
        const double rr = sqrt((dx * dx + dy * dy) + dz * dz);
        e += (0.5 * kb) * ((rr - r0) * (rr - r0));
    }
    e = block_sum(e, red);
    if (threadIdx.x == 0) U[w] = e;
}
"""

_MODULE = None


def _module():
    global _MODULE
    if _MODULE is None:
        import cupy as cp
        _MODULE = cp.RawModule(code=CUDA_SRC, options=("--std=c++14", "--fmad=false"))
    return _MODULE


def source_sha256():
    with open(os.path.abspath(__file__), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def resolve_cfg(**overrides):
    c = dict(DEFAULT_CFG)
    for k, v in overrides.items():
        if k not in DEFAULT_CFG:
            raise KeyError(f"unknown cfg key {k!r}")
        c[k] = v
    if float(c["deposit_clip"]) != 8.0 * float(c["abf_force_clip"]):
        raise ValueError("deposit_clip must be 8 x abf_force_clip (the campaign's rule)")
    return c


def physics(c):
    """Derived constants of a cfg (units A, kJ/mol, overdamped time units)."""
    T = float(c["T_K"])
    beta = 1.0 / (KB * T)
    O, a, L = E.framework(c["framework_npz"])
    rc = float(c["rc"])

    def lj(eps_K, sig):
        eps = float(eps_K) * KB
        sr6c = (float(sig) / rc) ** 6
        return dict(s2=float(sig) ** 2, c24=24.0 * eps, c4=4.0 * eps, vrc=4.0 * eps * (sr6c * sr6c - sr6c))

    ng = int(c["n_grid"])
    dphi = TWO_PI / ng
    return dict(T=T, beta=beta, O=O, a=a, L=L, invL=1.0 / L, rc2=rc * rc, go=lj(c["eps_go_K"], c["sigma_go"]),
                gg=lj(c["eps_gg_K"], c["sigma_gg"]), kb=float(c["k_bond"]), r0=float(c["r0_bond"]),
                h=float(c["h"]), ns=math.sqrt(2.0 * float(c["h"]) / beta), cphi=TWO_PI / a, cf=-(a / TWO_PI),
                gpx=PI / a, ng=ng, dphi=dphi, grid0=-PI + 0.5 * dphi, qv=1.0 / (ng * dphi),
                fclip=float(c["abf_force_clip"]), dclip=float(c["deposit_clip"]))


def seed_material(seed, N, T_K, n):
    """Philox keys and the three PCG64 streams (IC, -, FR) of one run, from SeedSequence([seed, N, T, n])."""
    ss = np.random.SeedSequence([int(seed), int(N), int(round(float(T_K))), int(n)])
    keys = ss.generate_state(2, dtype=np.uint64)
    kids = ss.spawn(3)
    gens = [np.random.Generator(np.random.PCG64(k)) for k in kids]
    return keys, gens


def initial_positions(gen, N, n, a, L, r0):
    """Each of the n guests of each walker by the single-guest law (uniform alpha cage + 0.5 A jitter,
    uniform orientation, bond r0); molecule 0 is the tagged one.  Overlaps are removed by the MC
    pre-equilibration."""
    q = E.initial_conditions(gen, N * n, a, L, r0=r0)
    return np.ascontiguousarray(q.reshape(N, n, 2, 3))


def _uniform_x_tagged(gen, qt, p):
    """Diagnostics only: tagged COM x uniform on [0, L) (rejection on the framework energy < 0), y, z kept on its
    cage axis, bond along x (the only orientation that threads the 8-ring window)."""
    O, L, go = p["O"], p["L"], p["go"]
    com = 0.5 * (qt[:, 0] + qt[:, 1])
    out = qt.copy()
    todo = np.arange(qt.shape[0])
    for _ in range(1000):
        if todo.size == 0:
            break
        x = gen.uniform(0.0, L, todo.size)
        c = com[todo].copy()
        c[:, 0] = x
        beads = np.stack([c - [0.5 * p["r0"], 0, 0], c + [0.5 * p["r0"], 0, 0]], axis=1)
        d = beads[:, :, None, :] - O[None, None]
        d -= L * np.rint(d / L)
        r2 = (d * d).sum(-1)
        s6 = np.where(r2 < p["rc2"], (go["s2"] / r2) ** 3, 0.0)
        U = np.where(r2 < p["rc2"], go["c4"] * (s6 * s6 - s6) - go["vrc"], 0.0).sum((1, 2))
        ok = U < 0.0
        out[todo[ok]] = beads[ok]
        todo = todo[~ok]
    if todo.size:
        raise RuntimeError(f"{todo.size} tagged guests found no non-overlapping x")
    return out


class Batch:
    """R seeds of one (method, N, n) run, batched on one GPU."""

    def __init__(self, seeds, method, N, n_steps, cfg=None, table=None, save_steps=None, verbose=True):
        import cupy as cp
        self.cp = cp
        self.c = cfg if cfg is not None else resolve_cfg()
        c = self.c
        if method not in ("abf", "fr", "frozen"):
            raise ValueError(method)
        self.method = method
        self.seeds = [int(s) for s in seeds]
        self.R = len(self.seeds)
        self.N = int(N)
        self.n = int(c["n_guests"])
        self.n_steps = int(n_steps)
        self.W = self.R * self.N
        if self.N > 8192:
            raise ValueError("N > 8192 overflows the fixed-point deposit")
        self.p = physics(c)
        p = self.p
        self.verbose = verbose
        self.is_fr = method == "fr"
        self.cap = max(int(c["cap_min"]), int(float(c["max_event_fraction"]) * self.N))
        self.fr_on = self.is_fr and self.cap >= 1 and float(c["fr_rate"]) > 0.0
        if self.is_fr and self.N < 2:
            raise ValueError("FR needs N >= 2")
        self.table = None
        if method == "frozen":
            if table is None:
                raise ValueError("frozen needs a bias table (bin forces Gamma_j)")
            self.table = cp.asarray(np.asarray(table, dtype=np.float64))
        mod = _module()
        self.k_forces = mod.get_function("lf_forces")
        self.k_fold = mod.get_function("lf_fold")
        self.k_move = mod.get_function("lf_move")
        self.k_cv = mod.get_function("lf_cv")
        self.k_mc = mod.get_function("lf_mc")
        self.k_energy = mod.get_function("lf_energy")

        q0 = np.empty((self.R, self.N, self.n, 2, 3))
        k0 = np.empty(self.R, np.uint64)
        k1 = np.empty(self.R, np.uint64)
        self.fr_gen = []
        for r, s in enumerate(self.seeds):
            keys, gens = seed_material(s, self.N, c["T_K"], self.n)
            k0[r], k1[r] = keys
            q0[r] = initial_positions(gens[0], self.N, self.n, p["a"], p["L"], p["r0"])
            if c.get("tagged_ic", "cage") == "uniform_x":
                q0[r, :, 0] = _uniform_x_tagged(gens[0], q0[r, :, 0], p)
            self.fr_gen.append(gens[2])
        self.q = cp.asarray(q0.reshape(self.W, self.n, 2, 3))
        self.key0 = cp.asarray(k0)
        self.key1 = cp.asarray(k1)
        self.O = cp.asarray(np.ascontiguousarray(p["O"].reshape(-1)))
        self.nO = int(p["O"].shape[0])
        self.F = cp.zeros_like(self.q)
        self.phi = cp.zeros(self.W)
        self.jb = cp.zeros(self.W, dtype=cp.int32)
        self.fl = cp.zeros(self.W)
        self.phin = cp.zeros(self.W)
        self.jn = cp.zeros(self.W, dtype=cp.int32)
        ng = p["ng"]
        self.stM = cp.zeros(self.R * ng, dtype=cp.uint64)
        self.stC = cp.zeros(self.R * ng, dtype=cp.uint64)
        self.acc = cp.zeros((self.R, 4, ng))
        self.cross = cp.zeros(self.R, dtype=cp.int32)
        self.mc_acc = cp.zeros(self.R, dtype=cp.int32)
        self.mc_att = cp.zeros(self.R, dtype=cp.int32)
        self.mc_dU = cp.zeros(self.W)
        self.cell = cp.zeros(self.W, dtype=cp.int64)
        self.anc = np.tile(np.arange(self.N, dtype=np.int64), (self.R, 1))
        self.repl = np.zeros(self.R, dtype=np.int64)
        self.nopp = 0
        self.score_std = np.zeros(self.R)

        B = 2 * self.n
        self.G = 1
        while self.G * 2 <= 32 and B * self.G * 2 <= 128:
            self.G *= 2
        self.bt_forces = 32 * ((B * self.G + 31) // 32)
        self.sh_forces = 8 * (3 * self.nO + 12 * self.n)
        self.save_steps = (np.asarray(save_steps, dtype=np.int64) if save_steps is not None
                           else E.budget_save_grid(self.n_steps)[0])
        if self.save_steps[0] != 0:
            self.save_steps = np.concatenate([[0], self.save_steps])
        # FR scratch (numba)
        if self.is_fr:
            self.Kk, _, _ = E.kde_matrix(ng, float(c["kde_bandwidth"]))
            self.fs = dict(cnt=np.zeros(ng), pk=np.empty(ng), raw=np.empty(self.N), sc=np.empty(self.N),
                           touched=np.empty(ng, np.int64), cand=np.empty(self.N, np.int64),
                           dsel=np.empty(self.N, np.int64), ssel=np.empty(self.N, np.int64),
                           dp=np.empty(self.N), cdf=np.empty(self.N))
        self.timers = dict(preeq_s=0.0, run_s=0.0)

    # ------------------------------------------------------------------ kernels
    def _i(self, v):
        return np.int32(v)

    def _d(self, v):
        return np.float64(v)

    def forces(self, deposit):
        p = self.p
        self.k_forces((self.W,), (self.bt_forces,),
                      (self.q, self.F, self.O, self.phi, self.jb, self.fl, self.stM, self.stC,
                       self._i(self.N), self._i(self.n), self._i(self.nO), self._i(self.G), self._d(p["L"]),
                       self._d(p["invL"]), self._d(p["rc2"]), self._d(p["go"]["s2"]), self._d(p["go"]["c24"]),
                       self._d(p["gg"]["s2"]), self._d(p["gg"]["c24"]), self._d(p["kb"]), self._d(p["r0"]),
                       self._d(p["cphi"]), self._d(p["cf"]), self._d(p["dphi"]), self._i(p["ng"]),
                       self._d(p["dclip"]), self._d(FSCALE), self._i(1 if deposit else 0)),
                      shared_mem=self.sh_forces)

    def fold(self, prod):
        n = self.R * self.p["ng"]
        self.k_fold(((n + 255) // 256,), (256,),
                    (self.stM, self.stC, self.acc, self._i(self.R), self._i(self.p["ng"]), self._d(1.0 / FSCALE),
                     self._i(1 if prod else 0)))

    def move(self, step, scl, freeze_tagged=False):
        p = self.p
        nt = self.W * self.n
        table = self.table if self.table is not None else self.acc
        self.k_move(((nt + 127) // 128,), (128,),
                    (self.q, self.F, self.jb, self.acc, table, self.key0, self.key1, self.cell, self.cross,
                     self._i(self.W), self._i(self.N), self._i(self.n), self._i(p["ng"]), np.uint64(step),
                     self._d(p["h"]), self._d(p["ns"]), self._d(scl), self._d(p["fclip"]), self._d(p["gpx"]),
                     self._d(p["a"]), self._i(1 if self.table is not None else 0),
                     self._i(1 if freeze_tagged else 0)))

    def cv(self):
        p = self.p
        self.k_cv(((self.W + 127) // 128,), (128,),
                  (self.q, self.phin, self.jn, self._i(self.W), self._i(self.n), self._d(p["cphi"]),
                   self._d(p["dphi"]), self._i(p["ng"])))

    def mc(self, step, m_fixed=-1, place=False):
        if self.n < 2:
            return
        p = self.p
        self.k_mc((self.W,), (64,),
                  (self.q, self.O, self.key0, self.key1, self.mc_acc, self.mc_att, self.mc_dU, self._i(self.N),
                   self._i(self.n),
                   self._i(self.nO), np.uint64(step), self._i(m_fixed), self._d(p["L"]), self._d(p["invL"]),
                   self._d(p["rc2"]), self._d(p["go"]["s2"]), self._d(p["go"]["c4"]), self._d(p["go"]["vrc"]),
                   self._d(p["gg"]["s2"]), self._d(p["gg"]["c4"]), self._d(p["gg"]["vrc"]), self._d(p["beta"]),
                   self._d(p["a"]), self._d(float(self.c["r_block"]) ** 2), self._i(1 if place else 0),
                   self._d(0.0)))

    def energies(self):
        p = self.p
        U = self.cp.zeros(self.W)
        self.k_energy((self.W,), (128,),
                      (self.q, self.O, U, self._i(self.n), self._i(self.nO), self._d(p["L"]), self._d(p["invL"]),
                       self._d(p["rc2"]), self._d(p["go"]["s2"]), self._d(p["go"]["c4"]), self._d(p["go"]["vrc"]),
                       self._d(p["gg"]["s2"]), self._d(p["gg"]["c4"]), self._d(p["gg"]["vrc"]), self._d(p["kb"]),
                       self._d(p["r0"])))
        return U

    # ------------------------------------------------------------------ phases
    def reset_cells(self):
        cp = self.cp
        x = 0.5 * (self.q[:, 0, 0, 0] + self.q[:, 0, 1, 0])
        self.cell[...] = cp.floor(x / self.p["a"]).astype(cp.int64)

    def preequilibrate(self):
        """Environment equilibration around the tagged guest held at its IC (outside the budget):
        placement sweeps (a reinsertion is taken whenever the new site does not overlap, U_inter <= 0: the IC puts
        2-3 guests on one cage centre, and plain Metropolis from there parks guests inside framework walls and
        leaves them there -- diagnosis 2026-10-28, results/lta_loading/validation/debug_preeq.json), then
        Metropolis sweeps, then dynamics with the tagged guest frozen + reinsertion MC."""
        t0 = time.perf_counter()
        c = self.c
        if self.n >= 2:
            ctr = 0
            for _ in range(int(c["place_sweeps"])):
                for m in range(1, self.n):
                    self.mc(PRE_MC_BASE + ctr, m_fixed=m, place=True)
                    ctr += 1
            for _ in range(int(c["preeq_sweeps"])):
                for m in range(1, self.n):
                    self.mc(PRE_MC_BASE + ctr, m_fixed=m)
                    ctr += 1
            for k in range(int(c["preeq_steps"])):
                self.forces(deposit=False)
                self.move(PRE_MD_BASE + k, 0.0, freeze_tagged=True)
                if (k + 1) % int(c["mc_every"]) == 0:
                    self.mc(PRE_MC_BASE + ctr)
                    ctr += 1
        self.mc_acc[...] = 0
        self.mc_att[...] = 0
        self.reset_cells()
        self.check_physical("after pre-equilibration")
        self.timers["preeq_s"] = time.perf_counter() - t0

    def check_physical(self, where, fmax_cap=1.0e4):
        """Raise unless every walker is finite and overlap-free (max |F| <= fmax_cap kJ/mol/A; typical ~ 70).
        A walker left overlapped is not merely improbable: its forces throw it to coordinates where the
        minimum image loses all precision, and it then corrupts the shared ABF estimator silently."""
        cp = self.cp
        self.forces(deposit=False)
        fw = cp.abs(self.F).reshape(self.W, -1).max(axis=1)
        bad = int((~cp.isfinite(fw) | (fw > fmax_cap)).sum().get())
        if bad or not bool(cp.isfinite(self.q).all().get()):
            raise FloatingPointError(f"{bad} of {self.W} walkers non-physical {where} "
                                     f"(max |F| {float(cp.nanmax(fw).get()):.3g})")

    def fr_opportunity(self):
        cp = self.cp
        c = self.c
        p = self.p
        self.cv()
        phin = self.phin.get()
        jn = self.jn.get().astype(np.int64)
        N = self.N
        fs = self.fs
        dst, src = [], []
        self.nopp += 1
        for r in range(self.R):
            sl = slice(r * N, (r + 1) * N)
            sstd, _ = E._fr_score(np.ascontiguousarray(phin[sl]), np.ascontiguousarray(jn[sl]), N, p["ng"],
                                  p["dphi"], self.Kk, p["grid0"], p["qv"], math.log(p["qv"]),
                                  float(c["score_clip"]), fs["cnt"], fs["pk"], fs["raw"], fs["sc"], fs["touched"])
            self.score_std[r] += sstd
            if not self.fr_on:
                continue
            ublk = self.fr_gen[r].random(N + 2 * self.cap)
            nev = E.fr_select_native(fs["sc"], N, self.cap, float(c["fr_rate"]),
                                     float(c["h"]) * int(c["fr_every"]), ublk, fs["cand"], fs["dsel"], fs["ssel"],
                                     fs["dp"], fs["cdf"])
            if nev:
                d = fs["dsel"][:nev].copy()
                s = fs["ssel"][:nev].copy()
                self.anc[r, d] = self.anc[r, s]
                self.repl[r] += nev
                dst.append(r * N + d)
                src.append(r * N + s)
        if dst:
            d = cp.asarray(np.concatenate(dst))
            s = cp.asarray(np.concatenate(src))
            self.q[d] = self.q[s]
            self.cell[d] = self.cell[s]

    def run(self, out_path=None, label=""):
        cp = self.cp
        c = self.c
        p = self.p
        ng = p["ng"]
        warm = max(int(c["warmup_steps"]), 1)
        burn = int(c["burn_in_steps"])
        fr_start = int(c["fr_start_steps"])
        fr_every = int(c["fr_every"])
        mc_every = int(c["mc_every"])
        bscale = float(c["abf_bias_scale"])
        S = len(self.save_steps)
        sv_acc = np.zeros((S, self.R, 4, ng))
        sv_hist = np.zeros((S, self.R, ng), dtype=np.int32)
        sv_i = np.zeros((S, self.R, 5), dtype=np.int64)          # repl, cross, mc_acc, mc_att, n_unique_anc
        sv_f = np.zeros((S, self.R, 2))                          # ess/N, max lineage share
        sv_wall = np.zeros(S)
        sv_U = np.zeros((S, self.R))                             # mean potential energy per seed (drift monitor)
        sp = 0
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        for k in range(self.n_steps + 1):
            self.forces(deposit=True)
            self.fold(k >= burn)
            if sp < S and self.save_steps[sp] == k:
                acc = self.acc.get()
                sv_acc[sp] = acc
                hist = cp.bincount(cp.arange(self.W, dtype=cp.int32) // self.N * ng + self.jb,
                                   minlength=self.R * ng).reshape(self.R, ng)
                sv_hist[sp] = hist.get()
                sv_i[sp, :, 0] = self.repl
                sv_i[sp, :, 1] = self.cross.get()
                sv_i[sp, :, 2] = self.mc_acc.get()
                sv_i[sp, :, 3] = self.mc_att.get()
                for r in range(self.R):
                    cnt = np.bincount(self.anc[r], minlength=self.N)
                    sv_i[sp, r, 4] = int((cnt > 0).sum())
                    sv_f[sp, r, 0] = self.N / float((cnt * cnt).sum())
                    sv_f[sp, r, 1] = cnt.max() / self.N
                sv_U[sp] = self.energies().reshape(self.R, self.N).mean(axis=1).get()
                sv_wall[sp] = time.perf_counter() - t0
                if not np.isfinite(acc).all():
                    raise FloatingPointError(f"non-finite accumulators at step {k}")
                fmax = float(self.cp.abs(self.F).max().get())
                if not fmax <= 1.0e6:
                    raise FloatingPointError(f"force blow-up at step {k}: max |F| {fmax:.3g}")
                if self.verbose and (sp % 25 == 0 or sp == S - 1):
                    print(f"  [{label}] step {k}/{self.n_steps} save {sp + 1}/{S} wall {sv_wall[sp]:.1f}s "
                          f"repl {int(self.repl.sum())} cross {int(sv_i[sp, :, 1].sum())} "
                          f"mc {int(sv_i[sp, :, 2].sum())}/{int(sv_i[sp, :, 3].sum())}", flush=True)
                sp += 1
            if k == self.n_steps:
                break
            self.move(k, bscale * min(1.0, k / warm))
            nxt = k + 1
            if self.is_fr and nxt >= fr_start and (nxt - fr_start) % fr_every == 0:
                self.fr_opportunity()
            if self.n >= 2 and nxt % mc_every == 0:
                self.mc(nxt)
        cp.cuda.Stream.null.synchronize()
        self.timers["run_s"] = time.perf_counter() - t0
        q = self.q.get()
        if not np.isfinite(q).all():
            raise FloatingPointError("non-finite positions at the end of the run")
        out = dict(save_step=self.save_steps, save_u=self.save_steps / float(max(self.n_steps, 1)),
                   sv_acc=sv_acc, sv_hist=sv_hist, sv_i=sv_i, sv_f=sv_f, sv_wall=sv_wall, sv_U=sv_U,
                   seeds=np.asarray(self.seeds), score_std_sum=self.score_std, n_opp=np.int64(self.nopp))
        meta = dict(engine=ENGINE, engine_sha256=source_sha256(), method=self.method, N=self.N, n_guests=self.n,
                    n_steps=self.n_steps, R=self.R, cap=self.cap, fr_on=self.fr_on, cfg=self.c,
                    timers=self.timers, label=label,
                    sv_i_cols=["repl", "cross", "mc_acc", "mc_att", "n_unique_anc"],
                    sv_f_cols=["ess_over_N", "max_lineage_share"])
        if out_path is not None:
            save_npz(out_path, out, meta)
        return out, meta


def save_npz(path, out, meta):
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = path + f".tmp{os.getpid()}.npz"
    np.savez_compressed(tmp, __meta__=np.asarray(json.dumps(meta, default=str)), **out)
    os.replace(tmp, path)


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        out = {k: z[k] for k in z.files if k != "__meta__"}
        meta = json.loads(str(z["__meta__"]))
    return out, meta
