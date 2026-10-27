// lta_cuda.cu -- A100 backend of the ethane/LTA histogram-ABF (+ uniform FR) engine (NSCC A100 study).
//
// Same algorithm as the validated CPU engines (src/lta_ladder_numba.py _run_chunk with diagnostics off, native FR
// mode 0; its walker-parallel form src/parallel/lta_parallel.py in counter mode), on one GPU:
//
//  * Walker i is handled by a GROUP of G threads (G a power of two); a block holds WB groups.  The group splits the
//    384 framework O atoms (shared memory) over its threads for both beads (768 shifted-LJ pair terms, minimum image
//    d - L rint(d / L), cutoff), then reduces the 6 force components (shuffle tree, then shared memory for G > 32);
//    the group leader adds the harmonic bond, computes the CV phi = wrap(2 pi x_COM / a), the clipped local mean
//    force and the bin.
//  * Per evaluated step k: forces at the pre-move positions; DEPOSIT of the clipped local force on the fixed-point grid
//    2^-fscale_log2 with 64-bit integer atomics (per block, then a triple-buffered global staging table and one grid
//    barrier); every block adds the exact integer bin sums of the step to its copy of the float64 all-steps and (k >=
//    burn-in) production accumulators; bias = ramp clip(M_j / max(C_j, 1), +-60) pi / a; save (host) at save steps
//    (state at the start of step k, accumulators incl. step k); Euler-Maruyama move with counter-mode Philox4x64-10
//    noise, counter (k, slot, word 0|1, 0), Box-Muller, the first 6 of 8 normals in [bead][xyz] order.
//  * FR at nxt = k + 1 >= fr_start, (nxt - fr_start) % fr_every == 0, on the post-move CV: grid barrier, then every
//    block computes the same selection redundantly: occupied-bin counts smoothed by the wrapped-Gaussian matrix Kk
//    (per output bin, ascending occupied source bins: numba's order), clamp / normalise / KL, periodic linear
//    interpolation, recentre, 3 x (clip, recentre); fr_select_native with the opportunity's block of N + 2 cap
//    uniforms (PCG64 child 2 of SeedSequence([seed, N, round(T)]), the validated stream): death probabilities,
//    candidates in slot order, partial Fisher-Yates to the cap, inverse-CDF sources over S-; whole-molecule copies
//    (sources never die, so the copies are race-free); the block owning a destination copies it.
//
// Deviations from the CPU engine (ulp-level, documented in docs/nscc_a100/VALIDATION.md): CUDA libm; the LJ sum
// order (thread split + tree; the CPU sum is a vectorised 'reassoc' reduction); the per-step deposit sum is exact over
// values rounded to 2^-43 (|error| <= 5.7e-14 per deposit, the ulp of the 480 clip); the means over walkers, the
// mass / KL sums and the S- prefix sum of FR are deterministic tree sums instead of sequential sums.
#include "gpu_common.cuh"
#include <future>
#include <thread>
#include <vector>
#include <algorithm>

#define LT_NO_MAX 512
#define LT_NG_MAX 256
#define LT_EPS 1.0e-12
#define LT_PI 3.141592653589793
#define LT_TWO_PI (2.0 * 3.141592653589793)
#define LT_NTICK 16

struct LtPar {
    int N, ng, nO, warm, burn, fr_start, fr_every, cap, is_fr, G, WB, timers;
    long long n_steps;
    double dt, ns, L, invL, rc2, s2, c24, kb, r0, cphi, cf, gpx, dphi, fclip, dclip, rate, dteff, sclip, bscale,
        grid0, qv, lq, fscale, inv_fscale;
    unsigned long long k0, k1;
};

struct LtDev {
    double *q;          // (N, 2, 3)
    double *F;          // (N, 2, 3) forces of the current state (kept across calls)
    int *jb;            // bin of the current state
    double *fl;         // clipped local force of the current state
    double *phin;       // post-move CV (FR exchange)
    int *jn;            // post-move bin (FR exchange)
    long long *anc;     // ancestor labels
    const double *Kk;   // (ng, ng)
    const double *O;    // (3, nO) framework: Ox, Oy, Oz
    // save block
    double *acc;        // (4, ng): FS, CS, FSP, CSP
    long long *ctr;     // repl, nopp, noppev, nscore, -, -, -, -
    double *fst;        // sstd sum, smax
    long long *evh;     // (cap + 1)
    int *err;
    double *bdh;        // (2, ng): birth hist, death hist
    long long *stM, *stC;
    long long *tick;
};

// ------------------------------------------------------------------------------------------------ physics
__device__ __forceinline__ void lt_pair(double x, double y, double z, double ox, double oy, double oz, double L,
                                        double invL, double rc2, double s2, double c24, double &fx, double &fy,
                                        double &fz)
{
    double dx = x - ox;
    dx = dx - L * rint(dx * invL);
    double dy = y - oy;
    dy = dy - L * rint(dy * invL);
    double dz = z - oz;
    dz = dz - L * rint(dz * invL);
    const double r2 = (dx * dx + dy * dy) + dz * dz;
    if (r2 < rc2) {
        const double inv = 1.0 / (r2 > LT_EPS ? r2 : LT_EPS);
        const double s = s2 * inv;
        const double sr6 = (s * s) * s;
        const double coef = (c24 * ((2.0 * sr6) * sr6 - sr6)) * inv;
        fx += coef * dx;
        fy += coef * dy;
        fz += coef * dz;
    }
}

__device__ __forceinline__ double lt_wrap(double p)
{
    const double t = p + LT_PI;
    double m = fmod(t, LT_TWO_PI);
    if (m != 0.0 && m < 0.0)
        m += LT_TWO_PI;
    return m - LT_PI;
}

__device__ __forceinline__ int lt_bin(double ph, double dphi, int ng)
{
    // ph is wrapped to [-pi, pi): floor(...) is in [0, ng] and fits an int (32-bit modulo, not the slow 64-bit one)
    int j = (int)floor((ph + LT_PI) / dphi);
    j %= ng;
    if (j < 0)
        j += ng;
    return j;
}

__device__ __forceinline__ int lt_pymod(int a, int n)
{
    int r = a % n;
    return r < 0 ? r + n : r;
}

// the 6 counter-mode normals of walker-step (step, slot): words 0 and 1, Box-Muller pairs, [bead][xyz]
__device__ __forceinline__ void lt_noise6(unsigned long long step, unsigned long long slot, unsigned long long k0,
                                          unsigned long long k1, double *z)
{
    unsigned long long o0[4], o1[4];
    philox4x64_10(step, slot, 0ULL, 0ULL, k0, k1, o0);
    philox4x64_10(step, slot, 1ULL, 0ULL, k0, k1, o1);
    box_muller(u01(o0[0]), u01(o0[1]), z[0], z[1]);
    box_muller(u01(o0[2]), u01(o0[3]), z[2], z[3]);
    box_muller(u01(o1[0]), u01(o1[1]), z[4], z[5]);
}

// one Box-Muller pair p (0, 1, 2) of the walker-step's 6 normals: z[2p], z[2p + 1] (words 0, 0, 1; uniforms (0,1), (2,3),
// (0,1)) -- the same numbers as lt_noise6, computed by three lanes in parallel
__device__ __forceinline__ void lt_noise_pair(unsigned long long step, unsigned long long slot, int p,
                                              unsigned long long k0, unsigned long long k1, double *z)
{
    unsigned long long o[4];
    philox4x64_10(step, slot, p == 2 ? 1ULL : 0ULL, 0ULL, k0, k1, o);
    const int a = (p == 1) ? 2 : 0;
    box_muller(u01(o[a]), u01(o[a + 1]), z[2 * p], z[2 * p + 1]);
}

// ------------------------------------------------------------------------------------------------ smem layout
struct LtLayout {
    size_t O, q, F, acc, sM, sC, red, jbw, flw, z, fr, bytes;
    size_t f_ph, f_jn, f_cnt, f_pk, f_sc, f_u, f_cdf, f_cand, f_ssel, f_scan, f_md, f_mi, f_oj, f_oc;
};

__host__ __device__ inline size_t lt_al8(size_t b) { return (b + 7) & ~(size_t)7; }

__host__ __device__ inline LtLayout lt_layout(int N, int ng, int nO, int WB, int G, int cap, int is_fr)
{
    LtLayout l;
    size_t o = 0;
    l.O = o;    o = lt_al8(o + sizeof(double) * 3 * nO);
    l.q = o;    o = lt_al8(o + sizeof(double) * 6 * WB);
    l.F = o;    o = lt_al8(o + sizeof(double) * 6 * WB);
    l.acc = o;  o = lt_al8(o + sizeof(double) * 4 * ng);
    l.sM = o;   o = lt_al8(o + sizeof(unsigned) * 4 * ng);    // 4 limbs per bin
    l.sC = o;   o = lt_al8(o + sizeof(unsigned) * ng);
    const int nwarp_g = G > 32 ? G / 32 : 1;
    l.red = o;  o = lt_al8(o + sizeof(double) * (6 * WB * nwarp_g + 64));
    l.jbw = o;  o = lt_al8(o + sizeof(int) * WB);
    l.flw = o;  o = lt_al8(o + sizeof(double) * WB);
    l.z = o;    o = lt_al8(o + sizeof(double) * 6 * WB);
    l.fr = o;
    if (is_fr) {
        const int Lb = N + 2 * cap;
        l.f_ph = o;   o = lt_al8(o + sizeof(double) * N);
        l.f_jn = o;   o = lt_al8(o + sizeof(int) * N);
        l.f_cnt = o;  o = lt_al8(o + sizeof(int) * ng);
        l.f_pk = o;   o = lt_al8(o + sizeof(double) * ng);
        l.f_sc = o;   o = lt_al8(o + sizeof(double) * N);
        l.f_u = o;    o = lt_al8(o + sizeof(double) * Lb);
        l.f_cdf = o;  o = lt_al8(o + sizeof(double) * N);
        l.f_cand = o; o = lt_al8(o + sizeof(int) * N);
        l.f_ssel = o; o = lt_al8(o + sizeof(int) * (cap + 1));
        l.f_scan = o; o = lt_al8(o + sizeof(int) * 64);
        l.f_md = o;   o = lt_al8(o + sizeof(double) * 128);
        l.f_mi = o;   o = lt_al8(o + sizeof(int) * 16);
        l.f_oj = o;   o = lt_al8(o + sizeof(int) * ng);
        l.f_oc = o;   o = lt_al8(o + sizeof(double) * ng);
    } else {
        l.f_ph = l.f_jn = l.f_cnt = l.f_pk = l.f_sc = l.f_u = l.f_cdf = l.f_cand = l.f_ssel = l.f_scan = l.f_md =
            l.f_mi = l.f_oj = l.f_oc = o;
    }
    l.bytes = o;
    return l;
}

// group reduction of 6 doubles over the G threads of a group (G power of two); the leader (gt == 0) gets the sums.
// Fully unrolled (compile-time offsets, the accumulators stay in registers); for G < 32 the shuffles stay inside the
// G-lane segment (offsets >= G are skipped), for G > 32 the warp totals are added through shared memory in warp order.
__device__ __forceinline__ void lt_group_reduce6(double v[6], int G, int gt, int wslot, double *red)
{
    const int width = G < 32 ? G : 32;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
#pragma unroll
        for (int c = 0; c < 6; c++) {
            const double t = __shfl_down_sync(0xffffffffu, v[c], o, width);
            if (o < width)
                v[c] += t;
        }
    }
    if (G > 32) {
        const int nw = G / 32, wg = gt >> 5, lane = gt & 31;
        if (lane == 0) {
#pragma unroll
            for (int c = 0; c < 6; c++)
                red[(wslot * nw + wg) * 6 + c] = v[c];
        }
        __syncthreads();
        if (gt == 0) {
#pragma unroll
            for (int c = 0; c < 6; c++) {
                double sacc = red[(wslot * nw) * 6 + c];
                for (int w = 1; w < nw; w++)
                    sacc += red[(wslot * nw + w) * 6 + c];
                v[c] = sacc;
            }
        }
        __syncthreads();
    }
}

// block-wide inclusive prefix sum of x[0..n) in place (contiguous chunks per thread; deterministic); sh >= 64 doubles
__device__ void lt_block_prefix(double *x, int n, double *sh)
{
    const int tid = threadIdx.x, nt = blockDim.x;
    const int E = (n + nt - 1) / nt;
    const int c0 = min(n, tid * E), c1 = min(n, c0 + E);
    double s = 0.0;
    for (int i = c0; i < c1; i++)
        s += x[i];
    // exclusive scan of the chunk sums: warp shuffle tree, then warp totals by warp 0
    const int lane = tid & 31, warp = tid >> 5, nw = nt >> 5;
    double incl = s;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const double y = __shfl_up_sync(0xffffffffu, incl, o);
        if (lane >= o)
            incl += y;
    }
    if (lane == 31)
        sh[warp] = incl;
    __syncthreads();
    if (warp == 0) {
        double w = (lane < nw) ? sh[lane] : 0.0;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const double y = __shfl_up_sync(0xffffffffu, w, o);
            if (lane >= o)
                w += y;
        }
        sh[32 + lane] = w;
    }
    __syncthreads();
    double run = (warp > 0 ? sh[32 + warp - 1] : 0.0) + (incl - s);
    for (int i = c0; i < c1; i++) {
        run += x[i];
        x[i] = run;
    }
    __syncthreads();
}

// ------------------------------------------------------------------------------------------------ FR (one block)
// Every thread participates.  Reads the post-move CV / bins of all N walkers and the opportunity's uniform block.
// Returns n (the replacement count); dsel = cand[0..n), ssel[0..n).  Writes score stats into md[8], md[9].
// Scratch: md + 32 (96 doubles: block_sum needs 32, lt_block_prefix 64).
__device__ __noinline__ int lt_fr_block(const LtPar &P, const LtDev &D, const double *__restrict__ ug, unsigned char *smem,
                           const LtLayout &lay)
{
    const int N = P.N, ng = P.ng, cap = P.cap, tid = threadIdx.x, nt = blockDim.x;
    double *ph = (double *)(smem + lay.f_ph);
    int *jn = (int *)(smem + lay.f_jn);
    int *cnt = (int *)(smem + lay.f_cnt);
    double *pk = (double *)(smem + lay.f_pk);
    double *sc = (double *)(smem + lay.f_sc);
    double *u = (double *)(smem + lay.f_u);
    double *cdf = (double *)(smem + lay.f_cdf);
    int *cand = (int *)(smem + lay.f_cand);
    int *ssel = (int *)(smem + lay.f_ssel);
    int *scan = (int *)(smem + lay.f_scan);
    double *md = (double *)(smem + lay.f_md);
    int *mi = (int *)(smem + lay.f_mi);
    const int Lb = N + 2 * cap;
    for (int k = tid; k < ng; k += nt)
        cnt[k] = 0;
    for (int i = tid; i < N; i += nt) {
        ph[i] = D.phin[i];
        jn[i] = D.jn[i];
    }
    for (int k = tid; k < Lb; k += nt)
        u[k] = ug[k];
    __syncthreads();
    for (int i = tid; i < N; i += nt)
        atomicAdd(&cnt[jn[i]], 1);
    __syncthreads();
    // smoothed counts: per output bin, ascending occupied source bins (numba's order), clamp at 0.  The occupied
    // bins are compacted first (ascending), and the matrix entries are fetched 8 at a time ahead of the in-order
    // accumulation (the L2 latency is paid once per 8 terms; the sum and its order are unchanged)
    int *oj = (int *)(smem + lay.f_oj);
    double *oc = (double *)(smem + lay.f_oc);
    int nocc;
    {
        const int Eg = (ng + nt - 1) / nt;
        const int g0 = min(ng, tid * Eg), g1 = min(ng, g0 + Eg);
        int c_t = 0;
        for (int j = g0; j < g1; j++)
            c_t += (cnt[j] != 0);
        int off = block_excl_scan(c_t, scan, &nocc);
        for (int j = g0; j < g1; j++)
            if (cnt[j] != 0) {
                oj[off] = j;
                oc[off] = (double)cnt[j];
                off++;
            }
        __syncthreads();
    }
    double vloc = 0.0;
    for (int k = tid; k < ng; k += nt) {
        double v = 0.0;
        int t = 0;
        for (; t + 8 <= nocc; t += 8) {
            double kv[8];
#pragma unroll
            for (int u = 0; u < 8; u++)
                kv[u] = __ldg(&D.Kk[(size_t)oj[t + u] * ng + k]);
#pragma unroll
            for (int u = 0; u < 8; u++)
                v += oc[t + u] * kv[u];
        }
        for (; t < nocc; t++)
            v += oc[t] * __ldg(&D.Kk[(size_t)oj[t] * ng + k]);
        if (v < 0.0)
            v = 0.0;
        pk[k] = v;
        vloc += v;
    }
    double *red = md + 32;      // 96 doubles of scratch (md has 128; md[8], md[9] hold the score stats)
    const double tot = block_sum(vloc, red);
    double mass = tot * P.dphi;
    if (mass < LT_EPS)
        mass = LT_EPS;
    double kloc = 0.0;
    for (int k = tid; k < ng; k += nt) {
        const double v = pk[k] / mass;
        pk[k] = v;
        kloc += v * (log(v > LT_EPS ? v : LT_EPS) - P.lq);
    }
    const double kl = block_sum(kloc, red) * P.dphi;     // (block_sum syncs: pk complete)
    // raw scores and the first recentring
    double sloc = 0.0;
    for (int i = tid; i < N; i += nt) {
        const double xx = (ph[i] - P.grid0) / P.dphi;
        const double f0 = floor(xx);
        const double fr = xx - f0;
        const int i0 = (int)f0;
        const double p0 = pk[lt_pymod(i0, ng)];
        const double p1 = pk[lt_pymod(i0 + 1, ng)];
        const double pat = (1.0 - fr) * p0 + fr * p1;
        const double qat = (1.0 - fr) * P.qv + fr * P.qv;
        const double r = (log(pat > LT_EPS ? pat : LT_EPS) - log(qat > LT_EPS ? qat : LT_EPS)) - kl;
        sc[i] = r;
        sloc += r;
    }
    double m = block_sum(sloc, red) / N;
    for (int i = tid; i < N; i += nt)
        sc[i] = sc[i] - m;
    __syncthreads();
    for (int it = 0; it < 3; it++) {
        double s2 = 0.0;
        for (int i = tid; i < N; i += nt) {
            double v = sc[i];
            if (v < -P.sclip)
                v = -P.sclip;
            else if (v > P.sclip)
                v = P.sclip;
            sc[i] = v;
            s2 += v;
        }
        m = block_sum(s2, red) / N;
        for (int i = tid; i < N; i += nt)
            sc[i] = sc[i] - m;
        __syncthreads();
    }
    // diagnostics: std (ddof 0) and max |s|
    double a1 = 0.0, amax = 0.0;
    for (int i = tid; i < N; i += nt) {
        a1 += sc[i];
        amax = fmax(amax, fabs(sc[i]));
    }
    const double mm = block_sum(a1, red) / N;
    const double am = block_max(amax, red);
    double v2 = 0.0;
    for (int i = tid; i < N; i += nt) {
        const double d = sc[i] - mm;
        v2 += d * d;
    }
    const double var = block_sum(v2, red) / N;
    // fr_select_native: death probabilities, S- weights, guard
    const int E = (N + nt - 1) / nt;
    const int c0 = min(N, tid * E), c1 = min(N, c0 + E);
    double sb = 0.0, sd = 0.0;
    int nc_t = 0;
    for (int i = c0; i < c1; i++) {
        const double s = sc[i];
        const double dw = s > 0.0 ? s : 0.0;
        const double bw = -s > 0.0 ? -s : 0.0;
        const double dp = dw > 0.0 ? 1.0 - exp(((-P.rate) * dw) * P.dteff) : 0.0;
        cdf[i] = bw;
        sb += bw;
        sd += dw;
        nc_t += (u[i] < dp);
    }
    const double sbw = block_sum(sb, red), sdw = block_sum(sd, red);
    if (tid == 0) {
        md[8] = sqrt(var);
        md[9] = am;
    }
    if (sbw <= LT_EPS || sdw <= LT_EPS) {
        __syncthreads();
        return 0;
    }
    int n;
    int off = block_excl_scan(nc_t, scan, &n);
    for (int i = c0; i < c1; i++) {
        const double s = sc[i];
        const double dw = s > 0.0 ? s : 0.0;
        const double dp = dw > 0.0 ? 1.0 - exp(((-P.rate) * dw) * P.dteff) : 0.0;
        if (u[i] < dp)
            cand[off++] = i;
    }
    __syncthreads();
    if (n == 0)
        return 0;
    if (n > cap) {
        if (tid == 0) {
            for (int t = 0; t < cap; t++) {
                const int mm2 = n - t;
                long long jj = (long long)(u[N + t] * (double)mm2);
                if (jj > mm2 - 1)
                    jj = mm2 - 1;
                jj += t;
                const int tmp = cand[t]; cand[t] = cand[jj]; cand[jj] = tmp;
            }
        }
        n = cap;
        __syncthreads();
    }
    lt_block_prefix(cdf, N, red);
    const double totc = cdf[N - 1];
    for (int t = tid; t < n; t += nt) {
        const double x = u[N + cap + t] * totc;
        int lo = 0, hi = N - 1;
        while (lo < hi) {
            const int mid = (lo + hi) >> 1;
            if (cdf[mid] > x)
                hi = mid;
            else
                lo = mid + 1;
        }
        ssel[t] = lo;
    }
    __syncthreads();
    (void)mi;
    return n;
}

// ------------------------------------------------------------------------------------------------ the step kernel
// One call: [bias + move + FR of step a] if a_move, then for k = a+1 .. b: [eval + deposit of k], and for k < b
// [bias + move + FR of k].  A call with a_move == 0 evaluates and deposits step a == b only (the run's prologue).
__global__ void __launch_bounds__(512) lt_step_kernel(LtPar P, LtDev D, long long a, long long b, int a_move,
                                                       const double *__restrict__ ubuf, long long opp_base)
{
    cg::grid_group grid = cg::this_grid();
    extern __shared__ __align__(16) unsigned char smem[];
    const int N = P.N, ng = P.ng, nO = P.nO, G = P.G, WB = P.WB;
    const int tid = threadIdx.x, nt = blockDim.x, B = gridDim.x;
    const LtLayout lay = lt_layout(N, ng, nO, WB, G, P.cap, P.is_fr);
    double *Os = (double *)(smem + lay.O);
    double *qs = (double *)(smem + lay.q);
    double *Fs = (double *)(smem + lay.F);
    double *acc = (double *)(smem + lay.acc);
    unsigned *sL = (unsigned *)(smem + lay.sM);
    unsigned *sCn = (unsigned *)(smem + lay.sC);
    double *red = (double *)(smem + lay.red);
    int *jbw = (int *)(smem + lay.jbw);
    double *flw = (double *)(smem + lay.flw);
    double *zs = (double *)(smem + lay.z);
    const int g = tid / G, gt = tid % G;              // group (walker slot in block) and thread in group
    const int wi = blockIdx.x * WB + g;               // walker index
    const bool act = (g < WB) && (wi < N);
    const bool lead = act && gt == 0;
    // the noise of a move is produced by a dedicated NOISE WARP (the block's last warp) during the force
    // evaluation, so it is ready before the leader's bias read / move and never diverges a leader's warp; task
    // t = lane + 32 i is the Box-Muller pair t % 3 of walker slot t / 3 (three lanes per walker)
    const int zl = tid - G * WB;                                   // lane in the noise warp (or < 0)
    auto noise_tasks = [&](long long step) {
        for (int t = zl; t < 3 * WB; t += 32) {
            const int w = t / 3, wn = blockIdx.x * WB + w;
            if (wn < N)
                lt_noise_pair((unsigned long long)step, (unsigned long long)wn, t % 3, P.k0, P.k1, zs + w * 6);
        }
    };
    for (int t = tid; t < 3 * nO; t += nt)
        Os[t] = D.O[t];
    for (int t = tid; t < 4 * ng; t += nt)
        acc[t] = D.acc[t];
    for (int t = tid; t < ng; t += nt) {
        sL[4 * t] = sL[4 * t + 1] = sL[4 * t + 2] = sL[4 * t + 3] = 0u;
        sCn[t] = 0u;
    }
    if (lead) {
        for (int c = 0; c < 6; c++) {
            qs[g * 6 + c] = D.q[(size_t)wi * 6 + c];
            Fs[g * 6 + c] = D.F[(size_t)wi * 6 + c];
        }
        jbw[g] = D.jb[wi];
        flw[g] = D.fl[wi];
    }
    int bad = 0;
    const bool tim = P.timers && blockIdx.x == 0 && tid == 0;
    long long c_force = 0, c_dep = 0, c_sync = 0, c_move = 0, c_fr = 0, ta = 0, tb = 0;
    long long c_pair = 0, c_red = 0, c_noise = 0, tc = 0;     // split of c_force (block 0, thread 0 = leader 0)
    const long long c_start = tim ? clock64() : 0;
    if (a_move && zl >= 0)
        noise_tasks(a);
    __syncthreads();
    const double *Ox = Os, *Oy = Os + nO, *Oz = Os + 2 * nO;
    for (long long k = a; k <= b; k++) {
        if (k > a || !a_move) {
            // ---------------- evaluation of step k at the current positions
            if (tim)
                ta = clock64();
            double v[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
            if (zl >= 0 && k < b)
                noise_tasks(k);
            if (act) {
                const double x0 = qs[g * 6 + 0], y0 = qs[g * 6 + 1], z0 = qs[g * 6 + 2];
                const double x1 = qs[g * 6 + 3], y1 = qs[g * 6 + 4], z1 = qs[g * 6 + 5];
                const double pL = P.L, pinvL = P.invL, prc2 = P.rc2, ps2 = P.s2, pc24 = P.c24;
                for (int o = gt; o < nO; o += G) {
                    const double ox = Ox[o], oy = Oy[o], oz = Oz[o];
                    lt_pair(x0, y0, z0, ox, oy, oz, pL, pinvL, prc2, ps2, pc24, v[0], v[1], v[2]);
                    lt_pair(x1, y1, z1, ox, oy, oz, pL, pinvL, prc2, ps2, pc24, v[3], v[4], v[5]);
                }
            }
            if (tim) { tc = clock64(); c_pair += tc - ta; }
            lt_group_reduce6(v, G, gt, g, red);
            if (tim) { const long long t2 = clock64(); c_red += t2 - tc; tc = t2; }
            if (lead) {
                const double *q = qs + g * 6;
                const double dx = q[0] - q[3], dy = q[1] - q[4], dz = q[2] - q[5];
                double r = sqrt((dx * dx + dy * dy) + dz * dz);
                if (r < LT_EPS)
                    r = LT_EPS;
                const double c = (-P.kb) * (r - P.r0);
                const double fbx = (c * dx) / r, fby = (c * dy) / r, fbz = (c * dz) / r;
                double *F = Fs + g * 6;
                F[0] = fbx + v[0];
                F[1] = fby + v[1];
                F[2] = fbz + v[2];
                F[3] = (-fbx) + v[3];
                F[4] = (-fby) + v[4];
                F[5] = (-fbz) + v[5];
                const double x = (q[0] + q[3]) * 0.5;
                const double phv = lt_wrap(P.cphi * x);
                double f = P.cf * (F[0] + F[3]);
                if (f < -P.dclip)
                    f = -P.dclip;
                else if (f > P.dclip)
                    f = P.dclip;
                const int j = lt_bin(phv, P.dphi, ng);
                jbw[g] = j;
                flw[g] = f;
                if (!(fabs(f) <= P.dclip))
                    bad |= 1;
            }
            if (lead)
                limb_add(sL, sCn, jbw[g], (unsigned long long)__double2ll_rn(flw[g] * P.fscale), 1u);
            if (tim) { tb = clock64(); c_force += tb - ta; ta = tb; }
            __syncthreads();
            // ---------------- deposit: exact integer bin sums of the step into every block's accumulators
            const bool prod = k >= P.burn;
            if (B == 1) {
                for (int t = tid; t < ng; t += nt) {
                    const unsigned cc = sCn[t];
                    if (cc) {
                        const double tf = (double)limb_total(sL, t) * P.inv_fscale, tc = (double)cc;
                        acc[t] += tf;
                        acc[ng + t] += tc;
                        if (prod) {
                            acc[2 * ng + t] += tf;
                            acc[3 * ng + t] += tc;
                        }
                        sL[4 * t] = sL[4 * t + 1] = sL[4 * t + 2] = sL[4 * t + 3] = 0u;
                        sCn[t] = 0u;
                    }
                }
                if (tim) { tb = clock64(); c_dep += tb - ta; ta = tb; }
            } else {
                const int s3 = (int)(k % 3);
                long long *gM = D.stM + (size_t)s3 * ng, *gC = D.stC + (size_t)s3 * ng;
                for (int t = tid; t < ng; t += nt) {
                    const unsigned cc = sCn[t];
                    if (cc) {
                        atomicAdd((unsigned long long *)&gM[t], (unsigned long long)limb_total(sL, t));
                        atomicAdd((unsigned long long *)&gC[t], (unsigned long long)cc);
                        sL[4 * t] = sL[4 * t + 1] = sL[4 * t + 2] = sL[4 * t + 3] = 0u;
                        sCn[t] = 0u;
                    }
                }
                if (tim) { tb = clock64(); c_dep += tb - ta; ta = tb; }
                grid.sync();
                if (tim) { tb = clock64(); c_sync += tb - ta; ta = tb; }
                for (int t = tid; t < ng; t += nt) {
                    const long long cc = *((volatile long long *)&gC[t]);       // both loads issued together
                    const long long mm = *((volatile long long *)&gM[t]);
                    if (cc) {
                        const double tf = (double)mm * P.inv_fscale, tc = (double)cc;
                        acc[t] += tf;
                        acc[ng + t] += tc;
                        if (prod) {
                            acc[2 * ng + t] += tf;
                            acc[3 * ng + t] += tc;
                        }
                    }
                }
                if (blockIdx.x == 0) {
                    const int z3 = (int)((k + 2) % 3);
                    for (int t = tid; t < ng; t += nt) {
                        D.stM[(size_t)z3 * ng + t] = 0;
                        D.stC[(size_t)z3 * ng + t] = 0;
                    }
                }
                if (tim) { tb = clock64(); c_dep += tb - ta; ta = tb; }
            }
            __syncthreads();
        }
        if (k == b)
            break;                                  // save point (host) / end of the call
        // ---------------- bias read + move of step k
        if (tim)
            ta = clock64();
        const long long nxt = k + 1;
        const bool fr_now = P.is_fr && nxt >= P.fr_start && ((nxt - P.fr_start) % P.fr_every) == 0;
        if (lead) {
            double ramp = (double)k / (double)(P.warm > 1 ? P.warm : 1);
            if (ramp > 1.0)
                ramp = 1.0;
            const double scl = P.bscale * ramp;
            const int j = jbw[g];
            const double c = acc[ng + j];
            double Gm = 0.0;
            if (c > 0.0)
                Gm = acc[j] / (c > 1.0 ? c : 1.0);
            if (Gm < -P.fclip)
                Gm = -P.fclip;
            else if (Gm > P.fclip)
                Gm = P.fclip;
            const double bxi = (scl * Gm) * P.gpx;
            const double *zz = zs + g * 6;
            const double z0 = zz[0], z1 = zz[1], z2 = zz[2], z3 = zz[3], z4 = zz[4], z5 = zz[5];
            double *q = qs + g * 6;
            const double *F = Fs + g * 6;
            q[0] = (q[0] + P.dt * (F[0] + bxi)) + P.ns * z0;
            q[1] = (q[1] + P.dt * F[1]) + P.ns * z1;
            q[2] = (q[2] + P.dt * F[2]) + P.ns * z2;
            q[3] = (q[3] + P.dt * (F[3] + bxi)) + P.ns * z3;
            q[4] = (q[4] + P.dt * F[4]) + P.ns * z4;
            q[5] = (q[5] + P.dt * F[5]) + P.ns * z5;
            if (fr_now) {
                for (int cc = 0; cc < 6; cc++)
                    D.q[(size_t)wi * 6 + cc] = q[cc];
                const double x = (q[0] + q[3]) * 0.5;
                const double phv = lt_wrap(P.cphi * x);
                D.phin[wi] = phv;
                D.jn[wi] = lt_bin(phv, P.dphi, ng);
            }
        }
        if (tim) { tb = clock64(); c_move += tb - ta; ta = tb; }
        // the leader's new positions (qs) are read by every thread of its group in the next force evaluation: a
        // group of G > 32 threads spans several warps, so a block barrier is required here
        __syncthreads();
        if (fr_now) {
            __threadfence();
            grid_barrier(grid);
            const long long iopp = (nxt - P.fr_start) / P.fr_every - opp_base;
            const double *ug = ubuf + iopp * (long long)(N + 2 * P.cap);
            const int n = lt_fr_block(P, D, ug, smem, lay);
            const int *cand = (const int *)(smem + lay.f_cand);
            const int *ssel = (const int *)(smem + lay.f_ssel);
            // copies into this block's destinations (sources never die: no ordering issue)
            for (int t = 0; t < n; t++) {
                const int d = cand[t], s = ssel[t];
                if (d >= blockIdx.x * WB && d < blockIdx.x * WB + WB) {
                    const int gd = d - blockIdx.x * WB;
                    if (tid < 6)
                        qs[gd * 6 + tid] = D.q[(size_t)s * 6 + tid];
                    if (tid == 0)
                        D.anc[d] = D.anc[s];
                }
            }
            if (blockIdx.x == 0 && tid == 0) {
                const double *md = (const double *)(smem + lay.f_md);
                D.ctr[1] += 1;          // nopp
                D.ctr[3] += 1;          // nscore
                D.fst[0] += md[8];
                if (md[9] > D.fst[1])
                    D.fst[1] = md[9];
                D.evh[n] += 1;
                if (n > 0) {
                    D.ctr[2] += 1;
                    D.ctr[0] += n;
                }
                const int *jnS = (const int *)(smem + lay.f_jn);
                for (int t = 0; t < n; t++) {
                    D.bdh[jnS[ssel[t]]] += 1.0;
                    D.bdh[ng + jnS[cand[t]]] += 1.0;
                }
            }
            __syncthreads();
            if (tim) { tb = clock64(); c_fr += tb - ta; ta = tb; }
        }
    }
    // exit at the save point / end: state at the start of step b, its forces, bin and local force
    grid_barrier(grid);
    if (lead) {
        for (int c = 0; c < 6; c++) {
            D.q[(size_t)wi * 6 + c] = qs[g * 6 + c];
            D.F[(size_t)wi * 6 + c] = Fs[g * 6 + c];
        }
        D.jb[wi] = jbw[g];
        D.fl[wi] = flw[g];
    }
    if (blockIdx.x == 0)
        for (int t = tid; t < 4 * ng; t += nt)
            D.acc[t] = acc[t];
    if (bad)
        atomicOr(D.err, bad);
    if (tim) {
        D.tick[0] += clock64() - c_start;
        D.tick[1] += c_force;
        D.tick[2] += c_dep;
        D.tick[3] += c_sync;
        D.tick[4] += c_move;
        D.tick[5] += c_fr;
        D.tick[6] += 1;
        D.tick[7] += c_pair;
        D.tick[8] += c_red;
        D.tick[9] += c_noise;
    }
}

// FR test kernel (one block): the selection of lt_fr_block on given post-move CVs / bins and a uniform block
__global__ void lt_fr_test_kernel(LtPar P, LtDev D, const double *ug, int *out, double *dout)
{
    extern __shared__ __align__(16) unsigned char smem[];
    const LtLayout lay = lt_layout(P.N, P.ng, 0, 0, 1, P.cap, 1);
    const int n = lt_fr_block(P, D, ug, smem, lay);
    const int *cand = (const int *)(smem + lay.f_cand);
    const int *ssel = (const int *)(smem + lay.f_ssel);
    const double *sc = (const double *)(smem + lay.f_sc);
    const double *md = (const double *)(smem + lay.f_md);
    if (threadIdx.x == 0)
        out[0] = n;
    for (int t = threadIdx.x; t < n; t += blockDim.x) {
        out[1 + t] = cand[t];
        out[1 + P.cap + t] = ssel[t];
    }
    for (int i = threadIdx.x; i < P.N; i += blockDim.x)
        dout[i] = sc[i];
    if (threadIdx.x == 0) {
        dout[P.N] = md[8];
        dout[P.N + 1] = md[9];
    }
}

// force test kernel: forces / CV / local force / bin of given positions (one group of G threads per walker)
__global__ void lt_force_test_kernel(LtPar P, const double *O, const double *q, double *F, double *aux)
{
    extern __shared__ __align__(16) unsigned char smem[];
    const int nO = P.nO, G = P.G;
    double *Os = (double *)smem;
    double *red = Os + 3 * nO;
    for (int t = threadIdx.x; t < 3 * nO; t += blockDim.x)
        Os[t] = O[t];
    __syncthreads();
    const int g = threadIdx.x / G, gt = threadIdx.x % G;
    const int wi = blockIdx.x * (blockDim.x / G) + g;
    const bool act = wi < P.N;
    double v[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    if (act) {
        const double *qq = q + (size_t)wi * 6;
        for (int o = gt; o < nO; o += G) {
            lt_pair(qq[0], qq[1], qq[2], Os[o], Os[nO + o], Os[2 * nO + o], P.L, P.invL, P.rc2, P.s2, P.c24, v[0],
                    v[1], v[2]);
            lt_pair(qq[3], qq[4], qq[5], Os[o], Os[nO + o], Os[2 * nO + o], P.L, P.invL, P.rc2, P.s2, P.c24, v[3],
                    v[4], v[5]);
        }
    }
    lt_group_reduce6(v, G, gt, g, red);
    if (act && gt == 0) {
        const double *qq = q + (size_t)wi * 6;
        const double dx = qq[0] - qq[3], dy = qq[1] - qq[4], dz = qq[2] - qq[5];
        double r = sqrt((dx * dx + dy * dy) + dz * dz);
        if (r < LT_EPS)
            r = LT_EPS;
        const double c = (-P.kb) * (r - P.r0);
        const double fbx = (c * dx) / r, fby = (c * dy) / r, fbz = (c * dz) / r;
        double *Fo = F + (size_t)wi * 6;
        Fo[0] = fbx + v[0];
        Fo[1] = fby + v[1];
        Fo[2] = fbz + v[2];
        Fo[3] = (-fbx) + v[3];
        Fo[4] = (-fby) + v[4];
        Fo[5] = (-fbz) + v[5];
        const double x = (qq[0] + qq[3]) * 0.5;
        const double phv = lt_wrap(P.cphi * x);
        double f = P.cf * (Fo[0] + Fo[3]);
        if (f < -P.dclip)
            f = -P.dclip;
        else if (f > P.dclip)
            f = P.dclip;
        aux[(size_t)wi * 3 + 0] = phv;
        aux[(size_t)wi * 3 + 1] = f;
        aux[(size_t)wi * 3 + 2] = (double)lt_bin(phv, P.dphi, P.ng);
    }
}

__global__ void lt_normals_kernel(unsigned long long k0, unsigned long long k1, long long step0, long long n_steps,
                                  int N, double *out)
{
    const long long n = n_steps * N;
    for (long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x; t < n; t += (long long)gridDim.x * blockDim.x) {
        const long long s = t / N, i = t % N;
        unsigned long long o0[4], o1[4];
        philox4x64_10((unsigned long long)(step0 + s), (unsigned long long)i, 0ULL, 0ULL, k0, k1, o0);
        philox4x64_10((unsigned long long)(step0 + s), (unsigned long long)i, 1ULL, 0ULL, k0, k1, o1);
        double *z = out + t * 6;
        box_muller(u01(o0[0]), u01(o0[1]), z[0], z[1]);
        box_muller(u01(o0[2]), u01(o0[3]), z[2], z[3]);
        box_muller(u01(o1[0]), u01(o1[1]), z[4], z[5]);
    }
}

// ------------------------------------------------------------------------------------------------ host
// fp: lta_ladder_numba._params layout (F_DT .. F_LQ); ip: [N, ng, n_steps, warm, burn, fr_start, fr_every, cap, is_fr,
// G, WB, timers, chunk_opps, device, fscale_log2, nO]
static LtPar lt_make_par(const long long *ip, const double *fp, const unsigned long long *u64)
{
    LtPar P;
    memset(&P, 0, sizeof P);
    P.N = (int)ip[0];
    P.ng = (int)ip[1];
    P.n_steps = ip[2];
    P.warm = (int)ip[3];
    P.burn = (int)ip[4];
    P.fr_start = (int)ip[5];
    P.fr_every = (int)ip[6];
    P.cap = (int)ip[7];
    P.is_fr = (int)ip[8];
    P.G = (int)ip[9];
    P.WB = (int)ip[10];
    P.timers = (int)ip[11];
    P.nO = (int)ip[15];
    // numba _params indices: F_DT, F_NS, F_A, F_L, F_INVL, F_RC2, F_S2, F_C24, F_KB, F_R0, F_C4, F_VRC = 0..11;
    // F_CPHI, F_CF, F_GPX, F_DPHI, F_FCLIP, F_DCLIP, F_RATE, F_DTEFF, F_SCLIP = 12..20; F_WH, F_CM, F_BSCALE, F_GRID0,
    // F_QV, F_LQ = 21..26
    P.dt = fp[0];
    P.ns = fp[1];
    P.L = fp[3];
    P.invL = fp[4];
    P.rc2 = fp[5];
    P.s2 = fp[6];
    P.c24 = fp[7];
    P.kb = fp[8];
    P.r0 = fp[9];
    P.cphi = fp[12];
    P.cf = fp[13];
    P.gpx = fp[14];
    P.dphi = fp[15];
    P.fclip = fp[16];
    P.dclip = fp[17];
    P.rate = fp[18];
    P.dteff = fp[19];
    P.sclip = fp[20];
    P.bscale = fp[23];
    P.grid0 = fp[24];
    P.qv = fp[25];
    P.lq = fp[26];
    P.fscale = ldexp(1.0, (int)ip[14]);
    P.inv_fscale = ldexp(1.0, -(int)ip[14]);
    P.k0 = u64[0];
    P.k1 = u64[1];
    return P;
}

extern "C" {

double lt_gpu_now(void) { return host_now(); }

int lt_gpu_device_info(int dev, long long *out, char *name, int namelen)
{
    cudaDeviceProp pr;
    if (cudaGetDeviceProperties(&pr, dev) != cudaSuccess)
        return -1;
    out[0] = pr.multiProcessorCount;
    out[1] = pr.clockRate;
    out[2] = (long long)pr.sharedMemPerBlockOptin;
    out[3] = pr.major;
    out[4] = pr.minor;
    out[5] = pr.ECCEnabled;
    out[6] = (long long)(pr.totalGlobalMem >> 20);
    snprintf(name, (size_t)namelen, "%s", pr.name);
    return 0;
}

int lt_gpu_normals(const unsigned long long *key, long long step0, long long n_steps, int N, double *out,
                   char *errbuf, int errlen)
{
    try {
        DBuf<double> d;
        d.alloc((size_t)n_steps * N * 6);
        lt_normals_kernel<<<256, 256>>>(key[0], key[1], step0, n_steps, N, d.p);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(out, d.p, sizeof(double) * n_steps * N * 6, cudaMemcpyDeviceToHost));
    } catch (GpuError &e) {
        snprintf(errbuf, (size_t)errlen, "%s", e.msg);
        return -1;
    }
    return 0;
}

int lt_gpu_force_test(const long long *ip, const double *fp, const double *O, const double *q, double *F_out,
                      double *aux_out, char *errbuf, int errlen)
{
    try {
        unsigned long long z[2] = {0, 0};
        LtPar P = lt_make_par(ip, fp, z);
        const int G = P.G, per = 1024 / G;
        DBuf<double> dO, dq, dF, dA;
        dO.alloc((size_t)3 * P.nO);
        dq.alloc((size_t)6 * P.N);
        dF.alloc((size_t)6 * P.N);
        dA.alloc((size_t)3 * P.N);
        CUDA_CHECK(cudaMemcpy(dO.p, O, sizeof(double) * 3 * P.nO, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dq.p, q, sizeof(double) * 6 * P.N, cudaMemcpyHostToDevice));
        const size_t sm = sizeof(double) * (3 * P.nO + 6 * per * (G > 32 ? G / 32 : 1) + 64);
        CUDA_CHECK(cudaFuncSetAttribute(lt_force_test_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm));
        lt_force_test_kernel<<<(P.N + per - 1) / per, per * G, sm>>>(P, dO.p, dq.p, dF.p, dA.p);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(F_out, dF.p, sizeof(double) * 6 * P.N, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(aux_out, dA.p, sizeof(double) * 3 * P.N, cudaMemcpyDeviceToHost));
    } catch (GpuError &e) {
        snprintf(errbuf, (size_t)errlen, "%s", e.msg);
        return -1;
    }
    return 0;
}

int lt_gpu_fr_test(const long long *ip, const double *fp, const double *Kk, const double *phin, const int *jn,
                   const double *ublk, int threads, int *out, double *dout, char *errbuf, int errlen)
{
    try {
        unsigned long long z[2] = {0, 0};
        LtPar P = lt_make_par(ip, fp, z);
        DBuf<double> dK, dph, du;
        DBuf<int> djn, dout_i;
        DBuf<double> dout_d;
        dK.alloc((size_t)P.ng * P.ng);
        dph.alloc(P.N);
        djn.alloc(P.N);
        du.alloc((size_t)P.N + 2 * P.cap);
        dout_i.alloc((size_t)1 + 2 * P.cap);
        dout_d.alloc((size_t)P.N + 2);
        CUDA_CHECK(cudaMemcpy(dK.p, Kk, sizeof(double) * P.ng * P.ng, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dph.p, phin, sizeof(double) * P.N, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(djn.p, jn, sizeof(int) * P.N, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(du.p, ublk, sizeof(double) * (P.N + 2 * P.cap), cudaMemcpyHostToDevice));
        LtDev D;
        memset(&D, 0, sizeof D);
        D.Kk = dK.p;
        D.phin = dph.p;
        D.jn = djn.p;
        const LtLayout lay = lt_layout(P.N, P.ng, 0, 0, 1, P.cap, 1);
        const size_t sm = lay.bytes;
        CUDA_CHECK(cudaFuncSetAttribute(lt_fr_test_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm));
        lt_fr_test_kernel<<<1, threads, sm>>>(P, D, du.p, dout_i.p, dout_d.p);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(out, dout_i.p, sizeof(int) * (1 + 2 * P.cap), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(dout, dout_d.p, sizeof(double) * (P.N + 2), cudaMemcpyDeviceToHost));
    } catch (GpuError &e) {
        snprintf(errbuf, (size_t)errlen, "%s", e.msg);
        return -1;
    }
    return 0;
}

/*
 * lt_gpu_run: one complete run (the timed part of run_arm_gpu).
 *   ip int64[16] (lt_make_par), fp double[27] (lta_ladder_numba._params), u64: Philox k0, k1, PCG64 child-2 state
 *   hi, lo, inc hi, lo; O (3, nO); Kk (ng, ng); q0 (N, 2, 3); save_steps (S) in [0, n_steps]
 *   outputs: sv_acc (S, 4, ng); sv_i (S, 3): repl, nopp, noppev; sv_evh (S, cap + 1); sv_nfe (S) = N (k + 1);
 *   sv_wall (S); q_out (N, 6); anc_out (N); acc_out (4, ng); bdh_out (2, ng); evh_out (cap + 1);
 *   misc double[8]: [0] fst sstd sum, [1] fst smax, [2] nscore, [3] repl, [4] nopp, [5] noppev
 *   tim double[24]: [0] t0, [1] t_end, [2] setup s, [3] save s, [4] FR-uniform wait s, [6] calls, [7] chunks,
 *   [8] clock kHz, [9..15] cycles: total, force, deposit, barrier, move, FR, calls, [16] blocks, [17] threads,
 *   [18] max co-resident blocks, [19] dyn smem, [20] err
 */
int lt_gpu_run(const long long *ip, const double *fp, const unsigned long long *u64, const double *O,
               const double *Kk, const double *q0, const long long *save_steps, long long S,
               double *sv_acc, long long *sv_i, long long *sv_evh, long long *sv_nfe, double *sv_wall,
               double *q_out, long long *anc_out, double *acc_out, double *bdh_out, long long *evh_out,
               double *misc, double *tim, char *errbuf, int errlen)
{
    const double t_setup0 = host_now();
    try {
        LtPar P = lt_make_par(ip, fp, u64);
        const int N = P.N, ng = P.ng, nO = P.nO, cap = P.cap;
        const long long n_steps = P.n_steps;
        const long long chunk_opps = ip[12] > 0 ? ip[12] : 4096;
        const int dev = (int)ip[13];
        const int G = P.G, WB = P.WB, T = G * WB + 32;      // + the noise warp
        if (N < 1 || ng < 1 || ng > LT_NG_MAX || nO < 1 || nO > LT_NO_MAX || G < 1 || (G & (G - 1)) || T > 512 ||
            ((G * WB) & 31) || WB < 1 || WB > 32 || (P.is_fr && (cap < 1 || N < 2 || P.fr_every < 1)))
            gpu_fail("lt_gpu_run: bad request (G a power of two, G WB a multiple of 32, WB <= 32, G WB + 32 <= 512)");
        const int B = (N + WB - 1) / WB;
        CUDA_CHECK(cudaSetDevice(dev));
        cudaDeviceProp pr;
        CUDA_CHECK(cudaGetDeviceProperties(&pr, dev));
        const LtLayout lay = lt_layout(N, ng, nO, WB, G, cap, P.is_fr);
        const size_t smem = lay.bytes;
        if (smem > (size_t)pr.sharedMemPerBlockOptin)
            gpu_fail("lt_gpu_run: shared memory request exceeds the device limit");
        CUDA_CHECK(cudaFuncSetAttribute(lt_step_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
        int occ = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, lt_step_kernel, T, smem));
        const long long maxco = (long long)occ * pr.multiProcessorCount;
        if (B > maxco)
            gpu_fail("lt_gpu_run: grid exceeds the co-resident block limit of a cooperative launch");
        const int cap1 = cap + 1;
        const int Lb = N + 2 * cap;
        const size_t save_bytes = sizeof(double) * 4 * ng + sizeof(long long) * 8 + sizeof(double) * 2 +
                                  sizeof(long long) * cap1 + 16;
        DBuf<double> dq, dF, dfl, dph, dK, dO, dbdh, dU;
        DBuf<int> djb, djn;
        DBuf<long long> danc, dSTM, dSTC, dTick;
        DBuf<unsigned char> dSave;
        dq.alloc((size_t)6 * N);
        dF.alloc((size_t)6 * N);
        dfl.alloc(N);
        dph.alloc(N);
        djb.alloc(N);
        djn.alloc(N);
        danc.alloc(N);
        dK.alloc((size_t)ng * ng);
        dO.alloc((size_t)3 * nO);
        dbdh.alloc((size_t)2 * ng);
        dSTM.alloc((size_t)3 * ng);
        dSTC.alloc((size_t)3 * ng);
        dTick.alloc(LT_NTICK);
        dSave.alloc(save_bytes);
        const size_t ublk = P.is_fr ? (size_t)chunk_opps * Lb : 1;
        dU.alloc(ublk);
        LtDev D;
        D.q = dq.p;
        D.F = dF.p;
        D.jb = djb.p;
        D.fl = dfl.p;
        D.phin = dph.p;
        D.jn = djn.p;
        D.anc = danc.p;
        D.Kk = dK.p;
        D.O = dO.p;
        D.acc = (double *)dSave.p;
        D.ctr = (long long *)(D.acc + 4 * ng);
        D.fst = (double *)(D.ctr + 8);
        D.evh = (long long *)(D.fst + 2);
        D.err = (int *)(D.evh + cap1);
        D.bdh = dbdh.p;
        D.stM = dSTM.p;
        D.stC = dSTC.p;
        D.tick = dTick.p;
        std::vector<long long> anc0(N);
        for (int i = 0; i < N; i++)
            anc0[i] = i;
        CUDA_CHECK(cudaMemcpy(dq.p, q0, sizeof(double) * 6 * N, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(danc.p, anc0.data(), sizeof(long long) * N, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dK.p, Kk, sizeof(double) * ng * ng, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dO.p, O, sizeof(double) * 3 * nO, cudaMemcpyHostToDevice));
        HBuf<unsigned char> hSave;
        hSave.alloc(save_bytes);
        HBuf<double> hU[2];
        hU[0].alloc(ublk);
        hU[1].alloc(ublk);
        cudaStream_t st;
        CUDA_CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
        cudaEvent_t evU[2];
        CUDA_CHECK(cudaEventCreateWithFlags(&evU[0], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&evU[1], cudaEventDisableTiming));
        Pcg64 rng;
        rng.s = ((unsigned __int128)u64[2] << 64) | u64[3];
        rng.inc = ((unsigned __int128)u64[4] << 64) | u64[5];
        // FR opportunities: index o = (nxt - fr_start) / fr_every for nxt = fr_start + o fr_every <= n_steps;
        // chunk c holds opportunities [c chunk_opps, (c + 1) chunk_opps)
        const long long n_opp_total = (P.is_fr && n_steps >= P.fr_start) ? (n_steps - P.fr_start) / P.fr_every + 1 : 0;
        const long long nchunks = (n_opp_total + chunk_opps - 1) / chunk_opps;
        auto fill = [&](long long c, double *buf) {
            const long long o0 = c * chunk_opps, o1 = std::min(n_opp_total, o0 + chunk_opps);
            const long long n = (o1 - o0) * Lb;
            for (long long i = 0; i < n; i++)
                buf[i] = rng.next_double();
        };
        const double *acc_h = (const double *)hSave.p;
        const long long *ctr_h = (const long long *)(acc_h + 4 * ng);
        const double *fst_h = (const double *)(ctr_h + 8);
        const long long *evh_h = (const long long *)(fst_h + 2);
        const int *err_h = (const int *)(evh_h + cap1);
        double t_save = 0.0, t_uwait = 0.0;
        long long n_calls = 0, sp = 0;
        auto record = [&](long long k, double t0) {
            const double ta = host_now();
            CUDA_CHECK(cudaMemcpyAsync(hSave.p, dSave.p, save_bytes, cudaMemcpyDeviceToHost, st));
            CUDA_CHECK(cudaStreamSynchronize(st));
            const double w = host_now() - t0;
            if (*err_h)
                gpu_fail("lt_gpu_run: device error flag (deposit outside the clip)");
            memcpy(sv_acc + sp * 4 * ng, acc_h, sizeof(double) * 4 * ng);
            sv_i[sp * 3 + 0] = ctr_h[0];
            sv_i[sp * 3 + 1] = ctr_h[1];
            sv_i[sp * 3 + 2] = ctr_h[2];
            memcpy(sv_evh + sp * cap1, evh_h, sizeof(long long) * cap1);
            sv_nfe[sp] = (long long)N * (k + 1);
            sv_wall[sp] = w;
            sp++;
            t_save += host_now() - ta;
        };
        CUDA_CHECK(cudaDeviceSynchronize());
        tim[2] = host_now() - t_setup0;
        // ------------------------------------------------------------------ timed start
        const double t0 = host_now();
        std::future<void> fut;
        long long cur_chunk = -1;
        auto ensure_chunk = [&](long long c) {          // upload chunk c (sequential: c == cur_chunk + 1)
            if (c == cur_chunk)
                return;
            if (c == 0) {
                const double tw = host_now();
                fill(0, hU[0].p);
                t_uwait += host_now() - tw;
            } else {
                const double tw = host_now();
                fut.get();
                t_uwait += host_now() - tw;
            }
            const long long o0 = c * chunk_opps, o1 = std::min(n_opp_total, o0 + chunk_opps);
            CUDA_CHECK(cudaMemcpyAsync(dU.p, hU[c & 1].p, sizeof(double) * (o1 - o0) * Lb, cudaMemcpyHostToDevice,
                                       st));
            CUDA_CHECK(cudaEventRecord(evU[c & 1], st));
            if (c + 1 < nchunks)
                fut = std::async(std::launch::async, [&, c]() {
                    cudaEventSynchronize(evU[(c + 1) & 1]);
                    fill(c + 1, hU[(c + 1) & 1].p);
                });
            cur_chunk = c;
        };
        auto launch = [&](long long a, long long b, int a_move, long long opp_base) {
            const double *ub = dU.p;
            void *args[] = {(void *)&P, (void *)&D, (void *)&a, (void *)&b, (void *)&a_move, (void *)&ub,
                            (void *)&opp_base};
            CUDA_CHECK(cudaLaunchCooperativeKernel((void *)lt_step_kernel, dim3(B), dim3(T), args, smem, st));
            n_calls++;
        };
        // prologue: evaluate + deposit step 0
        launch(0, 0, 0, 0);
        while (sp < S && save_steps[sp] == 0)
            record(0, t0);
        long long k = 0;
        while (k < n_steps) {
            // the call [k, e]: moves of steps k .. e-1, i.e. opportunities nxt in [k + 1, e]
            long long e = n_steps;
            if (sp < S && save_steps[sp] < e)
                e = save_steps[sp];
            long long opp_base = 0;
            if (P.is_fr && n_opp_total > 0) {
                // first opportunity with nxt >= k + 1
                long long lo = k + 1 < P.fr_start ? P.fr_start : k + 1;
                long long o_first = (lo - P.fr_start + P.fr_every - 1) / P.fr_every;
                if (o_first < n_opp_total) {
                    const long long c = o_first / chunk_opps;
                    ensure_chunk(c);
                    opp_base = c * chunk_opps;
                    // the call must not cross the chunk's last opportunity
                    const long long o_last = std::min(n_opp_total, (c + 1) * chunk_opps) - 1;
                    const long long nxt_last = P.fr_start + o_last * P.fr_every;
                    if (o_last + 1 < n_opp_total && nxt_last < e)
                        e = nxt_last;              // opportunities up to and including nxt_last
                }
            }
            launch(k, e, 1, opp_base);
            while (sp < S && save_steps[sp] == e)
                record(e, t0);
            k = e;
        }
        if (fut.valid())
            fut.get();
        CUDA_CHECK(cudaMemcpyAsync(hSave.p, dSave.p, save_bytes, cudaMemcpyDeviceToHost, st));
        CUDA_CHECK(cudaMemcpyAsync(q_out, dq.p, sizeof(double) * 6 * N, cudaMemcpyDeviceToHost, st));
        CUDA_CHECK(cudaStreamSynchronize(st));
        const double t_end = host_now();
        // ------------------------------------------------------------------ timed end
        CUDA_CHECK(cudaGetLastError());
        if (*err_h)
            gpu_fail("lt_gpu_run: device error flag (deposit outside the clip)");
        if (sp != S)
            gpu_fail("lt_gpu_run: not every save fired");
        CUDA_CHECK(cudaMemcpy(anc_out, danc.p, sizeof(long long) * N, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(bdh_out, dbdh.p, sizeof(double) * 2 * ng, cudaMemcpyDeviceToHost));
        memcpy(acc_out, acc_h, sizeof(double) * 4 * ng);
        memcpy(evh_out, evh_h, sizeof(long long) * cap1);
        misc[0] = fst_h[0];
        misc[1] = fst_h[1];
        misc[2] = (double)ctr_h[3];
        misc[3] = (double)ctr_h[0];
        misc[4] = (double)ctr_h[1];
        misc[5] = (double)ctr_h[2];
        long long tk[LT_NTICK];
        CUDA_CHECK(cudaMemcpy(tk, dTick.p, sizeof tk, cudaMemcpyDeviceToHost));
        tim[0] = t0;
        tim[1] = t_end;
        tim[3] = t_save;
        tim[4] = t_uwait;
        tim[6] = (double)n_calls;
        tim[7] = (double)nchunks;
        tim[8] = (double)pr.clockRate;
        for (int q = 0; q < 7; q++)
            tim[9 + q] = (double)tk[q];
        tim[21] = (double)tk[7];
        tim[22] = (double)tk[8];
        tim[23] = (double)tk[9];
        tim[16] = B;
        tim[17] = T;
        tim[18] = (double)maxco;
        tim[19] = (double)smem;
        tim[20] = (double)*err_h;
        cudaEventDestroy(evU[0]);
        cudaEventDestroy(evU[1]);
        cudaStreamDestroy(st);
    } catch (GpuError &e) {
        snprintf(errbuf, (size_t)errlen, "%s", e.msg);
        return -1;
    }
    return 0;
}

}  // extern "C"
