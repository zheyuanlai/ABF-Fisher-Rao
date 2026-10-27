// gw_cuda.cu -- A100 backend of the entropic-gateway histogram-ABF (+ uniform FR) engine (NSCC A100 study).
//
// Same algorithm as the validated CPU engines (src/gateway_ladder_numba.py _advance, diagnostics off; its parallel
// form src/parallel/gateway_parallel.py + gw_kernel.c in counter mode), on one GPU:
//
//  * One PERSISTENT cooperative kernel per host segment (a segment ends at a save, at the end of an FR-uniform
//    chunk, or at the end of the run); every loop step runs inside the kernel, FR included, so the host only
//    stamps saves.  Walker i lives in registers of thread (i - w0) % blockDim of block i / wpb (WPT walkers per
//    thread); X, Y in global memory are a communication buffer written before an FR barrier / at kernel exit.
//  * Per loop step: force/bin of the pre-move state; DEPOSIT: f is rounded to the fixed-point grid 2^-fscale_log2
//    and added with 64-bit INTEGER atomics into a per-block shared table, then (gridDim > 1) into a global staging
//    table (triple-buffered by step mod 3, one grid barrier per step); every block adds the step's exact integer
//    bin sums to its own copy of the running float64 accumulators Mh += S 2^-fscale, Ch += count (identical in every
//    block: deterministic, order-independent).  Bias read b = Mh[j] / (Ch[j] + min_count) from the table holding
//    every deposit of the step; Euler-Maruyama move with counter-mode Philox4x64-10 noise, counter (step, slot, 0,
//    0), Box-Muller (zx, zy) -- the CPU counter mode's integers and uniforms; reflection into [-1.8, 1.8] with
//    Python's float mod.
//  * FR at loop steps sl with sl % fr_every == 0 on the post-move X: grid barrier, then EVERY block computes the
//    same selection redundantly (no second barrier): KDE on the 181-node grid with reflection images (per-node
//    sums in numba's order: ascending source node, image 0/1/2), mass and KL by one thread in numba's sequential
//    order, scores, fr_resample with the opportunity's fixed block of L = N + cap + max(N, cap) uniforms of
//    PCG64(SeedSequence([seed, N, 1])) (drawn on the host, the validated stream), proportional cap, partial
//    Fisher-Yates, survivor pool + clones, the pool shuffle; gather new = old[sel] only when kd + kc > 0 (no slot
//    permutation on a zero-event opportunity).  Block 0 keeps the counters and death histograms.
//
// Deviations from the CPU engine (all ulp-level or below, documented in docs/nscc_a100/VALIDATION.md):
//  (a) CUDA's exp/log/sin/cos (<= 2 ulp) instead of glibc's;  (b) the per-step bin sum is exact over values
//  rounded to 2^-34 (|error| <= 2.9e-11 per deposit) and added to the running total once per step (the CPU
//  gateway engine adds deposits one by one);  (c) none in the FR integer logic (identical comparisons on the same
//  doubles up to (a)).
#include "gpu_common.cuh"
#include <future>
#include <thread>
#include <vector>
#include <algorithm>

#define GW_XMIN (-1.8)
#define GW_XMAX (1.8)
#define GW_G 181
#define GW_EPS 1e-30
#define GW_KMAX 1024
#define GW_NCTR 8          // deaths, kd, kc, nopp, nopp_ev, nopp_d, -, -
#define GW_NTICK 8         // cycles: total, deposit, sync, move, fr, steps timed, calls, -

__constant__ double c_kern[GW_KMAX];

struct GwPar {
    int N, nb, wpb, fe, cap, L, do_fr, timers, pad, koff;
    double h, amp, two_s2, s2, dom_amp, H, oout, delta, min_count;
    double gamma, ramp_steps, score_clip, dt_fr, dx, logq, fscale, inv_fscale, fmax;
    unsigned long long k0, k1;
};

struct GwDev {
    double *X, *Y;
    double *Mh, *Ch;               // inside the save block
    long long *ctr, *evh, *dth;    // inside the save block
    int *err;                      // inside the save block
    double *mb;                    // per-block running max |bias|
    long long *stM, *stC;          // staging tables, 3 x nb
    long long *tick;
};

// ------------------------------------------------------------------------------------------------ dynamics
__device__ __forceinline__ double gw_pymod(double vx, double wx)
{
    double mod = (fabs(vx) < fabs(wx)) ? vx : fmod(vx, wx);
    if (mod != 0.0 || mod != mod) {
        if ((wx < 0.0) != (mod < 0.0))
            mod = mod + wx;
    } else {
        mod = (wx < 0.0) ? -0.0 : 0.0;
    }
    return mod;
}

__device__ __forceinline__ double gw_reflect(double q)
{
    const double span = GW_XMAX - GW_XMIN;
    double qm = gw_pymod(q - GW_XMIN, 2.0 * span);
    if (qm > span)
        qm = 2.0 * span - qm;
    return qm + GW_XMIN;
}

__device__ __forceinline__ void gw_force_bin(const GwPar &P, double x, double y, double &f, double &fy, int &j)
{
    const double e = exp(((-x) * x) / P.two_s2);
    const double om = P.oout + P.dom_amp * e;
    const double dom = ((-P.dom_amp) * (x / P.s2)) * e;
    f = (((4.0 * P.H) * x) * ((x * x) - 1.0)) + (((om * dom) * y) * y);
    fy = (om * om) * y;
    long long jj = (long long)floor(((x - GW_XMIN) / P.delta) + 1e-9);
    if (jj < 0)
        jj = 0;
    else if (jj > P.nb - 1)
        jj = P.nb - 1;
    j = (int)jj;
}

__device__ __forceinline__ void gw_noise(unsigned long long step, unsigned long long slot, unsigned long long k0,
                                         unsigned long long k1, double &zx, double &zy)
{
    unsigned long long o[4];
    philox4x64_10(step, slot, 0ULL, 0ULL, k0, k1, o);
    box_muller(u01(o[0]), u01(o[1]), zx, zy);
}

// ------------------------------------------------------------------------------------------------ smem layout
struct GwFrLayout {
    size_t xs, u, p, curv, md, hist, flag, die, clone, sel, pool, ocnt, scan, mi, bytes;
};

__host__ __device__ inline size_t gw_al8(size_t b) { return (b + 7) & ~(size_t)7; }

__host__ __device__ inline GwFrLayout gw_fr_layout(int N, int L, int cap)
{
    GwFrLayout l;
    size_t o = 0;
    l.xs = o;    o = gw_al8(o + sizeof(double) * N);
    l.u = o;     o = gw_al8(o + sizeof(double) * L);
    l.p = o;     o = gw_al8(o + sizeof(double) * GW_G);
    l.curv = o;  o = gw_al8(o + sizeof(double) * GW_G);
    l.md = o;    o = gw_al8(o + sizeof(double) * 4);
    l.hist = o;  o = gw_al8(o + sizeof(int) * GW_G);
    l.flag = o;  o = gw_al8(o + sizeof(int) * N);
    l.die = o;   o = gw_al8(o + sizeof(int) * N);
    l.clone = o; o = gw_al8(o + sizeof(int) * N);
    l.sel = o;   o = gw_al8(o + sizeof(int) * N);
    l.pool = o;  o = gw_al8(o + sizeof(int) * (N + cap + 1));
    l.ocnt = o;  o = gw_al8(o + sizeof(int) * N);
    l.scan = o;  o = gw_al8(o + sizeof(int) * 64);
    l.mi = o;    o = gw_al8(o + sizeof(int) * 16);
    l.bytes = o;
    return l;
}

__host__ __device__ inline size_t gw_base_bytes(int nb)
{
    return gw_al8(gw_al8(2 * sizeof(double) * nb + 5 * sizeof(unsigned) * nb + 8) + sizeof(double) * 32 +
                  sizeof(int) * (nb + 2));
}

// ------------------------------------------------------------------------------------------------ FR (one block)
// Every thread of the block participates.  Reads the post-move X of all N walkers from global memory and the
// opportunity's uniform block ug (L doubles).  Returns kd + kc (the gather index is in sel when > 0).  If
// `counters` (block 0), also counts the realised deaths (old slots not selected) into mi[4].
__device__ __noinline__ int gw_fr_block(const GwPar &P, const double *__restrict__ Xg, const double *__restrict__ ug, double g,
                           unsigned char *area, bool counters)
{
    const int N = P.N, L = P.L, cap = P.cap, tid = threadIdx.x, nt = blockDim.x;
    const GwFrLayout lay = gw_fr_layout(N, L, cap);
    double *xs = (double *)(area + lay.xs);
    double *u = (double *)(area + lay.u);
    double *p = (double *)(area + lay.p);
    double *curv = (double *)(area + lay.curv);
    double *md = (double *)(area + lay.md);
    int *hist = (int *)(area + lay.hist);
    int *flag = (int *)(area + lay.flag);
    int *die_c = (int *)(area + lay.die);
    int *clone_c = (int *)(area + lay.clone);
    int *sel = (int *)(area + lay.sel);
    int *pool = (int *)(area + lay.pool);
    int *ocnt = (int *)(area + lay.ocnt);
    int *scan = (int *)(area + lay.scan);
    int *mi = (int *)(area + lay.mi);
    const double dx = P.dx;

    for (int k = tid; k < GW_G; k += nt)
        hist[k] = 0;
    for (int i = tid; i < N; i += nt)
        xs[i] = Xg[i];
    for (int k = tid; k < L; k += nt)
        u[k] = ug[k];
    __syncthreads();
    // (1) KDE: counts on the nearest grid node
    for (int i = tid; i < N; i += nt) {
        int k = (int)rint((xs[i] - GW_XMIN) / dx);
        if (k < 0)
            k = 0;
        else if (k > GW_G - 1)
            k = GW_G - 1;
        atomicAdd(&hist[k], 1);
    }
    __syncthreads();
    // per output node: contributions in numba's order (source node ascending; image 0, 1, 2)
    const int pad = P.pad, koff = P.koff;
    for (int i = tid; i < GW_G; i += nt) {
        double pv = 0.0;
        // every source node with a contribution to node i (image 0, 1 or 2) lies in [i - pad, i + pad]: the
        // reflected images reach node i only from k <= pad - i (image 1) or k >= 2(G-1) - i - pad (image 2), both
        // inside that window; ascending k, images 0/1/2 per k: numba's order, the same terms
        const int k0 = i - pad > 0 ? i - pad : 0, k1 = i + pad < GW_G - 1 ? i + pad : GW_G - 1;
        for (int k = k0; k <= k1; k++) {
            const int c = hist[k];
            if (c == 0)
                continue;
            const double cd = (double)c;
            int o = k;
            if (abs(o - i) <= pad)
                pv += cd * c_kern[koff + (o - i + pad)];
            if (k >= 1 && k <= pad) {
                o = -k;
                if (abs(o - i) <= pad)
                    pv += cd * c_kern[koff + (o - i + pad)];
            }
            if (k <= GW_G - 2 && k >= GW_G - 1 - pad) {
                o = 2 * (GW_G - 1) - k;
                if (abs(o - i) <= pad)
                    pv += cd * c_kern[koff + (o - i + pad)];
            }
        }
        p[i] = pv;
    }
    __syncthreads();
    if (tid == 0) {
        const double inv_n = 1.0 / (double)N;
        double s = 0.0;
        for (int k = 0; k < GW_G; k++) {
            p[k] *= inv_n;
            s += p[k];
        }
        double mass = dx * (s - 0.5 * (p[0] + p[GW_G - 1]));
        if (mass < GW_EPS)
            mass = GW_EPS;
        md[0] = mass;
    }
    __syncthreads();
    {
        const double mass = md[0];
        for (int k = tid; k < GW_G; k += nt) {
            const double v = p[k] / mass;
            const double pk = v > GW_EPS ? v : GW_EPS;
            p[k] = pk;
            curv[k] = pk * (log(pk) - P.logq);
        }
    }
    __syncthreads();
    if (tid == 0) {
        double kl = 0.0, prev = curv[0];
        for (int k = 1; k < GW_G; k++) {
            const double cur = curv[k];
            kl += 0.5 * (cur + prev) * dx;
            prev = cur;
        }
        md[1] = kl;
    }
    __syncthreads();
    // (2) scores and the firing decisions (contiguous chunks per thread: compaction keeps ascending slot order)
    const double kl = md[1];
    const int E = (N + nt - 1) / nt;
    const int c0 = min(N, tid * E), c1 = min(N, c0 + E);
    int nd_t = 0, nc_t = 0;
    for (int i = c0; i < c1; i++) {
        double pos = (xs[i] - GW_XMIN) / dx;
        if (pos < 0.0)
            pos = 0.0;
        else if (pos > GW_G - 1.0)
            pos = GW_G - 1.0;
        int i0 = (int)floor(pos);
        if (i0 > GW_G - 2)
            i0 = GW_G - 2;
        const double frac = pos - i0;
        double v = p[i0] + frac * (p[i0 + 1] - p[i0]);
        if (v < GW_EPS)
            v = GW_EPS;
        double s = log(v) - P.logq - kl;
        if (s > P.score_clip)
            s = P.score_clip;
        else if (s < -P.score_clip)
            s = -P.score_clip;
        const double uu = u[i];
        int fl = 0;
        if (s > 0.0) {
            const double pd = 1.0 - exp(-g * s * P.dt_fr);
            if (uu < pd)
                fl = 1;
        } else if (s < 0.0) {
            const double pc = 1.0 - exp(g * s * P.dt_fr);
            if (uu < pc)
                fl = 2;
        }
        flag[i] = fl;
        nd_t += (fl == 1);
        nc_t += (fl == 2);
    }
    int tot;
    const int off = block_excl_scan(nd_t + (nc_t << 16), scan, &tot);
    {
        int od = off & 0xFFFF, oc = off >> 16;
        for (int i = c0; i < c1; i++) {
            if (flag[i] == 1)
                die_c[od++] = i;
            else if (flag[i] == 2)
                clone_c[oc++] = i;
        }
    }
    const int nd = tot & 0xFFFF, nc = tot >> 16;
    __syncthreads();
    // (3) proportional cap and the two partial Fisher-Yates shuffles (sequential: <= cap steps each)
    if (tid == 0) {
        const int nev = nd + nc;
        int kd, kc;
        if (nev > cap) {
            const int den = nev > 1 ? nev : 1;
            kd = (int)rint((double)(cap * nd) / (double)den);
            if (kd > nd)
                kd = nd;
            kc = cap - kd;
            if (kc > nc)
                kc = nc;
        } else {
            kd = nd;
            kc = nc;
        }
        int k = N;
        if (kd + kc > 0) {
            for (int t = 0; t < kd; t++) {
                const double uu = u[k++];
                int j = t + (int)(uu * (double)(nd - t));
                if (j > nd - 1)
                    j = nd - 1;
                const int tmp = die_c[t]; die_c[t] = die_c[j]; die_c[j] = tmp;
            }
            for (int t = 0; t < kc; t++) {
                const double uu = u[k++];
                int j = t + (int)(uu * (double)(nc - t));
                if (j > nc - 1)
                    j = nc - 1;
                const int tmp = clone_c[t]; clone_c[t] = clone_c[j]; clone_c[j] = tmp;
            }
        }
        mi[0] = kd;
        mi[1] = kc;
        mi[2] = k;
        mi[3] = nd;
        mi[5] = nc;
    }
    __syncthreads();
    const int kd = mi[0], kc = mi[1], ku = mi[2];
    if (kd + kc == 0)
        return 0;
    // (4) survivor pool (ascending) + clones
    for (int i = tid; i < N; i += nt)
        sel[i] = 0;
    __syncthreads();
    for (int t = tid; t < kd; t += nt)
        sel[die_c[t]] = 1;
    __syncthreads();
    int ns_t = 0;
    for (int i = c0; i < c1; i++)
        ns_t += (sel[i] == 0);
    int n_surv;
    int os = block_excl_scan(ns_t, scan, &n_surv);
    for (int i = c0; i < c1; i++)
        if (sel[i] == 0)
            pool[os++] = i;
    for (int t = tid; t < kc; t += nt)
        pool[n_surv + t] = clone_c[t];
    const int Psz = n_surv + kc;
    __syncthreads();
    // (5) the pool shuffle / fill
    if (Psz >= N) {
        if (tid == 0) {
            int k = ku;
            for (int t = 0; t < N; t++) {
                const double uu = u[k++];
                int j = t + (int)(uu * (double)(Psz - t));
                if (j > Psz - 1)
                    j = Psz - 1;
                const int tmp = pool[t]; pool[t] = pool[j]; pool[j] = tmp;
            }
        }
        __syncthreads();
        for (int t = tid; t < N; t += nt)
            sel[t] = pool[t];
    } else {
        for (int t = tid; t < N; t += nt) {
            if (t < Psz) {
                sel[t] = pool[t];
            } else {
                const double uu = u[ku + (t - Psz)];
                int j = (int)(uu * (double)n_surv);
                if (j > n_surv - 1)
                    j = n_surv - 1;
                sel[t] = pool[j];
            }
        }
    }
    __syncthreads();
    if (counters) {
        for (int i = tid; i < N; i += nt)
            ocnt[i] = 0;
        __syncthreads();
        for (int i = tid; i < N; i += nt)
            atomicAdd(&ocnt[sel[i]], 1);
        __syncthreads();
        int z = 0;
        for (int i = tid; i < N; i += nt)
            z += (ocnt[i] == 0);
        const int nde = block_sum_int(z, scan);
        if (tid == 0)
            mi[4] = nde;
        __syncthreads();
    }
    return kd + kc;
}

// ------------------------------------------------------------------------------------------------ step kernel
// HELP = 1: the second half of the block are HELPER threads that compute the next step's noise into shared memory
// while the walker threads run the bias read / move / force / deposit chain (the noise leaves the critical path);
// HELP = 0: each walker thread computes its next noise itself next to the force evaluation (instruction-level
// overlap).  Both give bitwise the same numbers (the noise is a function of (step, slot) only).
template <int WPT, int HELP>
__global__ void __launch_bounds__(512 / WPT) gw_step_kernel(GwPar P, GwDev D, long long s0, long long s1,
                                                       const double *__restrict__ ubuf, long long fr_base)
{
    cg::grid_group grid = cg::this_grid();
    extern __shared__ __align__(16) unsigned char smem[];
    const int nb = P.nb, N = P.N, tid = threadIdx.x, nt = blockDim.x, B = gridDim.x;
    double *Mloc = (double *)smem;
    double *Cloc = Mloc + nb;
    unsigned *sL = (unsigned *)(Cloc + nb);            // 4 limbs per bin
    unsigned *sCn = sL + 4 * nb;
    double *red = (double *)(smem + gw_al8(2 * sizeof(double) * nb + 5 * sizeof(unsigned) * nb + 8));
    int *sTch = (int *)(red + 32);                               // bins touched in this step (any order)
    int *sNT = sTch + nb;                                        // touched-bin counters, by step parity
    double *zb = (double *)(smem + gw_base_bytes(nb));          // HELP: noise [step parity][local slot][zx|zy]
    unsigned char *fr_area = smem + gw_base_bytes(nb) + (HELP ? gw_al8(sizeof(double) * 4 * P.wpb) : 0);
    const int w0 = blockIdx.x * P.wpb;
    const int w1 = min(N, w0 + P.wpb);
    const int TW = HELP ? nt / 2 : nt;                           // walker threads
    const bool helper = HELP && tid >= TW;
    const int lt = helper ? tid - TW : tid;
    for (int t = tid; t < nb; t += nt) {
        Mloc[t] = D.Mh[t];
        Cloc[t] = D.Ch[t];
        sL[4 * t] = sL[4 * t + 1] = sL[4 * t + 2] = sL[4 * t + 3] = 0u;
        sCn[t] = 0u;
    }
    if (tid < 2)
        sNT[tid] = 0;
    // zx, zy: the noise of the NEXT move, computed next to the force evaluation (two independent dependency chains
    // the scheduler can interleave); the noise is a function of (step, slot) only, so an FR gather does not touch it
    double x[WPT], y[WPT], f[WPT], fy[WPT], zx[WPT], zy[WPT];
    int jb[WPT], wi[WPT];
#pragma unroll
    for (int r = 0; r < WPT; r++) {
        const int slot = w0 + lt + r * TW;
        wi[r] = helper ? N : slot;                               // helpers own no walker
        x[r] = y[r] = f[r] = fy[r] = zx[r] = zy[r] = 0.0;
        jb[r] = 0;
        if (wi[r] < w1) {
            x[r] = D.X[wi[r]];
            y[r] = D.Y[wi[r]];
            if (!HELP)
                gw_noise((unsigned long long)s0, (unsigned long long)wi[r], P.k0, P.k1, zx[r], zy[r]);
            gw_force_bin(P, x[r], y[r], f[r], fy[r], jb[r]);
        }
        if (helper && slot < w1) {
            double *z = zb + ((size_t)(s0 & 1) * P.wpb + (slot - w0)) * 2;
            gw_noise((unsigned long long)s0, (unsigned long long)slot, P.k0, P.k1, z[0], z[1]);
        }
    }
    double mb = 0.0;
    int bad = 0;
    const bool tim = P.timers && blockIdx.x == 0 && tid == 0;
    long long c_dep = 0, c_sync = 0, c_move = 0, c_fr = 0, ta = 0, tb = 0;
    const long long c_start = tim ? clock64() : 0;
    __syncthreads();
    for (long long step = s0; step < s1; ++step) {
        const int par = (int)(step & 1);
        if (tim)
            ta = clock64();
#pragma unroll
        for (int r = 0; r < WPT; r++) {
            const bool act = wi[r] < w1;
            if (act && !(fabs(f[r]) <= P.fmax))
                bad |= 1;
            if (act) {
                const int j = jb[r];
                const unsigned long long u = (unsigned long long)__double2ll_rn(f[r] * P.fscale);
                if (atomicAdd(&sCn[j], 1u) == 0u)
                    sTch[atomicAdd(&sNT[par], 1)] = j;
                atomicAdd(&sL[4 * j + 0], (unsigned)(u & 0xFFFFULL));
                atomicAdd(&sL[4 * j + 1], (unsigned)((u >> 16) & 0xFFFFULL));
                atomicAdd(&sL[4 * j + 2], (unsigned)((u >> 32) & 0xFFFFULL));
                atomicAdd(&sL[4 * j + 3], (unsigned)((u >> 48) & 0xFFFFULL));
            }
        }
        __syncthreads();
        // only the bins touched in this step (the order of the list is irrelevant: each bin is updated once, by
        // exact integer sums); the other parity's counter is reset for the next step (its readers are done)
        const int ntch = sNT[par];
        if (tid == 0)
            sNT[par ^ 1] = 0;
        if (B == 1) {
            for (int i = tid; i < ntch; i += nt) {
                const int t = sTch[i];
                const unsigned c = sCn[t];
                Mloc[t] = Mloc[t] + (double)limb_total(sL, t) * P.inv_fscale;
                Cloc[t] = Cloc[t] + (double)c;
                sL[4 * t] = sL[4 * t + 1] = sL[4 * t + 2] = sL[4 * t + 3] = 0u;
                sCn[t] = 0u;
            }
            if (tim) { tb = clock64(); c_dep += tb - ta; ta = tb; }
        } else {
            const int s3 = (int)(step % 3);
            long long *gM = D.stM + (size_t)s3 * nb;
            long long *gC = D.stC + (size_t)s3 * nb;
            for (int i = tid; i < ntch; i += nt) {
                const int t = sTch[i];
                const unsigned c = sCn[t];
                atomicAdd((unsigned long long *)&gM[t], (unsigned long long)limb_total(sL, t));
                atomicAdd((unsigned long long *)&gC[t], (unsigned long long)c);
                sL[4 * t] = sL[4 * t + 1] = sL[4 * t + 2] = sL[4 * t + 3] = 0u;
                sCn[t] = 0u;
            }
            if (tim) { tb = clock64(); c_dep += tb - ta; ta = tb; }
            grid.sync();
            if (tim) { tb = clock64(); c_sync += tb - ta; ta = tb; }
            for (int t = tid; t < nb; t += nt) {
                const long long c = *((volatile long long *)&gC[t]);      // both loads issued together
                const long long m = *((volatile long long *)&gM[t]);
                if (c) {
                    Mloc[t] = Mloc[t] + (double)m * P.inv_fscale;
                    Cloc[t] = Cloc[t] + (double)c;
                }
            }
            if (blockIdx.x == 0) {
                const int z3 = (int)((step + 2) % 3);
                for (int t = tid; t < nb; t += nt) {
                    D.stM[(size_t)z3 * nb + t] = 0;
                    D.stC[(size_t)z3 * nb + t] = 0;
                }
            }
            if (tim) { tb = clock64(); c_dep += tb - ta; ta = tb; }
        }
        __syncthreads();
        if (helper && step + 1 < s1) {
#pragma unroll
            for (int r = 0; r < WPT; r++) {
                const int slot = w0 + lt + r * TW;
                if (slot < w1) {
                    double *z = zb + ((size_t)((step + 1) & 1) * P.wpb + (slot - w0)) * 2;
                    gw_noise((unsigned long long)(step + 1), (unsigned long long)slot, P.k0, P.k1, z[0], z[1]);
                }
            }
        }
#pragma unroll
        for (int r = 0; r < WPT; r++) {
            if (wi[r] < w1) {
                if (HELP) {
                    const double *z = zb + ((size_t)(step & 1) * P.wpb + (wi[r] - w0)) * 2;
                    zx[r] = z[0];
                    zy[r] = z[1];
                }
                const int j = jb[r];
                const double den = Cloc[j] + P.min_count;
                const double b = (den > 0.0) ? Mloc[j] / den : 0.0;
                const double ab = fabs(b);
                if (ab > mb)
                    mb = ab;
                const double xq = (x[r] + (((-f[r]) + b) * P.h)) + (P.amp * zx[r]);
                x[r] = gw_reflect(xq);
                y[r] = (y[r] - (fy[r] * P.h)) + (P.amp * zy[r]);
            }
        }
        if (tim) { tb = clock64(); c_move += tb - ta; ta = tb; }
        if (P.do_fr && (step % P.fe) == 0) {
#pragma unroll
            for (int r = 0; r < WPT; r++)
                if (wi[r] < w1) {
                    D.X[wi[r]] = x[r];
                    D.Y[wi[r]] = y[r];
                }
            __threadfence();
            grid_barrier(grid);
            const double *ug = ubuf + ((step - fr_base) / P.fe) * (long long)P.L;
            const double g = (P.ramp_steps > 0.0) ? P.gamma * (1.0 - exp(((double)(-step)) / P.ramp_steps))
                                                  : P.gamma;
            const bool cnt = (blockIdx.x == 0);
            const int nev = gw_fr_block(P, D.X, ug, g, fr_area, cnt);
            if (nev > 0) {
                const GwFrLayout lay = gw_fr_layout(N, P.L, P.cap);
                const int *sel = (const int *)(fr_area + lay.sel);
#pragma unroll
                for (int r = 0; r < WPT; r++)
                    if (wi[r] < w1) {
                        const int s = sel[wi[r]];
                        x[r] = D.X[s];
                        y[r] = D.Y[s];
                    }
            }
            if (cnt && tid == 0) {
                const GwFrLayout lay = gw_fr_layout(N, P.L, P.cap);
                const int *mi = (const int *)(fr_area + lay.mi);
                D.ctr[3] += 1;
                D.evh[nev] += 1;
                int nde = 0;
                if (nev > 0) {
                    D.ctr[4] += 1;
                    D.ctr[1] += mi[0];
                    D.ctr[2] += mi[1];
                    nde = mi[4];
                    D.ctr[0] += nde;
                    if (nde > 0)
                        D.ctr[5] += 1;
                }
                if (nde > P.cap)
                    bad |= 2;
                else
                    D.dth[nde] += 1;
            }
            __syncthreads();
            if (tim) { tb = clock64(); c_fr += tb - ta; ta = tb; }
        }
        if (step + 1 < s1) {
#pragma unroll
            for (int r = 0; r < WPT; r++)
                if (wi[r] < w1) {
                    if (!HELP)
                        gw_noise((unsigned long long)(step + 1), (unsigned long long)wi[r], P.k0, P.k1, zx[r],
                                 zy[r]);
                    gw_force_bin(P, x[r], y[r], f[r], fy[r], jb[r]);
                }
            if (tim) { tb = clock64(); c_move += tb - ta; ta = tb; }
        }
    }
    // exit: every gather of the last step must be complete before the state is written back
    grid_barrier(grid);
#pragma unroll
    for (int r = 0; r < WPT; r++)
        if (wi[r] < w1) {
            D.X[wi[r]] = x[r];
            D.Y[wi[r]] = y[r];
        }
    if (blockIdx.x == 0)
        for (int t = tid; t < nb; t += nt) {
            D.Mh[t] = Mloc[t];
            D.Ch[t] = Cloc[t];
        }
    const double bm = block_max(mb, red);
    if (tid == 0 && bm > D.mb[blockIdx.x])
        D.mb[blockIdx.x] = bm;
    if (bad)
        atomicOr(D.err, bad);
    if (tim) {
        D.tick[0] += clock64() - c_start;
        D.tick[1] += c_dep;
        D.tick[2] += c_sync;
        D.tick[3] += c_move;
        D.tick[4] += c_fr;
        D.tick[5] += s1 - s0;
        D.tick[6] += 1;
    }
}

// FR test kernel (one block): the selection of gw_fr_block on given X / uniforms
__global__ void gw_fr_test_kernel(GwPar P, const double *X, const double *ug, double g, int *sel_out, int *info)
{
    extern __shared__ __align__(16) unsigned char smem[];
    const int nev = gw_fr_block(P, X, ug, g, smem, true);
    const GwFrLayout lay = gw_fr_layout(P.N, P.L, P.cap);
    const int *sel = (const int *)(smem + lay.sel);
    const int *mi = (const int *)(smem + lay.mi);
    for (int i = threadIdx.x; i < P.N; i += blockDim.x)
        sel_out[i] = (nev > 0) ? sel[i] : i;
    if (threadIdx.x == 0) {
        info[0] = mi[0];   // kd
        info[1] = mi[1];   // kc
        info[2] = mi[3];   // nd
        info[3] = mi[5];   // nc
        info[4] = (nev > 0) ? mi[4] : 0;   // realised deaths
    }
}

__global__ void gw_normals_kernel(unsigned long long k0, unsigned long long k1, long long step0, long long n_steps,
                                  int N, double *out)
{
    const long long n = n_steps * N;
    for (long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x; t < n; t += (long long)gridDim.x * blockDim.x) {
        const long long s = t / N, i = t % N;
        double zx, zy;
        gw_noise((unsigned long long)(step0 + s), (unsigned long long)i, k0, k1, zx, zy);
        out[(2 * s) * N + i] = zx;
        out[(2 * s + 1) * N + i] = zy;
    }
}

// ------------------------------------------------------------------------------------------------ host
static GwPar gw_make_par(const long long *ip, const double *dp, const unsigned long long *u64, int nkern)
{
    GwPar P;
    memset(&P, 0, sizeof P);
    P.N = (int)ip[0];
    P.nb = (int)ip[1];
    P.fe = (int)ip[6];
    P.cap = (int)ip[7];
    P.L = (int)ip[8];
    P.do_fr = (int)ip[9];
    P.timers = (int)ip[10];
    const double h = dp[0], beta = dp[1], s = dp[5];
    P.h = h;
    P.H = dp[2];
    P.oout = dp[3];
    P.min_count = dp[6];
    P.amp = sqrt(2.0 * h / beta);
    P.two_s2 = 2.0 * s * s;
    P.s2 = s * s;
    P.dom_amp = dp[4] - dp[3];
    P.delta = (GW_XMAX - GW_XMIN) / (double)P.nb;
    P.gamma = dp[7];
    P.ramp_steps = dp[8];
    P.score_clip = dp[9];
    P.dt_fr = dp[10];
    P.dx = dp[11];
    P.logq = log(1.0 / (P.dx * (GW_G - 1)));
    P.fscale = ldexp(1.0, (int)ip[13]);
    P.inv_fscale = ldexp(1.0, -(int)ip[13]);
    P.fmax = ldexp(1.0, 62) / (P.fscale * (double)P.N);
    const int r = (nkern - 1) / 2;
    P.pad = r < GW_G - 1 ? r : GW_G - 1;
    P.koff = r - P.pad;
    P.k0 = u64[0];
    P.k1 = u64[1];
    return P;
}

typedef void (*gw_kfun)(GwPar, GwDev, long long, long long, const double *, long long);

static gw_kfun gw_pick(int wpt, int help)
{
    if (help) {
        switch (wpt) {
        case 1: return gw_step_kernel<1, 1>;
        case 2: return gw_step_kernel<2, 1>;
        case 4: return gw_step_kernel<4, 1>;
        default: return nullptr;
        }
    }
    switch (wpt) {
    case 1: return gw_step_kernel<1, 0>;
    case 2: return gw_step_kernel<2, 0>;
    case 4: return gw_step_kernel<4, 0>;
    default: return nullptr;
    }
}

extern "C" {

double gw_gpu_now(void) { return host_now(); }

// device properties: [sm count, clock kHz, max smem optin, cc major, cc minor, ecc, mem MB]
int gw_gpu_device_info(int dev, long long *out, char *name, int namelen)
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

int gw_gpu_normals(const unsigned long long *key, long long step0, long long n_steps, int N, double *out,
                   char *errbuf, int errlen)
{
    try {
        DBuf<double> d;
        d.alloc((size_t)n_steps * 2 * N);
        gw_normals_kernel<<<256, 256>>>(key[0], key[1], step0, n_steps, N, d.p);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(out, d.p, sizeof(double) * n_steps * 2 * N, cudaMemcpyDeviceToHost));
    } catch (GpuError &e) {
        snprintf(errbuf, (size_t)errlen, "%s", e.msg);
        return -1;
    }
    return 0;
}

// PCG64 doubles from a numpy state (tests): st = [state_hi, state_lo, inc_hi, inc_lo]
void gw_pcg64_doubles(const unsigned long long *st, long long n, double *out)
{
    Pcg64 g;
    g.s = ((unsigned __int128)st[0] << 64) | st[1];
    g.inc = ((unsigned __int128)st[2] << 64) | st[3];
    for (long long i = 0; i < n; i++)
        out[i] = g.next_double();
}

int gw_gpu_fr_test(const long long *ip, const double *dp, const unsigned long long *u64, const double *kern,
                   int nkern, const double *X, const double *ublk, double g, int threads, int *sel_out, int *info,
                   char *errbuf, int errlen)
{
    try {
        GwPar P = gw_make_par(ip, dp, u64, nkern);
        CUDA_CHECK(cudaMemcpyToSymbol(c_kern, kern, sizeof(double) * nkern));
        DBuf<double> dX, dU;
        DBuf<int> dS, dI;
        dX.alloc(P.N);
        dU.alloc(P.L);
        dS.alloc(P.N);
        dI.alloc(8);
        CUDA_CHECK(cudaMemcpy(dX.p, X, sizeof(double) * P.N, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dU.p, ublk, sizeof(double) * P.L, cudaMemcpyHostToDevice));
        const size_t sm = gw_fr_layout(P.N, P.L, P.cap).bytes;
        CUDA_CHECK(cudaFuncSetAttribute(gw_fr_test_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm));
        gw_fr_test_kernel<<<1, threads, sm>>>(P, dX.p, dU.p, g, dS.p, dI.p);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(sel_out, dS.p, sizeof(int) * P.N, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(info, dI.p, sizeof(int) * 5, cudaMemcpyDeviceToHost));
    } catch (GpuError &e) {
        snprintf(errbuf, (size_t)errlen, "%s", e.msg);
        return -1;
    }
    return 0;
}

/*
 * gw_gpu_run: one complete run (the timed part of run_arm_gpu).
 *   ip  int64[16]: N, nb, n_steps, blocks, threads, wpt, fr_every, cap, L, do_fr, timers, chunk_opps, device,
 *       fscale_log2, helpers (0/1: half of each block computes the noise), -
 *   dp  double[16]: h, beta, H, omega_out, omega_in, s, min_count, gamma, ramp_steps, score_clip, dt_fr, dx, ...
 *   u64 uint64[8]: Philox key k0, k1; PCG64 FR stream state hi, lo, inc hi, lo
 *   kern (nkern = 2 r + 1): the normalised KDE kernel
 *   x0, y0 (N); save_at (S) sorted completed-step counts in [0, n_steps]
 *   outputs: o_M, o_C (S, nb); o_fr (S, 6): deaths, kd, kc, nopp, nopp_ev, nopp_d; o_evh, o_dth (S, cap + 1);
 *            o_nfe (S) = N k; o_wall (S) seconds since the timed start; x_out, y_out (N); evh, dth final (cap + 1)
 *   tim double[24]: [0] t0 (CLOCK_MONOTONIC), [1] t_end, [2] setup s (before t0), [3] save s (sync + D2H + copy),
 *       [4] FR-uniform wait s (host producer), [5] max |bias|, [6] kernel calls, [7] chunks, [8] clock kHz,
 *       [9..15] device cycles of block 0 thread 0: total, deposit, barrier, move+force, FR, steps timed, calls,
 *       [16] blocks used, [17] threads, [18] max co-resident blocks, [19] dyn smem bytes, [20] err flags
 * Returns 0, or -1 with errbuf set.
 */
int gw_gpu_run(const long long *ip, const double *dp, const unsigned long long *u64, const double *kern, int nkern,
               const double *x0, const double *y0, const long long *save_at, long long S,
               double *o_M, double *o_C, long long *o_fr, long long *o_evh, long long *o_dth, long long *o_nfe,
               double *o_wall, double *x_out, double *y_out, long long *evh_out, long long *dth_out, double *tim,
               char *errbuf, int errlen)
{
    const double t_setup0 = host_now();
    try {
        GwPar P = gw_make_par(ip, dp, u64, nkern);
        const int N = P.N, nb = P.nb;
        const long long n_steps = ip[2];
        const int B = (int)ip[3], T = (int)ip[4], wpt = (int)ip[5];
        const long long chunk_opps = ip[11] > 0 ? ip[11] : 256;
        const int dev = (int)ip[12];
        if (N < 1 || nb < 1 || nb > 4096 || B < 1 || T < 32 || T > 1024 || (T & 31) || nkern > GW_KMAX ||
            nkern < 1 || (P.do_fr && (P.fe < 1 || P.cap < 1 || N < 2)))
            gpu_fail("gw_gpu_run: bad request");
        const int help = (int)ip[14];
        P.wpb = (N + B - 1) / B;
        if ((long long)(help ? T / 2 : T) * wpt < P.wpb || T > 512 / wpt || (help && (T & 63)))
            gpu_fail("gw_gpu_run: walker threads x wpt < walkers per block (or a bad helper split)");
        gw_kfun kf = gw_pick(wpt, help);
        if (!kf)
            gpu_fail("gw_gpu_run: wpt must be 1, 2 or 4");
        CUDA_CHECK(cudaSetDevice(dev));
        cudaDeviceProp pr;
        CUDA_CHECK(cudaGetDeviceProperties(&pr, dev));
        size_t smem = gw_base_bytes(nb) + (help ? gw_al8(sizeof(double) * 4 * P.wpb) : 0) +
                      (P.do_fr ? gw_fr_layout(N, P.L, P.cap).bytes : 0);
        if (smem > (size_t)pr.sharedMemPerBlockOptin)
            gpu_fail("gw_gpu_run: shared memory request exceeds the device limit");
        CUDA_CHECK(cudaFuncSetAttribute(kf, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
        int occ = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kf, T, smem));
        const long long maxco = (long long)occ * pr.multiProcessorCount;
        if (B > maxco)
            gpu_fail("gw_gpu_run: grid exceeds the co-resident block limit of a cooperative launch");
        CUDA_CHECK(cudaMemcpyToSymbol(c_kern, kern, sizeof(double) * nkern));
        // device state
        const int cap1 = P.cap + 1;
        const size_t save_bytes = sizeof(double) * 2 * nb + sizeof(long long) * (GW_NCTR + 2 * cap1) + 64;
        DBuf<double> dX, dY, dMB, dU;
        DBuf<long long> dSTM, dSTC, dTick;
        DBuf<unsigned char> dSave;
        dX.alloc(N);
        dY.alloc(N);
        dMB.alloc(B);
        dSTM.alloc((size_t)3 * nb);
        dSTC.alloc((size_t)3 * nb);
        dTick.alloc(GW_NTICK);
        dSave.alloc(save_bytes);
        const size_t ublk = P.do_fr ? (size_t)chunk_opps * P.L : 1;
        dU.alloc(ublk);
        GwDev D;
        D.X = dX.p;
        D.Y = dY.p;
        D.Mh = (double *)dSave.p;
        D.Ch = D.Mh + nb;
        D.ctr = (long long *)(D.Ch + nb);
        D.evh = D.ctr + GW_NCTR;
        D.dth = D.evh + cap1;
        D.err = (int *)(D.dth + cap1);
        D.mb = dMB.p;
        D.stM = dSTM.p;
        D.stC = dSTC.p;
        D.tick = dTick.p;
        CUDA_CHECK(cudaMemcpy(dX.p, x0, sizeof(double) * N, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dY.p, y0, sizeof(double) * N, cudaMemcpyHostToDevice));
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
        const int fe = P.fe;
        // FR chunks: chunk c covers loop steps [c * chunk_opps * fe, min(n_steps, (c + 1) * chunk_opps * fe))
        const long long cstep = P.do_fr ? chunk_opps * fe : n_steps;
        const long long nchunks = n_steps > 0 ? (n_steps + cstep - 1) / cstep : 0;
        auto n_opp_of = [&](long long c) -> long long {
            const long long a = c * cstep, b = std::min(n_steps, a + cstep);
            return ((b + fe - 1) / fe) - ((a + fe - 1) / fe);
        };
        auto fill = [&](long long c, double *buf) {
            const long long n = n_opp_of(c) * P.L;
            for (long long i = 0; i < n; i++)
                buf[i] = rng.next_double();
        };
        const double *Mh_h = (const double *)hSave.p;
        const double *Ch_h = Mh_h + nb;
        const long long *ctr_h = (const long long *)(Ch_h + nb);
        const long long *evh_h = ctr_h + GW_NCTR;
        const long long *dth_h = evh_h + cap1;
        const int *err_h = (const int *)(dth_h + cap1);
        double t_save = 0.0, t_uwait = 0.0;
        long long n_calls = 0, sptr = 0;
        auto record = [&](long long k, double t0) {
            const double ta = host_now();
            CUDA_CHECK(cudaMemcpyAsync(hSave.p, dSave.p, save_bytes, cudaMemcpyDeviceToHost, st));
            CUDA_CHECK(cudaStreamSynchronize(st));
            const double w = host_now() - t0;
            if (*err_h)
                gpu_fail(*err_h & 1 ? "gw_gpu_run: |f| beyond the fixed-point range" : "gw_gpu_run: deaths > cap");
            memcpy(o_M + sptr * nb, Mh_h, sizeof(double) * nb);
            memcpy(o_C + sptr * nb, Ch_h, sizeof(double) * nb);
            for (int q = 0; q < 6; q++)
                o_fr[sptr * 6 + q] = ctr_h[q];
            memcpy(o_evh + sptr * cap1, evh_h, sizeof(long long) * cap1);
            memcpy(o_dth + sptr * cap1, dth_h, sizeof(long long) * cap1);
            o_nfe[sptr] = (long long)N * k;
            o_wall[sptr] = w;
            sptr++;
            t_save += host_now() - ta;
        };
        CUDA_CHECK(cudaDeviceSynchronize());
        tim[2] = host_now() - t_setup0;
        // ------------------------------------------------------------------ timed start
        const double t0 = host_now();
        if (P.do_fr && nchunks > 0) {
            const double tw = host_now();
            fill(0, hU[0].p);                   // the first FR block is drawn inside the timed region (as the CPU engines)
            t_uwait += host_now() - tw;
        }
        while (sptr < S && save_at[sptr] == 0)
            record(0, t0);
        std::future<void> fut;
        for (long long c = 0; c < nchunks; c++) {
            const long long a = c * cstep, b = std::min(n_steps, a + cstep);
            if (P.do_fr) {
                if (c > 0) {
                    const double tw = host_now();
                    fut.get();
                    t_uwait += host_now() - tw;
                }
                const long long n = n_opp_of(c) * P.L;
                if (n > 0)
                    CUDA_CHECK(cudaMemcpyAsync(dU.p, hU[c & 1].p, sizeof(double) * n, cudaMemcpyHostToDevice, st));
                CUDA_CHECK(cudaEventRecord(evU[c & 1], st));
                if (c + 1 < nchunks) {
                    // the next block is drawn while this chunk runs (PCG64 is sequential: one producer at a time)
                    fut = std::async(std::launch::async, [&, c]() {
                        cudaEventSynchronize(evU[(c + 1) & 1]);   // the copy that last read this host buffer
                        fill(c + 1, hU[(c + 1) & 1].p);
                    });
                }
            }
            const long long fr_base = ((a + fe - 1) / fe) * fe;
            long long s = a;
            while (s < b) {
                long long e = b;
                if (sptr < S && save_at[sptr] < e)
                    e = save_at[sptr];
                const double *ub = dU.p;
                void *args[] = {(void *)&P, (void *)&D, (void *)&s, (void *)&e, (void *)&ub, (void *)&fr_base};
                CUDA_CHECK(cudaLaunchCooperativeKernel((void *)kf, dim3(B), dim3(T), args, smem, st));
                n_calls++;
                while (sptr < S && save_at[sptr] == e)
                    record(e, t0);
                s = e;
            }
        }
        if (fut.valid())
            fut.get();
        CUDA_CHECK(cudaMemcpyAsync(hSave.p, dSave.p, save_bytes, cudaMemcpyDeviceToHost, st));
        CUDA_CHECK(cudaMemcpyAsync(x_out, dX.p, sizeof(double) * N, cudaMemcpyDeviceToHost, st));
        CUDA_CHECK(cudaMemcpyAsync(y_out, dY.p, sizeof(double) * N, cudaMemcpyDeviceToHost, st));
        CUDA_CHECK(cudaStreamSynchronize(st));
        const double t_end = host_now();
        // ------------------------------------------------------------------ timed end
        CUDA_CHECK(cudaGetLastError());
        if (*err_h)
            gpu_fail(*err_h & 1 ? "gw_gpu_run: |f| beyond the fixed-point range" : "gw_gpu_run: deaths > cap");
        if (sptr != S)
            gpu_fail("gw_gpu_run: not every save fired");
        memcpy(evh_out, evh_h, sizeof(long long) * cap1);
        memcpy(dth_out, dth_h, sizeof(long long) * cap1);
        std::vector<double> mbv(B);
        CUDA_CHECK(cudaMemcpy(mbv.data(), dMB.p, sizeof(double) * B, cudaMemcpyDeviceToHost));
        long long tk[GW_NTICK];
        CUDA_CHECK(cudaMemcpy(tk, dTick.p, sizeof tk, cudaMemcpyDeviceToHost));
        double mbmax = 0.0;
        for (int q = 0; q < B; q++)
            mbmax = std::max(mbmax, mbv[q]);
        tim[0] = t0;
        tim[1] = t_end;
        tim[3] = t_save;
        tim[4] = t_uwait;
        tim[5] = mbmax;
        tim[6] = (double)n_calls;
        tim[7] = (double)nchunks;
        tim[8] = (double)pr.clockRate;
        for (int q = 0; q < 7; q++)
            tim[9 + q] = (double)tk[q];
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
