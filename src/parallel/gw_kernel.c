/*
 * gw_kernel.c -- walker-parallel OpenMP kernel of the entropic-gateway histogram-ABF engine (Experiment B).
 *
 * Final campaign, docs/final_campaign/PREREGISTRATION.md section 2.2 and EXPB_BACKEND_DESIGN.md.  The kernel
 * advances the loop steps [step0, step1) of ONE arm exactly as src/gateway_ladder_numba.py::_advance does with
 * diagnostics off (MODEL_ORIG: alpha = 1, lam = 1), for the steps between two FR opportunities / saves; FR, saves
 * and all bookkeeping stay in the numba driver (src/parallel/gateway_parallel.py), which calls the validated numba
 * FR functions.  Every floating-point expression below is evaluated in the order numba evaluates the Python source
 * (left to right; unary minus binds tighter than * and /); the library is built with
 *   gcc -O3 -march=native -fopenmp -fno-fast-math -ffp-contract=off -fPIC -shared
 * so there is no FMA contraction and no reassociation; exp, log, sin, cos, fmod are glibc's (the same libm the
 * numba engine calls in the same process).
 *
 * Per loop step (walker i, slot order; thread t owns the walker block [w0, w1) and the bin block [b0, b1)):
 *   force/bin of own walkers at the pre-move state (computed at kernel entry for step0, otherwise fused into the
 *     previous step's move loop: same walker, same thread);
 *   BARRIER A (all forces/bins of this step are written);
 *   DEPOSIT, bin-owner: thread t scans ALL N walkers in slot order and adds those whose bin is in [b0, b1):
 *     Ch[j] += 1.0; Mh[j] += f.  Per bin this is the serial slot-order summation, bitwise, at any thread count;
 *   BARRIER B (every deposit of the step is complete before any bias read);
 *   BIAS READ of own walkers: den = Ch[j] + min_count; bias = Mh[j]/den (0 if den <= 0); running max |bias|
 *     (thread-local, reduced by max after the region: exact, order-independent);
 *   MOVE of own walkers with the step's noise (replay: buffer filled by the driver from numba's MT19937 in the
 *     validated order; counter: Philox4x64-10, counter (step, slot, 0, 0)), then (if not the segment's last step)
 *     the force/bin of the moved walker for the next step.
 * After BARRIER B a walker's fx/fy/jb/X/Y are touched only by its owner, and Mh/Ch are written again only after
 * the next BARRIER A, so the two barriers per step are the only synchronisation needed.
 *
 * Variants (ip[7] bit flags; 0 = the design brief's kernel above, the default):
 *   bit 0 REPLICATED deposit: every thread keeps a private copy of (Mh, Ch) (copied from the shared table at
 *     kernel entry) and adds ALL N walkers' deposits into it in slot order (so every copy is bitwise the serial
 *     table), then reads its walkers' bias from its own copy: no barrier between deposit and bias read, ONE barrier
 *     per step.  fx/jb are double-buffered by step parity (thread u may still read step k's buffer while thread t
 *     writes step k+1's; t cannot write step k+2's = step k's buffer before u passes barrier k+1, i.e. after u's
 *     deposit of step k).  Thread 0 writes its copy back to (Mh, Ch) at the end of the call.
 *   bit 1 DISSEMINATION barrier (C11 release/acquire flags, ceil(log2 T) rounds, monotone episode numbers) in place
 *     of `#pragma omp barrier` inside the same OpenMP parallel region.
 * Every variant gives bitwise the same result (the per-bin summation order is the slot order in all of them).
 *
 * Timers (ip[6] = timer mode: 0 off; m >= 1 on with phase reads on a SAMPLE of the loop steps, mean gap ~2^(m-1);
 * the driver's default is m = 9, ~1 step in 256; m = 1 reads every step).  Per thread and per call, the rdtsc
 * stamps at region entry, after the replicated-table copy, after the entry force/bin, after the step loop and
 * after the copy-back are EXACT (7 reads per call incl. the master's launch / join stamps).  Inside the step loop
 * the phase reads (barrier A wait, deposit, barrier B wait, bias read, move + next force) are taken only on the
 * sampled steps s_0 = 0, s_{i+1} = s_i + 1 + (mix64(s_i) >> (64 - m)), mix64(s) = the splitmix64 output
 * finaliser of (s + 1) * 0x9E3779B97F4A7C15 (gaps ~uniform on [1, 2^m], mean ~2^(m-1); m = 1: every step).
 * The sequence is a function of the run's global loop step only (the next sampled step is carried across calls
 * in acc[12]), so every thread samples the same steps at any thread count and segmentation, and the
 * pseudo-random gaps do not alias with the FR period or the save grid.  Between two sampled steps the loop runs
 * gw_run_untimed, the kernel's only step loop (also the whole call with timers off), with no per-step test; the
 * sampled step runs gw_step with the timing flag set.  Rationale: reading rdtsc 3-6 times every step cost +27 %
 * at N = 1, +8 % at N = 2, +4 % at N = 4 (t = 1), which the validated serial engine does not pay (sampled:
 * <= 0.33 % at N = 1-16, timers on vs off, interleaved, one idle core).  The driver splits the
 * exact step-loop ticks over the phases in proportion to the sampled phase ticks (ratio estimator), so force/move
 * + ABF + sync = the exact region wall time (team mean); only the split inside the loop is estimated.
 * acc[] layout: gw_advance comment.
 * The driver converts ticks to seconds with the TSC frequency measured over the run (constant_tsc).
 */
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <stdatomic.h>
#include <sched.h>
#include <omp.h>
#include <x86intrin.h>

#define GW_XMIN (-1.8)
#define GW_XMAX (1.8)
#define GW_MAXT 512           /* maximum team size */
#define GW_PAD 16             /* 16 doubles = 128 bytes per thread slot of the scratch arrays (no false sharing) */
/* timer sampling-gap hash: splitmix64's output function of (s + 1) * golden (evaluated on sampled steps only) */
static inline uint64_t gw_mix64(uint64_t s)
{
    uint64_t z = (s + 1ULL) * 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/* ---------------------------------------------------------------------------------------------------------- */
/* Philox4x64-10 (Random123 constants) and Box-Muller                                                          */
/* ---------------------------------------------------------------------------------------------------------- */
#define PHILOX_M0 0xD2E7470EE14C6C93ULL
#define PHILOX_M1 0xCA5A826395121157ULL
#define PHILOX_W0 0x9E3779B97F4A7C15ULL
#define PHILOX_W1 0xBB67AE8584CAA73BULL

static inline void philox4x64_10(uint64_t c0, uint64_t c1, uint64_t c2, uint64_t c3, uint64_t k0, uint64_t k1,
                                 uint64_t out[4])
{
    for (int r = 0; r < 10; r++) {
        if (r > 0) {
            k0 += PHILOX_W0;
            k1 += PHILOX_W1;
        }
        const unsigned __int128 p0 = (unsigned __int128)PHILOX_M0 * (unsigned __int128)c0;
        const unsigned __int128 p1 = (unsigned __int128)PHILOX_M1 * (unsigned __int128)c2;
        const uint64_t hi0 = (uint64_t)(p0 >> 64), lo0 = (uint64_t)p0;
        const uint64_t hi1 = (uint64_t)(p1 >> 64), lo1 = (uint64_t)p1;
        c0 = hi1 ^ c1 ^ k0;
        c1 = lo1;
        c2 = hi0 ^ c3 ^ k1;
        c3 = lo0;
    }
    out[0] = c0; out[1] = c1; out[2] = c2; out[3] = c3;
}

static inline double gw_u01(uint64_t o)
{
    return (double)(o >> 11) * 0x1.0p-53;       /* exact: 53-bit integer times 2^-53, in [0, 1 - 2^-53] */
}

static inline void gw_box_muller(double ua, double ub, double *z0, double *z1)
{
    const double r = sqrt(-2.0 * log(1.0 - ua));
    const double th = (2.0 * M_PI) * ub;        /* 2.0 * M_PI is the exact double 2 * pi (= Python 2.0 * math.pi) */
    *z0 = r * cos(th);
    *z1 = r * sin(th);
}

/* gateway counter-mode noise of one walker-step: word 0, (u0, u1) -> (zx, zy) */
static inline void gw_counter_noise(uint64_t step, uint64_t slot, uint64_t k0, uint64_t k1, double *zx, double *zy)
{
    uint64_t o[4];
    philox4x64_10(step, slot, 0ULL, 0ULL, k0, k1, o);
    gw_box_muller(gw_u01(o[0]), gw_u01(o[1]), zx, zy);
}

/* ---------------------------------------------------------------------------------------------------------- */
/* dynamics primitives (gateway_ladder_numba._advance / _reflect, op for op)                                   */
/* ---------------------------------------------------------------------------------------------------------- */
/* numba's float % (numba/cpython/numbers.py real_divmod_func_body = CPython float_divmod): fmod, then the sign of
 * the divisor; a zero remainder takes the divisor's sign. */
static inline double gw_pymod(double vx, double wx)
{
    /* fmod(vx, wx) = vx - n wx with n = trunc(vx / wx); for |vx| < |wx| n = 0 and the result is vx exactly (sign of
     * zero included), so the library call is skipped there (bitwise the same; NaN / inf take the fmod path) */
    double mod = (fabs(vx) < fabs(wx)) ? vx : fmod(vx, wx);
    if (mod != 0.0 || mod != mod) {             /* unordered != : non-zero or NaN */
        if ((wx < 0.0) != (mod < 0.0))
            mod = mod + wx;
    } else {
        mod = (wx < 0.0) ? -0.0 : 0.0;
    }
    return mod;
}

static inline double gw_reflect(double q, double lo, double hi)
{
    const double span = hi - lo;
    double qm = gw_pymod(q - lo, 2.0 * span);
    if (qm > span)
        qm = 2.0 * span - qm;
    return qm + lo;
}

typedef struct {
    double h, min_count, amp, two_s2, s2, dom_amp, H, oout, delta;
    int64_t nb;
} gw_par;

static inline void gw_force_bin(const gw_par *P, double x, double y, double *f_out, double *fy_out, int32_t *j_out)
{
    const double e = exp(((-x) * x) / P->two_s2);
    const double om = P->oout + P->dom_amp * e;
    const double dom = ((-P->dom_amp) * (x / P->s2)) * e;
    const double f = (((4.0 * P->H) * x) * ((x * x) - 1.0)) + (((om * dom) * y) * y);
    *f_out = f;
    *fy_out = (om * om) * y;
    int64_t j = (int64_t)floor(((x - GW_XMIN) / P->delta) + 1e-9);
    if (j < 0)
        j = 0;
    else if (j > P->nb - 1)
        j = P->nb - 1;
    *j_out = (int32_t)j;
}

/* walker block of thread tid: blocks of 8 walkers (64 bytes of doubles) when every thread gets >= 1 block */
static inline void gw_wpart(int64_t N, int nth, int tid, int64_t *w0, int64_t *w1)
{
    if (N >= 8 * (int64_t)nth) {
        const int64_t nblk = (N + 7) / 8;
        const int64_t a = nblk * tid / nth, b = nblk * (tid + 1) / nth;
        *w0 = 8 * a;
        *w1 = (8 * b < N) ? 8 * b : N;
    } else {
        *w0 = N * tid / nth;
        *w1 = N * (tid + 1) / nth;
    }
}

static double g_mb[GW_MAXT * GW_PAD] __attribute__((aligned(128)));
static uint64_t g_tk[GW_MAXT * GW_PAD] __attribute__((aligned(128)));

/* dissemination barrier: flag[round][thread] holds the last episode signalled to that thread in that round */
#define GW_MAXROUNDS 10
typedef struct {
    _Alignas(128) atomic_llong v;
} gw_flag_t;
static gw_flag_t g_fl[GW_MAXROUNDS][GW_MAXT];
static long long g_episode = 0;      /* episode number reached by every thread at the end of the last call */

static inline void gw_dbar(int nth, int tid, long long *ep)
{
    const long long e = ++*ep;
    for (int r = 0, d = 1; d < nth; r++, d <<= 1) {
        const int partner = (tid + d) % nth;
        atomic_store_explicit(&g_fl[r][partner].v, e, memory_order_release);
        unsigned spins = 0;
        while (atomic_load_explicit(&g_fl[r][tid].v, memory_order_acquire) < e) {
            /* spin; after ~2^14 pauses (tens of microseconds, i.e. the partner is descheduled: an oversubscribed
             * core) yield the cpu so that the partner can run instead of losing a whole time slice per barrier */
            if (++spins < (1u << 14))
                _mm_pause();
            else
                sched_yield();
        }
    }
}

/* (an orphaned `omp barrier` inside gw_step binds to the enclosing parallel region of gw_advance) */
#define GW_BARRIER()                                      \
    do {                                                  \
        if (dissem)                                       \
            gw_dbar(nth, tid, ep);                        \
        else {                                            \
            _Pragma("omp barrier")                        \
        }                                                 \
    } while (0)

/* per-thread state of one gw_advance call (read-only inside a step except mb, ep and the tick sums) */
typedef struct {
    const gw_par *P;
    int64_t N, step0, step1, w0, w1;
    int32_t b0, b1;
    int nth, tid, dissem, replicated, mode;
    uint64_t k0, k1;
    double *X, *Y, *Mh, *Ch, *Cr, *Mr, *fx, *fy, *bias;
    int32_t *jb;
    const double *zbuf;
} gw_thr;

typedef struct {
    uint64_t fm, dep, bias, sync, n;    /* sampled phase ticks and the number of sampled steps */
} gw_ticks;

/* ONE loop step of one thread (force/bin of this step already in the parity buffer).  Always inlined and called
 * with a compile-time constant `smp`, so the compiler emits two copies: the untimed one (no rdtsc at all: the
 * code of timers off) and the timed one, taken only on sampled steps. */
static inline __attribute__((always_inline)) void gw_step(const int smp, const gw_thr *S, const int64_t step,
                                                          double *mbp, long long *ep, gw_ticks *tk)
{
    const gw_par *const P = S->P;
    const int64_t N = S->N, w0 = S->w0, w1 = S->w1;
    const int32_t b0 = S->b0, b1 = S->b1;
    const int nth = S->nth, tid = S->tid, dissem = S->dissem;
    double *const X = S->X, *const Y = S->Y, *const Mh = S->Mh, *const Ch = S->Ch, *const Cr = S->Cr,
           *const Mr = S->Mr, *const fy = S->fy, *const bias = S->bias;
    const double h_ = P->h, amp = P->amp, min_count = P->min_count;
    const double *fc = S->fx + (step & 1) * N;          /* this step's forces / bins */
    const int32_t *jc = S->jb + (step & 1) * N;
    double *fn = S->fx + ((step + 1) & 1) * N;          /* next step's */
    int32_t *jn = S->jb + ((step + 1) & 1) * N;
    double mb = *mbp;
    uint64_t last = 0, now = 0;
    if (smp)
        last = __rdtsc();
    if (nth > 1) {                                      /* (a team of one needs no barrier) */
        GW_BARRIER();                                   /* A: every force/bin of this step is written */
        if (smp) { now = __rdtsc(); tk->sync += now - last; last = now; }
    }
    if (S->replicated) {
        /* all walkers in slot order into the private copy */
        for (int64_t i = 0; i < N; i++) {
            const int32_t j = jc[i];
            Cr[j] = Cr[j] + 1.0;
            Mr[j] = Mr[j] + fc[i];
        }
        if (smp) { now = __rdtsc(); tk->dep += now - last; last = now; }
    } else {
        /* bin owner: all walkers in slot order, own bins only */
        for (int64_t i = 0; i < N; i++) {
            const int32_t j = jc[i];
            if (j >= b0 && j < b1) {
                Ch[j] = Ch[j] + 1.0;
                Mh[j] = Mh[j] + fc[i];
            }
        }
        if (smp) { now = __rdtsc(); tk->dep += now - last; last = now; }
        if (nth > 1) {
            GW_BARRIER();                               /* B: every deposit of this step is complete */
            if (smp) { now = __rdtsc(); tk->sync += now - last; last = now; }
        }
    }
    /* bias read (from a table holding every deposit of this step) */
    for (int64_t i = w0; i < w1; i++) {
        const int32_t j = jc[i];
        const double den = Cr[j] + min_count;
        const double b = (den > 0.0) ? Mr[j] / den : 0.0;
        bias[i] = b;
        const double ab = fabs(b);
        if (ab > mb)
            mb = ab;
    }
    *mbp = mb;
    if (smp) { now = __rdtsc(); tk->bias += now - last; last = now; }
    /* move (+ the next step's force/bin of the same walker) */
    const int nxt = (step + 1 < S->step1);
    if (S->mode == 0) {
        const double *zx = S->zbuf + (step - S->step0) * 2 * N;
        const double *zy = zx + N;
        for (int64_t i = w0; i < w1; i++) {
            const double xq = (X[i] + (((-fc[i]) + bias[i]) * h_)) + (amp * zx[i]);
            const double xn = gw_reflect(xq, GW_XMIN, GW_XMAX);
            const double yn = (Y[i] - (fy[i] * h_)) + (amp * zy[i]);
            X[i] = xn;
            Y[i] = yn;
            if (nxt)
                gw_force_bin(P, xn, yn, &fn[i], &fy[i], &jn[i]);
        }
    } else {
        const uint64_t k0 = S->k0, k1 = S->k1;
        for (int64_t i = w0; i < w1; i++) {
            double zx, zy;
            gw_counter_noise((uint64_t)step, (uint64_t)i, k0, k1, &zx, &zy);
            const double xq = (X[i] + (((-fc[i]) + bias[i]) * h_)) + (amp * zx);
            const double xn = gw_reflect(xq, GW_XMIN, GW_XMAX);
            const double yn = (Y[i] - (fy[i] * h_)) + (amp * zy);
            X[i] = xn;
            Y[i] = yn;
            if (nxt)
                gw_force_bin(P, xn, yn, &fn[i], &fy[i], &jn[i]);
        }
    }
    if (smp) { now = __rdtsc(); tk->fm += now - last; tk->n++; }
}

/* the untimed loop steps [s0, s1) of one thread: the ONLY step loop of the kernel, used with timers off (the whole
 * call) and with timers on (the runs between two sampled steps), so the timed and untimed builds of a step are the
 * same machine code.  Not inlined, and the thread state is a by-value copy, so the loop-invariant flags (mode,
 * replicated, team size, barrier kind) are locals the compiler can hoist / unswitch on. */
static __attribute__((noinline)) void gw_run_untimed(const gw_thr S, const int64_t s0, const int64_t s1,
                                                     double *mbp, long long *ep)
{
    double mb = *mbp;
    long long e = *ep;
    gw_ticks tk = {0, 0, 0, 0, 0};
    for (int64_t step = s0; step < s1; step++)
        gw_step(0, &S, step, &mb, &e, &tk);
    *mbp = mb;
    *ep = e;
}

/* ---------------------------------------------------------------------------------------------------------- */
/* exported API                                                                                                */
/* ---------------------------------------------------------------------------------------------------------- */
double gw_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

uint64_t gw_rdtsc(void)
{
    return __rdtsc();
}

int gw_max_threads(void) { return GW_MAXT; }

int gw_omp_proc_bind(void) { return (int)omp_get_proc_bind(); }

int gw_omp_num_places(void) { return omp_get_num_places(); }

/* the OS cpu ids of place p (writes at most nmax ids, returns the count) */
int gw_omp_place_ids(int p, int *ids, int nmax)
{
    int n = omp_get_place_num_procs(p);
    if (n > nmax)
        n = nmax;
    if (n > 0) {
        int buf[1024];
        omp_get_place_proc_ids(p, buf);
        for (int k = 0; k < n; k++)
            ids[k] = buf[k];
    }
    return n;
}

/* team probe: for a team of T threads, the place number of each thread (place[tid]); returns the team size */
int gw_team_places(int T, int *place)
{
    int got = 0;
#pragma omp parallel num_threads(T)
    {
        const int tid = omp_get_thread_num();
        place[tid] = omp_get_place_num();
        if (tid == 0)
            got = omp_get_num_threads();
    }
    return got;
}

/* raw Philox4x64-10 blocks: ctr (n, 4), out (n, 4) */
void gw_philox_many(int64_t n, const uint64_t *ctr, const uint64_t *key, uint64_t *out)
{
    for (int64_t i = 0; i < n; i++)
        philox4x64_10(ctr[4 * i], ctr[4 * i + 1], ctr[4 * i + 2], ctr[4 * i + 3], key[0], key[1], out + 4 * i);
}

/* counter-mode gateway normals of the moves step0 .. step0 + n_steps - 1, layout (n_steps, 2, N) = [step][zx|zy][slot]
 * (the kernel's exact noise; OpenMP over steps, nthreads <= 0 = runtime default) */
void gw_counter_normals(const uint64_t *key, int64_t step0, int64_t n_steps, int64_t N, int nthreads, double *out)
{
    const uint64_t k0 = key[0], k1 = key[1];
    const int T = nthreads > 0 ? nthreads : omp_get_max_threads();
#pragma omp parallel for schedule(static) num_threads(T)
    for (int64_t s = 0; s < n_steps; s++) {
        double *zx = out + 2 * N * s, *zy = zx + N;
        for (int64_t i = 0; i < N; i++)
            gw_counter_noise((uint64_t)(step0 + s), (uint64_t)i, k0, k1, zx + i, zy + i);
    }
}

/*
 * gw_advance: advance loop steps [step0, step1) in place.
 *   ip  int64[8]: N, nb, nthreads, step0, step1, noise (0 replay, 1 counter), timer mode (0 off; m in [1, 32]:
 *       on, sampled steps with gaps uniform on [1, 2^m]; m = 1 every step), variant flags
 *   dp  double[8]: h, beta, H, omega_out, omega_in, s, min_count, unused
 *   key uint64[2]: Philox key (counter mode)
 *   X, Y (N): state; Mh, Ch (nb): ABF accumulators; scratch fx (2N), jb int32 (2N) (step-parity double
 *   buffers), fy (N), bias (N)
 *   zbuf: replay noise, (step1 - step0, 2, N) = [step][zx|zy][slot] (unused in counter mode)
 *   fctr double[1]: running max |bias| (in/out)
 *   acc double[16], accumulated over calls (ticks are team means of per-thread rdtsc ticks):
 *     EXACT  [0] entry force/bin, [1] region launch wait + join wait, [2] step loop, [11] replicated-table copy
 *            in / copy back (ABF bookkeeping), [7] region wall (master, launch to join);
 *     SAMPLED (sampled steps only) [3] move + next force/bin (incl. counter noise), [4] deposit, [5] bias read,
 *            [6] barrier waits;
 *     COUNTS [8] kernel calls, [9] loop steps, [10] sampled loop steps (always accumulated, timers on or off;
 *            [10] stays 0 with timers off).
 *     STATE  [12] the next sampled loop step (in/out; 0 at the start of a run, so step 0 is sampled).
 *     [0] + [1] + [2] + [11] = [7] up to rounding (per thread: launch wait + work + join wait = launch to join).
 * Returns the team size actually used (the driver checks it equals nthreads), or -1 on a bad request.
 */
int gw_advance(const int64_t *ip, const double *dp, const uint64_t *key,
               double *X, double *Y, double *Mh, double *Ch,
               double *fx, double *fy, int32_t *jb, double *bias,
               const double *zbuf, double *fctr, double *acc)
{
    const int64_t N = ip[0], nb = ip[1], step0 = ip[3], step1 = ip[4];
    const int T = (int)ip[2], mode = (int)ip[5];
    const int replicated = (int)(ip[7] & 1), dissem = (int)((ip[7] >> 1) & 1);
    if (N < 1 || nb < 1 || nb > 8192 || T < 1 || T > GW_MAXT || step1 < step0 || (mode != 0 && mode != 1)
        || ip[7] < 0 || ip[7] > 3 || ip[6] < 0 || ip[6] > 32)
        return -1;
    const int timers = (ip[6] != 0);
    const int smp_all = (ip[6] == 1);                       /* every step */
    const int smp_shift = 64 - (int)ip[6];                  /* gap - 1 = mix64 >> (64 - m), on [0, 2^m) */
    const int64_t smp_next0 = (int64_t)acc[12];             /* next sampled step (carried across calls) */
    int64_t smp_next_end = smp_next0;
    if (step1 == step0)
        return T;
    gw_par P;
    const double h = dp[0], beta = dp[1], s = dp[5];
    P.h = h;
    P.H = dp[2];
    P.oout = dp[3];
    P.min_count = dp[6];
    P.amp = sqrt(2.0 * h / beta);               /* numba: math.sqrt(2.0 * h / beta) = sqrt((2.0*h)/beta) */
    P.two_s2 = 2.0 * s * s;                     /* (2.0*s)*s */
    P.s2 = s * s;
    P.dom_amp = dp[4] - dp[3];                  /* oin - oout */
    P.delta = (GW_XMAX - GW_XMIN) / (double)nb;
    P.nb = nb;
    const uint64_t k0 = key[0], k1 = key[1];
    const double mb0 = fctr[0];
    const long long ep0 = g_episode;
    long long ep_end = ep0;
    int nth_used = 0;
    const uint64_t t_launch = timers ? __rdtsc() : 0;

#pragma omp parallel num_threads(T)
    {
        const int tid = omp_get_thread_num();
        const int nth = omp_get_num_threads();
        long long ep = ep0;
        /* exact per-call stamps; sampled phase tick sums */
        uint64_t t_start = 0, t_cp = 0, t_l0 = 0, t_l1 = 0, t_end = 0;
        uint64_t s_fm = 0, s_dep = 0, s_bias = 0, s_sync = 0, n_smp = 0;
        if (timers)
            t_start = __rdtsc();
        if (tid == 0)
            nth_used = nth;
        int64_t w0, w1;
        gw_wpart(N, nth, tid, &w0, &w1);
        const int32_t b0 = (int32_t)(nb * tid / nth), b1 = (int32_t)(nb * (tid + 1) / nth);
        double mb = mb0;
        /* replicated variant: this thread's private copy of the table (Ch at [0, nb), Mh at [nb, 2 nb)) */
        double rep[replicated ? 2 * nb : 1];
        if (replicated) {
            for (int64_t j = 0; j < nb; j++) {
                rep[j] = Ch[j];
                rep[nb + j] = Mh[j];
            }
        }
        double *const Cr = replicated ? rep : Ch;
        double *const Mr = replicated ? rep + nb : Mh;
        if (timers)
            t_cp = __rdtsc();

        {
            double *f0 = fx + (step0 & 1) * N;
            int32_t *j0 = jb + (step0 & 1) * N;
            for (int64_t i = w0; i < w1; i++)
                gw_force_bin(&P, X[i], Y[i], &f0[i], &fy[i], &j0[i]);
        }
        if (timers)
            t_l0 = __rdtsc();

        {
            const gw_thr S = {.P = &P, .N = N, .step0 = step0, .step1 = step1, .w0 = w0, .w1 = w1, .b0 = b0,
                              .b1 = b1, .nth = nth, .tid = tid, .dissem = dissem, .replicated = replicated,
                              .mode = mode, .k0 = k0, .k1 = k1, .X = X, .Y = Y, .Mh = Mh, .Ch = Ch, .Cr = Cr,
                              .Mr = Mr, .fx = fx, .fy = fy, .bias = bias, .jb = jb, .zbuf = zbuf};
            gw_ticks tk = {0, 0, 0, 0, 0};
            if (!timers) {
                gw_run_untimed(S, step0, step1, &mb, &ep);
            } else {
                /* untimed runs between the sampled steps (the same sequence on every thread) */
                int64_t step = step0, nsmp = smp_next0;
                while (step < step1) {
                    const int64_t e = (nsmp < step1) ? (nsmp > step ? nsmp : step) : step1;
                    if (e > step)
                        gw_run_untimed(S, step, e, &mb, &ep);
                    step = e;
                    if (step < step1) {                      /* step == nsmp (or a stale nsmp < step0) */
                        gw_step(1, &S, step, &mb, &ep, &tk);
                        nsmp = step + 1 + (smp_all ? 0 : (int64_t)(gw_mix64((uint64_t)step) >> smp_shift));
                        step++;
                    }
                }
                if (tid == 0)
                    smp_next_end = nsmp;
            }
            s_fm = tk.fm;
            s_dep = tk.dep;
            s_bias = tk.bias;
            s_sync = tk.sync;
            n_smp = tk.n;
        }
        if (timers)
            t_l1 = __rdtsc();
        if (replicated && tid == 0) {
            for (int64_t j = 0; j < nb; j++) {
                Ch[j] = Cr[j];
                Mh[j] = Mr[j];
            }
        }
        g_mb[tid * GW_PAD] = mb;
        if (tid == 0)
            ep_end = ep;
        if (timers) {
            t_end = __rdtsc();                    /* end of this thread's work (join wait measured from here) */
            uint64_t *tk = g_tk + tid * GW_PAD;
            tk[0] = t_start - t_launch;           /* region launch wait */
            tk[1] = (t_cp - t_start) + (t_end - t_l1);   /* replicated-table copy in / copy back */
            tk[2] = t_l0 - t_cp;                  /* entry force/bin */
            tk[3] = t_l1 - t_l0;                  /* step loop (exact) */
            tk[4] = s_fm;
            tk[5] = s_dep;
            tk[6] = s_bias;
            tk[7] = s_sync;
            tk[8] = t_end;
            tk[9] = n_smp;
        }
    }
    const uint64_t t_join = timers ? __rdtsc() : 0;
    g_episode = ep_end;                            /* every thread executed the same number of barriers */
    double mb = mb0;
    for (int t = 0; t < nth_used; t++) {
        const double v = g_mb[t * GW_PAD];
        if (v > mb)
            mb = v;
    }
    fctr[0] = mb;
    if (timers) {
        double e_fb = 0.0, e_wait = 0.0, e_loop = 0.0, e_cp = 0.0, s_fm = 0.0, s_dep = 0.0, s_bias = 0.0,
               s_sync = 0.0;
        for (int t = 0; t < nth_used; t++) {
            const uint64_t *tk = g_tk + t * GW_PAD;
            e_wait += (double)tk[0] + (double)(t_join - tk[8]);
            e_cp += (double)tk[1];
            e_fb += (double)tk[2];
            e_loop += (double)tk[3];
            s_fm += (double)tk[4];
            s_dep += (double)tk[5];
            s_bias += (double)tk[6];
            s_sync += (double)tk[7];
        }
        const double inv = 1.0 / (double)nth_used;
        acc[0] += e_fb * inv;
        acc[1] += e_wait * inv;
        acc[2] += e_loop * inv;
        acc[3] += s_fm * inv;
        acc[4] += s_dep * inv;
        acc[5] += s_bias * inv;
        acc[6] += s_sync * inv;
        acc[7] += (double)(t_join - t_launch);
        acc[10] += (double)g_tk[9];               /* sampled steps (the same on every thread) */
        acc[11] += e_cp * inv;
        acc[12] = (double)smp_next_end;           /* exact: steps < 2^53 */
    }
    acc[8] += 1.0;
    acc[9] += (double)(step1 - step0);
    return nth_used;
}
