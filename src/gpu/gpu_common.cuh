// gpu_common.cuh -- device/host helpers shared by the A100 backends (src/gpu/gw_cuda.cu, src/gpu/lta_cuda.cu).
//
// NSCC A100 study (docs/nscc_a100/PREREGISTRATION.md).  Philox4x64-10 and Box-Muller are op-for-op the CPU
// counter mode of src/parallel (gw_kernel.c / philox.py): identical integers and uniforms; the normals differ from
// glibc's only through CUDA's log/sin/cos (<= 2 ulp).  PCG64 is numpy's PCG64 (XSL-RR 128/64), used on the host to
// draw the FR uniform blocks of the validated stream (Generator.random: (next64 >> 11) 2^-53).
// Build: nvcc -O3 -arch=sm_80 --fmad=false (no FMA contraction, IEEE div/sqrt), float64 throughout.
#pragma once
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <ctime>

namespace cg = cooperative_groups;

#define PHILOX_M0 0xD2E7470EE14C6C93ULL
#define PHILOX_M1 0xCA5A826395121157ULL
#define PHILOX_W0 0x9E3779B97F4A7C15ULL
#define PHILOX_W1 0xBB67AE8584CAA73BULL

__device__ __forceinline__ void philox4x64_10(unsigned long long c0, unsigned long long c1, unsigned long long c2,
                                              unsigned long long c3, unsigned long long k0, unsigned long long k1,
                                              unsigned long long o[4])
{
#pragma unroll
    for (int r = 0; r < 10; r++) {
        if (r > 0) {
            k0 += PHILOX_W0;
            k1 += PHILOX_W1;
        }
        const unsigned long long hi0 = __umul64hi(PHILOX_M0, c0), lo0 = PHILOX_M0 * c0;
        const unsigned long long hi1 = __umul64hi(PHILOX_M1, c2), lo1 = PHILOX_M1 * c2;
        c0 = hi1 ^ c1 ^ k0;
        c1 = lo1;
        c2 = hi0 ^ c3 ^ k1;
        c3 = lo0;
    }
    o[0] = c0; o[1] = c1; o[2] = c2; o[3] = c3;
}

__device__ __forceinline__ double u01(unsigned long long o)
{
    return (double)(o >> 11) * 0x1.0p-53;       // exact
}

__device__ __forceinline__ void box_muller(double ua, double ub, double &z0, double &z1)
{
    const double r = sqrt(-2.0 * log(1.0 - ua));
    const double th = (2.0 * M_PI) * ub;
    z0 = r * cos(th);
    z1 = r * sin(th);
}

// grid-wide barrier (cooperative launch); a single block needs only __syncthreads
__device__ __forceinline__ void grid_barrier(cg::grid_group &grid)
{
    if (gridDim.x > 1)
        grid.sync();
    else
        __syncthreads();
}

// block-wide exclusive scan of one int per thread (blockDim.x a multiple of 32, <= 1024); sh >= 33 ints
__device__ __forceinline__ int block_excl_scan(int v, int *sh, int *total)
{
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    int x = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, x, o);
        if (lane >= o)
            x += y;
    }
    if (lane == 31)
        sh[warp] = x;
    __syncthreads();
    if (warp == 0) {
        int w = (lane < nw) ? sh[lane] : 0;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int y = __shfl_up_sync(0xffffffffu, w, o);
            if (lane >= o)
                w += y;
        }
        sh[lane] = w;
    }
    __syncthreads();
    const int base = (warp > 0) ? sh[warp - 1] : 0;
    *total = sh[nw - 1];
    __syncthreads();                            // sh may be reused by the caller right away
    return base + x - v;
}

// deterministic block sum of one double per thread (fixed shuffle tree, then warp totals in warp order); sh >= 32
__device__ __forceinline__ double block_sum(double v, double *sh)
{
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, o);
    if (lane == 0)
        sh[warp] = v;
    __syncthreads();
    double t = 0.0;
    if (warp == 0) {
        t = (lane < nw) ? sh[lane] : 0.0;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            t += __shfl_down_sync(0xffffffffu, t, o);
        if (lane == 0)
            sh[0] = t;
    }
    __syncthreads();
    t = sh[0];
    __syncthreads();
    return t;
}

__device__ __forceinline__ double block_max(double v, double *sh)
{
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        v = fmax(v, __shfl_down_sync(0xffffffffu, v, o));
    if (lane == 0)
        sh[warp] = v;
    __syncthreads();
    double t = 0.0;
    if (warp == 0) {
        t = (lane < nw) ? sh[lane] : 0.0;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            t = fmax(t, __shfl_down_sync(0xffffffffu, t, o));
        if (lane == 0)
            sh[0] = t;
    }
    __syncthreads();
    t = sh[0];
    __syncthreads();
    return t;
}

__device__ __forceinline__ int block_sum_int(int v, int *sh)
{
    int tot;
    block_excl_scan(v, sh, &tot);
    return tot;
}

// Fixed-point deposits into shared memory WITHOUT 64-bit shared atomics (on sm_80 those compile to compare-and-swap
// loops): a 64-bit two's-complement value u is split into four 16-bit limbs, each accumulated in a 32-bit counter
// with native ATOMS.ADD (a per-step bin sum of <= 2^11 limbs of < 2^16 cannot overflow); the exact step sum is
// recombined as sum_k L_k 2^(16 k) mod 2^64.  Integer arithmetic: independent of every order.
__device__ __forceinline__ void limb_add(unsigned *sL, unsigned *sCn, int j, unsigned long long u, unsigned cnt)
{
    atomicAdd(&sL[4 * j + 0], (unsigned)(u & 0xFFFFULL));
    atomicAdd(&sL[4 * j + 1], (unsigned)((u >> 16) & 0xFFFFULL));
    atomicAdd(&sL[4 * j + 2], (unsigned)((u >> 32) & 0xFFFFULL));
    atomicAdd(&sL[4 * j + 3], (unsigned)((u >> 48) & 0xFFFFULL));
    atomicAdd(&sCn[j], cnt);
}

__device__ __forceinline__ long long limb_total(const unsigned *sL, int j)
{
    const unsigned long long t = (unsigned long long)sL[4 * j] + ((unsigned long long)sL[4 * j + 1] << 16) +
                                 ((unsigned long long)sL[4 * j + 2] << 32) + ((unsigned long long)sL[4 * j + 3] << 48);
    return (long long)t;
}

// Warp-aggregated deposit: every lane of the warp calls it (inactive lanes with active = false); lanes holding the
// same bin are summed limb by limb with full-warp __reduce_add_sync, then ONE lane per distinct bin adds the limb sums
// and the count (limb_add).
__device__ __forceinline__ void warp_agg_deposit(bool active, int j, long long q, unsigned *sL, unsigned *sCn)
{
    const unsigned FULL = 0xffffffffu;
    const int lane = threadIdx.x & 31;
    unsigned todo = __ballot_sync(FULL, active);
    while (todo) {
        const int l = __ffs(todo) - 1;
        const int jl = __shfl_sync(FULL, j, l);
        const bool mine = active && (j == jl);
        const unsigned peers = __ballot_sync(FULL, mine);
        const unsigned long long u = mine ? (unsigned long long)q : 0ULL;
        const unsigned s0 = __reduce_add_sync(FULL, (unsigned)(u & 0xFFFFULL));
        const unsigned s1 = __reduce_add_sync(FULL, (unsigned)((u >> 16) & 0xFFFFULL));
        const unsigned s2 = __reduce_add_sync(FULL, (unsigned)((u >> 32) & 0xFFFFULL));
        const unsigned s3 = __reduce_add_sync(FULL, (unsigned)((u >> 48) & 0xFFFFULL));
        if (lane == l) {
            atomicAdd(&sL[4 * jl + 0], s0);
            atomicAdd(&sL[4 * jl + 1], s1);
            atomicAdd(&sL[4 * jl + 2], s2);
            atomicAdd(&sL[4 * jl + 3], s3);
            atomicAdd(&sCn[jl], (unsigned)__popc(peers));
        }
        todo &= ~peers;
    }
}

// ------------------------------------------------------------------------------------------------ host helpers
static inline double host_now()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

// numpy PCG64 (pcg_setseq_128_xsl_rr_64): step, then output on the new state
struct Pcg64 {
    unsigned __int128 s, inc;
    inline unsigned long long next64()
    {
        const unsigned __int128 mult = ((unsigned __int128)0x2360ED051FC65DA4ULL << 64) | 0x4385DF649FCCF645ULL;
        s = s * mult + inc;
        const unsigned long long x = (unsigned long long)(s >> 64) ^ (unsigned long long)s;
        const unsigned rot = (unsigned)(s >> 122);
        return (x >> rot) | (x << ((-rot) & 63u));
    }
    inline double next_double() { return (double)(next64() >> 11) * (1.0 / 9007199254740992.0); }
};

struct GpuError {
    char msg[512];
};

#define CUDA_CHECK(call)                                                                                 \
    do {                                                                                                 \
        cudaError_t e_ = (call);                                                                         \
        if (e_ != cudaSuccess) {                                                                         \
            GpuError ge_;                                                                                \
            snprintf(ge_.msg, sizeof ge_.msg, "%s:%d %s: %s", __FILE__, __LINE__, #call,                 \
                     cudaGetErrorString(e_));                                                            \
            throw ge_;                                                                                   \
        }                                                                                                \
    } while (0)

static inline void gpu_fail(const char *what)
{
    GpuError ge;
    snprintf(ge.msg, sizeof ge.msg, "%s", what);
    throw ge;
}

// owning device buffer
template <typename T>
struct DBuf {
    T *p = nullptr;
    size_t n = 0;
    void alloc(size_t count)
    {
        n = count > 0 ? count : 1;
        CUDA_CHECK(cudaMalloc((void **)&p, n * sizeof(T)));
        CUDA_CHECK(cudaMemset(p, 0, n * sizeof(T)));
    }
    ~DBuf()
    {
        if (p)
            cudaFree(p);
    }
};

// owning pinned host buffer
template <typename T>
struct HBuf {
    T *p = nullptr;
    size_t n = 0;
    void alloc(size_t count)
    {
        n = count > 0 ? count : 1;
        CUDA_CHECK(cudaMallocHost((void **)&p, n * sizeof(T)));
        memset(p, 0, n * sizeof(T));
    }
    ~HBuf()
    {
        if (p)
            cudaFreeHost(p);
    }
};
