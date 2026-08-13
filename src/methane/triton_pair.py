"""Fused Triton kernel for the pair term: switched LJ + real-space PME, one pass, no intermediates.

Why this exists
---------------
The tensor-op pair path is memory-bandwidth-bound: at ``B = 512`` it materialises ~50 GB of
``(B, chunk, N)`` intermediates per force call and costs ~27 ms of a ~35 ms step, drawing ~370 W
of a 600 W H200.  Every tensor-level restructuring was measured and lost (neighbour list 655 vs
744 ns/day, LJ/Coulomb split 509 — see ``nonbonded.py``).  The only remaining lever is fusion:
one program per (walker, i-tile) sweeps all ``j`` in registers, so DRAM traffic drops to the
positions and forces themselves (~20 MB) and the kernel becomes compute-bound.

Exclusions by molecule id, and why that is exact
------------------------------------------------
The kernel skips a pair iff ``mol_id[i] == mol_id[j]``.  Methanes get unique ids, each water's
three sites share one id — so equal-id is *exactly* the self-pair plus the 3 x 512 intramolecular
water exclusions, i.e. precisely ``PairTerms.excluded``.  This is not assumed:
:func:`build_mol_id` reconstructs the mask from the ids and **hard-asserts elementwise equality**
with the OpenMM-derived mask at enable time.

Numerical contract
------------------
Same math as ``PairTerms.energy_forces`` in float32: same minimum image, same OpenMM switch
polynomial and its derivative window, same ``erfc`` real-space Coulomb (libdevice), same
``clamp_min(1e-24)`` floor.  Summation *order* differs (register accumulation vs chunked
reduction), so agreement is float32-reassociation-level, not bitwise; the acceptance gates in
``tests/test_methane_triton.py`` bound it against both the float32 torch path and the float64
ground truth.

Deployment: gated as a performance-only change, like ``torch.compile`` and float32 before it.
"""
from __future__ import annotations

import numpy as np
import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as tld

from .nonbonded import ONE_4PI_EPS0


@triton.jit
def _pair_kernel(X, F, EBUF, Q, SIG, EPS, MOLID,
                 N, L, inv_L, cutoff, cutoff2, switch, inv_width,
                 alpha, two_a_sqrtpi, C4PIEPS,
                 BI: tl.constexpr, BJ: tl.constexpr, GI: tl.constexpr):
    pid = tl.program_id(0)
    b = pid // GI
    ti = pid % GI

    offs_i = ti * BI + tl.arange(0, BI)
    mask_i = offs_i < N
    base = b * N
    xi = tl.load(X + (base + offs_i) * 3 + 0, mask=mask_i, other=0.0)
    yi = tl.load(X + (base + offs_i) * 3 + 1, mask=mask_i, other=0.0)
    zi = tl.load(X + (base + offs_i) * 3 + 2, mask=mask_i, other=0.0)
    qi = tl.load(Q + offs_i, mask=mask_i, other=0.0)
    si = tl.load(SIG + offs_i, mask=mask_i, other=0.0)
    ei = tl.load(EPS + offs_i, mask=mask_i, other=0.0)
    mi = tl.load(MOLID + offs_i, mask=mask_i, other=-1)

    fx = tl.zeros((BI,), dtype=tl.float32)
    fy = tl.zeros((BI,), dtype=tl.float32)
    fz = tl.zeros((BI,), dtype=tl.float32)
    e_acc = 0.0

    for tj in range(0, N, BJ):
        offs_j = tj + tl.arange(0, BJ)
        mask_j = offs_j < N
        xj = tl.load(X + (base + offs_j) * 3 + 0, mask=mask_j, other=0.0)
        yj = tl.load(X + (base + offs_j) * 3 + 1, mask=mask_j, other=0.0)
        zj = tl.load(X + (base + offs_j) * 3 + 2, mask=mask_j, other=0.0)
        qj = tl.load(Q + offs_j, mask=mask_j, other=0.0)
        sj = tl.load(SIG + offs_j, mask=mask_j, other=0.0)
        ej = tl.load(EPS + offs_j, mask=mask_j, other=0.0)
        mj = tl.load(MOLID + offs_j, mask=mask_j, other=-2)

        dx = xi[:, None] - xj[None, :]
        dy = yi[:, None] - yj[None, :]
        dz = zi[:, None] - zj[None, :]
        # minimum image; floor(t + 0.5) == round except exactly at half-integers, which the
        # physics never reaches (the CV guard keeps r < 0.98 * L/2)
        dx = dx - L * tl.floor(dx * inv_L + 0.5)
        dy = dy - L * tl.floor(dy * inv_L + 0.5)
        dz = dz - L * tl.floor(dz * inv_L + 0.5)

        r2 = dx * dx + dy * dy + dz * dz
        live = (mask_i[:, None] & mask_j[None, :] & (r2 < cutoff2)
                & (mi[:, None] != mj[None, :]))
        r = tl.sqrt(tl.maximum(r2, 1e-24))
        inv_r = tl.where(live, 1.0 / r, 0.0)

        # -- Lennard-Jones with the OpenMM switch --------------------------------------------
        sig = 0.5 * (si[:, None] + sj[None, :])
        eps = tl.sqrt(ei[:, None] * ej[None, :])
        p = sig * inv_r
        p2 = p * p
        p6 = p2 * p2 * p2
        p12 = p6 * p6
        e_lj = 4.0 * eps * (p12 - p6)
        dlj = -24.0 * eps * (2.0 * p12 - p6) * inv_r

        xs = (r - switch) * inv_width
        xs = tl.minimum(tl.maximum(xs, 0.0), 1.0)
        xs2 = xs * xs
        xs3 = xs2 * xs
        s = 1.0 - 10.0 * xs3 + 15.0 * xs3 * xs - 6.0 * xs3 * xs2
        ds = (-30.0 * xs2 + 60.0 * xs3 - 30.0 * xs3 * xs) * inv_width
        ds = tl.where((r > switch) & (r < cutoff), ds, 0.0)

        # -- PME real space ------------------------------------------------------------------
        qq = C4PIEPS * qi[:, None] * qj[None, :]
        ar = alpha * r
        erfc_ar = tld.erfc(ar)
        e_el = qq * erfc_ar * inv_r
        del_ = -qq * (erfc_ar * inv_r * inv_r + two_a_sqrtpi * tl.exp(-ar * ar) * inv_r)

        e_pair = tl.where(live, e_lj * s + e_el, 0.0)
        dE = tl.where(live, dlj * s + e_lj * ds + del_, 0.0)
        e_acc += tl.sum(e_pair)

        coef = dE * inv_r
        fx += -tl.sum(coef * dx, 1)
        fy += -tl.sum(coef * dy, 1)
        fz += -tl.sum(coef * dz, 1)

    tl.store(F + (base + offs_i) * 3 + 0, fx, mask=mask_i)
    tl.store(F + (base + offs_i) * 3 + 1, fy, mask=mask_i)
    tl.store(F + (base + offs_i) * 3 + 2, fz, mask=mask_i)
    tl.store(EBUF + pid, e_acc)


def build_mol_id(pair_terms):
    """Molecule ids whose equal-id mask is **provably** the exclusion mask.

    Each site starts as its own molecule; every exclusion pair is merged toward its smaller
    index (water triples are (O,H1),(O,H2),(H1,H2) with O smallest, so one sorted pass reaches
    the fixed point).  The reconstruction ``mol_id[i] == mol_id[j]`` (off-diagonal) plus the
    diagonal is then asserted elementwise against ``PairTerms.excluded`` — if the two ever
    disagree, enabling the kernel raises rather than silently mis-excluding.
    """
    n = pair_terms.n
    ex = pair_terms.exclusion_pairs.cpu().numpy()
    mol = np.arange(n, dtype=np.int64)
    for a, b in sorted(map(tuple, np.sort(ex, axis=1))):
        mol[b] = mol[a]
    rebuilt = torch.as_tensor(mol[:, None] == mol[None, :])
    rebuilt |= torch.eye(n, dtype=torch.bool)
    if not torch.equal(rebuilt, pair_terms.excluded.cpu()):
        raise RuntimeError("mol-id exclusion reconstruction does not match the OpenMM-derived "
                           "exclusion mask; the Triton kernel would mis-exclude pairs")
    return torch.as_tensor(mol, dtype=torch.int32, device=pair_terms.charge.device)


def pair_energy_forces_triton(pair_terms, x, mol_id, block_i=64, block_j=64, num_warps=4):
    """Drop-in replacement for ``PairTerms.energy_forces`` (float32, CUDA).

    Returns ``(E (B,), F (B, N, 3))`` with the same 0.5 ordered-pair energy convention.
    """
    B, N, _ = x.shape
    x = x.contiguous()
    F = torch.empty_like(x)
    GI = triton.cdiv(N, block_i)
    ebuf = torch.empty(B * GI, device=x.device, dtype=x.dtype)
    _pair_kernel[(B * GI,)](
        x, F, ebuf, pair_terms.charge, pair_terms.sigma, pair_terms.epsilon, mol_id,
        N, pair_terms.L, 1.0 / pair_terms.L, pair_terms.cutoff, pair_terms.cutoff ** 2,
        pair_terms.switch, 1.0 / (pair_terms.cutoff - pair_terms.switch),
        pair_terms.alpha, 2.0 * pair_terms.alpha / float(np.sqrt(np.pi)), ONE_4PI_EPS0,
        BI=block_i, BJ=block_j, GI=GI, num_warps=num_warps)
    return 0.5 * ebuf.view(B, GI).sum(-1), F
