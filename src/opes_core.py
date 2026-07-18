"""Engine-agnostic OPES_METAD (On-the-fly Probability Enhanced Sampling).

A single, self-contained implementation of the well-tempered OPES bias on a 1-D
reaction-coordinate (CV) grid.  It is deliberately decoupled from any physical
engine: it consumes CV values ``z`` (one per walker) and returns a biasing
*mean force* along the CV, ``-dA_n/dz``, that each engine applies through its own
existing bias channel (WCA ``add_abf_force``; the toys' ``interp1d`` bias term).
It reasons through the CV marginal like the Fisher--Rao correction, but acts
through a smooth adaptive bias instead of birth--death resampling.

Algorithm (Invernizzi & Parrinello, JPCL 2020; PLUMED ``OPES_METAD``)
--------------------------------------------------------------------
A weighted kernel-density estimate of the *unbiased* CV marginal is grown online.
At deposition step ``n`` each walker sample ``z_k`` enters with reweight factor
``w_k = exp(beta * A_{n-1}(z_k))`` (the bias felt so far), so the accumulated
estimate approximates the Boltzmann CV marginal ``P(z) ~ exp(-beta F(z))``:

    p_tilde_n(z) = sum_k w_k K_sigma(z - z_k) / sum_k w_k .

The well-tempered bias is

    A_n(z) = (1 - 1/gamma) * beta^{-1} * log( p_tilde_n(z) / Z_n + epsilon ),

with the regularization floor ``epsilon = exp(-beta * BARRIER / (1 - 1/gamma))``
(PLUMED's ``BARRIER`` parameterization) and ``Z_n`` the mean of ``p_tilde_n`` over
the so-far-explored region.  The applied biasing force along the CV is ``-A_n'(z)``.
For ``gamma -> inf`` the prefactor ``(1 - 1/gamma) -> 1`` and the target becomes
flat (uniform) -- the ``OPES`` flat-target ablation.

Native free-energy estimate: because ``p_tilde_n`` already IS the reweighted
(unbiased) CV marginal, the OPES free energy is ``F_hat_n(z) = -beta^{-1} log
p_tilde_n(z)`` (up to a constant), directly comparable to the ABF/mFR estimate on
the same grid.  A common frozen-bias evaluator can instead freeze the applied bias
``(1 - 1/gamma) F`` and reconstruct F from the biased marginal.

No reference free energy is ever consulted here (no-leakage): see
``assert_no_reference_leakage``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field
from typing import Optional

import torch

EPS = 1.0e-12
DEFAULT_DTYPE = torch.float32


@dataclass
class OPESConfig:
    """Hyperparameters for OPES_METAD on a 1-D CV grid.

    ``barrier`` (ΔE, the primary knob), ``pace`` and ``sigma`` are the three main
    parameters; the rest are secondary and left at PLUMED-like defaults.  All are
    metadata-serialisable (floats/ints/str) so a spec hash is stable.
    """
    # --- CV grid (must match the engine's evaluation grid for a fair compare) ---
    z_min: float = -0.2
    z_max: float = 1.2
    n_grid: int = 160
    beta: float = 1.0

    # --- primary OPES knobs ---
    barrier: float = 4.0            # BARRIER: free-energy barrier to overcome (energy units)
    pace: int = 500                 # deposit a batch of kernels every `pace` steps
    sigma: float = 0.05             # kernel bandwidth in CV units (>0), or use adaptive
    sigma_mode: str = "fixed"       # "fixed" | "adaptive"

    # --- well-tempering ---
    gamma: float = float("inf")     # bias factor; inf => flat (uniform) target ablation.
                                    # If <=0 or inf, treated as flat-target (prefactor 1).
    gamma_from_barrier: bool = True # if True and gamma not set finite, gamma = beta*barrier

    # --- kernel management (secondary; tuned only around the shortlist) ---
    compression_threshold: float = 1.0   # merge kernels closer than this * sigma (grid mode: n/a)
    sigma_min_factor: float = 0.0         # floor on adaptive sigma as a fraction of `sigma`
    adaptive_sigma_stride_mult: int = 10  # ADAPTIVE_SIGMA_STRIDE = mult * pace
    epsilon_floor: float = 0.0            # 0 => use PLUMED formula exp(-beta*barrier/(1-1/gamma))

    # --- application ---
    bias_force_clip: float = 40.0   # clip |−A'_n(z)| like the ABF force clip
    warmup_steps: int = 0           # ramp the applied bias linearly over this many steps
    fill_edges: bool = True         # edge-extrapolate the bias profile at grid ends

    def effective_gamma(self) -> float:
        """Resolve the well-tempering factor. inf/<=1 => flat target (prefactor 1)."""
        g = self.gamma
        if (g is None) or (g == float("inf")) or (g <= 1.0):
            if self.gamma_from_barrier and math.isinf(g):
                # BARRIER-derived bias factor: gamma = beta * BARRIER (PLUMED default heuristic).
                gb = self.beta * self.barrier
                return gb if gb > 1.0 else float("inf")
            return float("inf")
        return float(g)

    def config_hash(self) -> str:
        import hashlib, json
        return hashlib.md5(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]


def _gaussian_kernel(diff, bandwidth):
    bw = float(max(bandwidth, EPS))
    return torch.exp(-0.5 * (diff / bw) ** 2) / (bw * math.sqrt(2.0 * math.pi))


def _trapz(y, x):
    return torch.sum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1]))


class OPESState:
    """Online OPES_METAD bias on a fixed 1-D CV grid (grid = implicit kernel store).

    Grid accumulation keeps memory O(n_grid) irrespective of trajectory length or
    walker count -- essential for 1024 GPU walkers over 250k steps -- and reuses the
    reflected-Gaussian-KDE / trapezoid conventions of the ABF/mFR cores so the OPES
    marginal lives on the same grid with the same normalization (fair comparison).
    """

    def __init__(self, cfg: OPESConfig, device, dtype=DEFAULT_DTYPE):
        self.cfg = cfg
        self.device = device
        self.dtype = dtype
        self.grid = torch.linspace(cfg.z_min, cfg.z_max, cfg.n_grid, device=device, dtype=dtype)
        self.width = float(cfg.z_max - cfg.z_min)
        self.beta = float(cfg.beta)
        self.gamma = cfg.effective_gamma()
        # well-tempering prefactor (1 - 1/gamma); flat target => 1.0
        self.prefactor = 1.0 if math.isinf(self.gamma) else (1.0 - 1.0 / self.gamma)
        # regularization floor epsilon = exp(-beta*BARRIER/(1-1/gamma)); flat => exp(-beta*BARRIER)
        if cfg.epsilon_floor > 0.0:
            self.epsilon = float(cfg.epsilon_floor)
        else:
            denom = max(self.prefactor, EPS)
            self.epsilon = float(math.exp(-self.beta * cfg.barrier / denom))
        # weighted-KDE accumulators
        self.num = torch.zeros(cfg.n_grid, device=device, dtype=dtype)  # sum_k w_k K(grid - z_k)
        self.wsum = torch.zeros((), device=device, dtype=torch.float64)  # sum_k w_k
        self.w2sum = torch.zeros((), device=device, dtype=torch.float64) # sum_k w_k^2 (reweight ESS)
        self.n_deposits = 0
        self.n_samples = 0
        # adaptive-sigma running estimate (EMA of the CV std), seeded lazily
        self.sigma_cur = float(cfg.sigma)
        self._sig_ema = None
        # cached profiles (rebuilt after each deposit)
        self._rho = None          # normalized CV marginal (integrates to 1)
        self._bias = None         # applied bias potential A_n(z) on the grid
        self._bias_force = None   # -dA_n/dz on the grid
        self._rebuild_profiles()

    # -- reflected weighted KDE contribution of a batch of samples --------------
    def _kde_num(self, z_samples, weights, bandwidth):
        zc = z_samples.to(self.dtype)
        wc = weights.to(self.dtype)
        lo, hi = self.cfg.z_min, self.cfg.z_max
        reflected_z = torch.cat([zc, 2.0 * lo - zc, 2.0 * hi - zc])
        reflected_w = torch.cat([wc, wc, wc])
        diff = self.grid[:, None] - reflected_z[None, :]
        k = _gaussian_kernel(diff, bandwidth)          # (n_grid, 3B)
        return torch.sum(k * reflected_w[None, :], dim=1)

    # -- rebuild the normalized marginal, bias potential and bias force ---------
    def _rebuild_profiles(self):
        if self.n_deposits == 0 or float(self.wsum) <= EPS:
            # no data yet: flat marginal, zero bias
            self._rho = torch.full_like(self.grid, 1.0 / max(self.width, EPS))
            self._bias = torch.zeros_like(self.grid)
            self._bias_force = torch.zeros_like(self.grid)
            return
        raw = self.num / float(self.wsum)                    # ~ weighted kernel density
        raw = torch.clamp(raw, min=0.0)
        mass = _trapz(raw, self.grid)
        rho = raw / torch.clamp(mass, min=EPS)               # normalized: integral = 1
        self._rho = torch.clamp(rho, min=EPS)
        # dimensionless density relative to the uniform level 1/width => O(1) in basins
        u = self._rho * self.width
        # applied bias potential A_n(z) = prefactor * beta^{-1} * log(u + epsilon)
        A = self.prefactor * (1.0 / self.beta) * torch.log(u + self.epsilon)
        # anchor additive constant so the deepest-explored region has A ~ 0 (cosmetic;
        # forces/weights are invariant to this shift)
        A = A - A.max()
        self._bias = A
        # bias force = -dA/dz (central differences; edges one-sided)
        f = torch.empty_like(A)
        f[1:-1] = -(A[2:] - A[:-2]) / (self.grid[2:] - self.grid[:-2])
        f[0] = -(A[1] - A[0]) / (self.grid[1] - self.grid[0])
        f[-1] = -(A[-1] - A[-2]) / (self.grid[-1] - self.grid[-2])
        self._bias_force = torch.clamp(f, -self.cfg.bias_force_clip, self.cfg.bias_force_clip)

    def _weights_for(self, z_samples):
        """Reweight factor w = (u_{n-1}(z)+eps)^{prefactor} = exp(beta*A_{n-1}(z)).

        Uses the CURRENT bias (built from all prior deposits); on the first deposit
        the bias is zero so all weights are 1 (uniform), as in PLUMED.
        """
        if self.n_deposits == 0:
            return torch.ones_like(z_samples)
        A_at = self.evaluate_bias(z_samples)
        return torch.exp(torch.clamp(self.beta * A_at, max=50.0))

    def _update_sigma(self, z_samples):
        if self.cfg.sigma_mode != "adaptive":
            return float(self.cfg.sigma)
        s = float(torch.std(z_samples).item()) if z_samples.numel() > 1 else self.sigma_cur
        if not math.isfinite(s) or s <= 0:
            s = self.sigma_cur
        self._sig_ema = s if self._sig_ema is None else 0.9 * self._sig_ema + 0.1 * s
        floor = self.cfg.sigma_min_factor * float(self.cfg.sigma)
        self.sigma_cur = max(self._sig_ema, floor, EPS)
        return self.sigma_cur

    def deposit(self, z_samples):
        """Add a batch of walker CV samples to the weighted KDE and refresh the bias."""
        z_samples = z_samples.detach().to(self.dtype)
        w = self._weights_for(z_samples).detach()
        bw = self._update_sigma(z_samples)
        self.num += self._kde_num(z_samples, w, bw)
        self.wsum += float(w.sum().item())
        self.w2sum += float((w.to(torch.float64) ** 2).sum().item())
        self.n_deposits += 1
        self.n_samples += int(z_samples.numel())
        self._rebuild_profiles()

    # -- per-walker lookups (interp on the grid, edge-safe) --------------------
    def _interp(self, profile, z):
        dz = self.grid[1] - self.grid[0]
        x = (z - self.grid[0]) / dz
        i0 = torch.floor(x).long()
        i0c = i0.clamp(0, self.grid.numel() - 2)
        frac = (x - i0c.to(z.dtype)).clamp(0.0, 1.0)
        val = (1.0 - frac) * profile[i0c] + frac * profile[i0c + 1]
        if self.cfg.fill_edges:
            val = torch.where(z < self.grid[0], profile[0].expand_as(val), val)
            val = torch.where(z > self.grid[-1], profile[-1].expand_as(val), val)
        else:
            inside = (i0 >= 0) & (i0 < self.grid.numel() - 1)
            val = torch.where(inside, val, torch.zeros_like(val))
        return val

    def evaluate_bias(self, z):
        return self._interp(self._bias, z)

    def bias_force_at(self, z, step=None):
        """Applied biasing mean force -A'_n(z) per walker, with optional warmup ramp."""
        f = self._interp(self._bias_force, z)
        if step is not None and self.cfg.warmup_steps > 0:
            f = f * min(1.0, float(step) / float(self.cfg.warmup_steps))
        return f

    # -- native OPES estimates on the grid -------------------------------------
    def free_energy(self):
        """Native OPES free energy F_hat(z) = -beta^{-1} log rho(z), centred at midpoint."""
        Fz = -(1.0 / self.beta) * torch.log(torch.clamp(self._rho, min=EPS))
        idx = torch.argmin(torch.abs(self.grid - 0.5))
        return Fz - Fz[idx]

    def mean_force(self):
        F = self.free_energy()
        out = torch.empty_like(F)
        out[1:-1] = (F[2:] - F[:-2]) / (self.grid[2:] - self.grid[:-2])
        out[0] = (F[1] - F[0]) / (self.grid[1] - self.grid[0])
        out[-1] = (F[-1] - F[-2]) / (self.grid[-1] - self.grid[-2])
        return out

    def marginal(self):
        return self._rho.clone()

    def applied_bias(self):
        return self._bias.clone()

    def diagnostics(self):
        ws = float(self.wsum); w2 = float(self.w2sum)
        neff = (ws * ws / w2) if w2 > 0 else 0.0
        nker = int(torch.count_nonzero(self.num > self.num.max() * 1e-4).item()) if self.num.max() > 0 else 0
        return dict(
            neff=neff,
            neff_frac=(neff / max(self.n_samples, 1)),
            n_kernels=nker,
            n_deposits=self.n_deposits,
            zed=float(self.wsum),
            sigma_cur=float(self.sigma_cur),
            max_bias=float(self._bias.max().item()),
            min_bias=float(self._bias.min().item()),
            bias_range=float((self._bias.max() - self._bias.min()).item()),
        )


def assert_no_reference_leakage(has_reference_target: bool, method: str = "opes"):
    """OPES must never receive the TI/quadrature reference as a bias input.

    The reference is used ONLY for post-hoc L2 evaluation. This mirrors the
    ``assert_no_oracle_leakage`` guards in the ABF/mFR cores.
    """
    if has_reference_target:
        raise ValueError(
            f"method {method!r} must not receive a reference free energy as a bias "
            "target; OPES builds its bias purely from online reweighted samples.")
