"""Mollified SHUS accumulator (the ABP core), batched over a leading run axis.

Frozen conventions (docs/SPEC_SHUS_FR.md):

* Bias sign:            V_t(q) = V(q) - F_t(xi(q)),  so F_t -> F + const flattens xi.
* Estimator:            F_t = -beta^{-1} log R_t   on the grid.
* Deposit weight:       e^{-beta F_n(xi)} = R_n(xi), with F_n (equivalently R_n) FROZEN
                        over the adaptation block that is being deposited.  In this gauge
                        the reweighting factor is literally the interpolated accumulator.
* Update:               R_{n+1} = R_n + g_shus * (dt / K) * mollify(sum_of_deposits),
                        with g_shus the per-row adaptation gain (1 = frozen baseline).
                        then R <- R / max(R)  (per row).  The renormalization is a pure
                        gauge change: every future deposit weight scales by the same
                        factor, so the entire R trajectory is scale-invariant and the
                        rescaling only prevents floating-point overflow (SHUS mass grows
                        exponentially once the marginal is flat).
* Estimator protection: nothing in this class is called at an FR event.  Resampling
                        gathers walker arrays only; tests/test_fisher_rao.py enforces it.

The mollifier delta_eps is the Gaussian grid kernel of grid.gaussian_kernel (deposits are
scattered to the nearest grid point, then the block histogram is convolved once -- the
same discretization the validated ABF-Fisher-Rao engine used for densities).
"""
from __future__ import annotations

import torch

from .grid import (EPS, Grid1D, binned_density, central_diff, gaussian_kernel,
                   interp1d, nearest_bin, smooth)


class ShusAccumulator:
    """State: R (rows, G) accumulator and the current block's weighted deposit buffer.

    ``beta`` is a (rows, 1) tensor so one batch can carry a parameter sweep.
    """

    def __init__(self, rows: int, grid: Grid1D, beta: torch.Tensor, eps_bw: float,
                 device, dtype, gain=None):
        self.grid = grid
        self.beta = beta.reshape(rows, 1).to(device=device, dtype=dtype)
        self.kernel, self.krad = gaussian_kernel(eps_bw, grid.dx, device, dtype)
        self.R = torch.ones((rows, grid.n), device=device, dtype=dtype)
        self.buf = torch.zeros((rows, grid.n), device=device, dtype=dtype)
        # adaptation gain g_SHUS: per-row multiplier on the accumulator increment.
        # Gauge-compatible for any positive per-row constant (increment and deposit
        # weights both scale linearly in R) and fixed-point-shape-preserving (it
        # only rescales the approach rate).  gain=None is the frozen g=1 baseline.
        if gain is None:
            self.gain = torch.ones((rows, 1), device=device, dtype=dtype)
        else:
            self.gain = gain.reshape(rows, 1).to(device=device, dtype=dtype)
            assert bool((self.gain > 0).all()), "g_shus must be positive"
        self._refresh_bias()

    # -- bias seen by the dynamics -------------------------------------------------
    def _refresh_bias(self):
        self.F = -torch.log(torch.clamp(self.R, min=EPS)) / self.beta
        self.Fp = central_diff(self.F, self.grid.dx)

    def bias_force_at(self, X):
        """+dF_n/dx at walker positions: the x-force contribution of -grad(V - F_n)."""
        return interp1d(X, self.Fp, self.grid)

    # -- estimator ------------------------------------------------------------------
    def deposit(self, X):
        """Deposit one step of physically propagated positions into the block buffer.

        Weight is the block-frozen e^{-beta F_n(xi)} = R_n(xi); R_n only changes in
        update(), so all deposits inside a block see the same gauge.
        """
        w = interp1d(X, self.R, self.grid)
        idx = nearest_bin(X, self.grid)
        self.buf.scatter_add_(1, idx, w)

    def update(self, dt: float, K: int):
        """Close the adaptation block: mollify the buffer, accumulate, renormalize.

        Returns the raw (pre-renormalization) increment Delta R_n, which the engine
        uses for the deposition-feedback diagnostic d_n = Delta R_n / ||Delta R_n||:
        healthy SHUS deposits d_n ~ exp(-beta F); an over-flattened population
        deposits d_n ~ R_n (rich-get-richer feedback).
        """
        inc = smooth(self.buf, self.kernel, self.krad, self.grid.dx) \
            * (dt / K) * self.gain
        self.R = self.R + inc
        self.R = self.R / self.R.max(dim=1, keepdim=True).values
        self.buf.zero_()
        self._refresh_bias()
        return inc

    # -- reporting / persistence -----------------------------------------------------
    def f_estimate(self, eval_mask):
        """F_hat centered on the eval window (reporting gauge)."""
        return self.F - self.F[:, eval_mask].mean(dim=1, keepdim=True)

    def state_dict(self):
        return {"R": self.R.clone(), "buf": self.buf.clone()}

    def load_state_dict(self, sd):
        self.R = sd["R"].clone().to(self.R)
        self.buf = sd["buf"].clone().to(self.buf)
        self._refresh_bias()


def biased_marginal_estimate(X, eta_bw: float, grid: Grid1D):
    """KDE p_hat of the current walker reaction-coordinate values (bandwidth eta)."""
    k, r = gaussian_kernel(eta_bw, grid.dx, X.device, X.dtype)
    return binned_density(X, k, r, grid)
