"""Test systems  q = (x, y_1, ..., y_m)  with  xi(q) = x.

    V = U(x)                                   barrier profile in the CV
      + 1/2 omega(x)^2 y_1^2                   ACTIVE fiber coordinate
      + A(x) [ (y_1/c)^2 - 1 ]^2               optional hidden two-channel structure
      + delta(x) (y_1/c)                       x-dependent channel tilt
      + sum_{k>=2} 1/2 omega_s(x)^2 y_k^2      SPECTATOR fiber coordinates

  U(x)       'double_well': H (x^2-1)^2 | 'periodic': H/2 (1-cos(2 pi n_w (x-xmin)/L))
             | 'flat': 0
  omega(x)   = omega_out + (omega_in - omega_out) g(x)      g = normalized barrier shape
  omega_s(x) = oms_out  + (oms_in  - oms_out ) g(x)
  A(x)       = A_c [1 - (1-a_min) exp(-d(x,x_sw)^2 / 2 s_sw^2)]
  delta(x)   = delta0 tanh(x/s_delta)  or  delta0 sin(2 pi (x-xmin)/L)

Knob -> regime
  H              enthalpic barrier the adaptive bias must learn away
  omega_in/out   fiber relaxation time tau_fiber ~ 1/omega^2  (the lift-cost knob)
  L, n_wells     CV transport distance (the O(L) vs O(L^2) knob)
  A_c, a_min     hidden slow channel reachable only through a switch region at x_sw
  delta0         makes the CORRECT channel mixture x-dependent
  m_spec         SYSTEM SIZE: spectator dofs are thermodynamically trivial but they
                 make the replica-exchange energy gap grow like sqrt(m_spec), so
                 Hamiltonian-exchange acceptance decays while the RC-WFR lift does not

xi = x is linear with |grad xi| = 1, so there is no Fixman/Jacobian correction: the
fiber measure is the plain conditional in y, f = dV/dx, and F'(x) = E[dV/dx | X = x].
The spectator block contributes analytically:
    F_spec(x) = m_spec * beta^-1 * log omega_s(x),
    F'_spec(x) = m_spec * omega_s'(x) / (beta omega_s(x)).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import torch

from ..grid import DEVICE, DTYPE, EPS, Grid1D, reflect_into


@dataclass(frozen=True)
class SysParams:
    beta: float = 8.0
    barrier: str = "double_well"
    H: float = 2.5
    n_wells: int = 2
    omega_out: float = 1.0
    omega_in: float = 25.0
    s: float = 0.25
    A_c: float = 0.0
    c: float = 1.0
    a_min: float = 1.0
    x_sw: float = 0.0
    s_sw: float = 0.25
    delta0: float = 0.0
    s_delta: float = 0.5
    m_spec: int = 0
    oms_out: float = 1.0
    oms_in: float = 4.0
    y_max: float = 6.0
    n_yq: int = 4001


class SepSystem:
    def __init__(self, params: SysParams, grid: Grid1D, device=DEVICE, dtype=DTYPE):
        self.p = params
        self.grid = grid
        self.device, self.dtype = device, dtype
        self._L = grid.volume
        self._build_reference()

    # ---- profiles ----------------------------------------------------------
    def _theta(self, x):
        return 2.0 * torch.pi * self.p.n_wells * (x - self.grid.xmin) / self._L

    def U(self, x):
        p = self.p
        if p.barrier == "double_well":
            return p.H * (x * x - 1.0) ** 2
        if p.barrier == "periodic":
            return 0.5 * p.H * (1.0 - torch.cos(self._theta(x)))
        return torch.zeros_like(x)

    def dU(self, x):
        p = self.p
        if p.barrier == "double_well":
            return 4.0 * p.H * x * (x * x - 1.0)
        if p.barrier == "periodic":
            k = 2.0 * torch.pi * p.n_wells / self._L
            return 0.5 * p.H * k * torch.sin(self._theta(x))
        return torch.zeros_like(x)

    def _g_neck(self, x):
        if self.p.barrier == "periodic":
            return 0.5 * (1.0 - torch.cos(self._theta(x)))
        return torch.exp(-x * x / (2.0 * self.p.s ** 2))

    def _dg_neck(self, x):
        p = self.p
        if p.barrier == "periodic":
            k = 2.0 * torch.pi * p.n_wells / self._L
            return 0.5 * k * torch.sin(self._theta(x))
        return -(x / p.s ** 2) * torch.exp(-x * x / (2.0 * p.s ** 2))

    def omega(self, x):
        p = self.p
        return p.omega_out + (p.omega_in - p.omega_out) * self._g_neck(x)

    def domega(self, x):
        p = self.p
        return (p.omega_in - p.omega_out) * self._dg_neck(x)

    def omega_s(self, x):
        p = self.p
        return p.oms_out + (p.oms_in - p.oms_out) * self._g_neck(x)

    def domega_s(self, x):
        p = self.p
        return (p.oms_in - p.oms_out) * self._dg_neck(x)

    def _dist_x(self, x, x0):
        d = x - x0
        if self.grid.bc == "periodic":
            d = d - self._L * torch.round(d / self._L)
        return d

    def A_of(self, x):
        p = self.p
        if p.A_c == 0.0:
            return torch.zeros_like(x)
        d = self._dist_x(x, p.x_sw)
        return p.A_c * (1.0 - (1.0 - p.a_min) * torch.exp(-d * d / (2.0 * p.s_sw ** 2)))

    def dA_of(self, x):
        p = self.p
        if p.A_c == 0.0:
            return torch.zeros_like(x)
        d = self._dist_x(x, p.x_sw)
        return p.A_c * (1.0 - p.a_min) * (d / p.s_sw ** 2) * \
            torch.exp(-d * d / (2.0 * p.s_sw ** 2))

    def delta_of(self, x):
        p = self.p
        if p.delta0 == 0.0:
            return torch.zeros_like(x)
        if p.barrier == "periodic":
            return p.delta0 * torch.sin(2.0 * torch.pi * (x - self.grid.xmin) / self._L)
        return p.delta0 * torch.tanh(x / p.s_delta)

    def ddelta_of(self, x):
        p = self.p
        if p.delta0 == 0.0:
            return torch.zeros_like(x)
        if p.barrier == "periodic":
            k = 2.0 * torch.pi / self._L
            return p.delta0 * k * torch.cos(k * (x - self.grid.xmin))
        t = torch.tanh(x / p.s_delta)
        return p.delta0 * (1.0 - t * t) / p.s_delta

    # ---- potential (Y: (..., 1 + m_spec)) ----------------------------------
    def _split(self, Y):
        return Y[..., 0], Y[..., 1:]

    def V(self, X, Y):
        p = self.p
        y, S = self._split(Y)
        om = self.omega(X)
        v = self.U(X) + 0.5 * om * om * y * y
        if p.A_c != 0.0:
            u = y / p.c
            v = v + self.A_of(X) * (u * u - 1.0) ** 2 + self.delta_of(X) * u
        if p.m_spec > 0:
            oms = self.omega_s(X)
            v = v + 0.5 * (oms * oms).unsqueeze(-1) * (S * S)
            v = v.sum(-1) if v.dim() > X.dim() else v
        return v

    def energy(self, X, Y):
        """Total V; shape (R, N)."""
        p = self.p
        y, S = self._split(Y)
        om = self.omega(X)
        v = self.U(X) + 0.5 * om * om * y * y
        if p.A_c != 0.0:
            u = y / p.c
            v = v + self.A_of(X) * (u * u - 1.0) ** 2 + self.delta_of(X) * u
        if p.m_spec > 0:
            oms = self.omega_s(X)
            v = v + 0.5 * (oms * oms) * (S * S).sum(-1)
        return v

    def mean_force(self, X, Y):
        """f = dV/dx at (X, Y); shape (R, N)."""
        p = self.p
        y, S = self._split(Y)
        om, dom = self.omega(X), self.domega(X)
        g = self.dU(X) + om * dom * y * y
        if p.A_c != 0.0:
            u = y / p.c
            g = g + self.dA_of(X) * (u * u - 1.0) ** 2 + self.ddelta_of(X) * u
        if p.m_spec > 0:
            oms, doms = self.omega_s(X), self.domega_s(X)
            g = g + oms * doms * (S * S).sum(-1)
        return g

    def grad_y(self, X, Y):
        """dV/dy_k; shape (R, N, 1 + m_spec)."""
        p = self.p
        y, S = self._split(Y)
        om = self.omega(X)
        g0 = om * om * y
        if p.A_c != 0.0:
            u = y / p.c
            g0 = g0 + self.A_of(X) * 4.0 * u * (u * u - 1.0) / p.c + self.delta_of(X) / p.c
        if p.m_spec > 0:
            oms = self.omega_s(X)
            gs = (oms * oms).unsqueeze(-1) * S
            return torch.cat([g0.unsqueeze(-1), gs], dim=-1)
        return g0.unsqueeze(-1)

    # ---- reference ---------------------------------------------------------
    def _build_reference(self):
        p = self.p
        dev, dt = self.device, self.dtype
        xq = self.grid.x(dev, dt)
        yq = torch.linspace(-p.y_max, p.y_max, p.n_yq, device=dev, dtype=dt)
        X1, Y1 = xq.unsqueeze(1), yq.unsqueeze(0)
        om = self.omega(X1)
        Vact = self.U(X1) + 0.5 * om * om * Y1 * Y1
        if p.A_c != 0.0:
            u = Y1 / p.c
            Vact = Vact + self.A_of(X1) * (u * u - 1.0) ** 2 + self.delta_of(X1) * u
        w = -p.beta * Vact
        wmax = w.max(dim=1, keepdim=True).values
        e = torch.exp(w - wmax)
        dy = float(yq[1] - yq[0])
        Zc = torch.trapezoid(e, dx=dy, dim=1)
        F = -(torch.log(torch.clamp(Zc, min=EPS)) + wmax.squeeze(1)) / p.beta
        dom = self.domega(X1)
        fact = self.dU(X1) + om * dom * Y1 * Y1
        if p.A_c != 0.0:
            u = Y1 / p.c
            fact = fact + self.dA_of(X1) * (u * u - 1.0) ** 2 + self.ddelta_of(X1) * u
        dF = torch.trapezoid(fact * e, dx=dy, dim=1) / torch.clamp(Zc, min=EPS)
        if p.m_spec > 0:                       # analytic Gaussian spectator block
            oms, doms = self.omega_s(xq), self.domega_s(xq)
            F = F + p.m_spec * torch.log(oms) / p.beta
            dF = dF + p.m_spec * doms / (p.beta * oms)
        mask = self.grid.eval_mask(dev, dt)
        self.F_ref = (F - F[mask].mean()).unsqueeze(0)
        self.dF_ref = dF.unsqueeze(0)
        pdf = e / torch.clamp(Zc, min=EPS).unsqueeze(1)
        cdf = torch.cumulative_trapezoid(pdf, dx=dy, dim=1)
        cdf = torch.cat([torch.zeros((cdf.shape[0], 1), device=dev, dtype=dt), cdf], 1)
        self._cdf = cdf / torch.clamp(cdf[:, -1:], min=EPS)
        self._yq, self._pdf = yq, pdf
        tail = float(torch.max(pdf[:, 0] + pdf[:, -1]))
        assert tail < 1e-8, f"y_max too small: edge conditional density {tail:.3e}"
        dyq = float(yq[1] - yq[0])
        self.p_channel_ref = torch.trapezoid(pdf[:, yq > 0], dx=dyq, dim=1)
        self.tau_fiber_ref = 1.0 / (self.omega(xq) ** 2)

    # ---- oracle conditional refresh ---------------------------------------
    def sample_conditional(self, X, gen, block=65536):
        """Exact Y ~ nu^xi(. | X); returns (R, N, 1 + m_spec).

        The (M, Gy) interpolated-CDF tensor is materialized in blocks so that a
        large oracle draw cannot exhaust device memory.
        """
        g, p = self.grid, self.p
        shp = X.shape
        xf = X.reshape(-1)
        M = xf.numel()
        yout = torch.empty(M, device=X.device, dtype=X.dtype)
        for a in range(0, M, block):
            xb = xf[a:a + block]
            pos = torch.clamp((xb - g.xmin) / g.dx, 0.0, g.n - 1.0)
            i0 = torch.clamp(torch.floor(pos).long(), 0, g.n - 2)
            frac = (pos - i0.to(X.dtype)).unsqueeze(-1)
            cdf = self._cdf[i0] + frac * (self._cdf[i0 + 1] - self._cdf[i0])
            u = torch.rand(xb.shape, device=X.device, dtype=X.dtype, generator=gen)
            j = torch.clamp(torch.searchsorted(cdf.contiguous(),
                                               u.unsqueeze(-1).contiguous()),
                            1, cdf.shape[-1] - 1)
            lo = torch.gather(cdf, -1, j - 1).squeeze(-1)
            hi = torch.gather(cdf, -1, j).squeeze(-1)
            t = (u - lo) / torch.clamp(hi - lo, min=EPS)
            y0, y1 = self._yq[(j - 1).squeeze(-1)], self._yq[j.squeeze(-1)]
            yout[a:a + block] = y0 + t * (y1 - y0)
        y = yout.reshape(*shp).unsqueeze(-1)
        if p.m_spec == 0:
            return y
        sd = 1.0 / ((p.beta ** 0.5) * self.omega_s(X)).unsqueeze(-1)
        S = sd * torch.randn((*shp, p.m_spec), device=X.device,
                             dtype=X.dtype, generator=gen)
        return torch.cat([y, S], dim=-1)

    # ---- dynamics ----------------------------------------------------------
    def clamp_y(self, Y):
        return reflect_into(Y, -self.p.y_max, self.p.y_max)

    def _noise(self, shape, dt, gen, device, dtype):
        return ((2.0 * dt / self.p.beta) ** 0.5) * torch.randn(
            shape, device=device, dtype=dtype, generator=gen)

    def step_fiber(self, X, Y, dt, gen):
        return self.clamp_y(Y - dt * self.grad_y(X, Y)
                            + self._noise(Y.shape, dt, gen, Y.device, Y.dtype))

    def step_full(self, X, Y, dt, gen, bias_force_x=None):
        gx = self.mean_force(X, Y)
        if bias_force_x is not None:
            gx = gx - bias_force_x
        Xn = self.grid.enforce(X - dt * gx
                               + self._noise(X.shape, dt, gen, X.device, X.dtype))
        Yn = self.step_fiber(X, Y, dt, gen)
        return Xn, Yn

    def params_dict(self):
        return asdict(self.p)


Sep2DParams = SysParams          # backwards-compatible aliases
Sep2D = SepSystem
