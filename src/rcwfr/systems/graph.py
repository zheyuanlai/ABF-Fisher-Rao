"""Same potential as systems/base.py, NONLINEAR reaction coordinate.

    V(x, y)  -- unchanged, from SysParams
    xi(q)    = x + a sin(k y_1)          instead of   xi(q) = x

so `a` is a pure NONLINEARITY knob: a = 0 reproduces the existing systems bit for
bit, and every Chapter-3 object (det G, the Fixman factor, the divergence term of
the local mean force, the difference between the three lifts, the gap between the
standard and the rigid free energy) switches on continuously with a.

Everything stays exactly computable because Sigma(z) is a graph over y:

    Sigma(z) = { (z - a sin(k y_1), y) },   dsigma = sqrt(G) dy,   G = 1 + c^2,
    (det G)^{-1/2} dsigma = dy,

so with the m_spec Gaussian spectators integrated out analytically,

    Psi(y, z) = V_act(z - s(y), y) + (m_spec / beta) log omega_s(z - s(y)),
    nu(y | z) propto e^{-beta Psi(y, z)}                      TRUE conditional
    nu_rgd(y | z) propto e^{-beta Psi(y, z)} sqrt(G(y))       what a Fixman-less
                                                              constrained sampler
                                                              actually converges to
    F(z)     = -beta^{-1} log integral e^{-beta Psi} dy
    F_rgd(z) = -beta^{-1} log integral e^{-beta Psi} sqrt(G) dy
    F'(z)    = E_nu[ d_z Psi ]

Three lifts, all written as one fiber velocity  w = dy/dz  (the ambient lift is
then  dq = dz (1 - c w, w, 0...), which satisfies grad xi . dq = dz for any w):

    cartesian  w = 0                 move x only; the naive lift
    minnorm    w = c / G             the minimum-Euclidean-norm horizontal lift
    adiabatic  w = w*(y, z)          solves the fiber continuity equation
                                     d_z nu + d_y (nu w) = 0, i.e.
                                     w* = (beta / nu) int_{-inf}^{y} nu (d_z Psi - F')

w* is the only lift with zero conditional lag by construction, and it is built
from exactly the two objects a TI method already estimates: the fiber-conditional
density and the mean-force fluctuation about F'.
"""
from __future__ import annotations

import torch

from ..grid import DEVICE, DTYPE, EPS, Grid1D
from ..manifold import GraphCV
from .base import SepSystem, SysParams


class GraphSystem(SepSystem):
    """SepSystem potential + GraphCV reaction coordinate."""

    def __init__(self, params: SysParams, grid: Grid1D, cv: GraphCV,
                 device=DEVICE, dtype=DTYPE):
        self.cv = cv
        super().__init__(params, grid, device, dtype)

    # ---- ambient potential / gradient -------------------------------------
    def V_ambient(self, q):
        return self.energy(q[..., 0], q[..., 1:])

    def grad_V_ambient(self, q):
        X, Y = q[..., 0], q[..., 1:]
        gx = self.mean_force(X, Y).unsqueeze(-1)      # dV/dx
        gy = self.grad_y(X, Y)                        # dV/dy
        return torch.cat([gx, gy], dim=-1)

    # ---- fiber-frame potential Psi(y, z) ----------------------------------
    def _psi_parts(self, y, z):
        """Return (x, Psi, dPsi/dz, dPsi/dy) at fiber point y on Sigma(z)."""
        p, cv = self.p, self.cv
        x = z - cv.s(y)
        c = cv.c(y)
        yy = y.unsqueeze(-1)
        zeros = torch.zeros(yy.shape[:-1] + (p.m_spec,), device=y.device, dtype=y.dtype)
        Y = torch.cat([yy, zeros], -1) if p.m_spec else yy
        Vact = self.energy(x, Y)
        Vx = self.mean_force(x, Y)
        Vy = self.grad_y(x, Y)[..., 0]
        if p.m_spec:
            oms, doms = self.omega_s(x), self.domega_s(x)
            Vact = Vact + p.m_spec * torch.log(oms) / p.beta
            Vx = Vx + p.m_spec * doms / (p.beta * oms)
        # dPsi/dz = dPsi/dx * (dx/dz = 1);   dPsi/dy = -c * dPsi/dx + dV/dy
        return x, Vact, Vx, -c * Vx + Vy

    # ---- exact reference on a (z, y) quadrature grid -----------------------
    def _build_reference(self):
        p, cv, g = self.p, self.cv, self.grid
        dev, dt = self.device, self.dtype
        zq = g.x(dev, dt)
        yq = torch.linspace(-p.y_max, p.y_max, p.n_yq, device=dev, dtype=dt)
        Z1, Y1 = zq.unsqueeze(1), yq.unsqueeze(0)
        _, Psi, dPsi_dz, _ = self._psi_parts(Y1, Z1)
        w = -p.beta * Psi
        wmax = w.max(dim=1, keepdim=True).values
        e = torch.exp(w - wmax)
        dy = float(yq[1] - yq[0])
        Zc = torch.trapezoid(e, dx=dy, dim=1)
        F = -(torch.log(torch.clamp(Zc, min=EPS)) + wmax.squeeze(1)) / p.beta
        dF = torch.trapezoid(dPsi_dz * e, dx=dy, dim=1) / torch.clamp(Zc, min=EPS)
        # rigid (Fixman-less) counterpart
        sqG = torch.sqrt(cv.G(Y1)).expand_as(e)
        Zr = torch.trapezoid(e * sqG, dx=dy, dim=1)
        F_rgd = -(torch.log(torch.clamp(Zr, min=EPS)) + wmax.squeeze(1)) / p.beta

        mask = g.eval_mask(dev, dt)
        self.F_ref = (F - F[mask].mean()).unsqueeze(0)
        self.dF_ref = dF.unsqueeze(0)
        self.F_rgd_ref = (F_rgd - F_rgd[mask].mean()).unsqueeze(0)

        pdf = e / torch.clamp(Zc, min=EPS).unsqueeze(1)
        cdf = torch.cumulative_trapezoid(pdf, dx=dy, dim=1)
        cdf = torch.cat([torch.zeros((cdf.shape[0], 1), device=dev, dtype=dt), cdf], 1)
        self._cdf = cdf / torch.clamp(cdf[:, -1:], min=EPS)
        self._yq, self._pdf = yq, pdf
        self._dy = dy

        # adiabatic fiber velocity  w* = (beta / nu) * cumint[ nu (dPsi/dz - F') ]
        integrand = pdf * (dPsi_dz - dF.unsqueeze(1))
        cum = torch.cumulative_trapezoid(integrand, dx=dy, dim=1)
        cum = torch.cat([torch.zeros((cum.shape[0], 1), device=dev, dtype=dt), cum], 1)
        self._w_ad = p.beta * cum / torch.clamp(pdf, min=1e-14)
        # the tails are 0/0; clamp them to the value at the last well-populated node
        good = pdf > 1e-10
        self._w_ad = torch.where(good, self._w_ad, torch.zeros_like(self._w_ad))

        tail = float(torch.max(pdf[:, 0] + pdf[:, -1]))
        assert tail < 1e-7, f"y_max too small: edge conditional density {tail:.3e}"
        self.p_channel_ref = torch.trapezoid(pdf[:, yq > 0], dx=dy, dim=1)
        self.tau_fiber_ref = 1.0 / (self.omega(zq) ** 2)
        self.fixman_gap = float((self.F_ref - self.F_rgd_ref)[0, mask].abs().max())

    # ---- interpolation helpers --------------------------------------------
    def _z_index(self, z):
        g = self.grid
        pos = torch.clamp((z - g.xmin) / g.dx, 0.0, g.n - 1.0)
        i0 = torch.clamp(torch.floor(pos).long(), 0, g.n - 2)
        return i0, (pos - i0.to(z.dtype))

    def _interp_zy(self, table, z, y):
        """Bilinear read of a (G, n_yq) table at scattered (z, y)."""
        i0, fz = self._z_index(z)
        fz = fz.unsqueeze(-1).squeeze(-1)
        posy = torch.clamp((y - self._yq[0]) / self._dy, 0.0, len(self._yq) - 1.0)
        j0 = torch.clamp(torch.floor(posy).long(), 0, len(self._yq) - 2)
        fy = (posy - j0.to(y.dtype))
        t00 = table[i0, j0]; t01 = table[i0, j0 + 1]
        t10 = table[i0 + 1, j0]; t11 = table[i0 + 1, j0 + 1]
        return ((1 - fz) * ((1 - fy) * t00 + fy * t01)
                + fz * ((1 - fy) * t10 + fy * t11))

    # ---- fiber sampling ----------------------------------------------------
    def icdf(self, z, u, block=32_768):
        """y such that CDF_{nu(.|z)}(y) = u.  Blocked over the particle axis."""
        shp = z.shape
        zf, uf = z.reshape(-1), u.reshape(-1)
        M = zf.numel()
        out = torch.empty(M, device=z.device, dtype=z.dtype)
        for a in range(0, M, block):
            zb, ub = zf[a:a + block], uf[a:a + block]
            i0, fz = self._z_index(zb)
            cdf = self._cdf[i0] + fz.unsqueeze(-1) * (self._cdf[i0 + 1] - self._cdf[i0])
            j = torch.clamp(torch.searchsorted(cdf.contiguous(),
                                               ub.unsqueeze(-1).contiguous()),
                            1, cdf.shape[-1] - 1)
            lo = torch.gather(cdf, -1, j - 1).squeeze(-1)
            hi = torch.gather(cdf, -1, j).squeeze(-1)
            s_ = (ub - lo) / torch.clamp(hi - lo, min=EPS)
            y0, y1 = self._yq[(j - 1).squeeze(-1)], self._yq[j.squeeze(-1)]
            out[a:a + block] = y0 + s_ * (y1 - y0)
        return out.reshape(*shp)

    def lift_cdf(self, z, y, z_new):
        """The EXACT adiabatic lift: the monotone map that carries nu(.|z) onto
        nu(.|z_new), i.e. y' = CDF^{-1}_{z_new}( CDF_z(y) ).

        This is the flow generated by w*, integrated exactly.  Using the map
        rather than the velocity matters because w* = (beta/nu) * cumint(...)
        diverges wherever the fiber conditional has a low-density valley -- which
        is precisely the multimodal fiber the method is supposed to help with.
        """
        return self.icdf(z_new, torch.clamp(self.pit(z, y), 1e-12, 1 - 1e-12))

    # ---- fast path: every particle on the SAME fiber ----------------------
    def _cdf_at(self, z_scalar):
        """Index arithmetic in PYTHON: building a device tensor from a float here
        costs a host-device synchronization on every call, which dominates the
        whole sweep when this is in the inner loop."""
        g = self.grid
        pos = min(max((float(z_scalar) - g.xmin) / g.dx, 0.0), g.n - 1.0)
        i0 = min(int(pos), g.n - 2)
        f = pos - i0
        return (self._cdf[i0] * (1.0 - f) + self._cdf[i0 + 1] * f).contiguous()

    def pit_scalar(self, z_scalar, y):
        """PIT when all of `y` lives on one fiber: O(M log n_yq), not O(M n_yq)."""
        cdf = self._cdf_at(z_scalar)
        pos = torch.clamp((y - self._yq[0]) / self._dy, 0.0, len(self._yq) - 1.0)
        j0 = torch.clamp(torch.floor(pos).long(), 0, len(self._yq) - 2)
        fy = pos - j0.to(y.dtype)
        return cdf[j0] + fy * (cdf[j0 + 1] - cdf[j0])

    def lift_cdf_scalar(self, z_scalar, y, z_new_scalar):
        """The exact adiabatic lift, common-fiber fast path."""
        u = torch.clamp(self.pit_scalar(z_scalar, y), 1e-12, 1 - 1e-12)
        cdf1 = self._cdf_at(z_new_scalar)
        j = torch.clamp(torch.searchsorted(cdf1, u.contiguous()), 1, len(cdf1) - 1)
        lo, hi = cdf1[j - 1], cdf1[j]
        s_ = (u - lo) / torch.clamp(hi - lo, min=EPS)
        return self._yq[j - 1] + s_ * (self._yq[j] - self._yq[j - 1])

    def sample_fiber(self, z, gen, block=32_768):
        """Exact y ~ nu(.|z) by inverse CDF (batched, arbitrary shape).

        The (M, n_yq) interpolated-CDF tensor is materialized in blocks so a
        large oracle draw cannot exhaust device memory.
        """
        shp = z.shape
        zf = z.reshape(-1)
        M = zf.numel()
        out = torch.empty(M, device=z.device, dtype=z.dtype)
        for a in range(0, M, block):
            zb = zf[a:a + block]
            i0, fz = self._z_index(zb)
            cdf = self._cdf[i0] + fz.unsqueeze(-1) * (self._cdf[i0 + 1] - self._cdf[i0])
            u = torch.rand(zb.shape, device=z.device, dtype=z.dtype, generator=gen)
            j = torch.clamp(torch.searchsorted(cdf.contiguous(),
                                               u.unsqueeze(-1).contiguous()),
                            1, cdf.shape[-1] - 1)
            lo = torch.gather(cdf, -1, j - 1).squeeze(-1)
            hi = torch.gather(cdf, -1, j).squeeze(-1)
            t = (u - lo) / torch.clamp(hi - lo, min=EPS)
            y0, y1 = self._yq[(j - 1).squeeze(-1)], self._yq[j.squeeze(-1)]
            out[a:a + block] = y0 + t * (y1 - y0)
        return out.reshape(*shp)

    def pit(self, z, y):
        """Probability integral transform u = CDF_{nu(.|z)}(y) in [0, 1].

        Under the correct conditional u is Uniform[0,1] for EVERY z, so a single
        pooled histogram of u measures the conditional lag of the whole ensemble
        without needing a per-z KL estimate.  This is D_cond, made measurable.
        """
        shp = z.shape
        zf, yf = z.reshape(-1), y.reshape(-1)
        M = zf.numel()
        out = torch.empty(M, device=z.device, dtype=z.dtype)
        block = 32_768
        for a in range(0, M, block):
            zb, yb = zf[a:a + block], yf[a:a + block]
            i0, fz = self._z_index(zb)
            cdf = self._cdf[i0] + fz.unsqueeze(-1) * (self._cdf[i0 + 1] - self._cdf[i0])
            posy = torch.clamp((yb - self._yq[0]) / self._dy, 0.0, len(self._yq) - 1.0)
            j0 = torch.clamp(torch.floor(posy).long(), 0, len(self._yq) - 2)
            fy = (posy - j0.to(y.dtype)).unsqueeze(-1)
            lo = torch.gather(cdf, -1, j0.unsqueeze(-1))
            hi = torch.gather(cdf, -1, (j0 + 1).unsqueeze(-1))
            out[a:a + block] = (lo + fy * (hi - lo)).squeeze(-1)
        return out.reshape(*shp)

    # ---- fiber dynamics (intrinsic; invariant measure is exactly nu(.|z)) --
    def step_fiber_z(self, z, y, dt, gen):
        _, _, _, dPsi_dy = self._psi_parts(y, z)
        noise = torch.randn(y.shape, device=y.device, dtype=y.dtype, generator=gen)
        y = y - dt * dPsi_dy + ((2.0 * dt / self.p.beta) ** 0.5) * noise
        return torch.clamp(y, -self.p.y_max, self.p.y_max)

    def mean_force_z(self, z, y):
        """dPsi/dz -- the mean-force sample whose conditional average is F'(z)."""
        return self._psi_parts(y, z)[2]

    # ---- lifts -------------------------------------------------------------
    def fiber_velocity(self, z, y, mode):
        if mode == "cartesian":
            return torch.zeros_like(y)
        if mode == "minnorm":
            c = self.cv.c(y)
            return c / (1.0 + c * c)
        if mode == "adiabatic":
            return self._interp_zy(self._w_ad, z, y)
        raise ValueError(mode)

    def lift_fiber(self, z, y, dz, mode, n_sub: int = 1):
        """Transport y from Sigma(z) to Sigma(z + dz) with the chosen lift."""
        if mode == "oracle":
            raise ValueError("oracle lift is handled by sample_fiber")
        if mode == "adiabatic":
            return self.lift_cdf(z, y, z + dz)
        h = dz / n_sub
        for _ in range(n_sub):
            y = y + h * self.fiber_velocity(z, y, mode)
            z = z + h
            y = torch.clamp(y, -self.p.y_max, self.p.y_max)
        return y


def build_graph(name: str, a: float = 0.6, k: float = 1.4, **overrides) -> GraphSystem:
    from ..registry import SYSTEMS, torsion
    from dataclasses import replace
    cv = GraphCV(a=a, k=k, d=1)
    spec = SYSTEMS[name]
    p = spec["params"]
    if overrides:
        p = replace(p, **overrides)
    return GraphSystem(p, spec["grid"], cv)


def build_mfib(omega: float = 1.0, a: float = 0.6, k: float = 1.4, H: float = 2.5,
               beta: float = 8.0, xhalf: float = 1.8, evalhalf: float = 1.5,
               n: int = 361, **overrides) -> GraphSystem:
    """Uniform-stiffness fiber: omega_in = omega_out = omega, so the fiber
    relaxation time  tau_mix = 1 / omega^2  is CONSTANT in z.

    That makes omega a clean knob for the timescale condition tau_mix << tau_WFR:
    nothing else in the system changes with it except the fiber relaxation rate.
    The conditional still depends on z, because the barrier U(z - a sin k y_1)
    reaches the fiber through the nonlinear reaction coordinate.
    """
    from ..grid import Grid1D
    y_max = max(4.0, 7.0 / (beta ** 0.5 * omega))
    p = SysParams(beta=beta, barrier="double_well", H=H,
                  omega_out=omega, omega_in=omega, s=0.25,
                  y_max=y_max, n_yq=6001, **overrides)
    g = Grid1D(-xhalf, xhalf, n, -evalhalf, evalhalf, bc="reflect")
    return GraphSystem(p, g, GraphCV(a=a, k=k, d=1))


def log_nu_exact(sysG, z_scalar: float, y):
    """log nu(y | z) with the normalizer computed by quadrature on the same grid."""
    zq = torch.full_like(sysG._yq, z_scalar)
    _, Psi_ref, _, _ = sysG._psi_parts(sysG._yq, zq)
    w = -sysG.p.beta * Psi_ref
    wmax = w.max()
    logZ = torch.log(torch.trapezoid(torch.exp(w - wmax), dx=sysG._dy)) + wmax
    _, Psi, _, _ = sysG._psi_parts(y, torch.full_like(y, z_scalar))
    return -sysG.p.beta * Psi - logZ


def _fiber_ref(sysG, z_scalar, n_grid=8001):
    """(y, nu, delta-free basics) on a grid trimmed to the support of nu(.|z)."""
    ym = sysG.p.y_max
    y = torch.linspace(-ym, ym, n_grid, device=sysG.device, dtype=sysG.dtype)
    z = torch.full_like(y, z_scalar)
    _, Psi, _, _ = sysG._psi_parts(y, z)
    w = -sysG.p.beta * Psi
    nu = torch.exp(w - w.max())
    nu = nu / torch.trapezoid(nu, x=y)
    return y, nu, z


def lag_coefficients(sysG, z_scalar: float, mode: str, n_grid: int = 8001):
    """The two lift-lag coefficients, both in CLOSED FORM.

    Write delta = w - w* for the lift's fiber-velocity error and let
    L delta = (1/nu) d_y(nu delta).  Linearizing s = rho/nu - 1 about 0:

      frozen fiber (no relaxation), displaced by dz
          d_z s = -L delta                =>  D_cond = (dz^2 / 2) C,
          C = || L delta ||^2_nu = integral [ d_y(nu delta) ]^2 / nu dy

      fiber relaxing while z moves at speed v
          d_t s = L_nu s - v L delta,  steady state  L_nu s = v L delta.
          In one fiber dimension that ODE integrates exactly: the zero-flux
          solution is  s = v beta (Delta - <Delta>_nu)  with Delta = int delta,
          so
          D_cond = (v^2 / 2) C_eff,   C_eff = beta^2 Var_nu( integral delta ).

    No eigendecomposition and no fitted constant.  Their ratio defines the
    timescale that the condition "tau_mix << tau_WFR" is really about:

          tau_eff = sqrt(C_eff / C),

    which is the fiber relaxation time WEIGHTED BY WHERE THE LIFT ERROR LIVES --
    an error carried by a fast fiber mode gets a small tau_eff and is repaired
    for free, however large it looks instantaneously.
    """
    y, nu, z = _fiber_ref(sysG, z_scalar, n_grid)
    w = sysG.fiber_velocity(z, y, mode)
    w_star = sysG.fiber_velocity(z, y, "adiabatic")
    delta = torch.nan_to_num(w - w_star, nan=0.0, posinf=0.0, neginf=0.0)
    keep = nu > nu.max() * 1e-12
    delta = torch.where(keep, delta, torch.zeros_like(delta))

    flux = nu * delta
    d_flux = torch.gradient(flux, spacing=(y,))[0]
    integ = torch.where(keep, d_flux * d_flux / torch.clamp(nu, min=1e-300),
                        torch.zeros_like(nu))
    C = float(torch.trapezoid(integ, x=y))

    Delta = torch.cat([torch.zeros(1, device=y.device, dtype=y.dtype),
                       torch.cumulative_trapezoid(delta, x=y)])
    m1 = torch.trapezoid(nu * Delta, x=y)
    m2 = torch.trapezoid(nu * Delta * Delta, x=y)
    C_eff = float(sysG.p.beta ** 2 * (m2 - m1 * m1))
    tau_eff = (C_eff / C) ** 0.5 if C > 0 else 0.0
    return dict(C=C, C_eff=C_eff, tau_eff=tau_eff)


def lag_coefficient(sysG, z_scalar: float, mode: str, **kw):
    return lag_coefficients(sysG, z_scalar, mode, **kw)["C"]


# ---------------------------------------------------------------------------
# multi-dimensional fiber:  y = (y_1, S_1..S_m)
# ---------------------------------------------------------------------------
# y_1 is the mode that enters xi and is a candidate for PROMOTION to a second
# collective variable; the S_k are spectators that enter V but not xi.  Their
# conditional law given (z, y_1) is exactly Gaussian,
#
#     S ~ N( mu(x) , sigma(x)^2 I ),   sigma = 1/(sqrt(beta) omega_s(x)),
#     mu(x) = mu_amp * g_neck(x),                      x = z - s(y_1),
#
# so everything stays quadrature-exact while the fiber is genuinely
# (1 + m)-dimensional.
#
# The SHIFT mu is what makes the design rule testable, and the width is not.
# Measured (scripts/exp_block_lag.py), a width-only spectator block has a frozen
# lift-lag coefficient 0.1%-5% of the y_1 block's: its conditional barely moves
# along z, so lifting it naively is nearly right however slowly it relaxes, and it
# tests nothing.  A shifted block is different in exactly the right way:
#
#   * its partition function is mu-independent, so it contributes NOTHING to F and
#     leaves the y_1 physics and the exact reference untouched;
#   * its conditional does move along z, by a tunable amount, so a naive lift is
#     wrong by a tunable amount;
#   * a wrong lift BIASES the mean force -- E[-omega_s^2 (S - mu) mu'] is zero
#     under the correct conditional and nonzero under a stale one;
#   * its relaxation time 1/omega_s^2 is tunable INDEPENDENTLY of all of that.
#
# Two orthogonal knobs -- how wrong the naive lift is, and how fast the block
# repairs it -- which is what the design rule needs.

class _NDMixin:
    mu_amp = 0.0                      # spectator shift amplitude; 0 = width-only

    def spectator_sd(self, z, y1):
        """sd of each spectator given (z, y_1); shape = broadcast of the inputs."""
        x = z - self.cv.s(y1)
        return 1.0 / (self.p.beta ** 0.5 * self.omega_s(x))

    def spectator_mean(self, z, y1):
        """mu(x) = mu_amp * x -- a LINEAR shift.

        Linear on purpose.  A neck-localized profile was tried first and is useless
        here: its slope is near zero over most of the reaction coordinate, so the
        block's lag coefficient barely moves with the amplitude (measured: C_S goes
        0.041 -> 0.066 as the amplitude goes 0 -> 1). A constant slope makes the
        coefficient exactly calibratable,

            C_S = m_spec * beta * (mu_amp * omega_s)^2 ,

        so setting mu_amp = A / omega_s fixes C_S = m beta A^2 while the relaxation
        time 1/omega_s^2 is free to vary over orders of magnitude.
        """
        if self.mu_amp == 0.0:
            return torch.zeros_like(y1)
        return self.mu_amp * (z - self.cv.s(y1))

    def _dmu_dx(self, x):
        return torch.full_like(x, self.mu_amp)

    def sample_spectators(self, z, y1, gen):
        """Exact S ~ nu(. | z, y_1); returns (..., m_spec)."""
        m = self.p.m_spec
        sd = self.spectator_sd(z, y1).unsqueeze(-1)
        mu = self.spectator_mean(z, y1).unsqueeze(-1)
        return mu + sd * torch.randn((*y1.shape, m), device=y1.device,
                                     dtype=y1.dtype, generator=gen)

    def _psi_nd(self, z, y1, S):
        """Full fiber potential and its z / y_1 / S derivatives, spectators EXPLICIT.

        Psi(y_1, S, z) = V_act(x, y_1) + (1/2) omega_s(x)^2 |S|^2,   x = z - s(y_1)
        """
        p, cv = self.p, self.cv
        x = z - cv.s(y1)
        c = cv.c(y1)
        Y = y1.unsqueeze(-1)
        if p.m_spec:
            Y = torch.cat([Y, torch.zeros_like(S)], -1)
        Vact = self.energy(x, Y)
        Vx = self.mean_force(x, Y)
        Vy = self.grad_y(x, Y)[..., 0]
        oms, doms = self.omega_s(x), self.domega_s(x)
        mu = self.spectator_mean(z, y1)
        D = S - mu.unsqueeze(-1)
        d2 = (D * D).sum(-1)
        # d/dx of (1/2) omega_s(x)^2 |S - mu(x)|^2
        dV_S_dx = oms * doms * d2 - (oms * oms) * self._dmu_dx(x) * D.sum(-1)
        dPsi_dz = Vx + dV_S_dx
        dPsi_dy = -c * dPsi_dz + Vy
        dPsi_dS = (oms * oms).unsqueeze(-1) * D
        return Vact + 0.5 * oms * oms * d2, dPsi_dz, dPsi_dy, dPsi_dS

    def step_fiber_nd(self, z, y1, S, dt, gen):
        """Overdamped Langevin on the (1 + m)-dimensional fiber at fixed z.

        Invariant measure is exactly nu^xi(. | z) in these coordinates.
        """
        _, _, gy, gS = self._psi_nd(z, y1, S)
        amp = (2.0 * dt / self.p.beta) ** 0.5
        y1 = y1 - dt * gy + amp * torch.randn(y1.shape, device=y1.device,
                                              dtype=y1.dtype, generator=gen)
        S = S - dt * gS + amp * torch.randn(S.shape, device=S.device,
                                            dtype=S.dtype, generator=gen)
        return torch.clamp(y1, -self.p.y_max, self.p.y_max), S

    def mean_force_nd(self, z, y1, S):
        """dPsi/dz with the spectators EXPLICIT.

        The marginalized `mean_force_z` is also unbiased and has lower variance,
        but it integrates the spectators out analytically and would therefore hide
        any error a lift makes on them -- which is exactly what is under test here.
        """
        return self._psi_nd(z, y1, S)[1]

    def lift_spectators(self, z, y1, z_new, y1_new, S, mode, gen=None):
        """Transport the spectator block from (z, y_1) to (z_new, y_1_new).

        'cartesian'  leave S alone -- the naive lift
        'scaled'     rescale by the ratio of conditional widths, which is the EXACT
                     conditional transport for a Gaussian block
        'oracle'     redraw from the exact conditional
        """
        if mode == "cartesian":
            return S
        if mode == "scaled":                       # exact for a Gaussian block
            mu0 = self.spectator_mean(z, y1).unsqueeze(-1)
            mu1 = self.spectator_mean(z_new, y1_new).unsqueeze(-1)
            r = (self.spectator_sd(z_new, y1_new)
                 / self.spectator_sd(z, y1)).unsqueeze(-1)
            return mu1 + (S - mu0) * r
        if mode == "oracle":
            return self.sample_spectators(z_new, y1_new, gen)
        raise ValueError(mode)

    def pit_spectators(self, z, y1, S):
        """PIT of each spectator against its exact conditional; uniform if correct."""
        sd = self.spectator_sd(z, y1).unsqueeze(-1)
        mu = self.spectator_mean(z, y1).unsqueeze(-1)
        return 0.5 * (1.0 + torch.erf((S - mu) / (sd * 2.0 ** 0.5)))


class GraphSystemND(_NDMixin, GraphSystem):
    """GraphSystem with the spectator block carried explicitly rather than
    integrated out.  `F_ref` / `dF_ref` are unchanged: the reference already
    integrates the spectators analytically, which is what makes this exact."""


def build_graph_nd(name: str, a: float = 0.6, k: float = 1.4, m_spec: int = 4,
                   oms_out: float = 1.0, oms_ratio: float = 4.0,
                   mu_amp: float = 0.0, **overrides) -> GraphSystemND:
    """Nonlinear CV, one promotable slow mode, and `m_spec` spectators whose
    conditional WIDTH changes along z (ratio oms_in/oms_out fixed), so their
    overall stiffness is a clean fiber-timescale knob."""
    from ..registry import SYSTEMS
    from dataclasses import replace
    spec = SYSTEMS[name]
    p = replace(spec["params"], m_spec=m_spec, oms_out=oms_out,
                oms_in=oms_out * oms_ratio, **overrides)
    sysG = GraphSystemND(p, spec["grid"], GraphCV(a=a, k=k, d=1))
    sysG.mu_amp = mu_amp          # shifts do not enter the partition function,
    return sysG                   # so F_ref / dF_ref need no rebuild
