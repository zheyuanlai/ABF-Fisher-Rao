"""E1b: the Fixman term, decided by where the constrained sampler actually lands.

Showing that a Fixman-less constrained sampler "differs from nu^xi" is weak; the
sharp statement is that it converges to the RIGID measure

    nu_rgd(y|z)  propto  e^{-beta Psi(y,z)} sqrt(G(y))          (= e^{-beta V} dsigma)

instead of the physical conditional

    nu(y|z)      propto  e^{-beta Psi(y,z)}                     (= e^{-beta V} (det G)^{-1/2} dsigma).

So the test is a 2x2: run the ambient constrained dynamics with and without the
Fixman potential, and score each against BOTH targets with the PIT.  Each sampler
should sit at the histogram floor against its own target and well above it
against the other.
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import torch

from rcwfr.grid import DEVICE, DTYPE, EPS
from rcwfr.manifold import constrained_step
from rcwfr.systems.graph import build_mfib


def cdf_tables(s, z_scalar):
    """(standard, rigid) conditional CDFs on the reference y grid at one z."""
    y = s._yq
    z = torch.full_like(y, z_scalar)
    _, Psi, _, _ = s._psi_parts(y, z)
    w = -s.p.beta * Psi
    e = torch.exp(w - w.max())
    out = []
    for weight in (torch.ones_like(e), torch.sqrt(s.cv.G(y))):
        pdf = e * weight
        pdf = pdf / torch.trapezoid(pdf, dx=s._dy)
        c = torch.cumulative_trapezoid(pdf, dx=s._dy)
        c = torch.cat([torch.zeros(1, device=y.device, dtype=y.dtype), c])
        out.append((c / c[-1]).contiguous())
    return out


def pit_with(s, cdf, y):
    pos = torch.clamp((y - s._yq[0]) / s._dy, 0.0, len(s._yq) - 1.0)
    j = torch.clamp(torch.floor(pos).long(), 0, len(s._yq) - 2)
    f = pos - j.to(y.dtype)
    return cdf[j] + f * (cdf[j + 1] - cdf[j])


def kl(u, nbins=64):
    h = torch.histc(u, bins=nbins, min=0.0, max=1.0)
    n = float(h.sum()); p = h / n
    return float((p * torch.log(torch.clamp(p * nbins, min=1e-30))).sum()) \
        - (nbins - 1) / (2.0 * n)


def sample(s, z_scalar, fixman, M=120_000, n_steps=20_000, dt=1e-4):
    cv, beta = s.cv, s.p.beta
    gen = torch.Generator(device=DEVICE); gen.manual_seed(4 + int(fixman))
    z = torch.full((M,), z_scalar, device=DEVICE, dtype=DTYPE)
    y = s.sample_fiber(z, gen)
    q = torch.stack([z - cv.s(y), y], -1)
    keep = []
    for t in range(n_steps):
        q = constrained_step(cv, q, z, s.grad_V_ambient(q), dt, beta, gen, fixman=fixman)
        q[..., 1] = torch.clamp(q[..., 1], -s.p.y_max, s.p.y_max)
        q = cv.project(q, z)
        if t >= n_steps // 2 and t % 100 == 0:
            keep.append(q[..., 1].clone())
    return torch.cat(keep)


if __name__ == "__main__":
    s = build_mfib(omega=1.0, a=0.8, k=2.0)          # strong nonlinearity: sqrt(G) in [1, 1.9]
    print(f"sqrt(G) range: [{float(torch.sqrt(s.cv.G(s._yq)).min()):.3f}, "
          f"{float(torch.sqrt(s.cv.G(s._yq)).max()):.3f}]   "
          f"RMSE(F - F_rgd) = {s.fixman_gap:.5f}")
    out = []
    for z0 in (-0.9, -0.3, 0.3):
        c_std, c_rgd = cdf_tables(s, z0)
        row = dict(z=z0)
        for fix in (True, False):
            y = sample(s, z0, fix)
            row[f"fix{int(fix)}_vs_standard"] = kl(pit_with(s, c_std, y))
            row[f"fix{int(fix)}_vs_rigid"] = kl(pit_with(s, c_rgd, y))
        out.append(row)
        print(f"  z={z0:+.2f}   WITH Fixman: vs standard {row['fix1_vs_standard']:8.5f} | "
              f"vs rigid {row['fix1_vs_rigid']:8.5f}    "
              f"WITHOUT: vs standard {row['fix0_vs_standard']:8.5f} | "
              f"vs rigid {row['fix0_vs_rigid']:8.5f}", flush=True)
    json.dump(out, open("results/manifold/fixman_dynamic.json", "w"), indent=1)
