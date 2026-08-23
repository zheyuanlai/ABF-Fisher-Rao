"""High-precision reference free energies from massive unbiased Brownian dynamics.

    F_ref(z) = -beta^{-1} log p(z) + const

is the one reference that uses NONE of the constrained machinery it is meant to
validate.  The same trajectories also deposit the local mean force through the
Chapter-3 estimator, so `F_TI` vs `F_hist` from IDENTICAL samples is a direct
test of the mean-force formula (grad xi, the Gram matrix, the Hessian trace and
the divergence term) with the sampler taken out of the question.

Statistics are accumulated in `--blocks` independent blocks so the reference's
own uncertainty is measurable rather than assumed.
"""
from __future__ import annotations

import argparse, math, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rcwfr.mol import systems as S
from rcwfr.mol.dynamics import free_step
from rcwfr.mol.geom import TorsionCV


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="PEN")
    ap.add_argument("--B", type=int, default=131072)
    ap.add_argument("--steps", type=int, default=2_000_000)
    ap.add_argument("--burn", type=float, default=0.15)
    ap.add_argument("--nb", type=int, default=180)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--joint-nb", type=int, default=60)
    ap.add_argument("--init-from", default=None,
                    help="seed the torsions from a previous run's joint histogram; "
                         "the stationary law is unaffected, only the transient")
    ap.add_argument("--hist-every", type=int, default=25)
    ap.add_argument("--mf-every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=20260822)
    ap.add_argument("--out", default="results/mol/ref")
    a = ap.parse_args()

    dev, dt = torch.device("cuda"), torch.float64
    torch.manual_seed(a.seed)
    sy = S.REGISTRY[a.system](dev, dt)
    top, beta, h = sy.top, sy.beta, sy.h
    nt = top.tor_idx.shape[0]
    cvs = [TorsionCV(top.tor_idx[k:k + 1], top.mass, shift=sy.cv.shift)
           for k in range(nt)]

    if a.init_from:
        # A uniform start leaves a basin that the dynamics cannot reach out of
        # in the run's length populated by its initial share, not its Boltzmann
        # weight -- alanine's C7ax sat at EXACTLY 0.3548 in all eight blocks of
        # the first attempt.  Seeding from the previous estimate removes that
        # transient; it cannot bias the stationary law it converges to, and the
        # block spread reports whether it converged.
        H = np.load(a.init_from)["H2"].sum(0)
        pr = torch.as_tensor((H / H.sum()).reshape(-1), device=dev, dtype=dt)
        idx = torch.multinomial(pr, a.B, replacement=True)
        nbh = H.shape[0]
        i0, i1 = idx // nbh, idx % nbh
        w = 2 * math.pi / nbh
        u = torch.rand(a.B, 2, device=dev, dtype=dt)
        phis = torch.stack([(i0.to(dt) + u[:, 0]) * w - math.pi,
                            (i1.to(dt) + u[:, 1]) * w - math.pi], -1)
        if nt > 2:
            phis = torch.cat([phis, (torch.rand(a.B, nt - 2, device=dev, dtype=dt)
                                     * 2 - 1) * math.pi], -1)
    else:
        phis = (torch.rand(a.B, nt, device=dev, dtype=dt) * 2 - 1) * math.pi
    q = sy.ideal(phis)
    step = torch.compile(lambda q: free_step(top, q, h, beta,
                                            drift_cap=sy.drift_cap), dynamic=False)

    nb, nbk = a.nb, a.blocks
    Z = lambda *s: torch.zeros(s, device=dev, dtype=dt)
    H1, H2 = Z(nbk, nt, nb), (Z(nbk, nb, nb) if nt >= 2 else None)
    # full joint over (z, y_1, ..., y_F) at a coarser resolution: the |S| >= 2
    # refresh lift needs p(y_S | z) for arbitrary promoted subsets S
    nj = a.joint_nb
    # the full joint is only affordable up to three torsions; the PAIRWISE tables
    # (z, y_k) are what the diagnostic and the single-mode proposals actually use
    Hj = Z(*([nbk] + [nj] * nt)) if 2 <= nt <= 3 else None
    Hp = Z(nbk, nt - 1, nb, nb) if nt >= 2 else None
    ij_of = lambda x: torch.clamp(((x + math.pi) / (2 * math.pi) * nj).long(), 0, nj - 1)
    S0, S1, W0, W1 = Z(nbk, nt, nb), Z(nbk, nt, nb), Z(nbk, nt, nb), Z(nbk, nt, nb)
    burn = int(a.burn * a.steps)
    per = max(1, (a.steps - burn) // nbk)
    ib_of = lambda x: torch.clamp(((x + math.pi) / (2 * math.pi) * nb).long(), 0, nb - 1)
    t0 = time.time()
    for it in range(a.steps):
        q = step(q)
        if it < burn:
            continue
        blk = min(nbk - 1, (it - burn) // per)
        if it % a.hist_every == 0:
            phi = torch.stack([c.value(q)[:, 0] for c in cvs], -1)
            for k in range(nt):
                H1[blk, k].scatter_add_(0, ib_of(phi[:, k]), torch.ones(a.B, device=dev, dtype=dt))
            if H2 is not None:
                H2[blk].reshape(-1).scatter_add_(
                    0, ib_of(phi[:, 0]) * nb + ib_of(phi[:, 1]),
                    torch.ones(a.B, device=dev, dtype=dt))
                if Hj is not None:
                    flat = ij_of(phi[:, 0])
                    for kk in range(1, nt):
                        flat = flat * nj + ij_of(phi[:, kk])
                    Hj[blk].reshape(-1).scatter_add_(
                        0, flat, torch.ones(a.B, device=dev, dtype=dt))
                for kk in range(1, nt):
                    Hp[blk, kk - 1].reshape(-1).scatter_add_(
                        0, ib_of(phi[:, 0]) * nb + ib_of(phi[:, kk]),
                        torch.ones(a.B, device=dev, dtype=dt))
        if it % a.mf_every == 0:
            gV = top.grad(q)
            for k, c in enumerate(cvs):
                f, G = c.mean_force(q, gV, beta)
                ib = ib_of(c.value(q)[:, 0])
                w = G[..., 0, 0] ** -0.5
                S0[blk, k].scatter_add_(0, ib, torch.ones_like(f[:, 0]))
                S1[blk, k].scatter_add_(0, ib, f[:, 0])
                W0[blk, k].scatter_add_(0, ib, w)
                W1[blk, k].scatter_add_(0, ib, w * f[:, 0])
        if it % 200_000 == 0 and it:
            el = time.time() - t0
            print(f"  {it}/{a.steps} {el:.0f}s eta {el/it*(a.steps-it):.0f}s", flush=True)

    os.makedirs(a.out, exist_ok=True)
    edges = np.linspace(-math.pi, math.pi, nb + 1)
    p = os.path.join(a.out, f"{a.system}_ref.npz")
    np.savez_compressed(
        p, centers=0.5 * (edges[1:] + edges[:-1]),
        H1=H1.cpu().numpy(), H2=(H2.cpu().numpy() if H2 is not None else np.zeros((0, 0, 0))),
        S0=S0.cpu().numpy(), S1=S1.cpu().numpy(), W0=W0.cpu().numpy(), W1=W1.cpu().numpy(),
        Hjoint=(Hj.cpu().numpy() if Hj is not None else np.zeros((0,))), joint_nb=nj,
        Hpair=(Hp.cpu().numpy() if Hp is not None else np.zeros((0,))),
        beta=beta, h=h, B=a.B, steps=a.steps, burn=burn, T=sy.T, nb=nb, blocks=nbk)
    print(f"done {time.time()-t0:.0f}s -> {p}", flush=True)


if __name__ == "__main__":
    main()
