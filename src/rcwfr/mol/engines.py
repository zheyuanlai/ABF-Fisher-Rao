"""Molecular arms: constrained stratified TI, RC-WFR with its lifts, and ABF.

COST INVARIANT.  Cost is counted in gradient evaluations of V per replica:
one per dynamics step, plus one per mean-force deposit (the deposit also needs
the Hessian of xi, but that touches four atoms and is charged as nothing).
ABF deposits from the gradient its own dynamics already computed, so it pays
n_steps; a constrained arm pays n_steps * (1 + 1/dep_every).  `fe` is returned
per save so a comparison can be made on the shared axis rather than assumed.

Every arm reads the SAME MeanForceAccumulator, with the SAME bandwidth and the
same low-count ramp, and every constrained arm deposits the Chapter-3 local mean
force with the (det G)^{-1/2} Fixman weight -- so no comparison here is
contaminated by an estimator asymmetry.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from ..estimators import MeanForceAccumulator
from ..fisher_rao import kde_marginal, kl_to_uniform, selection_indices
from ..grid import EPS, Grid1D, interp1d
from ..resampling import ancestor_stats, surviving_ancestors
from ..wasserstein import w_step_sde, w_step_flow
from .dynamics import constrained_step, free_step
from .ff import _wrap, rotate_about_bond
from ..shus import ShusAccumulator
from .geom import TorsionCV
from .joint import JointRefresh
from .lift import AdaptiveFiberCDF, ReferenceFiberCDF


@dataclass
class MolCfg:
    N: int = 256                 # replicas per row
    n_steps: int = 100_000       # dynamics steps (the budget)
    n_cond: int = 20             # constrained steps between WFR events
    dep_every: int = 20          # deposit the mean force every this many steps
    save_every: int = 4_000
    n_eq: int = 2_000            # constrained equilibration before any deposit
    bw_mf: float = 0.10
    n_min: float = 1.0
    init: str = "point"          # point | grid_cold | grid_warm
    z0: float = 0.0
    n_windows: int = 0           # stratified arms: distinct windows (0 = N)
    # --- WFR ---------------------------------------------------------------
    kappa: float = 0.30
    w_mode: str = "sde"          # sde | flow | none
    w_flow_clip: float = 50.0
    theta: float = 0.30
    fr_rule: str = "fr"          # fr | count | none
    bw_kde: float = 0.25
    alpha_ess: float = 0.5
    n_bins_count: int = 45       # bins for the count-balancing control
    fr_jitter: float = 0.0
    # --- lift --------------------------------------------------------------
    lift: str = "shake"          # shake | rot | y{map,ref,mh}_{oracle,learned}
    lift_bw_z: float = 0.25
    lift_bw_y: float = 0.30
    lift_decay: float = 0.999
    lift_nmin: float = 150.0
    lift_substeps: int = 4
    promote: tuple = (1,)        # which fiber torsions the lift transports
    lift_start: float = 0.0      # engage the LEARNED lift only after this many
                                 # STEPS (absolute, so a single long run can be
                                 # read at several budgets); before it the arm
                                 # uses the plain
                                 # rotation lift while still accumulating
                                 # nu_hat(y|z).  A learned lift bootstrapped from
                                 # its own cold-start output is self-reinforcing:
                                 # the ensemble is a delta in y, so nu_hat is a
                                 # delta, so a refresh redraws the delta forever.
    # --- ABF ---------------------------------------------------------------
    abf_n_min: float = 200.0
    abf_wall_k: float = 400.0    # half-harmonic wall outside a restricted CV domain
    shus_gain: float = 1.0       # ABP / SHUS / OPES baseline
    shus_block: int = 200
    shus_eps_bw: float = 0.07
    n_newton: int = 6
    # --- two-stage: RC-WFR as an initialiser, then unbiased constrained TI ---
    t_switch: int = 0            # steps after which reaction-coordinate TRANSPORT
                                 # and Fisher-Rao selection are switched off and
                                 # the run becomes ordinary stratified constrained
                                 # TI at whatever z the replicas reached.  0 = never.
    snap_at_switch: bool = False   # at the switch, place the replicas on a UNIFORM
                                 # grid of windows (by sorted rank) instead of
                                 # leaving them where transport happened to drop
                                 # them.  Frozen-in-place walkers are an uneven
                                 # stratification, and the unevenness never
                                 # averages away because the same set is reused
                                 # for the whole production stage.
    freeze_lift_at_switch: bool = False  # stop UPDATING the learned conditional at
                                 # the switch (keep using it).  With z frozen, each
                                 # window sees only its own handful of walkers, so
                                 # a table that keeps learning collapses to a delta
                                 # per window and the Metropolis proposal dies.
    # A SECOND accumulator is always carried and zeroed at the switch, so one run
    # reports both estimators: `F` keeps every deposit, `F_prod` uses only
    # post-switch samples.  That separates "RC-WFR's transport bias is in the
    # deposits" from "it is in the configurations" at no extra sampling cost.
    fixman_weight: bool = True   # deposit with (det G)^{-1/2}.  Correct when the
                                 # samples come from the RIGID measure, i.e. from
                                 # constrained dynamics.  The conditional-library
                                 # arm draws from nu^xi directly and is refreshed
                                 # faster than it relaxes, so for THAT arm the
                                 # weight is applied to a measure that never had
                                 # the (det G)^{1/2} factor, and double-counts it.
    joint_nb: int = 0            # >0: accumulate a persistent (z, y_1, ...) table.
                                 # Used to build a STRATIFIED reference conditional:
                                 # every window is visited equally, so the table is
                                 # populated where an unbiased run is thin.


# ---------------------------------------------------------------------------
def _wrap_to(a, b):
    """Shortest signed increment from a to b on the circle."""
    return _wrap(b - a)


def _project_path(cvf: TorsionCV, q, zy0, zy1, K, n_newton):
    """SHAKE along K equal increments; a single large jump does not converge."""
    d = _wrap_to(zy0, zy1)
    for k in range(1, K + 1):
        q, _ = cvf.project(q, _wrap(zy0 + d * (k / K)), n_newton=n_newton,
                           n_outer=(2 if k == K else 1))
    return q


def _gather_conf(q, sel):
    A, D = q.shape[-2], q.shape[-1]
    return torch.gather(q, 1, sel[:, :, None, None].expand(-1, -1, A, D))


def _init_conf(sy, cfg, rows, gen, ref_table=None, full_cv=None):
    """Place replicas.  Cold starts put EVERY fiber coordinate in the trans basin,
    which is the honest enhanced-sampling scenario; the warm start draws y from
    the reference conditional and is an oracle diagnostic, never a claim."""
    dev, dt, g, N = sy.device, sy.dtype, sy.grid, cfg.N
    nt = sy.top.tor_idx.shape[0]
    if cfg.init == "point":
        z = torch.full((rows, N), cfg.z0, device=dev, dtype=dt)
    else:
        M = cfg.n_windows if cfg.n_windows else N
        assert N % M == 0
        zs = torch.linspace(g.xmin, g.xmax, M + 1, device=dev, dtype=dt)[:M]
        z = zs.repeat_interleave(N // M).unsqueeze(0).expand(rows, N).clone()
    phis = torch.full((rows, N, nt), sy.y0, device=dev, dtype=dt)
    phis[..., 0] = z
    if cfg.init == "grid_warm" and ref_table is not None and nt > 1:
        u = torch.rand((rows, N), device=dev, dtype=dt, generator=gen)
        phis[..., 1] = _invcdf(ref_table, z, u)
    if cfg.init == "grid_spread" and nt > 1:
        # windows on a z grid, every fiber torsion drawn UNIFORMLY.  Used to build
        # a stratified-TI reference: no oracle conditional is needed, and a
        # uniform start is far closer to any conditional than a delta is, so the
        # per-window relaxation transient is short and whatever is left of it
        # shows up in the between-row spread.
        phis[..., 1:] = (torch.rand((rows, N, nt - 1), device=dev, dtype=dt,
                                    generator=gen) * 2 - 1) * math.pi
    q = sy.ideal(phis.reshape(-1, nt)).reshape(rows, N, sy.top.n_atoms, 3)
    return q, z.unsqueeze(-1)


def _invcdf(table, Z, u):
    cdf, _ = table._rows_at(Z)
    j = torch.clamp(torch.searchsorted(cdf.contiguous(), u.unsqueeze(-1).contiguous()),
                    1, table.n_y - 1)
    a = torch.gather(cdf, -1, j - 1).squeeze(-1)
    b = torch.gather(cdf, -1, j).squeeze(-1)
    t = (u - a) / torch.clamp(b - a, min=EPS)
    y0, y1 = table.yv[(j - 1).squeeze(-1)], table.yv[j.squeeze(-1)]
    return y0 + t * (y1 - y0)


def _marginalise(Hjoint, keep):
    """Reference joint (nz, n1, ..., nF) -> (nz, n_keep...) for the promoted set."""
    H = Hjoint
    n_fib = H.dim() - 1
    for ax in reversed(range(n_fib)):
        if ax not in keep:
            H = H.sum(dim=ax + 1)
    return H


def _cond_kl(Hz, Href, dy):
    """z-resolved KL( p_hat(y|z) || p_ref(y|z) ), averaged over the realized p(z).

    NOT pooled over z: the manifold phase showed a pooled conditional diagnostic
    hides errors that cancel between fibers.
    """
    pz = Hz.sum(-1)
    p = Hz / torch.clamp(pz, min=EPS).unsqueeze(-1)
    r = Href / torch.clamp(Href.sum(-1), min=EPS).unsqueeze(-1)
    kl = (p * (torch.log(torch.clamp(p, min=1e-12))
               - torch.log(torch.clamp(r, min=1e-12)))).sum(-1)
    wz = pz / torch.clamp(pz.sum(-1, keepdim=True), min=EPS)
    return (kl * wz).sum(-1)


# ---------------------------------------------------------------------------
def run_constrained(sy, cfg: MolCfg, rows: int, seed: int, ref=None,
                    kappa_vec=None, theta_vec=None, decay_vec=None):
    """Stratified constrained TI and every RC-WFR arm (they differ only in flags)."""
    dev, dt, g = sy.device, sy.dtype, sy.grid
    top, cv, beta, h = sy.top, sy.cv, sy.beta, sy.h
    gen = torch.Generator(device=dev); gen.manual_seed(seed)
    torch.manual_seed(seed + 7717)
    nt = top.tor_idx.shape[0]
    full_cv = TorsionCV(top.tor_idx, top.mass, shift=cv.shift) if nt > 1 else cv
    gy = sy.y_grid if sy.y_grid is not None else g
    n_fib = max(nt - 1, 0)                       # fiber torsions (z takes slot 0)
    prom = tuple(int(k) for k in cfg.promote if k < nt)
    specs = [sp for sp in sy.y_specs if sp[0] in prom]

    ref_table = None
    if ref is not None and nt > 1:
        ref_table = ReferenceFiberCDF(rows, ref["gz"], gy, dev, dt, ref["H2"])
    qlib = None
    if cfg.lift == "qref_oracle" and ref is not None and "conflib" in ref:
        qlib, qfill = ref["conflib"], ref["conffill"]
    joint = None
    if ref is not None and specs:
        if len(specs) == 1 and "Hcond" in ref:
            # the pairwise table is finer than a marginal of the full joint and
            # exists for every chain length; prefer it for single-mode promotion
            joint = JointRefresh(ref["Hcond"][specs[0][0] - 1], dev, dt, smooth=3)
        elif "Hjoint" in ref:
            joint = JointRefresh(_marginalise(ref["Hjoint"],
                                              [sp[0] - 1 for sp in specs]),
                                 dev, dt, smooth=3)
    q, z = _init_conf(sy, cfg, rows, gen, ref_table, full_cv)
    # `sy.ideal` already places every torsion exactly, so this only cleans up
    # float error.  The target has to carry ALL of full_cv's components -- hexane
    # has three torsions, and hard-coding two silently broke every hexane arm.
    if nt > 1:
        tgt = full_cv.value(q).clone()
        tgt[..., 0] = z[..., 0]
        q, _ = full_cv.project(q, tgt, n_newton=6, n_outer=1)
    else:
        q, _ = cv.project(q, z, n_newton=8, n_outer=2)

    step = torch.compile(lambda q, z: constrained_step(
        top, cv, q, z, h, beta, n_newton=cfg.n_newton,
        drift_cap=sy.drift_cap), dynamic=False)
    gradV = torch.compile(lambda q: top.grad(q), dynamic=False)
    efun = torch.compile(lambda q: top.energy(q), dynamic=False)
    # mean_force is NOT compiled: its exact Hessian goes through
    # vmap(jacfwd(jacrev)), which dynamo cannot trace.  It runs every dep_every
    # steps, so the eager cost is amortised; `dep_every` is the knob.
    mfun = lambda q, gv: cv.mean_force(q, gv, beta)
    yfun = torch.compile(lambda q: full_cv.value(q), dynamic=False) if nt > 1 else None

    acc = MeanForceAccumulator(rows, g, cfg.bw_mf, cfg.n_min, dev, dt)
    acc_prod = MeanForceAccumulator(rows, g, cfg.bw_mf, cfg.n_min, dev, dt)
    mask = g.eval_mask(dev, dt)
    kap = (cfg.kappa if kappa_vec is None
           else torch.as_tensor(kappa_vec, device=dev, dtype=dt).view(rows, 1))
    learn = None
    y_src, y_mode = (cfg.lift.split("_")[1], cfg.lift.split("_")[0][1:]) \
        if cfg.lift.startswith("y") else (None, None)
    if cfg.lift.startswith("q"):
        y_src, y_mode = None, None
    assert y_mode in (None, "map", "ref", "mh"), cfg.lift
    if y_src == "learned":
        learn = AdaptiveFiberCDF(rows, g, gy, dev, dt, cfg.lift_bw_z,
                                 cfg.lift_bw_y, cfg.lift_nmin, cfg.lift_decay)
        if decay_vec is not None:
            learn.decay = torch.as_tensor(decay_vec, device=dev,
                                          dtype=dt).view(rows, 1, 1)
    table = ref_table if y_src == "oracle" else learn

    n_outer = cfg.n_steps // cfg.n_cond
    lift_it0 = int(cfg.lift_start / cfg.n_cond)
    save_outer = max(1, cfg.save_every // cfg.n_cond)
    n_saves = n_outer // save_outer
    dtau = cfg.n_cond * h
    theta0 = (torch.full((rows,), cfg.theta, device=dev, dtype=dt) if theta_vec is None
              else torch.as_tensor(theta_vec, device=dev, dtype=dt).view(rows))
    ar = torch.arange(cfg.N, device=dev).unsqueeze(0).expand(rows, cfg.N)
    anc = ar.clone()
    Z = lambda *s: torch.zeros(s, device=dev, dtype=dt)
    out = {"F": Z(n_saves, rows, g.n), "kl": Z(n_saves, rows), "cov": Z(n_saves, rows),
           "ess_fix": Z(n_saves, rows), "ess_anc": Z(n_saves, rows),
           "dcond": Z(n_saves, rows), "resid": Z(n_saves, rows), "fe": Z(n_saves),
           "lift_cov": Z(n_saves, rows), "F_prod": Z(n_saves, rows, g.n)}
    nkz = nky = (int(ref["Hdiag"].shape[-1]) if (ref is not None and "Hdiag" in ref)
                 else 36)
    Hdiag = Z(rows, max(n_fib, 1), nkz, nky)
    njb = cfg.joint_nb
    Hjt = Z(njb, *([njb] * max(n_fib, 1))) if njb else None
    out["dcond_all"] = Z(n_saves, rows, max(n_fib, 1))
    fe, si, out_w = 0.0, 0, None
    eq_outer = cfg.n_eq // cfg.n_cond
    sw_outer = (cfg.t_switch // cfg.n_cond) if cfg.t_switch else None
    did_reset = False
    out["fe_switch"] = torch.zeros((), device=dev, dtype=dt)
    for it in range(n_outer):
        transporting = sw_outer is None or it < sw_outer
        if sw_outer is not None and it == sw_outer and not did_reset:
            did_reset = True
            out["fe_switch"] = torch.tensor(fe * cfg.N, device=dev, dtype=dt)
            acc_prod.S0.zero_(); acc_prod.S1.zero_()
            if cfg.snap_at_switch:
                order = torch.argsort(z[..., 0], dim=1)
                rank = torch.empty_like(order)
                rank.scatter_(1, order, torch.arange(cfg.N, device=dev)
                              .unsqueeze(0).expand(rows, cfg.N))
                ztgt = g.xmin + (rank.to(dt) + 0.5) * g.volume / cfg.N
                if cfg.lift == "shake":
                    q, _ = cv.project(q, ztgt.unsqueeze(-1),
                                      n_newton=cfg.n_newton, n_outer=3)
                else:
                    q = rotate_about_bond(q, sy.z_bond[0], sy.z_bond[1],
                                          list(sy.z_movers),
                                          -_wrap(ztgt - z[..., 0]))
                z = ztgt.unsqueeze(-1)
        for k in range(cfg.n_cond):
            q = step(q, z)
            fe += 1.0
            if it >= eq_outer and ((k + 1) % cfg.dep_every == 0):
                gv = gradV(q); fe += 1.0
                f, G = mfun(q, gv)
                w = (G[..., 0, 0] if cv.m == 1 else torch.linalg.det(G)) ** -0.5
                if not cfg.fixman_weight:
                    w = torch.ones_like(w)
                acc.deposit(z[..., 0], f[..., 0], weights=w)
                acc_prod.deposit(z[..., 0], f[..., 0], weights=w)
                out_w = w
        yall = yfun(q) if yfun is not None else None
        y = yall[..., 1] if yall is not None else None
        if yall is not None:
            iz = torch.clamp(((z[..., 0] - g.xmin) / g.volume * nkz).long(), 0, nkz - 1)
            for kf in range(n_fib):
                iy = torch.clamp(((yall[..., kf + 1] + math.pi) / (2 * math.pi) * nky).long(),
                                 0, nky - 1)
                Hdiag[:, kf].reshape(rows, -1).scatter_add_(
                    1, iz * nky + iy, torch.ones_like(iz, dtype=dt))
            if Hjt is not None:
                jz = torch.clamp(((z[..., 0] - g.xmin) / g.volume * njb).long(),
                                 0, njb - 1)
                flat = jz
                for kf in range(n_fib):
                    jy = torch.clamp(((yall[..., kf + 1] + math.pi) / (2 * math.pi) * njb)
                                     .long(), 0, njb - 1)
                    flat = flat * njb + jy
                Hjt.reshape(-1).scatter_add_(
                    0, flat.reshape(-1),
                    torch.ones(flat.numel(), device=dev, dtype=dt))
            if learn is not None and not (cfg.freeze_lift_at_switch
                                          and not transporting):
                # the constrained sampler produces nu_rgd, so the conditional is
                # accumulated with the same (det G)^{-1/2} weight the mean force
                # carries -- otherwise nu_hat(y|z) is the rigid conditional
                learn.deposit(z[..., 0], y, weight=(out_w if it >= eq_outer else None))
        # --- Wasserstein transport of the labels ---------------------------
        if not transporting:
            zn = z[..., 0]
        elif cfg.w_mode == "sde":
            zn = w_step_sde(z[..., 0], kap, dtau, g, gen)
        elif cfg.w_mode == "flow":
            zn = w_step_flow(z[..., 0], kap, dtau, g, cfg.bw_kde, cfg.w_flow_clip)
        else:
            zn = z[..., 0]
        # --- lift ----------------------------------------------------------
        if cfg.w_mode != "none" and transporting:
            lift_cov = torch.zeros(rows, device=dev, dtype=dt)
            if qlib is not None:
                # ABSOLUTE CEILING: replace the whole configuration by an exact
                # conditional draw at z'.  No lift error of any kind survives, so
                # whatever this arm still shows is the estimator plus the
                # z-marginal -- the bound every other lift is measured against.
                nzb = qlib.shape[0]
                b = torch.clamp(((zn - g.xmin) / g.volume * nzb).long(), 0, nzb - 1)
                u = torch.rand(zn.shape, device=dev, dtype=dt, generator=gen)
                j = (u * qfill[b].to(dt)).long().clamp_(0, qlib.shape[1] - 1)
                q = qlib[b, j].to(dt)
                # The library is bucketed, so a drawn configuration sits SOMEWHERE
                # in bin b, not at z'.  Within a bucket the library follows the
                # Boltzmann conditional while the labels follow the WFR target, so
                # depositing f at the label without re-projecting mismatches the
                # two by O(bucket width) -- worth 0.09 kcal/mol here, which is
                # more than the lift error this arm is supposed to bound.
                q, _ = cv.project(q, zn.unsqueeze(-1), n_newton=cfg.n_newton, n_outer=1)
                lift_cov = torch.ones(rows, device=dev, dtype=dt)
            elif cfg.lift == "shake":
                # minimum-norm HORIZONTAL lift: move along M^{-1} grad xi.  This
                # is the Chapter-3 geometric answer and the negative control.
                q, _ = cv.project(q, zn.unsqueeze(-1), n_newton=cfg.n_newton, n_outer=2)
            else:
                # exact internal-coordinate lift: rotate the distal fragment.
                # Bond lengths, bond angles and every other torsion are preserved
                # exactly, so nothing in the fiber is distorted by the move itself.
                q = rotate_about_bond(q, sy.z_bond[0], sy.z_bond[1],
                                      list(sy.z_movers), -_wrap(zn - z[..., 0]))
                engaged = (table is not None and y is not None
                           and (y_src == "oracle" or it >= lift_it0))
                if engaged and joint is not None and y_src == "oracle" \
                        and y_mode in ("ref", "mh"):
                    # promote an arbitrary subset of fiber torsions at once
                    u = torch.rand(zn.shape + (len(specs),), device=dev, dtype=dt,
                                   generator=gen)
                    ys = joint.sample(zn, u)
                    ycur = torch.stack([yall[..., ti] for ti, _b, _m in specs], -1)
                    qp = q
                    for i, (ti, bd, mv) in enumerate(specs):
                        qp = rotate_about_bond(qp, bd[0], bd[1], list(mv),
                                               -_wrap(ys[..., i] - ycur[..., i]))
                    if y_mode == "mh":
                        logA = (-beta * (efun(qp) - efun(q))
                                + joint.log_pdf(zn, ycur) - joint.log_pdf(zn, ys))
                        fe += 2.0
                        a_mh = (torch.rand(zn.shape, device=dev, dtype=dt, generator=gen)
                                < torch.exp(torch.clamp(logA, max=0.0)))
                        q = torch.where(a_mh[..., None, None], qp, q)
                        lift_cov = a_mh.to(dt).mean(1)
                    else:
                        q = qp
                        lift_cov = torch.ones(rows, device=dev, dtype=dt)
                elif engaged and y_mode == "mh":
                    # INDEPENDENCE METROPOLIS on the slow torsion, proposal
                    # nu_hat(. | z').  A rigid rotation about the torsion axis is
                    # an isometry of R^{3A}, leaves the internal-coordinate
                    # Jacobian (which depends on bonds and angles, never on a
                    # torsion) alone, and does not touch the four atoms that
                    # define xi -- so det G is invariant too and the move targets
                    # the constrained ensemble EXACTLY.  The learned conditional
                    # then only sets the acceptance rate, never the answer: a
                    # degenerate nu_hat makes the move a no-op instead of a
                    # catastrophe.
                    u = torch.rand(zn.shape, device=dev, dtype=dt, generator=gen)
                    yp, okl = table.sample(zn, u)
                    lp_old = table.log_pdf(zn, y)
                    lp_new = table.log_pdf(zn, yp)
                    qp = rotate_about_bond(q, sy.y_bond[0], sy.y_bond[1],
                                           list(sy.y_movers), -_wrap(yp - y))
                    dE = efun(qp) - efun(q)
                    fe += 2.0        # two energy evaluations per replica per event
                    logA = -beta * dE + lp_old - lp_new
                    a_mh = (torch.rand(zn.shape, device=dev, dtype=dt, generator=gen)
                            < torch.exp(torch.clamp(logA, max=0.0))) & okl
                    q = torch.where(a_mh[..., None, None], qp, q)
                    lift_cov = a_mh.to(dt).mean(1)      # reported acceptance rate
                elif engaged:
                    if y_mode == "ref":
                        u = torch.rand(zn.shape, device=dev, dtype=dt, generator=gen)
                        yn, okl = table.sample(zn, u)
                    else:
                        yn, okl = table.map(z[..., 0], y, zn)
                    q = rotate_about_bond(q, sy.y_bond[0], sy.y_bond[1],
                                          list(sy.y_movers), -_wrap(yn - y))
                    lift_cov = okl.to(dt).mean(1)
            z = zn.unsqueeze(-1)
        else:
            lift_cov = torch.zeros(rows, device=dev, dtype=dt)
        # --- Fisher-Rao reallocation ---------------------------------------
        if cfg.fr_rule != "none" and transporting:
            sel, info = selection_indices(z[..., 0], g, cfg.fr_rule, theta0, gen,
                                          bw=cfg.bw_kde, n_bins=cfg.n_bins_count,
                                          alpha_ess=cfg.alpha_ess)
            q = _gather_conf(q, sel)
            z = torch.gather(z[..., 0], 1, sel).unsqueeze(-1)
            anc = torch.gather(anc, 1, sel)
            if cfg.fr_jitter > 0:
                zj = g.enforce(z[..., 0] + cfg.fr_jitter * torch.randn(
                    z[..., 0].shape, device=dev, dtype=dt, generator=gen))
                q, _ = cv.project(q, zj.unsqueeze(-1), n_newton=cfg.n_newton, n_outer=2)
                z = zj.unsqueeze(-1)
        if (it + 1) % save_outer == 0 and si < n_saves:
            p = kde_marginal(z[..., 0], g, cfg.bw_kde)
            out["F"][si] = acc.free_energy(mask)
            out["F_prod"][si] = acc_prod.free_energy(mask)
            out["kl"][si] = kl_to_uniform(p, g)
            out["cov"][si] = _coverage(z[..., 0], g)
            out["ess_fix"][si] = _ess(out_w) if it >= eq_outer else 1.0
            e, _ = ancestor_stats(anc, cfg.N)
            out["ess_anc"][si] = e / cfg.N
            out["resid"][si] = cv.dz_residual(cv.value(q), z).abs().amax(dim=(1, 2))
            if ref is not None and nt > 1:
                for kf in range(n_fib):
                    out["dcond_all"][si, :, kf] = _cond_kl(Hdiag[:, kf],
                                                           ref["Hdiag"][kf], 1.0)
                out["dcond"][si] = out["dcond_all"][si, :, 0]
            out["lift_cov"][si] = lift_cov
            out["fe"][si] = fe * cfg.N
            Hdiag.zero_()
            anc = ar.clone()
            si += 1
    out["z_final"], out["q_final"] = z, q
    if Hjt is not None:
        out["Hjoint"] = Hjt
    return out


def _ess(w):
    return (w.sum(1) ** 2) / torch.clamp((w * w).sum(1) * w.shape[1], min=EPS)


def _coverage(Zt, grid, n_bins=45):
    idx = torch.clamp(((Zt - grid.xmin) / grid.volume * n_bins).long(), 0, n_bins - 1)
    hh = torch.zeros((Zt.shape[0], n_bins), device=Zt.device, dtype=Zt.dtype)
    hh.scatter_add_(1, idx, torch.ones_like(Zt))
    return (hh > 0).to(Zt.dtype).mean(1)


# ---------------------------------------------------------------------------
def run_abf(sy, cfg: MolCfg, rows: int, seed: int, ref=None):
    """Multiple-walker ABF: unconstrained dynamics biased by the running F'.

    The bias is refreshed every step (it is one interpolation); the mean-force
    DEPOSIT is subsampled at `dep_every`, the same rate every constrained arm
    uses, so no arm is handed a denser estimator than another.  At h = 0.002 the
    torsion decorrelates over thousands of steps, so subsampling at 20 costs
    essentially nothing statistically -- it only saves the Hessian.
    """
    dev, dt, g = sy.device, sy.dtype, sy.grid
    top, cv, beta, h = sy.top, sy.cv, sy.beta, sy.h
    gen = torch.Generator(device=dev); gen.manual_seed(seed)
    torch.manual_seed(seed + 331)
    nt = top.tor_idx.shape[0]
    full_cv = TorsionCV(top.tor_idx[:2], top.mass, shift=cv.shift) if nt > 1 else cv
    phis = torch.full((rows, cfg.N, nt), sy.y0, device=dev, dtype=dt)
    phis[..., 0] = cfg.z0
    q = sy.ideal(phis.reshape(-1, nt)).reshape(rows, cfg.N, top.n_atoms, 3)
    acc = MeanForceAccumulator(rows, g, cfg.bw_mf, cfg.n_min, dev, dt)
    acc_prod = MeanForceAccumulator(rows, g, cfg.bw_mf, cfg.n_min, dev, dt)
    mask = g.eval_mask(dev, dt)

    kw = (cfg.abf_wall_k if g.bc == "reflect" else 0.0)
    step = torch.compile(lambda q, bg: _abf_step(top, cv, q, bg, g, h, beta,
                                                 drift_cap=sy.drift_cap, k_wall=kw),
                         dynamic=False)
    gradV = torch.compile(lambda q: top.grad(q), dynamic=False)
    efun = torch.compile(lambda q: top.energy(q), dynamic=False)
    yfun = torch.compile(lambda q: full_cv.value(q), dynamic=False) if nt > 1 else None

    n_saves = cfg.n_steps // cfg.save_every
    Z = lambda *s: torch.zeros(s, device=dev, dtype=dt)
    out = {"F": Z(n_saves, rows, g.n), "kl": Z(n_saves, rows), "cov": Z(n_saves, rows),
           "ess_fix": Z(n_saves, rows) + 1.0, "ess_anc": Z(n_saves, rows) + 1.0,
           "dcond": Z(n_saves, rows), "resid": Z(n_saves, rows), "fe": Z(n_saves),
           "lift_cov": Z(n_saves, rows), "F_prod": Z(n_saves, rows, g.n)}
    nkz = nky = (int(ref["Hdiag"].shape[-1]) if (ref is not None and "Hdiag" in ref)
                 else 36)
    Hdiag = Z(rows, nkz, nky)
    bias_n = torch.full((rows, 1), cfg.abf_n_min, device=dev, dtype=dt)
    si, fe = 0, 0.0
    for n in range(cfg.n_steps):
        bg = acc.mean_force(bias_n)
        q, zc = step(q, bg)
        fe += 1.0
        if (n + 1) % cfg.dep_every == 0 and n >= cfg.n_eq:
            f, G = cv.mean_force(q, gradV(q), beta)
            acc.deposit(zc, f[..., 0])
        if yfun is not None and n % 20 == 0:
            y = yfun(q)[..., 1]
            iz = torch.clamp(((zc - g.xmin) / g.volume * nkz).long(), 0, nkz - 1)
            iy = torch.clamp(((y + math.pi) / (2 * math.pi) * nky).long(), 0, nky - 1)
            Hdiag.view(rows, -1).scatter_add_(1, iz * nky + iy,
                                              torch.ones_like(iz, dtype=dt))
        if (n + 1) % cfg.save_every == 0 and si < n_saves:
            p = kde_marginal(zc, g, cfg.bw_kde)
            out["F"][si] = acc.free_energy(mask)
            out["kl"][si] = kl_to_uniform(p, g)
            out["cov"][si] = _coverage(zc, g)
            if ref is not None and nt > 1:
                out["dcond"][si] = _cond_kl(Hdiag, ref["Hdiag"][0], 1.0)
            out["fe"][si] = fe * cfg.N
            Hdiag.zero_()
            si += 1
    out["z_final"], out["q_final"] = zc.unsqueeze(-1), q
    return out


def _abf_step(top, cv, q, bias_grid, g, h, beta, drift_cap=None, k_wall=0.0):
    """One unconstrained step with the running mean force applied as a bias.

    The bias enters as the gradient of -A_hat(xi(q)), i.e. the adaptive-biasing
    POTENTIAL form; the deposit is taken from the UNBIASED gradient, so the
    estimator is the same object every other arm reports.
    """
    gv = top.grad(q)
    gs = cv.grad_local(q)
    zc = cv.value(q)[..., 0]
    b = interp1d(zc, bias_grid, g)
    if k_wall > 0.0:
        # A CV with an inaccessible arc is studied on a restricted domain.  The
        # constrained arms are pinned inside it by construction; an unconstrained
        # one has to be held there by walls, or its walkers leave, get clamped
        # into the edge bin and poison the estimator.  Half-harmonic, the
        # standard construction.
        over = (torch.clamp(zc - g.xmax, min=0.0) + torch.clamp(zc - g.xmin, max=0.0))
        b = b + k_wall * over
    gb = gv.clone()
    gb[..., cv.support, :] = gb[..., cv.support, :] - b[..., None, None] * gs[..., 0, :, :]
    minv = (1.0 / top.mass).view(-1, 1)
    nz = torch.randn(q.shape, device=q.device, dtype=q.dtype)
    amp = ((2.0 * h / beta) ** 0.5) * torch.sqrt(minv)
    drift = -h * minv * gb
    if drift_cap is not None:
        drift = torch.clamp(drift, -drift_cap * amp, drift_cap * amp)
    return q + drift + amp * nz, zc


def run_opes(sy, cfg: MolCfg, rows: int, seed: int, ref=None):
    """Adaptive-biasing-POTENTIAL baseline, the ABP / SHUS / OPES family.

    ABF pushes with the running mean FORCE; this family builds a bias POTENTIAL
    from the visited density and pushes with its gradient, targeting a uniform
    marginal.  Both are adaptive and unconstrained, and they fail differently, so
    a comparison that has only one of them is comparing against half the field.

    The free-energy estimate is NOT the bias potential: it is the same Chapter-3
    mean-force accumulator every other arm reports, so no comparison here turns
    on an estimator difference.
    """
    dev, dt, g = sy.device, sy.dtype, sy.grid
    top, cv, beta, h = sy.top, sy.cv, sy.beta, sy.h
    torch.manual_seed(seed + 1777)
    nt = top.tor_idx.shape[0]
    full_cv = TorsionCV(top.tor_idx, top.mass, shift=cv.shift) if nt > 1 else cv
    phis = torch.full((rows, cfg.N, nt), sy.y0, device=dev, dtype=dt)
    phis[..., 0] = cfg.z0
    q = sy.ideal(phis.reshape(-1, nt)).reshape(rows, cfg.N, top.n_atoms, 3)
    acc = MeanForceAccumulator(rows, g, cfg.bw_mf, cfg.n_min, dev, dt)
    shus = ShusAccumulator(rows, g, beta, cfg.shus_eps_bw, dev, dt, cfg.shus_gain)
    mask = g.eval_mask(dev, dt)
    kw = (cfg.abf_wall_k if g.bc == "reflect" else 0.0)
    step = torch.compile(lambda q, bg: _abf_step(top, cv, q, bg, g, h, beta,
                                                 drift_cap=sy.drift_cap, k_wall=kw),
                         dynamic=False)
    gradV = torch.compile(lambda q: top.grad(q), dynamic=False)
    yfun = torch.compile(lambda q: full_cv.value(q), dynamic=False) if nt > 1 else None
    n_saves = cfg.n_steps // cfg.save_every
    Z = lambda *s: torch.zeros(s, device=dev, dtype=dt)
    out = {"F": Z(n_saves, rows, g.n), "F_prod": Z(n_saves, rows, g.n),
           "kl": Z(n_saves, rows), "cov": Z(n_saves, rows),
           "ess_fix": Z(n_saves, rows) + 1.0, "ess_anc": Z(n_saves, rows) + 1.0,
           "dcond": Z(n_saves, rows), "resid": Z(n_saves, rows), "fe": Z(n_saves),
           "lift_cov": Z(n_saves, rows), "fe_switch": torch.zeros((), device=dev,
                                                                  dtype=dt)}
    nkz = nky = (int(ref["Hdiag"].shape[-1]) if (ref is not None and "Hdiag" in ref)
                 else 36)
    Hdiag = Z(rows, nkz, nky)
    si, fe = 0, 0.0
    for n in range(cfg.n_steps):
        q, zc = step(q, shus.Fp)
        fe += 1.0
        shus.deposit(zc)
        if (n + 1) % cfg.shus_block == 0:
            shus.update(h * cfg.shus_block, cfg.shus_block)
        if (n + 1) % cfg.dep_every == 0 and n >= cfg.n_eq:
            f, G = cv.mean_force(q, gradV(q), beta)
            w = G[..., 0, 0] ** -0.5
            acc.deposit(zc, f[..., 0], weights=w)
        if yfun is not None and n % 20 == 0:
            y = yfun(q)[..., 1]
            iz = torch.clamp(((zc - g.xmin) / g.volume * nkz).long(), 0, nkz - 1)
            iy = torch.clamp(((y + math.pi) / (2 * math.pi) * nky).long(), 0, nky - 1)
            Hdiag.reshape(rows, -1).scatter_add_(1, iz * nky + iy,
                                                 torch.ones_like(iz, dtype=dt))
        if (n + 1) % cfg.save_every == 0 and si < n_saves:
            out["F"][si] = acc.free_energy(mask)
            out["F_prod"][si] = out["F"][si]
            out["kl"][si] = kl_to_uniform(kde_marginal(zc, g, cfg.bw_kde), g)
            out["cov"][si] = _coverage(zc, g)
            if ref is not None and nt > 1:
                out["dcond"][si] = _cond_kl(Hdiag, ref["Hdiag"][0], 1.0)
            out["fe"][si] = fe * cfg.N
            Hdiag.zero_()
            si += 1
    out["z_final"], out["q_final"] = zc.unsqueeze(-1), q
    return out


ARMS = {"constrained": run_constrained, "abf": run_abf, "opes": run_opes}
