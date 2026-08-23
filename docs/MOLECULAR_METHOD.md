# RC-WFR on molecules: the construction

## The decomposition

Write the sampled measure as marginal times conditional,

    rho_t(dq) = p_t(z) rho_t(dq | z),      z = xi(q),

and give the two factors different jobs:

    p_t(z)          --- WFR --->            eta(z)          (uniform, or any target)
    rho_t(dq | z)   --- constrained MD ---> nu^xi(dq | z)   (the physical conditional)

Only the second has to be right for thermodynamic integration; the first only
has to be spread out.  That split is what makes RC-WFR different from ABF or
OPES, which reshape the marginal by modifying the *forces* and leave the
conditional to look after itself.

## One outer iteration

Given `N` replicas, each a configuration `q_i` with `xi(q_i) = z_i`:

1. **Wasserstein transport** of the labels.  For a uniform target the W-gradient
   flow of `KL(p || u)` is heat flow, realised either as `z <- z + sqrt(2 kappa dtau) eta`
   or as the deterministic probability flow `z <- z - kappa dtau grad log p_hat(z)`.
2. **Lift** each configuration from `Sigma(z_i)` to `Sigma(z_i')`.  This is the
   step the whole campaign is about; see below.
3. **Fisher-Rao reallocation.**  Weights `a_i = (u(z_i)/p_hat(z_i))^theta`
   followed by exact-N systematic resampling.  Selection depends on `z` only, so
   given `z` it cannot touch the conditional and cannot bias a conditional
   average.
4. **Constrained relaxation.**  `n_cond` steps of projected Brownian dynamics on
   `Sigma(z_i')`, then one mean-force deposit at the END of the window.

## The lift

Any `dq` with `grad xi . dq = dz` moves a configuration between fibers; they
form an affine space over `T_q Sigma(z)`.  Four members, in increasing order of
what they know:

| lift | knows | on a torsion |
|---|---|---|
| minimum-norm horizontal | the ambient (mass) metric | SHAKE along `M^-1 grad xi`; bends bonds to buy the constraint |
| internal-coordinate rotation | the molecule's internal coordinates | rotate the distal fragment about the torsion axis; exact, distortion-free |
| conditional map / refresh | `nu(y \| z)` for a promoted slow mode `y` | rotate `y` to `F^-1_{z'}(F_z(y))`, or to a draw from `nu(. \| z')` |
| **Metropolis conditional move** | `nu_hat(y \| z)` as a *proposal* only | propose, then accept by the exact ratio |

The last one is the practical answer and the only one whose correctness does not
depend on the quality of what it learned.

### Why the Metropolis move is exact here

Rotating the distal fragment of torsion `y` about its central bond:

* is an isometry of `R^{3A}`, so Lebesgue measure is preserved;
* changes `y` and nothing else -- every bond length, bond angle and other
  torsion is invariant, because the two planes that define any other dihedral
  are both carried by the same rotation;
* leaves the internal-coordinate Jacobian alone, since that Jacobian is a
  product of bond and angle factors and never depends on a torsion.  The
  reference measure along the curve `y -> q(y)` is therefore flat in `y`;
* does not move any of the four atoms that define `xi`, so `grad xi` and hence
  `det G` are invariant, and the move preserves the RIGID measure the
  constrained sampler actually produces as well as the conditional one.

Hence

    accept  with  min(1, exp(-beta[V(q') - V(q)]) * nu_hat(y|z') / nu_hat(y'|z'))

is an independence Metropolis move whose invariant law is exactly the
constrained ensemble on `Sigma(z')`.  `nu_hat` sets the acceptance rate and
nothing else.  It carries a 2% uniform background, which bounds the proposal
away from zero, makes the chain ergodic, and lets the same density serve both
the draw and the ratio.

This is what removes the failure mode that killed the learned refresh: a lift
built from the ensemble's own samples is self-referential, and an uncorrected
refresh from a degenerate `nu_hat` *destroys* the fiber relaxation the dynamics
would otherwise have made.  Corrected, the same degenerate `nu_hat` merely makes
the move a no-op.

## Estimators

Constrained Brownian dynamics with mobility `M^-1` on the bare potential samples
the RIGID measure `e^{-beta V} sigma^M_Sigma`, not `nu^xi`.  Rather than
simulate the Fixman-corrected potential (which needs second derivatives of `xi`
every step), each deposit carries the weight `(det G)^{-1/2}` and the shared
accumulator's self-normalisation does the rest:

    F'(z) = E_rgd[w f] / E_rgd[w],     w = (det G)^{-1/2},
    f     = (grad xi^T M^-1 grad V)/G - beta^{-1} div(M^-1 grad xi / G).

The divergence term is assembled from Hessian contractions of `xi` alone, and a
torsion depends on four atoms, so that Hessian is 12 x 12 no matter how large
the molecule is.  Measured reweighting ESS: 0.98.

## Cost

Counted in gradient evaluations of `V` per replica: one per dynamics step, one
per mean-force deposit, and two energy evaluations per Metropolis move.  Every
comparison is made on that axis, and `fe` is stored per save rather than assumed.
