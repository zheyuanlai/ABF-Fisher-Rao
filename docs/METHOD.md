# RC-WFR-TI: Reaction-Coordinate Wasserstein-Fisher-Rao Thermodynamic Integration

## The object we want

Physical canonical measure `nu(dq) = Z^-1 exp(-beta V(q)) dq`, reaction coordinate
`z = xi(q)`, free energy `F(z) = -beta^-1 log (dnu^xi/dz)`.  Disintegration

    nu(dq) = exp(-beta F(z)) dz  nu^xi(dq | z),

mean force  `F'(z) = E_{nu^xi(.|z)}[ f(Q) ]`  (for `xi(q)=x` linear, `f = dV/dx`).

RC-WFR targets the ARTIFICIAL joint law

    nubar(dq, dz) = u(z) dz  nu^xi(dq | z),      u = uniform on M,

i.e. uniform coverage of `z` with the CORRECT physical conditional inside every
fiber.  No bias potential is ever estimated.  `F` is reconstructed by TI from the
conditional mean force.

## The marginal flow

Population density `p_t(z)` of replica labels.  Gradient flow of `KL(p||u)` in the
Wasserstein-Fisher-Rao geometry:

    d_t p = kappa div( p grad log(p/u) ) - lambda p ( log(p/u) - E_p log(p/u) )

For uniform `u` this is exactly

    d_t p = kappa Laplacian(p)  -  lambda p ( log p - E_p log p )
            \_____ W ______/      \________ FR ________________/

Dissipation identity: d_t KL(p||u) = -kappa I(p|u) - lambda Var_p(log(p/u)) <= 0.

* W  = local transport / support expansion / DISCOVERY   (rate ~ kappa pi^2 / L^2)
* FR = nonlocal reallocation / ESTABLISHMENT             (rate ~ lambda, L-independent)

FR cannot enlarge `supp p`; W can.  That is the complementarity being tested.

## Particle algorithm (one outer iteration)

    1. CONDITIONAL   n_cond steps of constrained Langevin for Q_i on Sigma(Z_i);
                     accumulate mean-force samples f(Q_i) at Z_i.
    2. W             Z_i <- Z_i + sqrt(2 kappa dtau) eta_i   (reflect / wrap)
                     (or deterministic probability flow Z_i <- Z_i - kappa dtau grad log p)
    3. LIFT          move Q_i from Sigma(Z_i) to Sigma(Z_i^+); relax.
    4. FR            p_hat by KDE (or GMM);  a_i = (u/p_hat(Z_i))^theta;
                     systematic-resample exactly N replicas (carry Q_i with Z_i).
    5. TI            F_hat(z) = integral of the binned mean-force estimate.

`theta = 1 - exp(-lambda dtau)` is the exact finite-time FR step for a frozen target.

## Cost model (the only fair currency)

One FORCE EVALUATION = one evaluation of grad V at one configuration.
Every arm is compared at MATCHED total force evaluations.  W steps, FR steps,
KDE/GMM fits and the TI quadrature cost ZERO force evaluations; the lift costs
whatever relaxation steps it consumes.  Wall-clock is reported separately.

## What is genuinely new vs what is not

NOT new: WFR sampling (Lu-Lu-Nolen; Lu-Slepcev-Wang), birth-death enhanced MD
(Pampel et al. 2023), diffusion+selection in a CV (Lelievre-Rousset-Stoltz
parallel adaptive dynamics with selection), GMM/WFR location-vs-mass splitting
(Yan-Wang-Rigollet), stratified constrained TI, umbrella sampling / REUS,
weighted ensemble.

Narrow claim under test: *a bias-free free-energy method that evolves the replica
population by a WFR flow on the low-dimensional reaction-coordinate marginal while
preserving physical conditional sampling on the fibers and reconstructing F by TI.*

## Falsifiable hypotheses (preregistered; see PREREGISTRATION.md)

H0 (mechanism)  W+FR beats W-only and FR-only at matched cost on a designed
                discovery+establishment-limited system.
H1 (vs ABF/ABP) RC-WFR beats ABF and SHUS/OPES at matched force evaluations on
                >= 2 systems.
H2 (vs classical) RC-WFR beats fixed-window stratified TI and REUS/WE.
H3 (validity)   the lift does not introduce a free-energy bias floor that
                dominates at the budgets where H1/H2 are claimed.
H4 (geometry)   smooth FR is not reproduced by plain count balancing.
                (Prior from the ABF/ABP campaign: expect H4 to FAIL.)

The decisive predictor is the LIFT RATIO

    R_lift = tau_cond_needed / tau_transport_under_optimal_bias

RC-WFR can only win where R_lift << 1.
