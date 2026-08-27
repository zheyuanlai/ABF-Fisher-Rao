# q-r decoupling: what is frozen across every stage

Frozen protocol: `docs/QR_DECOUPLING_PREREGISTRATION.md`.

Transferred **unchanged** from the clean-v2 backbone, so the only thing that
differs between this campaign and the one it follows is the allocation:

    beta 4.0 | dt 0.002 | K 256 | n_steps 50000 | eval_every 500
    ABF: binned_smooth, post_propagation, h 0.05, update_every 10, min_count 1.0
    domain x [-3, 3], y [-2.5, 3.5]; profile grid 401; x_tilt 0.1021665783

`post_propagation` is not a preference here: the qr arms read the same eligible
observation stream the accumulator does, and the engine only builds it on that
path.

## Frozen allocation settings, shared by every arm

    n_cells 32              ~8 replicas per cell at K = 256
    opportunity_every 500   an opportunity is not an obligation
    burnin 0.20 / stop 0.80 three phases; the window closes before the run does
    floor_fraction 0.25     structural: a_j is 0 off-mask and the walls are real
    benefit_threshold 0.10  predicted risk reduction that pays for genealogy
    eps_gene 0.1            genealogy inflation the rejuvenation hold allows
    history_capacity 400    tau fit window, in ABF observations
    theta 1.0               exact mass projection; the FR flow is theta < 1

A3, A4a, A4b and A5 share every one of these. **Only `r` differs.** An
arm-specific anything else would make a margin unattributable, which is how the
previous four campaigns lost their positives.

## The arms

    A0   qr absent                    plain ABF
    A1   clean-v2 physical BD         historical failure control (its own config)
    A2   arm: A2                      mass only -- must be identical to A0
    A3   arm: A3                      count balancing, the incumbent
    A4a  arm: A4a                     r propto sqrt(a)        static leverage
    A4b  arm: A4b                     r propto sqrt(a Gamma)  + measured difficulty
    A5   arm: A5, rho 0.5             ESS-constrained

## Seeds

    5100-5115   Stage 1 calibration (16, ABF-only, per kappa cell)
    5200-5231   Stage 2 confirmatory (32 paired)

Fresh blocks: no seed here has been used by v2, v3, v4 or clean-v2.
