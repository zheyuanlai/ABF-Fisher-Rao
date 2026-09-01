# T_info retrospective test — PREDICTION, frozen before computing anything

Written 2026-09-01, **before** `src/information_adequacy.py` was run on any
cell. Commit this file before the results commit; the point is that the clock's
definition and the expected ordering are fixed in advance, so the test cannot be
rescued by trying definitions until one fits (the "unequal tuning budget"
defect this project has recorded).

## Status: EXPLORATORY, not confirmatory

Every cell below is already closed and its ΔI_F is already known to me. This can
therefore only *fail to falsify* the new predictor; it cannot confirm it. A
prospective test needs the 250/350 K screens with T_info recorded before
unblinding.

## The frozen definition

ABF estimates a CONDITIONAL mean force
`F'(z) = E_nu[ f(q) | xi(q)=z ]`, whose accuracy is governed by
`SE[F'(z)] ~ sqrt( Var(f|z) / n_eff(z) )`, NOT by how uniform `p^xi(z)` is.

From the saved cumulative kernel-weighted accumulators, per save t and bin g:

    W(t,g) = smooth(csum_p)(t,g)  = eff_counts(t,g) - eff_counts(t_burn,g)
    Y(t,g) = smooth(fsum_p)(t,g)  = mean_force(t,g) * (W(t,g) + min_count)

Per-block increments dW, dY over blocks of `block_saves` consecutive saves.
Block-bootstrap the blocks available up to t, form
`F'* = sum(dY*)/(sum(dW*) + min_count)`, integrate to a PMF, remove the additive
gauge, and take

    U_F(t) = sqrt( mean_g Var_*( F*_t(z_g) ) )        [reference-FREE]

`T_info` is then the SAME relative rule already frozen for T_marg: with
`U_inf = median of U_F over the last 20% of the run` and `U_0 = U_F` at the end
of the ABF warm-up,

    T_info = first t with U_F(t) <= U_inf + 0.2 (U_0 - U_inf), sustained 0.1 T.

Instrument check, required before any claim: the single-run block bootstrap
`U_F^block` must track the across-seed spread `U_F^seed` (which is the honest
uncertainty, since the seeds are genuinely independent runs). If they disagree
badly the block bootstrap is not measuring what it claims and the test is void.

## The prediction

Headroom for FR is `H = (T_info - t_FR) / T`. The hypothesis is that FR can only
help while the mean-force estimator is still information-starved, so:

| cell | known ΔI_F | predicted T_info | predicted H |
|---|---|---|---|
| LTA T-sweep (4 cells) | **strong positive**, −14.8% to −35.1% | LATE | large > 0 |
| CHA ethene/propene | small positive, ≈ −6% | INTERMEDIATE | small > 0 |
| **ZIF-8 300 K** | **HARMFUL, +3.67%** | **EARLY** | **≈ 0 or < 0** |

Concretely, ranked: `H(LTA) > H(CHA) > H(ZIF-8)`, and the sign of ΔI_F should
track the sign of H.

**What would falsify it:** ZIF-8 showing large positive headroom, or the LTA
cells showing none, or no monotone relation between H and ΔI_F across the
cells. If the relation fails, STOP — do not invent a third predictor. That
outcome would say the premise "marginal birth-death is a robust ABF
accelerator" is wrong in general, and the project should be reframed around the
mechanism map rather than another tuned classifier.

## What this does NOT test

T_info is computed here from ABF-arm data. Whether *gating* FR on it recovers
the WCA benefit while removing the ZIF-8 harm is a separate, prospective,
fresh-seed experiment and is not addressed by this audit.
