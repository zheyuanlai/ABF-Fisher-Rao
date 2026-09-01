# Does a mean-force information-adequacy clock explain the campaign?

**No. Definition and prediction were frozen first (`configs/information_campaign/
prediction_v1.md`, commit `0d56426`); this is the result.** Exploratory by
construction — every cell was already closed and its ΔI_F already known.

## The proposal

After ethane/ZIF-8 came back HARMFUL in the regime the marginal clock selected,
the diagnosis was that the campaign had conflated *marginal establishment* with
*estimator establishment*. ABF estimates a conditional mean force whose accuracy
is governed by `SE[F'(z)] ~ sqrt(Var(f|z)/n_eff(z))`, not by how uniform
`p^xi(z)` is. The proposed replacement predictor was `T_info`: the time at which
a reference-free, block-bootstrapped PMF uncertainty `U_F(t)` reaches the same
relative convergence criterion already used for `T_marg`. FR would then be
gated to the window where the estimator is genuinely information-starved.

## Result: the predictor fails, for two independent reasons

Nine closed cells spanning the full outcome range, all computed from their own
saved accumulators with no new simulation:

| cell | ΔI_F | T_info | headroom H |
|---|---|---|---|
| lta_T150 | −31.92 % | 52.2 | +0.803 |
| cha_ethene_450 | −5.96 % | 57.6 | +0.657 |
| lta_T225 | −21.28 % | 43.2 | +0.653 |
| lta_T300 | −14.84 % | 43.2 | +0.653 |
| cha_propene_600 | −5.96 % | 49.6 | +0.557 |
| lta_v1_late | −0.21 % | 40.2 | +0.537 |
| cha_propene_450 | −5.72 % | 45.6 | +0.508 |
| **lta_T80** | **−35.14 %** | 31.2 | **+0.453** |
| **zif8_T300** | **+3.67 %** | 150.0 | **+0.400** |

**Spearman(H, ΔI_F) = −0.437, p = 0.240, n = 9.** The sign is as predicted and
ZIF-8 does have the least headroom, but the relation is not significant and —
decisively — it fails to separate the *strongest positive* (LTA T80, H = 0.453)
from the *harmful* cell (ZIF-8, H = 0.400). The frozen prediction
`H(LTA) > H(CHA) > H(ZIF-8)` is false: CHA overlaps LTA throughout.

### Reason 1 — it is a VARIANCE clock, and these endpoints are BIAS

| | U_F (bootstrap uncertainty) | e_F (true error) | U_F/e_F |
|---|---|---|---|
| ZIF-8 300 K, final | 0.0266 | 0.2747 | **0.097** |
| CHA ethene, final | 0.2872 | 1.8182 | **0.158** |

**99 % (ZIF-8) and 98 % (CHA) of the final squared error is not variance.** Of
ZIF-8's residual, reference noise explains only 5.7 %, leaving ~93 % as genuine
estimator bias. `U_F` falls 3.4× over the run while `e_F` falls 1.1× — the
uncertainty keeps improving long after the error has stopped, because the error
is set by a bias that more sampling does not touch. A clock built on `SE[F']`
therefore cannot say when the estimator is finished.

This is not a new discovery so much as the fifth independent arrival at the same
place. The mechanism campaign found η_bias never below 0.121 on any system and
concluded kernel-ABF free-energy endpoints are bias-dominated *by construction*;
the fibre-horizon audit found 96–97 % of FR's damage was bias; and the ZIF-8
result itself found 98 %.

### Reason 2 — it is not robust to an arbitrary analysis knob

| block length | 2 ps | 5 ps | 10 ps | 20 ps |
|---|---|---|---|---|
| U_F final | 0.0234 | 0.0259 | 0.0282 | 0.0300 |
| **headroom H** | **+0.260** | **+0.400** | **+0.567** | **+0.733** |

The *variance* estimate is stable (28 % across a 10× change), so bias dominance
is not an artifact. But `T_info` swings **3×**, because the relative-convergence
rule is applied to a curve whose shape depends on how many blocks exist at each
t. A predictor that moves 3× on a choice with no physical content is not a
property of the run.

## Instrument validation, and a bug it caught

Before any claim, the single-run block bootstrap was required to track the
across-seed spread (the honest uncertainty, since seeds are independent runs).
It initially disagreed by **28–54×** on the LTA cells. That was not the
hypothesis failing — it was two errors in my own cell table: LTA uses a
**periodic** CV and has **no `min_count` regularizer** (a plain
`smooth(f)/smooth(c)` ratio), so the reconstructed block moments were garbage.
After correcting both, every cell sits at 0.66–1.77. Without the pre-committed
instrument check this would have been reported as a result.

## Conclusion, per the frozen falsifier

The frozen prediction stated: *"If the relation fails, STOP — do not invent a
third predictor."* It failed. So:

**Do not build the information-gated FR campaign.** Its gate is a variance clock
on a bias-dominated endpoint, and it is not robust.

More constructively, the recurring diagnosis now has five independent
confirmations, and it points somewhere specific. If ~95 % of the endpoint error
is kernel-smoothing bias, then *where the walkers are* is close to irrelevant and
the dominant lever is the estimator's **bandwidth** — which the mechanism
campaign already measured as a **58× endpoint-MSE win** (h 0.07 → 0.005), against
the 1.4–1.7× that allocation was ever worth. A consistency check on ZIF-8: the
ABF arm underestimates the barrier by 0.335 kJ/mol, and Gaussian kernel bias at
h = 0.20 Å with the observed peak curvature predicts ≈ 0.8 kJ/mol — the same
order.

The licensed next experiment is therefore **not** another allocation predictor,
and **not** ZIF-8 at 250/350 K. It is a direct bandwidth sweep on a cell where
the reference is already trusted.
