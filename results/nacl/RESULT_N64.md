# NaCl / water — the `N = 64` cell: **ABF-SUFFICIENT**

Frozen by `docs/SPEC_nacl_water.md` and Amendments 14–15. Pin `53dfb30`. All eight preregistered
seeds (4000–4007), `N·T = 100 ns` per ensemble, analysed only once all eight existed.

## Verdict

```
reference    ACCEPTED   ratio 0.0907 <= 0.5, complete, 250 ps x 61 r x 3 builds x 4 fam x 3 rep
Gate 0       PASS       0.0075 global / 0.0483 barrier   (WCA passes at 0.040; deca FAILED 0.61)
Gate A       PASS       max TV 1.000 (preregistered p(xi|Y) direction) vs 0.30
Gate B       PASS       8/8 seeds, both states           -- NON-BINDING, see below
Gate C       NO DEFICIT 0/8 seeds -- on SSIP, the only state with the power to say so
             CIP is NON-BINDING (lambda = 1.72 expected walkers; needs >= 16)
-> N = 64 is ABF-sufficient. No mFR arm is licensed for this cell.
```

## Gate C in detail — the gate the verdict rests on

Occupancy against the bias-aware target `Q*_k(t) ∝ ∫_Ck exp(−β[F_ref − B_t])`, over the second
half, 79 checkpoints × 8 seeds:

| state | min P/Q\* | median P/Q\* | % below 0.5 Q\* | longest contiguous deficit |
|---|---|---|---|---|
| CIP | 0.000 | 1.176 | 12.8 % | 10–70 ps (required: 312.5 ps) |
| SSIP | 0.866 | 0.995 | 0.0 % | 0 ps |

**Gate C at CIP has no power, and the CIP claim rests on a different statistic, not on the
gate.** With `lambda_min = Q*_CIP x N = 1.57` expected walkers, "occupancy < 0.5 Q*" is
arithmetically identical to "the state is empty right now", and the smallest deficit resolvable
at 2 sigma is **160 %** -- i.e. none. The gate is reported NON-BINDING at CIP and excluded from
the verdict; **SSIP (lambda_min = 61.45) is the state that carries it.**

**An unpowered gate cannot support "no deficit" any more than it supports "deficit", so the
replacement statistic has to be powered on its own terms -- measured, not asserted.** An earlier
draft of this document cleared CIP with a time average it called "well estimated" and never put
an error bar on it. Done properly (`scripts/nacl_audit_cip_power.py`): the gate does not ask
whether CIP is empty at time `t`, it asks for a deficit **sustained 312.5 ps**, so the powered
form averages occupancy over a 312.5 ps sliding window (33 checkpoints) instead of one
checkpoint. The error bar needs no Poisson assumption and no autocorrelation model, because the
**8 seeds are independent ensembles** and their spread is the standard error directly.

| state | worst of 47 windows | ratio P/Q\* | 2-sigma band | worst single seed-window |
|---|---|---|---|---|
| CIP | 810-1130 ps | 1.111 +- 0.078 | **[0.954, 1.267]** | 0.765 |
| SSIP *(positive control)* | 1240-1560 ps | 0.980 +- 0.003 | [0.974, 0.986] | 0.960 |

The worst CIP window sits **1.9x the 0.5 threshold at its lower 2-sigma bound**, and the worst
single seed-window is 0.765. A sustained deficit of the size Gate C tests is excluded at CIP with
power, so the cell is ABF-sufficient on both states rather than on SSIP with CIP unknown. SSIP is
carried as the positive control on purpose: a check exercised only where it passes cannot be
distinguished from one that always passes.

Supporting counting argument:

**The CIP zeros are counting noise, not a deficit.** CIP's mean target is 0.0311 of 64 walkers —
**1.99 walkers expected** — so `P(zero) = e^−1.99 = 13.7 %` against 12.8 % observed. Mean
occupancy 0.0417 versus mean target 0.0311 means CIP is on average **1.34× over-populated**. The
312.5 ps contiguity requirement filters exactly this fluctuation, which is what it is for.

SSIP tracks its target essentially perfectly (median 0.995) and never approaches the threshold.

## Within the basin that carries the verdict, not just its integral

Gate C is **basin-integrated**, and SSIP spans 88 % of the domain. Any redistribution *inside*
SSIP that preserves its integral is invisible to the gate -- walkers jammed against a wall would
still "hold their target population". Prompted by the C60 session's finding that a relaxation
guard set at the explosion scale could not fire on a sterically jammed water, whose general form
is that **a guard at the catastrophic scale cannot protect against the metastable scale**.
Measured (`scripts/nacl_audit_within_basin.py`, `within_basin_audit.json`):

| basin | integrated P/Q | q1 | q2 | q3 | q4 | shape TV |
|---|---|---|---|---|---|---|
| SSIP | 0.9890 | 0.942 | 0.965 | 0.991 | 1.026 | **0.025** |
| CIP | 1.3424 | *(no target mass)* | 1.773 | 0.800 | 0.724 | 0.215 |

**SSIP is clean**: a gentle monotone gradient, quarter masses matching target to within a point
or two, and the outermost three grid points holding 4.45 % of walkers against 4.10 % of target.
No jam. The verdict's load-bearing state is not a coarse-graining artifact.

**CIP's larger TV is real structure, not a wall effect**: 74 % of its walkers sit in
[0.24, 0.27] nm around the 0.26 nm minimum against a 56 % target, thinning to 0.80 and 0.72
toward the barrier. **No quarter falls below the 0.5 deficit threshold.** Its inner quarter holds
0.42 % of walkers against 0.00 % of target, and restricting CIP to SPEC §2.1's frozen 15 kT
physical window changes its integrated ratio by nothing at all (1.3424 either way).

*The first version of this audit reported `P/Q = 1274` for that inner quarter* -- 0.0175 % of
walkers divided by a target a thousand times smaller. A ratio needs **both** arguments'
populations; the tool now carries the masses and suppresses the ratio outright where the target
has none. The spectacular number was an artifact of the instrument, and the basin ratio it was
supposedly qualifying never moved.

## Gate B is non-binding and is reported as such

`T_hit = 0.5 ps` on every seed is the **first recordable frame**: the published start
(`r = 0.30 nm`) is 0.040 nm from the first state boundary, 0.095 ps of ballistic transit against
a 0.5 ps trace interval. Gate B could not have failed. Independent far-threshold arrivals,
against an exact `Φ⁻¹(q^(1/n))` fastest-of-64 floor, do carry information:

| threshold | floor | observed | × floor |
|---|---|---|---|
| 0.52 nm (SSIP min) | 0.14 ps | 1.0–1.5 ps | 6.9–10.4× |
| 0.70 nm (outer) | 0.26 ps | 2.5–3.0 ps | 9.5–11.4× |
| 1.00 nm (dissociated) | 0.46 ps | 6.5–9.5 ps | 14.2–20.7× |

Discovery is genuinely diffusive and genuinely fast; the Gate B *number* is not what shows it.

## Physics

CIP minimum at 0.26 nm, desolvation barrier **5.34 kT** at ~0.35 nm, and beyond it a landscape
flat within ~0.9 kT. The SSIP minimum at 0.52 nm is real but merges under the frozen 2 kT rule:
**NaCl has one genuine metastable state.** Our independently built constrained-TI reference
reproduces the published 100 ns ABF PMF to **better than 0.15 kT** across the entire physical
region — with the caveat that both were computed in the same box, so this validates the
implementation and is silent on finite-size systematics.

## The caveat, pre-committed before this verdict was read (commit `addfbed`)

NaCl's hydration varies **14–83× more across `r` than at fixed `r`**, against methane's 5.4×. So
NaCl has little structure orthogonal to the reaction coordinate, and **"mFR had nothing to work
with" is a live alternative to "mFR was not needed".** This null is therefore **weaker than
methane's** and must not be reported as a second independent null of equal strength. Gate A at
1.000 is a statement about NaCl's physics, not a strong gate.

## Status of the study

**The N ladder is exposed and the guard now runs before classification.** `lambda = Q* N` falls
with `N`, so Gate C loses power as the ladder descends. Using the corrected statistic (the
**minimum** of `Q*(t) N` over the judged window, `lambda_min` = 61.45 at SSIP and 1.57 at CIP --
the earlier 62.3 / 1.72 were the pre-fix means over all checkpoints), the ladder projects to
CIP 1.57 / 0.78 / 0.39 / 0.20 and SSIP 61.45 / 30.72 / 15.36 / 7.68 for N = 64 / 32 / 16 / 8.

**N = 16 and N = 8 are structurally unclassifiable, and this is arithmetic rather than a
projection.** The targets partition, so `Q*_k <= 1` for every state, so `lambda_k = N Q*_k <= N`.
Any cell with `N <= 16` therefore has `lambda < 16` for *every* state -- with equality only if a
single basin held the entire target, which CIP's nonzero target forbids. Sampling cannot raise a
bound that does not depend on the sample. The threshold is not arbitrary either: `lambda = 16` is
exactly where the resolvable deficit `2/sqrt(lambda)` equals the 50 % deficit the gate tests, so
below it Gate C is asked to detect an effect smaller than its own resolution.

**Gate C is therefore NOT COMPUTABLE at N = 16 and N = 8, and those cells cannot be classified.**
Note the shape of the temptation: N = 16 misses by 0.64 walkers, and relaxing the threshold from
16 to 15.36 would admit it. That is precisely the forbidden move -- retuning a frozen threshold
against a result -- and the guard is worth having only if it binds when it is inconvenient.
Without the guard, §8.2's "smallest N passing every gate" rule searches directly into the cells
where P(empty) reaches 0.78 and would license an mFR arm on counting noise -- which is exactly
the defect this repository already retracted once
(`results/deca/screen_RETRACTED_no_min_count_guard/`: a state that could not hold walkers, on
which "Gate C fired", `licenses_mfr: true`).

`N = 64` is decided. §8.2 requires the **entire** map, so `N = 8, 16, 32` are running and the
study is not closed until they report: the frozen rule is the *smallest* `N` passing every gate,
so a smaller cell could still be establishment-limited. **The N = 64 verdict is final for N = 64
and provisional for NaCl.**
