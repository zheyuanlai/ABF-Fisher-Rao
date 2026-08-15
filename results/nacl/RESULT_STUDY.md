# NaCl / water: **ABF-SUFFICIENT.** The study closes without licensing an mFR arm.

Closed 2026-08-15 23:40 UTC. Frozen by `docs/SPEC_nacl_water.md` and Amendments 14-16. Sampler
pinned `53dfb30`; analysis at `a268630` (the report carries `analysis_provenance` — a
`gates_report.json` lacking that block came from the superseded worktree and is not a verdict).

## Verdict

```
reference    ACCEPTED   ratio 0.0907 <= 0.5, 250 ps x 61 r x 3 builds x 4 fam x 3 rep
Gate 0       PASS       0.0075 global / 0.0483 barrier   (campaign best; deca FAILED at 0.61)
Gate A       PASS       max TV 1.000 (preregistered p(xi|Y) direction) vs 0.30
Gate B       PASS       8/8 seeds, both states, both cells   -- NON-BINDING, see RESULT_N64.md
Gate C       NO DEFICIT 0/8 seeds on SSIP, the powered state, at BOTH classifiable cells
-> the map is COMPLETE and no cell is eligible.
   NaCl is not an mFR candidate under the preregistered budget. STOP.
```

The frozen §8.2 rule is the **smallest N passing every gate**. The classifiable ladder is
`{32, 64}` and neither cell is under-established, so there is nothing for mFR to repair.

## The map, and why it has two cells rather than four

| cell | Gate C, SSIP | lambda (SSIP / CIP) | status |
|---|---|---|---|
| N = 64 | 0/8 deficient | 61.45 / 1.57 | classified: ABF-sufficient |
| N = 32 | 0/8 deficient | 30.75 / 0.81 | classified: ABF-sufficient |
| N = 16 | — | <= 16 / — | **NOT COMPUTABLE, a priori** |
| N = 8 | — | <= 8 / — | **NOT COMPUTABLE, a priori** |

`N = 16` and `N = 8` were struck **without being run**, and the exclusion is arithmetic rather
than a judgement: the basin targets partition, so `Q*_k <= 1`, and strictly `< 1` whenever a
second basin carries positive target — hence `lambda_k = N Q*_k < N`. A cell with `N <= 16`
cannot reach the `lambda >= 16` power floor for **any** state, whatever the sampling. That floor
is not arbitrary: `lambda = 16` is exactly where the resolvable deficit `2/sqrt(lambda)` equals
the 50 % deficit Gate C tests, so below it the gate is asked to detect an effect smaller than
its own resolution. `nacl_gates.map_completeness()` derives this and refuses to emit a
study-level verdict while any *classifiable* cell is missing. `N = 16` misses by 0.64 walkers;
relaxing the threshold to admit it is the forbidden retune-against-a-result, and was refused.

**Projection recorded before the cell ran** (`CLOSURE_PRECOMMIT.md`): SSIP `lambda ~ 30.7`,
CIP `~ 0.78`. **Realised: 30.75 and 0.81.** No material departure, so the bias-aware target
behaved as the reference predicted.

## Gate C is unpowered at CIP in both cells, and a second statistic carries it

`lambda_CIP` is 1.57 and 0.81, so "occupancy < 0.5 Q\*" is arithmetically "is CIP empty right
now" and the gate is reported NON-BINDING there. **An unpowered gate cannot support "no deficit"
any more than it supports "deficit"**, so CIP is cleared by a statistic that is powered on its
own terms: the gate asks for a deficit *sustained* over 0.2 T, so occupancy is averaged over a
sliding window of exactly that length, with the standard error taken from the spread across the
8 **independent** seeds — no Poisson assumption, no autocorrelation model.

| cell | state | worst window ratio | 2-sigma band | worst single seed-window |
|---|---|---|---|---|
| N = 64 | CIP | 1.111 +- 0.078 | [0.954, 1.267] | 0.765 |
| N = 64 | SSIP | 0.980 +- 0.003 | [0.974, 0.986] | 0.960 |
| N = 32 | CIP | 1.371 +- 0.106 | [1.160, 1.582] | 0.679 |
| N = 32 | SSIP | 0.982 +- 0.004 | [0.974, 0.991] | 0.957 |

Every worst-case window sits above the 0.5 threshold by more than 2 sigma, and no single
seed-window in any cell falls below 0.679. A sustained deficit of the size Gate C tests is
excluded **with power** at both states of both cells.

## The basins are sampled inside, not merely on aggregate

Gate C is basin-integrated and SSIP spans 88 % of the domain, so a redistribution *within* SSIP
that preserved its integral would be invisible. Measured (`nacl_audit_within_basin.py`), SSIP is
clean in both cells — quarter ratios 0.942-1.026 (N=64) and 0.949-1.018 (N=32), shape TV 0.025
in both, outermost three grid points holding 4.3-4.5 % of walkers against 4.1 % of target. No
wall accumulation. CIP's larger TV (0.215, 0.205) is real structure and not a jam: ~72-74 % of
its walkers sit around the 0.26 nm minimum against a ~56 % target, thinning to 0.72-0.75 toward
the barrier, and **no quarter in either cell falls below the 0.5 threshold**.

## Physics

CIP minimum at 0.26 nm, desolvation barrier **5.34 kT** at ~0.35 nm, and beyond it a landscape
flat within ~0.9 kT. The SSIP minimum at 0.52 nm is real but merges under the frozen 2 kT rule,
so **NaCl has one genuine metastable state**. The independently built constrained-TI reference
reproduces the published 100 ns ABF PMF to better than **0.15 kT** across the physical region —
computed in the same box, so this validates the implementation and is silent on finite-size
systematics.

## The caveat, pre-committed at `addfbed` before any of this was read

NaCl's hydration varies **14-83x more across `r` than at fixed `r`**, against methane's 5.4x. NaCl
therefore has little structure orthogonal to the reaction coordinate, and **"mFR had nothing to
work with" is a live alternative to "mFR was not needed."** This null is **weaker than
methane's** and must not be reported as a second independent null of equal strength. Gate A at
1.000 is a statement about NaCl's physics, not a strong gate.

## What this contributes to the campaign

A fourth system that is **ABF-sufficient**: the ABF baseline already establishes every state the
gates can resolve, so marginal reallocation has no deficit to repair. It joins methane as a
literature-anchored negative, with the weakness above attached, and does **not** add a positive
to the four-regime map. See `docs/V2_PREREGISTRATION.md` and the campaign report.

**What would have changed the verdict:** a sustained occupancy deficit on SSIP at either cell,
or a CIP windowed band straddling 0.5 (which would have made CIP UNKNOWN rather than clear and
weakened the closure to "ABF-sufficient on the state that can be measured"). Neither occurred.
Both branches were written down in `CLOSURE_PRECOMMIT.md` before the number existed.
