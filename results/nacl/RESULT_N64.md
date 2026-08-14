# NaCl / water — the `N = 64` cell: **ABF-SUFFICIENT**

Frozen by `docs/SPEC_nacl_water.md` and Amendments 14–15. Pin `53dfb30`. All eight preregistered
seeds (4000–4007), `N·T = 100 ns` per ensemble, analysed only once all eight existed.

## Verdict

```
reference    ACCEPTED   ratio 0.0907 <= 0.5, complete, 250 ps x 61 r x 3 builds x 4 fam x 3 rep
Gate 0       PASS       0.0075 global / 0.0483 barrier   (WCA passes at 0.040; deca FAILED 0.61)
Gate A       PASS       max TV 1.000 (preregistered p(xi|Y) direction) vs 0.30
Gate B       PASS       8/8 seeds, both states           -- NON-BINDING, see below
Gate C       NO DEFICIT 0/8 seeds, both states
-> N = 64 is ABF-sufficient. No mFR arm is licensed for this cell.
```

## Gate C in detail — the gate the verdict rests on

Occupancy against the bias-aware target `Q*_k(t) ∝ ∫_Ck exp(−β[F_ref − B_t])`, over the second
half, 79 checkpoints × 8 seeds:

| state | min P/Q\* | median P/Q\* | % below 0.5 Q\* | longest contiguous deficit |
|---|---|---|---|---|
| CIP | 0.000 | 1.176 | 12.8 % | 10–70 ps (required: 312.5 ps) |
| SSIP | 0.866 | 0.995 | 0.0 % | 0 ps |

**The CIP zeros are counting noise, not a deficit.** CIP's mean target is 0.0311 of 64 walkers —
**1.99 walkers expected** — so `P(zero) = e^−1.99 = 13.7 %` against 12.8 % observed. Mean
occupancy 0.0417 versus mean target 0.0311 means CIP is on average **1.34× over-populated**. The
312.5 ps contiguity requirement filters exactly this fluctuation, which is what it is for.

SSIP tracks its target essentially perfectly (median 0.995) and never approaches the threshold.

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

`N = 64` is decided. §8.2 requires the **entire** map, so `N = 8, 16, 32` are running and the
study is not closed until they report: the frozen rule is the *smallest* `N` passing every gate,
so a smaller cell could still be establishment-limited. **The N = 64 verdict is final for N = 64
and provisional for NaCl.**
