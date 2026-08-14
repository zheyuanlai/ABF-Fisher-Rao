# Finite-size gate: INCONCLUSIVE, not FAIL — the criterion cannot resolve what it tests

SPEC §1.3, run **after** the reference and screen were built. 1024-water box
(`L = 3.134273 nm`, `rho = 0.9966 g/cm^3`, half-box 1.567 nm), 3 r-points × 16 replicas ×
(50 + 200) ps, 1.18 h on GPU 3. Same constrained-TI estimator as the accepted reference.

---

## What the script printed, and why it is the wrong headline

| `r` (nm) | `f` 1024-water | `f` 512-water | difference | preregistered verdict |
|---|---|---|---|---|
| 0.70 | −10.909 | −10.912 | **+0.0013 kT/nm** | PASS |
| 0.80 | +1.058 | +0.795 | **+0.1060 kT/nm** | FAIL |
| 0.90 | −5.497 | −8.362 | **+1.1563 kT/nm** | FAIL |

The preregistered rule compares point estimates against a 0.1 kT/nm tolerance and, read
literally, **fails**. That reading does not survive contact with the uncertainties.

| `r` (nm) | difference | combined SEM | significance |
|---|---|---|---|
| 0.70 | +0.0013 | ±0.670 | **0.00 σ** |
| 0.80 | +0.1060 | ±0.500 | **0.21 σ** |
| 0.90 | +1.1563 | ±0.671 | **1.72 σ** |

> **The 0.1 kT/nm tolerance is roughly 5–7× smaller than the uncertainty of the measurement it
> gates.** At 16 replicas × 200 ps the combined SEM on the difference is ~0.5–0.67 kT/nm, so the
> criterion cannot distinguish a real shift from zero at the threshold it specifies. Not one of
> the three points is resolved at 2σ, including both nominal failures.

The honest verdict is therefore **INCONCLUSIVE**. `r = 0.90` at 1.72 σ is the one point worth
treating as suspect — it is the largest, and it lies in the direction expected for a box too
small at large separation — but it is not a resolved finding.

**Making it conclusive is infeasible.** Resolving 0.1 kT/nm needs SEM ~0.05, a 10–13× reduction,
i.e. **100–170× more sampling**: roughly 120–200 h against the 1.18 h spent. The threshold was
set without checking it was measurable at any budget this study would spend, which is the
sixth failure class the NaCl session named — *a check whose result does not entail the thing it is
relied on to establish* — here in the form of a criterion finer than its own instrument.

## Does the verdict move? No, and this was tested rather than argued

Worst-case sensitivity: the reference mean force was perturbed by the **full measured difference**
at 0.80 and 0.90 (zero below 0.70, where agreement is 0.0013), re-integrated, and Gate C re-run
against the perturbed target across all 8 seeds and the entire second half of every run.

```
worst-case F distortion, outer region                     0.062 kT
min occupancy / Q*  over 8 seeds x 3 states x 2nd half    0.733   (seed 5002, t = 130.8 ps)
deficit threshold                                         0.50
                                                          -> STILL NO DEFICIT
```

The distortion is small because a mean-force error acts on `F` only through its integral over the
outer ~0.2 nm. **The ABF-sufficient verdict is unchanged**, and it was unchanged under the most
adverse reading of the data rather than under the central one.

## Status of the reference

* **interior (`r ≤ 0.70`): confirmed.** Agreement 0.0013 kT/nm — the two box sizes are
  indistinguishable where the population actually sits and where every state boundary lies
  (the outermost tercile edge is 0.71 nm).
* **outer points (`r = 0.80, 0.90`): provisional.** Point estimates exceed the tolerance;
  neither is resolved; `r = 0.90` is suspect at 1.72 σ. Any future use of `W(r)` or `F(r)` in that
  range — a binding free energy, a comparison to a published tail — should carry this caveat.

## What this gate could and could not do, run late

§1.3 specifies the check *"before the reference is built"*. It was run after the reference, the
screen and the verdict. Run then, it could have truncated the evaluation domain to
`r ≤ 0.70` before anything depended on the outer points. Run now it cannot: the domain is
embedded in an accepted reference and a completed 8-seed screen, and re-truncating retrospectively
would be reshaping a result to fit a check performed afterwards.

So the outcome is a **recorded caveat plus an explicit judgement**, which is the weaker of the two
things a preregistered check can deliver. **Retrospectively performing a preregistered check is not
the same as fulfilling it, and this write-up says which one happened.**

Artifacts: `finite_size.json`, `r0.70.npz`, `r0.80.npz`, `r0.90.npz`, `../finite_size_run.log`.
