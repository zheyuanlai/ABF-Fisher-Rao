# Methane reference accepted, and Gate 0 PASSES

Constrained-TI reference on the batched engine, 29 `r`-points × 16 replicas (8 wet / 8 dry) ×
3 independent builds, 50 ps equilibration + up to 200 ps production under Amendment 12.2's
checkpoint retirement rule. **13.78 h** on one H200.

**No mFR arm exists and no mFR result has been seen.** Gate 0 is evaluated first, per
Amendment 10, and this document reports it before any screen has run.

---

## 1. Reference acceptance — ACCEPTED

| | |
|---|---|
| independent builds | 3 |
| max pairwise `L2` between builds | **0.092 kJ/mol** |
| consensus `F` span | 3.11 kT |
| `ratio = max L2 / (0.10 × span)` | **0.1198** |
| acceptance threshold (§4.5) | `≤ 0.5` |

Accepted with **4× margin** on the criterion.

## 2. Gate 0 — PASSES

The controlled experiment Amendment 9 identifies as *the* instrument: independently prepared
solvent families held at the same `r`, either agreeing or not.

| | cross-family `|wet − dry|` / mean`|F'|` |
|---|---|
| **methane, global** | **0.048** |
| **methane, core 0.42–0.70 nm** | **0.081** |
| methane, worst single point (`r = 0.90`) | 0.158 |
| WCA dimer | 0.040 — PASS |
| gateway | 0.036 global / 0.189 constriction — PASS (marginal) |
| deca-alanine | 0.61 — FAIL |
| R15 β=2 | 0.564 / 0.593 — FAIL |

**No numerical threshold is set** (Amendment 9 refused to fix one after seeing R15's number, and
that refusal binds here). Argued against the ladder: methane sits at the WCA level, **12.7×
below deca**, and its worst single point is below the gateway's constriction value, which the
campaign already accepted as a marginal pass.

**The decisive evidence is dynamic, not just the final number.** The wet/dry spread *collapsed*
as sampling grew, which is the direct signature of a conditional ensemble that mixes:

| `r` (nm) | 50 ps | 200 ps |
|---|---|---|
| 0.56 | 7.32 | **2.42** |
| 0.58 | 6.24 | **2.13** |
| 0.60 | 5.18 | **2.04** |
| 0.66 | 5.70 | **2.64** |

A conditional-equilibration failure does the opposite — deca's 61 % error persisted at up to
2 × 10⁶ effective counts per bin.

> **Methane is not conditional-equilibration-limited.** The solvent cage equilibrates at fixed
> methane separation within the budget, including through the desolvation region.

## 3. The PMF reproduces the literature structure — at sub-kT depth

`W_ref` shows all three classic hydrophobic-association features, at positions that match the
methane literature:

| feature | our `r` | literature | our `W` |
|---|---|---|---|
| contact minimum | **0.38 nm** (3.8 Å) | ~3.9 Å | **−1.11 kT** |
| desolvation barrier | **0.56 nm** (5.6 Å) | ~5.5–6.0 Å | **+0.40 kT** |
| solvent-separated minimum | **0.72 nm** (7.2 Å) | ~7.0–7.5 Å | **−0.27 kT** |

`n_gap` rises monotonically 0.20 → 2.75 across the domain, so the wetting transition is resolved
and the descriptor behaves as designed.

**Every feature is sub-kT.** Measured from the higher of the two minima, the desolvation barrier
is `0.40 − (−0.27) = 0.67 kT`, far below Amendment 3's 2 kT merge threshold, so the basin rule
merges everything into a **single basin** and the preregistered tercile fallback fires.

> **This is stated before the screen and is not a screen verdict.** A sub-kT barrier means there
> is no metastability in `xi` for ABF to be slow about, which is the shape §9 of the
> preregistration anticipated when it called methane a likely null. But **Gates B and C are
> decided by ABF-only screen data, not by reading the reference**, and the screen has not run.
> Recording the expectation here, in advance, is what stops it being presented afterwards as a
> prediction that was confirmed.

## 4. Outstanding — the finite-size gate has NOT been run

SPEC §1.3 preregisters a gate that is still open: evaluate `⟨f_loc⟩` at
`r ∈ {0.70, 0.80, 0.90} nm` in a **1024-water box** and truncate the evaluation domain if any
point moves by more than 0.1 kT/nm.

It matters here specifically. At `L = 2.4908 nm` the top of the domain sits at 72 % of the
minimum-image half-box, and `r = 0.90` is both the worst Gate 0 point (0.158) and the point with
the largest wet/dry spread (4.26 kJ/mol/nm). **The reference is therefore accepted for the
interior and provisional in its outermost points**, and this is recorded as open rather than
quietly assumed away.

## 5. Artifacts

```
results/methane/box/        NPT box: L = 2.490832 ± 0.000497 nm, rho = 0.9946 g/cm^3
results/methane/baths/      wet/dry baths + 174 minimised per-r starts
results/methane/ti_torch/   raw TI cells and per-checkpoint states
results/methane/ref/        reference.json / reference.npz, this file
```
