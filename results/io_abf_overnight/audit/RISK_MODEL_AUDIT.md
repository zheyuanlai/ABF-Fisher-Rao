# Offline audit of the IO-ABF risk model

2026-08-28. **No simulations were run.** Every number below comes from profiles already on
disk. Code: `scripts/audit_io_abf_risk_model.py`; raw output `risk_model_audit.json`.

## The question

Why does `Σ_j a_j Γ_j / r_j` predict a full-domain **improvement** (0.757 / 0.770 / 0.430 relative
to A0) where the measurement shows **degradation** (1.399 / 1.031 / 1.880)?

The derivation uses only `E[e_A²] ≈ tr(Q diag Σ) = Σ_j a_j Var[f̂_j]`. The exact identity is

    E[e_A²] = b'Q b + tr(Q Σ),      b = E[f̂] − f,   Σ = Cov(f̂).

So the model can be missing the finite-time mean-force **bias** (assumed zero) or the
**off-diagonal covariance** (assumed zero). Both were measured.

## 0. The identity closes, so this is a theory problem and not a scoring problem

`Q` was built from each engine's own reconstruction — cumulative trapezoid, then centring, then
the engine's own evaluation window — and the reconstruction offset `A f'_ref − F_ref` reported
separately rather than folded into the bias.

| | worst relative error of `b'Qb + tr(QΣ)` vs measured | worst reconstruction offset |
|---|---|---|
| all systems, arms, scopes | **3.0e-16** | 9.9e-07 (1.2e-17 on WCA) |

And the audit reproduces the campaign's own inside-mask ratios: **0.646 / 0.887 / 0.920 / 1.468**
against the reported 0.652 / 0.879 / 0.923 / 1.441. Nothing is being decomposed except the number
the campaign actually reported.

## 1. The answer: the model describes under a third of one percent of the error

**`η_bias = 0.930 – 0.999` in every system, every arm, both scopes.**

| system | arm | scope | η_bias | η_cov | R_bias | tr(Q diagΣ) | tr(QΣ) | **modelled share of MSE** |
|---|---|---|---:|---:|---:|---:|---:|---:|
| β=4 | A0 | primary | 0.963 | 0.915 | 2.39e-3 | 7.80e-6 | 9.17e-5 | **0.31 %** |
| β=4 | A6b | primary | 0.930 | 0.895 | 9.63e-4 | 7.66e-6 | 7.29e-5 | 0.74 % |
| β=8 | A0 | primary | 0.991 | 0.945 | 4.22e-2 | 2.19e-5 | 3.98e-4 | **0.05 %** |
| β=8 | A6b | primary | 0.983 | 0.933 | 3.29e-2 | 3.86e-5 | 5.73e-4 | 0.12 % |
| gateway | A0 | primary | 0.992 | 0.887 | 1.11e-4 | 1.08e-7 | 9.48e-7 | **0.10 %** |
| gateway | A6b | primary | 0.985 | 0.897 | 9.32e-5 | 1.49e-7 | 1.44e-6 | 0.16 % |
| WCA | A0 | primary | 0.998 | 0.927 | 5.83e-3 | 8.16e-7 | 1.12e-5 | **0.01 %** |
| WCA | A6b | primary | 0.999 | 0.907 | 1.26e-2 | 1.42e-6 | 1.53e-5 | 0.01 % |
| β=4 | A0 | full | 0.976 | 0.941 | 5.92e-2 | 8.85e-5 | 1.49e-3 | 0.15 % |
| β=8 | A0 | full | 0.996 | 0.903 | 1.52e0 | 5.77e-4 | 5.92e-3 | 0.04 % |
| gateway | A0 | full | 0.983 | 0.912 | 8.47e-4 | 1.29e-6 | 1.47e-5 | 0.15 % |
| WCA | A0 | full | 0.998 | 0.922 | 8.70e-3 | 1.07e-6 | 1.37e-5 | 0.01 % |

**This is Case B, decisively.** The quantity `Σ_j a_j Γ_j / r_j` *is* `tr(Q diag Σ)`, and it
accounts for **0.01 % – 0.74 %** of the measured mean squared error. Optimising it can only ever
move that fraction. The other 99+ % is `b'Qb` — a finite-time mean-force bias the derivation
assumes away.

**Case A is also true but subordinate.** `η_cov = 0.887 – 0.958`: the diagonal approximation
captures only 4–11 % of the variance term. That is a real defect in the derivation, but it is a
correction to a term worth well under 1 % of the endpoint.

## 2. And A6b moved the bias — which is the whole observed effect

| system | scope | R_bias (A0) | R_bias (A6b) | change |
|---|---|---:|---:|---:|
| β=4 | primary | 2.39e-3 | 9.63e-4 | **0.40× (better)** |
| β=8 | primary | 4.22e-2 | 3.29e-2 | **0.78× (better)** |
| gateway | primary | 1.11e-4 | 9.32e-5 | **0.84× (better)** |
| WCA | primary | 5.83e-3 | 1.26e-2 | **2.15× (worse)** |
| β=4 | full | 5.92e-2 | 1.17e-1 | 1.97× (worse) |
| β=8 | full | 1.52e0 | 1.61e0 | 1.06× (worse) |
| gateway | full | 8.47e-4 | 2.92e-3 | 3.45× (worse) |
| WCA | full | 8.70e-3 | 2.59e-2 | 2.97× (worse) |

So **the inside-mask gain and the outside-mask damage are the same phenomenon: the reallocation
moved the finite-time bias.** It lowered it where the allocation concentrated effort and raised it
where the allocation withdrew effort. The variance term barely moved and was never large enough to
produce effects of the observed size.

**Consequence: the campaign's A6b results are neither evidence for nor against Neyman allocation.**
The channel Neyman optimises contributed under 1 % of the error in every case. Whatever A6b did, it
did through a channel the theory does not model.

## 3. Where the bias lives, and how it moves

**Outside the mask it is bias too**, not starvation noise: `η_bias = 0.951 – 0.999` there, and
`R_bias` rises 1.32× / 1.13× / 2.00× / (0.25× on WCA) from A0 to A6b. So the full-domain guard was
catching a real effect, but not the one the risk model would attribute it to.

**It appears immediately on activation, then grows** (WCA, the only system storing profile time
series):

| t/T | R_bias A0 | R_bias A6b | ratio |
|---:|---:|---:|---:|
| 0.3 | 7.74e-3 | 1.13e-2 | 1.46 |
| 0.5 | 6.64e-3 | 1.21e-2 | 1.82 |
| 0.7 | 6.12e-3 | 1.22e-2 | 2.00 |
| 1.0 | 5.83e-3 | 1.26e-2 | 2.15 |

`η_bias ≥ 0.992` at every one. The allocation window opens at 0.2T and the bias gap is already
1.46× by 0.3T — this is not slow starvation accumulating, the reallocation relocates the bias
essentially at once.

## 4. Supporting mechanism: the estimator's correlation length is ~2 allocation cells

| system | corr length (1/e) | ABF bandwidth h | cell width | corr len / h | corr len / cell |
|---|---:|---:|---:|---:|---:|
| β=4 | 0.2100 | 0.070 | 0.1125 | 3.0 | **1.87** |
| β=8 | 0.2100 | 0.070 | 0.1125 | 3.0 | **1.87** |
| gateway | 0.2100 | 0.070 | 0.1125 | 3.0 | **1.87** |
| WCA | 0.0817 | 0.025 | 0.0438 | 3.3 | **1.87** |

Identical to three significant figures across two engine families and a 2.6× change in bandwidth.
The mean-force estimate is smeared over roughly **two allocation cells**, which is the direct
support for the kernel-smoothing explanation of `η_cov` — and it also means the allocator is
resolving `r` more finely than the estimator can resolve `f̂`.

## 5. The 1/r law is approximate at best, and undefined where an arm evacuates

Predicting `Σ_A6b` from `K̂` estimated on **A0 alone** (no fitting to A6b), restricted to cells both
arms actually occupy:

| system | scope | predicted / measured |
|---|---|---:|
| β=4 | primary | 3.00 |
| β=8 | primary | 0.78 |
| β=8 | full | 0.68 |
| gateway | primary | 1.31 |
| gateway | full | 2.73 |
| WCA | primary | 0.49 |
| WCA | full | 0.63 |

Order of magnitude only — factors of 0.49 to 3.0 in both directions. **Case C is partly true**, but
like Case A it is a correction to a term worth under 1 %.

One structural finding from this test: on β=4 the unrestricted extrapolation diverged, because
**A6b drove at least one cell to exactly zero realised occupancy** (A0's minimum was 1.2e-3).
`FLOOR_FRACTION` floors the *target* `r*`; nothing floors the realised occupancy, and `D(r)^{-1/2}`
is undefined when a cell empties.

## 6. What this licenses and forbids

* **Do not** repair the leverage operator. `a_j → 0` near a boundary is mathematically correct for
  the integrated, centred endpoint, and the audit gives no evidence against it.
* **Do not** implement the full-domain KKT constraint. It constrains `Σ a_full Γ/r`, which is a
  0.15 %-of-MSE quantity, and λ = 0 on all three damaged systems anyway.
* **Do not** build a covariance-aware `R(r) = tr[Q D(r)^{-1/2} K D(r)^{-1/2}]` yet. It is the right
  generalisation of the variance term, and the variance term is not the problem.
* **The realizability finding stands untouched** — it is about whether a target can be imposed at
  all, not about which target is optimal, and the audit says nothing against it.
* **The open question is now `b(r)`**: a model for how the finite-time mean-force bias depends on
  the occupancy. Without it there is nothing to optimise, because
  `MSE(r) = b(r)'Q b(r) + tr(QΣ(r))` is ≥ 99 % first term.

## 7. A data gap this audit hit

EB and gateway store only the **final** `Fp_hat`, so items 4 and 6 could only be time-resolved on
WCA. This is the same defect the WCA Stage-A audit recorded ("Case IX raw retains only already-
scored scalars, so no rescore is possible"). The fix is one line per engine: store `Fp_hat` at
every save, as `wca_abffr_core` already does.
