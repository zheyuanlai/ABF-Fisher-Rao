# Phase 5 verdict, and what the gate found instead

2026-08-28. Prereg: `docs/MECHANISM_CAMPAIGN_PREREGISTRATION.md`. Verdict logic committed
before the data existed (`scripts/analyze_tau_arms.py`, commit e6c12cb).

## Phase 5: the gate FAILED — and that is the result

16 paired seeds × {A0, A6a, A6b}, T = 160 (16 M steps), on the validated constant-σ²
resolvable-τ benchmark. At T:

| arm | MSE | b'Qb | tr(QΣ) | η_bias |
|---|---:|---:|---:|---:|
| A0 | 2.890e-05 | 2.859e-05 | 3.09e-07 | **0.9893** |
| A6a | 9.297e-06 | 9.016e-06 | 2.81e-07 | 0.9697 |
| A6b | 1.085e-05 | 1.058e-05 | 2.66e-07 | 0.9755 |

**Gate η_bias < 0.1: FAIL.** Per the preregistration no Neyman claim may be made either way,
and none is. (For the record the arms behave sensibly — A6b's tr(QΣ) is the smallest of the
three, 0.945× A6a — but at 2.5 % of the endpoint that cannot decide anything.)

## Why it failed, and why no benchmark can fix it

The decomposition through time on A0 is unambiguous:

| t | b'Qb | tr(QΣ) | η_bias |
|---:|---:|---:|---:|
| 1.6 | 2.674e-05 | 2.41e-05 | 0.526 |
| 6.4 | 2.752e-05 | 1.02e-05 | 0.729 |
| 19.2 | 2.979e-05 | 2.93e-06 | 0.911 |
| 57.6 | 2.995e-05 | 9.40e-07 | 0.970 |
| 160.0 | 2.859e-05 | 3.09e-07 | 0.989 |

Second-half scaling: **d log b'Qb / d log t = −0.018** (a floor, flat to 2 %) and
**d log tr(QΣ) / d log t = −0.717** (decaying). The bias reaches its floor by the *second*
frame; the variance decays past it and keeps going.

**The minimum η_bias over the entire run is 0.121, at t = 0.** The gate is never satisfied at
any time, on a system built specifically to make Γ heterogeneous and resolvable. And the frozen
allocation window opens at 0.2 T = 32, by which time η_bias is already > 0.93 — **the allocator
only ever acts in the bias-dominated regime.**

So the conclusion is structural, not a property of these benchmarks:

> For a fixed-bandwidth kernel ABF estimator, the free-energy endpoint is bias-dominated at
> essentially all times. `Σ_j a_j Γ_j / r_j` is not the leading-order term, and no choice of
> system makes it one.

## The bandwidth ladder: the tradeoff runs the other way, and there is a large win in it

Extending the h ladder (exploratory, **not** preregistered), prescribed-r passive estimator,
same target, T = 40:

| h | b'Qb | tr(QΣ) | total MSE | η_bias |
|---:|---:|---:|---:|---:|
| 0.005 | 4.067e-05 | 2.35e-07 | **4.090e-05** | 0.9943 |
| 0.010 | 4.197e-05 | 2.35e-07 | 4.220e-05 | 0.9944 |
| 0.020 | 5.934e-05 | 2.54e-07 | 5.960e-05 | 0.9957 |
| 0.035 | 1.984e-04 | 4.38e-07 | 1.988e-04 | 0.9978 |
| **0.070** | 2.380e-03 | 2.28e-06 | **2.382e-03** | 0.9990 |
| 0.140 | 3.236e-02 | 2.69e-05 | 3.238e-02 | 0.9992 |

Two things, and I had the first one backwards before measuring it:

1. **Integrated variance *increases* with h** (exponent +2.97), it does not decrease. Smoothing
   lowers pointwise variance but lengthens the correlation range, and the endpoint integrates
   `F'` — so correlated errors accumulate coherently through the cumulative trapezoid. This is
   the same fact as the audit's η_cov = 0.89–0.96. There is therefore **no interior
   bias–variance optimum in h**; both terms fall together as h shrinks.
2. **The production default h = 0.07 is 58× worse in endpoint MSE than h = 0.005** on this
   system. The bias flattens below h ≈ 0.01 onto a residual, h-independent floor
   (4.07e-05) — binning/pseudocount, not kernel — so the win saturates rather than continuing.

η_bias stays ≥ 0.994 across the whole 28× range of h. **Shrinking h does not open a
variance-dominated regime; it just makes the bias floor much lower.**

## What this means for the project

* The Neyman/`Γ` line is not refuted — it is **out of scope for this estimator's endpoint**.
  Its proper claim (Phase 0: tr(QΣ) falls 2.5–4× under `r ∝ sqrt(aΓ̂)`) is measured and true,
  and it governs a term worth ≲ 3 % of the error.
* The largest available win in this codebase is not allocation. It is one estimator constant:
  **58× on endpoint MSE from `h: 0.07 → 0.005`**, against the 1.4–1.7× time-to-accuracy and
  ~35 % error reductions the allocation campaigns were chasing.
* **Caveat, and it is load-bearing**: measured in the *prescribed-r passive* setting (exact
  static bias, N = 4096, EB potential, T = 40). In an adaptive run the bias feeds back into the
  dynamics and small h means fewer counts per bin, where the pseudocount term bites. The
  end-to-end check — adaptive ABF, the h ladder, production walker counts, all four systems —
  is cheap and is the obvious next experiment.

## Status of the preregistered phases

| phase | status |
|---|---|
| 0 K-family re-audit | done — mixed regime, variance obeys Neyman, margins bias-carried in K2/K3 |
| 1 b[r] validation | done — PASS on 6/7 targets, corr 0.98–0.999 |
| 2 h and m factorials | done — h-exponent 1.90, pseudocount confirmed where defined |
| 3 held-out ranking | **not run** — needs a variance model beside b[r]; lower priority now |
| 4 realizability | done — C_force ∝ β⁻² exact, TV flat, WCA bracketed |
| 5 large-τ arm test | done — **gate failed; no Neyman claim; the failure is the result** |
| 6 new allocator | **not started, and should not be until the bandwidth question is settled** |
