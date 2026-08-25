# v3 pilot result: consistency did not make Fisher–Rao self-limiting

Frozen protocol: `docs/V3_PREREGISTRATION.md` (v3.1) with Amendments 1–6.
21 arms × 8 matched seeds, executed 2026-08-25. Thresholds frozen beforehand from
plain ABF alone (`results/v3/V3_THRESHOLDS.json`, scope R₁₂).

**Verdict: 0/11 candidates mechanism-positive, 0/11 advancement-positive.**
Both tracks fail their frozen gates.

## The decomposition, reported first (Amendment 6c)

At ε_F,2, median over 8 matched seeds:

| arm | R_shape (bias) | R_FR (FR increment) | R_total |
|---|---|---|---|
| C_capped8 + FT ρ=0.70 | **1.906** | **0.411** | 0.830 |
| C_capped8 + FT ρ=0.85 | **1.906** | **0.411** | 0.830 |
| C_capped12 + FT ρ=0.70 | **1.617** | **0.583** | 1.000 |
| C_capped12 + FT ρ=0.85 | **1.617** | **0.583** | 1.000 |
| C_tempered8 + FT ρ=0.85 | 1.327 | 0.599 | 0.828 |
| C_flat + FT ρ=0.85 | 1.000 | 0.828 | 0.828 |

**These three medians do not factorize, and must not be presented as if they
do.** The identity R_total = R_shape × R_FR is exact *per seed* — verified on all
8 seeds — but median(XY) ≠ median(X)·median(Y): for capped-12,
1.617 × 0.583 = 0.943 against a reported median R_total of 1.000. Amendment 6c
is a per-seed decomposition plus a reporting order, not an algebraic identity
between the three summary numbers. The registered statistic is the median and it
stays. As a purely descriptive companion, geometric means of the paired ratios do
factorize exactly: 2.086 × 0.548 = 1.144.

The one-line summary the campaign was built to be able to make:

> **Capping the bias is a large, real gain. Adding Fisher–Rao reallocation
> destroys it.** Every consistent-target arm has R_FR < 1; the best combined
> method merely ties plain ABF.

The no-FR controls reach ε_F,2 on **8/8 seeds** where plain ABF manages 5/8, and
end with *lower* error than ABF on scope R₁₂ (0.585–0.826 of it). Had we reported
R_total alone we would have said "no effect"; had we reported the controls alone
we would have said "1.6–1.9× speedup". Neither is the finding.

## P3 is falsified: the dose does not self-limit

The central prediction was that a bias-consistent target would make FR turn
itself off as A_t → F. Registered threshold: dose decay ≥ 5× from the first
quarter of the FR window to the last.

| arm | dose decay | ancestral ESS/K | seeds passing ESS | replacements |
|---|---|---|---|---|
| C_capped8 + FT | 1.10–1.11 | 0.126–0.132 | **0/8** | ~1150 |
| C_capped12 + FT | 1.00–1.02 | 0.138–0.141 | **0/8** | ~1200 |
| C_tempered8 + FT | 1.05 | 0.134 | **0/8** | 1112 |
| C_flat + FT | 1.05 | 0.118 | **0/8** | 1443 |

Observed decay is ~1.0 against a required 5.0: the dose is **flat across the
whole window**. Ancestral ESS lands at 0.12–0.16 K against a required 0.5, on
0/8 seeds for every arm. Algebraic consistency with the current estimate did not
buy practical self-limitation at finite K and T.

Per Amendment 4b this was never expected to reach exactly zero — carrier error,
finite-time non-equilibrium and finite-K/KDE fluctuation all contribute — but a
decay factor of 1.0 is not a noise floor, it is no decay at all.

## The decisive diagnostic: the damage is target error, not Fisher–Rao

`C_capped12_oracle_target` applies the candidate's own estimated bias and
oracle-izes **only** the target to the true frozen-bias marginal
q ∝ exp(−β(F_ref + B_t)) (Amendment 2). Everything else is identical to the
deployable arm:

| quantity | deployable capped12+FT | oracle-target | plain ABF |
|---|---|---|---|
| S at ε_F,2 | 1.000 | **2.389** (best of all 20 arms) | 1 |
| final e_F on R₁₂ (vs ABF) | 2.086 | **0.490** | 1 |
| final barrier e_F′ (vs ABF) | 1.527 | **0.518** | 1 |
| seeds censored at ε_F,2 | 6/8 | **0/8** | 3/8 |
| dose decay | 1.02 | 1.01 | — |
| ancestral ESS/K | 0.141 | 0.152 | 1 |
| replacements | 1201 | 1123 | 0 |

What is held fixed is the **operator, the governor parameter ρ, the opportunity
schedule, and the applied-bias construction**; only the target construction
changes. It is *not* the same dose — changing q changes the FT weights, and the
realized replacement counts differ (1201 vs 1123). Claiming matched dose would
require verifying the θ_t and ESS trajectories, which this campaign did not
persist (see the known gap below).

The correct conclusion is therefore:

> **Wrong-target FR reallocation causes the accuracy damage**, and target
> construction is load-bearing for it.

Not "target error rather than reallocation": reallocation is precisely the
mechanism by which an erroneous target corrupts A_t, and with a correct target
the same kind of reallocation *improves* accuracy substantially. The proposed
feedback loop — poor F̂ → poor target → FR movement → worse F̂ — is consistent
with these endpoints but is not yet demonstrated causally; the diagnostic replay
below is what would show it in time order.

Consistency was defined against the *current estimate*, and Amendment 2 already
established that this equals the physical stationary marginal only as A_t → F.

Note also that this finite-time target error is **independent of g**: with
B_t = g(A_t) − A_t and q_t ∝ exp(−βg(A_t)),
log(p*_{B_t}/q_t) = −β[F − A_t] + const. Moving between capped, tempered and
physical cannot repair it. And because the target exponentiates the carrier,
q_t/q_t^oracle ∝ exp(−β(A_t − F)): an estimate accurate enough to serve as a
force bias can still be far too inaccurate to serve as an exponentiated
population target. That is plausibly the deepest lesson of this pilot.

This is the same signature as the entropy-dominant bottleneck study's
target-limited result: the oracle target gains where the deployable one does not.

## Track P behaves exactly as registered (P2)

| arm | S at ε_F,2 | final e_F on R₁₂ | ancestral ESS/K | dose decay |
|---|---|---|---|---|
| P_BD p_max=0.02 | 0.992 | 1.122 | 0.393 | 0.99 |
| P_BD p_max=0.10 | 1.000 | 1.757 | 0.119 | 1.02 |
| P_FT ρ=0.70 | 2.059 | **4.215** | 0.033 | 1.00 |
| P_FT ρ=0.85 | 2.068 | **3.140** | 0.055 | 1.00 |

P_FT reaches the accuracy threshold *faster* than ABF and ends at 3–4× ABF's
error, with ancestral ESS at 3–6 % — the transient-gain-then-repayment pattern
v2 showed, now unmistakable. Time-to-accuracy alone would have called this a
2× win; the mandatory final-frame companion is what exposes it, which is the
whole reason v3.1 demoted AUC and required non-inferiority.

## Registered predictions, scored

| prediction | verdict |
|---|---|
| P2 Track P: transient gain, endpoint damage, no dose decay, genealogy failure | **confirmed** |
| P3 Track C: dose decays ≥5×, genealogy passes | **falsified** (decay ≈ 1.0, ESS 0/8) |
| P3 Track C: no-FR control captures ≥50 % of the gain | **confirmed, emphatically** — it captures *more* than all of it |
| P4 physical/tempered underperform capped on barrier F′ | **confirmed** (capped 1.46–1.54, tempered 2.97, physical 7.26) |
| P5 exact ≤ hold-out ≤ oracle-refresh on final F′ | **partial** — hold-out 1.296 and refresh 1.426 both beat exact 1.527, but refresh is not better than hold-out |
| P7 consistent-physical endpoint is the worst arm | **confirmed** (final e_F 9.205, worst of 20) |
| P8 Track C effects are small | **wrong in sign** — the effect is a large negative, not a small positive |
| P9 governor self-throttles under Track P's inconsistent target | **refuted** — P_FT made 2706–4486 replacements against P_BD's 202–851; FT churned *more*, not less |

## A defect in the preregistration itself

Advancement condition 7 (full-domain final e_F ratio ≤ 1.25) is **structurally
unsatisfiable for the capped family**, and this was not discovered until the data
existed. The capped bias deliberately stops flattening beyond c_cut, so its
free-energy estimate in the far tails is poor by design: the *no-FR controls*
already sit at 15.3–18.9× on that ratio. The cap was written to charge tail
sacrifice and was set without anticipating its magnitude.

It is **not** amended after the fact, and it does not change the verdict: every
candidate fails on 2–4 *independent* grounds (dose decay, genealogy, final
non-inferiority on R₁₂, barrier) with condition 7 removed entirely. Recorded so
that a future protocol either sets that bound from a pilot control or scopes it
to the family it can meaningfully constrain.

## Two independent failure modes, not one

The oracle arm separates them cleanly:

1. **Accuracy failure — target feedback.** Driven by F − A_t, independent of the
   family member g, and removed entirely by an oracle target.
2. **Finite-particle selection failure — genealogy.** *Not* removed by the oracle
   target: dose decay stays ≈ 1.0 and ancestral ESS/K ≈ 0.15 even when accuracy
   is excellent.

A perfect deployable target estimator would therefore be expected to yield
excellent F and a still-failed genealogy gate. Any successful successor has to
solve both.

## What this licenses

Nothing downstream. A mechanism-positive would have licensed the β = 8 rung;
none was obtained. The frozen rule stands: no new ρ search, no retuned c_cut, no
rescue of the schedule.

The scientifically live question the data does raise — and which is *not* a
licensed conclusion from this pilot — is whether the damage can be removed by
attacking target-estimation error rather than the operator, since the
oracle-target arm shows the operator itself is not what breaks accuracy. Any such
study needs its own preregistration.
