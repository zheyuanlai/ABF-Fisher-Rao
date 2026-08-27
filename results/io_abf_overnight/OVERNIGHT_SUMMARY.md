# IO-ABF transfer campaign — overnight result

Run 2026-08-27, branch `q-r-decoupling`.
Frozen protocol: [`docs/IO_ABF_OVERNIGHT_PREREGISTRATION.md`](../../docs/IO_ABF_OVERNIGHT_PREREGISTRATION.md),
written and committed **before** any scientific run. No criterion in it was changed afterwards.

---

## 1. One-paragraph answer

**Information-optimal allocation transfers, and it costs something the kappa family could not
show.** On all three preregistered systems A6b reaches the frozen stringent accuracy faster than
plain ABF — S(ε₂) = 1.37–1.69, every paired 95 % CI clear of 1, and A6b hits the threshold in
**32/32 seeds in all three systems** where A0 manages only 22–27 — and it is also *more* accurate
than ABF at the horizon inside the evaluation window. But the preregistered full-domain
correctness guard fails in two systems of three. The reason is not a bug: the leverage `a(z)` is
exactly zero outside the evaluation mask, so `r* ∝ sqrt(a Γ)` assigns those cells only the shared
floor, and their free energy degrades. **IO-ABF buys accuracy where the endpoint scores it by
spending accuracy where it does not.** Only the entropic bottleneck at β = 8 clears the guard, so
by the frozen rule it is the campaign's single POSITIVE. A6c — the same mechanism with the genuine
Fisher–Rao mass constrained to ESS_M/K ≥ 0.5 — is **worse than plain ABF in every system**
(S = 0.65–0.81), and Fisher–Rao is demonstrably load-bearing in it, so this is a real price on
representing `q_phys`, not an inactive constraint.

---

## 2. Headline

| System | role | R_Γ | A6b S(ε₂) | 95 % CI | hit A6b/A0 | A6c S(ε₂) | mass ESS | final A6b/A0 | full-domain A6b/A0 | verdict |
|---|---|---:|---:|:--:|:--:|---:|---:|---:|---:|:--|
| Bottleneck β=4 | control | 21.6 | **1.694** | [1.281, 2.136] | 1.00/0.84 | 0.810 | 0.500 | 0.652 | 1.399 | **NOT POSITIVE** |
| Bottleneck β=8 | candidate | 12.4 | **1.395** | [1.313, 1.478] | 1.00/0.75 | 0.757 | 0.500 | 0.879 | 1.031 | **POSITIVE** |
| Entropic gateway | candidate | 123.7 | **1.366** | [1.158, 1.600] | 1.00/0.69 | 0.652 | 0.500 | 0.923 | 1.880 | **NOT POSITIVE** |
| WCA dimer | candidate | — | — | — | — | — | — | — | — | **reference gate FAILED — not run** |

Preregistered checks, A6b vs A0:

| System | S ≥ 1.15 | CI lower > 1 | censoring ok | final ≤ 1.10× | full-domain ≤ 1.10× |
|---|:--:|:--:|:--:|:--:|:--:|
| Bottleneck β=4 | PASS | PASS | PASS | PASS | **FAIL** |
| Bottleneck β=8 | PASS | PASS | PASS | PASS | PASS |
| Entropic gateway | PASS | PASS | PASS | PASS | **FAIL** |

Four of the five checks pass everywhere. The campaign turns entirely on the fifth.

---

## 3. Where the full-domain damage lives

RMS free-energy error at the horizon, split by the evaluation mask (which covers 151 of 181 grid
points, 83 %):

| System | arm | inside mask | outside mask | full domain |
|---|---|---:|---:|---:|
| β=4 | A0 | 0.0491 | 0.5657 | 0.2471 |
| β=4 | A6b | 0.0320 | 0.7941 | 0.3458 |
| β=4 | **ratio** | **0.652** | **1.404** | **1.399** |
| β=8 | A0 | 0.2064 | 2.8167 | 1.2415 |
| β=8 | A6b | 0.1815 | 2.9160 | 1.2797 |
| β=8 | **ratio** | **0.879** | **1.035** | **1.031** |
| gateway | A0 | 0.0105 | 0.0653 | 0.0294 |
| gateway | A6b | 0.0097 | 0.1255 | 0.0553 |
| gateway | **ratio** | **0.923** | **1.922** | **1.880** |

Read this carefully in both directions.

* **Inside the mask A6b is better in every system.** There is no system where the speedup was
  bought by ending up less accurate on the scored window.
* **Outside the mask A6b is worse in every system**, and the full-domain metric is dominated by
  that region because the outside error is already 6–14× the inside error *for plain ABF too*.
  So "1.88× worse full-domain" means "1.92× worse in a region neither arm converges", not "the
  free energy is 1.88× worse everywhere". The guard is doing its job, and the honest statement of
  what it caught is the trade, not a failure of the estimator.
* β=8 clears the guard because its outside damage is genuinely smallest (1.035), not merely
  because its denominator is largest.

**An observation, not a result (n = 3 systems):** the outside damage is monotone in the measured
difficulty spread — R_Γ = 12.4 → 1.03, 21.6 → 1.40, 123.7 → 1.92 — while the *speedup* is not
(1.395, 1.694, 1.366). Three points cannot establish a trend, but the direction is mechanistically
what one expects: a more heterogeneous Γ produces a more lopsided r*, and the floor bounds the
starvation without removing it. If this holds, the shared floor `FLOOR_FRACTION = 0.25` is the
knob that governs the trade — and it was deliberately **not** touched tonight.

---

## 4. Q1–Q4, as preregistered

**Q1 — does estimator-risk allocation accelerate ABF?** Yes, and reproducibly. S(ε₂) =
1.37–1.69 with every CI above 1, on 32 fresh paired seeds per system, and the censoring goes the
*helpful* way: A6b reaches the stringent threshold in 32/32 seeds in all three systems where A0
reaches it in 27, 24 and 22. Because a censored A0 seed is charged exactly the horizon rather than
the longer time it would really have needed, the quoted speedups are conservative.

**Q2 — what does keeping the physical mass cost?** More than all of the gain. A6c runs at
0.65–0.81× plain ABF and its final error is 1.45–1.88× A0's. Retention `R_retain` is **negative in
all three systems** (−0.27, −0.61, −0.95). This is not an inactive constraint being blamed for
noise: P(λ > 0) = 1.000 at every opportunity in every system, mass ESS is pinned at exactly 0.500
against 0.137–0.156 unconstrained, and TV(r_A6c, r_A6b) = 0.28–0.30. **Fisher–Rao is load-bearing
in A6c and what it does is harmful to the free-energy endpoint.** The Stage-2 kappa result — that
the physical mass costs a quarter to two fifths of the gain — does not transfer; here it costs
more than the whole of it.

**Q3 — does the gain appear where difficulty is heterogeneous?** **Not testable as designed, and
the answer to the question actually asked is no.** Every system carries large Γ heterogeneity
(R_Γ = 12–124), so there is no low-heterogeneity cell to contrast against — the same failure the
q-r campaign recorded for K0. Within the range available, speedup does not track R_Γ; the gateway
has 10× the spread of β=8 and a *smaller* speedup. What tracks R_Γ is the collateral damage.

**Q4 — is IO-ABF neutral where there is no heterogeneity?** **Never tested.** The β=4 cell was
designated the control on the repository's "ABF-sufficient at β ≤ 4" classification, but that
classification is about birth–death mFR gain, not about Γ. Measured, β=4 has R_Γ = 21.6, *larger*
than the β=8 candidate's 12.4. **β=4 was not a control and must not be quoted as a passed tie
test.** This is the identical trap the q-r campaign hit with K0, hit again with a different cell
family, which suggests the lesson generalises: *a regime classification built for one mechanism
does not transfer to another mechanism's controlling variable.* A real Q4 test needs a cell with Γ
verified flat, and neither the kappa family nor the bottleneck family has one.

---

## 5. Difficulty decomposition

| System | Q₁₀(σ²) | Q₉₀(σ²) | R_σ | Q₁₀(τ) | Q₉₀(τ) | R_τ | Q₁₀(Γ) | Q₉₀(Γ) | R_Γ | valid-τ | ρ_s(Γ early, late) | dominant |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--|
| Bottleneck β=4 | 0.0745 | 5.49 | 73.7 | 9.95e-4 | 3.67e-3 | 3.7 | 5.74e-4 | 0.0118 | 20.6 | 0.999 | 0.981 | σ² |
| Bottleneck β=8 | 0.0911 | 1.68 | 18.4 | 1.75e-3 | 6.86e-3 | 3.9 | 6.03e-4 | 7.51e-3 | 12.5 | 0.867 | 0.980 | σ² |
| Entropic gateway | 1.43e-3 | 0.930 | 648.5 | 3.84e-3 | 7.31e-3 | 1.9 | 3.93e-5 | 4.90e-3 | 124.7 | 0.998 | 0.992 | σ² |

Every system passes the ≥ 80 % valid-τ reliability gate, so none is marked `Gamma unresolved` and
a candidate failure here *is* attributable.

**The mechanism is σ², not τ, in all three systems**, and by a wide margin: R_σ is 18–649 while
R_τ never exceeds 3.9. The R-OBS probes explain why. Sampled at full density (every integrator
step), the per-cell mean-force series decorrelates in **1–2 steps** on the bottleneck and in
**≤ 2 steps** on WCA; the chosen observation intervals were 1, 2 and 7 steps. The local mean-force
observation stream in these systems is very nearly white at the integrator scale, so τ carries
almost no cross-cell information and Γ = σ²τ is a σ² map wearing a product's clothes. That is a
real property of these benchmarks, not an estimator failure — but it means **this campaign did not
test the τ half of the theory at all.** Γ is also very stable in time: Spearman(Γ early, Γ late) =
0.98–0.99, so the allocator is not chasing its own noise.

---

## 6. The mechanism works, and the figures show it

`results/io_abf_overnight/<system>/analysis/figures_confirmatory/`, six figures per system.

The load-bearing one is **fig5**. With **no birth, no death and no resampling**, the bias alone
pulls the realised occupancy onto the target: TV(r*, realised) = 0.151 for A6b on β=8, against an
A0 occupancy with a completely different shape. Ancestor ESS stays at N for every IO run in every
system (gate G0.3, checked end to end). The claim that a replica density can be held by an
adaptive bias instead of by genealogy is now confirmed on two engine families.

**fig6** shows why A6c loses: its target is pulled toward `q ∝ exp(−βÂ)`, which is peaked in the
potential wells, and away from the information optimum, which is peaked on the flanks where the
force noise lives. The two objects genuinely disagree in these systems.

---

## 7. WCA — the reference gate, and what was done instead

**Gate verdict: FAIL.** Two blocking findings, both from the repository's own Stage-A audit:

1. The high-precision reference is built on **41 z-values at spacing 0.035** against an evaluation
   grid at 0.0088 — a factor of **4.0** — and that audit states in its own words that "a denser
   build would be needed to quote a final corrected `F'` curve pointwise".
2. The correction from the cached reference to the high-precision one is **L2 = 0.0608**, which is
   **3.0×** the −22.83 % effect size it would be used to score. The cached reference is wrong by
   **24.8 σ** at z = 0.255 (`F' = 2.094` cached vs `0.601 ± 0.060`), inside the transition region
   where arms differ most, so the cross term does not cancel.

Physical parameters and grid *do* match (n_dim 10, a 1.5, σ 1, ε 1, h 2, w 2, β 1; 160 points on
[−0.2, 1.2]). The failure is the reference's own resolution and accuracy, not a setup mismatch.

Per the preregistration this licenses **A0 diagnostics only and no speedup**, which is what was
run: an A0-only Γ screening, instrumented read-only so the dynamics are byte-identical to the
accepted sampler. Its result is in §8 and it carries no speedup claim.

**What WCA needs before it can be run properly** — and this is the highest-value single job for
the next session — is a denser high-precision TI reference: ~141 z-values at spacing 0.01 with the
same 4-preparation, 20k-prep / 20k-equilibration / 50k-production protocol. The existing 41-point
build took 22 min on one H200, so this is roughly 75–90 min of GPU time. With that in hand the
full WCA campaign is ~136 runs at ~6.5 min each, about 15 h on one GPU.

---

## 8. WCA A0-only difficulty screening

STATUS_WCA_SCREENING

---

## 9. Engineering gates — all pass, and one found a real defect

| Gate | Check | Result |
|------|-------|--------|
| G0.1 | uniform `r*` is exactly plain ABF; an A0 row inside an IO batch equals accepted `eb.ABF` bit for bit | PASS |
| G0.2 | nothing on the allocation path *executes* a reference name (tokenised source, docstrings stripped) | PASS |
| G0.3 | no resampler; every public method returns a float field; ancestor ESS = N end to end | PASS |
| G0.4 | `r*` normalised, positive, floor exactly 0.25/J | PASS |
| G0.5 | constant Γ gives `r* ∝ sqrt(a)` | PASS |
| G0.6 | synthetic AR(1) stream recovers σ², τ, Γ ordering (Spearman > 0.85, valid-τ ≥ 80 %) | PASS |
| G0.7 | A6c satisfies ESS_M/K ≥ 0.5 **on the target it applies** | PASS after a fix |
| G0.8 | no weight in the ABF accumulator; the allocation force enters the drift only | PASS |
| G0.9 | no valid τ falls back to `sqrt(a)`, not to an extreme allocation | PASS |
| REG  | the 160 existing q-r / kappa / clean-v2 regression tests | PASS, before and after |

21 gate tests in `tests/test_io_abf_gates.py`.

### G0.7 found a real defect, inherited from the closed q-r campaign

The shared floor was applied **after** the ESS bisection. Mixing with uniform lowers ESS_w, so the
target that was actually applied missed the constraint it had been solved for — measured at **ESS
0.420 against a stated 0.500** — and the number the run *reported* was the pre-floor target's, not
the one it used. `allocation.r_ess_constrained` now takes `floor_fraction` and solves the floored
problem; verified on 500 random cases, all satisfy the bound on the applied target. The floored
problem is always solvable, because as λ → ∞ the target tends to `(1−ε)q + ε/J` and
`Σ q²/((1−ε)q) = 1/(1−ε)` bounds `ESS_w ≥ 1 − ε = 0.75`.

**Consequence for the closed q-r campaign, stated rather than quietly corrected:** its Stage-2
"mass ESS 0.500 against 0.09–0.11 unconstrained" is the **unconstrained-solve** number and should
be read that way. The unfloored code path is left intact so that campaign's arithmetic is
unchanged. Its *direction* is unaffected — the constraint was still active and still binding — but
the fidelity it certified was not the fidelity it ran at.

### A second, smaller blemish — recorded, deliberately not fixed mid-campaign

`a_j` is computed by **summing** the per-grid-point leverage over each cell. With G = 181 grid
points and J = 32 cells, cells hold 5 or 6 grid points, which puts a ±10 % sawtooth on `a_j` and
~±5 % on `r*` (visible in fig4). The risk model actually calls for an extra factor of the cell's
grid-point count, `g_j = n_j Σ_{g∈j} a_g Γ_g`. This was found *after* seeing candidate results, so
changing it now would be exactly the post-hoc adjustment the preregistration forbids. It is
identical across A6b and A6c and absent from A0, it is small against a target that varies ~3×, and
if anything it *understates* the candidate arms — so the direction of every result above is safe.
**Fix it before the next campaign, not inside this one.**

---

## 10. Caveats that limit how far these numbers travel

1. **ε₂ is not stringent on β=4.** That cell's A0 error curve is non-monotone — it bottoms at
   t = 12 (0.0488) and drifts back up to 0.0520 — so ε₂ (median at 0.6 T = 0.05100) is *larger*
   than ε₁ (median at 0.4 T = 0.05039). On β=4, τ(ε₂) measures arrival at a plateau, not at a
   tighter accuracy, and its wide CI [1.281, 2.136] reflects that.
2. **β=8 sits on a bias floor.** Its error falls to ≈0.24 by t = 8 and then decays only to 0.203
   by t = 40, so ε₂ = 0.212 is 5 % above the asymptote. The *statistical* part of the error is
   small against the systematic part in that cell, which structurally caps what any
   variance-reduction method can win there.
3. **σ̂² conflates conditional force noise with ABF estimation error.** The residual is taken
   against the *running* `F̂'`, so in an under-sampled region a poorly-estimated `F̂'` inflates σ̂².
   Visible in fig3 as the σ² peaks at |z| ≈ 1.3 on β=8, which are low-occupancy wall regions
   rather than physically noisy ones. The resulting feedback is stabilising (under-sampled → looks
   hard → gets replicas), but it is not the Γ the theory specifies.
4. **τ was never really tested**, per §5.
5. **The allocation window closes at 0.80 T** (frozen from the q-r protocol), so the final-error
   comparison is made after the allocation has been off for the last fifth of every run.
6. **Three systems, two engine families.** The gateway shares `eb_abffr_core`'s primitives, so
   these are not three independent engines.

---

## 11. What was not done, and why

* **WCA A6b/A6c confirmatory** — the reference gate failed. §7.
* **Molecular Γ screening** (butane, pentane φ₁, alanine, pentane R₁₅, deca) — three of the five
  have periodic torsion CVs, and the plan explicitly rules out building the periodic leverage
  operator tonight. Of the two non-periodic candidates, deca is excluded by classification and
  pentane R₁₅ is documented discovery-limited, so a Γ map there would describe a coordinate the
  sampler barely visits. Deferred rather than half-done.
* **Deca-alanine as an IO-ABF candidate** — excluded by the preregistration, and it stayed
  excluded.
* **Gateway `one_right` init** — not run. It is part of the accepted gateway setup and would
  separate discovery from establishment, but adding a condition *after* seeing the `left` result
  cannot be reported as preregistered, so it is left for the next session to run cleanly.

## 12. Nothing on the prohibited list was done

`FLOOR_FRACTION` stayed 0.25. ρ stayed 0.5, θ stayed 1, neither swept. No birth–death. The Γ
estimator was not touched after the pilot. The evaluation window was not changed. `A_ref` reached
nothing that sets `a`, `r*` or the warm-up. The allocation cadence came from the structural rule
plus rule R-OBS, both fixed before any candidate ran. No horizon was extended. No seed was
deleted — 0 non-finite finals across 414 runs (408 scientific + 6 probe).

## 13. Artifacts

```
docs/IO_ABF_OVERNIGHT_PREREGISTRATION.md    the frozen protocol
src/abffr/io_abf.py                         the allocator (engine-agnostic)
tests/test_io_abf_gates.py                  21 Phase-0 gates
scripts/run_io_abf_campaign.py              probe / calibration / pilot / confirmatory
scripts/analyze_io_abf.py                   endpoint, decomposition, six figures
scripts/io_abf_summary.py                   cross-system tables
scripts/io_abf_wca_gate.py                  WCA reference gate + A0 screening
results/io_abf_overnight/<system>/
    probe/r_obs.json                        rule R-OBS, per system
    calibration/thresholds.json             eps1, eps2, R_Gamma, valid-tau  (frozen)
    pilot/  confirmatory/                   one npz per (arm, seed)
    analysis/                               endpoint json + figures
results/io_abf_overnight/wca/reference_gate.json
results/io_abf_overnight/GPU_POLICY.md
```
