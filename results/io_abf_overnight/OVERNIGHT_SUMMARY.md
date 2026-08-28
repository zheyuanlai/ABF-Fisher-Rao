# IO-ABF transfer campaign — overnight result

Run 2026-08-27, branch `q-r-decoupling`.
Frozen protocol: [`docs/IO_ABF_OVERNIGHT_PREREGISTRATION.md`](../../docs/IO_ABF_OVERNIGHT_PREREGISTRATION.md),
written and committed **before** any scientific run. No criterion in it was changed afterwards.

---

## 1. One-paragraph answer

**Information-optimal allocation transfers to three systems of four, and both the way it pays and
the way it breaks are now measured.** On the two entropic-bottleneck cells and the gateway, A6b
reaches the frozen stringent accuracy faster than plain ABF — S(ε₂) = 1.37–1.69, every paired 95 %
CI clear of 1, threshold reached in **32/32 seeds in all three** where A0 manages 22–27 — and it is
also *more* accurate than ABF at the horizon inside the evaluation window. But the preregistered
full-domain correctness guard fails in two of those three: the leverage `a(z)` is exactly zero
outside the evaluation mask, so `r* ∝ sqrt(aΓ)` gives those cells only the shared floor and their
free energy degrades. **IO-ABF buys accuracy where the endpoint scores it by spending accuracy
where it does not.** Only the entropic bottleneck at β = 8 clears the guard, so by the frozen rule
it is the campaign's single POSITIVE. On the fourth system, the WCA dimer, **A6b is 44 % *worse*
than plain ABF** — and the reason is not the difficulty theory (which was inert there) but the
realisation: the bias term `β⁻¹∇log r*` is unbounded and scales as 1/β, so at WCA's β = 1 it
applies a force 2.35× the physical mean force. A6c — the same mechanism with the genuine
Fisher–Rao mass constrained to ESS_M/K ≥ 0.5 — is **worse than plain ABF in all four systems**,
with Fisher–Rao demonstrably load-bearing, so that is a real price on representing `q_phys`, not an
inactive constraint.

---

## 2. Headline

| System | role | R_Γ | A6b S(ε₂) | 95 % CI | hit A6b/A0 | A6c S(ε₂) | mass ESS | final A6b/A0 | full-domain A6b/A0 | verdict |
|---|---|---:|---:|:--:|:--:|---:|---:|---:|---:|:--|
| Bottleneck β=4 | control | 21.6 | **1.694** | [1.281, 2.136] | 1.00/0.84 | 0.810 | 0.500 | 0.652 | 1.399 | **NOT POSITIVE** |
| Bottleneck β=8 | candidate | 12.4 | **1.395** | [1.313, 1.478] | 1.00/0.75 | 0.757 | 0.500 | 0.879 | 1.031 | **POSITIVE** |
| Entropic gateway | candidate | 123.7 | **1.366** | [1.158, 1.600] | 1.00/0.69 | 0.652 | 0.500 | 0.923 | 1.880 | **NOT POSITIVE** |
| WCA dimer | candidate | 1.8 | *0.595* | *unpaired* | 0.00/0.92 | *0.595* | 0.500 | **1.441** | **1.708** | **NEGATIVE** (pilot, unpaired) |

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

## 3b. What S(ε₂) is actually measuring — read this before quoting the speedups

The frozen threshold is `ε₂ = median e_A(0.6 T)` from A0 calibration. On all three systems the ABF
error curve has **already plateaued well before 0.6 T**, so ε₂ lands essentially *on* A0's own
asymptote:

| System | ε₂ | A0 final | A6b final | A0 final relative to ε₂ |
|---|---:|---:|---:|---:|
| β=4 | 0.05100 | 0.04911 | 0.03205 | **−3.7 %** |
| β=8 | 0.21211 | 0.20637 | 0.18149 | **−2.7 %** |
| gateway | 0.01063 | 0.01055 | 0.00973 | **−0.8 %** |

So `τ(ε₂)` for A0 is "when does A0 first sit three frames at its own asymptote" — a fragile,
noise-dominated quantity — while A6b, whose asymptote is 7–35 % lower, crosses early and in every
seed. That is exactly the 32/32-vs-22/32 hit pattern, and it means **S(ε₂) is largely a
re-expression of the asymptote gap rather than an independent measurement of rate.** The three
speedups order the same way as the three final-accuracy ratios (1.694/0.652, 1.395/0.879,
1.366/0.923), which is what one would expect if they are the same fact seen twice.

The cleaner statistic is the error ratio at fixed times, which needs no threshold:

| t / T | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| β=4 | 1.000 | 0.787 | 0.712 | 0.702 | 0.676 | 0.676 | 0.659 | 0.653 | 0.652 |
| β=8 | 1.000 | 1.001 | 0.980 | 0.960 | 0.950 | 0.927 | 0.909 | 0.892 | 0.879 |
| gateway | 1.000 | 0.999 | 0.962 | 0.983 | 0.954 | 0.932 | 0.921 | 0.928 | 0.923 |

Two things to take from it. The ratio is **exactly 1.000 at 0.2 T** in every system, which is the
allocation window opening — the arms really do share their burn-in identically, so nothing before
that point can be confounding the comparison. And the advantage then **grows monotonically and
never reverses**, in contrast to the transient-gain-then-reversal signature that v2, v3 and
clean-v2 all produced. On this evidence the honest claim is *"IO-ABF reaches a better free energy
inside the scored window, and the advantage widens with time"* rather than *"IO-ABF converges
1.4–1.7× faster"*. The preregistered endpoint stays the headline because it was frozen, but it
should be quoted with this paragraph attached.

A cleaner endpoint for the next campaign: set ε from a **fraction of the horizon at which the
curve is still falling**, or use the time-integrated error, or run long enough that 0.6 T is not
already asymptotic. All three thresholds here were frozen honestly and all three turned out to sit
in the flat part of the curve.

---

## 4. Q1–Q4, as preregistered

**Q1 — does estimator-risk allocation accelerate ABF?** **It improves ABF, reproducibly and in
every system; whether "accelerate" is the right word is settled by §3b, and the answer there is
"partly".** On the frozen endpoint, S(ε₂) = 1.37–1.69 with every CI above 1 on 32 fresh paired
seeds per system, and the censoring goes the *helpful* way: A6b reaches the stringent threshold in
32/32 seeds in all three systems where A0 reaches it in 27, 24 and 22, and a censored A0 seed is
charged only the horizon rather than the longer time it would really have needed, so the quoted
speedups are conservative *as speedups on that endpoint*. But ε₂ sits within 0.8–3.7 % of A0's own
asymptote, so most of that number is the asymptote gap rather than a rate difference. The
threshold-free reading is that the error ratio starts at exactly 1.000 when the allocation window
opens and falls monotonically to 0.652 / 0.879 / 0.923 without ever reversing. **Both statements
are true and the second is the one to build on.**

**Q2 — what does keeping the physical mass cost?** More than all of the gain. A6c runs at
0.65–0.81× plain ABF and its final error is 1.45–1.88× A0's. Retention `R_retain` is **negative in
all three systems** (−0.27, −0.61, −0.95) — though the ratio is the wrong statistic to read here,
because it varies mostly through its denominator. The *absolute* cost is much steadier:
`S_A6c − 1` is −0.19, −0.24, −0.35 while `S_A6b − 1` is +0.69, +0.40, +0.37. **A6c lands in about
the same place regardless of how much A6b gained**, which is what one expects if the constraint,
not the difficulty map, is what sets its target.

Why it is this expensive is visible in the mechanism: `q ∝ exp(−βÂ)` on a 10–20 kT barrier is
nearly two delta functions in the wells, and demanding `ESS_M/K ≥ 0.5` against a target that
peaked forces `r` to be nearly as peaked. That is close to the *unbiased* occupancy — precisely
the allocation ABF exists to escape. **ρ = 0.5 is a far more aggressive fidelity demand on a tall
barrier than the same number was on the kappa family**, and that, not a defect, is the most likely
reason Stage 2's 61–78 % retention does not survive the transfer. ρ was not swept, as preregistered;
the ρ–speed Pareto curve is a clean next experiment, not a rescue of this one. This is not an inactive constraint being blamed for
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

## 5b. The gateway question: is the slow region the expensive region?

The protocol asks specifically whether the gateway's slow-establishment region coincides with high
`a(z)Γ(z)` — and warns that if it does not, the old gateway FR positive was not an IO mechanism
and that has to be accepted. Measured on A0:

| System | share of total `aΓ` in the constriction (\|z\| < 0.5) | share of domain | enrichment | top cells by `aΓ` |
|---|---:|---:|---:|---|
| **Gateway** | **97.7 %** | 25 % | **3.91×** | z = ±0.169, ±0.281 |
| Bottleneck β=8 | 27.0 % | 25 % | 1.08× | z = ±0.506, ±0.619 |

**On the gateway the answer is yes, and emphatically.** Nearly all of the information cost sits in
the constriction, on its immediate shoulders at |z| ≈ 0.17–0.28 with a dip exactly at z = 0 — which
is where `ω'` is largest and the `ω ω' y²` force term has its variance. The premise the theory
rests on, that the region which is slow to establish is the region that is statistically expensive,
holds in the system built to be establishment-limited.

**But this does not show that the old gateway mFR positive was the IO mechanism, and the honest
reading is that it probably was not.** The two arms aim at different objects: the mFR target is
`q ∝ exp(−β(F_target − B))`, a *flattening* target that puts no special weight on the
constriction, while `r*` here is sharply peaked on the constriction shoulders and at the shared
floor almost everywhere else. Co-location of the *difficulty* with the constriction is not
co-location of the two *targets*. Settling it needs the old mFR arm's realised occupancy measured
against `r*` on the same seeds, which this campaign did not run. **Recorded as open, not as
confirmation.**

The contrast with β=8 is worth keeping. Same potential family, and its `aΓ` is essentially flat
(1.08× enrichment): its Γ peaks on the *flanks* at |z| ≈ 0.5 and at the walls near |z| ≈ 1.3, not
at the constriction. A narrow, severe gateway (s = 0.10, r = 32) concentrates difficulty; a wide,
mild one (s = 0.25, ω_in = 25 at β = 8) spreads it. That is the axis a follow-up should sweep.

---

## 6. The mechanism works, and the figures show it

`results/io_abf_overnight/<system>/analysis/figures_confirmatory/`, six figures per system.

The load-bearing one is **fig5**. With **no birth, no death and no resampling**, the bias alone
pulls the realised occupancy onto the target: TV(r*, realised) = 0.151 for A6b on β=8, against an
A0 occupancy with a completely different shape. Ancestor ESS stays at N for every IO run in every
system (gate G0.3, checked end to end). The claim that a replica density can be held by an
adaptive bias instead of by genealogy is now confirmed on two engine families.

**fig6 is the mechanism figure the project has been trying to draw for four campaigns**, and it
puts a number on the thesis. The two objects the campaign set out to separate are not merely
different, they are on different scales entirely:

| System | dynamic range of `q` (FR mass) | dynamic range of `r*` | TV(q, r*) | λ |
|---|---:|---:|---:|---:|
| β=4 | **1.4 × 10¹⁶** | 10.7 | 0.780 | 8.5e-4 |
| β=8 | **1.8 × 10¹⁹** | 8.1 | 0.774 | 7.2e-4 |
| gateway | **2.7 × 10¹³** | 16.1 | 0.775 | 1.2e-4 |

`q ∝ exp(−βÂ)` on a 10–20 kT barrier is two near-delta functions in the wells and 10⁻¹⁰ at the
barrier top; the information-optimal `r*` varies by less than a factor of 20 across the whole
domain. **Where probability mass should be and where sampling effort should go differ by thirteen
to nineteen orders of magnitude**, at a total variation of 0.78. Equal-weight birth–death imposed
`r = q`; asking for `ESS_M/K ≥ 0.5` asks for a weakened version of the same identification, and
that is why A6c pays what it pays. This is the clearest statement the project has produced of why
the two objects had to be decoupled, and it is a measurement rather than an argument.

---

## 7. WCA — the reference gate

**Gate verdict: PASS**, against `results/v2_validity_audits/wca_hp_v3/` (drop-in cache
`cache/phase_hp_v3/`).

**I got this wrong on the first pass and the correction matters.** My first gate run read
`wca_hp_reference/` — the 41-point build — and returned FAIL on its resolution. That build is
**superseded**. The repository already contains a full-resolution rebuild: 160 acquisition
z-values *on the evaluation grid itself* (interpolation factor exactly 1.0), no smoothing, PCHIP,
4 preparations × 96 replicas, 20 k prep / 20 k equilibration / 80 k production. The accepted
five-arm runner already defaults to `--cache-dir cache/phase_hp_v3`. Checking *a* reference is
not checking *the current* reference, and the two answers were opposite.

| Check | Value | |
|---|---|:--|
| physical parameters | n_dim 10, a 1.5, σ 1, ε 1, h 2, w 2, β 1 | match |
| grid / CV | 160 points on [−0.2, 1.2] | match |
| acquisition resolution | dz 0.00881 vs evaluation dz 0.00881 → factor **1.00** | PASS |
| smoothing | none | PASS |
| reference's own uncertainty on F′ | max se_prep 0.0747, se_replica 0.0610 | — |
| worst-case propagated uncertainty on F | 0.0217 RMS (fully-correlated bound) | — |
| against ABF's own final error (0.0901) | **24 %** | caveat, not blocking |
| Gate 0 (conditional equilibrates at fixed z) | rel spread 0.042 all / 0.048 transition | **PASS** |

Two caveats travel with the pass:

1. The reference's own uncertainty is 24 % of ABF's final error under a *fully-correlated*
   worst case. It is common to all arms and largely cancels in a paired threshold-crossing
   endpoint, but it bounds how fine an effect may be claimed.
2. **The default cache `cache/wca_ti_reference.npz` is the defective build** — wrong by 24.8 σ at
   z = 0.255. Any WCA run in this campaign must be pointed at `cache/phase_hp_v3`.

So a WCA A0/A6b/A6c comparison **is** licensed. What blocked it tonight was time, not validity:
WCA runs one method per process at ~13 min each under the R-OBS cadence, so the preregistered
design (16 calibration + 24 pilot + 96 confirmatory) is ~29 h on the single GPU this campaign is
restricted to. What was run is recorded in §8; the confirmatory is the next session's job.

## 8. WCA A0-only difficulty screening

**A0 calibration complete: 16 seeds, scored against `cache/phase_hp_v3`**, never the default
cache. The 8-seed paired pilot is running behind it. Frozen from this calibration:
ε₁ = 0.08340, ε₂ = 0.07917, T = 500.

| | WCA (n=16) | β=4 | β=8 | gateway |
|---|---:|---:|---:|---:|
| R_σ² | **2.06** | 73.7 | 18.4 | 648.5 |
| R_τ | **2.18** | 3.7 | 3.9 | 1.9 |
| R_Γ | **1.79** | 20.6 | 12.5 | 124.7 |
| valid-τ | **0.676** | 0.999 | 0.867 | 0.998 |
| ρ_s(Γ early, late) | **0.781** | 0.981 | 0.980 | 0.992 |
| e_A(T) median | 0.0761 | 0.0522 | 0.2024 | 0.0106 |

### WCA is marked `Gamma unresolved`

valid-τ = 0.676 against the 0.80 reliability gate. **A candidate failure on WCA may not be
attributed to the theory** — only to the fact that half of `Γ = σ²τ` is unmeasurable there. The
estimator was not adjusted to rescue it.

The reason is a hard limit, not a tuning failure: WCA's τ comes in at 3.9e-4 – 8.6e-4 time units
against a timestep of **2.0e-3** — a fifth to a half of *one integration step*. The lag-1 AR(1) fit
therefore sits in its `φ → 0` failure mode. No sampling cadence fixes this; you cannot sample
faster than `dt`. R-OBS was run at full density (`obs_every = 1`) and returned its own floor,
which is the estimator saying "below my resolution", not a measurement.

### What the WCA arm actually tests

The decisive statistic is not R_Γ but whether Γ moves the target at all. Both targets computed
from A0 data alone — the full `r* ∝ sqrt(aΓ̂)` and the pure-geometry `r* ∝ sqrt(a)`, both floored:

| System | n | R_Γ | TV(r_aΓ, r_a) | max ratio |
|---|---:|---:|---:|---:|
| β=4 | 32 | 20.1 | 0.199 | 2.03 |
| β=8 | 32 | 11.1 | 0.158 | 1.92 |
| gateway | 32 | 113.7 | 0.341 | 3.20 |
| **WCA** | 16 | **1.57** | **0.012** | **1.08** |

**On WCA the information-optimal target *is* the pure-leverage target, to within 1.2 % total
variation.** So the WCA arm tests the *static geometric* half of the theory and not the difficulty
half — a different experiment, and a useful one, but it must not be reported as a Γ-heterogeneity
result either way.

The defensible characterisation, stated with the limit rather than after it:

> **WCA's conditional force noise is nearly homogeneous across the reaction coordinate (σ² spans
> 2.1×), and its correlation time is unmeasurable at this timestep.**

Not "WCA has flat Γ". The σ² half is solidly measured — an instantaneous spread needs no window —
and on its own it bounds R_Γ small unless τ were strongly heterogeneous, which cannot be checked.

### The prediction registered before any candidate arm

`wca/PREDICTION_BEFORE_CANDIDATES.md`, written with two A0 records on disk and no A6b or A6c run
started, then amended (still pre-candidate) once the target-displacement number was in: **A6b
should improve *less* on WCA than on the heterogeneous systems — error ratio at the horizon
≥ 0.92, against 0.652 / 0.879 / 0.923 elsewhere — and its full-domain damage should be the
smallest of the four.** Falsifier: a ratio below 0.65 would mean the gain does not follow Γ. The
check is implemented in `analyze_io_abf_wca.py`, committed before the pilot produced a single
record, so it cannot be tuned to the answer.

One correction to an earlier reading of this section: at n = 6 the WCA threshold looked distinctly
less degenerate than the others. At n = 16 it is **3.8 %** above A0's own final, against 0.8–3.7 %
elsewhere — the top of the same range, not a different regime. §3b's caveat applies to WCA too.

---

## 8a. The WCA pilot — a clear negative, and a defect it exposed in the engine

**A6b and A6c are both substantially worse than plain ABF on the WCA dimer**, and this is the
system the plan called the most important physical benchmark.

| arm | n | e_A(T) median | IQR | full-domain | hit ε₂ | ratio to A0 | 95 % CI | Mann–Whitney |
|---|---:|---:|---:|---:|---:|---:|:--:|---:|
| A0 | 24 | 0.07593 | 0.00284 | 0.09273 | 22/24 | 1.000 | — | — |
| **A6b** | 8 | 0.10945 | 0.00415 | 0.15887 | **0/8** | **1.441** | [1.412, 1.489] | p = 9.5e-08 |
| **A6c** | 8 | 0.09733 | 0.00291 | 0.12499 | **0/8** | **1.282** | [1.253, 1.316] | p = 9.5e-08 |

The error ratio is above 1 at every fraction of the horizon and grows: 1.111 / 1.288 / 1.353 /
1.391 / 1.448 at t/T = 0.2 … 1.0. Note also that **A6c is *better* than A6b here** — the reverse of
every other system. The ESS constraint pulls the target back toward `q`, partially undoing a
reallocation that is doing harm.

### The pairing did not hold, and that is an engine defect, not an analysis choice

The 1.111 ratio at 0.2 T — where the allocation window has only just opened and the arms should
still be identical — was the tell. Tested directly: **two runs of plain ABF with the same seed in
the same process differ by 0.53 in the PMF.** The WCA sampler is not reproducible run to run at
all, presumably through non-deterministic CUDA atomics in the force accumulation amplified by
chaotic dynamics in float32. The instrumentation is *not* the cause — an A0-instrumented run
differs from an uninstrumented one by 0.22, which is *less* than two uninstrumented runs differ
from each other.

So the WCA arms were never paired, and the paired bootstrap in `analyze_io_abf_wca.py` does not
apply to them. **The table above is therefore the unpaired comparison**: 24 independent A0 runs
(16 calibration + 8 pilot) against 8 each of A6b and A6c, Mann–Whitney and an unpaired bootstrap.
The negative survives easily — plain ABF's own run-to-run spread is 10–90 % within [0.0730,
0.0791], about ±4 %, against a 44 % degradation.

> **This corrects a note in my own memory** which recorded WCA determinism as holding *within* a
> process and failing only across processes. Measured tonight, it fails within a process too. Any
> future WCA arm comparison must be analysed as unpaired, or the engine must be made deterministic
> first.

### Why it fails: the allocation force is unbounded, and it scales as 1/β

The realisation is `B_t = Â_t + β⁻¹ log r*`, so the applied force carries `β⁻¹ ∇log r*`. **Nothing
bounds that term, and it grows as β falls.** WCA runs at β = 1; the other three at β = 4, 8, 16.

| System | β | rms alloc force | max alloc force | rms \|F′_ref\| | rms ratio | max ratio |
|---|---:|---:|---:|---:|---:|---:|
| β=4 | 4 | 0.620 | 1.330 | 13.617 | 0.046 | 0.098 |
| β=8 | 8 | 0.421 | 1.292 | 13.552 | 0.031 | 0.095 |
| gateway | 16 | 0.264 | 0.873 | 2.732 | 0.097 | 0.320 |
| **WCA** | **1** | **4.813** | **16.864** | 7.179 | **0.670** | **2.349** |

On the three systems where IO-ABF helped, the allocation is a 3–10 % perturbation of the physical
mean force. **On WCA it is 67 % in RMS and 235 % at its peak** — it is not a reweighting of the
sampling, it is the dominant force in the reaction coordinate. The target itself is unremarkable
(`r*` spans 0.0078–0.0509 against a uniform 0.0312, a log-range of 1.88 over a domain of width
1.4); what differs is that at β = 1 the same target costs sixteen times the force it would at
β = 16.

**This is a real gap in the method as specified, and it is the single most actionable result of the
night.** The ABF part of the bias already carries `abf_force_clip = 40`; the allocation part
carries nothing. A cap, or a β-aware limit on target sharpness, is the obvious fix — and it was
deliberately **not** applied tonight, because inventing a knob after seeing the result is exactly
what the preregistration forbids. It is the next experiment, not a rescue of this one.

### The registered prediction, checked mechanically

Registered before any candidate ran: *"A6b should improve less on WCA — error ratio at the horizon
≥ 0.92"*, with *"ratio < 0.65 falsifies the mechanism"*. Observed: **1.448**. The stated one-sided
bound is satisfied and the falsifier is not triggered, so the check formally passes — but **the
spirit of the prediction failed**: I predicted a *modest gain* from the geometric factor alone and
what happened was substantial *harm*. Recording it as a pass and moving on would be the wrong
reading. What the prediction got right is that WCA behaves differently from the heterogeneous
systems; what it missed is the direction, because it reasoned about the *target* and the damage
came from the *force needed to realise it*.

### What may and may not be concluded

* WCA is marked **`Gamma unresolved`** (valid-τ 0.676), so this failure **may not be attributed to
  the difficulty theory**. And it should not be: on WCA the IO target *is* the pure-leverage target
  (TV 0.012), so the difficulty channel was inert and had nothing to fail at.
* What did fail is the **realisation**: holding a geometric reallocation with the bias, at β = 1,
  applied a force comparable to the physics.
* This is a **pilot** (8 seeds per arm). Under the protocol a pilot is an implementation check and
  no algorithm change follows it. The confirmatory was not run.

---

## 8b. The pilots, and a winner's-curse note

The 8-seed pilots were implementation checks only and **no algorithm change followed any of
them**, as preregistered. All clean: 0 non-finite finals in 72 pilot runs, valid-τ 0.77–1.00,
occupancy sane, no leakage.

| System | arm | S(ε₂) | hit | final/A0 | valid-τ | NaN |
|---|---|---:|---:|---:|---:|---:|
| β=4 | A6b | 2.147 | 1.000 | 0.642 | 0.946 | 0 |
| β=4 | A6c | 0.871 | 0.625 | 1.602 | 0.940 | 0 |
| β=8 | A6b | 1.435 | 1.000 | 0.887 | 0.805 | 0 |
| β=8 | A6c | 0.789 | 0.125 | 1.408 | 0.774 | 0 |
| gateway | A6b | 1.600 | 1.000 | 0.908 | 0.997 | 0 |
| gateway | A6c | 0.718 | 0.000 | 1.857 | 0.992 | 0 |

Every direction survived to the confirmatory. But **every magnitude shrank** — A6b went
2.147 → 1.694, 1.435 → 1.395, 1.600 → 1.366 on fresh seeds, and A6c 0.871 → 0.810, 0.789 → 0.757,
0.718 → 0.652. Three of three in the same direction is a small sample, but it is the shape of a
winner's curse and it is exactly what the fresh confirmatory block exists to catch. **Had the
pilot been reported as the result, all three speedups would have been overstated by 3–27 %.**

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

* **WCA A6b/A6c confirmatory** — the reference gate passes and the A0 calibration and 8-seed
  pilot both ran (§8, §8a); the 32-seed confirmatory is ~18 h more on one GPU. Given the pilot's
  size and direction (44 % worse, p = 9.5e-08 unpaired) the confirmatory should not be run until
  the unbounded-allocation-force problem in §8a is addressed — it would only measure the same
  defect more precisely.
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
