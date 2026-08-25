# v3.1: Consistent-target ABF–FR — preregistered two-track toy campaign

## Material Passport

- Artifact type: code-experiment protocol
- Status: **FROZEN 2026-08-25, before any v3 scientific run.** Further changes require a numbered amendment appended to this file.
- Revision: v3.1 (v3.0 draft red-lined on six technical points; see Revision log)
- Supersedes: `docs/PHYSICAL_TARGET_PULSE_V2_PREREGISTRATION.md` (pilot completed negative; the 2026-08-25 audit found the v2 operator did not faithfully implement its target)
- Primary system: the existing two-dimensional `xi(x,y)=x` ABF benchmark, geometry unchanged
- Evidence boundary: the 2026-08-25 audit is background evidence. Its three mechanistic claims (score-clip collapse, target–bias stationary conflict, clone-noise permanence) enter the *predictions* section only. They do not enter the gates.

## Background: why v2 does not close the question

The v2 pilot returned 0/54 schedules passing. The audit reproduced the gate
arithmetic exactly but found the operator unfaithful: with `score_clip = 5`
against a physical target spanning ~78 nats, ~88 % of particles sat outside the
clip and the applied score collapsed to a two-level step ("kill |x| ≳ 1.8, clone
uniformly from the interior"). The apparent +11 % cell is attributable to
evacuating the ~16 % of the ensemble that flat ABF parks outside the evaluation
mask. Final-frame F′ error was worse than plain ABF in every cell; every gainful
cell failed the genealogy gate; turnover never decayed inside any pulse. v2
therefore neither confirmed nor refuted physical-target FR.

## Research questions

- **Q-P (Track P — the original idea, faithfully implemented):** with standard
  ABF always on (estimator *and* full applied bias), does unclipped Fisher–Rao
  reallocation toward the estimated physical marginal accelerate convergence of
  F and F′ without genealogical collapse?
- **Q-C (Track C — the consistency-corrected family):** does Fisher–Rao
  reallocation toward a target that is *exactly the stationary marginal of the
  applied bias* **[wording superseded by Amendment 2: consistent under the
  current estimate, physically consistent asymptotically as A_t → F]**
  accelerate convergence relative to the same-bias no-FR control,
  without genealogical collapse?
- **Q-D (discretization):** for the same FR flow, which finite-K realization
  best preserves the continuum contraction per unit genealogy cost?

The claim under test is finite-time acceleration measured by time-to-accuracy,
with final-frame non-inferiority required. Integrated AUC is demoted to a
secondary descriptive statistic (v2 lesson: integrals bank transients).

## The family: one carrier, one law

**Carrier.** A single estimated free energy carries both the bias and the target.
It is defined at the force level and integrated:

    A′_t(z) := F̂′_t(z)   (the running ABF mean-force estimate)
    A_t(z)  := ∫ A′_t     (cumulative trapezoid; gauge irrelevant, see below)

**The v2 target EMA is retired** (`target_ema_alpha` removed from the scientific
method). Its measured memory was ~20 steps, negligible against the estimator's
own cumulative-from-t=0 averaging, and keeping two different smoothings of F was
the source of the v3.0 inconsistency. One carrier, no EMA.

**Family law.** Choose a shape function g. Then

    applied bias potential   B_t(z) := g(A_t(z)) − A_t(z)
    applied bias force       −∂_z B_t = A′_t(z) · [1 − g′(A_t(z))]
    FR target                q_t(z)  ∝ exp[−β g(A_t(z))]

so that q_t is *exactly* exp[−β(A_t + B_t)], the frozen-bias marginal under the
current estimate. Both the force and the target are functionals of the same A_t,
which is what makes the consistency exact rather than approximate.
**[Superseded by Amendment 2: this is exactness *in the estimated model*. The
true stationary marginal is p* ∝ q_t · exp[−β(F − A_t)], equal to q_t only as
A_t → F. Track P's mismatch, by contrast, survives a perfect estimator.]** Gauge: A_t is
defined up to an additive constant; the force is a derivative and the target is
normalized, so no gauge convention is needed except inside g (see below, where g
is written in terms of A_t − min A_t).

| Member | g(A) | applied bias force | q_t ∝ | Consistent |
|---|---|---|---|---|
| plain ABF (baseline) | 0 | A′_t (= standard ABF) | uniform | yes |
| P: physical | A | 0 … **overridden**: see note | exp(−β A_t) | **no, by design** |
| C-capped (primary) | (aβ)⁻¹ log(1 + e^{a u}) | A′_t [1 − σ(a u)] | (1 + e^{a u})^(−1/a) | yes |
| C-tempered (secondary) | A/γ_wt | (1 − 1/γ_wt) A′_t | exp(−β A_t/γ_wt) | yes |
| consistent-physical endpoint | A | 0 (estimator still on) | exp(−β A_t) | yes |

with the dimensionless depth variable

    u_t(z) := β [ A_t(z) − min_z A_t(z) ] − c_cut,     a := 2 (dimensionless sharpness)

**Note on Track P.** Track P is the deliberate violation: it applies the *full*
flattening force A′_t (g ≡ 0) while targeting exp(−β A_t) (g = A). It is the only
arm whose bias and target disagree, and that disagreement is the object of study.
Plain ABF is the g ≡ 0 member of the family exactly, so no carrier confound
exists between the baseline and any Track-C arm.

**Parameter units and transfer.** Every threshold below is dimensionless (kT).
`c_cut` is a declared method parameter meaning *the intended flattening depth*:
the method flattens the landscape wherever the estimate says it lies within
`c_cut` kT of the minimum, and lets physics suppress everything beyond. It is not
basin knowledge — no well or barrier location enters — but it must exceed the
largest barrier the user intends to cross, exactly as a metadynamics bias factor
must. Transfer rule to any later system: `c_cut` is set by this declared rule,
reported, and never tuned post hoc.

## Frozen backbone (identical in every arm)

Torch engine, CUDA, float64. Per step: (1) propagate all replicas under the
current applied bias; (2) accumulate force/count statistics from the propagated
configurations (post-propagation order; clones never counted before a
propagation); (3) update F̂′_t (binned_smooth, h = 0.05, update stride 10) and
integrate to A_t; (4) if an FR opportunity is scheduled, apply the FR operator;
(5) continue with independent physical noise from the matched noise bank.

**The ABF estimator runs in every arm, from the first step to the last.**
Justification, stated correctly: any *frozen* bias that is a function of ξ alone
leaves the equilibrium conditional measure on each fibre unchanged,
π_B(dq | ξ = z) = π(dq | ξ = z), because e^{−βB(z)} is constant on the fibre.
Continuously updating the mean-force estimator therefore remains structurally
valid under every applied-bias schedule in this campaign. It does **not** follow
that the estimator is unbiased in finite time: the realized conditional law of
the ensemble can be far from that equilibrium conditional, and this campaign
measures exactly that residue (clone-policy ladder, barrier F′). Finite-time
conditional non-equilibrium remains part of the estimator error — the failure
mode already on record in this project's pentane and alanine studies.

Constants: V(x,y) + a·x with a = 0.1021665783, β = 4, dt = 0.002,
n_steps = 50 000 (T = 100), K = 256, domain x ∈ [−3,3], y ∈ [−2.5,3.5], uniform
initial conditions, Langevin noise matched across arms per seed, KDE bandwidth
η = 0.10, profile grid 401 nodes. Pilot seeds 0–7; confirmation seeds 100–131.

**Domain decision.** The simulation domain stays [−3,3]. Shrinking it to the
useful support would use knowledge the deployable method is defined not to have
and would delete the very inefficiency this family addresses (flat targets waste
~43 % of the domain here). The v2 edge loophole is closed in the metric instead.

## FR operators — three discretizations of one flow

The continuum flow is ∂_τ p = −p [log(p/q) − E_p log(p/q)]. **No score clipping
anywhere; `score_clip`, `max_event_fraction` and the v2 event-cap machinery are
removed from the scientific method.** All weight arithmetic is in log space
(normalize by the max log-weight). Score:
S_i = log p̂(z_i) − log q(z_i) − (1/K)Σ_j [log p̂(z_j) − log q(z_j)].

**BD-standard** (the reference particle scheme; used online). Each particle
carries event rate |S_i|. If S_i > 0, particle i dies and a uniformly chosen
other replica is duplicated; if S_i < 0, particle i is duplicated and a uniformly
chosen other replica dies. Fixed K. Timestep per opportunity:

    Δτ = −log(1 − p_max) / Q_0.90(|S|)

where `p_max` is the event-probability cap **for the central 90 % of score
magnitudes**; the upper decile may exceed it, and that saturation is faithful to
the flow rather than a distortion. (v3.0 wrongly wrote Q_0.90(S⁺) and wrongly
labelled a different scheme "standard".)

**BD-paired** (offline only). Deaths Bernoulli from S⁺, births multinomial with
weight max(−S_i, 0), matched 1:1. Its mean-field limit is the same flow —
−p̂S⁺ + p̂S⁻ = −p̂S, using E[S⁺] = E[S⁻] under mean-centering — and it is expected
to have lower variance than BD-standard because births are placed where the flow
wants them rather than uniformly. It is a third discretization, not the standard
one, and it is evaluated only in the offline benchmark.

**FT (exact finite-time step)** (used online). Exact solution for fixed q:
p⁺ ∝ p^(1−θ) q^θ. Particle realization: log w_i = θ[log q(z_i) − log p̂(z_i)],
then systematic resampling to fixed K. Step size from the ESS governor: θ is the
largest value in [0,1] with ESS(w) = 1/Σ w̄_i² ≥ ρK (bisection, with a
verification scan — monotonicity of ESS in θ is an engineering gate, not an
assumption). When p̂ ≈ q the weights are near-uniform and systematic resampling
approaches the identity: the operator is self-limiting by construction.

**Matched-dose contrast.** BD-vs-FT at matched Δτ/θ is answered **only** by the
offline benchmark. Online P-BD and ESS-governed P-FT are a comparison between two
*deployable realizations* and must not be reported as a dose-matched
discretization contrast.

**FR schedule (pilot):** fixed window t ∈ [0.2T, 0.8T], opportunity stride
L_FR = 500 steps, in every FR arm. No discrepancy trigger and no adaptive gate:
self-limitation is a **measured outcome** (replacements per opportunity versus
time), never an assumed mechanism. Triggers are a downstream optimization
contingent on this pilot.

**Clone policies:** (a) exact copy (default, jitter 0); (b) hold-out — the clone
propagates normally but contributes to the ABF accumulators only after
L_hold = 500 steps; (c) oracle conditional refresh — y_child drawn exactly from
π(y | x) by grid inverse-CDF (diagnostic only; never part of a deployable arm).

## Arms: candidates, controls, diagnostics

Every arm shares the backbone; 8 matched seeds each.

**Baseline (1):** plain ABF (g ≡ 0, no FR).

**Track P — 6 configs** (candidates 5, diagnostic 1)
- P-BD-standard, p_max ∈ {0.02, 0.05, 0.10} — 3 candidates
- P-FT, ρ ∈ {0.70, 0.85} — 2 candidates
- P-oracle-target (q from F_ref, FT ρ = 0.85) — 1 diagnostic

**Track C — 13 configs** (candidates 6, same-bias controls 3, diagnostics 4)
- *Controls (not candidates; they cannot pass an FR gate — they are the
  reference for one):* C-capped no-FR, c_cut ∈ {8, 12} — 2; C-tempered no-FR,
  γ_wt = 8 — 1. (C-flat's control is plain ABF itself, the g ≡ 0 member.)
- *Candidates:* C-capped + FT, c_cut ∈ {8, 12} × ρ ∈ {0.70, 0.85} — 4;
  C-tempered + FT (γ_wt = 8, ρ = 0.85) — 1; C-flat-full-domain + FT (ρ = 0.85) — 1
- *Diagnostics:* C-oracle-target (capped c_cut = 12, ρ = 0.85) — 1; clone
  hold-out — 1; clone oracle-refresh — 1; K = 1024 sensitivity — 1

**Bridge (1, diagnostic):** consistent-physical endpoint (g = A ⇒ zero applied
bias, estimator on, FT ρ = 0.85).

**Total 21 configs × 8 seeds. Declared tuning budget** (the unequal-budget defect
class: count the configs each arm was allowed before comparing arms): plain ABF 1;
Track P 5 candidates; Track C 6 candidates. Selection compares candidates only.

**Choice of c_cut ∈ {8, 12}, declared rationale.** The reference barrier is
7.68 kT above the global minimum (6.83 above the shallower well). c_cut = 12
clears it by 4.3 kT; c_cut = 8 places the barrier inside the softplus shoulder,
where ~35 % of the flattening force is removed. The pair therefore *maps the
starvation boundary* rather than sampling two safe values. This uses reference
knowledge to design the experiment, which is declared here and does not enter any
deployable arm; the deployable rule for c_cut is the declared flattening depth.

**Infrastructure runs (not campaign arms):** plain-ABF and C-capped(12) runs at
K = 1024, 4 seeds, no FR, for offline cloud harvesting.

## Stage 0 — forensic closeout of v2 (runs on the v2 engine, unchanged)

Confirms the audit's mechanistic claims before v3 builds on them. Winning v2 cell
(γ = 0.1, bi = 0.2, sf = 0.3, fe = 100) with score_clip ∈ {5, 20, ∞}; the
γ = 0.1 / fe = 500 cell with max_event_fraction ∈ {0.1, 1.0}; the v2
`physical_oracle` arm. 8 seeds each. Log raw and applied score quantiles
(1/10/50/90/99 %), fraction clipped, and edge-strip population fraction. Stage 0
is an audit, not a method search; its only decision is the wording of the v2
post-mortem.

## Offline discretization benchmark (no MD)

Harvest ~50 particle clouds from the **K = 1024** infrastructure runs across
seeds and times (K = 512 and 1024 cannot be obtained by subsampling a K = 256
cloud — a v3.0 error). Subsample to K ∈ {64, 128, 256, 512, 1024}. For fixed q
(physical and capped variants), apply one BD-standard, one BD-paired and one FT
step per cloud at matched Δτ/θ, 100 FR random seeds per cell.

**Evaluation must not reuse the operator's own density estimate.** The FR weights
are built from the specified KDE (η = 0.10); the before/after marginals are
evaluated on a fixed fine grid with an *independent* leave-one-out-bandwidth KDE,
and KL to the known q is computed from that representation. A bandwidth-free
companion metric (1-Wasserstein distance to q) is reported alongside. Without
this split, the discretization that best overfits the operator's KDE wins
artificially.

**Summary statistic, defined in advance:**

    C_gene := [1 − ESS_anc⁺/K] / [KL⁻ − KL⁺]        (genealogy cost per nat)

computed only for cells with KL⁻ − KL⁺ ≥ 0.01 nat; cells below that threshold are
reported as "no measurable contraction" and excluded from the ratio, with their
count stated (no silent caps).

## Engineering gates (must pass before scientific runs)

Inherited from v2: target normalization; free-energy-offset invariance of every
target; oracle target agrees with the quadrature reference; zero-strength FR
(p_max = 0 / θ = 0) is trajectory-identical to the same-bias no-FR arm; FR never
touches accumulators directly; no FR activity outside the window; CPU/GPU score
agreement on a fixed cloud; estimator-pair agreement; noise keyed by seed and
step only, independent of method, batch size and shard membership; deterministic
whole-run restart; completion markers only after durable flush.

New for v3:
1. **Consistency gate — Track C families only, must pass.** For each Track-C
   (bias, target) family, a long no-FR run under the frozen bias must show
   p̂ → q_t within a stated tolerance. Track P is *excluded* from this gate: its
   mismatch is the scientific object of Q-P, recorded as a diagnostic, and its
   failure is a result rather than an engineering fault. (v3.0 wrongly listed a
   gate that all arms must pass while predicting one arm would fail it.)
2. **Family-law gate:** for each g, the applied force equals the numerical
   derivative of −B_t = A_t − g(A_t) to tolerance, and q_t equals
   exp[−β(A_t + B_t)] normalized — the exact-consistency check.
3. **Governor gate:** ESS(w(θ)) verified monotone in θ on fixed clouds; the
   bisection returns the boundary value within tolerance; a non-monotone cloud
   falls back to a grid scan and is logged.
4. **Log-space gate:** weights computed at score ranges up to 100 nats without
   overflow or underflow (synthetic cloud).
5. **Clone-policy gate:** hold-out clones contribute exactly zero observations for
   L_hold steps; oracle-refresh y-draws match the quadrature conditional on a
   fixed x-slice.
6. **Diagnostics gate:** conditional-diagnostics rows carry `config_id`; every
   figure caption and count is derived from data (the v2 latent-hazard fixes,
   verified by the new gate-logic tests).

## Outcomes

**Evaluation scopes (dimensionless; closes the v2 edge loophole in the metric,
not the physics):**
- **Primary scope R₁₂ := {z : β(F_ref(z) − min F_ref) ≤ 12}**, i.e. x ∈ [−1.74, 1.69],
  a single connected interval containing both wells and the 7.68 kT barrier with
  4.3 kT of margin, covering 57 % of the domain. Post-hoc, from the reference;
  never visible to any algorithm. (v3.0's "min F_ref + 6" was in energy units
  = 24 kT; the same numeral read thermally would have excluded the barrier and
  disconnected the scope — the error this revision exists to fix.)
- **Secondary: the full domain [−3,3]** — tail sacrifice is charged here.
- Legacy |x| ≤ 2.5: reported for v2 comparability only.
- Barrier region |x| ≤ 0.4: F′ error reported separately (evaluation-only).

**Primary endpoints:** time-to-accuracy in iterations,
τ_ε = min{n : e(m) ≤ ε for all m ∈ [n, n + Δn]}, Δn = 5 saved frames, on scope
R₁₂, for both e_F and e_F′, at two thresholds ε₁, ε₂ fixed **method-blind** from
the pilot plain-ABF median curves at 60 % and 80 % of budget and frozen before
any FR curve is viewed. Speedup S_ε = τ_ε^ABF / τ_ε^method.

**Mandatory companions:** final-frame non-inferiority on R₁₂
(e_F(T) ≤ 1.05 e_F^ABF(T) and e_F′(T) ≤ 1.05 e_F′^ABF(T)); barrier-region final
F′ ratio ≤ 1.10; genealogy (ancestor ESS, max family weight, cumulative
replacements); the **dose trajectory** (replacements per opportunity versus time);
the discrepancy trajectory KL(p̂_t ‖ q_t) with per-event drop and between-event
regrowth. AUC integrals: secondary, descriptive.

**Attribution rule.** All attribution is within-FR and within-family: an FR
candidate is measured against its own same-bias no-FR control; support versus
shape is separated by the C-flat / C-capped / C-tempered / P contrast;
estimation error by the oracle-target arms; clone handling by the clone-policy
ladder. **No non-FR reallocation mechanism is run** — project principle: Fisher–Rao
is the only reallocation operator studied, and it is the only member of the
marginal-only family defined for an arbitrary target.

## Two outcome tiers

v3.0 registered an expectation of single-digit effects while gating advancement at
≥ 10 %; that contradiction is resolved by separating the tiers.

**Mechanism-positive** (Track C, per candidate; all required):
1. dose decays by ≥ 5× from the first quarter of the FR window to the last;
2. genealogy gates pass (median ancestor ESS ≥ 0.5K and max family weight ≤ 0.10,
   each on ≥ 6/8 seeds);
3. final-frame non-inferiority on R₁₂ for both F and F′;
4. barrier F′ final ratio ≤ 1.10;
5. median S_ε ≥ 1.05 at ε₂ for both F and F′ **against its same-bias no-FR
   control**, favorable on ≥ 6/8 seeds.

A mechanism-positive is a genuine success: it establishes that a consistent FR
target produces self-limiting, genealogy-safe, harm-free reallocation. It
licenses **exactly one** next step — the β = 8 entropic-bottleneck toy, the
registered home of larger effects, with the algorithm frozen. It does **not**
license WCA or any molecular system.

**Advancement-positive** (per candidate, either track; all required):
1. median S_ε ≥ 1.10 at both thresholds for F **and** both for F′ (four medians),
   against **plain ABF**;
2. for Track C additionally, the mechanism-positive criteria above;
3. favorable sign on ≥ 6/8 seeds at ε₂ for both F and F′;
4. final-frame non-inferiority (both, R₁₂) on the median and on ≥ 6/8 seeds;
5. barrier F′ final ratio ≤ 1.10 (median);
6. genealogy gates as above;
7. full-domain final e_F ratio ≤ 1.25 (declared tail-sacrifice cap).

An advancement-positive licenses the full benchmark ladder. Among passing
candidates choose the gentlest: lowest cumulative replacements, then larger ρ (or
smaller p_max), then smaller c_cut. If no candidate passes in a track, that
track's hypothesis is reported unsupported at this gate.

## Registered predictions (written before any v3 run)

- **P1** (Stage 0): removing the clip restores a multi-level score distribution
  and materially changes behavior ⇒ v2 did not faithfully test physical-target FR.
- **P2** (Track P): per-event KL drop followed by between-event regrowth; dose
  does **not** decay across the window; every cell with measurable gain fails the
  genealogy gate; final F′ non-inferiority fails or ties. If Track P instead
  passes its gates, the original idea stands and Track C becomes the ablation.
- **P3** (Track C capped): dose decays ≥ 5× across the window; genealogy passes at
  ρ = 0.85; the no-FR capped control captures ≥ 50 % of the support-related gain;
  FR's own increment appears mainly in τ_ε at ε₁.
- **P4** (shape): physical and tempered targets underperform capped on barrier F′;
  c_cut = 8 shows measurable barrier starvation relative to c_cut = 12.
- **P5** (clones): exact ≤ hold-out ≤ oracle-refresh on final F′, with the gap
  concentrated at the barrier (within-fibre y relaxation ≈ 303 ≫ T = 100).
- **P6** (offline): FT dominates BD-standard on variance and on C_gene at matched
  dose; BD-paired sits between them; all gaps shrink with K.
- **P7**: the consistent-physical endpoint is the worst arm (crossing starvation).
- **P8** (tier expectation): Track-C effects on this benchmark are small
  (single-digit % in τ_ε). This toy is a **mechanism** test; effect-size claims
  belong to the harder rungs. A mechanism-positive here is a success and licenses
  only the β = 8 rung — it does not satisfy the advancement threshold.
- **P9** (governor as a consistency detector): under Track P's inconsistent
  target the ESS governor self-throttles to θ ≈ 0, making P-FT nearly inert while
  P-BD churns — the governor refuses to move when the target disagrees with where
  the particles are. If observed, this is a reportable property of the FT
  realization, not a null result.

## Fresh-seed confirmation

If and only if a candidate is advancement-positive: freeze it, seeds 100–131, no
retuning. Arms: plain ABF; the frozen candidate; its same-bias no-FR control; the
oracle-target diagnostic. Confirmation requires median S_ε ≥ 1.10 at both
thresholds for F and F′ with paired bootstrap 95 % CIs wholly favorable, plus all
companion gates. Both tracks passing ⇒ both confirmed on the same fresh seeds.
Whole confirmatory blocks run in ONE process (cross-process determinism trap on
record).

## Downstream ladder (algorithm frozen after confirmation)

1. β = 8 entropic-bottleneck toy — licensed by a mechanism-positive or better.
2. WCA dimer — requires an advancement-positive. Solvent-clone decorrelation measured.
3. Molecular torsion (pentane first: its hidden-conditional harm regime is where
   the clone-policy result matters). No per-system FR redesign.

Adverse and null cells remain in every figure. Figures are generated from saved
CSVs with data-derived captions (no literals), exported as matching PNG and PDF.

## Amendments

**Amendment 1 (2026-08-25, before any v3 scientific run) — identity gates need a
stated tolerance; the GPU engine is not bitwise reproducible.**

Measured while verifying that the Stage 0 score-shape instrumentation does not
perturb trajectories: running *identical code with an identical command and
identical batch layout twice* on the same GPU gives
max |Δ l2_F| = 3.9e-7, max |Δ l2_F′| = 3.4e-7, max |Δ ΔF̂| = 1.2e-6 over the
50 000-step horizon, while **every discrete counter (ancestor ESS, cumulative
replacements, barrier crossings) is exactly identical**. The cause is
non-deterministic reduction order in CUDA `scatter_add_` inside the ABF
accumulators, accumulated over 50 000 steps; it is ~1e-5 relative to a final
l2_F of ~0.03, i.e. far below the seed-to-seed spread (0.0144–0.0598).

Consequences, binding on this campaign:
1. No engineering gate in this document may be read as a *bitwise* identity gate.
   The zero-strength-FR gate, the CPU/GPU agreement gate, the estimator-pair gate
   and the deterministic-restart gate are all satisfied at
   **tolerance 1e-5 absolute on profile-derived quantities**, together with
   **exact equality on all discrete counters** (ancestry, event and replacement
   counts, crossings). The discrete-counter half is the sharper test and is
   required, not optional.
2. Any future claim that two configurations produced "identical" runs must state
   which of the two halves it means.
3. This does not affect paired comparisons: arms share Langevin variates per
   seed, and the paired differences under study are 10^3–10^5 times this noise.

**Amendment 2 (2026-08-25, before any v3 scientific run) — "exact stationary
consistency" overstated; the consistency gate as frozen would fail correct code.**

The family law states q_t ∝ exp[−β g(A_t)] with B_t = g(A_t) − A_t, and v3.1
described q_t as "exactly the stationary marginal of the applied bias". That is
exact only *in the estimated model*. Under a frozen bias B_t the true stationary
marginal involves the true free energy F:

    p*_{B_t}(z) ∝ exp[−β(F(z) + B_t(z))]
               = exp[−β g(A_t(z))] · exp[−β(F(z) − A_t(z))]
               ∝ q_t(z) · exp[−β(F(z) − A_t(z))]

so **p*_{B_t} = q_t if and only if A_t = F + const.** The correct statement is:

> Track C is *algebraically consistent with its current estimate* at every
> instant, and *physically consistent asymptotically* as A_t → F.

Verified numerically on the reference profile: with A = F_ref, max|p* − q| =
5.6e-16; with a perturbed carrier A = F_ref + 0.3·sin(2.1x) + 0.15·cos(3.7x),
max|p* − q| = 0.28 while the corrected identity above holds to 6.7e-16.

**This is not a weakening of the Track P / Track C distinction.** For Track P the
applied bias is B = −A_t, so even at A_t = F exactly the stationary marginal is
uniform while the target is exp(−βF): the conflict survives a perfect estimator
(verified: discrepancy 1.81 at A = F_ref). Track C's residual mismatch vanishes as
the estimate converges; Track P's does not. Stage 0 already demonstrated this
experimentally — its oracle-target arm, which removes estimation error entirely,
was as harmful at the endpoint as the estimated one (final F′ ratio 1.379).

**Gate replacement.** New-for-v3 engineering gate 1 (the consistency gate) is
withdrawn as stated — a correct implementation with an imperfect frozen carrier
would fail it, which is this project's "spec error faithfully implemented"
defect class. It is replaced by two physical-stationarity gates, both Track C
only, both required:

- **Gate 1A (arbitrary frozen carrier).** For any frozen A and B = g(A) − A, a
  long no-FR run must converge to
  p_expected(z) ∝ exp[−β(F_ref(z) + B(z))], normalized on the profile grid.
  This tests the biased-dynamics implementation itself.
- **Gate 1B (oracle carrier).** Setting A = F_ref gives B = g(F_ref) − F_ref and
  hence p*_B = q exactly. A long no-FR run must converge to q within tolerance.
  This tests the family's intended asymptotic consistency.

The algebraic family-law gate (new-for-v3 gate 2) is retained unchanged: it tests
the code's algebra, whereas 1A/1B test physical stationarity. Track P remains
excluded from all three; its mismatch is the object of Q-P.

**`C-oracle-target` redefined.** To isolate *target*-estimation error alone, the
Track-C oracle arm keeps the candidate's own estimated bias B_t and oracles only
the target:

    q_t^oracle(z) ∝ exp[−β(F_ref(z) + B_t(z))]      (= q_t · exp[−β(F_ref − A_t)])

i.e. the true stationary marginal of the bias actually applied. Replacing A_t by
F_ref inside *both* the bias and the target would change two things at once and
is not what this diagnostic is for.

Note that the two oracle arms now mean deliberately different things, and this is
correct: **P-oracle-target** oracles the *idea's* target (exp(−βF_ref), the
mismatch tested with a perfect target, unchanged from v3.1), while
**C-oracle-target** oracles the *achievable* target. Neither enters selection.

**Interpretive refinement to P3 (no threshold change).** Because the mismatch is
only asymptotically zero, the FR dose under Track C should decay *as the
estimator converges*, not to zero immediately; the residual dose floor is a
readout of ‖F − A_t‖. The registered ≥5× decay threshold is unchanged; this
records what that number measures.

**No other change.** The 21 arms, c_cut ∈ {8,12}, γ_wt = 8, L_FR = 500, ρ, p_max,
clone policies, evaluation scopes, mechanism/advancement thresholds, Track P, the
offline benchmark and the downstream ladder are all unchanged.

**Amendment 3 (2026-08-25, before any v3 scientific run) — the FR-time/θ mapping
is frozen; "matched Δτ/θ" is now exactly defined.**

v3.1 specified the offline benchmark at "matched Δτ/θ" without saying how the
two are related. θ is *not* Fisher–Rao time. The exact solution of the flow for
fixed q is p_τ ∝ p₀^{e^{−τ}} q^{1−e^{−τ}}, so with the FT parameterization
p⁺ ∝ p^{1−θ} q^{θ},

    θ = 1 − exp(−τ)          τ = −log(1 − θ)

and the matched FT dose for a BD step of length Δτ_BD is

    θ_matched = 1 − exp(−Δτ_BD),     **not**   θ = Δτ_BD.

The two agree to O(Δτ²) but diverge at the doses this campaign uses: the
relative error of the naive identification is 2.5 % at Δτ = 0.05, 10.3 % at
Δτ = 0.2, 27.1 % at Δτ = 0.5 and 58.2 % at Δτ = 1.0.

The same mapping fixes the composition law, which becomes a sharp unit test that
the exponent has not been reversed:

    T_{θ₂}(T_{θ₁}[p]) = T_{θ₁₂}[p],   θ₁₂ = 1 − (1−θ₁)(1−θ₂) = θ₁ + θ₂ − θ₁θ₂

verified numerically to 2.2e-16, against 1.4e-1 for the naive θ₁ + θ₂.

Binding: every dose-matched offline BD/FT comparison uses θ = 1 − exp(−Δτ_BD).
The online arms are unaffected — P-BD is parameterized by p_max and P-FT by the
ESS governor, and v3.1 already forbids reading the online pair as a dose-matched
contrast. No arm, parameter or threshold changes.

**ESS monotonicity is a theorem, not an assumption (gate strengthened).** With
a_i = log q(z_i) − log p̂(z_i) and Z(s) = Σ_i e^{s a_i}, the governor's ESS is

    ESS(θ) = Z(θ)² / Z(2θ),     d/dθ log ESS = 2[m(θ) − m(2θ)],  m(s) = Z′(s)/Z(s)

and m′(s) = Var_{w(s)}(a) ≥ 0, so m is non-decreasing, m(2θ) ≥ m(θ) for θ ≥ 0,
and **ESS is non-increasing on [0,1]** for any fixed cloud. The numerical
monotonicity gate is retained, but a violation is now an engineering anomaly
(numerical pathology) rather than a property of the cloud, and the bisection's
grid-scan fallback must therefore **log loudly** whenever it fires rather than
silently substituting a scan.

**Amendment 4 (2026-08-25, before any v3 scientific run) — two interpretive
corrections, and diagnostics-only logging at every FR opportunity.**

**4a. rho implies no genealogy bound.** The ESS governor constrains the current
particle *weights*, ESS_w = 1/sum_i w_i^2. The genealogy gate measures mass
aggregated by initial ancestor, W_a = sum_{i: anc(i)=a} w_i and
ESS_anc = 1/sum_a W_a^2. These are different objects: with all K particles
descended from one ancestor but uniform weights, ESS_w = K while ESS_anc = 1
(verified). Even one realistic FT step at rho = 0.85 was measured to leave
ESS_w/K = 0.850 but ESS_anc/K = 0.762. Therefore any statement of the form
"rho^m is the worst case" is a **compounding heuristic with no bound behind
it**, and must be labelled as such. The correct statement:

> rho controls the violence of one FT weighting step and provides no cumulative
> genealogy guarantee.

This strengthens rather than weakens the frozen design: the governor is
deliberately local, and genealogy safety is an independent global gate. Both
remain separately required.

**4b. Residual FR dose is not a readout of ||F - A_t||.** Amendment 2's
interpretive note on P3 said the residual dose floor measures the carrier error.
That is wrong as stated and is corrected here. The measured FR discrepancy
carries three contributions:

    FR discrepancy = carrier error (F - A_t)
                   + finite-time physical non-equilibrium
                   + finite-K / KDE fluctuation

Even at A_t = F exactly, a finite ensemble at finite time does not satisfy
p_hat_t = q_t. P3 is therefore to be read as:

> As A_t -> F the *systematic* target-bias inconsistency disappears, so the FR
> dose should decay toward a finite-sampling noise floor.

The registered >= 5x decay threshold is unchanged.

**4c. Diagnostics-only logging (never gates).** Because this toy has F_ref, the
three contributions above can be separated after the fact. Logged at every FR
opportunity, for diagnosis only and never entering selection:

- carrier error e_A(t) = ||A_t - F_ref - c_t||_{L2(R12)} at the optimal gauge c_t;
- consistency mismatch D_cons(t) = KL(p*_{B_t} || q_t) with
  p*_{B_t} propto exp(-beta(F_ref + B_t)) -- the exact Amendment-2 residue;
- KL(p_hat || q) immediately before and after the FR operator;
- FT: theta_t, ESS_w/K, R_t/K;  BD: dtau_t, Q_0.90(|S_t|), mean and max event
  probability, R_t/K;
- both: ESS_anc/K and w_max before and after, and the **genealogy retention
  factor** G_t = ESS_anc^+ / ESS_anc^-.

G_t distinguishes ancestry lost to a few catastrophic early events from mild
persistent coalescence -- mechanistically very different, and indistinguishable
from the endpoint value alone.

**4d. Genealogy damage is irreversible.** Once two ancestral lineages coalesce
through selection, later dose decay cannot recreate them. A candidate may
therefore show a textbook theta_t -> 0 curve and still fail the genealogy gate
because the damage happened early. That is not a contradiction in the protocol;
it is why v3.1 requires self-limitation and genealogy safety as separate
conditions.

## Implementation appendix A (frozen 2026-08-25, before the infrastructure runs)

Clarifications that make already-frozen text unambiguous. They add no arm, no
parameter and no threshold, and are recorded here because each of them could
otherwise be settled *after* seeing data.

**A.1 Offline cloud sampling — no post-hoc selection.** v3.1 says "~50 clouds
across seeds and times". Frozen exactly as

    4 seeds x 2 infrastructure families x 6 normalized times = 48 clouds

with the two families being plain ABF and C-capped(c_cut = 12) at K = 1024, and
the six normalized times fixed in advance at

    t/T = 0.15, 0.30, 0.45, 0.60, 0.75, 0.90

identical for both families. Clouds are never selected, weighted or dropped on
the basis of their KL, score shape or any benchmark outcome. Q-D is a
preregistered prediction (P6), so choosing "representative" clouds after
inspecting them would decide the prediction it is meant to test.

**A.2 Offline dose set.** Every offline comparison is run at the three
registered online BD strengths, p_max ∈ {0.02, 0.05, 0.10}. For each cloud the
FR time is

    dtau(p_max) = -log(1 - p_max) / Q_0.90(|S|)

computed from that cloud's own score, and the matched FT dose is Amendment 3's

    theta(p_max) = 1 - exp(-dtau(p_max)).

BD-standard, BD-paired and FT therefore represent the same nominal FR time on
every cloud, which is what makes P6 interpretable.

**A.3 FR opportunity indices.** With n_steps = 50 000 and the window [0.2T, 0.8T]
at stride L_FR = 500, the opportunity set is exactly

    J_FR = {10000, 10500, ..., 40000},   |J_FR| = 61

both endpoints included. FR fires *after* the propagation and estimator update
of that step, per the frozen backbone order. The schedule gate asserts this
entire index array, not merely "no events outside the window" — that weaker
assertion passes for a first event at 10500, a missing event at 40000, or a
stride applied to estimator updates instead of physical steps.

**A.4 Hold-out counter semantics.** h_i in {0, ..., L_hold} per slot. At each
physical step the slot propagates normally; if h_i > 0 its observation is not
deposited in the ABF accumulators; then h_i is decremented. So h_i = L_hold
excludes exactly the next L_hold propagated observations. Under FT, one
descendant per parent is the continuation and inherits the parent's remaining
counter; the other N_j - 1 descendants are new clones and receive a fresh
L_hold, including clones of an already-held-out replica.

**A.5 Oracle refresh acts only on the fibre.** A new clone from parent
(x_j, y_j) keeps x_child = x_j exactly and draws y_child ~ pi(y | x_j). It must
not move x, change the FR offspring count, change ancestry, or deposit an ABF
observation at the instant of refresh; the child becomes estimator information
only after a physical propagation, subject to its clone policy.

**A.6 Three independent random streams.** Physical noise is keyed by
(seed, physical step, slot) and by nothing else — not ancestry, not FR event
count, not the number of FR draws consumed, not method, not batch or shard
layout. FR randomness (event Bernoullis, event ordering, uniform partners,
systematic-resampling offset) is keyed by (seed, opportunity, draw). Oracle
refresh uses a third stream keyed by (seed, opportunity, child slot). The
invariant under test:

> FR may change which configuration occupies slot i, but never which future
> Langevin variates belong to slot i.

The regression gate generates the MD noise bank once with FR disabled and once
after consuming a large number of FR and oracle draws, and requires the two to
be *exactly* equal.

## Revision log

**v3.1 (2026-08-25), six corrections to the v3.0 draft, all adopted:**
1. Track C was not exactly consistent: the bias used F̂′ while the target used the
   EMA F̄. Fixed by a single carrier A_t and the family law B = g(A) − A,
   q ∝ exp(−βg(A)); the EMA is retired, which also makes plain ABF the exact
   g ≡ 0 member and removes the carrier confound without an extra arm.
2. "Unbiased under every applied-bias schedule" replaced by the correct
   frozen-bias fibre-invariance statement, with finite-time conditional
   non-equilibrium explicitly retained as estimator error.
3. BD is now the standard birth–death realization (rate |S_i|, uniformly chosen
   partner); the paired variant is relabelled BD-paired and moved to the offline
   benchmark; the percentile timestep uses Q_0.90(|S|) and p_max is described as a
   central-90 % cap.
4. Mechanism-positive and advancement-positive tiers separated, resolving the
   contradiction between P8 and the ≥ 10 % gate; each tier's licensing scope stated.
5. All thresholds made dimensionless: scope R₁₂ (was an energy-unit "+6"),
   c_cut ∈ {8, 12} kT chosen to bracket the measured 7.68 kT barrier, sharpness
   a = 2 dimensionless, and a declared transfer rule for c_cut.
6. Bookkeeping: the consistency gate applies to Track C only (Track P's mismatch
   is a diagnostic, not an engineering failure); offline clouds come from
   dedicated K = 1024 runs; matched-dose BD/FT is offline-only; Track C is split
   into candidates versus same-bias controls. Additionally, offline KL is now
   evaluated with an independent leave-one-out-bandwidth estimator plus a
   bandwidth-free companion, and C_gene is defined with a stated exclusion rule.
