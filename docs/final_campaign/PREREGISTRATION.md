# Final mechanism and computational-validation campaign — preregistration

*Written 2026-10-11, before any engine change, validation run, smoke run, benchmark or production run of this
campaign. Revised once before freezing after a five-lens adversarial review (spec compliance, statistics, numerics,
Experiment B fairness, factual accuracy; findings and their resolution in `PREREG_REVIEW_RESOLUTION.md`). Frozen in git
together with `configs/final_campaign/{expA.json, expA_validation.json, expB.json}`, `EVIDENCE_LEDGER.md` and
`results/final_campaign/seed_check.json`. Later changes are appended as dated amendments below the final line; nothing
above it is edited after the freeze commit.*

Repository state at writing: `main` = `origin/main` = `189c2408`. The current scientific state is summarised in
`EVIDENCE_LEDGER.md` (§2 claims C1–C8; §1 WCA reconciliation). This campaign asks two questions and nothing else:

* **Experiment A.** On the fixed shifted-fibre potential, does faster transverse mobility λ — which leaves the Gibbs
  law, F\*, the conditional law and Var(f | x) unchanged — rescue FR?
* **Experiment B.** With a genuinely walker-parallel CPU backend, does FR at large N beat the best ABF allocation in
  wall-clock time on the same hardware?

No WCA run is authorised. No FR strength, rate, bandwidth, cap, schedule, estimator setting or mobility value is
tuned or added after any performance outcome is seen. A neutral or negative result is not a stopping reason.

Every statement in the reports of this campaign is labelled **[identity]**, **[observed]** or **[hypothesis]**, and
the four claim types of `EVIDENCE_LEDGER.md` §2 (fixed-N benefit, equal-force-evaluation benefit, wall-clock benefit
on stated hardware, correctness of the limiting estimate) are never merged.

---

## 1. Experiment A — does faster transverse mobility rescue FR on the shifted fibre?

### 1.1 Model, and what the intervention does and does not change

The existing shifted fibre of the mechanism campaign (`docs/mechanism/SCIENTIFIC_PLAN.md` §1.1), κ = 1, with every
gateway physical constant unchanged (β = 16, H = 0.5, s = 0.1, ω(x) = 1 + 31 exp(−x²/(2s²)), x reflected into
[−1.8, 1.8], y unbounded):

  V_shift(x, y) = F\*(x) + (κ²/2)(y − m(x))²,  m(x) = (√(2/β)/κ) log ω(x),  F\*(x) = H(x² − 1)² + β⁻¹ log ω(x).

Dynamics (Euler–Maruyama, the engine `src/gateway_family_numba.py`, `variant = "shift"`, `kappa = 1`, `lam = λ`):

  X ← reflect(X + (−∂ₓV + F̂′(X)) h + √(2h/β) ξₓ),  Y ← Y − λ ∂ᵧV h + √(2λh/β) ξᵧ.

**Cells:** λ ∈ {1, 4, 16} as specified by the task, plus **λ = 128 as a preregistered extension** (reason below).

**[identity] Unchanged by λ.** Scaling the transverse drift and noise together leaves exp(−βV_shift) invariant, so for
every λ > 0: F = F\* + C; E[f | x] = F\*′(x) with f = ∂ₓV the deposited local force; Y | x ~ N(m(x), 1/(βκ²));
Var(f | x) = κ² m′(x)²/β = (2/β²)(ω′/ω)² (max 2.062 at |x| = 0.210). The x-update formula, the ABF estimator and FR
contain no λ. At frozen x the EM y-update is an exact AR(1) for d = y − m(x) with coefficient φ = 1 − λκ²h, so the
frozen-x relaxation time is 1/(λκ²) = 1, 0.25, 0.0625, 0.0078, and the EM stationary transverse variance is the exact
one times 1/(1 − λκ²h/2), a relative excess of 1.25e-5, 5.0e-5, 2.0e-4 and 1.6e-3 at h = 2.5e-5.

**[identity + observed] Changed by λ — more than conditional relaxation.** With x free, d relaxes at rate
κ²(λ + m′(x)²), because x itself moves to put the walker back on the channel centre (the x-force contains
−κ² d m′(x)); m′² reaches 33.0 at |x| = 0.210, exactly where Var(f | x) peaks and where the mechanism campaign located
FR's biased young-clone deposits. Hence, relative to λ = 1, the coupled flank relaxation is faster by only **1.09×,
1.44× and 4.74×** at λ = 4, 16, 128 (frozen-x: 4×, 16×, 128×; the 1/λ scaling holds only where m′ ≈ 0, i.e. in the
wells and at the gate centre, where Var(f | x) ≈ 0). Conversely, along the curved channel x can advance only as fast
as y follows m(x): the local x-mobility factor λ/(λ + m′²) is 0.029, 0.108, 0.327, 0.795. **[observed, scouting
before the freeze]** Flat-bias chains of the frozen validation kernel (no ABF, no FR; 512 walkers, 30 t.u. per λ,
`scripts/final_campaign/expA_scout_transport.py`, `results/final_campaign/expA/scout/`) give hysteresis side
changes between x < −0.5 and x > 0.5 of 0.0043 ± 0.0005, 0.0108 ± 0.0009, 0.0204 ± 0.0015, 0.0298 ± 0.0018 per walker
per t.u. at λ = 1, 4, 16, 128 (×2.5, ×4.8, ×6.9), and flank (x₀ = −0.21) 1/e conditional times 0.037, 0.033,
0.023, 0.0066 t.u. (×1.1, ×1.6, ×5.6; coupled prediction 1/(κ²(λ + m′²)) = 0.029, 0.027, 0.020, 0.0062), while in the
wells and at the gate centre they follow 1/λ. **So transverse mobility changes gate transport (and with it ABF's own
establishment) at least as much as conditional relaxation. This design cannot isolate conditional relaxation from transport**; it tests
"transverse mobility", and every attribution to conditional relaxation alone is a [hypothesis]. λ = 128 ≫ max m′² is
added because it is the only mobility at which the flank relaxation changes materially; it changes transport further
as well. **The engine is not modified** (`src/gateway_family_numba.py`, sha256 `69087f2c…`, supports any λ > 0;
λ ≠ 1 changes only the y update).

### 1.2 Hypotheses, manipulation checks and contrasts (frozen)

* **HA1 (primary, task-specified).** Faster transverse mobility improves FR's performance relative to ABF on the
  integrated free-energy error: the FR/ABF Ī_F ratio is smaller at λ = 16 than at λ = 1.
* **HA1x (extension).** The same for λ = 128 vs λ = 1.
* **HA3 (secondary).** FR improves the reaction-coordinate marginal at every λ.
* **Manipulation checks** (not hypotheses about FR; they describe what the intervention did): **MC1** coupled
  conditional relaxation (1/e time and integrals to fixed cuts of ρ_y, ρ_f at x₀ = −1, −0.21, 0); **MC2** gate
  transport (side-change rate between x < −0.5 and x > 0.5 per walker per t.u.), both from the G-A1 flat-bias chains,
  per λ with jackknife CIs; **MC3** the change from λ = 1 of the matched excess bias of young-clone deposits (class
  [0.01, 0.1) t.u.) over ABF's deposits, from production (D2). MC3 passes ("young-clone bias reduced") iff the
  Holm-adjusted (over N) `B_excess_did_p` < 0.05 with a negative point estimate.
* **ABF-only cross-λ descriptors:** ABF Ī_F, ABF gate transitions per run, ABF τ(TV_half ≤ 0.1) and first arrival,
  per λ, with seed-bootstrap CIs of the λ/λ = 1 ratios.

Contrasts, at each N ∈ {128, 512} separately, seeds coupled across λ (§1.4):

  E(λ_a vs λ_b; N): R = exp(Δ), Δ = median over seeds s of [ L_{λa}(s) − L_{λb}(s) ],  L_λ(s) = log(Ī_F^FR(s) / Ī_F^ABF(s)),

with a 10 000-resample seed bootstrap (`default_rng(20261010)`) percentile 95 % CI and two-sided bootstrap p (the
existing `scripts/mechanism/cross_cell.py` construction). R < 1 means FR's relative performance is better at λ_a.
This is a median of per-seed log-ratio differences, not a ratio of median G values.

* **E1 (primary):** λ 16 vs λ 1, **Holm over the two N** (family size 2).
* **E2 (extension):** λ 128 vs λ 1, Holm over the two N, reported as a separate family.
* **T1, T2, T3 (trend, unadjusted):** λ 4 vs 1, λ 16 vs 4, λ 128 vs 16.
* **Decomposition (reported with every E):** median over seeds of log(Ī_F^ABF(λa)/Ī_F^ABF(λb)) and of
  log(Ī_F^FR(λa)/Ī_F^FR(λb)), with CIs.

**Resolution (stated now).** From the Exp I shifted-fibre per-seed L (sd 0.144 at N 512, 0.270 at N 128) and no
coupling benefit (Exp II: L uncorrelated across λ), 32 seeds detect R ≈ 0.87 (N 512) and R ≈ 0.78 (N 128) with 80 %
power at the first Holm step. Smaller effects will read as inconclusive.

### 1.3 Numerical validation (frozen; all before any production run)

The common timestep is chosen by this section alone, before any ABF/FR performance of Experiment A is read.
**Production starts only after the gate G-A1/G-A2 and all nine checks below pass. Any failure stops Experiment A for
investigation; it is reported, and any fix is a dated amendment written before production.**

**Candidate h_A = 2.5 × 10⁻⁵** (the validated mechanism-campaign h); refinement 1.25 × 10⁻⁵ for all four λ cells if
any new dynamics fails.

**G-A1. Timestep gate for the three new dynamics** shift κ = 1 at λ = 4, 16 and 128, using the frozen mechanism
harness (`src/gateway_family_validation.py`, `scripts/mechanism/run_validation.py`, `analyze_validation.py`) and its
frozen thresholds, unchanged: V1 (closed-form EM variance inflation ≤ 0.02 and measured ≤ 0.02 + 2 se at the two
central fine bins; shifted-fibre conditional-mean 3-se test; pooled |x| ≤ 0.3 variance within 2 %), V2 (D ≤ 0.00185,
95 % upper ≤ 0.0028 for F_density and F_MF), V3 (D ≤ 0.0115, upper ≤ 0.017), V3b (debiased relative RMS of
Var(f | x) ≤ 5 %), V4 (MALA at h 1e-4 and exact i.i.d. samples; an h-independent precondition), V5 (finite states,
wall reflections < 1e-6 per walker-step, every group's crossing fraction ≥ 0.5; ABF-bias smoke: 2 seeds, N 2048,
4 t.u., live ABF + FR, finite, final max|Γ| < 2 max|F\*′| on bins with ≥ 100 deposits).
* **Multiplicity.** All 3-se clauses (V1 shifted-fibre means at each h; V4 MALA and exact clauses) of the new dynamics
  are decided by two-sided p (t₁₅, 16-group jackknife) with Holm at family-wise 5 % — one family for V1 per h and one
  for V4 — so that noise alone cannot trigger refinement.
* **Harness extension, inert for the seven old dynamics.** New dynamics enter only through an explicit registry that
  is accepted only together with `--out results/final_campaign/expA/validation` (refused for the frozen
  `results/mechanism/validation`); defaults remain the seven old dynamics. They take global `DYNAMICS_ORDER` indices
  7, 8, 9 and new seed bases (§4). Run length T = 500 / min(λ, 1) t.u. (500 t.u. for every λ ≥ 1: x coverage
  limits V2/V3; the old rule 500/λ would cut λ = 16 to 31 t.u.); trace stride 0.05/(min(λ, 1) h); the conditional-ACF
  horizon of the new dynamics is ≥ 6 t.u. absolute, so MC1 compares the same cuts (0.3125, 1.25, 5 t.u.) and the 1/e
  time at every λ. **Inertness test (before any G-A1 run):** for each of the seven old dynamics a short chain run
  through the harness at HEAD 189c2408 and through the extended harness gives bitwise identical arrays; the
  settings, seed and ACF-plan functions return identical values for every old (dynamics, chain, h, group); the
  analyzer run on `results/mechanism/validation` reproduces `summary.json` byte for byte apart from its generation
  timestamp; `run_validation --dry-run` with defaults lists exactly the old jobs.

**G-A2. λ = 1.** Its gate record at h = 2.5e-5 (all clauses PASS, `results/mechanism/validation/summary.json`,
`dynamics.shift`) is part of the Experiment A gate. If refinement is triggered, λ = 1 is re-gated at 1.25e-5.

**Selection.** h_A = 2.5e-5 iff λ = 4, 16 and 128 all PASS V1, V2, V3, V3b and V5 and V4 holds. Otherwise all four
cells move to 1.25e-5, all four are gated there, and B = 2048 · 40 / h_A (same T_N). A failure at 1.25e-5 stops
Experiment A as UNRESOLVED.

**The nine mandatory checks** (each must pass before production):

| # | check | how | pass rule |
|---|---|---|---|
| 1 | analytic forces vs finite differences | unit test: ∂ₓV, ∂ᵧV of `grad_V` vs central differences at ≥ 200 (x, y) points over [−1.8, 1.8] × [m − 6σ, m + 6σ] (ε 1e-6); kernel-extracted deposit force = `grad_V` at every λ | relative error ≤ 1e-7 (FD) / ≤ 1e-12 (kernel) |
| 2 | F\*, F\*′ identical across λ | `build_references.py` on the four cells | primary arrays bitwise equal to each other and to `results/equal_budget_v2/references/gateway_reference.npz` |
| 3 | conditional law and Var(f \| x) λ-independent | [identity] §1.1; measured by G-A1 V1, V3b at λ = 4, 16, 128 (and the λ = 1 record) | gate PASS |
| 4 | transverse integrator vs exact | (a) G-A1 V1/V4 (exact i.i.d. and MALA references); (b) **frozen-x AR(1) test through the production engine** (`run_arm(..., _freeze_x_test=True)`, as `tests/test_gateway_family_numba.py::test_f4`): one x₀ = −0.21 (κ constant, so other x₀ are replicates of the same process), 64 chains started from the EM stationary law, 200 relaxation times (200/(λκ²) t.u.) each, at λ ∈ {1, 4, 16, 128} and h_A, plus a **resolvable-inflation case** at λ ∈ {1, 16} with h = 0.1/λ (φ = 0.9, EM excess 5.3 %); stationary variance and the ACF at lags ℓh ∈ {0.25, 0.5, 1, 2}/(λκ²), estimated with the known mean 0 and normalised by the known EM variance, se from the between-chain spread (t₆₃) | every variance and ACF point consistent with (1/(βκ²))/(1 − λκ²h/2) and φ^ℓ, φ = 1 − λκ²h: Holm over the 30 two-sided p in one family, FWER 5 %. This verifies the y integrator (drift and noise scaling with λ); it is **not** evidence about coupled relaxation, which is MC1 |
| 5 | timestep bias at λ = 16 (and 128) | G-A1 | gate PASS |
| 6 | ABF/FR pairing before the first birth–death | on the check-9 smoke runs (pre-production gate) and on all production runs (audit): all shared saved arrays bitwise equal between the arms up to the last save before the first realised FR death | every pair, each with ≥ 1 compared save (number recorded); a pair with none is checked by comparing the full state at the step of the first FR opportunity with a non-zero event in a dedicated short replay. Production: 256/256 pairs |
| 7 | independent noise and FR streams | existing engine tests (Langevin MT19937 vs FR PCG64 `SeedSequence([seed, N, 1])`), re-run; the FR arm's Langevin stream equals the ABF arm's | tests pass |
| 8 | old configurations unchanged | engine sha256 = `69087f2c…`; re-run 4 stored production jobs (Exp I shift N 128 seed 8100 abf/fr; Exp I α = 1 N 512 seed 8100 abf/fr) at HEAD; plus the harness inertness test above | bitwise equal on all shared arrays |
| 9 | smoke test of every new cell | smoke seeds (§4), all 8 cells × 2 arms, T = 4 t.u. | finite, result contract and diagnostic consistency pass (Σ age classes = C_all; D1 totals = deaths); check 6 passes; never interpreted |

### 1.4 Production design (frozen)

* Cells: λ ∈ {1, 4, 16, 128} × N ∈ {512, 128} × {ABF, ABF + uniform FR}; **32 fresh seeds 8300–8331** in every cell:
  4 × 2 × 2 × 32 = **512 runs** (384 for the task-specified λ ∈ {1, 4, 16}).
* h = h_A (§1.3). **B = 2048 · 40 / h** walker-steps per arm and seed = 3 276 800 000 at h = 2.5e-5. T_N = Bh/N = 160
  (N 512), 640 (N 128).
* Initial conditions coupled across λ and arms: x from `init_left(seed, N)`, y = m(x) + z/(√β κ) from the same
  standard normals z (`init_family`); λ does not enter. Langevin noise MT19937 seeded with
  `default_noise_seed(seed, N)` (independent of λ: coupled across λ); FR uniforms PCG64 `SeedSequence([seed, N, 1])`.
  ABF and FR arms share initial state and noise.
* Algorithm constants (frozen gateway values, never varied): 180 bins, own-bin P0 estimator, min_count 1, γ 1.5,
  η 0.1, score clip 3, FR interval 0.004 t.u., ramp 4 t.u., cap max(1, ⌊0.08 N⌋), genealogy window 1.6 t.u.
* Save grid `prereg_save_grid`; diagnostics D1–D4 on (as in the mechanism campaign).
* Reference: the common analytic F\*, F\*′ (bitwise the equal-budget gateway reference), the accepted gateway scorer
  (151-node eval window [−1.5, 1.5], profile centred). The secondary EM-consistent reference equals F\*′ exactly for
  the shifted fibre at every λ.
* Driver: new `scripts/final_campaign/` wrappers reusing `scripts/mechanism/run_cells.py` functions; outputs under
  `results/final_campaign/expA/`; per-run ledger with wall time. No old config, result, reference or synthesis file is
  written. Jobs run seed-major (all cells of a seed before the next seed), so a truncated production leaves complete
  coupled seed blocks.

### 1.5 Measurements (frozen)

**Primary:** Ī_F = (1/200) Σ_{k=1}^{200} e_F(u = k/200) — the integrated L² free-energy error of the existing pipeline
(`eqb_metrics.run_metrics`, rectangle rule over the 200 uniform budget fractions).

**Secondary** (existing analysis code, unchanged): (1) final e_F; (2) Ī_F′, final e_F′, and the floor-free Ī_F′_stat,
final e_F′_stat (floor 0.03227); (3) Ī_TV_half, final TV_half; (4) persistent τ at e_F 0.002 / 0.0055 / 0.02,
e_F′_stat 0.004 / 0.0136 / 0.0946 and TV_half 0.1, in u and t, with censoring and the exact sign test; (5) marginal
discovery (first arrival x > 0.5) and establishment (instantaneous and cumulative rules); (6) MC1 conditional
relaxation times; (7) age-dependent bias of deposits by clone age (D2 classes [0, 0.01), [0.01, 0.1), [0.1, 1),
≥ 1 t.u., never): debiased RMS B_c with the reflected seed-bootstrap CI, and the matched excess over ABF's deposits;
a descriptive decomposition of FR's gate-region excess bias into age-class contributions at each λ; (8) genealogy:
unique ancestors, windowed ancestor ESS/N — its time average = the mean of `gen_ess_win` over the 200 uniform-u saves
with t ≥ 1.6 t.u. — and its minimum, maximum windowed lineage fraction (`gen_maxfam_win`), realised deaths per walker
per t.u.; (9) cost per successful threshold for each (cell, N, arm, threshold): (a) Σ_seeds full-run cost /
#successes and (b) oracle stopping (Σ_successes τ·cost + Σ_failures full cost) / #successes, in walker-steps (exact)
and in core-seconds (τ_u × run wall time); "∞ (0 successes)" when none succeed.

### 1.6 Statistics (frozen)

* Within a cell: G = (FR − ABF)/ABF per seed; median, 10 000-resample seed-bootstrap 95 % CI
  (`default_rng(20261010)`), wins; mean-based companions reported.
* E1, E2 as in §1.2; T1–T3 unadjusted; the same construction on final e_F, Ī_F′_stat and Ī_TV_half as labelled
  secondary contrasts (unadjusted). The Holm-consistent CI for the first Holm step (97.5 %) is reported next to each
  primary 95 % CI; equivalence uses the 95 % CI.
* Equivalence of two cells' FR gains only if the 95 % CI of exp(Δ) lies in [0.90, 1.11].
* HA3 holds at (λ, N) iff the upper 95 % CI of G(Ī_TV_half) < 0; it "persists" iff this holds in every cell.
* Replication: the λ = 1 cells repeat the Exp I shifted-fibre cells on fresh seeds; reported side by side
  (descriptive, not tested).
* Cells with < 10 complete seed pairs make no claim.

### 1.7 Outcome classification (frozen)

Per N and per contrast family (E1 primary; E2 extension). Let R, p_Holm be the contrast's ratio and Holm-adjusted p,
and G_hi = the upper end of the 95 % CI of G(Ī_F) at the higher λ.

| label | rule | wording allowed |
|---|---|---|
| **A** strong rescue | p_Holm < 0.05, R < 1, G_hi < 0, and guards g1–g4 hold | "faster mobility rescues FR" |
| **A-qualified** | p_Holm < 0.05, R < 1, G_hi < 0, but a guard fails (named) | "FR better than ABF but with a new problem: gX" |
| **B** partial rescue | p_Holm < 0.05, R < 1, G_hi ≥ 0 | "reduces FR's disadvantage; FR not better than ABF" |
| **I** indeterminate | not (p_Holm < 0.05 and R < 1), but G_hi < 0 | "FR better than ABF at the higher λ; improvement over λ = 1 not supported" |
| **C-equivalent** | p_Holm ≥ 0.05, 95 % CI of R inside [0.90, 1.11], G_hi ≥ 0 | "no rescue" |
| **C-reverse** | p_Holm < 0.05, R > 1 | "faster mobility makes FR relatively worse" |
| **C-inconclusive** | every other case | "no detectable rescue; R bounded by its CI" — never "no rescue" |

Guards (at the higher λ, same N): **g1** the CI of G(final e_F) does not lie entirely above 0; **g2** the CI of
G(Ī_F′_stat) does not lie entirely above 0; **g3** genealogy: g3 fails iff the upper 95 % CI of the seed-median ratio
(higher λ / λ 1) of FR's time-averaged windowed ESS/N is < 0.8, or the lower 95 % CI of the ratio of FR's maximum
windowed lineage fraction is > 1.25; **g4** no non-finite state and no cap violation in any run.

**Combination across N (primary question).** "Faster mobility rescues FR" requires A, A-qualified or B at both N
(the weaker label governs the wording). "Rescue at N = n only" when one N is A/A-qualified/B and the other is
C-equivalent, C-inconclusive or I. Opposite significant signs (one A/B, one C-reverse) are reported as N-dependent,
with no global claim. "No rescue" globally only if both N are C-equivalent or C-reverse.

**Interpretation map** (decided now):
* Every outcome is reported together with MC1, MC2, the ABF-only descriptors and the E decomposition. Because λ
  changes transport (MC2) as well as relaxation (MC1), a change in R is attributed to **transverse mobility**; an
  attribution to conditional relaxation alone is a [hypothesis] in every branch.
* ABF improves strongly with λ (ABF Ī_F ratio CI < 1) while R changes → the change may come from ABF's own improved
  transport/establishment rather than from FR's clones; stated as a confound.
* A rescue (A/A-qualified/B) with MC3 passing → consistent with the clone fibre-lag hypothesis. A rescue without MC3
  passing → the rescue is not explained by reduced young-clone bias (transport or other); stated so.
* C with MC3 passing → young-clone bias falls but FR's relative performance does not improve: young-clone lag is not
  what limits FR here; candidate explanations (establishment share of ABF's error, entry-side over-population, ABF's
  own lag) named as hypotheses.
* C without MC3 passing → the intervention did not reach the mechanism at the tested λ, read against MC1 (flank
  speed-up 1.09× / 1.44× predicted at λ 4 / 16).
* E1 and E2 disagreeing (e.g. C at λ 16, rescue at λ 128) → consistent with flank relaxation (which changes
  materially only at λ ≫ m′²) mattering, with transport also changed: [hypothesis].
* Any F effect without a marginal effect points to the force estimate, not to establishment.

### 1.8 Figures and report (frozen)

`figures/final_campaign/expA/`: (A1) seed-median Ī_F of both arms vs λ, per N, with CIs; (A2) final e_F vs λ;
(A3) FR/ABF ratio exp(median L) vs λ with CIs and the E/T contrasts; (A4) conditional relaxation (MC1): ρ_y, ρ_f at
x₀ = −1, −0.21, 0 for every λ, with the frozen-x and coupled-OU predictions, and τ vs λ; (A4b) gate transport (MC2)
vs λ; (A5) young-clone bias B_c by clone-age class vs λ (FR) with ABF's deposits (MC3); (A6) marginal Ī_TV_half vs
λ and its G; (A7) median e_F(u) convergence curves, ABF vs FR, one panel per (λ, N); (A8) genealogy (ESS/N, maximum
lineage fraction, deaths per walker) vs λ; plus the standard per-cell A–F figures of `plot_config.py`. Every figure
is checked for legibility (log-ratio axes where single seeds would flatten a linear axis).

Report: **`docs/mechanism/EXPA_SHIFTED_FIBRE_MOBILITY_v1.md`** (header "Version 1, commit <hash>"; any revision is a
new `_v2` file). Seed-level tables (per seed, cell, arm: Ī_F, final e_F, G, τ) as CSV and markdown under
`results/final_campaign/expA/tables/`.

---

## 2. Experiment B — fair parallel wall-clock comparison

### 2.1 Question and the claim it can support

Can FR at large N beat the best ABF allocation in **wall-clock time to a persistent free-energy accuracy**, when both
are given the same hardware? The answer is a statement about **specified hardware and a specified implementation**,
distinct from the fixed-N and equal-force-evaluation claims (`EVIDENCE_LEDGER.md` §2).

### 2.2 Backend requirements (frozen)

A new walker-parallel shared-memory CPU backend under `src/parallel/` implementing the **same mathematical
algorithm** as the validated engines (`src/gateway_ladder_numba.py` = `src/gateway_family_numba.py` at α = 1, λ = 1,
and `src/lta_ladder_numba.py`):
* the same histogram ABF estimator and conventions (gateway: Γ_j = M_j/(C_j + 1), all deposits of a step before any
  bias read; LTA: per-step staged deposits into all-steps accumulators, ramp, ±60 bias clip, ±480 deposit clip,
  production accumulators from burn-in), shared by all walkers;
* the same FR operator, score, selection, cap, ramp and schedule, using the validated engines' FR functions and
  block-structured PCG64 FR uniform streams; the per-walker part of the LTA score may be evaluated in the parallel
  region into a per-slot array and summed serially in slot order (bitwise the serial result);
* the same initial conditions and accumulator conventions;
* synchronisation: every deposit of step k completes before any bias read of step k;
* **deterministic reduction order:** deposits are added to each bin in slot order, so results do not depend on the
  thread count;
* two noise modes: **replay** (the validated engine's exact serial Langevin stream: numba MT19937 for the gateway,
  PCG64 for LTA) and **counter** (Philox4x64-10 keyed by SeedSequence([seed, N, 2]) for the gateway and
  SeedSequence([seed, N, 150, 2]) for LTA, counter = (global step, slot, word, 0), Box–Muller normals), generated
  inside the parallel region;
* timing runs carry no passive diagnostics; saves record M_all, C_all (LTA also M_prod, C_prod), FR counters, a
  wall-clock stamp and the force-evaluation count; in-engine timers separate force/move, the ABF deposit + bias-read
  phase, synchronisation (barrier wait), FR and saves; JIT and thread-pool warm-up are timed separately and excluded;
* **implementation choice and recorded deviation.** The task asks to reuse the validated numba engines. For LTA the
  backend is a numba `parallel=True` kernel reusing `_mol_force`, `_fr_score` and `fr_select_native`. For the gateway
  the per-step dynamics and the ABF deposit/bias read are re-implemented in a C/OpenMP kernel (gcc -O3,
  `-fno-fast-math -ffp-contract=off`) called from a driver that runs the validated numba FR step, **because** this
  host's measured synchronisation costs (OpenMP barrier 0.5–1.1 µs at 8–16 threads; numba `prange` launch ≥ 3.8 µs at
  8 threads) are comparable to the gateway's whole per-step work (≈ 40 µs serial at N = 512). This deviation is
  licensed only by V-B1 (bitwise replay identity). If V-B1 cannot be met, the numba `parallel=True` variant is the
  fallback, recorded in an amendment before the feasibility benchmark. The backend commit is frozen before the first
  benchmark; every benchmark attempt is logged and the first complete attempt decides.

### 2.3 Backend validation (frozen; all must pass before the feasibility benchmark is interpreted; a failure stops Experiment B for that system)

* **V-B1 replay identity.** Replay mode reproduces the validated engine (diagnostics off) **bitwise** on every shared
  result array at thread counts {1, 4, 16}, both arms, for gateway N ∈ {16, 64, 512} and LTA 150 K N ∈ {2 (ABF only),
  16, 64, 512}. Run lengths: gateway ≥ 12 t.u. at N 16 and ≥ 6 t.u. at N 64 and 512; LTA ≥ 200 000 steps at N 16 and
  64, ≥ 60 000 at N 512, ≥ 40 000 at N 2 — each with ≥ 20 saves and, for FR arms, ≥ 50 realised deaths (counts
  reported). Fallback for LTA only: if identity fails solely because the validated engine's fastmath-reassociated
  LJ sum compiles differently inside the parallel kernel (shown by first-step per-walker forces agreeing to
  ≤ 1e-13 relative), V-B1 for LTA is met by identical integer outputs (bins, FR candidates and events) over the whole
  test run and max |Δq| ≤ 1e-8 Å; this is recorded as an amendment.
* **V-B2 thread-count invariance.** Counter mode gives bitwise identical results for thread counts {1, 2, 4, 8, 16} on
  the same configurations.
* **V-B3 RNG.** (i) Philox4x64-10 raw outputs equal numpy's `Philox` bit generator for ≥ 10⁴ counters × 4 keys
  (exact). (ii) The counter map (step, slot, word) is injective over every production (N, n_steps, words per
  walker-step) (exhaustive assertion). (iii) From 10⁸ counter-mode normals: mean, variance, skewness, excess kurtosis;
  lag-1 correlation across steps, across adjacent slots and across words; the correlation of the two Box–Muller
  partners; KS against N(0, 1) — each converted to a two-sided p, one Holm family (9 tests), FWER 1 %.
* **V-B4 statistical equivalence with the validated engine.** For each system and each arm in {core small-N ABF
  (gateway 16, LTA 2), ABF 512, FR 512}: counter mode at full budget on 32 fresh seeds vs the validated engine
  (gateway: the 32 equal-budget production runs 8100–8131; LTA: the 16 production runs 15000–15015 plus 16 new
  validated-engine runs on fresh labels). Pass iff the 95 % bootstrap CI of the ratio of median Ī_F
  (counter / validated) lies within **[0.85, 1.18]** for every arm, and the 95 % CI of the difference of median paired
  G(Ī_F) at N 512 (counter minus validated) lies within ±0.10. If a CI exceeds the band while its point estimate lies
  in [0.93, 1.075] (resp. ±0.05), one pre-specified extension adds 32 seeds per side and the test is re-evaluated once.
  The Exp III R1–R5 consistency battery is reported alongside as a descriptive non-rejection check, never cited as
  equivalence.
* **V-B5 pairing.** In counter mode the ABF and FR arms of one (seed, N) are bitwise identical before FR's first
  realised death.

### 2.4 Feasibility gate (frozen thresholds; decided before any benchmark)

**Hardware allocation: k = 16 physical cores** — two L3 groups (CCDs, 8 cores each) of one NUMA node, SMT siblings
left idle. Rationale: a modest, realistic share of a shared node, inside one memory domain; k = 8 and k = 32 are
reported as descriptive throughput points (benchmark only).

**Execution environment for every timed run (benchmark and production).** A run with t ≤ 8 threads gets one CCD to
itself; t = 16 gets two CCDs of one node; no other process of this campaign runs on those CCDs or their SMT siblings
during the run. CCDs are chosen at launch with all their cores and siblings ≤ 10 % busy over a 2 s sample; if none
qualifies the launcher waits (poll 30 s, up to 30 min), then uses the least-loaded qualifying-as-far-as-possible CCD
set and flags the run. A monitor process pinned outside all allocations samples /proc/stat at 1 Hz for every core
and sibling of each active run's CCDs, and `scaling_cur_freq`; each run records the foreign busy fraction on its
CCDs' cores and siblings and the involuntary context switches of its threads.

**Benchmark protocol.** Production configurations truncated to a fixed walker-step count (gateway 1.6384 × 10⁸ = 5 %
of B; LTA 3.072 × 10⁷ = 10 % of B; FR is active in both). Candidates: ABF at N ∈ {1, 2, 4, 8, 16, 32, 64, 128, 256,
512} and FR at N ∈ {64, 128, 256, 512}, each on the parallel engine (counter mode) at t ∈ {1, 2, 4, 8, 16} (t ≤ N)
and on the validated serial engine (diagnostics off, one core); 3 repeats in rotated order; median walker-steps per
second. Recorded: throughput, peak memory, and the split of wall time into force/move, ABF deposit + bias read,
synchronisation, FR and saves.

**Gate, per system (both required):**
* **F1 (absolute scaling):** for both N = 512 arms, the best parallel throughput over t ≤ 16 ≥ **6.0 ×** the
  validated serial engine's throughput at N = 512 on one core.
* **F1b (genuine thread scaling, same RNG):** for the FR 512 arm, the best parallel throughput over t ≤ 16 ≥
  **4.0 ×** the parallel engine's own t = 1 throughput. (The RNG-only speed-up, parallel t = 1 vs serial engine, is
  reported separately.)

A system that fails stops here: its production is not run, the measured numbers and the reason are reported, and no
production budget is spent on it.

### 2.5 Production design (frozen; only for systems that pass §2.3 and §2.4)

**Arms.** Core arms (task-specified): gateway ABF 16, ABF 64, ABF 512, ABF + FR 512; LTA 150 K ABF 2, ABF 64,
ABF 512, ABF + FR 512. **Competitor completion (mechanical, frozen):** for every benchmark candidate (method, N),
projected median wall time to the mid threshold W_proj = Q50_s[τ_mid,u(N, method, s)] × B × c\*(N, method), where
τ_mid,u are the equal-budget production per-seed persistent mid-threshold times (`results/equal_budget_v2/<system>/
analysis/summary.json`, censoring-aware median) and c\* is the best measured wall time per walker-step over the
candidate implementations. The ABF candidate with the smallest W_proj and the FR candidate with the smallest W_proj
are added as arms if they are not already core arms (at most one extra arm per method and system). The projection
and the added arms are recorded before production. This makes the decisive comparison "best FR allocation vs best
ABF allocation on this hardware" rather than "FR 512 vs three hand-picked ABF arms".

**Implementation per arm.** The fastest median throughput among {validated serial engine, parallel engine at
t ∈ {1, 2, 4, 8, 16}, t ≤ N}; within 3 % of the best counts as tied, and ties go to fewer cores (the serial engine
counts as one core and wins a tie with parallel t = 1). An arm on the serial engine runs its validated replay noise
(licensed by V-B4's small-N arm). Recorded before production.

| system | seeds | budget per run |
|---|---|---|
| gateway (α = 1, λ = 1, h 2.5e-5) | **8400–8463** (64) | B = 3 276 800 000 walker-steps |
| LTA 150 K (h 2e-4) | **15200–15231** (32) | B = 307 200 000 walker-steps |

* Frozen production physics, timestep, references, scorer and algorithm constants of the equal-budget campaign;
  diagnostics off.
* **Save grid (Experiment B only, identical in u for every arm):** the 200 uniform fractions u = k/200 ∪ log-spaced u
  with ratio 1.02 from 10⁻⁴ to 1 (≈ 466 points); steps = clip(round(u n_steps), 1, n_steps), unique. Ī_F uses the 200
  uniform saves as before.
* Same seed labels across all arms; **only ABF 512 and FR 512 are paired** (shared initial state and noise); cross-N
  arms are independent.
* **Run order:** seed-major blocks; within a seed the arm order is permuted by `default_rng(20261011 + seed)`; runs
  are dispatched in that order to free CCD sets. A fixed drift reference (gateway FR 512, t = 16, 2 % of B) is run every
  30 min and reported.
* **Contention:** a run is contended for an endpoint if, over the endpoint's wall window ([0, W] for a W endpoint, the
  whole run for Ī_F^wall), any core or sibling of its CCDs shows foreign busy fraction > 0.10, or its threads incur
  > 5 involuntary context switches per thread-second. Counter mode is deterministic, so a contended run is repeated
  (same science, new timing) until an uncontended replicate exists, at most 3 repeats; all records are kept; the
  primary analysis uses the first uncontended replicate; runs with none are flagged, excluded from the primary
  analysis and included in a sensitivity analysis.
* **Cost accounting:** used core-hours = t_arm × wall (the resource ceiling, §3); allocated core-hours = 16 × wall
  (every arm is notionally given the full allocation) reported alongside.

### 2.6 Endpoints (frozen)

**Primary:** W_mid = wall-clock seconds from the timed start to the first save of the persistent window
(e_F ≤ threshold at that save and every later save of the run) at the frozen **mid** threshold — gateway 0.0055,
LTA 150 K 0.13 kJ/mol. A run that never persistently reaches it within its budget is **censored** (W = ∞); no
time-to-accuracy is imputed. Sensitivity: W interpolated log-linearly in e_F between the last failing save and the
first persistent save.

**Secondary:**
1. W at the strict and loose thresholds (gateway 0.002 / 0.02; LTA 0.03 / 0.5 kJ/mol). A contrast at any threshold
   is labelled "not comparable (budget artefact)" if a censored arm's completion wall time is shorter than the
   comparator's W;
2. integrated free-energy error against wall-clock, Ī_F^wall(W\*) = (1/W\*) ∫₀^{W\*} e_F(w) dw, with e_F(w) the
   error of the latest save at or before w (before the first save: the error of the zero-bias flat profile; after a
   run completes: its final error); W\* = the smallest of the arms' median completion wall times over uncontended
   runs, fixed at its observed value before the bootstrap; full curves shown;
3. force evaluations to accuracy (N × completed steps at the threshold save);
4. CPU core-hours to accuracy, used (t_arm × W) and allocated (16 × W);
5. effective parallelism: serial-equivalent speed-up = validated-serial wall per walker-step × walker-steps / wall,
   and parallel efficiency = speed-up / t_arm;
6. per-step ABF overhead (deposit + bias read), synchronisation and FR overhead from the in-engine timers, and FR's
   paired added wall time at N = 512 with ABF 512 and FR 512 at a common thread count (FR 512's t_arm) as well as at
   their own;
7. fraction of runs reaching each threshold;
8. seed-bootstrap 95 % CIs for every quantity (10 000 resamples, `default_rng(20261010)`).

### 2.7 Statistics and decision rule (frozen)

* One new estimator module (`scripts/final_campaign/expB_stats.py`), unit-tested on censored fixtures before any
  production data exist. `boot_median_ci`, `paired_contrast` and the Exp III `W1_RULE` are **not** used for any W
  quantity (they drop non-finite values).
* Per-arm medians are ∞-aware (`quantile_inf` on the full vector including ∞).
* Bootstrap: 10 000 resamples, `default_rng(20261010)`; seeds resampled independently per arm, except that ABF 512
  and FR 512 share one index draw.
* Ratio ρ(f, a) = median W_f / median W_a with the conventions finite/∞ = 0, ∞/finite = ∞, ∞/∞ = undefined; a contrast
  with > 2.5 % undefined resamples is "unresolved"; undefined resamples are counted, never dropped silently.
* **Decision (per system, mid threshold).** Let 𝔽 be the FR arms and 𝔸 the ABF arms (core + completion).
  **FR advantage** iff there is an f ∈ 𝔽 such that, for every a ∈ 𝔸, the upper end of the two-sided
  (1 − 0.05/|𝔽|) percentile CI of ρ(f, a) is < 1 (intersection–union over 𝔸, Bonferroni over 𝔽).
  **ABF advantage** iff there is an a ∈ 𝔸 such that, for every f ∈ 𝔽, the lower end of the (1 − 0.05/|𝔸|) CI of
  ρ(f, a) is > 1. Otherwise **unresolved**, with the CIs as bounds. The two advantages cannot both hold.
* Descriptive: ρ_best = min_f median W_f / min_a median W_a, re-selecting both sides in every resample, with its
  percentile CI and the fraction of resamples with ρ_best < 1 (not a probability).
* Cross-N contrasts are unpaired (independent bootstraps; Gehan–Wilcoxon on W with ∞ ranked last, reported); the
  paired sign test is used only for ABF 512 vs FR 512.
* **Resolution (stated now).** From resampling the equal-budget per-seed τ_mid at fixed throughput: at 64 gateway
  seeds a ratio of medians has a 95 % CI of about ×/÷1.3, and at 32 LTA seeds about ×/÷1.22. Decisions are therefore
  resolvable only for ρ outside ≈ [0.77, 1.3] (gateway) and ≈ [0.82, 1.22] (LTA); closer values read as unresolved.
* A model-based projection to other core counts (measured throughput curves × equal-budget τ distributions) may be
  reported only as a labelled projection, never as a measurement.

### 2.8 Figures and reports (frozen)

`figures/final_campaign/expB/`: (B1) throughput vs threads per (system, N, method), with the F1/F1b lines and the
serial engine; (B2) wall time per walker-step vs N; (B3) W_mid per arm (seed points, medians, CIs, censored marked);
(B4) e_F vs wall-clock median curves per arm per system; (B5) ρ bootstrap distributions with the decision bands;
(B6) cost (force evaluations, used and allocated core-hours to accuracy); (B7) overhead split (force/move, ABF,
synchronisation, FR, saves). Reports: `docs/final_campaign/EXPB_PARALLEL_BACKEND.md` (implementation, validation,
feasibility) and `docs/final_campaign/EXPB_WALLCLOCK.md` (production, if licensed). Seed-level tables under
`results/final_campaign/expB/tables/`.

---

## 3. Compute resources, ceilings and stopping rules (frozen)

Host at the freeze (measured, timestamp in `EXECUTION_LOG.md`): 2 × AMD EPYC 9554 (128 physical cores, SMT-2,
siblings c and c + 128; 16 L3 groups of 8 cores), 1.5 TB RAM, load average ≈ 20–24 from other users' long jobs
(≈ 22–30 busy cores, unpinned and migrating; often no fully idle CCD), cpufreq governor schedutil with boost (per-core
1.25–3.9 GHz), **disk 100 % full with ≈ 6 GB free and falling**, GPUs 0–2 busy with other users' jobs (GPU 3 idle at
the freeze; unused by this campaign).

* **Ceiling for the whole campaign: 250 CPU core-hours, counted as used core-hours** (threads × wall, including
  validation, smoke, tests, benchmarks, production, repeats and failed attempts), tracked in
  `results/final_campaign/compute_ledger.csv`; allocated core-hours for Experiment B are reported but not counted
  against the ceiling (idle cores of an allocation remain available to other users). Experiment A ≤ 110 (≈ 13
  validation + ≈ 47 production at h 2.5e-5; ≈ 100 with refinement); Experiment B ≤ 110 (≈ 25 validation and
  equivalence incl. extension, ≈ 6 benchmark, ≈ 60–75 production with up to two completion arms); slack 30.
  Experiment B production wall time ≤ 12 h.
* **Projection gates:** before each production stage the stage cost is projected from measured throughput × budget ×
  seeds × arms; if it exceeds the remaining stage cap, the stage is not launched and the reason is reported.
* **0 GPU-hours.** No GPU work is authorised: the previous campaign measured the existing torch engines slower than
  one CPU core at N = 512, and no new GPU feasibility argument exists.
* At most 64 concurrent physical cores of this campaign; SMT siblings of used cores left idle; no other user's
  process is ever signalled or reniced.
* Disk: ≤ 2 GB of new data in total. The root filesystem is shared and was being filled by other users at up to
  ≈ 8 GB/h at writing (14 → 6 GB free in one hour; `/data1`, `/data2` are not writable for this account). Jobs of
  this campaign **pause** (do not start new runs or saves) while free space is below 2 GB and resume above 3 GB;
  the pause is logged. Raw arrays are stored compressed.
* All jobs resumable (completed runs skipped by signature; long jobs checkpoint).
* **Stopping rules:** (i) Experiment A — a failed check or gate (§1.3) not resolved by a pre-production amendment, a
  failed gate at both candidate h, or a numerical instability; (ii) Experiment B, per system — a V-B1/V-B2/V-B3/V-B4
  failure, or a failed feasibility gate; (iii) the resource ceiling or a projection gate; (iv) a disk pause that does not clear within 24 h. An
  incomplete stage is reported with its actual status; nothing is dropped silently.

## 4. Seeds and pairing (frozen)

| use | seeds | notes |
|---|---|---|
| Exp A production | 8300–8331 | every cell; coupled across λ and arms |
| Exp A cell smoke (check 9, check 6) | 8390, 8391 | never production |
| Exp A ABF-bias smoke (V5) | 8392, 8393 | N 2048, 4 t.u. |
| Exp A frozen-x AR(1) test (check 4b) | 8650–8661 | engine runs; one label per (λ, case) |
| Exp A validation chains, d′ = DYNAMICS_ORDER index − 7 ∈ {0, 1, 2} (λ 4, 16, 128), k = index in the fixed list (2.5e-5, 1.25e-5) | em_flat 400000 + 1000d′ + 100k + g; MALA 410000 + 100d′ + g; em_unbiased 411000 + 100d′ + 10k; exact_flat 420000 + 100d′ + g; exact_unbiased 421000 + 100d′ + g | g = group 0–15; raw seeds seed numba MT and the x₀ `default_rng` |
| Exp B gateway production | 8400–8463 | all arms |
| Exp B gateway equivalence (V-B4) | counter 8500–8531; extension counter 8532–8563, extension validated 8700–8731 | vs production 8100–8131 |
| Exp B LTA 150 K production | 15200–15231 | all arms |
| Exp B LTA 150 K equivalence (V-B4) | counter 15300–15331; new validated 15340–15355; extension counter 15360–15391, extension validated 15460–15491 | vs production 15000–15015 + 15340–15355 |
| Exp B benchmarks, bitwise tests, drift reference | 8600–8609, 15400–15409 | never scored |
| Exp B counter-mode Philox keys | gateway SeedSequence([seed, N, 2]); LTA SeedSequence([seed, N, 150, 2]) | no existing stream uses third element 2 |

Disjointness from every previously used gateway and LTA seed label, the absence of gateway noise-seed collisions
((1 000 003 · seed + 7919 · N) mod (2³¹ − 1)) with any MT seed recorded in the stored gateway run and validation files,
raw-seed disjointness of the new validation chains, and the absence of duplicate SeedSequence states over all FR /
LTA / IC / counter streams (old and new) are checked by `scripts/final_campaign/check_seeds.py`; its PASS output is
committed with this file (`results/final_campaign/seed_check.json`). [identity] For 0 < |ΔN| ≤ 4095 the gateway noise
map is injective on any label window narrower than 418 747, so distinct (seed, N) pairs in these blocks cannot share a
noise seed.

## 5. Reproducibility

* Commit hashes of the preregistration, each implementation, each amendment and each result set are logged in
  `docs/final_campaign/EXECUTION_LOG.md` with measured (not estimated) UTC times.
* Every run file records engine sha256, git commit, seed, h, precision (float64), device (CPU), thread count, CPU set
  and library versions; raw arrays are gitignored and listed with sha256 in `results/final_campaign/DATA_MANIFEST.json`;
  summaries, ledgers, tables and figures are committed.
* Frozen v1 artifacts, historical results and historical analysis scripts are never modified.

## 6. Execution order

Preregistration commit → Experiment A tooling + tests (incl. harness inertness) → checks 1, 2, 4b, 7, 8 → G-A1/G-A2
gate → smoke (checks 6, 9) → Experiment A production → analysis and report → Experiment B backend, tests and review →
V-B1…V-B5 → feasibility gate → competitor completion and thread rule recorded → Experiment B production (systems that
pass) → analysis → final synthesis.

## 7. Final synthesis (frozen structure)

`docs/final_campaign/FINAL_SYNTHESIS.md`, one section per task question, each answer typed [identity] / [observed] /
[hypothesis] and tied to one of the four claim types: (Q1) does FR help when an already-discovered region is
under-established; (Q2) does that benefit require fast conditional relaxation; (Q3) is marginal improvement alone
predictive of free-energy improvement; (Q4) does the shifted-fibre mobility intervention rescue FR; (Q5) does FR beat
the best ABF allocation at equal force evaluations; (Q6) does genuine hardware parallelism change that conclusion;
(Q7) what the corrected WCA negative results imply for generality. Plus: a table of WCA and every negative or neutral
control (alanine, valine, small-N cells, shifted fibre, λ = 0.1) next to the positives; the five final-decision items
of the task (publishability as a conditional finite-time accelerator; surviving claims; computational advantage vs
fixed-N acceleration only; stop or continue; publication figure and table set); and `docs/final_campaign/
PUBLICATION_READINESS.md`.

---
*Amendments (dated, appended below; nothing above is edited after the freeze commit).*
