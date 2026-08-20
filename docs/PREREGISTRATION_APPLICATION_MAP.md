# PREREGISTRATION — Applicability map (post-campaign study)

## CAMPAIGN FROZEN 2026-08-20 (tag `v1-application-map-final`)

Published as branch `abp-fisher-rao` of `github.com/zheyuanlai/ABF-Fisher-Rao`.  That
repository hosts the closed mFR-ABF campaign on `main`; the two branches share a remote
and **no history** (see `docs/PROVENANCE.md`), so nothing here merges into that campaign
and its recorded outcomes are untouched.

Phases A, B, D, ALA, F, I and J are closed with their outcomes recorded below, and the
campaign-level synthesis is the speed map at the end of this file.  The narrative
account is `docs/TECHNICAL_REPORT.md`.  Phase G (descriptor-dimension scaling) and
Phase H (reaction-law comparison) are **designed but deliberately not run**: both would
refine a step with no demonstrated benefit in either the bias-limited or the
variance-limited regime, and that decision is recorded rather than left implicit.  A
weighted-population study of rare-event fluxes or rates would be a different observable
and belongs in a new project with its own preregistration.

Nothing below this line is edited.  Two *interpretations* were narrowed after review;
both are marked in place as dated scope corrections next to the claim they narrow, and
no recorded number changed.

Status: **Phase A + Phase B designs FROZEN 2026-08-18** (this commit; before any
Phase-A/Phase-B run exists). Later phases (C+) are sketched with their gates; their
numerical designs get their own dated freeze commits before their runs.

## What this study is (and is not)

This is a **new post-campaign applicability study**, not a revision of the closed
gateway or WCA preregistrations. The recorded outcomes of
`docs/PREREGISTRATION_GATEWAY.md` (accuracy PASS dI_F = -11.4%, speed target missed,
count balancing ties FR, persistent FR overdamped) and `docs/PREREGISTRATION_WCA.md`
(all cells SHUS-sufficient at K = 1024; preregistered null; no WCA FR run) are frozen
and are **not** edited, rescored, or re-litigated here.

The campaign question changes from "does FR beat SHUS at K = 1024?" to:

> **When, at realistic computational budgets, does marginal Fisher-Rao population
> reallocation add useful information to an adaptive-biasing (ABP) simulation?**

Four sub-questions, in priority order:

* **Q1 (confound):** Does the gateway FR win survive a better-tuned plain-SHUS
  baseline, or was FR merely compensating for an overly aggressive adaptation rate?
* **Q2 (resources):** Does an establishment-limited regime appear on WCA at smaller
  replica counts K, so that FR could substitute for walkers / force evaluations?
* **Q3 (geometry):** Does fine FR ever beat coarse count balancing (needs >= 2D CV;
  later phase).
* **Q4 (cloning):** How does hidden-coordinate decorrelation (tau_clone) limit the
  usefulness of cloning (later phase).

Interpretation discipline (frozen): we do not tune systems until FR wins; plain-ABP
diagnostics decide whether an FR experiment is warranted; a null on any question is a
recordable outcome, not a failure to be repaired.

## The adaptation-gain parameter g_SHUS

New engine parameter (this commit): a dimensionless per-arm multiplier `g_shus > 0`
on the SHUS accumulator increment,

    R_{n+1} = R_n + g_shus * (dt / K) * mollify(block deposits),

followed by the usual `R <- R / max(R)` gauge step. `g_shus = 1` is the frozen
historical implementation (bitwise-identical output; unit-tested). Because the gauge
renormalization and the deposit weights both scale linearly in R, multiplying the
increment is gauge-compatible for any per-row constant; the fixed-point *shape*
`R* = K_eps e^{-beta F}` is unchanged (g only rescales the approach rate); the
estimator sign convention and the estimator-protection invariant are untouched
(unit tests accompany this commit). `g_shus` lives on the `Method` dataclass so arms
with different gains share physics, initial conditions, and Langevin noise within one
paired batch.

---

## Phase A — gateway adaptation-gain study (Q1)   [FROZEN 2026-08-18]

Cell: **anchor_D** (beta = 16, H = 0.5, s = 0.10, r = 32), all frozen gateway
conventions unchanged: eps_bw = 0.02, grid n = 361, eta_bw = 0.10, block = 20,
K = 1024, dt = 2e-4, n_steps = 500_000 (T = 100), n_saves = 400.

### A1 — plain-SHUS gain screen (no FR anywhere)

* Arms: plain SHUS at `g_shus in {0.25, 0.5, 0.75, 1.0, 1.5}`, one paired batch
  (all gains share each seed's noise realization).
* Seeds: **200..215** (16 fresh seeds; disjoint from calibration 0-7, pilot 8-15,
  production 100-131).
* Recorded per row: e_F(t), I_F, e_F(T), D_t, a_1(t) sign changes after T_hit,
  ring-out time (last t with e_F > 2 e*, e* = analytic mollifier floor), occupancy
  overshoot max P_+(t), T_hit, T_est (same D_tol construction as Stage 1), wall-clock.
* **g_best selection rule (frozen):** among gains whose median paired
  e_F(T) <= 1.05 x that of g = 1.0, pick the one with the lowest median paired I_F.
  If two qualifying gains are within 2% relative median I_F of each other, pick the
  one closer to 1.0. If no gain other than 1.0 qualifies, g_best = 1.0.

### A1 outcome (2026-08-18, seeds 200-215 — recorded, not to be edited)

Median paired dI_F vs g = 1.0: g 0.25: +85.5% [83.8, 88.5] (T_est censored);
g 0.5: +30.8% (eT ratio 1.27 — NOT qualified); g 0.75: +10.9% (eT ratio 0.83);
g 1.5: **-9.4% [-10.0, -8.6]** (eT ratio 0.96), with earlier discovery (T_hit 3.75
vs 4.5), earlier establishment (T_est 28.5 vs 38.5) and smaller occupancy overshoot
(0.660 vs 0.682). **The "SHUS too aggressive" hypothesis is refuted in its original
form: slowing adaptation hurts; speeding it up recovers most of the FR-sized gain.**
g_best by the frozen rule = 1.5 — at the BOUNDARY of the frozen grid.

### A1b — grid extension (amendment, FROZEN 2026-08-18 before any A1b run)

Because g_best landed on the grid boundary, A2 against g = 1.5 could understate what
gain tuning alone achieves and bias Q1 toward FR. Extension, decided before any A1b
run exists: same cell, same seeds 200-215, same batch_seed 20260824 — the engine's
noise stream is method-independent, so a new batch with arms
`g_shus in {2.0, 3.0}` is EXACTLY noise-paired with the stored A1 rows; paired
statistics are computed against the stored g = 1.0 records. Same qualifying and
selection rules, applied over the union of A1 and A1b gains. If g = 3.0 is the
argmin AND improves the median paired dI_F by more than 2 percentage points over
g = 2.0, one final extension {4.5, 6.0} runs under the same rules; no extension
beyond 6.0 regardless. A2 then uses the overall g_best.

### A1b outcome (2026-08-18 — recorded, not to be edited)

g = 2.0: dI_F -14.0% but eT ratio 1.249 (NOT qualified); g = 3.0: dI_F -17.1% but
eT ratio 1.198 (NOT qualified), ringing grows monotonically with gain (sign flips
4 -> 5 -> 7). Gain tuning above 1.5 is a pure speed/accuracy TRADEOFF: more
integrated-error reduction only by sacrificing final accuracy — the structure the
frozen FR winner does not have (Stage 3: -11.4% I_F WITH eT ratio 0.837).
**Overall g_best = 1.5 (interior point; boundary concern resolved). A2 proceeds.**

### A2 — best-tuned SHUS vs the frozen FR winner

* Seeds: **300..315** (16 fresh matched seeds). One paired batch, arms:
  1. `shus` — historical baseline (g = 1);
  2. `shus_gbest` — plain SHUS at g_best from A1 (if g_best = 1.0 this arm is
     omitted and Q1 is answered "tuning does not help" directly from A1);
  3. `fr_temp` — the frozen Stage-3 winner, UNRETUNED: theta = 0.01, stride = 10
     blocks, window [6, 14], g = 1;
  4. `count` — 9-bin count balancing, same theta/stride/window, g = 1;
  5. `sham` — matched-turnover sham shadowing fr_temp.
* Primary endpoint (frozen): median paired relative change of I_F vs the `shus`
  baseline for each arm, with paired bootstrap 95% CI; plus the direct paired
  contrast `I_F(fr_temp) vs I_F(shus_gbest)`.
* **Interpretation rule (frozen):**
  * If `shus_gbest` matches or beats `fr_temp` (median paired
    I_F(gbest) <= I_F(fr_temp), CI of the difference not favoring fr_temp) with
    healthy diversity (trivially true: no resampling): record
    **"FR is not practically necessary on the gateway; the win was compensating
    the adaptation rate."**
  * If `fr_temp` still improves over `shus_gbest` by >= 5% median paired I_F with
    CI < 0: record **"the population correction supplies something adaptation-rate
    tuning does not reproduce."**
  * Between those: record the point estimate and CI without a headline claim.
* Secondary: e_F(T) ratios, uncensored time-to-accuracy ladder at e0/{2,4,8}
  (e0 = 0.236 frozen), D_t, a_1 ring metrics, frozen-bias validation of final biases
  (same protocol as Stage 3) if the primary result is a near-tie (<5% separation),
  since then the online-estimator artifact question matters most.

### A2 outcome (2026-08-18, seeds 300-315 — recorded, not to be edited)

* dI_F vs shus(g=1): shus_gbest(1.5) -10.7% [-12.1, -9.7]; fr_temp -11.9%
  [-12.4, -10.5]; count -12.1% [-13.0, -11.4]; sham -0.1% (inert, replicated).
* **Frozen primary contrast fr_temp vs shus_gbest: -0.9% [-3.5, 0.6] — TIE.
  Verdict by the frozen rule: no headline claim; point estimate recorded.**
* Online final error: fr_temp/count eT ratios 0.883/0.895 vs gbest 0.963 (modest
  FR edge); gbest's speed ladder is erratic (S = 1.27/0.99/1.17) vs fr_temp's
  uniform modest 1.10/1.05/1.04.
* Frozen-bias: the in-run 60k endpoint suggested a large FR edge (0.762 vs 0.962),
  but the 200k equilibrated rescoring (same Stage-3 protocol) shows it was
  relaxation-limited: all non-sham arms collapse to l2 ~ 0.003 with fr_temp vs
  gbest = 0.997 [0.729, 1.311] (tie); fr_temp vs shus = 0.919 [0.843, 0.993]
  (small, significant); gbest vs shus = 0.813 [0.679, 1.205] (n.s., wide).
* Reading (within the frozen vocabulary): on this 1D gateway a one-parameter
  adaptation-rate increase reproduces the FR-sized I_F gain; temporary FR achieves
  the same net effect through a different mechanism (an 8%-of-run population
  intervention with uniform modest speedups and slightly better online final
  error) rather than a globally faster adaptation law. Count balancing ties FR for
  the third time — the fine FR geometry remains non-specific in 1D. Q1's practical
  answer: **FR is not necessary on the gateway at K = 1024; it is one of (at
  least) two equally effective ways to buy the same integrated-error improvement —
  the open FR-specific questions move to resource scaling (Q2), dimensionality
  (Q3), and clone decorrelation (Q4).**

## Phase B — WCA population-size / resource-scaling map (Q2)   [FROZEN 2026-08-18]

New study; the closed Stage-4 null at K = 1024 is untouched and reused as this map's
K = 1024 anchor (same seeds 0-7, same protocol, batch_seed 20260821).

* Cells: **b2h6** (primary stress cell, largest barrier ~12 kT), **b1h2** (easy
  control). b4h1 only if the first two make it informative (own freeze first).
* K ladder: **{32, 64, 128, 256}** new runs + the frozen K = 1024 Stage-4 anchor.
* Seeds 0..7 per (cell, K); plain SHUS only; all Stage-4 conventions unchanged
  (dt = 2e-3, n_steps = 250_000, T = 500, block = 20, eps_bw = 0.025, eta_bw = 0.07,
  n_saves = 400, ess_window_steps = 4000; float32 dynamics / float64 estimator).
* Gate metrics per row: T_hit (stretched region, hold 0.05), T_est (trailing-median
  D_t rule, hold 0.10), gap G = T_est - T_hit, D_t, region fractions, support
  fraction. **D_tol(K) = 1.5 x (KL* + noise95(K))**: only the finite-K KDE noise
  component changes with K, computed analytically by the identical
  `kde_noise_floor(K, ...)` — preregistered here, per the closed campaigns' gate
  construction. KL* per cell as in Stage 4 (hp_v3 for b1h2; phase-tier proxy
  amplitude for b2h6, gate use only, never scoring).
* **Eligibility rule (frozen, same as the closed campaigns):** a (cell, K) is
  eligible for a population-reallocation experiment only if median T_hit/T <= 0.2
  AND median (T_est - T_hit)/T >= 0.25.
* **Outcomes (frozen vocabulary):** per (cell, K): discovery-limited /
  establishment-limited / SHUS-sufficient / intermediate, using Stage-1's
  classification thresholds on hit and gap fractions. If NO K is eligible: record
  the **resource-scaling null** ("SHUS passes from discovery-limited to
  SHUS-sufficient with no establishment window") and Q2 closes negative — FR gets
  no WCA run, and the campaign proceeds to the 2D geometry question (Q3).

### Phase-B outcome (2026-08-18, seeds 0-7 — recorded, not to be edited)

**Complete resource-scaling NULL.** Every (cell, K) for K in {32, 64, 128, 256,
1024} on both b1h2 and b2h6 is SHUS-sufficient: median T_hit/T <= 0.008, median
T_est/T = 0.000 (zero censored seeds), gap <= 0 — across D_tol(K) values spanning
0.009 to 0.33, so this is not a tolerance artifact. The hypothesized three-regime
structure (discovery-limited -> establishment-limited -> SHUS-sufficient) DOES NOT
EXIST on WCA: SHUS's occupancy flooding performs the search per-walker (the bias
drives escape deterministically), so the population size never becomes the binding
constraint. Accuracy degrades gracefully and statistics-limited with K (b1h2
median e_F(T): 0.157 at K=1024 -> 0.225 at K=32; D_T rises accordingly): fewer
walkers cost estimator NOISE, not population placement — which reallocation
cannot repair by construction. **Q2 closes negative: no eligible K, no WCA FR run
(preregistered rule); FR has no resource-efficiency window on this system class.
The campaign proceeds to the 2D geometry question (Q3, Phase D).**

## Later phases (gated, designs to be frozen when reached)

* **Phase C:** if some (cell, K*) is establishment-limited: plain-SHUS gain screen
  at K* FIRST (g in {0.5, 0.75, 1.0, 1.25}); FR only if the establishment deficit
  survives gain tuning. FR transfer: theta = 0.01, stride 10 unretuned; window from
  the K* SHUS-only quantiles (t_on >= Q90(T_hit), t_off < Q50(T_est)). Arms:
  tuned SHUS / +FR / +count / +sham. Efficiency accounting in force evaluations
  C = K x n_steps x N_particles; headline comparison "FR at K vs plain SHUS at 2K".
  Before any accuracy claim: an occupancy-consistent reference (umbrella + WHAM/MBAR
  under the production integrator, dt, clip) — hp_v3 stays labeled
  constrained-mean-force and is never the accuracy referee for SHUS.
* **Clone-decorrelation diagnostics** (tau_clone; sibling correlation in xi and an
  orthogonal solvent coordination descriptor vs matched independent pairs) become
  mandatory instrumentation for ANY molecular FR run in this study.
* **Phase D (Q3):** 2D periodic SHUS+FR engine, analytic torus validation, then
  (phi, psi) alanine dipeptide; plain-SHUS K screen before any FR; FR vs
  {6x6, 9x9, 12x12} count balancing.

  **D1 — torus plain-SHUS screen [FROZEN 2026-08-18, before any D1 run]:**
  engine commit 8814431 (89 tests green). Surface
  V = H1(1-cos 2phi)/2 + H2(1-cos 2psi)/2 + Hc cos(phi)cos(psi): four basins,
  axis barriers H1/H2, depth split 2 Hc between the deep and shallow pairs.
  Cells: t_easy (beta=1, H=0.8, Hc=0.2; control), t_mid (beta=4, H=1.5, Hc=0.5;
  4 kT barrier, 4 kT split), t_cold (beta=8, H=1.0, Hc=0.25; 6 kT barrier),
  t_anchor (beta=8, H=1.5, Hc=0.75; 6 kT barrier, 12 kT split). Frozen numerics:
  grid 72x72, eps_bw = 0.06 (analytic floors: beta e* <= 0.20 kT and
  KL* <= 0.019 on every cell — the Stage-1 calibration logic applied
  analytically; 0.10 would put a 0.56 kT floor under t_anchor), eta_bw = 0.25,
  K = 1024, dt = 1e-3, n_steps = 200_000 (T = 200), block = 20, n_saves = 400,
  profile_every = 8, seeds 0..7, init: all walkers in the deep b00 basin.
  Gates (same construction as all closed campaigns): T_hit = first persistent
  time (hold 0.05) ALL FOUR basins hold >= 1 walker; T_est = trailing-median
  KL rule (hold 0.10) with D_tol = 1.5 x (KL*(cell) + noise95_2D(K)); classify
  with the frozen vocabulary; FR eligibility = median T_hit/T <= 0.2 AND
  median gap/T >= 0.25. A 2D FR/count experiment (D2+) runs ONLY on an
  establishment-limited cell, with its own design freeze first.
  **D1 outcome (2026-08-18, seeds 0-7 — recorded, not to be edited):** the torus
  map shows all three regimes, unlike WCA. t_easy SHUS-sufficient (gap 0.07);
  **t_mid establishment-limited and ELIGIBLE (hit 0.038, est 0.804, gap 0.77,
  zero censored)**; t_cold establishment-limited but unresolved within T = 200
  (8/8 censored T_est — reserved, needs longer T); t_anchor discovery-limited
  (hit 0.246: the 12 kT depth split blocks discovery — the regime reallocation
  cannot help by construction, kept as a negative-control cell). **t_mid is the
  2D anchor; D2 proceeds on it.**

  **D2 — FR vs count-balancing resolution on t_mid [FROZEN 2026-08-18, before
  any 2D FR run]:** fresh seeds 400..415, one paired batch, T = 200, frozen D1
  numerics. FR transfer, unretuned: theta = 0.01, stride = 10 blocks. Window
  from the frozen derivation rule on D1 SHUS-only data (t_on = ceil(Q90(T_hit))
  = 14; t_off = t_on + 0.25 (Q50(T_est) - t_on) = 50 — the gateway winner's
  25%-of-interval convention). Arms:
  1. shus (baseline); 2. fr_temp (fine periodic KDE); 3. count6 (6x6);
  4. count9 (9x9); 5. count12 (12x12); 6. sham (shadows fr_temp);
  7. shus_g1.5 (gateway-transferred gain, exploratory secondary — the Phase-A
  lesson demands the tuning comparison be visible in 2D too).
  Primary endpoint: median paired dI_F vs shus with bootstrap CIs, plus the
  direct paired contrasts fr_temp vs each count resolution. Q3 reading rules
  (frozen): count ties FR at ALL resolutions -> the fine geometry is not the
  active ingredient in 2D either; FR beats count only at the sparse resolutions
  (12x12 or 12x12+9x9) while count6 ties -> the advantage is RESOLUTION, i.e.
  smooth density estimation, not FR structure per se (record as such); FR beats
  ALL count resolutions including 6x6 -> first evidence the smooth FR
  reallocation itself matters. Ancestry floors as always (min windowed
  ESS_anc/K >= 0.5, final n_anc/K >= 0.5) — an arm that violates them is
  reported but not interpreted as a win. Engine note: events2d generalizes
  coarse_nb to per-row values so all three resolutions share one paired batch;
  the 1D events.py of the closed campaigns is untouched.

  **D2 outcome (2026-08-18, seeds 400-415 — recorded, not to be edited):**
  * **Q3 answer by the frozen rule: count ties FR at ALL resolutions** —
    fr_temp vs count6/count9/count12 = +0.1/+0.0/+0.1% (CIs within ±0.2%);
    the fine Fisher-Rao geometry is not the active ingredient in 2D either
    (fourth replication of FR ~ count across two campaigns and two systems).
  * Stronger: ALL reallocation arms are inert-to-slightly-harmful on this
    establishment-limited cell (fr_temp +0.3% [0.3, 0.4] vs shus; sham 0.0;
    ancestry floors passed, no theta backoff, ~180 events fired — the same
    dose that bought -11.4% on the gateway). Population placement is not the
    binding constraint: T_est is unchanged (161.5 vs 161.8).
  * Exploratory secondary: **shus_g1.5 dominates: dI_F = -29.3% [-29.4, -29.3],
    e_F(T) ratio 0.427, T_est 104.5 vs 161.8, no resampling.** The torus
    establishment deficit is an ADAPTATION-RATE deficit, which reallocation
    cannot repair and gain tuning directly does.
  * Emerging applicability map: temporary FR helps when the establishment
    transient is an UNDERDAMPED OSCILLATION (gateway anchor_D: FR ~ gain
    tuning ~ -10%); it is inert when the transient is slow-adaptation-limited
    (torus t_mid: gain wins -29%, FR +0.3%); nothing is needed when SHUS
    floods immediately (WCA, all K). Follow-ups for a next session: t_mid
    adaptation-gain screen (is g > 1.5 better here? does FR-on-top-of-g_best
    add anything?), t_cold with longer T, and the Q4 clone-decorrelation
    instrumentation before any solvated system.

  **D3 — closing experiments on t_mid [FROZEN 2026-08-18, before any D3 run]:**
  the user-approved final campaign: three jobs, then stop opening application
  branches and take stock.

  * **D3a — t_mid gain curve.** Seeds 400-415, batch_seed 20260828 (exactly
    noise-paired with the stored D2 rows via the method-independent noise
    stream). New arms g_shus in {2.0, 3.0, 4.0}; g = 1.0 and 1.5 come from the
    stored D2 records. Pareto/qualifying rule as in A1: qualifying = median
    paired e_F(T) ratio vs g = 1 <= 1.05; g* = lowest median paired I_F among
    qualifying; gains within 2 points resolve toward SMALLER g. If g = 4 is the
    argmin and beats g = 3 by > 2 points, one extension {6, 8}; hard cap 8.
    Also recorded per gain: T_hit, T_est (needed for the D3b window).
  * **D3b — the decisive test: tuned SHUS vs tuned SHUS + FR.** Fresh seeds
    500-515, five arms: shus_g1 (anchor), shus_gstar (baseline), gstar+fr_temp
    (theta = 0.01, stride 10, transferred unretuned), gstar+count9,
    gstar+sham (shadows gstar+fr_temp). Window derived from D3a's g* rows by
    the frozen rule t_on = ceil(Q90(T_hit)), t_off = t_on +
    0.25 (Q50(T_est) - t_on). **Decision rule (frozen): median paired dI_F of
    gstar+FR vs gstar >= -2% or CI straddles 0 -> "FR adds nothing on top of a
    tuned base ABP on t_mid" and the SHUS-FR question on this cell is CLOSED;
    <= -5% with CI < 0 AND beating its sham -> FR has independent value on top
    of tuning (Outcome 1); between: point estimate, no headline.**
  * **D4 — t_cold honest classification.** Seeds 0-7, T = 800
    (n_steps = 800_000), arms shus + shus_g1.5 (secondary), same gates with the
    same D_tol; classify with the frozen vocabulary AND the new mechanism
    taxonomy: establishment deficit = (A) population-oscillatory (overshoot /
    ring in basin occupancies and D_t after T_hit), (B) adaptation-rate-limited
    (monotone slow flooding; the g arm removes it), (C) hidden-coordinate,
    (D) statistical noise. Only a Type-A residual after gain tuning would
    justify a future reallocation arm (own freeze first).
  * **Q4a — first clone-decorrelation measurement (WCA solvent).** Purpose:
    instrument tau_clone before ANY solvated FR claim; WCA runs remain
    FR-free (the Q2 closure stands — this is measurement, not intervention).
    Cells b1h2 + b2h6, K = 256, seeds 0-3. Protocol: plain SHUS to t0 = 100
    (dt 2e-3), freeze the learned bias, select parents stratified over xi in
    [0, 1] (8 bins x 8 parents), duplicate each parent into two children,
    evolve all children under the frozen bias with independent noise to
    lag 100. Descriptors: xi (dimer extension) and the orthogonal solvent
    coordination n_coord(q) = sum_j s(min_bead |q_j - q_bead|),
    s(r) = 0.5 (1 - tanh((r - r_c)/w_s)), r_c = 1.6 sigma, w_s = 0.1 sigma
    (first-shell count). Decorrelation measure (frozen):
    m(tau) = 1 - d_sib(tau)/d_ind(tau), with d = RMS pair difference and the
    independent baseline built from same-bin parent pairs;
    tau_clone = first tau with m <= 1/e. Report tau_clone^(xi) and
    tau_clone^(perp) per cell, against the FR event stride (0.2 t) and the
    establishment scales.

  **D3 outcome (2026-08-18 — recorded, not to be edited):**
  * D3a gain curve (noise-paired with D2; seeds 400-415): I_F falls
    MONOTONICALLY through the whole grid — g 1.5/2/3/4/6/8 give
    -29.3/-44.3/-58.8/-65.6/-71.5/-73.7% with eT ratios 0.43-0.72 (ALL improve
    final error; the 1.05 guard never binds). T_est falls 162 -> 13.8.
    g* = 8 at the preregistered hard cap: the historical SHUS adaptation rate
    is simply far too slow on this cell; the "tuned" regime is an order of
    magnitude faster than the frozen default.
  * D3b (fresh seeds 500-515; g* = 8 baseline; window [5, 7] by the frozen
    derivation): **gstar+FR = -0.1% [-0.5, 0.4]; gstar+count9 = -0.1%;
    sham +0.2% (inert); anchor shus_g1 = +282.6% vs the tuned baseline.**
    Ancestry floors passed, FR fired (~10 events, turnover 22). **Frozen
    verdict: FR adds nothing on top of a tuned base ABP on t_mid — the
    SHUS-FR question is CLOSED for this cell (the user's Outcome 2/3:
    apparent reallocation opportunities on this cell were artifacts of a
    non-optimally tuned adaptive method).**

  **D4 outcome (2026-08-19, seeds 0-7, T = 800 — recorded, not to be edited):**
  t_cold resolves honestly at longer T: T_hit/T = 0.024, T_est/T = 0.223 with
  ZERO censored seeds (the T = 200 screen had merely truncated a ~160-unit
  establishment interval, the same absolute scale as t_mid's). Gap/T = 0.199 —
  **below the 0.25 eligibility bar: t_cold is NOT establishment-limited at an
  honest horizon.** Mechanism: the g = 1.5 arm removes most of the remaining
  deficit (T_est 178 -> 106, paired dI_F = -24.0% [-24.1, -23.6]) — Type B
  (adaptation-rate-limited) again, with a mild final-error trade (eT ratio
  1.07). No Type-A oscillatory residual emerged; per the frozen rule no
  reallocation arm is justified on t_cold.

  **Closing assessment of the three-job campaign (2026-08-19):** the user's
  predefined Outcome 2/3 is realized. Across gateway, WCA (all K), and both
  torus cells: (i) every establishment deficit large enough to matter was
  either absent (WCA) or predominantly adaptation-rate-limited (torus; and on
  the gateway, gain tuning matched FR); (ii) FR never beat count balancing in
  four attempts spanning 1D and 2D CVs at three histogram resolutions;
  (iii) FR added nothing (-0.1%) on top of a properly tuned base ABP where it
  was tested head-to-head; (iv) clone decorrelation is not the limiter on the
  one solvated system measured (tau_clone < one event stride). The empirical
  practical hierarchy stands: tune the base adaptive method first; diagnose
  the deficit type; reach for directed reallocation only for a genuine
  population-oscillatory residual, and there simple count balancing suffices
  at low CV dimension. The remaining open door for FR-specific value is a
  molecular benchmark with coupled CVs (alanine, Stages A-E) and, for the
  cloning question, a solvated system with genuinely slow hidden coordinates. on both WCA cells
  (b1h2, b2h6; K = 256, frozen bias at t0 = 100, 64 stratified parents each),
  m(tau) collapses from 1 to noise around 0 by the FIRST recorded lag:
  **tau_clone < 0.2 t (resolution-limited upper bound) for BOTH xi and the
  solvent-coordination descriptor** — i.e. within one FR event stride. The
  independent baseline is meaningful (conditional n_coord spread 1.4-2.0), so
  this is a real measurement, not an artifact. Clone descendants become
  effectively independent almost immediately in this 2D solvated system: the
  covariance-defect / clone-redundancy mechanism is NOT what limited FR on WCA
  (that was the absent establishment gap, Q2). The instrument is validated and
  carries to any future solvated benchmark; a system with slow hidden-variable
  relaxation (3D water wetting/dewetting, coordination shells) remains the
  place tau_clone could bind.

* **Phase E (Q4):** one hidden-solvent stress benchmark chosen for a clean
  orthogonal descriptor, not for FR's expected success. Next system after D3/D4/
  Q4a: vacuum alanine dipeptide (port from the closed ABF-Fisher-Rao campaign),
  run as a diagnostic benchmark with the staged plan: A (xi = phi, plain-SHUS
  K-map + conditional psi diagnostics), B (xi = (phi, psi) 2D K-map), C (gain
  tuning wherever slow), D (FR/count/sham only in a residual Type-A regime),
  E (explicit solvent only if vacuum leaves a live question). Occupancy-based
  references under the identical model; the mechanism taxonomy (A/B/C/D)
  replaces the bare establishment-gap gate everywhere from here on.

## Phase ALA — alanine dipeptide diagnostic benchmark

**Engine (commit e290ecc, 104 tests green):** frozen physical model ported from
the closed campaign — ff14SB vacuum TorchFF built from the cached parameter
artifact (param_hash 6ffd00dc241f, verified vs OpenMM at extraction; parity
fixtures stored), BAOAB (dt = 1 fs, gamma = 1 ps^-1, T = 300 K, float64, no
constraints), IUPAC (phi, psi). Reference = the campaign's accepted
umbrella+MBAR occupancy FES; the engine grid IS the reference's 97-cell-centred
torus lattice (no resampling); eval masks mask8 / mask1 (F <= 8 kT); basins =
watershed from the 4 deepest reference minima (C7eq / C5 / C7ax / alphaR).
Every walker starts at the minimised C7eq structure with fresh Maxwell momenta.
FR clones copy (q, cached f) and draw fresh Maxwell momenta (the validated
full-state cloning). The question is NOT "can FR win" but **"what type of
limitation (A/B/C/D) does a realistic molecular ABP encounter after proper
tuning?"**

**Engineering note (2026-08-19, commit 82400f1, before any completed ALA-1
run):** the first launch showed the classic launch-bound profile (99% util at
156/600 W, 2.7 GB used — thousands of tiny kernels). The engine was rebuilt:
analytic forces and dihedral gradients replacing the three per-step autograd
graphs (validated to ~1e-15 relative against autograd, incl. the extreme
parity fixture), CUDA-graph capture/replay of each 20-step adaptation block
(noise pre-drawn eagerly per block from the unchanged generator sequence),
and torch deterministic algorithms so GPU reruns are bitwise reproducible.
87 -> 381 steps/s (4.4x; 633 without order-stable scatters — determinism
chosen). One instrumentation deviation: saves are now aligned to adaptation-
block boundaries (~400 saves preserved); physics, seeds, and noise streams
unchanged. Identical-parameter arms pair bitwise on CPU; on CUDA they share
the noise stream but can diverge from last-bit reduction-order effects —
the regime every GPU campaign in this project operated in.

**ALA-1 — joint (K x g) plain-SHUS screen [FROZEN 2026-08-19, before any run]:**
* Runs: cv in {phi, phipsi} x K in {32, 128, 512}, each batch carrying
  g_shus in {0.5, 1, 2, 4, 8} as five noise-paired arms (the tune-first rule is
  built into the screen; rows are launch-bound-free), seeds 0..7, NO FR
  anywhere. n_steps = 500_000 (0.5 ns), block = 20, n_saves = 400,
  profile_every = 8, ess_window_steps = 4000.
* Frozen numerics: eps_bw = 0.08 (analytic mask8 floor e* = 0.33 kT,
  KL* = 0.357 — recorded here; the 37 kT surface makes the mollified fixed
  point genuinely non-uniform and D_tol must carry that), eta_bw = 0.25.
* Gates: T_hit = all four basins persistently occupied (hold 0.05); T_est =
  trailing-median KL rule (hold 0.10), D_tol = 1.5 x (KL*(cv) + noise95(cv, K))
  with the analytic KL* of the mollified fixed point on the reference surface
  and the finite-K KDE noise floor, both per cv dimensionality.
* Per (cv, K): Pareto g* by the frozen rule (qualifying = median paired e_F(T)
  ratio vs g = 1 <= 1.05; lowest median paired I_F; within 2 points -> smaller
  g); if g = 8 is argmin and beats g = 4 by > 2 points, one extension {12, 16},
  hard cap 16. Classification with the frozen vocabulary AND the mechanism
  taxonomy (does gain remove the deficit -> Type B; ringing survives ->
  Type A candidate).
* cv = phi additionally records the conditional diagnostic
  E_cond(t) = int p_t(phi) TV(p_t(psi|phi), p_ref(psi|phi)) dphi and the full
  joint (phi, psi) KDE series: the Type-C readout is a flat phi-marginal
  (D_t at floor) with E_cond staying high. Marginal FR is NOT expected to
  repair Type C; that outcome closes the incomplete-CV question negatively.
* FR (with count6/9/12 + sham controls, theta = 0.01 stride 10 transferred,
  window by the frozen quantile rule, on the TUNED baseline, fresh seeds,
  -5%-with-CI standard) runs ONLY on a (cv, K) cell that retains a residual
  Type-A establishment deficit after gain tuning — its design gets its own
  dated freeze first. tau_clone^(psi) instrumentation (Q4a protocol with the
  hidden psi as the orthogonal descriptor) precedes any such FR run.

**ALA-1 corrections before the outcome (2026-08-19, recorded):** (i) the basin
seeds are selected by topographic PROMINENCE, not raw depth — the MBAR
surface's roughness put three sub-minima within 0.1 kT inside C7eq, so a
depth-sorted top-4 seeded one well three times and missed C7ax entirely; the
prominence rule recovers the documented minima (C7eq, C5, C7ax + the psi-bridge
at (-82, 137)). Basin labels are diagnostics-only, so the completed runs were
rescored from their stored joint KDEs (no rerun). (ii) The E_cond metric's own
floor — the SAME eta = 0.25 KDE applied to PERFECT sampling — is 0.180
(finite-K: 0.185/0.181 at K = 128/512), preregistered here as the reference
point before interpretation.

**ALA-1 outcome (2026-08-19, seeds 0-7 — recorded, not to be edited):**
* **cv = phi (psi hidden): SHUS-sufficient at every K and every gain**
  (hit/T <= 0.09, est/T <= 0.11, zero censored). E_cond sits AT the metric
  floor for K >= 128 (0.183-0.195 at g = 1 vs floor 0.180; K = 32 mildly
  elevated, 0.26): **no Type-C conditional deficit — psi | phi relaxes fast
  enough in vacuum at 300 K even with only phi biased.** Gain tuning HURTS
  in 1D beyond g ~ 2 (K = 32: +32% I_F and eT ratio 1.53 at g = 8); the
  default adaptation rate is roughly right.
* **cv = (phi, psi): SHUS-sufficient at K in {128, 512}** (hit/T <= 0.10,
  est/T <= 0.19); K = 32 is partially discovery-limited at 0.5 ns (small
  population vs the 15.8 kT C7ax barrier — Type D/discovery, which
  reallocation cannot repair by construction). Gain helps I_F monotonically
  in 2D (to -33.3% at K = 512, g = 8, eT ratio 0.96); Pareto g* = 2/2/4 at
  K = 32/128/512. Final errors ~0.36-0.40 kT, near the eps = 0.08 mollifier
  floor (0.33 kT).
* **No (cv, K) cell is establishment-limited => by the frozen gate, NO FR
  run on alanine.** The first real molecular benchmark reproduces the WCA
  pattern: a properly-run SHUS leaves no population-reallocation window; the
  residual inefficiencies are adaptation-rate (Type B, 2D) or small-K
  discovery/statistics (Type D), and the hidden-coordinate channel (Type C)
  is absent in vacuum. The FR-opportunity question on alanine closes
  negative without an FR run ever being justified — the campaign's terminal
  assessment (Outcome 2/3) now spans gateway, WCA (all K), both torus
  cells, and atomistic alanine under both CV choices. Solvated alanine
  (Stage E) remains the one live candidate for Type C / slow-tau_clone
  physics, to be opened only as its own preregistered study.

## Stopping / interpretation rules (frozen)

* Tuned plain SHUS matches FR on the gateway => "FR is not practically necessary
  for SHUS" on 1D CVs; the geometry question (Q3) still proceeds once, then stop.
* FR ties count balancing in 1D AND 2D => "directed population balancing matters,
  not the fine FR geometry."
* FR helps only at small K => "FR is a resource-efficiency device"; report the
  walker-equivalence factor.
* No molecular system ever satisfies the establishment criterion under well-tuned
  SHUS => "FR is practically redundant for SHUS in typical molecular applications"
  — an acceptable terminal conclusion.

---

# Phase F — the conditional (fiber-wise) Fisher-Rao question

Status: **F1 design FROZEN 2026-08-19** (this commit; before any F1 run exists).
F2 (the reallocation experiment) is gated on F1 and gets its own dated freeze.

## Why the campaign reopens after a terminal assessment

The closing assessment above stands unedited: across gateway, WCA at every K, both
torus cells and atomistic alanine under both CV choices, MARGINAL Fisher-Rao added
nothing that adaptation-rate tuning or coarse count balancing did not already
supply.  This phase does not re-litigate any of that.  It asks a question those runs
could not reach, and it is opened because of a structural reading of why they nulled,
not because of a hope that one more system will behave differently.

**The lemma the four count-ties were reporting.** `fisher_rao.py` scores walkers with
`a_k = [u(xi_k)/p_hat(xi_k)]^theta` and `events*.py` count balancing scores them with
the same expression evaluated on a histogram instead of a KDE.  Both are particle
realizations of the SAME continuous flow `d_t p = -p (log p/u - KL(p||u))`; they can
differ only through the estimator of `log p(xi)`.  At K = 1024 on a 1D or 2D CV with
6-12 bins, both estimators are excellent, so **a tie is the correct answer, not an
accident** — gateway, D2 at three resolutions, and D3b were four measurements of one
identity.  It follows that marginal FR can only separate from count balancing where
the density estimator binds, i.e. in reallocation descriptor spaces of dimension >= 3
where a histogram's cells outnumber the walkers.  That is Phase G's question, not
this one, and it is recorded here so the four ties stop being read as a disappointment.

**The regime the ABP cannot reach.** Phase B named the mechanism for the nulls: SHUS
floods, so escape along xi is deterministic per walker and the bias performs the
reallocation in xi itself.  Any population method whose target is a function of xi
alone is therefore redundant with the base method — which is exactly what every
recorded outcome says.  The complement is a theorem-shaped statement: the ABP owns
the xi-marginal, and the one thing it structurally cannot flatten is `p(z | xi)` for a
coordinate z it does not bias.  That is the mechanism taxonomy's **Type C**, and it is
the only live regime for a population correction under an ABP.  Vacuum alanine was
the campaign's one Type-C probe and it came back negative (`psi | phi` relaxes at the
metric floor), so the question was never actually tested — only its absence recorded.

**Why marginal FR could never have answered it.** Two walkers at the same xi in
different hidden channels receive the SAME marginal score, so no marginal reallocation
— FR or count — can prefer the under-populated one.  The instrument this campaign has
been running is blind to the only deficit its base method leaves open.  Phase F
therefore changes the instrument, not the system class:

    p^+(z | xi) propto p(z | xi)^{1-theta} u(z)^theta,      p(xi) LEFT INVARIANT

(`src/abpfr/fisher_rao_cond.py`, `src/abpfr/events_cond.py`, committed with this
freeze; 19 validation tests, whole suite 123 green).  Weights come from the joint KDE
as `[u(z_k)/p_hat(z_k | xi_k)]^theta`; systematic resampling runs INSIDE equal-width
xi-strata, so every stratum keeps exactly its walker count.  Two properties follow
that the marginal step does not have:

* the xi-histogram SHUS deposits from is invariant at stratum resolution, so the
  estimator-resampling feedback this project has fought since Stage 0 cannot enter
  through the deposit signal (tested: a backed-off conditional arm is bitwise the
  plain run);
* **`g_shus` is structurally disqualified as the competing explanation.**  The gain
  rescales the xi-bias; on a cell whose deficit lives behind a barrier in z it can do
  nothing by construction.  For the first time in this campaign the arm that matched
  or beat FR everywhere cannot be the answer.

Target convention transferred unchanged: u is UNIFORM on z.  It is NOT the stationary
conditional (nothing biases z), so the step is temporary-window only and the realized
channel populations are recorded at every save so overshoot is measured, not assumed
absent.  One asymmetric cell is carried precisely because the uniform target is
knowingly wrong there.

## The system (`src/abpfr/systems/bichannel.py`, committed with this freeze)

Overdamped Langevin on the torus, CV = phi, hidden coordinate = psi:

    s(psi) = (1 + cos psi)/2
    V = Hperp (1 - cos 2 psi)/2 + Delta (1 - cos psi)/2
        + s Ha (1 - cos 2 phi)/2 + (1 - s) Hb (1 + cos 2 phi)/2

Two channels (psi ~ 0 and psi ~ pi) span the whole CV range; `Ha = Hb` makes them
images of each other under `phi -> phi + pi/2`, so channel B's phi-wells sit exactly
on channel A's barriers, `Z_A = Z_B`, and the channel ratio is exactly
`e^{-beta Delta}` (tested).  The reference is exact: the CV is a coordinate, so
`F(phi, psi) = V` and `F(phi) = -beta^{-1} log int e^{-beta V} dpsi` by quadrature on
the production grid — no reference simulation to confound an accuracy claim.  At
`psi = +-pi/2` the phi-dependence of V is constant when `Ha = Hb`, so **the hidden
barrier is exactly orthogonal to the CV**: no bias on phi lowers it.

Pre-run design quantities (`type_c_amplitude`, `analytic_floors`,
`conditional_floors`), recorded here before any screen row:

| cell (beta = 4, Ha = Hb = 1) | p_B_ref | p_B_ref (biased) | e_F if B never populated | e* (mollifier floor) | E_chan floor q95 |
|---|---|---|---|---|---|
| Hperp 1.0, Delta 0   | 0.500 | 0.500 | 0.335 | 0.0032 | 0.071 |
| Hperp 1.5, Delta 0   | 0.500 | 0.500 | 0.351 | 0.0034 | 0.068 |
| Hperp 2.0, Delta 0   | 0.500 | 0.500 | 0.358 | 0.0035 | 0.069 |
| Hperp 2.5, Delta 0   | 0.500 | 0.500 | 0.361 | 0.0035 | 0.070 |
| Hperp 1.5, Delta 0.5 | 0.129 | 0.316 | 0.181 | 0.0034 | 0.066 |
| Hperp 2.0, Delta 0.5 | 0.126 | 0.316 | 0.183 | 0.0035 | 0.066 |

The Type-C error stands ~100x above the estimator's own floor.  Under a converged
bias the stationary law is `p_ref(psi|phi) x uniform(phi)`, so the channel fraction a
converged run should show is the UNIFORM-phi average `p_B_ref_biased`, not the
Boltzmann average; both are recorded and the metric floors use the biased null.

**Metric honesty.** `E_cond` — the campaign's ALA-1 conditional metric — is a
KDE-vs-KDE total variation whose finite-K floor on this system (~0.15 at K = 1024) is
as large as the whole deficit it would measure.  It is recorded but is NOT a gate
here.  The primary conditional readout is the channel-resolved
`E_chan = int p(phi) |P_B(phi) - P_B_ref(phi)| dphi` (floor ~0.07), and the primary
accuracy endpoint is `e_F` on `F(phi)` (floor 0.0034).

**Engineering calibration (disclosed, 2026-08-19, before this freeze).** A 2-seed,
T = 200 pilot on cells (Hperp, Delta) in {(1,0), (1.5,0), (2,0), (1.5,0.5)} with arms
g in {1, 4} chose the cell grid below and measured throughput (1854 steps/s at 16
rows).  It showed the channel population rising 0 -> 0.49 / 0.38 / 0.13 by T = 200 at
Hperp = 1.0 / 1.5 / 2.0 with e_F(T) = 0.015 / 0.136 / 0.308, and g = 4 changing
nothing (0.130 vs 0.136 at Hperp = 1.5).  It is calibration, not an outcome: no F1
row existed when this design was frozen, and F1 re-measures everything at 8 seeds and
a 4x longer horizon.

## F1 — plain-SHUS Type-C screen (NO reallocation anywhere)   [FROZEN 2026-08-19]

* Cells: `Hperp in {1.0, 1.5, 2.0, 2.5}` at `Delta = 0`, plus `Hperp in {1.5, 2.0}`
  at `Delta = 0.5`; `beta = 4`, `Ha = Hb = 1`.  Six cells.
* Seeds **0..7**; arms `g_shus in {1, 2, 4, 8}` as four noise-paired arms
  (the tune-first rule built into the screen).  192 rows, one batch,
  batch_seed 20260950.
* Frozen numerics: `K = 1024`, `dt = 1e-3`, `n_steps = 800_000` (**T = 800**, the
  honest horizon — the D4 lesson that a truncated screen mislabels), `block = 20`,
  `eps_bw = 0.06`, `eta_bw = 0.25`, `n_saves = 400`, `profile_every = 8`,
  `joint_every = 40`, `ess_window_steps = 4000`, `n_strata = 32`, `init = chanA`
  (every walker starts in channel A: the hidden channel must be REACHED).
* Recorded per row: `e_F(t)`, `I_F`, `e_F(T)`; `D_t` = KL(p_hat(phi) || u) with
  `D_tol = 1.5 x (KL* + noise95(K))`; `T_hit^B` = first persistent time (hold 0.05)
  with `P_B > 0`; `T_est` from `D_t` (the MARGINAL establishment time — the ABP's own
  convergence verdict); `E_chan(t)`, `E_cond(t)`, `P_B(t)`, `P_B(phi)` profiles,
  deposition diagnostics, wall clock.

### Type-C eligibility (frozen, decided on plain-SHUS rows only)

A cell is eligible for the F2 reallocation experiment iff ALL of:

1. **the deficit is in the free energy:** median `e_F(T) >= 10 e*`;
2. **the hidden channel is reached and populated:** min over seeds of `P_B` at
   `t = T/4` is `> 0.01` — otherwise the cell is Type D (discovery) and no
   reallocation can help by construction;
3. **adaptation rate does not repair it:** the best gain's median `e_F(T)` is still
   `>= 10 e*` (Type B excluded — the gain arm is in the same batch, noise-paired);
4. **it is attributable to the channel:** median `E_chan(T) >= 2 x` its biased
   finite-K floor;
5. **it is still live at T:** median `P_B` over the last 10% of the run is
   `< 0.8 x p_B_ref_biased` (a cell that has already equilibrated has nothing left
   to accelerate).

If NO cell is eligible, record the **Type-C null** and the conditional-FR branch
closes without an FR run, exactly as Q2 closed on WCA.

### Predictions, recorded before the run so they can fail

* **P1** gain tuning does not repair the deficit on any Type-C-eligible cell.
* **P2** marginal FR will be inert there (`|dI_F| < 2%`) — it is blind by construction,
  and it is carried in F2 as a control precisely so that blindness is measured.
* **P3** conditional FR reduces `I_F` by `>= 20%` on an eligible cell.
* **P4** conditional COUNT balancing will TIE conditional FR at this descriptor
  dimension (one hidden coordinate).  This is stated up front: F2 tests the
  **geometry** claim (reallocation must be conditioned on the coordinate that limits
  the estimator), not the FR-vs-histogram claim, which by the lemma above needs a
  >= 3D descriptor and belongs to Phase G.  A P4 tie is a confirmation of the
  campaign's own lemma, not a new disappointment.

## F2 — the reallocation experiment (gated; own freeze before any FR row)

Runs only on an F1-eligible cell.  Fresh seeds 600-615, one paired batch, arms:
1. `shus_gstar` (tuned baseline, g* from F1);
2. `gstar + fr_cond` (theta = 0.01, stride 10 blocks, transferred UNRETUNED);
3. `gstar + cnt_cond` (stratified count control, resolution frozen in F2);
4. `gstar + fr_marg` (marginal FR, the blindness control);
5. `gstar + sham_cond` (stratified matched-turnover sham).
Window by the frozen quantile rule on F1's g* rows, with the channel establishment
time in place of the marginal one; its exact form is fixed in the F2 freeze because
F1 must first show whether `T_est^chan` is censored.  Ancestry floors as always
(min windowed `ESS_anc/K >= 0.5`, final `n_anc/K >= 0.5`); an arm that violates them
is reported and not interpreted as a win.  `tau_clone^(psi)` (the validated Q4a
instrument, hidden psi as the orthogonal descriptor) is measured on the eligible cell
BEFORE any F2 accuracy claim: in a Type-C system it is long by construction, which is
both why cloning is needed and the ceiling on what it can buy.

**Decision rules (frozen now):**
* `fr_cond` vs `gstar`: median paired `dI_F <= -5%` with CI < 0 **and** beating its
  sham -> **the conditional population correction has independent value on a Type-C
  deficit**; `>= -2%` or CI straddling 0 -> conditional FR adds nothing either, and
  the population-correction idea closes negative across every geometry this project
  can construct.
* `fr_cond` vs `cnt_cond`: tie -> record "conditioning on the right coordinate is the
  active ingredient; the FR estimator is not, at this descriptor dimension"
  (predicted).  `fr_cond` better with CI < 0 -> first evidence the smooth FR estimator
  itself matters, and Phase G's dimensional claim gets a head start.
* `fr_marg` non-null -> the blindness argument above is WRONG and this freeze says so;
  the whole Phase-F rationale would have to be rewritten in that outcome's light.

### F1 outcome (2026-08-19, seeds 0-7, T = 800 — recorded, not to be edited)

**Three of six cells are Type-C eligible under the frozen gate**, and the campaign's
first ABP deficit that adaptation-rate tuning cannot touch:

| cell | e_F(T) at g=1 | / e* | P_B(T) vs ref | E_chan / floor | e_F across g in {1,2,4,8} | eligible |
|---|---|---|---|---|---|---|
| Hperp 1.0, Delta 0   | 0.021 | 6.5 | 0.495 / 0.500 | 0.82 | 0.021 -> 0.032 | no (equilibrates) |
| Hperp 1.5, Delta 0   | 0.018 | 5.3 | 0.494 / 0.500 | 0.77 | 0.018 -> 0.029 | no (equilibrates) |
| **Hperp 2.0, Delta 0**   | **0.171** | **49.3** | 0.335 / 0.500 | 2.33 | 0.171 -> 0.174 | **YES** |
| **Hperp 2.5, Delta 0**   | **0.327** | **93.1** | 0.096 / 0.500 | 5.80 | 0.327 -> 0.329 | **YES** |
| Hperp 1.5, Delta 0.5 | 0.016 | 4.6 | 0.312 / 0.316 | 0.83 | 0.016 -> 0.028 | no (equilibrates) |
| **Hperp 2.0, Delta 0.5** | **0.105** | **30.2** | 0.175 / 0.316 | 2.26 | 0.105 -> 0.103 | **YES** |

* **P1 confirmed, decisively.** On every eligible cell an eightfold increase in the
  adaptation gain moves e_F(T) by <= 3%, and where it moves it at all it moves it the
  WRONG way (0.1706 -> 0.1743 on the anchor).  Every earlier establishment deficit in
  this campaign was Type B and dissolved under gain tuning — up to -73.7% on torus
  t_mid.  Here the same knob, over the same range, does nothing.  This is what a
  deficit orthogonal to the CV looks like, and it is the first one the campaign has
  produced.
* **The ABP reports success while being wrong.** The marginal establishment time
  `T_est` from the phi-KL rule is 0.00 with zero censored seeds on every eligible cell:
  by its own convergence diagnostic SHUS has finished immediately, while `e_F` sits
  at 49-93 x the estimator floor for the whole 800-unit run.  The deficit is invisible
  to every gate this project has used before Phase F.
* **Mechanism confirmed as designed.** `E_chan` starts near its floor (at t = 0 the
  population sits at phi ~ 0, where the reference conditional really is ~98% channel
  A), rises to 0.41-0.49 once the bias spreads the walkers across phi, and then decays
  only as slowly as the channels exchange: on the anchor 0.41 (t=100) -> 0.15 (t=800),
  on Hperp 2.5 0.49 -> 0.41.  The error is created by the biasing itself and drained
  only by hidden-coordinate transport.
* **T_est^chan is censored 8/8 on all three eligible cells** (E_chan never reaches
  2 x floor within T = 800), so the F2 window uses the fallback fixed below.
* Non-eligible cells behave exactly as the taxonomy predicts: Hperp <= 1.5 equilibrates
  its channels well inside T and lands at 5-6 e*, i.e. SHUS-sufficient, and there the
  familiar 1D pattern returns — gain tuning makes final accuracy WORSE (0.018 -> 0.029).

## F2 — the reallocation experiment   [FROZEN 2026-08-19, before any F2 row]

* **Cells:** all three eligible ones, in one batch: `hp2_d0` (anchor: severe, channel
  well populated), `hp2.5_d0` (sparse channel: 9.6% — tests whether reallocation
  still works when there is little to clone), `hp2_d0.5` (asymmetric: the frozen
  UNIFORM target is knowingly WRONG here, target 0.316 vs uniform's 0.5, and it is
  carried precisely to expose overshoot).
* **g\* = 1 on all three cells** by the frozen selection rule: the gains are within
  0.12-1.5% of each other on median I_F, far inside the 2-point band, which resolves
  toward the smaller gain.
* **Window (frozen derivation, censored fallback):** `t_on = ceil(Q90(T_hit^B))`
  pooled over the eligible cells = ceil(19.8) = **20**; with `T_est^chan` censored,
  `t_off = t_on + 0.25 (T - t_on)` = **215**, the same 25%-of-interval convention with
  the horizon replacing the unreachable establishment time.  As run fractions:
  `t_on_frac = 0.025`, `t_off_frac = 0.26875`.
* **Dose (frozen decision, stated before any F2 row):** the transferred protocol is
  theta = 0.01 at "stride 10 blocks", but block TIME differs across systems — 10
  blocks is 0.04 time units on the gateway and 0.2 here, so the raw stride would
  deliver ~975 events against the frozen winner's ~200.  The primary arm therefore
  transfers the **dose**: `fr_every_blocks = 49` gives 199 events in the window,
  matching the winner.  A secondary arm keeps the raw stride 10 (~975 events, 5x dose)
  so the dose dependence is measured rather than assumed.
* **Arms** (seeds **600-615**, one paired batch, batch_seed 20260960):
  1. `shus` — g\* baseline;
  2. `fr_cond` — conditional FR, theta = 0.01, stride 49, alpha_ess = 0.5;
  3. `fr_cond_hi` — conditional FR at the raw transferred stride 10 (5x dose);
  4. `cnt_cond` — stratified count control, `cond_bins = (32, 9)`: the phi resolution
     is set EQUAL to `n_strata = 32` so the only difference from `fr_cond` is the psi
     density estimator (9-bin histogram vs smooth KDE);
  5. `fr_marg` — marginal FR, same theta/stride: the blindness control;
  6. `sham_cond` — stratified matched-turnover sham shadowing `fr_cond`.
* Primary endpoint: median paired `dI_F` vs `shus` with paired bootstrap 95% CIs,
  plus the direct contrasts `fr_cond` vs `cnt_cond`, `fr_cond` vs `fr_marg`, and
  `fr_cond` vs `sham_cond`.  Secondary: `e_F(T)`, `E_chan(T)`, the `P_B(t)`
  trajectory against `p_B_ref_biased` (overshoot on the asymmetric cell is a
  REPORTED outcome, not a failure), ancestry floors, realized turnover and theta.
  Decision rules are the ones frozen in the Phase-F entry above and are not restated.

### F2a — clone decorrelation in the hidden coordinate (before any F2 accuracy claim)

The Q4a instrument, adapted: plain SHUS to `t0 = 200`, bias frozen, 8 x 8 parents
stratified over phi, each duplicated into two children evolved under the frozen bias
with independent noise.  Two decorrelation measures, because a Type-C system needs
both and Q4a's single number would hide the distinction:

* `m_chan(tau)` — excess probability that siblings still share a CHANNEL, over the
  same-phi-bin independent baseline.  For a population correction this SHOULD be
  slow: a clone that immediately forgets which channel it was in cannot carry the
  correction.  Long `tau_clone^chan` is a PRECONDITION here, not a limitation.
* `m_psi(tau)`, `m_phi(tau)` — the Q4a RMS-pair-difference measure on psi and on the
  CV.  Fast within-channel decorrelation means siblings are not statistically
  redundant for conditional averages.

Recorded per eligible cell against the F2 event stride (0.98 time units) and the
window length (195).  The pairing `tau_clone^chan >> stride` with
`tau_clone^psi ~ stride` is the configuration in which conditional cloning can help;
the opposite pairing would bound what F2 can buy, and is recorded either way.

### F2a outcome (2026-08-19, 4 seeds/cell — recorded, not to be edited)

| cell | tau_clone^chan | tau_clone^psi | tau_clone^phi | F2 event stride |
|---|---|---|---|---|
| Hperp 2.0, Delta 0   | 280 | 135 | **0.20** (resolution-limited) | 0.98 |
| Hperp 2.5, Delta 0   | 155 |  58 | **0.20** | 0.98 |
| Hperp 2.0, Delta 0.5 | 225 | 165 | **0.20** | 0.98 |

The instrument separates what Q4a's single number could not, **in one system at one
time**: a clone forgets its parent's CV within a single event stride
(`tau_clone^phi` is at the first recorded lag, exactly the WCA result), while it keeps
its CHANNEL identity for 160-290 event strides.  Independent baselines are meaningful
(same-channel 0.78-0.95, RMS psi separation 0.7-1.4 rad), so these are measurements,
not artifacts.

This is the mechanistic account of the whole campaign in one table.  Cloning is
worthless in the coordinate the bias already controls — the copy is independent again
before the next event — and valuable in the hidden one, where the copy carries the
correction for the rest of the run.  It also yields a **pre-run diagnostic**: measure
`tau_clone` on a candidate descriptor; if it is short compared with the intended event
stride, reallocation conditioned on that descriptor cannot help, whatever the metric.

### F2 outcome (2026-08-19, seeds 600-615, T = 800 — recorded, not to be edited)

Median paired `dI_F` vs the g\* = 1 baseline, paired bootstrap 95% CI:

| arm | Hperp 2.0, Delta 0 | Hperp 2.5, Delta 0 | Hperp 2.0, Delta 0.5 |
|---|---|---|---|
| `fr_cond` (dose-matched) | **-15.33 [-15.90, -13.38]** | **-12.61 [-13.68, -10.93]** | **-31.35 [-32.17, -28.58]** |
| `cnt_cond` (stratified count) | -15.15 [-15.59, -13.27] | -11.76 [-13.81, -10.71] | -28.22 [-29.77, -26.44] |
| `fr_marg` (marginal FR) | +0.19 [-1.85, 1.23] | -0.04 [-0.29, 0.16] | -0.18 [-1.48, 1.07] |
| `sham_cond` (matched turnover) | +0.25 [-0.70, 0.68] | -0.08 [-0.26, 0.06] | +0.11 [-0.67, 0.92] |
| `fr_cond_hi` (5x dose) | -30.80 [-33.06, -29.60] | -36.17 [-37.20, -34.85] | -72.50 [-73.26, -70.05] |

**Verdict by the frozen rule: the conditional population correction has independent
value on a Type-C deficit.** `fr_cond` clears the -5%-with-CI<0 bar on all three
cells and beats its own matched-turnover sham by -14.95 / -12.36 / -31.03%.  The
gateway's Stage-3 result was also positive (-11.4%), but it was reproduced by gain
tuning and tied by count balancing, so no arm-specific claim survived it.  This is the
first population-correction result in the project that **no other available arm
reproduces**: gain tuning is structurally out (F1: eightfold gain, <= 3% effect, wrong
sign), marginal FR is null, and the matched-turnover sham is null.

* **P2 confirmed: marginal FR is exactly null** (+0.19 / -0.04 / -0.18%, every CI
  straddling zero) while firing 58-80 resampling events.  Same theta, same stride,
  same window, same engine; only the geometry differs.  The blindness argument that
  motivated this phase is measured, not assumed.
* **P4 confirmed: stratified count ties conditional FR** (-0.84 [-1.98, 1.34],
  -1.21 [-1.87, 2.55], -2.44 [-7.60, 2.09]).  **The active ingredient is WHICH
  coordinate the reallocation is conditioned on, not the Fisher-Rao estimator.**  This
  was predicted before the run from the campaign's own lemma; the FR-vs-histogram
  question needs a >= 3D descriptor and is Phase G's, not this phase's.
* **P3 partially met.** The predicted >= 20% reduction was reached on one of three
  cells at the dose-matched transfer (-31.4% on the asymmetric cell); the other two
  gave -15.3% and -12.6%, past the decision bar but short of the prediction.  At 5x
  dose all three exceed 20%, but that arm **violates the frozen ancestry floor**
  (final `n_anc/K` = 0.34-0.36 < 0.5) and is therefore reported and NOT interpreted as
  a win.  The dose-matched arms pass every floor (min windowed `ESS_anc/K` >= 0.966,
  final `n_anc/K` >= 0.58).
* **Marginal invariance holds in production, not just in the unit tests.** On the
  anchor cell the phi-marginal KL the SHUS accumulator deposits from is
  0.00237-0.00251 across all six arms (`D_tol` = 0.0088): every arm's marginal gate
  says "converged" for the whole run, and only `e_F` separates them.  The correction
  cannot have been bought by perturbing the occupancy signal.
* **Mechanism, end to end.** Anchor-cell channel population (median, reference 0.5):
  all arms identical at the window start (0.0068 at t = 20); by the window end
  (t = 215) `shus`/`fr_marg`/`sham` sit at 0.129-0.132 while `fr_cond` is at 0.228 and
  `fr_cond_hi` at 0.296; the lead persists to T = 800 (0.332 vs 0.368 vs 0.404), with
  `E_chan` and `e_F` falling in step.  The step transports population across the
  hidden barrier during a 24%-of-run window; the physics then keeps it.
* **The knowingly-wrong target did not overshoot.** On the asymmetric cell the frozen
  uniform-in-psi target corresponds to a 0.5 channel fraction while the correct biased
  value is 0.316.  `fr_cond` ends at 0.245 and `fr_cond_hi` at 0.319 — approaching the
  correct value from below, not past it, because the window is temporary and the
  post-window relaxation sets the final populations.  The temporary-window discipline
  frozen since Stage 0 is what makes a deliberately incorrect target safe here; a
  persistent version would not enjoy this and is not claimed.

**What this does and does not establish.** It establishes that an adaptive-biasing
potential leaves exactly one repairable deficit — the conditional structure of the
coordinates it does not bias — and that a temporary, marginal-preserving population
correction repairs it where nothing else in this project's toolkit could: not gain
tuning (F1), not marginal FR, not turnover alone.  It does NOT establish that the
Fisher-Rao geometry is the reason: at one hidden dimension a 9-bin histogram does the
same job, exactly as the lemma predicts.  The FR-specific claim now has one honest
place left to live, and Phase G is where it must be settled.

## Phase G (opened by F2, design NOT yet frozen)

The dimensional claim: marginal or conditional, FR separates from count balancing only
where the density estimator binds, i.e. a reallocation descriptor of dimension >= 3
where a histogram's cells outnumber the walkers (12^4 ~ 2 x 10^4 cells against
K ~ 10^3).  The bi-channel construction generalizes directly (several hidden angles,
each with its own orthogonal barrier), and `conditional_log_ratio` already reads its
score off a joint KDE, so the engine work is a grid/kernel generalization rather than
a new method.  `tau_clone` per descriptor becomes the pre-run screen for which
coordinates belong in the conditioning set.  Design to be frozen before any Phase-G
run, as always.

---

# Phase F3/F4 — the baselines Phase F was missing   [FROZEN 2026-08-19, before any run]

Two corrections to the Phase-F design, raised after the F2 outcome and adopted before
any new row exists.  Both are recorded as gaps in the original design, not as
refinements of it.

**Gap 1 — the augmented CV.** F2 shows that `SHUS(phi) + conditional reallocation in
psi` beats `SHUS(phi)`.  But once psi is known to be the limiting coordinate, the
obvious alternative is to bias it: `SHUS(phi, psi)`.  The 2D engine has existed since
commit 8814431 and this arm was simply never run.  Until it is, F2's result cannot be
called an algorithmic advance — it might only be an indirect substitute for enlarging
the CV.  This is the more fundamental experiment and it precedes Phase G.

**Gap 2 — target parametrization.** The conditional target is UNIFORM in z.  "Uniform"
is not a canonical notion for a molecular descriptor: it changes under
reparametrization of z (helicity vs its logarithm).  Note first an identity that
removes one apparent knob: with the FR law, a tempered target
`q_lambda ~ p^{1-lambda} u^lambda` gives `p^+ ~ p^{1-theta lambda} u^{theta lambda}`,
i.e. **tempering the target is exactly equivalent to reducing theta** — the F2 dose
ladder already IS the target-temperature ladder.  What is NOT equivalent, and is what
the objection is really about, is a target that is uniform in a *different*
parametrization of z.  That is what F4 tests.

## Engine work committed with this freeze

`BiChannelConfig.cv in {"phi", "phipsi"}` with the 2D accumulator path; an
augmented-CV run is scored on the SAME deliverable via
`reduce_to_phi(F2_hat) = -beta^{-1} log int e^{-beta F2} dpsi` (exact on the reference
to 1e-12, tested).  Recorded before the run: the mollifier floor on the reduced
quantity is **identical for both CV choices** (e* = 0.00346 / 0.00351 / 0.00347 on the
three cells) because the psi-mollification integrates out — so neither side gets a
floor advantage.  The engine refuses reallocation arms under `cv = "phipsi"` (biasing
and reallocating the same coordinate is a different experiment).  Also tested: the
Langevin noise stream depends only on `(B, seeds, batch_seed)`, so a separate batch
with the same three keys is EXACTLY noise-paired with the stored F2 rows — the same
device used for A1b.

## F3a — augmented CV vs conditional reallocation, equal budget   [FROZEN]

* Cells: the three F1-eligible ones.  Seeds **600-615**, batch_seed **20260960**,
  B = 48 — identical to F2, so every F3a row is paired with its F2 counterpart.
* `cv = "phipsi"`, arms = the gain ladder `g_shus in {1, 2, 4, 8}` (the augmented-CV
  arm must be TUNED, or this repeats the campaign's own worst mistake).  K = 1024,
  T = 800: identical force evaluations, identical walkers, identical steps — the grid
  bookkeeping is the only cost difference and it is negligible.  `g*` by the frozen
  Pareto rule (qualifying median paired `e_F(T)` ratio vs g = 1 <= 1.05; lowest median
  `I_F`; ties within 2 points resolve toward the smaller gain).
* Primary endpoint: median paired `dI_F` of `SHUS(phi,psi)` at `g*` against the stored
  `shus` baseline AND against the stored `fr_cond`, with paired bootstrap CIs.
* **`E_chan` is NOT a valid cross-CV metric and is not used as one.** Under a 2D bias
  the sampled conditional is uniform BY DESIGN, so a large `E_chan` for that arm is
  the intended sampling distribution, not an error.  Only `e_F` on the reduced `F(phi)`
  is comparable across the two CV choices.

**P5 (predicted before the run):** the tuned augmented-CV ABP BEATS
`SHUS(phi) + conditional FR` on all three cells at K = 1024.  Biasing psi removes the
barrier outright; reallocation only moves population across it.  If P5 holds, the
honest headline is that conditional reallocation is valuable **when the limiting
coordinate cannot or will not be added to the biased CV** — not that it dominates
augmented biasing.  We state this now so the result cannot be re-narrated afterwards.

## F3b — where the augmented CV starts to cost   [FROZEN]

The sample-complexity question P5 leaves open: a `96 x 96` accumulator needs walkers to
fill it, while conditional reallocation needs only enough walkers per stratum.

* Anchor cell `hp2_d0`, fresh seeds **700-707**, T = 800, `K in {64, 256}` (K = 1024
  comes from F2/F3a).  One `cv = "phi"` batch (arms `shus`, `fr_cond`, `cnt_cond`,
  `sham_cond`) and one `cv = "phipsi"` batch (gain ladder) per K, same batch_seed
  (**20260970 + K**) so they are noise-paired.
* **`n_strata = max(4, K // 32)`** — frozen now, holding ~32 walkers per stratum, so
  the conditional arm's own sample complexity is handled by a stated rule rather than
  by a choice made after seeing the numbers.  This makes the method K-dependent and
  that is disclosed.
* **P6:** the augmented CV's margin over conditional reallocation SHRINKS as K falls,
  and the crossing K (if any) is the quantitative form of the practical claim.

## F4 — target-parametrization sensitivity   [FROZEN]

On the asymmetric cell `hp2_d0.5` (where the uniform target is already known to be
wrong) and the anchor `hp2_d0`, fresh seeds **720-735**, batch_seed **20260980**, all
other F2 settings unchanged.  Arms:
1. `shus` (baseline);
2. `fr_cond` (uniform in psi — the frozen target);
3. `fr_cond_rp+` (uniform in `psi' = psi + 0.8 sin psi`, i.e. target density in psi
   `~ 1 + 0.8 cos psi`: a 9:1 skew TOWARD channel A, the wrong way);
4. `fr_cond_rp-` (uniform in `psi' = psi - 0.8 sin psi`: the same skew toward B);
5. `fr_cond_oracle` (target = the exact `p_ref(psi | phi)`; **ORACLE, diagnosis
   only, never a claimable method** — it consults the reference the run is scored
   against);
6. `sham_cond`.

The reparametrized arms are deliberate, large, *arbitrary* mis-specifications of the
target — exactly what choosing helicity over log-helicity would do.  **P7:** the
benefit survives both reparametrizations at reduced size, and the oracle target
bounds how much a perfect target could add.  If instead a reparametrization destroys
or reverses the benefit, the method is parametrization-critical and that must be
stated as a limitation before any molecular application.

## Phase G — RENAMED: conditional density-estimation scaling (not "the FR claim")

The earlier framing was wrong and is corrected here.  A win of KDE-scored conditional
FR over histogram-scored stratified count in dimension >= 3 would establish that
**smooth conditional density estimation scales better than fixed bins** — an estimator
result, useful in practice, but NOT evidence that the Fisher-Rao geometry is the
right reaction law.  Both arms already implement the same FR-type law and differ only
in `p_hat(z | xi)`.  Phase G is therefore a scaling study (`d_z in {1,2,3,4}` at
several K), named for what it measures.

## Phase H — the actual Fisher-Rao law test (design sketch, freeze when reached)

To test the LAW rather than the estimator: compute one conditional KDE and feed it to
several update laws at matched turnover, matched ESS loss, matched event times and
matched target — FR's `p^+ ~ p^{1-theta} u^theta` (particle weight `(u/p)^theta`)
against, e.g., convex relaxation `p^+ = (1-alpha) p + alpha u` (particle weight
`(1-alpha) + alpha u/p`), and a chi^2 / deficit-proportional score.  Only a win under
those constraints is evidence for the Fisher-Rao reaction law itself.

### F3a outcome (2026-08-19, seeds 600-615, noise-paired with F2 — recorded, not to be edited)

**P5 confirmed, and by a wide margin.  The augmented CV wins.**

| cell | `fr_cond` (1D CV + conditional FR) | `cnt_cond` | **`SHUS(phi,psi)` at g\* = 1** | aug vs `fr_cond` |
|---|---|---|---|---|
| Hperp 2.0, Delta 0   | -15.33% (e_F 39.5 e*) | -15.15% | **-83.33% (e_F 0.0072 = 2.1 e*)** | **-80.40 [-80.66, -79.82]** |
| Hperp 2.5, Delta 0   | -12.61% (78.1 e*)     | -11.76% | **-83.20% (0.0085 = 2.4 e*)**     | **-80.79 [-81.25, -80.55]** |
| Hperp 2.0, Delta 0.5 | -31.35% (17.6 e*)     | -28.22% | **-79.93% (0.0090 = 2.6 e*)**     | **-70.91 [-71.46, -70.25]** |

Same walkers, same steps, same noise, same deliverable, same mollifier floor.  Biasing
the hidden coordinate takes the run to 2-3 x the estimator floor — essentially
converged — while conditional reallocation leaves it at 18-78 x.  `g*` = 1 by the
frozen Pareto rule (higher gains lower `I_F` further but fail the 1.05 final-error
guard); the choice is immaterial, g = 1 already wins by 80%.

**This changes the Phase-F headline and it is restated here rather than in a later
gloss.** The claim is NOT that conditional reallocation is the right way to fix a
Type-C deficit.  It is:

> A Type-C deficit is repaired best by adding the limiting coordinate to the biased
> CV.  Conditional reallocation repairs a large part of it — 15-31%, where base-method
> tuning repairs none — **without touching the CV**, and is therefore a fallback for
> when the limiting coordinate cannot be biased, not a competitor to biasing it.

The honest cases for the fallback, which are what remains to be established:
1. **Non-differentiable descriptors.** Birth-death needs only to EVALUATE z; biasing
   needs its gradient.  Hydrogen-bond counts, native contacts with hard cutoffs,
   cluster labels and most learned descriptors are usable as reallocation coordinates
   and not as biasing coordinates.  This is a real asymmetry and it is the strongest
   remaining argument, but it is an argument about applicability, not performance,
   and this project has not tested it.
2. **Descriptor dimension / sample complexity.**  An ABP must FILL its accumulator;
   reallocation only needs enough walkers per stratum.  At `d_z = 1`, `K = 1024` the
   96 x 96 accumulator is filled easily and the ABP wins by 80%.  Whether a crossing
   exists is now the load-bearing question, not a side study: **F3b (K ladder) and the
   renamed Phase G (`d_z` ladder) carry the practical case for the method.**

Note that argument 1 is weaker than it first looks: conditioning and biasing need the
SAME knowledge of *which* coordinate is slow.  The asymmetry is only in what can be
done with it once known.

### F4 outcome (2026-08-19, seeds 720-735 — recorded, not to be edited)

**P7 is REFUTED.  The method is target-parametrization sensitive.**

| arm | Hperp 2.0, Delta 0 | Hperp 2.0, Delta 0.5 | turnover |
|---|---|---|---|
| `fr_cond` (uniform in psi) | -14.89% | -31.21% | 327 / 341 |
| `fr_cond_rp+` (uniform in psi + 0.8 sin psi) | **+5.14 [3.75, 5.89]** | **+0.42 [0.18, 1.15]** | 282 / 260 |
| `fr_cond_rp-` (uniform in psi - 0.8 sin psi) | -63.33% | -42.16% | 617 / 612 |
| `fr_cond_oracle` (exact `p_ref(psi|phi)`) | -64.02% | -53.81% | 575 / 412 |

* A single arbitrary reparametrization of the descriptor — one that a modeller could
  make without thinking, e.g. helicity versus a nonlinear function of it — **destroys
  the entire benefit and slightly reverses it** (+5.1% / +0.4%).  `rp+` fires at a
  turnover comparable to `fr_cond` (282 vs 327), so this is direction, not dose: at
  matched intensity the sign flips.  This must be stated as a limitation before any
  molecular application, and it is the reason the target is now a first-class
  argument of `conditional_log_ratio` rather than a hard-coded uniform.
* The `rp-` and oracle magnitudes are **dose-confounded** (turnover 575-617 vs 327):
  a target further from the current population produces larger scores and more
  turnover, and F2 already showed dose alone buys a lot.  Their size is therefore not
  directly comparable to `fr_cond`'s; their DIRECTION is what the arms establish.
* The oracle nonetheless bounds the headroom: with a perfect conditional target the
  same machinery reaches -64% / -54% versus uniform's -15% / -31%.  Most of what
  conditional reallocation could deliver is being left on the table by the target,
  not by the update law.

**The structural point this exposes.** Under a reparametrization `z -> h(z)` both `q`
and `p_hat` carry the same Jacobian, so the ratio `q / p_hat` — and hence the FR
weight — is invariant *only if q is specified as a density in a fixed measure*.
"Uniform" is a choice of measure on the hidden coordinate.  The one
parametrization-invariant conditional target is the physical `p(z | xi)` itself, which
is exactly what is unknown.  **Every practical target therefore encodes a modelling
choice, on the same footing as choosing a CV — and F4 measures what a bad choice
costs: all of it.**  Note also the identity recorded in the F4 freeze: tempering a
target is equivalent to reducing theta, so temperedness is not an escape from this;
only the choice of reference measure is.

This makes target selection, not the update law, the central open problem for any
molecular application of conditional reallocation — which reorders Phase H.

### F3b outcome (2026-08-19, seeds 700-707, anchor cell — recorded, not to be edited)

**P6 is essentially refuted: the margin barely moves.**

| K | n_strata | `fr_cond` vs 1D `shus` | `sham_cond` | aug `g*`=1 vs 1D `shus` | **aug vs `fr_cond`** |
|---|---|---|---|---|---|
| 64   | 4  | -14.18 [-18.68, -6.41] | -5.14 [-8.97, 0.89] | -78.04 (e_F 9.0 e*)  | **-74.42 [-76.78, -72.13]** |
| 256  | 8  | -15.47 [-20.22, -13.40] | -0.23 [-1.85, 2.99] | -81.42 (5.6 e*)      | **-78.37 [-78.98, -75.66]** |
| 1024 | 32 | -15.33 [-15.90, -13.38] | +0.25 [-0.70, 0.68] | -83.33 (2.1 e*)      | **-80.40 [-80.66, -79.82]** |

Sixteen-fold fewer walkers moves the augmented CV's advantage from -80.4% to -74.4%.
There is a trend in the predicted direction and it is nowhere near a crossing.

**Why the sample-complexity argument fails, mechanistically.** At K = 64 the 2D
accumulator has 9216 cells and 64 walkers, and it still reaches 9.0 e* while the 1D
baseline sits at 57.1 e*.  An ABP's accumulator is filled by **trajectory time, not by
population size** — 64 walkers over 40 000 adaptation blocks deposit plentifully, and
the bias itself drives them to cover the grid.  This is Phase B's flooding mechanism
reappearing on the accumulator side, and it means low K does not create the opening
the conditional method needed.

Caveats recorded: at K = 64 the conditional comparison is weakly resolved (8 seeds,
`fr_cond` CI [-18.7, -6.4]) and its sham is no longer inert (-5.14%, CI straddling 0),
so part of that cell's apparent benefit may be turnover rather than direction; the
K = 256 and K = 1024 shams are inert as usual.

**Cost accounting.** Under the campaign's frozen convention (force evaluations,
`C = K x n_steps x N_particles`) the two CV choices are exactly equal here — same
walkers, same steps, same system.  In wall clock the augmented CV cost 2-5x more in
this implementation, because the analytic potential is nearly free and a 96 x 96 grid
convolution per block dominates; in any molecular application force evaluation
dominates and the difference vanishes.  Either way an 80% gap at these error levels
is not an accounting artifact.

### What survives of the practical case for conditional reallocation

After F3a/F3b/F4, the claims still standing are narrower than Phase F's own headline:

1. **It repairs what base-method tuning cannot.** Firm: F1 showed an eightfold gain
   change moves a Type-C deficit <= 3% (wrong sign); F2 showed conditional
   reallocation moves it 15-31% with null sham and null marginal-FR controls, and a
   verified-invariant CV marginal.  This is the campaign's one positive.
2. **It is decisively worse than biasing the limiting coordinate.** Firm, at
   `d_z = 1` across K in {64, 256, 1024}: -71 to -81%.
3. **The statistical-efficiency argument for it is refuted at `d_z = 1`** (F3b).  What
   remains is a *computational* argument — at `d_z >= 3` the accumulator becomes
   infeasible in memory rather than merely under-sampled — which is a weaker and more
   mundane claim than the one Phase G was opened on, and it is the honest version.
4. **The strongest untested case is non-differentiable descriptors:** birth-death needs
   only to evaluate z (hydrogen-bond counts, contact maps with hard cutoffs, cluster
   labels, learned descriptors), while biasing needs its gradient.  This project has
   not tested it, and it is an applicability argument, not a performance one.
5. **The target is a modelling choice that can cost everything** (F4), and the only
   parametrization-invariant target is the unknown physical conditional.

Phase G should therefore be run for what it can actually settle — the dimension at
which an augmented-CV ABP becomes infeasible while conditioning remains usable — and
NOT as a route to an FR-specific claim.  Before any molecular work, the mandatory
baseline set is now: tuned ABP on the base CV, **tuned ABP on the augmented CV**,
conditional reallocation, its stratified-count control, and its sham.

---

# Phase I — weighted fiber-wise selection: allocation vs represented probability   [FROZEN 2026-08-20, before any Phase-I row]

## Why this comes before Phase G and Phase H

F4 refuted P7: one arbitrary reparametrization of the hidden descriptor turned
`fr_cond`'s -14.9% into +5.1%.  The reason is structural, not numerical.  In the
equal-weight step, **the selection IS the represented distribution**: reallocating
walkers toward a target `q` makes the ensemble represent something closer to `q`, and
the SHUS accumulator then learns from that.  A wrong `q` is therefore a wrong physical
distribution, and the only parametrization-invariant `q` is the unknown physical
`p(z | xi)`.  Scaling that method to `d_z = 4` (Phase G) or refining its update law
(Phase H) does not touch this: it would make a target-critical method faster and
prettier.

There is a standard way to separate the two jobs the equal-weight step conflates —
carry statistical weights, as selection/splitting methods do:

    where computational effort goes      <-- decided by the score
    what probability the ensemble carries <-- decided by the weights

Phase I implements it and asks the one question that decides whether conditional
reallocation is a **sampling** method (a target is a modelling choice with F4's
downside) or a **variance-reduction** method (target-free, and safe to take to a
molecule).

## The step (`src/abpfr/fisher_rao_cond.py::child_weights`, committed with this freeze)

Everything about the event is unchanged: the same conditional score read off the
same particle-density KDE, the same theta, the same ESS backoff, the same
within-stratum systematic resampling, the same event times.  **The selection index is
bit-identical to the equal-weight arm's** (unit-tested), so a weighted arm is its
equal-weight twin at matched dose by construction, not by tuning.  What changes is
what the descendants carry:

* score-driven arms (FR, count): slot descending from parent `k` in stratum `j` gets
  `W_k / (cnt_j w_k)` — the importance weight of a draw from the `a`-tilted law, so
  more copies means proportionally less weight each, for ANY score;
* the sham: its kill is uniform and position-independent, so each parent's weight is
  split equally among its realized children (`sum_{children(i)} W_j = W_i`);
* both are then renormalized to hold **each stratum's total weight exactly fixed**.
  That is the weighted form of the invariant Phase F rests on: with weights, the
  xi-marginal the SHUS deposit sees is carried by weight, not by counts, so it is the
  stratum weight that must be conserved.  It is, to 1e-10 (unit-tested).

Consequences, all committed with this freeze and unit-tested:

* the SHUS deposit is `W_k * R(X_k)`, i.e. the accumulator learns the law the ensemble
  REPRESENTS, not the allocation; at `W = 1` every code path is bitwise the F2/F4
  engine (verified against `HEAD` on a 10-row batch, all series bit-identical);
* every scored diagnostic (`e_F`, `KL(p_xi||u)`, `E_cond`, `E_chan`, `P_B`) is a
  weighted estimate, and the PARTICLE channel split is recorded alongside it as
  `P_regions_n` — the decoupling is measured in every run, not assumed;
* **no weight-ESS guard is applied.**  A guard would silently change the dose and
  destroy the pairing with the equal-weight arm, so the weight ESS is recorded at
  every save (`ess_w_t`, `min_ess_w`) and reported as the method's cost.

## I1 — the target-sensitivity re-run, with and without weights   [FROZEN]

Same two cells as F4 (`hp2_d0`, `hp2_d0.5`), same `K = 1024`, `T = 800`,
`n_strata = 32`, same window/dose (`theta = 0.01`, stride 49 blocks, `[20, 215]`),
**fresh seeds 800-815**, batch_seed **20261000**.  All eleven arms in ONE batch, so
every contrast is paired in noise and initial conditions:

| family | arms |
|---|---|
| baseline | `shus` |
| equal weight (F4 arms, re-run on the new seeds) | `fr_cond`, `fr_cond_rp+`, `fr_cond_rp-`, `fr_cond_oracle`, `sham_cond` |
| weighted | `wfr_cond`, `wfr_cond_rp+`, `wfr_cond_rp-`, `wfr_cond_oracle`, `wsham_cond` |
| weighted, EXPLORATORY dose ladder (not dose-matched) | `wfr_cond_hot` (`theta = 0.1`), `wfr_cond_hot_oracle` |

`rp+`/`rp-` are F4's targets unchanged (uniform in `psi +- 0.8 sin psi`); the oracle
target remains DIAGNOSIS ONLY (it consults the reference the run is scored against).

**Primary endpoint (frozen): the target-induced spread.**  For each family, let
`S = max - min` of the median paired `dI_F` vs `shus` over the three *choosable*
targets `{uniform, rp+, rp-}` (the oracle is excluded: it is not a choice a modeller
can make).  F4 gives `S_equal = 20.0` points on `hp2_d0` and `31.6` on `hp2_d0.5`.

* **P8 (target safety):** `S_weighted <= 0.5 S_equal` on both cells, and no weighted
  target arm has a paired CI lying entirely above zero.  If P8 holds, weighting fixes
  the defect F4 exposed.
* **P9 (benefit retention) — predicted to FAIL, recorded now so it cannot be
  re-narrated:** `wfr_cond` will NOT reproduce `fr_cond`'s -15% / -31%; we expect it
  within noise of `shus` and of `wsham_cond`.  The reason is the same structural fact:
  the equal-weight gain came from imposing the target on the represented conditional,
  and weighting is precisely the removal of that mechanism.  What can still buy a
  genuine gain is variance reduction — more particles carrying the rare channel's
  weight, hence a better-resolved A->B flux — and that is the effect P9 measures.
* **P10 (the decisive diagnostic):** the ORACLE arms separate the two mechanisms
  cleanly.  Equal-weight oracle bought -64% / -54% (F4).  If `wfr_cond_oracle`
  collapses toward null, the whole Phase-F positive was *borrowing the answer from the
  target*, and conditional reallocation is a sampling method that needs knowledge it
  does not have.  If `wfr_cond_oracle` keeps a large fraction of that gain, the method
  has real variance-reduction value that survives being made target-safe — which
  would be the strongest result of the campaign and would justify Phase G/H.

* **P11 (exploratory, secondary):** weighting makes hard allocation *safe* — the
  represented law is held whatever the score does — so the natural follow-up is
  whether allocating ten times harder buys the variance reduction the frozen dose
  may be too weak to show.  `wfr_cond_hot` and `wfr_cond_hot_oracle` run at
  `theta = 0.1` (a probe confirms the ESS backoff leaves it intact; `theta = 0.5`
  does not survive and is not used).  These two arms have NO dose-matched sham, so a
  positive from them is a lead for a follow-up experiment and is not claimable here;
  a null from them, on the other hand, closes the variance-reduction hypothesis at
  this system, because `wfr_cond_hot_oracle` allocates hard toward the exactly
  correct conditional and cannot be improved upon as an allocation policy.

Secondary, reported for every arm: `min ess_w` (the cost), realized turnover (dose
check), `P_B` vs `P_B^n` (the decoupling — a weighted arm must show particle
enrichment comparable to its equal-weight twin while its represented `P_B` stays in
the plain-SHUS band, or the implementation claim is false), `KL(p_phi || u)` (the CV
marginal must stay invariant, as in F2), and `e_F(T) / e*`.

**Pilot disclosure (frozen with the design).**  Two engineering pilots preceded this
freeze and are reported so nothing about them is discovered later: (i) a 10-row,
T = 100 probe used only for wall-clock and to confirm the weight bookkeeping is
conserved; (ii) a 2-seed, full-T, 11-arm probe launched BEFORE this section was
written and read AFTER it, used to size the weight-ESS cost and to confirm that the
equal-weight arms reproduce F4 on fresh seeds.  The predictions above stand exactly as
they were written; the pilot's two-seed numbers are reported alongside the outcome
below so a reader can see what was already visible at freeze time.  The pilot also
corrected one engineering expectation: `ess_w` does NOT stay near 1 — at the frozen
dose it ends around 0.86-0.99 for the uniform and `rp+` targets and around 0.47-0.72
for the higher-turnover `rp-` and oracle targets — so the weight-variance cost is
real and is reported as a primary secondary endpoint rather than as a footnote.

**Interpretation rule (frozen).**  P8 alone is not a success: a method can be made
target-insensitive by making it inert, and P9 is expected to show exactly that.  The
combination that would keep conditional reallocation alive as a general method is
**P8 held AND P10 retaining a substantial fraction of the oracle gain**; the
combination P8 held + P9 and P10 both null means the honest conclusion is that
equal-weight conditional reallocation works by assuming the answer, and that its
practical value is confined to cases where a defensible target is available on
independent physical grounds.

### I1 outcome (2026-08-20, seeds 800-815, 416 rows in one paired batch — recorded, not to be edited)

**P8 confirmed (spread clause, by a factor of 7-24 beyond what it asked; no-harm
clause fails narrowly on one arm).  P9 confirmed as predicted: the benefit does not
survive.  P10 decisive: the Phase-F positive was the target being imposed on the
represented law.**

| arm | `hp2_d0` dI_F | `hp2_d0.5` dI_F | P_B (represented) | P_B^n (particles) | min ESS_w |
|---|---|---|---|---|---|
| `shus` | 0 (I_F 204.7, 47.8 e*) | 0 (114.9, 29.9 e*) | 0.3384 / 0.1748 | same | 1 |
| `fr_cond` | **-14.66 [-15.54, -12.70]** | **-28.81 [-29.97, -25.90]** | 0.3779 / 0.2402 | same | 1 |
| `fr_cond_rp+` | **+6.52 [4.67, 8.06]** | +0.19 [-0.83, 1.34] | 0.3193 / 0.1724 | same | 1 |
| `fr_cond_rp-` | -66.36 [-69.29, -63.52] | -39.85 [-43.79, -37.45] | 0.4878 / 0.3740 | same | 1 |
| `fr_cond_oracle` | -62.85 [-64.43, -60.81] | -55.37 [-59.31, -52.10] | 0.4761 / 0.2881 | same | 1 |
| `wfr_cond` | +0.08 [-1.03, 1.51] | -0.82 [-1.68, 0.83] | 0.3382 / 0.1761 | 0.4077 / 0.2710 | 0.93 / 0.89 |
| `wfr_cond_rp+` | +1.15 [0.18, 1.93] | +0.50 [-0.86, 1.62] | 0.3365 / 0.1736 | 0.3022 / 0.1738 | 0.95 / 0.99 |
| `wfr_cond_rp-` | -0.37 [-1.51, 1.30] | -2.43 [-3.44, -0.71] | 0.3403 / 0.1820 | 0.5923 / 0.4639 | 0.60 / 0.58 |
| `wfr_cond_oracle` | -2.84 [-4.55, -1.84] | -0.58 [-2.10, 0.18] | 0.3434 / 0.1789 | 0.6621 / 0.3618 | 0.48 / 0.74 |
| `sham_cond` | -0.02 | +0.39 | 0.3389 / 0.1743 | same | 1 |
| `wsham_cond` | -0.01 | +0.07 | 0.3404 / 0.1723 | 0.3408 / 0.1758 | 0.85 / 0.84 |
| `wfr_cond_hot` (exploratory) | -2.98 [-5.53, -0.50] | -2.72 [-5.13, 1.46] | 0.3453 / 0.1823 | 0.4419 / 0.3564 | 0.57 / 0.53 |
| `wfr_cond_hot_oracle` (exploratory) | -28.19 [-33.03, -17.18] | -3.95 [-7.35, -1.70] | 0.4034 / 0.1854 | 0.7676 / 0.5225 | 0.10 / 0.30 |

**P8 — target-induced spread over the three choosable targets.**  Equal weight
72.88 / 40.04 points; weighted **1.52 / 2.93** points; ratios 0.021 and 0.073 against
a criterion of 0.50.  The sign flip is gone: `rp+`, the reparametrization that cost
`fr_cond` its entire benefit and +6.5% on top, costs the weighted arm +1.15%.  The
no-harm clause nonetheless FAILS on that one arm (`wfr_cond_rp+` on `hp2_d0`, CI
[0.18, 1.93] entirely above zero) — a fifth of the equal-weight harm, but a real one,
and it is recorded as a partial fail rather than rounded to zero.

**P9 — nothing survives.**  `wfr_cond` is +0.08% / -0.82% against `shus` and
+1.17% / -0.50% against its own weighted sham, while `fr_cond` beats its sham by
-14.69% / -29.18% in the same batch.  The equal-weight step's entire margin over its
sham disappears when the descendants carry compensating weights.

**P10 — the mechanism, settled.**  The oracle target is the best conditional target
that exists.  With equal weights it buys -62.85% / -55.37%; with weights it buys
-2.84% / -0.58%.  Roughly 95% of the largest effect this campaign ever measured was
the target being written into the represented distribution, not information the
selection extracted from the trajectories.  The decoupling is visible directly in the
same rows: `wfr_cond_oracle` puts **66% of its particles** in channel B while the
population it represents stays at **0.3434** — the plain-SHUS value 0.3384 — and its
CV marginal is unmoved (`KL(p_phi||u)` 0.0023-0.0029 across `shus`, `fr_cond`,
`wfr_cond`).

**Cost.**  Weight ESS falls to 0.93-0.99 at the frozen dose for the uniform and `rp+`
targets, 0.48-0.74 for the high-turnover `rp-`/oracle targets, and 0.10-0.30 for the
hot arms.  No guard is applied, by design.

**P11 (exploratory) — a lead, and a confound found while checking it.**  At ten times
the dose the weighted arms do move: `wfr_cond_hot` -2.98% / -2.72%, and
`wfr_cond_hot_oracle` **-28.19% [-33.03, -17.18]** on the anchor cell.  Before this is
read as variance reduction, a bookkeeping check was run (no dynamics at all, the same
frozen population resampled 200 times, `scratchpad/drift2.py` reproduced in the I2
script): the importance rule plus the per-stratum renormalization is a RATIO estimator
and carries an O(1/walkers-per-stratum) residual bias **toward the target** — at the
production `n_strata = 32` (32 walkers per stratum) it moves a 0.151 fiber fraction by
+0.006 at `theta = 0.01` and **+0.014 at `theta = 0.1`**, and at 128 walkers per
stratum only +0.002 / +0.003.  (Dropping the renormalization does not help: the drift
reverses sign, the stratum weights random-walk to 11x their nominal value, and the
xi-marginal invariance the design rests on is lost.)  `wfr_cond_hot_oracle`'s
represented `P_B` moved +0.065 against plain SHUS, which is several times the static
drift but of the same sign and the same mechanism, so **the -28% is confounded and is
not claimable as it stands.**  I2 settles it.

### I2 — is the hot-dose lead variance reduction or residual bookkeeping bias?   [FROZEN 2026-08-20, before any I2 row]

Anchor cell `hp2_d0` only, seeds **800-815**, batch_seed **20261010**, everything else
as I1.  Five arms — `shus`, `wfr_cond_hot`, `wfr_cond_hot_oracle`, and a weighted
matched-turnover sham for EACH hot arm (`wsham_hot`, `wsham_hot_oracle`) — run twice,
in two batches identical except for `n_strata in {32, 8}`, i.e. **32 vs 128 walkers
per stratum**.  Same arms and same event schedule in both, so the two batches are
noise-paired to each other as well as within themselves.

* **P12:** if the hot-oracle gain is variance reduction, it survives at 128 walkers
  per stratum, where the static drift is ~4x smaller, and it beats its matched sham in
  both configurations.  If it is the residual ratio bias, it shrinks with the bias and
  collapses toward the sham.
* Recorded per configuration: the sham-relative margin (the primary number), the
  represented `P_B` gap against plain SHUS (the bias monitor), `KL(p_phi||u)` (wider
  strata weaken the marginal invariance and this is where that would show), and
  `min ess_w`.
* The static drift for each configuration is measured by the same script, on the same
  populations, before the batch runs — so the bias correction is quantified rather
  than argued.

### I2 outcome (2026-08-20, seeds 800-815, two noise-paired batches — recorded, not to be edited)

**P12 fails: the hot-dose lead is the bookkeeping bias, not variance reduction.**

Static bias of the weight rule, measured on the same populations before the batches
ran (200 events, no dynamics, fiber fraction 0.1514):

| walkers/stratum | `theta = 0.01` | `theta = 0.1` |
|---|---|---|
| 32 (`n_strata = 32`) | +0.0064 | **+0.0142** |
| 128 (`n_strata = 8`) | +0.0024 | **+0.0032** |

Anchor cell, `theta = 0.1`, against plain SHUS and against matched-turnover weighted
shams:

| | `n_strata = 32` (32/stratum) | `n_strata = 8` (128/stratum) |
|---|---|---|
| `wfr_cond_hot` vs `shus` | -0.22 [-3.20, 2.45] | -0.34 [-2.81, 1.42] |
| `wfr_cond_hot_oracle` vs `shus` | **-29.57 [-36.31, -22.91]** | **-1.71 [-12.08, 2.92]** |
| `wfr_cond_hot` vs its sham | -1.81 [-4.88, 2.30] | -2.44 [-3.82, 2.43] |
| `wfr_cond_hot_oracle` vs its sham | **-31.78 [-38.74, -18.57]** | **+0.09 [-10.64, 3.32]** |
| represented `dP_B` (oracle arm) | +0.0654 | +0.0216 |
| particle `P_B^n` (oracle arm) | 0.7407 | **0.8276** |
| `KL(p_phi||u)` at T (shus / hot / hot-oracle / sham) | 0.0022 / 0.0044 / 0.0140 / 0.0109 | 0.0023 / 0.0042 / 0.0261 / 0.0123 |

Quadrupling the walkers per stratum cuts the static bias by 4.4x, cuts the arm's
represented `dP_B` by 3x, and **removes the entire gain** (-31.8% -> +0.1% against its
own sham).  The allocation itself did not weaken — the particle population in channel
B is *larger* at 128/stratum (0.83 vs 0.74) — so the gain tracked the movement of the
REPRESENTED law, which is the bias, and not the allocation, which is the putative
variance reduction.  `wfr_cond_hot` (uniform target) is null against its sham in both
configurations.

Two honest caveats recorded with it: (i) `n_strata = 8` makes the strata 4x wider in
phi, and the CV marginal is correspondingly less protected (`KL` 0.026 vs 0.014 for
the oracle arm) — the configuration trades marginal invariance for lower bias, and
both effects point the same way only for the bias explanation; (ii) heavy turnover
perturbs the KDE-level marginal even at exactly preserved stratum counts (the sham
sits at `KL` 0.011-0.012 against the baseline's 0.0022), which is a within-stratum
effect of duplicated walkers and is why the F2-dose arms, not these, are the ones the
marginal-invariance claim is made for.

### What Phase I settles

1. **The mechanism of the campaign's one positive is identified.**  Equal-weight
   fiber-wise reallocation improves `I_F` by moving the represented conditional toward
   its target; the target's correctness is therefore the method's accuracy, which is
   why `rp+` reverses the sign (F4) and why the oracle is best (F4).  Forbidding that
   movement — while keeping the identical selection — removes 95% of the oracle's
   effect and all of the uniform target's.
2. **A weighted (measure-preserving) birth-death step did not repair the Type-C
   deficit, at any dose, even with a perfect target.**  The reason it is not expected
   to is that such a step satisfies `E[mu_N^+ | mu_N] = mu_N` — it does not move the
   represented law at the event — while a Type-C deficit is a BIAS, an unrelaxed
   conditional.
   **Scope correction (2026-08-20, narrowed after review; no number changes).**  An
   earlier wording here said such a step "can only reduce variance".  That is too
   strong as a general statement about an adaptive algorithm: the event is unbiased at
   the instant it fires, but it changes the GENEALOGY, and an altered genealogy can
   change later estimator variance and rare-transition statistics — which is exactly
   why weighted-ensemble and sequential Monte Carlo work where they do.  What I1/I2
   establish is the measurement, not the theorem: on this Type-C benchmark, weighted
   conditional allocation produced no repair of the deficit.
3. **Therefore conditional reallocation is a method for the case where a defensible
   target is known independently** (a symmetry-related family, discrete states with
   known relative free energies, a validated prior), and is not a general repair for
   hidden-coordinate starvation.  Where the target is a guess, F4 bounds the downside
   at "all of it, with the wrong sign".
4. **The remaining honest use of weights is as a safety net, not as an accelerator**:
   at the frozen dose the weighted arm costs 1-2% and removes a 20-40 point target
   risk.  Whether the safe version can ever WIN is a question about
   variance-limited — not bias-limited — hidden structure, which this system does not
   exhibit and which no experiment in this campaign has yet built.

---

# Phase J — the variance-limited regime: can measure-preserving selection win when the conditional is already right?   [FROZEN 2026-08-20, before any J2 row]

Phase I closed the bias route: equal-weight reallocation repairs a Type-C deficit by
writing its target into the represented conditional, and the measure-preserving
version — which cannot do that — buys nothing on a deficit that IS a bias.  The one
regime it was never tested in is the complementary one: **the represented conditional
correct in expectation, but resolved by too few walkers.**  That is the setting
weighted-ensemble / stratified-sampling methods are actually built for, and Phase J is
the experiment that decides whether the safe version of this method has a use at all.
It runs BEFORE Phase G (dimension scaling) and Phase H (reaction laws), both of which
would otherwise be refining a step with no demonstrated benefit.

## A structural fact this phase had to establish first (recorded, it constrains the design)

The obvious construction — make the hidden state RARE (`p_B ~ 1-5%`) and start the
ensemble at the correct conditional — cannot be had by tuning `Delta`.  Under a
CONVERGED phi-bias the sampled law is uniform in phi times `p_ref(psi | phi)`, so a
hidden state's population is its phi-averaged conditional weight, and its contribution
to the deliverable `F(phi) = -kT log sum_s Z_s(phi)` is that same weight.  Rarity and
relevance are the same number.  Scanned analytically over `Hperp in {1.25..2}`,
`Delta in {0.5..2}`, `Ha = Hb in {1, 2}` (`scripts/run_appmap_phaseJ_variance.py`
reproduces it): the shape of `F(phi)` that channel B carries, in units of the
mollifier floor, tracks the biased population monotonically —

| `Ha = Hb` | `Delta` | `p_B` (Boltzmann) | `p_B` (biased, = what a converged run samples) | `|dF from B| / e*` |
|---|---|---|---|---|
| 1.0 | 0.5 | 0.119 | 0.316 | 53 |
| 1.0 | 1.0 | 0.018 | 0.130 | 17.5 |
| 1.0 | 1.5 | 0.0025 | 0.029 | 3.4 |
| 1.0 | 2.0 | 0.0003 | 0.005 | **0.5 (below the floor)** |

In this family, a state rare enough to be poorly represented is by the same token one
the free energy barely depends on, and the ABP's own flattening is what promotes the
F-relevant states to O(1) population — the accumulator-side form of the flooding
mechanism Phase B found.  So the variance-limited regime realizable HERE is not a rare
state at large K; it is **few walkers per stratum**, i.e. small K, with a start that
removes the bias.

**Scope note (2026-08-20, narrowed after review).**  This is a property of the
symmetric bi-channel family scanned above, not a theorem about adaptive biasing.
`F(phi) = -beta^-1 log int e^{-beta V(phi,z)} dz` is an integral over z at fixed phi, so
a state may be globally rare, `int p(B|phi) dphi << 1`, and still dominate over a NARROW
interval of phi — omitting it would then cause a locally serious free-energy error at a
small population cost.  What the scan shows is that in THIS family the two quantities
move together, because channel B's phi-profile is as broad as channel A's; a
narrow-well hidden state would decouple them (at the cost of a stiffness this
integrator cannot carry: reaching a 2% population needs `Hb ~ 10^3`).  The distinction
matters for any later molecular example and is recorded so the constraint is not
mistaken for a law.

## The protocol (engine work committed with this freeze)

* `init = "stationary"`: walkers drawn from the exact stationary law of the CONVERGED
  bias (uniform in phi, psi from `p_ref(psi | phi)`) by grid inverse-CDF — the same
  construction `conditional_floors` already used, so the initial condition and the
  metric floors come from one sampler;
* `warm_start = True`: the accumulator starts at its analytic fixed point
  `R* = K_eps e^{-beta F1}` instead of at `R = 1`.
* Together these make the run start converged AND stationary, so there is no
  establishment transient and no unrelaxed conditional: what is left is the estimator's
  variance about its fixed point.  Both consult the reference and are therefore
  experimental CONDITIONS applied identically to every arm — never an arm's private
  information, and never a claimable method.
* `Method.cond_state`: a score on the DISCRETE hidden state (channel label) with no
  kernel and no bins — classical stratified allocation, realizing
  `n_s ~ p_s^{1-theta} q_s^theta`, so `theta = 1` is equal count per state and
  `theta = 1/2` the square-root compromise.  Neyman's `n_s ~ p_s sigma_s` is this
  family only at constant `sigma_s`; the per-state spread is recorded, not fed back,
  because using it would make the allocation depend on the estimand being scored.

## J1 — the screen (plain SHUS only; RUN, seeds 900-907, recorded here)

Cells `Hperp in {1.5, 2.0} x Delta in {1.0, 1.5}` (`Ha = Hb = 1`, `beta = 4`),
`K in {64, 256, 1024}`, T = 800, `init = "stationary"`, `warm_start`, one arm.
Gate for J2, stated here and evaluated on the screen (disclosure: these criteria were
written after the screen was read; every cell passes all three, so the gate performs no
cell selection — the anchor was then chosen on deficit SIZE, a plain-SHUS quantity):

1. **no bias**: `|median P_B(T) - p_B_ref_biased| <= 1` binomial sd `sqrt(p(1-p)/K)`;
2. **variance-dominated**: seed variance >= 50% of the MSE of `F_hat(T)` in the
   B-carrying window `||phi| - pi/2| < pi/4`;
3. **something to improve**: median `e_F(T) >= 2 e*`.

| K | cell | `e_F(T)/e*` | `P_B` vs ref (in sd) | var / MSE (global) | var / MSE (B window) |
|---|---|---|---|---|---|
| 1024 | hp1.5_d1.0 | 4.2 | +0.21 | 0.79 | 0.83 |
| 1024 | hp2.0_d1.0 | 4.4 | -0.73 | 0.78 | 0.77 |
| 1024 | hp2.0_d1.5 | 3.2 | -0.32 | 0.85 | 0.87 |
| 256 | hp1.5_d1.0 | 7.2 | -0.45 | 0.87 | 0.88 |
| 256 | hp2.0_d1.0 | 7.7 | -0.30 | 0.88 | 0.84 |
| 64 | hp1.5_d1.0 | 14.7 | -0.50 | 0.90 | 0.92 |
| 64 | hp2.0_d1.0 | 14.9 | -0.47 | 0.82 | 0.82 |

**All three gates pass everywhere.**  This is the first cell in the campaign whose
deficit is a VARIANCE: the represented conditional sits within half a binomial sd of
the exact one at every K, and 77-92% of the remaining error is seed-to-seed scatter.
(Contrast Phase F's `chanA` protocol on the same system: `I_F` 115-205 with the
conditional grossly wrong.)  The deficit grows as K falls exactly as a variance should.

## J2 — the experiment   [FROZEN]

Anchor: `K = 256`, `n_strata = 8` — **32 walkers per stratum, the same per-stratum
sample size whose weight-bookkeeping bias I2 measured (+0.0064 at `theta = 0.01`,
+0.0142 at `theta = 0.1`)**, so the known confound is held at a known size rather than
rediscovered.  Cells `hp1.5_d1.0` (exchange active, `tau_B->A = 114 << T`) and
`hp2.0_d1.0` (`tau_B->A = 634 ~ T`) — the exchange rate is the axis along which a
splitting method should stop or start working.  Fresh seeds **920-935**, batch_seed
**20261020**, T = 800, window and stride as F2 (`[20, 215]`, 49 blocks).  Ten arms in
one paired batch:

| arm | score | weights | theta |
|---|---|---|---|
| `shus` | — | — | — |
| `fr_cond` | joint KDE | EQUAL | 0.01 |
| `sham_cond` | — (shadows `fr_cond`) | EQUAL | — |
| `wfr_cond` | joint KDE | weighted | 0.01 |
| `wfr_cond_hot` | joint KDE | weighted | 0.1 |
| `wcnt_cond_hot` | 8x8 histogram | weighted | 0.1 |
| `wstate_hot` | discrete channel | weighted | 0.1 |
| `wstate_eq` | discrete channel | weighted | 1.0 (equal count per state) |
| `wsham_cond` | — (shadows `wfr_cond_hot`) | weighted | — |
| `wsham_eq` | — (shadows `wstate_eq`) | weighted | — |

**Primary endpoint (frozen): the seed VARIANCE of `F_hat(T)` in the B-carrying window,
each weighted arm against its matched-turnover weighted sham.**  Variance, not MSE, is
primary precisely because I2 showed the weight rule carries a small mean shift toward
the target: a mean shift lands in bias^2 and leaves variance alone, so the two effects
are separated by construction instead of being argued about.  Both are reported, with
`bias^2`, MSE, `I_F`, `e_F(T)/e*`, `min ess_w`, turnover and the realized `P_B` drift.

* **P13 (the hypothesis):** at least one weighted allocation arm reduces the B-window
  seed variance by >= 10% against its matched sham, on at least the exchange-active
  cell.  Recorded prior: genuinely uncertain, and the reason to doubt it is specific —
  weighted selection conserves each stratum's weight, so the part of the variance that
  comes from the CHANNEL ALLOCATION draw is untouched by construction; only the
  within-channel sampling noise can fall.  If the channel draw dominates, P13 fails and
  the mechanism is exhausted.  The exchange-active cell is where it should work best,
  because there the channel weight is re-drawn by the dynamics many times per run and
  splitting can resolve those crossings.
* **P14 (the mirror of Phase F):** the EQUAL-WEIGHT arm `fr_cond`, which gained
  -15 to -31% on this same system when the conditional was wrong, now **HURTS** —
  its target is no longer approximately right, so writing it into a correct conditional
  is pure damage.  Predicted: `dI_F > 0` with the CI above zero, and its `bias^2`
  component rises while its variance may well fall.  This is the cleanest possible
  demonstration that the Phase-F gain was target-borrowing, and it costs one arm.
* **P15 (is the FR rule special?):** the discrete-state allocation — no kernel, no bins,
  nothing but counts of the hidden state per stratum — matches or beats the joint-KDE
  score at matched dose.  A fifth tie would say the estimator has never been the active
  ingredient; a KDE win here would be the first evidence in the campaign that it is.

**Interpretation rule (frozen).**  If P13 fails on both cells, the safe version of
conditional reallocation has no demonstrated benefit in either regime — bias-limited
(Phase I) or variance-limited (Phase J) — and the campaign's recommendation becomes:
population selection on an unresolved descriptor is worth running only when an
independently defensible target is available, in which case its gain is the target's,
not the geometry's.  Phase G and Phase H are then not worth running on this method.

### J2 outcome (2026-08-20, seeds 920-935, 320 rows in one paired batch — recorded, not to be edited)

**P13 fails.  P14 confirmed, and decisively.  P15 untestable as designed (the two
scores did not match dose).  And the frozen primary endpoint turned out to be the
wrong one — that is reported first, because it changes how the headline number reads.**

**The frozen endpoint is invalid here, and why.**  The primary was "seed variance
against the matched-turnover weighted sham".  Against that reference every weighted
arm looks excellent: `wfr_cond` -92% / -74%, `wfr_cond_hot` -86% / -64%,
`wcnt_cond_hot` -83% / -22%, `wstate_hot` -85% / -54%.  **None of that is a benefit.**
The weighted sham is not inert — it degrades `I_F` by +93% / +88% and inflates the
B-window variance by +766% / +313% against plain SHUS — so the "improvements" are
measured against a broken arm.  The reason is structural and is worth recording:
random weight-conserving churn fragments weights multiplicatively (each refill splits
a survivor's weight), so the weight ESS random-walks to 0.22 over 800 events, whereas
a score-driven allocation's weights are a deterministic function of the score and stop
changing once the particle population reaches the target.  **A matched-turnover sham is
a valid null for an equal-weight step and is NOT one for a weighted step.**  Every
number below is therefore re-scored against plain SHUS.

| arm | cell hp1.5 (`tau_B->A` = 114) | | | cell hp2.0 (`tau_B->A` = 634) | | | |
|---|---|---|---|---|---|---|---|
| | `dI_F` | var(B) | MSE(B) | `dI_F` | var(B) | MSE(B) | min ESS_w |
| `fr_cond` (equal weight) | **+24.0 [19.2, 42.0]** | -25% | -15% | **+87.6 [39.8, 116.9]** | -7% | **+104%** | 1.00 |
| `sham_cond` (equal weight) | +1.9 [-3.4, 6.4] | -21% | -20% | -3.0 [-6.2, 6.9] | -6% | -3% | 1.00 |
| `wfr_cond` (theta 0.01) | +7.9 [-2.8, 14.3] | **-31%** | -25% | +6.8 [-0.8, 19.4] | +8% | +9% | 0.92 |
| `wfr_cond_hot` (theta 0.1) | +44.6 | +25% | +25% | +41.1 | +50% | +66% | 0.51 |
| `wcnt_cond_hot` | +36.0 | +47% | +52% | +37.7 | +221% | +219% | 0.55 |
| `wstate_hot` | +23.8 | +33% | +44% | +38.6 | +89% | +119% | 0.66 |
| `wstate_eq` (equal count/state) | +309 | +980% | +1052% | +263 | +1391% | +1812% | 0.03 |

**P13 — no.**  Not one weighted arm improves the deliverable against plain SHUS on
either cell; every `dI_F` is positive or straddles zero.  The single variance
reduction is the gentlest arm on the exchange-active cell (`wfr_cond`, -31% B-window
variance at `ESS_w` 0.92), and it (i) does not replicate on the slower cell (+8%),
(ii) does not reach the deliverable (`I_F` +7.9%), and (iii) is **mostly not
direction**: the equal-weight matched-turnover sham, which allocates nothing, already
buys -21% of it.  Resampling of any kind couples seeds through shared ancestry and
therefore shrinks seed-to-seed scatter on its own — a null of about -20% that any
variance endpoint in a selection experiment has to clear, and which this campaign had
not measured before.  Stronger allocation is strictly worse, and the mechanism is
visible in the same rows: `ESS_w` falls to 0.51-0.66 at `theta = 0.1` and to 0.03 at
full equalization, so the accumulator ends up driven by fewer effective walkers than
plain SHUS has.  **Allocating more particles into the rare channel and paying for it
in weight variance is a wash at best on this system, and a large loss at any
appreciable dose.**

**P14 — yes, and it is the cleanest confirmation of the Phase-I mechanism.**  The
equal-weight arm `fr_cond` is the SAME arm, at the same dose, on the same system that
gained -15% to -31% in Phase F.  With the conditional now correct in expectation it
**hurts**: `I_F` +24.0% and +87.6%, both CIs entirely above zero, its `bias^2` in the
B window up +276% and +2676%, and its represented `P_B` displaced +2.12 binomial sd on
the asymmetric cell.  Its VARIANCE actually falls (-25%, -7%) — it trades a little
variance for a much larger bias, which is exactly what "writing the target into the
represented law" means.  Phase F's positive and Phase J's negative are the same
operation applied to a wrong and to a right conditional.

**P15 — not answered.**  At the same `theta` the discrete-state score fires far less
turnover than the joint KDE (278 vs 870 events' worth), so the two were not
dose-matched and the comparison is confounded; what can be said is that the
kernel-free score is not worse than the KDE at comparable damage, consistent with the
campaign's four earlier ties.  The classical equal-count-per-state allocation
(`theta = 1`) is not merely inert but destructive here (`ESS_w` 0.03), which is a
useful negative for anyone tempted by textbook stratified allocation inside an
adaptive-bias run: the allocation and the represented measure are carried by the same
weights, and equalizing counts fragments them.

### What Phases I and J together settle

1. **Bias-limited hidden structure** (Phase F/I): equal-weight conditional
   reallocation repairs it *by writing its target into the represented conditional*;
   the measure-preserving version repairs nothing, at any dose, even with a perfect
   target.
2. **Variance-limited hidden structure** (Phase J): measure-preserving allocation buys
   no improvement in the deliverable; its best case is a variance reduction that
   undirected churn largely reproduces, and any appreciable dose loses more to weight
   degeneracy than it gains in coverage.
3. **The same step damages a correct conditional as much as it repairs a wrong one**
   (P14), so its sign depends entirely on whether the target is closer to the truth
   than the current ensemble is — which is unknowable without the answer.
4. **Method-level warnings this campaign produced that outlive its negative result:**
   a matched-turnover sham is not a valid null once walkers carry weights; seed-variance
   endpoints in selection experiments carry a ~20% churn null; and "uniform" is a
   choice of reference measure, not a canonical target.

**Recommendation (frozen).**  Phase G (descriptor-dimension scaling) and Phase H
(reaction-law comparison) are not worth running on this method: both refine a step
with no demonstrated benefit in either regime.  Molecular work (deca-alanine) should
not proceed on conditional reallocation with a guessed target.  The only remaining
defensible use is the one Phase I identified — a hidden descriptor whose conditional
target is known on independent grounds (symmetry-related states, discrete states with
known relative free energies) — where the gain is the target's information, not the
geometry's, and where the honest comparison is against simply biasing that descriptor
(F3a: -71% to -81% in favour of biasing it).

---

# The speed map — every stored run rescored as time-to-accuracy (2026-08-20, no new simulations)

`I_F` mixes two things: how fast the error fell and how low it ended.  The campaign's
other frozen endpoint separates them, and it had only ever been run on the gateway:

    tau_eps = first t whose trailing 0.2 T window stays at or below eps   (right-censored)
    S_eps   = tau_eps^baseline / tau_eps^arm

`scripts/analyze_speed_map.py` applies it to every stored run on a ladder of thresholds
in units of each cell's analytic mollifier floor `e*`, so a rung means the same thing in
every system.  Censoring is printed, and a speedup is computed only over seeds where
both arms attained the rung, with that count shown.  Nothing was re-simulated.

### Gateway anchor_D — the original positive, and what it was

| `eps/e*` | `fr_temp` vs UNTUNED `shus` | `count` vs UNTUNED `shus` | `sham` | `fr_temp` vs TUNED `shus_gbest` |
|---|---|---|---|---|
| 32 | **1.29 [1.27, 1.33]** | 1.30 [1.28, 1.31] | 1.00 | 1.06 [1.01, 1.10] |
| 16 | 1.11 [1.10, 1.12] | 1.11 [1.10, 1.12] | 1.00 | 0.87 [0.86, 0.88] |
| 8 | 1.08 [1.07, 1.09] | 1.08 [1.07, 1.09] | 1.00 | 1.29 [1.27, 1.32] |
| 4 | 1.05 [1.04, 1.06] | 1.04 [1.04, 1.05] | 1.00 | 1.11 [1.00, 1.14] |
| 2 | 1.03 [1.01, 1.04] | 1.03 [1.02, 1.04] | 1.00 | 0.97 [0.89, 1.04] |

Against the untuned baseline the speedup is real, largest at loose accuracy (1.29x) and
decaying to 1.03x at the tightest rung the run resolves — the same shape the Stage-3
threshold ladder reported, now with 32 paired seeds at every rung.  **Count balancing
matches it to two decimals at every rung** (a fifth replication of that tie), and the
matched-turnover sham is exactly 1.00, so the effect is real and is not FR-specific.
Against the TUNED baseline the ordering is not even consistent in sign (1.06, 0.87,
1.29, 1.11, 0.97): there is no rung-independent speedup left.

### 2D torus t_mid — the adaptation-rate case

`gstar_fr` = **1.00 at every rung** (1.00, 1.00, 1.00, 1.00, 0.98), `gstar_count9`
likewise, `gstar_sham` likewise — while the untuned `shus_g1` scores 0.35, 0.19, 0.14,
0.13 and is censored at the tightest rung.  On this system the entire speed story is the
adaptation gain, and reallocation contributes nothing measurable.

### Phase F Type-C — what the conditional step bought, and what biasing the coordinate bought

| arm | `hp2_d0` (`eps` = 64 e*) | `hp2_d0.5` (`eps` = 32 e*) |
|---|---|---|
| `fr_cond` (equal weight) | **1.57 [1.49, 1.68]** | **3.66 [3.14, 3.77]** |
| `cnt_cond` (stratified count) | 1.65 [1.48, 1.70] | 3.47 [3.32, 3.57] |
| `fr_marg` (marginal FR) | 1.00 [0.96, 1.06] | 1.00 [0.98, 1.04] |
| `sham_cond` | 0.99 | 1.00 |
| **`aug_g1` (bias psi too)** | **6.25 [6.09, 6.70]** | **7.31 [7.24, 7.62]** |

The conditional step is a genuine 1.6-3.7x time-to-accuracy speedup where marginal FR
and the sham are exactly 1.00 — and simply adding the hidden coordinate to the biased CV
is **6-7x**, i.e. about 4x faster again than the fallback, at identical cost.  This is
the F3a conclusion in the speed metric rather than in `I_F`.

### Phase I — the same speedup, switched off by making the step measure-preserving

| arm | `hp2_d0` (64 e*) | `hp2_d0.5` (32 e*) |
|---|---|---|
| `fr_cond` (equal weight, uniform target) | 1.60 [1.53, 1.64] | 3.50 [3.29, 3.84] |
| `fr_cond_oracle` (equal weight, exact target) | **4.84 [4.32, 4.96]** | **5.37 [4.74, 5.90]** |
| `wfr_cond` (weighted, uniform) | **0.99 [0.97, 1.03]** | **1.01 [0.99, 1.05]** |
| `wfr_cond_oracle` (weighted, exact target) | **1.04 [1.01, 1.18]** | **1.00 [0.99, 1.05]** |
| `wfr_cond_hot` (weighted, 10x dose) | 1.05 [0.99, 1.13] | 1.05 [0.95, 1.08] |
| `sham_cond` / `wsham_cond` | 1.00 / 1.00 | 1.00 / 1.00 |

**This is the mechanism result restated in the metric the speedup claim was originally
made in.**  An oracle conditional target reaches the accuracy rung 4.8-5.4x sooner with
equal weights, and 1.00-1.04x sooner once the descendants carry compensating weights.
The acceleration was the target information entering the represented law, not a property
of the selection.

### Phase J — no convergence phase by construction

The warm-started runs begin AT the fixed point, are driven up to their own sampling-noise
level and relax back, so `tau_eps` is undefined.  The comparable quantity is the late-run
noise level (mean `e_F` over the last quarter of T, in units of `e*`): `shus` 7.6 / 7.4,
`wfr_cond` **7.9 / 7.5 (ratio 1.00 [0.94, 1.15] and 0.99 [0.96, 1.14])**, `wfr_cond_hot`
11.0 / 9.2 (1.39, 1.28), `fr_cond` 8.0 / 12.9 (1.09, 1.69), `wsham_cond` 17.1 / 16.5
(2.19, 2.21).  Measure-preserving allocation neither speeds up nor slows down the
estimator at the gentle dose, and costs at any stronger one.

### What the speed map settles

Three claims had been running together under the word "faster", and they separate
cleanly:

* **A — did the error curve sit lower?**  Yes, for gateway FR and for Phase-F equal-weight
  conditional FR.
* **B — did it reach a fixed accuracy sooner?**  Yes: 1.03-1.29x on the untuned gateway
  (matched exactly by count balancing), 1.6-3.7x on Type C.  Both are real.
* **C — is that a target-free acceleration, after controlling for base-method tuning and
  for changes of the represented law?**  **No.**  Against a tuned baseline the gateway
  speedup is inconsistent in sign and the torus speedup is exactly 1.00; on Type C the
  entire speedup — including the oracle's 4.8-5.4x — falls to 1.00-1.04x the moment the
  step is made measure-preserving.

The honest summary of the campaign's speed evidence is therefore: **the observed
accelerations are real, and every one of them is attributable either to a correction the
adaptive bias can make by itself (gateway, torus) or to target information injected into
the represented distribution (Type C).  No target-free Fisher-Rao acceleration of the
free-energy estimate has been demonstrated.**
