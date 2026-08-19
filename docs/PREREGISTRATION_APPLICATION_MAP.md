# PREREGISTRATION — Applicability map (post-campaign study)

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
