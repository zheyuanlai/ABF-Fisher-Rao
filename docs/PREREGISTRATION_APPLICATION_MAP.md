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
