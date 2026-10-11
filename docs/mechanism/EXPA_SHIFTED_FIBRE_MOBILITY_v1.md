# Experiment A: does faster transverse mobility rescue FR on the shifted fibre?

**Version 1** — 2026-10-11. Results commit `1ec25ca5` (production and analysis), tooling `014c4be0`, gate records
`ef8ebc20`. Preregistration `docs/final_campaign/PREREGISTRATION.md` (frozen `c5a2264f`; Amendments 1–2 written
before any production run). Any revision of this report is a new `_v2` file; this file is not edited.

Every statement is typed **[identity]** (exact), **[observed]** (measured, seed-bootstrap CIs) or **[hypothesis]**.

**Data and code.** Runs `results/final_campaign/expA/<cell>/N<N>/s<seed>_<arm>.npz` (gitignored; sha256 in the run
ledger); per-cell analysis `results/final_campaign/expA/<cell>/analysis/`; synthesis
`results/final_campaign/expA/synthesis/{cross_cell,mech,outcome}.{json,md}`; seed tables
`results/final_campaign/expA/tables/`; validation `results/final_campaign/expA/validation/`, checks
`results/final_campaign/expA/checks/`; figures `figures/final_campaign/expA/` (A1–A8, per-cell A–F) and
`figures/final_campaign/expA/validation/` (A4, A4b). Tools `scripts/final_campaign/{expA_*,make_cells,run_cells}.py`.

## 1. Design (as frozen)

* **Model.** The shifted fibre V = F\*(x) + ½κ²(y − m(x))², m = (√(2/β)/κ) log ω, κ = 1, all gateway constants (β 16,
  H 0.5, s 0.1, ω = 1 + 31 e^{−x²/2s²}). Only the transverse mobility changes: Y ← Y − λ∂ᵧV h + √(2λh/β) ξ, with
  **λ ∈ {1, 4, 16}** (task-specified) **and λ = 128** (preregistered extension, added before any data because of §2.2).
* [identity] For every λ the Gibbs law, F = F\* + C, the conditional law Y | x ~ N(m, 1/(βκ²)) and
  Var(f | x) = (2/β²)(ω′/ω)² are unchanged; the x-update, the ABF estimator and FR do not involve λ.
* **Runs.** N ∈ {512, 128} × {ABF, ABF + uniform FR} × 4 λ × 32 fresh seeds 8300–8331 = **512 runs**, B = 2048·40/h =
  3.2768 × 10⁹ walker-steps per run, h = 2.5 × 10⁻⁵, frozen gateway ABF/FR constants (180 bins, min_count 1, γ 1.5,
  η 0.1, clip 3, FR every 0.004 t.u., ramp 4 t.u., cap ⌊0.08N⌋). Seeds coupled across λ and arms (same initial
  normals, same Langevin noise).
* **Primary contrast** E1(N) = exp(median_s[L_16(s) − L_1(s)]), L = log(Ī_F^FR/Ī_F^ABF), Holm over N; **E2** the same
  for λ 128 vs 1 (separate family); trends T1–T3 unadjusted. Outcome labels per prereg §1.7 (A / A-qualified / B / I /
  C-equivalent / C-reverse / C-inconclusive), guards g1–g4, combination across N.

## 2. Numerical validation and manipulation checks

### 2.1 Gate and checks (all before production) — **PASS**

| item | result |
|---|---|
| G-A1 timestep gate, λ 4 / 16 / 128 at h 2.5e-5 | V1, V2, V3, V3b, V5 (+ ABF-bias smoke) PASS; V4 holds; V1 Holm 0/6, V4 Holm 0/24 rejections → **h_A = 2.5e-5**, no refinement |
| G-A2 λ = 1 | frozen record PASS; its 16 em_flat chains replayed through the extended harness **bitwise identical** (full-length inertness) |
| 1 forces vs FD | max rel. error 2.7e-10 (286 points per λ); kernel deposit force = `grad_V` to 3.4e-16 |
| 2 F\*, F\*′ across λ | bitwise equal across the 4 cells and to the equal-budget gateway reference |
| 3 conditional law, Var(f\|x) | [identity]; measured V1 / V3b PASS at every λ |
| 4a / 4b integrator | V1/V4 vs exact and MALA PASS; frozen-x AR(1) through the production engine: 0/30 Holm rejections, incl. the resolvable-inflation case (φ = 0.9, EM excess 5.3 %) at λ 1 and 16 |
| 5 timestep at λ 16, 128 | gate PASS |
| 6 ABF/FR pairing | smoke 16/16 and production **256/256** pairs bitwise equal before the first realised death (frozen rule; 4 pairs via replay) |
| 7 streams | Langevin stream identical across arms and λ; FR stream PCG64 SeedSequence([seed, N, 1]) |
| 8 old configurations | engine sha256 `69087f2c…` unchanged; 4/4 stored production jobs re-run bitwise; harness inertness PASS |
| 9 smoke | 32/32 runs, contract and diagnostic consistency PASS |

### 2.2 What λ changes (manipulation checks, flat-bias validation chains; jackknife 95 % CIs)

**MC1 — coupled conditional relaxation** (1/e time of the conditional ACF of y; the force ACF is identical up to sign):

| λ | wells (x₀ = −1) | gate centre (x₀ = 0) | **flank (x₀ = −0.21)** | flank speed-up vs λ 1 | flank integral to 5 t.u. |
|---|---|---|---|---|---|
| 1 | 0.99 | 0.544 | **0.0391** [0.0387, 0.0396] | 1 | 0.362 |
| 4 | 0.249 | 0.162 | **0.0344** | ×1.14 | 0.208 |
| 16 | 0.0627 | 0.054 | **0.0238** | ×1.65 | 0.061 |
| 128 | 0.0078 | 0.0078 | **0.0063** | ×6.2 | 0.0077 |

[identity + observed] With x free, the residual d = y − m(x) relaxes at κ²(λ + m′²); m′² = 33 at the flank, where
Var(f | x) peaks, so mobility barely changes the fast flank relaxation until λ ≫ 33 (coupled prediction 0.029, 0.027,
0.020, 0.0062), while in the wells and at the gate centre the time scales as 1/λ. The long x-driven tail of the flank
ACF (integral to 5 t.u.) shrinks much faster: 0.36 → 0.21 → 0.061 → 0.0077.

**MC2 — gate transport** (hysteresis side changes x < −0.5 ↔ x > 0.5 per walker per t.u.): 0.00612, 0.0125 (×2.05),
0.0226 (×3.70), 0.0322 (×5.27) at λ 1, 4, 16, 128. [identity] Along the curved channel x advances only as fast as y
follows m(x) (local mobility factor λ/(λ + m′²) = 0.029, 0.108, 0.327, 0.795).

**Consequence (preregistered).** λ changes gate transport at least as much as conditional relaxation; the design tests
**transverse mobility** and cannot by itself attribute an effect to conditional relaxation alone.

## 3. Results

### 3.1 Primary outcome — **A (strong rescue) at both N, in both families** [observed]

| family | N | R [95 % CI] | Holm p | FR better at higher λ (G_hi) | guards g1–g4 | label |
|---|---|---|---|---|---|---|
| **E1 λ 16 vs 1** | 512 | **0.65** [0.63, 0.78] | 0.0004 | yes (−0.30) | all hold | **A** |
| **E1 λ 16 vs 1** | 128 | **0.37** [0.29, 0.46] | 0.0004 | yes (−0.07) | all hold | **A** |
| E2 λ 128 vs 1 | 512 | 0.78 [0.66, 0.82] | 0.0004 | yes (−0.30) | all hold | A |
| E2 λ 128 vs 1 | 128 | 0.32 [0.27, 0.38] | 0.0004 | yes (−0.23) | all hold | A |

Combined across N (frozen rule): **"faster mobility rescues FR"** for E1 and for E2. Seed wins: 29/32, 29/32, 30/32,
31/32. The Holm-consistent 97.5 % CIs exclude 1 in every row.

### 3.2 Per cell: FR vs ABF (paired G = (FR − ABF)/ABF, median [95 % CI], FR wins)

| λ | N | ABF Ī_F | FR Ī_F | **G(Ī_F)** | G(final e_F) | G(Ī_F′_stat) | G(Ī_TV_half) |
|---|---|---|---|---|---|---|---|
| 1 | 512 | 0.0990 | 0.0949 | −7.1 % [−14.0, −0.0] (21) | **+45.9 %** [+31.3, +57.3] (1) | +15.4 % | −88.0 % |
| 1 | 128 | 0.0374 | 0.0734 | **+92.3 %** [+74.4, +141.4] (1) | **+1094 %** (0) | +128 % | −92.0 % |
| 4 | 512 | 0.0434 | 0.0476 | +10.1 % [+2.8, +20.6] (7) | +146 % (0) | +4.7 % | −84.3 % |
| 4 | 128 | 0.0140 | 0.0295 | **+125 %** [+102, +135] (0) | +811 % (0) | +96.8 % | −85.5 % |
| 16 | 512 | 0.0134 | 0.0088 | **−33.7 %** [−39.7, −29.9] (32) | −26.8 % [−37.6, −5.7] (25) | −34.5 % | −73.8 % |
| 16 | 128 | 0.00428 | 0.00347 | **−22.9 %** [−35.1, −7.1] (24) | +45 % [−9, +139] (11), n.s. | −18.4 % | −73.7 % |
| 128 | 512 | 0.00509 | 0.00325 | **−34.2 %** [−38.0, −30.2] (30) | −58.2 % [−68.9, −23.7] (25) | −30.4 % | −75.7 % |
| 128 | 128 | 0.00145 | 0.00101 | **−32.0 %** [−43.8, −23.3] (28) | +5.0 % [−39.2, +51.5] (16), n.s. | −32.1 % | −74 % |

* **Replication of Experiment I at λ = 1** (fresh seeds): N 512 −7.1 % (Exp I +2.0 % [−4.6, +7.8]), N 128 +92 %
  (Exp I +105 %); final error and marginal effects agree in sign and size. [observed]
* **Trend.** λ = 4 is *not* intermediate: FR is relatively worse than at λ = 1 at N 512 (T1 R = 1.17 [1.12, 1.28]);
  between 4 and 16 the relative performance changes sharply (T2 R = 0.57 at N 512, 0.33 at N 128); 128 vs 16 is flat
  (T3 R = 1.01 [0.88, 1.17] and 0.82 [0.68, 1.07]). [observed]
* **Secondary contrasts** (unadjusted) agree with E1/E2 on final e_F (R 0.52 and 0.095) and on Ī_F′_stat (0.58 and
  0.35). On the marginal, FR's *relative* gain shrinks with λ (TV R 2.2–3.4) because ABF's own marginal improves, but
  **HA3 holds in every cell**: FR improves the reaction-coordinate marginal at every λ and N (G −74 % to −92 %).

### 3.3 Convergence behaviour (figure A7) [observed]

At λ = 1 and 4, FR leads early and then **plateaus at an error floor while ABF keeps converging**: at N 128, λ 1, FR
stalls near e_F ≈ 0.06–0.07 while ABF reaches 0.005; at N 512 the curves cross near u ≈ 0.5 (λ 1) and 0.3 (λ 4). At
λ = 16 and 128 the floor is gone: FR's curve lies at or below ABF's for the whole budget at N 512, and the two meet at
the end at N 128. This is the same signature (FR floor, ABF converging) as the validated WCA negative
(`EVIDENCE_LEDGER.md` W8).

### 3.4 ABF itself, discovery and establishment [observed]

| ABF descriptor | λ 1 | λ 4 | λ 16 | λ 128 |
|---|---|---|---|---|
| Ī_F, N 512 (ratio vs λ 1) | 0.0990 | 0.0434 (0.44) | 0.0134 (0.136) | 0.0051 (0.051) |
| Ī_F, N 128 | 0.0374 | 0.0140 (0.38) | 0.00428 (0.115) | 0.00145 (0.039) |
| gate transitions per run, N 512 | 617 | 1085 | 1850 | 2578 |
| τ(TV_half ≤ 0.1), N 512 (u) | censored | censored | 0.21 | 0.27 |
| first arrival, N 512 (u) | 0.0108 | 0.0098 | 0.0093 | 0.0100 |

Mobility makes ABF 7–25× more accurate and lets it establish the marginal within the budget; discovery (first arrival)
is unchanged. The E decomposition shows FR improving even faster than ABF: from λ 1 to 16, ABF's Ī_F falls to 0.133
(N 512) / 0.121 (N 128) of its λ = 1 value, FR's to 0.092 / 0.044.

**Cost per successful mid threshold** (e_F ≤ 0.0055, oracle stopping): λ 1 — FR 0/32 at both N, ABF 0/32 (N 512) and
16/32 (N 128); λ 16 — both 32/32, FR needs 0.49× (N 512) and 0.70× (N 128) of ABF's walker-steps; λ 128 — both 32/32,
0.52× and 0.70×.

## 4. Mechanism read-outs [observed; attribution = hypothesis]

**MC3 (young-clone manipulation check): passes at every λ.** The matched excess of deposits from walkers cloned
0.01–0.1 t.u. earlier over ABF's deposits at the same bins: 0.141 / 0.149 (λ 1, N 512 / 128) → 0.062 / 0.040 (λ 4) →
0.004 / 0.000 (λ 16) → 0.000 / 0.001 (λ 128); the change from λ 1 has Holm p 0.0004 everywhere.

**Bias by clone age** (debiased RMS deviation of class deposits from the bin-averaged F\*′, end snapshot):

| λ, N | FR [0,0.01) | [0.01,0.1) | [0.1,1) | ≥ 1 t.u. | never cloned | FR all | ABF all |
|---|---|---|---|---|---|---|---|
| 1, 512 | 0.216 | 0.254 | 0.259 | 0.168 | 0.315 | **0.191** | **0.084** |
| 1, 128 | 0.144 | 0.160 | 0.159 | 0.127 | 0.313 | **0.139** | **0.010** |
| 16, 512 | 0.010 | 0.010 | 0.007 | 0.003 | 0.044 | **0.0045** | **0.0062** |
| 16, 128 | 0.000 | 0.002 | 0.003 | 0.002 | 0.059 | **0.0022** | **0.0016** |

At λ = 1 FR's deposits are biased in **every** age class — not only the youngest — and the gate-region squared-bias
excess over ABF is carried mostly by clones of age 0.1–1 t.u. and ≥ 1 t.u. on the entry side of the gate (descriptive
decomposition, `mech.md`). At λ ≥ 16 the excess vanishes in every class and FR's all-deposit bias is at ABF's level
(slightly below at N 512). **Genealogy is unaffected by λ**: windowed ancestor ESS/N 0.77–0.78 (N 512) and 0.57–0.58
(N 128), maximum lineage fraction 0.03–0.04 / 0.07, deaths per walker per t.u. 0.115–0.121 / 0.232–0.235.

## 5. Interpretation (against the frozen map)

* **[observed] Faster transverse mobility rescues FR on the shifted fibre.** The task-specified primary test (E1) gives
  outcome A at both N, and the extension (E2) agrees. At λ ≥ 16 FR lowers the integrated free-energy error by 23–34 %
  and halves the cost to the mid threshold — the size of the original gateway's gain (−30 to −41 %) — with no new
  genealogy or endpoint problem (g1–g4 hold).
* **[observed] The intervention is not conditional relaxation alone.** It also multiplies gate transport (×3.7 at
  λ 16) and makes ABF 7–9× better. By the frozen map, the rescue is attributed to transverse mobility; ABF's improved
  transport/establishment is a stated confound. What argues against a pure "ABF got easier" reading is that FR's
  relative performance improves (R 0.37–0.65) even though ABF itself establishes the marginal far better at high λ —
  i.e. in the regime where FR's establishment advantage should matter *less*.
* **[observed → hypothesis] The harm at low mobility is a conditional bias carried by FR's population.** At λ ≤ 4, FR's
  deposits are biased in every clone-age class, FR's error plateaus at a floor while ABF converges, and both disappear
  at λ ≥ 16, together with the young-clone excess (MC3 passes). This is *consistent with* the fibre-lag hypothesis in an
  extended form: score-directed selection keeps a population whose transverse coordinates lag the curved channel
  (m(x)) — through recent copies and through the slow, x-motion-coupled relaxation tail (flank integral 0.36 t.u. at
  λ 1) that only high mobility removes. Because the young-clone excess is a minority of the bias at λ = 1, "young
  clones alone" does not explain the harm; the bias is population-wide. Which part of the rescue is due to faster
  conditional relaxation and which to faster transport cannot be separated by this design.
* **[observed] Non-monotonicity at λ = 4.** Transport doubled and the young-clone excess halved, but the flank
  relaxation changed by only 14 %, and FR got relatively worse at N 512 (T1 R 1.17). The sharp change between λ 4 and 16
  coincides with the long flank tail collapsing (0.21 → 0.061 t.u.), not with the 1/e time. [hypothesis: the relevant
  relaxation for FR is the slow, x-coupled component, not the fast local one.]
* **[observed] Marginal improvement is not predictive by itself.** FR improves the marginal by 74–92 % in every cell,
  including the four cells where it worsens the free energy by up to +125 % (Ī_F) and +1094 % (final error).
* **Link to WCA [hypothesis].** The low-mobility FR floor here has the same phenomenology as WCA's validated FR floor
  (stationary tilt, ABF converging). It is plausible, but not tested, that the same population-wide conditional bias
  operates there; WCA is not re-run in this campaign.

## 6. What is established and what is not

* **Established** [observed, validated numerics]: on the shifted fibre, raising λ from 1 to 16 (or 128) turns uniform FR
  from neutral/harmful (−7 %/+92 % at λ 1; +10 %/+125 % at λ 4) into a 23–34 % improvement, with the preregistered
  primary test at outcome A at both N; FR's harm at low mobility comes with a population-wide transverse force bias and
  an error floor, both of which vanish at high mobility.
* **Not established**: that conditional relaxation (as opposed to transport) is the operative variable; why λ = 4 is
  worse than λ = 1 at N 512; whether the same mechanism explains WCA.

## 7. Figures

`figures/final_campaign/expA/`: A1 Ī_F vs λ; A2 final e_F vs λ; A3 FR/ABF ratio and contrasts; A5 deposit bias by
clone age and MC3; A6 marginal TV; A7 convergence curves; A8 genealogy; `validation/` A4 (MC1) and A4b (MC2); per-cell
A–F figures in `<cell>/N<N>/`. All passed the legibility check.

## 8. Compute, files, commits

* Used core-hours (threads × wall, from the ledgers): production 40.5; validation (G-A1 + λ = 1 replay) 15.2; smoke,
  checks, analysis ≈ 1.5; **total ≈ 57** of the 110 core-h Experiment A cap; 0 GPU-h.
* Commits: preregistration `c5a2264f`; tooling + Amendment 1 `014c4be0`; gate records + Amendment 2 `ef8ebc20`;
  production + analysis `1ec25ca5`.
* Limitations: one potential geometry (κ = 1); two replica counts; transport and relaxation co-vary with λ; the
  clone-age decomposition is descriptive.
