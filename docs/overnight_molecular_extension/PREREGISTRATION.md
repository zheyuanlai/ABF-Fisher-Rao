# Overnight molecular extension — preregistration

*Frozen 2026-10-16 17:57 UTC (commit 41b5684e; an earlier draft line said "18:25", a clock error corrected here) for candidate A′. Amendments are appended below and never edit the text above them.
**Two adaptive arms only**: histogram ABF (arm A) and the identical histogram ABF + uniform marginal FR (arm B). No
sham, oracle, OPES or other algorithmic arm. Reference calculations (MC, umbrella/WHAM) are not arms.*

**Seen before freezing:**
* the literature reports (`review/`);
* the ITE build: 768 Si / 1536 O, coordination OK;
* single-bead percolation: 1-D channels along b;
* Gate 0 (5 tests pass);
* an MC umbrella pilot at 300 K (36 windows, κ 300, 2 molecules × 20k steps): βΔF ≈ 15.8 on φ, ΔU ≈ +20.7 kJ/mol;
* a correlation-time pilot: τ_int(η) ≈ 0.87 t.u. in the cage and ≤ 0.06 t.u. on the flank / window (EM dt 1e-4).

No ABF or FR run of ITE exists.

## 1. System A′ (labelled model, NOT a reproduction of Combariza et al. 2011)

| item | value | source |
|---|---|---|
| framework | rigid Si-ITE, ITQ-3 experimental coordinates (Camblor et al. 1997; RASPA2 `ITQ-3.cif`, sha256 in `cache/eri/framework_ite_itq3.json`), Cmcm a 20.622, b 9.7242, c 19.623 Å | IZA type material; RASPA2 |
| box | X = 3b, Y = 2c, Z = 2a = 29.17 × 39.25 × 41.24 Å; 768 Si, 1536 O | `scripts/overnight/save_ite.py` |
| guest | united-atom ethane, two CH3 beads, harmonic bond k/kB 96 500 K/Å², r0 1.54 Å | Dubbeldam et al. JPCB 108, 12301 (2004), Table 3 |
| host–guest | CH3–O 93.00 K / 3.48 Å, oxygen only | same; PRL 93, 088302 Table I |
| guest–guest | CH3–CH3 108.00 K / 3.76 Å, intermolecular only (loaded cases) | same |
| cutoff | 12 Å, energy-shifted, no tail | JPCB Eq. 14 |
| T | 300 K | fixed in `CANDIDATES.md` before any computation |
| CV | φ = wrap(2π (X_com − x_w)/b), X = crystal b; x_w = 4.1979 Å, the mean X of the 8-ring O atoms; windows at φ = 0, cages at ±π. Linear, so no geometric term; local mean force f = −(b/2π)(F_0x + F_1x) | `scripts/overnight/save_ite.py`, Gate 0 test |
| channel family | every molecule starts in, and stays in, the translation-related channel family A (axis Y = c/4 + m c, Z = n a). The other three families are related to A by C-centring (windows shifted b/2) or the c-glide (y reversed). They are not mixed. Ethane cannot change channel (single-bead percolation in Y needs > 100 kJ/mol, and Z never percolates; Gate 0 test). This is checked in every run (§6) | – |
| hidden-variable diagnostics | η = cos(2π Z_com/a) (lateral position in the cage, +1 on the axis); u_x² (alignment with the channel) | – |

## 2. Gate 1 — exact MC reference

* **Design:** the ERI design with new seeds. 40 windows at linspace(−π, π, 40, endpoint=False), κ 300 kJ/mol/rad².
  16 groups, each with 16 molecules per (group, window), 5000 MC pre-steps, 5000 burn-in, 200 000 MC steps sampled
  every 5. Moves d_bead 0.10 Å, d_trans 0.30 Å, d_rot 0.60 rad. Seed 8 100 000 + 1000 g + w.
* **Estimators and D statistic:** as in the ERI prereg §3 (fine WHAM, pooled mean force, group jackknife, 2000
  group-bootstrap resamples).
* **Checks (PASS required):**
  1. min pooled fine-bin count ≥ 200;
  2. D(F_density, F_MF) ≤ 0.03 kJ/mol, upper ≤ 0.045;
  3. split-half D ≤ 0.03;
  4. first-half vs second-half D ≤ 0.03;
  5. |circular mean of Γ| < 3 se;
  6. the noise is reported.

  The tolerance is the LTA 300 K validation tolerance, 0.03 kJ/mol = 0.012 kT.
* **Decomposition:** U(φ) and −TS(φ) = F − U. The barrier is the mean over the 4 bins straddling φ = 0 minus the
  5-bin neighbourhood of the minimum.

## 3. Gate 2 — timestep

* **Design:** the same windows, 16 groups × 16 molecules, 5000 MC pre-steps, 1 t.u. burn-in, 32 t.u. per molecule,
  samples every 1e-3 t.u. Seed 9 100 000 + 1000 g + w.
* **Candidates:** EM 2e-4, EM 1e-4, EM 5e-5, LM 2e-4.
* **ADMISSIBLE:** D(F_density, MC) ≤ 0.03 kJ/mol with upper ≤ 0.045, AND D(F_MF, MC) ≤ 0.03 with upper ≤ 0.045,
  AND mean |Δ⟨η|φ⟩| ≤ 0.02.
* **Production dt** = the largest admissible EM dt. Second clause (FR-blind): D_MF ≤ ⅓ × median ABF-only e_F(T) at
  the frozen Gate 3 configuration (all K of the screen). If it fails, take the next smaller admissible dt.

## 4. Gate 3 — histogram ABF calibration (ABF only)

* **Estimator:** fixed as in ERI: P0 on 180 bins, all deposits from step 0, own-bin force, per-bin trust ramp
  min(1, C_j/200), global ramp, no clip, pre-equilibration with unbiased EM.
* **To be selected on ABF-only runs only (K 64 and 1024, 3 seeds, fresh seeds 940 000+):**
  * **(a) horizon T.** The smallest T in {50, 100, 200} t.u. at which the median ABF e_F(T) at K 1024 is within
    1.5× of its value at 2T, or 200 if none.
  * **(b) ramp and pre-equilibration.** Ramp {1, 4} t.u. × pre-eq {0, 5} t.u. Minimise the mean over K of the
    normalised median Ī_F, with a 5 % tie broken toward the shorter pre-eq, then the shorter ramp. This is the ERI
    rule, written before any ITE ABF data. Audit F6 noted that the ERI rule weighted K 64 equally; for ITE the K 1024
    weight is doubled (weights 1 : 2), because FR is relevant at large K.
  * **(c) bin ladder {90, 180, 360}.** Plateau rule: 180 is kept unless 360 lowers the median e_F(T) at K 1024 by
    more than 10 % AND the P0 floor of 180 exceeds 10 % of e_F(T).
* **Error reference:** MC F_MF on the run's grid (180; 90 by merging; 360 from the smooth Fourier fit), as in ERI.

## 5. ABF-only screen and classifier

* **Screen:** K ∈ {64, 256, 1024}, 8 seeds each (seed 950 000 + s), frozen Gate 3 configuration.
* **Regions:** window W = {|φ| < 0.5} and cage core C = {|φ| ≥ 2.0}. Both are verified from the Gate 1 profile before
  the screen. If W does not satisfy βF ≥ βF_max − 1.5 or C does not satisfy βF ≤ βF_min + 1.5, the region thresholds
  are re-derived from the reference with exactly those rules, still before the screen.
* **Classifier:** ERI Amendment 2, verbatim.
  1. **Discovery:** median first true translocation > 0.1 T, or any seed without one by 0.5 T → DISCOVERY-LIMITED.
  2. **Conditional:**
     * τ_int(η) ≥ 0.1 T, or
     * a time-averaged lag (η or u_x²) > 3 × its iid floor

     → CONDITIONAL-LIMITED.
  3. **Establishment:** median longest run of deficient 2 t.u. blocks (block-mean occupancy / Q\* < 0.75 AND
     deficit > 3 sd_null) ≥ 0.2 T, AND (I) [window deposit ratio < 0.5 OR window share of the final squared MF error
     ≥ 0.5] → ESTABLISHMENT-LIMITED, eligible for FR.
  4. Otherwise ABF-SUFFICIENT.

  Block length and n_eff use t.u.; for ITE the block is 2 t.u. if T ≤ 50 and 0.04 T otherwise (≥ 25 blocks). The
  null control and the power test are re-run at the ITE block length before the screen.
* **Secondary, reported for every K:** T_hit (first window entry), T_est (end of the last deficient block), the
  uniform-target occupancy, TV(p̂, Q\*) minus its floor, the late-deposit ratio, and the clone decorrelation ratio
  τ_hidden / mean interval between replacement events (FR arm only).

## 6. Two-arm FR protocol (applies only to an ELIGIBLE cell; frozen here, before any FR run)

* **FR mechanism:** the validated LTA-ladder machinery unchanged. Uniform target on the circle, KDE 0.10 rad, score
  clip ±2, cap max(1, ⌊0.02 K⌋) per opportunity, whole-replica cloning with label inheritance, independent noise per
  slot after cloning.
* **Schedule (physical time):**
  * FR opportunities every 0.001 t.u. (fr_every = 0.001/dt);
  * FR starts at the end of the bias ramp (fr_start = ramp);
  * the genealogy window is 0.05 T.
* **Rate, safety-only:**
  * Ladder {0.05, 0.1, 0.2, 0.5, 1, 2} per t.u., at the eligible K, 3 calibration seeds (960 000+), ABF+FR arm only,
    over the full horizon T.
  * Selected rate: the LARGEST rate whose median over seeds satisfies, at every save after fr_start:
    * windowed ancestor ESS/K ≥ 0.30;
    * windowed max lineage share ≤ max(0.05, 3/K);
    * total replacements ≤ 2 K per t.u.
  * The calibration script must not compute any free-energy, mean-force or marginal-error quantity.
* **Production:**
  * fresh paired seed blocks; exploratory 8 seeds (970 000+), then confirmatory 16 seeds (980 000+);
  * arms A/B share the IC and Langevin streams slot by slot and are bitwise identical until the first FR event
    (tested).
* **Primary endpoint:** paired per-seed ΔĪ_F = Ī_F(B)/Ī_F(A) − 1, with Ī_F = I_F/T against the Gate 1 MC reference.
  * Median and 95 % percentile-bootstrap CI over seeds (10 000 resamples).
  * The number of seeds where B < A.
  * The useful-effect threshold is −10 %, with the CI excluding 0. One family (one cell), so there is no
    multiplicity correction. If several K are eligible, Holm over K.
* **Secondary endpoints:**
  * Δe_F(T), Δe_F′(T), ΔTV;
  * Δ(late-deposit e_F), i.e. deposits after the fr_start;
  * true translocations;
  * T_est;
  * ancestor ESS / max lineage;
  * replacement fraction;
  * conditional lag;
  * force evaluations (identical by construction);
  * wall-clock per arm (reported, not a claim).
* **Long-time correctness:** pooled final accumulators of each arm vs MC (D statistic). The FR limit vs the ABF limit
  must have D ≤ 3 × noise, otherwise it is flagged as a stationary tilt.
* **Claim types kept separate:**
  1. fixed-K finite-time accuracy (this endpoint);
  2. equal force evaluations with K free (only if (1) is positive: K-ladder at fixed B = K·n_steps, both arms, best
     allocation compared with a bootstrap over seeds that re-selects K);
  3. wall-clock (not claimed);
  4. long-time correctness (as above).

---

## Amendment 1 — Gate 1/2 verdicts and the Gate 3a horizon (2026-10-16 18:50 UTC; ABF-only data only, no FR run exists)

* **Gate 1 (MC, 16 groups, 2000 bootstrap resamples): PASS on every check.**
  * Overlap: min fine count 186 793.
  * D(F_density, F_MF) = 0.0099 [0.012].
  * Split-half: 0.0047 [0.011].
  * Equilibration: 0.000 [0.0095].
  * Closure: 0.0002 ± 0.0008.
  * Noise: 0.004 kJ/mol.
  * Barrier: 39.70 kJ/mol = **15.92 kBT**. ΔU = 20.61 and −TΔS = 19.09 kJ/mol, so **48 % entropic**.
* **Gate 2:**

  | candidate | D_density [upper] | D_MF [upper] | verdict |
  |---|---|---|---|
  | EM 2e-4 | 0.0357 [0.051] | 0.0301 [0.039] | **NOT ADMISSIBLE** (D_density > 0.03) |
  | EM 1e-4 | 0.0148 [0.030] | 0.0129 [0.021] | **ADMISSIBLE** (bond var +8.8 %) |

  **Production dt = 1e-4.** LM 2e-4 and EM 5e-5 are reported in VALIDATION.md.
* **Gate 3a horizon rule:** at K 1024 (3 seeds; ramp 1, pre-eq 5), the median e_F is 0.0625 / 0.0457 / 0.0344 /
  0.0234 at 50 / 100 / 200 / 400 t.u. The ratio e_F(50)/e_F(100) = 1.37 ≤ 1.5, so **T = 50 t.u.** (500 000 steps).
  * The dt second clause is provisionally met: D_MF 0.0129 ≤ ⅓ × 0.0625 = 0.021. It is rechecked on the screen.
  * The establishment block length is 2 t.u. (T ≤ 50).
* **Gate 3b launched:** ramp {1, 4} × pre-eq {0, 5} t.u. at T 50, K 1024 and 64, 3 seeds (941 000+). The rule is
  §4(b) with weights K 1024 : K 64 = 2 : 1.

## Amendment 2 — Gate 3 frozen; screen launched (2026-10-16 18:38 UTC; ABF-only data only)

* **Gate 3b** (`results/overnight/ite/calib_warm/warm_choice.json`): weighted normalised Ī_F (K 1024 : 64 = 2 : 1)

  | ramp, pre-eq | score |
  |---|---|
  | 1, 0 | 1.000 |
  | **1, 5** | **0.914** |
  | 4, 0 | 1.324 |
  | 4, 5 | 1.192 |

  → **ramp 1 t.u., pre-eq 5 t.u.**
* **Gate 3c:** the P0 floor of 180 bins is 0.0046 kJ/mol, ≈ 5 % of e_F(T) at K 1024. The switch condition (floor
  > 10 % AND 360 better by > 10 %) cannot hold, so **180 bins**. The 90/180/360 ladder runs are reported anyway.
* **Frozen ABF configuration:**

  | knob | value |
  |---|---|
  | dt | 1e-4 |
  | T | 50 t.u. |
  | bins | 180 |
  | trust | 200 |
  | ramp | 1 t.u. |
  | pre-eq | 5 t.u. |
  | saves | every T/400 |

  Regions verified on the reference: W bins βF ≥ 14.69 (needs ≥ 14.43); C bins βF ≤ 1.04 (needs ≤ 1.5).
* **Null control and power** of the establishment test at T 50, 2 t.u. blocks, threshold 10 t.u.:
  * FP 0/300 at K 64/256/1024.
  * Power for a 12 t.u. deficit starting at t = 2:

    | K | 30 % | 40 % | 50 % | 80 % |
    |---|---|---|---|---|
    | 1024 | 54 % | 100 % | – | – |
    | 256 | 30 % | 84 % | 100 % | – |
    | 64 | – | 3 % | 18 % | 93 % |
* **Screen launched:** K 1024 / 256 / 64 × 8 seeds (950 000+).
* **FR job stages written before any classification:** `scripts/overnight/ite_jobs.py` `frcal` / `prod`, exactly §6.
  They run only for an ELIGIBLE K.

---

## Amendment 3 — second cell A′-150: the same ITE model at 150 K (2026-10-16 ~18:50 UTC, before any 150 K computation)

**Result that triggered it.** The A′ (300 K) screen was classified ABF-sufficient at K 64/256/1024. FR is not licensed
there, and no FR run of A′-300 will be made.

**Why this cell.** The 2026-10-16 instruction (§11) allows another temperature when resources permit and the model
stays valid.
* The Hamiltonian is unchanged; only β changes.
* The repo's validated ethane/LTA results used the same O–CH3 pair (93 K / 3.48 Å). That system was most clearly
  establishment-limited at 150 K (FR −28.5 % Ī_F against exact MC, `docs/numerical_validation/LTA_VALIDATION.md`).
* A′-150 therefore tests whether that regime appears in a 1-D 8-ring channel system at the same temperature.
* Prediction (written now, without data):
  * the barrier in kBT roughly doubles to about 25–30;
  * ABF discovery slows;
  * the classification could be discovery-limited, establishment-limited or ABF-sufficient. No direction is assumed.

**Protocol.** Identical to sections 2–6 at T = 150 K, with these differences:
* Seeds: MC 8 200 000 + 1000 g + w; Langevin 9 200 000 + 1000 g + w; ABF calibration 943 000+ / 944 000+ / 945 000+;
  screen 951 000+; FR calibration 961 000+; production 971 000+ (explore) and 981 000+ (confirm).
* Umbrella design: chosen by a short MC design pilot (recorded below before the full reference runs). Requirements:
  min fine-bin count ≥ 200 and adjacent-window overlap. Window count and κ may change; tolerances may not.
* Gate 2 run length per molecule: 64 t.u. (twice 300 K, since D ∝ kT).
* Every gate tolerance and every classifier threshold is unchanged. The horizon candidates are {50, 100, 200} t.u.
  (else 200), and the Gate 3a horizon runs are 400 t.u. long.
* Results directory: `results/overnight/ite150/`.

**Budget estimate.** Gates 1–3 plus the screen ≈ 45–60 core-h. FR calibration plus explore and confirm blocks, if
eligible, ≈ 35–45 core-h. The total stays under the remaining 143 core-h.
* If eligible, the equal-budget ladder (§6, claim type 2) runs only if ≥ 25 core-h remain after the confirmatory
  block.
* Otherwise it is reported as not done.

**A′-150 design pilot (recorded before the full reference runs):**
* Pilot: 64 windows, κ 250, 2 molecules × 20k MC steps. Acceptance: bead 0.64, translation 0.46, rotation 0.72;
  min fine count 194; βΔF ≈ 23.9.
* **Frozen:** 64 windows, κ 250 kJ/mol/rad², d_bead 0.07, d_trans 0.20, d_rot 0.40. 16 groups × 16 molecules ×
  200 000 MC steps. Langevin runs 64 t.u. per molecule.
* Gate 2 candidates are EM 2e-4 and EM 1e-4. LM 2e-4 and EM 5e-5 run only if both fail (budget). Config:
  `configs/overnight/ite150/reference.json`.

## Amendment 4 — A′-150: Gate 1/2 verdicts and horizon (2026-10-16 ~19:58 UTC; ABF-only data only)

* **Gate 1 (MC, 64 windows, κ 250, 16 groups): PASS on every check.**
  * Min fine count: 298 429.
  * D(F_density, F_MF): 0.0103 [0.0115].
  * Split-half: 0.000 [0.0034].
  * Equilibration: 0.000 [0.0040].
  * Closure: 0.0013 ± 0.0007 (< 3 se).
  * Noise: 0.0023 kJ/mol.
  * Barrier: **29.97 kJ/mol = 24.03 kBT**. ΔU = 20.16 and −TΔS = 9.81 kJ/mol, so **33 % entropic**. ΔU is
    unchanged from 300 K (20.61); the entropic part halves with T, as expected for a T-independent ΔS.
* **Gate 2:**

  | candidate | D_density [upper] | D_MF [upper] | bond-variance error | verdict |
  |---|---|---|---|---|
  | EM 2e-4 | 0.0161 [0.024] | 0.0168 [0.022] | +19.1 % | **ADMISSIBLE** |
  | EM 1e-4 | 0.0026 [0.011] | 0.0107 [0.016] | – | admissible |

  **Production dt = 2e-4** (the largest admissible EM dt). LM / 5e-5 were not needed.
* **Gate 3a horizon:** at K 1024 (3 seeds), the median e_F is 0.140 / 0.0815 / 0.0493 / 0.0352 at 50 / 100 / 200 / 400
  t.u. The ratios are 1.72 / 1.65 / 1.40, so **T = 200 t.u.** (1 000 000 steps). The establishment block is
  0.04 T = 8 t.u., and the deficient-run threshold is 0.2 T = 40 t.u.
* **dt second clause (provisional):** D_MF 0.0168 ≤ ⅓ × 0.0493 = 0.0164? It **fails narrowly** on the horizon-run
  median. The frozen rule says it is checked on the screen median e_F(T) at the frozen Gate 3 configuration. If it
  still fails there, the next smaller admissible dt (1e-4) is taken and Gate 3 plus the screen are re-run at 1e-4.
* **P0 floor** at 180 bins: 0.0040 kJ/mol.
* **Gate 3b launched:** ramp {1, 4} × pre-eq {0, 5}, T 200, seeds 944 000+.

## Amendment 5 — A′-150: the dt second clause fails → dt 1e-4 (2026-10-16 ~20:05 UTC; no 150 K screen and no FR data exist)

* **Gate 3b at dt 2e-4** (`results/overnight/ite150/calib_warm/warm_choice.json`): weighted scores

  | ramp, pre-eq | score |
  |---|---|
  | 1, 0 | 1.000 |
  | **1, 5** | **0.933** |
  | 4, 0 | 1.208 |
  | 4, 5 | 1.235 |

  → **ramp 1, pre-eq 5.** The K 1024 median e_F(T = 200) at that setting is 0.0411.
* **Second clause:** D_MF(2e-4) = 0.0168 > ⅓ × 0.0411 = 0.0137 → **FAILS**. By §3, the production dt becomes the
  next smaller admissible EM dt, **1e-4**: D_MF 0.0107 [0.016] and D_density 0.0026 [0.011]. The clause is checked
  again on the 1e-4 screen.
* **Deviation from the letter of §3 ("Gate 3 is repeated"), decided now and blind to any screen or FR result:**
  * The Gate 3a/3b choices (T = 200 t.u., ramp 1 t.u., pre-eq 5 t.u.) concern ABF dynamics in physical time. They are
    carried over to dt 1e-4.
  * They are verified by the Gate 3c bin ladder, which is run at dt 1e-4 (K 1024 / 64, 3 seeds, seeds 945 000+). Its
    180-bin K 1024 runs must give a median e_F(T) within ±25 % of the dt 2e-4 Gate 3b value (0.0411). If they do
    not, the full Gate 3 (3a, 3b) is repeated at dt 1e-4.
  * Reason: budget. Repeating 3a and 3b at 1e-4 costs about 35 core-h. The verification is part of 3c, which is
    needed anyway.
* **Budget consequence.** An eligible cell at dt 1e-4 can afford the FR safety calibration and the exploratory
  8-seed block (≈ 60 core-h). The 16-seed confirmatory block (≈ 60 core-h more) would exceed the 200 core-h ceiling.
  If eligible, the exploratory block runs, the confirmatory block is reported as not run, and no confirmatory claim
  is made.

## Amendment 6 — A′-150: Gate 3c at dt 1e-4; screen launched (2026-10-16 ~20:22 UTC; no 150 K screen or FR data exist)

* **Bin ladder at dt 1e-4, T 200, ramp 1, pre-eq 5** (3 seeds, 945 000+). References follow §4: MC F_MF for 90 and
  180 bins, the smooth Fourier P0-compatible reference for 360.

  | K | 90 bins | 180 bins | 360 bins |
  |---|---|---|---|
  | 1024 | 0.0380 | **0.0390** | **0.0348** |
  | 64 | 0.0565 | 0.0650 | 0.0724 |

  Median e_F(T); P0 floors 0.0158 / 0.0040 / 0.0010.
* **Amendment 5 consistency check: PASS.** The 180-bin K 1024 median at dt 1e-4 is 0.0390 against 0.0411 at 2e-4
  (ratio 0.95). The carried-over T / ramp / pre-eq stand.
* **dt second clause at 1e-4: PASS.** D_MF 0.0107 ≤ ⅓ × 0.0390 = 0.0130.
* **Bin rule (§4c): SWITCH to 360 bins.** The 180-bin floor is 10.2 % of e_F(T) (> 10 %), and 360 lowers the K 1024
  median by 10.7 % (> 10 %).
  * Both margins are thin. The per-seed (noise-paired) differences are +3 %, −6 % and −15 %. This is recorded but
    the rule is applied as written.
  * Every 360-bin read-out uses the smooth reference on the 360-bin grid (`metrics.reference_for_grid`). It differs
    from the MC on-grid reference by 0.004 kJ/mol RMS at 180 bins.
* **Frozen A′-150 configuration:**

  | knob | value |
  |---|---|
  | dt | 1e-4 |
  | T | 200 t.u. (2 000 000 steps) |
  | bins | **360** |
  | trust | 200 |
  | ramp | 1 t.u. |
  | pre-eq | 5 t.u. |
  | saves | every 0.5 t.u. |
  | establishment blocks | 8 t.u. (deficient-run threshold 40 t.u.) |

* **Screen:** K 1024 / 256 / 64 × 8 seeds, seeds 951 000+.

## Amendment 7 — A′-150 classified (2026-10-16 ~20:40 UTC)

* **Null control at the A′-150 block design** (8 t.u. blocks, 40 t.u. threshold, T 200), run before the
  classification:
  * FP 0/300 at K 64/256/1024.
  * Power for a 48 t.u. deficit:

    | K | 30 % | 40 % | 50 % |
    |---|---|---|---|
    | 64 | 17 % | 68 % | 96 % |
    | 256 | 42 % | 98 % | – |
    | 1024 | 88 % | 100 % | – |

* **Screen** (`results/overnight/ite150/screen/classification.json`): **ABF-SUFFICIENT at K 64 / 256 / 1024.**
  * Discovery: median first translocation 9.0 / 6.2 / 4.5 t.u. (threshold 20).
  * Establishment: longest deficient run 8 t.u. (one block; threshold 40); median block ratio 0.96 / 1.00 / 0.99.
  * Information: window deposit share 1.03 / 1.03 / 1.04; window share of the final squared MF error
    0.011 / 0.013 / 0.006.
  * Conditional lag / floor: η 1.50 / 1.27 / 1.77, u_x² 0.90 / 1.17 / 1.62 (threshold 3).
  * e_F(T) 0.072 / 0.040 / 0.041 kJ/mol.
* **dt clause on the screen:** D_MF 0.0107 ≤ ⅓ × min-K median e_F(T) = 0.0132. PASS.
* **Decision:** no FR run for A′-150. Every cell of this extension (ERI 600 K, ITE 300 K, ITE 150 K) is
  ABF-sufficient, so no ABF vs ABF + FR comparison was licensed.

---

## Amendment 8 — audit findings and an EXPLORATORY predictor-validation FR experiment (committed 2026-10-16 21:04:11 UTC as 88543cac — an earlier draft said "~21:20", a clock error — before any FR production or calibration run of any cell; FR calibration launched 21:04)

**Audit** (`review/AUDIT_OVERNIGHT.md`, independent, read-only). No classification changes. Every Gate 1/2/3 number
reproduced, and no walker ever left its channel family. Findings acted on:

* **F1 (high).** The frozen classifier fails a positive control.
  * Applied approximately to the repo's LTA 150 K ABF data (FR −28.5 % Ī_F against exact MC), it returns
    ABF-SUFFICIENT.
  * E′ fires there (21.6 t.u. against 12), but the information clause (I) fails (window deposit share 0.85, window
    error share 0.07).
  * So "ABF-sufficient" from this gate is **not** evidence that FR is unnecessary. Amendment 3's question cannot be
    answered by the gate.
* **F2.** The 150 K region check of §5 was not performed. Both regions fail the rule (W min βF 22.30 < 22.55; C max
  βF 2.17 > 1.5). Re-derived regions leave the label unchanged (audit recomputation). Recorded as a deviation.
* **F3.** Most of the "34–57 % early-phase I_F" accrues during the bias ramp, before FR can start. The share
  addressable after the ramp is 16–26 %. The HANDOFF recommendation is corrected.
* **F4.** The 150 K switch to 360 bins compared errors against two different references. With one common reference,
  360 improves on 180 by 7.1 % (< 10 %), so 180 would have been kept. The label is unaffected; recorded.
* **F5–F9:** wording, timestamp, reproducibility and horizon-dependence items, corrected in the docs.
  * F7: Amendment 3's text was written before the full 150 K MC launch (18:48) but committed at 18:49. The 150 K
    null control ran after the screen and before the classification.

**Why an FR experiment now.**
* The 2026-10-16 instruction (§11) asks whether ABF population establishment, information starvation, conditional
  relaxation and replica diversity **predict FR performance**.
* With the eligibility gate shown to give a false negative on the known positive, the only way to test the predictors
  is to measure FR on characterised cells.
* This is a **predictor-validation experiment, labelled EXPLORATORY**. It is NOT licensed by the frozen eligibility
  gate, and it makes no claim of the "FR beats ABF on a confirmatory, gate-licensed cell" type. The frozen
  classifications are unchanged.

**Design (frozen now; the §6 FR protocol, verbatim):**

| cell | system | K | config | FR calibration seeds | paired block seeds | pre-specified predictor values |
|---|---|---|---|---|---|---|
| X1 | ITE 300 K | 1024 | A′ frozen: dt 1e-4, T 50, ramp 1, pre-eq 5, 180 bins | 960 000–960 002 | 16 seeds, 980 000+ | E′ run 4 / 10 t.u.; early I_F share 52 %; window error share 0.009; conditional lag 0.87 / 0.78 |
| X2 | ITE 150 K | 256 | A′-150 frozen: dt 1e-4, T 200, ramp 1, pre-eq 5, 360 bins | 961 000–961 002 | 16 seeds, 981 000+ | E′ run 8 / 40 t.u. (LTA-like horizon: 9.6 / 12); early I_F share 47 %; window error share 0.013; lag 1.27 / 1.17 |

* X1 is the largest E′ ratio among our cells at the frozen horizons. X2 is where the audit found LTA-like deficits at
  an LTA-like horizon. Both are chosen by the predictors, not by any FR data.
* **FR rate:** the largest SAFE rate on {0.05, 0.1, 0.2, 0.5, 1, 2} by `scripts/overnight/fr_calibrate.py` (blind).
* **Block:** one paired 16-seed block per cell. The separate 8-seed exploratory block is dropped: the design is fixed,
  and the budget is better spent on n = 16.
* **Analysis:** `scripts/overnight/analyze_two_arm.py`.
  * Primary: ΔĪ_F, median with 95 % bootstrap CI and wins.
  * Secondary: Δe_F(T), Δe_F′, Δ late e_F, ΔTV, translocations, genealogy, pooled long-time limits.
* **Prediction from the predictors (recorded now):**
  * E′ never fires and (I) is far from firing in both cells → expect |ΔĪ_F| < 10 %.
  * X2 is closer to LTA's regime than X1 → if any cell shows an FR gain, X2 should show the larger one.
  * Both directions are reported.
* **Two arms only.** ABF and ABF + uniform FR share IC and noise streams slot by slot. Pairing is checked:
  bitwise-identical states before the first FR event.
* **Budget:** calibration about 11.7 (X1) + 4.7 (X2) core-h; blocks about 20.8 + 8.3 core-h; total ≈ 46 core-h,
  giving a cumulative ≈ 163 of 200.

**Amendment 8, FR rate calibration** (`results/overnight/ite/frcal/frcal_K1024.json`,
`results/overnight/ite150/frcal/frcal_K256.json`; blind script, no error metric read):
* **X1 (K 1024).** Every rate is safe. Min median windowed ESS/K is 0.97 / 0.93 / 0.87 / 0.73 / 0.57 / 0.31 for rates
  0.05 → 2 → **rate 2.0**.
* **X2 (K 256).** Rates 0.05–0.5 are safe. Rate 1 (ESS 0.20, lineage 0.070) and rate 2 (0.15, 0.090) are unsafe →
  **rate 0.5**.
* Paired blocks launched 21:14:33 UTC: X1 seeds 980 000–980 015, X2 seeds 981 000–981 015, both arms each.
