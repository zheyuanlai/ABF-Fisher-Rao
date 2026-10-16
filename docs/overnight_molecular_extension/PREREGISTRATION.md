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
