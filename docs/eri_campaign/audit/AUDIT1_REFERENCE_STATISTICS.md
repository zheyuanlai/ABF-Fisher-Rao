# AUDIT 1 — reference and statistics lens

*Independent adversarial audit, 2026-10-16 ~17:00 UTC. Read-only; no existing file modified, nothing committed, CPU
only (2 threads). Scratch code: `/tmp/claude-1008/.../scratchpad/audit_refstats/` (`common.py` is an independent
re-implementation of WHAM / F_density / F_MF / jackknife that does not import `refanalysis`).*

Scope: `src/eri/reference.py`, `src/eri/refanalysis.py`, `scripts/eri_campaign/analyze_reference.py`,
`scripts/eri_campaign/reference_fine.py`, `src/eri/metrics.py`, and (because it appeared untracked during the audit
and implements A1.4) `scripts/eri_campaign/classify_screen.py`. Data read: Gate 1 MC, Gate 2 EM/LM per-job npz,
`reference_profile.npz`, `reference_fine.npz`, calibration and pilot runs, and the ABF-only screen
(`screen_dilute/`, 24 files, ABF only).

## Verdict in one paragraph

The **reference itself is sound**: the MC moves are symmetric and exact, the Rodrigues rotation is right, EM and LM
are implemented correctly, WHAM converges to the right fixed point, and I reproduce Gate 2 D for EM 2e-4 to every
printed digit. The 180-bin reweighting in `bead_profile` is harmless (0.0005 kBT). The defects are in the
**decision statistics**. The main one: the frozen establishment test A1.4-3(E) **cannot fire at K = 64**. Its
threshold is ≤ 0 for 87–97 % of [2, T] in all 8 screen seeds, and the longest stretch with a positive threshold is
1.1–5.1 t.u., shorter than the required 8 t.u. So "ABF-sufficient at K = 64" is decided by algebra, not by the data.
The conditional-lag test 2(b) has very low power at small K. The FR-blind second clause of Gate 2 passes on a thin
and partly mis-referenced margin. The paired comparisons in Gate 1 are debiased as if independent. None of these
reverse a verdict already recorded, but F1 and F2 decide what the screen is able to conclude.

## Findings (most severe first)

### F1 (HIGH) — A1.4-3(E) has zero power at K = 64 and needs an ~80 % deficit at K = 256
* `metrics.establishment` (l. 129) uses `thr = 0.5 N Q − 2 sqrt(N Q (1−Q))`, exactly as frozen. thr > 0 requires
  N Q/(1−Q) > 16. For N = 64 that means Q\*_W > 0.20, and the uniform share of the 28 window bins is 0.156.
* Algebra: at N = 64, Q\* = 0.156 gives thr = −0.81. At N = 256 thr = 8.4 against an expected 40 walkers (an 80 %
  deficit is needed). At N = 1024 thr = 56.6 against 160.
* Screen data (ABF only, 8 seeds per K), longest contiguous span in [2, T] with thr > 0, at K = 64:
  **1.3, 4.5, 2.7, 3.8, 1.3, 5.1, 1.9, 1.1 t.u.** The positive-threshold fraction of [2, T] is 3–13 %. The span the
  rule requires is 8 t.u. It is therefore impossible for any window trajectory, including zero walkers in the
  window for the whole run.
* Consequence: the K = 64 cell, the smallest N and the one most likely to be starved, is classified ABF-sufficient
  by construction. A report that "no K is establishment-limited" would overstate what the data show. The bias runs
  toward the null (no FR licensed).
* Fix (an amendment, before any classification is read): test a **time-pooled** deficit. One option: block sums of
  n_W over 1–2 t.u. against 0.5 N Q\*_W per save, summed, with the sd inflated by the measured autocorrelation of
  n_W. Another: a relative criterion n_W/(N Q\*_W) < 0.5 on a running mean with a power check from a binomial null
  simulation at each K. At minimum, record that (E) is non-informative at K = 64.

### F2 (MEDIUM) — conditional-lag test 2(b): very low power at small K; the floor is not from the reference accumulators
* `conditional_lag` returns d_η = sqrt(Σ_c w_c (η̄_c − η_ref,c)²). Under the null, d_η ≈ floor = sqrt(Σ_c var_c/N).
  Measured var(η|c) is 0.27–0.48 in cage bins. Observed median d_η/floor over [T/2, T] is **0.92–1.02** in the
  pilot (N 64/256/1024) and calibration (N 64/1024) runs I checked. That number is pure finite-N noise.
* The rule fires at d_η > 3·floor, i.e. a systematic RMS lag ≥ 2.83·floor: ≥ 0.18 at N = 1024 and **≥ 0.73 at
  N = 64**, while ⟨η|c⟩ itself only spans 0.43–0.98. Taking the median of per-save ratios over [T/2, T] does not
  average the noise down, because each save is compared with the single-save floor.
* The prereg says the floor uses "the reference conditional variances of η". The Gate 1 accumulators store no Σ η²
  (`ACC_KEYS` has `Meta` only). `classify_screen.hidden_stats` instead takes var(η|c) from the molecule-0 time series
  of the **EM 2e-4** umbrella runs, pooled over umbrella-weighted samples. That source is an undocumented
  substitution (dt-biased, not MC).
* Fix: test the time-averaged, floor-debiased lag (mean over [T/2, T] of d_η² − floor², with a block-bootstrap
  CI). Store Σ η² in the reference, or document the ts-based substitute in an amendment.

### F3 (MEDIUM) — the FR-blind second clause (A1.1) passes on a fragile, partly mis-referenced margin
* A1.1 quotes e_F(T) = 0.129 at N = 1024 (3 seeds). Recomputed: **0.1273 with the frozen primary reference**
  (`reference_on_grid`) and 0.1291 with `reference_smooth`. The quoted number is the smooth-reference one.
* The three seeds give 0.058 / 0.127 / 0.130. On seed 0 alone the clause FAILS (⅓ × 0.058 = 0.019 < 0.037).
* The 8 screen seeds at K = 1024 give median e_F(T) 0.1245, so ⅓ is 0.0415 against D_MF 0.0367: it still passes,
  with a 12 % margin. The D upper95 of 0.059 exceeds 0.0415, so dt bias up to ~47 % of e_F is not excluded.
* Consequence: scored against MC, up to ~5 % of e_F at K = 1024 is a common dt offset in both arms
  (sqrt(0.124² − 0.037²) = 0.119), which dilutes relative arm differences.
* Fix: restate the clause with the primary reference and the 8-seed screen median. Make the secondary
  (dt-consistent) scoring co-primary for any claim that relies on relative e_F differences below ~10 %.

### F4 (LOW) — `compare()` treats same-data comparators as independent (Gate 1 checks 2 and 4)
* `compare` uses noise² = mean(seA² + seB²) and bootstraps A and B with **independent** group indices
  (`refanalysis.py` l. 182, 190–191). Check 2 compares F_density and F_MF built from the *same* groups.
* Their per-bin jackknife correlation is 0.67, and the paired jackknife noise is **0.0055 against the 0.0084
  used**. D is debiased to 0.000 (paired: 0.0014), and the bootstrap upper bound is biased low by ≈ 2·cov.
* Check 4 (first vs second half of the same molecules) is barely affected: paired noise 0.0158 against 0.0150.
* Verdicts unchanged, because the tolerance is 0.05.
* Fix: when A and B share groups, use the jackknife of the difference and a common bootstrap index.

### F5 (LOW) — Gate 2 conditional-observable check is not the preregistered statistic
* The prereg asks for RMS z-scores of (u_x², u_z², η, bond) per bin. The code reports mean |Δ| and a bond-variance
  ratio. z-scores cannot be computed because no per-bin second moments of η, u_x² or u_z² are stored.
* The reported bond-variance error at the frozen dt, **+19.2 %**, is exactly the EM harmonic artefact
  1/(1 − k_b dt) = 1/(1 − 802.3 × 2e-4) = 1.191 for the relative coordinate. It is real dt distortion of the
  intramolecular distribution that D does not see.
* Fix: store second moments, or amend the reporting rule. State the bond artefact as a known production-dt
  distortion.

### F6 (LOW) — the A1.2 bin-ladder "plateau" is reported for N = 1024 only
* Recomputed with reference_smooth on all grids (3 seeds each):
  * N = 1024, 180 → 360: e_F(T) −4.6 %, Ī_F −0.9 %. This matches the quoted figures.
  * N = 64, 180 → 360: **e_F(T) −13.5 %, Ī_F +4.9 %**.
  * N = 64, ng 90: e_F(T) 0.150 against 0.242 at ng 180.
* With 3 seeds these are noisy, but the plateau statement is N = 1024-specific.

### F7 (LOW) — Q\* ignores the non-conservative part of the applied bias
* `applied_bias_potential` integrates ramp·w·Γ with its circular mean removed. The real applied force has a
  circulation of 2π⟨ramp·w·Γ⟩. In the N = 64 calibration run (`calib_bins/abf_N64_ng180_s0`) it is −13.2 kJ/mol
  (−2.6 kBT) at t = 2 t.u. and −0.43 kBT at T. At N = 1024 it is ≤ 0.35 kBT.
* The exact stationary law of the tilted periodic diffusion differs from Q\* by **TV 0.095** at t = 2 (Q\*_W 0.323
  against 0.288) at N = 64. The difference is negligible at N = 1024 and late.
* Effect: the early-time TV(p̂, Q\*) and the (E) target at small N.
* Fix: compute Q\* with the Stratonovich formula for the tilted 1-D periodic diffusion, or at least report the
  circulation.

### F8 (LOW) — the test pipeline that decides the campaign has no unit tests
* There are no tests for `refanalysis` (wham, compare, bead_profile), for `metrics` (establishment, tv_floor,
  applied_bias_potential, conditional_lag), for `reference.run` (an MC detailed-balance or LM/EM toy), or for
  `classify_screen.py`.
* F1 is exactly the class of defect that a null/power test of the classifier would have caught.

### F9 (LOW) — small bookkeeping defects
* (a) `applied_bias_potential` silently ignores `bias_clip`, which the engine supports. It is frozen off today;
  add an assert.
* (b) The span loop in `establishment` (l. 139–144) counts the interval *before* the first under-established save,
  including time before `t_from`. That biases each span up by one save interval (0.1 t.u.).
* (c) Region statements:
  * The cage core C = {|φ| ≥ 2} has max βF **0.442**, against the stated "within 0.4 kBT".
  * The window region on the grid is |φ| < 0.489 (28 bins), while the engine counters use the exact |φ| < 0.5.
  * (I) uses 28/180 = 0.156 rather than |W|/2π = 0.159.
* (d) The `reference_on_grid` docstring promises a density route for grids that do not divide 180. The code raises
  instead and names a non-existent `reference_density_on_grid`. The `metrics` docstring cites "Amendments 1–2",
  but only Amendment 1 exists.

### F10 (LOW) — secondary reference noise is half the bias it is meant to isolate
* The EM-2e-4 umbrella F_MF has jackknife noise **0.0174 kJ/mol**, 3.8× the primary's 0.0046. rms(EM_MF − MC_MF)
  is 0.0405, of which the bias is ≈ 0.036. The secondary reference therefore resolves the dt bias at about 2σ and
  cannot "isolate" statistical error below ~0.017.

### F11 (INFO) — the literature observable's folding Jacobian is larger than the band
* The Voronoi-folded |s| has a geometric measure that falls linearly to zero at the cage axis (Monte Carlo over the
  cell: 1.00 → 0.245 at the minimum bin 45 → 0.006 at bin 60).
* That alone contributes **−1.41 kBT** to βF(bin 0) − βF(bin 45). The band is ±0.35.
* The two computed conventions happen to agree: |s| 5.98, X-projection 6.13. Schlenker is out of the band under
  both (5.29 / 5.44), so the framework choice is robust. "Reproduces 6.13" is still convention-conditional.

## Checked and found correct (with evidence)

* **MC detailed balance.**
  * Bead and translation moves are symmetric cubes; the bead is chosen with p = ½.
  * The rotation draws a uniform axis by normalised Gaussian and θ ~ U[−d, d]. The kernel is conjugation- and
    inversion-invariant, hence symmetric on S², and the rotation is about the COM (equal masses).
  * Rodrigues is v cos θ + (k×v) sin θ + k(k·v)(1 − cos θ), coded correctly at l. 279–285.
  * Acceptance is exp(−β Δ(U + W)) in consistent kJ/mol units, and f is recomputed at sample time in MC
    (l. 163–165).
* **Umbrella.** W = κ/2 wrap(φ − c)². The force on each bead is −κ d·π/P = −∂W/∂φ · ∂φ/∂x_b. WHAM uses the same
  wrap convention.
* **EM / LM.** LM is the Leimkuhler–Matthews "BAOAB-limit" form x + h F + sqrt(2hkT)(R_n + R_{n+1})/2. Its memory is
  initialised with an independent normal (l. 148–150), and its bond variance error of +0.04 % confirms it.
* **WHAM.**
  * An independent Newton solver reproduces F_density to 2.4e-9 and F_MF to 7e-15.
  * The 180-bin bias-at-centre approximation in `bead_profile` shifts the φ marginal by ≤ 0.0017 kBT and the |s|
    barrier by **−0.0005 kBT** (5.9810 → 5.9805; groups 0–3: 5.9620 → 5.9615).
* **Gate 2 D for EM 2e-4.** Reproduced independently: D_density 0.021952 (repo 0.021952), D_MF 0.036655 (repo
  0.036655). Upper95 with 300 resamples: 0.0465 / 0.0558 (repo with 2000: 0.0477 / 0.0588).
* **Debiasing.** For independent comparators the bootstrap variance is 0.90–0.95 × the jackknife variance, so
  subtracting 2·noise² slightly over-subtracts: the upper bound is anti-conservative by < 0.001 kJ/mol, which is
  negligible.
* **Checks 1–6 and Gate 2 criteria.** The thresholds and profiles match PREREGISTRATION §3–4 (check 2 alone carries
  the upper bound; the η hard condition uses mean |Δ|, the stricter reading). Exceptions are F4 and F5.
* **Metrics.**
  * `applied_bias_potential` equals the engine's applied bias: same k, ramp min(1, k/warm), trust min(1, C/200),
    and Γ = 0 in empty bins.
  * `tv_floor` is exact for N iid draws, by linearity and binomial marginals.
  * The P0-compatibility of `reference_on_grid` holds: rms vs the smooth reference is 0.0042, about its noise of
    0.0046.
  * The count-weighted merge at ng = 90 differs from the unweighted merge by 0.0002.
  * The coarse-bin reweighting systematic in `conditional_lag` is 0.0004–0.008, against a floor of 0.065.
