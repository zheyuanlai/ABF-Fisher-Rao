# WCA dimer replica ladder at equal force-evaluation budget, and the dt = 0.002 integrator problem

*2026-10-04. A companion to `docs/GATEWAY_REPLICA_LADDER.md`.*

* Engine: `src/wca_numba.py`, a numba CPU port of `run_sampler_gpu` with the histogram estimator and
  uniform FR. Tests are in `tests/test_wca_numba.py` (40 pass). The port was checked by three adversarial
  reviewers; all 11 minor findings were fixed.
* Runners: `scripts/run_wca_replica_ladder.py`, `run_wca_fr_dose.py`, `run_wca_unbiased_arbiter.py`,
  `run_wca_consistency.py`.
* Analyzers: `scripts/analyze_wca_replica_ladder.py`, `analyze_wca_arbiter.py`.
* Configs (each frozen before the data it governs): `configs/wca_replica_ladder/*.json`.
* Data: `results/wca_replica_ladder/`.

## 1. The port reproduces the accepted runs

Validation used the accepted confirmation seeds 3100-3115, N = 1024, 120000 steps, saves every 2500 steps.

| | ABF I_F | ABF e_F(T) | FR I_F | FR e_F(T) | FR effect, integrated | FR effect, final | FR replacements |
|---|---|---|---|---|---|---|---|
| accepted (GPU, float32) | 41.01 | 0.0893 | 31.26 | 0.0449 | -23.1 % [-28.6, -18.0] 16/16 | -50.5 % 16/16 | 3619 |
| numba, float32 emulation | 40.99 | 0.0912 | 32.35 | 0.0454 | -22.9 % [-26.6, -19.9] 16/16 | -49.8 % 16/16 | 3563 |
| numba, float64 | 41.74 | 0.0933 | 31.43 | 0.0441 | -24.1 % [-25.1, -23.4] 16/16 | -50.9 % 16/16 | 3554 |

The gate passes on all four clauses. The port review found that every accepted WCA histogram run stores
its counts in float32 and saturates bin 159 (the wall tail) at 2^24, unequally between the arms. The
float64 row shows this did not materially move the accepted numbers.

## 2. The dt = 0.002 ladder

Design: `configs/wca_replica_ladder/design.json`. B = 1024 x 120000; N = 1024, 256, 64, 16, 4 and 1;
16 seeds per N.

**Scored against the v2 TI reference (`scoreboard.md`).** FR wins 16/16 at every N from 16 to 1024
(integrated -27 to -46 %, final -43 to -52 %). ABF at N <= 64, serial included, floors at e_F ~ 0.075.
Taken at face value, this would make WCA the opposite of the gateway: FR helps at every N >= 16, and
serial ABF loses.

It cannot be taken at face value:

1. **Seed-pooled long-run profiles.**
   * ABF converges to one profile for every N from 1 to 256 (pairwise RMS 0.001-0.003).
   * ABF + FR (N = 16-1024) converges to a *different* profile: a smooth tilt with RMS 0.06-0.09. The
     tilt is stationary across time windows; the noise is 0.002-0.004.
   * The v2 TI reference lies between the two (0.076 from ABF's limit, 0.04 from FR's), so FR's tilt
     points toward it.
2. **Dose-response** (`fr_dose/`; N = 16, T = 3840, 16 seeds). RMS(FR - ABF) is 0.058 / 0.024 / 0.011
   at FR rates 0.1 / 0.03 / 0.01 (1194 / 373 / 122 replacements per run). The FR operator causes the
   tilt. Dose-response alone cannot say whether FR biases a correct ABF or corrects a biased one.
3. **A dynamics-consistent reference** (`reference/`; 32 independent serial ABF seeds, 160 and 640 bins
   agree to 0.0022). Against it, FR loses 0/16 at every N >= 16, including the accepted shape
   (`scoreboard_dynref.md`).
4. **Arbiter: plain unbiased MD** (`unbiased/`, `figures/fig_wca_arbiter.png`; 32 seeds x 1.2288e8 steps,
   F_density = -kT log P(z), noise 0.0008). It disagrees with *every* mean-force-based profile: ABF limit
   0.13, FR limits 0.17-0.20, v2 0.20. It also disagrees with the integrated mean force of the *same
   unbiased runs* (0.29). The estimator's entropic term -2w/(beta r) is correct for the 2-D distance CV,
   so the sampler's stationary measure is not Boltzmann.
5. **Consistency vs dt** (`consistency/`; unbiased, 32 seeds, 30720 tu each). RMS(F_mf - F_density):

| dt | RMS(F_mf - F_density) | noise |
|---|---|---|
| 0.002 (accepted) | **0.284** | 0.004 |
| 0.001 | 0.051 | 0.004 |
| 0.0005 | 0.018 | 0.003-0.009 |

The density-based free energy itself moves by 0.12 kT RMS between dt 0.002 and 0.0005: the compact well
goes from -0.78 to -0.46 kT, and the mean-force well from -1.22 to -0.49. Removing the force clips
(min_r 0.5) blows the integrator up.

**Conclusion for dt = 0.002.** The accepted WCA cell carries an integrator bias of 0.1-0.7 kT in the free
energy. That is larger than every ABF-vs-FR effect being compared (0.04-0.09). Because the bias depends on
the drift and on the population dynamics, different samplers (unbiased, ABF, ABF + FR) and the TI
reference converge to different profiles. No dt = 0.002 WCA verdict is therefore reference-free. That
includes the accepted WCA positives, whose reference (v2 TI) was built inside the same artefact. The
dt = 0.002 ladder cannot answer the replica question.

## 3. The dt = 0.0005 ladder (done 12:25 UTC)

Design: `configs/wca_replica_ladder/design_dt0.0005.json`, frozen before the run.

* Same physical times: every step-counted knob x4.
* Budget x4: B = 4.9152e8 replica-steps per arm.
* 16 paired seeds per N (3700-3715).

**Reference.** The pooled serial ABF limit of 32 independent seeds (3800-3831; seed-half noise 0.0015).
It agrees with -kT log P(z) of the dt 0.0005 unbiased runs to **0.0094**, which is within the density's
own noise of 0.009. At this dt the mean-force and density routes agree, so the reference is the free
energy of the simulated system. The v2 TI reference is **0.268** from it: v2 was built at dt 0.002.
Scored against v2, every arm reads e_F ~ 0.26 (`scoreboard_dt0.0005.md`), which measures the reference
error and nothing else.

Results against the dt-consistent reference (`scoreboard_dt0.0005_dynref.md`;
`figures/fig_wca_ladder_*_dt0.0005_dynref.png`). Effects are ABF + FR vs ABF, median paired change
[bootstrap 95 % CI] (wins out of 16).

| N | T | ABF Ibar_F | FR Ibar_F | dIbar_F (wins) | ABF e_F(B) | FR e_F(B) | d e_F(B) (wins) | pooled RMS(FR limit - ABF limit) |
|---|---|---|---|---|---|---|---|---|
| 1024 | 240 | 0.0462 | 0.0516 | +12.0 % [+9.1, +19.7] (1) | 0.0036 | 0.0168 | +377 % (0) | 0.018 |
| 256 | 960 | 0.0150 | 0.0288 | +86 % (0) | 0.0032 | 0.0211 | +563 % (0) | 0.022 |
| 64 | 3840 | 0.0078 | 0.0210 | +161 % (0) | 0.0044 | 0.0192 | +358 % (0) | 0.021 |
| 16 | 15360 | 0.0072 | 0.0133 | +93 % (2) | 0.0036 | 0.0128 | +231 % (0) | 0.015 |
| 4 | 61440 | 0.0083 | 0.0078 | -13 % [-36, +36] (10) | 0.0041 | 0.0047 | +6 % (8) | 0.004 |
| 1 | 245760 | 0.0077 | -- | -- | 0.0032 | -- | -- | -- |

* **ABF + FR follows ABF until it reaches a floor of about 0.015-0.02, then stays there.** ABF keeps
  converging to about 0.0035. The floor is FR's stationary bias. It is about 4x smaller than at dt 0.002,
  where the tilt was 0.06-0.09, but not gone.
* There is no establishment-limited phase in which FR gets ahead. The only transient advantage is
  around u ~ 0.2 at N = 1024.
* FR is harmful at every N >= 16 and neutral at N = 4, where the event cap of 1 keeps it nearly inert.
* ABF itself: Ibar_F falls from 0.046 at N = 1024 to 0.007-0.008 at N <= 64. The final error is
  ~0.0035 at every N, so serial ABF is as good as any split at the endpoint and the best early on.

Predictions:

| | prediction | verdict |
|---|---|---|
| D1 | the FR limit agrees with the ABF limit to <= 0.015 | **FAIL**: 0.018-0.022 at N >= 64, 0.015 at N = 16 |
| D2 | ABF improves as N falls | **PASS** on Ibar_F; final error flat |
| D3 | FR's benefit is largest at large N and vanishes at small N | **FAIL**: no benefit at any N; the harm is smallest at N = 1024 (+12 %) |
| D4 | serial ABF ends more accurate than 1024-replica ABF + FR | **PASS** (0.0032 vs 0.0168) |

## 4. What WCA says about the replica question

On the WCA dimer at a trustworthy time step, uniform FR helps at no replica count, and it is least
harmful at the largest N. At equal force-evaluation budget, few-replica or serial ABF is the best (N, T)
split. This matches the gateway's budget conclusion. The gateway's FR benefit at large N does not
transfer: there, ABF at N >= 128 is establishment-limited; here, ABF converges smoothly and FR only adds
a bias floor.

Open: does FR's residual bias (0.015-0.022 at dt 0.0005) vanish as dt -> 0, or is it a finite-N
resampling (fibre-lag) bias? It is comparable to the remaining estimator-density inconsistency (0.018) at
this dt, so the dt 0.00025 test is the next step. Either way, at any practical dt it dominates ABF's
statistical error after a modest budget.

**Consequence for earlier WCA results.** Every accepted WCA positive (kernel and histogram campaigns,
movies, FR-start ladder) was measured at dt 0.002 against a reference built inside the same integrator
artefact. FR's dose-dependent tilt points toward that reference. These positives should be treated as
**not established** until they are re-measured at a small dt against a dt-consistent reference.
