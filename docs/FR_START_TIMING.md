# FR-start timing experiment — preregistration

**Frozen 2026-09-04, before any production run of this experiment.**  Branch `main`.
GPU 3 (H200) only, per the standing user instruction.

## Question

Two of the project's null cells had uniform-target Fisher–Rao (FR) switched on long
after the ABF warm-up had ended:

| cell | ABF warm-up ends | FR start (closed arm) | closed verdict |
|---|---|---|---|
| alanine dipeptide, vacuum (φ,ψ), campaign Stage 3 | 5 ps | **20 ps** | EQUIVALENT, ΔI_F −1.13 % [−3.50, +1.54] 10/16 (repaired endpoint) |
| pentane R15, β = 1.4 / 1.6 (P1 mid-β follow-up) | 5 000 steps (2.5 t.u.) | **12 000 steps (6.0 t.u.)** | null, ΔI_F +0.49 % / +0.61 %, final −0.85 % / +0.40 % |

The ethane/LTA sweep showed that this can matter: moving `fr_start` from 40 000 to
20 000 steps (= warm-up end) turned a null (−0.21 %) into −14.8 % at 300 K with the
same rate and seeds (`configs/uniform_campaign/lta_sweep_prereg.json`).  The
mechanism claim of the whole campaign is that uniform FR is a finite-time
**establishment accelerator**: it can only act in the window
T_discover < t < T_establish(ABF).  Starting after that window has closed reads as a
null by construction.

**H1 (timing):** the alanine and R15 nulls are artefacts of a late FR start; FR started
at the end of the ABF warm-up accelerates ABF's convergence in the 5–20 ps (alanine) /
2.5–20 t.u. (R15) window.

**H0 (ABF-sufficient):** the nulls persist at every start time — the cells are
ABF-sufficient and the closed verdicts stand.

## Existing-data motivation (computed before any new run, no GPU)

Alanine, campaign ABF arm (16 seeds, N = 2048), kernel-matched equilibrium-weighted
aligned-L2 error, medians over seeds:

| t (ps) | e_F (kJ/mol) | KL(p̂‖uniform) | C7ax fraction |
|---|---|---|---|
| 4 | 4.30 | 2.16 | 0.000 |
| 5 | **6.92** (peak) | 1.83 | 0.024 |
| 8 | 2.83 | 1.97 | 0.273 (overshoot) |
| 10 | 1.62 | 1.96 | 0.261 |
| 20 | 0.44 | 1.68 | 0.069 |
| 100 | 0.21 | 1.67 | 0.053 |

C7ax is first hit at 3.98–4.56 ps in **all 16 seeds**, and the error falls 16× between
5 and 20 ps.  So the window H1 needs (rare basin discovered, marginal far from uniform,
error still large, FR not yet on) exists and is exactly 5–20 ps.

But the frozen alanine dose is nearly inert: at rate 0.02 the campaign arm fired a
median 2.3 replacements per opportunity (0.11 % of N; 370 cumulative = 18 % of N over
80 ps) and KL(p̂‖uniform) did not move (1.680 → 1.674).  The oracle rate ladder at the
20 ps start (N = 4096, seeds 10–13) was flat from 0.02 to 0.45 while age-aware ESS fell
0.97 → 0.81 (rate 0.15) → 0.60 (rate 0.45), all above the 0.30 floor.

R15 β = 1.4, closed ABF arm: compact R first discovered at a median 1 932 steps
(≈ 1 t.u.), error 3.08 at t = 2.5 → 1.94 at 10 → 1.23 at 20 → 1.06 plateau from
t ≈ 30.  The closed FR arm fired **90 replacements per seed in 13 601 opportunities**
(6.5 × 10⁻⁶ per opportunity, 9 % of N cumulative).

Consequence for the design: a timing-only ladder at the frozen rates cannot separate
"too late" from "too gentle", so a **dose factor** is added a priori at rates the
project has already run (alanine 0.15 = the oracle ladder's middle rung; R15 0.10 =
the closed study's own `fr_active` probe).  The dose arms are secondary; the primary
contrast changes only the start time.

## Arms

Everything not listed is inherited verbatim from the closed frozen configurations
(`configs/uniform_campaign/alanine_uniform.yaml`, `configs/uniform_campaign/r15_midbeta_methods.yaml`).

**Alanine** (`configs/fr_start_timing/alanine.yaml`), seeds 0–15, N = 2048, 100 ps, init
c7eq / init_seed 4242, rng_seed 20260903, fr_every 0.5 ps, score_clip 2.0, cap 0.05:

| stage | method | fr_start | fr_rate | role |
|---|---|---|---|---|
| abf | abf | – | – | fresh baseline, same seeds |
| u02_t5 | fr_uniform | **5 ps** | 0.02 | **PRIMARY** (= warm-up end, the LTA rule) |
| u02_t2 | fr_uniform | 2 ps | 0.02 | exploratory: before C7ax discovery, ABF ramp at 40 % |
| u02_t10 | fr_uniform | 10 ps | 0.02 | intermediate |
| u02_t20 | fr_uniform | 20 ps | 0.02 | campaign arm re-run (replication + same process family) |
| u15_t5 | fr_uniform | 5 ps | 0.15 | dose check |
| u15_t20 | fr_uniform | 20 ps | 0.15 | dose check |
| o02_t5 | fr_oracle | 5 ps | 0.02 | secondary target check; runs last, only if time permits |

**Pentane R15** (`configs/fr_start_timing/r15.yaml`), β ∈ {1.4, 1.6}, seeds 200–215,
N = 1024, 80 000 steps (40 t.u.), init trans, rng_seed 20260830, fr_every 5, cap 0.01,
score_clip 2.0, `save_every` 5 000 → **1 000** (diagnostics only; 81 checkpoints):

| method | fr_start (steps) | fr_rate | role |
|---|---|---|---|
| abf | – | – | fresh baseline at the new save cadence |
| u02_s5000 | **5 000** | 0.02 | **PRIMARY** (= warm-up end) |
| u02_s3000 | 3 000 | 0.02 | exploratory: mid-warm-up (ramp at 60 %) |
| u02_s8000 | 8 000 | 0.02 | intermediate |
| u02_s12000 | 12 000 | 0.02 | closed arm re-run |
| u10_s5000 | 5 000 | 0.10 | dose check |
| u10_s12000 | 12 000 | 0.10 | dose check |

Pairing.  R15: the init (all-trans + 1e-3 jitter from the seeded generator) and the
fixed-shape per-step noise draw are functions of `rng_seed` only, so every arm is paired
with ABF by construction across processes.  Alanine: the 20 ps thermalised ensemble is
built once by the `abf` stage and cached (`--init-cache`), so every arm starts from the
bitwise-identical ensemble; the dynamical noise stream is per-step fixed-shape from
`rng_seed`.  (The campaign's own two arms already differed in the 4th digit at 5 ps,
before any FR event, because the estimator's float64 `scatter_add` is order-
nondeterministic; "paired" has always meant same init + same noise stream, never
bitwise trajectories.)

## Engine changes (implementation only; arithmetic unchanged)

* `src/alanine/graphed.py`: CUDA-graph replay of the physical force and of the CV local
  mean force.  Same eager kernels in the same order; outputs **bitwise identical**
  (`tests/test_alanine_graphed.py`, `torch.equal`); random draws stay eager.  Step time
  at R·N = 32 768: 45.8 → 17.6 ms in a real 600-step run, final PMF agreement 3e-14,
  identical event counts.
* `alkanes/cv2d.py`: the constant atom-index tensors are cached per device instead of
  rebuilt from Python lists each call (no arithmetic); `torch.linalg.inv` →
  `torch.linalg.inv_ex(...).inverse` (same routine, bitwise-equal result, no host
  sync; the adaptive ridge already excludes the singular case `inv` would raise on);
  the regularisation counter accumulates in place.
* `scripts/run_alanine_study.py`: `--init-cache`, `--cuda-graph` (both recorded in the
  artifact meta together with `fr_start_steps`, `fr_every`, `fr_rate`).
* R15 engine untouched.

## Endpoints

Let e(t) be each system's frozen error norm against its frozen reference: alanine
`eF_km_equilibrium` (kernel-matched, equilibrium-weighted aligned L2, the 2026-09-02
repaired endpoint); R15 the thermal-window aligned interval L2 `l2_F_t` (as stored by
`jobs_cv.execute_dist`).

1. **Primary: I_F = ∫ e(t) dt over the primary window** W₁ = [5, 100] ps (alanine) and
   [2.5, 40] t.u. (R15) — from the primary arm's own start to the end — for EVERY arm
   including ABF and the late-start arms.  Statistic: per-seed paired relative change
   ΔI_F = 100·(I_F^arm − I_F^abf)/I_F^abf, median over the 16 seeds, 10 000-resample
   BCa bootstrap CI of the median (`alanine.metrics_ala.paired_bootstrap`), win rate.
2. Secondary windows, same statistic: the closed windows W₂ = [20, 100] ps / [0, 40]
   t.u. (comparability with the closed verdicts), and each arm's own window
   [t_start, T].
3. **Final error** e(T), paired relative change (transient vs persistent).
4. **Time-to-accuracy** τ_ε: first t in W₁ with e ≤ ε sustained for 0.2·T, for
   ε ∈ {e₀/2, e₀/4, e₀/8, ABF's own final error}, e₀ = median ABF e at the start of W₁.
   Speed-up S_ε = τ_ε^abf / τ_ε^arm; censoring reported, never imputed.
5. Mechanism: alanine KL(p̂_t‖uniform) and basin fractions (C7eq, C5, C7ax); R15
   `frac_compact`/`frac_extended` and `kl_pq` vs t; profile snapshots.
6. Genealogy / dose: alanine age-aware ESS/N ≥ 0.30, max lineage share ≤ 0.05, per-
   opportunity event fraction < 0.05 (cumulative reported); R15 final ancestor ESS ≥
   0.30 N, max ancestor fraction ≤ 0.05, event fraction < 0.05.  A floor violation is
   reported next to the verdict and blocks the "safe" label.
7. Replication check: the new `abf` vs the campaign's `abf` and the new `u02_t20` vs
   the campaign's `fr_uniform` (same seeds, equal-in-distribution init).  Expected:
   ΔI_F within the bootstrap CI of zero.  Also the new R15 `abf`/`u02_s12000` vs the
   closed pair on the coarse (5 000-step) checkpoints, which are a subset of the new ones.

## Decision rules (frozen now; the campaign's thresholds)

Per arm vs ABF on the primary window: **acceleration-positive** if median ΔI_F ≤ −10 %
AND CI95 upper < 0; **safe** if additionally the median final change ≤ +5 %; **neutral**
if |median ΔI_F| < 10 % and the final change within ±5 %; **harmful** if CI95 lower > 0
on I_F or the final change > +5 % with CI excluding 0.

*Amendment A1 (2026-09-04 12:50 UTC, before any new arm had finished; found while
validating the analysis script on the CLOSED data only).*  The three labels above are
evaluated in the order positive → neutral → harmful, as in the campaign, so a change
whose CI excludes zero but lies inside the ±10 % band is **neutral** (reported as
NEUTRAL_SIG with its CI), not harmful; the closed R15 arms (+0.55 % [+0.22, +0.67]) are
the case in point.  No threshold changes.

Reading of the ladder:

* **H1 supported (burn-in was too long):** `u02_t5` / `u02_s5000` acceleration-positive
  while the re-run 20 ps / 12 000-step arm is neutral (as closed).
* **Timing trade-off:** 2 ps / 3 000 harmful or worse than 5 ps / 5 000, 5 ps / 5 000
  best, 20 ps / 12 000 neutral.
* **Dose, not timing:** the rate-0.15 / 0.10 arms positive at BOTH start times with the
  timing contrast between them neutral.
* **Timing × dose:** only the early high-dose arm positive — timing matters but the
  frozen rate is too gentle to show it.
* **H0 (close the cell as ABF-sufficient):** every rate-0.02 arm neutral AND the
  dose arms neutral or harmful.  If the dose arms flatten the marginal (KL falls) while
  ESS drops and I_F does not improve, that is count balancing without conditional
  information, reported as such.

The primary contrast is one number per system (u02_t5 / u02_s5000 vs abf on W₁); all
other arms are secondary and are reported in full regardless of sign.

## Compute

GPU 3 only (`CUDA_VISIBLE_DEVICES=3`, enforced by the alanine runner).  Alanine arms run
as separate processes from the cached init, two at a time; R15 stages `b14` and `b16`
run as two concurrent processes.  Estimated: alanine ≈ 30 min per arm (graphed), R15
≈ 20–50 min per arm; whole experiment ≈ 4–5 h wall.

## Held fixed / prohibited

* No FR parameter is tuned against e, I_F or τ; the two dose rates are the a-priori
  values named above and nothing else is added after data are seen.
* No additional arms, windows, seeds or thresholds after the first production run starts.
* Every arm is reported (including the exploratory and secondary ones) with its
  genealogy numbers; a floor violation is never hidden behind an error improvement.
* The closed verdicts are not edited; this document and its results file
  (`results/fr_start_timing/RESULTS.md`) are additive.
