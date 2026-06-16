# Handoff — Entropy-dominant bottleneck: `target_ema_rate` tuning sweep

_Sweep: `results/entropy_dominant_bottleneck_ema_tuning/ema_sweep_20260615_175039/`_
_Written 2026-06-15. All numbers below were verified directly against the CSVs in this directory; nothing is estimated or fabricated._

> This document is a hand-written analyst handoff for the **EMA-rate tuning** follow-up.
> It is distinct from the auto-generated `handoff.md` / `report_addendum_entropy_dominant.md`
> that the analyzer drops into the **original** sweep
> `results/entropy_dominant_bottleneck/sweep_20260614_015145/`.

---

## 1. Project objective

We study whether adding a **Fisher–Rao (FR) birth–death correction** on top of
**Adaptive Biasing Force (ABF)** helps recover the 1-D marginal free energy
`F(x)` faster / more accurately, and specifically whether the benefit is
**larger for *entropic* bottlenecks than for *energetic* ones**.

The entropy-dominant model (see `src/edb_abffr_core.py`) promotes the transverse
coordinate to `m` dimensions so the entropic share of the marginal free energy
can be dialed up while the **total** barrier `B0` is held fixed:

```
V(x, y) = H (x^2 - 1)^2 + 1/2 omega(x)^2 ||y||^2,    y in R^m
omega(x) = omega_out + (omega_in - omega_out) exp(-x^2 / 2 s^2)
F_ref(x) = H (x^2 - 1)^2 + (m/beta) log omega(x) + C
```

Barrier decomposition (thermal units): `B0 = beta*H + m*log(omega_in/omega_out)`,
entropic share `phi = m*log(omega_in/omega_out) / B0`. `H` and `omega_in` are
**derived** to hold `B0` fixed: `H = (1-phi) B0 / beta`,
`omega_in = omega_out exp(phi B0 / m)`. Sweeping `phi` trades energetic barrier
for entropic barrier at fixed total `B0`.

Fixed physics for this study: `beta=4`, `m=2`, `B0=8`, `omega_out=1`, `s=0.25`.

### Method registry (`src/edb_abffr_core.py`)
- `abf` — ABF only (no birth–death). Baseline.
- `fr_estimated` — **deployable** FR. Target `q_est(x) ∝ exp[-beta(F_EMA(x) - B_ABF(x))]`,
  where `F_EMA` is an online EMA of the ABF bias `B_ABF` itself (engine line ~537:
  `F_target = (1-ema)*F_target + ema*Bbias`).
- `fr_uniform` — FR toward a flat target (control for "shape steering" vs "balanced resampling").
- `fr_oracle` — **non-deployable** FR. Target built from the analytic `F_ref`.
  Diagnostic only; `assert_no_oracle_leakage` guarantees no other method ever reads `F_ref`.

---

## 2. Current scientific question

The original `phi` sweep (`sweep_20260614_015145`) produced a **nuanced** headline:

- The **oracle** target (knows `F_ref`) shows a large FR gain that **grows
  strongly with `phi`** — there is genuine entropic-specific Fisher–Rao headroom.
- The **deployable** `fr_estimated` target does **not** capture that headroom:
  its gain is small, statistically indistinguishable from `fr_uniform`, and
  even negative in the mid-entropic regime.

The open question this sweep answers: **is the estimated-target shortfall just a
badly chosen `target_ema_rate`?** If a different EMA rate exists that closes the
gap to the oracle, the deployable method would be salvageable as-is.

**Answer: no.** Tuning the EMA rate does not fix it (Section 5).

---

## 3. Exact experiments already run

Two stages, both at the locked physics (`beta=4, m=2, B0=8, omega_out=1, s=0.25`),
`N=512` walkers, `dt=5e-4`, `n_steps=80000` (physical time `T=40`), `gamma=15`,
`fr_every=10`, `eta=0.10`, `score_clip=3.0`, `max_event_fraction=0.08`,
`ramp_steps=2000`, `ess_window_steps=4000`, **20 matched seeds** (0–19).

### Stage A — EMA-rate sweep (`abf` + `fr_estimated`)
- `phi ∈ {0.5, 0.75, 0.9}`
- `target_ema_rate ∈ {0.00025, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05}` (8 rates)
- Configs: `experiments/entropy_dominant_bottleneck/configs/ema_tuning/ema_0p*.yaml`
  (one YAML per rate).
- 8 rates × 3 phi × 20 seeds × 2 methods = **960 runs** (120 per rate-job).

### Stage B — matched oracle/uniform baseline (`abf` + `fr_uniform` + `fr_oracle`)
- Same `phi ∈ {0.5, 0.75, 0.9}`, same seeds, single config
  `configs/ema_tuning/oracle_uniform_baseline.yaml` (`target_ema_rate=0.005`).
- 3 phi × 20 seeds × 3 methods = **180 runs**.
- This provides the matched oracle and uniform reference columns used in
  `matched_oracle_uniform_summary.csv` and `ema_tuning_best_by_phi.csv`.

`aggregate_metadata.json` records `n_records: 960`, `baseline_loaded: true`.

> NOTE on orchestration: the per-rate jobs were launched 8-wide (one GPU each,
> see `manifest.tsv`), then aggregated. The aggregation/plotting driver script
> itself is **not** committed to the repo (no `*.py` under `experiments/` or
> `scripts/` references `ema_tuning`/`ema_sweep`); only its YAML inputs, the
> per-rate `run_entropy_dominant_bottleneck.py` outputs, and the aggregated CSVs
> are present. If the EMA sweep must be reproduced end-to-end, the aggregation
> step would need to be re-created (it reads the per-rate `raw/main/*.npz` and
> the baseline, then emits the `ema_tuning_*.csv` files and `plots/`).

---

## 4. GPU usage / runtime

From the per-job `run_metadata.json` files:

- **GPU:** NVIDIA H200 NVL, host `atlas`. Each Stage-A rate-job pinned to one
  GPU (`CUDA_VISIBLE_DEVICES` 0–7, one rate each — see `manifest.tsv`).
- **Per-job wall time:** ~74–86 s for each 120-run rate-job; 180-run baseline ~77 s.
- **Summed GPU wall-time:** ~715 s (~12 min) across the 9 jobs.
- **Real elapsed:** the 8 rate-jobs ran in parallel across GPUs 0–7, so the
  practical end-to-end time is ~the slowest single job (~86 s) plus the baseline
  (~77 s) plus aggregation — i.e. a few minutes, not 12.
- Single-GPU is more than enough; these are small batched runs. `torch 2.12.0+cu130`.

For reference, the **original** `phi` sweep
(`sweep_20260614_015145`, 5 phi × 20 seeds, main + rate-sweep, 800 runs total)
took ~222 s on one H200 NVL.

---

## 5. Key numerical findings (verified)

### 5.1 Best EMA rate per `phi` (`ema_tuning_best_by_phi.csv`)

"Best" = the EMA rate minimizing the **median final `L2(F)`** of `fr_estimated`
(not the rate maximizing gain%). `gain%` is the matched-seed median
`100*(L2_abf - L2_fr)/L2_abf`; positive = FR better.

| phi | best `target_ema_rate` | ABF med final L2 | best est-FR med final L2 | est-FR gain % | est-FR win rate | matched oracle med L2 | oracle gain % | oracle win |
|----:|----:|----:|----:|----:|----:|----:|----:|----:|
| 0.5  | 0.05  | 0.02845 | 0.03126 | **−6.9%**  | 0.15 (3/20) | 0.02464 | **+11.1%** | 0.95 |
| 0.75 | 0.05  | 0.02305 | 0.02480 | **−9.8%**  | 0.35 (7/20) | 0.01403 | **+37.4%** | 1.00 |
| 0.9  | 0.005 | 0.03149 | 0.02552 | **+18.0%** | 0.90 (18/20)| 0.01752 | **+41.6%** | 1.00 |

### 5.2 Every EMA rate, per `phi` (`ema_tuning_matrix.csv`, gain % vs ABF)

| phi | 0.00025 | 0.0005 | 0.001 | 0.002 | 0.005 | 0.01 | 0.02 | 0.05 |
|----:|----:|----:|----:|----:|----:|----:|----:|----:|
| 0.5  | −24.5 | −14.8 | −14.0 | −11.6 | −8.9  | −11.5 | −12.6 | −6.9 |
| 0.75 | −39.5 | −22.1 | −22.6 | −12.6 | −12.4 | −23.8 | −10.3 | −9.8 |
| 0.9  | −8.3  | +11.8 | +19.6 | +17.1 | +18.0 | +17.5 | +10.2 | +8.8 |

### 5.3 What the table says
- **`phi=0.5`:** all 8 EMA rates give **negative** gain. estimated FR is worse
  than ABF at every rate. Best (least bad) is `−6.9%` at rate `0.05`.
- **`phi=0.75`:** same — **all 8 rates negative**, best `−9.8%` at rate `0.05`.
- **`phi=0.9`:** estimated FR helps for 7 of 8 rates; best **min-L2** at rate
  `0.005` (`+18.0%`). Note the *max-gain%* rate is `0.001` (`+19.6%`) but its
  median L2 (0.02622) is slightly worse than rate `0.005` (0.02552), which is
  why `best_by_phi` selects `0.005`. Either way it stays far above the oracle
  (`+41.6%`, median L2 0.01752).
- The matched oracle gain (`+11.1 / +37.4 / +41.6%`) **rises strongly with phi**;
  the best deployable estimated gain (`−6.9 / −9.8 / +18.0%`) does **not**.
- `fr_uniform` ≈ `fr_estimated` at the matched point
  (`matched_oracle_uniform_summary.csv`: uniform gains `−13.7 / −15.1 / +12.5%`),
  i.e. the self-estimated free-energy *shape* adds essentially nothing over a
  flat target.

### 5.4 Plots
- `plots/ema_tuning_gain.png` — matched-seed gain % vs EMA rate, one curve per phi.
- `plots/ema_tuning_final_l2.png` — median final `L2(F)` vs EMA rate, per phi.

Supporting CSVs: `ema_tuning_summary.csv` (per rate×phi×method medians, IQR, SE,
ESS, replacement fraction, score std, win rates), `ema_tuning_raw_scalars.csv`
(960 per-run scalar rows), `matched_oracle_uniform_summary.csv`.

---

## 6. Interpretation — why EMA-estimated FR underperforms for entropic barriers

The deployable target is `q_est(x) ∝ exp[-beta(F_EMA(x) - B_ABF(x))]`, where
`F_EMA` is an **EMA of the ABF bias `B_ABF`** that the same run is producing.
Two structural problems compound specifically in the entropic regime:

1. **Self-referential / near-uniform target.** Because `F_EMA` is a lagged copy
   of `B_ABF`, the exponent `F_EMA - B_ABF` is the *difference between the bias
   and a smoothed version of itself*. Once the ABF bias has roughly converged,
   that difference is small and slowly varying, so `q_est` collapses toward a
   nearly flat distribution. This is exactly why `fr_estimated` and `fr_uniform`
   behave almost identically (Section 5.3): the estimated target carries little
   usable free-energy *shape* information — the FR mechanism reduces to balanced
   resampling / variance reduction, not shape-steering.

2. **Noisy, bias-prone mean force at the bottleneck.** For an entropic barrier
   the marginal free energy is the smooth `(m/beta) log omega(x)` bump, and the
   instantaneous force along `x` is `dV/dx = 4Hx(x^2-1) + omega(x) omega'(x) ||y||^2`.
   The entropic part depends on the **conditional transverse fluctuation `||y||^2`**,
   whose variance is largest exactly near the channel (small `omega`, large
   `1/(beta omega^2)`). So the ABF mean-force estimate — and therefore `F_EMA`
   and `q_est` — is noisiest precisely where steering would have to be accurate.
   Birth–death toward a noisy target injects resampling noise instead of
   correcting coverage, which is why mid-entropic gains go **negative**.

The oracle removes problem 1 entirely (it is handed the true `F_ref` shape) and
problem 2 partially (its target is exact), so it realizes the entropic-specific
headroom — `+11 / +37 / +42%` rising with phi. The gap between oracle and
estimated is therefore a **target-quality gap**, not an FR-mechanism gap, and it
is **not** closed by tuning the single EMA hyperparameter.

> Sanity context: the FR variants preserve the conditional law `Y|X` (empirical
> `Var(Y_j|X)` matches the analytic `1/(beta omega^2)` across ~3 orders of
> magnitude in the original sweep), so the shortfall is not FR corrupting the
> orthogonal coordinates — it is purely the marginal-target estimate being weak.

---

## 7. What should NOT be claimed in the paper

- ❌ **Do not claim** the deployable (estimated-target) FR realizes an
  entropic-specific gain. At `phi=0.5` and `phi=0.75` it is **net-harmful at
  every EMA rate tested**; only `phi=0.9` shows a positive deployable gain.
- ❌ **Do not claim** the estimated-target shortfall is a hyperparameter artifact
  ("just needs a better EMA rate"). This sweep tested 8 rates over two decades
  and the gap to the oracle persists at all of them.
- ❌ **Do not present oracle (or any `F_ref`-informed) numbers as achievable in
  practice.** The oracle reads the analytic `F_ref`; it is a diagnostic upper
  bound on FR headroom, not a deployable method. Always label it as such.
- ❌ **Do not claim** the self-estimated free-energy *shape* helps over a flat
  target — `fr_estimated ≈ fr_uniform` here.
- ❌ **Do not over-read `phi=0.9`'s `+18%`** as evidence the method works
  generally; it is one of three phi values and the two lower ones are negative.
- ✅ **Safe / honest claim:** FR birth–death reliably *repairs sample-starved
  ABF and accelerates transient convergence* (integrated-error gain is positive
  across the board in the original sweep), independent of whether the starvation
  is energetic or entropic. The entropic-specific headroom is **real but
  target-limited** (oracle proves it exists; the deployable estimator cannot yet
  reach it), and that is the clearest open problem to flag for future work.

---

## 8. Recommended next steps (non-oracle, deployable)

The goal of every option below is to **improve the deployable marginal-target
estimate** so it approaches the oracle headroom, without ever reading `F_ref`.

**(a) Early FR + late ABF cleanup.** Use FR aggressively early (when ABF is
sample-starved and the coverage benefit is largest), then ramp birth–death off
and let pure ABF refine the mean force late, when a noisy target would only add
resampling noise. Cheap to try: schedule `gamma(t)` to decay (the engine already
has a ramp; add a down-ramp). Tests whether the negative mid-entropic gain is a
late-time target-noise artifact.

**(b) Lagged / windowed target estimator.** Replace the single-EMA `F_EMA` with
a target built from a **time-lagged or sliding-window** block of the ABF bias
(e.g. the bias as of `t - tau`, or a window average) so the target is not a
near-instantaneous copy of `B_ABF` — directly attacks the self-referential
collapse in Section 6.1. Low risk, reuses the existing pipeline.

**(c) Cross-fitted target estimator.** Split walkers into folds; build each
fold's FR target from the *other* folds' ABF statistics. Decorrelates the target
from the walkers it steers, reducing the feedback bias near the bottleneck.
Moderate engineering; principled variance-reduction.

**(d) Confidence-gated / tempered target.** Gate or temper the target by local
estimator confidence: where the ABF mean-force count/variance is poor (the
bottleneck, problem 6.2), shrink `q_est` toward uniform or raise an effective
temperature so FR does not steer hard toward a noisy estimate. Use the existing
per-bin smoothed count (`final_denom0` / the `C` accumulator) as the confidence
signal.

**(e) eABF / CZAR-style target estimation.** Estimate the marginal free energy
with an extended-system / CZAR estimator instead of raw binned ABF bias, then
feed that into the FR target. This is the most likely to genuinely close the gap
because CZAR gives a lower-variance, less-biased free-energy estimate exactly in
the regimes where binned ABF struggles. Largest engineering lift; highest upside.

**Suggested ordering:** (b) and (a) first (cheap, reuse pipeline) → (d) (uses
existing confidence signals) → (c) → (e) (largest lift). Re-run the `phi`
sweep with each and compare the deployable gain trend against the oracle curve.

### Diagnostic-only ablations (NOT deployable methods)

Analytic **conditional-variance / entropy targets** — i.e. targets that use the
known `Y|X ~ N(0, 1/(beta omega^2) I_m)` law or the analytic entropic term — may
be informative as *model-informed ablations* to bound how much of the gap is due
to the marginal estimate vs the entropic force noise. **They must be treated as
diagnostic only, on the same footing as `fr_oracle`**, and never reported as a
deployable method: real applications do not know the conditional law (or `omega(x)`)
analytically. Keep them clearly separated from options (a)–(e) in any writeup.

---

## 9. Important file paths

Sweep root: `results/entropy_dominant_bottleneck_ema_tuning/ema_sweep_20260615_175039/`

| File | Contents |
|---|---|
| `ema_tuning_best_by_phi.csv` | best EMA rate per phi + matched oracle/uniform gains (Section 5.1) |
| `ema_tuning_matrix.csv` | gain % and median L2 for every rate×phi (Section 5.2) |
| `ema_tuning_summary.csv` | full per rate×phi×method medians/IQR/SE/ESS/win-rate |
| `ema_tuning_raw_scalars.csv` | 960 per-run scalar rows |
| `matched_oracle_uniform_summary.csv` | abf / fr_uniform / fr_oracle at the matched point |
| `aggregate_metadata.json` | sweep manifest (`n_records:960`, `baseline_loaded:true`) |
| `manifest.tsv` | per-rate GPU/pid/config/out_dir/log mapping |
| `plots/ema_tuning_gain.png`, `plots/ema_tuning_final_l2.png` | summary figures |
| `ema_0p*/run_metadata.json`, `oracle_uniform_baseline/run_metadata.json` | per-job GPU/runtime |
| `ema_0p*/raw/main/*.npz` | per-run raw arrays (profiles, time series, conditional diagnostics) |

Code / config:
- `src/edb_abffr_core.py` — m-dim ABF+FR engine (generalizes `src/eb_abffr_core.py`).
- `experiments/entropy_dominant_bottleneck/run_entropy_dominant_bottleneck.py` — per-config runner.
- `experiments/entropy_dominant_bottleneck/analyze_entropy_dominant_bottleneck.py` — analyzer for the **original** sweep (not the EMA aggregation).
- `experiments/entropy_dominant_bottleneck/configs/entropy_dominant_default.yaml` — original 5-phi config.
- `experiments/entropy_dominant_bottleneck/configs/ema_tuning/ema_0p*.yaml` — 8 per-rate configs.
- `experiments/entropy_dominant_bottleneck/configs/ema_tuning/oracle_uniform_baseline.yaml` — Stage-B baseline config.

Related prior context: original phi sweep
`results/entropy_dominant_bottleneck/sweep_20260614_015145/` (with its own
auto-generated `handoff.md` and `report_addendum_entropy_dominant.md`).

---

## 10. Status / reproducibility caveats

- ✅ All numbers in Sections 5.1–5.3 verified against the CSVs (2026-06-15).
- ✅ No NaNs; all 960 + 180 runs completed (`completed: true` in metadata).
- ⚠️ The EMA-sweep **aggregation/plotting driver is not committed** (Section 3
  note). The per-rate runs are reproducible via `run_entropy_dominant_bottleneck.py`
  with the `ema_0p*.yaml` configs; the `ema_tuning_*.csv` aggregation step would
  need to be re-created from the per-rate `raw/main/*.npz` if regenerated.
- These files are currently **untracked** in git (`results/...ema_tuning/` and
  `configs/ema_tuning/`). Commit if they should be preserved.




