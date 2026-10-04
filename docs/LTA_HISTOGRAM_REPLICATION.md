# Ethane in LTA under the histogram estimator: replication, the 350 K point, a matched sham, and the movies

**Date:** 2026-10-03 (night; the user asleep, autonomous session). **Status:** design, predictions and
schedule written BEFORE any run; the frozen design is `configs/lta_histogram/campaign.json` (18:50 UTC,
git base `56ab774`). Results are appended below as they close. **GPU:** 3 only.

## 1. Where this starts from

The ethane/LTA system (`src/lta/core_lta.py`: TraPPE-UA ethane in rigid all-silica LTA, CV = COM
position along the cage–window–cage axis, overdamped BD) is the campaign's molecular entropic-barrier
example. Its closed kernel-estimator temperature sweep (`configs/uniform_campaign/lta_sweep_prereg.json`,
`results/uniform_campaign/RESULTS.md` Stage 5) found uniform-target FR a safe accelerator at every T,
with the benefit growing as T falls while the entropic share of the barrier falls:

| T (K) | ΔF‡ (kT) | entropic share | ΔI_F | final Δe_F | ABF crossings / replica |
|---|---|---|---|---|---|
| 300 | 10.8 | 72 % | −14.8 % [−17.0, −11.7] 16/16 | −19.7 % | 2.69 |
| 225 | 11.7 | 68 % | −21.3 % | −28.2 % | 1.80 |
| 150 | 13.5 | 61 % | −31.9 % | −56.3 % | 0.94 |
| 80 | 18.1 | 47 % | −35.1 % | −74.7 % | 0.23 |

and the offline read-out audit (`docs/LTA_READOUT_SWEEP.md`) showed the gain survives every read-out
bandwidth down to raw bins. Three things were missing, and a fourth was asked for tonight:

1. **Estimator.** The presented algorithm has moved to the textbook online histogram (P0) mean-force
   estimator; EB, WCA and the gateway were replicated under it (`docs/HISTOGRAM_ABF_REPLICATION.md`),
   LTA was not. Every LTA movie / figure should be made under the same estimator as the others.
2. **A hotter point.** The two readings of the sweep (establishment starvation vs entropic share)
   disagree in direction as T rises further: starvation predicts the benefit keeps shrinking at 350 K
   (ABF feeds the window ever more easily), entropy share predicts it should grow (−TΔS‡ share rises).
   350 K is a cheap, sharp, falsifiable new point. (The temperatures proposed in the planning note,
   150–350 K, overlap the closed sweep except for this one.)
3. **Attribution.** The LTA sweep had two arms. The campaign's method rule (gateway) is one matched
   sham per FR arm and the DIRECT arm-vs-sham contrast as the attribution statistic. The sham replays
   the FR arm's realised per-opportunity event counts with random direction.
4. **Movies.** Ethane/LTA is the example where "entropic barrier" has a concrete physical meaning
   (large α-cage, 4.1 Å 8-ring window: the window is not high in energy, it has few admissible
   configurations). Requested: a presentation movie with the physical picture, the F / U / −TS
   decomposition, the ABF vs FR split screen with the replica marginal, and the FR events.

### The screening rule, checked on data we already had

The planning note proposes choosing T* from ABF-only data by T_hit/T_run < 0.1 and
0.25 < T_est/T_run < 0.75. From the kernel sweep's ABF arms (computed before any new run; T_run = 60):

| T (K) | first window visit (median seed) | t at which the median ABF error first comes within 1.5× of its final value | 2× | τ(e₀/8) | KL(p‖uniform) at t = 4 / 12 / 60 |
|---|---|---|---|---|---|
| 80 | 3.0 | 6.0 (non-monotone curve: 0.227 at t = 12, 0.255 at t = 30, 0.164 at 60) | 4.8 | 4.2 | 0.60 / 0.45 / 0.010 |
| 150 | 2.4 | 40.8 | 30.0 | 3.6 | 0.43 / 0.11 / 0.011 |
| 225 | 2.1 | 39.0 | 31.8 | 3.0 | 0.33 / 0.03 / 0.009 |
| 300 | 1.8 | 36.6 | 27.6 | 2.4 | 0.26 / 0.01 / 0.008 |

So with T_est tied to the FINAL error the rule already classifies 150–300 K as establishment-limited
(T_hit/T_run 0.03–0.05, T_est/T_run 0.61–0.68); tied to the initial error (the campaign's e₀/8
convention) it would call every T "too easy" (τ 2.4–4.2) although FR gains 15–35 %. The threshold
must be tied to the endpoint the method is judged on. (At 80 K ABF never converges in the budget.)

## 2. Design (frozen in `campaign.json`)

* **Engine additions** (all off by default; `tests/test_lta_histogram.py`, 8 tests: the legacy kernel
  engine is bit-identical to a fixture generated from the pre-change engine and every legacy
  `config_hash` is unchanged):
  `abf_estimator = "histogram"` — own-bin P0 bias `Γ_j = M_j / C_j` on the engine's 180 circular bins
  (0 on empty bins, the WCA rule), no smoothing, no interpolation; reported F = exact integral of the
  piecewise-constant force, closed on the circle by the engine's existing periodic-closure convention,
  at the cell centres; the ±60 clip and the 20 000-step ramp unchanged.
  `fr_sham` — replays its partner's `event_counts` (deaths uniform without replacement, sources uniform
  among survivors, `wca_abffr_core.uniform_birth_death_torch` semantics), no score.
  `store_snapshots` — report-only movie record (configurations, CV, walker ids, reported PMF, events),
  no RNG, no arithmetic change.
* **Cell and knobs:** the sweep's sampler verbatim except the estimator (N 1024, 300 k steps, ramp /
  burn-in / FR start 20 000, fr_every 5, clip 2, cap 0.02, FR KDE 0.10); per-T FR rates inherited from
  the sweep's safety-only ladders (0.20; 80 K 0.10); **350 K**: reference by the sweep's rule
  (κ 594 kJ/mol/rad², 128 571 steps, 25 714 burn-in, seed 20260883) and its own safety-only ladder
  under the histogram sampler, frozen into `campaign.json` before the 350 K production.
* **Pairing:** per-T `rng_seed` = the kernel sweep's, so the histogram ABF arm runs on the same initial
  conditions and noise stream as the kernel ABF arm (noise-paired across estimators); fresh seed
  labels 1000–1095. Three arms of one T in ONE process, 16 seeds.
* **Width:** 180 bins (the deposit grid of the kernel estimator; raw-bin read-out on the plateau in
  the audit). Clause (a) of the histogram campaign's rule (deterministic P0 floor share ≤ 0.20) is
  evaluated from the reference before any FR result is read; clause (b) (plateau to 360 bins) by an
  ABF-only ladder {90, 180, 360} at the end of the schedule, reported, not used to re-select.
* **Endpoints:** the sweep's (paired ΔI_F and Δe_F(T) medians, 10 k bootstrap CI seed 20260829, wins;
  τ atlas convention; genealogy median-across-seeds floors 0.30 / 0.05); replication clauses and the
  +25 % absolute-accuracy clause of the histogram campaign; attribution = direct FR-vs-sham contrast.
* **Movies:** 300 K and 150 K, one seed (label 1200, rng 20261000 + T), both arms in one process,
  snapshots every 100 steps; renderer `scripts/lta_movie_render.py`, simulate stage
  `scripts/run_lta_histogram.py movie`.
* **Schedule (GPU 3, `scripts/lta_histogram/launch.sh`):** movie 300 K → reference 350 K → ladder
  350 K → production 300 → movie 150 → production 150 → 350 → 225 → 80 → width ladder.

## 3. Predictions (recorded before any run)

- **P1 replication:** histogram FR vs histogram ABF has the kernel sweep's sign and a magnitude within
  10 points at every replicated T, CIs excluding zero, ≥ 15/16 wins.
- **P2 baseline:** histogram ABF own e_F(T) within ±10 % of kernel ABF's legacy read-out
  (0.087 / 0.134 / 0.095 / 0.164 kJ/mol at 300 / 225 / 150 / 80 K); I_F within +15 %.
- **P3 350 K:** the benefit keeps shrinking (ΔI_F between −12 % and 0; final within ±10 %; ABF
  crossings per replica > 2.7; entropic share > 72 %): NEUTRAL or marginal. If ΔI_F < −15 % the
  entropy-share reading gains support instead.
- **P4 sham:** the matched sham is neutral to slightly harmful vs ABF; FR beats it directly by at least
  two thirds of its FR-vs-ABF margin with a CI excluding zero.
- **P5 screening rule:** the histogram ABF arms reproduce the table above (rule satisfied at 150–300 K
  with the final-error threshold, not with e₀/8).
- **P6 health:** histogram FR fires 30–70 % more replacements than kernel FR (WCA precedent); median
  min ESS/N ≥ 0.30 at 150–350 K, possibly < 0.30 at 80 K.

## 4. Results

(appended as stages close)

### 4.0 Before any FR result was read (19:15 UTC)

* **Width clause (a), deterministic:** P0 floor of the 180-bin histogram, from a periodic cubic
  interpolant of each reference (`scripts/analyze_lta_histogram.py`, `results/lta_histogram/pre_run_kernel_only/`):
  0.0031 / 0.0035 / 0.0041 / 0.0048 kJ/mol at 80 / 150 / 225 / 300 K against kernel ABF e_F(T) of
  0.164 / 0.134 / 0.095 / 0.087 → share 0.02–0.06 ≤ 0.20. **180 bins admissible.** (90 bins: 0.0166 at
  300 K, share 0.19, borderline; 360 bins: 0.0011.) Clause (b) awaits the end-of-night ladder.
* **Kernel reproduction:** the new analyzer recomputes the closed sweep's ΔI_F, Δe_F(T) and both CIs
  from the kernel production files to exactly 0.0 at every T.
* **Screening rule on the kernel ABF arms** (T_est = persistent time-to-1.5×-final, τ convention):
  satisfied at all four T (T_hit/60 = 0.03–0.05; T_est/60 = 0.55–0.68); with the first-hit reading
  80 K drops out (6.0, non-monotone curve).
* **300 K movie (seed 1200), already closed:** ABF e_F(T) 0.0934 kJ/mol, FR 0.0839 (−10.1 %),
  I_F −12.4 %, 672 replacements, min ESS/N 0.431 (first record; the genealogy re-simulation of 2026-10-04 gives 0.0915 / 0.0813, −11.1 %, −12.8 %, 665 events, see §4.9); 129/145 of the deaths before t = 10 are in the cage
  and 136/145 births go to neck/window; FR holds ABF's final accuracy from t = 49.6 (ABF 59.7).
  `results/lta_movie/T300/README.md`. Configuration cloud (ABF arm, t ≥ 4): transverse rms 2.78 Å in the
  cage vs 0.22 Å in the window, ⟨|cos θ|⟩ 0.53 vs 0.955 — ln(160) + ln(~9) ≈ 7.3 k_B against the
  measured −TΔS‡/kT = 7.8 (`figures/fig_lta_configuration_cloud_T300`).

### 4.1 The 350 K reference and its FR rate (19:12–19:23 UTC, before the 350 K production)

Umbrella/WHAM by the sweep's rule (κ 594 kJ/mol/rad², 40 × 256, 128 571 steps, seed 20260883):
**ΔF‡ = 10.34 kT** (split halves 10.34 / 10.33), ΔU‡ = 2.70 kT, **−TΔS‡ = 7.64 kT = 74 % of the barrier**;
unbiased 16 384 × 100 k cross-check overlaps 159/180 bins with 0.154 kJ/mol RMS. So the entropic share
keeps rising with T (47 → 61 → 68 → 72 → 74 %) while the barrier in kT keeps falling (18.1 → 10.3):
the two readings of the sweep still point in opposite directions at this T.
`results/uniform_campaign/lta/reference/reference_T350.npz`; `figures/fig_lta_decomposition_vs_T` updated.

Safety-only rate ladder under the histogram sampler (seeds 1110–1111, 120 k steps): min ESS/N
0.929 / 0.858 / 0.739 / 0.607 and wmax ≤ 0.006 at rates 0.02 / 0.05 / 0.10 / 0.20 → **0.20 selected**
(the same rate as 150–300 K), frozen into `campaign.json` at 19:22:54 UTC; no error metric read.

### 4.2 300 K, three arms, 16 seeds (production 19:23–20:30 UTC; `results/lta_histogram/scoreboard.md`)

| arm (histogram, 180 bins) | e_F(T) median (kJ/mol) | I_F median | events | crossings / replica | median min ESS/N (worst) | wmax |
|---|---|---|---|---|---|---|
| abf | 0.0866 | 22.98 | 0 | 2.66 | – | – |
| fr_uniform (rate 0.20) | 0.0654 | 19.85 | 10 492 | 2.74 | 0.414 (0.391) | 0.009 |
| fr_sham (replays fr_uniform's counts) | 0.0838 | 23.36 | 10 492 | 2.67 | 0.439 (0.408) | 0.008 |
| kernel abf (sweep, same noise) | 0.0869 | 23.03 | 0 | 2.69 | – | – |
| kernel fr_uniform (sweep) | 0.0664 | 19.56 | 10 483 | 2.75 | 0.418 (0.380) | 0.009 |

Paired contrasts (median, 95 % CI, wins):

| contrast | ΔI_F | Δe_F(T) |
|---|---|---|
| **hist FR vs hist ABF** | **−14.10 % [−15.10, −10.98] 16/16** | −7.20 % [−15.42, +0.87] 11/16 |
| hist sham vs hist ABF | −0.24 % [−2.17, +1.35] 8/16 | −1.17 % [−4.85, +11.09] 10/16 |
| **hist FR vs hist sham (direct)** | **−13.17 % [−14.62, −12.48] 16/16** | −8.18 % [−15.28, −2.63] 12/16 |
| kernel FR vs kernel ABF (sweep) | −14.84 % [−17.00, −11.70] 16/16 | −19.68 % [−25.11, −7.00] 14/16 |

* **Baseline (P2 PASS):** histogram ABF = kernel ABF: e_F(T) −0.4 %, I_F −0.2 %; per-seed paired ratio
  (same noise stream) 0.988 / 1.058. Unlike EB (where the histogram repaired a 23× kernel stall) the LTA
  kernel baseline was already on the plateau, as the read-out audit said.
* **Replication (campaign clauses: REPLICATES; P1 as worded: integrated PASS, endpoint FAIL):** the
  integrated gain is reproduced to within 0.7 points (−14.1 vs −14.8 %, 16/16, CI excludes zero). The
  endpoint gain shrinks from −19.7 % to a paired median of −7.2 % whose CI touches zero (11/16), while
  the ratio of the MEDIAN final errors is −24 % for both estimators: the unsmoothed per-seed endpoint is
  noisier, so the paired per-seed ratio is pulled toward zero. Same pattern as EB and the gateway under
  this estimator ("integrated replicates, endpoint shrinks"). Time to ABF's final accuracy 1.32× (kernel 1.56×).
* **Attribution (P4 substantively PASS):** the matched sham — identical event schedule, random direction —
  is exactly neutral (−0.24 % [−2.2, +1.4], 8/16; its curve lies on the ABF curve) with the same
  genealogy load as FR (ESS 0.44 vs 0.41). FR beats its own sham directly by −13.17 % [−14.6, −12.5]
  16/16, i.e. 93 % of its FR-vs-ABF margin: the gain is the Fisher–Rao DIRECTION, not resampling per se.
  (P4's clause "sham final in [0, +10 %]" is missed by a hair: −1.2 %.)
* **Health (P6 FAIL on the event count, PASS on ESS):** histogram FR fires the SAME number of events as
  kernel FR (10 492 vs 10 483, +0.1 %), not +30–70 % as on WCA — the LTA event count is set by the FR
  marginal KDE (object B, unchanged), not by the estimator. Median min ESS/N 0.414 ≥ 0.30, wmax 0.009.
* **Screening rule (P5 PASS):** histogram ABF: first window visit t = 1.8, time to 1.5× final 37.2 of 60
  (kernel 36.6); satisfied with the final-error threshold, not with e₀/8 (τ 2.4).
* **Verdict at 300 K:** histogram FR = SAFE_ACCELERATOR (ΔI_F −14.1 %, CI < 0, final non-inferior,
  health ok); sham = NEUTRAL; attribution = FR beats sham (integrated and final).

### 4.3 The 150 K movie (seed 1200; `results/lta_movie/T150/README.md`)

ABF e_F(T) 0.1763 kJ/mol, FR 0.0897 (−49.1 %), I_F −30.7 %, 837 replacements, min ESS/N 0.322; FR
holds ABF's final accuracy from t = 21.6 (ABF only at the end); window crossings 917 (ABF) vs 1 085 (FR) (first record; the genealogy re-simulation gives 0.1765 / 0.0951, −46.2 %, −30.3 %, 819 events, see §4.9).
The starved phase is on screen: of the 254 replacements between t = 4 and 10, 251 deaths are in the cage
and 249 births go to the neck / window; the FR marginal reaches the uniform level by t = 20 while ABF's
needs until t ≈ 30. Configuration cloud at 150 K (ABF arm): transverse rms 2.97 Å (cage) vs 0.18 Å
(window), ⟨|cos θ|⟩ 0.55 vs 0.97 — the window is tighter and more aligned than at 300 K (0.22 Å, 0.955),
consistent with −TΔS‡ 8.2 vs 7.8 kT (`figures/fig_lta_configuration_cloud_T150`).

### 4.4 150 K, three arms, 16 seeds (production 20:43–21:50 UTC)

| arm (histogram) | e_F(T) median | I_F median | events | crossings / replica | median min ESS/N (worst) |
|---|---|---|---|---|---|
| abf | 0.1378 | 28.08 | 0 | 0.92 | – |
| fr_uniform (0.20) | 0.0713 | 20.69 | 12 562 | 1.09 | 0.325 (0.294) |
| fr_sham | 0.1360 | 27.63 | 12 562 | 0.93 | 0.400 (0.370) |
| kernel abf / kernel fr (sweep) | 0.1342 / 0.0628 | 25.12 / 17.25 | 0 / 12 521 | 0.94 / 1.10 | – / 0.319 |

| contrast | ΔI_F | Δe_F(T) |
|---|---|---|
| **hist FR vs hist ABF** | **−28.07 % [−29.21, −26.25] 16/16** | **−52.60 % [−56.63, −44.01] 16/16** |
| hist sham vs hist ABF | −0.94 % [−2.34, +1.15] 9/16 | −2.92 % [−7.32, +2.67] 10/16 |
| **hist FR vs hist sham (direct)** | **−27.03 % [−30.03, −24.86] 16/16** | **−50.65 % [−56.90, −43.29] 16/16** |
| kernel FR vs kernel ABF (sweep) | −31.92 % [−33.45, −28.30] 16/16 | −56.31 % [−59.94, −43.24] 16/16 |

* **P1 PASS at 150 K** (gaps 3.8 / 3.7 points, both CIs exclude zero, 16/16): here the endpoint gain
  replicates too — the starved cell's endpoint is a large effect, not a per-seed-noise one.
* **P2 PASS:** histogram ABF e_F(T) +2.7 %, I_F +11.8 % vs kernel ABF (paired 1.020 / 1.134): the own-bin
  bias costs a little integrated accuracy here, inside the +25 % clause.
* **Attribution:** sham neutral (−0.9 % integrated, −2.9 % final), same genealogy load; FR beats its sham
  directly by −27.0 % / −50.7 %, 16/16 on both endpoints — 96 % of the FR-vs-ABF margin.
* **Mechanism:** FR raises the window traffic (1.09 vs 0.92 crossings per replica, +18 %; the sham 0.93
  = ABF): the direction, not the resampling, generates the extra crossings.
* τ to ABF's final accuracy: 25.2 vs 60.0 (2.38×; kernel 2.56×). ESS 0.325 ≥ 0.30 (worst 0.294); events
  +0.3 % vs kernel FR (P6's +30–70 % again not seen).
* **Verdict at 150 K:** SAFE_ACCELERATOR; sham NEUTRAL; FR beats sham.

### 4.5 350 K — the new point (production 21:51–22:58 UTC)

| arm (histogram) | e_F(T) median | I_F median | events | crossings / replica | median min ESS/N (worst) |
|---|---|---|---|---|---|
| abf | 0.0623 | 22.20 | 0 | 3.23 | – |
| fr_uniform (0.20) | 0.0583 | 19.46 | 10 291 | 3.29 | 0.436 (0.403) |
| fr_sham | 0.0628 | 22.45 | 10 291 | 3.26 | 0.442 (0.415) |

| contrast | ΔI_F | Δe_F(T) |
|---|---|---|
| **hist FR vs hist ABF** | **−11.77 % [−15.23, −9.47] 16/16** | −9.07 % [−22.03, −0.15] 12/16 |
| hist sham vs hist ABF | +0.16 % [−2.60, +1.74] 7/16 | −6.28 % [−15.24, +14.17] 9/16 |
| **hist FR vs hist sham (direct)** | **−12.62 % [−14.27, −9.04] 16/16** | −12.12 % [−19.25, +8.12] 10/16 |

* **P3 — the predictor contrast.** The entropic share is the largest of the five temperatures (74 %)
  and the benefit is the SMALLEST: −11.8 % integrated vs −14.1 % at 300 K (histogram, same estimator),
  −28.1 % at 150 K; ABF's own window traffic the largest (3.23 crossings per replica vs 2.66 at 300 K,
  0.92 at 150 K). The starvation reading predicted the direction and roughly the size (ΔI_F in
  [−12, 0]: −11.8); the entropy-share reading predicted growth and is refuted a fifth time. Where P3's
  wording erred: the benefit did not reach "NEUTRAL" — at −11.8 % with a CI excluding zero and
  16/16 wins the frozen rule still returns SAFE_ACCELERATOR (the analyzer therefore marks P3 as not
  passed on its verdict clause; the quantitative clauses pass). So the shrinkage is gradual: across
  150 → 300 → 350 K the integrated gain goes −28 → −14 → −12 % as the crossings per replica go
  0.9 → 2.7 → 3.2.
* **Endpoint:** −9.1 % paired [−22.0, −0.15], 12/16 — small, borderline; τ to ABF's final accuracy
  1.03× (58.2 vs 60.0): at this T the endpoint is essentially the same, the gain is in the transient.
* **Attribution holds:** sham +0.2 % (neutral), FR beats sham directly by −12.6 % 16/16 on the integrated
  endpoint; on the final the direct contrast is not resolved (−12 % [−19, +8]).
* Health: ESS 0.436, wmax 0.009; events 10 291 (fewer than at 300 K: 10 492, and 150 K: 12 562 — the
  event count falls as the marginal flattens sooner). P0 floor share 0.095 at the smaller 350 K ABF error.
* Screening rule on the histogram ABF arm: T_hit 1.2, T_est 40.2 of 60 — satisfied, like 150 and 300 K.

### 4.6 225 K, three arms, 16 seeds (production 22:58–00:06 UTC)

| arm (histogram) | e_F(T) median | I_F median | events | crossings / replica | median min ESS/N (worst) |
|---|---|---|---|---|---|
| abf | 0.0870 | 25.27 | 0 | 1.77 | – |
| fr_uniform (0.20) | 0.0607 | 20.32 | 11 040 | 1.86 | 0.390 (0.351) |
| fr_sham | 0.0848 | 25.51 | 11 040 | 1.77 | 0.428 (0.406) |
| kernel abf / kernel fr (sweep) | 0.0948 / 0.0608 | 23.04 / 18.48 | 0 / 11 151 | 1.80 / 1.87 | – / 0.385 |

| contrast | ΔI_F | Δe_F(T) |
|---|---|---|
| **hist FR vs hist ABF** | **−19.90 % [−22.23, −17.26] 16/16** | **−30.82 % [−44.92, −18.89] 16/16** |
| hist sham vs hist ABF | +0.12 % [−2.24, +1.24] 7/16 | −7.09 % [−13.38, +4.29] 11/16 |
| **hist FR vs hist sham (direct)** | **−20.80 % [−22.84, −17.38] 16/16** | **−32.37 % [−44.38, −16.87] 15/16** |
| kernel FR vs kernel ABF (sweep) | −21.28 % [−23.21, −18.13] 16/16 | −28.17 % [−35.15, −24.14] 15/16 |

P1 PASS (gaps 1.4 / 2.7 points, CIs exclude zero, 16/16); P2 PASS (hist ABF e_F(T) −8.2 %, I_F +9.7 %
vs kernel; paired 1.051 / 1.106); sham neutral; FR beats its sham on both endpoints; τ to ABF's final
accuracy 1.45× (kernel 1.67×); ESS 0.390; events 11 040 (kernel 11 151, −1 %). Verdict SAFE_ACCELERATOR.

### 4.7 80 K, three arms, 16 seeds (production 00:06–01:13 UTC; rate 0.10 as in the sweep)

| arm (histogram) | e_F(T) median | I_F median | events | crossings / replica | median min ESS/N (worst) |
|---|---|---|---|---|---|
| abf | 0.1741 | 26.53 | 0 | 0.226 | – |
| fr_uniform (0.10) | 0.0289 | 17.47 | 10 605 | 0.433 | 0.262 (0.237) |
| fr_sham | 0.1709 | 26.53 | 10 605 | 0.226 | 0.432 (0.413) |
| kernel abf / kernel fr (sweep) | 0.1638 / 0.0406 | 23.96 / 15.48 | 0 / 10 744 | 0.234 / 0.432 | – / 0.257 |

| contrast | ΔI_F | Δe_F(T) |
|---|---|---|
| **hist FR vs hist ABF** | **−35.14 % [−36.23, −33.35] 16/16** | **−83.36 % [−84.59, −82.57] 16/16** |
| hist sham vs hist ABF | −1.16 % [−3.48, +1.11] 9/16 | −3.44 % [−9.44, +3.11] 9/16 |
| **hist FR vs hist sham (direct)** | **−34.45 % [−36.52, −33.56] 16/16** | **−83.63 % [−84.08, −81.99] 16/16** |
| kernel FR vs kernel ABF (sweep) | −35.14 % [−37.31, −32.71] 16/16 | −74.74 % [−77.23, −73.17] 16/16 |

* The integrated gain reproduces to the second decimal (−35.14 vs −35.14 %); the endpoint gain is
  LARGER under the histogram (−83.4 vs −74.7 %), exactly what the offline read-out audit predicted for
  raw bins (−82.0 %, `docs/LTA_READOUT_SWEEP.md`): the kernel smoothing was hiding part of FR's endpoint
  advantage in the starved cell. P1 PASS; P2 PASS (hist ABF e_F(T) +6.3 %, I_F +10.7 %).
* **Mechanism, cleanly attributed:** ABF alone makes 0.226 crossings per replica in the whole run; FR
  makes 0.433 (1.9×) and the sham, with the identical replacement schedule, 0.226 = ABF. The extra window
  traffic is generated by WHERE the clones are placed, not by the resampling itself.
* τ to ABF's final accuracy 11.4 vs 60.0 (5.26×; kernel 6.67×). Health: median min ESS/N 0.262 < 0.30
  (kernel 0.257) → the SAFE label is withheld by the frozen rule, as in the sweep: ACCELERATION_POSITIVE.
  The sham's ESS (0.432) is much higher than FR's at the same event count — directed selection
  concentrates lineages, random selection does not.

### 4.8 Width ladder, clause (b) (01:13–01:34 UTC; ABF only, 300 K, R = 4 noise-paired across widths)

| n_grid | Δφ (Å) | e_F(T) median | I_F median | P0 floor (kJ/mol) | floor share |
|---|---|---|---|---|---|
| 90 | 0.132 | 0.0905 | 23.25 | 0.0166 | 0.18 |
| **180 (frozen)** | 0.066 | **0.0862** | **23.26** | 0.0048 | 0.055 |
| 360 | 0.033 | 0.0915 | 23.71 | 0.0011 | 0.012 |

180 bins is the minimum of the ladder on both endpoints: 360 is WORSE by +7.2 % [+2.4, +15.2] on
e_F(T) (0/4 wins) and +1.5 % on I_F (variance of the finer bins), 90 by +5 % on e_F(T). The formal
plateau clause "≤ 5 % change to the next finer width" is therefore not met by 180 — because the finer
width degrades, not because 180 under-resolves. Every width passes the floor clause. The frozen choice
stands; nothing was re-selected (`figures/fig_lta_hist_width_ladder`).

## 5. Synthesis (all five temperatures, 16 seeds × 3 arms each; `results/lta_histogram/scoreboard.md`)

| T (K) | ΔF‡ (kT) | entropic share | ABF crossings / replica | hist FR vs ABF, ΔI_F | kernel sweep | hist FR vs ABF, Δe_F(T) | kernel sweep | sham vs ABF, ΔI_F | FR vs sham (direct), ΔI_F | τ speed-up | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 80 | 18.1 | 47 % | 0.23 | **−35.1 %** 16/16 | −35.1 % | −83.4 % 16/16 | −74.7 % | −1.2 % | −34.5 % 16/16 | 5.3× | ACCEL. POSITIVE (ESS 0.26) |
| 150 | 13.5 | 61 % | 0.92 | **−28.1 %** 16/16 | −31.9 % | −52.6 % 16/16 | −56.3 % | −0.9 % | −27.0 % 16/16 | 2.4× | SAFE |
| 225 | 11.7 | 68 % | 1.77 | **−19.9 %** 16/16 | −21.3 % | −30.8 % 16/16 | −28.2 % | +0.1 % | −20.8 % 16/16 | 1.45× | SAFE |
| 300 | 10.8 | 72 % | 2.66 | **−14.1 %** 16/16 | −14.8 % | −7.2 % 11/16 | −19.7 % | −0.2 % | −13.2 % 16/16 | 1.32× | SAFE |
| **350 (new)** | 10.3 | 74 % | 3.23 | **−11.8 %** 16/16 | – | −9.1 % 12/16 | – | +0.2 % | −12.6 % 16/16 | 1.03× | SAFE |

1. **The LTA result survives the estimator change (P1).** Under the textbook histogram estimator the
   integrated gain reproduces the kernel sweep at every replicated T to within 0.0–3.8 points, 16/16
   everywhere, and the endpoint gain reproduces at 80–225 K (at 80 K it GROWS to −83 %, as the raw-bin
   read-out audit predicted). Only the 300 K endpoint shrinks (paired −7 %, CI touching zero; the ratio
   of median final errors is −24 % for both estimators): at the mildest cell the per-seed endpoint is
   noise-limited. Histogram ABF itself equals kernel ABF within 1–8 % on e_F(T) and +10–12 % on I_F (P2).
2. **The matched sham settles attribution (P4).** With the identical replacement schedule and random
   direction the sham is neutral at every T (|ΔI_F| ≤ 1.2 %, its error curve lies on ABF's, its window
   traffic equals ABF's), while FR beats its sham directly by 93–98 % of its FR-vs-ABF margin, 16/16 at
   every T. The gain is the Fisher–Rao direction. At 80 K FR doubles the window crossings; the sham does
   not touch them.
3. **The 350 K point (P3) resolves the predictor question in the same direction as the sweep.** The
   entropic share is largest there (74 %) and the benefit smallest (−11.8 %), continuing the monotone
   relation with ABF's own window traffic (0.23 → 3.23 crossings per replica maps onto −35 → −12 %).
   The entropy-share reading predicted growth and is refuted a fifth time; the starvation reading
   predicted the direction and the size. P3's wording was too strong in one respect: the benefit did
   not reach neutrality — at −11.8 % with 16/16 wins the frozen rule still returns SAFE_ACCELERATOR.
   Within the budget the ABF endpoint is essentially reached by both arms (τ 1.03×); the gain is in the
   transient.
4. **The screening rule works with the right threshold (P5).** On both estimators' ABF arms the first
   window visit is at t = 1.2–3.0 (T_hit/T_run 0.02–0.05) and the time to come within 1.5× of the FINAL
   error is 34–43 of 60 (0.56–0.71) at every T — inside the proposed window. Tied to the INITIAL error
   (τ(e₀/8) = 1.8–4.2) the same rule would call every T "too easy". For this system the relevant slow
   variable is the tail of marginal establishment (the KL-to-uniform curves in
   `figures/fig_lta_hist_establishment`), and the threshold must be tied to the endpoint.
5. **Health (P6).** The histogram FR fires the SAME number of replacements as the kernel FR at every T
   (−1 to +0.3 %), unlike WCA (+65 %): on LTA the event count is set by the FR marginal KDE (object B,
   unchanged), not by the estimator. ESS floors met at 150–350 K (0.33–0.44); 80 K at 0.26 as in the
   sweep. The sham, at equal event count, keeps ESS 0.40–0.44 — directed selection concentrates lineages.
6. **Width.** 180 bins is admissible by the floor clause (P0 floor share 0.018–0.055 at 80–300 K and
   0.095 at 350 K, limit 0.20) and is the optimum of the ABF-only ladder; the plateau clause is formally
   unmet only because 360 bins is worse.

**Verdict.** Ethane/LTA under the histogram estimator: uniform-target Fisher–Rao is a safe accelerator
at 150–350 K and an unsafe-by-ESS but large accelerator at 80 K, the gain is attributable to the
Fisher–Rao direction (sham neutral), and the benefit shrinks monotonically with ABF's own ability to feed
the window — the fifth temperature extends the trend rather than reversing it. The two movies
(`results/lta_movie/T300`, `T150`) show the mechanism on single seeds: deaths in the over-populated cages,
births into the neck and window while the ABF bias is still learning the barrier.

**Not done / caveats.** The full CPU test suite was not re-run to completion (a 45-min run timed out; only
`tests/test_lta_histogram.py` (8) and the existing histogram tests (28) were verified; no other module
imports the LTA engine). The temperature axis remains a budget axis as much as a landscape axis (the
sweep's caveat). The sham's clone sources are drawn among survivors with replacement (WCA semantics);
the FR arm's multinomial can also repeat a source. 350 K has no kernel arm; its comparison is within the
histogram estimator only. P3 and P4 are marked "not passed" by the analyzer on their label / range
clauses (verdict SAFE rather than NEUTRAL; sham final −1 to −7 % rather than in [0, +10]); the
quantitative claims they encode hold.

## 6. Artifacts

`configs/lta_histogram/campaign.json` (frozen design + execution log), `scripts/run_lta_histogram.py`,
`scripts/lta_histogram/launch.sh`, `scripts/analyze_lta_histogram.py`, `scripts/lta_movie_render.py`,
`scripts/plot_lta_configuration_cloud.py`, `scripts/plot_lta_decomposition_ladder.py`,
`tests/test_lta_histogram.py` + `tests/fixtures/lta_pre_histogram_fixture.npz`;
`results/lta_histogram/{summary.json, scoreboard.md, comparison_T*.csv, figures/, calibration/, pre_run_kernel_only/}`
(raw `production_T*/*.npz` and ladder npz are git-ignored, reproducible from the launcher);
`results/lta_movie/T{300,150}/{*.mp4, README.md, movie_summary.json}`;
`results/uniform_campaign/lta/reference/reference_T350.npz`. Timeline (UTC): design frozen 18:50,
chain 18:51–01:34 (movie 300 K 13 min, reference 350 K 7 min, ladder 11 min, five productions 67.5 min
each, width ladder 21 min), GPU 3 only.

## 4.9 The ensemble movie (2026-10-04): visualising the population, not the molecules

**Why.** In the particle movies both columns show 1024 dots filling two cages; at any instant the arms
differ only by a shift of window occupancy (11 % vs 20 % at t ≈ 9, 150 K), invisible in a cloud. The method
acts on the empirical measure p̂_t = N⁻¹ Σ δ_{ξ(q_i)}, so the movie should show that measure and the
genealogy of the population. Design agreed with the user (three acts: the physical problem with one
replica → the ensemble as a replica strip with the p̂/q heat strip, a window-occupancy meter and highlighted
families → the space–time kymographs with lineage trees and R_W(t)).

**Genealogy record.** `store_snapshots` now also stores the initial-ancestor label of every slot at every
snapshot and, per event, (dying slot, source slot) and (child id, parent id); bit-inert, tested
(`tests/test_lta_histogram.py`). Both movie seeds were re-simulated with it (GPU 3, 13 min each). A CUDA
re-run is not bitwise identical to the first record (scatter-add order): identical until t = 4, the first
254 (150 K) / 145 (300 K) events land in the same regions, endpoints drift (150 K FR −49.1 → −46.2 %,
300 K −10.1 → −11.1 %). Particle movies and READMEs were re-rendered from the new records; the first
records are archived in `raw_pre_genealogy/`.

**What the genealogy says (150 K, FR arm, 819 replacements in 1024 slots):**

| t | copies among window replicas | copies among cage replicas | families alive (of 1024) | largest family |
|---|---|---|---|---|
| 6 | 40 % | 0 % | 904 | 6 |
| 8 | 52 % | 4 % | 822 | 8 |
| 10 | 48 % | 6 % | 775 | 9 |
| 20 | 40 % | 24 % | 681 | 10 |
| 60 | 57 % | 54 % | 516 | 13 |

Generation depth of the copies: 492 first-generation, 220 second, 73 third, 22 fourth, 12 fifth. So the
"one discovery → many descendants" picture is real but **collective**: the window is populated by copies
of the walkers that found it (half of the window population at t = 8, none of the cage population) while
originals trickle in, through hundreds of small families; the largest family ever has 13 members. That is
exactly what keeps the ancestor ESS/N at 0.31–0.45 — the dose is set so that no lineage takes over. At
300 K the same quantities are 20–22 % vs 1–6 % (t = 6–10) and the largest family has 6 members. The
first three walkers to visit the window (t = 1.3, 1.7, 1.9, before FR starts, hence the same walkers in
both arms) end with families of 3, 0 (extinct) and 6; the three largest families (13, 11, 10) descend from
walkers that sat in or near the window at the first FR opportunities (t = 4.2–4.3).

**Renderer** `scripts/lta_ensemble_movie_render.py` (acts I–III, beeswarm strip with originals grey and
copies tinted, the three largest families coloured, p̂/q heat strip, R_W meters, kymographs with lineage
trees, a copies-in-window curve, data-driven closing line). Prototype static view
`scripts/plot_lta_ensemble_kymograph.py` → `figures/fig_lta_ensemble_kymograph_T150`.

**Movies rendered** (04:40 UTC): `results/lta_movie/T150/lta_ensemble_abf_vs_fr.mp4` and
`results/lta_movie/T300/lta_ensemble_abf_vs_fr.mp4` (≈34 s each, 1920×1080, 30 fps; READMEs updated).
Highlighted families (rule: the three largest final FR families): 150 K — initial walkers 728 / 595 / 75
(first in the window at t = 4.2–4.3, i.e. the walkers that sat near the window at the first FR
opportunities; final 13 / 11 / 10 members); 300 K — 770 / 637 / 868 (6 / 6 / 6; largest ever 7; 22 % of
the window population are copies at t = 8). Design rule learnt: colour the *copies* and the *largest*
families, not the first discoverers (3 / extinct / 6 members at 150 K).
