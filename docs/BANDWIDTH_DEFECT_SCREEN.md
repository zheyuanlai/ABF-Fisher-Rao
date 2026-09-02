# Bandwidth-defect screen for every headline system, and what it turned up

**Date:** 2026-09-02. **Status:** screen complete (no GPU); one exploratory re-score
(not preregistered); one analysis defect found in a closed study (chipped for repair).
**Scripts:** `scripts/audit_readout_smoothing_all.py` → `results/information_campaign/readout_smoothing_screen_all.json`;
`scripts/rescore_kernel_matched.py` → `results/information_campaign/kernel_matched_rescore.json`.

## 1. The screen

Take the system's reference mean force, smooth it with the engine's **own** legacy read-out
kernel (same kernel form, boundary treatment and normalisation as the engine's
Nadaraya–Watson estimator), re-integrate, and measure the aligned-RMS deviation in the
endpoint's units. Divided by the measured ABF `e_F(T)` that is the **share** of the baseline's
error that is deterministic kernel bias. The bias-only upper bound on what a sharper read-out
can buy is `predicted MSE gain = 1 / (1 − share²)`. It costs seconds.

Calibration against the three sweeps that were actually run:

| system | share | predicted | measured |
|---|---|---|---|
| ZIF-8 300 K (0.20 Å) | 0.88 | 4.61× | 5.82× |
| WCA Case IX (0.025) | 0.16 | 1.03× | 1.04× |
| LTA 80 K (0.05 rad) | 0.19 | 1.04× | 0.96× |

LTA shows why it is an upper bound: removing the kernel also raises variance. It is a
**bandwidth-defect screen** — "is a bandwidth matrix worth GPU hours on this system?" — not an
optimal-bandwidth predictor.

## 2. Every headline system at its legacy read-out

| system | h / bin | roughness | share | pred. gain | verdict on the baseline |
|---|---|---|---|---|---|
| ZIF-8 300 K | 1.34 | 0.912 | 0.88 | 4.6× | read-out-limited (confirmed 5.8×) |
| **alanine (φ,ψ)** | 1.24 | 0.938 | **0.84** | **3.4×** | read-out-limited |
| **gateway s=0.1 r=32** | 3.50 | 0.944 | **≥ 1 (saturated)** | — | read-out-limited (see §3) |
| LTA 300 K | 1.43 | 0.974 | 0.48 | 1.30× | moderate |
| LTA 225 K | 1.43 | 0.964 | 0.39 | 1.18× | moderate |
| CHA ethene 450 K | 1.26 | 0.970 | 0.35 | 1.14× | mild |
| CHA propene 600 K | 1.26 | 0.961 | 0.28 | 1.08× | mild |
| LTA 150 K | 1.43 | 0.967 | 0.25 | 1.07× | mild |
| CHA propene 450 K | 1.26 | 0.963 | 0.21 | 1.05× | negligible |
| LTA 80 K | 1.43 | 0.966 | 0.19 | 1.04× | negligible (confirmed 0.96×) |
| WCA Case IX | 2.84 | 0.950 | 0.16 | 1.03× | negligible (confirmed 1.04×) |
| R15 pentane β=1.4 / 1.6 | 4.45 | 0.65 / 0.67 | 0.06 | 1.00× | negligible (its error is conditional-limited) |

For the non-periodic engines (gateway, CHA, R15) the JSON also carries `floor_vs_ref` (the
whole deterministic floor against the scoring reference) and `disc_floor` (the trapezoid
discretisation of F′ alone, which no bandwidth removes). CHA's discretisation floor is
0.042–0.060 kJ/mol — 11–16 % of its error — and is **not** a bandwidth defect.

**Reading.** The two strong positives (WCA, LTA 80 K) and the CHA cells sit on adequate
baselines; the campaign's three "interesting" non-positives — ZIF-8 (harm → neutral once
corrected), alanine (neutral) and the gateway (transient with reversal) — are the three
read-out-limited baselines in the project.

## 3. Gateway: the share saturates, and the "reversal" is a read-out artefact

The gateway's measured ABF **mean-force** error (`final_l2_fp` 0.0694) equals the deterministic
kernel bias of its own estimator (0.0703) to 1.3 %: the baseline's F′ error is *entirely* kernel
bias. Its F error (0.0105) sits *below* the uniform-density fixed point (0.0119) because ABF's
accumulated occupancy is non-uniform and the NW kernel weights follow it — the estimator is
partly rescued from its own bias by *where the walkers were*. Marginal FR changes exactly that.

So the closed comparison (−11.8 % integrated, **+9.8 % worse at the end**, ratio crossing 1 at
t≈17–19) was re-scored against the estimator's own fixed point — the reference mean force
smoothed with the engine kernel (`h = 0.07`, reflect-padded, normalised), re-integrated.
Exploratory; the 64 pairs, the eval window and the statistics are the closed study's.

| reference | ΔI_F | Δe_F(T) | wins (final) | ratio uni/abf at t = 5 / 10 / 20 / 40 |
|---|---|---|---|---|
| raw F_ref (published) | −11.81 % [−14.04, −9.02] | **+9.82 %** [+8.20, +11.74] | 1/64 | 0.53 / 0.78 / 1.02 / **1.10** |
| kernel-matched | **−31.16 %** [−33.22, −28.41] | **−55.17 %** [−56.23, −53.47] | 64/64 | 0.52 / 0.47 / 0.46 / **0.45** |

Against the target the estimator can actually reach, uniform-FR is a **persistent 2×
accelerator** on the gateway and never reverses. The published reversal is the sign of the
occupancy-dependent kernel bias — the same mechanism that manufactured ZIF-8's +3.67 % "harm"
(corrected → +0.80 %, neutral), with the opposite sign. The frozen-bias endpoint the campaign
already reported (FR −11.15 % better) pointed the same way.

**What this does not license.** Kernel matching is not a sharper read-out; the gateway runs
saved no raw accumulators, so the corrected-read-out contrast is untested. Settling it needs a
read-out bank in `eb_abffr_core` (the `ReadoutBank` pattern from `wca_abffr_core`) and a fresh
preregistration; the system is a batched toy (128 rows in one process), so the run is minutes,
not hours. Proposed, not run.

## 4. The same re-score leaves WCA and CHA where they were

| system | ΔI_F raw → kernel-matched | Δe_F(T) raw → kernel-matched |
|---|---|---|
| WCA Case IX | −21.91 % → −24.00 % (16/16 both) | −41.76 % → −49.16 % |
| CHA ethene 450 | −5.96 % → −6.38 % | −12.20 % → −16.77 % |
| CHA propene 450 | −5.72 % → −5.92 % | −19.20 % → −22.35 % |
| CHA propene 600 | −5.96 % → −6.17 % | −26.62 % → −33.17 % |

Where the share is ≤ 0.35 the contrast moves by a few points and never changes verdict (CHA
stays NEUTRAL by the frozen −10 % bar; WCA stays SAFE). Where it is ≳ 0.8 the contrast can
flip sign. The screen tells you which case you are in before any GPU is spent.

## 5. Alanine: the closed study's primary endpoint was arm-insensitive by construction

`src/alanine/metrics_ala.py::smooth_reference` builds the "kernel-matched" reference with the
**unnormalised** wrapped Gaussian (`density2d.smooth2` of `periodic.wrapped_gaussian_kernel_matrix`;
row sum 3.096 per axis, ×9.58 in 2-D), so the reference it compares against is ≈9.6× a smoothed
F_ref. The deterministic `aligned_l2(K_unnorm F_ref, F_ref)` under equilibrium weighting is
**25.63 kJ/mol**; the closed study's measured ABF `final_eF_km_equilibrium` is **25.71 kJ/mol**.
99.7 % of the primary endpoint was a fixed reference-scaling constant common to both arms, which
is why abf and fr_uniform agreed to +0.01 % [−0.01, +0.02].

The verdict survives on the un-matched endpoints (16 paired seeds, N=2048): ΔI_F **−0.17 %**
[−0.52, +0.35], final **−0.15 %** [−0.49, +0.38] — EQUIVALENT by the ±10 % band, but with an
honest CI width. The engine's own estimator is a normalised NW ratio, so the fix is a one-line
normalisation in the analysis code, followed by a re-derivation of the alanine decision files.
That repair is scoped as a separate task (chip). With a correctly normalised kernel the
deterministic bias is 0.478 kJ/mol against a measured 0.568 — share 0.84, the second-most
read-out-limited baseline in the project.

## 6. How to use the screen

```bash
CUDA_VISIBLE_DEVICES="" python scripts/audit_readout_smoothing.py        # calibrated three
CUDA_VISIBLE_DEVICES="" python scripts/audit_readout_smoothing_all.py    # every headline system
CUDA_VISIBLE_DEVICES="" python scripts/rescore_kernel_matched.py         # exploratory re-score
```

Before running any bandwidth matrix on a new system: compute the share. Below ≈0.35 the legacy
read-out is not the story and the arm contrast is stable; above ≈0.8 fix the estimator first —
any arm contrast measured against the unsmoothed reference is contaminated, in either direction.
