# WCA dimer: audit of the free-energy references and of the historical FR result

*2026-10-09, branch `research/langevin-validation-oct2026`. Phase III of the overnight
numerical-validation campaign.*

* Scripts: `scripts/numerical_validation/analyze_wca_reference_audit.py` and
  `scripts/numerical_validation/rescore_wca_history.py`.
* Data: `results/numerical_validation/wca/reference_audit.json`, `rescore_history.json`.
* Figure: `results/numerical_validation/wca/figures/fig_wca_reference_audit.png`.

The historical scores are not modified. The published scoreboards in `results/wca_replica_ladder/`
and `results/histogram_abf/wca/` stay as they are. This document adds a score against an exact
reference and labels which conclusions survive.

## 1. The arbiter

The arbiter is exact Gibbs sampling of the intended potential (WCA + double-well dimer + RC wall, no
force clip, no `min_r` clamp) by Metropolis Monte Carlo:

* 64 independent chains, 3e7 sweeps each after 2e4 burn-in sweeps;
* single-particle moves at acceptance 0.59, plus dimer-stretch moves with the 2-D Jacobian r′/r at
  acceptance 0.48;
* integrated autocorrelation time of z below 300 sweeps.

Compared on the 160 bin centres in the evaluation window [−0.1, 1.1], each profile centred:

* **Exact-sampler checks.** MC reproduces the analytic dimer-only density: well population 0.2250 ± 0.0002
  against 0.2252. In the full system, the mean-force route and the density route of the *same* MC chains
  agree to D = 0.0002 kT (95 % upper bound 0.0007). This validates the estimator
  f = (w/r) d·(F1 − F0) − 2w/(βr), Jacobian term included, independently of any dynamics.
* **Precision.** MC F_density has noise 0.0006 kT RMS (jackknife over chains).

D is the debiased RMS difference from the MC profile (prereg statistic). All values are in kT.

## 2. Every historical reference against exact Gibbs

| reference / profile | how it was built | D vs exact | max abs diff |
|---|---|---|---|
| **v2 TI = the accepted reference** (`cache/phase_hp_v3/…g160.npz`) | constrained (projected) EM, dt 0.002, accepted clips, 61 nonuniform z, PCHIP | **0.288** | 0.90 |
| v1 TI (`cache/wca_ti_reference.npz`) | same scheme, 51 z, smoothed 1 cell | 0.282 | 0.84 |
| phase TI cache (`cache/phase/…g160.npz`) | same as v1 | 0.280 | 0.84 |
| dt-consistent ref, dt 0.002 (pooled serial ABF, 32 seeds) | production ABF at dt 0.002 | 0.225 | 0.79 |
| dt-consistent ref, dt 0.0005 (pooled serial ABF, 32 seeds) | production ABF at dt 0.0005 | 0.0145 | 0.044 |
| unbiased production EM dt 0.002: F_density / F_MF | −kT log P(z) / ∫⟨f⟩ | 0.132 / 0.308 | 0.41 / 0.88 |
| unbiased production EM dt 0.001: F_density / F_MF | | 0.022 / 0.060 | 0.08 / 0.16 |
| unbiased production EM dt 0.0005: F_density / F_MF | | 0.0063 / 0.020 | 0.023 / 0.051 |
| unbiased production EM dt 0.000125: F_density / F_MF | | 0.0024 / 0.0019 | 0.007 / 0.008 |

**The accepted WCA reference is wrong by 0.29 kT RMS and up to 0.9 kT**, with the compact well too
deep. It agrees instead with the *mean-force route of the dt = 0.002 dynamics* (0.31 from exact). The
TI references were built with the same clipped Euler–Maruyama integrator at dt 0.002 (projected onto
fixed z), so they inherit its stationary measure. They are a timestep-consistent reference for the
dt 0.002 chain, not the free energy of the model.

## 3. Separating the error sources

| source | size | evidence |
|---|---|---|
| (1) wrong physical force/potential | **none** in the smooth region; non-gradient regularisations act only when the integrator overshoots | `WCA_FORCE_AUDIT.md` |
| (2) invariant-measure discretisation (dt, with the clips) | **dominant**: 0.31 / 0.060 / 0.020 / 0.0019 kT (mean-force route) at dt 0.002 / 0.001 / 0.0005 / 0.000125 | §2; `WCA_LANGEVIN_VALIDATION.md` |
| (3) mean-force formula and Jacobian | **none**: 0.0002 kT under exact sampling | §1 |
| (4) reference statistical error | 0.001-0.006 kT (jackknife) | §2 |
| (5) smoothing / integration / interpolation conventions | ≤ 0.008 kT: v1 (smoothed) vs v2 (unsmoothed, PCHIP) differ by 0.006-0.008 | §2 |
| (6) FR finite-population / resampling bias | 0.06-0.09 kT at dt 0.002 and 0.015-0.022 at dt 0.0005, beyond the integrator's own bias; dose-dependent | §4 |

The two free-energy routes see the dt bias differently. The density route converges faster (0.13 →
0.022 → 0.006 → 0.002) than the mean-force route ABF actually uses (0.31 → 0.060 → 0.020 → 0.002). The
mean-force route is the one that matters for ABF.

## 4. The FR stationary tilt

Pooled long-run profile of ABF + FR (16 seeds) minus pooled ABF at the same dt and N (`reference_audit.json`, `fr_tilt`):

| N | dt 0.002 | dt 0.0005 |
|---|---|---|
| 1024 | 0.087 | 0.018 |
| 256 | 0.090 | 0.022 |
| 64 | 0.083 | 0.021 |
| 16 | 0.061 | 0.015 |
| 4 (cap 1, nearly inert) | 0.005 | 0.004 |

Against exact Gibbs, at dt 0.0005:

* the ABF limits are 0.013-0.014 off, a smooth tilt of −0.04 → +0.015 kT across the window, which is the
  mean-force-route dt bias;
* the ABF + FR limits are 0.025-0.034 off, roughly double the tilt in the same direction.

So at dt 0.0005, FR adds a stationary bias of its own, on top of the integrator's.

**Can the existing evidence tell an integrator × FR interaction from an intrinsic FR bias? No.** Between
dt 0.002 and 0.0005 the tilt fell 4-5×, roughly in proportion to dt. Two points cannot separate c·dt from
a + b·dt, and the dt 0.002 point lies in the clip-dominated, non-asymptotic regime. Separating them needs
the tilt at a validated dt. That measurement is the conditional Phase-V experiment
(`configs/numerical_validation/wca_confirmatory_prereg.json`, prediction P1).

Even the smaller dt 0.0005 tilt is not harmless. It is 1.3-2× the ABF limit's own bias, and it is what
makes FR harmful at every N ≥ 16 against exact Gibbs (§5).

## 5. The historical FR results, rescored against exact Gibbs

Accepted read-out (`wca_numba.score_ladder_result`). Only the reference changes. Each cell gives the
median paired change FR vs ABF [95 % bootstrap CI] (FR wins out of 16).

| set | reference | ABF e_F(T) | FR e_F(T) | ΔI_F | Δe_F(T) |
|---|---|---|---|---|---|
| **accepted histogram confirmation** (float64 replay, N 1024, dt 0.002) | v2 TI (published) | 0.093 | 0.044 | **−24.1 %** [−25.1, −23.4] (16) | −50.9 % (16) |
| | **exact Gibbs** | 0.200 | 0.269 | **+18.5 %** [+17.9, +19.6] (0) | **+33.9 %** (0) |
| ladder dt 0.002, N 1024 / 256 / 64 / 16 | v2 TI | | | −25 / −37 / −38 / −45 % (16 each) | −52 / −44 / −43 / −47 % |
| | exact Gibbs | | | **+18 / +29 / +28 / +21 %** (0 each) | +32 / +32 / +29 / +21 % |
| ladder dt 0.0005, N 1024 / 256 / 64 / 16 | v2 TI | 0.265 | 0.255 | −3 / −5 / −5 / −3 % | |
| | exact Gibbs | 0.013 | 0.028 | **+15 / +65 / +89 / +57 %** (0 each) | +130 / +131 / +136 / +80 % |
| ladder dt 0.0005, N 4 | exact Gibbs | | | −10 % [−27, +4] (12) | −18 % [−32, +7] |

**Verdict on the accepted WCA FR gain: REFUTED as a free-energy acceleration.** The −24 % integrated and
−51 % final gains exist only against the dt-0.002 TI reference. That reference was built inside the same
integrator artefact, and FR's dose-dependent tilt points toward it, so the apparent gain is a
reference-matching artefact. Against the free energy of the model:

* FR is harmful in the accepted cell (+18.5 %, 0/16);
* FR is harmful at every N ≥ 16 at both dt 0.002 and dt 0.0005;
* FR is neutral only at N = 4, where its cap of 1 event leaves it nearly inert.

Neither the kernel-estimator campaigns, the FR-start ladder, the WCA movies nor the OT/repair WCA studies
used a validated time step or reference. Their FR-vs-ABF conclusions are **not established**; they are
not refuted individually here.

Root cause, in one sentence: the accepted dt = 0.002 is beyond Euler–Maruyama's linear stability limit
for ordinary WCA collisions. The per-particle force clip keeps the chain bounded but makes its stationary
law a different fluid (mean potential energy about 1000 kT instead of 11 kT, pairs inside 0.65σ in about
3 % of steps). The reference was built with the same chain, so every profile, reference included,
inherited a 0.1-0.9 kT artefact larger than every effect being compared.

## 6. What survives

* The engine, the estimator and the mean-force formula are correct (Phase I, §1).
* At the validated time step (`WCA_LANGEVIN_VALIDATION.md`), production Euler–Maruyama reproduces exact
  Gibbs to ≤ 0.0024 kT. A WCA FR claim can now be tested properly. Whether FR helps there is the
  conditional Phase-V experiment.
* The dt 0.0005 replica-ladder conclusions (FR harmful at every N ≥ 16; serial or few-replica ABF best at
  equal budget) were drawn against the dt-consistent reference. They **also hold against exact Gibbs**,
  with larger relative harm, because the dt-consistent reference itself was only 0.0145 kT from exact.
