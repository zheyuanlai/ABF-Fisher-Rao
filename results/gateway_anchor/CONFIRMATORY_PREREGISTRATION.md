# Confirmatory preregistration — gateway anchor, fresh seeds

**Committed before any of the seeds below have been run.** Everything numeric in this
document is frozen; nothing here may be changed after the runs start. If the result fails
the rule in §4 it is reported as a failure.

## 0. Why this run exists

The first anchor comparison (`results/gateway_anchor/production/`) selected each arm's FR
rate from a four-point ladder **and** reported that rate's confidence interval on the same
runs. That is tuning-selection bias: the reported interval is conditioned on having chosen
the best-looking rung. The numbers were useful for calibration and are not a confirmatory
result.

This run fixes the rates chosen there, changes nothing else, and measures once on seeds that
have never been used.

Two further defects in the first comparison are also fixed here:

* **One sham was not enough.** The single `sham` arm shadowed `fr_oracle` only. The oracle and
  practical arms build different targets and therefore fire different numbers of events
  (measured: 69/40 versus 128/70 deaths/clones in a short check), so the oracle's shadow was
  never an intensity-matched control for the *practical* method — which is the arm carrying
  the deployable claim. Each FR arm now has its own sham.
* **"No interval excluded zero" is not equivalence.** Failure to detect an effect is not
  evidence that the effect is negligible. The sham is now judged by a two-one-sided-tests
  (TOST) equivalence criterion against a predeclared margin.

## 1. Frozen configuration

Taken verbatim from the calibration run; **no value below may be re-tuned.**

| quantity | value |
|---|---|
| cell | `beta = 16`, `s = 0.10`, `r = 32` (`beta*H = 8 kT`, so `H = 0.5`) |
| `gamma` practical | **1.5** |
| `gamma` oracle | **0.5** |
| walkers `N` | 2048 |
| run length | `n_steps = 100000`, `dt = 4e-4`, `T = 40` |
| FR stride | `fr_every = 10`, `fr_burnin = 0`, `ramp_fraction = 0.10` |
| event cap | `max_event_fraction = 0.08` |
| score clip | `score_clip = 3.0` |
| KDE bandwidth | `eta = 0.10` |
| ABF estimator | `h = 0.07`, `min_count = 1.0`, grid 181 on `[-1.8, 1.8]` |
| target estimator | online EMA of the bias, `target_ema_rate = 0.005` |
| lineage window | `ess_window_steps = 4000` |
| initialisation | `left` (primary); `one_right` (secondary mechanism control) |
| health gates | `ESS_anc/N >= 0.30` **and** `w_max <= 0.05` |
| primary metric | `I_F = int_0^T ||F_hat_t - F||_{L2} dt` |
| seeds | **32 fresh seeds, 100–131**, disjoint from the calibration seeds 0–15 |

## 2. Arms

| arm | description |
|---|---|
| `abf` | baseline |
| `fr_estimated` | **practical mFR** — target from the online EMA of the bias. *Primary claim.* |
| `sham_practical` | matched sham for `fr_estimated`: same event times, same realised death and birth counts, same cap, identities uniform |
| `fr_oracle` | oracle mFR — target from the analytic `F`; non-deployable diagnostic |
| `sham_oracle` | matched sham for `fr_oracle` |

All five arms inside a seed share initial conditions and Langevin noise, so every comparison
is paired.

## 3. Endpoints

**Primary.** Paired relative change in `I_F` against `abf` on the same seed. Negative is
better. Reported as the median over seeds with a paired bootstrap CI (10 000 resamples, seed
`20260803`).

**Estimator-independent confirmation.** `I_F` is computed from the same accumulators the FR
mechanism perturbs, and birth–death makes replicas correlated descendants, so a gain could in
principle be a change in the estimator's statistics rather than a better bias. After each run
the learned mean force is **frozen at the same physical time** for every arm, a fresh
population — *identical across arms* — is launched under it with **no adaptation and no
birth–death**, and the free energy is reconstructed from the sampled density,
`F_hat = B - beta^{-1} log p_B`. Endpoint: `||F_hat - F||_{L2}` in `kT`, paired.

**Secondary.** `I_{F'}`, `T_hit`, `T_est`, the integrated deficit
`int_{T_hit}^{T} [Q*_+(t) - P_+(t)]_+ dt`, `ESS_anc/N`, `w_max`, realised event counts, and
the `one_right` mechanism control.

## 4. Success rule — frozen

The **practical** arm carries the claim. It succeeds if and only if **all four** hold:

1. median paired relative change in `I_F` **≤ −10 %**;
2. paired **95 % CI upper endpoint < −5 %**;
3. **≥ 24 of 32** paired seeds improve;
4. health gates pass: `ESS_anc/N ≥ 0.30` and `w_max ≤ 0.05`.

The **sham** must be shown *equivalent*, not merely undetected. Equivalence margin
**±5 %**, tested by TOST at α = 0.05, i.e.:

5. the **90 % CI** of the sham's paired relative change lies entirely within `[-5 %, +5 %]`.
   (The 95 % CI is also reported, and it is stated whenever it does not also fit.)

The headline claim — *directional Fisher–Rao selection, not generic turnover* — requires
1–4 **and** 5 for the matched sham of that arm.

**Predeclared readings of every outcome:**

| outcome | reading |
|---|---|
| practical passes 1–4, `sham_practical` passes 5 | the deployable method's gain is attributable to Fisher–Rao direction |
| practical passes 1–4, `sham_practical` fails 5 in the *improving* direction | the gain is at least partly generic turnover; the directional claim is not supported |
| practical fails 1–4 | the calibration result did not replicate on fresh seeds; report as such |
| practical passes the accuracy rule but fails the health gates | report as a rate that cannot be deployed, and do not quote its gain as the headline |
| frozen-bias endpoint disagrees with `I_F` | the online gain is an estimator effect, not a better bias; the frozen-bias reading wins |

## 5. What this run does *not* do

No rate ladder. No hyperparameter search. No additional `(s, r, beta)` cells. No new system.
If any of those turn out to be wanted, they are a separate, separately preregistered
experiment.

## 6. Standing correction to how the map is described

The `(s, r, beta)` table is a **finite-budget establishment map**, not a phase diagram over
physically different equilibrium problems. Because `beta*H` was held fixed, `beta F(x)` is
identical in every cell; under `tau = t/beta`, `ytilde = sqrt(beta) y` the longitudinal SDE is
exactly `beta`-free, and the measured `tau_est` varies 1.21× against an 8× change in `beta`
while `tau_hit` is identical to four digits
(`results/gateway_phase/production/beta_scaling_audit.json`). The anchor remains a legitimate
finite-budget comparison — methods are always evaluated at finite budget — but no claim is
made that varying `beta` produced different landscapes.

---

# Amendment 1 (2026-08-03) — a baseline-pairing defect, found by code inspection

**Appended, not edited.** Everything above stands as originally committed.

## The defect

The two FR arms carry different rates, and `gamma` is a per-configuration quantity, so the
run is executed as two batches: `[abf, fr_estimated, sham_practical]` at
`gamma = 1.5` and `[abf, fr_oracle, sham_oracle]` at `gamma = 0.5`. The runner keeps only the
**first** batch's `abf` copy, so the oracle arms are compared against a baseline drawn from a
different batch.

Initial conditions are seeded from the seed alone and therefore *do* match across batches.
The Langevin noise stream is seeded from `batch_seed` and therefore does **not**. So:

| comparison | pairing | affected? |
|---|---|---|
| `fr_estimated` vs `abf` (**primary claim**) | same batch, same noise | no |
| `sham_practical` vs `abf` | same batch, same noise | no |
| `fr_estimated` vs `sham_practical` (**attribution**) | same batch, same noise | no |
| `fr_oracle` vs `sham_oracle` (attribution) | same batch, same noise | no |
| `fr_oracle` vs `abf` | matched ICs, **unmatched noise** | **yes** |
| `sham_oracle` vs `abf` (**the TOST**) | matched ICs, **unmatched noise** | **yes** |

The primary claim and both direct arm-versus-sham contrasts are unaffected. What is affected
is the oracle arm's comparison against ABF and — importantly — the `sham_oracle` equivalence
test, whose interval is inflated by noise that does not cancel.

## Why this is being fixed, and the awkward fact about it

`sham_oracle` is the one test in the confirmatory run that **failed** (90 % CI
$[-5.63, +1.05]$, just past the $-5\%$ margin). Fixing a statistic after it fails is exactly
the shape of an analysis that gets fitted to its result, so the reasoning is recorded here
before the re-run:

* the defect is identifiable **from the code alone**, without reference to any outcome —
  two batches, one discarded baseline, a noise stream keyed to `batch_seed`;
* it is a defect in the *direction* of an inflated interval, which is precisely what a
  failed equivalence test looks like, so leaving it in place would also be a choice that
  favours a particular conclusion;
* the primary claim is untouched either way, so nothing about the headline depends on this.

## What will be re-run, and what will be reported

Re-run **the oracle batch only**, with `abf` retained and each arm compared against its own
batch's baseline. The practical batch is not re-run and its numbers do not change. Because
adding a method changes how noise is broadcast across the batch, the oracle arms' realisations
change; the corrected `fr_oracle` and `sham_oracle` numbers replace the current ones.

**Declared now, before the re-run:** the corrected `sham_oracle` TOST result is reported
whichever way it comes out. If it still fails, the manuscript continues to say that only the
practical arm supports an equivalence claim. If it now passes, the manuscript says that it
passes *only after* this correction, and this amendment is cited. The original numbers are
retained in `confirmatory_summary.json` alongside the corrected ones.
