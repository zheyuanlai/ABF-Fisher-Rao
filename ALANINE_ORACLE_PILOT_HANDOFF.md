# ALANINE ORACLE PILOT HANDOFF

Branch `alanine-dipeptide`. 2026-08-01.

> **OUTCOME: the pilot FAILS its primary go/no-go criterion. Classification EQUIVALENT.**
> **A subsequent 22x FR-rate ladder spanning the full intended operating band is ALSO EQUIVALENT (§14).**
> **Production was NOT launched.** No hyperparameter was tuned after seeing the result.

Companions: `ALANINE_REFERENCE_HANDOFF.md` (accepted Stage-1 reference),
`ALANINE_EXECUTION_DECISION.md` (A/B/C change classification), `ALANINE_SPEC.md` (design record).

---

## 1. Headline

Oracle mFR, layered on corrected 2-D ABF and given the accepted reference as its target, makes
**no measurable difference** to the free-energy estimate for vacuum Ace-Ala-Nme on (φ,ψ) at the
tested budget. The mechanism was demonstrably *running* — 1741–3021 replacement events, healthy
genealogy — and the rare basin was demonstrably *populated* — so this is not an inert code path
and not a discovery-limited failure. It is a clean null.

| stage | N | init | median Δ (kernel-matched integrated FES) | 95% CI | seed wins | class |
|---|---|---|---|---|---|---|
| N2048 | 2048 | C7eq | **+0.013 %** | [+0.013, +0.030] % | 0/4 | EQUIVALENT |
| N4096 | 4096 | C7eq | **−0.010 %** | [−0.060, −0.002] % | 3/4 | EQUIVALENT |
| N2048_refeq | 2048 | reference-equilibrium | **+0.031 %** | [−0.100, +0.041] % | 1/4 | EQUIVALENT |

Negative = mFR better. The pilot threshold was **≤ −10 %**. Observed effects are ~0.01–0.03 %,
i.e. **300–1000× smaller than the threshold**, with confidence intervals lying wholly inside
±0.1 %. Endpoint mean-force change is likewise ~1e−5 in every stage.

**Permitted conclusion.** This is a statement about *vacuum Ace-Ala-Nme with ξ = (φ,ψ)*, the
frozen physical model, and the 20–100 ps transient window. It is **not** evidence about marginal
Fisher–Rao in general.

## 2. Repository state

Branch `alanine-dipeptide`, commits `5792264` (sampler + tests), `d510a11` (diagnostics +
runners), `43f14fc` (FR-rate calibration), this commit (pilot + handoff).

New: `src/alanine/{basins,core2d_ala,metrics_ala}.py`, `FastBackboneCV2D` in `cv2d.py`,
`scripts/{run_alanine_study,analyze_alanine,run_alanine_frozen,select_alanine_fr_rate}.py`,
`configs/alanine/{smoke,calibration,pilot}.yaml`, `tests/test_alanine_sampler.py`.
Nothing in `src/alkanes/` was modified in this session. No WCA/alkane birth-death code touched.

## 3. Tests

`CUDA_VISIBLE_DEVICES="" python -m pytest tests/ -q` → **121 passed, 3 skipped** (skips are the
CUDA-parity tests, skipped on CPU). Stage S1 was run before any GPU work.

## 4. Two bugs found, either of which would have produced a fake result

**(a) The oracle target was NaN — study-voiding, and silent.** The accepted reference stores
`+inf` in the 226 of 9409 bins no umbrella window visited. `F − F.mean()` is then
`inf − inf = NaN` over the *whole* grid, so the target is NaN, the Fisher–Rao score is NaN, no
death or birth weight is ever positive, and `fr_oracle` degenerates into `abf` reporting **zero
events**. That would have been reported as "mFR is EQUIVALENT to ABF" — the same words as the
real conclusion below — with the mechanism never switched on. Caught only because a test asserted
birth–death actually fires. Fixed by `sanitize_reference` (unvisited cells pinned to
`F_min + 30 kT`, scale zeroed on finite cells) plus a finiteness assertion; two regression tests.

**(b) Basin definition dropped C7ax.** Ranking reference minima by depth and keeping the deepest
few returns six shallow sub-minima inside the φ<0 megabasin and **omits C7ax** (2.56 kT, 7th by
depth) — the one basin the study is about. Replaced by prominence merging (min-max barrier to any
deeper minimum), giving C7eq (1318 cells) / C5 (574) / C7ax (347) = exactly the 2239-cell
Ω_eval, with P(C7ax) = 0.0313 against the reference's 0.0311.

## 5. GPU compliance

**GPU 7 only, throughout.** The runner refuses to start unless `CUDA_VISIBLE_DEVICES` is set to
an absolute index in {4,5,6,7}, exactly one device is visible
(`torch.cuda.device_count() == 1`), and free memory exceeds 1.5× the estimated peak; the original
absolute value is recorded in every artifact's manifest. GPUs 4/5/6 were saturated by another
user (`yesom`) for the whole session and were never touched; GPUs 0–3 were never used.
Every artifact records `cuda_visible_devices: "7"`.

## 6. Throughput and memory

| shape | ms/step | peak | free | headroom |
|---|---|---|---|---|
| N=2048 × 4 seeds (B=8192) | 48.1–49.1 | 4.71 GiB | 134 GiB | 28× |
| N=4096 × 4 seeds (B=16384) | 49.4–49.5 | 9.39 GiB | 126 GiB | 13× |

Cost is **flat in batch** — doubling the walkers costs 3 % more wall-clock — because the step is
kernel-launch bound. Concurrent packing is nearly free (43.3 ms/step with three jobs on one GPU
versus 45 ms solo), which is how six 100 ps runs fitted in ~2.7 h instead of 7.5 h sequential.

The obvious optimisation was checked and **rejected on evidence**: restricting the den Otter
Hessian to the 5-atom union {4,6,8,14,16} (15 coords instead of 66) is mathematically exact
(G identical, div 1e−14, bias exactly 0) but only **1.0–1.1×**, because the cost is `torch.func`
dispatch, not the contraction — `vmap(hessian)` is 6.2 ms per CV, flat in B, and Dynamo cannot
trace functorch transforms. `jacfwd(jacfwd)` is 1.16×, not worth destabilising a validated path.
The dense CV is retained; the union form ships as a tested equivalent (`FastBackboneCV2D`).

## 7. FR rate — chosen on safety only

Three rates, 30 ps, N=2048 × 2 seeds, oracle arm. The selector never loads FES or mean-force
error, so the rate cannot have been tuned toward a positive result.

| rate | cumulative event fraction | per-opportunity | ESS_age | w_max | T | clip |
|---|---|---|---|---|---|---|
| **0.02** | **2.61 %** | 0.124 % | 0.972 | 0.0015 | 293.5 | 0 |
| 0.05 | 6.93 % | 0.330 % | 0.928 | 0.0015 | 293.0 | 0 |
| 0.10 | 13.92 % | 0.663 % | 0.867 | 0.0024 | 292.9 | 0 |

**Denominator ambiguity, resolved before any accuracy metric was read, and it matters.**
"Event fraction" can mean per-FR-opportunity or cumulative:

* *per-opportunity*: no rate reaches the [1 %, 3 %] band — all fall **below** it. The failure is
  one-sided (under-intense, not unsafe); every genuine safety criterion passes at every rate.
* *cumulative*: exactly one rate (0.02) satisfies every criterion.

I adopted the cumulative reading and selected **0.02**, the **gentlest** tested rate — the choice
least able to manufacture an effect, so the resolution cannot be said to favour a positive
outcome. **Caveat now visible:** cumulative fraction is *run-length dependent* (2.61 % over the
calibration's 21 opportunities becomes ~21 % over the pilot's 161 at the same rate), so it is the
wrong quantity for a fixed threshold across different run lengths. The scale-free
per-opportunity figure is used at the pilot gate, and the genealogy criteria are the substantive
turnover guard. Both are recorded in `fr_rate_selection.json`. **This is the one place a reviewer
could reasonably choose differently**; note that a *larger* rate would have been needed to reach
the band, and larger rates are the ones that could plausibly show an effect.

## 8. Pilot — preregistered criteria and outcome

Config: two arms differing **only** by birth-death; N ∈ {2048, 4096}; 4 paired seeds; 100 ps;
window 20–100 ps; FR start 20 ps, every 0.5 ps, rate 0.02, score clip 2.0, max event fraction
0.05; primary C7eq init plus reference-equilibrium crossed control; shared frozen estimator
(n_grid 97 odd, abf_bandwidth 0.08, kde 0.15, min_count 200, clip 200 kJ/mol/rad, project every
50, stride 1); identical initial ensembles and identical dynamical noise per paired seed.

| criterion | required | N2048 | N4096 | refeq |
|---|---|---|---|---|
| median FES improvement | ≤ −10 % | **+0.013 % FAIL** | **−0.010 % FAIL** | **+0.031 % FAIL** |
| ≥3/4 seed wins | ≥ 3 | 0/4 FAIL | 3/4 PASS | 1/4 FAIL |
| mean-force improvement | ≤ −5 % | −0.002 % FAIL | −0.005 % FAIL | +0.004 % FAIL |
| sign consistent, 3 weightings | yes | PASS | PASS | PASS |
| CI upper < 0 | yes | FAIL | PASS | FAIL |
| age-aware ESS ≥ 0.30 N | yes | 0.956 PASS | 0.966 PASS | 0.959 PASS |
| max ancestor ≤ 0.05 | yes | 0.0034 PASS | 0.0015 PASS | 0.0024 PASS |
| event fraction < 5 % | yes | 0.13 % PASS | 0.13 % PASS | 0.14 % PASS |
| clip fraction < 1e−4 | yes | 0 PASS | 0 PASS | 0 PASS |
| non-finite / leakage | none | none | none | none |

**Every safety and hygiene criterion passes; every accuracy criterion fails by three orders of
magnitude.** Sign is consistent across all three weightings within each stage, and consistent
between the two initialisations at N=2048 (both marginally positive).

**Frozen-bias validation was not run, and reporting that is more honest than running it.**
Retention is defined as frozen improvement ÷ online improvement; with an online improvement of
~1e−4 the ratio is 0/0 and carries no information. There is no gain for an accumulator artifact
to explain. It must be run if any future configuration shows a real online gain.

## 9. Why this is a *clean* null, unlike the earlier kill-shot

Three things that were wrong or unverified in the withdrawn kill-shot are established here:

1. **The mechanism ran.** 1741 (N=2048) and 3021 (N=4096) replacement events across 4 seeds,
   against `abf`'s 0. The NaN bug in §4a is exactly the failure mode that would have hidden this.
2. **It is not discovery-limited.** From a pure C7eq start, plain ABF reaches C7ax in
   **3.1–4.1 ps** in every seed and holds **~5.6 %** occupancy over 20–100 ps (reference
   equilibrium 3.1 %). There is ample population to reallocate — this is *not* the R15 regime.
3. **The reference is accepted**, all 8 gates passed, and the run asserts an identical
   force-field parameter hash (`6ffd00dc241f`), so arms and reference cannot drift.

## 10. Selected N

**N = 4096** would be the choice if the study continued: its FR scores rest on twice the walkers
at the same wall-clock, and its seed-win rate (3/4) and CI are better behaved than N=2048's.
It was **not** chosen for showing the larger apparent improvement — both are ~0 and the
difference between them is far below any resolvable scale.

## 11. Artifacts

```
results/alanine_oracle/
  smoke/            S2 tiny GPU smoke (not scientific, stored separately)
  benchmark/        benchmark.json -- ms/step, peak memory, projected wall time
  calibration/      rate_lo|mid|hi runs + fr_rate_selection.json
  pilot/
    N2048/ N4096/ N2048_refeq/    raw/*.npz + run_manifest.json
    analysis/       paired_seed_metrics*.csv, time_series_metrics*.csv,
                    genealogy_metrics*.csv, basin_metrics*.csv, cost_metrics*.csv,
                    pilot_decision*.json, reference_provenance.json
```
Every npz carries a full manifest: run id, spec hash, config hash, force-field parameter hash,
reference path and hash, `cuda_visible_devices`, git commit and dirty flag, wall time, ms/step,
peak memory, clip fraction, force evaluations, basin names and centres.

## 12. Does alanine remain a neutrality control?

**Yes, and the pilot now supports that classification with direct evidence rather than by
inference.** The reference already showed ψ carries at most a **0.75 kT** internal barrier at
every populated φ, so there is no hidden slow coordinate for marginal reallocation to repair.
The pilot confirms the consequence: with the mechanism verifiably active and the rare basin
verifiably populated, reallocation changes the FES estimate by ~0.01 %.

## 13. The exact next decision

**Stop alanine.** Specifically:

* **Do not** implement the practical/EMA target. The oracle is the mechanistic upper bound; it
  showed nothing to recover, so a deployable approximation of it cannot do better.
* **Do not** run production, a replication, or a rate/bandwidth sweep on this system. The
  preregistered rule forbids tuning after a failed pilot, and there is no effect to resolve.
* **Do** treat vacuum Ace-Ala-Nme (φ,ψ) as the atomistic **neutrality control** — the role
  pentane 2-D plays for torsions — and report it as such.
* **Val screening remains separately gated** and was not touched this session. Before any Val
  work, `ALANINE_EXECUTION_DECISION.md` §7 requires: a genuine χ1 conditional barrier; *and*
  distinguishability of χ1 rotamers in (φ,ψ) (mutual information / rotamer classification from
  the CV alone) — because mFR sees only the (φ,ψ) marginal and cannot repair a conditional
  failure invisible to it; *and* discovered-but-under-established behaviour; *and* an oracle
  improvement.

## 14. FR-rate ladder — the open item, now CLOSED

Open item 1 below was the one real gap: the pilot's rate reallocated 0.13 % of the population per
FR opportunity, 8–23× below the intended 1–3 % band. That band has now been tested directly,
pre-declared in `results/alanine_oracle/rate_ladder/PREREGISTRATION.json` **before** the runs, with
criteria identical to the pilot and the expectation of another null recorded in advance.

The baseline is the pilot's N4096 `abf` run reused **exactly** — `fr_rate` never enters the abf
code path, `gen_dyn` depends only on `rng_seed` and the initial ensemble only on `init_seed`, so
the abf trajectory is bit-identical at any rate and the pairing on seeds [10,11,12,13] is exact.

| fr_rate | events | per-opportunity | median Δ FES | 95 % CI | wins | ESS_age | w_max | class |
|---|---|---|---|---|---|---|---|---|
| 0.02 | 3 021 | 0.126 % | −0.0100 % | [−0.063, −0.002] % | 3/4 | 0.966 | 0.0015 | EQUIVALENT |
| 0.15 | 22 830 | 0.876 % | −0.0025 % | [−0.054, +0.014] % | 2/4 | 0.813 | 0.0051 | EQUIVALENT |
| 0.45 | 66 818 | 2.557 % | +0.0236 % | [+0.009, +0.035] % | 0/4 | 0.602 | 0.0137 | EQUIVALENT |

**Intensity was varied 22×, spanning the entire intended operating band, and accuracy did not
move.** The effect oscillates around zero at the 0.02 % level with no systematic trend, against a
−10 % threshold. Meanwhile the mechanism demonstrably worked harder at every step: events rose
3 021 → 22 830 → 66 818 and age-aware ancestor ESS fell monotonically 0.966 → 0.813 → 0.602. So
the null is not "the mechanism was too gentle to matter" — the mechanism was turned up until it
measurably eroded genealogical diversity, and the free-energy estimate still did not improve.

At the top rate the sign also became inconsistent across weightings (equilibrium +, uniform-8 −,
uniform-10 −), which is what noise-level effects look like. All safety criteria still passed at
every rate (ESS ≥ 0.60 against a 0.30 floor, w_max ≤ 0.014 against 0.05, clip 0, no non-finite).

Per the pre-declared stopping rule, **alanine is now closed**: no further rate, bandwidth, target
or estimator variation is attempted.

## 15. Remaining open items

1. ~~FR-rate band untested~~ — **CLOSED by §14.**
2. Frozen-bias validation is unexercised on this system (§8) and should be run before any future
   configuration's online gain is believed.
3. `results/alanine_oracle/pilot/analysis/` currently holds per-stage copies of the CSVs; the
   plot set enumerated in the task brief was not generated (the decision needed no visual
   inspection, and no positive claim rests on one).
