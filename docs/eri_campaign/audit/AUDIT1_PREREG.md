# AUDIT 1 — preregistration, design and inference lens

*Independent adversarial audit, 2026-10-16. Read-only. No existing file modified, nothing committed, no GPU.*
*Scratch scripts: `/tmp/claude-1008/-home-zheyuanlai-ABF-Fisher-Rao/d4317313-dc1c-4e55-8b95-fc34dea3264c/scratchpad/audit_prereg/s1..s9`.*

**Blindness statement.** I did NOT open, load or classify any file in `results/eri_campaign/screen_dilute/`. Every
classifier statistic below is computed on data the coordinator had already seen before Amendment 1: the Gate 3a/3b
calibration runs (`calib_warmup`, `calib_bins`, seeds 910000/920000) and the pilot (`pilot_abf`), plus the Gate 1/2
reference. They use the frozen configuration (or state where they do not) but different seeds from the screen.

## 0. Bottom line

1. **Every decision so far followed the frozen rule as written** (framework band, Gate 1 checks, Gate 2 admissibility,
   the dt second clause, the Gate 3a selection rule). I reproduced each one. Two are fragile or use a non-frozen
   reference (§1).
2. **The A1.4 classifier cannot return ELIGIBLE in this system as frozen.** Its establishment test (E) is
   structurally vacuous at N = 64 and needs a ≥ 74 % (N = 1024) or ≥ 91 % (N = 256) window deficit sustained for
   8 t.u. Its false-positive rate is 0 at every N. On every calibration run at the frozen config, (E) span = 0 t.u.
3. **The classifier has no category for the error mechanism that dominates ABF here.** At N = 1024 under the frozen
   config, e_F(T) is 1.5–3.6× the error of the same run's deposits after t = 2 t.u. That is an early-deposit
   (conditional/transient) bias. Test 2a was thresholded at 4× a τ already known (it cannot fire). Test 2b looks only at
   [T/2, T] and only at η, which is pinned in the window. The run would be labelled "ABF-SUFFICIENT". That label is
   wrong as a description, although "STOP FR" is still the right action.
4. **The frozen ABF baseline still leaves about 2× on the table at N = 1024.** The Gate 3a rule (written with the
   N = 64 half visible) picked the 1 t.u. ramp. At N = 1024 alone, the 4 t.u. ramp gives a 31 % lower Ī_F. Both
   effects bear on any later FR claim at large N.
5. **Before any FR run, the plan must freeze the FR knobs, the rate-selection rule, the endpoint, the test and the
   claim type.** It must also add a blind safety-only analysis path. Today these are absent or deferred.

## 1. Did each decision follow the frozen rule? (recomputed)

| decision | rule | recomputed | verdict |
|---|---|---|---|
| framework (§2) | \|βΔF\*−6.13\| ≤ 0.35, groups 0–3 | G&T 5.96 (in), Schlenker 5.29 (out) → G&T | followed |
| Gate 1 checks 1–6 | as §3 | min fine count 200 040; D 0.000/0.000/0.013; closure −0.0013 ± 0.0016 | followed (see F18 on the debiasing) |
| Gate 2 admissible | D ≤ 0.05, up ≤ 0.075, \|Δη\| ≤ 0.02 | EM 2e-4: 0.022 [0.048], 0.037 [0.059], 0.0035 | followed; the z-score diagnostic was not implemented (F12) |
| dt second clause | D ≤ ⅓ median e_F(T), frozen config | **on-grid (frozen primary) reference: median 0.1273 → 0.0424.** Smooth reference: 0.1291 → 0.0430 (the amendment's 0.129 is the smooth one). D_MF 0.0367 passes | followed, fragile (F5) |
| Gate 3a selection | min mean over N of normalised median Ī_F, 5 % tie → shorter eq, then warm-up | (25000, 5000) 0.3717; (25000, 20000) 0.3760 (+1.2 %, tie → shorter warm-up); (10000, 20000) 0.3948 | reproduced; design concern (F6) |
| Gate 3b bins | no written rule; 180 from the prompt | N 1024 smooth: 180→360 e_F −4.6 %, Ī −0.86 % ✓. **N 64: 90 bins e_F 0.152 vs 180 0.243 (−38 %); 360 −13.5 %** | the "plateau" holds at N 1024 only (F13) |

Per-seed e_F(T) at N = 1024 (calib_bins, ng 180, on-grid reference) is 0.0583, 0.1273 and 0.1303. One seed in three
fails the clause on its own (0.019 < 0.037). The mean of the three fails (0.0351 < 0.0367). The upper bound 0.059
fails against every seed.

## 2. Power and null controls of the A1.4 rules

### (E) window count < 0.5 N Q\*_W − 2 sd for ≥ 8 t.u.

* The threshold is ≤ 0 whenever N < 16(1−Q)/Q: N < 86 at Q\* = 0.157 (the uniform share of W) and N < 48 at
  Q\* = 0.25. **At N = 64 the test is vacuous.** In the N = 64 calibration runs, 87–96 % of saves in [2, T] have a
  threshold ≤ 0.
* Per-save null FP: 7e-11 (N 256), 8e-24 (N 1024). A simulated correlated null (independent two-state walkers,
  τ = 0.3 / 1 / 3 t.u., 400 reps each) gave a longest span of 0.0 t.u. and FP = 0 in every case.
* Deficit needed for a 50 % per-save detection, and for a 50 % chance of an iid 80-save run:

  | N | Q\* = 0.157 | Q\* = 0.25 |
  |---|---|---|
  | 64 | never / never | 90 % / 99 % |
  | 256 | 79 % / 91 % | 71 % / 84 % |
  | 1024 | 64 % / 74 % | 61 % / 69 % |
  | 4096 | 57 % / 63 % | 55 % / 60 % |

* Calibration runs at the frozen config:

  | runs | span (t.u.) | deposit-share ratio | window share of the final squared MF error |
  |---|---|---|---|
  | N 1024 (5 runs) | 0.0 | 1.03–1.05 | 0.03–0.08 |
  | N 64 (5 runs) | 0.0 | 0.96–1.07 | 0.03–0.21 |

  The window population sits at 0.9–1.2× uniform. Q\*_W itself starts above uniform (0.25–0.44 at t = 2) because
  the early bias overshoots.

### (I) information

* Null (uniform deposits, variance-only error, from the MC Var(f|φ)): the window share of Σ Var is 0.238 and the
  deposit ratio is 1.0. **FP = 0.**
* Clause 2 (≥ 50 % of the error in W) needs the window deposits cut about 3.2× relative to the rest. At that point
  clause 1 (< 0.5) has already fired, so clause 2 is redundant unless the window error is bias-dominated.
* The dt bias is not concentrated in W: W carries 12.5 % of the squared Γ(EM) − Γ(MC) on 15.6 % of the bins. So (I)
  is not contaminated by the dt bias.

### 2a τ_int(η) ≥ 0.1 T

From the Gate 2 EM-2e-4 series, the cage windows' median τ_int(η) is 0.58 t.u. The largest window median is 0.83
and the largest single value 1.96. The threshold is 4 t.u., so the rule cannot fire. It was set knowing the
1.09 t.u. pilot.

### 2b d_η > 3× noise floor, median over saves in [T/2, T]

* Floor (uniform population, 18 occupied coarse bins): 0.259 / 0.130 / 0.065 at N 64 / 256 / 1024. Three times the
  floor is 0.78 / 0.39 / 0.195.
* Power for a lag δ in ⟨η⟩ in the 10 high-variance coarse bins (200 reps):

  | N | P(fire) |
  |---|---|
  | 1024 | 0 at δ = 0.2, 1.0 at δ = 0.3 |
  | 256 | 0.79 at δ = 0.5 |
  | 64 | 0 even at δ = 0.5 |

  The conditional sd of η in the cage is about 0.68.
* Calibration ratios are 0.93–1.02, i.e. the null.
* η is pinned in the window: ⟨η⟩ = 0.983 and Var(η | coarse bin) ≈ 0.000–0.001. The window's conditional structure is
  in u_x² (0.347 → 0.663), which is tracked but unused.

### Discovery

This is the population-first translocation. Calibration values are 1.6–2.4 t.u. (N 64) and 0.5–1.4 t.u. (N 1024),
against a threshold of 4 t.u. The test is not vacuous, but no seed is near the threshold.

## 3. The mechanism the classifier misses

e_F(T) against the late-deposit error at the frozen config (eq 25000, w 5000):

| run | e_F(T) | after t = 2 | after t = 4 |
|---|---|---|---|
| calib_bins N1024 s0 | 0.058 | 0.016 | — |
| calib_bins N1024 s1 | 0.127 | 0.060 | — |
| calib_bins N1024 s2 | 0.130 | 0.071 | — |
| calib_warmup N1024 s0 | 0.181 | 0.122 | 0.097 |
| calib_warmup N1024 s1 | 0.131 | 0.065 | 0.037 |

At N = 64 the ratio is about 1.0–1.1, i.e. variance-limited. With the 4 t.u. ramp, the ratio after t = 8 is still
about 2.

The cumulative deposit-weighted ⟨η⟩ bias (from `Seta`) is 0.06 without pre-equilibration and 0.005–0.019 with it. So
pre-equilibration removed the η part, and the residual transient is carried by something else: the within-bin φ tilt,
orientation, or transverse position. An η-only test cannot see it.

## 4. Findings

See the JSON returned to the orchestrator; it is mirrored here in short form.

| id | sev | summary |
|---|---|---|
| F1 | high | (E) has zero power at N 64 and very low power at N 256 / 1024 (FP = 0): the screen cannot return ELIGIBLE |
| F2 | high | no classifier category for the dominant early-deposit/conditional transient; 2a cannot fire; 2b is late-only. The "ABF-SUFFICIENT" label is misleading |
| F3 | medium | 2b uses η only (pinned in W) and per-snapshot medians against 3× floor: low power; u_x² unused |
| F4 | medium | (E) is tested against the bias-aware Q\*, not against FR's uniform target: deficits caused by an under-built bias are invisible |
| F5 | medium | dt second clause: median of 3 seeds, smooth rather than frozen reference, fails under the mean or the upper bound; no rule for re-checking on 8 seeds or at the ladder cells |
| F6 | medium | Gate 3a rule written with N 64 visible; frozen warm-up is 31 % worse in Ī_F at N 1024 than the 4 t.u. alternative; eq at the grid edge → weaker large-N baseline |
| F7 | medium | the frozen ABF keeps a 1.5–3.6× early-deposit excess at N 1024; the FR start (default 2 t.u., not frozen) decides whether FR can touch it; no late-deposit endpoint |
| F8 | medium | provenance: the amendment header says 16:50 but the file is 16:41:55 / commit 16:42:07; the log has no entries after 16:25 although the amendment cites "16:00–16:48"; classify_screen.py written after N 64 screen outputs existed, uncommitted |
| F9 | medium | the FR confirmation design is not frozen: FR knobs, rate selection among safe rates, safety evaluated at T only, N-dependent lineage cap, endpoint/test/multiplicity, seed blocks, blind safety analysis |
| F10 | medium | the four claim types are not separated: fixed-N = equal force evals here, the equal-budget allocation ladder is absent, wall-clock is on a shared host, long-time correctness is unplanned |
| F11 | medium | the classifier is not validated: no positive or negative control, no unit tests of establishment / conditional_lag |
| F12 | low | Gate 2 conditional-observable z-scores not implemented (mean abs diffs instead) |
| F13 | low | the Gate 3b "plateau" is quoted from N 1024 only; N 64 is non-monotone (90 bins −38 %) |
| F14 | low | the 2b floor "from the reference conditional variances" is not computable (no Meta2 stored); the script substitutes the EM-2e-4 first-molecule series |
| F15 | low | cage core C = \|φ\| ≥ 2.0 is not "within 0.4 kBT" (boundary bins 0.44–0.49); the property holds for \|φ\| ≥ 2.08 |
| F16 | low | classification uses medians of 8 seeds with no stability rule; no rule for which eligible K go to FR |
| F17 | low | the conditional threshold is absolute (0.1 T), not relative to the establishment time (the project's own mechanism result) |
| F18 | low | the Gate 1 checks 2/4 debias by se_A² + se_B² for correlated estimates (D biased to 0); outcome robust (raw RMS 0.0057 / 0.0199) |
| F19 | info | the secondary dt-consistent reference is not implemented in metrics.py; the umbrella-EM dt bias is not exactly the ABF-biased one |
| F20 | info | gate2.json and gate2_EM_2e-04.json differ (upper 0.0477 vs 0.0485): the analysis is not bitwise reproducible (WHAM warm start) |

## 5. Amendments needed BEFORE the classifier is applied to the screen

Freeze these as Amendment 2, with the sha256 of `classify_screen.py`, before any screen file is read. Report the
original A1.4 label **and** the amended label side by side.

1. **Replace (E) with a powered statistic.**
   * Statistic: the time-integrated window deficit D_W = 1 − Σ_t n_W(t) / Σ_t N Q_W(t) over [2 t.u., T].
   * Compute it against both Q\* (bias-aware) and the uniform share (FR's target).
   * Noise: a block bootstrap over saves (block ≥ 5 τ_int of n_W) within each seed, plus the across-seed spread.
   * Fire if the median D_W ≥ 0.30 and the lower 95 % bound > 0.
   * Before use, show by simulation (the s6 two-state null and a 30 % injected deficit) FP ≤ 5 % and power ≥ 80 % at
     each K. A K with no power is reported as "UNTESTABLE AT THIS N", never as ABF-sufficient.
2. **Add an estimator-level transient/conditional-memory test** (2c).
   * Fire if, in the median seed, e_F(T, all deposits) / e_F(T, deposits after t0 = 4 t.u.) ≥ 1.5, and in at least
     6 of 8 seeds.
   * Also report a reference-free version: RMS(F̂_all − F̂_late) against its across-seed noise.
   * Label: "TRANSIENT/CONDITIONAL-MEMORY-LIMITED (not FR-eligible)".
3. **2b: use a time-averaged lag.** Average each coarse bin's population mean over the saves in [2, T] before
   comparing it with the reference, with noise from a block bootstrap. Include u_x², and the deposit-weighted
   cumulative bias (Seta/C, Sux2/C) against the MC conditional means. Define the floor from stored data (MC ts per
   window) and say so.
4. **Rename the fallback.** "ABF-SUFFICIENT" becomes "NOT ESTABLISHMENT-LIMITED", with the sub-diagnosis from 2c/2b,
   so that no report claims ABF is at its floor when it carries a removable transient.
5. **Classification stability:** for each label, report the fraction of 2000 seed-bootstrap resamples that return it.
   A label below 0.8 is INDETERMINATE → no FR.
6. **Positive and negative controls before use.** Run the classifier on synthetic inputs, a null and an injected
   deficit, as unit tests. If archived ABF-only data from a validated FR-positive cell exist (gateway N ≥ 16, LTA
   N ≥ 16), apply the statistic to them too. If a known positive is not classified ELIGIBLE, the classifier is
   miscalibrated.
7. **dt clause:** state now, before reading the screen, whether the clause is re-evaluated on the 8 screen seeds at
   N 1024 with the frozen primary (on-grid) reference. Because the clause is fragile, also make the EM-2e-4 umbrella
   reference co-primary for every N 1024 result (implement it in metrics.py).
8. **Log the missing steps:** Gate 3a result 16:31, Gate 3b 16:32–16:40, the clause computation, the screen launch
   16:41:55, the classifier script 16:43:49 and whether any screen file was opened. Correct the amendment's time
   (16:41:55, not 16:50).

## 6. Must exist before any FR run (including the safety calibration)

* **FR knobs frozen:** fr_start, fr_every 10, score_clip 2, kde_bandwidth 0.10, max_event_fraction 0.02, cap_min 1,
  the uniform target and the genealogy window. Justify fr_start against the early-deposit transient. At 2 t.u., FR
  cannot affect deposits before 2 t.u., which carry most of the N 1024 error.
* **Rate selection rule:** e.g. the largest rate in {0.02, 0.05, 0.1, 0.2} satisfying min over saves in
  [fr_start + window, T], not at T only, of ESS_w/N ≥ 0.30, in ≥ 7/8 seeds. Make the lineage cap N-aware: at N 64,
  0.05 means ≤ 3 walkers.
* **Blind safety analysis:** the safety runs' analysis must never load sv_M / M accumulators. Strip them from the
  outputs or use a dedicated script, so that no FR error exists before the confirmatory freeze.
* **Endpoints:** one primary (e.g. e_F(T) against the MC on-grid reference). Secondary: Ī_F, the late-deposit
  e_F(t0 = 4), and the EM-2e-4-consistent e_F.
* **Statistics:** a paired test on the per-seed log ratio FR/ABF (Wilcoxon signed-rank, or 95 % bootstrap CI), with
  a one-sided α. Add a TOST equivalence margin (e.g. ±10 %) for a neutral verdict, and Holm across K and the primary
  endpoint.
* **Seed blocks:** safety 940000+, exploratory 950000+, confirmatory 960000+, equal-budget 970000+. Never reuse the
  screen's ABF runs (930000+) as an FR comparator; that would be a selection bias.
* **Claim types, each with its own design:**
  * (i) fixed-N paired. Here this is also equal force evaluations at the same N: FR adds 0 force evals, and the
    N × 25 000 pre-equilibration counts in both.
  * (ii) equal total budget: the best ABF over (N, T) allocations, including the pre-equilibration cost, against the
    best FR. This is the claim the project's history says fails.
  * (iii) wall-clock: interleaved paired timing on pinned cores, because the host load is 8–40.
  * (iv) long-time correctness: a long run (e.g. T 200 t.u. at N 256) of both arms against the dt-consistent
    reference, with TOST. FR stationary tilts were found in WCA.
