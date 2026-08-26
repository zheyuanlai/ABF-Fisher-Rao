# Clean-v2: intermittent physical-target Fisher--Rao as an ABF accelerator

## Material passport

- Artifact type: code-experiment protocol
- Status: frozen before scientific runs
- Date frozen: 2026-08-26
- Branch: `unflattened-target`
- Supersedes: `docs/V3_PREREGISTRATION.md`, `docs/V4A_PREREGISTRATION.md`,
  and the schedule/target grid of
  `docs/PHYSICAL_TARGET_PULSE_V2_PREREGISTRATION.md`
- Primary system: the existing two-dimensional `xi(x, y) = x` ABF benchmark,
  70/30 physical mass split (`x_tilt = 0.1021665783`)
- Code: `src/abffr/clean_v2.py` (algorithm), `src/abffr/accel.py` (endpoint),
  `tests/test_clean_v2_gates.py`, `tests/test_clean_v2_accel.py`

**Amendments, all made 2026-08-26 before any scientific run:**

1. `gamma` grid widened from `{0.02, 0.05, 0.10}` to `{0.002, 0.01, 0.05}`
   (section 7), and the post-failure lower-dose escape hatch deleted (section 9).
2. The censoring claim corrected. An earlier draft asserted that restriction
   "cannot manufacture a speedup". **That was backwards**: restriction replaces a
   censored `tau` by `T`, the smallest value it could have had, so censoring the
   *arm* inflates `S^(T)`. Section 6 now states the direction per side, hit
   fractions are mandatory, and `accel.confirms` refuses a threshold where the
   arm is censored more than the baseline.
3. Primary statistic renamed `S^(T)`, the *restricted* speedup at horizon `T`,
   rather than an estimate of the unrestricted ratio.
4. The oracle arm demoted from an implied upper bound to what it actually
   licenses: target-estimation error is a plausible limiting factor (section 8).
5. Section 1 rewritten: v2 already stated a finite-time acceleration claim, so
   clean-v2 returns to and sharpens that question rather than replacing it.
6. Gate A repeated at production scale on GPU, and the `abf.ema_alpha` removal
   shown to be a target change and not an estimator change (section 5).
7. A turnover-matched sham was considered and deliberately excluded. The
   exclusion stands; its stated justification was **corrected** -- the sham is
   perfectly runnable from the algorithm's own estimated scores, and the real
   reason is scope: this campaign tests efficacy, not a causal decomposition
   (section 9).
8. **Logic fix:** `pilot_promising` was blind to censoring while `confirms` was
   not, so an inflated Stage-2 cell could have won selection and been sent to
   fresh seeds. It now takes `Speedup` objects and applies the same censoring
   refusal (section 8).
9. The Stage-2 dose table was labelled "expected events per replica" but held the
   **integrated hazard** `gamma T_FR |S|`. Each replica fires at most once per
   opportunity, so with 30 opportunities the stated "120 expected events" was not
   merely approximate but impossible. Section 7 now gives
   `E[N] = N_opp (1 - exp(-gamma |S| L_FR dt))` per cell, and notes that the
   matched-dose approximation holds in R12 and breaks in the walls.
10. The boundary-extension rule was **deleted** rather than formalised: it was
    circular, and no code implemented it. A boundary winner is now reported as
    boundary-limited and taken to fresh seeds unchanged (section 9).
11. `--ignore-screen` no longer writes Stage 3/4 configs. It previously claimed
    not to authorise a confirmation run while creating the files that start one.

---

## 1. What changed, and why

It would be inaccurate to say v2 asked the wrong question. The v2
preregistration states in its own words that "the claim under test is finite-time
acceleration", and it already used a delayed-on/temporary-off schedule. What went
wrong was narrower and more specific:

- v2's **endpoint hierarchy** still leaned on integrated and final `L2` error, so
  selection and the verdict were driven by the wrong statistic even though the
  stated question was about time;
- v2's **operator** was not birth--death: a score clip and an event-fraction cap
  distorted it, and neither was reported as changing the flow;
- v3 then moved the question somewhere else entirely -- to whether ABF and FR can
  share a *stationary* distribution -- and answered no, attributing the failure to
  a target--bias conflict (ABF wants a flat reaction-coordinate marginal, a
  physical target wants a Boltzmann one, and a permanently coupled sampler cannot
  serve both).

The v3 objection is sound, but it rests on a premise this campaign drops: it is
only a conflict if ABF and FR must coexist forever and share a stationary
distribution. Clean-v2 therefore does not introduce a new question. It **returns
to v2's original finite-time acceleration question and sharpens it** -- faithful
unclipped birth--death, the unflattened physical target, and time-to-accuracy
promoted from an aspiration to the primary endpoint:

> Can intermittent Fisher--Rao act as a finite-time **population accelerator**,
> so that ABF reaches a prescribed free-energy accuracy in **less physical
> simulation time**?

Under that question FR does not need to share ABF's long-run marginal. FR is
switched off before the end of every run, so the arm and its baseline are
literally the same plain-ABF algorithm over the final segment and the
`t -> infinity` limit is ABF's by construction. FR is a non-equilibrium
transient population intervention, not a replacement sampler.

Two consequences follow immediately, and they are the whole reason this document
exists rather than an amendment:

1. **The target becomes the unflattened physical marginal.** There is no reason
   to interpolate, cap or temper a target that is only applied in pulses.
2. **The endpoint becomes time-to-accuracy, not final error.** Plain ABF is a
   convergent estimator; given enough time it is *supposed* to be excellent, and
   an accelerator's advantage is *supposed* to shrink. Selecting on final error
   was measuring the wrong thing.

---

## 2. Frozen hypothesis

Let `A'_t(z) = Fhat'_t(z)` be the ordinary ABF mean-force estimator and
`A_t(z) = integral A'_t` the current free-energy estimate. The physical
reaction-coordinate target is

```
q_t(z) = exp(-beta A_t(z)) / integral exp(-beta A_t(u)) du.
```

After a pure-ABF learning phase, standard particle birth--death toward `q_t` is
applied periodically. With

```
e_F(t)   = min_c || A_t - F_ref - c ||_{L2},      tau_eps = inf { t : e_F(t) <= eps },
S_eps    = E[tau_eps^ABF] / E[tau_eps^{ABF+FR}],
```

the hypothesis is `S_eps > 1`. A value `S_eps = 1.5` reads as: *the same
free-energy accuracy for two thirds of the simulation time.* That ratio is the
headline quantity of the project.

The hypothesis is **not** `e_F(T)^{FR} < e_F(T)^{ABF}`. The desired picture is
FR ahead early, first to the threshold in the middle, and the two curves
converging late. That is what acceleration looks like.

---

## 3. Frozen algorithm

For `K` replicas, at every integration step:

1. **Propagate.** Each replica takes one biased overdamped Langevin step under
   the current ABF bias.
2. **Update ABF** from the *propagated* configurations
   (`abf.observation_order: post_propagation`), then integrate to `A_{n+1}`.
   The resampling operation never contributes an observation: a clone first
   speaks after its next physical propagation.
3. **Phase I, `n < n_burn`:** nothing else happens. `gamma_FR = 0`.
   `exp(-beta A_n)` is not a credible physical target at `t = 0`, and
   exponentiating an unconverged estimator amplifies its error.
4. **Phase II, `n_burn <= n < n_off` and `(n - n_burn) mod L_FR == 0`:** one FR
   pulse (below).
5. **Phase III, `n >= n_off`:** pure ABF again.

With `t_burn = 0.2 T` and `t_off = 0.8 T`:
`[0, 0.2T)` ABF, `[0.2T, 0.8T)` ABF + sparse FR, `[0.8T, T]` ABF.

### The pulse

Marginal, from the existing reflected/binned KDE at bandwidth `eta = 0.10`:

```
phat_n(z) = (1/K) sum_j K_eta(z - xi(q_n^j)).
```

Score, computed **without ever exponentiating or normalising a target**. Since
`log q_n(z) = -beta A_n(z) + C_n` and the score is mean-centered, `C_n` cancels
identically:

```
r_i = log phat_n(z_i) + beta A_n(z_i),        S_i = r_i - (1/K) sum_j r_j.
```

This is exactly `log(phat/q_n) - E_phat[log(phat/q_n)]` with no normalising
integral, no free-energy gauge convention and no `exp(-100)` underflow. It is
gauge invariant by construction (Gate C). `phat` is only ever evaluated at
particle positions, where a KDE is bounded below by its own self-contribution
`1/(K eta sqrt(2 pi))`, so the score can never be produced by a floored density.

Operator: `fr_v3.bd_standard`, unchanged and reused rather than rewritten.
`S_i > 0` means over-represented relative to the physical target, so `i` dies and
a **uniformly chosen** other replica duplicates; `S_i < 0` means
under-represented, so `i` duplicates and a uniformly chosen replica dies. Event
probability

```
P_i = 1 - exp(-gamma |S_i| dtau_FR),          dtau_FR = L_FR * dt.
```

No cap on the tail, no ceiling on the event count, population exactly fixed. No
`bd_paired`, no finite-time map, no systematic resampling, no persistent weights.

The interval-scaled clock matters: over a fixed window the number of
opportunities is `T_FR / (L_FR dt)` and each carries reaction time `L_FR dt`, so
the **integrated FR dose is `gamma * T_FR` regardless of `L_FR`**
(`test_dtau_is_interval_scaled_so_total_dose_depends_only_on_gamma`). Varying
`L_FR` therefore studies *frequent-and-weak* against *sparse-and-strong* rather
than handing one arm more total FR time.

### The only two scientific knobs

```
L_FR  (fr_every)      and      gamma.
```

Burn-in (`0.2T`), switch-off (`0.8T`), `eta`, the target and the operator are
frozen and are not degrees of freedom.

---

## 4. What is retired

Removed from the scientific method, and **rejected** by
`abffr.clean_v2.validate_config` rather than defaulted -- a config carrying the
key at all is refused, because `score_clip: null` still tells a reader the
operator has a clip:

| Removed | Why |
| --- | --- |
| `fr.score_clip` | truncated 55.4% of particle scores in v2 and compressed a 27.19-nat raw span to 8.11 nats. The clipped operator is not birth--death. |
| `fr.max_event_fraction` | the 10% cap demonstrably bound and silently changed the FR dose. If FR is too strong, reduce `gamma`; do not truncate the operator. |
| `fr.target_ema_alpha`, `abf.ema_alpha` | ABF's accumulator is already a cumulative estimator. A second smoothed free-energy memory is a second thing to tune. |
| `fr.ramp_fraction`, `fr.jitter` | must be exactly 0. The operator is applied as written. |
| `v3:` block | capped/tempered/consistent target family, FT step, ESS governor, clone policies, discrepancy trigger. Closed: see `docs/V3_POST_MORTEM.md`. |
| `v4:` block | persistent mass sidecar. Closed. |
| `selection.write_generic_best` | must be `false`. The generic selector ranks on integrated and final L2 error, which is the endpoint this campaign moved away from; it must not be able to write a file from a clean-v2 stage. |

Also forbidden by the protocol (not by a validator, because they were never
implemented on this path): target interpolation, flat/physical mixtures,
adaptive target mixing, ESS-governed FR, adaptive `gamma`, and any retuning
after fresh confirmatory seeds are seen.

The v3/v4 code and configs are **kept, not deleted**: they are the record of a
closed negative result. They are unreachable from the clean-v2 path -- the
config gate refuses a `v3:`/`v4:` block, and the engine asserts exclusivity.

---

## 5. Engineering gates

No scientific run may start until `tests/test_clean_v2_gates.py` and
`tests/test_clean_v2_accel.py` pass. Each gate is paired with a positive control
asserting that the quantity it inspects actually moves, because a gate that
cannot fire reads as reassurance.

| Gate | Statement | Test |
| --- | --- | --- |
| A | `gamma = 0` reproduces plain ABF in `q_t`, `Fhat'_t`, `A_t` under matched physical noise -- exactly (`atol=0`) on CPU at gate scale | `test_gate_A_zero_gamma_is_plain_abf` |
| B | one FR event with no propagation changes no ABF accumulator: at the pulse step `F'` and `F` match the baseline exactly while the marginal already differs | `test_gate_B_resampling_contributes_no_abf_observation` |
| C | `A_t -> A_t + C` leaves every `S_i` identical | `test_gate_C_score_is_invariant_to_the_free_energy_gauge` |
| D | nothing clips: `S_applied == S_raw` at every recorded quantile, `score_clipped_fraction == 0`, and the `log phat` floor never binds | `test_gate_D_*` |
| E | the applied probability is `1 - exp(-gamma \|S\| dtau_FR)` with no later cap, and the operator fires at that rate | `test_gate_E_*` |
| F | FR fires exactly on `[n_burn, n_off)` at stride `L_FR`, first pulse on `n_burn`, none at or after `n_off` | `test_gate_F_*`, `test_schedule_specification_is_half_open` |
| G | both arms share initial conditions and Langevin noise; FR draws from a separate stream, and adding an FR row to a batch does not perturb a baseline row | `test_gate_G_*` |

Gate F is checked against `abffr.clean_v2.firing_steps`, which is the schedule
*specification*; the engine reproduces it inline and the gate compares the two,
so a drift in either shows up as a failure rather than as a plot.

**Scope of what the gates verify.** They run on CPU at engineering scale
(hundreds of steps, tens of replicas), which is what makes `atol = 0` a
meaningful assertion. The GPU engine is **not** bitwise reproducible at
production scale -- reduction order varies, and v3 Amendment 1 measured `3.9e-7`
in `l2_F` over 50k steps while discrete counters stayed exact.

**Gate A was therefore also run at production scale on GPU** (50,000 steps,
`K = 256`, `float64`, one H200, seeds 1000-1001), comparing plain ABF against a
`gamma = 0` physical arm:

| quantity | seed 1000 | seed 1001 |
| --- | --- | --- |
| max abs delta `l2_F` (R12) | `1.64e-07` | `2.30e-07` |
| max abs delta `l2_F'` (R12) | `4.01e-07` | `6.18e-07` |
| max abs delta `F'` on the profile grid | `4.08e-06` | `5.38e-06` |
| FR events / replacements | 0 / 0 | 0 / 0 |

The discrete counters are exactly zero and the continuous residuals sit at the
engine's own non-determinism floor. **That floor is measured here, not assumed**:
the `gamma = 0` comparison is identical by construction (the FR branch is never
entered), so its residual *is* the scale against which any other "identical" claim
in this protocol is judged.

**The plain-ABF baseline is the same ABF the earlier campaigns ran.** Clean-v2
forbids `abf.ema_alpha`, so it must be shown that this removed a *target*, not an
*estimator*. Running `abf_only` at the same seed under a clean-v2 config and
under a legacy config still carrying `abf.ema_alpha = 0.05`, `fr.score_clip = 5`
and `fr.max_event_fraction = 0.10` gives, at production scale on GPU, max abs delta `3.2e-07` in
`l2_F` (R12) and `9.5e-06` in `F'` on the profile grid --
against a measured floor of `5.4e-06`, so within 2x of it -- and
**bit-for-bit equality on CPU**
(`test_removing_ema_alpha_did_not_change_the_abf_estimator`). Those knobs only
ever fed `Fhat_target`, which only ever fed the FR target, and never the
accumulators, `F'`, `F` or the applied bias.

Both checks are **reproducible from committed configs and stored as data**, not
only as prose: `configs/clean_v2/identity_clean.yaml` and `identity_legacy.yaml`
produce the runs, `scripts/verify_clean_v2_identity.py` emits
`results/clean_v2/identity_checks.json`, and the script exits non-zero if the FR
counters are not exactly zero or the `ema` delta exceeds twice the measured
floor. Re-running gives numbers of the same order but not the same digits, which
is the point: the floor is a property of the engine, so it is measured on every
run rather than hard-coded.

---

## 6. Evaluation

### Scope

Primary: **R12**, the dimensionless thermal scope `beta (F_ref - min F_ref) <= 12`.
On this benchmark it spans `x` in `[-1.74, +1.69]`: both basins and the *whole
barrier*, excluding only the reflecting-wall strips where the reference free
energy exceeds 12 kT and no sampler has data. It was frozen for the v3 campaign,
before the question this campaign asks existed, and it cannot hide barrier
damage -- the one thing an unflattened target is most likely to cause.

Secondary, always reported: `legacy` (`x` in `[-2.49, 2.49]`, v2 comparability)
and `full`. These retain the wall strips, so any damage the primary scope
excludes is still visible.

### Thresholds

Frozen from Stage 1, which contains no FR run at all, by a mechanical rule:

```
eps_{F,1} = median over calibration seeds of e_F^ABF(0.4 T)
eps_{F,2} = median over calibration seeds of e_F^ABF(0.6 T)
```

and likewise for `F'`. Thresholds are frozen for the primary *and both secondary*
scopes at once, so a later scope switch would be visible as a choice among values
frozen at one moment rather than as a fresh calculation. The output file is
write-once; `--force` prints what it discards.

### Time to accuracy

```
tau_{F,eps} = min { t : e_F(t_j) <= eps for CONSECUTIVE_FRAMES = 3 saved frames },
```

recorded at the **first** frame of the qualifying run. A single downward
fluctuation is not convergence. A consequence stated in advance: a threshold
first reached in the last two frames cannot start a qualifying run and is
censored.

A seed that never reaches the threshold is **right-censored at `T`, not
dropped**. Dropping censored seeds would compare the subset of ABF seeds that
converged against the subset of FR seeds that converged. The statistic is the
**restricted speedup at horizon `T`**:

```
S^(T)_eps = E[min(tau^ABF, T)] / E[min(tau^{ABF+FR}, T)],
```

with a **paired** bootstrap over matched seeds (the arms share initial conditions
and noise; resampling seeds jointly is what the pairing bought).

`S^(T)` answers "within the budget `T`, who got there first?". It is **not** an
estimate of the unrestricted `E[tau^ABF] / E[tau^FR]`, and it is **not**
unconditionally conservative. Restriction replaces a censored `tau` by `T`, the
*smallest* value it could have had, so:

| censored side | effect on `S^(T)` |
| --- | --- |
| the **arm** only | denominator shrinks -> `S^(T)` **inflated** (flatters FR) |
| the **baseline** only | numerator shrinks -> `S^(T)` **deflated** |
| both | direction indeterminate |

The one safe case is the useful one: with **no arm censoring**, `S^(T)` is a
lower bound on the unrestricted speedup. So every reported ratio carries `n`, the
censored count, **and the hit fraction `P(tau <= T)` on each side**; and a
free-energy threshold at which the arm is censored more often than the baseline
**cannot carry the verdict** (`accel.confirms` refuses it). Such a run is not a
negative result -- it is an unresolved one, and the horizon is the thing to fix.

Stage 1 reports `P_ABF(tau_eps <= T)` for every frozen threshold at freeze time.
If plain ABF reaches the stringent threshold in under half its calibration seeds,
the horizon rather than the method is the binding constraint, and the result must
say so instead of calling the threshold hard.

### Safety diagnostics -- never a verdict, never a ranking

Final and integrated `e_F(T)`, `e_F'(T)`, the final/baseline ratios, ancestral
ESS, `w_max`, replacements per pulse, and `C_accel = (S_{F,2} - 1) /
replacement fraction`. `e_F^{FR}(t) ~ e_F^{ABF}(t)` late is not a failure --
that is exactly acceleration. Only a pathology is rejected: `e_F^{FR}(T)`
much larger than `e_F^{ABF}(T)` **with the curve still diverging after FR is
off**.

Genealogy is a mechanism and safety diagnostic, not a research objective.
Birth--death necessarily spends genealogy. `ESS_anc / K >= 0.5` is **not** a veto
on a method that converges measurably faster; it becomes one only at
`ESS_anc ~ 1` with accompanying accuracy degradation.

---

## 7. Stage plan and run budget

| Stage | Purpose | Seeds | Runs |
| --- | --- | --- | --- |
| 0 | engineering gates | 0-1 | small |
| 1 | ABF-only threshold calibration | 1000-1015 | 16 |
| 2 | 9-schedule pilot + baseline, 8 matched seeds | 2000-2007 | 80 |
| 3 | fresh-seed confirmation, 3 arms x 32 seeds | 3000-3031 | 96 |
| 4 | long-horizon sanity at 2T, 2 arms x 8 seeds | 4000-4007 | 16 |
| | **total before transfer** | | **~210** |

Seed blocks are disjoint by stage, so "fresh seeds" in Stage 3 is literally true.

Stage 2 grid: `fr_every` in {100, 500, 1000} x `gamma` in
**{0.002, 0.01, 0.05}**. Nothing else varies.

`gamma` is log-spaced over 25x deliberately, so the pilot is a genuine
dose-response scan -- from "barely touches the thermally relevant region" to
"heavy churn" -- rather than three neighbouring points that could all sit on the
same side of the optimum.

What the interval-scaled clock matches across `L_FR` is the **integrated hazard**
`gamma T_FR |S|`, not the expected number of events. Each replica fires at most
once per opportunity, so with `N_opp` opportunities the expected count is

```
E[N_events] = N_opp * (1 - exp(-gamma |S| L_FR dt)),      N_opp = 300 / 60 / 30
```

for `L_FR` = 100 / 500 / 1000. The two agree only while the per-pulse probability
is small. Expected events per replica over the active window:

| `gamma` | hazard `gamma T_FR \|S\|` | R12 (`\|S\| ~ 3`), L = 100 / 500 / 1000 | walls (`\|S\| ~ 40`), L = 100 / 500 / 1000 |
| --- | --- | --- | --- |
| 0.002 | 0.36 / 4.8 | 0.36 / 0.36 / 0.36 | 4.8 / 4.6 / 4.4 |
| 0.01 | 1.8 / 24 | 1.79 / 1.77 / 1.75 | 23.1 / 19.8 / 16.5 |
| 0.05 | 9.0 / 120 | 8.87 / 8.36 / 7.78 | 98.9 / 51.9 / 29.5 |

In R12 -- where the primary endpoint lives -- the matched-dose approximation is
excellent: at most 12% spread across `L_FR` at the strongest `gamma`, and 0.4% at
the middle one. It breaks in the wall strips, where per-pulse probabilities
saturate (`1 - exp(-4) = 0.98` at `gamma = 0.05`, `L = 1000`) and the sparse
schedules therefore cannot evacuate the walls as thoroughly as the frequent ones
at the same nominal dose: 98.9 against 29.5 events.

That is a property of the birth--death tau-leap, not a defect, and it is worth
stating because it means **`L_FR` is a clean frequent-weak/sparse-strong axis
inside R12 and additionally a realised-turnover axis outside it.** A wall-driven
difference between `L_FR` cells should be read that way rather than as a pure
scheduling effect.

This replaces an earlier `{0.02, 0.05, 0.10}`, which the engineering measurements
in section 9 showed would have been saturated (`P ~ 1` for wall replicas) in most
cells. **The change was made before any scientific run**, which is why it is a
grid choice and not a rescue.

Stage 3 arms: plain ABF (primary baseline), clean physical-target intermittent
BD (primary method), oracle physical-target intermittent BD (diagnostic,
`q propto exp(-beta F_ref)`, **never** a candidate method and **not** an upper
bound -- it answers only whether target-estimation error is a plausible limiting
factor).

**The Stage-3 and Stage-4 configs do not exist in this repository.** They are
generated by `scripts/select_clean_v2_schedule.py` from the Stage-2 pilot, so a
confirmation config cannot be authored before the pilot has spoken. "No retuning
after seeing confirmatory seeds" is enforced by the order in which files come
into existence.

---

## 8. Decision rules, declared in advance

### Stage-2 screen (`accel.pilot_promising`)

A schedule is promising if `S^(T)_{F,1} >= 1.15` **and** `S^(T)_{F,2} >= 1.15`,
at least one of `S^(T)_{F',1}, S^(T)_{F',2}` is `>= 1.10`, neither `F'` speedup
falls below `0.95`, **and neither free-energy threshold has more arm censoring
than baseline censoring**. Final error and AUC appear nowhere in this screen.

The censoring condition is the same one the Stage-3 verdict applies, and it is
here for a specific reason: selection filters on this predicate, so a screen
blind to censoring would let a cell whose `S^(T)` is inflated by the arm failing
to converge win the pilot and consume 96 fresh runs before the verdict caught it.
`pilot_promising` therefore takes `Speedup` objects, exactly as `confirms` does,
and the two predicates cannot drift apart about what censoring means.

### Selection (`accel.rank_key`)

Rank by `S_{F,2}`; among schedules within 5% of the leader, prefer fewer
replacements, then larger `fr_every`, then smaller `gamma` -- the sparsest
intervention that buys essentially the same acceleration.

### Stage-3 verdict (`accel.confirms`)

Both `S^(T)_{F,1}` and `S^(T)_{F,2}` at least `1.15` **with the paired-bootstrap
95% CI strictly above 1**; both `F'` speedups above 1 with no clear slowdown;
and **no free-energy threshold at which the arm is censored more often than the
baseline**, because restriction inflates `S^(T)` exactly there. A run that fails
only the last condition is not a negative result -- it is unresolved, and the
horizon is what to change.

### Oracle decision tree

- **Case A** -- estimated positive, oracle positive: the pipeline works.
- **Case B** -- estimated null, oracle positive: **target-estimation error is a
  plausible limiting factor.** That is the whole inference the oracle licenses.
  It does not by itself establish that the burn-in was too short, and the oracle
  is not an upper bound on what the deployable method can reach. Run **one**
  predeclared diagnostic at `t_burn = 0.4T` with the identical target, operator,
  interval and `gamma`; only if that closes the oracle gap is "`0.2T` burn-in was
  too short" supported. Do not invent a new target. Concretely: copy the
  generated
  `configs/clean_v2/stage3_confirmation.yaml`, set

  ```yaml
  fr:
    burnin_fractions: [0.40]
    duration_fractions: [0.40]   # NOT 0.60 -- stop = burnin + duration, and
                                 # t_off must stay at 0.8T
  ```

  and nothing else. `stop_fraction >= 1.0` is refused at load, because a window
  that silently ran to the end of the run is what invalidated every v3 FR arm.
- **Case C** -- neither accelerates: physical-target intermittent FR does not
  accelerate this benchmark at this dose and schedule. No interpolation rescue.

---

## 9. Pre-declared expectations and known risks

These are recorded before the scientific runs so that neither outcome can be
narrated as expected after the fact.

**The score span is large, and that is the point.** The physical target is
`exp(-beta A)`, and on this landscape `beta` times the range of `F_ref` over the
*sampled* domain is **80.7 nats** (over R12 alone it is only 11.7). Because ABF
flattens, replicas occupy the whole domain including the high-`F` wall strips, so
the score genuinely spans tens of nats. Engineering-scale measurement
(`n_steps = 4000`, `K = 64`) puts the median per-pulse span at 30--61 nats and
replacements at 3--30% of the population per pulse. Two independent confirmations
of the v2 post-mortem fall out of this: `score_clip = 5` was discarding a
30--60-nat signal, and `max_event_fraction = 0.10` would have bound in half of
those cells.

**Predicted mechanism, and the risk it carries.** Roughly 38% of the plain-ABF
marginal sits outside R12. The pulse will preferentially kill those replicas, so
the campaign should expect `e_F` on the `full`/`legacy` scopes to *degrade*
during Phase II while R12 may improve, with ABF re-populating the wall strips
during Phase III. If the primary endpoint improves while the secondary scopes
degrade and do not recover after `t_off`, the honest reading is a scope-limited
accelerator, and it must be reported as such rather than as an unqualified
speedup.

**Dose range, and why there is no post-failure escape hatch.** An earlier draft
of this protocol used `gamma in {0.02, 0.05, 0.10}` and pre-registered a
lower-dose diagnostic to be run *if* every cell turned out saturated. That is a
bad design even when it is declared in advance: the measurements in this section
already told us the grid would very likely be saturated, so the diagnostic was
not a contingency but a deferred correction, and it would read to any reader as
"the first grid failed, so another was searched". The grid was therefore widened
to `{0.002, 0.01, 0.05}` **before any scientific run**, and the escape hatch is
gone.

**Nothing replaces it: the grid does not get extended.** An earlier draft
allowed one factor-of-5 extension if the winner landed on a `gamma` boundary.
That rule was circular -- selection is what identifies the boundary case, and the
selector writes the Stage-3 config in the same step -- and no code implemented
it, which is precisely the "prose describing behaviour the code cannot perform"
defect this protocol exists to avoid.

The frozen rule is simpler: **if the selected schedule sits at `gamma = 0.002` or
`gamma = 0.05`, report the dose optimum as boundary-limited, freeze that winner,
and take it to fresh seeds unchanged.** `select_clean_v2_schedule.py` prints this
and records `gamma_at_grid_edge` in `selected_schedule.json`. This campaign tests
whether the method accelerates, not where the optimal `gamma` is; a boundary
winner is a reported limitation, not a reason to search further.

**No turnover-matched sham -- a scope decision, not a feasibility one.** A
score-permuted sham (`S~_i = S_{pi(i)}`: same score distribution, same
positive/negative counts, same expected turnover, directions scrambled) would
separate "directed Fisher--Rao reallocation" from "generic clone/delete churn".
It is deliberately **not** in this campaign.

The reason is scope, and it must be stated that way. **This campaign tests
efficacy, not a causal decomposition of the acceleration.** The claim under test
is "ABF + intermittent physical-target FR reaches a prescribed accuracy faster
than ABF", and the essential comparison for that claim is plain ABF, which is
already the baseline. A sham is also not a competing method: nobody would deploy
a permuted-score resampler, so it does not belong to the set of things a user
chooses between.

An earlier draft argued that a sham *could not be constructed* because it needs
the true per-particle score assignment. **That was wrong** and is withdrawn: the
algorithm computes `S_i` itself from `log phat` and `beta A_t`, so those
estimated scores could plainly be permuted. The sham is runnable; it is simply
out of scope here.

The cost must be stated in any paper rather than left implicit: a positive result
attributes acceleration to **intermittent physical-target FR as a package**, and
does not on its own separate directed reallocation from generic population
turnover. Two things partly stand in, and neither is a substitute. The
`gamma` x `L_FR` grid is a dose-response scan, so acceleration varying
structurally with FR dose is harder to explain by undirected churn than a single
positive cell would be; and the oracle arm varies the target's *direction* while
holding the turnover mechanism fixed. If a full attribution is later wanted, the
score-permuted sham is the right instrument and would be its own preregistered
study.

**Censoring pressure.** If the frozen thresholds are hit by only a minority of
seeds within `T`, `S_eps` is dominated by restriction at `T` and loses
resolution. Stage 1 must report the censoring rate of plain ABF against its own
frozen thresholds; a rate above 50% at `0.6T` means the horizon, not the method,
is the binding constraint, and that must be said in the result.

---

## 10. Figures

1. `e_F(t)` against physical time, plain ABF vs the FR arm, median + IQR, log
   `y`, with `t_burn` and `t_off` marked. Schedules are **never averaged into one
   curve**.
2. `S^(T)_eps` at both frozen thresholds with paired-bootstrap intervals. The
   most important quantitative figure. Rows carry the arm name, `(gamma, L_FR)`
   and `n`.
3. `e_{F'}(t)`, the quantity ABF actually learns.
4. Mechanism: `phat_t`, `q_t` and `p_ref^phys` on one axis, plus the genealogy
   appendix (`ESS_anc(t)`).
5. Threshold-reaching survival `P(tau_eps > t)` per arm, one panel per frozen
   threshold, annotated with the hit fraction. This is the figure that shows a
   speedup is a curve stepping down *earlier* rather than a curve that ends
   lower because its seeds ran out of budget -- the visual form of the censoring
   argument in section 6.

---

## 11. Reproduction

```bash
# Stage 0 -- gates
python -m pytest tests/test_clean_v2_gates.py tests/test_clean_v2_accel.py -q
python scripts/run_reference_2d.py --config configs/clean_v2/stage0_gates.yaml
python scripts/run_clean_v2.py --config configs/clean_v2/stage0_gates.yaml \
    --stage gates --device cpu

# Stage 1 -- calibration, then freeze the thresholds (write-once)
python scripts/run_reference_2d.py --config configs/clean_v2/stage1_calibration.yaml
python scripts/run_clean_v2.py --config configs/clean_v2/stage1_calibration.yaml \
    --stage calibration
python scripts/freeze_clean_v2_thresholds.py \
    --stage-root results/clean_v2/stage1_calibration/calibration \
    --out results/clean_v2/thresholds.json

# Stage 2 -- pilot, analysis, selection (generates the Stage 3/4 configs)
python scripts/run_reference_2d.py --config configs/clean_v2/stage2_pilot.yaml
python scripts/run_clean_v2.py --config configs/clean_v2/stage2_pilot.yaml --stage pilot
python scripts/analyze_clean_v2.py \
    --stage-root results/clean_v2/stage2_pilot/pilot \
    --thresholds results/clean_v2/thresholds.json --screen pilot
python scripts/select_clean_v2_schedule.py \
    --acceleration results/clean_v2/stage2_pilot/pilot/acceleration.csv

# Stage 3 -- fresh-seed confirmation
python scripts/run_reference_2d.py --config configs/clean_v2/stage3_confirmation.yaml
python scripts/run_clean_v2.py --config configs/clean_v2/stage3_confirmation.yaml \
    --stage confirmation
python scripts/analyze_clean_v2.py \
    --stage-root results/clean_v2/stage3_confirmation/confirmation \
    --thresholds results/clean_v2/thresholds.json --screen confirm
python scripts/plot_clean_v2.py \
    --stage-root results/clean_v2/stage3_confirmation/confirmation \
    --thresholds results/clean_v2/thresholds.json
```

---

## 12. Transfer (only after a positive 2-D confirmation)

Order: entropic-bottleneck hard regime, then the WCA dimer, then one molecular
torsion benchmark (pentane before any peptide). The schedule is **transferred,
not retuned**, and it travels in dimensionless form -- burn-in fraction, active
window fraction, number of FR opportunities, integrated FR dose -- because 500
MD steps do not mean the same thing on two simulators.
`scripts/select_clean_v2_schedule.py` writes exactly those quantities into
`selected_schedule.json` for that purpose.
