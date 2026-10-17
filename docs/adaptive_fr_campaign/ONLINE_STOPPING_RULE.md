# Online FR stopping rule R1 (Experiment B)

*Code: `src/eri/fr_controller.py` (the single implementation, used by both engines).*
*Engine hooks: `src/eri/engine.py` (`Run._check` and `advance_controlled`) and `src/lta_ladder_numba.py` (`run_arm`:
`next_check`, `do_check`, `I_FRSTOP`).*
*Tests: `tests/test_fr_controller.py` (15), `tests/test_lta_fr_stop.py` (4), `tests/test_fr_stop.py` (7), plus both
engines' existing suites.*

## 1. What the rule decides, and from what

FR starts at the validated fixed start (the end of the bias ramp). R1 only decides **when to stop**. Once stopped,
FR never restarts and the run is exactly histogram ABF continuing from the current state. This is bitwise by test.

At every check (t = t_start + n × 0.25 t.u., n ≥ 0), the controller receives only the *current* cumulative state of
*this* run:
* birth and death histograms on the CV grid;
* deposit counts C_j and force sums M_j (the accumulators the bias reads);
* the windowed ancestor labels.

It imports only `math` and `numpy` and does no I/O. It never sees a reference free energy, a reference mean force,
another trajectory or the future. Tests check all of this: on the syntax tree, determinism, checkpoint/resume, and
chunking invariance.

## 2. Definition (frozen spec R1)

| symbol | meaning | value |
|---|---|---|
| check_every | check spacing | 0.25 t.u. (must be a multiple of dt; asserted) |
| W | trailing window | 1.0 t.u. (a multiple of check_every; asserted) |
| R | coarse regions of the CV circle | 18 equal arcs |
| θ | relative end-of-phase threshold | 0.25 |
| z_floor | noise floor in s.d. | 2 |
| persist | consecutive passing checks | 4 (= 1 t.u.) |
| stop_factor | stop time / t1 | 2 |
| support_count | every fine bin must have C_j ≥ | 200 (the ABF trust count) |
| stability_tol | relative change of Γ over W | 0.05 |
| essw_min | per-run windowed ancestor ESS/K backstop | 0.15 |
| maxlin | per-run windowed max-lineage backstop | max(0.10, 6/K) |

* **Window counts.** b_r and d_r are the FR births and deaths in region r during (t − W, t].
* **Directedness**, the systematically relocated fraction of the population per t.u.:
  j(t) = max(0, ½ Σ_r |b_r − d_r| − ½ Σ_r √(2 (b_r + d_r)/π)) / (K W).
  The subtracted term is the expectation of |b − d| under a symmetric null.
* **Noise s.d.:** s_j(t) = ½ √(Σ_r (b_r + d_r)(1 − 2/π)) / (K W).
* **Running maximum:** j_max(t) is the largest j so far.
* **support(t):** every fine bin has C_j ≥ 200.
* **stability(t):** ‖Γ_t − Γ_{t−W}‖ / ‖Γ_{t−W}‖ ≤ 0.05, over the bins already supported at t − W.
* **low(t):** j ≤ max(θ j_max, z_floor s_j), **and** support, **and** stability, **and** t ≥ t_start + W.
* **t1** is the first check of the first run of `persist` consecutive low checks. The decision is taken at the check
  that completes that run, and **FR stops at t_stop = max(2 t1, the decision time)**.
* **Safety backstop.** It is evaluated at every check until FR has actually stopped. If windowed ESS/K < 0.15 or the
  max lineage share exceeds max(0.10, 6/K), FR stops at that check. Safety can only bring a stop earlier.
* **Runs that never stop.** A run in which neither condition fires keeps FR to the horizon. It is reported as
  "never stopped", not hidden and not capped.

## 3. Rationale

* **Directed phase.** While ABF's bias is still too weak to keep the barrier (window) region populated, FR moves
  walkers *systematically* from the over-populated cages into the window: a large, directed net flux.
* **End of that phase.** Once the bias maintains the population, FR's events are mostly noise corrections, and the
  directed flux collapses to a small residual.
* **Why not use the residual.** The residual can stay statistically significant, for example while FR fights ABF's
  own window overshoot. It is exactly the regime in which continuous FR accumulated its late-time cost in the
  previous campaign. R1 therefore detects the **end of the directed phase relative to its own peak**, not the
  disappearance of all FR activity.
* **Why 2 × t1.** It is the online counterpart of the earlier offline rule, "stop at 2 × ABF's window-establishment
  time". That rule produced the confirmed early-stopping results, 10 t.u. at ITE 300 K and 15 at ITE 150 K.
* **Absolute time, on purpose.** t1 is measured, like that rule, in absolute time from t = 0, because ABF's
  establishment clock starts at 0. This is deliberate and documented (review finding 1).
* **Guards.** Support and stability stop the marginal from looking "established" while the mean-force estimator is
  still unsupported or moving. A uniform marginal alone never passes (tested).

## 4. Development (ITE only; LTA and ERI are held out)

* **Data.**
  * Historical ITE exploratory data: X1, X2 and the E1x ladder outcomes (from the previous campaign).
  * Nine new monitor runs of continuous FR on designated exploratory seeds 1200000–1200500, covering ITE 300/150 K,
    K256/1024, and FR rates 0.2–2. They recorded every online observable every 0.25 t.u.
    (`scripts/adaptive_fr/dev_monitor.py`).
* **Not used.** No LTA or ERI FR trajectory and no Experiment A outcome was used to choose any element of R1.
* **Replay of R1 on the development runs** (`dev_replay_rule.py`). The in-engine controller reproduces the replay
  exactly (`dev_engine_check.py`: t1 2.75, stop 5.5, zero events after the stop).

| run (T, K, rate, seed) | j_max | t1 | decision | t_stop | reason |
|---|---|---|---|---|---|
| T150_K1024_rate0.2_s1200500 | 0.036 | 5.25 | 6.0 | **10.5** | established |
| T150_K256_rate0.2_s1200400 | 0.021 | 5.25 | 6.0 | **10.5** | established |
| T150_K256_rate0.5_s1200100 | 0.067 | 5.0 | 5.75 | **10.0** | established |
| T150_K256_rate0.5_s1200101 | 0.051 | 5.0 | 5.75 | **10.0** | established |
| T300_K1024_rate0.2_s1200300 | 0.037 | 3.25 | 4.0 | **6.5** | established |
| T300_K1024_rate2_s1200000 | 0.219 | 2.75 | 3.5 | **5.5** | established |
| T300_K1024_rate2_s1200001 | 0.212 | 2.75 | 3.5 | **5.5** | established |
| T300_K256_rate1_s1200200 | 0.115 | 3.0 | 3.75 | **6.0** | established |
| T300_K256_rate1_s1200201 | 0.085 | 3.25 | 4.0 | **6.5** | established |

* **Reading.** R1 stops at about 5.5–6.5 t.u. at 300 K and about 10–10.5 t.u. at 150 K. It adapts to temperature
  and hardly depends on K (256 / 1024) or FR rate (0.2–2).
* **Comparison with the fixed schedules.** The confirmed fixed schedules stopped at 10 (300 K) and 15 (150 K). In the
  exploratory E1x ladder, stopping at 5 was about as good as 10 at 300 K. R1 sits at the early end of the good range.

**Revisions made during development, before any held-out data** (independent review, then fixed):
1. An absolute floor (2 % of K per t.u.) was replaced by the noise-scaled floor z_floor × s_j. An absolute floor does
   not scale with K, FR rate or time unit.
2. The per-run lineage backstop now scales with K.
3. Safety is now evaluated until FR has actually stopped, and can only bring the stop earlier.
4. Grid assertions were added.
5. The controller code is now part of both engines' provenance hashes.
6. `advance()` refuses to bypass the controller.

**Not changed, by decision.** The absolute-time stop factor (consistent with the validated rule), and no cap on FR
duration. A never-stopped run is reported as such.

## 5. Known limits, stated before evaluation

* **Weak directed phase.** If FR's directed phase is weak relative to its residual flux, `low` may not persist and FR
  runs long.
* **Flat profiles.** For near-flat mean-force profiles the relative-stability guard converges slowly.
* **Very small K (< 64).** Not covered by development.
* **In-sample cells.** C1 (ITE 300 K) and C2 (ITE 150 K) are in-sample at the cell level: the rule was developed on
  ITE, although with other seeds. LTA 150/300 K and ERI 600 K are **held out**.

## 6. How R1 behaved after the freeze (evaluation, not development; nothing retuned)

| cell | system / K | R1 stop, t.u.: median (range) | never stopped | S_geo [95 % CI] | verdict |
|---|---|---|---|---|---|
| C1 | ITE 300 K / K1024 | 5.5 (5.5–6.0) | 0 | 1.67 [1.45, 1.98] | SUPPORTED |
| EA_R1 | ITE 300 K / K256 | 6.5 (5.5–9.0) | 0 | 1.31 [1.07, 1.47] | SUPPORTED |
| EB64 | ITE 300 K / K64 (rate 0.2) | 7.5 (6.5–9.0) | 0 | 1.11 [0.92, 1.33] | UNRESOLVED |
| EB16 | ITE 300 K / K16 (rate 1.0) | 11.0 (9.0–14.0) | 0 | 0.93 [0.70, 1.29] | UNRESOLVED |
| C2 | ITE 150 K / K256 | 8.25 (7.0–11.5) | 0 | 1.64 [1.29, 1.90] | SUPPORTED |
| C3 | LTA 150 K / K1024 (held out) | 21.5 (20.0–24.5) | 0 | 2.52 [2.33, 2.60] | SUPPORTED |
| C4 | LTA 300 K / K1024 (held out) | 16.0 (14.0–18.5) | 0 | 1.46 [1.37, 1.60] | SUPPORTED |
| C5 | ERI 600 K / K1024 (held out) | 15.25 (5.5–25.5) | 0 | 0.61 [0.53, 0.71] | CONTRADICTED |

* **The stop time moves with the system.** It runs from 5.5 t.u. (ITE 300 K) to 21.5 t.u. (LTA 150 K), tracking the
  ABF establishment time it is meant to detect. Every run stopped; none were stopped by the safety backstop.
* **Below the development range of K, R1 still stops**, but later: 7.5 at K64 and 11 at K16. Those cells show no
  establishment gain from FR at all (paired difference 0), so the stop time there has nothing to protect.
* **The failure mode** is ERI (CROSS_SYSTEM_RESULTS.md §5). Its directed flux never decays below 30–60 % of its peak.
  R1 therefore stops late and spread out (5.5–25.5 t.u.), and FR's ongoing bias (MECHANISM.md §1) accumulates in the
  meantime.
