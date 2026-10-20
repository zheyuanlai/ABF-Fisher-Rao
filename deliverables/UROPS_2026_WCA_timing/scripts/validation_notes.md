## Manual audit notes (complement the automated table)

**Before production (all before the 16:31 UTC launch).**

* Baseline: the original 43 WCA engine tests pass on the unmodified engine (main checkout, 13e77275).
* Backward compatibility:
  * the same 43 tests pass on the extended engine;
  * 12 new tests pass (`tests/test_wca_numba_stop.py`). They include bitwise equality with the original module for
    ABF + FR + sham jobs with stopping disabled.
* Logs: `data/summary/tests/*.txt`, written 16:19 UTC. `*.log` is git-ignored, so the freeze commit e25edb3e
  silently skipped them. They were added unchanged, as `.txt`, in the results commit.
* Regression: the extended engine reproduces the stored Phase-V seed-5100 trajectories bitwise for every saved array
  at t = 0, 5, …, 45:
  * ABF;
  * FR from t = 40, including its 16 FR replacements over 40–45.
  Evidence: `data/summary/regression/*.json`. That run was also the cost benchmark: 575 s per 360 000 steps.
* Cost: about 0.85 core-h per run, so 7 × 16 runs came to about 99 core-h, under the 200 core-h cap. The secondary
  arms were therefore included **before** any production run (`frozen_config.json → arm_inclusion_decision`).
* Time-to-accuracy tolerances were taken from the validated Phase-V ABF-only arm alone.

**Production.**

* 112 / 112 runs complete; none rejected, none rerun, no non-finite state.
* One commit, one engine and one frozen config for every run.
* About 110 core-h in total (0.98–0.99 h per run, 112 workers concurrently).

**Post-data changes (presentation and formatting only; no statistic, window, rule or data changed).**

1. `analyze_campaign.py`: the `scalars.csv` header contained keys such as `I_F[0,20]` written unquoted. It now uses
   `csv.writer`. Every statistic is identical before and after (P1, P2 and all contrasts re-checked).
2. `validate_campaign.py`: the tolerance for recomputing the P1 / P2 medians from the 10-significant-digit per-seed
   CSVs was set to 1e-8 instead of 1e-12. The measured differences are 3e-11 and 1e-12.
3. `plot_figures.py`, after visual inspection: the annotation now names the coincidence of the t = 20 reset with the
   stop of the stop-20 arm, and the main-figure legend was moved off the annotation.
4. Log files are stored as `.txt` (git ignores `*.log`).

**Interpretation checks.**

* `fr_start40` equals ABF bitwise before t = 40 (validated per seed). Its I_F^{[0,40]} differs from ABF's in 1 of 16
  seeds, through the first FR event at the t = 40 checkpoint itself.
* 72 % of ABF's I_F^{[0,240]} lies in the warm-up window [0, 20], before the reset of the reported estimator. Totals
  over [0, 240] are therefore dominated by the warm-up estimate, which is why the preregistration uses [20, 80] for
  early convergence.
* The [20, 80] CIs are wide: for FR stopped at t = 20, −11.8 % [−29.2, +7.4]. With 16 seeds, modest post-reset early
  gains or losses cannot be excluded. The result is "not resolved", not "no effect".
* Arms differ in their FR streams by design, one per arm, so contrasts between two FR arms include FR-randomness
  variance. Langevin noise and initial conditions are shared.

**Figure inspection.**

* All five PDFs were viewed. No spurious spikes, clipping or mislabelled units.
* The t = 20 spike in e_F / e_F′ is the documented estimator reset (genuine).
* The genealogy sawtooth is the 8 t.u. windowed-label reset.
* After an FR stop, the windowed ESS stays at its last value (no more opportunities).
