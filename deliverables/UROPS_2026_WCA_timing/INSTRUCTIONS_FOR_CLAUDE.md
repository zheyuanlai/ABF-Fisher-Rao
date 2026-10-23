# Instructions for the manuscript-writing Claude (WCA timing study)

This package supplies data and figures only. Do not change the numbers. Definitions are in `FIGURE_PROVENANCE.md`;
the statistics and their exact wording are in `RESULTS_SUMMARY.md`.

## Recommendation for the WCA subsection: keep the negative result, and add that an earlier start only changes the early transient

* **Keep the validated WCA figure (FR from t = 40) as the headline WCA result.**
  * Its negative conclusion is replicated here on 16 fresh seeds: I_F +8.9 % [+1.4, +16.5], e_F(240) +172 %.
  * To show the replication, use `data/threepanel/wca_start40_*` (133 checkpoints instead of 49).
* **Add one paragraph, and optionally `figures/main_threepanel.pdf` or the `wca_start0` / `wca_stop20` CSV sets as a
  secondary figure.**
  * Starting FR at t = 0 lowers the early integrated error, I_F^{[0,40]} −7.5 % (preregistered P1, supported). That
    gain sits in the ABF warm-up window [0, 20], whose estimate is discarded at the t = 20 reset of the reported
    estimator.
  * After the reset, no schedule improves the main early window I_F^{[20,80]} with a resolved CI.
  * Continuous FR ends +150 % to +200 % above ABF whatever its start.
  * Switching FR off at t = 20 removes that late penalty (P2: final error −61.8 % vs continuous FR, supported).
  * The final error of FR stopped at t = 20 is then ABF-like (−6.0 % [−22.3, +10.3]; inconclusive at the +5 %
    margin). Its total integrated error is −8.1 % [−12.6, −4.4], a secondary, descriptive comparison driven mainly
    by the warm-up window.
* **Do NOT replace the figure with the early-stopping comparison as an improvement claim.**
  * The preregistered early-convergence criterion and safe-early-stopping criterion are NOT met by any schedule.
  * FR stopped at t = 20 is the best of seven schedules. Promoting it post hoc as "FR accelerates WCA" is not
    supported.

## Using the CSVs (`data/threepanel/`)

* Schema: `t,abf_median,abf_q25,abf_q75,fr_median,fr_q25,fr_q75`.
* There are 133 actual checkpoints per file. All values are finite and positive, so log axes are safe.
* The shading is the IQR across 16 seeds, not a CI.
* Fixed comparisons against ABF:
  * `wca_start0`: FR from t = 0, continuous;
  * `wca_stop20`: FR on 0 ≤ t < 20;
  * `wca_stop40`: FR on 0 ≤ t < 40;
  * `wca_start40`: FR from t = 40, the validated schedule;
  * `wca_start5`, `wca_start20`: secondary starts.
* `*_exi.csv` is the **full-domain** marginal error. It is dominated by replicas held in the soft walls (clamped into
  the edge bins, about 16 % of K), so do not read it as interior uniformity. The interior diagnostic is in
  `data/summary/master_series.csv` (`exi_interior`).

## Caption qualifications (required)

* State "validated dt = 1.25 × 10⁻⁴, exact Metropolis-MC reference, K = 256, 16 paired seeds".
* **Mark t = 20 as the end of the ABF warm-up and the reset of the reported ABF estimator.** It also explains the
  spike in e_F and e_F′ there.
* For the stop arms, mark the FR stop. For the stop-20 arm it coincides with the reset.
* Do not say that the early benefit carries through to the converged estimate: after the reset it is not resolved.
* The marginal panel shows uniformity of the population, not accuracy. Continuous FR improves the full-domain
  marginal (−4 % to −7 % on [20, 80]) while its free-energy and mean-force errors end worse.
* Distinguish the established findings (P1, P2, the late penalty, the replication) from the mechanistic hypotheses:
  * the stationary FR tilt ("fibre lag");
  * stopping at the reset keeps FR-period deposits out of the reported accumulators.
  Neither was tested.
* The earlier dt = 0.002 start-time ladder (`docs/WCA_FR_START.md`) is invalid as a free-energy result. Do not cite
  its gains.

## Do not

* Do not use the preview figures as data.
* Do not present the seven arms as tuned options.
* Do not describe the stop-20 schedule as a confirmed acceleration.
