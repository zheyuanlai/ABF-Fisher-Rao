# Ethane/LTA under the histogram (P0) ABF estimator — replication, 350 K, matched sham

Study of 2026-10-03/04 (overnight, autonomous). Prereg, predictions and results narrative:
`docs/LTA_HISTOGRAM_REPLICATION.md`; frozen design: `configs/lta_histogram/campaign.json`.

| file | what |
|---|---|
| `scoreboard.md`, `summary.json` | all endpoints, contrasts, clauses, verdicts (`scripts/analyze_lta_histogram.py`) |
| `comparison_T{80,150,225,300,350}.csv` | per-seed integrated / final errors of abf, fr_uniform, fr_sham and the paired deltas |
| `production_T*/{abf,fr_uniform,fr_sham}.npz` | raw runs (git-ignored; `scripts/lta_histogram/launch.sh` reproduces them) |
| `calibration/fr_rate_selection_T350.json` | the 350 K safety-only rate ladder (0.20 selected) |
| `calibration/width_ladder_T300/n{90,180,360}.npz` | ABF-only width ladder (git-ignored) |
| `pre_run_kernel_only/` | the analyzer's output BEFORE any FR result existed (floor clause, kernel reproduction, screening rule) |
| `figures/fig_lta_hist_benefit_vs_T` | the headline: ΔI_F and Δe_F(T) vs T, histogram vs kernel vs sham vs direct FR-vs-sham |
| `figures/fig_lta_hist_convergence_T*`, `fig_lta_hist_profiles_T*` | median e_F(t) of all arms (histogram solid, kernel dotted); final profiles |
| `figures/fig_lta_hist_establishment` | KL(p‖uniform) vs t and the ABF-only discovery / establishment timings vs T |
| `figures/fig_lta_hist_width_ladder` | the width ladder with the deterministic P0 floor |
| `figures/fig_lta_decomposition_vs_T` | F, U, −TS vs z for 80–350 K and the barrier decomposition vs T |
| `figures/fig_lta_configuration_cloud_T{300,150}` | COM cross-section and orientation in the cage vs in the window (from the movie records) |
| `logs/` | chain driver and per-stage logs (git-ignored) |

Headline (16 seeds × {abf, fr_uniform, fr_sham} per T, paired medians, 95 % bootstrap CIs):

| T (K) | hist FR vs ABF ΔI_F | kernel sweep | sham vs ABF | FR vs sham (direct) |
|---|---|---|---|---|
| 80 | −35.1 % [−36.2, −33.4] 16/16 | −35.1 % | −1.2 % | −34.5 % 16/16 |
| 150 | −28.1 % [−29.2, −26.3] 16/16 | −31.9 % | −0.9 % | −27.0 % 16/16 |
| 225 | −19.9 % [−22.2, −17.3] 16/16 | −21.3 % | +0.1 % | −20.8 % 16/16 |
| 300 | −14.1 % [−15.1, −11.0] 16/16 | −14.8 % | −0.2 % | −13.2 % 16/16 |
| 350 (new) | −11.8 % [−15.2, −9.5] 16/16 | – | +0.2 % | −12.6 % 16/16 |

Movies: `results/lta_movie/T300/` and `results/lta_movie/T150/` (ABF vs ABF + FR, one seed each, READMEs).
