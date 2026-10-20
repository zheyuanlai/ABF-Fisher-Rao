# Figure and data provenance

## Common to every figure and CSV

* **Runs.** 112 confirmation runs: 7 arms × 16 paired seeds 5600–5615.
  * Raw files: `results/urops_2026_wca_timing/raw/s<seed>_<arm>.npz` in the main research checkout (git-ignored,
    69 MB).
  * Every run was produced at commit **e25edb3e** on branch `urop-wca-timing-validated`, which holds the
    preregistration and was pushed before the runs. That commit's engine is `src/wca_numba.py`; its sha256 is
    recorded in each file.
  * Launched 2026-10-20 16:31 UTC. Each run took 0.98–0.99 h single-threaded, about 110 core-h in total.
* **Physics** (the validated Phase-V cell, unchanged):
  * WCA solvent + double-well dimer, 100 particles per replica, β = 1;
  * float64 Euler–Maruyama with the accepted force clips and soft reaction-coordinate wall;
  * dt = 1.25 × 10⁻⁴, K = 256, T = 240 (1 920 000 steps);
  * lattice initial condition and Langevin noise of the seed, identical in every arm.
* **ABF.**
  * P0 histogram, 160 bins on [−0.2, 1.2], edge-extrapolated.
  * Linear warm-up ramp to t = 20.
  * The reported estimator switches to the production accumulators at t = 20. **The t = 20 checkpoint is the reset**:
    the production estimator holds one step of deposits there, which causes the spike in e_F and e_F′.
* **FR.**
  * Uniform target on [−0.2, 1.2], KDE bandwidth 0.07.
  * Constant rate 0.1 while active, no ramp.
  * Opportunities every 0.01 t.u. (80 steps); score clip 2; cap 5 per opportunity, cap_min 0.
  * Death-driven, score-weighted refill.
  * Each arm has its own FR stream: SeedSequence child 1 + id, with ids abf —, start40 0, start0 1, stop20 2,
    stop40 3, start5 4, start20 5.
  * Stopping: no FR opportunity at or after t_stop. The replicas and estimators then continue with plain ABF.
* **Reference.** Exact Gibbs by Metropolis MC, 64 chains, `rescore_wca_history.mc_reference`. Frozen in
  `data/summary/reference_mc.json` (array sha256 f58f069c…). The analysis asserts equality with it.
* **Errors.** All from the Phase-V read-out, `wca_numba.score_ladder_result`.
  * e_F(t) = `l2_f_t`: aligned L² on [−0.1, 1.1], in kT.
  * e_F′(t) = `l2_fp_t`: the histogram arms' own mean-force error on [−0.1, 1.1], in kT per unit z.
  * e_ξ(t), full domain: ‖p − 1/1.4‖ on [−0.2, 1.2] from the instantaneous replica histogram `z_hist` (160 bins).
    The engine clamps replicas in the soft-wall tails into the two edge bins.
  * Interior diagnostic: the same over bins 1–158, renormalised.
  * Edge fraction: (z_hist[0] + z_hist[159]) / K.
* **Statistics.**
  * CSV curves: median and 25 / 75 % percentiles across the 16 seeds at every saved checkpoint (numpy linear
    percentile).
  * Contrasts: the median per-seed relative change with a 10 000-resample percentile bootstrap CI.
* **Time grid.** All 133 saved checkpoints: every 0.5 t.u. on [0, 20], every 1 t.u. on (20, 80], every 5 t.u. on
  (80, 240]. There is no interpolation.

## Data files

| file | content | source |
|---|---|---|
| `data/threepanel/wca_start0_{eF,eFp,exi}.csv` | ABF vs `fr_start0` (FR from t = 0, continuous) | `analyze_campaign.py` → `export_figures.py` |
| `data/threepanel/wca_stop20_*.csv` | ABF vs `fr_start0_stop20` (FR on 0 ≤ t < 20) | same |
| `data/threepanel/wca_stop40_*.csv` | ABF vs `fr_start0_stop40` (FR on 0 ≤ t < 40) | same |
| `data/threepanel/wca_start40_*.csv` | ABF vs `fr_start40` (the validated schedule, fresh seeds) | same |
| `data/threepanel/wca_start5_*.csv`, `wca_start20_*.csv` | ABF vs the secondary starts | same |
| `data/summary/master_series.csv` | median / IQR series of every arm and series | `export_figures.py` |
| `data/per_seed/*.csv`, `scalars.csv` | per-seed series and scalars (unit of inference) | `analyze_campaign.py` |
| `data/summary/analysis.json`, `contrasts.csv` | all contrasts, verdicts, time to accuracy, genealogy, run checks | `analyze_campaign.py` |

The `exi` panel of every three-panel CSV is the **full-domain** marginal error (the task's definition). The interior
diagnostic and edge fraction are in `data/per_seed/<arm>__exi_interior.csv` and `__edge_frac.csv`, and in the master
table.

## Figures (`figures/`, rendered by `scripts/plot_figures.py` from `data/` only)

| figure | content | arms | annotations |
|---|---|---|---|
| `main_threepanel.pdf` | e_F, e_F′, e_ξ (full); median and IQR; log y | ABF, FR from 0 (continuous), FR on [0, 20) | dotted grey t = 20: ABF warm-up end + estimator reset; dash-dot green t = 20: FR stop of the green arm (the two coincide) |
| `timing_comparison.pdf` | paired change vs ABF of I_F^{[0,40]}, I_F^{[20,80]}, e_F(240), with 95 % CI and FR-better counts | all six FR schedules | circles = continuous FR, squares = early-stopped FR |
| `marginal_vs_free_energy.pdf` | e_ξ full domain, e_ξ interior (diagnostic), edge-bin fraction, e_F | ABF, FR from 0, FR on [0, 20), FR from 40 | t = 20 reset |
| `supp_all_arms.pdf` | e_F, e_F′, e_ξ medians | all seven arms | t = 20 reset (= stop of the stop-20 arm); cyan t = 40 = stop of the stop-40 arm |
| `supp_genealogy.pdf` | cumulative replacements; windowed ancestor ESS / K | six FR arms | after a stop, the windowed labels are no longer reset (ESS frozen) |

Caption requirement: identify t = 20 as the ABF warm-up end **and** the reported-estimator reset in every figure
that shows e_F or e_F′.
