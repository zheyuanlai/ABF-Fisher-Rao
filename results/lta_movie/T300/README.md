# Ethane in LTA at 300 K, histogram estimator: ABF vs ABF + Fisher-Rao birth-death

`lta_abf_vs_fr_histogram.mp4` (1920x1080, 30 fps, 35.4 s, 26 MB) is one seed of the ethane/LTA
system (`src/lta/core_lta.py`) run under the textbook online histogram (P0) ABF mean-force estimator
with the frozen knobs of `configs/lta_histogram/campaign.json`, both arms in ONE process from the same
initial ensemble and Langevin noise stream, recorded every 100 steps (3 001 snapshots) and drawn three
snapshots per frame. Made 2026-10-03 (docs/LTA_HISTOGRAM_REPLICATION.md).

| item | value |
|---|---|
| system | TraPPE-UA ethane (two CH3 beads, bond 1.54 Å) in rigid all-silica LTA (IZA framework, 2x2x2 pseudo-cells, a = 11.919 Å), CH3-O LJ eps/kB 93 K, sigma 3.48 Å, rc 10 Å; overdamped Brownian dynamics dt 2e-4; T = 300 K |
| CV | ethane COM position along the cage-window-cage axis, folded onto one period: window plane at z = 0, alpha-cage centres at z = +-a/2 = +-5.96 Å |
| reference | umbrella sampling + WHAM (`results/uniform_campaign/lta/reference/reference_T300.npz`): dF‡ = 10.8 kT, of which -TdS‡ = 7.8 kT (72 % entropic), dU‡ = 3.0 kT |
| sampler | N = 1024 replicas, 300 000 steps (T = 60), ABF bias ramp 20 000 steps (t = 4), reported estimator restarts after the 20 000-step burn-in, bias force clip 60 |
| estimator | histogram, 180 circular bins (0.066 Å); bias force = own-bin M_j / C_j; reported F = exact piecewise-linear integral (own read-out; e_F = full-circle aligned RMS vs the reference) |
| FR arm | `fr_uniform`: rate 0.20 (the sweep's safety-frozen rate), every 5 steps from step 20 000 (t = 4), score clip 2, max event fraction 0.02, marginal KDE 0.10 |
| seed | label 1200, rng 20261300 (fresh); featured replica: slot 0 (never replaced in this run) |
| this seed | e_F(T): ABF 0.0915 kJ/mol, FR 0.0813 (-11.1 %); I_F 25.4 vs 22.2 (-12.8 %); 665 replacements; min ancestor ESS/N 0.454, max lineage share 0.007; cage-to-cage crossings 2 721 vs 2 762 |
| kernel sweep medians at 300 K (16 seeds, results/uniform_campaign/lta/summary_T300.json) | -14.8 % integrated [-17.0, -11.7], -19.7 % final, 16/16; this seed's histogram-estimator numbers sit inside that spread; the 16-seed histogram replication is `results/lta_histogram/` |
| time to ABF's final accuracy (this seed) | FR holds e_F <= 0.0915 from t = 49.6, ABF from t = 59.7 |
| pairing | identical trajectories until the first FR opportunity at t = 4.02; paired in distribution afterwards. Re-simulated 2026-10-04 with the genealogy keys (same seed; a CUDA re-run differs slightly after t ~ 10 through scatter-add order: the first 145 events are identical, the endpoint moved from -10.1 % to -11.1 %); first record archived in `raw_pre_genealogy/` |

## What each panel shows

* **Top row, per column** - the x-y projection of the framework slab through the cage and window centres
  (O light grey, Si dark grey; two alpha-cages at x = a/2 and 3a/2, 8-ring windows at x = 0, a, 2a),
  with the COM of every replica folded into the two cages of the periodic box (x mod 2a, y mod a) and the
  featured replica drawn as its two CH3 beads. Right column only: Fisher-Rao deaths (red x) and births
  (green ring, at the source replica) flashed for the last four snapshots; cumulative counters.
* **Second row** - the replica histogram along the CV in units of the uniform target (dashed), the fraction
  in the window (|z| < 1.5 Å) and the cumulative cage-to-cage crossings.
* **Third row** - the learned free energy (the histogram estimator's own PMF) against the reference, in kT,
  with e_F(t).
* **Bottom left (a)** - the reference decomposition F = U - TS: F (black), U (blue dashed), -TS (vermilion
  dash-dot) in kT, cursors at the two featured replicas. The window is not high in ENERGY (dU‡ 3 kT);
  it has few admissible configurations (-TdS‡ 7.8 kT).
* **Bottom middle (b)** - the transverse cross-section (y, z mod a) of the replicas currently in the cage
  (|z_CV| > 4 Å) and in the window (|z_CV| < 1.5 Å), pooled over both arms and the last 40 snapshots, over
  the 8-ring O atoms, with the mean |cos theta| of the molecular axis against the channel axis.
  Over the whole run (ABF arm, t >= 4): transverse rms 2.78 Å in the cage vs 0.22 Å in the window
  (~160x less cross-section area), <|cos theta|> 0.53 (isotropic) vs 0.955 (aligned);
  ln(160) + ln(~9) ~ 7.3 k_B, close to the measured -TdS‡/kT = 7.8
  (`results/lta_histogram/figures/fig_lta_configuration_cloud_T300.png`).
* **Bottom right (c)** - e_F(t) of both arms (log), the ramp end / estimator restart at t = 4 (= FR start
  here), and the current window fractions against the uniform level 0.25.

## What to point at while it plays

1. t < 4 (ABF bias ramping, FR not yet on, both columns identical): the replicas sit in the cages;
   the first window visit is at t = 1.0; by t = 4 only 3 % of replicas are in the window and 14
   crossings have happened in 1024 replicas. The cage cloud in (b) fills the cross-section and the
   window cloud, when it appears, is a dot: the entropic barrier made visible.
2. t = 4 to 10: FR switches on while the marginal is still far from flat. 145 replacements in this
   interval; 129 of the 145 deaths are in the CAGE (over-represented) and 136 of the 145 births go to
   the neck / window (under-represented): selection pushes population toward the bottleneck while ABF
   is still learning it. Window fraction at t = 6: ABF 0.13, FR 0.17; at t = 10: 0.20 vs 0.23.
3. t = 10 to 30: e_F(FR) / e_F(ABF) is 0.67-0.77; the FR marginal reaches the uniform level first
   (0.243 at t = 20 vs the target 0.252; ABF overshoots to 0.257-0.271 because the bias flattens the
   window and the cage centres differently). Deaths now spread over cage / neck / window
   (285 / 232 / 148 over the whole run) and births favour the neck (319 / 175 / 171): the steady
   state is a redistribution within an already-flat marginal, which is why the gain saturates.
   Genealogy (new record): at t = 6-10 about 20-22 % of the window replicas are Fisher-Rao copies vs
   1-6 % in the cages (150 K: 40-52 % vs 0-4 %); the largest family ever has 6 members.
4. t > 30: the two curves converge (ratio 0.81 at t = 50, 0.89 at the end); FR is at ABF's final
   accuracy from t = 49.6. At 300 K the LTA cell is the mildest point of the temperature sweep;
   `results/lta_movie/T150/` shows the same system where ABF starves (0.9 crossings per replica).

Provenance: `scripts/run_lta_histogram.py movie --temperature 300` (simulate; raw records in `raw/`,
ignored by git), `scripts/lta_movie_render.py --out results/lta_movie/T300` (frames + ffmpeg via
imageio-ffmpeg), engine record `store_snapshots` (bit-inert, `tests/test_lta_histogram.py`).

## The ensemble movie: `lta_ensemble_abf_vs_fr.mp4` (≈34 s; `scripts/lta_ensemble_movie_render.py`)

Same three-act construction as `../T150/README.md` (replica strip with originals grey and Fisher–Rao
copies orange, p̂/q heat strip, R_W meter, kymographs with lineage trees). Highlighted families: initial
walkers 770 (first in the window at t = 2.5, before FR starts, so the same walker in both arms), 637 and
868 (t = 5.1, 6.0); they end with 6, 6 and 6 members (largest family ever 7). At 300 K the collective
mechanism is much weaker than at 150 K: at t = 8, 22 % of the window replicas are copies (150 K: 52 %),
629 of 1024 families are alive at the end (150 K: 516), and the meters read 0.63× vs 0.75× at t = 6 and
converge by t ≈ 12 — consistent with the −11 % endpoint here against −46 % at 150 K. Watch this movie
after the 150 K one: it is the same machinery on a cell where ABF alone is not starved.
