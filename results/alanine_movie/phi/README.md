# Alanine dipeptide, CV = phi alone (psi hidden): ABF vs ABF + Fisher-Rao

`alanine_phi_abf_vs_fr.mp4` (1920x1080, 30 fps, ~35 s): one paired seed of the closed incomplete-CV
study (`docs/ALANINE_1D_CV.md`), recorded every 50 steps (0.05 ps), two snapshots per frame. Only phi
is biased; psi is the hidden coordinate and is shown on the Ramachandran plot unbiased. Both columns
share the initial C7eq ensemble and the Langevin noise; they differ only by the birth-death step.

| item | value |
|---|---|
| system / sampler | vacuum Ace-Ala-Nme, 300 K, dt 1 fs, gamma 1 ps^-1; N 2048 walkers, 100 ps; 1-D histogram estimator on phi (97 bins, adaptive boxes, c_min 800, 5 ps ramp); den Otter 1-D force |
| FR arm | `fr_uniform` on the phi marginal: uniform target on the circle, from 0.5 ps, rate 0.15, fr_every 500, cap 5 % of N |
| seed / init | seed 0, C7eq, `init_seed` 4242, `rng_seed` 20260903 |
| this seed | error of F(phi) vs the exact marginal of the 2-D reference: ABF 0.348, FR 0.621 kJ/mol at 100 ps; integrated 96.7 vs 120.1 (+24 %); 4,118 deaths / 4,118 births; C7ax occupancy at the end 13.0 % vs 15.2 % |
| the closed study (16 seeds, `results/alanine_1d/analysis/scoreboard.md`) | phi alone, FR rate 0.15 from 0.5 ps: +18.9 % [+12.9, +21.1] integrated, +67 % at the end, 0/16, HARMFUL; rate 0.02: -1.3 % integrated / +5.7 % final, inconclusive |

Provenance: `configs/alanine_movie/ala1d.yaml` (stages `phi_abf`, `phi_fr`), engine
`src/alanine/core1d_ala.py` with the report-only `store_snapshots` record (bit-inert,
`tests/test_alanine_movie.py`), renderer `scripts/make_alanine_movie.py --case phi`.

## What each panel shows

* **Top left / middle** - every walker on the Ramachandran plot over the reference 2-D free energy on
  its full 0-90 kJ/mol range (colour bar on the left panel; hatched = cells the reference never sampled;
  both angles are periodic, so clouds continue across the ±180° edges)
  (white wells, grey barriers), coloured by well (C7eq blue, C5 green, C7ax orange, elsewhere
  purple). Only phi is biased, so the cloud spreads along phi and stays in the psi valleys the
  physics allows. In the FR column: deaths of the last six snapshots as red crosses, births as green
  rings, cumulative counters.
* **Bottom left / middle** - the learned F(phi) (coloured) against the exact marginal of the 2-D
  reference (black), on the full 0-70 kJ/mol range with the walkers' phi histogram shaded underneath (the FR target is a flat
  histogram).
* **Right column** - the error of F(phi) against time; the KL distance of the phi marginal to the
  uniform circle (reachable here: it falls to ~0.03, unlike the torus case); the well occupancies.

## What to point at while it plays

* By ~4 ps both clouds have crossed into C7ax (orange) through the flattened phi barrier; psi
  relaxes within picoseconds inside each well, so the hidden coordinate is not a problem for phi.
* Once the phi marginal is flat (~20 ps) FR has nothing left to redistribute, yet at rate 0.15 it
  keeps killing and cloning ~20 walkers per opportunity; the red learned curve drifts above the
  black reference at the barrier while the blue (ABF) curve settles on it, and the error curves
  separate for good. This is the closed study's HARMFUL verdict on the 1-D phi marginal.
