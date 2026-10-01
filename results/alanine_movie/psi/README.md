# Alanine dipeptide, CV = psi alone (phi hidden): ABF vs ABF + Fisher-Rao

`alanine_psi_abf_vs_fr.mp4` (1920x1080, 30 fps, ~35 s): one paired seed of the closed incomplete-CV
study (`docs/ALANINE_1D_CV.md`), recorded every 50 steps (0.05 ps), two snapshots per frame. Only psi
is biased; phi, the slow angle, is hidden and is shown on the Ramachandran plot unbiased. Both
columns share the initial C7eq ensemble and the Langevin noise; they differ only by the birth-death step.

| item | value |
|---|---|
| system / sampler | vacuum Ace-Ala-Nme, 300 K, dt 1 fs, gamma 1 ps^-1; N 2048 walkers, 100 ps; 1-D histogram estimator on psi (97 bins, adaptive boxes, c_min 800, 5 ps ramp) |
| FR arm | `fr_uniform` on the psi marginal: uniform target on the circle, from 0.5 ps, rate 0.15, fr_every 500, cap 5 % of N |
| seed / init | seed 0, C7eq, `init_seed` 4242, `rng_seed` 20260903 |
| this seed | error of F(psi) vs the exact marginal of the 2-D reference: ABF 2.09, FR 2.16 kJ/mol at 100 ps (the phi<0-side conditional reference is matched to ~0.4); integrated 217 vs 219; 1,551 deaths / 1,551 births; C7ax occupancy at the end 12.5 % (ABF) vs 5.7 % (FR), reference 3.1 % |
| the closed study (16 seeds, `results/alanine_1d/analysis/scoreboard.md`) | psi alone: ABF converges to the phi<0 conditional (error 0.51 vs it, 2.07 vs the full marginal); FR at 0.15 NEUTRAL on the PMF (+1.3 %) but the walkers' mass on the C7ax side halves (0.129 -> 0.064) and the hidden conditional distance rises 14.5 % |

Provenance: `configs/alanine_movie/ala1d.yaml` (stages `psi_abf`, `psi_fr`), engine
`src/alanine/core1d_ala.py` with the report-only `store_snapshots` record (bit-inert,
`tests/test_alanine_movie.py`), renderer `scripts/make_alanine_movie.py --case psi`.

## What each panel shows

* **Top left / middle** - every walker on the Ramachandran plot over the reference 2-D free energy on
  its full 0-90 kJ/mol range (colour bar on the left panel; hatched = cells the reference never sampled;
  both angles are periodic, so clouds continue across the ±180° edges),
  coloured by well (C7eq blue, C5 green, C7ax orange, elsewhere purple). Only psi is biased, so the
  clouds are vertical columns at the phi values the physics allows; the slow phi crossing to C7ax
  (right) happens only occasionally. FR column: deaths of the last six snapshots as red crosses,
  births as green rings, cumulative counters.
* **Bottom left / middle** - the learned F(psi) (coloured) against the exact marginal of the 2-D
  reference (solid black) and against the conditional marginal of the phi < 0 side only (dashed),
  with the walkers' psi histogram shaded; the title gives the error against each.
* **Right column** - the error of F(psi) against time: solid vs the full marginal, dashed vs the
  phi<0-side reference; the KL distance of the psi marginal to the uniform circle; the well occupancies.

## What to point at while it plays

* By ~5 ps the psi marginal is flat (KL ~0.03) and the learned curve has locked onto the DASHED
  reference, i.e. the free energy of the phi < 0 half of the molecule: between psi = -130 and -50 deg
  it reports a barrier ~9 kJ/mol too high because the C7ax side, which dominates the true marginal
  there, has barely been visited. The error against the full marginal stays near 2 kJ/mol for the
  whole run while the error against the phi<0 reference falls to ~0.4.
* Watch the orange C7ax occupancy (bottom right) creep up through the hidden phi: ABF reaches ~9 %
  by 70 ps and keeps climbing; the FR column stays near 4 %. The deaths (red crosses) fall in the
  psi band where the C7ax walkers sit, because the lagging bias leaves that band transiently
  over-populated relative to the uniform target: FR removes exactly the walkers carrying the new
  phi information. This is the closed study's mechanism for "FR slows the hidden coordinate".
