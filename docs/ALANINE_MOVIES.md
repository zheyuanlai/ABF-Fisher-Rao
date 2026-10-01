# Alanine dipeptide movies: ABF vs ABF + Fisher-Rao on the Ramachandran plot

**Date:** 2026-10-01. Three presentation movies of real trajectories, one paired seed each, under the
selected 300 K configuration of the closed alanine studies (histogram estimator, adaptive boxes,
c_min 800, 5 ps ramp; `docs/ALANINE_HISTOGRAM_ABF.md`), with the torus-uniform Fisher-Rao arm from
0.5 ps at rate 0.15 -- the dose at which the birth-death events are visible (the campaign's frozen
rate 0.02 fires ~3 events per opportunity, 0.15 fires ~20-30). Both columns of every movie share the
initial ensemble and the Langevin noise stream, so until the first replacement the two clouds are
the same walkers.

| movie | CV biased | hidden angle | what the closed study found (16 seeds) |
|---|---|---|---|
| `ala2d` | (phi, psi) | none | FR NEUTRAL at every start and dose; both arms loop through the three wells identically |
| `phi` | phi alone | psi (fast) | ABF fine; FR at 0.15 HARMFUL (+19 % integrated, the flat 1-D marginal is churned) |
| `psi` | psi alone | phi (slow) | F(psi) converges to the phi<0 conditional; FR slows the hidden phi (mass on the C7ax side halves) |

## Engine record (bit-inert)

`AlaSimConfig.store_snapshots = k` / `Ala1DSimConfig.store_snapshots = k` records, every k steps and
for ONE seed (`snapshot_seed_index`), the walker angles (both angles in the 1-D engine: the CV and the
hidden one), persistent walker ids (a birth mints a fresh id), the 2-D watershed basin label of every
walker, the live PMF, and every realised death (at the dying walker's angles) and birth (at the
source's angles) against the snapshot they precede. It consumes no RNG and changes no arithmetic
(`tests/test_alanine_movie.py`: outputs bit-identical with the record on and off; every event of the
featured seed appears exactly once; the 1-D record's CV / hidden angles equal the 2-D engine's).

## Runs

`configs/alanine_movie/ala2d.yaml` (stages `abf`, `fr`) and `ala1d.yaml` (`phi_abf`, `phi_fr`,
`psi_abf`, `psi_fr`): seed 0, N 2048, 100 ps, snapshots every 50 steps (0.05 ps; 2001 per run),
init C7eq thermalised 20 ps and cached per config; `scripts/alanine_movie/launch.sh` (GPU 3).
Raw records `results/alanine_movie/<case>/<stage>/raw/*.npz` (ignored, ~60 MB each; reproduce with
the launcher); movies and READMEs in `results/alanine_movie/<case>/`.

## Frames (`scripts/make_alanine_movie.py`)

Top row: the Ramachandran plot per arm -- the reference free energy as the grey background with
contours, the three wells labelled, every walker as a dot coloured by its basin (C7eq, C5, C7ax,
elsewhere), FR deaths flashed as red crosses and births as green rings for the last few snapshots,
with running death / birth counts. Middle row: the learned free energy (2-D: the live PMF on the
same colour scale with the reference contours overlaid; 1-D: the live PMF profile against the full
marginal reference and, for psi, the phi<0 conditional reference, with the walkers' CV histogram).
Bottom row: the FES error of both arms against time with a moving marker, and the basin occupancies
of both arms (C7eq, C5, C7ax) against time with the reference equilibrium populations dotted -- the
"looping between the three wells" and, for psi alone, the slow filling of C7ax through the hidden phi.

## Movies (rendered 2026-10-01; 1920x1080, 30 fps, 35 s each; seed 0, rate 0.15)

| movie | FES error at 100 ps, ABF / FR (kJ/mol) | integrated error, ABF / FR | FR events | C7ax occupancy at the end, ABF / FR |
|---|---|---|---|---|
| `ala2d` | 0.53 / 0.53 | 96 / 93 (-3 %) | 4,183 | 4.9 % / 4.5 % |
| `phi` | 0.35 / 0.62 | 97 / 120 (+24 %) | 4,118 | 13.0 % / 15.2 % |
| `psi` | 2.09 / 2.16 | 217 / 219 (+1 %) | 1,551 | 12.5 % / 5.7 % |

Files: `results/alanine_movie/<case>/alanine_<case>_abf_vs_fr.mp4`, `movie_summary.json`, `README.md`
(what each panel shows and what to point at). The 2-D movie shows the closed NEUTRAL verdict (the
two learned surfaces and error curves stay on top of each other while the walkers loop through the
three wells identically); phi alone shows the HARMFUL verdict of the high dose on a flat 1-D
marginal (the learned barrier drifts above the reference once FR has nothing left to redistribute);
psi alone shows the hidden-coordinate mechanism (the learned F(psi) locks onto the phi<0-side
conditional, 9 kJ/mol too high at the barrier, while C7ax fills slowly through the hidden phi and
the FR deaths land exactly in the psi band where the C7ax walkers sit).
