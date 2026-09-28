# WCA-dimer movie, histogram estimator: ABF vs ABF + Fisher-Rao birth-death

`wca_abf_vs_fr_histogram.mp4` (1920x1080, 30 fps, 42.0 s, 30 MB) is one seed of the accepted corrected
Case IX WCA cell run with the textbook ABF histogram (P0) mean-force estimator of the replication
campaign (docs/HISTOGRAM_ABF_REPLICATION.md), both arms in one process from the same lattice
configuration and random seed, recorded every 50 steps and drawn two snapshots per frame.

| item | value |
|---|---|
| cell | beta 1, h 2, w 2 (bare dimer barrier beta*h = 2 kT), 10x10 = 100 WCA particles, a 1.5 (box 15 x 15), sigma = epsilon = 1 |
| sampler | N 1024 replicas, dt 0.002, 120 000 steps (T = 240), ABF bias ramp 10 000 steps, reported estimator starts after a 10 000-step burn-in, bias force clip 40 |
| estimator | histogram, 160 bins on [-0.2, 1.2] (Delta xi = 0.00875, the frozen width `configs/histogram_abf/selected_bins.json`); bias force = own-bin M_j / C_j; reported F = exact piecewise-linear integral of the bins (own read-out; e_F = window-aligned RMS over [-0.1, 1.1] against the constrained-TI reference `cache/phase_hp_v3/wca_ti_b1_h2_w2_n10_a1.5_g160.npz`, v2) |
| FR arm | `fr_uniform`: rate 0.1, every 5 steps from step 20 000 (t = 40), score clip 2, max event fraction 0.02, marginal KDE 0.07 (the accepted knobs; untouched by the estimator) |
| seed | 3200 (fresh label; the campaign used 3000-3007 / 3100-3115); featured replica: slot 0 |
| this seed | e_F(T): ABF 0.0863, FR 0.0443 (-49 %); I_F 38.3 vs 30.9 (-19 %); 3 511 replacements (every death is one birth); windowed ESS/N 0.717, max lineage share 0.040; round trips 705 k vs 732 k |
| confirmation medians (16 seeds, results/histogram_abf/wca/confirmation/summary.json) | histogram FR -50.5 % final [-52.3, -45.4], -23.1 % integrated [-28.6, -18.0], 16/16; kernel at h_read* -47.2 % / -19.0 % |
| time to ABF's final accuracy (this seed) | FR reaches e_F <= 0.0863 for good at t = 63, ABF at t = 240 (3.8x; campaign median 3.9x) |
| pairing | same lattice init and seed; the trajectories are bitwise identical only until t = 0.4 (the force kernel's scatter-add atomics are order-nondeterministic, docs/WCA_CORRECTED_CONFIRMATION.md), so the columns are paired in distribution, not step by step.  At t = 40, before FR has acted, the FR column happens to be 1.4x WORSE than ABF; by t = 80 it is 0.61x, by t = 120 0.45x |

Provenance: frozen design `configs/histogram_abf/campaign.json` (asserted against the production
YAML and the Case IX prereg by `scripts/run_histogram_abf_wca.py:load_frozen`), engine
`src/wca_abffr_core.py` (`SimConfig.abf_estimator = 'histogram'`) with the report-only
`store_snapshots` movie record added for this movie (bit-inert: `tests/test_wca_snapshots.py`), job
plumbing `src/wca_phase_jobs.py:execute_run`.  Data in `movie_data.npz`; the drawn / scored free
energy per frame is the reported estimator's own PMF, checked against the engine's `l2_f_t` at
every save (`scripts/make_wca_movie.py simulate`).

## What each panel shows

* **Top row, left of each column** - the physical configuration of replica 0: the two dimer particles
  (coloured, bonded) in their WCA solvent (grey discs of radius sigma/2), the periodic box re-centred
  on the dimer midpoint.  The normalised bond length xi = (|q1 - q2| - r0) / 2w of that replica is
  printed with its state (compact xi < 0.25, transition, stretched xi > 0.75).  Right column only: when
  slot 0 dies and is refilled by a copy of another replica (t = 90, 118, 168, 185, 204) the panel
  flashes "replaced by a copy of another replica" and the configuration jumps.
* **Top row, right of each column** - every replica drawn as a bead on the reference free-energy
  profile beta*F_ref(xi) (thick grey, constrained TI; the vertical jitter is a fixed per-slot offset
  so the beads do not shimmer).  The dotted curve is the bare dimer potential V_S(xi) =
  16 h xi^2 (1 - xi)^2 shifted to F_ref(0): the solvent lowers the barrier and deepens the stretched
  well, i.e. F != V_S.  Right column only: every Fisher-Rao death (red cross, at the dying replica's
  xi) and birth (green ring, at the source replica's xi) is flashed and fades over ~4 snapshots; the
  counters are cumulative.
* **Second row** - histogram of the replicas along xi (56 bins) in units of the uniform target
  (dashed), with the fraction in the stretched state.
* **Third row** - the learned free energy (the histogram estimator's own PMF, 160 bins) against
  the reference, both centred on the evaluation window [-0.1, 1.1] (the grey margins are outside it),
  with the current e_F(t).
* **Bottom left** - e_F(t) for both arms; solid = past, faint = the rest of the run, cursor at the
  current time.  The spike at t = 20 is the reported estimator restarting after the burn-in (the
  campaign's convention: the bias estimator sees every step, the reported one only post-burn-in).
* **Bottom right** - fraction of replicas in the stretched state against the uniform-target level
  0.32.

## What to point at while it plays

1. t < 7: from the compact lattice start the ABF bias ramps in and both marginals flatten on their
   own within ~6 time units (first stretched replica at t = 0.5, stretched fraction at 90 % of the
   target by t = 6.5): the WCA dimer is not barrier-limited under ABF, and the FR arm is still plain
   ABF (no events before t = 40).
2. t = 20: the reported estimator restarts (spike); t = 40: FR switches on.  Watch WHERE the
   events go: 3 472 of the 3 511 deaths are in the over-represented stretched state (xi > 0.75, where
   ABF alone leaves ~1.4x the uniform density), the births land in the transition region (1 607) and
   the compact state (1 452).  Selection enforces the uniform target that the bias alone does not.
3. t ~ 40-120: the FR error falls from 1.4x ABF's (seed noise at the switch-on) to 0.61x at t = 80
   and 0.45x at t = 120; the ABF curve has plateaued near 0.1 kT since t ~ 60.
4. t > 120: the ratio stays ~0.45-0.5 (final -49 %); replica 0 in the FR column is replaced five
   times, each time appearing in a new solvent cage.  FR reaches ABF's final accuracy at t = 63.

## Reproduce

```bash
CUDA_VISIBLE_DEVICES=3 python scripts/make_wca_movie.py simulate                # 163 s + 193 s on one H200
python scripts/make_wca_movie.py render --workers 24                             # ~1 min, 24 CPUs
python scripts/make_wca_movie.py render --only 700 --frames-dir /tmp/x           # one frame, layout check
CUDA_VISIBLE_DEVICES= python scripts/make_wca_movie.py simulate --smoke --out /tmp/smoke   # tiny CPU pipeline check
```
Other seeds: `simulate --seed 3201`; `--featured k` picks the replica drawn in the configuration panel;
`--n-bins` overrides the frozen width.
