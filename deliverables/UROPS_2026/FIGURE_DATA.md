# Figure data: format and per-figure guide

## CSV format (all 21 files in `figs/data/`)

```csv
t,abf_median,abf_q25,abf_q75,fr_median,fr_q25,fr_q75
```

* `t`: a saved simulation checkpoint, strictly increasing. Every row is a checkpoint the run actually saved. There
  is no interpolation, smoothing or thinning: all saved checkpoints of the run are kept.
* `*_median`, `*_q25`, `*_q75`: median and 25th / 75th percentiles **across seeds** of the per-seed error at that
  checkpoint (numpy `percentile`, linear interpolation between order statistics). They are not confidence bounds.
* ABF and FR use the same seeds and the same time grid. The three CSVs of a figure share one grid.
* **No zeros, no NaN.** Every value in all 21 files is finite and strictly positive (checked by
  `scripts/validate_exports.py`), so `ymode=log` needs no special handling and no value was replaced or clipped.
* Numbers are written with 10 significant digits.
* Per-seed values: `per_seed/<fig>_<metric>_<arm>.csv`, columns `t, seed_<id>…`. These are the unit of inference.

## Per-figure guide

| figure | rows | t range | t unit | e_F unit | e_F′ unit | e_ξ unit, domain, u | panel (c) type | FR on / off |
|---|---|---|---|---|---|---|---|---|
| `gateway` | 225 | 0.004 – 40 | reduced time | reduced energy (β = 16) | energy per unit x | x^−1/2; [−1.8, 1.8]; 1/3.6 | instantaneous histogram (180 bins) | on t = 0 (rate ramp τ = 4); never off |
| `lta150` | 101 | 0 – 60 | engine t.u. (EM dt 2e-4) | kJ/mol | kJ/mol/rad | rad^−1/2; circle; 1/(2π) | instantaneous **KDE** (0.1 rad) | on t = 4; never off |
| `lta300` | 101 | 0 – 60 | engine t.u. | kJ/mol | kJ/mol/rad | rad^−1/2; circle; 1/(2π) | instantaneous **KDE** (0.1 rad) | on t = 4; never off |
| `ite300` | 401 | 0 – 50 | engine t.u. (EM dt 1e-4) | kJ/mol | kJ/mol/rad | rad^−1/2; circle; 1/(2π) | instantaneous histogram (180 bins) | **on t = 1, off t = 10** |
| `meta` | 401 | 0 – 200 | reduced time | reduced energy (β = 4) | energy per unit x | x^−1/2; [−3, 3]; 1/6 | instantaneous histogram (120 bins) | on t = 0 (rate ramp τ = 20); never off |
| `alanine` | 101 | 0 – 100 | ps | kJ/mol | kJ/mol/rad | rad^−1; torus; 1/(2π)² | instantaneous histogram (97² cells) | on t = 20 ps; never off |
| `wca` | **49** | 0 – 240 | reduced LJ time | kT (β = 1) | kT per unit z | z^−1/2; [−0.2, 1.2]; 1/1.4 | instantaneous histogram (160 bins) | on t = 40; never off |

All panels (c) are instantaneous marginals of the current replicas at the save. None is interval-averaged. Two
are KDE-smoothed: LTA's, because the engine saves only its KDE marginal.

### Floors to draw or mention (dotted horizontal lines in the previews)

| figure | panel | floor | meaning |
|---|---|---|---|
| gateway | (b) | 0.0323 | hard piecewise-constant floor of the frozen e_F′ (within-bin variation of F′ at the gate) |
| gateway | (c) | 0.156 | finite-N multinomial floor (N = 2048 exactly-uniform walkers on 180 bins); FR dips slightly below it |
| ite300 | (c) | 0.167 | finite-N floor (N = 1024, 180 bins); both arms reach it by t ≈ 4–5 |
| meta | (b) | 0.139 | piecewise-constant floor (120 bins; steep quartic walls) |
| meta | (c) | 0.141 | finite-N floor (N = 1000, 120 bins) |
| alanine | (a) | 0.559 kJ/mol | bootstrap noise of the umbrella / MBAR reference |
| alanine | (b) | ≈ 8.7 kJ/mol/rad (indicative) | reference cell noise through the central difference |
| alanine | (c) | 0.341 | finite-N floor (N = 2048 on 97² cells) |
| wca | (c) | 0.666 | finite-N floor (N = 256, 160 bins); the curves sit above it because of edge-bin pile-up (below) |

The LTA KDE marginals have no analytic floor. They level off at ≈ 0.05 (finite-N KDE scatter).

### Genuine features (not processing artefacts)

* **gateway**:
  * initial plateau while walkers are confined to the left well;
  * ABF's late drop in (b) at t ≈ 4 when the gate is first crossed;
  * FR's earlier drop at t ≈ 3.
* **lta**: kink at t = 4, where the reported estimator switches to the production accumulators (estimator
  burn-in) and the ABF ramp ends.
* **alanine**: e_F rises during the 5 ps ABF ramp, then falls.
* **wca**: spike at t = 20 in (a) and (b), the reported-estimator burn-in reset. FR's e_F plateaus above ABF's after
  t ≈ 50.
* **meta**: FR's (c) reaches the floor by t ≈ 40, against t ≈ 150 for ABF, while FR's (a) ends ≈ 4× higher.

## Supplementary data (`figs/data_supplementary/`, same seven-column format; NOT part of the 21 required files)

| file | content | use |
|---|---|---|
| `alanine_exi_kde.csv` | e_ξ of FR's own KDE marginal (0.15 rad) on the torus | shows that even the smoothed marginal ends at its finite-N floor (0.042 vs 0.0416); the steric floor (0.025 for the reference's finite support) is below both |
| `wca_exi_interior.csv` | e_ξ over the 158 interior bins only | removes the edge-bin spikes of the replicas held in the soft-wall tails (≈ 16 % of N late in the run); sits at the finite-N floor in both arms |

## Preview figures

`figs/previews/<fig>.pdf` are matplotlib renderings of these CSVs, for checking only:

* ABF blue, FR orange dashed;
* IQR shading, log y axis;
* FR on / off markers and floors as dotted lines.
