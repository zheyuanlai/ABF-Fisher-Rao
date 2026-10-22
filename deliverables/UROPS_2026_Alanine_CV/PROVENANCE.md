# Source-data provenance: alanine dipeptide with an incomplete (1-D) collective variable

**No new simulation was run.** Every number is post-processing of six existing runs and of the project's 2-D MBAR
reference.

## Raw runs

They are git-ignored and live in the main research checkout `/home/zheyuanlai/ABF-Fisher-Rao`.

| stage | CV | method | raw file (`results/alanine_1d/<stage>/raw/`) | config hash | sha256 (first 12) |
|---|---|---|---|---|---|
| `phi_abf` | φ | histogram ABF | `phi_abf__abf__c7eq__N2048__T100000__ns16__4ed0df09e857.npz` | 8090ac4d8992 | 762901df8b28 |
| `phi_u02_t0` | φ | ABF + uniform FR, rate 0.02 | `phi_u02_t0__fr_uniform__c7eq__N2048__T100000__ns16__ebb1c849ddc4.npz` | 8090ac4d8992 | 28f1543955f5 |
| `phi_u15_t0` | φ | ABF + uniform FR, rate 0.15 | `phi_u15_t0__fr_uniform__c7eq__N2048__T100000__ns16__5a9d3c87b62e.npz` | 9d5e2dd8fe88 | 05c86af8ca30 |
| `psi_abf` | ψ | histogram ABF | `psi_abf__abf__c7eq__N2048__T100000__ns16__18090ea3eb6c.npz` | a004d4e5ad7f | d73759f86126 |
| `psi_u02_t0` | ψ | ABF + uniform FR, rate 0.02 | `psi_u02_t0__fr_uniform__c7eq__N2048__T100000__ns16__ac0dd1bf7ac6.npz` | a004d4e5ad7f | 949b2f392c67 |
| `psi_u15_t0` | ψ | ABF + uniform FR, rate 0.15 | `psi_u15_t0__fr_uniform__c7eq__N2048__T100000__ns16__f3fd1beaa055.npz` | ee0602cf41c2 | eec3c13b0f46 |

The full sha256 of each file is in `meta/export.json → arms.<stage>.raw_sha256`.

**Common to all six runs** (2026-09-30):

* engine `src/alanine/core1d_ala.py`, run from commit ebc123bb with `git dirty: true`, on GPU 3;
* alanine dipeptide in vacuum, AMBER ff14SB, BAOAB Langevin at dt 1 fs, γ 1/ps, 300 K, float64 (CUDA graphs);
* N = 2048 walkers, 100 000 steps = 100 ps, saves every 1 ps (101 saves), 16 paired seeds 0–15;
* `rng_seed` 20260903 and the cached C7eq initial ensemble (`results/fr_start_timing/alanine/init_c7eq_seeds0-15_N2048.npz`).

**ABF.** Adaptive-box histogram estimator on 97 circular bins of the biased dihedral (IUPAC):

* 1-D boxes of 1–9 bins, c_min = 800 counts, 5 ps bias ramp;
* the walker feels its own bin's force;
* the reported PMF is the exact integral of the piecewise-constant bin force after removing its circular mean.

**FR.**

* Uniform target 1/(2π) on the CV circle; wrapped-Gaussian KDE, bandwidth 0.15 rad; score clip 2.
* `fr_start_steps` 0 and `fr_every` 500, so the first FR opportunity is at step 500 = **0.5 ps** and then every
  0.5 ps. FR is never stopped.
* Cap ⌊0.05 N⌋ = 102 per opportunity; full-state kill-and-clone (`core2d_ala`); 6 ps age-aware genealogy window.

**Pairing.** All arms of a CV share the initial ensemble and the dynamical noise stream. The curves are identical at
t = 0 and differ by ≤ 4 % at t = 1 ps. The CUDA `scatter_add` is order-nondeterministic
(`docs/FR_START_TIMING.md`), so the pairing is statistical, not bitwise.

The original study is documented in `docs/ALANINE_1D_CV.md` (§7 results), with analysis in
`scripts/analyze_alanine_1d.py` and outputs in `results/alanine_1d/analysis/`. The matched kernel-estimator check
arms (`*_abf_kernel`) are not part of this export.

## Reference

* **2-D reference.** `results/alanine/reference/reference.npz`: umbrella sampling with 24 × 24 windows × 16 copies,
  κ 200 kJ/mol/rad², and MBAR on the 97 × 97 grid (300 frames per window). Never-visited cells are +∞.
* **1-D free-energy references** (`alanine.core1d_ala.marginal_reference_1d`, as in the original analysis):
  * F(ξ) = −kT log Σ_hidden exp(−F₂/kT), for ξ = φ and ξ = ψ;
  * the φ < 0 conditional marginal of ψ, used for the "visited-side" comparison.
  * The 8 kT mask and equilibrium (Boltzmann) weights come from `metrics_ala.build_masks` applied to the 1-D
    reference.
* **Uncertainty** (`scripts/reference_1d.py` → `data/reference/reference_1d.json`). The project's own copy-level
  block bootstrap (`scripts/bootstrap_alanine_reference.py`) was re-run:
  * whole copies resampled within each window, MBAR re-solved, 200 replicates, seed 20260805, 185 s on GPU 3
    (idle);
  * each replicate propagated through the marginalisation and the derivative;
  * the re-solved point estimate reproduces the stored reference to 1.4 × 10⁻¹⁴ kJ/mol.

**1-D mean-force reference.** There is **no independent mean-force reference**: the umbrella data store angles, not
forces.

* It is the periodic central difference of the 1-D cell free energies. To O(dz²) this equals the cell-averaged mean
  force that the histogram ABF reports.
* It is a numerical derivative of a noisy grid, **not exact**. Its uncertainty has two parts:

| CV | statistical (bootstrap, correlations included) | discretisation (3- vs 5-point stencil) | total | ABF final e_F′ | ratio | verdict (rule fixed before computing e_F′) |
|---|---|---|---|---|---|---|
| φ | 1.17 kJ/mol/rad | 0.47 | 1.26 | 1.35 | 0.93 | **FLAGGED**: φ e_F′ is reference-limited after ≈ 10 ps |
| ψ | 1.58 | 0.62 | 1.70 | 4.03 | 0.42 | RELIABLE (reference < 25 % of the squared error) |

* Free-energy reference-noise floor (aligned L², equilibrium weight): **0.205 kJ/mol for φ**, 0.267 for ψ, 0.264 for
  the φ < 0 conditional. The φ-only ABF endpoint, 0.374, is therefore partly reference-limited.

## 2-D reference surface (for the Ramachandran panel)

The data are in `data/reference/` and are produced by `scripts/export_reference_2d.py`.

* **`reference_F2_kJ.csv`.** F(φ, ψ) in kJ/mol, exactly the stored MBAR surface, shifted so that its global
  minimum is 0.
  * **Rows are φ and columns are ψ.** Both use the grid −π + (i + ½)·2π/97, listed in degrees in
    `reference_F2_grid_deg.csv`.
  * `NaN` marks the 226 never-visited cells.
  * Written with `np.savetxt` defaults, so float64 round-trips exactly.
  * The φ, ψ and ψ|φ<0 marginals recomputed from this CSV equal the 1-D references used by every panel exactly.
* **`reference_F2_se_kJ.csv`.** The copy-level bootstrap standard error per cell. The median on the 8 kT mask is
  0.61 kJ/mol.
* **`reference_F2_minima.csv`.** Cell-grid local minima whose prominence (the lowest barrier to lower ground)
  exceeds twice the median cell s.e.:

  | basin | φ, ψ (deg) | F − F_min (kJ/mol) | prominence (kJ/mol) |
  |---|---|---|---|
  | C7eq | −74.2, 55.7 | 0 | — |
  | C5 | −152.2, 155.9 | 1.12 | 6.35 |
  | C7ax | 63.1, −48.2 | 6.39 | 32.9 |

  * A fourth entry at (137.3°, 40.8°), 84 kJ/mol, lies in an essentially unsampled region with no bootstrap s.e.
    It is an artefact, not a basin.
* **There is no resolved αR minimum.** The only cell-level local minimum in the αR region is (−55.7°, −18.6°) at
  12.2 kJ/mol, with a prominence of only 0.12 kJ/mol, below the cell noise. If αR is labelled, it should be labelled
  as a region or shoulder, not as a minimum of this surface.

## Definitions of the exported errors

* **e_F(t).** Equilibrium-weighted, additive-constant-aligned L² of the arm's 1-D PMF against the full 1-D marginal
  reference on its 8 kT mask (`metrics_ala.aligned_l2`). This is the original primary read-out, in kJ/mol.
* **e_F′(t).** Equilibrium-weighted RMS, over the stencil-complete cells of the same mask, of the estimator's
  reported bin mean force minus the reference mean force, in kJ/mol/rad.
  * The bin force is taken **raw**, as the walkers feel it, including its non-zero circular mean ("circulation",
    about −0.6 to −1.0 at 100 ps).
  * Supplementary variants:
    * *closed*: the bin force minus its circular mean, i.e. the derivative of the reported PMF;
    * *dispersion*: per seed, the RMS distance from the arm's across-seed mean force. It is reference-free and
      measures precision only. It is exactly 0 at t = 0, where every seed holds the same empty estimator.
    * *debiased*: √max(e_F′² − σ_noise², 0). Indicative only: the discretisation part is not removed.
* **e_ξ(t).** ‖p_t − 1/(2π)‖_L² on the circle, from the saved **instantaneous** walker marginal `marg_hist`. The
  finite-N floor for exactly uniform walkers is 0.086.
* **Hidden φ > 0 mass.**
  * Instantaneous: the fraction of the 2048 walkers with φ > 0 at each save, from the saved joint (φ, ψ) histogram.
  * Window-pooled (the original statistic): joint histograms summed over the saves in each window.
  * Target under a uniform-ψ bias: the ψ-uniform average of the reference conditional p(φ > 0 | ψ), **0.321**. The
    unbiased Gibbs population, 0.032, is shown only for contrast.
* **D_cond.** Window-pooled, not instantaneous, for the windows [1, 5], [5, 20], [20, 50] and [50, 100] ps.
  * Joint histograms are summed over the saves in a window and pooled to about 15° bins on both axes (4 cells).
  * For every pooled CV bin with ≥ 200 samples, compute the TV between the walkers' p(hidden | ξ) and the reference
    conditional; D_cond is the equal-weight mean over those bins.
  * This is the definition the original code implements. The preregistration text of `docs/ALANINE_1D_CV.md` §5 says
    "≥ 50 walkers per CV bin", but the published numbers come from the code, which uses ≥ 200 samples per pooled bin.
  * The multinomial sampling floor is exported alongside.
* **Paired statistics.** These are the original ones (`analyze_fr_start_timing.paired_stats`): the per-seed relative
  change (FR − ABF)/ABF, its median, a BCa bootstrap CI and the win rate.
  * Own window: [1, 100] ps for the integrated errors.
  * D_cond change: the paired change on the [50, 100] window.

## Scripts (`scripts/`)

| script | role |
|---|---|
| `common_ala.py` | paths (`ABFFR_DATA_ROOT`), stage list |
| `reference_1d.py` | bootstrap propagation to the 1-D references and the mean-force reference uncertainty |
| `export_alanine_cv.py` | all figure CSVs, per-seed files and `meta/export.json`, using the original `analyze_alanine_1d` functions |
| `plot_previews.py` | preview PDFs |
| `validate_exports.py` | validation report |
| `build_package.py` | manifest and ZIP |

The analysis code was imported from the main checkout at commit 13e77275, with unmodified tracked files.
