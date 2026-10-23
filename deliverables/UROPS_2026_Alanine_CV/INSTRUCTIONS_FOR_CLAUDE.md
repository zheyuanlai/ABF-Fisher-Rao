# Instructions for the manuscript-writing Claude (alanine, incomplete CV)

This package provides data only, from existing runs. Do not change the numbers. Definitions are in `PROVENANCE.md`;
quotable statistics are in `RESULTS_SUMMARY.md`.

## Figure A: 2 × 3 convergence figure (the `\ConvergenceFigure` CSVs)

* **Files.** Copy `figs/data/{phi,psi}_{eF,eFp,exi}.csv` into `UROPS-report/figs/data/`.
  * The schema is `t,abf_median,abf_q25,abf_q75,fr_median,fr_q25,fr_q75`.
  * Each file has 101 saved checkpoints (every 1 ps), and all values are finite and positive.
  * Row 1 is φ alone, row 2 is ψ alone. The columns are e_F, e_F′ and e_ξ.
  * The primary pair is histogram ABF vs ABF + uniform FR at rate 0.02. The rate-0.15 pair is in
    `figs/data_supplementary/*_rate015.csv`.
* **Required caption content.**
  * "Alanine dipeptide in vacuum, one dihedral biased; median and interquartile range over 16 paired seeds
    (N = 2048, 100 ps)."
  * "Errors are equilibrium-weighted and measured against the full one-dimensional marginal of the two-dimensional
    MBAR reference, on its 8 kT mask."
  * "FR first acts at 0.5 ps; the ABF bias ramp ends at 5 ps."
  * Panel (c) is the **instantaneous** walker marginal against the uniform density 1/(2π). Its finite-N floor is
    0.086.
  * Mean-force panels:
    * The reference mean force is a central difference of the reference free energy, not an exact reference.
    * For φ, its uncertainty (1.26 kJ/mol/rad) is about the size of the endpoint error. After about 10 ps the φ
      mean-force curve is reference-limited, so do not compare the arms on it.
    * For ψ, the reference is adequate (uncertainty 1.70 vs error 4.0).
  * Mention the free-energy reference-noise floor of 0.20 kJ/mol for φ. The φ endpoint, 0.374, is about 30 %
    reference noise in squared terms.

## Figure B: ψ-only hidden-coordinate diagnostic

Source files are in `figs/data_supplementary/`; the preview is `figs/previews/psi_hidden_coordinate.pdf`.

1. **`psi_hidden_pmf.csv`:** the learned F(ψ) at 100 ps (median and IQR, aligned per seed to the full reference),
   with two references: the full marginal and the φ < 0 conditional. The learned profile lies on the conditional
   profile and is about 9 kJ/mol too high between ψ ≈ −130° and −50°.
2. **`psi_hidden_mass_phipos_instantaneous.csv`:** the fraction of walkers with φ > 0 at each save, an instantaneous
   value (median and IQR).
   * The target is **0.321**, the reference p(φ > 0 | ψ) averaged uniformly over ψ, which is what a flat-ψ bias
     produces.
   * **Do not use 0.032**, the unbiased Gibbs population, as the target. If shown, label it "unbiased population".
   * The window-pooled values used in the original study are in `psi_hidden_mass_phipos_windows.csv`.
3. **`psi_hidden_dcond_windows.csv`:** D_cond per window, [1, 5], [5, 20], [20, 50] and [50, 100] ps, with the
   multinomial floor.
   * Each value pools the joint walker histogram over a window. Plot one value per window, never as an
     instantaneous time series.
   * Definition: the equal-weight mean TV of p(φ | ψ) against the reference, over pooled 15° bins with ≥ 200
     samples in the window.

## Ramachandran surface (data/reference/reference_F2_kJ.csv)

* **Orientation and units.** Rows are φ, columns are ψ, on the grid −π + (i + ½)·2π/97. Values are in kJ/mol with
  the global minimum at 0; `NaN` marks the 226 never-visited cells.
* **Edge profiles.** The marginals computed from this file are exactly the panel references for F_φ and F_ψ.
* **Basin labels.** Use `reference_F2_minima.csv`. The surface's own resolved minima are:
  * C7eq at (−74°, 56°);
  * C5 at (−152°, 156°), +1.1 kJ/mol;
  * C7ax at (63°, −48°), +6.4 kJ/mol.
* **αR is not a resolved minimum here.** Its region near (−56°, −19°) is a 12 kJ/mol shoulder with 0.12 kJ/mol
  prominence. Label it as a region, or omit it.
* **Uncertainty.** The per-cell bootstrap s.e. is in `reference_F2_se_kJ.csv` (median 0.61 kJ/mol). Contours finer
  than about 1 kJ/mol are below the reference noise.

## Claims supported, and claims to avoid

* **Supported:**
  * φ alone works: 0.374 kJ/mol, C7ax first reached at 2.3–3.5 ps in all 16 seeds.
  * ψ alone converges to the φ < 0 conditional: 2.070 vs 0.514 kJ/mol.
  * The hidden φ is not equilibrated by 100 ps: φ > 0 mass 0.13–0.16 vs the 0.321 target; D_cond 0.25–0.32.
  * Gentle FR is neutral on F: φ −1.34 % integrated (endpoint +5.7 %); ψ +0.24 %.
  * Strong FR is harmful on φ (+18.9 % integrated, +67 % final) and on ψ degrades the hidden conditional
    (D_cond +14.5 %, φ > 0 mass 0.064 vs 0.129).
* **Avoid:**
  * Do not equate C7ax being "visited in all seeds" with equilibration.
  * Do not claim FR improves the φ mean force: that panel is reference-limited and the closed variant reverses the
    sign.
  * Do not call the strong-FR mechanism ("kills where the bias lags") demonstrated: it is the original study's
    hypothesis.
  * Do not present the 2 × 3 figure as an accuracy gain from FR's flatter marginal.
