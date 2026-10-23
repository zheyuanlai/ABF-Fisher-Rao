# Instructions for the manuscript-writing Claude

This package supplies **data only**. Do not change the numbers. The definitions behind them are in
`FIGURE_PROVENANCE.md` and `FIGURE_DATA.md`.

## Integration

1. Copy the 21 files `figs/data/*.csv` into `UROPS-report/figs/data/`. They replace any earlier copies, including
   the four LTA files of the draft.
2. The names match the `\ConvergenceFigure` interface: `<fig>_eF.csv`, `<fig>_eFp.csv`, `<fig>_exi.csv` for `<fig>`
   in gateway, lta150, lta300, ite300, meta, alanine, wca.
3. The columns are `t,abf_median,abf_q25,abf_q75,fr_median,fr_q25,fr_q75`. Plot the median with an IQR band. The
   band is the 25–75 % range across seeds, **not** a confidence interval.
4. All values are finite and strictly positive, so log y axes are safe.
5. Put FR on / off markers at the times in the table of `FIGURE_DATA.md`. For ITE, mark both t = 1 (on) and t = 10
   (off).
6. Quote statistics from `RESULTS_SUMMARY.md`. Always distinguish integrated from final error, and use each
   experiment's own integration window.

## Caption qualifications (required)

* **All figures.** Write "Median over N paired seeds; shading = interquartile range across seeds." Seed counts:
  gateway 32, meta 32, the others 16. Both arms share initial conditions and Langevin noise. Panel (c) is the
  **instantaneous** marginal of the current replicas, never interval-averaged.
* **Gateway.**
  * Validated timestep h = 2.5 × 10⁻⁵ against the exact analytic F.
  * Ī_F −31.4 % [−34.7, −27.7] (31/32); e_F(T) −61.4 %.
  * The often-quoted −29.7 % comes from an independent dt-study run set (seeds 7100–7131). Either may be cited if it
    is labelled.
  * (b) has a hard piecewise-constant floor 0.0323.
  * In (c) FR reaches the finite-N floor 0.156 by t ≈ 6.
  * It is a constructed toy.
* **LTA 150 / 300 K.**
  * Exact Metropolis MC reference.
  * I_F −28.5 % (150 K) and −13.7 % (300 K).
  * The **300 K final-error change (−7.9 % [−14.6, +0.7]) is not resolved**. Do not claim an endpoint benefit at
    300 K.
  * Panel (c) is a **KDE-smoothed** (0.1 rad) instantaneous marginal, the only marginal the engine saved. Say so.
* **ITE 300 K.**
  * Confirmatory block E1: FR active only on 1 ≤ t < 10, then plain ABF.
  * S_geo 1.59 [1.31, 1.85]; post-ramp Ī_F −38 %; e_F(T) −32 %.
  * Do **not** present it together with continuous FR as one three-arm experiment: they are different seed blocks.
  * In (c) both arms sit at the finite-N histogram floor from t ≈ 4–5.
* **Energetic metastability (meta).** This is a **new** experiment with a frozen protocol (preregistered timestep
  gate, ABF-only resolution choice, safety-only FR rate) and 32 fresh seeds.
  * **FR is harmful**: I_F +51.9 % [+40.2, +60.7], 0/32; e_F(T) +334 %.
  * FR flattens the marginal faster (panel c), but the free-energy error ends ≈ 4× higher.
  * A diagnostic locates the extra error in the barrier bins, where the hidden y-conditional is bimodal. Present
    this as a diagnostic, not a demonstrated mechanism.
  * It does **not** reproduce the archived kernel-ABF result, which used 5 seeds and outcome-selected FR settings.
  * No temperature sweep was run. Do not claim a numerically demonstrated pathway switch.
* **Alanine.**
  * Equilibrium-weighted error on the 8 kT mask against the raw reference.
  * **Neutral** (dI_F +0.04 % [−1.36, +0.64]); the curves overlap almost exactly.
  * e_F ends at the reference-noise floor 0.56 kJ/mol. e_F′ is reference-noise limited.
  * (c) uses the full-torus uniform target and reaches the finite-N floor 0.341 (N = 2048 on 97² cells) by about
    20 ps.
  * The steric floor is real but small (0.025 for the reference's finite support) and is **not resolvable at this
    N**. Do not attribute the (c) plateau to steric exclusion.
  * The arms are statistically paired (same initial ensemble and noise stream), not bitwise, because of GPU
    reduction nondeterminism.
* **WCA.**
  * Validated dt* = 1.25 × 10⁻⁴, exact MC reference: a **validated negative**.
  * I_F +9.9 % [+2.3, +13.2] (4/16 FR wins); e_F(T) +131 %.
  * The earlier positive WCA figures (dt = 0.002) are invalid and must not be used.
  * Only 49 checkpoints were saved.
  * The spike at t = 20 is the estimator burn-in reset.
  * Panel (c) is dominated by the ≈ 16 % of replicas held in the soft-wall tails, which are counted in the two edge
    bins. The interior-only version (`figs/data_supplementary/wca_exi_interior.csv`) sits at the finite-N floor.
* **Equal cost.** These figures compare at equal N and T. Do not describe them as compute-equal allocation
  comparisons (see `RESULTS_SUMMARY.md`, last section).

## Do not

* Do not recompute curves from percentages, or use the preview PDFs as data.
* Do not combine experiments, or the matched-sham arms (historical, not exported).
* Do not describe panel (c) as interval-averaged.
