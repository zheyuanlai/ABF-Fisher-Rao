## Manual notes

**Data sufficiency.** All six required stages exist with 16 seeds, N = 2048 and 100 ps. Each raw file contains
everything needed, so **no simulation was launched**:

* per-save PMF and mean force;
* the instantaneous walker marginal;
* the joint (φ, ψ) walker histogram;
* basin fractions and first hits.

**Reproduction.**

* Every published number of `docs/ALANINE_1D_CV.md` is reproduced by the original `scripts/analyze_alanine_1d.py`
  functions (imported, not re-implemented), and equals `results/alanine_1d/analysis/summary.json` to 10⁻¹².
* The new metrics (e_F′, e_ξ, instantaneous φ > 0 mass) are computed from the same raw arrays. Their definitions are
  in `PROVENANCE.md`.

**Mean-force reference.**

* The reliability rule was fixed in `scripts/reference_1d.py` before any ABF e_F′ was computed: reliable if the
  reference uncertainty is < 0.5 × the median ABF final e_F′.
* Result: ψ is reliable (0.42); φ is flagged (0.93).
* For φ, two alternative diagnostics are provided:
  * the seed dispersion (reference-free precision) is 0.09 at 100 ps;
  * the noise-debiased error is 0.67, indicative only.
* The raw and closed variants of the φ e_F′ disagree in the sign of the FR-0.15 effect. The φ mean-force comparison is
  therefore not interpretable after about 10 ps.

**D_cond definition.** The doc's §5 says "≥ 50 walkers per CV bin". The implementation that produced the published
values uses ≥ 200 samples per pooled 15° bin, summed over the window. The export follows the implementation, since the
validation targets are its numbers.

**Pairing.** The arms share the initial ensemble and the noise stream, and are identical at t = 0. They differ by up
to about 4 % at t = 1 ps because CUDA reductions are nondeterministic (`docs/FR_START_TIMING.md`). The pairing is
statistical, not bitwise.

**Hidden coordinate.**

* The φ > 0 target 0.321 is the ψ-uniform average of the reference conditional. The unbiased population 0.032 is
  labelled as such and not used as a target.
* C7ax is first *visited* in 16/16 seeds in every arm (ψ: 3.4–5.6 ps). The hidden coordinate is not equilibrated:
  φ > 0 mass is 0.13–0.16 vs 0.321.

**Zeros in supplementary data.** The seed-dispersion series is exactly 0 at t = 0 (an empty, identical estimator in
every seed). The previews omit t = 0 on log axes; the CSV keeps the 0. The six primary CSVs contain no zero.

**Provenance caveat.** The six runs were made from commit ebc123bb with a dirty working tree (recorded in their
metadata). The engine source at that time is not separately archived beyond the repository history.
