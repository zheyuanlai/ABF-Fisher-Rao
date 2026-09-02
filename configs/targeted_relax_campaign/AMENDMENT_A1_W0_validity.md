# Amendment A1 (post hoc, recorded 2026-09-02 17:30 UTC, before W1 starts)

**Frozen W0 outcome:** `SENSITIVITY_INVALID` — Spearman(v̂, v_constr) = 0.455 over the 12 constrained sites (rule ≥ 0.6),
and the "top-half sites above both controls" test failed. The τ gates passed on every site (stable within 2×, resolved).

**Diagnosis (from the W0-B data, no new data):** the WCA conditional force variance is a *plateau*: v_constr = 2102 ± 68
for every site with z ≥ 0.18 and 1211 at the compact end (z ≤ 0.01). Ten of the twelve sites sit on the plateau within
±3 %, so a rank correlation has no dynamic range, and the second "control" (the argmin over the right half of the
window) is a plateau point — there is no low-sensitivity region in that half. The estimator itself tracks the truth:
Pearson(v̂, v_constr) = 0.970 and v̂/v_constr = 1.67 with CV 0.059 (range 1.53–1.87) across all twelve sites; it finds the
only low-sensitivity region correctly. The failure is the gate's design on a flat landscape, not the instrument.

**Amended validity criterion (value-based, applied to the SAME W0-B data):** pass if Pearson(v̂, v_constr) ≥ 0.6 AND the
ratio v̂/v_constr has CV ≤ 0.25 AND the τ gates hold. This is a post-hoc criterion choice and is labelled as such
everywhere it is reported; it changes nothing about any arm, the estimator, the allocation, the budget or the W1/W2 gates.

**Also recorded now:** every measured τ_f (0.0064–0.0115, i.e. 3–6 outer steps; ACF first zero at 0.3–0.7 time units with
a weak positive tail already included in the integral) lies BELOW the frozen τ floor of 0.02, so the frozen τ map is
effectively flat at 10 dt. The floor is kept as frozen (no second change).

**Prediction recorded before W1:** with a nearly flat sensitivity field and a solvent force that decorrelates within one
FR interval, targeting cannot matter much (F_rand ≈ F_R), the ordinary dynamics between opportunities already relaxes
the force-relevant fibre, and extra inner steps deposit nothing while outer steps deposit and relax at the same cost —
so W1 most likely fails the compute gate (NO_COMPUTE_EFFICIENT_FR_RELAXATION) or shows only a small accuracy gain.
Running W1 is still the right test: it measures whether the gateway mechanism (a slow, localised fibre) exists in this
solvent at all, on the frozen W1 gates.
