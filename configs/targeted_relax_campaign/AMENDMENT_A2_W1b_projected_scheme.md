# Amendment A2 (recorded 2026-09-03 00:30 UTC, after W1 closed with STOP, before W1b starts)

**Frozen W1 outcome:** `NO_COMPUTE_EFFICIENT_FR_RELAXATION` (commit 7ef04e5). Every relaxed arm is *worse* than its
unrelaxed partner, monotonically in ρ (F_ρ vs F +13.0 / +15.5 / +31.1 % integrated, 0/4; final +34 / +72 / +119 %),
with the positive control replicating (F vs A −18.1 %, 4/4). The extra error sits at z < 0.25 (mean-force error
−0.58 for abf_targ1 vs −0.26 for abf at raw bins).

**Diagnosis (post hoc, `results/targeted_relax_campaign/wca/W1/diagnostic/operator_consistency.json`):** at fixed z,
the frozen-dimer scheme at dt = 2e-3 gives a mean force −1.5 to −2.1 below the reference in the compact region and
+0.8 above at the trough; the reference's own scheme (all particles move, dimer re-projected) at dt = 2e-3
reproduces the reference; and at dt = 5e-4 the two schemes AGREE with each other (e.g. z = 0.012: −3.29 vs −3.08
relative to the reference; z = 0.176: −2.05 vs −1.90). The operators share a continuum limit; the accepted reference
(and the outer ABF dynamics, which converges to it) carry an O(dt) discretisation bias of order 3 in the compact
mean force; the frozen-dimer inner steps sit closer to the continuum and are therefore scored as error. W1's harm is
an operator-consistency confound: the inner operator's discretised stationary law differed from the outer's.

**Amendment:** a new stage **W1b** identical to W1 except that the inner relaxation uses the reference's scheme
(`RelaxConfig.scheme = 'projected'`: every particle moves one Euler–Maruyama step at the outer dt, then the dimer is
re-projected to its z), which reproduces the reference at dt = 2e-3 at every diagnostic site. Arms
`abf_ptarg{ρ}`, `fr_ptarg{ρ}` for ρ ∈ {0.25, 0.5, 1}, same seeds 820–823, same τ map, same budget accounting,
same read-out rule and ρ\* rule, paired against the existing W1 `abf` / `fr_uniform` runs. The frozen-dimer W1 result
stands as recorded; W1b is reported alongside it, labelled as an amendment.

**Prediction recorded now:** with a consistent operator the harm should disappear; on a flat sensitivity field with a
five-step force-correlation time the targeted relaxation will most likely be neutral (|ΔI_F(F_ρ, F)| < 10 %) and fail
the compute gate, i.e. STOP again — that is the honest test of whether the gateway mechanism exists in this solvent.

**Also recorded:** the dt = 2e-3 discretisation bias of the accepted WCA reference (~3 in F′ at z ≈ 0, ~0.6 kT in the
compact well depth) does not affect any of the campaign's *relative* comparisons (every arm and the reference share
it), but it is a benchmark caveat the paper must state, and any future operator with a different discretisation must
be validated against the reference's scheme first (this stage's lesson).
