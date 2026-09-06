# Ethane / flexible ZIF-8: Wasserstein reallocation + constrained gate repair — preregistration

**Frozen 2026-09-06 12:40 UTC, at the launch of the full Z1–Z3 run (a 0.3-min smoke run with 16
unequilibrated replicas per site had been executed to test the code; its numbers informed the
predictions below and are labelled as such).**  GPU 1 only.  Design source: the reviewer's plan
Z1–Z5 (WCA → pentane R15 → ethane/ZIF-8).  Parents: `docs/WCA_OT_CONFIRMATORY_M3.md` (OT+R wins on a
repairable bath fibre), `docs/PENTANE_R15_OT_REPAIR.md` (OT wins, repair is pure cost on a frozen
torsional fibre; τ⊥ ≈ 10⁶–10⁷ steps), `configs/uniform_campaign/zif8_prereg.json` and
`docs/ZIF8_CORRECTED_BASELINE.md` (the ZIF-8 model, reference and corrected ABF baseline).

## Question

WCA's fibre (solvent shell, τ_f 3–6 steps) was repairable and repair paid; pentane's fibre (torsional
families) was frozen and repair was pure cost.  ZIF-8's hidden fibre is a **real flexible molecular
gate**: the six-ring aperture A_gate (mean radius of the six bottleneck ring-H about the Zn₆ centroid)
and the guest orientation.  **If Wasserstein transport moves ethane along the channel axis, does the
gate lag, how large is the mean-force bias that lag injects, and how many constrained steps does it
take the gate to re-adapt?**  τ_gate(ξ) decides the regime: A (WCA-like, 10–300 steps, affordable),
B (pentane-like, ≫ 10⁴ steps or an irreparable residual), C (instantaneous, OT alone suffices).

## System, reference, operators (frozen; nothing in `core_zif8` changes)

* Model, cell, CV, thermostat, dt = 0.5 fs, γ = 1/ps, T = 300 K, force kernel f32 / positions f64,
  exactly the corrected-baseline production engine.  φ = wrap(2πξ/L), ξ = (COM_guest − win_centre)·n,
  L = 14.2953 Å, one ABF grid bin = 0.1489 Å (n_grid 96).  Determinism flags OFF for Z1–Z3 (no
  pairing needed; 1.75× faster) — they stay ON for any Z5 arm.
* Reference: the accepted umbrella/WHAM reference `results/uniform_campaign/zif8/reference/reference_T300.npz`
  (64 windows × 64 replicas × 150 ps; barrier peak 39.30 kJ/mol = 15.8 kT at ξ = +0.97 Å; cage minimum
  at ξ = +6.03 Å; split-half F′ RMS 0.16 kJ/mol/Å) — F′_ref(ξ) by periodic central differences,
  and p_ref(A_gate | ξ sub-bin) for the eight 0.25 Å sub-bins of |ξ| < 1 Å.
* **Lift** (`ot_repair_zif8.lift_guest`): translate the whole ethane by (ξ′ − ξ) n — bond, orientation,
  every framework atom and every velocity untouched (exact, linear CV).
* **Repair** (`ConstrainedBAOAB`): one BAOAB step of the outer dynamics (physical force only,
  framework-COM pinned), then the guest COM re-projected along n to ξ′ and its COM velocity component
  along n removed.  Linear constraint ⇒ the projected measure is the exact conditional p(·|ξ′) with the
  tangent-space Maxwellian; framework, aperture, guest orientation and bond all move.  Nothing deposited;
  one force evaluation per inner step.  Tests: `tests/test_zif8_ot_repair.py` (4).

## Z1 — fixed-ξ operator validation (`scripts/zif8_ot_z123.py`)

Six sites chosen mechanically from the reference F(ξ): **cage minimum** (argmin F), **left half-height**
(F = F_min + ΔF/2 on the approach side), **left band sub-bin** [−0.50, −0.25), **window plane**
[−0.25, +0.25) (two central sub-bins), **peak sub-bin** [+0.75, +1.00) (contains argmax F), **right
half-height**.  Point sites are one grid bin wide.  Each of 256 replicas per site keeps its own ξ′
drawn uniformly in the site interval (like-for-like with the reference sub-bin).  Initial states: the
init pool's configurations nearest in circular ξ, lattice-shifted (exact symmetry), pulled to ξ′ over
4 000 steps (≤ 1.3 Å/ps), equilibrated 40 000 steps (20 ps), recorded 40 000 steps.
Outputs: b_inf = ⟨f_ξ⟩ − ⟨F′_ref(ξ′)⟩ with SE over replicas, first/second-half drift, ⟨A_gate⟩, sd,
⟨θ_gate⟩, TV of the recorded A_gate histogram (96 bins) against the reference sub-bin (band sites) and
between halves (all sites), integrated autocorrelation time τ_A of A_gate at fixed ξ (last 8 000 steps).

**Operator gate (frozen):** mean force |b_inf| ≤ max(1.0 kJ/mol/Å, 3 SE) at every site AND
|half-drift| ≤ 2 SE; conditional TV(rec, ref) ≤ TV(halves) + 0.05 and |⟨A⟩ − ⟨A⟩_ref| ≤ 0.02 Å at the
three band sites.  **Hard stop** if |⟨A⟩ − ⟨A⟩_ref| > 0.10 Å at any band site after full
equilibration (the constrained operator would then be inconsistent with the umbrella reference).
Mean-force offsets that pass are the operator's own signature and are subtracted in Z2 (WCA M1 rule).

## Z2 — single OT event

From the Z1-equilibrated ensembles, lift Δξ ∈ ±{½, 1, 2, 4, 8} bins (0.074–1.19 Å; the campaign cap
will be 2 bins = 0.30 Å), then repair at fixed ξ′ for M = 400 steps (200 fs) recording every step
b(m) = ⟨f_ξ(m)⟩ − F′_ref(ξ′) and ⟨A_gate⟩(m), sd; then 3 600 tail steps and a 2 000-step stationary
window → b_inf(ξ′), A_inf, sd_inf, and the coarse-bin (0.1 Å) gate law for D_gate(m) at
m ∈ {0, 5, 20, 100, 400} against the stationary law (its 256-sample floor is reported).  Lift audit:
ΔU, max guest force, minimum host–guest distance.  Fractions remaining
(b(m) − b_inf)/(b(0) − b_inf) and the same for ⟨A_gate⟩; times to 20 % / 10 % and the e-fold.

## Z3 — τ_gate(ξ)

Three readings, all reported in steps and fs: τ_A at fixed ξ (Z1 autocorrelation), the ⟨A_gate⟩
relaxation time after a 2-bin lift (Z2), the mean-force relaxation time after a 2-bin lift (Z2).
**Regime rule (frozen):** with τ* = the median over flank/band sites of the 2-bin mean-force 20 %-time,
and inj = median |b(0) − b_inf| / |F′_ref| at the 2-bin lift over the flank/band sites:
A (WCA-like, repair meaningful) if 10 ≤ τ* ≤ 300 steps and inj ≥ 0.2; B (pentane-like) if
τ* > 10⁴ steps or an irreparable residual |b_inf(ξ′) − b_inf,site(ξ′)| > 0.2 |F′| persists after the
tail; C (OT only) if τ* < 10 steps or inj < 0.1.  Affordability is then a number, not a judgement:
c·τ* inner steps per opportunity against the 5-step opportunity cadence.

## Z4 / Z5 (outline; not run in this round without a separate freeze)

Z4: ABF-only regime screen (temperature and/or budget ladder, screen classifier of the ZIF-8 prereg,
per-bin adequacy diagnostics) to find a cell with an establishment deficit; the cell is chosen from
ABF-only diagnostics, never from where OT looks best.  Z5: A/F/T/R/F+R/T+R on the chosen cell with
the corrected baseline (h_bias 0.10 Å, h_read 0.05 Å); the repair dose per opportunity is set from
τ* (Z3) with the cadence chosen so that the inner/outer force-evaluation ratio stays ≤ 1.  A
production arm at the corrected-baseline scale costs ~9.5 GPU-hours on GPU 1; the six-arm design
will need a reduced scale, agreed with the user.

## Predictions (recorded now; informed by the 0.3-min smoke run)

* Z1 passes: constrained BAOAB reproduces the umbrella gate conditional (the smoke run's
  ⟨A_gate⟩ 3.00 vs 2.86 Å at the window plane is an unequilibrated pull artefact and will close).
* τ_A at fixed ξ ≈ 20–60 fs (40–120 steps); the mean-force relaxation after a lift is slower,
  50–200 fs, because guest orientation/lateral position relax with the gate: **regime A but not
  cheap** — 5 inner steps remove < 20 % of a 2-bin injection; ~τ* ≈ 100–300 inner steps are needed.
* Injection at the 2-bin cap is of the order of F′ itself (smoke: +13.6 vs F′ 16 at the left band,
  −49 vs −26 at the right flank), i.e. inj ≈ 0.5–2 — ten times pentane's relative injection.  Hence,
  unlike pentane, unrepaired OT at a 2-bin cap should visibly bias the mean force on the flanks;
  whether that hurts e_F(T) at the corrected baseline (already 0.084 kJ/mol) is the Z5 question.
* Every number is reported regardless of sign; nothing is tuned after data.
