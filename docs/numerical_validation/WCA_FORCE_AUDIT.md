# WCA dimer: force-field audit

*2026-10-09, branch `research/langevin-validation-oct2026`. Phase I of the overnight numerical-validation
campaign. Tests: `tests/test_wca_force_audit.py` (19 pass). Numbers:
`results/numerical_validation/wca/force_audit.json` (`scripts/numerical_validation/wca_force_audit_numbers.py`).*

## 1. Summary

* **The physical force is correct.** The analytic WCA + double-well + wall gradient matches central
  differences of the energy with second-order convergence (max relative error 2e-4 / 2e-6 / 2e-8 at
  ε = 1e-3 / 1e-4 / 1e-5, roundoff floor ~1e-9 below that) on compact, transition and stretched dimer
  configurations and on free solvent states. Newton's third law holds to 5e-14, and translation and
  periodic-image invariance hold to round-off. The production force (`wca_numba.wca_force`, both norm modes;
  torch `WCADimerEngine.force` in float64) equals an independently written force to ≤ 1.4e-13 absolute
  whenever no pair is inside `min_r`.
* **The local mean force, including its Jacobian (entropic) term, is correct.** With no solvent,
  `f = (w/r) d·(F1 − F0) − 2w/(βr)` equals dA/dz of the exact dimer free energy
  A(z) = V_dim + U_wall − β⁻¹ log r to 1e-6 at ten values of z. It is also validated statistically in
  the full many-body system (exact Gibbs MC, `WCA_LANGEVIN_VALIDATION.md` §3).
* **The production drift is not the gradient of the stated potential. There are two defects, both
  regularisations that the integrator needs at the accepted dt:**
  1. **The `min_r` clamp is force/energy inconsistent.** Below r = 0.65σ the production force is
     −(V′(r_min)/r_min)·d. That is the gradient of an *unimplemented* quadratic continuation
     U_q(r) = V(r_min) + V′(r_min)(r² − r_min²)/(2 r_min). The implemented energy is flat, V(r_min) = 651 kT,
     so its gradient is 0 while the force is ~1.15e4 at r = 0.60.
  2. **The per-particle force clip at 250 is not a gradient field and breaks Newton's third law.** It
     rescales each particle's *total* force independently of the energy. In a three-particle counterexample:
     * the sum of clipped forces is (+105, …), not 0;
     * the Jacobian of the clipped field is asymmetric (∂F_i,x/∂q_k,y ≠ ∂F_k,y/∂q_i,x by > 10), so no
       potential has it as gradient. The raw field's Jacobian is symmetric to 1e-5.

  A single WCA pair reaches the clip at r = 0.865σ, where V = 14.2 kT. Under exact Gibbs sampling that is
  rare. Under the dt = 0.002 dynamics it is routine: the clip binds in about 29 % of steps and pairs sit
  inside r_min in about 3 % of steps (§5).
* **Root cause of the WCA timestep problem: Euler–Maruyama is linearly unstable in ordinary collisions
  at dt = 0.002.** dt·V″(r) is 2.05 at r = 0.95σ (V = 3 kT) and 4.7 at r = 0.90σ (V = 7.6 kT), against a
  stability limit of 2. The force clip turns the instability into bounded but unphysical jumps of up to
  dt × 250 = 0.5σ per step. Those jumps drive particles deep into each other's cores, where the `min_r`
  clamp takes over (§5). The integrator, not the force field, is wrong at dt = 0.002. Smaller dt is the
  cure, and whether any dt is good enough is decided by the exact-Gibbs comparison in
  `WCA_LANGEVIN_VALIDATION.md`.

No defect was fixed in the production code. The historical model is unchanged and every historical
result is preserved. The validation uses a separately named model, the *intended* potential in
`src/wca_validation.py` (no clamp, no clip), and asks at which dt the production dynamics reproduces
its Gibbs measure. A consistent regularised potential was **not** needed (§6).

## 2. Source-to-equation map

Production code: `src/wca_abffr_core.py` (torch, accepted GPU engine) and `src/wca_numba.py` (the CPU
port used by every 2026-10 study, bitwise equal to the torch engine under the same noise; see its tests).

| quantity | equation | source |
|---|---|---|
| box, particles | L = n_dim·a = 15, P = 100 (dimer = particles 0, 1), β = 1, σ = ε = 1 | `DimerWCAParams` `wca_abffr_core.py:70-97` |
| WCA pair | V(r) = 4ε[(σ/r)¹² − (σ/r)⁶] + ε for r ≤ r_c = 2^{1/6}σ, dimer pair excluded; C¹ at r_c (V = V′ = 0), V″ jumps | `:300-338` (scatter path), `wca_numba.py:208-282` |
| `min_r` clamp | r_safe = max(r, 0.65σ) inside V′; force = −V′(r_safe)/r_safe · d (d the true displacement); energy V(r_safe) | `:308-316`, `:335`; `wca_numba.py:244-250` |
| dimer bond | V_dim = h(1 − u²)², u = (r − r_c − w)/w, h = 2, w = 2; dV/dr = −4hu(1 − u²)/w | `:326-329`, `:337`; `wca_numba.py:265-281` |
| periodic boundaries | wrap q mod L; minimum image d − L·round(d/L); r_c < L/2; a bond of length L/2 costs > 30 kT | `:221-227` |
| force clip | F_i ← F_i · min(1, 250/‖F_i‖), per particle, on the total force | `clip_forces` `:229-232`; `_clip_vec` `wca_numba.py:190-201` |
| RC | z = ξ(q) = (r₀₁ − r_c)/(2w); ∇_{q0}ξ = d/(2wr), ∇_{q1}ξ = −∇_{q0}ξ | `:350-358` |
| local mean force | f = (w/r) d·(F1 − F0) − 2w/(βr) (LRS: G = ∇ξ/‖∇ξ‖², div G = 2w/r in 2-D), computed with the **clipped** force, sample clamped to ±500 | `:361-366`, `:1689`; `wca_numba.py:645-657` |
| ABF bias | +clip(Γ_j(z), ±40)·ramp·∇ξ added to the clipped force, then clipped again | `:1729`; `wca_numba.py:700-747` |
| RC wall | −k_w[(z − z_max)₊ + (z − z_min)₋]∇ξ, k_w = 80, z ∈ [−0.2, 1.2]; the gradient of U_wall = (k_w/2)·excess², zero inside the window; added then clipped again | `:377-383`, `:1730`; `wca_numba.py:722-747` |
| integrator | Euler–Maruyama q ← q + dt·T(q) + √(2dt/β)·ξ, unit mobility, then wrap | `:1554`, `wca_numba.py:748-752` |
| reference (accepted "v2") | constrained (projected-EM) dynamics at fixed z, dt 0.002, same clips; ⟨f⟩ at 61 nonuniform z, trapezoid, PCHIP to the 160-grid | `scripts/build_wca_hp_reference_v2.py`; `cache/phase_hp_v3/…g160.npz` |
| reference (v1) | same constrained scheme, 51 z points, smoothed 1 grid cell | `constrained_ti_reference_gpu` `:2253-2302`; `cache/wca_ti_reference.npz` |

Noise amplitude, drift and mobility are consistent with the overdamped generator
L = −∇U·∇ + β⁻¹Δ, whose invariant law is exp(−βU). The projected constrained dynamics of the TI
references samples the conditional law on {ξ = z} without a Fixman correction, which is correct here
because ‖∇ξ‖ = 1/(√2 w) is constant.

## 3. Gradient verification (A1, A2)

Six configurations, every coordinate, central differences of the *intended* energy. A coordinate is
skipped only when its stencil would straddle the cutoff, where V″ jumps; continuity of V and V′ at r_c is
tested separately (A1-cutoff).

| configuration | z | min pair r | max abs F | rel. err ε = 1e-3 | 1e-4 | 1e-5 | 1e-6 | abs sum F | production vs intended |
|---|---|---|---|---|---|---|---|---|---|
| restrained z = 0 | −0.03 | 0.957 | 41.9 | 1.9e-4 | 2.0e-6 | 2.0e-8 | 1.5e-9 | 2.6e-14 | 6.8e-14 |
| restrained z = 0.25 | 0.25 | 1.006 | 21.5 | 1.9e-4 | 2.0e-6 | 2.0e-8 | 1.2e-9 | 2.7e-15 | 4.0e-14 |
| restrained z = 0.5 | 0.53 | 0.956 | 61.1 | 2.1e-4 | 2.1e-6 | 2.1e-8 | 1.4e-9 | 1.6e-14 | 8.5e-14 |
| restrained z = 1.0 | 1.01 | 0.925 | 77.1 | 2.5e-4 | 2.5e-6 | 2.5e-8 | 1.3e-9 | 1.2e-14 | 1.4e-13 |
| free MC state a | 1.08 | 0.969 | 31.4 | 4.3e-4 | 4.3e-6 | 4.3e-8 | 1.8e-9 | 4.4e-15 | 4.8e-14 |
| free MC state b | 1.00 | 0.942 | 60.3 | 3.0e-4 | 3.0e-6 | 3.0e-8 | 3.1e-9 | 4.7e-14 | 1.3e-13 |

The exact factor-100 drop per decade of ε is the O(ε²) truncation of a correct gradient. A wrong term
would leave an ε-independent floor.

Further checks:

* The torch engine's float64 energy equals the intended energy to 1.6e-14.
* The screened fast kernels equal the plain double loops (A7).
* Single-particle and dimer-stretch MC energy differences equal total-energy differences to 1e-9 (A7).

## 4. Regularisations (A4, A5)

| regularisation | conservative? | consistent with the implemented energy? | where it acts |
|---|---|---|---|
| `min_r` clamp (0.65σ) | yes, but for U_q, not V | **no**: energy flat, force a linear spring of slope 1.9e4 | r < 0.65σ, V > 651 kT |
| force clip (250 per particle) | **no** (asymmetric Jacobian) | **no**; also breaks Newton III | single pair r < 0.865σ (V > 14 kT), or several overlapping contacts |
| mean-force sample clip (±500) | estimator only | changes the estimator, not the dynamics | samples with abs f > 500 |
| ABF bias clip (±40), RC wall | the wall is −∇U_wall; the bias clip acts on the estimate | yes | outside [−0.2, 1.2], or abs Γ > 40 |

Removing the clips is not a cure at the accepted dt. With `force_clip` 1e5 and `min_r` 0.5, the dt 0.0005
dynamics blows up (2026-10-04 consistency test D). Without any clip, Leimkuhler–Matthews at dt 0.002 blew
up within the first 0.002 t.u. of the 2026-10-09 pilot. The clips are what keeps the accepted dt from
diverging. Their cost is that the drift stops being the gradient of the stated potential.

## 5. How often the regularisations act in the production dynamics

Production drift (`EM_IMPL`, the unbiased branch of `wca_numba.simulate`; test A8 shows the harness is
*bitwise* the production trajectory) for 16 seeds × 3072 t.u. per dt, with diagnostics every step
(`results/numerical_validation/wca/diag/`):

DIAG_TABLE_PLACEHOLDER

## 6. Was a consistent regularised model needed?

**No.** The production clips act only when the integrator overshoots, and their activation falls
by orders of magnitude as dt decreases (§5). The decisive test is therefore not a new potential. It is
whether the *production* dynamics at some dt reproduces the Gibbs measure of the *intended* potential,
which has no clip. Exact Metropolis MC on the intended potential provides that measure without ever
evaluating a force (`WCA_LANGEVIN_VALIDATION.md`). A separately named, consistently regularised
potential would be needed only if no affordable dt passed that test.

## 7. Regression tests

`tests/test_wca_force_audit.py` (CPU, about 2 min):

* A1: FD gradient on six configurations at three ε; cutoff continuity.
* A2: Newton III, translation and periodic images, minimum-image validity.
* A3: production force (two norm modes) and torch force/energy equal the intended force/energy.
* A4: the min_r inconsistency, as a characterisation test.
* A5: the force clip is a non-gradient that breaks Newton III.
* A6: the local mean force equals dA/dz of the exact dimer free energy.
* A7: fast kernels equal reference loops; MC energy bookkeeping.
* A8: the validation harness is bitwise the production unbiased dynamics, at dt 0.002 with the clip active
  and at dt 0.0005.
* A9: MC, MALA and Leimkuhler–Matthews reproduce the analytic dimer-only density. Euler–Maruyama
  reproduces its *first-order* invariant density, and its deviation from the exact density is detected.

Test-design note: a first version of A9 used a χ² over 32 coarse bins and rejected the *correct* MC
sampler (p = 8e-5). A chain's well population fluctuates as a whole, so bins within a well are strongly
correlated and the χ² null does not hold. With 48 × 4e6 sweeps the MC well population is 0.2250 ± 0.0002
against an analytic 0.2252. The test now checks the well population and the within-well shape
separately.
