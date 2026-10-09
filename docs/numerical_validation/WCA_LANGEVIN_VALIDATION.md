# WCA dimer: validation of the unbiased Langevin dynamics against exact Gibbs sampling

*2026-10-09, branch `research/langevin-validation-oct2026`. Phase II of the overnight numerical-validation
campaign.*

* Design, statistic and gates were frozen before any profile was computed:
  `configs/numerical_validation/wca_validation_prereg.json`, commit 6ec7c01.
* Runner: `scripts/numerical_validation/run_wca_validation.py`.
* Analysis: `scripts/numerical_validation/analyze_wca_validation.py`.
* Data: `results/numerical_validation/wca/` (`summary.json`, `figures/fig_wca_timestep_validation.png`).

## 1. Verdict

| gate (frozen) | result |
|---|---|
| G_F_force | **PASS**: `WCA_FORCE_AUDIT.md` |
| G_MC_self | **PASS**. The analytic dimer-only density is reproduced (MC well population 0.2250 ± 0.0002 vs 0.2252). In the full system D(MALA, MC) = 0.000 kT (upper bound 0.0059 ≤ 0.0075). |
| G_MF_formula | **PASS**. Exact-Gibbs mean-force route vs density route: D = 0.0002 kT (upper 0.0007). |
| **G_dyn: production EM** | **FAIL at dt 0.002, 0.001, 0.0005; MARGINAL at 0.00025; ADMISSIBLE at 0.000125** |
| **timestep choice** | **RESOLVED: dt* = 0.000125**, the largest admissible dt in the ladder. That is 16× smaller than the accepted 0.002. |

The WCA dynamics gate is resolved. The production integrator with its clips, at dt = 0.000125,
reproduces the exact Gibbs free energy of the intended potential along z to ≤ 0.003 kT (upper bounds
0.0074 and 0.0049). That holds for both the density route and the mean-force route ABF uses. At the
accepted dt = 0.002 it misses by 0.13 kT (density) and 0.31 kT (mean force).

## 2. The arbiters

* **Metropolis MC** (exact; energy only, never a force). 64 chains × 3e7 sweeps after 2e4 burn-in,
  each sweep = 100 single-particle moves (δ 0.4, acceptance 0.59) + 20 dimer-stretch moves with the 2-D
  Jacobian r′/r (acceptance 0.48).
  * Profile noise is 0.0006 kT.
  * ⟨U⟩ = 11.0981 ± 0.0002; P(z > 0.5) = 0.70525 ± 0.00023.
  * The tests check it against the analytic dimer-only density (`tests/test_wca_force_audit.py` A9).
* **MALA** (exact; EM proposal + Metropolis–Hastings on U, dt 2.5e-4). 32 chains × 30720 t.u.
  * Acceptance 0.870 (pilot: 0.67 at dt 5e-4, 0.96 at 1e-4). MALA is practical on the full 100-particle
    system.
  * Agrees with MC: D = 0 on both routes; ⟨U⟩ 11.0971 ± 0.0007; P(z > 0.5) 0.7054 ± 0.0018; the
    closest-approach distribution matches MC (P(r_min < 0.865) 2.4e-6 vs 2.5e-6).

The two exact samplers use different ingredients: MC uses the energy only, MALA uses energy and force.
Their agreement is the strongest evidence that the reference is the Gibbs measure of the stated
potential.

## 3. Timestep refinement (gates use the debiased RMS D vs MC; kT; 95 % bootstrap upper bound in brackets)

| scheme | dt | chains | D(F_density, MC) | D(F_MF, MC) | D(F_MF, own F_density) | blow-ups | gate |
|---|---|---|---|---|---|---|---|
| production EM (clipped) | 0.002 | 64 | 0.131 [0.134] | 0.308 [0.310] | 0.285 | 0 | FAIL |
| | 0.001 | 64 | 0.022 [0.025] | 0.060 [0.061] | 0.051 | 0 | FAIL |
| | 0.0005 | 64 | 0.0078 [0.011] | 0.020 [0.022] | 0.017 | 0 | FAIL |
| | 0.00025 | 64 | 0.0025 [0.0074] | 0.0065 [0.0078] | 0.0058 | 0 | MARGINAL |
| | **0.000125** | 64 | **0.0032 [0.0074]** | **0.0021 [0.0049]** | 0.0040 | 0 | **ADMISSIBLE** |
| Leimkuhler–Matthews (no clip) | 0.001 | 32 | — | — | — | **32** | FAIL |
| | 0.0005 | 32 | 0.0036 [0.012] (22 finite) | 0.0061 [0.013] | 0.000 | **10** | FAIL |
| | 0.00025 | 32 | 0.0030 [0.0097] | 0.0036 [0.0080] | 0.000 | 0 | MARGINAL |
| BAOAB, γ = 1 (no clip) | 0.005 | 32 | 0.000 [0.011] | 0.000 [0.0078] | 0.000 | 0 | MARGINAL (precision) |
| | 0.0025 | 32 | 0.000 [0.013] | 0.000 [0.010] | 0.008 | 0 | MARGINAL (precision) |
| | 0.00125 | 32 | 0.000 [0.012] | 0.000 [0.010] | 0.003 | 0 | MARGINAL (precision) |
| MALA (exact) | 0.00025 | 32 | 0.000 [0.0059] | 0.000 [0.0037] | 0.000 | 0 | ADMISSIBLE |

Notes:

* The 2026-10-04 values (RMS(F_MF − F_density) 0.284 / 0.051 / 0.018 at dt 0.002 / 0.001 / 0.0005) are
  reproduced by the new read-out: 0.284 / 0.050 / 0.017 on the old 32 seeds alone.
* **Routes.** The density route converges faster than the mean-force route. ABF uses the mean-force
  route, so its error is the relevant one: 0.31 → 0.060 → 0.020 → 0.0065 → 0.0021 kT, a factor of about 3
  per halving of dt.
* **Convergence order.** The apparent order of the mean-force route is about 1.6 between dt 0.001 and
  0.000125, and 2.4 between 0.002 and 0.001.
  * This is faster than the textbook weak order 1 of EM. The likely reason is that part of the error comes
    from the clip, whose activity falls by an order of magnitude per halving (`WCA_FORCE_AUDIT.md` §5).
  * The two smallest-dt D values are within 1-2× their noise, so the order there is not resolved. I do not
    claim an asymptotic order.
  * The robust statements: (i) the error is monotone in dt and falls below tolerance at 0.000125; (ii) the
    potential energy, a stiff-contact observable, shows clean first order: ⟨U⟩ − ⟨U⟩_MC = +0.61 / +0.27 /
    +0.12 kT at dt 0.0005 / 0.00025 / 0.000125.
* **Leimkuhler–Matthews.** Without a clip it is second-order accurate, with ⟨U⟩ bias 0.0006 ± 0.0007 at
  dt 0.00025. But it is not robust: 32/32 chains blew up at dt 0.001 and 10/32 at 0.0005.
* **BAOAB** (independent kinetic-Langevin check, no clip):
  * Point estimates agree with MC on both routes at every dt down to 0.005, which is **40× the admissible
    EM step**. ⟨U⟩ bias: +0.005 ± 0.0015 (dt 0.005), +0.0045 ± 0.002 (0.0025), +0.0001 ± 0.0018 (0.00125).
  * Its gate reads MARGINAL only because z mixes more slowly at γ = 1, so the profile noise is 0.008-0.010
    and the upper bounds exceed 0.0075.
  * Momenta: mean 4e-5 to 2e-4; variance 0.99982 / 1.00006 / 0.99992 (exact 1); standardised skewness
    ≈ −1e-4; excess kurtosis ≤ 2e-4. The O step's Gaussian momentum law is intact.
  * The configurational marginal is correct at the much larger underdamped step. This matches the known
    superiority of BAOAB's configurational accuracy over overdamped EM.
  * The two schemes' admissible steps should **not** be compared through a dt_od ≈ dt_ud² rule: the
    timesteps were chosen independently for each scheme.
  * **Hamiltonian part** (`scripts/numerical_validation/wca_nve_check.py`, `nve_check.json`). Velocity
    Verlet with γ = 0, from 16 exact-Gibbs configurations with Maxwell momenta, 100 t.u. each:

    | dt | RMS fluctuation of H (kT, 200 dof) | drift (kT per t.u.) |
    |---|---|---|
    | 0.005 | 0.0166 | 6.7e-5 ± 1.0e-4 |
    | 0.0025 | 0.0040 | 0 ± 2.5e-5 |
    | 0.00125 | 0.0011 | −1.1e-5 ± 1.0e-5 |

    The fluctuation ratio per doubling of dt is 3.7 / 4.1, the O(dt²) of a symplectic scheme with a
    smooth force. There is no secular drift. The intended force is integrable without any clip at these
    steps, consistent with the force audit.

Smooth secondary observables (seed jackknife se):

| ensemble | P(z > 0.5) | ΔF_wells = −log[P(stretched)/P(compact)] | ⟨U⟩ |
|---|---|---|---|
| MC (exact) | 0.70525 ± 0.00023 | −0.9673 ± 0.0012 | 11.0981 ± 0.0002 |
| MALA (exact) | 0.7054 ± 0.0018 | −0.968 ± 0.009 | 11.0971 ± 0.0007 |
| EM 0.002 | **0.6435** ± 0.0016 | **−0.689** ± 0.007 | 4.5e13 (deep overlaps) |
| EM 0.001 | 0.6976 ± 0.0015 | −0.926 ± 0.008 | 13.258 ± 0.003 |
| EM 0.0005 | 0.7028 ± 0.0013 | −0.949 ± 0.007 | 11.712 ± 0.003 |
| EM 0.00025 | 0.7048 ± 0.0014 | −0.958 ± 0.007 | 11.364 ± 0.003 |
| EM 0.000125 | 0.7046 ± 0.0012 | −0.956 ± 0.006 | 11.219 ± 0.003 |
| BAOAB 0.005 | 0.7052 ± 0.0036 | −0.967 ± 0.018 | 11.103 ± 0.002 |

At the accepted dt the compact/stretched balance is off by 0.28 kT, and the probability of the stretched
state is 0.643 instead of 0.705.

## 4. Trajectory sanity checks (necessary, not sufficient)

`figures/fig_wca_dynamics_sanity.png`:

* (a) z(t) is stable and confined at every dt, with frequent transitions between the wells. A stable
  trajectory is therefore no evidence of correct sampling: the dt 0.002 chain looks entirely healthy and
  is 0.3 kT wrong.
* (b) The closest-approach distribution separates the integrators from exact Gibbs by orders of magnitude
  (P(r_min < 0.865σ): 0.31 / 3.0e-2 / 2.7e-3 / 2.8e-4 / 4.2e-5 vs 2.5e-6).
* (c) Clip, min_r and sample-clip activation fall by about 10× per halving of dt.
* (d) ⟨U⟩ bias is first order in dt.
* (e) Independent lattice starts per seed show no dependence on the initial condition beyond the seed
  scatter. At dt 0.002 the whole seed cloud is displaced.
* (f) The z histograms are all finite and smooth. Only the dt 0.002 one is visibly wrong.

No positions, forces or energies were non-finite in any production-EM chain. Non-finite values occurred
only in the unclipped LM chains at dt ≥ 0.0005.

## 5. Precision and statistics

* Every uncertainty comes from *independent chains*: a leave-one-chain-out jackknife for per-bin noise and
  a chain bootstrap for the upper bounds. No within-chain sample is treated as independent.
* The z autocorrelation time of an MC chain is below 300 sweeps, so each chain holds about 1e5
  decorrelated z samples.
* Precision floor of the gate: about 0.003 kT. The bootstrap upper bounds at the admissible dt (0.0074,
  0.0049) are dominated by that noise, not by bias. D at dt 0.000125 is within 1-1.5× the noise.

## 6. Exact-measure vs dt-consistent measure

At dt 0.0005, the 2026-10-04 replica ladder used a *dt-consistent* reference: the pooled serial ABF of the
same chain. Against exact Gibbs that reference is 0.0145 kT off (`WCA_REFERENCE_AUDIT.md`). It was a
useful debugging reference, but it is not the physical free energy: its error is 3× the admissibility
tolerance, and comparable to the FR effects studied at that dt. Physical claims need the exact reference,
or a dt at which the chain's own reference is within tolerance of it. That dt is 0.000125.

