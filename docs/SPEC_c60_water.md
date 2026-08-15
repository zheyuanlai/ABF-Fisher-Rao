# SPEC — C60–C60 association in explicit TIP4P-Ew water (ABF eligibility study)

**Status: FROZEN at the commit carrying this file.** Licensed by Amendment 16 of
`docs/V2_PREREGISTRATION.md`. No C60 trajectory may run before this file is committed, and no
clause below may be edited after the ABF-only screen runs except by a further numbered amendment
stating what changed, why, and what had been seen at the time.

**What had been seen when this was written:** the methane study was CLOSED (ABF-sufficient,
8/8 seeds); the NaCl study had a parity-passed engine, an accepted TI reference, Gate 0 and
Gate A passes, and a running N = 64 screen with **no Gate B/C verdict**; the Zangi 2014 paper
(`cache/zangi2014/zangi2014_jpcb.pdf`) had been read for its model and protocol. **No C60
trajectory, box, reference, screen result, gate verdict or mFR result of any kind existed.**
No NaCl screen outcome influenced any clause below.

**This is an ABF eligibility study, not an mFR experiment.** The four-way classification of
Amendment 10 is applied in the frozen order **Gate 0 → A → B → C → D, and the first failing gate
is the verdict.** mFR is prohibited until an establishment-limited cell is frozen:
`results/c60/calibration/` and `results/c60/production/` **must not exist** before that.

**Preregistered expectation: literature-motivated candidate, not a predicted positive.** C60 is
opened because large-solute hydrophobic association couples the pair separation to collective
interfacial-water reorganisation — Zangi identifies a distinct class of confined interfacial
waters (I2) that exists **only** in the dimeric contact state, and explicitly notes that small
solutes like methane are too small to host them. That makes an establishment gap *plausible
enough to test*, on the far side of the hydrophobic length-scale crossover from the methane null.
The ABF-only screen decides whether the realised bottleneck is ABF-sufficient,
discovery-limited, conditional-equilibration-limited, or establishment-limited. Every one of the
four is a publishable outcome; a null is closed without an mFR arm, exactly as methane was.

---

## 0. The question

```
  CH4/H2O   ABF-sufficient   (closed 2026-08-14: worst Q/Q* ~ 0.83, no window)
  NaCl/H2O  screen running
  C60/H2O   ?                <- this study
  WCA       establishment-limited (the constructed positive)
```

Does a published large-hydrophobe explicit-water system naturally produce
`T_hit << T_est <= T_run` — states discovered early by multiwalker ABF whose bias-aware
populations nevertheless take a substantial fraction of the run to establish? Methane answered
no with a wholly local solvent response; C60 presents ~0.71 nm hydrophobic spheres whose
association reorganises structured interfacial water collectively. Same CV type, same solvent,
qualitatively different solvent length scale — the controlled progression, not a convenient one.

---

## 1. Physical model (frozen): Zangi, J. Phys. Chem. B 118, 12263 (2014)

Primary source: Zangi, *Are Buckyballs Hydrophobic?*, JPCB 118:12263–12270 (2014),
doi `10.1021/jp508174a`, cached at `cache/zangi2014/zangi2014_jpcb.pdf`. Internal units:
**nm, kJ/mol, ps, amu, K, e**.

| quantity | value | provenance |
|---|---|---|
| solutes | 2 rigid C60, internal geometry fixed | paper (positions held fixed) |
| water model | TIP4P-Ew, rigid, 4-site | paper (ref 45) |
| water count `N_w` | 1282 | paper |
| total sites | 120 C + 4 × 1282 = **5248** | — |
| C charge | 0 | paper |
| `sigma_CO` | 0.319 nm | paper (Werder graphite/water, ref 40) |
| `epsilon_CO` | 0.392 kJ/mol | paper |
| C–C rule | **geometric combination from C–O** | paper ("used to extract the LJ parameters (using the geometric combination rule) between two carbon atoms") |
| intra-cage interactions | **excluded** | paper |
| cage orientation | pentagon rings facing, parallel, out of registry | paper |
| LJ cutoff | 1.0 nm, **no switch**, tail correction NPT-only | paper (GROMACS convention; see deviations) |
| electrostatics | PME, real-space cutoff 1.0 nm | paper |
| temperature | **300 K** (primary; the paper's 5-T ladder is out of scope) | paper |
| pressure (box equilibration only) | 1.0 bar | paper |
| timestep | 2 fs, subject to the §3.4 dt gate | paper |
| box | rectangular, `Lx = Ly < Lz` | paper |
| reaction coordinate | axial COM separation `d_com`, along z | paper |
| reference domain | `[0.908, 2.428]` nm, 68 uniform windows | paper |
| reference protocol | 2 ns equil + 12 ns/window equivalent budget (§5) | paper |
| PMF anchor | `F(2.428 nm) = 0` | paper |

### 1.1 Parameters are derived from what is actually simulated

TIP4P-Ew is taken from `amber14/tip4pew.xml` as OpenMM stores it — `sigma_O = 0.316435 nm`,
`epsilon_O = 0.680946 kJ/mol`, `q_H = +0.52422`, `q_M = −1.04844`, `r_OH = 0.09572 nm`,
`r_HH = 0.15139006545247014 nm`, M-site a `ThreeParticleAverageSite` with weights
`(0.786646558, 0.106676721, 0.106676721)` — and the torch engine reads every parameter back out
of the OpenMM `System` (the methane `sigma_O` rule; `tests/test_c60_engine.py` pins the values).

Carbon LJ parameters are **derived** from the paper's cross terms and the stored water values:

```
  sigma_C   = sigma_CO^2   / sigma_O    = 0.319^2   / 0.316435   (~0.3215865 nm)
  epsilon_C = epsilon_CO^2 / epsilon_O  = 0.392^2   / 0.680946   (~0.2256641 kJ/mol)
```

so that the **geometric** combination reproduces the paper's `sigma_CO`, `epsilon_CO`, and its
geometric-rule C–C parameters exactly. OpenMM's `NonbondedForce` applies Lorentz–Berthelot,
whose arithmetic sigma mean then gives `sigma_CO = 0.3190104 nm` — **1.04e-5 nm (3.3e-5
relative) above the paper's 0.319**, while C–C and everything else is exact. This is Declared
Deviation 1; its energetic size (~1e-3 kJ/mol at contact) is four orders below `kT`.

### 1.2 Cage geometry (declared choice)

The paper does not print the internal C60 geometry. `src/c60/geometry.py` builds the exact
two-bond-length truncated icosahedron with the gas-phase electron-diffraction values of Hedberg
et al., *Science* 254:410 (1991): `r_66 = 0.1401 nm`, `r_65 = 0.1458 nm` (Declared Deviation 2).
Construction is closed-form (icosahedron edge `s = 2 r_65 + r_66`, truncation fraction
`lam = r_65/s`); `validate_cage` asserts both bond classes to 1e-9 nm and a single vertex radius
(measured 0.356208 nm, cage diameter 0.7124 nm). Orientation: pentagon centred on each pole of
`z`; cage B is a **pure translate** of cage A along `+z`, which realises the paper's "parallel
and out of registry" exactly — the facing pentagons are staggered by 36.000 degrees, measured,
and at `d = 0.968 nm` exactly 25 inter-cage C–C pairs sit below 0.40 nm (5 facing 5, consistent
with the paper's "only five carbon atoms from each fullerene touch other carbons").

### 1.3 Rigid-solute kinematics

The cages never rotate and never move laterally. The **only** solute degree of freedom is

```
  xi(q) = Z_B - Z_A = d_com          (cage COMs; midpoint fixed at the box centre)
```

In the constrained reference (§5) and every conditional pool, both cages are held entirely
fixed, as in the paper. In the ABF screen (§7), `xi` is dynamical with effective mass

```
  mu = M_C60 / 2 = 60 x 12.011 / 2 = 360.33 amu
```

(two rigid bodies of mass `M` moving `±xi/2` about a fixed midpoint), propagated by the same
BAOAB Langevin scheme as the water, and the 120 carbon positions are reconstructed from `xi`
each step. Fixing the midpoint, the lateral offsets and the orientations is Declared Deviation 3
— it removes rigid-body degrees of freedom the paper also froze (positions fixed), and makes
the sampled measure exactly the conditional canonical measure on the `xi` fibre product.

### 1.4 Box and ensemble

NPT is used **only** to fix the volume; all reference, pool and screen dynamics is NVT at the
frozen box (Declared Deviation 4, the campaign convention — the paper ran NPT production with a
Berendsen barostat; its measured `Delta V` across the association is small and positive).

1. build exactly 1282 TIP4P-Ew waters around the cages at `d_ref = 2.428 nm` (the PMF anchor
   state) in a rectangular box of built aspect `A = Lz/Lx` (initial guess `2.64 x 2.64 x
   5.65 nm`, consistent with the paper's Figure 3 z-extent and 1282 waters at ~1 g/cm^3);
2. NPT at 300 K, 1 bar (isotropic MC barostat, aspect preserved; tail correction ON; cages
   fixed and asserted unmoved), 0.5 ns discard + 1.0 ns measurement;
3. freeze `Lx = Ly = (<V>/A)^(1/3)`, `Lz = A Lx`, record both in the run manifest;
4. everything downstream is NVT at that box; PME `alpha` and grid are pinned at the frozen box
   (`ewaldErrorTolerance 5e-4`) and recorded. Grid spacing must satisfy the paper's <= 0.12 nm.

**Geometric guards (asserted, not assumed):** `Lx/2 > 1.0 nm` cutoff; the outer cage poles at
`xi = 2.428` leave `> 2 nm` of water between periodic images along z; the interfacial shells
(1.082 nm from each cage COM, the paper's I1 boundary) fit inside `Lz/2` so a bulk region
exists, as in the paper's Figure 3.

**Finite-size exposure, stated in advance:** the lateral image-to-image cage surface separation
is ~1.9 nm. This is the paper's own geometry (Declared: we inherit its finite-size errors as
part of reproducing it). §10 adds a finite-size gate after the primary reference, with
worst-case propagation through the Gate C classification, methane-style.

---

## 2. Reaction coordinate, mean force, ABF discretisation

`xi` is **linear** in the coordinates: `grad_{q_j} xi = ±(1/60) e_z` for cage B/A carbons,
`|grad xi|^2 = 1/30`, and the divergence of the den Otter field vanishes. Therefore

```
  f(q)   = (1/2) (Fphys_A,z - Fphys_B,z)         Fphys_X,z = total physical z-force on cage X
  F'(d)  = E_nu [ f(q) | xi = d ]
```

with **no Jacobian / radial-entropy term**: `F` is the paper's PMF directly, and the methane
`W`-vs-`F` distinction does not arise (the identity test of the other specs is vacuous here and
is replaced by a finite-difference test of `dV/dxi` against `f`, same tolerance). Only physical
forces enter the estimator — never ABF, wall or constraint forces.

ABF bias: Cartesian force `-A'_hat(xi)/60 e_z` on every cage-A carbon and `+A'_hat(xi)/60 e_z`
on every cage-B carbon... applied to the `xi` degree of freedom directly as the generalised
force `+A'_hat(xi)` (equal and opposite on the two cages by construction, zero net force).

**Evaluation domain** `Omega = [0.908, 2.428] nm` (the paper's), **`n_grid = 153`** (spacing
0.01 nm, the NaCl standardised bin rule; odd, no Nyquist row). Soft walls **at** 0.908 and
2.428 nm with the NaCl constant `k_wall = 41840 kJ/mol/nm^2`, identical on every arm.
`fullSamples`-equivalent trust ramp `abf_min_count = 500` (the NaCl published value; applied to
the bias only, never the estimate). Bandwidth ladder `h ∈ {0.008, 0.012, 0.016} nm`, selected by
the methane §6.2 deterministic rule on ABF-only + reference data, then frozen; `h` is never
selected by whether mFR improves.

**Scoring mask:** `Omega_thermal` = largest contiguous interval containing `argmin F_ref` on
which `F_ref - min F_ref <= 15 kT` (the NaCl rule; scoring mask only, never a sampling
restriction).

---

## 3. Stage 0 — engine equivalence gate (binding, blocks everything)

The batched torch engine (`src/c60/`) extends the methane architecture: rectangular periodic
cell, 4-site water (M-site force redistribution by the OpenMM virtual-site weights, M
reconstructed after every position update), unswitched LJ, PME over the 3846 charged sites,
M-SHAKE/RATTLE on the (O, H1, H2) triples. `src/methane/` is **not modified** (Amendment 14.1
discipline); shared stateless functions are imported, everything structural is owned by
`src/c60/`.

### 3.1 Parity (tolerance 1e-6, OpenMM Reference platform, float64)

Over **>= 16 configurations** spanning `d ∈ {0.908, 0.95, 0.968, 1.00, 1.05, 1.10, 1.20, 1.30,
1.50, 1.70, 2.00, 2.20, 2.428} nm` plus thermal perturbation draws, including **distinct solvent
structures at the same `d`** (the clause the NaCl test pool under-implemented, fixed here):

| quantity | target |
|---|---|
| `V_torch` vs `V_openmm` (total; LJ-only; electrostatics-only) | rel < 1e-6 |
| all forces, both engines | rel < 1e-6 |
| cage forces `Fphys_A`, `Fphys_B`, hence `f(q)` | rel < 1e-6 |
| `xi` | exact to float64 round-off |
| `f` vs finite-difference `dV/dxi` (cages displaced ±h/2 along z) | rel < 5e-6 |
| batched vs single walker | rel < 1e-9 |

Plus: constraint satisfaction `<= 1e-8 nm` over a production-length run; M-site position error
vs the OpenMM virtual-site placement `<= 1e-12 nm`; equipartition per §3.4.

**If the gate fails, C60 does not run.**

### 3.2 Physical validation (after parity, before any free energy)

Bulk TIP4P-Ew: stable T and density at the frozen box, `g_OO(r)` against the TIP4P-Ew
literature curve (sanity, not curve-identity). Single-cage hydration: the C–O RDF first peak
and the paper's I1 water-layering scale (~0.35 nm surface offset) must be physically sensible.

### 3.3 Determinism and process discipline (inherited, measured elsewhere)

Whole blocks run in **one process on one GPU** (the WCA within-process determinism trap).
Nothing imports torch in an OpenMM-CUDA process (measured deadlock). Every run records
`CUDA_VISIBLE_DEVICES`, device idle state, commit, and the §11 launch ladder applies.

### 3.4 dt gate (frozen decision rule)

Candidate `dt = 2 fs` (paper) vs fallback `1 fs`: accept 2 fs iff over a 50 ps pilot at the
frozen box (i) kinetic temperature within 1 K of target under the §3.1-validated KE convention,
(ii) constraint drift `<= 1e-8 nm`, (iii) `<f>` at 3 separations agrees between the two dt
values within combined block error. Decided from the measurement, before the reference runs;
the decision and numbers are recorded in `results/c60/parity/RESULT.md`.

---

## 4. The orthogonal descriptor `n_gap`

The paper's I2 waters — confined between the two convex surfaces in the contact state, fewer
hydrogen bonds, lower entropy, absent for small solutes — are the physical reason C60 is worth
opening. The descriptor is frozen from the paper's own analysis region:

```
  n_gap(q) = sum_j s(|u_j|; xi/2) * s(w_j; R_cyl)        R_cyl = 0.62 nm  (paper, Figure 3)
```

water oxygens only; `u`, `w` axial/radial offsets from the cage midpoint; `s` the standard
rational switch of `methane.observables` (smooth, differentiable). `R_cyl` is the paper's I2
cylinder radius and is **not tuned**. Secondary recorded descriptors (diagnostic only, never
gate inputs): `n_shell` = smooth count of oxygens within 1.082 nm of either cage COM (the
paper's I1+I2 interfacial band).

**Orthogonality ratio (the NaCl pre-committed caveat, inherited):** report
`R_orth = across-d sd / within-d sd` of `n_gap` on the reference. Methane read 5.4x, NaCl
hydration 14–83x. If `R_orth` is enormous, `n_gap` is a re-spelling of `d` and a Gate A pass is
weak evidence; the number is reported next to every Gate A statement.

---

## 5. Stage I — constrained mean-force reference (before any ABF)

The paper computes the PMF by fixed-separation average-force integration — the same instrument
as the campaign's TI references. Ours, batched on the torch engine after Stage 0:

```
  68 uniform windows on [0.908, 2.428] nm       (the paper's layout; uniformity declared)
  4 solvent families per window: wet / dry / bulk / hot   (the §6 pool recipe)
  3 replicas per family, 100 ps equilibration + 250 ps production each
  3 independent builds, differently initialised
  aggregate: 68 x 4 x 3 x 3 x 0.35 ns ~ 857 ns
```

Cages fully fixed per window (the paper's convention); only water propagates. Accumulate
`f(q)` block means (5 ps blocks), the family means separately, and `n_gap` / `n_shell` traces.

```
  F_ref(d) = integral from d to 2.428 of -<f>(s) ds        (trapezoid on the 68-point grid,
                                                            F_ref(2.428) = 0, paper's anchor)
```

Family disagreement per window is reported, never averaged away — it is the same instrument as
Gate 0. Linear interpolation to the 153-point grid for gate/endpoint use; **no smoothing enters
any gate decision** (the WCA lesson).

**Acceptance (all must hold, else STOP and rebuild):**

```
  R_ref = (max pairwise L2 between builds) / (0.10 x span of consensus F)  <=  0.5
  OpenMM CUDA spot-check: >= 6 windows x 2 replicas, <f> within combined block error
  block length exceeds the measured f autocorrelation time
  family disagreement reported per window
```

**Reproduction gate against the paper (preregistered tolerances):** on the consensus reference,
(i) the contact minimum lies in `[0.94, 1.00] nm` (paper: 0.968–0.97); (ii)
`F(contact) - F(2.428)` lies within `± 3 kJ/mol` of the paper's ~`-16 kJ/mol` total at 300 K
(read from Fig. 1a: `Delta G ~ -14.5 kJ/mol` plus anchor conventions; tolerance covers
digitisation and our declared deviations); (iii) the direct (vacuum) cage–cage LJ energy
reproduces the paper's ~`-18.5 kJ/mol` within `0.5 kJ/mol` — **definition fixed by Amendment
16.6 before any solvent datum existed: the untruncated sum at the direct term's own minimum
over `d`** (measured `-18.78 kJ/mol` at `0.982 nm`: PASS);
(iv) a solvent-induced repulsive barrier exists between contact and 1.4 nm (paper Fig. 1d:
peak ~+6 kJ/mol at 300 K). Failing (i), (ii) or (iv) is a **model-reproduction failure: STOP**,
reported, no screen. (iii) failing is an implementation bug.

States: the Amendment 3 rule on the consensus `F_ref` — local minima, merge across barriers
`< 2 kT` measured from the higher minimum, boundaries at the intervening maxima; if a single
minimum survives, the fallback is the frozen equal-width tercile partition, declared as a
partition of the coordinate. Gate C additionally runs on the fine 153-bin partition, so the
classification is not hostage to one state definition. **Never from ABF data.**

---

## 6. Stage II — Gate 0 pools and `tau_perp` (before the screen)

At `d_k ∈ {d_contact, d_barrier, d_mid, 1.60, 2.20 nm}` — the first three from **our**
reference (contact minimum, principal barrier, and the solvent-separated minimum if one exists,
else 1.30 nm) — four solvent families per point, **>= 32 replicas per family**, cages fixed:

| family | preparation |
|---|---|
| `wet` | equilibrated at `2.428 nm`, cages teleported to `d_k`, brief re-settle |
| `dry` | equilibrated at the contact minimum, cages teleported to `d_k` |
| `bulk` | equilibrated at `d_k` directly (control) |
| `hot` | solvent randomised and briefly re-equilibrated (destroyed interface) |

* **Gate 0:** cross-family spread of `<f>` against `|F'_ref|` — **both** statistics
  (`G_ref = spread/|F'_ref|`, `G_pool = spread/|pool mean|`, the WCA divergence lesson), judged
  against the frozen calibration ladder **WCA 0.040 pass / gateway 0.036–0.189 pass / deca 0.61
  fail / R15 0.564 fail**, argued in `RESULT.md` in the error-carrying region; no numerical
  threshold, per Amendment 9. Fail ⇒ **conditional-equilibration-limited, STOP** (the
  Amendment 8 theorem: no marginal selection can repair `p(y|xi)`).
* **`tau_perp` (family):** `inf{ t : max_{a,b} TV[p_t(n_gap | d_k, a), p_t(n_gap | d_k, b)]
  <= 0.2 }`, maxed over `k`.
* **`tau_perp` (clone twin, the Gate D instrument):** exactly duplicated configurations,
  independent noise, `C_ngap(t) <= 1/e`. Both computed; disagreement > 2x reported as an open
  finding, not resolved by choosing one.

---

## 7. Stage III — ABF-only eligibility screen

**No Fisher–Rao anywhere in this stage.**

```
  fixed compute per cell:  B_MD = N x T = 128 ns
  N ∈ {8, 16, 32, 64}  ->  T ∈ {16, 8, 4, 2} ns
  8 ensemble seeds per cell: 7000–7007
  execution order: N = 64 first, then 32, 16, 8, sequentially
  frozen stopping rule: the ladder stops after N = 64 unless its verdict is
  establishment-limited (the methane Amendment 12.5 rule); a partial map is reported as partial
```

> **SUPERSEDED IN PART BY AMENDMENT 16.8 (before any reference datum):** because
> `Q*_k <= 1`, the 16.7 floor `min-span N Q*_k >= 16` is unsatisfiable at `N = 8` and
> degenerate at `N = 16` — those cells cannot classify and **are not run**. The executable
> map is `N = 64`, then `N = 32` only if 64 is establishment-limited. The Amendment 3
> partition and its `lambda_k(N)` table (unbiased and flat-bias brackets) are recorded from
> the accepted reference **before any screen cell launches**, and the partition is never
> revisited after any occupancy is read.

* **Initial condition (frozen):** every walker starts at the **global minimum of `F_ref`**
  (expected contact, ~0.97 nm — verified against our own reference before the screen so the
  NaCl "start was not AT the minimum" caveat cannot recur). Each walker receives an
  independently equilibrated solvent environment: >= 50 ps restrained equilibration per walker
  outside the ABF budget, `assert_distinct_solvent` enforced; initial baths drawn from the
  §6 `bulk` pool machinery at `d_0`. Declared bias: a contact start can only push toward
  discovery-limited, never manufacture establishment-limited.
* Estimator: `alkanes.interval` machinery unchanged; §2 grid, walls, ramp, bandwidth rule.
* Traces: `xi` every 0.5 ps, `n_gap` every 2 ps, bias profile checkpoints every 25 ps —
  cadences chosen so the §8 ballistic-floor comparison is resolvable
  (`sqrt(kT/mu) ~ 0.083 nm/ps` ⇒ ~5 ps ballistic transit to the barrier region, 10x the trace
  interval).
* Checkpoint/resume with full integrator + estimator + RNG state, the methane machinery.

---

## 8. Gates B and C (frozen thresholds, inherited unchanged)

* **Gate B (discovery):** persistent `T_hit,k < 0.1 T` on **>= 6 of 8 seeds**, per relevant
  state; persistence 2.0 ps (anti-flicker); the verdict instrument **refuses a partial seed
  block**. Additionally, fixed far-distance thresholds `d ∈ {1.2, 1.4, 1.6, 1.8, 2.0, 2.2} nm`
  are scored against the extreme-value ballistic floor (the NaCl `_fastest_of_n_p99`
  instrument) — diagnostic only, and the vacuity check: if the first state boundary is inside
  the ballistic floor or the trace resolution, Gate B is reported as non-binding there, the
  NaCl precedent. Fail ⇒ **discovery-limited, STOP.**
* **Gate C (establishment):** bias-aware target
  `Q*_k(t) = int_{C_k} exp(-beta[F_ref - B_t]) / int_Omega exp(-beta[F_ref - B_t])`;
  under-established = occupancy `< 0.5 Q*_k(t)` for a contiguous `>= 0.20 T` within the second
  half. Early hit + no deficit ⇒ **ABF-sufficient, STOP** (the methane outcome; entirely
  acceptable). Early hit + persistent deficit ⇒ establishment-limited, continue to Gate D.
  * **Small-N safeguard — SUPERSEDED BY AMENDMENT 16.7 before any Gate C datum:** the
    originally frozen `N Q*_k >= 3` admits zero-power cells (at `lambda = 3` even an empty
    state is 1.73 sigma; the deca retraction is this class). The binding guard is
    **`min over the judged span of N Q*_k(t) >= 16`** (two-sigma floor for the 0.5 ratio,
    the NaCl convention; minimum not mean, so a healthy stretch cannot mask a starved one).
    A cell where no state clears the floor is **UNCLASSIFIABLE at that N**, reported as such,
    never extended until it classifies.
* **Gate D (clone decorrelation):** `lambda_rep x tau_perp <= 0.1` with activity floor
  `N_repl >= 0.5 N`; no active safe rate ⇒ C3 failure, STOP.
* **Eligible cell if several pass every gate: the smallest `N`, mechanically** (the campaign
  §8.2 rule; the ChatGPT plan's largest-N proposal is rejected in favour of the frozen
  convention, and the `N Q* >= 3` safeguard covers its motivation).

Classification is implemented in `scripts/c60_gates.py`, which refuses to classify an
incomplete seed block, exactly like `scripts/nacl_gates.py`.

---

## 9. Stages IV–V — only if an establishment-limited cell is frozen

Everything inherited unchanged from the campaign; nothing C60-specific is invented:

* **Calibration:** seeds **7100–7103**, the §3 preregistration ladder spanning
  `lambda_rep tau_perp ∈ {0.01, 0.03, 0.10, 0.30}`, inactive rates struck, genealogy gates
  `ESS_anc/N >= 0.30`, `w_max <= 0.05`; rate selected by turnover/decorrelation/genealogy
  criteria only, never by `F` error.
* **Production:** 16 fresh matched seeds **7200–7215**; arms `abf`, `mfr_practical`,
  `mfr_sham` (one sham per FR arm, replaying realised event times and replacement counts with
  randomised identities; the direct arm-vs-sham contrast is the attribution statistic). The
  prior-art arms (`count_balancing`, `book_laplacian`) run as a declared closure on the same
  frozen setting conditional on a positive Q3, and are reported whatever they show (the Q1
  discipline; WCA's tie makes this mandatory, not optional).
* **Endpoints:** primary `I_F` (time-integrated `L2(Omega_thermal)`), secondary `I_F'`,
  `T_hit`/`T_est`/`tau_perp`, genealogy set, round trips, and the **conditional-fidelity
  endpoint** `TV(p_method(n_gap | d), p_ref(n_gap | d))` — a marginal gain paid for by fibre
  distortion is not a success. Frozen-bias validation mandatory for any positive (~25 % budget).
  Equal wall-clock primary, equal force evaluations secondary.
* **Success:** §4.3 unchanged — `median <= -10 %`, CI upper `< -5 %`, `>= 12/16` seeds,
  genealogy gates, sham contrast CI `< 0`. Equivalence claims by TOST, never by
  "no CI excluded zero".

---

## 10. Finite-size gate (after the primary reference, before the screen verdict is final)

One enlarged box (`Lx' = Ly' ~ 1.25 Lx`, same `Lz`, waters re-solvated at the frozen density),
`<f>` at 3 windows: the contact minimum, the principal barrier, `2.20 nm`; `>= 3` replicas
each, compared with the primary values with propagated uncertainty. The worst statistically
plausible correction is propagated through `F_ref -> Q*_k(t) ->` the Gate C classification
(methane's worst-case machinery). If the classification is invariant, the exposure is recorded
and harmless; if not, the affected verdict is reported as finite-size-inconclusive. No
tolerance tighter than the measurement SEM is declared (the methane lesson).

---

## 11. Compute, launch discipline, and concurrency

* **Device: GPU 3** (H200 NVL), assigned by Amendment 16 (which renegotiates the 15.4
  scheduling clause: NaCl keeps GPU 2; C60 takes GPU 3). Pinned with `CUDA_VISIBLE_DEVICES=3`
  everywhere; idleness re-checked and recorded before each stage; one process per GPU; whole
  seed blocks in one process.
* **Launch ladder (Amendment 15.3, inherited):** pinned detached worktree at the commit in
  `results/c60/PINNED_COMMIT`; preflight (commit match, clean tree, tests pass, device idle,
  manifest) → correctness → idle-device throughput → smoke → dt gate → checkpoint-resume
  verification → STOP for review. A defect found mid-ladder restarts the ladder at a new
  pinned commit.
* **Budget (aggregate MD, frozen):** parity/validation ~5 ns; NPT 1.5 ns; reference ~857 ns;
  Gate 0 pools ~5 x 4 x 32 x 0.15 ns ~ 96 ns + twin ~20 ns; screen 128 ns x up to 4 cells;
  finite-size ~10 ns; production (only if licensed) 3 arms x 16 seeds x 128 ns-equivalent +
  25 % frozen-bias. Throughput (ms/step vs batch) is measured and reported before the
  reference; **if the frozen budget is infeasible at the measured cost, that is an amendment,
  not a quiet reduction.**

---

## 12. Stop conditions — every one is a reported result

| condition | verdict |
|---|---|
| engine-equivalence gate fails | C60 does not run; defect reported |
| box guards of §1.4 fail | box redesign amendment before any reference |
| reference acceptance fails | STOP, rebuild; no screen interpreted |
| reproduction gate (i)/(ii)/(iv) fails | model-reproduction failure; STOP, no screen |
| Gate 0 fails in the error-carrying region | **conditional-equilibration-limited**; STOP |
| Gate A fails | states invisible through `xi`; STOP — a stop for the CV, never a licence to tune |
| Gate B fails | **discovery-limited**; STOP (harder ≠ better for mFR) |
| Gate C finds no persistent deficit | **ABF-sufficient**; STOP — the methane outcome, publishable |
| deficit only where `N Q* < 3` | finite-population discreteness; STOP, reported as such |
| Gate D admits no active safe rate | C3 failure; STOP |

**If C60 fails, the successor is not a retuned C60.** No temperature, box, force-field or
budget adjustment until it passes; a force-field robustness experiment (a second published
fullerene parameterisation) is permitted only **after** the primary verdict and never replaces
it. A primary null stays a null.

---

## 13. Gate A (stated with the others for completeness, in the corrected orientation)

On the reference joint `(xi, n_gap)` samples: labels `Y` = terciles of `n_gap` **within the
thermally relevant region**, and

```
  max over label pairs of TV( p(xi | Y=a), p(xi | Y=b) )  >=  0.30
```

— the **preregistered direction** `p(xi|Y)`, not the transposed `p(Y|state)` both earlier
studies first implemented (their correction commits are the precedent; C60 starts correct).
`MIN_SAMPLES_PER_BASIN = 12`-style guards make under-sampled labels NOT COMPUTABLE, never a
silent 0. Reported always with `R_orth` (§4) so a pass cannot be oversold.

---

## 14. Declared deviations, collected

1. **Lorentz–Berthelot C–O in OpenMM** vs the paper's geometric rule: `sigma_CO` high by
   1.04e-5 nm (3.3e-5 rel); C–C and `epsilon` exact (§1.1).
2. **Cage internal geometry** Hedberg et al. 1991 (`0.1401/0.1458 nm`); the paper does not
   print its own (§1.2).
3. **Frozen midpoint / lateral offsets / orientations** for the dynamical screen; the paper
   fixed all cage coordinates in every simulation (§1.3).
4. **NVT at the NPT-frozen box** vs the paper's NPT production; MC barostat vs Berendsen for
   the equilibration; single frozen box for all windows vs per-window NPT volumes (§1.4).
5. **Langevin thermostat** (`gamma = 1 ps^-1`, BAOAB) vs the paper's velocity-rescaling; both
   canonical.
6. **PME internals** (order-5 B-splines, pinned alpha/grid at tol 5e-4) vs GROMACS PME with
   0.12 nm spacing and "quadratic interpolation"; both converged PME schemes; our real-space
   cutoff matches the paper's 1.0 nm.
7. **Reference replica structure** (68 windows x 4 families x 3 replicas x 250 ps x 3 builds,
   ~857 ns, family-hysteresis instrument built in) vs the paper's 68 x (2 + 12) ns single
   trajectories (~952 ns); comparable aggregate, ours adds the acceptance and Gate 0
   instruments the campaign requires.
8. **dt gate** may fall back from the paper's 2 fs to 1 fs (§3.4).
9. **Uniform window spacing** on `[0.908, 2.428]` assumed; the paper states the range and
   count but not the spacing.

---

## 15. References

Zangi, *Are Buckyballs Hydrophobic?*, J. Phys. Chem. B 118, 12263 (2014) — the frozen model.
Werder et al., J. Phys. Chem. B 107, 1345 (2003) — origin of the C–O parameters.
Horn et al., J. Chem. Phys. 120, 9665 (2004) — TIP4P-Ew.
Hedberg et al., Science 254, 410 (1991) — C60 gas-phase geometry.
Li, Bedrov & Smith, J. Chem. Phys. 123, 204504 (2005); Makowski et al., J. Phys. Chem. B 114,
993 (2010) — prior C60 pair-PMF literature (context; not reproduced).
Remsing & Weeks (arXiv:1502.05220) — the small/large hydrophobe crossover motivating the study.
Lelièvre, Rousset & Stoltz (2010) — the mathematical framework, unchanged.
