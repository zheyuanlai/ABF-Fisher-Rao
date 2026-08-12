# SPEC — methane–methane association in explicit SPC/E water

**Status: FROZEN at the commit carrying this file.** Licensed by Amendment 11 of
`docs/V2_PREREGISTRATION.md`. No methane dynamics may run before this file is committed, and no
clause below may be edited after the ABF-only screen runs except by a further numbered amendment
to the preregistration stating what changed, why, and what had been seen at the time.

**Preregistered expectation: LIKELY NULL.** §9 of the preregistration classifies methane as a
falsification benchmark, not as the next positive. This document is written so that an
ABF-sufficient verdict is a publishable result rather than a disappointment, and so that no
parameter can be moved afterwards to convert one into the other.

**Nothing in this specification had been measured when it was written.** There was no methane
engine, no box, no reference, no screen and no gate result.

---

## 0. The question

Two united-atom methanes in liquid water associate through the classic hydrophobic structure

```
  contact minimum  <->  desolvation barrier  <->  solvent-separated minimum
```

The reaction coordinate is trivial — one interparticle distance — while the orthogonal degrees of
freedom are real water molecules that must rearrange to open or close the gap. That combination
is what makes methane worth asking, and it is also what makes it dangerous: after
Amendments 8 and 10, "the orthogonal coordinates are interesting" is as much a description of the
**conditional-equilibration-limited** regime, where marginal selection provably cannot help, as
of the **establishment-limited** regime, where it can.

The hypothesis under test is the three-timescale window of Amendment 10:

```
  T_hit  <<  tau_perp  <<  T_est
```

* `T_hit` — time for ABF to discover the solvent-separated state;
* `tau_perp` — relaxation time of the water at approximately fixed methane separation;
* `T_est` — time for the replica population to establish the correct bias-aware allocation there.

`tau_perp` is the quantity that decides which of the two regimes methane is in, and it is
measured from ABF-only and restrained data, before any mFR arm exists.

---

## 1. Physical model (frozen)

United-atom methane in rigid SPC/E water, following Asthagiri, Merchant & Pratt,
*J. Chem. Phys.* **128**, 244512 (2008) where that paper states the model.

Project internal units are **nm, kJ/mol, ps, amu, K**. Literature values are quoted in Å and
kcal/mol because that is how they are published, and converted on load
(`1 kcal/mol = 4.184 kJ/mol`, `1 Å = 0.1 nm`).

| quantity | literature units | internal units |
|---|---|---|
| temperature `T` | 298 K | 298 K |
| `kT` | 0.59219 kcal/mol | **2.47771 kJ/mol** |
| water model | SPC/E, rigid | — |
| water molecules `N_w` | 512 | 512 |
| methane molecules | 2 | 2 |
| methane sites | 1 (neutral LJ) | 1 |
| `sigma_M` | 3.73 Å | **0.3730 nm** |
| `epsilon_M` | 0.294 kcal/mol | **1.230096 kJ/mol** |
| `sigma_O` | 3.166 Å (nominal) | **0.31657195050398826 nm** (as simulated — see below) |
| `epsilon_O` | 0.1553 kcal/mol | **0.649775 kJ/mol** |
| `q_O` | −0.8476 e | −0.8476 e |
| `q_H` | +0.4238 e | +0.4238 e |
| `r_OH` (rigid) | 1.0 Å | **0.1 nm** |
| `theta_HOH` (rigid) | 109.47° | 109.47° |
| methane mass | 16.043 amu | 16.043 amu |
| electrostatics | PME | PME |
| LJ switch → cutoff | 10.0 → 10.5 Å | **1.00 → 1.05 nm** |
| thermostat | Langevin, `gamma = 1 ps^-1` | `gamma = 1 ps^-1` |
| integrator | BAOAB + SETTLE/RATTLE | — |
| timestep `dt` | 0.5 fs | **0.0005 ps** |
| production ensemble | NVT at the NPT-equilibrated volume | — |

`epsilon_M = 0.294 kcal/mol` and `sigma_M = 3.73 Å` are the standard united-atom methane
parameters (`epsilon/k_B = 148 K`), consistent between the cited source and TraPPE-UA.

> **`sigma_O` is AMBER's, not the nominal published value, and the difference is recorded rather
> than rounded away.** The SPC/E paper quotes `3.166 Å`; `amber14/spce.xml` — the file actually
> simulated — stores a pair coefficient unpacking to `0.31657195050398826 nm`, **9.0e-5 relative
> below** it. That is physically negligible and numerically decisive: §3.2 demands `1e-6` parity,
> so a torch engine written from `0.3166` would miss by ~90× for a reason that is not a bug.
>
> The rule this fixes, for NaCl too: **the torch engine reads every parameter back out of the
> OpenMM `System` it is being compared against** (`methane.system.site_parameters`), never out of
> a constant transcribed from a paper. `tests/test_methane_stage0.py` pins the gap so it can
> neither drift nor be silently "corrected".

### 1.1 Interactions

Methane–methane, a single LJ pair:

```
  V_MM(r) = 4 eps_M [ (sig_M/r)^12 - (sig_M/r)^6 ]
```

Water–water, LJ on oxygen only plus Coulomb on all three sites:

```
  V_WW = 4 eps_O [ (sig_O/r_OO)^12 - (sig_O/r_OO)^6 ]  +  sum_{a in j} sum_{b in k} q_a q_b / (4 pi eps_0 r_ab)
```

Methane–water, LJ against oxygen only (methane is neutral, so it carries no electrostatics):

```
  V_MW = sum_{i=1,2} sum_j 4 eps_MO [ (sig_MO/|Q_i - O_j|)^12 - (sig_MO/|Q_i - O_j|)^6 ]
```

> **DECLARED DEVIATION — the unlike-pair rule is ours, not the paper's.** The cited source
> specifies the methane model, the SPC/E solvent and the numerical protocol, but an explicit
> methane–oxygen mixing rule could not be verified from it. Lorentz–Berthelot is applied:
>
> ```
>   sig_MO = (sig_M + sig_O)/2 = 0.344786 nm      (from the simulated sig_O above)
>   eps_MO = sqrt(eps_M eps_O)  = 0.894022 kJ/mol  (= 0.21368 kcal/mol)
> ```
>
> This is an implementation choice and is reported as one. If the original topology or the
> Garnier *et al.* simulation files are obtained, they replace this — they do not silently
> modify it. The claim this study may make is therefore: **the published methane/SPC-E model with
> a declared Lorentz–Berthelot unlike-pair rule**, not a bit-exact reproduction.

Total configurational potential: `V = V_MM + V_MW + V_WW`.

### 1.2 Canonical measure

```
  nu(dq) = Z^-1 exp(-beta V(q)) dq          beta = 1/(kT)
```

with `q = (Q_1, Q_2, q_water)`, `Q_i in R^3` the methane centres. This is the measure of
Lelièvre–Rousset–Stoltz, unchanged.

### 1.3 Box and ensemble

NPT is used **only** to fix the volume; all reference, ABF and mFR production is NVT, so that the
target measure is exactly `nu` above and no barostat variable enters the mathematics.

1. equilibrate 512 SPC/E waters + 2 methanes at `T = 298 K`, `P = 1 bar`;
2. measure `<V_box>` over a preregistered 1 ns window after 0.5 ns discard;
3. set `L = <V_box>^(1/3)`, **freeze it**, and record it in the run manifest;
4. all production is NVT at that `L`.

The expected `L` is ≈ 2.48–2.50 nm. **It is measured, not assumed**, and the measured value is
what is frozen.

> **Finite-size exposure, stated in advance.** At `L ≈ 2.49 nm` the minimum-image half-box is
> `1.245 nm`, so the top of the evaluation domain (`0.90 nm`) sits at ~72 % of it, and the pair's
> nearest periodic images are ~1.59 nm apart. The 1.05 nm cutoff is safely below `L/2`, but the
> outer tail of the PMF carries finite-size error at this box size, and the cited source sampled
> only 4–8 Å in the same 512-water box.
>
> **Gate:** the constrained-TI reference (§6) is additionally evaluated at
> `r in {0.70, 0.80, 0.90} nm` in a **1024-water box** (`L ≈ 3.14 nm`). If `<f_loc>` differs from
> the 512-water value by more than `0.1 kT/nm` at any of the three, the evaluation domain is
> truncated to the largest `r` that passes, **before** the reference is built. Truncating the
> domain is permitted here and only here — before any `F_ref` exists — and the decision is
> recorded with the measured numbers.
>
> This costs three extra TI points and removes an exposure that would otherwise be discovered
> only when a reviewer asks.

---

## 2. Reaction coordinate and mean force

```
  xi(q) = |Q_2 - Q_1| = r          minimum-image displacement under PBC
```

**Evaluation domain** `Omega_xi = [0.33, 0.90] nm` (`[3.3, 9.0] Å`), `n_grid = 115` (odd, so no
Nyquist row exists; spacing `0.004956 nm`). Soft walls at `0.34` and `0.89` nm, identical on every
arm.

The boundaries are **not** basin definitions. Contact minimum, desolvation barrier and
solvent-separated minimum are located from *our own* reference in §7.

### 2.1 Minimum image

`R = Q_2 - Q_1` is wrapped into `[-L/2, L/2)^3` before `r = |R|` and `e = R/r` are taken. This is
the only change `alkanes.distance_cv.DistanceCV` needs; it is validated against the autodiff path
already in that module, with the wrap applied inside the differentiated function so the gradient
is the gradient of the wrapped distance.

Because `r <= 0.90 nm < L/2`, the wrap is a no-op on physically sampled configurations. It is
implemented anyway so that no configuration can silently produce a coordinate from the wrong
image.

### 2.2 Geometry and local mean force

With `grad_{Q_1} xi = -e`, `grad_{Q_2} xi = +e`, `|grad xi|^2 = 2`, and the den Otter field
`v = grad xi / |grad xi|^2`:

```
  div v = div_R (R/|R|) = 2/r
```

so the instantaneous local mean force whose conditional average is `F'(r)` is

```
  f(q) = (1/2) (grad_{Q_2} V - grad_{Q_1} V) . e  -  2/(beta r)
       = (1/2) (Fphys_1 - Fphys_2) . e            -  2/(beta r)
```

with `Fphys_i = -grad_{Q_i} V`, and

```
  F'(r) = E_{nu_xi(. | r)} [ f(q) ]
```

**This is already implemented.** `alkanes.distance_cv.DistanceCV.local_mean_force` computes
exactly `f_R = grad V . v - beta^-1 div v` with `div_v = 2/R`, validated against
`torch.func` autodiff by `DistanceCV.geometry_autodiff`. The methane engine reuses it; it does
not reimplement it.

### 2.3 `F(r)` is not the radial PMF `W(r)` — both are kept

The chemistry literature plots `W(r) = -beta^-1 log g_MM(r)`. The shell volume element gives
`p_xi(r) ∝ r^2 g_MM(r)`, hence

```
  F(r)  = W(r) - 2 beta^-1 log r + C
  W'(r) = F'(r) + 2/(beta r)
```

so the geometric term cancels in `W`, leaving

```
  W'(r) = E[ (1/2)(Fphys_1 - Fphys_2) . e | xi = r ]
```

— the bare average force along the line of centres.

Both are reported everywhere. `F` is what ABF/mFR flattens and what the primary endpoint scores;
`W` is what the methane literature can be compared against. **Every reference build and every arm
emits both**, and the identity above is a hard implementation test:

```
  || W'_hat - (F'_hat + 2/(beta r)) ||_inf  <  1e-10 kJ/mol/nm      (exact, same samples)
```

Keeping only one of the two is how a reference, an ABF force and a published PMF can differ by a
systematic radial-entropy term while all three look plausible.

### 2.4 ABF bias force

With `A_hat_t` the learned bias, the Cartesian force added to the two methanes is

```
  Fbias_1 = -A'_hat_t(r) e          Fbias_2 = +A'_hat_t(r) e
```

equal and opposite, so no net translational force acts on the pair. Applied through
`alkanes.distance_cv.dist_bias_force`, unchanged. The `abf_min_count` / `fullSamples` guard
(Amendment 5, Defect 2) is applied and is covered by a unit test that fails if the config value is
ignored.

---

## 3. Stage I — the engine, and the gate it must pass

This is the largest piece of work in the study and the only genuinely new infrastructure. The
project has **no** periodic, solvated sampler; §8.1 of the preregistration gates NaCl behind
exactly this engine, and methane is now its shakedown (Amendment 11.2).

### 3.1 Design

A batched `(walker, site, 3)` torch engine, in the style of the existing samplers:

* **periodic boundaries**, orthorhombic, minimum-image;
* **LJ** with a switching function from 1.00 to 1.05 nm, matching OpenMM's `LJ switch` form
  exactly;
* **electrostatics by smooth PME** — real space with `erfc`, reciprocal space by FFT, self term,
  and the intramolecular exclusion correction;
* **rigid SPC/E** by analytic **SETTLE** for positions and **RATTLE** for velocities inside a
  BAOAB splitting.

> **Why PME rather than reaction field.** The usual argument for reaction field is cost, and here
> it does not apply: at `L ≈ 2.49 nm` with a 1.05 nm cutoff the real-space part dominates and is
> shared by both schemes, while the reciprocal part is a batched FFT on a ~24³ grid, which is
> cheap. PME is what the cited model uses, it is what NaCl will need for a charged solute, and
> the parity gate is satisfiable either way — so the approximation buys nothing and costs the
> literature claim. Reaction field is retained only as a **declared fallback** if PME parity or
> throughput fails, and adopting it would be an amendment, not a silent substitution.

**Deviation from §1's batching rule, with its reason.** §1 requires every batch packed to
`>= 2048` states because the deca step is launch-bound below that. Deca is 112 atoms in vacuum;
one methane box is **1538 sites** with PME, so the step is compute-bound at any batch size and
the rationale does not transfer. Batch size here is chosen by measured throughput, and the
measurement is reported. The compute policy itself is unchanged: **exactly one GPU at a time,
pinned with `CUDA_VISIBLE_DEVICES`, from GPUs 0–3.**

### 3.2 Gate — engine equivalence (binding, blocks everything downstream)

Against OpenMM 8.5.2 on the same configurations, with cutoffs, switching, PME parameters
(`alpha`, grid, spline order) and exclusion conventions matched explicitly rather than by
default:

| quantity | target |
|---|---|
| `V_torch` vs `V_openmm` | max rel. discrepancy `< 1e-6` |
| `grad V_torch` vs `grad V_openmm` | max rel. discrepancy `< 1e-6` |
| `xi_torch` vs `xi_openmm` | exact to float64 round-off |
| `f_local` (§2.2) both engines | max rel. discrepancy `< 1e-6` |
| `F_ABF` bias force both engines | max rel. discrepancy `< 1e-6` |

over at least 12 configurations spanning the sampled energy range, including contact
(`r = 0.34 nm`), barrier and dissociated (`r = 0.89 nm`) geometries, matching the deca precedent
(§6.1, which achieved 2.9e-8 / 8.7e-9).

Plus the two rigid-water clauses of Amendment 11.3: **constraint satisfaction** to `1e-8 nm` over
a production-length run, and **equipartition** at both `0.5` and `1 fs`.

**If the gate fails, methane does not run.**

### 3.3 Physical validation (after the gate, before any free energy)

* **bulk water** — 512 SPC/E waters alone, stable `T` and density, and `g_OO(r)` against the NIST
  SPC/E benchmark;
* **methane hydration** — one methane in the box, `g_MO(r)`; the first hydration shell must be
  physically sensible (the cited source identifies ~3.3 Å as an inner-shell scale). This is a
  sanity gate on the potential, **not** a curve-identity requirement.

---

## 4. Stage II — the reference, built twice and independently of ABF

> **SUPERSEDED IN PART BY AMENDMENT 12 (§12.2–12.3).** Constrained TI (§4.2) is now the
> **primary** reference — 29 points × 16 replicas (8 wet / 8 dry) × 250 ps × 3 builds = 348 ns —
> and umbrella + MBAR (§4.1) is demoted to a **sparse anchor** at the contact minimum, the
> desolvation barrier and the solvent-separated minimum. The layout, bracketing and acceptance
> machinery below still governs that sparse build. Reference acceptance now follows Amendment
> 12.2.

The preregistration's §4.5 reference-quality rule is binding, and the WCA experience is the reason
it is taken seriously: a cached TI reference sitting 0.264 rms from a three-replica consensus
halved a related contrast, and Stage A has since confirmed it wrong at `z ≈ 0.25` by 23σ.
Methane gets **two independent constructions** that must agree before either is used.

### 4.1 Reference A — umbrella sampling + MBAR

**Window centres strictly bracket the evaluation domain** (Amendment 11.4):

```
  evaluation domain   [0.33, 0.90] nm
  window centres      [0.30, 0.93] nm
  64 windows, spacing 0.01 nm (0.1 Å)
  U_k(q) = (kappa/2) (xi(q) - r_k)^2
```

**`kappa` is chosen by a preregistered smoke calibration, not by a formula**, following the
Amendment 1 procedure. Three candidate layouts, 8 replicas per window:

| layout | windows | spacing | `kappa` (kJ/mol/nm²) | `kappa` (kcal/mol/Å²) | predicted sd/spacing |
|---|---|---|---|---|---|
| **L1** | 64 on [0.30, 0.93] | 0.01 nm | 8368 | 20 | 1.72 |
| **L2** | 64 on [0.30, 0.93] | 0.01 nm | 14700 | 35 | 1.30 |
| **L3** | 43 on [0.30, 0.93] | 0.015 nm | 8368 | 20 | 1.15 |

Measured and reported for each: `sd/spacing`, minimum neighbour overlap, sampled `xi` range,
**fraction of samples outside the evaluation domain**, and the maximum displacement of a window's
mean `r` from its centre. Acceptance: sampled range brackets `[0.33, 0.90]` on both edges,
minimum neighbour overlap `>= 0.75`, and no window mean displaced by more than half a spacing.
Among layouts that pass, take the **cheapest**. L2 is quoted at deca's accepted `sd/spacing` and
is the expected winner; the choice is made on the measurement.

**Two initial families per window**, deliberately: one pulled **outward from contact** and one
pulled **inward from the dissociated pair**. This is a direct test for wet/dry hysteresis at the
desolvation barrier, and it is the same instrument that Gate 0 uses in §9. Windows whose two
families disagree beyond their block error are extended; the disagreement is reported per window,
never averaged away.

```
  per window, per family:  0.5 ns equilibration  +  2.0 ns production
  32 replicas per window
  3 independent builds, differently initialised
```

MBAR (`pymbar` 4.0.3) gives `p_ref(r)`, hence `F_ref = -beta^-1 log p_ref + C` and
`W_ref = F_ref + 2 beta^-1 log r + C'`.

**Stopping rule**, on the Amendment 2 model — checkpoints at 0.5, 1.0, 1.5, 2.0 ns per replica,
with

```
  ratio = (max pairwise L2 between builds) / (0.10 x consensus F span)
```

Production stops at the first checkpoint where **both** `ratio <= 0.5` **and** sampling
`>= 1.0 ns` per replica. If the rule never fires by 2.0 ns, the run completes and reports
`ratio`; `ratio > 1.0` means the reference is **not accepted** and is rebuilt longer or with more
windows. Every checkpoint is retained, so the §4.5 convergence-versus-compute trace is a
by-product.

### 4.2 Reference B — constrained mean-force / TI audit

On a coarser grid, `r = 0.34, 0.36, ..., 0.90 nm` (29 points), hold `xi` fixed and accumulate the
**same** `f_loc` estimator of §2.2:

```
  F_TI(r) = F_TI(r_0) + integral_{r_0}^{r} <f_loc>(s) ds
```

This is the instrument Amendment 9 identifies as the one that actually settles a conditional
question — "Gate 0 is a screen for when to run it, not a substitute for it" — so it is built into
the reference stage rather than kept in reserve.

### 4.3 Acceptance

The reference is frozen only when **all** hold:

```
  || F_umbrella - F_TI ||  <=  0.1 kT       over the thermally relevant region
  ratio (3 builds, §4.1)   <=  0.5
  || W'_hat - (F'_hat + 2/(beta r)) ||_inf  <  1e-10 kJ/mol/nm        (§2.3 identity)
  out-of-domain sample fraction reported
  finite-size gate of §1.3 passed
```

Failing any of these, **STOP and rebuild the reference.** No screen result is interpreted against
a reference that failed §4.5.

---

## 5. The orthogonal descriptor `n_gap` and `tau_perp`

At fixed `r`, mFR sees only `r`. It cannot distinguish two configurations with the same `r` and
different water structure, and Amendment 8 proves that no `xi`-only score can. The study therefore
needs an explicit orthogonal descriptor, and it is the physically motivated one: **water occupancy
between the methanes.**

### 5.1 Definition (frozen, smooth)

Let `m = (Q_1 + Q_2)/2` and `e = (Q_2 - Q_1)/r`. For each water oxygen `O_j`, with
`d_j = O_j - m`, decompose into `u_j = d_j . e` (axial) and `w_j = |d_j - u_j e|` (radial):

```
  n_gap(q) = sum_j  s_axial(|u_j| ; r/2)  *  s_radial(w_j ; R_cyl)
```

with rational switching functions of the SPC/E-standard form
`s(x; x_0) = (1 - (x/x_0)^6) / (1 - (x/x_0)^12)` and `R_cyl = 0.20 nm`. A smooth count is used
rather than a hard one so that `n_gap` is differentiable and its time correlation is not dominated
by boundary crossings.

Reading: `n_gap ≈ 0` is a dry, contact-like gap; `n_gap > 0` means water has inserted. `R_cyl` is
fixed here in advance and is **not** tuned; its only requirement is that the reference joint
`p_ref(n_gap | r)` is bimodal across the barrier, which is checked once and reported.

### 5.2 `tau_perp` by the fixed-`r` conditional-mixing experiment

At `r_k in {r_A, r_barrier, r_B}` — taken from **our own** reference (§7) — prepare independent
solvent families at the same `r`:

| family | preparation |
|---|---|
| `wet` | equilibrated at the solvent-separated minimum, then projected to `r_k` |
| `dry` | equilibrated at contact, then projected to `r_k` |
| `bulk` | equilibrated at `r_k` directly (control) |
| `hot` | solvent randomised and re-equilibrated briefly (destroyed cage) |

This is the WCA Gate 0 pool design (`scripts/audit_wca_gate0.py`), which is the instrument that
discharged Amendment 10's obligation, transplanted to methane. Hold `r = r_k` constrained,
propagate many replicas per family under independent noise, and measure

```
  D_TV( p_t(n_gap | r_k, family a),  p_t(n_gap | r_k, family b) )
  tau_perp(r_k) = inf { t : max over pairs (a,b) of D_TV  <=  0.2 }
  tau_perp      = max over k of tau_perp(r_k)
```

The `0.2` threshold is a judgement fixed here in advance.

**Gate D's twin experiment (§2.5) uses the same machinery** with `C_Y(t) = Corr(n_gap^(i),
n_gap^(j))` on exactly-duplicated pairs and `tau_perp = inf{t : C_Y <= 1/e}`. Amendment 10
records that Gate D's clone-decorrelation time and the estimator-side `tau_perp` are the same
quantity; **both are computed, and if they disagree by more than 2× that disagreement is reported
as an open finding rather than resolved by picking one.**

---

## 6. Stage III — plain ABF only

**No Fisher–Rao anywhere in this stage.** This is the stage that decides the study.

### 6.1 Initial conditions

Amendment 4 established that the initial condition decides Gate B outright. All walkers start in
the **contact basin `A`**, so discovery is a genuine question rather than defined out of
existence.

**Walkers must not be clones.** Each walker receives an independently equilibrated water
environment:

1. restrain `xi` at contact;
2. assign independent Maxwell–Boltzmann velocities;
3. equilibrate each replica **50 ps** independently;
4. release the restraint and begin the shared ABF run.

The equilibration is **outside** the ABF budget and accumulates no free-energy statistics. Without
it, "many walkers" would mean "many clones" at `t = 0` and the mechanism test would be
contaminated at exactly the point it matters.

**Declared bias:** a contact start makes discovery harder, so it can only push the classification
*toward* discovery-limited. It cannot manufacture an establishment-limited verdict — the direction
that would license mFR.

### 6.2 Estimator and bandwidth

The binned/kernel machinery of `alkanes.interval`, unchanged. Bandwidths
`h in {0.008, 0.012, 0.016} nm` are tested **on ABF-only and reference data only**, and `h` is
selected by a deterministic rule fixed here:

> the **smallest** `h` for which the deterministic convolution floor
> `|| K_h * F_ref - F_ref ||_L2` is below 25 % of the ABF error at `T`, and every bin in the
> thermally relevant region carries `>= abf_min_count` effective samples.

Then `h` is frozen. **`h` is never selected by whether mFR improves.**

### 6.3 Screen design

```
  N in {128, 256, 512} walkers
  T_run = 200 ps per walker
  8 independent ensemble seeds per N   (seeds 5000-5007)
```

> **AMENDMENT 12.5: the ladder is executed sequentially, `N = 512` first.** The set of `N` and
> every threshold are unchanged; only the order is fixed, with a frozen dominance rule that stops
> at `N = 512` unless the verdict is establishment-limited. A partial map is reported as partial.

The `N` ladder is the direct test of the two-timescale mechanism: for independent walkers
discovery should accelerate roughly as `T_hit ~ 1/N` while establishment should not, so raising
`N` is what *creates* the window `T_hit << T_est`. **The entire map is reported**, as §8.2
requires of NaCl. If several `N` pass every gate, take the **smallest** — mechanically, not by
looking at errors.

`N` is never chosen by an mFR result, and `T_run` is never raised to manufacture a deficit.

---

## 7. States from our own reference

Amendment 3's rule, unchanged, applied to `F_ref`:

1. locate all local minima on the frozen grid;
2. merge adjacent pairs whose separating barrier, measured from the **higher** minimum, is below
   `2 kT`; repeat to convergence;
3. if `>= 2` minima survive they are the states, with boundaries at the intervening maxima;
4. **fallback** — if one minimum survives, use the frozen equal-width tercile partition of
   `Omega_xi`, declared as a partition of the coordinate and not as a claim of metastability.

Expected, but **not** assumed: `A` = contact, `B` = solvent-separated, `C` = dissociated. The
literature basin positions are not imposed; they move with the water model and the unlike-pair
rule, and §1.1 declares ours.

---

## 8. Gate order

Amendment 10 fixes the order, and classification is **by the first failing gate**:

```
  Gate 0  : is the ABF conditional mean force trustworthy?        <- leads
  Gate A  : can the states be distinguished through xi?
  Gate B  : were they discovered?
  Gate C  : were they established?
  Gate D  : do clones decorrelate faster than they are replaced?
```

### Gate 0 — ABF baseline validity

Amendment 9 **retracted the span clause**; it does not apply and is not evaluated. What is
evaluated:

* **retained pinning clause** — no seed with `> 0.90` of walkers in one state over the whole
  second half (deca read 0.951–0.9996; R15, which is fine, read 0.46–0.74);
* **the controlled experiment**, which Amendment 9 identifies as the actual instrument: the
  cross-family spread of `<f_loc>` at fixed `r` from §5.2, against `|F'_ref|`.

```
  calibration:  WCA 0.040 (passes)      gateway 0.036 global / 0.189 in the constriction (passes, marginal)
                deca 0.61 (fails)       R15 beta=2  0.564 / 0.593 (fails)
```

**No numerical threshold is set on the cross-family spread**, deliberately, following
Amendment 9's refusal to set one after seeing R15's number. The verdict is made against the
calibration ladder above and is argued explicitly in the run's `RESULT.md`, with the numbers, in
the region responsible for the free-energy error.

If Gate 0 fails in that region, methane is **conditional-equilibration-limited**, whatever the
visitation looks like — and Amendment 8's theorem `d/dt p_t(y|xi)|_FR = 0` says no marginal
selection rule repairs it. **STOP.**

### Gate A — CV visibility

```
  max over relevant pairs of TV( p(n_gap | state a), p(n_gap | state b) )  >=  0.30
```

evaluated on the reference. Below it, marginal mFR provably cannot preferentially correct one
state over the other: **STOP, and it is a stop for the CV, never a licence to tune mFR.**

### Gate B — discovery

```
  T_hit,B  <  0.1 T_run = 20 ps    on >= 6 of 8 screening seeds
```

with a persistence criterion so a single-frame boundary touch is not counted as discovery.
Failing: **discovery-limited, STOP** — the R15 outcome, where reallocation converted a support
deficit into a diversity deficit and mFR failed on 0/8 seeds, oracle included.

### Gate C — establishment

Against the **bias-aware** target of §2.1 — a state can be rare at equilibrium and perfectly
populated under the current bias:

```
  Q*_k(t) = int_{C_k} exp(-beta[F_ref - B_t]) dr  /  int exp(-beta[F_ref - B_t]) dr
```

A state is under-established if its occupancy stays below `0.5 Q*_k(t)` for a contiguous span of
at least `0.20 T_run = 40 ps`, over the second half.

```
  T_hit < 0.1 T  and  persistent deficit   ->  establishment-limited;  CONTINUE to Gate D
  T_hit < 0.1 T  and  no deficit           ->  ABF-sufficient;         STOP
```

### Gate D — clone decorrelation

```
  lambda_rep = (total replacements) / (N T_active)
  lambda_rep * tau_perp  <=  0.1
```

If no **active** rate (§3.2 of the preregistration, `N_replacements >= 0.5 N`) satisfies it, that
is a **C3 failure: STOP**, reported as a predicted R15-type outcome.

---

## 9. Stage IV — mFR, only if every gate passes

Mechanism unchanged from the rest of the project — no methane-specific modification:

```
  S_t^i = log( p_hat_t(r_i) / q_hat_t(r_i) )  -  int p_hat_t log( p_hat_t / q_hat_t ) dr
```

`S > 0` over-represented (death candidate), `S < 0` under-represented (clone candidate).

**Rate calibration** follows §3 of the preregistration exactly: 4 held-out seeds (**5100–5103**),
a four-point ladder spanning `lambda_rep tau_perp ~ {0.01, 0.03, 0.10, 0.30}`, inactive rates
struck before selection, genealogy gates `ESS_anc/N >= 0.30` and `w_max <= 0.05`, then minimum
calibration `L2(F)` with the gentler rate taken on ties within 2 pp.

> **No numerical rate is transferred from WCA, the gateway or the toys**, and **`gamma` is never
> selected by whichever value gives the lowest `F` error.** Selection is by
> turnover/decorrelation/genealogy criteria only.

**Arms** — **DECIDED BY AMENDMENT 12.7: three arms first** (`abf`, `mfr_practical`, `mfr_sham`),
answering Q3. The prior-art arms `book_laplacian` and `count_balancing` run as a separate,
declared prior-art closure on the same frozen setting and the same seeds, with nothing retuned,
**conditional on a positive Q3** — and are reported whatever they show. **One sham per FR arm**,
replaying the *realized* replacement counts and event times of
its matched seed with uniformly random death/clone identities. **The direct arm-vs-sham contrast
is the attribution statistic.**

**Seeds** — 16 fresh confirmatory seeds **5200–5215**, none reused from the reference, the screen
or calibration. Every comparison is seed-paired.

---

## 10. Endpoints

**Primary** — `I_F = int_{t_0}^{T} || F_hat_t - F_ref ||_{L2(w)} dt`, with the additive constant
aligned by `alkanes.interval.align_additive_constant`.

**Reported in parallel** — `I_W` on the radial PMF. The two differ analytically (§2.3), so they
must tell a consistent story; **`sign(I_F effect) != sign(I_W effect)` is a red flag that stops
interpretation**, not a result to choose between.

**Mean force** — `I_F'` and `e_F'(T)`; this is what ABF actually learns.

**Physical secondaries**, preregistered so the result is chemically interpretable:

```
  Delta W_{A->B} = W(r_B) - W(r_A)          contact -> solvent-separated
  Delta W_dagger = W(r_barrier) - W(r_A)    desolvation barrier
  basin probabilities
```

**Mechanism** — `T_hit`, `T_est`, `tau_perp`, contact↔solvent-separated transitions, round trips,
`KL(p_hat_t || q_t)`, `TV(p_hat_t, q_t)`.

**Diversity** — `N_anc(t)`, `ESS_anc(t)`, `w_max(t)`, cumulative replacements, per-event
replacement fraction, score clipping fraction. Computed **globally and inside the
solvent-separated basin**: global ESS can look healthy while the rare basin descends almost
entirely from one lucky discovery. Also age-aware over windows of length `tau_perp`.

**Clone decorrelation** — for each clone event at `t_0`, `C_clone(t) = Corr(n_gap^(i)(t_0+t),
n_gap^(j)(t_0+t))` and `tau_clone`. This tests the mechanism condition directly rather than
inferring it from ESS.

**Conditional fidelity** — `TV( p_method(n_gap | r), p_ref(n_gap | r) )`.

> **A gain in `F` accompanied by a significant worsening of conditional fidelity is NOT a
> success.** Marginal flatness is not the objective.

**Frozen-bias validation** (§4.4, mandatory for any positive) — stop adaptation, freeze `B_T`,
**discard the adaptive population**, start fresh independent methane–water trajectories, no ABF
update and no birth–death, and reconstruct `F(r) = B_T(r) - beta^-1 log p_{B_T}(r) + C`. Budget
~25 % of the adaptive force-evaluation budget. This separates "a better learned bias" from "an
online estimator that counted cloned samples repeatedly".

**Equal compute** — the primary comparison is **equal wall-clock on identical hardware**; ABF
receives additional MD steps until it matches the mFR budget including KDE/resampling overhead.
Equal force evaluations is the secondary.

---

## 11. Success and failure

**§4.3 of the preregistration is binding**, unchanged:

```
  median_s Delta_s             <=  -10 %
  95 % bootstrap CI upper end  <   -5 %
  #{ Delta_s < 0 }             >=  12 / 16
  ESS_anc / N                  >=  0.30
  w_max                        <=  0.05
  + 95 % CI of the direct mFR-vs-sham contrast below zero
```

Secondary label `STRONG POSITIVE` (Amendment 11.7) if additionally median `<= -15 %` vs ABF,
median `<= -10 %` vs sham, and `>= 13/16`.

Novelty (Q1), if the prior-art arms are run: `median (I_F(mFR) - I_F(prior)) / I_F(prior)
<= -5 %` with 95 % CI `< 0` against **both** baselines. A tie is reported as a tie, tested by
**TOST**, never by "no CI excluded zero".

### 11.1 The result that would matter

Not `I_F(mFR) < I_F(ABF)` on its own. The scientifically strong outcome is

```
  T_hit << tau_perp << T_est,  then on fresh seeds  I_F(mFR) < I_F(ABF) and < I_F(sham),
  while the round-trip increase stays small and genealogy stays healthy.
```

That would say the same thing WCA says, in a real molecular solvent: **FR does not principally
make the rare transition happen; once the physical dynamics has found a useful solvent
configuration, FR establishes enough population there for ABF to learn the mean force faster.**

### 11.2 Stop conditions — every one is a reported result

| condition | verdict |
|---|---|
| engine-equivalence gate fails | methane does not run; engine defect reported |
| reference acceptance fails (§4.3) | STOP, rebuild reference; no screen interpreted |
| finite-size gate fails at every `r` | domain truncated, or methane withdrawn |
| Gate 0 fails in the error-carrying region | **conditional-equilibration-limited**; STOP |
| Gate A fails | CV-visibility negative; STOP — a stop for the CV, not for mFR |
| Gate B fails | discovery-limited; STOP |
| Gate C finds no persistent deficit | **ABF-sufficient — the preregistered likely outcome**; STOP |
| Gate D admits no active safe rate | C3 failure; STOP |

**If methane fails, the next benchmark is not a retuned methane.** Per §9 of the preregistration,
neither the temperature nor the potential nor the run length is adjusted until it passes. The
preregistered successor is the next member of the published hydrophobe-size series — ethane, then
propane — which changes solvent-coupled difficulty in a literature-backed way rather than
constructing a favourable benchmark.

---

## 12. Budget and compute

**Compute policy unchanged: exactly one GPU at a time from GPUs 0–3, pinned with
`CUDA_VISIBLE_DEVICES`.** Per Amendment 12.4, GPUs 0, 1 and 2 carry other users' processes and
methane runs on **GPU 3**, with the device's idle state re-checked before each stage and recorded
with every throughput number. A contended device reads ~28× slow and is indistinguishable from a
code defect if not checked.

Measured throughput, frozen 1538-site system, one replica, verified-idle H200 NVL:

| platform | ms/step | ns/day |
|---|---|---|
| OpenMM CUDA, mixed | 0.093 | 462 |
| OpenMM CUDA, double | 0.125 | 345 |
| OpenMM OpenCL | 0.26 | 169 |
| OpenMM CPU | 13.9 | 3.1 |
| OpenMM Reference | 60.3 | 0.7 |

Engine split (Amendment 12.4): **OpenMM CUDA** carries the parity oracle, the NPT box and the
constrained-TI reference (~348 ns ≈ 18 h serial); the **batched torch sampler** carries the ABF
screen and every population arm, because `N` walkers sharing one estimator with birth–death is
days per seed as serial OpenMM contexts.

Physical budget, frozen:

| stage | aggregate MD |
|---|---|
| box equilibration (NPT) | 1.5 ns |
| reference A — umbrella | 64 windows × 2 families × 32 replicas × 2.5 ns × 3 builds |
| reference B — constrained TI | 29 points × replicas × production |
| finite-size gate (1024 waters) | 3 points |
| screen | 3 `N` × 8 seeds × `N` × 200 ps |
| `tau_perp` / Gate 0 pools | 3 `r` × 4 families × replicas |
| production (only if licensed) | arms × 16 seeds × `N` × 200 ps, + 25 % frozen-bias |

> **The physical budget is frozen; its cost is not yet known.** No throughput measurement for a
> 1538-site PME system in this codebase exists, because the engine does not exist. Stage I
> **measures and reports** ms/step against batch size, exactly as §1 of the preregistration
> tabulates for deca, and the cost follows from the measurement.
>
> **If the measured cost makes the frozen budget infeasible, that is a redesign recorded as an
> amendment — not a quiet reduction of the physics.** Reducing replicas, shortening windows or
> trimming the `N` ladder after seeing that the full design is expensive is exactly the move this
> document exists to prevent.

---

## 13. References

**Direct ABF precedent.** Garnier, Devémy, Bonal & Malfreyt, *Calculations of potential of mean
force: application to ion-pairs and host–guest systems*, **Mol. Phys.** 116, 1998–2008 (2018),
`10.1080/00268976.2018.1442593` — applies ABF, TI, finite-difference TI and umbrella sampling to
methane association in water.

**Concrete model.** Asthagiri, Merchant & Pratt, *Role of attractive methane-water interactions in
the potential of mean force between methane molecules in water*, **J. Chem. Phys.** 128, 244512
(2008), `10.1063/1.2944252` — united-atom methane, `eps = 0.294 kcal/mol`, `sigma = 3.73 Å`,
SPC/E, 512 waters, 298 K, PME. Cited for the model and protocol only; see §1.1 for the declared
unlike-pair deviation, and Amendment 11.8 for a timestep claim **not** carried forward.

**Hydrophobic-association benchmark.** Sobolewski, Makowski, Czaplewski, Liwo, Ołdziej &
Scheraga, *Potential of Mean Force of Hydrophobic Association: Dependence on Solute Size*,
**J. Phys. Chem. B** 111, 10765–10774 (2007), `10.1021/jp070594t` — establishes the methane/alkane
PMF family, and defines the preregistered successor series if methane fails.

**SPC/E.** Berendsen, Grigera & Straatsma, *The Missing Term in Effective Pair Potentials*,
**J. Phys. Chem.** 91, 6269–6271 (1987). NIST maintains SPC/E benchmark data used for the §3.3
code validation.

**Mathematical framework.** Lelièvre, Rousset & Stoltz, *Free Energy Computations: A Mathematical
Perspective*, Imperial College Press, 2010 — Chapters 3 and 5 for the reaction-coordinate free
energy, the conditional mean force and ABF.

---

## 14. Declared deviations, collected

Every deviation from the literature or from the preregistration, in one place, so none has to be
found by a reader:

1. **Lorentz–Berthelot methane–oxygen mixing rule** (§1.1) — our implementation choice; the cited
   source's unlike-pair rule could not be verified.
2. **NVT at the NPT-equilibrated volume** rather than NPT production (§1.3) — so the target is
   exactly the canonical measure with no barostat variable.
3. **`dt = 0.5 fs`** (Amendment 11.8) — our conservative choice justified by our own parity test,
   not by the citation originally attached to it.
4. **Batch size not packed to 2048** (§3.1) — the §1 rule addresses a launch-bound 112-atom vacuum
   step; a 1538-site PME step is compute-bound at any batch size.
5. **Execution order** — methane runs ahead of NaCl (Amendment 11.2).
6. **Evaluation domain may be truncated once**, before any `F_ref` exists, if the finite-size gate
   of §1.3 fails.
