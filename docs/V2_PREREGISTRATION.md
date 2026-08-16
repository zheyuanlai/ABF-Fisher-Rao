# v2 preregistration — does the Fisher–Rao direction beat prior directed selection, and does the establishment mechanism transfer?

**Status: FROZEN at the commit carrying this file. No v2 production dynamics may run before it.**

v1 is closed and frozen at `v1-regime-map-final` (`CLOSURE_v1.md`). Nothing in v2 alters a v1
raw artifact, a v1 number, or a v1 conclusion. v2 is an additive campaign on a new branch,
`v2-campaign`.

This document fixes, in advance, everything that a later look at an mFR result could otherwise
bias: which systems run, in what order, under what physical model, at what budget, how the
regime is classified, what counts as success, and what causes a benchmark to stop. The single
principle carried forward from the gateway study, which is what made it convincing:

> **The regime is diagnosed without looking at an mFR result.**

---

## 0. What v2 is asking

| | Question |
|---|---|
| **Q1** | Is the Fisher–Rao direction better than the directed-selection rules ABF already had? |
| **Q2** | Does that distinction survive on a published molecular system where selection-enhanced ABF is already known to help? |
| **Q3** | Does the establishment-limited mechanism transfer to an independent explicit-solvent atomistic system? |

Q1 is the highest-value question in the queue and the one v1 never asked. v1 measured mFR
against ABF and against matched *random* turnover. It never measured it against a *directed*
alternative. Chapter 6 of Lelièvre–Rousset–Stoltz applies selection to ABF on the WCA dimer
itself, with a Laplacian rule and a simpler density-balancing remark; Comer et al. ship a
`1/(bin count + 1)` balancing rule in NAMD. If plain count balancing matches the Fisher–Rao
score, the claim v2 can make is *"FR is a principled selection rule consistent with the
establishment mechanism"*, not *"FR improves on prior ABF selection"* — and v2 must be able to
tell those apart.

### Execution order

```
1.  deca-alanine        Stage 0 -> reference -> Gate A -> ABF-only screen -> production if licensed
2.  WCA Case IX         re-run against a high-precision TI reference          [queued, see §7]
3.  NaCl / water        engine -> engine-equivalence gate -> fixed-compute ABF map -> production if licensed
4.  methane / water     ABF-only screen only; preregistered likely-null       [optional]
5.  phi^4               T_hit(N) vs T_est(N) scaling                          [optional, cheap]
```

Experiment 0 of the source plan (WCA prior-art closure) is **skipped as a gating step** by
explicit instruction. §7 records what that leaves open and what is queued instead.

---

## 1. Compute policy

The node is shared. **Exactly one GPU at a time**, pinned with `CUDA_VISIBLE_DEVICES`, chosen
from an idle device among GPUs 0–3. At freeze time GPU 2 is idle (0 MiB) and GPUs 0, 1 and 3
are in use by others; v2 runs on **GPU 2**. If every GPU is occupied by others, v2 may expand
to **two** GPUs, and no further. One process per GPU: packing several processes onto one device
gave no speedup in the WCA study and will not here.

Measured throughput, deca-alanine, H200 NVL, float64, full BAOAB step including CV geometry,
binned accumulation and Cartesian bias force:

| batch `B` | ms/step | aggregate ns/day |
|---|---|---|
| 512 | 2.01 | 22 000 |
| 2048 | 2.03 | 87 000 |
| 4096 | 3.35 | 106 000 |
| 8192 | 6.46 | 110 000 |

The step is launch-bound below `B ≈ 2048` and compute-bound above it, so **every batch is
packed to at least 2048 states** — all seeds and all arms of a stage share one batch, one
process and one noise stream, as in the v1 gateway confirmatory design. `torch.compile` is
mandatory (6.6× at `B = 2048`) and is gated as a performance-only change by
`tests/test_deca_stage0.py::test_compiled_forces_are_numerically_indistinguishable_from_eager`.

---

## 2. Universal gates

Every new benchmark passes through the same three gates, in this order, before any mFR arm is
permitted. All three are evaluated from **ABF-only** data and a reference; none may consult an
mFR result.

### 2.1 Bias-aware target population

At time `t` ABF has learned a bias `B_t`, so the correct population target for region `C_k` is
not its unbiased equilibrium mass but

```
                integral over C_k of exp(-beta [F_ref(z) - B_t(z)]) dz
  Q*_k(t)  =  ---------------------------------------------------------
                  integral over all z of exp(-beta [F_ref(z) - B_t(z)]) dz
```

A state can be rare under the unbiased ensemble and perfectly populated under the current bias.
This is the quantity the valine study established; v2 reuses it unchanged.

### 2.2 Gate A — CV visibility (run FIRST)

Before asking whether a state is starved, ask whether its deficit can be seen by the marginal
`p_t^xi` at all. Construct richer structural labels `Y` and compare the conditional densities
`p(xi | Y = a)` against `p(xi | Y = b)` for the relevant states.

**Rule.** For each pair of relevant structural states `(a, b)`, compute the total-variation
distance between `p(xi | Y = a)` and `p(xi | Y = b)` on the frozen evaluation grid. If

```
  max over relevant pairs of  TV( p(xi | Y=a), p(xi | Y=b) )  <  0.30
```

then the states are not separated by the collective variable and **marginal mFR cannot
preferentially correct one over the other**.

> **STOP — the CV fails, not the method.** This is a stop for the proposed CV. It is never a
> licence to tune mFR harder. This is the pentane `phi_1` lesson.

The 0.30 threshold is a judgement, fixed here in advance rather than after seeing the number.
It is deliberately permissive: it stops only near-total overlap.

### 2.3 Gate B — discovery

For each relevant state `k`, with `T` the per-walker run length:

```
  T_hit,k  <  0.1 T     on at least 6 of 8 ABF-only screening seeds
```

**If not satisfied: discovery-limited. STOP. Do not run mFR.** Running it would repeat the
pentane `R15` experiment, where reallocation converted a support deficit into a diversity
deficit and mFR failed on 0/8 seeds — the oracle too.

### 2.4 Gate C — establishment

A state is **under-established** if its occupancy stays below `0.5 Q*_k(t)` for a contiguous
span of at least `0.20 T`, evaluated over the second half of the run.

```
  T_hit < 0.1 T  and  no persistent deficit   ->  ABF-sufficient;        STOP
  T_hit >= 0.1 T                              ->  discovery-limited;     STOP
  T_hit < 0.1 T  and  persistent deficit      ->  establishment-limited; CONTINUE
```

These are the thresholds the dipeptide campaign already used. Valine failed them decisively
(all states discovered by 5.4 ps and established by 52 ps of a 300 ps run, worst second-half
relative deficit 0.223), which is the calibration for what "clearly ABF-sufficient" looks like.

### 2.5 Gate D — clone decorrelation (new in v2)

This is what separates WCA from aggressive `R15`, and it is the gate v1 lacked. Selection must
be slow enough that the dynamics decorrelates siblings between events; otherwise the procedure
amplifies a few replicas and destroys information.

**Twin experiment.** Take configurations in the region where cloning will occur, duplicate them
exactly (`q1_0 = q2_0`), evolve both under independent thermal noise with the same frozen or
representative ABF bias and **no birth–death**. For an orthogonal descriptor `Y(q)`:

```
  C_Y(t)   = Corr( Y(q1_t), Y(q2_t) )
  tau_perp = inf { t : C_Y(t) <= 1/e }
```

**Hazard.** `lambda_rep = (total replacements) / (N * T_active)`.

**Safety condition.**

```
  lambda_rep * tau_perp  <=  0.1
```

The constant is conservative rather than theoretical: roughly a tenfold separation between the
clone-mutation timescale and the next expected replacement. If no *active* rate (§3.2) can
satisfy it, that is a **C3 failure: STOP**, and it is a predicted `R15`-type outcome, reported
as such.

---

## 3. Rate calibration

**No numerical `fr_rate` or selection intensity is transferred from WCA, the gateway or the
entropic toy to a new system.** The v1 WCA rate ladder already shows why: raising the rate
eventually collapses ancestor ESS and reverses the accuracy gain.

### 3.1 Calibration set

**4 held-out calibration seeds per system. None may appear in confirmatory production.**
Seed blocks are fixed in §5 so the two sets cannot silently overlap.

### 3.2 Four-point ladder and the activity requirement

Four logarithmically spaced rates, whose numerical values are chosen by a **short preregistered
smoke test** so that they span

```
  lambda_rep * tau_perp  ~  0.01,  0.03,  0.10,  0.30
```

The last rung is deliberately expected to approach or exceed the safe region.

**A rate may not win by turning mFR off.** Require median cumulative turnover over the active
part of the run

```
  N_replacements  >=  0.5 N
```

Rates failing this are **inactive** and are struck from the ladder before selection, not
compared and rejected. Without this clause an adverse ladder selects the arm that does nothing.

### 3.3 Genealogy health gates

Carried over unchanged from the gateway confirmatory preregistration:

```
  ESS_anc / N  >=  0.30        w_max  <=  0.05
```

### 3.4 Selection rule

Among rates that are **active**, **decorrelation-safe** and **genealogy-safe**, minimise
calibration integrated `L2(F)`. If two candidates differ by less than **2 percentage points**,
take the **gentler** one. Then freeze it. Baseline selection intensities `c` for the prior-art
arms (§4.2) are tuned by the same procedure on the same calibration seeds; the mFR
configuration is *not* retuned when a baseline is added.

---

## 4. Arms and metrics

### 4.1 Common metric schema

Every production run emits the same quantities.

**Primary endpoint.**

```
  I_F  =  integral from 0 to T of  || F_hat_t - F_ref ||_{L2(Omega)}  dt
```

**Accuracy secondaries.** `e_F(T) = ||F_hat_T - F_ref||_L2`; `I_F'` and `e_F'(T)`, the same on
the mean force.

**Mechanism.** `T_hit`, `T_est`, per-state occupancies, `TV(p_hat_t, q_t)`, `KL(p_hat_t || q_t)`,
transition counts, round trips.

**Diversity.** `N_anc(t)`, `ESS_anc(t) = 1 / sum_a w_a(t)^2`, `w_max(t)`, cumulative
replacements, per-event replacement fraction, score clipping fraction.

**Conditional fidelity.** For the system's orthogonal descriptor `Y`,

```
  TV( p_method(Y | xi),  p_ref(Y | xi) )
```

> **A gain in `F` accompanied by a significant worsening of conditional fidelity is NOT counted
> as a success.** Marginal flatness is not the objective. This is the deca-alanine analogue of
> the pentane conditional-TV check.

### 4.2 Arms

Every arm shares the physical model, walker count `N`, starting population, run length, ABF
estimator, seeds and force-evaluation budget.

| Arm | Purpose |
|---|---|
| `abf` | baseline (shared multiple-walker ABF) |
| `mfr_practical` | the proposed method, estimated target |
| `mfr_sham` | matched-turnover random-direction control, **one sham per FR arm** |
| `book_laplacian` | historical directed selection, `S = c * d2p/dz2 / p` |
| `count_balancing` | strongest simple alternative, `S = c * (1 - p_hat/p_bar)` |

Two v1 method rules are binding here: **one sham per FR arm**, and **the direct arm-vs-sham
contrast is the attribution statistic** (not "no CI excluded zero").

### 4.3 Success rules

Per paired seed `s`, with negative better:

```
  Delta_s = 100 * ( I_F,s(method) - I_F,s(ABF) ) / I_F,s(ABF)
```

**Primary (16-seed molecular confirmatory).** mFR is a meaningful positive only if **all** hold:

```
  median_s Delta_s              <=  -10 %
  95 % bootstrap CI upper end   <   -5 %
  #{ Delta_s < 0 }              >=  12 / 16
  ESS_anc / N                   >=  0.30
  w_max                         <=  0.05
```

This is the 32-seed gateway rule translated to 16 seeds.

**Attribution.** Additionally require the 95 % CI of the direct mFR-vs-sham contrast to lie
below zero.

**Novelty (Q1).** To claim FR is *better than* prior directed selection:

```
  median ( I_F(mFR) - I_F(prior) ) / I_F(prior)  <=  -5 %      with 95 % CI < 0
```

against **both** `book_laplacian` and `count_balancing`. If mFR instead ties them, that is
reported as a tie, and the claim becomes "a principled selection rule consistent with the
establishment mechanism". **Equivalence is tested by TOST**, never by "no CI excluded zero".

### 4.4 Frozen-bias validation for every positive

Any positive gets the estimator-independent check: stop adaptation at `T`, freeze each arm's
`B_T`, **discard the final walker population**, start an identical fresh population for every
arm, no ABF update, no birth–death, sample under the frozen bias, and reconstruct
`F(z) = B_T(z) - beta^-1 log p_{B_T}(z) + C`. Budget: **~25 % of the adaptive-stage
force-evaluation budget**; it is a secondary check and need not match the adaptive budget.

### 4.5 Reference-quality rule

**No arm-level effect is trustworthy unless the reference uncertainty is much smaller than the
effect discussed.** For every numerical reference, v2 provides: at least **three** independently
initialised reference calculations; block/bootstrap uncertainty; pairwise `L2` discrepancy among
replicas; a convergence plot against reference compute; a fixed support and evaluation domain;
and propagation of reference uncertainty into the primary `I_F` sensitivity analysis.

This is in the generic protocol because of what the v1 parallel audit found on WCA — a cached TI
reference sitting 0.264 rms from a three-replica high-precision consensus, about 10× the arm
effect, halving a related contrast from −4.75 % to −2.41 %.

---

## 5. Seeds

Fixed here so calibration and production cannot overlap.

| System | screening (ABF-only) | calibration | confirmatory production |
|---|---|---|---|
| deca-alanine | 3000–3007 (8) | 3100–3103 (4) | 3200–3215 (16) |
| NaCl / water | 4000–4007 (8) | 4100–4103 (4) | 4200–4215 (16) |
| methane / water | 5000–5007 (8) | — | — |
| C60 / water | 7000–7007 (8) | 7100–7103 (4) | 7200–7215 (16) |
| WCA Case IX re-run | — | — | reuses v1 seeds 400–415 |

---

## 6. System 1 — deca-alanine

### 6.1 Physical model (frozen)

| | |
|---|---|
| Molecule | Ace-(Ala)10-Nme, **112 atoms** |
| Force field | AMBER **ff14SB**, vacuum, `NoCutoff`, **zero constraints** |
| Thermostat | BAOAB Langevin, `T = 300 K`, `gamma = 1 ps^-1`, `dt = 1 fs` |
| Terms | 111 bonds, 198 angles, 303 torsions, 575 nonbonded exceptions; total charge 5.6e-16 e |
| Engine | `deca.engine.DecaEngine` over `alanine.forcefield.TorchFF`, `torch.compile`d |
| Parity | vs OpenMM: max rel energy **2.9e-8**, max rel force **8.7e-9** over 12 configurations spanning 100–22 288 kJ/mol |

`dt`, `gamma` and `T` match the alanine and valine studies exactly, so the integrator is not a
new variable.

> **DECLARED DEVIATION FROM THE LITERATURE.** Minoukadeh–Chipot–Lelièvre (JCTC 6:1008, 2010) and
> Comer et al. (JCTC 10:5276, 2014) use CHARMM in NAMD. v2 uses ff14SB in a batched torch
> sampler, because ff14SB carries no CMAP term and is therefore covered *exactly* by the energy
> path that already passed the alanine parity gate; CHARMM36 would require implementing and
> validating a bicubic CMAP term first. **The claim v2 may make is therefore: same molecule,
> same collective variable, same finite budget, a different modern protein force field — not a
> reproduction of the published PMF.** Closing this gap means implementing CMAP and is recorded
> as open work, not quietly assumed away.

### 6.2 Collective variable

```
  xi(q) = | r(C of ACE)  -  r(C of Ala10) |          atoms 4 and 104
```

the distance between the terminal carbonyl carbons — the end-to-end coordinate the deca-alanine
ABF literature uses. Analytic geometry and generalized mean force come from
`alkanes.distance_cv.DistanceCV`, validated against autodiff.

**Domain, from measurement.** An unbiased 50 ps run at 300 K from 256 helical and 256 extended
starts gives `xi` in [0.34, 3.66] nm, with the extended half collapsing to ~1.63 nm within
50 ps and the helical half sitting at ~1.68 nm. Frozen evaluation domain and soft walls:

```
  R_lo = 1.20 nm    R_hi = 3.60 nm    walls at 1.25 / 3.55 nm    n_grid = 129  (ODD)
```

Identical walls on every arm, as in the `R15` study. `n_grid` is odd so no Nyquist row exists.
`k_wall` is set by the preregistered smoke test of §3.2 and recorded before the reference run.
Below 1.20 nm the termini are in contact and `xi` no longer resolves conformation; that region
is excluded by the wall, not by post-hoc filtering.

### 6.3 Reference

**1-D umbrella sampling + MBAR**, independent of ABF by construction.

```
  64 windows, uniformly spaced on [1.20, 3.60] nm
  harmonic restraint k = 2000 kJ/mol/nm^2   (window sd ~ 0.035 nm, matched to spacing 0.038 nm)
  32 replicas per window, 4 ns each  ->  8192 ns aggregate per build
  3 independent builds, differently initialised  ->  24 576 ns total
```

Estimated cost at `B = 2048`: **~2.3 h per build, ~6.8 h for all three**. Aggregate sampling is
~64× the literature's 128 ns standard, which is what the reference-quality rule of §4.5 demands.
Windows are seeded from a deliberately diverse pool (helical, extended, and collapsed
structures) because vacuum deca-alanine's difficulty is *hidden* conformational structure, and a
reference seeded only from the helix would inherit exactly the bias under test.

**Reference acceptance:** all three builds agree within the §4.5 criteria, and the reference
uncertainty is small against a 10 % effect on `I_F`. Failing that, the reference is rebuilt with
more windows or longer windows before any screen result is interpreted.

### 6.4 Structural labels `Y` (for Gate A and conditional fidelity)

Computed from the frozen definitions:

* number of `i -> i+4` backbone hydrogen bonds (`O_i · H_{i+4} < 0.25 nm`);
* per-residue backbone basin (`alpha`, `beta`, `alpha_L`) from `(phi, psi)`;
* fraction of residues in `alpha`;
* radius of gyration;
* RMSD to the ideal `(-57, -47)` helix.

`Y` for Gate A is the pair (helical H-bond count bucket, `alpha` fraction bucket).

### 6.5 Budget

The **historically meaningful** budget, not one invented around mFR:

```
  16 walkers  x  0.5 ns  =  8 ns aggregate per ensemble
  8 independent ABF-only screening ensembles      (seeds 3000-3007)
```

At `B = 8 x 16 = 128` states and 500 000 steps this is ~17 min of GPU time. The screen is cheap;
its role is to *decide*, and it decides before any mFR arm exists.

### 6.6 Stop conditions

* Gate A fails (max pairwise TV `< 0.30`) → **STOP**, report as a CV-visibility negative.
* Gate B fails → **STOP**, discovery-limited.
* Gate C finds no persistent deficit → **STOP**, ABF-sufficient.
* Gate D admits no active safe rate → **STOP**, C3 failure.
* Reference acceptance fails → **STOP** and rebuild the reference; no screen result is
  interpreted against a reference that failed §4.5.

A stop is a result and is reported as one. Deca-alanine is where prior-art selection is
strongest, so it is explicitly **not** preregistered as a predicted positive.

---

## 7. System 2 — WCA Case IX against a high-precision reference

`CLOSURE_v1.md` §5a records the v1 headline's largest open exposure: the **−22.83 %** Case IX
contrast is scored against a cached TI reference that a parallel audit found inflates a related
WCA contrast by roughly 2×. That audit measured a *different endpoint* on *different arms* in a
*different run tree*, so it refutes nothing directly — but whether the same inflation applies to
integrated `L2(F)` at cell `b1_h2` **has never been measured**.

**A cheap re-score is not available.** The Case IX raw artifacts store the already-scored
`l2_f_t` scalars and `final_pmf` only — not `F_hat_t(z)` over time — so changing the reference
requires re-running the dynamics.

**Queued behind deca-alanine**, by instruction. Design, frozen now:

* build a consensus high-precision constrained-TI reference for cell `b1_h2` from **5**
  independently initialised replicas (the audit line used 3 at 960 k samples per `z`; its work
  survives in the worktree at `.claude/worktrees/abf-fisher-rao-audit-fdbcfb`);
* accept it under §4.5;
* re-run the 16 Case IX seeds (400–415) with **nothing retuned**, scoring against both the
  cached and the high-precision reference;
* report both, and propagate reference uncertainty into the contrast.

Preregistered readings: the effect survives intact; the effect shrinks but stays negative
(mechanism survives, effect size revised); or the effect vanishes (WCA no longer supports a
physical positive). **All three are reported. None is hidden.**

---

## 8. System 3 — NaCl in water

Gated behind a new engine and its equivalence gate; nothing here runs until both pass.

### 8.1 Engine

v1 has no periodic, solvated sampler. v2 extends the batched `(seed, walker, atom)` torch design
with periodic boundaries, a validated electrostatics treatment, and rigid-water constraints.
Before **any** NaCl production:

**Engine-equivalence gate** against OpenMM on the same configurations —

```
  V_torch ~ V_openmm      grad V_torch ~ grad V_openmm      xi_torch = xi_openmm
  f_local,torch ~ f_local,openmm                            F_ABF,torch ~ F_ABF,openmm
```

> An approximate reimplementation is **not** a literature reproduction and will not be called
> one. If the gate fails, NaCl does not run.

### 8.2 Fixed-compute regime map

Total simulated MD time is fixed at the literature ABF budget and **`T` is never chosen because
it produces a deficit**:

```
  B_MD = N * T = 100 ns        N in {8, 16, 32, 64}   ->   T in {12.5, 6.25, 3.125, 1.5625} ns
```

Every cell costs the same total MD time. This is a direct test of the two-timescale theory:
`T_hit ~ 1/(N k01)` should accelerate with `N` while `T_est ~ 1/(k01 + k10)` should not.

Run **ABF only**, 8 seeds × 4 `N`-values, and **report the entire map**. Do not select the cell
with the largest ABF error. If several cells pass every gate, choose mechanically: **the
smallest `N` that satisfies every gate**. If none does:

> **NaCl is not an mFR candidate under the preregistered budget. STOP.**

**The eligibility decision is made by the ABF-only data, never by a Kramers estimate.** The
ABF bias makes the rates time-dependent, `k_ij = k_ij[B_t]`, so an equilibrium barrier is a
prior, not a decision.

### 8.3 Solvent coordinates

The Na–Cl distance is not the complete dynamical coordinate. Record `n_coord(Na)`,
`n_coord(Cl)` and `n_bridge`, and use them for the conditional fidelity
`p(n_coord, n_bridge | r_NaCl)` and for the twin estimate of `tau_perp` in Gate D.

---

## 9. Systems 4 and 5 — optional

**Methane pair in water.** Preregistered **likely-null / falsification benchmark**, not "the
next positive". ABF-only screen first; if ABF is sufficient, **STOP and report that prediction**
as a literature-anchored negative control. The run length is **not** tuned until it passes.

**phi^4 scaling.** Cheap, tensor-native, no mFR arm. At fixed physical `T` and
`N in {32, 64, 128, 256, 512, 1024}`, measure `T_hit(N)`, `T_est(N)` and their ratio. The
two-state reading predicts `log T_hit = -log N + C` with a much flatter establishment curve —
direct evidence for *why* the mFR window exists, independent of any mFR result. Runs only if
time remains.

---

## 10. Explicitly dropped

Rotated Wolfe–Quapp; another bi-channel toy; ethanol interface transfer; membrane permeation;
cavity–ligand binding; chignolin. The first two duplicate mechanisms v1 already has; the
interface is not cleanly described by the two-state establishment story; membranes and cavities
introduce very slow environmental coordinates that would need a separate project to interpret a
negative; chignolin is far too expensive for what it would add now.

---

## 11. What each outcome would mean

Fixed in advance so no result can be re-narrated after the fact.

| Result | Conclusion |
|---|---|
| mFR beats ABF + sham + book selection + count balancing | the FR score adds something genuinely new to classical selection |
| mFR ties book/count selection | FR is a principled selection rule; superiority is **not** established |
| deca-alanine fails Gate A | marginal mFR provably cannot repair a conditional deficit — a clean CV-visibility negative |
| deca-alanine is ABF-sufficient | a literature-anchored negative where prior art reports selection helping; the discrepancy is the result |
| WCA effect survives the high-precision reference | mechanism and effect size both stand |
| WCA effect shrinks but stays negative | mechanism survives, effect size revised |
| WCA effect vanishes | WCA no longer supports a physical positive; **say so** |
| NaCl establishment-limited and mFR wins | strongest independent molecular validation |
| NaCl establishment-limited, mFR harms, clones stay correlated | direct experimental validation of the Gate D condition |
| NaCl ABF-sufficient | the screen did its job; the Kramers prior was wrong |
| methane ABF-sufficient | useful literature-anchored negative control |
| `T_hit ~ N^-1`, `T_est ~ N^0` | direct validation of the two-timescale mechanism |

---

## 12. Amendment procedure

Any change to this document after freeze is a numbered amendment appended below, stating what
changed, why, and **whether any result had been seen at the time**. v1's Amendment 2 to the
gateway confirmatory preregistration is the template: it fixed in advance what to quote when
both passes succeeded, before either had been read.

### Amendment 1 — deca-alanine umbrella window layout (2026-08-11)

**No arm result, no screen result and no reference PMF had been seen. Nothing downstream of the
reference existed at the time of this change.**

§6.3 froze 64 windows on `[1.20, 3.60]` at `k = 2000 kJ/mol/nm²`. A 12-window smoke run appeared
to show that layout sampling only up to 3.49 nm, leaving the top of the evaluation domain
uncovered. **That appearance was an artifact of the smoke run's 4 000-step pull, and the
calibration says so.** With the preregistered 20 000-step pull, the frozen layout covers the
domain on both edges (sampled `xi` in `[1.089, 3.643]`) with healthy neighbour overlap. The
original motivation for changing anything was therefore wrong, and is recorded as wrong.

Measured on three candidate layouts, 8 replicas per window, 20 k pull / 30 k equilibration /
60 k production:

| layout | spacing | sd/spacing | min neighbour overlap | sampled range | cis bonds |
|---|---|---|---|---|---|
| **frozen** 64w `[1.20, 3.60]` k=2000 | 0.0381 | 1.33 | 0.785 | [1.089, 3.643] | 0/512 |
| **cand-A** 96w `[1.10, 3.80]` k=3200 | 0.0284 | 1.31 | 0.821 | [1.003, 3.725] | 0/768 |
| **cand-B** 128w `[1.10, 3.80]` k=6000 | 0.0213 | 1.15 | 0.830 | [1.040, 3.750] | 1/1024 |

All three are viable. One **genuine** defect survives the correction, and it is the only reason
this amendment exists: in the frozen layout the highest window centre coincides with `R_hi`, and
under the steep end of the PMF it displaces downward by 0.105 nm, so the top bin of the
evaluation domain rests on the tail of a single displaced window rather than sitting between two
windows. That is a structural asymmetry in the estimator, not a tuning preference.

**Change:** deca-alanine umbrella windows become

```
  96 windows uniformly on [1.15, 3.70] nm      k_umbrella = 3200 kJ/mol/nm^2      n_rep = 32
```

so that the evaluation domain `[1.20, 3.60]` is strictly **interior** to the window range. The
stiffness follows the window spacing, and is the rung the calibration shows producing zero cis
peptide bonds (cand-B's stiffer pull produced one, so cand-B is rejected on structural integrity,
not on cost). Cost: `B = 3072`, ~2.9 h per build, ~8.7 h for the three required builds;
12 288 ns aggregate per build.

**Additional gate added at the same time:** replicas are screened with
`deca.system.validate_thermal` **after the pull and again after equilibration**, and any replica
carrying a cis peptide bond or a chirality flip is excluded from the reference with its count
reported. A hard pull is exactly the operation that can produce one, and an excluded replica must
be visible, never silently averaged in.

`R_lo`, `R_hi`, `n_grid`, the evaluation domain and every gate threshold are **unchanged**.

### Amendment 2 — reference stopping rule (2026-08-11)

**Written before any checkpoint had been computed. No reference PMF, no acceptance statistic
and no Gate A result existed at the time. Nothing downstream of the reference existed.**

§6.3 fixes 4 ns per replica, giving 12 288 ns per build — about **96× the deca-alanine ABF
literature benchmark of 128 ns**, and ~288× across three builds. §4.5 does not ask for a fixed
amount of sampling; it asks that reference uncertainty be small against the effect being
measured, and it *already* requires a convergence trace against reference compute. Running a
fixed 4 ns and then reporting that trace does the work twice and never acts on it.

**Structural change.** The three builds now run **interleaved in one batch**
(`B = 3 × 96 × 32 = 9216`) rather than sequentially. §4.5 acceptance is a statement about the
spread *between* independent builds, so it cannot be evaluated until every build has reached
the same sampling; sequential builds forbid stopping early by construction. Interleaving also
puts the batch past the measured per-state cost knee (0.99 µs/state-step at `B = 2048` against
0.79–0.82 µs at `B ≥ 4096`). Each build keeps its own RNG and its own diverse pool, so
"independently initialised" still means what it says.

**Stopping rule.** At checkpoints of **1, 2, 3 and 4 ns** per replica, compute each build's
`F_b` and the §4.5 statistic

```
  ratio  =  max pairwise L2 between builds  /  ( 0.10 x consensus F span )
```

Production stops at the first checkpoint where **both**

```
  ratio <= 0.5           and           sampling >= 2 ns per replica
```

The 2 ns floor exists so the run cannot stop on a single lucky checkpoint. The 0.5 margin
exists so it cannot stop sitting on the acceptance boundary at 1.0 — the reference must be
**twice** as good as the minimum, not barely good enough. If the rule is never met by 4 ns, the
run completes the full 4 ns and reports `ratio`; a final `ratio > 1.0` means the reference is
**not accepted** and must be rebuilt longer or with more windows, exactly as §4.5 already says.

Every checkpoint is retained and written out, so the convergence-versus-compute trace §4.5
requires is a by-product of the run rather than separate work — including the checkpoints after
the stop would have fired, when the rule does not fire.

**No gate threshold, no success rule, no arm definition and no evaluation domain changes.** The
only thing this amendment can affect is how much reference compute is spent, and it can only
spend less when the preregistered acceptance criterion is already exceeded by 2×.

### Amendment 3 — how the states `C_k` are defined (2026-08-11)

**Written while the reference was still equilibrating. `F_ref` did not exist, no checkpoint had
been computed, and no occupancy, `T_hit` or Gate result existed.**

§2 refers to "relevant metastable regions `C_1, …, C_K` in the chosen CV space" without fixing
how they are found. Every one of Gates B and C is a statement about those regions, so leaving
the rule open until after `F_ref` is in hand would let the states be chosen to produce a
verdict. The rule is therefore fixed here, in advance, **including its fallback**, because the
shape of the deca-alanine PMF is not yet known and a rule that only covers the convenient case
is not a rule.

**Primary rule — basins of the reference.**

1. Evaluate `F_ref` on the frozen evaluation grid over `[R_lo, R_hi]`.
2. Find all local minima.
3. Merge any adjacent pair whose separating barrier, measured from the **higher** of the two
   minima, is below **2 kT**. Repeat to convergence.
4. If **two or more** minima survive, they are the states `C_k`, with boundaries at the
   intervening barrier maxima and at the domain edges.

**Fallback — a single-basin PMF.** If only one minimum survives, the end-to-end coordinate has
no multi-basin structure and there is nothing for a basin rule to find. The states are then the
frozen **equal-width tercile partition** of `[R_lo, R_hi]` — compact / intermediate / extended —
which is the convention the alkane `R15` distance-CV study already used. This is declared as a
partition of the coordinate, **not** as a claim that the three regions are metastable.

**What each branch means, fixed now.** A single-basin PMF is itself informative: it says the
deca-alanine difficulty the literature reports is not a multi-basin structure *in `xi`*, which
is precisely the case Gate A exists to catch and which would push the interesting question into
the conditional `p(Y | xi)`. That reading is recorded here so it cannot later be presented as a
prediction that was confirmed.

The 2 kT merge threshold is a judgement fixed in advance, not tuned. It is deliberately low:
merging aggressively would manufacture a single basin and force the fallback.

### Amendment 4 — initial conditions for the ABF-only screen (2026-08-11)

**Written before the screen ran, before the reference finished, and before any `F_ref`,
basin structure, `T_hit` or occupancy existed.**

§6.5 fixes the screen budget (16 walkers × 0.5 ns, 8 ensembles) but not where the walkers
start. That omission is not cosmetic: **the initial condition decides Gate B outright.**

Two defensible conventions exist and they give opposite answers by construction:

* **Distributed along `xi`** — the usual multiple-walker ABF deployment, walkers spread across
  the reaction coordinate. Under it every state is occupied at `t = 0`, so `T_hit = 0`
  everywhere and **Gate B can never fail**. Discovery-limitation would be defined out of
  existence rather than tested.
* **All from the folded basin** — the natural physical initial condition for deca-alanine and
  the one the Park–Schulten helix-coil setup uses. Discovery becomes a genuine question.

**Chosen: all walkers start from the equilibrated α-helix.** Reasons, in order:

1. it keeps Gate B a real test rather than a formality;
2. it is the physically natural starting state, not a construction;
3. it matches how v1 seeded its screens — the valine V3 screen started from one region and
   *measured* when the others were reached (all 8 by 5.4 ps), which is the calibration for what
   "discovered easily" looks like.

**Protocol.** Each of the 8 ensembles (seeds 3000–3007) builds 16 walkers from the ideal
`(-57, -47)` helix with independent thermal jitter, relaxes them under
`deca.umbrella.relax_pool`, and runs **20 ps of unbiased Langevin equilibration**. That
equilibration is **outside** the 0.5 ns ABF budget and is declared as such — it thermalises the
structure, it does not advance the free-energy estimate, and no ABF accumulation occurs during
it.

**The bias this introduces, stated rather than discovered later.** A folded start makes
discovery *harder* than a distributed start, so this choice can only push the classification
**toward** discovery-limited. If deca-alanine comes out discovery-limited, that verdict must be
read with this in mind and reported alongside it — it would be a statement about the folded
start plus the 8 ns budget, not about the coordinate in general. It cannot manufacture an
establishment-limited verdict, which is the direction that would license an mFR arm.

### Amendment 5 — withdrawal of the first deca-alanine screen verdict (2026-08-11)

The first ABF-only screen reported `REGIME: ESTABLISHMENT-LIMITED`, `licenses_mfr: true`.
**That verdict is WITHDRAWN, not amended.** Two independent implementation defects, both in v2
code written for this campaign, invalidate it. The run is retained at
`results/deca/screen_RETRACTED_no_min_count_guard/` with a `RETRACTED.md`; nothing in it may be
cited.

**Defect 1 — out-of-domain samples clamped into edge bins.**
`alkanes.interval.bin_counts` clamps out-of-range samples into bin 0 / bin *n*−1, which is
harmless in the alkane study because soft walls make it rare. **Amendment 1 of this document
deliberately bracketed the umbrella centres at `[1.15, 3.70]` around the evaluation domain
`[1.20, 3.60]`**, so 4.82 % of reference samples lay outside the domain *by design* and were
piled into bin 0. That carved a spurious 2.65 kT well at `grid[0]` against neighbours at
~5.3 kT. The Amendment 3 basin finder read it as a genuine second minimum and split off a
0.056 nm "state" lying below the screen's soft wall at 1.25 nm — a region that can never be
populated — so Gate C reported a persistent deficit. **A fix introduced by this preregistration
manufactured the artifact it then certified.**

**Defect 2 — `abf_min_count` declared and never applied.**
`deca.core.DecaSimConfig` carried `abf_min_count` and no code read it.
`alkanes.interval.mean_force_profile` guards only `den > EPS`, so a bin holding a single sample
contributed that one instantaneous local mean force as its conditional average. The applied bias
ran away:

| | measured on the withdrawn run |
|---|---|
| learned `A_hat` span | **102.5 kT** against a 72.0 kT reference (+42 %) |
| walkers above 2.80 nm, second half | **97.9 %** |
| occupancy of the folded basin (holds the 1.64 nm minimum) | **0.008** |

`Q*` is computed *from* the learned bias, so Gate C compared a wrong occupancy against a wrong
target. This is the standard ABF `fullSamples` guard; `alanine.core2d_ala` applies it correctly
(`trust = den >= min_count`) and the deca sampler did not.

**What survives.** The reference is umbrella + MBAR and independent of ABF: rebuilt with the
edge fix it gives ratio **0.0337**, span **72.0 kT**, minimum **1.637 nm**, accepted. **Gate A
survives at 0.754** against the 0.30 threshold. Amendment 3's single-basin fallback fired as
preregistered.

**Nothing is retuned in response to the invalid run.** No physical parameter, seed, budget,
evaluation domain, state definition or gate threshold changes because of it. The corrected screen
re-runs seeds 3000–3007 at 16 walkers × 0.5 ns from the equilibrated helix, against the same
`F_ref` and the same thresholds. The only permitted differences are the two correctness fixes and
the corroboration gate of Amendment 6.

### Amendment 6 — structural corroboration is required to license a deca-alanine mFR arm (2026-08-11)

**Written before the corrected screen was analysed. The eligible label set below was frozen from
the reference alone, with no screen result in hand.**

The accepted `F_ref` is **single-basin and monotone**, spanning ~72 kT with its minimum near
1.64 nm, so Amendment 3's fallback partitions `xi` into three terciles. That partition is
objective, but it is **not sufficient to license an mFR production experiment**, because a
tercile can be underpopulated simply for being the far tail of a 72 kT climb. Tail starvation is
not the establishment mechanism this project is about, and the deca-alanine literature problem
concerns *conformations in parallel valleys*, not thirds of a distance interval.

Gate A has already supplied the better object: the structural labels are strongly separated in
`xi`, `max TV = 0.754` against a 0.30 threshold.

**New requirement.** The tercile Gates B and C remain the preregistered coordinate-level
diagnostic. In addition, a deca-alanine mFR arm is licensed **only if** a persistent
establishment deficit is also present in at least one **physically meaningful structural state
`Y` that is visible in `xi`**.

**Bias-aware structural target.** With `p_ref(xi, y)` the reference joint and `B_t(xi)` the bias
ABF has applied,

```
                integral  p_ref(xi, y) exp(beta B_t(xi)) dxi
  Q*_y(t)  =  ---------------------------------------------------
              sum_y'  integral  p_ref(xi, y') exp(beta B_t(xi)) dxi
```

the exact structural analogue of the coordinate target `Q*_k(t)` of §2.1 (the bias depends only
on `xi`, so it reweights the joint pointwise in `xi`). The establishment criterion is unchanged:
`R_y(t) = Q_y(t) / Q*_y(t) < 0.5` persistently for at least `0.20 T`.

**Eligible labels, frozen now.** Reference weight shares over the 9-state composite label:

| label | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| share | 5.93e-2 | 7.89e-3 | 7.37e-2 | 5.72e-2 | 6.60e-3 | 2.45e-1 | 4.94e-2 | **2.88e-4** | 5.01e-1 |

Applying the already-frozen 1e-3 reference-weight floor, the eligible set is
**`{0, 1, 2, 3, 4, 5, 6, 8}`**; only label 7 is excluded, for carrying 2.9e-4 of the reference
weight. **This set is fixed here and may not be revised after seeing the screen.**

**Decision rule.**

```
  coordinate deficit AND structural deficit   ->  establishment-limited; continue to Gate D
  coordinate deficit, NO structural deficit   ->  STOP: no physically corroborated deficit
  no coordinate deficit                       ->  ABF-sufficient or discovery-limited; STOP
```

**Passing Gate C does not license production.** The clone-decorrelation gate (§2.5) and the rate
calibration of §3 still stand between a licensed classification and any five-arm run.

### Amendment 7 — Gate 0: the ABF baseline must itself be valid (2026-08-11)

**Added after the corrected screen, and it does not change any existing threshold.** It adds a
gate that must be passed *before* Gates B, C and 6 are meaningful at all.

**What happened.** The corrected screen — with both defects of Amendment 5 fixed — still produced
`A_hat` spanning **86.7–110.3 kT** against a 72.0 kT reference, with **95.1–99.96 %** of walkers
pinned above 2.80 nm and a one-way population trace (`[1.00, 0, 0]` at 0 ps → `[0, 0, 1.00]` by
100 ps, never returning). Its learned mean force disagreed with `dF_ref/dR` by **61 %** in bins
holding up to **2 × 10⁶ effective counts**, so sampling volume cannot be the explanation.

**What the audit established.** `results/v2_validity_audits/deca_mean_force/` accumulated the
*same* `f_loc = ∇V·v − β⁻¹ ∇·v` estimator inside **umbrella-restrained** windows, whose
conditional sampling at fixed `xi` was already validated. Result: **8.4 % relative error**,
sub-1 % agreement through the well-converged interior (e.g. 103.66 vs 103.20, 127.08 vs 127.40,
133.04 vs 132.49), and integrating `⟨f_loc⟩` gives **69.4 kT against the reference's 67.1 kT** on
the same range.

> **The estimator, the CV geometry, the reference and the integration are mutually consistent.
> There is no bug in the mean force. What fails is ABF's *conditional* equilibration:** the
> peptide's hidden conformational degrees of freedom do not relax at fixed end-to-end distance
> within 16 walkers × 0.5 ns, so ABF averages over a non-equilibrium conditional ensemble, its
> mean force is systematically wrong, the bias overshoots, and the population makes a one-way
> trip to the extended end.

**Gate 0 (ABF baseline validity), evaluated before Gates B and C.** From ABF-only data plus the
accepted reference:

```
  span(A_hat_T) within [0.75, 1.25] x span(F_ref)          and
  no seed with > 0.90 of walkers in one tercile over the whole second half
```

Failing either, the run is classified **`ABF-baseline-invalid`** and

> **STOP. Gates B, C and 6 are not evaluated and no regime is assigned.**

The reason is not conservatism. `Q*_k(t)` and `Q*_y(t)` are both computed **from the applied
bias `B_t`**. If `B_t` is 20–53 % wrong, the targets are wrong, and a deficit measured against
them is an artifact of the baseline rather than evidence about establishment. The withdrawn
screen and the corrected screen both "passed" Gate C for precisely this reason.

**Why this is not a budget excuse.** The 8 ns aggregate is the literature benchmark and was run
faithfully, with 16 walkers sharing one estimator — i.e. genuinely multiple-walker ABF. The
budget is **not** raised in response. Raising it until the baseline works would be tuning the
benchmark to manufacture a workable control.

**The mechanistic consequence, recorded now.** A conditional-equilibration failure is a **third**
failure mode, distinct from discovery-limited and establishment-limited, and **mFR cannot repair
it by construction**: mFR reallocates population *along* `xi`. If the conditional ensemble at
each `xi` is wrong, moving walkers along `xi` does not fix it — no marginal reallocation rule
can, and that includes the book-Laplacian and count-balancing baselines. This belongs in the
regime map as its own category. Amendment 8 makes it one.

### Amendment 8 — a fourth regime, and why marginal selection cannot reach it (2026-08-11)

**Terminology adopted: `conditional-equilibration-limited`.** The defining situation is

```
  marginal sampling in xi looks abundant, but  p_t(q | xi)  is NOT  mu(q | xi)
```

so ABF's conditional mean force is biased at finite time no matter how many samples a bin holds.
Deca-alanine exhibited it at up to 2×10⁶ effective counts per bin with a 61 % mean-force error.
The name is chosen because it says exactly what fails; `orthogonal-relaxation-limited` is an
acceptable synonym.

**Why no marginal selection rule can repair it — proof.** The mFR score depends on the CV alone,
`S_t = S_t(xi)`. Writing the joint law of CV and hidden conformation as `p_t(xi, y)`, the
birth–death contribution is, up to the sign convention of `deca.selection`,

```
  d/dt p_t(xi, y) |_FR  =  -gamma S_t(xi) p_t(xi, y)
```

Marginalising over `y` gives `d/dt p_t^xi = -gamma S_t(xi) p_t^xi`. Then for the conditional
`p_t(y | xi) = p_t(xi, y) / p_t^xi(xi)`, the quotient rule gives

```
  d/dt p_t(y | xi) |_FR
      = [ (-gamma S p(xi,y)) p^xi  -  p(xi,y) (-gamma S p^xi) ] / (p^xi)^2
      = 0
```

> **The mean-field selection step leaves the conditional distribution exactly invariant.**

This holds for **any** score that is a function of `xi` alone, so it covers the book-Laplacian
rule and count balancing as well as Fisher–Rao: it is a statement about the entire
marginal-selection family, not about mFR specifically.

**Stated limit of the claim.** This is a mean-field statement. Finite-population cloning followed
by independent propagation can perturb the conditional distribution indirectly, so the claim is
**not** that mFR can never under any circumstances affect `p(y | xi)` — it is that there is **no
systematic selection pressure at fixed `xi`** to repair it. That distinction must survive into
the manuscript.

**The regime map becomes four-way.**

| regime | what fails | mFR expectation |
|---|---|---|
| ABF-sufficient | nothing important | neutral |
| discovery-limited | the state is never reached | cannot clone what does not exist |
| **establishment-limited** | state reached, population relaxes slowly | **the useful regime** |
| conditional-equilibration-limited | `p_t(q \| xi)` is wrong | marginal selection cannot directly repair it |

**Gate 0 becomes universal.** It applies to every benchmark from here, before Gates A/B/C, and
no future result may be forced into "discovery" or "establishment" when the honest answer is
that the ABF baseline was never valid. The classification for **all** systems — including the
R15 re-audit — is now four-way, not three-way.

**Deca-alanine is closed as a mechanistic boundary result**, not deleted and not retried at a
larger budget. Its value is precisely that it demonstrates something the alkanes and dipeptides
did not: a CV can separate the structural states well (Gate A = 0.754) and ABF can still fail,
because the *conditional* equilibrium along that CV is not reached in the available time.
Extending `16 × 0.5 ns` until ABF becomes tractable and then looking for an mFR gain would be
searching for a budget that manufactures the desired comparison, and is refused. A preregistered
budget-scaling study (8 / 32 / 128 ns aggregate) asking only whether `|F'_t − F'_ref|` falls as
conditional mixing catches up would be a legitimate appendix diagnostic — **a study of ABF
conditional equilibration, not an mFR search** — and is not a current priority.

### Amendment 9 — Gate 0's span clause is retracted (2026-08-11)

**My own Amendment 7 was wrong on one clause. The R15 audit found it, and I verified it
independently from the raw artifacts before accepting it.**

**Retracted: the span clause.** Amendment 7 required
`span(A_hat_T) in [0.75, 1.25] x span(F_ref)`. Measured span ratios:

| system / cell | span ratio | independent verdict |
|---|---|---|
| deca-alanine screen | 1.20 – 1.53 | baseline invalid |
| R15 dispersed β1 | **1.311** | *easy*, `normL2(F) = 0.068` |
| R15 trans β1 | **1.356** | *easy*, `normL2(F) = 0.076` |
| R15 dispersed β2 | 1.467 | starved |
| R15 trans β2 | 1.503 | starved |

**The ranges overlap.** The statistic cannot separate a baseline that is demonstrably fine from
one that is demonstrably broken, so it is not a validity test. `max − min` over 183–256 bins is
a two-order-statistic quantity dominated by tail noise, and it says nothing about whether the
learned bias is trustworthy. The clause is deleted, not re-tuned.

**Retained: the pinning clause.** `no seed with > 0.90 of walkers in one tercile over the whole
second half` does discriminate cleanly — deca **0.951–0.9996** against R15 **0.46–0.74**.

**Not adopted: a threshold on mean-force agreement.** The relative error
`mean|mf_hat − F'_ref| / mean|F'_ref|` on well-supported bins is the statistic that actually
diagnosed deca (0.61). But it does **not** cleanly separate on its own either: R15 β=1 reads
0.264–0.265 while being demonstrably fine (`normL2(F) = 0.068`), because mean force is far
noisier than the free energy it integrates to. Setting a threshold now would also fix R15's
verdict **after** seeing R15's number, which is the failure mode this document exists to
prevent. **No threshold is set.**

**How conditional-equilibration is attributed, then.** By the controlled experiment, not by a
screen statistic. For deca the question was settled by accumulating the *same* `f_loc` estimator
inside restrained windows with validated conditional sampling: 8.4 % error there against 61 % in
the ABF run exonerated the estimator, the CV geometry, the reference and the integration, leaving
conditional equilibration as the only remaining explanation. **That controlled test is the
instrument. Gate 0 is a screen for when to run it, not a substitute for it.**

**R15 status after the audit.**

* **Primary question answered:** the missing `fullSamples` guard is immaterial for R15 — deltas
  `normL2(F) <= 0.0004`, `span ratio <= 0.003`, `lowSupport = 0.000`. The v1 R15 numbers stand
  as published, and the discovery-limited pillar is **not** invalidated by the guard.
* **Newly open:** the β=2 cells show relative mean-force error **0.564 / 0.593**, essentially
  deca's 0.61. Whether they are *discovery-limited* or *conditional-equilibration-limited* is
  **unresolved**, and Amendment 8 forbids forcing it into the three-way box. A confound must be
  handled: the R15 reference is importance-sampling based, a different object from deca's
  umbrella+MBAR reference, so part of that error could be reference error.
* **Resolvable by:** the same controlled restrained-sampling test, run on R15. If restrained
  `<f_loc>` reproduces the R15 reference, the reference is sound and ABF's conditional sampling
  is implicated; if it does not, the reference is implicated instead.

### Amendment 10 — Gate 0 leads, the classifier is four-way, and there are three timescales (2026-08-11)

**Frozen before the R15 conditional-mixing experiment ran.**

**Gate order.** Gate 0 is evaluated **first**, ahead of Gate A:

```
  Gate 0 : is the ABF conditional mean force trustworthy?   <-- NEW, leads
  Gate A : can the relevant states be distinguished through xi?
  Gate B : were they discovered?
  Gate C : were they established?
```

Classification is by the **first failing gate**. A benchmark is never labelled discovery-limited
merely because rare states also have poor visitation: if Gate 0 fails in the region responsible
for the free-energy error, the benchmark is **conditional-equilibration-limited**, whatever the
visitation looks like.

**Three timescales, not two.** The favourable mFR window is not `T_hit << T <~ T_est`. It is

```
  T_hit  <<  tau_perp  <<  T_est
```

with `tau_perp` the relaxation time of the hidden coordinates *at fixed* `xi`:

| | |
|---|---|
| `T_hit` too large | discovery-limited |
| `tau_perp` too large | **conditional-equilibration-limited** |
| both small, `T_est` large | **mFR opportunity** |
| all small | ABF-sufficient |

> **mFR works when the system locally forgets a cloned configuration faster than the marginal
> population equilibrates.**

This is the same quantity as Gate D's clone-decorrelation time, arrived at from the estimator
side rather than the genealogy side; the two conditions are now recognised as one.

**Retrospective obligation.** The WCA dimer and the entropic gateway were classified
establishment-limited **before Gate 0 existed**. Their interpretation is therefore not yet
established: it must be *backfilled*, not assumed. Until each passes a Gate 0 audit, the phrase
"establishment-limited positive" is provisional for both. Neither the WCA prior-selection
comparison nor NaCl proceeds before that backfill.

**Not a setback to be engineered around.** `d/dt p_t(y|xi)|_FR = 0` (Amendment 8) makes this
limitation **structural** to marginal selection, covering count balancing and the Chapter-6
Laplacian rule equally. The project's claim is therefore not "birth-death added to ABF sometimes
helps" but *"we characterise when marginal population selection can and cannot accelerate
adaptive free-energy estimation"*. No attempt will be made to "fix" mFR against this regime.

**Deca-alanine is frozen where it is.** The 8/32/128 ns budget-scaling study answers "how long
does ABF need before deca conditionally equilibrates" — an ABF question, not an mFR question —
and is demoted to an optional appendix after the core campaign.

### Amendment 11 — methane/water is promoted to a gated benchmark, and the periodic engine is built for it first (2026-08-12)

**Written before any methane code, engine, box, reference, screen, gate result or arm result
existed.** At the time of writing a repository-wide search for a periodic, solvated, Ewald,
reaction-field or rigid-water code path returns nothing: there is no methane simulation in this
project and never has been. Nothing downstream of this amendment exists, so nothing in it can
have been chosen to produce a verdict.

The design being adopted is recorded in full in `docs/SPEC_methane_water.md`, which this
amendment licenses and which may not be edited after the screen runs except by a further
numbered amendment.

#### 11.1 What changes, and what explicitly does not

§9 preregisters the methane pair as **"ABF-only screen only; preregistered likely-null"**, with
screening seeds 5000–5007 and *no* calibration or production block — i.e. it was scoped so that
it could never produce an mFR result at all. It is promoted here to a full benchmark under the
universal gates: the ABF-only screen runs first, and an mFR arm exists **only if Gate 0 → A → B
→ C → D all pass**, on exactly the terms deca-alanine and NaCl are held to.

**What does not change is the prediction.** Methane remains preregistered as **likely-null**.
Promotion licenses a production *conditional on the gates*; it is not a forecast that they will
pass, and it may not be read as one afterwards. §9's stop clause is carried over verbatim and is
binding:

> if ABF is sufficient, **STOP and report that prediction** as a literature-anchored negative
> control. **The run length is not tuned until it passes.**

That last sentence is restated because the adopted design proposes `T_run = 200 ps`, and 200 ps
is frozen **now**, before an engine exists to measure anything with. If methane comes out
ABF-sufficient at 200 ps, the result is "ABF-sufficient at 200 ps"; the budget is not raised
until a deficit appears. The deca-alanine precedent (Amendment 7, "why this is not a budget
excuse") governs.

#### 11.2 Execution order: methane moves ahead of NaCl

§0 orders the queue deca → WCA → NaCl → methane. Methane and NaCl are swapped.

The reason is that **they need the same engine**, and methane is the better shakedown for it: the
solute is neutral and is a single Lennard-Jones site, so the periodic solvated engine can be
validated with no ionic finite-size correction, no charged-solute Ewald artifact, and no solute
internal degrees of freedom to confound the parity gate. NaCl then inherits an engine that has
already passed §8.1 against OpenMM on a system where any disagreement is unambiguously the
water. Building the engine for NaCl first and discovering a defect there would leave the defect
entangled with the charge treatment.

This reordering is licensed by the fact that Amendment 10's retrospective obligation is
**discharged**: WCA Gate 0 passes (`results/v2_validity_audits/wca_gate0/`, pool spread 0.040 of
`|F'_ref|`) and the gateway passes (`.../gateway_gate0/`, 0.036 global). Amendment 10 blocked
NaCl behind that backfill; the block is lifted for both.

**System 2 (WCA) is untouched and continues in parallel.** Nothing in this amendment or its spec
alters a WCA artifact, threshold or conclusion, and the open Stage-A finding — that the cached TI
reference is wrong at `z ≈ 0.25` by 23σ — is unaffected.

#### 11.3 The engine-equivalence gate of §8.1 is binding on methane

§8.1 was written for NaCl. It applies to methane unchanged, and **before any methane free energy
is computed**:

```
  V_torch ~ V_openmm      grad V_torch ~ grad V_openmm      xi_torch = xi_openmm
  f_local,torch ~ f_local,openmm                            F_ABF,torch ~ F_ABF,openmm
```

> An approximate reimplementation is **not** a literature reproduction and will not be called
> one. If the gate fails, methane does not run.

Two additions specific to rigid water, which NaCl will also need:

* **constraint satisfaction** — every O–H and H–H distance holds to `<= 1e-8 nm` of its rigid
  value over a production-length trajectory;
* **equipartition** — the configurational temperature and the kinetic temperature of the
  constrained system agree with the thermostat setpoint, measured per degree of freedom with the
  constrained DOF count, on both a `0.5 fs` and a `1 fs` timestep.

The parity gate is a statement about energies and forces at fixed configurations and is therefore
independent of the constraint algorithm; the two clauses above are what validate the constraint
algorithm, and they are gates, not diagnostics.

#### 11.4 A universal reference-construction rule (generalising Amendments 1 and 5)

The adopted design proposed umbrella centres on `[3.3, 9.0] Å` with the evaluation domain **also**
`[3.3, 9.0] Å`. **That is the exact configuration that produced the retracted deca-alanine
screen.** `alkanes.interval.bin_counts` clamps out-of-range samples into bin 0 / bin *n*−1; with
centres coincident with the domain edges the outermost windows spill mass outside the domain, it
piles into the edge bins, a spurious edge well appears, Amendment 3's basin finder reads it as a
state, and Gate C certifies a deficit in a region that can never be populated. Amendment 5 records
that this cost a screen verdict, and that the fix which caused it was introduced *by this very
document*.

The rule is therefore lifted out of the deca-specific text and made universal:

> **Umbrella window centres must strictly bracket the evaluation domain.** Any estimator
> consuming reference samples must either exclude out-of-domain samples or be demonstrated
> insensitive to them, and **the fraction of reference samples falling outside the evaluation
> domain is reported for every reference build.**

Methane's layout follows: evaluation domain `[3.3, 9.0] Å`, window centres on `[3.0, 9.3] Å`.
This applies retroactively to no accepted artifact — deca's accepted reference already carries
the edge fix — and prospectively to NaCl.

#### 11.5 Seeds

The §5 table gains the two blocks methane did not have. Screening is unchanged.

| System | screening (ABF-only) | calibration | confirmatory production |
|---|---|---|---|
| methane / water | 5000–5007 (8) | **5100–5103 (4)** | **5200–5215 (16)** |

The blocks are disjoint by construction, as for every other system.

#### 11.6 Deliberately left open: the arm set

§4.2 fixes **five** arms. The adopted design proposes **three** — `abf`, `mfr_practical`,
`mfr_sham` — dropping `book_laplacian` and `count_balancing`.

That is not a cosmetic difference. Those two arms are how the campaign answers **Q1**, which §0
calls "the highest-value question in the queue and the one v1 never asked": is the Fisher–Rao
direction better than the directed selection rules ABF already had? Deca-alanine is closed
without answering it and NaCl is now behind methane, so **methane is the only system in the queue
that could answer Q1 in explicit solvent.** Dropping the two arms forfeits that permanently for
this campaign; a three-arm positive could only be reported as "mFR beats ABF and beats matched
random turnover", which is what v1 already established twice.

**The decision is deferred, not made.** It changes nothing about the engine, the reference, the
screen or any gate, and those are the whole of the work until a production is licensed — which
the preregistration predicts will not happen. It must be **frozen before the calibration stage**
of §3, and this amendment records the recommendation: **keep all five.** The two prior-art arms
add two thirds to the cost of a production stage that only exists if five gates pass, and their
selection intensities `c` are tuned on the same calibration seeds by the same §3.4 procedure, so
they add no new tuning surface.

#### 11.7 Success rule: §4.3 stands

The adopted design proposes a stricter primary criterion (median `<= -15 %` vs ABF, `-10` to
`-15 %` vs sham, `>= 13/16` seed wins). **§4.3 remains the binding rule** — median `<= -10 %`,
95 % CI upper end `< -5 %`, `>= 12/16`, plus the §4.3 attribution clause — because a per-system
success threshold is not comparable across the campaign, and choosing one system's threshold
after seeing other systems' effect sizes is the failure mode this document exists to prevent.

Nothing is lost: the stricter numbers are preregistered here as a **secondary label**,
`STRONG POSITIVE`, reported alongside the §4.3 verdict. They can only ever be more demanding than
§4.3 and are never a substitute for it.

#### 11.8 One citation is not carried forward

The adopted design justifies `dt = 0.5 fs` by attributing a rigid-water timestep/equipartition
finding to Asthagiri *et al.*, **J. Chem. Phys.** 128, 244512 (2008) — the same reference used for
the methane model. That paper is about the role of attractive methane–water interactions in the
pair PMF, and the attribution could not be verified. **It is dropped rather than propagated.**

`dt = 0.5 fs` is retained as **our own conservative choice**, justified by the parity test the
design already specifies (§11.3 equipartition clause plus a `0.5` vs `1 fs` comparison of the
PMF and `tau_perp`), and by the fact that this study interprets solvent decorrelation times
mechanistically and so cannot afford an integrator artifact in `tau_perp`. If the parity test
shows `1 fs` indistinguishable, `1 fs` may be adopted for production and the change recorded.

The Lorentz–Berthelot methane–water mixing rule is likewise labelled **our implementation
choice**, not a literature value, exactly as the design proposed — carried into the spec as a
declared deviation on the model of §6.1's ff14SB-vs-CHARMM declaration.

### Amendment 12 — TI-first reference, sequential `N`, Q3 before Q1, and WCA is non-blocking (2026-08-12)

**Written before any methane dynamics, box, reference, mean force, screen, gate result or arm
result had been observed.** The only methane numbers in existence at the time were the
force-field constants of Amendment 11 and two engine *throughput* measurements (§12.4 below),
neither of which is a physical result. Nothing downstream of this amendment exists.

This amendment changes **execution strategy only**. The physical model, the reaction coordinate,
the evaluation domain, `T_run = 200 ps`, the Gate 0/A/B/C/D definitions and thresholds, the seed
blocks and the §4.3 success criterion are **untouched**.

#### 12.1 WCA is non-blocking and deferred

System 2 (the WCA Case IX re-run against a high-precision reference) is **deferred by
instruction** and is not a prerequisite for anything in methane. Stage A is left where it is:
the finding that the cached TI reference is wrong at `z ≈ 0.25` by 23σ stands, unactioned, and
Case IX remains scored against a reference now known to be defective. **That exposure is open
and is recorded here as open** — it is not resolved by this amendment and must not be described
as resolved.

Amendment 10's retrospective obligation was about *Gate 0*, and it is discharged for both WCA and
the gateway. The reference defect is a separate matter and blocks only the WCA effect size.

#### 12.2 Constrained TI becomes the **primary** reference; umbrella becomes a sparse cross-check

SPEC §4.1 makes umbrella + MBAR primary at

```
  64 windows x 2 families x 32 replicas x 2.5 ns x 3 builds  =  30.72 us aggregate MD
```

before methane has been asked whether it is interesting at all. For a benchmark preregistered as
**likely null** that is the wrong order of spending, and the alternative is not a compromise but
arguably the better instrument.

**Primary reference — constrained/restrained TI**, on the already-frozen grid
`r_j = 0.34, 0.36, ..., 0.90 nm` (29 points):

```
  29 points x 16 replicas x (50 ps equilibration + 200 ps production) x 3 independent builds
     =  348 ns aggregate MD          (88x less than the umbrella design)

  16 replicas per point split 8 wet-like / 8 dry-like initial solvent environments
  F_ref(r) = C + integral_{r_0}^{r} fbar(s) ds        W_ref(r) = F_ref(r) + 2 beta^-1 log r + C'
```

Three reasons this is *better*, not merely cheaper:

1. **It estimates the very object ABF learns.** `F'(r) = E[f(q) | xi = r]` with the local mean
   force already derived, implemented and autodiff-validated in `alkanes.distance_cv`. An
   umbrella reference estimates `F` and only implies `F'`.
2. **The wet/dry split makes the reference double as the Gate 0 instrument.** Amendment 9 records
   that a conditional-equilibration question is settled by restrained sampling with independently
   prepared families, not by a screen statistic — "that controlled test is the instrument". Here
   it is built into the reference stage instead of being a separate audit.
3. **It keeps the reference engine independent of the arm engine** (§12.4): the reference is
   computed in OpenMM, the population arms in the batched torch sampler. A shared-engine defect
   would cancel in an arm-vs-reference comparison and hide; separate engines cannot.

**The WCA lesson is honoured, not ignored.** What failed at WCA was a *cached, single-preparation,
inadequately validated* TI reference — `constrained_ti_reference_gpu` seeded every replica from
one lattice preparation and was structurally blind to a cage-equilibration failure. This design
is that reference's opposite: three independent builds, two deliberately opposed solvent
families, block uncertainty, and a convergence trace.

> **Declared structural exposure of TI, stated because it is real.** `F` is obtained by
> *integrating* `fbar`, so a systematic error at one `r` propagates to every larger `r`; umbrella
> + MBAR has no such accumulation. This is mitigated, not eliminated, by §12.3.

**Acceptance, replacing SPEC §4.3's umbrella clauses:** at production checkpoints of
`50, 100, 200, 400 ps`, compare the three builds' integrated `F` and the wet-vs-dry conditional
mean forces. **Extend only the points** whose build spread or family disagreement remains
materially above the target tolerance — not the whole grid. Acceptance keeps the §4.5 form:
three independent builds agreeing within `ratio <= 0.5`, block/bootstrap uncertainty reported,
out-of-domain fraction reported, and the `W' = F' + 2/(beta r)` identity holding to `1e-10`.

#### 12.3 Umbrella + MBAR survives as a sparse independent anchor

Retained, **after** TI and at a fraction of the cost: a small umbrella/MBAR build at sparse
windows around the three physically meaningful locations found from `F_ref` — contact minimum,
desolvation barrier, solvent-separated minimum — used to anchor the integrated TI curve and to
test the accumulation exposure declared above. It is a cross-check on an independent estimator,
not a second full reference. Its window centres still bracket the evaluation domain
(Amendment 11.4).

If TI and the sparse umbrella anchor disagree beyond their combined uncertainty, **the reference
is not accepted** and SPEC §11.2's stop applies.

#### 12.4 Engine strategy: two measured shortcuts, and what they did and did not buy

**Shortcut 1 — a CUDA-enabled OpenMM. TAKEN, and it works.** The `abffr` environment ships
OpenMM with `Reference/CPU/OpenCL` only. A conda-forge install pinned to a CUDA build
(`conda create -n methane-cuda -c conda-forge python=3.11 openmm cuda-version=12`) exposes the
`CUDA` platform. Measured on the frozen 1538-site system, **on an otherwise idle GPU**:

| platform | ms/step | ns/day (one replica) |
|---|---|---|
| CUDA, mixed | 0.093 | **462** |
| CUDA, double | 0.125 | **345** |
| OpenCL | 0.26 | 169 |
| CPU | 13.9 | 3.1 |
| Reference | 60.3 | 0.7 |

> **A contended GPU reads 28× slower and looks like a code defect.** The first CUDA benchmark
> returned 16 ns/day; the cause was another user's job arriving on that device mid-measurement
> (100 % utilisation, 11 GB). **Every throughput number in this campaign is measured on a
> verified-idle device and the device's idle state is recorded with it.** The compute policy is
> unchanged — exactly one GPU at a time, pinned — but §1's designation of *GPU 2* no longer
> holds: GPUs 0, 1 and 2 carry other users' processes and methane runs on **GPU 3**, re-checked
> before each stage.

At 462 ns/day the 348 ns TI reference is **~18 h of serial OpenMM**, which is why §12.2 is
affordable at all. The reference therefore runs in OpenMM and does **not** wait for the torch
engine.

**Shortcut 2 — `torch-pme` instead of hand-written smooth PME. To be tested, not trusted.**
`torch-pme` 0.5.0 provides differentiable PyTorch PME/P3M/Ewald with GPU support and installs as
a pure-python wheel. It is adopted **only if** it reproduces OpenMM's reciprocal electrostatics
*plus* the self and intramolecular-exclusion corrections to the frozen `1e-6` of SPEC §3.2 on the
12-configuration parity set. Otherwise the hand-written PME already planned proceeds. This is a
short spike, taken before any FFT code is written.

**The batched torch engine is still required, and the reason is measured.** OpenMM runs one
system at a time, and 1538 atoms cannot saturate an H200. The ABF/mFR arms need `N` walkers
sharing one estimator with birth–death across the population; served by `N` serial OpenMM
contexts that is `~2.05e8` walker-steps per seed at 0.093 ms each — **days per seed**. Batching
the population into one step is the whole reason this project has its own samplers. So:

```
  OpenMM (CUDA)   ->  parity oracle, NPT box, constrained-TI reference
  batched torch   ->  ABF screen and every population arm
```

#### 12.5 The `N` ladder is executed sequentially, starting at `N = 512`

SPEC §6.3 screens `N in {128, 256, 512}`. The ladder is unchanged; only its **order** is fixed,
and it is fixed here in advance:

```
  run N = 512 first, on seeds 5000-5007
```

Under the mechanism being tested `T_hit ~ 1/N` while `T_est` receives no comparable
acceleration, so `N = 512` is the setting with the best chance of exhibiting `T_hit << T_est`.
It is also a **dominance screen** for both null modes, which is what makes stopping early
legitimate rather than convenient:

| `N = 512` verdict | action | why it dominates |
|---|---|---|
| discovery-limited | **STOP** | fewer walkers cannot discover *faster* |
| ABF-sufficient | **STOP** | the preregistered likely outcome; fewer walkers make discovery noisier, which is not the discovered-but-under-established mechanism |
| conditional-equilibration-limited | **STOP** | Amendment 8: no marginal score acts on `p(y\|xi)` |
| **establishment-limited** | run `N = 128, 256`, then take the **smallest** eligible `N` by the frozen §8.2 rule | — |

> **The map is reported as partial when it stops early.** SPEC §6.3 says "the entire map is
> reported"; under sequential execution that becomes "every cell that was run is reported, and
> the cells not run are named, with the frozen rule that skipped them". A partial map is never
> presented as a complete one.

#### 12.6 First reported methane result is the three timescales, not an `F` error

After the `N = 512` ABF-only screen the first quantity reported is

```
  T_hit ,   tau_perp ,   T_est
```

and the Gate 0 verdict — **nothing about mFR**. This is already the preregistered gate order
(Amendment 10); it is restated because it is also the reporting order, so that no `F`-error
comparison can be seen before the regime is classified.

#### 12.7 Q3 first, Q1 conditional on a positive Q3

Amendment 11.6 left the arm set open and recommended all five. **It is now decided: three arms
first.**

```
  ABF        mfr_practical        matched-turnover sham
```

That triple already separates the baseline from generic resampling from directed Fisher–Rao
reallocation, which is exactly Q3 — does the establishment mechanism transfer to an independent
explicit-solvent atomistic system. Q1 (Fisher–Rao versus prior directed selection) is answered by
`book_laplacian` and `count_balancing`, and those run **as a separate prior-art closure on the
already-frozen physical setting, conditional on a positive Q3.**

**Where the line sits, stated precisely.** This is outcome-dependent *execution ordering*. It is
**not** outcome-dependent modification of the physical experiment, the mFR parameters, the gates
or the success rule, none of which may move. When the Q1 arms run they run on the same frozen
setting, the same seeds `5200–5215`, with **nothing retuned**, and their baseline intensities `c`
calibrated by the same §3.4 procedure on `5100–5103`.

**The cost of the ordering, declared.** Because the decision to run the Q1 arms is taken after
seeing Q3, the Q1 comparison is **not** part of the same preregistered primary analysis as Q3.
It is a declared follow-up, reported as such, and **reported whatever it shows** — including the
outcome where it ties or reverses and the novelty claim of §0 dies. A tie is tested by TOST, as
§4.3 already requires.

If methane is null, the three-arm design has saved the two extra arms and Q1 remains open for
NaCl, which is where it was always going to be answered if methane fails.

#### 12.9 Analysis-code failure classes (2026-08-13)

**Written before any screen verdict existed.** A cross-audit between the methane and NaCl
sessions found live defects in shipped analysis code in both, none of which raised an error and
all of which would have been read as physics. They fall into three classes, recorded here
because the classes generalise beyond either study.

> **CORRECTED SAME DAY.** The first version of this section claimed that the asymmetric defects
> "all leaned toward the positive result — none toward a null", and drew a structural argument
> from it. **That claim is false, and my own defect table below falsifies it.** The Gate C
> defect leans toward the *null*: a nan `Q*` makes `occupancy < 0.5 Q*` return False, which
> silences the detector, which is `ABF-sufficient — STOP`. I had written exactly that sentence
> one paragraph earlier and still asserted the opposite generalisation, because it had a
> satisfying shape and I did not check it against the table. The NaCl session enumerated its own
> six and refused to mirror the paragraph; its two class-1 defects also produced stops. This is
> the second claim in this campaign that I liked the shape of and did not check — the first was
> a density cross-check, corrected the same way. The retraction is kept in place of a quiet edit
> because the failure mode is the point.

| class | rule |
|---|---|
| 1 | **"No data" must never take the branch a PASS would take.** |
| 2 | **"No data" must never read as a *smaller* number** where that loosens a bound. |
| 3 | **A quantity defined as a partition must be tested as one** — on the sampled values, not only on the grid. |

**This study's defects, with the verdict each pushes**, derived per defect rather than assumed.
No combined count across the two studies is stated: each session can audit its own code and not
the other's, so a shared tally would be a number neither of us verified.

| # | defect | what the comparison does | verdict pushed |
|---|---|---|---|
| 1 | Gate A uncomputable, skipped by an `isfinite` guard | avoids the Gate A stop, falls through to Gate B | mislabels one stop as another — **neutral in pass/fail, wrong in classification** |
| 2 | Gate C non-finite `Q*` | `occ < 0.5·nan` → False, detector **silenced** | ABF-sufficient — STOP → **NULL** |
| 3 | grid partition drops `grid[-1]` | `Q*` understated by 0.1 %, deficit fires *less* | toward null, **second order / negligible** |
| 4 | out-of-domain walkers dropped from every basin | occupancy under-counted ~20 %, deficit fires *more* | establishment-limited → **POSITIVE** |

One toward the null, one toward the positive, two neutral. The same balance the NaCl session
measured independently.

Class 1 examples: an uncomputable Gate A skipped by an `isfinite` guard and falling through as
though it had passed; a non-finite `Q*` making `occupancy < 0.5 Q*` False at every checkpoint, so
no deficit is ever flagged and the cell classifies **ABF-sufficient — STOP** on the strength of
one missing reference point. The same nan is also reachable from *complete* inputs, because
`exp(-beta [F_ref - B_t])` overflows or underflows once the applied bias grows; the exponent is
now stabilised by subtracting its maximum.

Class 2 example: `tau_perp` estimated as a max over descriptors, with never-decorrelating ones
dropped rather than censored. That biases `tau_perp` **down** and the Gate D ceiling
`0.1 / tau_perp` **up**, licensing a faster selection rate than the physics justifies. Censored
points must enter at their lower bound and the ceiling reported as conservative.

Class 3 example, and the one that generalises furthest: basin masks closed on both ends
double-count shared boundaries (targets summing to 1.024); half-open everywhere drops the top
edge (`Q*` summing to 0.9988). Both are benign on a grid. The **same masks applied to walker
positions were not benign** — walkers leave the domain through the soft walls (measured range
`[0.322, 0.922]` against a domain of `[0.33, 0.90]`), so out-of-domain walkers were dropped from
every basin while `Q*` stayed normalised over the whole grid, and occupancies summed to **0.8**.
A partition assertion that only ever sees the grid passes while the real defect sits one line
away.

> **The direction must be derived, not assumed.** Missing or degraded data does not push a
> verdict in a consistent scientific direction. It lands in whichever branch the comparison
> operator defaults to — a `nan` *silences* a `<` test, an under-counted numerator *fires* one —
> and that default is a property of how the test was written, not of what the study hoped to
> find. Defects 2 and 4 above are the same class and point opposite ways, because the missing
> value landed on opposite sides of the comparison.
>
> What survives without qualification is the weaker and more useful statement: **analysis code
> deserves the same auditing discipline as the sampler**, which this campaign had been applying
> almost entirely to the latter. Every defect here was silent, none raised an error, and each
> would have been read as physics.

**A fourth class, from the tooling rather than the analysis.** Two paths to one resource, one of
which silently wins. Found once in each session within the same hour, with unrelated-looking
consequences drawn entirely from what the resource was: here, seeds 5005–5007 belonged only to a
process scheduled for termination and would simply never have run, surfacing as a 5-of-8 screen
against a 6-of-8 Gate B; in the NaCl session, two writers to one config file would have set the
study's production timestep by last-writer. The review question is the same for both — *can this
be triggered from two paths, and what happens if both fire?*

**And a testing lesson that is more general than the bug that produced it.** A guard written to
prevent the above was `pgrep -f "<script path>"`. Checked from an interactive shell it appeared
to self-match, which is this project's documented `pgrep` footgun; that appearance was **false**,
caused by the checking shell's own argv containing the pattern. Re-checked from inside a script
file — the way the code actually runs — the self-match vanished **and a real defect appeared that
the false one had concealed**: `pgrep -f` also matches the harness wrapper shells, one of which
outlives the python process it launched, so the guard would have aborted forever and blocked the
very work it was protecting. Both sessions had this, independently, in guards written that hour.

> **A contaminated test can report a false positive that conceals a true one — and the false
> positive is the more comfortable finding, so it ends the investigation.** "Self-match, as
> documented" felt like a confirmed diagnosis and stopped the search. The fix in both cases was
> to require the matched process's `comm` to start with `python`, and to verify it from inside a
> script file rather than from a shell whose command line contains the pattern.

**Two further classes, both about the *choice* of test rather than its execution.**

**A check whose result cannot entail the claim it is cited for.** Two forms appeared, and both
produce a PASS that means nothing:

* *wrong quantity.* The NaCl study's `R_hi <= 0.97 L/2` is a **geometric** criterion standing in
  for a **thermodynamic** one; passing it by 0.003 nm is not weak evidence about box-size
  convergence, it is none. Gate A in both studies was computed as `TV(p(Y | xi))` where §2.2
  specifies `TV(p(xi | Y))` — the transpose, and partly tautological since the descriptor grows
  with the coordinate by construction.
* *right quantity, unattainable precision.* Methane's finite-size gate specifies a
  `0.1 kT/nm` tolerance on a measurement whose combined SEM is `0.5-0.67 kT/nm`. Resolving it
  needs 100-170x more sampling than the check itself costs. Neither outcome entails anything.

> **A gate that cannot fire reads as reassurance**, which is worse than no gate.
> Every gate should be asked: *does passing this entail the claim it is cited for, or merely fail
> to contradict it?* — and, separately, *can this fire at all, given the precision available?*
> Set a tolerance from the achievable uncertainty, and state the detectable effect size.

**An error in the specification, faithfully implemented.** The NaCl study's Gate A transpose was
written into `SPEC_nacl` §6 and the code implemented that spec correctly. Every check operating
*within* the study therefore passes: the code matches its spec, the tests match the code, review
finds nothing. Only auditing a document against the document above it reaches this, and nothing
does that routinely — it was found because a peer with a different specification noticed the same
transpose in their own *code*. Distinct from every other class, which are all defects in
execution.

**A corollary about what a null is worth.** A clean statistic is a reason for suspicion, not
satisfaction: NaCl's Gate A of exactly 1.000 turned out to reflect hydration being nearly a
function of `r` (across-`r` variation 14-83x the within-`r` variation), so the gate nearly could
not fail. Methane's ratio is **5.4x** — the descriptor retains an sd of 0.37 at fixed `r`. This
**inverts the natural intuition**: a system with *more* orthogonal structure is not a harder case
for a null, it is the only case where a null means anything, because the null only says something
if there was something for reallocation to act on. Methane is therefore the load-bearing null and
a system with little orthogonal structure can corroborate at best.

**Ratios, fully specified.** A ratio presents as one clean number while hiding the provenance of
both arguments, which is what makes it quotable and unauditable. Checking that numerator and
denominator come from the same **population** is necessary and is only the first of three
independent axes. Working out a single quantity — *how fast could the fastest of `n` walkers
be* — took the two sessions four passes to get right, and each error passed a commensurability
check:

| axis | the error made | effect |
|---|---|---|
| — | rms speed of **one** walker, not an extreme value at all | floor ~3x too slow |
| **statistic** | `E[max]`, the *mean* fastest walker, where a floor wants a *quantile* | half of runs beat the floor |
| **sampling** | `max abs(v)` two-sided, admitting the fastest **inbound** walker as a candidate for an **outbound** arrival | wrong population sampled |
| approximation | `sqrt(2 ln n)`, the leading extreme-value term without its correction | +23 % on `E[max]`, inflating the diffusive claim |

> **Identity, statistic and sampling are three independent ways to get a denominator wrong, and
> commensurability tests only the first.** *Commensurable and wrong are independent failures.*

The estimator was then removed rather than improved: `P(max of n <= x) = Phi(x)^n`, so the
`q`-quantile is exactly `Phi^-1(q^(1/n))`. Verified here against 400 k-trial Monte Carlo to
**7e-4 sigma at n = 64** and **4e-5 at n = 512**. A closed form ends the argument that successive
approximations were extending.

**And the rule that supersedes the altitude formulation above.** The extreme-value *statistic*
argument was the hard half and was correct; the arithmetic that followed it — that a higher
quantile gives a *larger* ratio, not a smaller one — was the easy half and was inverted.

> **The step you found easy is the step you did not verify.** Difficulty is a poor proxy for error
> rate. This explains the altitude cases too: a summary feels easy after the item-level work, which
> is exactly why it goes unchecked.

**Where the retracted claim actually went wrong**, in a form worth reusing: the per-defect
statements were exact; the failure entered at the moment of summarising *across* defects. A
sentence whose subject is a **set** ("all of them", "none of them", "the pattern is") is no
longer checkable against anything in front of the writer, while a sentence about an item still
is. That is the altitude at which to demand a re-derivation — not at the point a claim becomes
quotable, but at the point its subject stops being a thing you can look at. Both of this
campaign's retracted claims (this one and the density cross-check of §12.2's discussion) were
set-level sentences resting on item-level evidence that had never been enumerated.

Pinned by `tests/test_methane_gates.py` (14 gates) and the NaCl session's equivalents, so each
is a regression test rather than a fact someone would have to re-derive.

#### 12.8 Correction to 12.4: the engine split is right, its stated reason was wrong

**Written the same day as 12.4, after building the engine and measuring it, and before any
methane box, reference, trajectory or physical result existed.** Only throughput and parity
numbers were in hand.

§12.4 justified the batched torch engine by asserting that serving `N` walkers with serial
OpenMM contexts is "days per seed". **That reasoning is wrong and is withdrawn.** Measured on an
idle H200, `M` OpenMM CUDA contexts stepped in a shared-estimator pattern (100 steps per round,
positions and forces read each round) give:

| `M` contexts | setup | aggregate |
|---|---|---|
| 16 | 7.9 s | 390 ns/day |
| 64 | 55.2 s | 385 ns/day |
| **128** | — | **fails: "No compatible CUDA device is available"** |
| **256** | — | **fails** |

390 ns/day is *faster* than the batched torch engine was at the time of writing §12.4, so cost
was never the discriminator.

**The real disqualifier is capability, and it is decisive.** OpenMM multi-context cannot create
more than ~64–127 contexts of this system on one device. Amendment 12.5 starts the screen at
**`N = 512`**, and §6.3 of the spec requires 128 as its smallest rung. A sampler that cannot hold
the population cannot run the preregistered design at any speed. The batched torch engine is
therefore required — but *because it is the only thing that reaches `N = 512`*, not because
OpenMM is slow.

**Measured engine throughput**, frozen 1538-site system, idle GPU 3, B = 512:

| engine | reaches `N = 512`? | aggregate ns/day |
|---|---|---|
| OpenMM CUDA, one context | no (single walker) | 462 |
| OpenMM CUDA, multi-context | **no** (fails ≥ 128) | 385–390 |
| torch batched, float64, eager | yes | 43 |
| torch batched, float64, compiled | yes | 34 |
| **torch batched, float32, compiled** | **yes** | **332** |

`torch.compile` is worth **8.1×** on this engine (float32, B = 256: 40 → 328 ns/day), consistent
with the 6.6× §1 records for deca, and it is mandatory here for the same reason.

**Precision, declared.** The parity gate of SPEC §3.2 is run in **float64**, where the engine
agrees with OpenMM to `2.9e-13` in energy and `1.4e-15` in force over 14 configurations spanning
−2 598 to +45 223 kJ/mol. **Production runs in float32**, which is what makes `N = 512` affordable
(float64 is ~10× slower and needs 63 GiB at B = 512). float32 is a **performance-only change and
is gated as one**, on the deca model
(`test_compiled_forces_are_numerically_indistinguishable_from_eager`): it must be shown
statistically indistinguishable from float64 on energy conservation, on the conditional mean
force, and on the PMF, before any production. If it is not, production reverts to float64 and the
budget is re-costed — the precision is not lowered to fit the budget.

**Cost implied.** The `N = 512` screen is `8 seeds x 512 walkers x 200 ps = 819 ns`; at 332 ns/day
that is **~2.5 days**. A neighbour list is the obvious next optimisation (the cutoff sphere is
~27 % of the box, so all-pairs wastes ~3.7×) and is a performance change gated the same way.

> **Both numbers in this paragraph were superseded within the day. See Amendment 13.2:** the
> throughput is **744 ns/day** (the 332 figure measured a path the engine does not take), and the
> neighbour list was implemented, measured **slower**, and rejected.

#### 12.10 Gate C has a minimum population, and it is now enforced (2026-08-14)

Class 6 of §12.9 — *a check whose result cannot entail the claim it is cited for* — applies to
**Gate C itself**, and the campaign had not noticed. Found by the NaCl session on 2026-08-14
when its `N = 64` cell reported a CIP deficit on 12.8 % of checkpoints; raised as a general
defect from this session; verified, implemented and extended by NaCl within the hour.

**The arithmetic.** Gate C fires on `occupancy < 0.5 Q*_k(t)`, applied to an **integer count of
walkers**. Its power is therefore set by `lambda_k = Q*_k N`, the expected count — not by the
number of checkpoints, and not by the run length. Once `0.5 lambda < 1`, the only integer
satisfying the test is **zero**, so the gate stops measuring a 50 % shortfall and starts
measuring *"is this state empty right now"*, whose frequency is `e^-lambda` **on physics alone**
and rises as `N` falls:

| | N = 64 | N = 32 | N = 16 | N = 8 |
|---|---|---|---|---|
| NaCl CIP `lambda` | 1.72 | 0.86 | 0.43 | 0.22 |
| NaCl SSIP `lambda` | 62.3 | 31.1 | 15.6 | 7.8 |
| `P(empty)` at CIP | 0.14 | 0.42 | 0.65 | 0.80 |

**The rule, binding on every system.** A state is judged by Gate C only where
`lambda_k = Q*_k N >= 16`, the value at which a 50 % deficit is a 2σ effect on one checkpoint
(`0.5 lambda >= 2 sqrt(lambda)`). Below it the state is reported **NON-BINDING with its
detectable effect size** and excluded from the verdict **in both directions** — it may neither
contribute a deficit (that would be counting noise) nor contribute a pass (that would be
arithmetic silence). If **no** state binds, the cell is **UNCLASSIFIABLE**, which is explicitly
*not* ABF-sufficient; raise `N` or coarsen the partition and re-run.

**Why the fall-through direction matters more than it looks.** §8.2's selection rule is *the
smallest `N` passing every gate*, so the search runs **down** the ladder — directly toward the
cells where the gate has no power. NaCl's `N = 16` and `N = 8` have **no state at all** reaching
the bar. Without the guard, "smallest N passing every gate" would have selected a cell whose
gate cannot fail, and reported it as a result.

**This repo has already paid for the mirror of this once.**
`results/deca/screen_RETRACTED_no_min_count_guard/RETRACTED.md`: edge clamping carved off a
0.056 nm "state" below the soft wall that could never hold a walker, *"Gate C fired on it"*, and
the run reported `licenses_mfr: true`. Retracted. **A state that cannot hold walkers is not a
state with a deficit** — and the quieter mirror, an unpopulatable state that happens *not* to
fire and so contributes a free pass, is equally fatal.

**Methane is unaffected, and that is a measurement, not an assumption.** Re-running all 8 seeds
through the guard gives `lambda = 127.6 / 147.0 / 224.2` on the three terciles, every state
BINDING, resolving 13–18 % deficits against the 50 % it tests for. Verdict unchanged:
**ABF-sufficient**. NaCl's `N = 64` verdict is also unchanged but its *support* is corrected —
SSIP (`lambda = 62.3`) carries it, and the CIP claim rests on the time-averaged 1.34× over-
population rather than on the gate, whose smallest resolvable deficit at `lambda = 1.72` is
**152 %**.

**Two general lessons, both already in §12.9 and both sharpened here.** (a) The question *does
this check's result entail the claim?* must be asked of the gate that **carries the verdict**,
not only of the gates that look weak — NaCl asked it of Gate B, declared Gate B non-binding, and
stopped one gate short. (b) When the guard was added, **four of NaCl's tests broke and every one
was the test's fault**, including a planted "establishment-limited" world at `lambda = 10.2`
asserting that the classifier must detect a world it correctly cannot, and a smallest-N test
that encoded the very search this amendment forbids. *A guard that breaks tests is evidence
about the tests.*

Implemented: `scripts/methane_gates.py` (`GATE_C_MIN_LAMBDA`), NaCl at `282e481`. Asserted:
`tests/test_methane_gates.py`, four tests including the deca retraction as an executable claim.

#### 12.11 Gate A's reported direction is now the preregistered one (2026-08-14)

The §12.9 class-6 correction to Gate A had been made **in prose only**. `methane_gates.py` still
computed the transpose and wrote `gateA_max_TV: 0.987` into `gates.json`, so the retracted number
was what the code emitted and what any re-run would report. Corrected: the script now computes
**`TV(p(xi | Y))` as §2.2 specifies** over 823 296 paired samples (dry / mid / wet by `n_gap`),
reporting **0.935** (pairs 0.805 / 0.935 / 0.474), and carries the transpose alongside it
explicitly labelled *diagnostic only, not the gate, do not quote*.

**A correction that lives only in a results file is not a correction.** The prose said 0.935 for
a day while every machine-readable artifact said 0.987.

`xi` bins are anchored to the **evaluation domain**, recovered from the grid's half-bin edges
rather than hard-coded. Anchored to the domain the statistic is flat at 0.9347 across 21–201
bins; anchored to the **data range** it wanders 0.931–0.936, because a data-anchored edge moves
with the most extreme walker. Asserted in `test_gate_a_reports_the_preregistered_direction_not_its_transpose`.

#### 12.12 Failure class 8: a reported action is not a performed action (2026-08-14)

Classes 1–7 are errors of **analysis**: a wrong statistic, a wrong quantity, a gate without
power, a spec faithfully implemented. Every one of them is in principle reachable by auditing
the artifacts, which is how the cross-audit found them. Class 8 is not.

**The class.** *A claim about an action, made in the same breath as the decision to take it,
with nothing checking that the decision became the action.* The NaCl session reported to this
one that its `N`-ladder map was "resuming from its 16:49 checkpoint" at the moment GPU 3 was
handed back. It had killed the map at 16:56 and did not start it until **23:34** — GPU 3 sat
idle for **5 h 14 m** while both sessions believed it was working. Self-reported, unprompted.

**Why it is categorically different.** No amount of auditing on the receiving side can catch it.
The receiver has the report and no independent channel to the machine state; the sender has the
machine state and no reason to re-check a claim it believes it already made true. Every other
class this week was caught by someone recomputing something. This one was caught only because
the session that made the claim went back and looked.

**The rule.** *A statement about the state of a running job is a measurement, not a memory.*
Before reporting that a run is live, resumed, queued, or killed, the reporting session must
observe it — `nvidia-smi`, the process table, a checkpoint mtime — **in the same turn as the
report**. This is not a counsel of perfection: it is one command, and its absence cost five
GPU-hours on a two-day run.

**Corollary, and it is the load-bearing half.** An unverified claim that turns out **correct**
is not a checked claim, **and the distinction is invisible from the outcome.** When this session
wrote that "both GPUs were returned to the NaCl session," it had not looked either; that claim
was true and the NaCl session's was false, but they were the *same act* with different luck.
This is why the rule cannot be "be careful when it matters" — whether it mattered is only
knowable afterwards. The receiving session must not launder an unverified claim onward.

**Second form: verifying the fact does not verify the rule.** Challenged on the GPU count, the
NaCl session measured the machine before answering — correct — and cited Amendment 15.4 as its
authority. 15.4 is quoted verbatim and **superseded**: 16.4 (2026-08-14) reassigns GPU 3 to C60
and directs NaCl's ladder to GPU 2 sequentially, and 16.2 flags the supersession explicitly. A
frozen amendment is only evidence if it is the current one, and a superseding clause seven pages
later is exactly as invisible as an idle device. **Cite the rule, then check nothing above it
moved.**

**Verified on writing this** (2026-08-14 23:43 UTC), and it immediately found something else:
GPUs 0 and 1 are running another user's five-day jobs, and **this group has two GPUs in use at
once** — `nacl_screen` (map, GPU 3, started 23:34) and `nacl_tau_perp` (GPU 2, started 23:38) —
against the standing one-at-a-time constraint. Neither session had looked. That is the class
demonstrating itself twice within one message.

**A threshold this catches, in a study that had not started.** Amendment 16.3 froze a C60 Gate C
guard before any C60 datum — the right instinct, adopted in advance — at `N Q*_k >= 3`. By
§12.10's arithmetic the smallest deficit resolvable at 2σ is `2/sqrt(lambda)`: **115 % at
`lambda = 3`, which exceeds 100 %, so no physically possible deficit is detectable there.**
`lambda >= 4` is the floor at which a *total* wipeout is marginally 2σ; `lambda >= 16` is where
the 50 % the gate tests for becomes a 2σ effect. The guard as frozen admits exactly the states
it was written to exclude, and `N Q*_k` must be the **minimum over the deficit span**, not its
mean. Raised to the C60 branch 2026-08-14; C60's threshold is C60's to set, and this clause
records the arithmetic, not a decision taken on its behalf.

**Also fixed by the same exchange.** The `lambda` guard of §12.10 was computing a **mean** over
**all** checkpoints rather than the minimum over the judged window — both optimistic in the same
direction. A state must be powered *throughout* the window it is judged on, not on average
across it, because the checkpoints producing the longest sub-threshold run are precisely the
ones the gate reads. NaCl CIP `1.72 -> 1.57` (smallest resolvable deficit `152 % -> 160 %`),
SSIP `62.3 -> 61.4`, verdicts unchanged; `scripts/methane_gates.py` already used the minimum.
Found because a number moved between two messages — **the tell that has worked all week.**

#### 12.13 The `lambda >= 16` guard keeps its threshold and loses its rationale (2026-08-15)

§12.10 justified `GATE_C_MIN_LAMBDA = 16` as *the `lambda` at which a 50 % deficit is a 2σ
effect*. Measured, that is false for the gate as implemented — **class 6 inside the guard
written to fix class 6**, and it survived review because the arithmetic was right about
something Gate C does not do.

`2/sqrt(lambda)` is a **single-checkpoint** criterion; Gate C requires a **contiguous
`0.20 T` run** below `0.5 Q*`. Contiguity suppresses false firing (intended) and true firing
(never accounted for). Closed by planting walker-conserving deficits in methane's **real**
8-seed traces — synthetic data cannot answer it, because the trace's correlation structure *is*
the quantity — and subsampling walkers to read the threshold across `lambda`:

| `lambda` | analytic 2σ | gate actually needs | | `lambda` | analytic 2σ | gate actually needs |
|---|---|---|---|---|---|---|
| 224.2 | 13 % | **60 %** | | 16.0 | 50 % | **65 %** |
| 147.0 | 16 % | **60 %** | | 9.2 | 66 % | **70 %** |
| 28.0 | 38 % | **65 %** | | 4.0 | 100 % | **75 %** |

0/8 seeds fire at a planted 50 %, 8/8 at 60 %.

**Binding changes.**

1. The threshold **stands at 16**; the rationale is corrected. `lambda >= 16` marks where
   counting noise stops dominating — not where a 50 % deficit becomes detectable.
2. **Every Gate C "no deficit" must be reported with the detection threshold**, not with the
   analytic figure. The licensed claim is *no deficit ≥ 60 % occurred*, and it is weaker than
   the direct occupancy bound that should accompany it.
3. **Detection is set by contiguity, not by `N`, above `lambda ~ 9`** — 60–75 % across a 50×
   range. Raising `N` to make Gate C more sensitive does not work; only the span rule moves it.
4. **A state near `lambda ~ 10` sitting at ratio 0.5 will not fire at any `lambda` on this
   ladder.** Where that is possible — C60's contact state is the live case — a windowed
   statistic must be **primary**, not secondary to Gate C.

**The planting was audited against the instrument bug that hit the replication, and the audit's
own strength is stated honestly.** The NaCl session replicated this on its own traces (0/8 at
50 %, 8/8 at **55 %** — same phenomenon, different geometry) after first getting "never fires up
to 90 %" from a planting that redistributed the removed mass proportionally to the *existing*
outside counts: at checkpoints where every walker is already inside the basin there is nothing to
be proportional to, the mass was dropped, and the basin's **share rose to 1.0000** — the opposite
of a deficit. Re-run here under three redistribution schemes (observed / bias-aware target /
uniform), the threshold is **identical to the last partial count** (0/8 at 50 %, 6-2-6 of 8 at
55 %, 8/8 at 60 %).

**That agreement is a consistency check, not independent evidence, and it should not be quoted as
the latter.** The per-state test reads only state `k`'s own column and `plant` sets that column to
`f * occ[:, k]` before touching anything else, so no redistribution scheme *can* move it: this
implementation is **structurally immune** to that bug class rather than empirically clear of it.
The immunity is the load-bearing claim; the three-way run is worth keeping only because it would
catch a future edit that reintroduces a renormalisation path. Asserted in
`test_the_planted_deficit_is_immune_to_the_redistribution_choice_by_construction`.

**The fire direction needs the floor too, symmetrically.** A *fire* is reliable
establishment-limited evidence only above the `lambda` floor: below it, `P < 0.5 Q*` is
arithmetically "the state is empty right now" and autocorrelation lets a well-established state
show a long empty run — `results/deca/screen_RETRACTED_no_min_count_guard/` is exactly that.
§12.10's exclusion is already two-sided for this reason; any downstream adoption that treats
contiguity as a one-sided detector must carry the floor as a precondition rather than assume it.

**Stated limitation:** the planted deficit is *stationary*. A deficit that decays as the bias
fills in would fire less readily, so **60 % is a floor on the detection threshold, not a
characterisation of it.** The decaying case was not measured and no claim is made about it.

**The verdicts are unchanged and better supported.** Methane's null rests on the direct
measurement — worst occupancy/`Q*` = 0.83 over 8 seeds × 3 states × the whole second half, so
the largest shortfall anywhere is **17 %**, 3.5× tighter than the gate could report. NaCl's
rests on its windowed band. Neither ever rested on the gate firing, which is the only reason
this correction costs nothing. Artifacts: `results/methane/screen_N512/gate_c_detection/`,
`scripts/methane_gate_c_detection.py`.

#### 12.14 The span frontier is a property of the SYSTEM, not of the gate (2026-08-16)

Two studies swept the required contiguous span at fixed system, planting 50 % deficits in real
traces. The result is not the frontier's location — it is that **the frontier does not transfer.**

| required span | methane FP / detect (24 state-seeds) | NaCl FP / detect (32 state-seeds) |
|---|---|---|
| 0.02 T | **0/24** / 21/24 | **9/32** / 31/32 |
| 0.05 T | 0/24 / 5/24 | 1/32 / 25/32 |
| 0.10 T | 0/24 / 2/24 | **0/32** / 11/32 |
| **0.20 T (prereg)** | 0/24 / **0/24** | 0/32 / **0/32** |

**Methane's frontier is at 0.02 T; NaCl's is at 0.10 T, five times tighter.** A study inheriting
methane's setting would have imported spurious fires on **28 %** of NaCl's state-seeds. The
false-positive rate is set by the system's own occupancy autocorrelation, so a campaign-level
recommendation of a *number* would be wrong.

**Binding rule.** Before freezing a span: **measure the frontier on the system's own traces, and
report the false-positive count beside the detection count.** A detection threshold quoted
without its false-positive count at the same span is not interpretable. Cheap once traces exist;
the positive control is the natural place to hang it.

**Both nulls survive a more sensitive gate, and the preregistered span is dominated in both.**
Methane fires 0/24 at 0.02 T where the gate catches 21/24 planted deficits (10x); NaCl 0/32 at
0.10 T against 11/32 (2x). In both, `0.20 T` gives identical zero false positives and zero
detection — it buys nothing. Reported as **post-hoc sensitivity analysis**: the preregistered
gate is 0.20 T, both verdicts rest on it, and this is what they *survive*. It matters more for
NaCl, the weaker null: it does not touch the orthogonality caveat, but it does remove *the gate
was too blunt to see it* as an alternative explanation.

**A candidate predictor, recorded BEFORE the second measurement exists.** Methane's occupancy
autocorrelation time over the judged window is **6.06 ps** (median of 24 state-seeds, range
0.84–22.39), and its frontier span of 4.0 ps is **0.7 tau_occ**; the preregistered 40 ps is
6.6 tau_occ. **Prediction: the frontier sits at order one tau_occ in any system, so NaCl's
0.10 T frontier should correspond to `tau_occ / T ~ 0.14`.** If it does, the rule becomes *set
the span near one occupancy autocorrelation time* — predictive rather than per-system
empirical. If it does not, this paragraph is the falsification and stands as written. The wide
per-state spread means the median is a crude summary and the prediction is order-of-magnitude,
not a fit.

#### 12.15 The `tau_occ` prediction is FALSIFIED as a coefficient, and my own number was censored (2026-08-16)

§12.14 predicted the frontier at order-one `tau_occ`, from methane's 4.0 ps against
`tau_occ = 6.06` ps, i.e. **0.7 tau**. NaCl measured `tau_occ = 42.2` ps (N-independent: 41.6 at
N=64, 42.2 at N=32 — a physical property behaving like one) and a frontier of **2.8–3.8 tau**.
That is a factor of 4–5, and the paragraph stands as its own falsification, as it said it would.

**It is worse than that, because my 0.7 was not a measurement.** §12.14 swept the span as a
*fraction of T* on a grid whose finest step was `0.02 T = 4.0 ps`, and the frontier came out at
the grid's own edge. Re-swept in **absolute time** down to 1 ps:

| N (fixed T = 200 ps) | frontier | in `tau_occ` |
|---|---|---|
| 512 | **<= 1 ps** | <= 0.17 |
| 128 | **<= 1 ps** | <= 0.17 |
| 32 | 8 ps | 1.32 |

**Methane's frontier is censored below 1 ps at production N, not 4 ps.** The coefficient
therefore spans **<= 0.17 to 3.8** across the two systems — at least 20x, not 5x. *A value equal
to your grid's boundary is a censoring, not a measurement*, and §12.14 reported one as the other
while carrying an explicit warning about crude summaries that did not cover this.

**What survives is the structural half, and it is NaCl's demonstration rather than my
prediction.** Re-sweeping in absolute time across two cells, they measure 120 ps and 160 ps where
a fixed-fraction rule predicts 120 and 240: **the frontier is approximately a physical time, not
a property of run length.** The `tau_occ` intuition was right about the *kind* of quantity and
useless about its size.

**A second contribution, measured here, that a single-cell sweep cannot see.** At fixed `T` the
frontier is flat from N = 512 to N = 128 and rises 8x by N = 32 — counting noise, the §12.10
mechanism, re-entering as `lambda` falls. So the frontier is **not** `tau_occ` alone but
approximately `max(physical floor from tau_occ, a counting-noise term growing as N falls)`.
Consequence for anyone reading NaCl's 120 -> 160 ps drift as residual `T` dependence: their two
cells differ in **both** `N` and `T`, and the drift is the same direction and plausible size as
the effect measured here. Separable by subsampling walkers within one cell, which costs nothing.

**The design defect NaCl found stands and is the practically important result.** With `N x T`
fixed, `span = 0.20 T` scales as `1/N`, so an N ladder judges its small cells with an instrument
`8x` blunter at N = 8 than at N = 64 — biased toward *hiding* establishment failure, worst
exactly where the ladder is meant to be most informative. **Methane is unaffected** (a single
cell at one N and one T, verified rather than assumed), but any ladder under a fixed `N x T`
budget inherits it. Fix: **set the span in absolute time from a short pilot, not as a fraction of
`T`** — which is `N`-invariant by construction and needs no coefficient to be transferable.

### Amendment 13 — two GPUs for methane, and the optimisation order (2026-08-12)

**Written before any methane box, trajectory, reference or gate result existed.** Only engine
parity and throughput numbers were in hand.

#### 13.1 Compute policy: methane runs on GPUs 2 and 3

§1 pins v2 to **exactly one GPU**, expandable to two **only if every GPU is occupied by others**.
That condition is not met and the expansion is taken anyway, so it is recorded here rather than
done quietly.

**Measured state of the node.** GPUs 0 and 1 carry another user's processes
(`run_OpenFWI_point_nobatch_{fno,cnn}.py`, elapsed 2 d 06 h and 2 d 10 h). GPUs **2 and 3 are
idle**. §1's rule was written when GPU 2 was the only free device and its purpose was to keep this
project off other people's hardware; taking the two devices nobody is using serves that purpose,
and GPUs 0 and 1 remain untouched.

**Authorised: methane may use GPUs 2 and 3, and no more.** One process per GPU, each pinned with
`CUDA_VISIBLE_DEVICES`, device idleness re-checked and recorded before each stage (Amendment
12.4). Pre-empting GPUs 0 or 1 is **not** authorised.

What this buys, and it is scheduling rather than efficiency: the confirmatory seeds are
independent, so a stage splits across the two devices at ~2× — and the OpenMM constrained-TI
reference can run on one device *while* the torch screen runs on the other, instead of queueing.

#### 13.2 Optimisation is deferred behind the Gate 0 verdict

Full-campaign cost at the measured 744 ns/day, one GPU:

| stage | aggregate MD | 1 GPU |
|---|---|---|
| constrained-TI reference (OpenMM) | 348 ns | 18 h |
| **`N = 512` screen — the Gate 0 verdict** | 819 ns | **1.1 d** |
| `N = 128, 256` screens (only if establishment-limited) | 614 ns | 0.8 d |
| FR rate calibration | 1 638 ns | 2.2 d |
| production, 3 arms x 16 seeds | 4 915 ns | 6.6 d |
| frozen-bias validation | ~1 230 ns | 1.7 d |

**~13 days to the end of the path, but 1.1 days to the decision that most likely ends it.**
Methane is preregistered as likely null (§9, Amendment 11.1), so the expected total is the first
two rows.

A hand-written fused Triton kernel for the pair term is the one remaining large win (the kernel is
memory-bandwidth bound; `torch.compile` already recovered 8.1× and tensor-op restructuring has
been measured to lose — Amendment 12.8). It is estimated at 3–10× for about a day of work, and it
is **deliberately not written yet**: it is scheduled only if the screen licenses production, where
it would pay for itself across the remaining 10+ days. If methane is ABF-sufficient the day is
never spent.

**This is scheduling, not scope.** No physical parameter, budget, gate or endpoint changes, and
the screen runs at the same `T_run`, `N` and seeds either way.

---

### Amendment 11 — the prior-art `c` ladder needs its own activity floor, and a wider span (2026-08-12)

**Status: adopted before any confirmatory five-arm run.** The calibration stage has run; the
confirmatory block (seeds 400–415) had not started when this was written, and
`results/wca_five_arm/confirm/raw/` was empty.

§3.4 says the prior-art selection intensities `c` are "tuned by the same procedure" as the mFR
rate ladder. Running that procedure surfaced two ways in which it does not transfer.

**(a) The `0.5 N` activity floor contradicts the matching objective.** §3.2 strikes a rung as
inactive unless `N_replacements >= 0.5 N`. With `N = 1024` that floor is **512**. The matching
target — `fr_estimated`'s own median turnover on the held-out seeds — is **457**.

So under the literal rule the proposed method is itself *inactive*, and every baseline is
required to select **harder** than the arm it is being matched to. That is precisely the
confound turnover matching exists to remove: §4.2's whole point is that no arm may win by
selecting more.

The floor was written for the mFR **rate** ladder, where it guards against a ladder that
"selects the arm that does nothing." That intent is preserved by taking the floor **relative to
the matched target**:

```
  N_replacements  >=  0.5 * target_turnover        (prior-art c ladder only)
```

The mFR rate ladder keeps `0.5 N` unchanged. On the calibration data this changes nothing for
`count_balancing` (it still selects `c = 1.0`; the rungs it strikes, `c = 0.1` and `0.3` at 73
and 168 replacements, are struck under either rule).

**(b) The ladder `(0.1, 0.3, 1.0, 3.0)` cannot match `book_laplacian`.** Measured:

| `c` | 0.1 | 0.3 | 1.0 | 3.0 |
|---|---|---|---|---|
| replacements | 2498 | 3282 | 3532 | 3592 |
| × target | 5.47 | 7.18 | 7.73 | 7.86 |

A **30×** change in `c` moves turnover **1.44×**. `S = c ∂²p/∂z² / p` is large enough that the
score sits at `score_clip = 2.0` across the entire original ladder, so `c` is nearly inoperative
there and the ladder bottoms out at 5.5× the target. This is consistent with the independently
measured clip behaviour in `tests/test_wca_prior_selection.py` (recentre+clip deviation
`2.6e-2` for `book_laplacian` against `1.9e-9` for `count_balancing`).

The ladder is extended **downward** to

```
  (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
```

The four original rungs are retained and their completed runs are reused unchanged, so this is
strictly an extension of the search, not a replacement of it.

**What is not changed.** The mFR configuration is still not retuned (§3.4). The held-out
calibration seeds (500, 501) are unchanged and still disjoint from the confirmatory block. The
endpoint, the confirmatory seeds and the arm list are unchanged.

**Reportable finding either way.** That `book_laplacian`'s turnover is nearly `c`-independent
over two decades is a property of the Chapter-6 rule on this system and is reported with the
Q1 result, whichever direction the comparison goes. If no rung matches the target, the arm is
reported as **unmatchable at fixed `score_clip`** rather than silently compared at 5.5×
turnover.

**Refinement, same day, before the confirmatory launched.** The extended ladder matched
`book_laplacian` at `c = 0.01` → 360 replacements, **0.79× target**: 21 % *under*-driven,
because turnover triples across the single rung `0.01 → 0.03` (360 → 1088). Three intermediate
rungs `(0.012, 0.015, 0.02)` were added. This can only move a baseline *closer* to mFR's
turnover — it strengthens the arm mFR is being compared against, never mFR — so it is
conservative with respect to the hypothesis under test.

Final matched intensities, on held-out seeds 500–501:

| arm | `c` | replacements | × target (457) |
|---|---|---|---|
| `book_laplacian` | 0.012 | 440 | **0.96** |
| `count_balancing` | 1.0 | 529 | **1.16** |

Both within 16 % of `fr_estimated`'s own turnover, so no arm in the confirmatory can win by
selecting harder. The saturation finding in (b) stands and is reported with Q1: `book_laplacian`
needs `c ≈ 0.012` to reach mFR's turnover, and is already fully clipped by `c = 0.1`.

---

### Amendment 12 — the genealogy gate was being read off the wrong statistic, and a reproducibility limit (2026-08-12)

Two findings surfaced while wiring the §4.3 checks into the Q1 analyzer. Both were found
*before* the confirmatory block was scored, and the first one applies retrospectively to a
already-reported result, so both are recorded here rather than in a results note.

#### (a) `ESS_anc / N >= 0.30` was never checked on WCA, and the obvious way to check it is wrong

`scripts/analyze_caseix_hp.py` verified four of §4.3's six primary conditions. It omitted **both**
genealogy gates. The earlier report that Case IX passed "all four preregistered checks" was
accurate as to the four it tested and **incomplete** as to §4.3, which lists six.

Applying the gate naively to the statistic the WCA runs already stored gives:

| arm | run-long `ESS_anc/N` | `w_max` |
|---|---|---|
| `fr_estimated` | 0.166 | 0.0195 |
| `fr_oracle` | 0.183 | 0.0195 |
| `sham_practical` | **0.191** | 0.0190 |

Read literally, Case IX fails at 0.166 < 0.30. But that reading cannot be right, and the sham row
is what proves it: `sham_practical` is **matched-turnover random-direction** selection, which is
not pathological by construction, and it sits at 0.191 — no better. A gate that fails its own
null control is measuring the wrong thing.

The cause is a definitional mismatch. `wca_abffr_core.py` initialises `ancestors` once and never
resets it, so `ancestor_ess` traces lineage from `t = 0` over the whole 120 000-step run; it
starts at exactly `N` and decays monotonically (1024 → 183). Run-long ancestral ESS decays toward
zero for **any** birth–death process, healthy or not, so a fixed floor on it is a cap on run
length, not a health check.

The gateway confirmatory that set the 0.30 floor recorded `ess_window_steps: 4000` and measured
`min_ess_frac` in the range **0.37–0.73**. It is a **windowed** statistic — coalescence over a
fixed horizon.

`SimConfig.ess_window_steps = 4000` and `min_ancestor_ess_window` are therefore added, reset on
that window and updated from the death/birth indices the birth–death routines already return.
It consumes **no RNG** and touches no dynamics, so it cannot perturb any arm; `abf` has no
genealogy and records `NaN`. Both statistics are stored side by side so they are never conflated
again, and §4.3's gate is evaluated on the windowed one.

**Retrospective obligation.** Case IX's stored artifacts predate this diagnostic, so its windowed
ESS is not recoverable without re-running. It is reported as **not evaluated** for that gate
rather than as passing. Its `w_max = 0.0195` — the gate that actually detects a lineage takeover,
and the one the gateway also passed at 0.0127 — does pass. The Q1 confirmatory, run on the same
system and the same frozen FR configuration, does record the windowed statistic, and is the
evidence for whether this configuration meets the gate.

#### (b) Runs are deterministic within a process, not across processes

Three `abf` runs at seed 400, identical spec hash `8df2be8349fc` and **zero** birth–death events,
returned integrated `L2(F)` of **37.63**, **47.93** and **39.15**. `torch.manual_seed` is set, so
the random stream is identical; the divergence enters through GPU kernel selection, which is
fixed within a process and varies between them, and is then amplified chaotically over 120 000
steps by `forces.scatter_add_`.

Consequences, stated plainly:

* **Within a process the sampler is deterministic** — the existing bit-identity test passes on
  back-to-back `abf` runs, which is also why the cause cannot be atomics.
* Every confirmatory block runs its arms and seeds **in one process**, so all arms share one
  kernel selection and the paired contrasts are internally consistent. The paired *differences*
  are far more stable than the levels: −11.88 % and −11.4 % from two independent processes whose
  `abf` levels differ by 4 %.
* **Absolute `I_F` levels are not reproducible across processes and must not be quoted as
  reproducible.** Only within-block contrasts are.

No result is retracted on this basis. It is a limitation on what may be claimed, and it is the
reason arms are never compared across separate invocations.

---

### Amendment 14 — NaCl opens as a concurrent branch on the published Talmazan model (2026-08-13)

**What had been seen when this was written:** the methane branch had a parity-passed engine, an
accepted reference and a running ABF-only screen with **no screen verdict yet**. For NaCl, the
Supporting-Information archive of Talmazan et al. 2025 had been downloaded and hashed, the
published model had been loaded into OpenMM and one single-point energy computed
(`results/nacl/stage0/`). **No NaCl trajectory, reference, screen, gate verdict or mFR result of
any kind existed.** No methane result influenced any NaCl clause, and nothing below changes any
methane clause.

#### 14.1 Concurrency and ownership

§0 (as amended by 11.2) ordered methane ahead of NaCl. NaCl development is now **concurrent**:
the methane session owns the generic periodic engine (`src/methane/{nonbonded,pme,dynamics,cv}`),
and the NaCl session **consumes** it — model layer, hydration observables and drivers only, in
`src/nacl/`. One narrow engine generalisation is authorised: making the `PairTerms` split-path
LJ-exclusion assertion graceful (CHARMM TIP3P hydrogens carry LJ; the assertion is
SPC/E-specific), with the methane execution path untouched.

#### 14.2 The physical model is the published tutorial system, extracted verbatim

`docs/SPEC_nacl_water.md`, committed with this amendment and frozen, fixes: the exact SI files
(hashed), parameters read back out of the OpenMM `System` (never transcribed), the published
protocol (300 K, 12→10 Å switched cutoff, PME, rigid TIP3P, `fullSamples 500`, colvar domain
[2, 14] Å at 0.1 Å bins, walls as published, 100 ns ABF budget), and the declared deviations
(NVT at frozen volume; BAOAB/M-SHAKE; no MTS; order-5 pinned PME; frozen-angle removal; no
local NAMD — the shipped tutorial outputs are the literature anchor).

#### 14.3 Everything already frozen stays frozen

Seeds 4000–4007 / 4100–4103 / 4200–4215 (§5), the fixed-compute map `N·T = 100 ns` over
`N ∈ {8,16,32,64}` (§8.2), the gate order 0→A→B→C→D and all thresholds (Amendments 7–10), the
§4.3 success and novelty criteria, and the §4.5 reference rule apply unchanged. NaCl runs **all
five production arms** if licensed (§3 primary arm set): after WCA's Q1 tie, the prior-art arms
are mandatory for the molecular Q1 answer. Amendment 12.7's three-arm reduction was and remains
methane-specific.

#### 14.4 Compute

Amendment 13.1 gave methane GPUs 2 and 3. From the first NaCl GPU stage onward the split is
**one device per study**: methane on GPU 2, NaCl on GPU 3 (or the reverse, recorded per stage),
each pinned with `CUDA_VISIBLE_DEVICES`, one process per GPU, idleness re-checked and recorded.
Until NaCl needs a device, methane may keep both. GPUs 0 and 1 remain untouched. This is
scheduling; no physics, seed, budget, gate or endpoint changes.

#### 14.5 Numbering note

Two later WCA amendments were appended with duplicate numbers 11 and 12 (2026-08-12, lines
following Amendment 13). They are left as committed; this amendment is numbered 14 as the next
in true sequence, and any citation of "Amendment 11/12" states which of the two it means.

---

### Amendment 15 — NaCl execution discipline: the dt decision rule, the window ordering, gated launches, and two GPUs (2026-08-13)

**What had been seen when this was written, stated because it matters here:** the first NaCl
dt-gate run and its retraction. That run printed `2 fs: dT = 2.67 K FAIL` with error bars later
shown to be ~2.2× too small (honest reading: 2.67 ± 1.07 K, 2.5σ); `1 fs: dT = 1.37 K PASS`.
The convention hypothesis was tested and refuted (`getKineticEnergy` ≡ explicit `Σ½mv²`, gap
+0.000 K). **No NaCl free-energy, reference, screen or mFR datum existed.** The rule in 15.1 is
frozen with the borderline 2 fs number already known, and is deliberately conservative so that
knowledge cannot be exploited in either direction.

#### 15.1 The timestep decision rule, frozen before the rerun

One preregistered run of the rebuilt gate; no extensions, no reruns on the same question:

* constraint clause (deterministic): float64, 4 walkers × 5 ps; `max violation <= 1e-8 nm`;
* equipartition clause (statistical): **production dtype** (float32), 32 walkers × 60 ps
  (10 ps warm) against 4 independent OpenMM replicas × 60 ps; both SEMs by **blocking** with
  10τ ≈ 5 ps blocks (calibrated on a known-truth AR(1) series: 2τ blocks understate by 26 %,
  10τ by 6 %; the residual bias is negative, so PASS is the weaker claim);
* with `σ = sqrt(sem_torch² + sem_openmm²)`:
  `PASS` iff `dT + σ <= 2.0 K`; `FAIL` iff `dT − σ > 2.0 K`; else `INDETERMINATE`.

Decision: **2 fs PASS → 2 fs. 2 fs FAIL or INDETERMINATE → 1 fs.** At 1 fs, `FAIL` of either
clause (confident, same definitions) → **STOP, engine defect**; `INDETERMINATE` at 1 fs → 1 fs
(the conservative endpoint of the preregistered ladder). The decision is written once to
`results/nacl/stage1/dynamics_gate.json` and never revisited because later free-energy results
look better or worse.

#### 15.2 `Omega_thermal` and reference acceptance: the ordering is one-way

SPEC §2.1 defines the window from "the accepted reference" while §5 uses the window inside
acceptance — a circularity if read as iterative. It is not iterative, and the operational order
is frozen: three raw builds → candidate consensus `F` → window from the fixed 15 kT rule on
that candidate → acceptance evaluated on that fixed mask → **PASS accepts reference and window
jointly; FAIL is a STOP.** There is no path on which the mask is adjusted and acceptance
re-evaluated. (This is what `nacl_ti_analyze.py` already computes; the amendment removes the
ambiguity in the spec text, not the behaviour.)

#### 15.3 Launch discipline: no autolaunch, pinned code, mechanical ladder

The GPU-idle autolaunch watcher is **retired** (nine defects were found in already-tested code
in one night; "GPU idle" is no longer a sufficient launch condition, and the shared tree is
edited concurrently by the methane session). Every expensive NaCl run now requires:

* a **pinned worktree**: NaCl production processes run from a detached worktree at the commit
  recorded in `results/nacl/PINNED_COMMIT`; the methane session's edits to the shared engine
  cannot reach it. Untracked data inputs/outputs are shared with the main tree by symlink;
  the cleanliness requirement applies to **code paths** (`src scripts tests docs`);
* the **launch ladder** (`scripts/nacl_launch_ladder.sh`): preflight (pinned commit matches,
  code tree clean, NaCl test suite passes, target GPU idle, launch manifest written) →
  Triton correctness → idle-device throughput → TI smoke run → the 15.1 dt gate →
  checkpoint-resume verification on the real driver → **STOP**. The ladder never launches the
  reference; that is a separate, reviewed action recorded in `docs/NACL_EXECUTION_STATE.md`;
* a failed step stops the ladder. A defect found mid-ladder is patched → tested → committed →
  **new pinned commit** → ladder restarted from the top. No opportunistic patch-and-continue.

`docs/NACL_EXECUTION_STATE.md` is the execution state machine: current pinned commit, stage,
last gate verdict, the single next permitted action, and forbidden actions. It is updated after
every stage and read at the start of every session.

#### 15.4 Compute: two GPUs for NaCl once methane concludes on GPU 3

By explicit user directive (2026-08-13): the campaign runs concurrently on GPUs 2 and 3 —
methane on GPU 3 until its screen and benchmark complete, NaCl on GPU 2 from the seed-5004
handover. **Once methane vacates GPU 3, NaCl may use both devices**, one process per GPU,
pinned with `CUDA_VISIBLE_DEVICES`, idle state recorded per stage. First use: the TI
reference's three independent builds split across the two devices. This is scheduling; no
physics, seed, budget, gate or endpoint changes.

### Amendment 16 — C60/water opens as a concurrent branch on the published Zangi model (2026-08-14)

**What had been seen when this was written:** methane was CLOSED (ABF-sufficient, 8/8 seeds,
Gate C worst ratio ~0.83, no establishment window). NaCl had a parity-passed engine, an
accepted TI reference, Gate 0 and Gate A passes, and the N = 64 screen mid-flight (seeds
4004–4007 in progress on GPU 2) with **no Gate B/C verdict**. For C60, the primary source —
Zangi, *J. Phys. Chem. B* 118, 12263 (2014), `10.1021/jp508174a` — had been downloaded and read
(`cache/zangi2014/zangi2014_jpcb.pdf`), and `src/c60/geometry.py` existed with its cage-geometry
unit checks. **No C60 trajectory, box, engine parity result, reference, screen, gate verdict or
mFR result of any kind existed.** No NaCl screen outcome influenced any clause below, and
nothing below changes any methane or NaCl clause except the 15.4 scheduling row noted in 16.4.

#### 16.1 Why C60, and why now

Methane closed as ABF-sufficient because its solvent response is local and fast: the null was
attributed, not merely observed. The literature contrast to that null is the large-solute side
of the hydrophobic length-scale crossover (Remsing & Weeks; Lum–Chandler–Weeks), and Zangi 2014
provides a published, unusually simple C60 pair system in explicit water whose contact state
hosts a distinct confined-interfacial-water population (I2) that methane is explicitly too
small to form. Whether that produces `T_hit << T_est` is **unknown and is the question**. C60
is opened as a blinded ABF eligibility study under the universal gates: the ABF-only screen
runs first, and an mFR arm exists **only if Gate 0 → A → B → C → D all pass**, on exactly the
terms deca-alanine, methane and NaCl were held to.

#### 16.2 The frozen specification

`docs/SPEC_c60_water.md`, **committed with this amendment and frozen**, fixes: the Zangi model
(2 rigid C60 + 1282 TIP4P-Ew, `sigma_CO = 0.319 nm`, `epsilon_CO = 0.392 kJ/mol`, geometric
C–C rule, intra-cage exclusions, pentagon-facing out-of-registry orientation, 1.0 nm cutoffs,
300 K), parameters read back out of the OpenMM `System` (never transcribed), the axial COM
separation CV on `[0.908, 2.428] nm` with its exact linear geometry, the 68-window constrained
reference with 4-family pools and 3-build acceptance, the Gate 0/A/B/C/D definitions by
reference to Amendments 7–10 with Gate A in the corrected `TV[p(xi|Y)]` orientation from the
start, the fixed-compute screen map `N·T = 128 ns` over `N ∈ {8, 16, 32, 64}` run N = 64
first with the methane 12.5 stopping rule, and the §4.3 success and §4.5 reference rules
unchanged. The declared deviations from the paper are collected in its §14.

#### 16.3 Seeds and gates

Seeds **7000–7007 / 7100–7103 / 7200–7215** join the §5 table (row added with this amendment);
disjoint from every other block by construction. **The gate order 0→A→B→C→D, every Amendment
7–10 threshold, the §4.3 success criteria and the §4.5 reference-quality rule apply
unchanged.** One C60-specific interpretability safeguard is frozen in advance (SPEC §8): a
Gate C deficit is headline-reportable only where `N Q*_k >= 3` over the deficit span — a
guard against small-`N` discreteness masquerading as establishment limitation, adopted before
any screen datum exists.

#### 16.4 Compute: GPU 3 reassigned from the NaCl option to C60 (scheduling only)

By explicit user directive (2026-08-14, "we have H200 on GPU 3"): **C60 takes GPU 3; NaCl
keeps GPU 2.** This supersedes the Amendment 15.4 clause that let NaCl expand onto GPU 3 once
methane vacated it — NaCl's remaining ladder (N = 32/16/8 map, ~2.4 GPU-days) continues on
GPU 2 sequentially after its N = 64 half-B block completes there. One process per GPU, pinned
`CUDA_VISIBLE_DEVICES`, idle state recorded per stage, whole seed blocks in one process. The
15.3 launch discipline (pinned worktree via `results/c60/PINNED_COMMIT`, mechanical ladder,
no autolaunch) applies to every expensive C60 run. This is scheduling; no physics, seed,
budget, gate or endpoint changes for any study.

#### 16.5 Engine ownership

The C60 session owns `src/c60/` and `scripts/c60_*`. `src/methane/` and `src/nacl/` are not
modified; stateless functions may be imported, structural adaptations (rectangular cell,
4-site water with virtual-site force redistribution, unswitched LJ, fixed-cage solute with a
single `xi` degree of freedom) are copied and owned by `src/c60/` so the methane and NaCl
execution paths are untouched (the Amendment 14.1 discipline).

#### 16.6 Clause (iii) of the SPEC §5 reproduction gate: definition fixed by measurement, before any solvent datum (2026-08-14)

**What had been seen:** vacuum cage-cage LJ arithmetic and the Stage-0 parity suite only.  No
reference window, no solvent trajectory statistic, no gate input of any kind existed.

The frozen clause read "the direct (vacuum) cage-cage LJ energy at the contact minimum
reproduces ~-18.5 kJ/mol within 0.5 kJ/mol" without fixing (a) truncated at the simulation
cutoff or untruncated, and (b) evaluated at the paper's contact separation or at the direct
term's own minimum.  Measured, with the frozen Hedberg geometry and the §1.1-derived
parameters: untruncated -17.75 at 0.968 nm and **-18.78 at its own minimum (0.982 nm)**;
1.0 nm-truncated -17.02 at 0.968.  Across published C60 geometries the same quantity moves by
~0.6 kJ/mol at fixed separation (David-geometry check: -18.36 at 0.968), which exceeds the
clause's own tolerance -- the clause as written tested the *cage geometry choice* (Declared
Deviation 2), not the parameter derivation it was meant to gate.

**Fixed definition:** clause (iii) is the **untruncated** vacuum cage-cage LJ sum at the
**minimum of the direct interaction over d**, compared to the paper's -18.5 kJ/mol within
0.5 kJ/mol.  Measured value: **-18.78 kJ/mol at d = 0.982 nm — PASS.**  The geometry
sensitivity is recorded above rather than absorbed into a wider tolerance.  Clauses (i), (ii)
and (iv) of the reproduction gate are untouched, and remain decided by the solvent reference
that does not yet exist.

#### 16.7 The Gate C small-N safeguard of 16.3 is replaced: N Q* >= 3 has zero discriminating power (2026-08-15)

**What had been seen:** no C60 reference, screen, or Gate C datum of any kind — the Stage-0
parity suite and the frozen box only.  The defect was pointed out by the NaCl session
(cross-session message, 2026-08-15 00:1x UTC), which had just applied the same arithmetic to
its own map and dropped two unclassifiable cells; the derivation is verified here
independently rather than adopted on authority.

**The defect.** Gate C fires on occupancy `< 0.5 Q*_k(t)` sustained over `>= 0.2 T`.
Occupancy is a count with expectation `lambda = N Q*_k`; its fluctuation scale is
`sqrt(lambda)`.  Resolving a 50 % deficit at two sigma requires `0.5 lambda > 2 sqrt(lambda)`,
i.e. **`lambda > 16`**.  At the frozen guard value `lambda = 3`, even a permanently *empty*
state sits only `3/sqrt(3) = 1.73` sigma from a healthy one — no deficit of any size is
resolvable, and a healthy state spends ~20 % of snapshots below the deficit line
(`P(count <= 1 | lambda = 3) = 0.199`), so persistent-looking spans arise by chance.  The
guard as frozen *admits precisely the cells in which the gate it guards cannot function*.
This repository has already retracted one verdict to this failure class
(`results/deca/screen_RETRACTED_no_min_count_guard/`): a "fired" Gate C licensing mFR on a
state that could not hold walkers at all.

**The replacement, frozen now:**

```
  a Gate C deficit is headline-reportable only where  min over the judged span of
  N Q*_k(t)  >=  16
```

* the statistic is the **minimum over the judged span, not a mean over checkpoints** — a mean
  lets a well-populated stretch mask exactly the configuration the guard exists to reject;
* `16` is the two-sigma working value for the frozen `0.5` deficit ratio, and matches the
  NaCl convention, keeping the two molecular studies comparable;
* a cell in which no reference-defined state clears the floor is reported as
  **UNCLASSIFIABLE at that N** — never silently classified, and never a licence to extend
  the run until it classifies (the frozen-budget rule is untouched);
* the persistence requirement is unchanged; its correlated-sample power is deliberately not
  credited toward a lower floor — leniency here is what the deca retraction bought.

16.3's `N Q* >= 3` clause and the SPEC §8 line implementing it are superseded; nothing else
in 16.3 changes.  This amendment precedes every Gate C datum by construction and cannot have
been tuned against one.

#### 16.8 Partition-freeze ordering, the lambda table, and the cells that cannot run (2026-08-15)

**What had been seen:** still no C60 reference, screen or Gate C datum — this extends 16.7
within the same pre-data window, again prompted by the NaCl session's cross-audit and adopted
after independent verification.

**(a) The partition is frozen before any occupancy is read, explicitly.** The Amendment 3
state rule (minima, < 2 kT merges from the higher minimum, boundaries at maxima) is evaluated
on the accepted reference, and the resulting partition **with its per-state lambda table** is
recorded in `results/c60/reference/RESULT.md` *before any screen cell launches*.  After any
occupancy has been read, the partition is never revisited: re-merging states because the
narrow ones came back unpowered would be choosing the partition against the result, turning
the 16.7 power guard into a licensing search.  "The merged state clears 16 but the contact
state does not, so C60 is classifiable only at the coarse partition" is an acceptable
recorded outcome; a post-hoc merge is not.

**(b) The lambda table is computed from the reference, before compute is spent.**
`lambda_k(N) = N Q*_k` needs no sampling.  It is emitted for every ladder N under both the
unbiased reference weights (`Q*_k` at `B = 0`, the screen's t = 0 target) and the
fully-flattened bias (`Q*_k` = width fraction, the late-time target), bracketing the
realised trajectory of `Q*_k(t)`.

**(c) Cells that cannot classify are not run.**  Because `Q*_k <= 1` identically,
`min-span N Q*_k >= 16` is unsatisfiable for **every** state at `N = 8` and degenerate at
`N = 16`: those two cells cannot produce a headline-reportable Gate C deficit under any
geometry, and — by the same arithmetic — their "no deficit" reading carries no evidential
weight either (the no-data-reads-as-PASS class).  Under the frozen sequential rule they
would only ever run after an establishment-limited `N = 64`, for an eligibility question
they cannot answer.  **They are therefore struck from the executable ladder**: the C60 map
is `N = 64` first, then `N = 32` only if 64 is establishment-limited; the `N in {8, 16}`
rows remain in the SPEC as documentation of why they cannot run (~0.75 GPU-days not spent
on unclassifiable cells, the NaCl precedent).  Gate B alone could still function at small
`N`, but a cell that can return at most half a verdict is not run on the frozen budget.

The 16.2/SPEC §7 ladder text is superseded to this extent; budgets, seeds, order and every
other clause are unchanged.

#### 16.9 Teleport preparation is replaced by constant-rate cage dragging (2026-08-15)

**What had been seen:** no reference window, gate statistic or screen datum — only three
preparation failures: (i) the ladder smoke died on a singular M-SHAKE matrix after the
dry-anchor teleport (cages placed at 0.968 nm into 2.428 nm-equilibrated water; a clipped-SD
reach of 0.06 nm cannot clear ~0.25 nm overlaps); (ii) the dt gate's first read was
RETRACTED because its 0.968/1.20 nm spots went NaN from the same teleport (equipartition
itself passed at 0.15 K); (iii) the whole-molecule clash pusher, tested on the same case,
**diverges structurally**: at contact the inter-cage gap is 0.256 nm wide, so no radial
ejection can clear both cages, and ejections buried waters in each other (max force rose
from 7e13 to 2e16 kJ/mol/nm, 14 waters unresolved).

**Replacement (mechanism only; family semantics, budgets, windows, seeds unchanged):**
every cage placement at a new separation is reached by a **constant-rate drag** — the cages
move linearly in `xi` at `<= 0.04 nm/ps` (up to 40 ps for the longest 1.46 nm traverse)
while the water propagates, with a per-site force clamp of `5e4 kJ/mol/nm` active during
the drag only; each drag is followed by the frozen settle/equilibration at fixed `d`, which
is what sets the sampled ensemble.  Family preparations become:

```
  wet(d_k)   drag DOWN from a 2.428 nm-equilibrated snapshot        (water-rich history)
  dry(d_k)   drag UP   from a 0.968 nm-equilibrated anchor          (water-poor history)
  bulk(d_k)  drag DOWN from an independent wet snapshot at HALF rate (most adiabatic control;
             replaces "equilibrated at d_k directly", which presumed an unintegrable teleport)
  hot(d_k)   dragged state + 0.05 nm water noise + clash push + SD  (destroyed interface)
```

The dry anchor itself is made by dragging 2.428 -> 0.968 and equilibrating.  The Gate 0 pool
preparations (SPEC §6) use the same mechanism.  Drag time does not scale in smoke mode (the
rate cap is physical).  Preparation costs ~35-70 ns aggregate per build, recorded as prep
outside the production accounting.  Force guards remain: a state with non-finite or
`> 1e6 kJ/mol/nm` forces after preparation raises rather than samples, and a NaN trajectory
raises rather than writing a verdict file.

**16.9 addendum (2026-08-15, before any reference datum):** the hot-family noise applies to
waters **outside the inter-cage gap cylinder** (`|u| < xi/2 + 0.1 nm` and `w < 0.45 nm`
exempt).  Measured: at near-contact windows one replica was unclearable across four
independent 0.05 nm draws — noise directed into a sterically closed pocket does not
"destroy an interface", it inserts a water no preparation can remove (the same 0.256 nm
geometric limit as the pusher).  The exterior noise, the frozen sigma and the
rejection-sampling loop are unchanged; the destroyed-interface semantics live in the I1
band, which is exterior.

#### 16.10 A pre-committed secondary statistic for sub-floor states (2026-08-15, before any reference or occupancy datum)

**What had been seen:** no C60 reference window, gate statistic, occupancy or screen datum.
NaCl had just closed ABF-sufficient, its unpowered CIP state cleared by a sliding-window
statistic that was a live branch of its pre-commitment; the construction is adopted here on
the same terms — frozen before the number can exist, with the inconclusive branch explicit.

For any reference-defined state whose 16.7 floor fails (`min-span N Q*_k < 16`) at an
executable cell, the following **secondary** statistic is computed alongside the
UNCLASSIFIABLE report, never in place of it:

```
  W_k(t)  =  occupancy of state k averaged over the sliding window [t, t + 0.2 T]
  D_k     =  min over t in the second half of  W_k(t) / <Q*_k over the same window>
  error   =  spread of D_k across the 8 independent seeds (SEM; no Poisson or
             autocorrelation assumption enters)
```

Pre-committed outcomes, decided by where the ±2 SEM band of the seed-mean D_k lies against
the frozen 0.5 ratio: entirely below -> **powered deficit evidence** (reported as secondary,
Gate C still does not fire on an unpowered state); entirely above -> **powered no-deficit
evidence**; straddling -> **INCONCLUSIVE, reported as such**.  The statistic is powered on
its own terms or it says so; it is never used to rescue an mFR licence — 16.7's floor
remains the licensing guard — only to let a sub-floor state say something quantitative
rather than nothing.  NaCl's realised example (CIP, worst window 1.371 ± 0.106 at N = 32)
is the calibration precedent.

**16.10 clarifications (2026-08-15, same pre-data window, from the NaCl session's caveat):**
(i) the cited NaCl precedent calibrates the statistic's **exclusion** side only (a band
clearing 0.5 from above); its detection side is validated against planted synthetic deficits,
never against physics that actually failed — if a C60 band lands NEAR 0.5, the INCONCLUSIVE
branch is doing real work; (ii) whenever the statistic is computed for a sub-floor state, it
is ALSO computed for the best-powered state of the same cell as a **positive control** — a
systematic offset must show up as the well-behaved state missing 1.0, not as the interesting
state looking odd; (iii) the window length is the gate's own contiguity requirement
(`0.2 T`) and is **never adjusted**; lengthening a window until a band separates is the same
move as relaxing the 16.7 floor, and is forbidden by the same reasoning.

#### 16.11 The windowed band becomes the PRIMARY establishment instrument; the contiguity gate is demoted to a one-sided detector (2026-08-16, before any C60 occupancy datum)

**What had been seen:** no C60 reference window, occupancy or screen datum (the reference
campaign had not launched).  The methane session measured, and the NaCl session
independently replicated, a planted-deficit calibration of the unmodified Gate C on the REAL
occupancy traces of both closed studies:

```
              45 %   50 %   55 %   60 %+     <- planted stationary deficit
  NaCl N=64   0/8    0/8    8/8    8/8
  NaCl N=32   0/8    0/8    8/8    8/8
  methane     ~60 % firing threshold; 0/8 at 50 %; 60-75 % across a 50x lambda range
```

**Zero seeds fire at the 50 % the gate nominally tests, at any lambda.**  The cause is
structural, not statistical power: at a 50 % deficit the mean occupancy sits exactly ON the
`0.5 Q*` line, noise lifts ~half the checkpoints above it, and the required contiguous span
almost never forms.  Contiguity, not counting noise, sets the detection floor — so the 16.7
floor's rationale ("lambda >= 16 resolves a 50 % deficit at 2 sigma") was correct arithmetic
about a single-checkpoint quantity **the gate does not compute**.  Both closed verdicts in
fact rest on band statistics (methane: worst ratio 0.83, a 17 % shortfall its gate could not
have reported; NaCl: SSIP worst-window band [0.974, 0.991], ~18x tighter than its gate).

**Frozen consequences for every executable C60 cell:**

1. **The 16.10 sliding-window band is the PRIMARY establishment instrument for every
   relevant state** (not only sub-floor states), with pre-committed branches on the seed-mean
   band (±2 SEM across the 8 seeds) against the frozen 0.5 ratio: entirely below ->
   **establishment-limited** (a measured sustained deficit); entirely above -> **powered
   no-deficit**, reported with the band's own exclusion strength ("excludes deficits
   > X %"); straddling -> **INCONCLUSIVE**, never converted to ABF-sufficient.
2. **An ABF-sufficient verdict requires the powered no-deficit branch for every relevant
   state.**  The mere absence of a contiguity fire is no longer evidence of anything.
3. The contiguity Gate C is computed and reported unchanged for cross-campaign
   comparability, as a **one-sided detector**: a FIRE is reliable evidence (its realized
   firing floor is a >= 55 % deficit) and is accepted as establishment-limited; a non-fire
   carries no weight.
4. The 16.7 floor continues to govern where the contiguity detector is binding; the windowed
   instrument self-reports its power through its band width and the mandatory positive
   control (16.10 clause ii).  The window length remains the gate's own `0.2 T`, immutable.
5. The detection-side caveats are carried, not waved off: the planted deficits are
   stationary (a floor on the real detection threshold — a decaying deficit fires less
   readily), and the windowed statistic's fire-side calibration still rests on planted data.
   The NaCl session's instrument bug (proportional redistribution silently producing a
   basin share of exactly 1.0000 — the tell) is recorded as the calibration's own audit
   trail.

The C60 gates code does not exist yet; it implements this amendment from the start rather
than being retrofitted.  Nothing here touches Gates 0/A/B, the budgets, seeds, or the mFR
prohibition.

**16.11 clarification (2026-08-16, same pre-data window, from the NaCl session's cross-read):**
clause 3's "a FIRE is reliable evidence" does not carry its own precondition and must:
at low lambda the fire direction degenerates to counting noise — when `0.5 lambda < 1` the
criterion is arithmetically "the state is empty right now", autocorrelated emptiness produces
long contiguous runs on perfectly established states (`P(empty) = 21 %` per checkpoint at
`lambda = 1.57`), and this is precisely the retracted deca verdict
(`results/deca/screen_RETRACTED_no_min_count_guard/`).  The symmetric statement, frozen:

* a contiguity **FIRE** is reliable establishment evidence **only where the 16.7 floor holds**
  (`min-span N Q*_k >= 16`); below it a fire is NOT interpretable and contributes nothing —
  the state's classification rests entirely on the windowed band;
* a **non-fire** carries no weight anywhere (16.11 clause 2);
* the **windowed band** is primary in both directions (16.11 clause 1).

The contiguity gate is thereby a detector that can only ever say "yes", and only in its
powered regime — reported alongside for campaign comparability, never decisive alone.

**16.11 second addendum — the contiguity span is a system property; C60 measures its own,
decides with neither (2026-08-16, still before any C60 occupancy datum):** the methane and
NaCl sessions measured the span's efficient frontier on their own real traces and disagree by
5x (methane ~0.02 T, NaCl ~0.10 T; at 0.02 T NaCl shows 9/32 false fires where methane shows
0/24) — the safe span is set by each system's own occupancy autocorrelation, i.e. by the
physics under measurement, which makes it exactly the kind of parameter 16.11 already
declines to let carry a verdict.  Frozen dispositions: (i) the C60 span REMAINS `0.2 T` for
both instruments — the row where both systems measured zero false fires — so the one-sided
detector's fire stays safe and the band stays conservative; the span is never re-chosen,
in either direction, for any reason; (ii) the "realized firing floor >= 55 %" quoted in
clause 3 is a property of span `0.2 T`, not of any system — the 55/60 % inter-study gap is
fully accounted for by span and is not carried as a C60 expectation; (iii) once the first
executable cell's traces exist, the span sweep is run on the UNMODIFIED traces and the
false-fire count per span is a **mandatory reported diagnostic** (rule-of-three bounds on
zero counts), explicitly non-decisional — it characterises the instrument on this system's
autocorrelation, alongside the verdict, never inside it.

#### 16.12 Absolute-time instrument windows; the contiguity statistic loses all verdict weight (2026-08-16, before any C60 occupancy datum)

**What had been seen:** no C60 reference or occupancy datum.  The NaCl session measured
`tau_occ = 42.2 ps` (median over 32 state-seeds, **N-independent**: 41.6 at N=64, 42.2 at
N=32) and located the false-fire frontier at **120 ps (N=64) / 160 ps (N=32)** — an
approximately absolute physical time (2.8-3.8 tau_occ), where a fixed-fraction rule predicts
1/N scaling (120/240).  Under `N x T = const`, the frozen `0.2 T` span therefore runs from
7.4 tau_occ (N=64) to 59 tau_occ (N=8): the instrument becomes up to **8x blunter at small
N, in the direction that hides establishment failure**, precisely where the ladder is most
informative.  Methane's frontier coefficient (0.7 tau) differs from NaCl's (2.8-3.8 tau), so
no coefficient transfers; only the parameterisation lesson does.

**Frozen dispositions:**

1. **Every Gate C persistence/window quantity is set in ABSOLUTE time: 400 ps** — the
   N = 64 anchor cell's `0.2 T`, i.e. the campaign-inherited value at the anchor, applied
   identically to every executable cell.  Cells are thereby compared on identical instrument
   terms; the 1/N blunting cannot arise.  The window is never re-chosen, in either
   direction, for any reason (the 16.10 immutability clause carries over to the absolute
   form).
2. **The contiguity statistic loses ALL verdict weight** (supersedes 16.11 clause 3's
   one-sided evidence): its nominal threshold is unreachable (16.11), its safe span is
   non-transferable (16.11 second addendum), and its absolute-time behaviour under the
   budget rule is N-dependent (this amendment).  It is computed and reported for
   cross-campaign comparability, with any fire arbitrated by the band: band-below ->
   the band already decides; band-straddle -> INCONCLUSIVE; band-above -> the fire is
   recorded as an instrument artefact.  **The windowed band is the sole establishment
   instrument**, and every claim it makes quotes its window explicitly ("no deficit
   sustained over >= 400 ps" — not "no deficit").
3. `tau_occ` for C60 is measured from the screen traces by the NaCl recipe (per-state-seed
   occupancy autocorrelation, median; reading a correlation time reveals no verdict) and
   **reported** alongside, with the 400 ps window expressed in tau units; the false-fire
   span sweep of the second addendum remains a mandatory, non-decisional diagnostic.  If
   400 ps proves to sit below C60's own frontier, that is reported as an instrument
   characterisation — it cannot corrupt a verdict, because the contiguity statistic no
   longer carries one.
4. The 16.8 strike of `N in {8, 16}` stands unchanged, now on budget-and-precedence grounds
   (the sequential rule reaches N = 32 only on an establishment-limited N = 64); it is not
   relitigated by the band's self-reporting power.

This is the terminal form of the Gate C instrument for C60: a measured band with an
absolute-time window and pre-committed branches, a reported-only contiguity statistic, and
every power and calibration number attached to the object it was measured on.
