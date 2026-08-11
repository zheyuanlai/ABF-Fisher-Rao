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
regime map as its own category.
