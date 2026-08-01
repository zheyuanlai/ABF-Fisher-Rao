# ACE-VAL-NME STAGE 0 HANDOFF

Branch `alanine-dipeptide` (Val work is additive; nothing under `src/alanine/` or
`src/alkanes/` was modified). 2026-08-01.

> **STATUS: Stage 0 complete and gated. Gate V1 PASSES. Gate §32 selects ξ = (φ, χ₁) and
> rejects (ψ, χ₁). The decisive gate V3 — discovery vs establishment — is NOT yet tested,
> so nothing here says mFR will help. Results in the RESULTS section at the end.**

Companions: `ALANINE_ORACLE_PILOT_HANDOFF.md` (the alanine null that motivates this),
`ALANINE_EXECUTION_DECISION.md` §7 (the V1–V4 screening gates this study must clear).

---

## 1. Headline

Ace-L-Val-Nme is built, validated against OpenMM, and gated by 32 new tests (18 Stage-0 +
14 restraint/MBAR/state-counting). The physical model is the alanine one **verbatim**. A cheap
minimum-energy-path pre-screen says χ₁ has three well-separated rotamer wells with
**10–17 kT MEP barriers** at every backbone region tested, and at least two rotamers within
~2.5 kT at each. The free-energy measurement has since confirmed this: **gate V1 passes** with
barriers of 11.3–17.9 kT, and the §32 screen selects **ξ = (φ, χ₁)**. See RESULTS below.

**The most consequential finding is not a number, it is a design correction that the
measurement forces.** See §3.

## 2. What was built

```
src/valine/__init__.py      re-exports
src/valine/system.py        28-atom topology, NeRF build, methyl-rotor relief,
                            restrained minimisation, exact 3-D rigid seed lattice, validate_seed
src/valine/cv.py            the three candidate 2-D CVs and their hidden coordinates
src/valine/umbrella.py      batched dihedral restraints, analytic dihedral gradient,
                            periodic 1-D MBAR
scripts/run_valine_chi1_profiles.py   Stage 2 driver (144 windows in one batched GPU run)
tests/test_valine_stage0.py           18 tests
tests/test_valine_umbrella.py         14 tests
```

Nothing else was touched. The validated alanine sampler, force-field extraction, BAOAB
integrator and birth–death machinery are **reused, not forked**.

System: 28 atoms, 27 bonds, 48 angles, 72 torsions, 134 exceptions, net charge −2.8e−17,
mass 172.228 (C₈H₁₆N₂O₂), **0 constraints**. `param_hash = 86622b245bb0`, which differs from
alanine's `6ffd00dc241f`, so the existing provenance gates will correctly refuse to run a Val
arm against the alanine reference.

Note on the plan's wording: "the same hydrogen-bond constraints" resolves, in this repo, to
**no constraints at all** — `extract_parameters` raises if the system carries any, because the
torch integrator implements no SHAKE. dt = 1 fs is therefore mandatory, not a choice.

## 3. The design correction: χ₁ must be VISIBLE, and the pre-screen shows why

The **frozen** design (`ALANINE_SPEC.md`, memory `alanine-study-status`) put Val's headline on
ξ = (φ, ψ) with **χ₁ hidden**. The plan now on the table instead makes χ₁ one of the two
selected coordinates: ξ = (φ, χ₁) or (ψ, χ₁). That is a material change, and the pre-screen
turns it from a preference into a requirement.

Adiabatic relaxed χ₁ scan (24 points, both sweep directions, φ and ψ restrained), kT = 2.4943
kJ/mol:

| backbone | (φ, ψ) | MEP barrier | hysteresis | wells (χ₁° : kT) |
|---|---|---|---|---|
| C7eq | (−80, 80) | 16.3 kT | 1.31 | −180:0.00, −75:1.75, +60:2.68 |
| C5 | (−140, 150) | 10.5 kT | 1.30 | −180:0.82, −60:0.62, +60:0.00 |
| αR | (−70, −30) | 10.1 kT | 0.01 | −180:1.05, −60:0.00, +75:0.82 |
| αL | (60, 40) | 16.7 kT | 0.02 | −150:0.12, −60:0.00, +75:2.48 |
| C7ax | (63, −48) | 11.9 kT | **26.9** | −180:0.00, −45:4.45, +75:4.34 |
| bridge | (−100, 0) | 14.1 kT | 1.25 | −165:4.59, −60:0.00, +75:2.42 |

This is a **minimum energy path with the backbone clamped**, so it is an *upper bound* on
F(χ₁|φ,ψ): entropy and the backbone relaxation these restraints forbid both lower it. The
C7ax row has 26.9 kT forward/backward hysteresis and is not trustworthy; the other five are.

**The implication.** A χ₁ barrier of order 10 kT is essentially never crossed by unbiased
dynamics (e^−10 ≈ 5e−5). So under the *frozen* χ₁-hidden design, Ace-Val-Nme would have been
**discovery-limited** — `T_hit ≈ T_run`, the R15 regime the repo already documented as a
failure mode for mFR, and the plan's own Outcome C. mFR cannot clone a state no walker has
reached. Making χ₁ visible puts the barrier *under the ABF bias*, which flattens it and
restores discovery; the remaining question then becomes the establishment transient, which is
exactly the regime mFR is supposed to address.

So the plan's change is not cosmetic. It is what keeps the system out of the R15 regime.

**Consequence for the gates.** Decision-doc gate **V2** ("χ₁ rotamers occupy meaningfully
different (φ,ψ) regions") is written for the χ₁-hidden design and becomes *tautological* once
χ₁ is a selected coordinate — the plan's §23 makes the same point. Its replacement is the
plan's Stage-3 test: cluster full states in (φ,ψ,χ₁), then measure how well the full-state
label is predicted from each candidate 2-D CV. V1, V3 and V4 are unaffected.

## 4. The cost fact that makes this affordable

**The selected CV stays two-dimensional.** `BackboneCV2D` is already parameterised by its two
atom quadruples, so (φ,χ₁) and (ψ,χ₁) drop into the validated alanine pipeline unchanged: the
FFT Poisson projection, the torus KDE, the birth–death score, the watershed basins and the
rank-4 MBAR factorisation all keep working, and a Stage-4 umbrella reference costs 24×24 = 576
windows exactly as alanine's did.

Only the *state map* (Stage 1) is three-dimensional, and that is clustering of samples, not
sampling on a 3-D grid. A genuinely 3-D **CV** would instead need `fftn`, a trilinear
interpolator, a rank-6 MBAR basis, a 97³ = 912,673-cell grid per seed and 24³ = 13,824 umbrella
windows. Nothing in this plan requires that, and it should not be built speculatively.

## 5. Three bugs found, two of which would have produced a wrong number

**(a) MBAR reweighted to the wrong ensemble — would have silently corrupted every χ₁ profile.**
`pymbar.MBAR.weights()[:, 0]` is the weight of each sample *in umbrella window 0*, not in the
unbiased ensemble; it still carries window 0's restraint. Using it returns the biased
distribution of the first window dressed up as F(χ₁). Caught by
`test_mbar_recovers_a_known_periodic_profile`, which failed by **459 kJ/mol** against a known
3-well potential. Fixed by computing `log w_n = −logsumexp_k[log N_k + f_k − u_kn]` explicitly
from pymbar's `f_k`.

**(b) Analytic dihedral-gradient sign error — and a vacuous test that hid it.** The closed-form
gradient (a 6.9× speedup over the `torch.func` primitive, which costs ~6 ms per dihedral and is
flat in batch) was wrong on the two middle atoms. **`Σ_a g_a = 0` is satisfied identically by
several wrong sign combinations**, so translation invariance could not detect it — two
successive wrong versions passed that check while being O(1) wrong. Only elementwise comparison
against `_grad_phi4` caught it. The corrected form agrees to 2.3e−13. Both the vacuous check
and the real one are kept in the test file, with a comment saying which is which.

**(c) The NeRF builder's ACE clash is inherited, not new.** The raw build leaves an
ACE HH32···O contact at 1.113 Å and ~6e4 kJ/mol. Measured **identically for alanine (59890)
and valine (59799)**, so it is the documented alanine defect, not a Val bug. It matters more
here: *unrestrained* minimisation dragged two of three rotamers from (−80, 80) to (+56, −38),
i.e. across backbone basins, which would have silently destroyed the seed lattice. Restrained
minimisation (plan §25) fixes it — all 12 screening seeds land within 0.6° of target.

## 6. Stage-0 gate — all checks pass

32 tests; full suite **153 passed, 3 skipped** (skips are the pre-existing CUDA-parity tests).

| plan §26 check | result |
|---|---|
| 1–2 OpenMM energy + force parity | 2.5e−10 relative on force; energy 9.6e−9 of the term scale |
| 3 CVs vs independent implementation | mdtraj, agreement 3e−7 rad (mdtraj is float32) |
| 4 periodic continuity across ±π | exact to 1e−9 over a full 720° sweep, all three angles |
| 5 CV gradients by finite difference | < 1e−5, all three candidate CVs; exactly 0 off-support |
| 6 Langevin stability | 5 ps × 32 walkers, T = 300 K, bonds bounded, no NaN |
| 7 cloning phase-space copy | position and cached force copied, momenta resampled — see below |
| 8 clones decorrelate | position spread and CV spread both grow |
| — L-valine chirality, mirror is D | pass |
| — sp² planarity, ω trans | ≤1.2° out-of-plane, ω 168–180°, 0/12 seeds flagged |
| — rigid seeding | bonds preserved to 1e−12; round-trip to 1e−8 deg |
| — three rotations independent | each moves only its own angle, to 1e−8 deg |

Two notes where I did **not** follow the plan's literal wording, both deliberate:

* **§26.7 "cloning positions and momenta".** The validated implementation copies the position
  and the cached physical force but draws **fresh Maxwell momenta** — exact, because the
  canonical density factorises. The test pins the implemented semantics, which is the one the
  alanine result was obtained under.
* **§26 "the same tolerances as alanine".** Kept, except that the alanine energy normaliser
  divides by the *total* energy and the force normaliser by ‖F_OpenMM‖. Val's total nearly
  cancels (bonded +69 against nonbonded −231, total −162) and the minimised seed has ‖F‖→0, so
  both normalisers divide by ~0. Energy is judged against Σ|per-force term| and the stationary
  point on an absolute scale. This is a corrected normaliser, not a loosened tolerance —
  underlying agreement is 2.5e−10.

## 7. Gap in the plan: the sham arm does not exist

Plan §39 requires arm **B: ABF + sham random resampling**, which performs the same number of
clone/delete operations independently of the FR score. `grep -i sham` over the repo returns
**nothing**; `core2d_ala.METHODS` is exactly `("abf", "fr_oracle")`. This must be implemented
before Stage 6, and it is a real requirement, not a formality — without it a positive oracle
result cannot be separated from the generic effect of perturbing the population.

Verified end to end: `run_sampler_ala('abf', ...)` with
`BackboneCV2D(PHI_ATOMS, CHI1_ATOMS, n_atoms=28)` runs unchanged and returns a finite PMF, so
the reuse claim in §4 is measured, not assumed.

**Second hazard in that path.** `core2d_ala.py:322` hardcodes the rare basin as
`c7 = (cur == 2)` rather than looking it up by name. For alanine that happens to be C7ax,
because prominence merging orders the basins C7eq, C5, C7ax. Val's basins will be different
and there is no reason index 2 is the interesting one, so **every `wmax_c7ax` /
`ess_age_c7ax` diagnostic would silently describe the wrong basin.** This must be
parameterised before any Val genealogy number is believed.

## 6b. Two more defects, both found by running the §32 test rather than by reading the code

**(d) The naive §32 gate would have been biased toward the answer I predicted.** Backbone-angle
scans hit genuinely inaccessible windows — the ψ scan loses 17/144 and the φ scan 20/144 to
steric contacts below 0.18 nm, twisted peptide bonds, and in one case an sp² centre 24.95° from
planar (the failure mode the alanine docs call unrepairable). Dropping those windows and then
reporting `max(F)` over what remains **understates** the barrier, and §32 wants *no* barrier —
so the error pushes the verdict toward a wrong PASS. The gate now counts **metastable states**
with unsampled bins treated as impassable separators, not as missing data. Pinned by
`test_count_states_two_arcs_split_by_inaccessible_regions`, a profile whose sampled `max(F)` is
under 3 kT but which is genuinely two states.

Counting states also exposed a circular-topology bug: linearising the torus at index 0 splits a
well straddling ±π into two, so the traversal must start at a separator (a NaN if one exists,
otherwise the highest ridge). `test_count_states` covers a well placed deliberately at ±π.

**(e) A tolerance artifact was deleting real windows, in the direction of my own prediction.**
At the Stage-0 `cv_tol_deg = 1.0`, 10 of the φ scan's 30 rejected windows were rejected for
"CV φ off target by 1.4 deg" — a *numerical placement miss*, not a structural defect. MBAR
builds its reduced potential from the **restraint centre**, not from where the seed landed, so
a 1.5° seed offset is harmless. Loosening `cv_tol_deg` to 5° for umbrella seeding (structural
checks untouched) recovers those 10 windows: φ drops fall 30 → 20, and the remaining ones are
16 ω-twists plus 4 steric. This matters because §8 records a standing prediction that
**(ψ,χ₁) should fail**, and (ψ,χ₁) is the CV whose hidden coordinate is φ. Deleting a third of
the φ windows on a numerical technicality would have helped deliver that prediction for the
wrong reason.

## 7b. Three further places the plan and the code disagree

**(i) "KDE grid 48×48" (plan §40) will hard-fail.** `require_odd_grid` raises on an even grid,
because the Nyquist row `k = n/2` has no representable derivative, so `gB == grad B` stops
holding exactly. Use 49×49, or keep alanine's 97. This is a `SystemExit`, not a silent
degradation, so it will be caught — but it should be fixed in the plan rather than discovered
at launch.

**(ii) The equal-budget arm (plan §44) rests on an assumption this implementation violates.**
§44 proposes `N=2048` for half the trajectory length, or `N=1024` for twice the length, "depending
on actual implementation cost". The alanine benchmark measured that **cost is flat in batch** —
doubling the walkers costs 3 % more wall-clock, because the step is kernel-launch bound. So
"N=2048 for half the length" is *not* an equal-compute arm at all: it is roughly **half** the
compute. Equal budget must be accounted in **force evaluations** (which the manifest already
records), and the honest larger-budget arm varies *time*, not N.

For the same reason, §44's principal `N=1024` is smaller than it needs to be: alanine ran
N=2048 and N=4096 at essentially the same wall-clock. Use at least 2048, both for statistical
strength and for comparability with the alanine null.

**(iii) The plan's genealogy gates are looser than the ones alanine already passed.** §42 asks
for `ESS_anc/N ≳ 0.2–0.25` and `w_max ≲ 0.1`; the alanine pilot used `ESS_age ≥ 0.30` and
`w_max ≤ 0.05` and passed at 0.956–0.966 and 0.0015–0.0034. Keeping the alanine thresholds
costs nothing and preserves comparability between the two studies.

## 8. Standing prediction, recorded before the screen runs

**(ψ, χ₁) is expected to fail the hidden-coordinate mixing gate (plan §32).** The coordinate it
hides is φ, and φ carries the dominant backbone barrier — alanine's accepted reference measured
a 15.79 kT min-max barrier between the φ<0 megabasin and C7ax. Under (φ, χ₁) the hidden
coordinate is instead ψ, which the same reference showed carries at most 0.75 kT at every
populated φ. So (φ, χ₁) is the expected winner and (ψ, χ₁) the expected casualty. Written down
in `src/valine/cv.py` as well, so it cannot be quietly reinterpreted afterwards.

## 9. Running now

```
CUDA_VISIBLE_DEVICES=7  scripts/run_valine_chi1_profiles.py --walkers-per-window 16
  144 windows (6 backbones x 24 chi1 centres, 15 deg spacing), 2 parent rotamers per window,
  batch 4608, 50 ps equilibration + 250 ps production, kappa_bb 500 / kappa_chi1 100,
  14.35 ms/step, ~72 min, peak 0.25 GiB
  -> results/valine/chi1_profiles/{meta.json, profiles.npz}
```

GPU compliance: GPU 7 only. GPUs 4/5/6 are saturated by another user and were never touched;
GPUs 0–3 were never used. The runner refuses to start unless `CUDA_VISIBLE_DEVICES` is a single
absolute index in {4,5,6,7}.

## 10. The exact next decision

The Stage-2 driver is generalised over which angle is scanned, so the same 144-window batched
run serves both gates — only the **verdict direction** differs:

```
--scan chi1   clamp (phi,psi)   gate V1        we WANT a barrier
--scan psi    clamp (phi,chi1)  gate sec.32    hidden coordinate of (phi,chi1) -- want NO barrier
--scan phi    clamp (psi,chi1)  gate sec.32    hidden coordinate of (psi,chi1) -- want NO barrier
```

The §32 modes are the direct test of the §8 prediction: a *large* barrier in the omitted
coordinate disqualifies that candidate CV, because ABF would then receive a conditionally
unequilibrated mean force and marginal mFR could not see the problem, let alone repair it.
Threshold set at 3 kT.

1. **Read the V1 verdict** from `results/valine/chi1_profiles/meta.json`. Note the in-flight
   run was launched before the scan-mode generalisation and writes the key `gate_v1`; the
   generalised script writes `gate`. Same numbers, same physics — only the key name moved. The MEP
   pre-screen predicts PASS; if the free-energy barrier comes out below ~2 kT the study stops
   here and Val joins alanine as a second neutrality control.
2. **Run `--scan psi` and `--scan phi`** (~75 min each). This decides the CV, and it tests the
   §8 prediction. Cheap, and it must happen before any reference is built.
3. **Stage 1** — 3-D state map in (φ,ψ,χ₁) by multi-start exploration + periodic clustering.
4. **Stage 3** — full-state distinguishability from the surviving CV. Not the tautological V2.
5. **Stage 5 before Stage 4.** The plan orders the umbrella reference (Stage 4) before the
   ABF-only mechanism screen (Stage 5). Run the **ABF discovery-vs-establishment screen first**:
   it is far cheaper, it is the gate that actually killed alanine, and building a
   publication-quality 576-window reference for a CV that then fails V3 would be wasted.
   Stage 5's deficit metric needs only a provisional reference, which Stage 1 supplies.
6. Only then Stage 4 → 6 → 7 → 8, and implement the sham arm (§7) before Stage 6.

## 11. Uncommitted

All of the above is in the working tree and **not committed** — no commit was requested. Suggested:

```
git add src/valine tests/test_valine_stage0.py tests/test_valine_umbrella.py \
        scripts/run_valine_chi1_profiles.py VALINE_STAGE0_HANDOFF.md
```

---

# RESULTS — gates V1 and §32 (2026-08-01)

## R1. Gate V1: **PASS**

`results/valine/chi1_profiles/` — 144 windows, **0 dropped**, 4608 walkers, 250 ps production,
82.9 min on GPU 7. Free energies from periodic 1-D MBAR at each of six clamped backbone points.

| backbone | (φ,ψ) | barrier | g− | g+ | t | wells (χ₁° : kT) |
|---|---|---|---|---|---|---|
| C7eq | (−80, 80) | 17.3 kT | 0.120 | 0.047 | **0.833** | −172:0.00, −72:1.96, +57:2.83 |
| C5 | (−140, 150) | 11.3 kT | 0.271 | **0.549** | 0.179 | −68:0.75, +67:0.00, +178:1.31 |
| αR | (−70, −30) | 11.4 kT | **0.616** | 0.224 | 0.160 | −178:1.54, −63:0.00, +67:0.99 |
| αL | (60, 40) | 17.9 kT | 0.460 | 0.035 | **0.505** | −158:0.00, −58:0.03, +78:2.64 |
| C7ax | (63, −48) | 12.9 kT | 0.012 | 0.012 | **0.975** | −178:0.00, −53:4.53, +83:4.74 |
| bridge | (−100, 0) | 15.5 kT | **0.916** | 0.071 | 0.013 | −168:4.91, −63:0.00, +72:2.53 |

Both V1 conditions are met by a wide margin: barriers of **11.3–17.9 kT** against a ≥2 kT
requirement, and a second rotamer above 2 % at five of six backbone points.

**The scientifically interesting part is the column of bolded maxima.** The dominant rotamer
*changes identity* with the backbone — t at C7eq/αL/C7ax, g⁺ at C5, g⁻ at αR/bridge — and the
margins are large (t 83 % at C7eq against g⁺ 55 % at C5). This is the backbone/side-chain
coupling the plan hypothesises in §9, measured directly rather than assumed, and it is what
makes (φ, χ₁) contain genuinely distinct populated cells.

**One honest surprise.** The MEP pre-screen was an *upper bound* — entropy and forbidden
backbone relaxation should both have lowered it — yet the free-energy barriers (11.3–17.9 kT)
came out **at or slightly above** the MEP values (10.1–16.7 kT). The pre-screen was not
conservative in the way expected. It does not change the verdict, which passes by 5×, but the
MEP should not be trusted as a bound for Ile/Leu without rechecking.

## R2. Gate §32: (φ,χ₁) **PASS**, (ψ,χ₁) **FAIL** — the recorded prediction holds

Same machinery, scanned axis permuted; the verdict criterion is the number of *populated*
states in the omitted coordinate, with inaccessible windows counted as impassable separators.

**(φ,χ₁) — hidden coordinate ψ — admissible.** One populated state at all six anchors. The
walls are high (16–25 kT) but the basin is single, and a single confined well mixes fast; §32
asks whether the omitted coordinate has *multiple slowly interconverting* states, not whether
it is confined. 17/144 windows inaccessible.

**(ψ,χ₁) — hidden coordinate φ — rejected.** At anchor ψ=−30°, χ₁=g⁻ the hidden φ carries
**two populated states**, wells at −63° and +57° separated by only 1.50 kT in depth but by a
**30.8 kT** barrier, with the minor state holding 26 % of the population. ABF on (ψ,χ₁) would
average its mean force over two φ basins that never interconvert, and marginal mFR — which sees
only p(ψ,χ₁) — could not detect that, let alone repair it. 20/144 windows inaccessible.

This is the outcome predicted in §8 before the measurement, for the stated reason: (ψ,χ₁) hides
φ, and φ carries the backbone barrier. Note the two failure modes are *not* symmetric — φ fails
by holding two comparably-deep basins, whereas ψ is merely deep and single.

## R3. Selected CV

```
xi = (phi, chi1)     atoms (4,6,8,20) and (6,8,10,12)
```

Enough of the plan is now settled to state what is NOT yet established: nothing here says mFR
will help. V1 and §32 establish only that the system has a real χ₁ barrier, that its rotamer
populations depend strongly on the backbone, and that (φ,χ₁) is an admissible selection
coordinate. **The gate that killed alanine — discovery followed by a persistent population
deficit (V3) — has not been tested.** That is the next experiment, and it should run before the
576-window Stage-4 reference.

## R4. Kinetic temperature runs ~2 % low — caused by the STIFF RESTRAINT, not by C–H

All three runs report a mean kinetic temperature of 293.2–293.3 K rather than 300 K. With
3·28·4608 degrees of freedom the sampling sigma is 0.68 K, so a 6.8 K deficit is ~10 sigma and
systematic.

Measured attribution (B=64, 30 ps, sigma_T = 5.79 K):

| restraint | dt | mean T | deviation |
|---|---|---|---|
| none | 1.00 fs | 296.41 | −3.59 |
| none | 0.50 fs | 296.78 | −3.22 |
| none | 0.25 fs | 298.45 | −1.55 |
| kappa = 500 / 100 | **1.00 fs** | **293.17** | **−6.83** |
| kappa = 500 / 100 | 0.50 fs | 299.00 | −1.00 |
| kappa = 500 / 100 | 0.25 fs | 299.81 | −0.19 |

The unrestrained system sits within ~0.6 sigma of 300 K at every step size. The restrained one
is 6.8 K low at dt = 1 fs and recovers to −1.0 K at dt = 0.5 fs and −0.2 K at 0.25 fs — the
O(dt²) signature of an under-integrated stiff mode. The kappa = 500 kJ/mol/rad² **backbone
restraint** is the fast mode, not the C–H stretches. The dt = 1 fs value reproduces the
production runs' 293.2 K exactly.

**Consequences, stated plainly.**

* It is a property of the *umbrella* runs, which carry stiff restraints. The ABF/mFR production
  runs carry no restraint — only a smooth learned bias — so they are not affected, and the
  alanine study's dt = 1 fs freeze stands.
* For the profiles reported above it is immaterial to the verdicts: a 2.3 % error in the kT
  scale moves a 17.3 kT barrier to ~16.9 kT, against a 2 kT threshold passed by 5×.
* **It is not immaterial for Stage 4.** The 576-window (φ,χ₁) umbrella reference is meant to be
  publication quality, and MBAR removes the restraint using its *analytic* potential, so a
  discretisation error in the sampled distribution is not fully unwound. Run the Stage-4
  reference at **dt = 0.5 fs**, or with a softer clamp, and verify the kinetic temperature
  before accepting it.

**Correction notice.** An earlier version of this section attributed the deficit to the known
BAOAB O(dt²) kinetic bias from unconstrained C–H stretches and asserted it was independent of
the restraint. That was wrong, and it was wrong because the first diagnostic computed its step
count as `60.0/(dt*1000)` — 60 steps instead of 60,000 — so it measured the first 60 fs before
thermalisation, found ~170 K everywhere, and showed no dt dependence *because nothing had
equilibrated in any of the runs*. The corrected sweep above shows a clean dt² recovery. The
original hypothesis (stiff restraint under-integration) was correct and was abandoned on the
strength of a broken measurement.
