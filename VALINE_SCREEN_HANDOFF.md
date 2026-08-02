# ACE-VAL-NME SCREENING HANDOFF — V3 FAILS ON BRANCH B; Val is a second neutrality control

Branch `alanine-dipeptide`. 2026-08-02. Companion to `VALINE_STAGE0_HANDOFF.md` (Stage 0, whose
§3, R1 and R4 carry inline corrections from this work) and `VALINE_SCREEN_SPEC.md` (the plan this
executes).

**Read in this order if you are picking this up cold:** §1 headline → §6c the V3 verdict → §7 what
is cancelled and why → §5 and §6 for the two Stage-0 claims that turned out to be wrong. §6b and
§6b-bis are the pilot's story, including a rejection that was itself a measurement error, and are
only worth reading if you are going to build another umbrella reference on a 2-D CV.

> **STATUS: the screening phase is COMPLETE and the answer is negative.** Stage 0 is frozen and
> tagged `valine-stage0-accepted`. S1, distinguishability and the pilot reference all pass. The
> decisive gate **V3 FAILS on branch B — ABF is already sufficient** (§6c): every one of the eight
> regions is discovered within 4.1 ps by 16/16 seeds and established within 45 ps, so there is no
> discovered-but-under-established state for mFR to repair. **Do not proceed to the Stage-4
> reference, the sham arm, or an mFR comparison.** Three Stage-0 claims are corrected by
> measurement along the way.

---

## 1. Headline

**Ace-Val-Nme is a second neutrality control, not a positive example.** ABF on ξ = (φ, χ₁)
discovers every metastable region in 0.5–4.1 ps and establishes every one of them within 45 ps of
a 300 ps run, with its free energy landing 0.247 kT from the reference. mFR has nothing to act on.
That verdict is worth more than a marginal positive would have been: Val was *selected* for a real
side-chain barrier and cleared every gate ahead of V3, so its failure is evidence about the
method's regime rather than about a badly chosen molecule.

The molecule has **seven metastable states** in (φ, ψ, χ₁), and they factor almost exactly as
**backbone megabasin × χ₁ rotamer**. The selected CV ξ = (φ, χ₁) recovers the full 3-D state
label with **97.3 % balanced accuracy**, so a population deficit would be visible to a marginal
Fisher–Rao score. That is the precondition for mFR being *able* to act, and it now holds.

**But the reason Val was thought to be interesting is wrong.** Stage 0 argued that χ₁ carries an
11–18 kT barrier that unbiased dynamics never crosses, and concluded that hiding χ₁ would make
the system discovery-limited — it called this "the single most important Stage-0 result".
Measured directly: χ₁ rotamers interconvert at **2.70 changes per walker per ns**, an
equilibrium rate. The slow coordinate is **φ**, which crossed its megabasin **4 times in 2581 ns**
of aggregate unbiased sampling. φ is in the CV. See §5.

---

## 2. What was run

| stage | artifact | cost |
|---|---|---|
| S1 state map | `results/valine/state_map/` | 8604 walkers × 300 ps = 2581 ns, 29 min GPU |
| distinguishability | `results/valine/state_map/distinguishability.json` | CPU |
| clustering sensitivity | `results/valine/state_map/state_sensitivity.json` | 18 settings, CPU |
| dt bias re-measurement | `results/valine/dt_bias/` | 3 timesteps × 3 restraints, one batch each |
| pilot reference | `results/valine/pilot_reference/` | 324 windows, 586 seeds × 8 copies, ~70 min |
| V3 ABF-only screen | `results/valine/v3_screen/` | 16 seeds x 2048, 300 ps, 4.5 h -- **FAIL-B, §6c** |
| figures | `results/valine/figures/valine_screen.png` | CPU |

One GPU at a time throughout. The node was re-partitioned mid-study (8 shared devices → this
group's own four, renumbered 0–3); everything before that ran on the old GPU 7, everything after
on GPU 0.

Measured step costs, for anyone sizing a follow-up: the ABF sampler is **~51 ms/step and flat in
batch from B = 8192 to B = 32768**, so seeds are nearly free and run *length* is the only real
cost lever — the screening plan's "equal-compute arm by varying N" is not equal compute. Two
optimisations found while sizing V3: `torch.linalg.eigvalsh` on the 2×2 Gram matrix was
dispatching to `cusolverDnXsyevBatched`, which **crashed above B = 32768** and reserved ~13 GiB
of workspace — replaced by the closed form, peak memory 20.1 → 7.3 GiB. And the union-block CV,
which an earlier benchmark had shown to be no faster, is 1.09–1.24× faster once cuSOLVER is no
longer masking it.

## 3. S1 — the state map, and why it is a lattice rather than a long run

Stage 0 measured barriers that unbiased trajectories cross rarely or never. A multi-start run
seeded at the structures we already know therefore returns exactly those structures and calls it
a discovery. The alternative usually proposed — an exploratory bias — buys coverage at the cost
of a second set of parameters to defend.

This run instead seeds a **dense regular lattice over the whole torus** (18 φ × 18 ψ × 9 χ₁ =
2916 points, of which **2151 survive structural validation**; 765 are genuinely inaccessible —
411 steric, 345 twisted peptide bond, 9 non-planar sp²) and lets each walker relax into whatever
state contains it. Coverage is then a property of the construction. The dynamics only has to
supply *local* relaxation, which unbiased dynamics does correctly.

**The price, stated plainly.** The resulting density is a **basin-of-attraction** measure, not a
Boltzmann one. It is the right measure for locating state *boundaries* and for state-*conditioned*
densities, and the wrong one for state *populations*, which come from the pilot free energy
instead. `src/valine/states.py` returns nothing called a population, deliberately.

### The seven states

| state | φ | ψ | χ₁ | rotamer | backbone | attraction | exits |
|---|---|---|---|---|---|---|---|
| B0 | +65 | −55 | −175 | t | φ>0 | 0.241 | 986 |
| B2 | +55 | −25 | −55 | g− | φ>0 | 0.101 | 373 |
| B6 | +55 | −35 | +85 | g+ | φ>0 | **0.008** | 1224 |
| B4 | −75 | +85 | −175 | t | φ<0 | 0.224 | 1338 |
| B1 | −145 | +145 | +65 | g+ | φ<0 | 0.218 | 816 |
| B3 | −75 | +45 | −65 | g− | φ<0 | 0.129 | 15593 |
| B5 | −135 | +155 | −65 | g− | φ<0 | 0.076 | 16161 |

Three rotamers × two backbone megabasins = six, plus one: the φ<0 g− state is **split into B3 and
B5** by the prominence threshold, and the transition matrix says they are kinetically **one**
state (30 769 exchanges between them, against 373–1338 exits for the genuinely separate states).

**The transition matrix is strictly block-diagonal in sign(φ)**: zero counts between the blocks
and infinite min-max barriers. Only 4 persistent φ crossings occur in the entire 2581 ns.

### Robustness

* **Split-half over walkers**: 7 states, same as the full sample.
* **Clustering knobs**: 16 of 18 settings (cells 30/36/44 × prominence 1.0/1.5/2.0 × ceiling
  5/8) give 7 states; the other 2 give 6, and what they merge is exactly the B3/B5 pair that is
  already known to be one state. Minimum 86 % of baseline states recovered. The screening plan's
  AMBIGUOUS branch is defined by this dependence, so it was measured rather than assumed away.
* Only 0.38 % of frames sit above the flood ceiling and are unassigned.

## 4. Distinguishability gate — **PASS**

This replaces decision-doc gate V2, which became tautological once χ₁ entered the CV, and it
globalises the §32 screen from six anchors to the whole plane. Marginal mFR sees exactly `p(ξ)`;
if two states projected onto overlapping regions of (φ, χ₁), a deficit in one of them would not
be a resolvable feature of `p(ξ)` and mFR could not preferentially clone into it.

| measurement | value | threshold | verdict |
|---|---|---|---|
| worst pairwise footprint overlap `Σ min(p_i, p_j)` | **0.189** | ≤ 0.30 | pass |
| cross-validated balanced accuracy | **0.973** | > 0.80 | pass |
| `H(B \| φ, χ₁)` | **0.111 bits** (prior 2.807) | — | — |
| selected-CV weight with ≥2 metastable ψ states | **0.018** | < 0.10 | pass |

Two details that decide whether these numbers mean anything:

* **Folds are split by walker, never by frame.** Consecutive frames of one walker are strongly
  correlated; splitting by frame inflates accuracy toward 1 and says nothing about generalisation.
* **Overlap is the headline because it is prior-free.** It compares state-conditioned footprints
  directly, so the non-Boltzmann weighting of the exploration cloud cannot bias it. The worst
  overlapping pair is B3/B5 — the pair that is really one state.

The machinery was validated on synthetic data first: three planted states, two of which share a
(φ, χ₁) footprint and differ only in the hidden ψ. It gave them overlap 0.975 and coin-flip
recall, and the separable one 0.097 and 0.999 — i.e. it detects exactly the failure mode it
exists to detect.

## 5. CORRECTION 1 — the χ₁ barrier Stage 0 reports is a *clamped* barrier

`VALINE_STAGE0_HANDOFF.md` §3 argues that a ~10 kT χ₁ barrier "is essentially never crossed by
unbiased dynamics (e^−10 ≈ 5e−5)", and concludes that a χ₁-hidden design would have been
discovery-limited — the R15 regime where mFR provably cannot act. The S1 exploration measures the
opposite:

```
chi1 rotamer changes: 2.70 per walker per ns
  FLAT from 30 ps to 300 ps
  identical for walkers seeded in a well (2.70) and on a barrier (2.69)
```

Flat in time and independent of where the walker started is an **equilibrium rate**, not a
seeding transient. It implies an effective barrier of **6–8 kT with the backbone free**, against
the **11.3–17.9 kT** Stage 0 measured with φ and ψ clamped at κ = 500 kJ/mol/rad². Even the
smallest clamped value predicts 0.12 crossings/walker/ns at a generous attempt frequency —
22× fewer than observed, and ~100× at a realistic one.

**Why.** The clamp forbids the backbone relaxation that accompanies the rotation. This also
explains the "honest surprise" recorded in Stage-0 R1 — that the umbrella barriers came out *at
or above* their own minimum-energy-path upper bound, when entropy should have lowered them. Both
measurements clamp the backbone, so they share the defect and agree with each other.

**What survives.**

* Gate **V1 still passes**: 6–8 kT is a real barrier against a ≥2 kT requirement.
* **(ψ, χ₁) is still rejected**, and now independently corroborated: φ crossed 4 times in 2581 ns.
* **(φ, χ₁) is kept** — but the discovery-limited argument for it does not survive, and should
  not be repeated.
* **V3 is reframed.** The coordinate ABF must flatten is **φ**, not χ₁.

## 6. CORRECTION 2 — the kinetic-temperature deficit is the integrator, not the restraint

`VALINE_STAGE0_HANDOFF.md` R4 attributes a 6.8 K kinetic deficit to the stiff dihedral clamp,
from a measurement at B = 64 whose sampling sigma was 5.79 K — too noisy to resolve the
unrestrained deficit, which was therefore read as absent. At B = 2048 per group (sigma 1.02 K):

| dt | T_kin − 300 K (unrestrained / κ500 / κ110) | T_bond − 300 K | T_angle − 300 K |
|---|---|---|---|
| 1.0 fs | **−7.01** / −6.63 / −6.88 | +25.8 / +27.7 / +26.6 | −59.9 / −60.1 / −59.7 |
| 0.5 fs | **−1.70** / −1.56 / −1.72 | +24.9 / +26.2 / +25.7 | −59.4 / −59.5 / −59.4 |
| 0.25 fs | **−0.25** / −0.60 / −0.57 | +24.5 / +25.7 / +25.9 | −59.0 / −59.6 / −59.1 |

The deficit is **independent of the restraint** (the clamp is marginally *warmer*) and scales as
**O(dt²)** — −7.01 → −1.70 → −0.25 K. It is the integrator.

**And it does not touch the free energies.** Across a **16× change in dt²**, the kinetic
temperature moves 6.76 K while the equipartition estimators on bonds and angles move **1.24 K
and 0.93 K** — of a 26 K and 60 K offset. Their offsets are therefore static properties of a
curvilinear-coordinate estimator (internal coordinates are not independent normal modes, so
`k⟨Δx²⟩ ≠ k_B T` for them even in an exactly canonical ensemble) and **not** a temperature error,
which would have scaled with dt exactly as `T_kin` does. The *configurational* distribution is
not corrupted by the timestep, and free energies at 1 fs are sound — which also retro-justifies
the alanine study's dt = 1 fs freeze, for the right reason this time.

> **A caught error in this very diagnostic.** The script's first automatic verdict tested
> `|T_bond − 300| < 3 K` and duly announced that the configurational distribution *was* off
> temperature — contradicting the script's own stated method, which is that a static estimator
> offset cancels in the dt comparison. The absolute offset is meaningless here; only the dt
> dependence is evidence. The verdict function now judges on the dt dependence and the saved
> artifact was recomputed from the unchanged measurements.

dt = 0.5 fs is kept for the restrained pilot anyway: it is the value the plan froze, and halving
an already-small artifact is cheap insurance on a reference MBAR cannot unwind. Relaxing a frozen
value mid-study to save wall-clock is the wrong trade.

## 6b. The pilot reference — and a diagnostic that failed it for the wrong reason

**Accepted: `results/valine/pilot_reference/`** — 324 windows on (φ, χ₁), ψ **free** and started
from **four** values (+150, +60, −40, −140), 951 (window, ψ-start) seeds × 8 copies, 400 ps at
dt = 0.5 fs. MBAR 91 iterations, residual 8.8e−9; overlap graph a single connected component;
85.3 % grid coverage; split-half over copies **0.31 kT**; kinetic temperature 298.31 K.

| region | centre (φ, χ₁) | population | | region | centre (φ, χ₁) | population |
|---|---|---|---|---|---|---|
| B0 | (−78, −178) | 0.4201 | | B4 | (−141, −71) | 0.0701 |
| B1 | (−74, −63) | 0.1898 | | B5 | (−82, +63) | 0.0380 |
| B2 | (−152, +67) | 0.1840 | | **B6** | (+56, −52) | **0.0055** |
| B3 | (+63, −174) | 0.0900 | | **B7** | (+56, +82) | **0.0014** |

The two rarest regions, B6 and B7, both sit at φ > 0 — behind the megabasin barrier. They are the
candidates for under-establishment in V3.

**The regions ARE the physical states.** Mapping every S1 exploration frame to both its 3-D state
and its pilot region (`region_state_map.json`) gives a near one-to-one correspondence — six of the
eight regions are 100 % pure:

> ⚠ **Name collision.** Pilot *regions* and S1 *states* are both labelled `B0…`. They are
> different labellings. Below, `S:` prefixes the 3-D state.

| region | = state | rotamer, backbone | purity | | region | = state | rotamer, backbone | purity |
|---|---|---|---|---|---|---|---|---|
| B0 | S:B4 | t, φ<0 | 1.00 | | B4 | S:B5 | g−, φ<0 | 0.98 |
| B1 | S:B3 | g−, φ<0 | 0.89 | | B5 | S:B1 | g+, φ<0 | 1.00 |
| B2 | S:B1 | g+, φ<0 | 1.00 | | **B6** | **S:B2** | **g−, φ>0** | 1.00 |
| B3 | S:B0 | t, φ>0 | 1.00 | | **B7** | **S:B6** | **g+, φ>0** | 1.00 |

The only impurity is the S:B3/S:B5 pair — the two states the transition matrix already showed to
be kinetically one. This is an independent confirmation of the distinguishability gate, measured
on a different artifact.

So the V3 prediction sharpens: **B7 = the g+ rotamer of the φ>0 backbone**, population 0.0014, is
the analogue of alanine's C7ax and the state to watch.

**Barriers, and this is the free-energy confirmation of §5:**

| transition | barrier |
|---|---|
| χ₁ rotamer, backbone free (2-D min-max path) | **1.1 – 7.4 kT** |
| crossing the φ megabasin | **9.7 – 14.1 kT** |
| χ₁ at *fixed* φ (1-D slice, ψ relaxed) | median 4.1 kT, up to 15.3 |

The last row is why the *effective* 2-D path is the number to quote: conditioning on φ can make
the χ₁ barrier look three times larger than what a walker free in φ actually pays. Stage 0's
11.3–17.9 kT conditions on φ **and** ψ, and is larger still.

### The diagnostic that was wrong

Two successive pilots were **rejected by a check that was measuring the wrong thing**, and the
error is worth stating precisely because it is easy to repeat.

Both ψ checks compared subsets that **do not cover the same windows**. Only 61 of 315 windows
survive structural validation for all four ψ starts, and β/PPII occupancy is mostly a property of
*which window* a walker sits in. Averaging over unmatched window sets converts a coverage
difference into an apparent equilibration failure — an unpaired comparison wearing a paired
comparison's clothes.

| | |
|---|---|
| start-memory spread, **all** windows | 0.169 → "ψ is not equilibrated" |
| start-memory spread, windows carrying **all four** starts | **0.010** |
| and by production quarter | 0.043 → 0.021 → 0.017 → **0.006** |

Replaced by a **paired** test that needs no MBAR at all: compare `p(ψ | window)` **across starts
within the same window**, calibrated against the same statistic between **copies of one start**,
which is pure sampling noise.

```
across-start worst-pair TV   0.025   (median over the 61 matched windows)
same-start noise floor       0.022
ratio                        1.16    ->  psi IS equilibrated
```

Corroboration that ψ was never the problem: in the **unrestrained** S1 exploration ψ changes basin
~226 times per walker per ns, roughly 83× faster than χ₁. A coordinate that fast does not fail to
equilibrate over 400 ps.

The confounded numbers (per-start FES RMSE 4.40 kT, unpaired start-memory spread 0.169) are still
printed, labelled as confounded, and **not** gated on. Pilot v1's 3.22 kT was very likely the same
artifact; it is superseded on sampling grounds regardless and kept at
`results/valine/pilot_reference_v1_rejected/`.

**A second, real lesson from v1 that still stands:** a split-half over *copies* is structurally
blind to the omitted coordinate, because copies of a window share its ψ start. It read 0.31–0.38 kT
in both pilots. Any reference on a 2-D CV needs a check that varies the omitted coordinate — but
that check must be **paired by window**.

## 6b-bis. (superseded — pilot v1's rejection narrative, kept for the record)

`results/valine/pilot_reference_v1_rejected/` — 324 windows on (φ, χ₁), ψ **free**, two ψ starts
(+120, −40), 8 copies each, 150 ps at dt = 0.5 fs, 65 min. MBAR converged cleanly (107 iterations,
residual 9.7e−9), 84.6 % of the 97² grid filled, kinetic temperature 298.36 K. It still fails:

| check | value | verdict |
|---|---|---|
| MBAR overlap graph connected | 1 component (even at threshold 0.001) | pass |
| grid coverage | 7958/9409 = 84.6 % | pass |
| split-half over **copies** | RMSE **0.38 kT** | pass |
| **ψ-start agreement** | RMSE **3.22 kT**, median \|d\| 1.4–1.9 kT where the population is | **fail** |

Those last two are not in tension, and the reason matters: **copies of a window share its ψ start**,
so a split-half over copies is blind to ψ by construction. It reads 0.38 kT while the thing it
cannot see is off by 3.22 kT. Any reference built this way and checked only by split-half would
look converged.

The failure is incomplete *equilibration* of the omitted coordinate, not trapping:

* ψ moves — **38.8 %** of walkers changed ψ basin during 150 ps;
* but remembers — mean β/PPII occupancy over the second half is **0.551** for walkers started at
  ψ = +120° and **0.463** for ψ = −40°.

A median 1.4–1.9 kT error is a factor ~5 in a state population, and V3 asks whether an occupancy
falls below **half** its target. The pilot has to be better than the threshold it is used to
evaluate, so v1 is kept as a record and **superseded**, not patched.

**v2 — ACCEPTED.** Four ψ starts (+150, +60, −40, −140), 400 ps, 951 seeds × 8 copies. MBAR
converged in 91 iterations, 85.3 % grid coverage, split-half 0.31 kT, kinetic temperature
298.31 K.

**And the check that rejected v1 was itself the wrong test.** v2's *unpaired* statistics look
just as bad as v1's — per-start FES RMSE 4.40 kT, start-memory spread 0.169 — but the paired
in-window test clears it outright:

| | across-start | same-start noise floor | ratio |
|---|---|---|---|
| TV of `p(ψ \| window)`, median over the 61 windows carrying all four starts | **0.025** | 0.022 | **1.16** |

ψ *is* equilibrated inside a window; the across-start difference is barely above pure sampling
noise. The unpaired comparison failed for a reason that has nothing to do with ψ: **the starts do
not cover the same windows** (only 61 of 315 survive structural validation for all four), and
β/PPII occupancy is mostly a property of *which window* a walker sits in. Averaging over
unmatched window sets converts a coverage difference into an apparent equilibration failure —
an unpaired comparison wearing a paired comparison's clothes. Over the matched windows the
start-memory spread is **0.010**, not 0.169.

That does not retroactively rescue v1 — v1's own paired statistic was never computed, and v2 is
better sampled regardless — but it does mean the *diagnostic*, not the reference, was the thing
that needed fixing. The unpaired numbers are still reported, and explicitly **not gated on**.

**The regions are the physical states, near one-to-one.** Mapping every S1 exploration frame to
both its 3-D state and its pilot region: six of the eight regions are **100 % pure** in a single
3-D state, and the two that are not (89 %, 98 %) mix only the B3/B5 pair already known to be
kinetically one state. So "region B7 is starved" is a statement about (φ>0, χ₁ = g⁺), not about
an artifact of the watershed.

Two method corrections came out of this:

* the per-start comparison now runs every **pair** and reports the **worst**. With more than two
  starts, an average would hide a single start that failed to equilibrate — which is the entire
  failure mode being tested for.
* acceptance gates on **connectivity** of the overlap graph, not on `min(overlap)`. v1 had a
  minimum pair overlap of 1e−4 and was nevertheless a single connected component; rejecting it
  for that number would have been rejecting a usable map for the wrong reason.

### What v1 does already establish, because it does not depend on ψ equilibration

The **effective** barriers — min-max paths through the 2-D (φ, χ₁) plane:

| transition | barrier |
|---|---|
| χ₁ rotamer, same backbone | **1.6 – 7.0 kT** |
| crosses the φ megabasin | **9.9 – 14.1 kT** |

This is the free-energy confirmation of §5. Stage 0's clamped 11.3–17.9 kT is roughly **twice**
the barrier a walker with a free backbone actually pays, and the φ megabasin — not χ₁ — carries
the large one.

## 6c. GATE V3 — **FAIL-B: ABF is already sufficient.** Val is a second neutrality control.

`results/valine/v3_screen/` — ABF only, 300 ps, 16 seeds × 2048 walkers, concentrated and
stratified arms as different seeds of one batch. clip fraction 0, zero non-finite.

**Every region is discovered almost immediately, and every region is established.**

| region | pilot pop | T_hit / T_run | T_est / T_run | occupancy / bias-aware target |
|---|---|---|---|---|
| B0 (φ<0, t) | 0.420 | 0.000 | 0.147 | 0.98 |
| B1 (φ<0, g⁻) | 0.190 | 0.000 | 0.073 | 0.94 |
| B2 (φ<0, g⁺) | 0.184 | 0.000 | 0.150 | 1.07 |
| B3 (φ>0, t) | 0.090 | 0.020 | 0.033 | 1.04 |
| B4 (φ<0, g⁻) | 0.070 | 0.000 | 0.137 | 1.13 |
| B5 (φ<0, g⁺) | 0.038 | 0.000 | 0.090 | 0.91 |
| B6 (φ>0, g⁻) | 0.0055 | 0.020 | 0.050 | 0.83 |
| B7 (φ>0, g⁺) | 0.0014 | 0.020 | 0.027 | 1.36 |

Discovery: **0.5–4.1 ps in 16/16 seeds**, against a threshold of 10 % of the run (30 ps) — five
to sixty times faster than required. Establishment: everything inside the ±50 % band by **45 ps**.
The worst relative deficit over the second half of the run is **0.23**, against a 0.50 threshold,
and no region is below half its target for more than 4.6 % of the run (threshold 20 %).

**The two arms agree.** Concentrated and stratified give the same verdict with the same numbers,
which is the diagnostic control doing its job: the result is not an artifact of where the walkers
started. And ABF's own free energy lands within **0.247 kT RMSE** of the pilot (marginal TV 0.069),
so this is not a case of a badly converged run flattering itself.

Even B7 — the rarest region at an unbiased population of 0.0014, the one predicted in §7 as the
candidate for starvation — is discovered in 4.1 ps by every seed and ends **above** its target.

**Verdict: FAIL-B.** There is no discovered-but-under-established state, so there is nothing for
mFR to repair. Under the screening plan's own rule this is a STOP, and explicitly *not* an
invitation to shorten the run or cut walkers until a deficit appears. Val joins alanine as a
second neutrality control — and it is a much stronger one, because unlike alanine it was chosen
for a real barrier, cleared every prior gate, and still shows nothing.

### Three caveats, stated because they qualify the numbers rather than the verdict

* **Condition 4 (omitted ψ) reads FAIL at worst-state TV 0.28, and that measurement is
  confounded.** It compares `p_ABF(ψ | region)` with `p_pilot(ψ | region)`, but ABF *flattens*
  within a region while the pilot is Boltzmann-weighted inside it, so the two weight the region's
  interior differently even when the ψ conditional at fixed (φ, χ₁) is identical. The signature
  fits: the two largest, most internally varied regions (B0 0.25, B1 0.28) disagree, while the
  six smaller ones sit at 0.02–0.15. This does not change the verdict — V3 fails on condition 2,
  and conditions 4 and 5 only gate a PASS — but the check needs re-deriving before it is quoted.
* **Condition 5 is a vacuous False**: there are no starved states, so "reproducible across seeds"
  has nothing to range over.
* **The `entries` column reads 0 for B3, B6 and B7** — the three φ>0 regions — although they are
  demonstrably reached. Transitions are only counted between consecutively-labelled frames, and
  reaching φ>0 means crossing a corridor above the 8 kT region ceiling, which is unlabelled. The
  zero is an artifact of the counter, and is itself consistent with the 9.9–14.1 kT φ barrier.

### One metric bug this exposed, worth not repeating

The first version of the establishment target capped cells the pilot never filled at
`F_min + 30 kT` and normalised over the whole torus. That put **97 % of the target mass in
exactly those cells**: ABF flattens, so `B_t` grows large where the pilot never sampled, and
`exp(−β(F_capped − B_t))` explodes there. The target concentrated wherever the reference was
least trustworthy. The fix is to define the target on the pilot's labelled support and condition
the observed fractions the same way. The `capped weight` diagnostic — added precisely so an
inadequate reference would show up as a number rather than a confident wrong answer — is what
caught it.

## 7. What happens next — and what must NOT

**V3 is decided (§6c): FAIL-B.** Under the screening plan's own rule that is a STOP, so the
things that were queued behind V3 are now cancelled, not deferred:

| queued work | status |
|---|---|
| full 24×24 Stage-4 reference | **cancelled** — it existed to support an mFR comparison |
| sham arm (`METHODS` still lacks it) | **cancelled** — only needed to defend a positive result |
| oracle mFR pilot | **cancelled** |
| practical mFR target | **cancelled** |

**What must not happen.** The plan anticipated this branch and named the temptation explicitly:
do not shorten the run, cut walkers, or lower the establishment band until a deficit appears.
Every one of those makes the *relative* thresholds easier — `T_hit < 0.1 T_run` and "starved for
≥ 0.2 T_run" both scale with run length — so a shorter run flatters the result rather than
testing it. The measured margins are not close: discovery is 5–60× faster than required and the
worst deficit is less than half the threshold.

### What the negative is worth, and what would follow it

Val is a **second neutrality control**, and a stronger one than alanine: alanine was neutral on a
CV that turned out to have no meaningfully rare state, whereas Val was *selected* for an
11–18 kT side-chain barrier and cleared V1, §32 and distinguishability (0.973) before failing V3.
Two independent systems now say the same thing — when ABF's CV contains the slow coordinate, ABF
establishes the populations on its own and marginal mFR has no deficit to repair.

The honest reading of the project so far is that the regime where mFR helps is *narrow*, and the
one system that ever showed a genuine support deficit (R15, `ALKANES_CV_EXTENSION_HANDOFF.md`)
was discovery-limited rather than establishment-limited — the regime where mFR provably cannot
act. Anyone continuing should be looking for a system that is establishment-limited by
construction, not hoping to find one by trying more peptides.

### Reusable machinery left behind

```
scripts/run_valine_state_map.py         T^3 state map from a torus-covering lattice
scripts/analyze_valine_distinguishability.py   can the 3-D state be read off the 2-D CV?
scripts/valine_state_sensitivity.py     re-cluster over the knobs; closes the AMBIGUOUS branch
scripts/run_valine_pilot_reference.py   coarse F(xi) with the omitted coordinate FREE
scripts/analyze_valine_pilot.py         acceptance; exits non-zero so a chain cannot ignore it
scripts/run_valine_v3_screen.py         ABF only, both init arms in one batch
scripts/analyze_valine_v3.py            V3 metrics and the decision rule
scripts/plot_valine_screen.py           the four-panel figure
```

Two design points in there worth not re-deriving:

* **The establishment target is bias-aware, and defined on the reference's support.** ABF moves
  the biased equilibrium as it learns, so the target is `q*_t ∝ exp(−β(F_pilot − B_t))` — but
  normalised over the cells the reference actually filled, and compared against observed
  fractions conditioned the same way. Getting either half wrong is a silent, confident error;
  see §6c.
* **Only discovered states can be under-established.** Counting an undiscovered state as a
  population deficit reports the R15 regime — where there is nothing to clone — as the regime
  mFR repairs. That distinction is the whole gate.

## 8. Code changes to shared machinery

All default-off, so the accepted alanine artifacts and behaviour are unaffected.

* `basins.BasinMap(..., name_hints=())` — Val takes neutral `B0, B1, …`. Alanine's Ramachandran
  boxes are written for (φ, ψ) and would attach a *backbone* name to a χ₁ rotamer; a wrong basin
  name propagates into every table and is harder to spot than a wrong number. Pinned by a test
  showing the same grid yields `C7eq` with hints and `B0` without.
* `run_sampler_ala(..., extra_angle_atoms=…)` — records the **omitted** coordinate and each
  walker's basin at every save. A 2-D CV can look converged while the coordinate it hides is not
  equilibrated, and nothing else in the sampler would notice. Per-walker basins matter because a
  *global* ψ check is nearly useless: two states can each be wrong in opposite directions and
  still sum to the right answer.
* `run_sampler_ala` dispatches on `scatter_bias`, enabling the **union-block CV**. For Val the
  union is 6 atoms — 18 of 84 coordinates — so the Hessian contraction shrinks ~22×. Equivalence
  is measured for Val, not inherited: local mean force to 1e−9, `G` bitwise, `div_v` to 1e−12,
  the scattered Cartesian bias force **bitwise**, and a 50-step end-to-end sampler run to 1e−8.
  That run is short on purpose — the two paths round differently in the last bits and Langevin
  dynamics amplifies that exponentially, so demanding agreement over picoseconds would be testing
  chaos, not correctness.
* `wmax_c7ax` / `ess_age_c7ax` → `wmax_rare` / `ess_age_rare`, with the tracked index recorded in
  the artifact; `metrics_ala` reads either spelling so the accepted alanine artifacts still load.

Full suite: **168 passed**.
