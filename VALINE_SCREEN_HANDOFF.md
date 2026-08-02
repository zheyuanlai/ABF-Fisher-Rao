# ACE-VAL-NME SCREENING HANDOFF — S1, distinguishability, and two Stage-0 corrections

Branch `alanine-dipeptide`. 2026-08-02. Companion to `VALINE_STAGE0_HANDOFF.md` (Stage 0) and
`VALINE_SCREEN_SPEC.md` (the plan this executes).

> **STATUS.** Stage 0 is frozen and tagged `valine-stage0-accepted`. The S1 state map and the
> global distinguishability gate are **done and both pass**. Two Stage-0 claims are **corrected
> by measurement**, one of them load-bearing. The pilot reference and the decisive V3 screen are
> **running / not yet run** — see §7. Nothing here says mFR will help.

---

## 1. Headline

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
| V3 ABF-only screen | `results/valine/v3_screen/` | *see §7* |
| figures | `results/valine/figures/valine_screen.png` | CPU |

All on GPU 7, one process at a time. Measured step costs, for anyone sizing a follow-up: the ABF
sampler is **50.5 ms/step at B = 16384** and **flat in batch** — so seeds are nearly free and run
*length* is the only real cost lever.

All on GPU 7. GPUs 4–6 were saturated by another user throughout and were never touched; 0–3
were never used.

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

## 6c. (superseded — v1's rejection narrative, kept for the record)

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

**v2** attacks both halves of the failure: **four** ψ starts (+150, +60, −40, −140) instead of
two, and **400 ps** of production instead of 150 — more independent initial conditions to average
over, and more time for each to forget. ~2 h; doubling the number of starts is nearly free per
step because the cost is flat in batch.

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

## 7. What is NOT yet done — and what decides the study

**Gate V3: does ABF discover every state and then leave one persistently under-established?**
That question is untested, and it is the gate that killed alanine. Everything above establishes
only that a deficit, if one exists, would be *visible* and *nameable*.

The machinery is built, smoke-tested end to end, and launched behind a gate:

```
scripts/run_valine_pilot_reference.py   coarse F_pilot(phi, chi1), psi FREE, dt = 0.5 fs
scripts/analyze_valine_pilot.py         acceptance; EXITS NON-ZERO if the pilot fails
scripts/run_valine_v3_screen.py         ABF only -- concentrated and stratified init
scripts/analyze_valine_v3.py            V3 metrics and the decision rule
```

The V3 launch is chained **behind** `analyze_valine_pilot.py`'s exit code, so a 14 h run cannot
start against a pilot that failed its own acceptance. Configuration:

| | |
|---|---|
| arms | concentrated **and** stratified, as different seeds of **one** batch |
| seeds | 8 per arm (16 total), N = 2048 → B = 32768 |
| length | 10⁶ steps = 1 ns at dt = 1 fs (unrestrained) |
| grid | 97 × 97, estimator frozen at the alanine values |
| cost | 52.7 ms/step → **14.6 h for both arms** |
| checkpoint | every 50 ps, analysable on its own (see §8) |

Running the arms in one batch is what makes the diagnostic arm affordable: the step cost is flat
from B = 16384 (51.0 ms) to B = 32768 (52.7 ms), so the second arm costs **3 %**, not 100 %. Each
seed of the sampler carries its own accumulators, bias field and genealogy, so they are genuinely
independent replicas that merely share a step loop; the analyzer selects an arm by seed and
refuses to mix them.

Two design points worth not re-deriving:

* **The establishment target is bias-aware.** ABF moves the biased equilibrium as it learns, so
  the target is `q*_t ∝ exp(−β(F_pilot − B_t))` with `B_t` the saved bias, not the unbiased
  population. Scoring against the unbiased population would flag a state as starved precisely
  when ABF had correctly flattened it — manufacturing the very signal mFR is supposed to remove,
  from a run in which nothing is wrong.
* **Only discovered states can be under-established.** Counting an undiscovered state as a
  population deficit would report the R15 regime — where there is nothing to clone — as the
  regime mFR repairs. This distinction is the whole gate.

Prediction worth recording before the run, given §5: **B6** (φ>0, χ₁ = g+, basin-of-attraction
weight 0.008 — by far the smallest) is the candidate for under-establishment, and it is
structurally the analogue of alanine's C7ax.

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
