# Methane is ABF-sufficient. The preregistered null, confirmed on 8/8 seeds.

ABF-only screen, `N = 512` walkers × 200 ps × 8 seeds (5000–5007), scored against the accepted
constrained-TI reference. **No Fisher–Rao arm was run, and none is licensed.**

§9 of the preregistration called methane a **likely-null / falsification benchmark** before any
methane code existed. It is a null. That is the preregistered outcome, reported as a result.

---

## Verdict

```
Gate 0   pinning clause: max tercile occupancy, 2nd half, any seed = 0.516   (<= 0.90)   OK
         conditional mean force (reference stage) = 0.048 global / 0.081 core            PASS
Gate A   max pairwise TV(p(xi | Y)) = 0.935  [as preregistered]              (>= 0.30)   PASS
Gate B   T_hit < 20 ps on 8/8 seeds, for all three states                    (>= 6/8)    PASS
Gate C   power: lambda = Q* N = 127.6 / 147.0 / 224.2   all three BINDING    (>= 16)     OK
         persistent deficit on 0/8 seeds, for all three states                           NO DEFICIT

VERDICT: ABF-sufficient -- STOP.
```

Gate C's power line was added 2026-08-14 (Amendment 12.10) and the numbers above are the re-run
through it. It is not decoration: Gate C tests an **integer walker count**, so at
`lambda = Q* N < 2` the threshold `occupancy < 0.5 Q*` becomes arithmetically identical to
*"the state is empty right now"*. Methane clears the bar by 8-14x on every state and resolves
**13-18 %** deficits against the 50 % it tests for, so the guard is inert here **by measurement**.
It is not inert everywhere: NaCl's `N = 16` and `N = 8` cells have no state reaching it at all.

Classification is by the first failing gate (Amendment 10). Nothing failed; Gate C found no
deficit, which is the ABF-sufficient branch.

## The numbers behind it

**Discovery is instant — but Gate B is nearly vacuous here, and that is a caveat on the gate,
not on the verdict.** Prompted by the NaCl session reporting the same sub-ps `T_hit` in a system
with a **5.34 kT** barrier, the numbers were checked against a ballistic estimate:

| | state 1 (0.520 nm) | state 2 (0.710 nm) |
|---|---|---|
| distance from the contact start (0.38 nm) | 0.140 nm | 0.330 nm |
| ballistic transit at `sqrt(2 kT/m)` = 0.556 nm/ps | **0.252 ps** | **0.594 ps** |
| observed `T_hit` | 0.25 ps | 0.5–0.8 ps |

The agreement is essentially exact. **`T_hit` here is measuring free-flight transit across a
tercile boundary, not a barrier-crossing rate** — the first `xi` trace frame after `t = 0` is at
0.25 ps, so state 1 is entered on the first frame it *could* be. Diffusive transit of the same
0.33 nm would be ~36 ps for a typical walker; the fastest of 512 arrives ballistically.

With a single-basin reference the states are the Amendment 3 **tercile fallback**, declared as a
partition of the coordinate rather than as metastable states — so a contact start sits 0.14 nm
from the first boundary and Gate B could not have failed.

That is a property of the partition, and it is why the verdict rests on **Gate C**, which is not
vacuous: occupancy tracks the bias-aware target across the whole domain for the entire second half.

One thing that *does* hold, and was checked rather than assumed after the NaCl session found the
opposite in its own run: **the start is exactly at the reference minimum** — `r = 0.380 nm`,
`W = -1.108 kT`, `0.000 kT` above the global minimum of `W`. SPEC §6.1's declared bias ("a contact
start makes discovery harder, so it can only push toward discovery-limited") therefore holds here
as written. Their start sat 2.64 kT above their CIP minimum, giving every walker two thirds of the
barrier for free — the same clause, written by them, invalidated by a configuration nobody had
checked against it.

**The far-threshold check the NaCl session used, applied here, goes further: methane's discovery
is not measurable as a rate anywhere in the domain.** First arrival at thresholds well past the
tercile boundary, against the correct ballistic floor — the *fastest of 512* walkers, which is
3.1 sigma of the relative-velocity distribution (1.72 nm/ps), not the rms 0.556:

| threshold | distance from start | fastest-of-512 ballistic | observed median |
|---|---|---|---|
| 0.60 nm | 0.220 nm | 0.13 ps | 0.25 ps |
| 0.70 nm | 0.320 nm | 0.19 ps | 0.50 ps |
| 0.80 nm | 0.420 nm | 0.24 ps | 0.88 ps |
| 0.89 nm | 0.510 nm | 0.30 ps | 1.75 ps |

Arrivals are **1.9–5.8x** the extreme-value ballistic floor and reach the far wall of a 0.51 nm
domain in under 2 ps. Methane's whole domain is only ~3 ballistic steps wide for the leading
walker, so there is no separation between "crossed a barrier" and "flew across the box".
**There is no transition to time.** That is not a defect in the measurement; it is what a 0.67 kT
barrier means.

**The discriminator is domain width in ballistic steps, not anything about the method.** The NaCl
study ran the identical gate on a **1.20 nm** domain — about 2 trace frames of free flight for its
leading walker — and measured **6.9–20.7x** its floor at far thresholds against an exact
extreme-value quantile: a resolvable, unambiguously diffusive rate. No budget would give methane
that separation. So the same gate, run the same way, is informative in one system and vacuous in
the other, and **Gate B's vacuity here is a property of the physics rather than a defect in the
gate.**

(The floor itself took four passes across both studies to get right — rms of one walker, then the
mean rather than a quantile, then two-sided rather than outward-only, then a leading-term
approximation — before being replaced by the exact `Phi^-1(q^(1/n))`. Recorded in Amendment 12.9;
none of it moves the verdict, which rests on Gate C.)

Raw per-seed numbers: all three states reached within 0.8 ps on every seed.

| seed | T_hit state 0 / 1 / 2 (ps) | worst deficit span | max tercile occupancy |
|---|---|---|---|
| 5000–5007 | 0.0 / 0.2 / 0.5–0.8 | **0.0 ps** on every seed | 0.498–0.516 |

## Gate C was shown to FIRE, and it needs 60 % — not the 13–18 % the power argument implied

Every methane state cleared Gate C, so nothing in this study had ever demonstrated the gate
*firing*. The power argument (`lambda = 127.6/147.0/224.2`, so `2/sqrt(lambda)` = 13–18 %)
characterises **counting noise and nothing else** — it is a single-checkpoint statement, while
the gate requires a **contiguous 40 ps run** below `0.5 Q*`. Contiguity suppresses false firing,
which is intended, and suppresses true firing, which was never accounted for.

Prompted by the NaCl session noting the same gap in its own CIP statistic — where it cannot be
closed, because no NaCl state had a real deficit either — this was closed here by **planting
walker-conserving deficits in the real traces** (scale one state's occupancy by `f`, redistribute
the removed population over the others so the partition still sums to 1) and running the
unmodified gate. Real traces, not synthetic: the correlation structure is the entire quantity in
question. Walkers are subsampled to read the threshold at the `lambda` a smaller cell would see.

| `lambda` | analytic 2σ, one checkpoint | **deficit the gate actually needs** | ratio |
|---|---|---|---|
| 224.2 | 13 % | **60 %** | 4.5× |
| 147.0 | 16 % | **60 %** | 3.6× |
| 127.6 | 18 % | **60 %** | 3.4× |
| 28.0 | 38 % | 65 % | 1.7× |
| 16.0 | 50 % | 65 % | 1.3× |
| 9.2 | 66 % | 70 % | 1.1× |
| 7.0 | 76 % | 75 % | 1.0× |
| 4.0 | 100 % | 75 % | 0.8× |

Firing is 0/8 seeds at a planted 50 % and 8/8 at 60 %, with 55 % partial (2–6 of 8).

**Three consequences, and the first is a correction to this project's own guard.**

1. **`GATE_C_MIN_LAMBDA = 16` does not deliver what its rationale claims.** It was derived from
   `0.5 lambda >= 2 sqrt(lambda)` — the `lambda` at which a 50 % deficit is a 2σ effect *on one
   checkpoint*. Measured, the gate at `lambda = 16` needs **65 %**, not 50 %. The threshold is
   still worth having and 16 is still a reasonable place for it, but **the claim attached to it
   was the wrong quantity** — failure class 6, in the guard written to fix failure class 6.
2. **The real detection threshold is 60–75 % across a 50× range in `lambda`.** It is set by the
   contiguity requirement, not by counting noise; `lambda` governs the gate only below ~9, where
   the two criteria cross and the analytic figure becomes the conservative one.
3. **The verdict does not rest on the gate, and this is why that matters.** "Gate C did not fire"
   licenses only *no deficit ≥ 60 % occurred*. The statement that carries the null is the
   **direct measurement**: the worst occupancy/`Q*` ratio over 8 seeds × 3 states × the entire
   second half is **0.83**, i.e. **the largest shortfall that ever occurred anywhere is 17 %**.
   That is a measured bound, not a gate output, and it is 3.5× tighter than what the gate could
   have told us.

### The span requirement is 10x stricter than it needs to be, and relaxing it strengthens the null

The NaCl session replicated the calibration and got **55 %** against this study's 60 %, and we
recorded the difference as confounded — two systems, two span settings. It is separable at fixed
system by varying the span, which is seconds of analysis on traces already on disk. **24
state-seeds (8 seeds x 3 states), planted 50 % deficit against the unmodified trace:**

| required span | fires with NO deficit | detects a real 50 % deficit |
|---|---|---|
| 0.02 T | **0/24** | **21/24** |
| 0.05 T | 0/24 | 5/24 |
| 0.10 T | 0/24 | 2/24 |
| **0.20 T (preregistered)** | **0/24** | **0/24** |
| 0.30 T | 0/24 | 0/24 |
| 0.40 T | 0/24 | 0/24 |

**The preregistered span buys nothing measurable and costs all of the detection power at 50 %.**
A 10x looser requirement detects 21 of 24 planted deficits at the *same* false-positive count.
The 55 % / 60 % difference between the two studies is therefore **fully accountable by the span
setting alone**, with no system difference required to explain it.

**This is a post-hoc sensitivity analysis and is labelled as one.** The preregistered gate is
`DEFICIT_SPAN = 0.20` and the verdict stands on it. What the sweep adds is that **the null
survives a 10x more sensitive version of its own gate**: run at 0.02 T, where the gate catches
21 of 24 planted 50 % deficits, the real traces still fire **0/24**. A ~50 % deficit was not
there to be found, and that is a stronger statement than the preregistered gate could make.

**Two honest limits.** `0/24` bounds the false-positive rate at **<= 12 %** by the rule of three,
not at zero. And the false-positive column is measured on a system with no deficit and methane's
own autocorrelation; a system whose orthogonal coordinates relax more slowly could produce
spurious runs at a loose span, so **the frontier is measured here and does not transfer**.

Full ladder: `gate_c_detection/ladder.json`, span frontier `gate_c_detection/span_frontier.json`;
script `scripts/methane_gate_c_detection.py`.

**Occupancy sits on the bias-aware target, not near it.** Final-frame occupancy divided by
`Q*_k(t)`, across all 8 seeds and 3 states: the **worst** ratio is **0.83** and the best is 1.09.
Gate C's deficit threshold is 0.5. Nothing came close to a deficit; ABF is populating every
region of the coordinate at the rate the applied bias asks for.

**Gate C is powered, checked rather than assumed.** A null from an underpowered test is
worthless, and the NaCl session found two cells of its own ladder where a 50 % deficit is not
*computable* at any sampling. Detecting `occ < 0.5 Q*` at 2 sigma against binomial counting noise
requires `lambda = N Q* >= 16(1 - Q*)`:

| state | `Q*` | `lambda = N Q*` | required | margin |
|---|---|---|---|---|
| 0 | 0.260 | 133.3 | 11.8 | **11.3x** |
| 1 | 0.292 | 149.4 | 11.3 | **13.2x** |
| 2 | 0.448 | 229.3 | 8.8 | **26.0x** |

At `N = 512` every state is powered by an order of magnitude. **Methane's null is not a
null-because-underpowered** — the test could have seen a 50 % deficit ten times smaller than the
one it was looking for. (For contrast, NaCl's remaining `N = 16` and `N = 8` cells give
`lambda` 15.6 and 7.8, below the threshold: those cells cannot return a verdict at any budget,
which is a property of the design rather than of the sampling.)

**The deficit test is per-checkpoint, not a mean.** `occ` is the instantaneous walker fraction at
each trace frame and the criterion is the longest *contiguous* run below `0.5 Q*`, so a
well-populated stretch cannot average away a stretch where a state fails to hold walkers — the
configuration the guard exists to catch.

**The walkers are not pinned.** Maximum occupancy of any one tercile over the whole second half
is 0.516 against a 0.90 clause — the population stays spread across the domain rather than
collapsing to an end, which is the failure deca-alanine showed at 0.951–0.9996.

## Why: the physics, measured beforehand

The accepted reference reproduces the literature structure at the right positions — contact
minimum 0.38 nm, desolvation barrier 0.56 nm, solvent-separated minimum 0.72 nm — but **every
feature is sub-kT**: −1.11, +0.40, −0.27 kT. The barrier measured from the higher minimum is
**0.67 kT**.

There is no metastability in `xi` for ABF to be slow about. `T_hit < 1 ps` is what a 0.67 kT
barrier looks like from the sampler side, and it was predicted from the reference **before the
screen ran** and recorded in `results/methane/ref/RESULT.md` at the time, precisely so it could
not be presented afterwards as a confirmed prediction.

## Why this is a good negative, not a wasted one

**Gate A passes, and the number to quote is 0.935, not 0.987.** The two differ because the
implementation computed the *transpose* of the preregistered quantity, found by asking the NaCl
session's question — *does passing this gate entail the claim it is cited for?*

§2.2 specifies `TV( p(xi | Y=a), p(xi | Y=b) )`: whether the **marginal in `xi`** can see the
difference between structural states, which is what licenses a marginal method to act on them.
`methane_gates.py` computed `TV( p(n_gap | tercile_a), p(n_gap | tercile_b) )` — whether the
descriptor differs between `xi`-terciles. That is a weaker and partly tautological statement,
since the gap volume grows with `r` by construction.

Recomputed as specified, over 823 296 paired samples with `Y` bucketed dry / mid / wet:

| pair | TV( p(xi \| Y) ) |
|---|---|
| dry (`n_gap` < 1) vs mid (1–2) | 0.805 |
| **dry vs wet (> 2)** | **0.935** |
| mid vs wet | 0.474 |

**Is 0.935 near-tautological?** The NaCl session found its own corrected Gate A reading exactly
1.000 and, rather than report it, measured why: its hydration descriptors vary 14–83× more across
`r` than at fixed `r`, so the label is nearly a function of the coordinate and the test nearly
cannot fail. The same question here, measured the same way over 0.01 nm bins:

```
n_gap variance BETWEEN r-bins   0.7363
n_gap variance WITHIN  r-bins   0.1373
orthogonality ratio             5.36x     (NaCl: 14-83x)
```

`n_gap` spans 0.21 → 2.73 across the domain but retains an sd of 0.37 at fixed `r`. At 5.4× the
descriptor is **substantially but not wholly determined by the coordinate**, so Gate A's 0.935 is
a real measurement rather than a restatement of the geometry — considerably more informative here
than NaCl's 1.000, which its own session correctly reports as physics rather than as a strong
gate. That difference is itself the point: methane has genuine solvent structure orthogonal to
`r`, which is why it is a meaningful test of whether ABF needs help, and it still does not.

**0.935 against a 0.30 threshold — Gate A passes as preregistered**, and passes on the quantity
that actually supports the claim. The verdict is unchanged; the number and its meaning are
corrected. The transposed 0.987 should not be quoted.

For a day that correction lived **in this file only**. `methane_gates.py` still computed the
transpose and wrote `gateA_max_TV: 0.987` into `gates.json`, so the retracted number was what the
code emitted and what any re-run would have reported. Fixed 2026-08-14 (Amendment 12.11): the
script now computes `TV(p(xi | Y))` over the 823 296 paired samples directly, reproducing
0.805 / 0.935 / 0.474 exactly, and keeps the transpose labelled *diagnostic only, do not quote*.
**A correction that lives only in a results file is not a correction.**

> The CV is excellent. Establishment is immediate. There is simply nothing for population
> reallocation to repair.

(Deliberately no longer citing "discovery is instant" as evidence: with a tercile partition and a
contact start, Gate B measures ballistic transit and could not have failed. The substantive
finding is Gate C.)

That is a cleaner negative control than one where a bad coordinate obscures the answer, and it is
what §9 asked methane to provide: a literature-anchored benchmark where selection-enhanced ABF has
no room to help, against which the campaign's positives can be read.

**Gate 0 passing matters too.** Methane is *not* conditional-equilibration-limited (0.048 global
against deca's 0.61), so this null is a statement about the free-energy landscape rather than
about a broken baseline. The four-regime map now has a case where every gate passes and the answer
is still "mFR is unnecessary".

## What does NOT happen next

Per §9 and Amendment 11.1, **binding**:

* no mFR arm, no calibration on seeds 5100–5103, no production on 5200–5215;
* **the run length is not raised** until a deficit appears — the result is "ABF-sufficient at
  200 ps", and tuning the budget until methane becomes interesting is refused;
* the preregistered successor, if the campaign wants a harder hydrophobe, is the published
  size series — ethane, then propane — not a retuned methane.

Q1 (Fisher–Rao versus prior directed selection) remains open and passes to NaCl, which was always
where it would be answered if methane failed.

> **Update 2026-08-15: NaCl closed ABF-sufficient too, so Q1 was not answered there either.**
> Its map is complete — `N = 64` and `N = 32` classified with no SSIP deficit on 8/8 seeds
> (`lambda` 61.45 and 30.75), `N = 16` and `N = 8` **not computable a priori** because `Q* < 1`
> forces `lambda = N Q* < N <= 16`. No mFR arm was licensed in either system.
>
> **The two nulls are not of equal weight, and the campaign must not present them as two
> independent negatives.** NaCl's hydration varies 14–83x more across `r` than at fixed `r`
> against methane's **5.4x**, so for NaCl "mFR had nothing to work with" survives as a live
> alternative to "mFR was not needed". Methane's margins (Gate C resolving 13–18 % on states
> holding `lambda = 127.6/147.0/224.2`) are what convert *we saw no deficit* into *a deficit of
> the size we care about was not there*. **Methane earns that; NaCl earns it for Gate C and not
> for the study.** The load-bearing null is this one.

## Finite-size gate: run after the fact, and inconclusive

SPEC §1.3's check was never run before the reference; it was run afterwards, on a 1024-water box.
Differences at `r = 0.70 / 0.80 / 0.90` are `+0.0013 / +0.1060 / +1.1563` kT/nm against a 0.1
tolerance — a literal FAIL at the outer two. **Nothing is resolved**: the combined SEMs are
±0.670 / ±0.500 / ±0.671, so the nominal failure at 0.80 is 0.21σ and the one at 0.90 is 1.72σ.
The tolerance is 5–7× finer than the uncertainty of the measurement it gates.

**The verdict was tested against the worst case rather than argued.** Perturbing the reference by
the *full* measured difference, re-integrating, and re-running Gate C over all 8 seeds and the
entire second half distorts `F` by 0.062 kT and leaves a minimum occupancy ratio of **0.733**
against the 0.5 threshold. **No deficit appears. The classification is unchanged.**

Interior (`r ≤ 0.70`) is confirmed at 0.0013 kT/nm, and every state boundary lies below it.
Outer points are provisional. Full detail: `../finite_size/RESULT.md`.

## Cost

| stage | wall |
|---|---|
| NPT box | 6 min |
| baths + 174 minimised per-r starts | ~35 min |
| constrained-TI reference (1392 trajectories) | 13.78 h |
| screen, 8 seeds × 512 walkers × 200 ps | 17.13 h + 11.5 h (first two seeds, other GPU) |

Engine: batched periodic LJ + smooth PME at **2.9e-13 / 1.4e-15** parity against OpenMM,
constrained BAOAB matching OpenMM's thermostat to 0.51 K.

---

## Closed state, verified 2026-08-15

Methane has **no armed automation and no GPU processes**. Verified rather than assumed:
`ps` shows no `methane_screen`, `methane_ti_torch`, `methane_baths`, `methane_box` or
`methane_triton` process, and no watcher waiting to launch one. GPUs 0/1 belong to another
user; GPU 2 is NaCl's `tau_perp`; GPU 3 is C60's `dt_gate` under Amendment 16.4.

This matters because of a failure the NaCl session hit and diagnosed: **a measurement can only
confirm the state of the world, and a rule is not part of the state of the world.** It launched
on a device that was genuinely free and genuinely not allocated to it, having consulted
`nvidia-smi` rather than the amendment that had superseded its allocation. Every watcher this
study armed had the same shape — they polled for facts (seed files present, compute apps absent,
process liveness) and none checked whether it was still *permitted* to do what it was armed to do.
Had the allocation moved overnight, they would have proceeded exactly as that launch did.

For a finished study the correct remedy is not a permission check but **disarmament**: a closed
study should own no process capable of starting work. That is now the state, and it is recorded
here so that "methane is closed" is a checkable claim about the machine rather than a statement
about intent. The NaCl session's `nacl_preflight.py` is the right remedy for a study still
running, and re-derives its governing clause from the preregistration at launch time rather than
from a cached copy.
