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
Gate A   max pairwise TV(p(n_gap | tercile)) = 0.987                         (>= 0.30)   PASS
Gate B   T_hit < 20 ps on 8/8 seeds, for all three states                    (>= 6/8)    PASS
Gate C   persistent deficit on 0/8 seeds, for all three states                           NO DEFICIT

VERDICT: ABF-sufficient -- STOP.
```

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

One thing that *does* hold, and was checked rather than assumed after the NaCl session found the
opposite in its own run: **the start is exactly at the reference minimum** — `r = 0.380 nm`,
`W = -1.108 kT`, `0.000 kT` above the global minimum of `W`. SPEC §6.1's declared bias ("a contact
start makes discovery harder, so it can only push toward discovery-limited") therefore holds here
as written. Their start sat 2.64 kT above their CIP minimum, giving every walker two thirds of the
barrier for free — the same clause, written by them, invalidated by a configuration nobody had
checked against it. That is a property of the partition,
and it is why the verdict below rests on **Gate C**, which is not vacuous: occupancy tracks the
bias-aware target across the whole domain for the entire second half.

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

**Occupancy sits on the bias-aware target, not near it.** Final-frame occupancy divided by
`Q*_k(t)`, across all 8 seeds and 3 states: the **worst** ratio is **0.83** and the best is 1.09.
Gate C's deficit threshold is 0.5. Nothing came close to a deficit; ABF is populating every
region of the coordinate at the rate the applied bias asks for.

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

**Gate A = 0.987 is the strongest CV-visibility number in the campaign.** `n_gap` separates the
terciles almost perfectly, so the collective variable sees the solvent physics clearly. Methane
therefore fails on the *state*, not on the *coordinate*:

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

## Cost

| stage | wall |
|---|---|
| NPT box | 6 min |
| baths + 174 minimised per-r starts | ~35 min |
| constrained-TI reference (1392 trajectories) | 13.78 h |
| screen, 8 seeds × 512 walkers × 200 ps | 17.13 h + 11.5 h (first two seeds, other GPU) |

Engine: batched periodic LJ + smooth PME at **2.9e-13 / 1.4e-15** parity against OpenMM,
constrained BAOAB matching OpenMM's thermostat to 0.51 K.
