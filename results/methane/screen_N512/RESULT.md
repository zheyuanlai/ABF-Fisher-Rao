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

**Discovery is instant.** All three states are reached within **0.8 ps** of a 200 ps run — 25×
faster than the 20 ps threshold, on every seed:

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

> The CV is excellent. Discovery is instant. Establishment is immediate. There is simply nothing
> for population reallocation to repair.

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
