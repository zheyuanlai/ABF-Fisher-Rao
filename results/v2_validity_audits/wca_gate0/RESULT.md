# WCA dimer Gate 0 — PASSES, and independently corroborates the cached-TI-reference defect

Last of the three provisional backfills required by Amendment 10. The WCA dimer carries the
project's strongest *physical* positive: Case IX, practical mFR **-22.83 %** vs ABF on 16/16
seeds, **-26.38 %** vs its own matched sham.

## Design — what the existing TI reference could not have caught

`constrained_ti_reference_gpu` seeds **every** replica from the same lattice plus a small
jitter. It measures the conditional mean force under **one** solvent preparation and never tests
whether the preparation matters. However precise, it is structurally blind to a cage-
equilibration failure.

Four pools, built to break it if it is breakable, all constrained at the same `z`:

| pool | preparation |
|---|---|
| `lattice` | the standard TI preparation (control) |
| `from_lo` | equilibrated at `z = -0.2` — compact dimer, **tight** cage — then projected |
| `from_hi` | equilibrated at `z = +1.2` — stretched dimer, **open** cage — then projected |
| `hot` | solvent randomised — a deliberately destroyed cage |

`from_lo` / `from_hi` is the physical failure mode: a walker driven quickly along `z` by the ABF
bias carries the wrong cage with it. 6 z-values x 4 pools x 256 replicas, 20 k prep + 20 k
equilibration + 80 k production each.

## Result 1 — Gate 0 PASSES

| | `<f_loc>` spread across pools / \|F'_ref\| |
|---|---|
| all z | **0.040** |
| transition region 0.25–0.75 (where mFR acts) | **0.039** |
| deca-alanine | 0.61 |
| R15 beta=2 | 0.564 / 0.593 |
| gateway | 0.036 global, 0.189 in the constriction |

Cages prepared compact, stretched, lattice and randomised all give the same conditional mean
force to ~4 %. **The solvent cage equilibrates at fixed `z`.** WCA is as clean as the gateway and
~15x cleaner than deca, with no degradation in the transition region — unlike the gateway, whose
constriction sat at 0.189.

**The establishment-limited classification, and the -22.83 % Case IX positive resting on it,
survive the backfill.**

## Result 2 — the cached TI reference is independently implicated

The same run separates pool *agreement* from agreement *with the reference*:

| z | pool mean | TI ref | pool spread | \|pool − ref\| | rel dev |
|---|---|---|---|---|---|
| 0.00 | 5.459 | 5.184 | 0.149 | 0.275 | 0.053 |
| **0.25** | **0.931** | **2.094** | **0.179** | **1.163** | **0.555** |
| 0.40 | 3.120 | 2.950 | 0.172 | 0.170 | 0.058 |
| 0.55 | -2.875 | -2.796 | 0.147 | 0.078 | 0.028 |
| 0.75 | -6.440 | -6.377 | 0.060 | 0.063 | 0.010 |
| **1.00** | **-0.521** | **-0.396** | 0.077 | 0.125 | **0.315** |

At `z = 0.25` four independently prepared cages agree with each other to **0.179** while sitting
**1.163 (56 %)** away from the cached reference. Agreement among independent preparations that
large cannot be explained by their own sampling error, so the discrepancy is in the **reference**,
not in the conditional ensemble. `z = 1.00` shows the same pattern at 32 %.

This independently corroborates `CLOSURE_v1.md` §5a, where a parallel audit found the cached TI
reference sitting 0.264 rms from a three-replica high-precision consensus and halving a related
WCA contrast (-4.75 % -> -2.41 %). **This audit reaches that conclusion by a different route —
conditional pool agreement rather than reference replication — and localises it: `z = 0.25` is
inside the transition region where the mFR effect lives.**

## What this does and does not license

* **Does:** WCA's establishment-limited classification is confirmed on mechanism. Conditional
  equilibration is not its problem, so the Case IX effect is not a deca-style artifact.
* **Does not:** validate the *magnitude* -22.83 %. That number is scored against the cached
  reference, which this audit finds locally wrong by 56 % in mean force at `z = 0.25`. The
  queued high-precision re-run (task: "Re-run WCA Case IX against a high-precision TI reference")
  is now better motivated, not less.

## Limitation

Six `z` values, not the full 160-point grid; `f_loc` is averaged over 256 replicas per pool with
80 k production steps. The pool-spread statistic is robust at this resolution, but the
per-`z` reference deviations should be confirmed on a denser grid before being quoted as a
correction to the reference itself.
