# NaCl execution state (Amendment 15.3)

Read this, the SPEC, and the amendments before doing anything. This is an execution/validation
campaign, not a design session. Update this file after every stage. Never run a later gate if
an earlier one failed; never search for a way around a STOP.

> **GPU 3 IS NO LONGER NaCl's (Amendment 16.4, 2026-08-14, user directive).** The 15.4
> "both devices once methane vacates" clause is superseded: the C60 study (Amendment 16)
> owns GPU 3; NaCl stays on GPU 2 — the N = 32/16/8 map restart runs there sequentially
> after half-B. Scheduling only; no physics, seed, budget, gate or endpoint changes.

```
CURRENT FROZEN COMMIT:  see results/nacl/PINNED_COMMIT (written when the worktree is cut)
CURRENT STAGE:          REFERENCE ACCEPTED (250 ps, 12.0 h). Gate 0 and Gate A PASS.
                        Screen N=64 launching: seeds 4000-4003 on GPU 2 now, 4004-4007
                        on GPU 3 when methane releases it (6/8 seeds done).
LAST COMPLETED GATE:    engine equivalence 11/11 <1e-6; box frozen L=2.892700 nm;
                        descriptors frozen; baths verified (9 baths, 549 starts)
VERDICT:                GATE 0 PASS (0.0075 global / 0.0483 barrier -- the campaign's
                        best; WCA passes at 0.040, deca FAILED at 0.61).
                        GATE A PASS (max TV 0.9959 vs 0.30; n_NaO 0.990, n_ClH 0.996,
                        n_bridge 0.947). Reference ACCEPTED, ratio 0.0907 <= 0.5, complete,
                        3 builds x 4 families x 3 replicas x 250 ps.
                        Basins: CIP [0.20,0.34] min 0.26; merged outer [0.34,1.40].
                        dW_CIP->outer 2.54 kT; barrier 5.34 kT.
                        dt = 2 fs DECIDED (Amendment 15.1, one run, never revisited):
                        2fs dT 0.74+-0.32 K PASS / 1fs dT 0.22+-0.30 K PASS; constraints
                        2.1e-15 nm. Triton PASSES both gates but is NOT adopted: measured
                        918 ns/day vs tensor 1020 at its best config, so the reference runs
                        the already parity-gated tensor path at max-batch 256.
NEXT PERMITTED ACTION:  wait for the N=8/16/32 map cells, then run nacl_gates.py over ALL
                        four cells. The frozen rule is the SMALLEST N passing every gate, so a
                        smaller cell could still be establishment-limited and the study is NOT
                        closed on N=64 alone. If no cell is eligible: NaCl is not an mFR
                        candidate under the preregistered budget -- write the closure with the
                        pre-committed weak-null caveat (commit addfbed) attached.
FORBIDDEN ACTIONS:      launching the TI reference (separate reviewed action after the ladder);
                        any screen cell before Gate 0/A; any mFR before Gates 0-D; editing the
                        SPEC except by numbered amendment; retuning anything against a result;
                        patch-and-continue inside a ladder run (patch -> test -> commit -> new
                        pin -> restart ladder)
EXPECTED OUTPUT FILES:  results/nacl/stage1/{launch_manifest.json, benchmark.json,
                        dynamics_gate.json, ladder.log}, results/nacl/ti_smoke/,
                        results/nacl/_resume_check/
```

## Stage ladder (each stage appends its outcome here)

| stage | status | outcome |
|---|---|---|
| 0A extract + parity | DONE | 11/11 at <1e-6 (`tests/test_nacl_engine.py`) |
| box (NPT) | DONE | L = 2.892700 nm; finite-size gate passes marginally, recorded |
| descriptor freeze | DONE | R0: NaO 0.315 / ClH 0.285 / ClO 0.375 nm; peaks at literature positions |
| baths + per-r starts | DONE | 9 baths, 549 starts, separations exact, forces safe |
| dt gate (first run) | **RETRACTED** | std/√n over correlated samples; 2 fs "FAIL" was 2.5σ |
| launch ladder | PENDING | — |
| dt gate (15.1 rerun) | PENDING | inside the ladder |
| TI reference | NOT PERMITTED YET | after ladder review; builds split over GPUs 2+3 per 15.4 |
| Gate 0 / Gate A | — | from the accepted reference |
| screen N=64 (alone first) | — | only if 0 and A pass; analyze B/C before any other cell |
| Gate D / calibration / mFR | — | only if licensed, in order |

## Provenance audit of pre-pin artifacts (2026-08-13 06:28, prompted by the methane session)

Everything before the pinned worktree ran from the working tree, so each input artifact was
audited against its producing script's commit time. Finding: **three artifacts were produced by
code launched minutes before (or between) the commits that captured it** — stage0 npz (run
01:43, script committed 01:48), the NPT box (launched ~02:01, committed 02:12), and the baths
(launched 02:22 from a tree state between 7667d48 and 7550390; the running interpreter is
immune to the later 03:09 edits). In every case the committed content is believed identical to
what ran, and in no case can that be *proven* — the same class of assurance as "I checked it
carefully", which this campaign has stopped accepting. None of these manifests recorded a git
commit, so no manifest points at a wrong commit (the methane session's sharper version of this
defect). Disposition: accepted as-is and recorded — the artifacts are inputs whose own
*content* is hashed in the launch manifest, and everything from the ladder onward runs from a
pinned worktree, which answers "what produced this input, and is that thing pinned?"
structurally rather than by care.

## Gate B is NON-BINDING for NaCl, established before it was read (2026-08-14, prompted by
## the methane session finding the same vacuity in its own)

`T_hit` is evidence about discovery only if the boundary is out of ballistic reach and above
the trace resolution. Measured for this study:

```
start (published equilibrate.coor)  r = 0.30 nm   F = 3.47 kT   <- NOT the CIP minimum (0.26)
first state boundary                r = 0.34 nm   F = 5.63 kT
distance                            0.040 nm
thermal speed sqrt(kT/mu)           0.423 nm/ps   (mu = 13.95 amu)
ballistic transit                   0.095 ps
xi trace interval                   0.500 ps      <- FIRST recordable frame
observed T_hit                      0.500 ps, every seed
effective climb from start          2.16 kT, NOT the 5.34 kT CIP->top barrier
```

So Gate B **cannot fail** with this initial condition: the boundary is 4.4x inside one
ballistic step and 5x below the trace resolution, and the published start already sits 2.64 kT
up the barrier. The SPEC §7 declared bias ("a contact start makes discovery harder, so it can
only push toward discovery-limited") assumed the start was AT the contact minimum; it is not.

**The conclusion survives on independent evidence, which is why this is a caveat and not a
retraction.** Far-threshold arrivals, where neither the ballistic floor nor the resolution can
manufacture the answer: SSIP minimum (0.52 nm) at 1.0-1.5 ps, outer region (0.70 nm) at
2.5-3.0 ps, well dissociated (1.00 nm) at 6.5-9.5 ps against a ballistic floor of 1.65 ps --
4-6x above ballistic, so genuinely diffusive, and all far under the 156.2 ps threshold.
Discovery really is fast; the Gate B *number* is not what shows it.

**Consequence: the regime verdict rests on Gate C.** Gate B is reported as passing and
non-binding, with these validity conditions attached in `gates_report.json` automatically
(`_diagnostics.T_hit_is_resolution_limited`) so no consumer can quote T_hit without them.

## Pre-committed caveat on the NaCl verdict, recorded BEFORE Gate C is read (2026-08-14 19:0x)

The methane session asked my orthogonality question of its own descriptor, and the contrast
constrains what a NaCl null can claim:

```
                     across-r sd / within-r sd     Gate A (preregistered)
  methane  n_gap                5.4x                    0.935
  NaCl     hydration        14 - 83x                    1.000
```

`n_gap` retains an sd of 0.37 at fixed r across a 0.21-2.73 range, so methane keeps genuine
solvent structure ORTHOGONAL to the coordinate -- there was something for marginal reallocation
to act on, and ABF still did not need it. NaCl's hydration is nearly a function of r.

**Therefore, committed in advance of reading Gate C:**

* if NaCl returns **ABF-sufficient**, the verdict carries the caveat that NaCl has little
  structure orthogonal to `r` (14-83x), so "mFR had nothing to work with" is a live alternative
  to "mFR was not needed", and the null is correspondingly WEAKER than methane's at 5.4x. It
  must not be reported as a second independent null of equal strength.
* if NaCl returns **establishment-limited**, the same fact makes the licence STRONGER: a
  population deficit in a system whose hydration is nearly determined by `r` is a deficit an
  r-marginal method can in principle reach, which is exactly the regime Amendment 8's theorem
  leaves open.
* Gate A at 1.000 is a statement about NaCl's physics, not a strong gate, and is reported as
  such either way.

This is recorded now so it cannot be attached selectively to whichever verdict arrives.
