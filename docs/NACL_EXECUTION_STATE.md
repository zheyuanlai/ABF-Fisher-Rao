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
CURRENT STAGE:          N=64 CELL VERDICT: ABF-SUFFICIENT (2026-08-14 23:26 UTC, 8/8 seeds).
                        Gate B 8/8 both states (non-binding, validity attached); Gate C 0/8
                        deficient either state.
                        MAP KILLED OFF GPU 3 at 2026-08-15 00:04 UTC to comply with 16.4.
                        The banner above was ALREADY in this file when the map was launched
                        on GPU 3 at 23:34 and resumed there; I read the machine and not the
                        rule, and cited 15.4 to a peer without checking it was still current.
                        The 30 min on GPU 3 produced NOTHING: no checkpoint was written after
                        the resume (state file still step 30000 / 60 ps), the kill landing
                        ~1 min before the first post-resume checkpoint came due.
                        REMAINING LADDER IS N=32 ALONE. N=16 and N=8 are NOT COMPUTABLE by
                        arithmetic, not by measurement: lambda = Q* N with Q* fixed by the
                        ACCEPTED reference gives max lambda 15.6 and 7.8, both < 16, for every
                        state. No amount of sampling changes a number that does not depend on
                        the sample. Running them costs ~2.5 GPU-days to produce two cells that
                        cannot be classified, so they are struck and NOT relaunched (user
                        decision 2026-08-15); N=32 (SSIP lambda ~30.7) is the only cell that can
                        still return a verdict. The exclusion is now DERIVED in code by
                        nacl_gates.map_completeness() rather than argued in prose, and is strict:
                        Q*_k < 1 whenever a second basin carries positive target, so lambda < N,
                        so N=16 cannot reach 16 either -- it is struck, not merely doubted.
                        N=32 therefore runs UNPACKED (256 walkers) rather than packed with 8/16
                        (448): measured 1394 steps/min = 1028 ns/day at 256 against an inferred
                        400-500 steps/min = 516-645 ns/day at 448, so packing looked 1.6-2.0x
                        WORSE in aggregate. Treat the 448 figure as INDICATIVE, not a clean A/B
                        -- inferred from checkpoint spacing under a >=20 min rule on a run whose
                        exclusive GPU access cannot now be confirmed. The decision it supports is
                        safe either way, since the packed run also spent 2/3 of its budget on
                        cells that cannot be classified.
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
2026-08-15 00:38 INCIDENT -- A RETIRED AUTOLAUNCH FIRED. tau_perp exited early and a watcher
                        armed BEFORE Amendment 16.4 launched the pre-16.4 TWO-GPU map split with
                        the pre-16.4 device assignment baked in: screen_map_A (seeds 4000-03) on
                        GPU 3 -- C60's device, co-resident with their dt gate -- and screen_map_B
                        (seeds 4004-07) on GPU 2. Both ran cells 8,16,32; both skipped preflight.
                        Killed at 00:39 and 00:40. THE GUARD DID NOT FAIL, THE LAUNCH WENT AROUND
                        IT: a permission check only protects launches that call it. Two remedies,
                        both needed -- the methane session's DISARMAMENT (a study should own no
                        process able to start work), and the C60 session's rule that a watcher
                        must point at the LADDER, never the payload, so a stale firing re-enters
                        the guard. scripts/nacl_launch_N32.sh complies with the second by
                        construction. C60's ladder took an unattributed SIGTERM inside the window;
                        no NaCl script contains any kill (verified by grep), origin undetermined,
                        they restarted from the top and are unharmed.
                        Half-block hazard worth keeping: screen_map_B alone was seeds 4004-07,
                        and require_full_block REFUSES a 4-seed verdict -- so the autolaunch's
                        output could not have been analysed even if it had been permitted.

RUNNING NOW:            N=32, PID 3835641 on GPU 2 since 2026-08-15 01:14 UTC, launched by
                        scripts/nacl_launch_N32.sh -- preflight PASSED all five checks including
                        the re-read of the governing compute clause; 80 tests; GPU 2 verified
                        idle. 256 walkers, seeds 4000-4007 in ONE process, 1562500 steps.
                        From the pinned worktree at 53dfb30 -- NOT a new pin: the sampler delta
                        53dfb30..HEAD is one diagnostic print plus a manifest field (src/nacl/
                        untouched, verified by diff), so the data-generating process is identical
                        across N=64 and N=32. ANALYSIS runs at HEAD, which is a DIFFERENT pin --
                        53dfb30 is not an ancestor of f88e434, so the worktree's nacl_gates.py
                        has no power guard, no cell-map guard and Gate A transposed. A
                        gates_report.json without an `analysis_provenance` block came from that
                        superseded tree and is not a verdict.
                        IN-SITU RATE 1394 steps/min = 1028 ns/day aggregate (steps 30000->90000
                        over 43 min). ETA ~19:56 UTC 2026-08-15, ~18.7 h total.

NEXT PERMITTED ACTION:  when N=32 completes, link cell_N32.npz and cell_N64.npz into one
                        directory and run nacl_gates.py over BOTH. REHEARSED 2026-08-15 on a
                        SYNTHETIC N=32 cell (scratchpad only, never under results/): the
                        two-cell path runs clean, map flips to COMPLETE, the verdict is emitted,
                        Gate B handles a 32-walker xi_trace, and the guard marks CIP non-binding
                        (lambda 0.78) with SSIP binding (30.72) at N=32. The rehearsal validates
                        PLUMBING ONLY -- the synthetic cell was built by scaling N=64 occupancy
                        by 32/64, so its lambda matching the projection is tautological, not
                        evidence. The real cell has its own bias trajectory and its own Q*(t). The map is then COMPLETE
                        (8 and 16 struck a priori) and the study-level verdict is emitted; until
                        then it is WITHHELD by map_completeness() and correctly so.
                        Read the outcome against results/nacl/CLOSURE_PRECOMMIT.md, which fixed
                        all four branches BEFORE the number existed -- including that CIP may
                        come back INCONCLUSIVE from the windowed audit and must then be reported
                        UNKNOWN, not folded into the null.
                        If N=32 shows no deficit on a POWERED state: NaCl is not an mFR candidate
                        under the preregistered budget -- write the closure with the pre-committed
                        weak-null caveat (commit addfbed) attached, unchanged.
                        ONLY if Gate C fires on a powered state: tau_perp must be re-run (~4 h,
                        GPU 2, no checkpointing) before Gate D. It is DOWNSTREAM of Gate C and
                        must not pre-empt the cell again.
DEFECT FOUND 2026-08-15, FIX DEFERRED: **caching a prerequisite skips the checks attached to
                        building it.** nacl_screen.py's `[init]` declared-bias check (commit
                        c21ef04, added *because* "a declared bias is an assumption about the
                        initial condition and nothing checked it") lives INSIDE the
                        population-BUILD branch. The N=32 run reused a cached populations.npz to
                        save 18 min, so the check never ran and manifest.initial_condition_check
                        will be null -- a check that did not run, reading as no problem.
                        NO EFFECT ON THE SCIENCE: the check depends only on R_CIP_NM and the
                        accepted reference, both frozen, so it was recomputed standalone ->
                        results/nacl/screen_N32/initial_condition_check.json. Start 0.300 nm sits
                        **2.64 kT** above the CIP minimum at 0.260 nm, declared_bias_holds FALSE,
                        identical to N=64 as the frozen inputs require.
                        FIX DEFERRED ON PURPOSE: hoisting the check out of the build branch is a
                        one-line diagnostic change, but editing the sampler mid-campaign risks a
                        launch from HEAD picking up a different sampler than the pinned worktree,
                        and ladder homogeneity is worth more than an early diagnostic fix. Do it
                        after closure. The general form is worth carrying: any check attached to
                        a build step is silently skipped by a cache hit.

OPEN ITEM (NOT ADOPTED): `scripts/nacl_screen_merge.py` carries an UNCOMMITTED change from the
                        session that stood down, replacing the hard-coded seed-axis table with
                        shape inference. Left in the working tree, not adopted, not reverted.
                        Two hazards before anyone commits it: (i) it classifies a field as
                        seed-indexed when `shape[0] == s0`, and s0 = 4 for a half-block, so any
                        field with a leading dimension of 4 is misclassified; (ii) its fallback
                        SILENTLY AVERAGES arrays it cannot classify instead of raising -- an
                        unknown layout reading as a result. Its own docstring's argument against
                        a stale table is fair, but a wrong inference is silent where a stale
                        table is loud. `nacl_merge_halves.py` (explicit axes, asserts grid/N/T/
                        dt/box/domain/schedule, requires disjoint seed sets unioning to
                        4000-4007) remains the AUTHORITATIVE merge. Off the critical path
                        regardless: N=32 runs all 8 seeds in ONE process and needs no merge.

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
