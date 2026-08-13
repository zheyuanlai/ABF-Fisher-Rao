# NaCl execution state (Amendment 15.3)

Read this, the SPEC, and the amendments before doing anything. This is an execution/validation
campaign, not a design session. Update this file after every stage. Never run a later gate if
an earlier one failed; never search for a way around a STOP.

```
CURRENT FROZEN COMMIT:  see results/nacl/PINNED_COMMIT (written when the worktree is cut)
CURRENT STAGE:          launch ladder PENDING (waiting for GPU 2 handover ~06:30 UTC)
LAST COMPLETED GATE:    engine equivalence 11/11 <1e-6; box frozen L=2.892700 nm;
                        descriptors frozen; baths verified (9 baths, 549 starts)
VERDICT:                dt UNDECIDED — first gate run RETRACTED (bad error bars, 2.5σ);
                        rerun governed by the frozen Amendment 15.1 rule
NEXT PERMITTED ACTION:  (ladder RUNNING from the pinned worktree, waiting on GPU 2 behind
                        methane seed 5004) -> after ladder review: launch the TI reference on
                        GPU 2 ALONE. GPU 3 is methane's until ~20:30-22:30 UTC Thu (six more
                        seeds + their benchmark); explicit "GPU 3 released" signal agreed.
                        Split-build parallelism is NOT decided: the per-point retirement rule
                        is a JOINT criterion over builds, so a split requires retirement-off
                        flat 250 ps -- decide from the ladder's measured throughput, record here
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
