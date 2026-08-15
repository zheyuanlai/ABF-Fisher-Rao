# C60 execution state machine

Read at the start of every session; updated after every stage.  Companion to
`docs/SPEC_c60_water.md` (frozen, Amendment 16) — this file records *where the study is*,
never a spec change.

## Current state (2026-08-14)

| item | state |
|---|---|
| pinned commit | see `results/c60/PINNED_COMMIT` (written at ladder time) |
| spec | FROZEN at `96bf011` (Amendment 16) |
| device | **GPU 3** (Amendment 16.4; NaCl keeps GPU 2) |
| NPT box | **FROZEN: Lx = Ly = 2.651139 nm, Lz = 5.673840 nm** (`results/c60/box/`) |
| PME | pinned: alpha 2.628260884878466 nm^-1, grid 24 x 24 x 48 (`pme_params.json`) |
| engine parity | in progress — `tests/test_c60_engine.py` |
| dt gate | **first read RETRACTED for cause** (2026-08-15): equipartition PASSED both dt (0.15 K / 0.08 K) and constraints 2e-15 nm, but the 0.968/1.20 nm mean-force spots went **NaN** — exploded trajectories from unrelaxed cage-teleport starts (SD reach 0.06 nm vs ~0.25 nm overlaps) silently forced the 1 fs fallback. Fix: whole-molecule clash pusher + force guards + NaN-raises (`src/c60/prep.py`); the rerun inside the next ladder is the decided-once run |
| reference | not started (3 builds pending) |
| Gate 0 pools | not started |
| screen | **prohibited** until reference accepted + Gates 0/A pass |
| mFR | **prohibited** until an establishment-limited cell is frozen |

## Next permitted action

Run `bash scripts/c60_launch_ladder.sh` at a pinned commit, then (separate, reviewed)
launch reference builds 1–3 sequentially on GPU 3, one process each:

```
CUDA_VISIBLE_DEVICES=3 python scripts/c60_reference.py --build 1
```

## Forbidden actions

* anything on GPUs 0–2 (0/1 other group; 2 is NaCl's);
* torch import in an OpenMM-CUDA process (measured deadlock, NaCl);
* any mFR/FR code, calibration, or pilot — `results/c60/calibration/` and
  `results/c60/production/` must not exist;
* editing `docs/SPEC_c60_water.md` (amendment only);
* extending any run past its frozen budget after seeing a result.

## Incident log

* **2026-08-15 00:38–00:39 UTC:** a stale NaCl watcher (armed pre-16.4) autolaunched the old
  two-GPU map split; one half sat on GPU 3 co-resident with our dt-gate torch phase for ~1
  minute before the NaCl session killed it. Our first ladder run died on an **unattributed
  external SIGTERM** in the same minute (NaCl session states it signalled only its own PID).
  Remedy: ladder restarted from the top per 15.3, so every measurement (including the
  idle-device throughput, the stage a co-tenant corrupts silently) is re-taken. Audit of our
  own automation found nothing armed carrying pre-16.4/16.7 state.
  **Addendum (NaCl session, 00:5x, logged at its stated strength):** a grep across all NaCl
  scripts and wrappers finds no `kill`/`pkill`/`killall` of any kind, so the SIGTERM did not
  come from a NaCl *script*; origin remains undetermined, and a harness- or self-side cause
  within the same minute is equally consistent. "Unattributed" is where the evidence sits.
* **Lesson (from the NaCl session, adopted):** a watcher armed before an amendment carries
  the allocation and thresholds current when it was armed and will act on them without
  consulting anything. Preflight guards protect launches that go through them; only
  **disarmament** protects against the ones that don't. Arm watchers late, point them at the
  guard (never directly at the payload), and kill them on any allocation change.

* **2026-08-15 01:0x UTC, recurrence:** the pre-16.4 NaCl A/B split appeared on GPUs 3+2
  again — this time launched by a **live sibling Claude session's shell command** from the
  worktree `ABF-Fisher-Rao-nacl-run` with `CUDA_VISIBLE_DEVICES=3` explicit (not a watcher).
  Our second ladder run ended at the same dt-gate phase boundary in the same window.
  Forensics sent to the NaCl session; escalated to the user. C60 relaunch armed behind a
  **10-minute continuous-idle** requirement on GPU 3, pointed at the ladder preflight.
* **Attribution upgraded (01:2x), strong circumstantial:** the NaCl session's own N=32 run
  (GPU 2, setsid-detached, ppid 1 — immune to launcher-timeout kills) died in the same
  window. Three processes across two studies died inside the two sibling launch windows;
  the only common element is the sibling's launch sequence, implying it **clears the target
  GPUs before launching, not restricted to its own processes**. Candidate mechanism from our
  own data: a `fuser -k /dev/nvidia{2,3}`-class device clearing fits every observation —
  GPU-selective (the GPU-0/1 OpenFWI jobs survived both windows), catches setsid-detached
  processes, matches "Terminated" (SIGTERM/SIGKILL to device holders), and lives in an
  interactive command line, which is why the NaCl session's repo-wide grep (correctly)
  found no kill in any script. No direct observation of the signal; logged at that strength.
  **Resolution:** by 01:2x both sibling processes were gone (stopped by the sibling or the
  user); GPUs 2 and 3 clear; the stability watcher proceeds.

## Measured facts worth keeping in view

* MC barostat scales massless carbons **individually** (cages are not molecules to it);
  the NPT script's per-ps cage projector is the remedy — do not reuse the barostat on
  fixed solutes without it.
* OpenMM State forces: parents redistributed, M row keeps the raw virtual-site force.
* The pip OpenMM wheel needs `openmm-cuda-13` plus `LD_LIBRARY_PATH` pointing at
  `site-packages/nvidia/cu13/lib` for the CUDA platform.
* Reference-platform eval at this box: 0.3 s — the parity oracle is affordable.
* Clash-inflated configurations (~1e9 kJ/mol) make FD tests ulp-limited; minimize first.
