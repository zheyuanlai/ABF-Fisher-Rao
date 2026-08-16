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
| dt gate | **DECIDED: dt = 1 fs (2026-08-15, third read, never revisited).** Read 1 RETRACTED (NaN spots from teleport prep); read 2 invalidated (2× drag-rate jam, guard fired). Read 3 procedurally valid: prep clean (0 jammed), equipartition PASS both dt (0.03 K / 0.66 K), constraints ~2e-15 nm, spots at 1.2/2.0 ok — the 0.968 spot missed at 2.17σ vs the frozen 2σ line, so the frozen rule decides 1 fs. A survivor-composition confound (14 vs 10 clean replicas between arms) is RECORDED BUT NOT EXPLOITED: redesigning the clause after seeing the borderline number would be result-directed. Cost is wall-clock only: reference ≈ 4.9 days idle / ~6.4 co-tenant at 175 ns/day-equivalent |
| reference | **CAMPAIGN RUNNING since 2026-08-16 03:24 UTC** — pin `761a4f7` (includes two peer-session fixes: post-SD hot acceptance, >=8/16 anchor floor + rate/8 approach), collision-proof worktree `/home/zheyuanlai/.c60-run-761a4f7`, PINNED_COMMIT written from the hash (the rev-parse-in-worktree convention was tautological). Build 1 ETA ~+42 h; 3 builds + spot-check + analysis ~4.5 days at 1 fs. Ladder step 6 discharged by demonstration (anchors-resume ran live in ladder14; checkpoint load-drill armed against build 1's first checkpoint) after external worktree removal killed two ladder tails (incident log) |
| Gate 0 pools | not started |
| screen | **prohibited** until reference accepted + Gates 0/A pass |
| mFR | **prohibited** until an establishment-limited cell is frozen |

## Next permitted action

Ladder9 running (pin = `results/c60/PINNED_COMMIT`, hot rejection sampling in). On a green
ladder: launch the reference campaign from the pinned worktree (separate, reviewed action):

```
cd /home/zheyuanlai/ABF-c60-worktree && setsid nohup bash scripts/c60_reference_campaign.sh > /home/zheyuanlai/ABF-Fisher-Rao/results/c60/reference/campaign.log 2>&1 &
```

At dt = 1 fs (DECIDED): ~4.9 idle days for 3 builds + spot-check + analysis. Then review
acceptance (R_ref <= 0.5) + the Zangi reproduction gate + the 16.8 lambda table, and only
then write/run the Gate 0 pools and the N = 64 screen (drivers not yet written -- model on
scripts/nacl_gates.py with the 16.7/16.8 constants).

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

* **2026-08-15 10:04 UTC: GPUs 0–3 are NOT exclusively this group's.** Third-party users
  measured resident: `juntingwu` on GPUs 1/2/3 (17.5 GB on 3 from 10:04:19), `yesom` on 0/1.
  Ladder5's throughput stage ran co-resident and its `throughput.json` under-reports
  uniformly by ~1.32× — superseded by `throughput_idle_reference.json`, which records the
  twice-replicated idle numbers from ladders 1/2 (B816_c128: 401.7/401.8 ms/step, 351
  ns/day agg). The dt gate/smoke/resume stages are ensemble- or mechanics-based and remain
  valid co-resident (wall clock only). The reference campaign launches co-resident with
  occupancy recorded in its manifest: ~3.2 days at current contention, 2.44 idle. The
  preflight idle check (memory-based, any process) would correctly refuse today; it passed
  at 09:5x because the foreign job landed at 10:04, mid-suite. Do not treat "preflight
  passed" as "stayed idle".

* **2026-08-16 01:1x–03:1x: worktree externally removed/recreated mid-ladder, twice** — the
  guessable path `/home/zheyuanlai/ABF-c60-worktree` was recreated at the moving branch head
  by a sibling session's maintenance; ladders 13/14 died on missing files at their tails
  while their early stages (suite 12/12, smoke build 1.75 h) passed. Remedies: uniquely
  named dot-prefixed production worktree; pin written from the hash; single-launcher rule
  declared to peers. A sibling session also landed two genuine C60 fixes on the branch
  (vacuous pre-SD hot acceptance; binomial anchor floor) — adopted, with the one-owner rule
  reasserted for execution (fixes by commit+message, never by launch).

## Measured facts worth keeping in view

* MC barostat scales massless carbons **individually** (cages are not molecules to it);
  the NPT script's per-ps cage projector is the remedy — do not reuse the barostat on
  fixed solutes without it.
* OpenMM State forces: parents redistributed, M row keeps the raw virtual-site force.
* The pip OpenMM wheel needs `openmm-cuda-13` plus `LD_LIBRARY_PATH` pointing at
  `site-packages/nvidia/cu13/lib` for the CUDA platform.
* Reference-platform eval at this box: 0.3 s — the parity oracle is affordable.
* Clash-inflated configurations (~1e9 kJ/mol) make FD tests ulp-limited; minimize first.
