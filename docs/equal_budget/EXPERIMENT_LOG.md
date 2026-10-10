# Equal-budget campaign: experiment log and resource ledger

All times are UTC, 2026-10-10. Ceilings: 1000 CPU core-hours and 24 GPU device-hours. No GPU was used: GPUs 0–1
carry another user's jobs, and the CPU engines made GPUs unnecessary.

## Branch decision

The task text asked for a dedicated branch. On 2026-10-10 the user had explicitly said: *"dont use the new
branch, just use the `main` branch and push to github `main`"*. All work is therefore committed on `main`.

## Timeline

Commit times are from `git log`, and run times from the per-run ledgers. Entries marked "~" are reconstructed
from file times.

| time (UTC) | event |
|---|---|
| ~04:12 | Resources checked: load ~18/256; GPUs 0–1 busy with another user, 2–3 idle. Phase 0A audit launched (3 readers: gateway, LTA, inventory), giving `audit/AUDIT_0A_*.md` |
| ~04:2x | Gateway validation harness and 6 tests written. Latent `gateway_numba.simulate` bug found: `ess_window ≤ 0` leaves the ancestor labels uninitialised and segfaults. Not used in production |
| 04:28 | Gateway timestep gate frozen before any validation run (e58edca) |
| ~04:29–04:37 | Gateway validation runs, 67 jobs, 3.5 core-h |
| 04:38 | **h_G = 2.5e-5 PASS**; 4e-4 and 1e-4 fail the 2 % fibre-variance gate (f917c49) |
| ~04:4x | Engine workflow launched: 2 implementers, 6 reviewers, 2 fixers |
| 04:54 | Scientific plan, production configs and frozen references committed: the **preregistration** (ea14f11) |
| ~06:0x | Analysis-pipeline workflow launched: builder, reviewers, fixer |
| 06:24 | Engines pass, 76 tests + 2 slow (65af1be) |
| ~06:25 | LTA smoke test under live ABF bias at both T: PASS |
| 06:28 | LTA timestep validation verified, h_L = 2e-4 frozen (badfbab). Production chain launched: LTA 300 K (coarse N first), then LTA 150 K, then gateway; 110 single-threaded workers on CPU |
| 06:33–06:47 | LTA 300 K production: 336/336 runs complete, 0 failed |
| 06:47–07:06 | LTA 150 K production (starts as LTA 300 K ends): 336/336 complete, 0 failed; arms finished 06:52:07–07:06:34 |
| before 07:06 | Integration incident during the analysis pipeline: an audit test wrote into the tracked `tests/fixtures/`, and the directory was then removed with `rm -rf`. The 9 tracked fixture files were **restored from git** (`git checkout -- tests/fixtures`) and verified. Test fixture paths moved to `tempfile.gettempdir()/eqb_fixtures`. No result file was affected |
| 07:06 | Analysis pipeline committed (55a4e78): 22 review findings fixed, including the unreachable gateway strict e_F′ threshold (P0 read-out floor 0.0323) and the establishment null calibration. Gateway production starts |
| 07:06–07:12 | LTA 300 K analysis and figures (20 per N × 11 N, png + pdf). Reviewed by eye: N = 1024 F-summary |
| 07:15 | LTA 300 K milestone committed and pushed (3798994) |
| 07:14–07:21 | LTA 150 K analysis, figures and synthesis; LTA 300 K synthesis |
| 07:21 | LTA 150 K milestone committed and pushed (a3b588a) |
| ~07:2x | Finite-N mechanism tests (`scripts/equal_budget/mechanism_analysis.py`). LTA completeness audit trial: PASS |
| 07:06–07:38 | Gateway production: 736/736 complete, 0 failed (12 N × 32 seeds; about 260–270 s per arm at most N (median 264 s over all 736 arms); the FR arm is slower at small N (median 291 s at N = 2, 278 s at N = 4)) |
| 07:38–08:07 | **Idle gap, my error.** The background waiter meant to signal the end of the gateway run tested for a last log line of exactly `done`, but the launcher appends `[…] chain done` after it. The condition could never match, and the analysis did not start until the user asked about progress at 08:07. No data was affected; only wall time was lost |
| 08:08 | Gateway analysis, 29 s |
| 08:08–08:09 | Gateway per-N figures (12 N rendered in parallel, 44 s) |
| ~08:10 | Synthesis S1–S8 for all systems plus cross-system X1. **Completeness audit `--system all`: PASS**, 0 failures. Gateway warnings: 2 unreachable-by-construction strict e_F′ thresholds, and the gateway reference file's F′ units are implicit (documented in FINAL_RESULTS §2) |
| ~08:12 | Gateway milestone committed and pushed (10a9e75). Report drafts written; these were **not** yet in git, because `docs/*` is gitignored |
| 08:15–08:31 | Report-verification workflow: 7 verifiers, 7 confirmers and a completeness critic. **62 confirmed corrections** and 14 completeness gaps |
| 08:35 | Fresh-seed best-allocation confirmation preregistered and committed before running (de3d20c) |
| 08:35–08:49 | Confirmation runs: 256/256 complete, 0 failed, 22.1 core-h. LTA ties; gateway ABF better (+24.0 % [+1.1, +52.9]) |
| 09:42–09:49 | Production figures regenerated with the legibility-checked scripts: S8 label overlap fixed, plus 8 N = 2 layout defects caught by the new check (C2 title collision, E panel (f) legend over data). **711/711 figures pass** the legibility check, and the **completeness audit passes** for all three systems. Equivalence-run data copied from the scratchpad to `results/equal_budget_v2/equivalence/` (/tmp was 98 % full) |
| ~08:4x–09:xx | Gap-closing workflow: appendix tables, FR overhead breakdown, figure legibility check and S8 label fix, data manifest. Corrections applied; reports force-added and committed (final SHA in the closing message) |

## Resource ledger

From `scripts/equal_budget/ledger_summary.py`, which writes `results/equal_budget_v2/ledger_summary.json`. A
production run is one single-threaded process, so wall seconds are core-seconds.

| item | runs | failed | core-h | force evaluations | max RSS (MB) | finished (UTC) |
|---|---|---|---|---|---|---|
| gateway timestep validation | 67 | 0 | 3.50 | -- | -- | -- |
| LTA 300 K production | 336 | 0 | 26.23 | 1.032e+11 | 280 | 06:33:12 .. 06:47:35 |
| LTA 150 K production | 336 | 0 | 26.47 | 1.032e+11 | 335 | 06:52:07 .. 07:06:34 |
| gateway production | 736 | 0 | 54.73 | 2.412e+12 | 156 | 07:10:56 .. 07:38:26 |
| best-allocation confirmation (LTA 300 K, 150 K, gateway) | 256 | 0 | 22.12 | | | 08:39:58 .. 08:49:15 |
| **total (ledgered)** | 1 731 | 0 | **133.0** | | | |
| LTA numba-vs-production equivalence runs (engine workflow, 05:07–06:14; not in a ledger file, timed from logs) | 176 | 0 | 13.6 | | | |
| FR-overhead benchmarks (pinned short runs, plus a reviewer re-run) | ~300 short | 0 | ≈ 1.5 | | | |
| regeneration / bitwise checks (data manifest) | 7 | 0 | 0.6 | | | |
| **total of measured runs** | | | **≈ 149** | | | |

**Not in the ledgers** (bounded estimates):
* engine test suites (76 + 2 slow tests, run several times by the workflow agents): ≤ 6 core-h;
* LTA validation tests (12 tests, 7.5 min) and smoke tests: ≤ 1 core-h;
* reference building from the 2026-10-09 MC data: ≤ 0.5 core-h;
* analysis pipeline tests and fixtures: ≤ 3 core-h;
* production analysis and figures: ≤ 2 core-h.

**Total ≈ 165 CPU core-hours (≈ 149 measured plus ≤ 12.5 bounded plus analysis and figure regeneration) of the
1 000 ceiling, and 0 GPU device-hours of the 24.** Peak concurrency was 110
single-threaded workers. Load averages observed during production were 101–126 on 256 cores. No other user's process was touched.

