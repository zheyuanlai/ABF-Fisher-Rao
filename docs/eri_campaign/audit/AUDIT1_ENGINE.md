# AUDIT 1: ABF/FR engine (`src/eri/engine.py`, `scripts/eri_campaign/run_jobs.py`)

*Independent adversarial audit, 2026-10-16. CPU only. No existing file was modified, nothing was committed and no GPU
was used. The scratch scripts are in
`/tmp/claude-1008/-home-zheyuanlai-ABF-Fisher-Rao/d4317313-dc1c-4e55-8b95-fc34dea3264c/scratchpad/audit_engine/`,
with an isolated `NUMBA_CACHE_DIR` so the repository's numba cache was never touched.*

## Bottom line

The engine **implements its own docstring exactly**. An independent pure-Python re-implementation of the documented
step (`ref_impl.py`) matches `_run_chunk` on every counter, and on positions to ≤ 1e-10:

* forces at pre-move positions;
* all deposits before the read;
* own-bin Γ including this step's deposit, with the global ramp × per-bin trust;
* `+b·π/P` on the x component of both beads of molecule 0 only;
* the EM move with slot-ordered noise;
* FR at post-move positions when `(nxt − fr_start) % fr_every == 0`, through the imported `_fr_score` /
  `fr_select_native` with LTA-identical `grid0`, `qv`, `lq` and bin convention;
* whole-replica copies with label inheritance;
* windowed-ancestor reset after the save;
* the saves.

It was checked on three configurations (FR M = 2, FR M = 1, ABF M = 1, N = 24, 1500 steps, 1500 K) and on two
high-T runs that exercise translocations (6000 K, N = 16: 3 and 7 translocations, 436 and 708 crossings). In every
run the following matched the reference **exactly**: q, C, M (to 3e-11), BH, DH, anc, row, lastcage, window
crossings, translocations, first-window step, first-translocation step, replacements, opportunities, NU/NUW,
ESS/N and windowed ESS/N.

Further checks:

* **Mean-force sign (finite difference):** |f − dU/dφ| / |dU/dφ| ≤ 9e-10. dφ/dx_bead = 0.473488, which equals the
  engine's π/P.
* **FR uniforms:** consumption is independent of events. The rate 0 and rate 5000 arms end with identical FR and
  noise RNG states.
* **Checkpoint/resume:** resuming through `run()` mid-run (with a different chunk size) reproduces **every** result
  field bitwise. The fields not checkpointed (`F`, `Kk`, `fp`, `ip`, `save_steps`) are all derived.
* **Force-evaluation accounting:** `n_force_evals` = N(n_steps+1) in all 24 screen outputs, and
  `n_equil_force_evals` = N × equil_steps.
* **Overflow and float32:** no integer overflow and no float32 anywhere.
* **Tests:** the 7 engine tests pass.

The defects found are therefore NOT in the step algorithm. They are in **provenance and reproducibility** (two
demonstrated, medium), in the **power of the planned FR stage** (medium, computed from the screen's own data), and
in robustness and test-coverage items (low/info).

## Findings

### E1 (medium): a numba cache across modules can silently run an OLD force field / FR machinery
`_run_chunk` is `@njit(cache=True)` (engine.py:131). It calls `model.config_energy_force` (another file) and
`lta_ladder_numba._fr_score` / `fr_select_native` (another file). Numba invalidates a cached function only when its
**own** source file changes.

**Demonstrated** (`cache_probe.py`, on a scratch copy of `src/`):
1. Run once.
2. Edit the copy's `model.py` so that every bead x-force gets +100 kJ/mol/Å.
3. Run again in a new process.

The engine's deposits are unchanged (sum −4.431066), while `model.energy_force` (recompiled) gives −849.2255; the
expected shift is −844.8. **The engine kept the old force field silently, and its sha (engine.py only) is
unchanged.**

**Status of the existing results:** not affected. `model.py` was last modified at 15:38:47, the engine cache was
built at 16:22:43, and `lta_ladder_numba.py` dates from 2026-10-10.

**Consequence:** any later model or FR-machinery fix (or regression) is invisible to production runs while Gate-0
tests and references (which call `model.*` directly) see it. The engine and its validation would then disagree, with
no signal.

**Fix:** do one of these:
* drop `cache=True` on `_run_chunk` (compile is about 2 s);
* key `NUMBA_CACHE_DIR` on sha256(engine.py + model.py + lta_ladder_numba.py);
* at start-up, compare one engine deposit against `model.local_mean_force(model.energy_force(q))` and abort on any
  mismatch.

### E2 (medium): the recorded engine sha is that of the file on disk at save time, not of the code that ran, and it covers engine.py only
`_engine_sha256()` (engine.py:82) re-reads `__file__` whenever `checkpoint()` or `result()` is called.

**Demonstrated** (`sha_test.py`, scratch copy): the code that ran had sha 37a91959…. After an edit during the run,
the result records fd7439fa…. `model.py`, `lta_ladder_numba.py` and `cache/eri/framework_gard_tait.npz` are never
hashed.

`run_jobs.py` uses `maxtasksperchild=8`, so recycled workers re-import whatever is on disk. One launch spanning an
edit therefore mixes code versions with identical recorded shas.

**Fix:**
* compute the hashes once at import, as module constants, for engine.py, model.py, framework.py,
  lta_ladder_numba.py and the framework npz;
* store all of them in the checkpoint and the result;
* make `restore()` compare the import-time hashes.

### E3 (medium, power): on the frozen FR rate ladder {0.02–0.2}/t.u. FR is nearly inert and mostly noise-driven, and the A1.5 safety criterion cannot bind
**Late-time score at the noise floor.** From the ABF-only screen's own walker histograms (`fr_power.py`,
`fr_noise.py`, `fr_early.py`), the late-time (t ≥ 2 t.u.) mean S⁺ is 0.156 / 0.097 / 0.054 for N = 64 / 256 / 1024.
For **N iid draws from the uniform target itself** it is 0.152 / 0.093 / 0.050. So FR events after 2 t.u. are
essentially finite-N KDE noise, with no marginal deficit to correct.

**When there is a systematic signal.** Only at t ≈ 1–1.5 t.u. (S⁺ 0.19 vs 0.05 at N = 1024), which is before the
default `fr_start_steps` = 10 000 = 2 t.u. At t = 3–5 t.u. the window is over-populated (0.19–0.20 vs a uniform
0.159), so uniform-target FR would then remove window walkers.

**Activity on the ladder.**
* Replacements per walker over T = 40 are 0.04–0.43 at N = 1024.
* Measured in the strongest-signal first t.u. at N = 128 (`ess_rate.py`):

  | rate (/t.u.) | replacements per walker | windowed ESS/N |
  |---|---|---|
  | 0.2 | 0.027 | 0.93–0.96 |
  | 2 | 0.32 | 0.45–0.59 |
  | 20 | 2.5 | 0.02 |

  The ESS/N ≥ 0.30 rule can only discriminate at rates of order 2–20, about 10–100× above the ladder.

**Consequence:** with no sham arm, any FR-vs-ABF difference at these rates is resampling noise, not marginal
correction, and cannot be attributed. The safety calibration is a formality on this ladder.

**Fix:** before any FR run, do all of these:
* report the FR score-to-noise-floor ratio as a mechanism diagnostic;
* define the rate ladder by the expected replacement fraction per t.u. (e.g. 0.05–1 per walker per t.u.);
* tie `fr_start` to the period with signal.

### E4 (low): step-denominated FR/genealogy knobs are not frozen, and their physical meaning changed with dt
`DEFAULT_CFG` (engine.py:59–69) assumes dt = 1e-4. At the production dt = 2e-4:
* `fr_start_steps` 10 000 means 2 t.u.;
* the default `genealogy_window_steps` 20 000 means 4 t.u., which matches LTA's `genealogy_window_tu` = 4.0;
* every job file so far uses 10 000, i.e. 2 t.u.

The A1.5 criterion "windowed ancestor ESS/N ≥ 0.30" depends on the window length, and a shorter window passes more
easily. **Fix:** freeze them in t.u. in the FR amendment and convert in `make_cfg`.

### E5 (low): `run_jobs.py` robustness
* **One failure kills the whole batch.** Any exception in one job (e.g. `restore()`'s "engine changed since the
  checkpoint" or "signature mismatch") propagates through `imap_unordered`. `Pool.__exit__` then terminates the pool,
  which kills in-flight siblings and abandons the queue. Toy demonstration in `pool_toy.py`: the siblings never
  finished.
* **No lock against duplicate launches.** Two launches of the same JOBS.json both run a job and write the same
  `out.ckpt.npz.tmp.npz` / `out.tmp.npz`, which can corrupt the checkpoint.

**Fix:**
* catch exceptions in `job()` and return an error status;
* take an O_EXCL lock file per job;
* use per-PID tmp names;
* on a signature mismatch, move the stale checkpoint aside.

### E6 (low): the pairing test does not test what its name says
`test_arms_bitwise_identical_before_fr_start…` calls `advance(fr_start)`, which already executes the first FR
opportunity (NOPP = 1). It passes only because that opportunity happened to have REPL = 0. Nothing asserts that the
arms DIFFER after the first event.

**Fix:**
* compare at `advance(fr_start − 1)` and assert NOPP == 0;
* add a check that the arms diverge after the first event;
* add `ref_impl.py`-style counter reconstruction as a regression test.

### E7 (info): smaller items
* **`evwin` is dead state.** It is maintained, inherited and checkpointed but never reported. LTA reported
  ever-visited-window counts as V_EVWIN.
* **Resumed runs under-report wall time.** `wall_seconds_session` covers only the last session after a resume, so a
  wall-clock comparison undercounts resumed runs. Each resume also re-runs `initial_conditions` and `_equilibrate`
  (N × equil_steps force evaluations, 25.6 M at N = 1024) before `restore()` overwrites q. The result is unchanged;
  only compute is wasted.
* **The pilot ran on a superseded engine.** `pilot_abf/*.npz` has engine sha 5a0c066b…, which is not the current
  37a91959…: it predates `equil_steps` and has no `n_equil_force_evals`. The 16:13 log reading came from that
  version. Do not pool the pilot with later runs.
* **Seed reuse would break freshness.** `SeedSequence([seed, N, M, T, variant])` excludes dt, warm-up, trust,
  equil_steps, n_grid, rate and method. Every configuration with the same seed therefore shares all streams (common
  random numbers). An FR block that reused a screen seed (930000+s) would make its ABF arm bitwise equal to the run
  that classified the cell. Assert that the seeds are fresh in the job builder.
* **Equilibration contaminates first-window entry.** `S_FIRSTWIN` = 0 in 2 of 8 N = 256 screen seeds, because a
  walker is already in the window after the 25 000-step unbiased pre-equilibration. The secondary statistic
  "first-window entry" is contaminated by that phase.
* **A docstring overstates determinism.** The `model.py` docstring says the sums use fastmath "'reassoc' only", but
  `_type_sum` is `fastmath={"reassoc","contract"}` (FMA allowed). Bitwise pairing therefore holds per host and
  compile target only.
