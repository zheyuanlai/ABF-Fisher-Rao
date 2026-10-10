# Entropic gateway sampler (`src/gateway_numba.py`): porting, diagnostics and checkpoint spec

Repo at HEAD 7bbbe23. `src/gateway_numba.py` has one commit (f4a847b), and its file date is 2026-10-04 06:30.

**Bottom line**
- Every diagnostic you listed can be recorded without any RNG draw or arithmetic change.
- Checkpoint/restart at any step boundary is exact if numba's own random-number state is saved and restored.
- I checked both on CPU. A scratch copy of `simulate` with the diagnostics added, split at arbitrary steps with the RNG state restored, gave **bitwise-identical** output to the original in all 4 configurations tested.

## 1. Potential, domain, boundaries, forces, reference

**Potential.** V(x,y) = H (x²−1)² + ½ ω(x)² y².
- ω(x) = ω_out + (ω_in − ω_out) exp(−x²/(2s²)) (`gateway_numba.py:393-394`; torch `eb_abffr_core.py:154-155`).
- The numba engine takes ω_in directly (`oin`). There is no `r` parameter there.
  - In torch, `GatewayConfig.omega_in = omega_out * r` (`gateway_core.py:132-133`).
  - In `design.json:3`, `"r": 32.0` is informational only.

**Frozen cell** (`design.json:3`): β = 16, H = 0.5 (βH = 8 kT), ω_out = 1, ω_in = 32, s = 0.1, dt = 4e-4.
- ω(0) = 32, ω(0.2) = 5.20, ω(0.3) = 1.344, ω(0.4) = 1.0104, ω(0.5) = 1.0001.
- Barrier βΔF ≈ 8 + ln 32 = 11.47 kT.

**Domain.** XMIN, XMAX = −1.8, 1.8 (`gateway_numba.py:49`, `eb_abffr_core.py:51`).
- Grid: N_GRID = 181, dx = 0.020000000000000018 (from linspace).
- y is unbounded. YMIN/YMAX at `eb_abffr_core.py:52` are informational only.

**x-boundary** (`_reflect`, `gateway_numba.py:81-87`):
```
span = hi - lo;  qm = (q - lo) % (2*span)   # float %, result in [0, 2*span)
if qm > span: qm = 2*span - qm
return qm + lo
```
This is identical to torch `reflect_into` (`eb_abffr_core.py:251-255`). It is applied to x only, after the full Euler–Maruyama update (EM below means this Euler–Maruyama integrator).

**Forces** (`gateway_numba.py:393-398`):
- ω'(x) = −(ω_in−ω_out)(x/s²) exp(−x²/2s²)
- f_x = 4Hx(x²−1) + ω ω' y²
- f_y = ω² y

**Update** (overdamped, unit mobility, kT = 1/β; `:361`, `:415-416`), fully explicit (old x and y on the right-hand side):
- x ← reflect(x + (−f_x + Γ_j) dt + √(2dt/β) ξ_x)
- y ← y − f_y dt + √(2dt/β) ξ_y

**Analytic reference** (`eb_abffr_core.py:170-184`):
- `F_ref = H(x²−1)² + log(ω)/β` (`:178`), then minus its mean over the eval-window nodes (`:179`). That centring is the only additive constant.
- `Fp_ref = 4Hx(x²−1) + ω'/(ω β)` (`:180`). `reference_mean_force` (`:273-277`) is the same expression at arbitrary x.
- Both equal H(x²−1)² + β⁻¹ log ω(x) + C and its derivative exactly.

**Eval window.** EVAL_LO, EVAL_HI = −1.5, 1.5 (`eb_abffr_core.py:54`). That is 151 grid nodes from −1.5 to 1.5. The gauge node is idx0 = 90 (x ≈ 6.9e-18).

**How e_F is scored.**
- Exact integral of the piecewise-constant Γ at the nodes (`HistogramABFEstimator.pmf_profile`, `eb_abffr_core.py:384-390`).
- Gauged at idx0, centred on the window, RMS against the centred F_ref (`l2_error`, `:267-270`).
- e_F′ is a Gauss–Legendre function-space RMS with 8 points per bin, inside the window (`fp_error`, `:392-398`).
- The analyzer hard-codes min_count = 1 (`analyze_gateway_replica_ladder.py:68`).

**Reference the EM integrator actually converges to** (post-hoc; `analyze_gateway_replica_ladder.py:35-46`):
- ⟨f|x⟩ = 4Hx(x²−1) + ω'/(βω(1−ω²dt/2)).
- The factor is 1.2575 at the gate at dt 4e-4. The analytic F floor is 0.00138 RMS (`docs/GATEWAY_REPLICA_LADDER.md:63-79`).

## 2. `simulate()` step order (`gateway_numba.py:329-465`)

```
setup: X[a,:]=x0, Y[a,:]=y0 for every arm (:338-341); Mh=Ch=0 (A,max_bins) (:342); anc empty (:343)
np.random.seed(noise_seed)                     (:360)  numba thread-local MT19937 (numba_rnd_init, uint32)
amp=sqrt(2dt/beta); dt_fr=dt*fr_every          (:361-362)
for step in range(n_steps):                    (:367)
  if ess_window>0 and step%ess_window==0: anc[a,i]=i   (:368-371)  [diagnostic only]
  zx[i]=std_normal(), i<N; then zy[i]=std_normal(), i<N (:377-381)  shared by ALL arms
  do_fr = step % fr_every == 0                 (:382)   no burn-in
  g = gamma*(1-exp(-step/ramp_steps)) if ramp_steps>0 else gamma  (:383-386); g=0 at step 0
  for a in arm order:                          (:387)
    delta = 3.6/nb_a
    DEPOSIT (all walkers first) (:391-406): f,fy from pre-move (x,y); j=clamp(floor((x+1.8)/delta+1e-9),0,nb-1);
           Ch[a,j]+=1; Mh[a,j]+=f; jb[i]=j
    MOVE (:409-416): Γ = Mh[a,jb[i]]/(Ch[a,jb[i]]+min_count) (0 if den<=0) -- OWN bin of the pre-move x,
           after this step's deposits by every walker incl. itself; piecewise constant, no interpolation;
           track max|Γ| (:413-414,417); x,y EM update + reflect
    FR if do_fr and arm_fr[a]==1 and N>=2 (:419), on post-move X[a]:
       p = kde_density(X[a]) (:421); S = uniform_scores (:423) [arm_score 1/2 = exploratory centred_scores]
       kd,kc = fr_resample(S,N,g,dt_fr,cap,...) (:426)
       if kd+kc>0: X,Y,anc[a,:] <- [a, sel] (gather); tot_die+=kd; tot_clone+=kc (:427-433)
  if do_fr: n_fr += 1   (:434-435)  counted even when no FR arm exists
  while ptr<n_saves and save_at[ptr]==step+1: SAVE (:437-464)
```

**ABF estimator.** Histogram P0 with [left, right) bins and the 1e-9 nudge. Γ_j = M_j/(C_j + min_count), with min_count = 1 (`design.json:12`). This matches `eb.HistogramABFEstimator` (`eb_abffr_core.py:319-324`, `:366-378`).

**FR knobs** (`design.json:13`): γ = 1.5, η = 0.1, fr_every = 10 (so dt_fr = 0.004), clip = 3, ramp_steps = 10000, cap = max(cap_min, ⌊0.08 N⌋) (`gateway_numba.py:488`).
- cap_min = 1 in production and 0 in the validation stage.
- Resulting caps: 1 for N ≤ 24, 2 at N = 32, 5 at 64, …, 163 at 2048 (measured from the production metadata).

**KDE** (`kde_density`, `:90-143`):
1. Nearest-node bin k = rint((x+1.8)/dx), clamped to [0, 180].
2. Gaussian kernel exp(−½(m dx/η)²)/(Σ·dx) with m ∈ [−20, 20] (r = round(4η/dx) = 20).
3. Reflect-padded: images at −k and 360−k, as torch 'reflect' does.
4. Divide by N, divide by the trapezoid mass (floored at EPS = 1e-30), then floor each value at EPS.

**Uniform score** (`uniform_scores`, `:146-177`):
- S_i = log max(interp(p̂, x_i), EPS) − log q − KL(p̂‖q), with q = 1/(dx·180) = 1/3.6 and KL by trapezoid.
- Linear interpolation, clamped at the grid ends. Then clipped to ±3.
- "Centring" is the grid KL, not the empirical mean. Empirical centring is exploratory mode 1.

**Death/clone law** (`fr_resample`, `:251-326`):
1. For each walker i in index order, draw u_i = U[0,1) (`:258`).
2. If S > 0, it is a death candidate when u < 1 − e^{−g S dt_fr}. If S < 0, it is a clone candidate when u < 1 − e^{g S dt_fr}. If S = 0, nothing happens.
3. Cap (`:270-281`): if nd + nc > cap, then kd = min(rint_half_even(cap·nd/(nd+nc)), nd) and kc = min(cap − kd, nc).
4. Return (0, 0) immediately if kd + kc = 0 (`:282-283`). **`sel` is then stale**, which is why the gather is gated.
5. Partial Fisher–Yates picks kd deaths and kc clones uniformly (`:285-294`).
6. Pool = survivors in index order + chosen clones (`:296-308`). P = N − kd + kc.
   - If P ≥ N: a uniform N-subset via partial Fisher–Yates, which scrambles all slot order even when kc = kd (`:309-317`).
   - If P < N: sel = pool, then N − P uniform survivors drawn **with replacement** (`:318-325`).
7. Consequence: tot_die and tot_clone are *capped candidate* counts, not realised changes.
   - Realised births always equal realised deaths (N is constant).
   - In my check (N = 2048, 4000 steps, fr_h180), kd = 206 and kc = 143, but the realised deaths (slots with zero offspring) were 289.

**Genealogy.**
- `anc` is reset at every multiple of ess_window = 4000 steps (`run_ladder_point` default at `:479`; hard-coded at `run_gateway_replica_ladder.py:79`). It is gathered with `sel` and read only at saves.
- `out_ess = N/Σcnt²` is **normalised ESS/N** (`:452`). Torch `ancestor_stats` returns absolute ESS (`gateway_core.py:713`).
- `out_wmax` = largest family / N (`:453`).
- The window length at a save is ((save_at−1) mod 4000) + 1 steps. At N = 2048 it varies over 500…4000 steps between saves (measured), so ESS is a save-phase sawtooth.

**Saves.**
- `save_at` = completed-step counts. It must be sorted; a value that is out of order or > n_steps silently never fires.
- `budget_save_grid` (`:468-475`): u = k/200 (k = 1..200) plus 24 log-spaced u in [1e-4, 5e-3), giving step = round(u·n_steps). That is 224 saves.
- What each save records per arm (`:437-464`):
  - out_M and out_C (cumulative bin sums; columns ≥ nb are zero)
  - ess, wmax
  - out_P: instantaneous left / gate / right fractions of the post-move, post-FR X, using X_BASIN = 0.5
  - cumulative kd and kc
- Also returned: max|Γ| per arm, final X and Y, n_fr (`:465`).
- The production runner writes all of these to an npz (`run_gateway_replica_ladder.py:56-59`). The dt runner omits wmax, P, maxbias, X_final and Y_final (`run_gateway_dt.py:44-45`).

## 3. RNG and reproducibility

- **One stream for everything.** Langevin normals and FR uniforms all come from numba's thread-local `np.random` MT19937.
  - It is seeded once per call (`:360`) with noise_seed = (1_000_003·seed + 7919·N) mod (2³¹−1) (`:493`). The validation stage adds +17 (`run_gateway_replica_ladder.py:74`).
  - This generator is separate from numpy's (`:505-508`).
  - Draw order per step: N normals (zx), N normals (zy), then for each FR arm in arm order at FR steps: N + kd + kc + (N if P ≥ N, else N − P) uniforms.
  - The step-0 opportunity (g = 0) still consumes N uniforms per FR arm.
  - Normals use the polar Box–Muller method with a cached second value (`numba/cpython/randomimpl.py:242-292`).
- **Slot-by-slot sharing.** Every arm gets the same (zx, zy) at each step, applied by slot index; after any FR gather, slot ≠ walker. But the stream position depends on all FR arms' draws. Torch, by contrast, uses separate `gen_n`/`gen_f` (`gateway_core.py:884-885`). Measured at N = 64 (the reorder test at 30000 steps):
  - ABF_h180 in a 4-arm job ≠ ABF_h180 run alone with the same noise_seed.
  - A job with two ABF arms equals the ABF arm run alone.
  - Changing only FR γ changes the ABF arm.
  - Reordering the arms changes both the ABF and the FR arm.
  - This is not listed in the module's "deliberate differences" (`:28-36`). `exploratory_score_fix.json:8` does acknowledge it.
  - INFERENCE: it is statistically harmless (an iid stream skipped by a past-dependent amount stays iid), but it is not bitwise-separable.
- **What sets the bits:** noise_seed, N, init seed, the arm list *and its order*, every cell and FR parameter, cap, the kernel, and min_count.
- **What does not change the bits** (all verified):
  - n_steps: a 3000-step run is an exact prefix of a 6000-step run.
  - ess_window and save_at: they are read-only.
- **Same-host reproducibility.** The dt = 4e-4 rerun (2026-10-09) of N2048_s7100 is bitwise equal (M, C, die) to the 2026-10-04 ladder file.
- INFERENCE: other machines or glibc versions could differ in exp/log. There is no fastmath and no `parallel` in the module.
- **Checkpoint state, exact.**
  - step (global: drives the anc reset, do_fr and g)
  - X, Y, Mh, Ch, anc (A×N or A×max_bins, float64/int64)
  - out_maxbias, tot_die, tot_clone, ptr, n_fr
  - the save arrays filled so far
  - RNG: `numba._helperlib.rnd_get_state(numba._helperlib.rnd_get_np_state_ptr())` returns (index, 624 uint32). This is a private API, and it is thread-local, so call it in the simulating thread.
  - Restore with `rnd_set_state(ptr, state)`, which forces has_gauss = 0. That is exact at step boundaries, because 2N normals per step (an even count) leave the gauss cache empty and uniforms never touch it (`numba/_random.c:293-318`).
  - Do **not** reseed on resume. A forked worker reinitialises its numba state from urandom, so restore immediately before calling (`_random.c:161-166`).
  - Scratch buffers need not be saved: all are overwritten before they are read.

## 4. `init_left` (`gateway_numba.py:68-78`)

1. rng = `np.random.default_rng(1000+seed)` (PCG64). This is the same stream as torch `init_conditions` (`gateway_core.py:517-543`); test V0 checks agreement to atol 1e-14.
2. x = rng.normal(−1, 0.05, N), then reflected into [−1.8, 1.8] (a no-op in practice).
3. z = rng.normal(0, 1, N), drawn *after* all N x values.
4. y = z / √(β ω(x)²): the continuum conditional, not the EM-stationary one.

Properties:
- x0 is a prefix in N; y0 is not (both verified).
- It is identical for every arm and independent of dt and noise_seed.
- Seed 7100, N = 1: x0 = −0.97466225, y0 = 0.01275033.

## 5. Diagnostics: present, missing, and where to record

| Diagnostic | Today | Where to record without changing dynamics |
|---|---|---|
| Instantaneous x-histogram per save | Missing (only 3-region `out_P`) | Save block `:437-464`, from `X[a]` (same state as `out_P`). Use the [left, right) rule with the 1e-9 nudge on the arm's bins or a fixed grid |
| Cumulative visitation | **Already `out_C`**: pre-move positions, every walker-step, at ABF-bin resolution | For another grid: in the deposit loop after `:404`, reading `x` |
| Region occupancy | Instantaneous `out_P` at saves, boundaries X_BASIN = 0.5 (`gateway_numba.py:52`, `gateway_core.py:77`) | **Time-averaged** occupancy is exact from the 180-bin `out_C` increments: bins 0–64 (x < −0.5), 65–114 (gate), 115–179 (x ≥ 0.5); differs only at x = +0.5 exactly. Optional report-only core \|x\| < 0.35 (FLANK_DIAG, `gateway_core.py:306`) |
| First visit to right well | Missing. Saves are too sparse: at N = 1 the linear saves are 1.024e6 steps = 410 time units apart | Move loop, right after `:415`: if `X[a,i] > X_BASIN` and the flag is unset, set first_right = step+1 |
| Cross-well transitions | Missing | Per-walker label with hysteresis (−1 if x < −0.5, +1 if x > 0.5), updated after `:415` (count a flip at update). **Gather the label with `sel` at `:429-431`** so clones inherit it. Initialise from x0 (all left) |
| Distinct ancestors / ancestor ESS / largest family | Windowed only (save-phase-dependent window) | Count `cnt>0` in the save block. For whole-run genealogy, add `anc0` (never reset), gathered with `sel` at `:429-431` |
| Realised deaths/births per save | Missing (`die`/`clone` are cumulative capped kd/kc) | Inside `if kd+kc>0` (`:427`; never read `sel` outside it): offspring counts from `sel`; deaths = #zero, births = Σmax(c−1, 0) = deaths |
| Walker traces | Missing (final X, Y only). Slot order is scrambled by every P ≥ N event | Persistent ids gathered with `sel`: first copy keeps the id, later copies get next_id++ (deterministic, no RNG). Record X, Y, id for selected ids after the arm loop (after `:433`) on a physical-time cadence |

**Verification.** All of the above was implemented in a scratch copy of `simulate` that can be chunked and resumed. It was compared against `gn.run_ladder_point` with the RNG deliberately scrambled between chunks and then restored. All 10 original outputs plus n_fr matched **bitwise** in every case:
- N = 16, 60000 steps, cut at 7777 and 31111
- N = 2048, 4000 steps
- N = 1, 200000 steps
- N = 3, 50000 steps (cut at 10, an FR step)

Diagnostic overhead was +6 to +10 % wall time.

## 6. Cost, threading, N = 1

**Threading.**
- `@njit(cache=True)` with no `parallel`/`prange`: single-threaded. The runners set NUMBA_NUM_THREADS = 1 (`run_gateway_replica_ladder.py:21`, `run_gateway_dt.py:6-7`).
- Parallelism comes from one process per (N, seed) job.

**Measured** (one core, shared host, load ~12/256):

| Configuration | ns per arm-walker-step | µs per step |
|---|---|---|
| N = 2048, 4 arms | 34.0 | 278 |
| N = 2048, 1 ABF arm | 75 | 154 |
| N = 2048, 1 FR arm | 78 | 159 |
| N = 16, 4 arms | 40.9 | — |
| N = 2, 4 arms | 82.7 | — |
| N = 1, 2 ABF arms | 68 | — |
| N = 1, 1 ABF arm | 110 | — |

INFERENCE from the 1- vs 2-arm difference at N = 2048:
- The shared noise costs about 117 µs per step (≈ 28 ns per normal).
- Each ABF arm costs about 18 ns per walker-step.
- FR adds about 3 ns per walker-step (≈ 58 µs per opportunity).

**Production job wall times** (median, 100 concurrent processes): 28.4 s at N = 2048 up to 70.8 s at N = 2 (4 arms); 33.2 s at N = 1 (2 arms).

**N = 1.**
- FR arms are dropped by the runner (`run_gateway_replica_ladder.py:115`) and skipped in the engine by `N >= 2` (`:419`), so no uniforms are drawn.
- n_fr still counts 20,480,000 opportunities.
- ESS = wmax = 1.

## 7. What a time-step change touches

**Physical-time knobs counted in steps** that must be rescaled. The dt runner rescales fr_every, ramp_steps and ess_window (`run_gateway_dt.py:20-22`, `:35`):
- **fr_every** sets both the schedule (`:382`) and the hazard dt_fr (`:362`). It must stay an integer: `int(round(10·4e-4/dt))`.
- **ramp_steps**: 4 time units.
- **ess_window**: 1.6 time units; it is hard-coded to 4000 in `run_with_save_at`.
- The save grid is in budget fractions, so it is invariant.
- **cap** is per opportunity, so it is preserved only if fr_every is rescaled.

**Not rescaled, and not dt-invariant:**
- **min_count = 1** is in samples. Counts per unit time scale as 1/dt, and the bias includes the walker's own current sample (`:405-412`). INFERENCE: this changes the early, low-count transient.
- **noise_seed** has no dt term, so runs at different dt are not pathwise coupled.

**Integrator:**
- The EM y-update needs ω_in²dt < 2, i.e. dt < 1.95e-3. At 4e-4, ω_in²dt = 0.41.
- The EM bias factor 1/(1−ω²dt/2) changes the floor from 0.00138 (dt 4e-4) to 0.00035 (1e-4) and 0.00009 (2.5e-5) (`gateway_dt_prereg.json:11`). The `em` reference reads `cell["dt"]`.
- y0 uses the continuum variance (negligible in the wells).

**Budget:** at fixed B = N·n_steps, T = B·dt/N moves with dt. Step-indexed diagnostics need t = step·dt.

## Files

**Scratch** (under `/tmp/claude-1008/-home-zheyuanlai-ABF-Fisher-Rao/b25e8a63-d0d8-4935-af2b-7d50d5f3ea3b/scratchpad/audit/`):
- `gw_diag.py`: the reference implementation of the chunked simulate with diagnostics and RNG get/set.
- `check_bitwise.py`, `check_stream.py`, `check_reorder.py`, `timing.py`, `timing_diag.py`.
- `tests/test_gateway_numba.py`: 15 passed on CPU.

**Repo side effects.** No tracked file was modified.
- My `python -I` run ignored `PYTHONDONTWRITEBYTECODE` and probably refreshed the gitignored `src/__pycache__/gateway_numba.cpython-314.pyc` (04:18:20).
- My numba caches went to the scratch directory.
- Another session is writing `src/gateway_validation.py` and `tests/test_gateway_validation.py`; its numba caches in `src/__pycache__` (04:16:50) predate my run.