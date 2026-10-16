# ERI campaign: code-infrastructure map and reuse plan

*Review agent, 2026-10-16, read-only, git `main` at 5f95131e. No existing file was modified. No GPU was used.*

This file maps the code of the closed ethane/LTA study that a new ethane/ERI engine can reuse. It covers
the torch engine, the numba ladder and walker-parallel engines, the independent validation model, the
alkanes FR machinery, the CHA engine and the analysis libraries. Every claim cites the source as `file:line`.
Everything marked **estimate** is arithmetic on repository numbers, not a measurement.

---------------------------------------------------------------------------------------------------------

## 0. Test environment (measured 2026-10-16)

| interpreter | Python | numba | numpy | torch | pytest | result of `pytest -q tests/test_lta_validation.py tests/test_lta_ladder_numba.py -x` (CPU only) |
|---|---|---|---|---|---|---|
| `/home/zheyuanlai/miniconda3/envs/abffr/bin/python` (= `../miniconda3/envs/abffr`) | 3.14.4 | 0.68.0 | 2.4.6 | 2.12.0+cu130 | 9.0.3 | **43 passed, 1 skipped in 149.07 s** (wall 150.4 s), exit 0 |
| `/home/zheyuanlai/miniconda3/bin/python` (base) | 3.13.13 | 0.67.0 | 2.5.0 | 2.12.1+cu130 | 9.1.1 | **43 passed, 1 skipped in 151.38 s** (wall 153.1 s), exit 0 |

* Both runs: `CUDA_VISIBLE_DEVICES=""`, `-p no:cacheprovider`. `torch.cuda.is_available()` was False in both.
* The one skip is `tests/test_lta_ladder_numba.py:1108` ("slow: set RUN_SLOW=1").
* The base-env run used `NUMBA_CACHE_DIR=<scratchpad>` and `PYTHONDONTWRITEBYTECODE=1`. Its numba compile
  was therefore cold and wrote nothing into the repository. The abffr run reused the existing on-disk
  numba cache; `find -mmin` showed no `__pycache__` file was modified.
* **Canonical environment is `abffr`.**
  * Every launcher pins it: `scripts/lta_histogram/launch.sh:7` (`PY=/home/zheyuanlai/miniconda3/envs/abffr/bin/python`),
    `scripts/equal_budget/launch_production.sh:4` (`conda activate abffr`).
  * The runbook does too: `docs/equal_budget/DATA_LOCATIONS.md:10`.
  * The production CPU convention is `CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_CACHE_DIR=/home/zheyuanlai/.cache/numba_eqb`
    (`docs/equal_budget/DATA_LOCATIONS.md:11`; set in code at `scripts/equal_budget/run_ladder.py:20-23`).
  * The base env also passes, but numba-bitwise claims hold only per (numpy, numba, compile target).
    That tuple is part of the checkpoint signature (`src/lta_ladder_numba.py:1157-1165`), so all ERI
    production must stay in one env.
* Host: `llvmlite.get_host_cpu_name()` = `znver4`, 256 logical CPUs (`os.cpu_count()`).

---------------------------------------------------------------------------------------------------------

## 1. Module map: responsibilities, key functions, conventions

### 1.1 Units and dynamics conventions (common to every LTA engine)

* **Units.** Energy kJ/mol, length Å, `KB = 0.008314462618` kJ/mol/K (`src/lta/core_lta.py:66`;
  `src/lta_ladder_numba.py:133`; `src/lta_validation.py:31`).
* **The CV and its force units.** The CV is φ in radians. The mean force and Γ are in kJ/mol/rad
  (`scripts/equal_budget/eqb_metrics.py:376` `units=dict(F="kJ/mol", Fp="kJ/mol/rad")`).
  * z = φ a / 2π in Å (`src/lta/core_lta.py:22`).
  * 1 kJ/mol/rad corresponds to (2π/a) kJ/mol/Å, i.e. 0.527 kJ/mol/Å at a = 11.919 Å.
* **Integrator.** Overdamped Euler–Maruyama with unit mobility, `q <- q + dt (F + bias) + sqrt(2 dt / beta) eta`.
  * Code: `src/lta/core_lta.py:14-17`, `:383`, `:521-522`; numba `F_NS = sqrt(2 h / beta)` at `src/lta_ladder_numba.py:1053`
    and the move at `:784-791`.
  * The **time unit** is therefore Å²·mol/kJ (dt·[F] = Å). The diffusion constant is D = kT Å²/t.u.
    (2.494 Å²/t.u. at 300 K, 1.247 at 150 K), and the per-coordinate noise per step is √(2 kT dt) = 0.032 Å at
    300 K, dt = 2e-4.
  * The docstring states the budget axis is force evaluations, not ps (`src/lta/core_lta.py:16-17`).
  * The ladder states its knobs in t.u., e.g. "warmup_steps=20000 (4 t.u.)" (`src/lta_ladder_numba.py:193`).
* **Coordinates are never wrapped.** The minimum image is applied only to bead–O displacements
  (`src/lta/core_lta.py:158-159`); the bond needs no minimum image (`:168`).
  * Far-from-origin agreement with torch is tested (`tests/test_lta_ladder_numba.py:692`).
  * No COM removal: the framework breaks translation invariance. Compare the alkanes engine, which
    removes COM drift (`src/alkanes/core.py:231,367`).
* **Precision.** float64 everywhere in LTA (`src/lta/core_lta.py:143`, numba float64 arrays).
* **RNG (torch).** `gen_dyn = torch.Generator().manual_seed(rng_seed)` supplies the initial conditions
  and the Langevin noise. `gen_fr` (seeded `rng_seed + 987654321`) supplies the birth–death draws
  (`src/lta/core_lta.py:347-348`).
  * The R seeds of one call share ONE stream. Seed values are labels (stated in
    `src/cha/core_cha.py:331-332`).
  * Pairing across arms comes from the same `rng_seed` and the same replica index
    (`scripts/analyze_lta_histogram.py:529-534`).

### 1.2 `src/lta/core_lta.py` (722 lines): the torch production engine

**Physics.**
* `LTAParams` (`:76-89`):
  * eps_go = 93.0 K·kB, sigma_go = 3.48 Å, rc = 10 Å, r0 = 1.54 Å, k_bond = 400 kJ/mol/Å² (½k(r−r0)²), 2 beads.
  * The docstring attributes the cross LJ to the "TraPPE-zeolite family, Dubbeldam et al.",
    truncated + shifted (`:12-13`).
  * Only framework O interact; "Si is screened" (`:6-8`).
* `LTASystem(params, device, dtype=float64, root)` (`:140-154`):
  * loads `cache/lta/framework.npz` (`a_pseudo`, scalar `box`, `o_pos`);
  * asserts `rc < L/2` (scalar, cubic: `:149`);
  * sets the shift `v_rc` (`:153-154`).
* Methods:
  * `_lj_pairs(q)` (`:156-162`): dense (B, 2, n_O, 3) displacements, `d - L*round(d/L)`.
  * `forces(q)` (`:164-180`): analytic bond + LJ. `coef = 24 eps (2 sr6² − sr6)/r²`, `F += coef·d`.
  * `potential_energy(q)` (`:182-193`): bond + shifted LJ.
  * `cv_value(q)` (`:196-199`): φ = `remainder((2π/a) x_COM + π, 2π) − π`, equal-mass COM.
  * `cv_local_mean_force(q, F)` (`:201-212`):
    * `f_loc = -(a/2π)(F_0x + F_1x)`;
    * `grad_full[:, :, 0] = π/a` (the TRUE ∇φ per bead);
    * linear CV, no geometric term.
  * `initial_conditions(R, N, gen)` (`:214-228`): random α-cage centre of the S³ cubic lattice + 0.5 Å Gaussian
    COM jitter, random orientation, bond r0.
* `region_index(phi, a, window_half, cage_min)` (`:231-239`): 0 = cage (|z| > 4 Å), 1 = neck, 2 = window (|z| < 1.5 Å).

**Histogram (P0) estimator** (added 2026-10-03, off by default for bit-identity: `:30-39`, `:119`, `:124-137`).
* `bin_index(phi, n_grid)` (`:245-249`): `floor((φ+π)/dφ) % n_grid`. This is EXACTLY the deposit rule of
  `alkanes.periodic.bin_counts/bin_sum`, so a walker reads back its own bin.
* `histogram_mean_force(M, C)` (`:252-254`):
  * Γ_j = M_j / C_j on populated bins.
  * **Empty-bin rule: Γ = 0** (`torch.where(count > 0, ..., 0)`; the "WCA rule", `:37`).
* `histogram_pmf(gamma, dphi)` (`:257-269`): exact integral of the piecewise-constant force at the cell centres.
  * **Periodic closure:** subtract the circular mean of Γ first (`g0 = Γ − mean Γ`).
  * Then `F_right = cumsum(g0 dφ)`, `F_c = F_right − ½ g0 dφ`, and centre F on its circular mean.
* Kernel (legacy) path instead uses:
  * `per.mean_force_profile` with K_abf (bandwidth 0.05 rad), read at φ by `per.circular_interp`;
  * `per.free_energy_from_mean_force` (trapezoid between centres) (`:363-374`, `:441-442`).

**`run_sampler(method, system, sim, seeds, oracle_free_energy=None, shadow_counts=None, verbose=True)`** (`:322-624`).
* Methods: `abf`, `fr_estimated`, `fr_uniform`, `fr_oracle`, `fr_sham` (`:67-71`).
* `assert_no_reference_leakage` (`:337`): only `fr_oracle` may receive the reference.
* `fr_sham` requires `shadow_counts` of shape (n_opps, R). They are the partner `fr_uniform`'s
  `event_counts`; the call refuses any other method (`:350-361`).
* **Step order** for step k = 0..n_steps (state at the start of the step):
  1. **Forces and CV at the pre-move positions.** `f_loc` is clamped to ±8·abf_force_clip = ±480 before
     the deposit (`:419-424`).
  2. **Deposits.** All-steps accumulators `fsum/csum` always; production accumulators `fsum_prod/csum_prod`
     and U(z) sums when `step >= estimator_burn_in_steps` (`:426-432`).
     * Note: `potential_energy` is a second full LJ pass at every production step (`:431`).
  3. **Bias**, from the ALL-STEPS accumulators INCLUDING this step's deposit (`:434-446`).
     * `ramp = min(1, step/abf_warmup_steps)`.
     * Histogram path: `mf_at = gather(Γ, own bin)` (`:439-440`).
     * Both paths: `mf_at.clamp(±abf_force_clip = 60)` (`:443`).
     * `bias_force = ramp·abf_bias_scale·mf_at · grad_full` (`:444-446`).
  4. **Target EMA** (`fr_estimated` only), from `step+1 >= fr_start_steps` (`:448-454`).
     * `F_ema` starts at Â.
     * `F_ema ← (1−r)F_ema + r Â` with r = target_ema_rate = 0.005 per step, then re-centred.
  5. **Region and crossing bookkeeping** (`:456-463`).
  6. **Save** every `save_every` steps and at n_steps (`:465-501`).
     * The REPORTED estimator uses the production accumulators once `csum_prod.sum() > 0`, else the
       all-steps ones (`:466-469`).
     * Diagnostics: KDE p̂, KL to uniform, target q, ‖p−q‖, KL(p‖q), ancestor ESS, region fractions.
  7. **Move** (`:521-522`).
  8. **FR opportunity** at `nxt = step+1` when `nxt >= fr_start_steps` and `(nxt − fr_start) % fr_every == 0`,
     on the POST-move CV (`:524-577`).
     * `_fr_target_lta` (`:272-291`) gives:
       * uniform (and the sham's diagnostic target): `normalize_density(ones)` = 1/(2π);
       * oracle and estimated: `q ∝ exp(−β(F − B_n))` with `B_n = ramp·Â`, the predicted biased marginal.
     * Then `_fr_score` → `_birth_death` (alkanes, §1.5), or `_sham_birth_death` (`:294-319`).
     * The sham does uniform deaths without replacement and uniform sources among the survivors, capped
       at N−1 (`:308`).
     * Bookkeeping arrays (`has_left_cage`, `rep_crossings`, `prev_reg`) are copied to the clones (`:558-561`).
* **Snapshot recorder** `store_snapshots = k` (`:401-411`, `:503-516`, `:562-577`, `:600-619`). It is
  report-only, consumes no RNG and is bit-inert (tested at `tests/test_lta_histogram.py:147`). For one
  seed index it records:
  * q, φ, persistent walker ids, the ancestor label per slot, the reported PMF, counts and cumulative
    crossings;
  * per FR event: (slot, φ, COM) of the deaths and the sources, (dying slot, source slot) and (child id, parent id).
* **Output keys** (`:579-619`): diagnostics series, `fsum_prod`, `u_counts` (= production counts),
  `u_of_z`, `event_counts` (n_opps, R), birth/death histograms, `total_replacement_events`, `n_cage_crossings`.
* Defaults differ from the campaign config:
  * `LTASimConfig` (`:92-122`) has fr_rate 0.10, fr_start 40000, estimator "kernel".
  * The histogram campaign uses fr_rate 0.2 (0.1 at 80 K), fr_start 20000, warmup 20000, burn-in 20000,
    n_grid 180, N 1024, 300 000 steps, 16 seeds (`configs/lta_histogram/campaign.json`, `sampler` and `per_T`).
  * Always build the config from the campaign file, as `scripts/run_lta_histogram.py:73-83` does.

**Reference tools.**
* `run_umbrella(system, sim, centers, kappa, n_steps, n_replicas, burn_in, sample_every, seed)` (`:627-671`):
  * all windows run in one batch, with bias ½κ wrap(φ−c)² applied via ∇φ (`:657-661`);
  * starting COM x is shifted onto the window centre (`:641-643`);
  * **a non-gradient force-norm clamp at 500 kJ/mol/Å for the first 5000 steps** (`:647`, `:653-655`).
* `wham_1d(phi_samples, centers, kappa, beta, n_bins=180)` (`:674-704`): periodic histogram WHAM.
  * Bias at the bin mids, from the wrapped distance.
  * p is normalised as a density.
  * Convergence when the max change of mean-shifted f_w is < 1e-10, ≤ 20 000 iterations.
  * F = −kT log max(p, 1e-300), mean zero. An empty bin would enter the mean at about +690 kT and shift
    the gauge.
* `conditional_u(phi, u, n_bins)` (`:707-722`): pooled per-bin mean of V (NaN for empty bins).

### 1.3 `src/lta_ladder_numba.py` (1556 lines): CPU numba port for the equal-budget ladder

**Purpose.** "The production torch engine ... is dispatch-bound for small N, so the long-thin end of the
ladder needs a compiled scalar loop" (`:8-9`). It is an op-for-op port of `core_lta.run_sampler`, methods
`abf` and `fr_uniform` only, histogram estimator only (`:11-34`). The porting audit is
`docs/equal_budget/audit/AUDIT_0A_lta.md` (`:12`).

**Kernels.**
* `_pair` (`:245-262`): one O on one bead in torch's operation order. Note `rint(d·invL)`; exact zero
  outside rc.
* `_lj_bead` (`:265-276`): the sum over all O, with `fastmath={"reassoc"}` so the sum vectorises.
* `_mol_force` (`:305-331`) and `_mol_energy` (`:334-347`): the 2-bead molecule.
* `_wrap` (`:350-357`): the fmod form, "bit-identical to ATen's CPU kernel".
* `_bin` (`:360-363`), `_region` (`:366-373`).
* `_true_event` (`:376-397`):
  * window-plane crossings are changes of `floor(x_COM/a)`;
  * a translocation is arrival in a cage of a different x-cell from the lineage's last cage.
* `_family_stats` (`:413-428`): number of unique ancestors, **ESS/N = N/Σc²**, max family fraction.
* `_fr_score` (`:431-515`): uniform target only. qv is a constant (`:483`); interpolation wraps periodically
  (`:479-481`); KDE sum over occupied bins in ascending order.
* `fr_select_native` (`:518-572`):
  1. dp = 1 − exp(−rate·S⁺·dt_eff);
  2. guard: if Σ S⁻ ≤ EPS or Σ S⁺ ≤ EPS, no event;
  3. candidates are walkers with u_i < dp_i;
  4. a partial Fisher–Yates picks a uniform ordered cap-subset;
  5. sources by inverse CDF ∝ S⁻.
* `_run_chunk` (`:575-883`) phases:
  1. forces/CV/clip (`:640-654`);
  2. per-step staged deposits then accumulators (`:655-681`);
  3. bias `ramp·clip(M/max(C,1), ±60)·(π/a)` from all-steps accumulators incl. this step (`:682-698`);
  4. regions and true events (`:699-716`);
  5. save (`:717-771`);
  6. EM move (`:784-791`);
  7. FR at the post-move CV with whole-molecule + ancestor + label copies (`:793-881`).

**Driver.**
* `DEFAULT_CFG` (`:186-215`): h 2e-4, B = 3.072e8, warmup = burn-in = fr_start = 20000, fr_every 5,
  **fr_rate 0.2**, score_clip 2, KDE 0.10, max_event_fraction 0.02, cap_min 1, abf_force_clip 60,
  deposit_clip = 8×60.
* `make_cfg(T_K, N, seed, **overrides)` (`:899-906`); `resolve_cfg` (`:909-946`).
  * `resolve_cfg` refuses a deposit_clip ≠ 8×abf_force_clip without an explicit override (`:916-926`).
  * cap = max(cap_min, floor(0.02 N)) (`:943`).
* `seed_streams(seed, N, T_K)` (`:949-957`): `SeedSequence([seed, N, round(T)]).spawn(3)` gives IC, noise and FR streams.
* `initial_conditions` (`:960-970`), `kde_matrix` (`:973-979`), `mean_force` (`:982-986`),
  `histogram_pmf` (`:989-994`), `budget_save_grid` (`:997-1004`).
* `run_arm(cfg, method, save_steps, *, checkpoint_path, checkpoint_every_s, chunk_steps, ...)` (`:1225-1405`):
  * atomic checkpoints, resumed bitwise;
  * the signature holds the job fingerprint, engine sha256, numpy/numba versions and compile target (`:1157-1172`, `:1308-1331`).
* `_build_result` (`:1408-1524`), `save_result`/`load_result` (`:1531-1556`): `.npz` without pickle.

**Deliberate differences from torch** (`:36-70`).
* Three PCG64 streams, against torch's one generator.
* A fixed FR uniform block of N + 2cap per opportunity.
* The finite-N cap extension (cap_min = 1). Torch's `int(0.02 N)` silently disables FR below N = 50 (`:48-49`).
* The 'reassoc' LJ sum is bitwise reproducible only per compile target (`:54-62`; measured znver4 vs
  skylake-avx512 |dq| ~ 1e-14 after 3000 steps).
* FR is refused at N = 1; N = 2 is FR-inactive by symmetry (`:65-69`).

**What it does NOT have.**
* No sham arm, no estimated or oracle target, no snapshot recorder.
* Instead it has traces of the first ≤ 32 slots and `hist_inst` (`:90-92`, `:772-777`).

### 1.4 `src/parallel/lta_parallel.py` (1554 lines) + `src/parallel/philox.py` (267 lines)

**Engine.**
* Walker-parallel `prange` over static contiguous walker blocks with the OpenMP threading layer
  (`numba.config.THREADING_LAYER = "omp"`, `:98`).
* It imports and calls the validated numba functions unchanged: `_mol_force, _wrap, _bin, _fr_score, fr_select_native`
  (`:106-112`).
* The move of step k is fused with the force evaluation of step k+1 (`:26-31`). Deposits, the bias read,
  saves and FR stay serial (`:32-33`).

**Noise modes.**
* `'replay'`: the PCG64 stream. Bitwise equal to `lta_ladder_numba.run_arm(diagnostics=False)` at any
  thread count (`:37-39`).
* `'counter'`: in-kernel Philox4x64-10 + Box–Muller, keyed `SeedSequence([seed, N, 150, 2])` (`:40-42`;
  `philox.py:85-131`, `lta_normals6` = 6 normals per (step, slot) for 2 beads × 3).

**Dependency-keyed numba cache** (`:177-238`). numba keys a cached function only by its defining file,
so a cross-file numba import would otherwise run stale machine code after an edit to the imported file.
A per-process `kernel_selfcheck` guards it (`:62-69`, `:849`).

**API and tests.** `run_arm_parallel(cfg, method, N, seed, n_steps, save_steps, *, threads, noise, ...)` (`:677`).
Tests: `tests/test_parallel_lta.py` (23 tests; not run here).

### 1.5 `src/alkanes/core.py` (FR machinery) and `src/alkanes/periodic.py` (periodic estimators)

These are imported unchanged by LTA (`src/lta/core_lta.py:61-63`) and by CHA (`src/cha/core_cha.py:43`).

**`core.py` functions.**
* `_fr_target(method, grid, dphi, F_ema, B_n, oracle, beta)` (`core.py:83-99`): uniform, oracle
  `exp(−β(F_ref − B_n))`, estimated `exp(−β(F_ema − B_n))`. Returns None before F_ema exists.
* `_recentered_clipped_score(raw, clip)` (`:102-107`): recentre, then 3 × (clip ±clip, recentre).
* `_fr_score(phi, grid, dphi, K_kde, q_grid, kde_bw, clip)` (`:110-121`):
  * p̂ = `normalize_density(bin_counts @ K_kde)`;
  * raw S_i = log p̂(φ_i) − log q(φ_i) − KL(p̂‖q), with p̂ and q read by periodic linear interpolation;
  * returns (score (R,N), p_grid, kl). `kde_bw` is unused.
* `_birth_death(q, score, ancestors, sim, gen_fr)` (`:124-160`):
  * dt_eff = dt·fr_every;
  * `max_events = int(max_event_fraction·N)`; no FR if < 1 (`:131`, `:137`);
  * death weight S⁺, birth weight S⁻, death prob 1 − exp(−fr_rate·S⁺·dt_eff);
  * guard on Σ weights (`:146`);
  * `randperm` cap subset (`:152-154`);
  * sources `multinomial(S⁻, n, replacement=True)` (`:155`);
  * **whole-configuration copy `q_new[r, di] = q[r, src]`**, generic in the trailing dimensions (`:156`), plus ancestor copy.
* `_ancestor_stats(ancestors, N)` (`:163-175`): **ESS = 1/Σw² in walkers** (not divided by N; analysis divides,
  `scripts/analyze_lta_histogram.py:414`), unique count, max fraction.
* `assert_no_reference_leakage(method, oracle_free_energy)` (`:178-186`).

**`periodic.py` functions.**
* `periodic_grid` (`:25-29`): the cell-centred grid on [−π, π).
* `circular_distance` (`:32-35`).
* `wrapped_gaussian_kernel_matrix` (`:38-41`): unnormalised, single nearest image.
* `bin_counts` / `bin_sum` (`:44-59`): `floor((φ+π)/dφ) % n`, `scatter_add_`.
* `smooth` (`:62-64`), `normalize_density` (`:67-71`), `kde_marginal` (`:74-77`), `circular_interp` (`:80-95`).
* `mean_force_profile` (`:98-107`), `free_energy_from_mean_force` (`:110-122`), `effective_counts` (`:125-127`).
* Metrics: `circular_l2` aligned (`:133-142`), `marginal_tv/kl/l2` (`:145-159`).

### 1.6 `src/alkanes/cv.py`: the nonlinear-CV template (den Otter geometric term)

* Formula: `f_loc = ∇V·v − β⁻¹ div v`, v = ∇ξ/|∇ξ|², `div v = Δξ/|∇ξ|² − 2 ∇ξᵀH∇ξ/|∇ξ|⁴` (`:3-24`).
* ∇ξ and H come from exact autodiff: `torch.func` grad + hessian under vmap (`:36-43`, `geometry` `:58-75`).
* `local_mean_force` (`:77-88`).
* The bias force is **`+A'(ξ) ∇ξ`**, "a conservative force −∇[−A(ξ(q))]" (`:22-24`, `abf_bias_force` `:91-93`).

### 1.7 `src/cha/core_cha.py` (596 lines): the nearest relative for ERI geometry

* **Orthorhombic box.** `box` is a (3,) tensor; `_min_image(d) = d − box·round(d/box)` (`:117`, `:140-141`).
* **Hexagonal → orthorhombic conversion** is done by `scripts/build_cha_framework.py:72-96`
  (A = a, B = a + 2b, C = c, so Lx = a, Ly = √3a, Lz = c).
* **Numerical cage/window finder** in `build_cha_framework.py`:
  * nearest-O distance field and cage maxima (`:119-150`);
  * cage-pair bottleneck search (`:153-176`);
  * 8-ring O count (`:181-193`);
  * best-fit SVD ring plane (`:196-206`).
* Guests:
  * a `GUESTS` table (`:50-59`) with mass-weighted COM (`:126`);
  * Lorentz–Berthelot to O (53.0 K, 3.30 Å) (`:60`, `:129-132`);
  * one `torch.compile`d energy, with forces by autograd (`:135-138`, `:177-182`).
* CV:
  * ξ = (COM − ring_centroid)·n, linear, non-periodic. `f_loc = −(F·∇ξ)/|∇ξ|²` with |∇ξ|² = Σ(m_i/M)² (`:184-200`).
  * Mass-weighted bias force `A'·(m_i/M)·n` (`:401-404`).
* Non-periodic estimator:
  * kernel-smoothed `num/(den + abf_min_count)` (`:259-262`), trapezoid PMF (`:265-268`), clamped
    interpolation (`:271-281`);
  * **`bin_index` clamps out-of-range ξ into the end bins** (`:236-238`).
* FR and target for a 1-D line: `fr_score_1d` (`:297-305`), `fr_target_1d` (`:308-323`).
* Reference: `wham_1d_line` (`:560-586`).
* **Two physics alterations that ERI must not copy silently:**
  * the sphere-union confinement wall (`:20-23`, `:155-159`);
  * a production-dynamics force-norm clamp for the first 2000 steps (`:374-381`).

### 1.8 `src/lta_validation.py` (347 lines): the independent float64 numba model (Gate-1 template)

* It re-implements the physics from the definition and never imports the production engine
  (`:1-21`). Parameters come from `params(T)` (`:42-48`); the kernels are `bead_lj` (`:51-78`) and
  `energy_force` (`:81-104`).
* It samples `exp(−β[U − A(φ)])` with a FIXED bias (`:13-16`), one of:
  * a Fourier series (`bias`, `:114-124`; `fourier_fit`, `:274-282`);
  * an umbrella: kappa > 0 gives `A = −(κ/2) wrap(φ − c)²` (`:169-172`).
* Schemes (`run_molecules`, `:137-267`):
  * **EM**: `q' = q + dt(F + A'∇φ) + √(2 dt kT) ξ` (`:198-205`).
  * **LM** (Leimkuhler–Matthews): `q_{n+1} = q_n + dt(F(q_n) + A'(φ_n)∇φ) + √(2 dt kT)·½(R_n + R_{n+1})`.
    The previous normal is carried in `R` (`:206-209`); second order for the invariant measure (`:19`).
  * **MC**: one step = a single-bead uniform move ±d_bead (0.08 Å) on a random bead + a rigid translation
    ±d_trans (0.3 Å). Both are symmetric; acceptance uses U − A (`:221-259`, defaults `:310-311`).
* Every chain is pre-relaxed by `n_pre_mc = 5000` exact MC steps, **with no force clamp** (`:141-143`, `:181`).
* Deposits every `sample_every` steps after burn-in (`:183-197`): fine histogram (`nfine` = 1800), 180-bin C, Mf, Mf²,
  bond moments, U. Window-plane cell crossings are counted.
* `wham(H, centres, kappa, kT, nfine=None, tol=1e-12, max_iter=200000, mode="fine")` (`:323-347`):
  * binned periodic WHAM in kT units, bias at the bin centres, p summing to 1, gauge f_0 = 0;
  * **`mode` and `nfine` are accepted but unused**. The fine-vs-mid180 distinction lives in the analysis script (§2.3).
* Tests (`tests/test_lta_validation.py`):
  * L1/L3 forces, energies, φ and f_loc equal `core_lta` to 1e-10 relative at 300 and 150 K (`:22-38`);
  * L2 forces equal central differences (`:41-56`);
  * L4 WHAM recovers a known density from synthetic multinomial umbrella histograms, RMS < 0.02 (`:59-78`).
* Result (`docs/numerical_validation/LTA_VALIDATION.md:182-187`): EM dt 2e-4 is ADMISSIBLE against MC at 300 K and
  150 K, and the published 300 K umbrella reference is MARGINAL (0.034 kJ/mol).

---------------------------------------------------------------------------------------------------------

## 2. Exact conventions the ERI engine must reproduce (or consciously change)

### 2.1 How the histogram-ABF force is applied (torch `core_lta.py:418-446`; numba `lta_ladder_numba.py:640-698`, `:784-791`)

1. **Instantaneous mean force.** `f_loc = −(F·∇φ)/|∇φ|² = −(a/2π)(F_0x + F_1x)` with F = −∇U the physical force.
   For a linear CV this is ∂U/∂φ, whose conditional mean is +dA/dφ (`core_lta.py:201-212`).
   * Deposit clip ±8·60 = ±480 kJ/mol/rad (`core_lta.py:423-424`; the rule is enforced in numba at `lta_ladder_numba.py:916-926`).
   * Possible bias source: (a/2π)·2·150 kJ/mol/Å ≈ 569 can exceed 480, given the
     "max abs F ≈ 150 kJ/mol/Å" snapshot figure in `docs/numerical_validation/LTA_VALIDATION.md:28`. The
     validation's unclipped MF route agreed with MC, so this was not material for LTA. ERI must count clip hits.
2. **Deposit before read.** The deposit goes into the bin j = floor((φ+π)/dφ) mod 180 of the pre-move φ. The
   bias then reads Γ_j from the all-steps accumulators that already include this step.
3. **Bias force.** `bias = ramp·scale·clip(Γ_own, ±60)·∇φ`, with ∇φ = π/a in x on each bead. The sign is
   **+**: the force +A'∇φ cancels the mean force −A'∇φ, so the biased marginal is ∝ exp(−β(F − A)).
   * There is no interpolation and no smoothing in the histogram path.
   * The read clip of 60 kJ/mol/rad is about 31.6 kJ/mol/Å in z at a = 11.919 Å.
4. **Ramp.** `ramp = min(1, step/warmup)` (`core_lta.py:436`; numba `:683-686`).
5. **No circular-mean correction of the APPLIED bias.** Only the read-out PMF subtracts mean(Γ)
   (`core_lta.py:266`). With a periodic CV the applied Γ(φ)∇φ has net circulation 2π·mean(Γ) ≠ 0 at
   finite samples, a small non-conservative tilt. The reference quantity is logged as
   `gamma_circular_mean` (`scripts/equal_budget/build_references.py:36`). ERI keeps the same convention for
   comparability, or changes it deliberately and documents it.

### 2.2 FR conventions (shared by torch and numba)

* **KDE.** Bandwidth 0.10 rad, unnormalised single-image wrapped Gaussian, normalised to a density.
* **Score.** log p̂ − log q − KL, recentred, then 3× (clip ±2, recentre).
* **Schedule and burn-in.** FR starts at `fr_start_steps` on the post-move CV, every `fr_every = 5` steps
  (dt_eff = 1e-3 t.u.). The production estimator burn-in is separate (`estimator_burn_in_steps`).
* **Cap.**
  * Torch: `int(0.02 N)` events per opportunity (20 at N = 1024).
  * Numba: max(1, floor(0.02 N)).
* **Sham.** It replays the partner's per-opportunity counts (`event_counts`). Run order abf → fr_uniform → fr_sham
  in one process (`scripts/run_lta_histogram.py:126-139`).
* **Genealogy.**
  * Torch ancestor labels are whole-run.
  * Numba also keeps windowed labels reset every 4 t.u. (`lta_ladder_numba.py:87-89`, `:778-780`).
  * Health: min ESS/N over FR-active saves ≥ 0.30 and max family fraction ≤ 0.05
    (`scripts/analyze_lta_histogram.py:91-93`, `:411-427`).
* **Rate calibration** is SAFETY-ONLY: the largest ladder rate meeting the genealogy floors, with no error
  metric read (`scripts/run_lta_histogram.py:154-192`).

### 2.3 Reference and WHAM modes (`scripts/numerical_validation/analyze_lta_validation.py`)

* **`density`.** WHAM on 1800 fine bins (bias at the fine-bin centres), summed to 180 bins, F = −kT log p
  (`:104-105`, `:116-122`). Fine bins matter because at κ = 300 kJ/mol/rad² the umbrella bias varies by
  about κ·|d|·dφ inside one 0.035-rad bin. That is about 5 kJ/mol at |d| = 0.5 rad (estimate).
* **`mid180`.** 180-bin WHAM with the bias at the bin midpoints, the published reference convention (`:105-106`, `:180`).
* **`mf`.** Pooled conditional mean force Σ Mf / Σ C over all windows, integrated with `hist_pmf` (`:113-115`).
  This is valid because the umbrella bias depends on φ only (`src/lta_validation.py:13-16`).
* **Noise.** Leave-one-group-out jackknife over 16 groups (`:108-110`).
* **`compare(A, B)`** (`:133-145`): D = √max(0, rms² − mean(se_A² + se_B²)), with a group-bootstrap 95 % upper
  bound debiased by 2× noise².
* **Gates.** `level` (`:156-159`): ADMISSIBLE if D ≤ 0.03 and upper ≤ 0.045 kJ/mol, MARGINAL if D ≤ 0.06,
  else FAIL. The MC self-consistency gate is D(MC mf, MC density) ≤ 0.03 (`:217`).
* **Run grid** (`scripts/numerical_validation/run_lta_validation.py:24-32`): 40 windows; κ 300 / 254 at 300 / 150 K;
  16 groups × 40 windows; 16 molecules per job (8 for EM 5e-5); seed = 7000 + 1000·tag + 50·group + window (`:41`).
* **Frozen references for the ladders.** `scripts/equal_budget/build_references.py:16-35` builds F_ref
  (MC density), its jackknife se, γ_ref = per-bin pooled mean force and F_ref_mf.

---------------------------------------------------------------------------------------------------------

## 3. How the three LTA engines differ

| aspect | torch `core_lta` | numba ladder `lta_ladder_numba` | validation `lta_validation` |
|---|---|---|---|
| role | production (GPU), all 5 methods, kernel or histogram | equal-budget ladder (CPU), `abf` / `fr` (uniform) only, histogram only | independent model (CPU) for Gate-1 references |
| sampling | adaptive ABF | adaptive ABF | FIXED bias (Fourier) or umbrella; no ABF |
| integrator | EM | EM (op-for-op port) | EM, **LM** (½(R_n + R_{n+1}) noise), **Metropolis MC** (bead + rigid moves) |
| start | random α-cage centre + 0.5 Å jitter | same law, own PCG64 stream | same law (`initial_states`) or window-centred COM (`window_states`, jitter 0.2 Å) + 5000 exact MC steps |
| clamps | f_loc ±480 deposit, Γ ±60 read; umbrella: force-norm clamp 500 for 5000 steps | same deposit/read clips; no umbrella | **none** |
| RNG | 1 torch generator for all R seeds + FR generator | `SeedSequence([seed, N, round T]).spawn(3)` PCG64 | PCG64(seed) per job |
| determinism | CPU exact; **CUDA not bitwise** (scatter-add) | bitwise per compile target; checkpoint resume bitwise | serial, deterministic |
| WHAM | `wham_1d` (density, bias at mids) | none | `wham` (prob., gauge f_0 = 0); fine/mid180 in the analysis |
| jackknife | none | none | analysis: 16-group LOO |

Equivalence evidence:
* E1 replay of torch-recorded draws (`tests/test_lta_ladder_numba.py:160-161`);
* law tests E2 (`:261-339`);
* the CUDA-vs-numba ΔI_F agree within CIs (`docs/equal_budget/IMPLEMENTATION_VALIDATION.md:75-79`).

---------------------------------------------------------------------------------------------------------

## 4. Metric and analysis functions

### 4.1 `scripts/analyze_uniform_lta.py` (imported by both analyses)

* `error_series(pmf, F_ref)`: the aligned full-circle RMS per (save, seed).
* `boot_median(x, seed)`: 10 000-resample bootstrap of the median, `N_BOOT, BOOT_SEED = 10000, 20260829` (`:28`).
* `tau(times, curve, eps, persist=0.2)`: the first time the curve stays ≤ eps for 0.2·T.
* `circular_interp_ref`: periodic linear interpolation of the reference onto the engine grid.

### 4.2 `scripts/analyze_lta_histogram.py`

* **`score_group(arms, ref, fr_start_steps, N, notes, tag)`** (`:368-429`), per arm:
  * the reference interpolated to the engine grid;
  * e_F(t) per save and seed;
  * **I_F = trapezoid of e_F over save times**, final e_F(T), and the median curve;
  * events and crossings per replica;
  * raw support (min / median / empty bins of `u_counts`);
  * a check that `histogram_pmf(fsum_prod/u_counts)` reproduces the last saved PMF;
  * genealogy health over FR-active saves.
* **`contrast(g, arm, base)`** (`:446-449`), built on `paired` (`:135-142`): the per-seed relative change
  100·(arm − base)/base of I_F and of e_F(T).
  * Reported: median, bootstrap 95 % CI (seed 20260829, +1 for the final endpoint), wins = #(Δ < 0).
* **`verdict_from_rule`** (`:478-490`): SAFE_ACCELERATOR / ACCELERATION_POSITIVE / NEUTRAL / NEGATIVE_OR_UNSAFE,
  with these thresholds (`:91-93`):
  * median ≤ −10 % and CI upper < 0;
  * final ≤ +5 %;
  * ESS/N ≥ 0.30 and w_max ≤ 0.05.
* Other functions:
  * `validate_arms` (`:334-365`): times, seeds, temperature and estimator must match;
  * `tau_table` (`:452-475`);
  * `establishment_diag` (`:554-604`): the T_hit/T_est screening rule;
  * `p0_floor` (`:607-633`): the deterministic P0 discretisation floor from a periodic spline of the reference.

### 4.3 `scripts/equal_budget/eqb_metrics.py` (`METRICS_VERSION = "eqb_metrics/2"`, `:66`)

* **Per run:** `run_metrics(res, system, P, scorer)` (`:645-867`). LTA errors use `lta_estimator` (`:320-331`) and `lta_errors_from` (`:334-341`):
  * Γ = M/max(C,1), 0 where empty, from the production accumulators once non-empty, F = histogram_pmf(Γ);
  * **e_F** = RMS of F − F_ref after removing the mean difference;
  * **e_Fp** = RMS of Γ − γ_ref, and **e_Fp_proj** the same with both circular means removed.
* **Curves and scalars:**
  * `Ibar_*` = mean over the 200 uniform budget fractions u = k/200 (`uniform_indices`, `:572-587`), and `final_*`;
  * `tau_persistent` (`:526-533`): first save after which the error stays ≤ ε at all later saves; inf = censored;
  * `tau_bgrid_*` on the common budget grid for cross-N comparisons;
  * TV_inst against uniform on 18 coarse bins, with the exact finite-N floor `tv_floor` (`:451-462`), and TV_half (`:488-513`).
* **Establishment:** the persistent far-state fraction ≥ ½ target, where the LTA far state is the window
  |z| < 1.5 Å with target 0.252 (`:92-95`). It carries a binomial null calibration, `establishment_null` (`:544-569`).
* **Activity:** genealogy minimum and final values, FR activity, and cap flags.
* **Group statistics:**
  * `boot_median_ci` (`:893-901`);
  * `paired_contrast` (`:904-916`): G = (FR − ABF)/ABF;
  * `sign_test_p` (`:919-926`);
  * `paired_tau_contrast` (`:929-948`): censoring-aware;
  * `best_allocation` (`:951-1044`): re-selects the best N in every bootstrap resample, ties share credit;
  * `max_transient_improvement` (`:1047-1058`).
* **Hard-wired to LTA/gateway:**
  * `SYSTEMS`, `FAR`, `LTA_KEYS` (`:86-105`); NB = 180 (`:67`); `LTAScorer` paths (`:344-385`); `far_bin_mask` (`:613-621`).
  * **The file's sha256 is part of every committed cache key** (`:79-84`). Editing it makes all committed
    equal-budget analyses STALE, which is why `scripts/equal_budget/eqb_family.py:4-12` extends it by a
    separate module. ERI must do the same.

---------------------------------------------------------------------------------------------------------

## 5. Reuse plan for an ERI engine

The target described in the brief:
* each replica holds a tagged ethane (two CH3 beads) plus optionally K−1 further ethanes (CH3–CH3 guest–guest LJ);
* the framework is rigid, with O only, in an orthorhombic supercell of about 45.9 × 39.8 × 44.4 Å;
* the cutoff is 12 Å.

### 5.1 What the downloaded structure files say about that box (data quoted as text; nothing executed)

* **IZA CIF.** `cache/eri/literature/structure/iza_europe/ERI.cif:14-23`: `_cell_length_a 13.0540(0)`,
  `_cell_length_c 15.1750(0)`, `_cell_angle_gamma 120.0000(0)`, `'P 63/m m c'`, hexagonal.
* **RASPA ERI example.** `cache/eri/literature/structure/raspa2_ex9_hexane_ERI_output/output_ERI_SI_3.3.3_573.000000_0.data`:
  * `:679` "Unit cell size: 13.270000 13.270000 15.050000";
  * `:686` "TRICLINIC Boundary conditions";
  * `:736-737` "Number of framework atoms: 2916" / "... in the unit cell: 108" (= Si36O72).
  * The RASPA example therefore uses a 3×3×3 *hexagonal* (triclinic-box) supercell, not an orthorhombic one.
* **Arithmetic** (not a literature claim). 45.9 ≈ 2·√3·13.27 = 45.97 and 39.8 ≈ 3·13.27 = 39.81. But
  44.4/3 = 14.80 Å matches neither c = 15.05 (RASPA) nor 15.175 (IZA).
  * **Open issue for the literature cross-check:** where do 44.4 Å and the orthorhombic setting come from?
  * Note the axis order too: x = 2√3a, y = 3a is the reverse of `build_cha_framework.py`'s convention
    (Lx = a, Ly = √3a, `:79`).
* **Estimate of size and cost.** A (2√3a × 3a × 3c) orthorhombic box = 18 C-centred orthorhombic cells
  = 36 hexagonal cells, giving **2592 O and 1296 Si**.
  * At a = 13.27, c = 15.05: V ≈ 82,600 Å³, ρ_O ≈ 0.0314 Å⁻³.
  * About 227 O lie inside a 12 Å sphere, against 2592 in a dense loop: an 11× waste.
  * LTA had 119 of 384 inside 10 Å, a 3.2× waste.

### 5.2 Import unchanged (no edits to existing files)

**Engine pieces.**
* `alkanes.periodic` (all) and `alkanes.core._fr_target, _recentered_clipped_score, _fr_score, _birth_death,
  _ancestor_stats, assert_no_reference_leakage`.
  * Only if the ERI CV is periodic. `_birth_death` copies `q[r, src]` over any trailing shape, so
    (R, N, K, 2, 3) multi-guest replicas work as-is.
* `lta.core_lta.bin_index, histogram_mean_force, histogram_pmf, _fr_target_lta, _sham_birth_death, wham_1d,
  conditional_u` (periodic; the sham is trailing-shape generic).
* `lta_ladder_numba`:
  * `_fr_score` and `fr_select_native` (periodic, uniform target only);
  * `kde_matrix, mean_force, histogram_pmf, budget_save_grid, seed_streams`;
  * the checkpoint, signature, provenance and RSS machinery (`run_signature`, `signature_mismatch`,
    `compile_target`, `_write_checkpoint`, `_read_checkpoint`, `_reset_peak_rss`, `save_result`, `load_result`).
  * Cross-file numba imports need the dependency-keyed cache of `parallel/lta_parallel.py:177-238`, or a
    copy with a recorded sha256. `lta_ladder_numba.py` is frozen, so the stale-cache risk is low but
    not zero.
* `parallel.philox`: `philox4x64_10`, `box_muller`, `raw_block` are generic. `lta_normals6` is hard-wired
  to 6 normals per slot, so a K-guest layout needs a new counter map.

**Statistics and analysis.**
* From `eqb_metrics` (pure functions, no system dependence): `boot_median_ci`, `paired_contrast`,
  `paired_tau_contrast`, `sign_test_p`, `best_allocation`, `tau_persistent`, `persistent_first`,
  `establishment_null`, `tv_floor`, `tv_to_uniform`, `tv_half`, `coarse_bins`, `uniform_indices`,
  `max_transient_improvement`, `json_safe`.
* From `analyze_uniform_lta`: `boot_median, tau, error_series`.
* From `analyze_lta_histogram`: `paired, contrast, verdict_from_rule`. `score_group` works for a periodic CV
  as long as the arm npz has `grid, times, pmf, steps, seeds, meta`.

**Validation tools.**
* `lta_validation.wham`, `fourier_fit`.
* The `analyze_lta_validation.Wham`/`Prof`/`compare`/`level` design. `CENTRES` (40), NB/NF and
  `barrier(a=11.919)` are LTA constants (`:26-30`, `:148-153`), so these are copied, not imported.

**Test and safety templates.**
* The L1–L4 test pattern of `tests/test_lta_validation.py`.
* The E1–E7/X* patterns of `tests/test_lta_ladder_numba.py`: replay equivalence, law tests, resume
  bitwise, fr_rate = 0 ≡ ABF, force-evaluation accounting.
* The ABF/FR bit-identity-before-fr_start check.

### 5.3 Must be generalised (new code in `src/eri/`)

1. **Framework builder** (`scripts/eri_campaign/build_eri_framework.py`).
   * Reuse the hexagonal→orthorhombic math, the Si:O/Si–O assertions and the numerical cage/window finder
     of `build_cha_framework.py:72-206`.
   * **Do not reuse its CIF op parser.** Both builders `eval()` the symmetry-op strings of the CIF
     (`scripts/build_lta_framework.py:41-44`, `scripts/build_cha_framework.py:49-51`), which would execute
     text from an untrusted download. Write a whitelist parser (±x/±y/±z/fractions) and run it with `python -I`.
   * Assert:
     * 2592 O for the planned box, or whatever the confirmed box implies;
     * `rc < min(box)/2` (12 < 19.9 Å for the stated box);
     * the 8-ring count of every window used by the CV.
   * Store `box` as a (3,) array plus cages, windows (centre, normal) and accessible-cage labels.
     Whether inaccessible cages must be blocked is a literature question; support an optional block list
     anyway.
2. **Physics.**
   * Per-component minimum image with a (3,) box. That is `core_cha._min_image` (`:140-141`); in numba,
     `d - L[k]*rint(d*invL[k])`.
   * Replace the scalar assert `core_lta.py:149` with `rc < min(L)/2`.
   * Multi-type LJ:
     * CH3–O cross parameters come from the literature extraction (the repo's LTA values 93.0 K/3.48 Å,
       `core_lta.py:80-81`, are not automatically ERI's);
     * CH3–CH3 guest–guest LJ, with min image, intramolecular pair excluded, truncated (and shifted or
       not, per the paper).
   * Tagged guest index 0, K guests per replica, state array (N, K, 2, 3).
3. **Framework pair search.** Dense all-O loops cost about 6.7× LTA per molecule-step (2·2592 against 768
   pair terms). Two options:
   * **Recommended: static voxel candidate lists.** Possible because the framework is rigid.
     * Voxel size s ≈ 1 Å; per voxel, the ascending-index list of O within rc + s√3/2 of the voxel centre.
     * Exact physics (no interpolation), no rebuilds, fixed summation order.
     * About 280 O per bead (estimate). Storage is about (box/s)³ × 280 int32 ≈ 90 MB, or one orthorhombic
       unit cell's worth if the voxel grid is commensurate.
   * *Optional:* a tricubic energy grid (RASPA-style). It is faster but adds interpolation error, and the
     force must be the exact gradient of the interpolated energy (the WCA lesson in §6). The Gate-1
     reference must then use the same grid potential, or Gate 0 must bound the grid error.
4. **CV and geometric term.** For any CV that is a function of the tagged COM, ξ = g(R), R = Σ w_b q_b, w_b = m_b/M:
   * ∇_{q_b}ξ = w_b ∇g and |∇ξ|² = c|∇g|² with c = Σw_b² (= ½ for ethane).
   * `f_loc = −(F_0 + F_1)·∇g/|∇g|² − kT [Δg/|∇g|² − 2∇gᵀH_g∇g/|∇g|⁴]`. The divergence term does not depend on c.
     * Linear g: it reduces to `core_lta.py:209`.
     * Radial g = |R − c₀|: it gives the familiar −2kT/r.
   * Bias force on bead b: `+A'(ξ)·w_b·∇g` (as `core_cha.py:401-404`).
   * Validate the analytic g-derivatives against a `torch.func` autodiff oracle (`src/alkanes/cv.py:36-88`)
     and against finite differences in Gate 0.
   * A **periodic** CV is possible only if the cage–window–cage path repeats along a box axis with a period
     commensurate with the box. Then the LTA machinery applies unchanged with a → that period.
   * Otherwise the CV is **non-periodic** and needs:
     * the non-periodic P0 estimator: `eb_abffr_core.HistogramABFEstimator` (`src/eb_abffr_core.py:327-402`),
       Γ = M/(C + min_count), exact PMF integration, own-bin evaluate;
     * a non-periodic FR score (as `core_cha.fr_score_1d`) and uniform target on [lo, hi];
     * an explicit out-of-range rule. Both `core_cha.bin_index` (`:236-238`) and `eb.histogram_bin_index`
       (`src/eb_abffr_core.py:319-324`) **clamp out-of-range samples into the edge bins**. Either confine
       (a declared wall, applied identically to all arms and the reference) or neither deposit nor bias
       outside the range.
5. **Multi-guest ABF semantics.**
   * Bias and deposit the TAGGED guest only. The ABF identity "biased marginal ∝ exp(−β(F − A))" holds for
     the tagged ξ.
   * Biasing every guest by A(ξ_k) changes the joint ensemble, and for interacting guests it does not
     flatten the single-guest marginal.
   * The FR score uses the tagged ξ across replicas, and clones copy the whole replica (all K guests).
     Numba copy loops are hard-coded to (2, 3) (`lta_ladder_numba.py:860-862`) and must become (K, 2, 3).
6. **Engine port.** Copy `_run_chunk` structurally into an ERI kernel and keep the step order (§2.1). Add:
   * a **sham arm** (numba has none; port the `_sham_birth_death` law and the shadow-count replay);
   * per-guest noise layout (n_moves, N, K, 2, 3);
   * ERI region and true-event bookkeeping (cage/window labels from the builder, not `floor(x/a)`);
   * accessible-cage initial conditions;
   * optional U(ξ) accumulation at a stride (torch `core_lta.py:431` doubles the LJ cost at every production step).
7. **Independent reference** (`src/eri/reference_numba.py`, the `lta_validation` pattern).
   * Its own physics code: umbrella windows on ξ, MC (per-guest bead and rigid moves, plus K > 1
     guest–guest energy) as the arbiter, EM and LM candidates.
   * 5000-step MC pre-relaxation, no clamps.
   * Fine-bin WHAM and pooled MF route, 16-group jackknife, the `compare`/`level` gates.
8. **Metrics.** A new `eri_metrics` module in the `eqb_family.py` style:
   * system, scorer, far state, reference paths and NB;
   * every statistic delegated to the unchanged `eqb_metrics` functions.

### 5.4 CPU-numba vs GPU-torch (observations in the repo, then estimates for ERI)

**Measured.**

| measurement | value | where |
|---|---|---|
| torch LTA on GPU 3, R = 16 × N = 1024, 300 001 steps, histogram ABF | 1305.9 s, i.e. **3.76e6 molecule-steps/s** (0.27 µs) | `results/lta_histogram/logs/production_T300.log` |
| same, fr_uniform | 1418.2 s (+8.6 %) | same log |
| numba serial LTA, any N | **0.86–0.91 µs per molecule-step** (≈ 1.1e6/s per core), linear in N; a 3.072e8 arm takes ≈ 5 min on one core | `docs/equal_budget/IMPLEMENTATION_VALIDATION.md:81-82`; `docs/equal_budget/LTA_TIMESTEP_VALIDATION.md:90` |
| numba walker-parallel LTA 150 K, N = 512 | 1.072e7 walker-steps/s at 16 threads = 9.13× serial (1.175e6); counter mode at 1 thread 0.37–0.87× serial | `docs/final_campaign/EXECUTION_LOG.md:59` |
| torch at small N | dispatch-bound (gateway: ≈ 0.9 ms per step regardless of N, so N = 1 would take 50 h) | `src/lta_ladder_numba.py:8-9`; `docs/GATEWAY_REPLICA_LADDER.md:59-61` |
| numba FR overhead, LTA | a ≈ 1.0 µs + 41 ns per walker per opportunity | `docs/equal_budget/FR_OVERHEAD.md:15` |
| GPU neighbour lists | methane: rejected, slower (box 2.5 cutoffs across); C60: physics-identical `SubsetNeighborList` with overflow-raise and drift-raise, but a measured negative because H libration forced rebuild storms | `docs/V2_PREREGISTRATION.md:1662-1667`, `:2746-2775`; `src/methane/nonbonded.py:34-39`; `src/c60/nonbonded.py:241-252` |

**Estimates for ERI dilute (K = 1).** These assume LTA's ≈ 1.1–1.2 ns per bead–O pair term (all overheads included).

| implementation | cost |
|---|---|
| numba dense | ≈ 6 µs per molecule-step |
| numba voxel lists (≈ 560 pair terms) | ≈ 0.7–1 µs per molecule-step |
| torch dense on GPU (5184 pair terms, ∝ LTA's 768) | ≈ 5.6e5 molecule-steps/s, with ≈ 2 GB per (B, 2, n_O, 3) f64 intermediate at B = 16 384 |

What follows from these numbers:
* **One CPU core with voxel lists would be about 2.5× the dense GPU engine.**
* An LTA-sized equal-budget arm (B = 3.072e8) would cost ≈ 5 min per core with voxel lists, ≈ 30 min dense.
* The LTA MC/EM validation cost ≈ 31 core-h per temperature (`docs/numerical_validation/LTA_VALIDATION.md:12`).
  A dense ERI version would be ≈ 210 core-h per temperature, more than the whole campaign ceiling of 200
  CPU core-h (`docs/eri_campaign/STATUS.md`). **Fast exact pair search is mandatory for the reference, not
  optional.**

**Recommendation.**
* Use the CPU numba path: deterministic pairing, linear in N, checkpointed, and validated FR selection
  code. Use ProcessPool over (seed, N, method) jobs with single-threaded workers, plus the walker-parallel
  backend pattern if wall-clock matters.
* Use torch on GPU only if a large R × N production needs it (24 GPU-h ceiling). If so:
  * f64 or deterministic f32;
  * deterministic binning;
  * the determinism flags;
  * fixed-capacity candidate lists.

---------------------------------------------------------------------------------------------------------

## 6. Known footguns (each one has cost this project before)

1. **CUDA `scatter_add_` atomics silently unpair the arms.**
   * The LTA torch engine bins with `scatter_add_` (`src/alkanes/periodic.py:44-59`). A CUDA re-run "is not
     bitwise identical ... (scatter-add order)" and the endpoints drift (`docs/LTA_HISTOGRAM_REPLICATION.md:399-402`).
   * Fix pattern: one-hot `scatter_` + fixed-order sum (`src/zif8/core_zif8.py:564-583`).
2. **`torch.compile` default CUDA reductions are nondeterministic** (f32 6.1e-05 per call).
   * Fix: `torch.use_deterministic_algorithms(True, warn_only=True)` + `torch._inductor.config.deterministic = True`
     (`src/zif8/core_zif8.py:90-102`).
   * CHA compiles on CUDA without these flags (`src/cha/core_cha.py:135-138`).
3. **float32 trap: `1/r2.clamp_min(1e-12)` → `sr6²` = inf, and `0·inf` = NaN.**
   * Floor r² at a physical value (1e-4 Å², `src/zif8/core_zif8.py:377`).
   * LTA's EPS clamp (`core_lta.py:176`) is safe only in f64.
4. **numba `fastmath reassoc` is bitwise only per compile target.**
   * Pin one host CPU (`znver4` here), record `compile_target`, and run both arms of a pair on one target
     (`src/lta_ladder_numba.py:54-62`).
5. **numba on-disk cache keys only the defining file.** Importing numba functions across files can leave
   stale machine code after an edit (`src/parallel/lta_parallel.py:177-238`).
6. **`pkill -f` / `pgrep -f` self-match** the calling shell or a harness wrapper.
   * Use a character class, as `scripts/launch_zif8_z5_finish.sh:8` does (`'run_zif8_o[t].py...'`).
   * Documented at `docs/V2_PREREGISTRATION.md:1518-1522`.
7. **Torch noise depends on R.** All R seeds share one generator (`core_lta.py:347`, `:521`), so a
   different R changes every seed's trajectory. Seed labels are not RNG seeds.
8. **Torch FR cap.** `int(0.02 N)` turns FR off below N = 50 (`src/alkanes/core.py:131,137`). The numba
   ladder adds cap_min = 1 (`src/lta_ladder_numba.py:48-49`).
9. **ESS units differ.** Torch reports ESS in walkers (`alkanes/core.py:172`); numba reports ESS/N
   (`lta_ladder_numba.py:428`).
10. **Defaults are not the campaign values.**
    * `LTASimConfig` has fr_rate 0.10, fr_start 40 000, kernel estimator; the ladder `DEFAULT_CFG` has fr_rate 0.2.
    * Build from the frozen config (`scripts/run_lta_histogram.py:73-83`).
11. **Non-gradient clamps in dynamics.**
    * `core_lta.run_umbrella` (`:647-655`) and `core_cha.run_sampler` (`:374-381`) clamp the force norm early on.
    * The WCA campaign was refuted by a non-gradient clip plus an unstable dt (`docs/numerical_validation/WCA_FORCE_AUDIT.md`).
    * Use exact MC pre-relaxation instead (`src/lta_validation.py:141-143`).
12. **Out-of-range CV samples are clamped into the edge bins** in the non-periodic estimators
    (`src/cha/core_cha.py:236-238`; `src/eb_abffr_core.py:324`).
13. **Applied periodic bias is not mean-corrected** (§2.1 item 5).
14. **The f_loc deposit clip (±480 kJ/mol/rad)** can silently bias the estimator. Record clip-hit counts in ERI.
15. **WHAM gauge with empty bins.** `wham_1d` and `F_from_p180` centre over all bins, so an empty bin
    (p → 1e-300) shifts the gauge (`core_lta.py:702-703`; `analyze_lta_validation.py:87-88`). Assert there
    are no empty bins.
16. **CIF parsing by `eval()`** in both builders (§5.3 item 1). Do not run on downloaded files.
17. **Editing a hashed analysis file stales committed results** (`eqb_metrics.py`, §4.3). Extend, never edit.
18. **Sign of the region coordinate.** z is measured from the WINDOW plane, not the cage centre. One
    version inverted it (`scripts/numerical_validation/analyze_lta_validation.py:148-153`). Define ERI's
    regions in the builder and assert them.

---------------------------------------------------------------------------------------------------------

## 7. Open issues handed to the coordinator

* **Box (§5.1).** The stated box's z = 44.4 Å is inconsistent with 3c for both c values in the downloaded
  files (15.05 Å RASPA, 15.175 Å IZA). Its orthorhombic setting also differs from the RASPA example's
  triclinic 3×3×3 cell. This needs resolving before the builder is written.
* **CV periodicity.** A periodic CV (LTA machinery unchanged) needs a straight cage–window–cage repeat along
  a box axis. Otherwise the CV is non-periodic and needs a declared confinement or out-of-range rule.
* **Missing pieces in the numba path.** It lacks a sham arm, an estimated/oracle target and a non-periodic
  score; ERI needs at least the sham.
* **Reference cost.** Fast exact pair search (voxel lists) is required to keep the MC reference inside 200 core-h.
* **Force-field values.** The CH3–O and CH3–CH3 parameters, truncation/shift and blocking must come from the
  literature extraction. The repo's LTA (93 K / 3.48 Å direct cross term) and CHA (Lorentz–Berthelot with
  O 53 K / 3.30 Å) sets differ from each other.
