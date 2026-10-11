# Experiment B — walker-parallel CPU backend: design brief

*2026-10-11. Implementation brief for `PREREGISTRATION.md` §2.2–2.3. Engineering detail only; every scientific rule
(validation gates, feasibility thresholds, endpoints) is in the preregistration and is not restated or changed here.*

## 1. What must be identical to the validated engines

### Gateway (validated engine: `src/gateway_family_numba.py` at variant α = 1, λ = 1 = MODEL_ORIG, `lam1`; bitwise the equal-budget engine `src/gateway_ladder_numba.py`)

Per loop step `step` (completed-step count of the deposited state), `_advance` lines 939–1130:
1. N normals `zx[i]`, then N normals `zy[i]` (numba MT19937, `np.random.standard_normal()`).
2. **Deposit, all walkers, slot order**: `e = exp(-x*x/two_s2)`, `om = oout + dom_amp*e`, `dom = -dom_amp*(x/s2)*e`,
   `f = 4.0*H*x*(x*x - 1.0) + om*dom*y*y`, `fy = om*om*y`, `j = int(floor((x - XMIN)/delta + 1e-9))` clamped,
   `Ch[j] += 1.0; Mh[j] += f`. `ictr[I_NFE] += N`.
3. **Move, all walkers** (reads the table that already holds all of this step's deposits): `den = Ch[j] + min_count`,
   `bias = Mh[j]/den` (0 if den ≤ 0), running max |bias|, `X = _reflect(X + (-fx + bias)*h + amp*zx, XMIN, XMAX)`,
   `Y = Y - fy*h + amp*zy` with `amp = sqrt(2h/β)`.
4. **FR** if `is_fr and N >= 2 and step % fr_every == 0` (step 0 included): `g = γ(1 − exp(−step/ramp_steps))`,
   `kde_density`, `uniform_scores`, `fr_resample` with the opportunity's fixed block of `L = N + cap + max(N, cap)`
   uniforms from `PCG64(SeedSequence([seed, N, 1]))`, offspring counts, gather X and Y, death counters, histograms,
   cap check.
5. Save if `save_at[sptr] == step + 1` (after FR): M_all, C_all, FR counters.

Floating-point expressions must be evaluated in exactly this order (numba evaluates left to right; no FMA
contraction; glibc `exp`). `_reflect` uses Python float-modulo semantics: `r = fmod(a, b); if r != 0 and
((r < 0) != (b < 0)): r += b` (and `r = copysign(0, b)` when r == 0).

### LTA 150 K (validated engine: `src/lta_ladder_numba.py`, `_run_chunk`, diagnostics off, native FR mode 0)

Per evaluated step k: (1) forces/CV/f/bin of all walkers at pre-move q (`_mol_force`, `_wrap`, `_bin`, deposit clip);
(2) per-step staged bin sums in slot order (`touched` order), then added to all-steps (and production, k ≥ burn-in)
accumulators; (3) bias from the all-steps accumulators incl. this step (ramp, ±60 clip, ·π/a); (5) save at
`save_steps[sp] == k`; (6) EM move with the [step][replica][bead][xyz] noise; (7) FR at `nxt = k + 1 ≥ fr_start` with
`(nxt − fr_start) % fr_every == 0` on the post-move CV: `_fr_score`, `fr_select_native` with the opportunity's block of
`N + 2 cap` uniforms (child 2 of `SeedSequence([seed, N, 150])`), copies.

## 2. Parallel decomposition

| phase | gateway | LTA |
|---|---|---|
| noise | per walker (counter) / pre-filled buffer (replay) | per walker (counter) / Python PCG64 array (replay) |
| force, CV, bin | parallel over walkers (static partition) | parallel over walkers (`prange`), ≈ 87 % of the work |
| deposit | **bin-owner parallel**: thread t owns a contiguous bin range and scans all N walkers in slot order — bitwise the serial summation order, any thread count | serial, slot order (N adds) |
| bias read + move | parallel over walkers | parallel over walkers |
| FR | serial, the validated numba functions | serial, the validated numba functions |
| save | serial | serial |

Gateway: one OpenMP parallel region per kernel call (a call runs the steps up to the next FR opportunity or save);
per step two barriers (after force/bin, after deposit); the move of step k and the force of step k + 1 for the same
walker run on the same thread. Build: `gcc -O3 -fopenmp -fno-fast-math -ffp-contract=off -fPIC -shared`, loaded with
`ctypes`; `OMP_WAIT_POLICY=active`, `OMP_PROC_BIND=close`, `OMP_PLACES` = the allocated cores. LTA: `numba.njit(
parallel=True)`, threading layer `omp`, `numba.set_num_threads(t)`.

## 3. Counter-mode noise

Philox4x64-10 (Random123 constants M0 = 0xD2E7470EE14C6C93, M1 = 0xCA5A826395121157, W0 = 0x9E3779B97F4A7C15,
W1 = 0xBB67AE8584CAA73B), key = two 64-bit words from `SeedSequence([seed, N, 2])` (gateway) or
`SeedSequence([seed, N, 150, 2])` (LTA) `.generate_state(2, np.uint64)`, counter = (global step, slot, word, 0).
Outputs o0..o3 → uniforms u = (o >> 11) · 2⁻⁵³; Box–Muller on (u0, u1) and (u2, u3): r = sqrt(−2 log(1 − u_a)),
θ = 2π u_b, z = (r cos θ, r sin θ). Gateway: word 0 → (zx, zy) from (u0, u1). LTA: words 0 and 1 → 8 normals, the
first 6 used in [bead][xyz] order. A clone continues with its slot's stream (as in replay mode). numpy's
`np.random.Philox(key=..., counter=...)` is the test oracle for the raw outputs (its counter-increment convention is
determined in the test).

## 4. Outputs

The common contract consumed by `scripts/equal_budget/eqb_metrics.py` scorers: `save_step`, `save_t`, `save_u`,
`M_all`, `C_all` (LTA also `M_prod`, `C_prod`), FR counters (`fr_deaths_cum`, `fr_opp_cum`, …), `n_force_evals`,
`X_final`/`Y_final` (gateway) or `q_final` (LTA), `max_abs_bias` (gateway), plus `wall_s_at_save`,
`n_force_evals_at_save`, `wall_total_s`, `warmup_s`, `t_kernel_s`, `t_fr_s`, `t_save_s`, `threads`, `cpus`,
`noise_mode`, provenance (engine sha256, git commit, compiler flags, library versions). Diagnostics-only keys are
absent.

## 5. Files

`src/parallel/philox.py` (numba), `src/parallel/gw_kernel.c` + `src/parallel/gateway_parallel.py`,
`src/parallel/lta_parallel.py`; tests `tests/test_parallel_backend.py`; harness `scripts/final_campaign/expB_*.py`.
The validated engines are imported, never edited.
