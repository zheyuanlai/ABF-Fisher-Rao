# Stage I: audit and numerical validation

*2026-10-18, 07:13–07:43 UTC.*
* Preregistration: `docs/hybrid_short_T/PREREGISTRATION.md`, frozen at `ef3b6453`; Amendment A1 at `839f7114`.
* Harness: `scripts/hybrid_short_T/validate.py`.
* Records: `results/hybrid_short_T/validation/*.json`; summary in `validation.json`.
* **Verdict: 41/41 checks PASS; no backend is blocked.**

## 1. Environment audit

| item | value |
|---|---|
| git | `0d683ccb` at the start; prereg `ef3b6453`; harness + Stage I `839f7114` |
| Python / numpy / numba / llvmlite | 3.14.4 / 2.4.6 / 0.68.0 / 0.50.0 (conda env `abffr`) |
| scipy / matplotlib | 1.17.1 / 3.10.9 |
| torch | 2.12.0+cu130. Not used in any run of this study |
| C compiler (gateway parallel kernel) | gcc 13.3.0, `-O3 -march=native -fopenmp -fno-fast-math -ffp-contract=off` |
| NVIDIA driver / CUDA | 580.105.08 / 13.0 |
| GPUs | 4 × H200 NVL (143 GB), **all busy at the start**: GPU 0 and 1 `yifanchen` 38 GB + `yesom` 2.4 GB, 100 % util; GPU 2 `hyt35` 24 GB, 81–100 %; GPU 3 `hyt35` 58 GB, 98–100 %. The `hyt35` jobs started 56 min earlier, the `yesom` jobs 51 days earlier |
| GPU decision | user instruction "CPU only" (prereg §1). No CUDA context was created by this study; `CUDA_VISIBLE_DEVICES=""` in every process |
| CPU | 2 × AMD EPYC 9554: 128 cores, SMT-2, 16 CCDs × 8 cores, 2 NUMA nodes. Nominal max 3.1 GHz, boost on. Telemetry shows 3.6–3.7 GHz on loaded cores |
| memory | 1.5 TB (≈ 1.0 TB available) |
| disk | `/` 3.5 TB, 141 GB free (96 % used). The DiskGuard pauses below 2 GB |
| host load | 1-min load ≈ 19–21 from ≈ 20 unpinned single-thread processes of other users (`yifanchen`, `zhengtan`, `yesom`, `hyt35`) roaming over every CCD |

## 2. CPU backends and what differs between them

| aspect | S1 (validated serial engine) | P_t (walker-parallel, counter mode) |
|---|---|---|
| code | `src/gateway_ladder_numba.py`, `src/lta_ladder_numba.py` `run_arm`, diagnostics off, unchanged | `src/parallel/gateway_parallel.py` + `gw_kernel.c` (variant `replicated_dissem`); `src/parallel/lta_parallel.py` (numba `parallel=True`, layer omp), unchanged |
| Langevin noise | gateway: numba MT19937 seeded `default_noise_seed(seed, N)`, polar normals. LTA: PCG64 child 1 of `SeedSequence([seed, N, round(T)])` | Philox4x64-10 keyed by `SeedSequence([seed, N, 2])` (gateway) / `[seed, N, round(T), 2]` (LTA), counter (step, slot, word, 0), Box–Muller. Replay mode reproduces S1's stream exactly (validation only) |
| initial conditions | gateway `init_left(seed, N)` (`default_rng(1000 + seed)`); LTA child 0 | identical code and streams |
| FR uniforms | gateway PCG64 `SeedSequence([seed, N, 1])` in fixed blocks of N + cap + max(N, cap) per opportunity; LTA child 2, N + 2 cap per opportunity | identical |
| floating point | float64. Numba, no FMA contraction. LTA `_lj_bead` has fastmath `reassoc` on its pair sum only | float64. The gateway C kernel uses the same expression order, `-fno-fast-math -ffp-contract=off`. LTA calls the same numba functions. Bitwise S1 in replay mode (V3) |
| ABF deposition | per step, all walkers, slot order | gateway: every thread adds all N deposits in slot order into its private table copy (bitwise the serial sum); LTA: serial, slot order |
| clipping | gateway none (P0, `min_count` 1); LTA bias ±60, deposit ±480 | identical code |
| FR selection / caps | `fr_resample` / `fr_select_native`; caps max(1, ⌊0.08 N⌋) / max(1, ⌊0.02 N⌋) | the same numba functions and caps (V8) |
| arm pairing | ABF and FR share IC and noise slot by slot, identical up to the first realised event (V5) | the same (V5) |
| timing | `SaveClock` hooks split the chunk kernel at saves (`perf_counter`); warm-up run excluded | in-kernel `CLOCK_MONOTONIC` stamps; engine warm-up and the LTA self-check excluded |

### The H200 torch path (documented from code, not run: prereg §1)

`scripts/mechanism/bench_torch_engines.py` drives `src/gateway_core.py` and `src/lta/core_lta.py`. It differs from
the CPU engines in these ways, all verified in the source on 2026-10-18:

* **Noise.**
  * Gateway: `torch.Generator(device).manual_seed(2000 + seed)` for the Langevin noise and `3000 + seed` for FR.
  * LTA: `manual_seed(seed)` for IC and noise and `seed + 987654321` for FR.
  * These are different numbers from the CPU streams, with the same law. Torch-CPU bitwise identity is impossible by
    construction.
* **Initial conditions.**
  * Gateway: identical x0 / y0 (`gateway_core.init_conditions` uses `default_rng(1000 + seed)`, as `init_left`
    does).
  * LTA: torch draws, independent of the numba streams.
* **Reductions.** `scatter_add_` (bin sums, `eb_abffr_core.py` l. 368–369, `alkanes/periodic.py` l. 49–58) uses
  float64 atomics on CUDA, so runs are not bitwise reproducible. ABF and FR arms can diverge before any FR event.
* **Gateway resampling** (`gateway_core.resample_indices` l. 621–700). Every FR opportunity draws random sort keys
  over the survivor + clone pool and gathers the first N. This **permutes all slots at every opportunity**, including
  the ramp's first opportunity where g = 0 and no event fires. The shared noise stream therefore drives different
  walkers in the two arms from step 1 on: the arms are **unpaired**. The numba engines copy in place and stay paired.
* **LTA birth–death** (`alkanes.core._birth_death`): in-place copies, paired until the first event (as numba).
* **Small-N caps.**
  * Torch gateway `floor(0.08 N)` vs numba `max(1, floor(0.08 N))`: they differ at N ≤ 12, where torch has cap 0
    and FR is off.
  * Torch LTA `int(0.02 N)` vs numba `max(1, int(0.02 N))`: they differ at N ≤ 49, where torch FR is silently off.
  * The drivers raise `EquivalenceError` there rather than run a different algorithm.
* **Measured speed (Experiment III, 2026-10-10, eager mode, one H200, N = 512).**

  | system | torch ABF | torch FR | one CPU core |
  |---|---|---|---|
  | LTA 300 K | 684 µs/step | 853 µs/step | 436 µs/step |
  | LTA 150 K | 681 µs/step | 857 µs/step | 436 µs/step |
  | gateway | 590 µs/step (step-cost benchmark only) | | 38.5 µs/step |

  FR overhead on the GPU was about 25 %. No new GPU measurement exists. Figure A shows these points as historical
  markers.

## 3. Gate results

| gate | configurations | result |
|---|---|---|
| **V1** budget / grid | all 34 rungs | **PASS**: n_steps = B/N integer; T_N equal to the short-T table; last save = n_steps; every save is on the original production grid; 87 / 195 / 101 budget-grid saves and 63 / 171 / 77 uniform saves at every N (gateway / LTA 300 K / LTA 150 K) |
| **V2** truncation identity (S1 through this harness vs the stored equal-budget production run, first production seed) | full short budget at N₀ = 2048 / 1024 / 1024, ABF and FR; windows at N 16 (gateway, 2 × 10⁶ steps) and N 4 (LTA, 2 × 10⁵ steps) | **PASS 12/12, bitwise** at every common save: M_all, C_all, FR deaths, and for LTA also M_prod, C_prod. The FR arms realised 2366 / 575 / 478 deaths over the full short budgets. The production runs had diagnostics on (bit-inert) |
| **V3** parallel replay = S1 | gateway N 2048 (20 000 steps), N 64 (2 × 10⁵); LTA 300 K N 1024 (25 000), 256 (40 000), 16 (1.2 × 10⁵); LTA 150 K N 1024; threads 1 / 4 / 16 (t ≤ N); both arms | **PASS 12/12, bitwise** on every science key (accumulators, FR counters and histograms, final states, force-evaluation counts). **First validation of the LTA parallel engine at 300 K** |
| **V4** counter-mode thread invariance | gateway N 2048, LTA 300 K N 256; threads 1 / 8 / 16; both arms | **PASS 4/4, bitwise** |
| **V5** arm pairing | S1 and P8: gateway N 64 (2 × 10⁵ steps), LTA 300 K and 150 K N 1024 (25 000 steps, FR from 20 000) | **PASS 6/6**: identical at every save before FR's first realised death, different from then on. Gateway: 2 saves before the first death, which happens early at N 64; the FR arm realised 54–64 deaths by the end of the window. LTA: 34 saves before it, 34–59 deaths after |
| **V6** hooks / timers inert | gateway N 64, LTA 300 K N 64, FR arm | **PASS 2/2**: S1 with SaveClock hooks bitwise equals plain `run_arm`; P_t with timers on bitwise equals timers off |
| **V7** scorer | 3 files per system: N₀ FR, N 16 ABF, N 2 FR | **PASS 9/9**: this harness's e_F equals the equal-budget metric cache **bitwise** at all 225–237 saves of each file |
| **V8** schedules / caps | static: N 2, 4, 16, 64, N₀; dynamic: LTA N 16 | **PASS**: gateway fr_every 160, ramp 160 000 steps, caps 1/1/1/5/163. LTA caps 1/1/1/1/20 in S1 and P_t; T_K carried to the parallel engine. In both engines cum_opps = 0 at steps 19 998 and 19 999 and 1 at 20 000: the first FR opportunity is the transition to step 20 000 |
| **V9** timing instrumentation | every V4 run | **PASS** (Amendment A1 wording): stamps finite and monotone; warm-up + timed loop ≤ the caller's span; untimed remainder 0.001 s (gateway) and 0.20 s (LTA, first call: self-check) / 0.002 s (later calls). `perf_counter` is `clock_gettime(CLOCK_MONOTONIC)`, the kernel's clock |
| seeds (`check_seeds.py`) | all new blocks | **PASS**: no reuse of 650 earlier gateway and 447 LTA labels; no collision of the 864 new gateway MT noise seeds with 1178 recorded ones; 4192 new SeedSequence states distinct from 35 268 earlier ones |

**Existing test suites** (run 2026-10-18 on this host):
* `tests/test_parallel_gateway.py` 36/36 and `tests/test_final_expB_harness.py` 46/46.
* `tests/test_parallel_lta.py` 37/37 on 16 cores. On 8 cores, 24 tests cannot set 16 threads (environment, not
  engine).
* `tests/test_parallel_benchmark.py` 23 pass, 1 skip, 2 fail. Its GPU guard refuses an unreadable / busy GPU 3,
  which is the intended behaviour.
* `tests/test_hybrid_short_T.py` 19/19.

**Scheduler smoke** (24 benchmark jobs + 3 contention repeats, scratch directory, 0.28 core-h):
* every record complete;
* contended runs repeated, with science sha256 identical across replicates;
* telemetry boundary holds seen.

## 4. Unresolved or only statistically covered

* **Counter-noise law.** Counter mode changes only the noise source. The algorithm is verified exactly (V3 replay =
  S1), and the RNG passed the Experiment B V-B3 battery. Statistical law equivalence:
  * LTA 150 K: established in Experiment B (V-B4 PASS);
  * gateway: the frozen V-B4 gate failed, a documented gate defect: the validated engine failed it against itself
    in 3/3 null controls;
  * LTA 300 K: never tested before.

  This study adds the post hoc law check of prereg §3 (campaign counter-mode runs vs the stored production runs
  truncated at u*; S1-routed rungs are the null control). The check is reported in `RESULTS.md`; it is not a gate.
* **GPU.** Nothing about H200 numerics is newly verified; §2 is code reading.
* **V5 at gateway N 64** has only 2 saves before the first realised death, so its pre-event pairing evidence is thin.
  Experiment B V-B5 checked N 16 / 64 / 512 over longer windows, and V2 / V3 are bitwise identities that subsume it
  for S1.

## 5. Amendment A2.9 additions (2026-10-18 ≈ 08:20–08:27 UTC; records in `results/hybrid_short_T/validation/`)

| gate | configurations | result |
|---|---|---|
| **V2d** truncation identity with the A2.1 dense save grid (552 / 647 saves per run, compared at the 88 / 201 production saves they contain; 2366 / 575 FR deaths) | gateway N 2048 FR and LTA 300 K N 1024 FR, full short budget, first production seed | **PASS 2/2, bitwise** at every save shared with the stored production run (M_all, C_all, deaths; LTA also M_prod, C_prod): the dense saves change no bit |
| **V4b** counter-mode thread invariance at t ∈ {1, 2, 4} | LTA 300 K N 16 and N 64, gateway N 16, both arms | **PASS 6/6, bitwise** |

**Verdict after A2.9: 49/49 checks PASS. No backend is blocked.**
