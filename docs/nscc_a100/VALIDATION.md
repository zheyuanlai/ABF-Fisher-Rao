# A100 engine validation (Stage I, V4 law check, Amendment A1)

Generated tables: `results/nscc_a100/analysis/validation_tables.md` (Stage I) and `law_tables.md` (V4). Records:
`results/nscc_a100/validation/{gateway,lta150,lta300}_validation.json`, `amendment1_check.json`, and
`results/nscc_a100/analysis/law_check.json`.

## 1. What is compared

The A100 engines (`src/gpu/`) run the validated algorithm of the CPU engines with the CPU engines' own random
inputs:
* the initial conditions (`init_left`; PCG64 child 0);
* the FR uniform blocks: numpy PCG64 reproduced in C++, checked bitwise;
* counter-mode Philox4x64-10 noise with the same keys and counter map.

Expected differences are ulp-level:
* CUDA libm vs glibc;
* the LJ force-sum order (LTA);
* tree sums in the LTA FR means;
* deposits rounded to a fixed-point grid (gateway 2^-34, LTA 2^-43) and then summed exactly.

The deposit rounding makes the histogram deterministic and independent of the launch configuration.

## 2. Stage I gates on the frozen code (2026-10-26 06:47-06:48 UTC): all PASS

| gate | gateway | LTA 300 K | LTA 150 K |
|---|---|---|---|
| RNG1: FR stream (2^20 doubles) | bitwise | bitwise | bitwise |
| RNG2: Philox normals vs CPU | max abs diff 4.4e-16 | 4.4e-16 | 4.4e-16 |
| FORCE: GPU vs numba `_mol_force`, G = 8 / 32 / 128 / 256 | -- | max rel 1.9e-16, bins identical | max rel 2.1e-16, bins identical |
| FR: selection on identical inputs | 600 opportunities (218 with events): 0 mismatches | 600 (180): 0 | 600 (181): 0 |
| TRAJ: GPU vs CPU counter mode | 4 runs: bin counts identical over the whole run, dM <= 6.4e-11, deaths equal | 4 runs: counts identical whole run, dM <= 1.4e-11 (first quarter), deaths equal | 4 runs: counts identical whole run, dM <= 5.0e-11, deaths equal |
| INV: launch configurations | 8 configurations bitwise identical (incl. helper warps) | 4 (walkers per block) bitwise identical | 4 bitwise identical |
| DET: repeat | bitwise | bitwise | bitwise |
| CTR: result contract | pass | pass | pass |

## 3. What the gates caught during development (before the freeze)

* **An LTA race condition.** The leader thread updated the walker's positions in shared memory. For groups of more
  than 32 threads per walker, the other warps of the group read them in the next force evaluation without a barrier.
  * The TRAJ gate showed it: configurations with G > 32 decoupled from the CPU within 50-100 steps, while every G <= 32
    configuration stayed coupled.
  * After adding the barrier, every TRAJ run of every configuration is coupled over the whole run.
  * An earlier, coarser save grid had hidden the failure as an apparent divergence at about 1 400 steps. It was
    first misattributed to chaos.
* **A tolerance question (LTA 300 K).** With the strict 1e-9 accumulator tolerance, LTA 300 K at N = 16 failed while
  bin counts and FR events were identical. The ulp-level force-sum differences grow in the coordinates that do not
  enter the CV.
  * The tolerance was set to 1e-6 for both systems, before the freeze, and is documented in the preregistration.
  * With the race fixed, the frozen-code LTA 300 K TRAJ runs pass at dM <= 1.4e-11 over the first quarter.
* **Performance defects, all timing-only and each re-validated:**
  * 64-bit shared-memory atomics compiled to CAS loops (replaced by 32-bit limbs);
  * warp-aggregated deposits that serialise when the walkers spread over many bins;
  * register spills at a 64-register launch bound;
  * the noise computed in the leader's warp;
  * a non-unrolled force reduction.

## 4. V4 law check (full budgets, 16 validation seeds per arm): PASS

GPU vs CPU P_t on the same seeds; ratio of medians GPU / CPU with a paired bootstrap 95 % CI.

| system | arm | final e_F | Ibar_F | u_mid |
|---|---|---|---|---|
| gateway | abf64 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| gateway | fr64 | 1.06 [0.57, 1.43] | 1.01 [0.81, 1.24] | 1.00 [0.80, 1.30] |
| gateway | abf512 | 0.995 [0.989, 1.001] | 1.00 [1.00, 1.00] | 0.99 [0.98, 1.00] |
| gateway | fr512 | 1.27 [0.89, 1.47] | 1.01 [0.95, 1.03] | 1.00 [0.86, 1.05] |
| LTA 300 K | abf64 | 1.07 [0.87, 1.21] | 1.03 [0.95, 1.13] | 1.16 [0.90, 1.43] |
| LTA 300 K | fr64 | 0.76 [0.61, 1.03] | 0.88 [0.71, 1.05] | 0.84 [0.49, 1.94] |
| LTA 300 K | abf256 | 0.86 [0.74, 1.22] | 0.99 [0.95, 1.02] | 1.01 [0.69, 1.12] |
| LTA 300 K | fr256 | 0.90 [0.72, 1.34] | 0.99 [0.90, 1.12] | 0.93 [0.54, 1.31] |
| LTA 150 K | abf64 | 0.97 [0.76, 1.25] | 1.02 [0.81, 1.08] | 1.07 [0.89, 1.27] |
| LTA 150 K | fr64 | 1.26 [0.81, 1.69] | 1.05 [0.88, 1.26] | 1.11 [0.71, 2.00] |
| LTA 150 K | abf256 | 0.98 [0.91, 1.09] | 1.00 [1.00, 1.01] | 0.98 [0.98, 1.04] |
| LTA 150 K | fr256 | 1.59 [0.94, 1.85] | 1.03 [0.97, 1.07] | 1.01 [0.89, 1.13] |

* Every CI contains 1.
* The gateway ABF runs stay essentially coupled to the CPU over the full budgets. For example, u_mid is identical on
  100 % (abf64) and 94 % (abf512) of the seeds.
* Once FR events have occurred, the GPU and CPU runs decorrelate, and the comparison becomes a comparison of laws.

## 5. Amendment A1 (FR path), timing only: PASS

The 7 law-check jobs below were re-run on the amended engines and reproduce the frozen-code records' science sha256
bitwise.

| system | arm | wall (s): frozen code -> A1 |
|---|---|---|
| gateway | fr64 | 33.2 -> 31.4 |
| gateway | fr512 | 8.27 -> 7.95 |
| LTA 150 K | fr64 | 26.9 -> 23.2 |
| LTA 150 K | fr256 | 8.11 -> 7.32 |
| LTA 300 K | fr64 | 60.2 -> 51.9 |
| LTA 300 K | fr256 | 18.2 -> 16.2 |
| LTA 150 K | abf64 | 16.26 -> 16.24 (unchanged, as expected) |
