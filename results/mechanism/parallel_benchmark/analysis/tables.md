# Experiment III benchmark: 2026-10-10T19:01:26Z

Root: `/home/zheyuanlai/ABF-Fisher-Rao/results/mechanism/parallel_benchmark`

Equivalence rule: reference = production per-seed values minus seeds sharing randomness with the backend runs; R1/R2: two-sample permutation test of the median Ibar_F difference per arm (20 000 relabellings); R3/R4: two-sided Mann-Whitney U per arm; R5: permutation test of the median paired G(Ibar_F) difference; Holm over R1-R5 at 0.05 -> CONSISTENT / INCONSISTENT, only with >= 8 complete backend pairs (else INCOMPLETE); minimum detectable shift (80 % power) reported with every verdict

W1 rule: W1 on one backend = the paired FR / ABF wall-clock time-to-accuracy ratio (median over both-finite pairs + seed-bootstrap 95 % CI, censoring-aware rank part) computed ONLY on the pairs in which NEITHER run was SMT-contended (each run pinned to one logical CPU, sibling busy <= 0.10 of its timed run, telemetry present); excluded pairs (contended or unknown) are counted; the same ratio on ALL pairs is reported as a sensitivity analysis and AGREES iff its median lies inside the primary 95 % CI; with no uncontended pair W1 is NOT EVALUABLE on that backend.  The same primary / all-pairs split is applied to the FR overhead and to the effective parallelism.

## gateway (mid threshold 0.0055)

* gpu_torch: **NOT TESTED** (no runs)
* cpu_numba_threads: **NOT TESTED** (no walker-parallel numba build exists (none written): NOT TESTED)

| backend | arm | n | budget frac | tau t | tau force evals | tau wall (s) | censored | Ibar_F | wall total (s) | us/step | warm-up (s) | SMT contended / unknown runs | host RSS max (MB) | GPU device max (MB) | GPU occupancy | bitwise vs prod |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cpu_numba_1core | abf | 8 | 1 | 38 | 7.78e+08 | 60.2 | 0 | 0.00779 | 246 | 38.5 | 0.181 | 0 / 0 | 166 | -- | n/a | equal |
| cpu_numba_1core | fr | 8 | 1 | 16 | 3.28e+08 | 24.6 | 0 | 0.00559 | 247 | 38.6 | 0.183 | 0 / 0 | 168 | -- | n/a | equal |

* **cpu_numba_1core**
  FR/ABF ratio, all pairs (median [95 % CI], n both finite; wins-losses-ties, sign p): tau_t 0.353 [0.298, 0.578] n=8 (8-0-0, p=0.00781); tau_fe 0.353 [0.298, 0.578] n=8 (8-0-0, p=0.00781); tau_wall 0.35 [0.299, 0.559] n=8 (8-0-0, p=0.00781)
  **W1** (wall, uncontended pairs): 0.35 [0.299, 0.559] n=8 (8-0-0, p=0.00781); 0 of 8 pairs excluded; all-pairs sensitivity 0.35, agrees: True
  FR overhead 0.548 % [0.0613, 1.17] (n=8; pairs without SMT contention: 0.548 % [0.0613, 1.17], n=8); G(Ibar_F) -0.283
  resources: timed 3969.3 core-seconds, allocated 3988.9
  Ibar_F / tau_t vs the equal-budget production summary: EQUAL

* effective parallelism (1 CPU core / 1 GPU wall, equal steps): NOT TESTED (needs runs of both cpu_numba_1core and gpu_torch)

## lta300 (mid threshold 0.094)

* cpu_numba_threads: **NOT TESTED** (no walker-parallel numba build exists (none written): NOT TESTED)

| backend | arm | n | budget frac | tau t | tau force evals | tau wall (s) | censored | Ibar_F | wall total (s) | us/step | warm-up (s) | SMT contended / unknown runs | host RSS max (MB) | GPU device max (MB) | GPU occupancy | bitwise vs prod |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cpu_numba_1core | abf | 8 | 1 | 56.1 | 1.44e+08 | 122 | 0 | 0.229 | 262 | 436 | 0.194 | 1 / 0 | 258 | -- | n/a | equal |
| cpu_numba_1core | fr | 8 | 1 | 48.3 | 1.24e+08 | 107 | 0 | 0.201 | 266 | 443 | 0.193 | 0 / 0 | 263 | -- | n/a | equal |
| gpu_torch | abf | 8 | 1 | 58.8 | 1.51e+08 | 198 | 0 | 0.229 | 410 | 684 | 0.271 | 0 / 0 | 1.02e+03 | 654 | exclusive | not applicable (torch backend: different random streams) |
| gpu_torch | fr | 8 | 1 | 48.3 | 1.24e+08 | 203 | 0 | 0.19 | 512 | 853 | 0.335 | 0 / 0 | 1.16e+03 | 752 | exclusive | not applicable (torch backend: different random streams) |

* **cpu_numba_1core**
  FR/ABF ratio, all pairs (median [95 % CI], n both finite; wins-losses-ties, sign p): tau_t 0.801 [0.652, 1.1] n=8 (6-2-0, p=0.289); tau_fe 0.801 [0.652, 1.1] n=8 (6-2-0, p=0.289); tau_wall 0.814 [0.662, 1.11] n=8 (6-2-0, p=0.289)
  **W1** (wall, uncontended pairs): 0.755 [0.662, 1.11] n=7 (5-2-0, p=0.453); 1 of 8 pairs excluded; all-pairs sensitivity 0.814, agrees: True
  FR overhead 1.55 % [1.12, 1.61] (n=8; pairs without SMT contention: 1.58 % [1.43, 1.61], n=7); G(Ibar_F) -0.108
  resources: timed 4226.5 core-seconds, allocated 4246.3
  Ibar_F / tau_t vs the equal-budget production summary: EQUAL
* **gpu_torch**
  FR/ABF ratio, all pairs (median [95 % CI], n both finite; wins-losses-ties, sign p): tau_t 0.857 [0.469, 1.57] n=8 (4-2-2, p=0.688); tau_fe 0.857 [0.469, 1.57] n=8 (4-2-2, p=0.688); tau_wall 1.04 [0.558, 1.97] n=8 (4-4-0, p=1)
  **W1** (wall, uncontended pairs): 1.04 [0.558, 1.97] n=8 (4-4-0, p=1); 0 of 8 pairs excluded; all-pairs sensitivity 1.04, agrees: True
  FR overhead 24.9 % [22.9, 26] (n=8; pairs without SMT contention: 24.9 % [22.9, 26], n=8); G(Ibar_F) -0.118
  resources: timed 7378.8 GPU-seconds, allocated 7507.2
  equivalence vs numba production: **CONSISTENT**
  minimum detectable shift (80 % power, per-test alpha 0.01 / 0.05): abf 21.2 / 17.3 % of the reference median; fr 32.3 / 26.5 % of the reference median; G_Ibar_F 0.1 / 0.0821 (absolute G)
  early-window consistency vs production (EXPLORATORY, not preregistered): **NOT APPLICABLE (needs shortened-budget runs of ONE n_steps)**

* effective parallelism (1 CPU core / 1 GPU wall, equal steps): abf 0.638 [0.628, 0.646] (n=8, 1 SMT-contended; uncontended 0.636, n=7); fr 0.52 [0.518, 0.522] (n=8, 0 SMT-contended; uncontended 0.52, n=8)

## lta150 (mid threshold 0.13)

* cpu_numba_threads: **NOT TESTED** (no walker-parallel numba build exists (none written): NOT TESTED)

| backend | arm | n | budget frac | tau t | tau force evals | tau wall (s) | censored | Ibar_F | wall total (s) | us/step | warm-up (s) | SMT contended / unknown runs | host RSS max (MB) | GPU device max (MB) | GPU occupancy | bitwise vs prod |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cpu_numba_1core | abf | 8 | 1 | 69.9 | 1.79e+08 | 152 | 0 | 0.281 | 261 | 436 | 0.193 | 1 / 0 | 258 | -- | n/a | equal |
| cpu_numba_1core | fr | 8 | 1 | 24.9 | 6.37e+07 | 55 | 0 | 0.184 | 265 | 442 | 0.194 | 2 / 0 | 263 | -- | n/a | equal |
| gpu_torch | abf | 5 | 1 | 61.8 | 1.58e+08 | 212 | 0 | 0.274 | 409 | 681 | 0.287 | 0 / 0 | 1.02e+03 | 654 | exclusive | not applicable (torch backend: different random streams) |
| gpu_torch | fr | 6 | 1 | 20.4 | 5.22e+07 | 83.1 | 0 | 0.173 | 514 | 857 | 0.352 | 0 / 0 | 1.16e+03 | 752 | exclusive, shared | not applicable (torch backend: different random streams) |

* **cpu_numba_1core**
  FR/ABF ratio, all pairs (median [95 % CI], n both finite; wins-losses-ties, sign p): tau_t 0.341 [0.318, 0.451] n=8 (8-0-0, p=0.00781); tau_fe 0.341 [0.318, 0.451] n=8 (8-0-0, p=0.00781); tau_wall 0.347 [0.321, 0.456] n=8 (8-0-0, p=0.00781)
  **W1** (wall, uncontended pairs): 0.361 [0.327, 0.518] n=6 (6-0-0, p=0.0312); 2 of 8 pairs excluded; all-pairs sensitivity 0.347, agrees: True
  FR overhead 1.56 % [1.47, 3.37] (n=8; pairs without SMT contention: 1.51 % [1.44, 1.72], n=6); G(Ibar_F) -0.334
  resources: timed 4254.3 core-seconds, allocated 4274.4
  Ibar_F / tau_t vs the equal-budget production summary: EQUAL
* **gpu_torch**
  FR/ABF ratio, all pairs (median [95 % CI], n both finite; wins-losses-ties, sign p): tau_t 0.376 [0.246, 0.583] n=5 (5-0-0, p=0.0625); tau_fe 0.376 [0.246, 0.583] n=5 (5-0-0, p=0.0625); tau_wall 0.449 [0.283, 0.713] n=5 (5-0-0, p=0.0625)
  **W1** (wall, uncontended pairs): 0.449 [0.283, 0.713] n=5 (5-0-0, p=0.0625); 0 of 5 pairs excluded; all-pairs sensitivity 0.449, agrees: True
  FR overhead 25.5 % [22.9, 28.1] (n=5; pairs without SMT contention: 25.5 % [22.9, 28.1], n=5); G(Ibar_F) -0.359
  resources: timed 5137.7 GPU-seconds, allocated 5226.3
  equivalence vs numba production: **INCOMPLETE (5 complete pairs < 8: tests reported, no verdict)**
  minimum detectable shift (80 % power, per-test alpha 0.01 / 0.05): abf 16.9 / 13.9 % of the reference median; fr 20.1 / 16.5 % of the reference median; G_Ibar_F 0.0905 / 0.0742 (absolute G)
  early-window consistency vs production (EXPLORATORY, not preregistered): **NOT APPLICABLE (needs shortened-budget runs of ONE n_steps)**

* effective parallelism (1 CPU core / 1 GPU wall, equal steps): abf 0.64 [0.627, 0.648] (n=5, 0 SMT-contended; uncontended 0.64, n=5); fr 0.517 [0.511, 0.523] (n=6, 1 SMT-contended; uncontended 0.517, n=5)

## Full-campaign projection (per-step wall x full budget x seeds)

* cpu_numba_1core: 3.46 core-h for 8 seeds x 2 arms
  * gateway/abf: 38.5 us/step x 6400000 steps + 1 s = 248 s/run [measured (8 runs); host SMT uncontended]
  * gateway/fr: 38.6 us/step x 6400000 steps + 1 s = 249 s/run [measured (8 runs); host SMT uncontended]
  * lta300/abf: 436.0 us/step x 600000 steps + 1 s = 263 s/run [measured (8 runs); host SMT contended, uncontended]
  * lta300/fr: 442.6 us/step x 600000 steps + 1 s = 267 s/run [measured (8 runs); host SMT uncontended]
  * lta150/abf: 435.6 us/step x 600000 steps + 1 s = 263 s/run [measured (8 runs); host SMT contended, uncontended]
  * lta150/fr: 442.2 us/step x 600000 steps + 1 s = 267 s/run [measured (8 runs); host SMT contended, uncontended]
  * plan Exp III CPU estimate 20 core-h: within: True
* gpu_torch: 21.04 GPU-h (one device, runs sequential) for 8 seeds x 2 arms (some rows from the harness fallback costs, not this root)
  * gateway/abf: 590.0 us/step x 6400000 steps + 20 s = 3796 s/run [fallback (FALLBACK_US_PER_STEP); host SMT contended]
  * gateway/fr: 590.0 us/step x 6400000 steps + 20 s = 3796 s/run [fallback (FALLBACK_US_PER_STEP); host SMT contended]
  * lta300/abf: 684.0 us/step x 600000 steps + 8 s = 418 s/run [measured (8 runs); host SMT uncontended]
  * lta300/fr: 853.0 us/step x 600000 steps + 8 s = 520 s/run [measured (8 runs); host SMT uncontended]
  * lta150/abf: 681.1 us/step x 600000 steps + 8 s = 416 s/run [measured (5 runs); host SMT uncontended]
  * lta150/fr: 857.2 us/step x 600000 steps + 9 s = 523 s/run [measured (6 runs); host SMT uncontended]
  * plan GPU ceiling 8 GPU-h (Exp III estimate 3): total within ceiling: False
    * gateway: 16.87 GPU-h -> equivalence gate CANNOT BE EVALUATED within the plan's 8 GPU-h ceiling (its 8 x 2 full-budget runs alone need 16.9 GPU-h)
    * lta300: 2.08 GPU-h -> equivalence gate evaluable within the GPU ceiling
    * lta150: 2.09 GPU-h -> equivalence gate evaluable within the GPU ceiling
