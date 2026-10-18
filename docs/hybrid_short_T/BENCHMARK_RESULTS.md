# Stage II: CPU throughput benchmark and the frozen routing

*2026-10-18, 07:46–09:54 UTC.*
* Harness: `scripts/hybrid_short_T/benchmark.py`, scheduler `hst_sched.py`.
* Records: `results/hybrid_short_T/bench/<system>/<config>/r<r>_s<seed>[.rep<k>].json` (telemetry sidecars `.npz`,
  gitignored).
* Summary: `bench/summary.json`; full tables: `bench/tables.md`.
* Routing: `results/hybrid_short_T/routing/routing.json`, frozen 09:54:52 UTC at `fe88826c` and committed at
  `62438653` **before any production run**.
* Figure: `figures/hybrid_short_T/A_throughput_crossover.{png,pdf}`.

**Claim types.** All numbers here are **measured timings** on this host, with its state at the time recorded. The
routing is an **identity** given the frozen rule. Production cost figures are **projections**.

## 1. What was measured

**H200.** **Not measured** (prereg §1: all four H200s were busy with other users' jobs; user instruction "CPU only").
Figure A shows the 2026-10-10 Experiment III torch-eager H200 points at N = 512 as historical markers only:
* LTA: 684 µs/step (ABF), 853 µs/step (FR), against 436 µs/step on one CPU core;
* gateway: a 590 µs/step step-cost estimate for ABF only.

**CPU configurations.** 342 configurations × 3 repeats (1026 jobs; 1224 attempts with contention repeats):
* every N of each ladder; ABF, and FR at N ≥ 2;
* S1 (the validated serial engine, one core) and P_t (walker-parallel engine, counter-mode noise, t ∈ {1, 2, 4, 8,
  16}, t ≤ N);
* production configurations truncated to 6.4 × 10⁷ (gateway) or max(1.2 × 10⁷, 25 000 N) (LTA) walker-steps.

**Long-window supplement (Amendment A2.5).** ABF, N ≥ 16, t ∈ {4, 8, 16}, at 5 × 10⁸ / 6 × 10⁷ walker-steps: 66
configurations × 3 repeats = 198 jobs. The routing uses these windows wherever they exist.

**Allocation.** Each run had its own exclusive CCD set and 1 Hz telemetry. Up to 6 CCD sets were in use at a time.

**Cost.** 22.8 used core-h for validation and benchmark together (ledger `results/hybrid_short_T/compute_ledger.csv`).

**Contention** (another user's process > 10 % busy on a run's core or SMT sibling):
* Fraction of attempts contended: t1 32/422, t2 13/199, t4 21/255, t8 39/253, t16 121/301.
* Mechanism: other users' unpinned processes sit one per CCD (the kernel spreads them over L3 domains). A run that
  fills a CCD pushes such a process onto an SMT sibling.
* Where a configuration had both contended and uncontended replicates of one repeat, the contended one was slower by
  a median 0.3 / 1.2 / 2.4 / 3.5 / 0.7 % (t = 1 / 2 / 4 / 8 / 16). Upper quartile ≤ 14.6 %, worst 34 %.
* Every configuration ended with at least one uncontended repeat.
* Allocation policy changes made during the benchmark (A1, A2.4 incl. the 08:58 revision, A2.4b) are in the
  preregistration amendments.

## 2. Throughput (steady-state wall per walker-step, ns; median over uncontended repeats)

| system | N | S1 | best parallel (threads) | frozen route | route speed-up vs S1 |
|---|---|---|---|---|---|
| gateway | 2048 | 75.2 | 5.84 (P16) | P16 | 12.9× |
| | 1024 | 75.0 | 6.51 (P16) | P16 | 11.5× |
| | 512 | 75.5 | 8.87 (P8) | P8 | 8.5× |
| | 64 | 74.9 | 13.95 (P8) | P8 | 5.4× |
| | 32 | 75.1 | 20.9 (P4) | P4 | 3.6× |
| | 16 | 75.5 | 34.7 (P2) | P2 | 2.2× |
| | 8 / 4 / 2 / 1 | 76.3 / 76.5 / 78.7 / 78.3 | 52.3 / 56.2 / 57.7 / 63.1 (P1) | **P1** | 1.46 / 1.36 / 1.36 / 1.24× |
| LTA 300 K | 1024 | 845 | 80.7 (P16) | P16 | 10.5× |
| | 512 | 834 | 93.2 (P16) | P16 | 9.0× |
| | 128 | 846 | 139 (P16) | P16 | 6.1× |
| | 64 | 841 | 177 (P8) | P8 | 4.8× |
| | 8 | 842 | 490 (P8) | P8 | 1.7× |
| | 4 | 848 | 852 (P4) | **S1** | 0.99× |
| | 2 / 1 | 844 / 854 | 1530 (P2) / 2340 (P1) | **S1** | 0.55 / 0.37× |
| LTA 150 K | 1024 / 128 / 64 / 8 / 4 / 1 | ≈ 843–855 | 81.5 / 133 / 178 / 494 / 840 / 2330 | P16 / P16 / P8 / P8 / **S1** / **S1** | 10.3 / 6.3 / 4.8 / 1.7 / 1.0 / 0.37× |

**Crossover (frozen rule: ≥ 5 % faster and every repeat faster).**
* **Gateway: no crossover.** The parallel engine is faster than S1 at every N, already on one thread (P1, 1.24–1.46×).
  - This is an **implementation effect**, not parallelism: a C kernel and counter-mode Philox normals instead of
    numba's MT19937 polar normals (prereg A2.8).
  - Parallelism starts to pay at N = 16 (P2), and the speed-up grows to 12.9× at N = 2048 on 16 cores.
* **LTA (both temperatures): the crossover is between N = 4 and N = 8.** S1 is fastest at N ≤ 4; P8 is 1.7× at N = 8.
  - The numba `parallel=True` engine has a per-step launch cost that dominates at small N: its P1 is 0.37× S1 at
    N = 1.
  - From N = 128 on, P16 gives 6–10.5×.
* **Thread count is rarely "more is better".** The gateway peaks at P8 for N = 64–512 and LTA at P8 for N ≤ 64:
  16 threads lose to the synchronisation and serial phases there.
* Long and short windows agree to within a few per cent, e.g. gateway N 256 P8: 9.80 (long) vs 9.87 (short); LTA
  300 K N 16 P8: 310 vs 309. The routes would have been the same at every N on the short windows alone; the 3 %
  tie rule is never decisive.

**FR overhead per step** (short windows, on the routed backend; median over repeats, c_ss(FR) / c_ss(ABF) − 1):
* Gateway: +4–7 % on P2–P16, +5 % at N 8 (P1), +8 % at N 4, +14.5 % at N 2.
* LTA 300 K: +9–14 % on P8 / P16, +7 % at N 4 (S1), +15 % at N 2 (S1).
* LTA 150 K: +10–12 % on P8 / P16, +7 % at N 4, +14.5 % at N 2.
* On the parallel engines, FR (KDE, score, selection, copies) runs serially while the forces are spread over 8–16
  threads. FR's share of the wall time therefore rises: Amdahl. On one core it was 1–14 % (`docs/equal_budget/
  FR_OVERHEAD.md`). Here it is larger on the parallel routes, but far below the ≈ 25 % of the historical H200 torch
  path.

**Start-up** (process start → timed start, median):
* S1 0.44 s (10–90 %: 0.43–0.48);
* LTA parallel 0.53 s (0.49–0.54; the self-check absorbs the JIT / cache load);
* gateway parallel 1.72 s (1.69–1.74), of which 1.45 s is the engine's own warm-up run.

This is why W_e2e is co-primary (A2.2): gateway W values are seconds.

CPU frequency (telemetry, run cores) was 3.3–3.75 GHz depending on the configuration. It is recorded but not
controlled.

## 3. Frozen routing (common backend for both methods at each N)

| N | 2048 | 1024 | 512 | 256 | 128 | 64 | 32 | 16 | 8 | 4 | 2 | 1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gateway | P16 | P16 | P8 | P8 | P8 | P8 | P4 | P2 | P1 | P1 | P1 | P1 |
| LTA 300 K | — | P16 | P16 | P16 | P16 | P8 | P8 | P8 | P8 | S1 | S1 | S1 |
| LTA 150 K | — | P16 | P16 | P16 | P16 | P8 | P8 | P8 | P8 | S1 | S1 | S1 |

* The deployment-optimised (per-method) routes equal the common routes at every N (`routing.json` `own`).
* Projected production cost (prod_seeds, expected repeats at E = 1 + p): 161 used core-h. Per seed block: gateway
  1.39, LTA 300 K 4.61, LTA 150 K 2.66.
* **Timing-model check on the 130-job production smoke stage.** Completion wall / (B × benchmark c_ss):
  - median 1.000 (range 0.95–1.21);
  - by backend, medians 0.994 (P16), 0.996 (P8), 1.001 (P1), 1.004 (P2), 1.006 (P4), 1.012 (S1).
  - The benchmark windows predict full-length production runs to within about 1 %.
