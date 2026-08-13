# Triton pair kernel: correct, and NOT deployed — it is slower

Measured on a verified-idle H200 (foreign-compute-app precondition enforced by the benchmark
itself), float32, both paths through the identical `MethaneNonbonded.energy_forces`.

| B | tensor ns/day | triton ns/day | ratio | tensor peak | triton peak |
|---|---|---|---|---|---|
| 512 | **741** | 666 | **0.90x** | 9.3 GiB | **1.6 GiB** |
| 960 | 729 | 672 | 0.92x | 17.4 GiB | 3.0 GiB |
| 1536 | **745** | 674 | 0.90x | 27.7 GiB | 4.7 GiB |
| 2200 | 603 | **675** | **1.12x** | 39.7 GiB | **6.8 GiB** |

**Verdict: not deployed.** Production runs `B = 512`, where the fused kernel is **10 % slower**
than the `torch.compile`d tensor path. Seeds 5005–5007 run on the tensor path.

## Why it lost

The comparison was never "hand-written CUDA vs naive tensor ops" — `torch.compile` already emits
fused Triton for the tensor path, and it emits *good* Triton. Recovering 8.1x over eager (measured
earlier) was inductor's work, not mine, and my hand-written kernel had to beat an optimising
compiler at its own output rather than beat unfused PyTorch. It did not.

## What it did win, and where it would matter

* **Memory: 5.8x lower** (1.6 vs 9.3 GiB at B=512; 6.8 vs 39.7 at B=2200). No `(B, chunk, N)`
  intermediates are ever materialised.
* **Flat scaling**: 63.96–64.83 µs/traj-step across a 4.3x range of B, against the tensor path's
  degradation at B=2200 (57.96 → 71.68 µs). The tensor path is memory-bound and falls off when
  the intermediates stop fitting; the kernel does not have any.
* Consequently the kernel **wins at B = 2200 (1.12x)** and its advantage grows with B.

So the honest statement is not "the kernel failed" but **"the kernel is the right tool for a
regime this study does not occupy"**. A larger system, a larger population, or a device with less
memory would invert the result. At `N = 512` walkers on a 143 GiB card, it does not.

## Status

Correctness: 5/5 static gates, 5.2e-6 max relative force against float64 — closer than the
float32 tensor path it loses to (6.2e-6). Retained, tested, and not deleted, on the same rule as
`energy_forces_split` and `VerletList`: a recorded performance negative with a consistency test
against the live path, so a future change that breaks it fails a test rather than rotting.

The 8 ps equipartition trajectory gate was **not run** — it exists (`scripts/methane_triton_gate.py`)
and would have gated deployment, but the benchmark decided deployment first and spending 15 min of
GPU on a gate for a kernel that will not ship is not a good trade. Recorded as not-run rather than
implied to have passed.
