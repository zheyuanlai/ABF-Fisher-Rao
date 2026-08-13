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

## Was this table itself a dynamo artifact? Checked, no.

The NaCl session lost a scheduling decision to `torch.compile` silently falling back to eager:
every `(B, chunk)` is a new shape, dynamo's `cache_size_limit` is 8, and past that it stops
recompiling — producing an apparent 8x throughput cliff that is the compiler giving up rather
than the hardware. Their tell was an identical `us/traj-step` across four different batch sizes.

This sweep is clean, on three independent checks:

1. **Shape count.** Each tag builds a fresh engine, so at most 4 distinct shapes reach any one
   compiled function, against a limit of 8. The tensor tag compiles `pair.energy_forces` and
   `recip.energy`; the triton tag compiles only `recip.energy`, since the kernel is not a dynamo
   function at all.
2. **The tell is absent.** Tensor `us/traj-step` spreads 13.72 (57.96–71.68) — a real memory-bound
   degradation, not a step change. The triton path *is* nearly flat (0.87 spread), but it is a raw
   kernel with no dynamo involvement, so flatness there is the expected result rather than a
   fallback signature.
3. **Independent cross-check.** The benchmark measures `energy_forces` alone at 29.85 ms/step for
   `B = 512`; the production seed 5003 log shows 38.6 ms/step for the full step. The 8.75 ms
   difference is the M-SHAKE/RATTLE solver and the integrator, which is the right order. A benched
   number running eager would not reconcile with the production rate this way.

`torch._dynamo.reset()` now runs per configuration regardless, so a larger sweep cannot hit the
limit silently.

## Caveat on the absolute numbers, and why the verdict survives it

The NaCl session found its own sweep under-reporting by **1.41x** against the in-situ rate of a
real stage — 30 timed steps after 5 warmup does not amortise per-call overhead, while a
25 000-step production block does. This sweep used **fewer** (15 timed after 3 warmup), so the
same bias is present here.

**It is bounded, from data already in hand.** The benchmark measures `energy_forces` alone at
29.85 ms/step for `B = 512`; the in-situ full step, including the constraint solver and
integrator, is 38.6 ms/step (seed 5003). Forces cannot exceed the full step, so any under-report
factor `f` satisfies `f × 29.85 ≤ 38.6`, i.e. **`f ≤ 1.29x`** — and less than that by whatever
the constraints actually cost. At NaCl's 1.41x the forces alone would be 42.1 ms, exceeding the
entire measured step, which is impossible.

**The verdict does not depend on the absolute numbers.** Both paths were timed identically in the
same process, so the 0.90x ratio at `B = 512` is insensitive to a common under-report factor. What
the absolute figures may not be used for is planning — and they are not: the screen ETA is derived
by `methane_status.py` from per-seed wall times parsed from production logs, which is in-situ by
construction.

Adopted as a standing rule, from the NaCl session: **the authority for planning is the in-situ
rate of a real stage; a sweep exists only to rank configurations against each other.** Warmup and
timed counts raised to 25/200 so future sweeps are closer to in-situ anyway.
