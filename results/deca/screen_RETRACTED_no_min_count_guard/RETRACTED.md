# RETRACTED — this screen's verdict is an artifact, not a result

**Do not cite anything in this directory.** Kept only for provenance.

This run reported `REGIME: ESTABLISHMENT-LIMITED` and `licenses_mfr: true`. Both are wrong.
Two independent defects, both fixed after this run:

1. **Edge clamping in the reference.** `alkanes.interval.bin_counts` clamps out-of-range
   samples into the edge bins. Amendment 1 deliberately bracketed the umbrella windows at
   `[1.15, 3.70]` around the evaluation domain `[1.20, 3.60]`, so 4.82 % of samples sat
   outside by design and were piled into bin 0. That carved a fake 2.65 kT well at
   `grid[0]` (neighbours ~5.3 kT), which the Amendment 3 basin finder read as a genuine
   second minimum and split off as a 0.056 nm "state" below the screen's soft wall —
   permanently unpopulated, so Gate C fired on it.

2. **`abf_min_count` was declared and never applied.** `mean_force_profile` guards only
   `den > EPS`, so a bin holding one sample contributed that single instantaneous local mean
   force as its conditional average. The bias ran away:

   | | |
   |---|---|
   | learned `A_hat` span | 102.5 kT vs a 72.0 kT reference (+42 %) |
   | walkers above 2.80 nm, second half | 97.9 % |
   | occupancy of the folded basin (holds the 1.64 nm minimum) | 0.008 |

   `Q*` is computed from that bias, so Gate C compared a wrong occupancy against a wrong
   target. This is the standard NAMD `fullSamples` guard; `alanine/core2d_ala.py:109` applies
   it correctly and this sampler did not.

Fixed in `deca.core` (per-bin trust ramp on the APPLIED bias; the estimate keeps the full mean
force) and `deca.umbrella` / `deca.labels` (drop out-of-domain samples instead of clamping).
`tests/test_deca_sampler.py::test_abf_min_count_actually_ramps_the_applied_bias` asserts the
config field now changes the trajectory.

Unaffected: the reference itself (umbrella + MBAR, independent of ABF) and Gate A.
