# Online-vs-readout bandwidth matrix — frozen before the runs

`h_bias` sets the mean force used ONLINE to bias the dynamics; `h_read` sets the
estimate used to REPORT the answer. They have been the same parameter for the
whole project. They need not be.

## Frozen before any h_bias run

- **Primary readout `h_read = 0.05 A`.** Chosen from the ALREADY-PUBLISHED
  offline sweep at h_bias = 0.20 (commit `cbcbf3c`), where e_F plateaus for
  h <= 0.05 A (0.1266 / 0.1251 / 0.1251 at 0.05 / 0.03 / 0.02). It is on the
  plateau but not below the bin width, so it is not a degenerate choice. Every
  h_bias arm is scored at this SAME h_read; letting each arm pick its own best
  readout would be tuning on the endpoint.
- **Primary endpoint**: e_F(T) and I_F at h_read = 0.05 A, versus the frozen
  umbrella reference. Other h_read values are sensitivity, not selection.
- **`abf_min_count = 20` is held fixed** at every h_bias. Changing the
  regularizer with the bandwidth would confound them; if a small bandwidth makes
  the damper active in some bins, that is measured and reported, not tuned away.
- **Arms**: h_bias in {0.20 (already run), 0.10, 0.05, 0.025} A. ABF only, no FR.
- 8 seed labels, shared init pool, identical physical model, 150 ps.

## The failure mode being watched for

Smaller h_bias means a noisier online bias force, not worse genealogy. Recorded
per arm: max |F'|, clipping fraction, profile roughness, transit counts, minimum
effective bin count, T_kin.

## Predictions, recorded now

- **A** (only readout matters): E(0.20, 0.05) ~ E(0.05, 0.05). Then keep smooth
  online ABF and simply read out sharply -- an essentially free fix.
- **B**: smaller h_bias also helps -> bandwidth is an algorithm-design issue.
- **C**: smaller h_bias hurts (noisy bias force degrades sampling) -> the two
  bandwidths should be deliberately decoupled, h_bias > h_read.
- **D**: everything plateaus below ~1/3 bin width -> the kernel was adding
  nothing and the raw bin estimator is the natural one.

I expect **A or C**: the online bias force is a smooth control signal that does
not need resolution, while the readout does.

## What this does NOT test

Whether mFR's gains survive a corrected estimator. That concern is separately
addressed and largely answered: across five closed cells the p'/p kernel-bias
term accounts for only 1-2% of mFR's measured effect in the strongest positives
(LTA), so those gains are NOT bandwidth-mediated.
