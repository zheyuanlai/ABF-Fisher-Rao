# FROZEN — the cached WCA reference misses a real narrow trough near z ~ 0.26

Immutable record of the observation, frozen **before** any reference-machinery change, so a
later implementation edit cannot make it impossible to reconstruct what exposed the problem.
`trough_discovery.npz` carries the dense scan, the sparse 41-point profile, the cached profile
and the Gate-0 pool values; `provenance.json` carries git rev, array hashes and every config.

## The observation

Dense unsmoothed constrained-TI scan, `dz = 0.010`, 4 independent preparations x 128 replicas,
20 k prep + 20 k equilibration + 60 k production:

| z | 0.220 | 0.230 | 0.240 | 0.250 | **0.260** | 0.270 | 0.280 | 0.290 | 0.300 |
|---|---|---|---|---|---|---|---|---|---|
| dense `F'` | 4.087 | 2.998 | 1.916 | 0.860 | **0.424** | 0.577 | 1.037 | 1.529 | 2.050 |
| `se_prep` | 0.073 | 0.035 | 0.044 | 0.038 | **0.022** | 0.045 | 0.048 | 0.044 | 0.055 |
| cached | 3.867 | 3.237 | 2.620 | 2.094 | **1.931** | 1.768 | 1.771 | 1.986 | 2.211 |

Minimum **0.424 at z = 0.260** against a cached **1.931** — deviation **-67 sigma**. Ten
consecutive points trace a monotone descent and recovery; this is a resolved feature, not an
isolated bad point.

All four preparations agree at the minimum -- `lattice 0.892, from_lo 0.780, from_hi 0.819,
hot 0.951`, spread 0.171 -- no larger than the spread elsewhere on the curve (0.15-0.34). The
feature is **not** preparation-dependent.

Three independent measurements agree once the steep local gradient (`dF'/dz ~ -100`) is
accounted for: dense 0.860 (z=0.250), sparse-41 1.090 (z=0.255 interp), Gate-0 pools 0.931
(z=0.250). An earlier note claiming a "5 sigma disagreement" between Gate 0 and the sparse build
was **wrong** -- it compared 0.250 against 0.255 on a curve moving 1.1 per 0.01 in z.

## Why the cached reference misses it

Two independent causes, both now established:

1. **Grid resolution.** The cached acquisition grid is `dz = 0.028`; the trough has structure at
   0.010. It cannot be represented.
2. **Grid-unit smoothing.** `smooth_profile_torch(..., sigma=1.0)` takes sigma in **grid cells**,
   not in the physical CV. On the 41-point build (`dz = 0.035`) it turned a raw 0.601 into a
   smoothed 2.166 -- erasing the feature. Two references on different grids are therefore *not*
   processed identically, which is a reproducibility defect independent of this trough.

## Convention audit — the feature is NOT a coordinate artifact

Verified in float64 against autodiff:

| check | result |
|---|---|
| `grad xi` analytic vs autodiff | 0.000e+00 |
| `\|grad xi\|^2` | 0.125000 = `1/(2w^2)` exactly |
| `div v` numeric vs code `2w/r` | 1.4e-9 |
| `local_mean_force` vs den Otter assembled from autodiff | 1.4e-9 |
| `project_dimer_to_z` -> `xi(q) = z` | 2.2e-16, including across periodic boundaries |
| spatial dimension | n = 2; `div v = 2w(n-1)/r = 2w/r` consistent |
| `dF/dz` vs `dF/dr` | `v` built from `grad xi` with `xi = z` -> returns `dF/dz`. **No missing 2w.** |

An initial 2.5 % mismatch in `div v` was **float32 finite-difference noise**, not physics; it
vanished to 1e-9 in float64.

**Incidental:** `core.DTYPE = torch.float32`. The WCA sampler, the cached TI reference and both
HP builds all run in single precision. Not wrong -- the mean force is a statistical average with
`SE ~ 0.04`, far above float32 resolution -- but it is why the projection holds `xi = z` to
~2e-7 rather than 2e-16 in production, and it belongs in reference metadata.

## Consequences, recorded here and acted on elsewhere

* The cached reference is wrong through `z in [0.23, 0.30]`, worst at `z = 0.26`.
* `-22.83 %` (Case IX) is an **uncalibrated** effect size; `L2(F_new - F_cached) = 0.0608` is
  ~3x the effect it was used to measure.
* Gate 0's WCA verdict (pool spread 0.040 / 0.039) is **not** affected in substance -- pools were
  compared with *each other* -- but the normalised numbers use `|F'_ref|` in the denominator and
  are therefore not reference-independent. At `z = 0.25`, spread/`|F'_cached|` = 0.086 while
  spread/`|pool mean|` = 0.192. Both are far from deca's 0.61, but the exact figures are not
  frozen until HP v2 exists.
