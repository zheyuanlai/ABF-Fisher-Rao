# LTA read-out sweep and the three-system smoothing audit (2026-09-02)

Re-analysis only; no new dynamics. Two questions left open by the ZIF-8
corrected-baseline result (`docs/ZIF8_CORRECTED_BASELINE.md`): does the LTA
positive survive a corrected **read-out**, and does WCA's or LTA's legacy
bandwidth under-resolve its mean-force profile the way ZIF-8's did?

## 1. The LTA raw accumulators are recoverable exactly

The LTA engine (`src/lta/core_lta.py`) bins the mean-force samples and smooths at
read time with a fixed wrapped-Gaussian matrix `K` (`abf_bandwidth` 0.05 rad,
1.43 bins): `mean_force = (K fsum)/(K csum)`, `eff_counts = K csum`. Both saved
arrays are float64 and `cond(K) = 1.25e4`, so solving the linear system returns
the raw binned accumulators at every save. Validation, all four temperatures,
both arms (`scripts/lta_readout_sweep.py`, `results/information_campaign/lta_readout_sweep.log`):

- recovered counts are integers to < 1e-5; per-seed totals are exactly 300000 x 1024
  and the burn-in block exactly 20000 x 1024;
- re-smoothing at the legacy bandwidth reproduces the saved profiles to 1e-8 and
  the saved e_F series to 1e-10;
- the published legacy dI_F is reproduced to 1e-10 at every T (-35.14 / -31.92 / -21.28 / -14.84 %).

WCA has no such route: `TorchKernelABFEstimator` accumulates kernel weights at the
sample positions, so nothing raw exists on disk (the `final_eff_counts` array is
already smoothed).

## 2. Stage 1 -- ABF arm alone: the legacy read-out is already on the plateau

Median ABF e_F(T) (kJ/mol) vs read-out bandwidth, legacy dynamics:

| h_read (rad) | h/bin | 80 K | 150 K | 225 K | 300 K |
|---|---|---|---|---|---|
| 0.100 | 2.86 | 0.1773 | 0.1576 | 0.1373 | 0.1467 |
| 0.075 | 2.15 | 0.1655 | 0.1394 | 0.1068 | 0.1049 |
| **0.050 (legacy)** | 1.43 | **0.1638** | **0.1342** | 0.0948 | 0.0869 |
| 0.035 | 1.00 | 0.1651 | 0.1345 | **0.0935** | **0.0846** |
| 0.025 | 0.72 | 0.1661 | 0.1351 | 0.0938 | 0.0846 |
| 0 (raw bins) | 0 | 0.1675 | 0.1360 | 0.0946 | 0.0854 |

Plateau rule (largest h with median e_F(T) within 2 % of the ladder minimum, the
rule that reproduces the frozen ZIF-8 choice; the legacy is kept if it is on the
plateau): legacy **on the plateau at 80, 150, 225 K**; at 300 K it is 2.7 % off
and the rule moves to 0.035 rad (1.06x MSE). Contrast ZIF-8, where the same
sweep gave 5.8x. Seed spread rises as h falls (0.0163 -> 0.0186 at 80 K), so
here there IS a bias-variance trade-off, unlike ZIF-8.

## 3. Stage 2 -- both arms: the LTA gain is not a read-out artefact

Paired median dI_F (16 seeds, CI95 bootstrap seed 20260829); every entry 16/16 wins:

| h_read | 80 K | 150 K | 225 K | 300 K |
|---|---|---|---|---|
| 0.100 | -22.4 | -24.1 | -14.7 | -9.1 |
| **0.050 (legacy)** | **-35.1** [-37.3, -32.7] | **-31.9** | **-21.3** | **-14.8** |
| 0.035 | -36.4 | -32.7 | -22.5 | -16.1 [-18.5, -12.5] |
| 0.0175 | -35.7 | -32.3 | -22.3 | -16.8 |
| 0 (raw bins) | -33.9 [-35.0, -31.9] | -30.4 | -20.6 | -16.0 |

Final-error gain grows monotonically as smoothing is removed (80 K: -74.7 % at the
legacy -> -82.0 % at raw bins). The benefit is largest exactly where kernel bias
is absent, i.e. it lives in the count/sampling term, consistent with the
establishment-starvation mechanism and inconsistent with a kernel-bias artefact.

**Settled:** the LTA positive survives every read-out from 2.9 bins to no kernel
at all. **Not settled:** the ONLINE half (h_bias). Re-scoring cannot change the
dynamics; whether a sharper bias force changes the FR contrast needs a run.

## 4. Deterministic smoothing audit: how much of each baseline's error is kernel bias

`scripts/audit_readout_smoothing.py` applies each engine's legacy read-out kernel
to its own reference mean force (no data): roughness ratio (the metric used on
ZIF-8's live bias force), barrier error, and the free-energy smoothing bias in
endpoint units next to the measured ABF e_F(T).

| system, legacy h | h/bin | roughness | barrier err | det. bias / measured e_F |
|---|---|---|---|---|
| ZIF-8 300 K, 0.20 A | 1.34 | 0.912 (live: 0.93) | -0.52 % | 0.2671 / 0.3018 = **0.88** |
| LTA 80 K, 0.05 rad | 1.43 | 0.966 | -0.12 % | 0.0303 / 0.1638 = **0.18** |
| LTA 300 K, 0.05 rad | 1.43 | 0.974 | -0.16 % | 0.0419 / 0.0869 = 0.48 |
| WCA IX, 0.025 (+0.5 bin) | 2.84 | 0.950 | -1.98 % | 0.0153 / 0.0931 = **0.16** |

On ZIF-8 the deterministic bias tracks the measured error across the whole
published ladder (share 0.95 -> 0.49 from h = 0.40 to 0.03 A), so the metric is
calibrated. ZIF-8's legacy error was 88 % kernel bias -- a read-out problem.
LTA's and WCA's are 16-18 % -- removing the read-out bias entirely could lower
their ABF e_F by at most ~1.5 % (if the variance term does not fall), which is
what the LTA sweep then measured directly. The remaining question for both
positives is therefore the online bias force, not the read-out.

WCA's legacy kernel does compress its reference barrier by 2 % (4x ZIF-8's
-0.52 %) with roughness 0.950, comparable to ZIF-8's legacy -- so the online
half is worth testing there first, as planned.

## 5. What is staged next (not run)

The WCA ONLINE-bandwidth audit is fully staged and unlaunched:
`configs/information_campaign/wca_baseline_audit_prereg.json` (draft; frozen when
committed), `scripts/run_wca_bandwidth_audit.py` (ABF-only arms h_bias 0.025 /
0.0125 / 0.00625, fresh seeds 600-615, read-out bank 0.0125 / 0.00625 / raw),
`scripts/analyze_wca_bandwidth_audit.py` (written before the data; exercised on
smoke output), and `scripts/smoke_wca_readout_bank.py` (bank proven inert
byte-for-byte on the deterministic CPU path). GPU 3 only.
