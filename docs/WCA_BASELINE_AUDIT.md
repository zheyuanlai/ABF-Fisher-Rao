# WCA Case IX on a corrected baseline: there was no baseline defect to correct

**Closed 2026-09-02.** Preregistered in
`configs/information_campaign/wca_baseline_audit_prereg.json`, frozen at commit
`723bc6e` before any run; outcomes A-D and a prediction recorded in advance.
48 ABF-only runs (3 online bandwidths x 16 fresh seeds 600-615, 120k steps,
N=1024, 3.35 h on GPU 3), analyzer `scripts/analyze_wca_bandwidth_audit.py`
committed with the prereg.

## Verdict: A_readout_only, and the read-out is worth 1.04x

### Stage 1 -- read-out ladder on the legacy arm alone

| read-out | median e_F(T) | vs ladder min | seed sd |
|---|---|---|---|
| **0.025 (legacy)** | **0.09088** | **+2.04 %** | 0.00329 |
| 0.0125 | 0.08930 | +0.26 % | 0.00341 |
| 0.00625 | 0.08913 | +0.07 % | 0.00344 |
| raw bins + 0.5-bin smoothing | 0.08909 | +0.03 % | 0.00345 |
| raw bins | 0.08906 | +0.00 % | 0.00345 |

The frozen plateau rule (2 % tolerance) put the legacy read-out **off the
plateau by 0.04 percentage points** -- 2.04 % against a 2.00 % cut -- so
h_read* = 0.0125. That is the whole of Outcome A: the rule fired, and the total
read-out gain available on this system is **1.041x in MSE**. ZIF-8's was 5.8x.

### Stage 2 -- online bandwidth, every arm scored at h_read* = 0.0125

| arm (h_bias) | median e_F(T) | Δe_F(T) | CI95 | wins | roughness | clipping | max abs F' |
|---|---|---|---|---|---|---|---|
| 0.025 (legacy) | 0.08930 | — | — | — | 0.929 | 0.00 % | 8.17 |
| 0.0125 | 0.08654 | −4.63 % | [−6.40, +1.65] | 11/16 | 0.942 | 0.00 % | 8.32 |
| 0.00625 | 0.09050 | +1.25 % | [−6.98, +5.31] | 7/16 | 0.947 | 0.00 % | 8.35 |

**Nothing is resolved.** Both CIs span zero, and the quarter-width arm is not
even nominally better than the legacy. Neither is anything broken: no clipping
at any arm, max abs F' = 8.3 against a clip at 40, and roughness stays at
0.93-0.95 -- the predicted failure mode (a noisy bias force at small h) does not
occur, exactly as on ZIF-8.

The recorded prediction was **B** (the first halving helps, modestly). The first
halving does lean the predicted way, −4.63 % on 11/16 seeds, but it is **not
resolved** and the prereg's rule does not license calling it a win. Prediction
not confirmed.

## What this settles

**WCA's −21.9 % was not bought from a deficient baseline.** The ZIF-8 harm was
an artifact of an ABF baseline that had given away 5.8x in read-out MSE and 7 %
of its online bias force. On WCA the same instruments find a baseline that is
already adequate on both halves: 1.04x available from the read-out, nothing
resolved online. There is no correction here large enough to move a −21.9 %
effect, so the headline positive stands as measured.

Together with the LTA read-out sweep (`docs/LTA_READOUT_SWEEP.md`, gain survives
to raw bins at all four temperatures), **both strong positives are now clear of
the baseline objection** -- LTA on the read-out half exactly, WCA on both halves.

## What this does not settle

The corrected read-out cannot be applied retroactively to the published Case IX
comparison: those runs (seeds 400-415) saved no raw accumulators and no bank, so
the −21.9 % is scored at the legacy read-out and cannot be re-scored. A direct
ABF-vs-mFR run on the corrected baseline would still be the cleanest possible
statement. It is now a **confirmatory nicety rather than a threat**: the
correction it would apply is worth 1.7 % in ABF's own e_F, and it applies to
both arms. On LTA, where the same correction could be measured on both arms, the
FR gain **grew** as smoothing was removed.

## Methodological result: the cheap audit predicts the expensive one

`scripts/audit_readout_smoothing.py` smooths a system's own reference mean force
with its legacy kernel and reports share = (deterministic bias)/(measured e_F).
If the endpoint error is bias + independent residue, the achievable read-out
gain is `1/(1 - share^2)` in MSE. Against the three measured sweeps:

| system | share | predicted MSE gain | measured |
|---|---|---|---|
| ZIF-8 300 K | 0.88 | 4.6x | 5.8x |
| WCA IX | 0.16 | 1.03x | 1.04x |
| LTA 80 K | 0.19 | 1.04x | 0.96x |

It is a **bias-only upper bound**: on LTA, removing the kernel costs more
variance than it saves bias, so the measured gain is below 1 and the prediction
is an over-estimate. As a screen -- "is a bandwidth correction worth GPU hours
on this system?" -- it costs seconds and would have called all three outcomes
correctly in advance.

### Step 2 closed the same day (2026-09-02)

The corrected-baseline confirmation ([WCA_CORRECTED_CONFIRMATION.md](WCA_CORRECTED_CONFIRMATION.md), prereg 06e03e8, seeds 700–715)
gave **R1_replicated**: ΔI_F −18.30 % [−26.27, −14.00], 16/16, final −47.05 % at h_read\* 0.0125; legacy read-out −16.30 %
[−25.43, −12.13] overlapping Case IX's interval; the gain grows as smoothing is removed.

### Step 2 closed the same day

The corrected-baseline confirmation ran on fresh seeds 700–715 ([WCA_CORRECTED_CONFIRMATION.md](WCA_CORRECTED_CONFIRMATION.md)):
at h_read\* = 0.0125 uniform-FR is **−18.30 % [−26.27, −14.00] integrated and −47.05 % [−49.25, −43.77] at
the end, 16/16**, SAFE_ACCELERATOR; the legacy read-out on the same trajectories reproduces Case IX's
interval, and the gain grows toward raw bins. Outcome R1_replicated.
