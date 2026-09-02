# Gateway on a corrected baseline: the late reversal was the read-out's, not the dynamics'

**Date:** 2026-09-02. **Status:** both steps preregistered, frozen by commit before their runs, closed.
**Preregs:** [`gateway_baseline_audit_prereg.json`](../configs/information_campaign/gateway_baseline_audit_prereg.json) (step 1, commit 7bbdb12),
[`gateway_corrected_confirmation_prereg.json`](../configs/information_campaign/gateway_corrected_confirmation_prereg.json) (step 2, commit e335000).
**Data:** `results/information_campaign/gateway_baseline_audit/` (analysis.json), `results/information_campaign/gateway_corrected_confirmation/` (summary.json, comparison.csv).
**GPU:** 3 only; step 1 235 s, step 2 152 s.

## Why this was run

The bandwidth-defect screen ([BANDWIDTH_DEFECT_SCREEN.md](BANDWIDTH_DEFECT_SCREEN.md)) found the gateway ABF
baseline fully read-out-limited: its measured F′ error equals the deterministic kernel bias of the
legacy `h = 0.07` read-out to 1.3 %. An exploratory kernel-matched re-score turned the closed
campaign's late reversal (−11.81 % integrated, **+9.82 % worse at the end**) into −55 %. That re-score
is not a sharper read-out, and the closed runs saved no raw accumulators — so the question was settled
the same way ZIF-8 and WCA were: correct the baseline by a frozen rule, then re-run the two arms.

The engine gained a report-only raw-accumulator record (`store_accumulators`: binned force sums and
counts at every save). It is bit-inert and the offline read-out reproduces the engine's own profiles
and errors to 1e-15 (`tests/test_gateway_readout.py`; analyzer self-checks).

## Step 1 — ABF-only bandwidth matrix (outcome C_online_hurts)

Three ABF arms `h_bias` ∈ {0.07, 0.035, 0.0175}, each one batch of the same 32 (init, seed) rows with
the same batch seed — identical Langevin noise, exactly paired. Seeds 300–315 × {left, one_right}.

**Stage 1, read-out ladder on the legacy arm (frozen 2 % plateau rule):**

| read-out | e_F(T) median | I_F median |
|---|---|---|
| 0.07 (legacy) | 0.01034 | 1.015 |
| 0.035 | 0.00550 | 0.854 |
| **0.0175** | **0.00528** | 0.869 |
| 0.00875 | 0.00536 | 0.882 |
| raw bins | 0.00538 | 0.894 |

The legacy read-out was off the plateau by 96 % — a **1.96× e_F, 3.8× MSE** read-out defect, as the
saturated share predicted. `h_read* = 0.0175` (largest plateau point).

**Stage 2, online arms at h_read\*:** a sharper *online* bias force makes ABF **worse**, resolved:
0.035 → +10.36 % [+2.83, +15.39] (9/32 wins), 0.0175 → +14.89 % [+6.70, +17.83] (5/32). Roughness
0.913 → 0.982 → 0.997; no clipping (max |F′| 3.75). My recorded prediction was B (online halving
helps); it was wrong on that half and right on the read-out half. Corrected baseline: **h_bias 0.07,
h_read\* 0.0175**, γ = 1.5 inherited under the pre-specified rule (h_bias unchanged).

## Step 2 — abf vs fr_uniform on the corrected baseline (outcome G1_persistent_positive)

Fresh seeds 400–415 × {left, one_right} = 32 pairs, both arms in one batch (shared initial
conditions and noise), γ = 1.5, frozen-bias stage inherited. Every read-out below is computed from the
**same** saved accumulators of the **same** trajectories.

| read-out | ABF e_F(T) | ΔI_F | Δe_F(T) | wins (final) | ratio uni/abf at t = 5 / 10 / 17 / 20 / 30 / 40 |
|---|---|---|---|---|---|
| 0.07 (legacy; closed convention) | 0.01047 | −10.14 % [−13.22, −7.89] | **+9.28 % [+7.97, +11.32]** | 4/32 | 0.50 / 0.77 / 0.98 / 1.03 / 1.08 / **1.09** |
| **0.0175 (h_read\*, primary)** | 0.00529 | **−31.90 % [−34.73, −29.28]** | **−59.38 % [−63.76, −56.73]** | **32/32** | 0.50 / 0.45 / 0.41 / 0.39 / 0.37 / **0.41** |
| raw bins | 0.00532 | −32.30 % [−36.03, −30.64] | −63.53 % [−67.25, −59.41] | 32/32 | — |

Closed campaign at the legacy read-out (64 pairs, seeds 100–131): −11.81 % [−14.04, −9.02] /
+9.82 % [+8.20, +11.74], crossing at t ≈ 17–19. **Reproduced here to within a point on fresh seeds** —
and on those very trajectories the corrected read-out shows a persistent 2.5× advantage that never
reverses. Per-init: left −30.7 % / −64.1 %, one_right −32.9 % / −58.8 %.

Frozen-bias endpoint (campaign convention, η = 0.10 KDE, not read-out-corrected): FR **−14.45 %**
[−18.67, −4.13], 23/32 — the same direction the campaign reported (−11.15 %). Genealogy (frozen
median convention): min ancestor ESS/N 0.394 ≥ 0.30, max lineage share 0.013 ≤ 0.05; the worst single
row dips to 0.250 during the early burst, as in the campaign (0.138). Time-to-accuracy at h_read\*:
e0/2 1.09×, e0/4 1.31×, e0/8 1.50×, ABF-final **3.51×** (11.4 vs 40). Verdict at h_read\*:
**SAFE_ACCELERATOR**; at the legacy read-out: ACCELERATION_POSITIVE (fails non-inferiority, as the
closed study did).

The recorded prediction (G1; ΔI_F in [−40, −20], final ≤ −30 %, ratio never crossing 1, closed
signature reproducing at the legacy read-out) was confirmed on every item.

## What it means

The gateway's "transient acceleration with late reversal" — which the campaign spent effort
explaining — was an artefact of scoring against the unsmoothed reference through a read-out whose
kernel bias was the whole ABF error. The mechanism is the Nadaraya–Watson weighting: the legacy
kernel's bias depends on the sampling marginal, ABF's non-uniform occupancy partially cancels it, and
uniform-FR removes that accidental cancellation by flattening the occupancy. Read out at a bandwidth
where the bias is gone, uniform-FR is a **persistent 2.5× accelerator** on this system, and the
reversal does not exist.

With this, the three read-out-limited baselines the screen flagged have each been corrected by the
same frozen rule and re-measured: ZIF-8's harm became neutral, alanine's neutral stayed neutral, and
the gateway's reversal became a persistent positive. The two adequate-baseline positives (WCA, LTA)
did not need correcting. The screen's reading — *audit estimator validity first, then judge
reallocation efficacy* — held in every case.

## Reproduce

```bash
CUDA_VISIBLE_DEVICES=3 python -u scripts/run_gateway_bandwidth_audit.py        # step 1 (235 s)
python scripts/analyze_gateway_bandwidth_audit.py
CUDA_VISIBLE_DEVICES=3 python -u scripts/run_gateway_corrected_confirmation.py  # step 2 (152 s)
python scripts/analyze_gateway_corrected_confirmation.py
```
