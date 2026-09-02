# WCA Case IX on the corrected baseline: the −21.9 % persists (R1)

**Date:** 2026-09-02. **Status:** preregistered, frozen by commit before the run (06e03e8), closed.
**Prereg:** [`wca_corrected_confirmation_prereg.json`](../configs/information_campaign/wca_corrected_confirmation_prereg.json).
**Data:** `results/information_campaign/wca_corrected_confirmation/` (summary.json, comparison.csv; 32 raw npz untracked).
**GPU:** 3 only; 32 runs in 2.45 h (shared with another user's job and, for ~7 min, the gateway audit).

## Design (all fixed by step 1, nothing re-derived here)

Step 1 ([WCA_BASELINE_AUDIT.md](WCA_BASELINE_AUDIT.md), outcome A) found no online arm resolved and the
legacy read-out 0.04 pp off the 2 % plateau, so the corrected baseline is **legacy dynamics
(h_bias 0.025) + read-out h_read\* = 0.0125**. Both arms — `abf` and `fr_uniform` — run at that
h_bias with every Case IX value inherited (N = 1024, 120 000 steps, FR rate 0.10 inherited because
h_bias is unchanged), on **fresh seeds 700–715**, both arms of a seed in one process. A read-out
bank recorded 0.0125, 0.00625 and raw binned sums at every save, so every read-out below is scored
from the same trajectories. The runner printed no error metric; the committed analyzer was the
first thing to read one.

## Result

| read-out | ABF e_F(T) | ΔI_F | Δe_F(T) |
|---|---|---|---|
| 0.025 | 0.09058 | -16.30 % [-25.43, -12.13], 16/16 | -42.39 % [-44.29, -38.76], 16/16 |  ← legacy (replication)
| 0.0125 | 0.08888 | -18.30 % [-26.27, -14.00], 16/16 | -47.05 % [-49.25, -43.77], 16/16 |  ← primary (h_read\*)
| 0.00625 | 0.08868 | -18.95 % [-26.35, -14.43], 16/16 | -48.05 % [-50.20, -45.25], 16/16 |
| raw+sigma | 0.08864 | -19.76 % [-26.52, -9.11], 15/16 | -48.41 % [-50.50, -45.64], 16/16 |
| raw | 0.08861 | -19.82 % [-26.51, -9.15], 15/16 | -48.60 % [-50.67, -45.86], 16/16 |

Recorded prediction R1 — ΔI_F in [−27, −15] with 16/16, Δe_F(T) near −40 %, ladder flat-to-growing
as smoothing is removed — held on every item. ABF's own correction from the sharper read-out is
−1.87 % (step 1: −1.7 %). Genealogy: min ancestor ESS/N 0.153 ≥ 0.10, max lineage
share 0.0273 ≤ 0.05 on every FR run. Round trips 781281 vs 791706 (flat, as in Case IX).
Time-to-accuracy at h_read\*: e0/2 1.00×, e0/4 1.00×, e0/8 1.40×, abf_final 3.43×. Verdict **SAFE_ACCELERATOR** at h_read\* and at the legacy read-out.

**Replication check.** The legacy read-out's ΔI_F CI [−25.43, −12.13] overlaps Case IX's
[−26.30, −19.04] (seeds 400–415); the fresh-seed median integrated gain is smaller (−16.3 % vs −21.9 %)
while the final-error gain is as large (−42.4 % vs −41.8 %). The 16 fresh seeds are a wider draw than
the closed block; both blocks say the same thing.

**The direction of the ladder is the point.** From the legacy 2.84-bin read-out down to raw bins the
FR advantage *grows* — integrated −16.3 → −19.8 %, final −42.4 → −48.6 % — exactly the LTA pattern
(−35.1 → −33.9 % / −74.7 → −82.0 %). Kernel smoothing was not buying any of the WCA gain; if anything
it was hiding some.

## Per seed (relative change fr_uniform vs abf, %)

| seed | ΔI_F @0.025 | ΔI_F @0.0125 | ΔI_F @raw | Δe_F(T) @0.0125 | min ESS/N | wmax |
|---|---|---|---|---|---|---|
| 700 | -30.9 | -33.1 | -33.4 | -46.2 | 0.165 | 0.021 |
| 701 | -29.6 | -30.7 | -34.3 | -49.2 | 0.202 | 0.016 |
| 702 | -0.8 | -0.0 | -6.5 | -40.6 | 0.159 | 0.023 |
| 703 | -15.4 | -18.0 | -17.0 | -47.2 | 0.169 | 0.021 |
| 704 | -17.1 | -18.6 | -20.5 | -43.8 | 0.152 | 0.027 |
| 705 | -16.3 | -17.8 | -20.6 | -38.7 | 0.167 | 0.018 |
| 706 | -2.8 | -4.2 | -8.2 | -44.7 | 0.169 | 0.019 |
| 707 | -16.6 | -18.8 | -9.1 | -49.9 | 0.170 | 0.019 |
| 708 | -25.4 | -26.3 | -26.5 | -49.0 | 0.184 | 0.014 |
| 709 | -15.6 | -17.0 | -12.0 | -46.9 | 0.166 | 0.017 |
| 710 | -16.3 | -19.4 | -19.2 | -53.6 | 0.168 | 0.019 |
| 711 | -28.6 | -30.3 | -27.0 | -51.2 | 0.172 | 0.023 |
| 712 | -0.4 | -1.3 | -8.6 | -43.5 | 0.166 | 0.021 |
| 713 | -28.5 | -31.5 | -34.9 | -48.4 | 0.166 | 0.016 |
| 714 | -9.4 | -10.4 | +5.4 | -49.8 | 0.173 | 0.021 |
| 715 | -12.1 | -14.0 | -22.7 | -43.0 | 0.176 | 0.018 |

At raw bins one seed's integrated gain shrinks to near zero (15/16 wins, CI [−26.5, −9.2]); its final
gain is intact. Raw bins carry the estimator's full variance at early saves, which is why the
integrated statistic, not the final one, is the one that widens.

## What this closes

With this the WCA positive has passed every test the ZIF-8 episode made necessary: the ABF-only
bandwidth audit (no defect capable of explaining it: 1.04× MSE against a 1.28× error reduction), the
kernel-matched re-score (−21.9 → −24.0 %), and now a direct fresh-seed replication on the audited
corrected baseline. Together with LTA 80 K (gain survives to raw bins) and the gateway (persistent
2.5× at the corrected read-out), three independent positives have survived estimator-bias scrutiny;
ZIF-8, alanine and R15/CHA remain neutral for distinct, understood reasons.

## Reproduce

```bash
CUDA_VISIBLE_DEVICES=3 python -u scripts/run_wca_corrected_confirmation.py   # 2.4 h alone on an H200
python scripts/analyze_wca_corrected_confirmation.py
```
