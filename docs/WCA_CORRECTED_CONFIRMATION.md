# WCA Case IX on the bandwidth-audited baseline: the −21.9 % persists on fresh seeds

**Date:** 2026-09-02. **Status:** preregistered (commit 06e03e8, before the run), closed. **Outcome: R1_replicated.**
**Prereg:** [`wca_corrected_confirmation_prereg.json`](../configs/information_campaign/wca_corrected_confirmation_prereg.json).
**Data:** `results/information_campaign/wca_corrected_confirmation/` (summary.json, comparison.csv; 32 raw npz untracked).
**GPU:** 3 only, 2.45 h (32 runs, both arms of each seed in one process; wall clock shared with the gateway batches for ~10 min).

## Design (all inherited, nothing tuned)

Step 1 ([WCA_BASELINE_AUDIT.md](WCA_BASELINE_AUDIT.md)) found no online arm resolved and the legacy
read-out 0.04 pp off the 2 % plateau, so the corrected baseline is **legacy dynamics (h_bias 0.025) +
sharper read-out (h_read\* 0.0125)**. Both arms run at the same h_bias; the FR rate 0.10 is inherited
because h_bias did not change (the re-earn rule was not triggered). Fresh seeds 700–715, 16 pairs,
read-out bank {0.0125, 0.00625, raw} in both arms, runner printing no error metric. Recorded
prediction: R1, ΔI_F in [−27, −15] with 16/16, Δe_F(T) near −40 %, ladder flat-to-growing.

## Result — every read-out, same 16 pairs

| read-out | ABF e_F(T) | ΔI_F | wins | Δe_F(T) |
|---|---|---|---|---|
| 0.025 (legacy; Case IX convention) | 0.09058 | −16.30 % [−25.43, −12.13] | 16/16 | −42.39 % [−44.29, −38.76] |
| **0.0125 (h_read\*, primary)** | 0.08888 | **−18.30 % [−26.27, −14.00]** | **16/16** | **−47.05 % [−49.25, −43.77]** |
| 0.00625 | 0.08868 | −18.95 % [−26.35, −14.43] | 16/16 | −48.05 % [−50.20, −45.25] |
| raw + 0.5-bin smoothing | 0.08864 | −19.76 % [−26.52, −9.11] | 15/16 | −48.41 % [−50.50, −45.64] |
| raw bins | 0.08861 | −19.82 % [−26.51, −9.15] | 15/16 | −48.60 % [−50.67, −45.86] |

- **Replication check (legacy read-out):** −16.30 % [−25.43, −12.13] against Case IX's −21.91 %
  [−26.30, −19.04] on seeds 400–415 — the intervals overlap; the fresh median sits at the softer end.
  Final −42.39 % vs Case IX's −41.76 %.
- **Corrected read-out (primary):** SAFE_ACCELERATOR. ABF's own error falls by 1.87 % from the read-out
  correction (step 1: 1.7 %) and the FR arm's advantage *grows* as smoothing is removed — the LTA
  pattern again, the opposite of what a smoothing-bought gain would do.
- Genealogy: min ancestor ESS/N 0.153 ≥ 0.10, max lineage share 0.027 ≤ 0.05. Round trips flat
  (781k vs 792k), 2160 replacement events per run — establishment, not extra crossings, as before.
- Time-to-accuracy at h_read\*: e0/8 1.40×, ABF-final **3.43×** (70 vs 240; Case IX 3.46×).

The recorded prediction held on every item.

## What it settles

The sentence the audit alone could not say is now said by a preregistered run: **the WCA gain persists
against a bandwidth-audited corrected ABF baseline on fresh seeds**, and it is larger, not smaller,
when the read-out kernel is removed. With the LTA sweep (gain survives to raw bins) and the gateway
replication (reversal was the read-out's), all three positives have been measured on baselines whose
estimator bias was audited and, where needed, corrected. The available read-out gain here (1.04×
MSE) and the mFR gain (1.9× in final MSE at h_read\*) are not the same lever.

## Reproduce

```bash
CUDA_VISIBLE_DEVICES=3 python -u scripts/run_wca_corrected_confirmation.py    # 2.45 h
python scripts/analyze_wca_corrected_confirmation.py
```
