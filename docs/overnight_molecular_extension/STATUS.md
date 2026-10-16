# Overnight molecular extension — STATUS

*Started 2026-10-16 17:12 UTC. Two adaptive arms only: histogram ABF and histogram ABF + uniform marginal FR. No sham,
oracle, OPES or any other algorithmic arm.*

## Current stage (2026-10-16 ~21:45 UTC): COMPLETE; nothing running

## What was done (two adaptive arms only, no sham or other arm)

| cell | barrier (exact MC) | entropic | frozen ABF-only classification | FR result |
|---|---|---|---|---|
| ethane / ERI 600 K | 6.67 kBT | 84 % | ABF-sufficient (K 64/256/1024) | not run |
| ethane / ITE 300 K | 15.9 kBT | 48 % | ABF-sufficient | **X1 (exploratory, K 1024): ΔĪ_F −14.9 % [−17.0, −10.1], 16/16; but e_F(T) +14 % [+0.8, +21]** |
| ethane / ITE 150 K | 24.0 kBT | 33 % | ABF-sufficient | **X2 (exploratory, K 256): ΔĪ_F −1.1 % (n.s.); e_F(T) +57 % [+29, +95], 0/16** |

**Bottom line.**
* In a realistic molecular entropic/energetic barrier system (ethane in rigid Si-ITE, 300 K, K 1024), **ABF + uniform
  FR reduces the integrated free-energy error by ~15 % at fixed K and equal force evaluations, on every one of 16
  paired seeds.**
* The gain comes from faster window establishment during the bias build-up.
* It is paid for with a **worse final estimate**. FR's late deposits are biased: in both cells the pooled late-half
  error vs exact MC is 3–4× ABF's, with the same profile shape. This is 2.5σ, below the frozen 3σ flag.
* At 150 K the late bias outweighs the early gain.
* The run was exploratory, not gate-licensed, and there is no equal-budget claim.

**Process findings.**
* The frozen eligibility classifier fails its positive control (it rejects LTA 150 K) and failed as a predictor here.
* An independent audit confirmed every gate decision and every number (`review/AUDIT_OVERNIGHT.md`).

## Validation gates

Every cell passes Gates 0–4. For ITE 150 K: dt 1e-4 (the second clause), 360 bins (on thin margins; with a common
reference 180 would have been kept, audit F4). See `VALIDATION.md`.

## Active jobs

None.

## Next action (needs the user)

1. Decide whether to follow up the X1/X2 lead, as a new preregistration:
   * a longer horizon and more seeds to establish whether the late FR bias is stationary;
   * an FR stopping time or a lower rate (the same two arms) to keep the early gain without the late cost;
   * an equal-budget ladder if a fixed-K gain survives.
2. Optionally, test the circulation hypothesis analytically or numerically. Note that a mean-corrected ABF would be a
   different algorithm and needs the user's approval.
3. Push the commits (on `main`, not pushed).

## Resource use

About 171 of 200 CPU core-h and 0 of 24 GPU-h. See `COMPUTE_LEDGER.md`.
