# Fresh-seed confirmation of the best equal-budget allocation (preregistration)

*Written 2026-10-10, after the production ladders were analysed and before any confirmation run. Frozen in git
together with `configs/equal_budget_v2/confirm_best_{lta300,lta150,gateway}.json`.*

## Why

The task specification (§20, S8) says not to select the best N from one noisy seed or an uncorrected minimum, and
asks for "independent confirmation of the best candidate" where necessary.

The production best-allocation bootstrap does re-select N in every resample, but the selection is unstable:
* the top bootstrap frequency is only 0.40–0.47;
* best FR − best ABF is unresolved in all three systems: +3.4 % [−13.2, +12.4], +5.2 % [−10.9, +24.1],
  +15.9 % [−4.2, +26.6].

Re-using the same seeds cannot remove the winner's curse. A fresh-seed paired comparison at the selected N can.

## What is frozen

* **N\*** = the N that the production Ī_F best-allocation analysis selected for **both** arms:
  * LTA 300 K: N\* = 16 (ABF and FR both best at 16);
  * LTA 150 K: N\* = 2 (both best at 2);
  * gateway: N\* = 16 (both best at 16).
* **Fresh seeds**, disjoint from production:
  * LTA 300 K: 30100–30131 (32);
  * LTA 150 K: 15100–15131 (32);
  * gateway: 8200–8263 (64; its per-seed scatter is about twice LTA's).
* **Everything else is identical to production:** engine, h, B, the save grid, every algorithm constant, and the
  references. The configs differ from production only in `seeds`, `N_ladder`, `N_coarse` and `out_dir`.
* **Primary endpoint.** The paired G(Ī_F) = (FR − ABF)/ABF at N\*: median, 10 000-resample seed-bootstrap 95 % CI,
  and wins. This is exactly the production statistic.
* **Secondary endpoints:** G for final e_F, Ī_F′ (gateway: also Ī_F′_stat), and τ(e_F mid).

## Decision rule (fixed before data)

* **CI excludes 0 on the negative side** (FR better). At its own best allocation FR beats ABF at its best
  allocation on fresh seeds. This would move the production reading from "A, unresolved" toward B, and it would be
  reported as such.
* **CI excludes 0 on the positive side.** ABF's best allocation is better: a strict outcome A.
* **CI includes 0.** The best allocations are statistically tied. Report the CI as the bound on any FR advantage
  at the best allocation, and report outcome A in its weak form ("FR's best does not beat ABF's best").
* The confirmation is reported **separately** from the production result and never pooled into it.
* No N other than N\* is run, and no algorithm setting is changed.

## Cost

(32 + 32 + 64) seeds × 2 arms = 256 runs of about 4.5 min each ≈ 19 core-h, i.e. about 15 min of wall time on 110
workers. Ledgers: `results/equal_budget_v2/confirm_best_*/ledger.csv`.
