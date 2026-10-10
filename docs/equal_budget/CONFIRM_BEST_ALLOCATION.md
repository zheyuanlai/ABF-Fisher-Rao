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

---

## Results (appended after the runs; nothing above this line was changed)

*Runs 08:35–08:49 UTC, 256/256 complete, 0 failed, 22.1 core-h.*

Analysis: `scripts/equal_budget/analyze_ladder.py --system <s> --config configs/equal_budget_v2/confirm_best_<s>.json`,
written to `results/equal_budget_v2/confirm_best_*/analysis/{summary.json,tables.md}`. G = (FR − ABF)/ABF at N\*:
the paired median, a 10 000-resample seed-bootstrap 95 % CI, and wins.

| system | N\* | seeds | G Ī_F (primary) | G final e_F | G Ī_F′ | Ī_TV_half (marginal) | verdict by the frozen rule |
|---|---|---|---|---|---|---|---|
| LTA 300 K | 16 | 32 | +6.2 % [−1.2, +22.4], 11/32 | +6.3 % [−24.7, +31.4] | −0.2 % [−3.1, +6.2] | −6.5 % [−13.0, −1.8], 24/32 | **tie** (weak A); any FR advantage ≤ 1.2 % |
| LTA 150 K | 2 | 32 | −3.0 % [−8.7, +3.3], 19/32 | −3.2 % [−17.5, +13.0] | −2.8 % [−5.4, +1.0] | −4.6 % [−11.4, +5.3], 20/32 | **tie** (weak A); any FR advantage ≤ 8.7 % |
| gateway | 16 | 64 | **+24.0 % [+1.1, +52.9]**, 22/64 | **+21.4 % [+2.0, +51.9]** | +0.3 % [+0.0, +0.6]; Ī_F′_stat **+12.0 % [+4.5, +17.2]** | −31.3 % [−34.7, −27.8], 63/64 | **ABF better: strict A** |

**τ(e_F mid)**
* LTA 300 K: 0.09 vs 0.09 (14/17/1, p = 0.72).
* LTA 150 K: 0.035 vs 0.0375 (6/6/20).
* gateway: 0.020 vs 0.025 (26/34/4, p = 0.37).
* No arm is censored.

**Reading**
* At the best allocation, FR never beats ABF on fresh seeds.
* In the gateway, FR at N\* = 16 is significantly *worse* on the free energy and on the floor-free mean force,
  while it still improves the marginal (−31 %, 63/64). That is outcome C at the best allocation.
* In LTA the two arms are statistically tied at N\*.
* This independent test agrees with the production best-allocation bootstrap: +3.4 % [−13.2, +12.4],
  +5.2 % [−10.9, +24.1], +15.9 % [−4.2, +26.6]. It also sharpens it: the gateway moves from "unresolved" to
  "ABF better", and the bounds on a possible FR advantage tighten to ≤ 1.2 % (300 K) and ≤ 8.7 % (150 K).
