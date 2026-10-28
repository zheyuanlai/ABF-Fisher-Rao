# Ethane/LTA at finite loading (150 K, 16 guests per box): results

*2026-10-27 to 2026-10-28, NSCC ASPIRE2A, one A100 per job. Design: `PREREGISTRATION.md` (frozen before any
trajectory) with Amendments A0–A4. Running record: `docs/experiments.md`.*

**Claim labels.** *[confirmatory]*: fresh seeds 32000–32063, arms frozen beforehand. *[exploratory]*: the 16-seed
ladder that chose the arms. *[measured]*: timings.

## 1. Answer

**Q1 (primary, walker-step budget): unresolved — FR does not beat the best ABF.** Both methods' best allocation is
N = 64 (the smallest rung). There, on 64 fresh seeds per arm:

| N = 64 [confirmatory] | ABF | ABF + FR |
|---|---|---|
| median walker-steps to sustained mid accuracy (0.13 kJ/mol) | 36.1M | 36.7M |
| censored runs | 0 / 64 | 0 / 64 |
| median final e_F (kJ/mol) | **0.033** | 0.043 |
| median ancestor ESS / N (minimum over the run) | — | **0.026** |

**R_B = 1.02 [0.81, 1.25]** (median b_mid FR / ABF, unpaired bootstrap): *unresolved*. The medians tie but the
distributions do not (`figures/lta_loading/fig_confirm`): FR's fastest seeds are slightly faster than ABF's, while
~20 % of FR seeds need 75–170M walker-steps against ABF's slowest 83M — the seeds whose genealogy collapsed. The two arms share seeds, so
the paired contrasts are also available: integrated error I_F −6.1 % [−12.6, +4.8] (FR better 35/64), time to mid
accuracy FR better 31/64, **final error +41.9 % [+7.5, +57.6] worse with FR** (FR better 21/64).

**Q3 (secondary, modelled wall-clock on one A100, Amendment A3): unresolved.** The wall-clock-optimal arms of the
ladder were ABF N = 64 and FR N = 512 (a walker-step at N = 64 costs ~4x one at N = 512 on the GPU). On the same 64
fresh seeds: median W_mid ABF 33.5 s vs FR 31.1 s, **R_W = 0.93 [0.80, 1.10]**; FR at N = 512 censored 3/64 and ended
with a median e_F of 0.079 vs ABF's 0.033. FR's exploratory b_mid at N = 512 (104.7M) shrank to 122.4M on fresh seeds.

**Q2 (fixed N, exploratory, paired ΔI_F FR vs ABF, 16 seeds):** FR lowers the integrated error at every N, and the
gain grows with N — the single-guest pattern survives at fixed N:

| N | ABF median b_mid | FR median b_mid | paired ΔI_F [95 % CI] | FR ancestor ESS/N |
|---|---|---|---|---|
| 64 | **34.9M** | 42.0M | −3.5 % [−16.3, +11.6] | 0.03 |
| 128 | 68.0M | 55.0M | −9.7 % [−19.6, −5.2] | 0.05 |
| 256 | 118.3M | 74.5M | −26.0 % [−33.2, −21.0] | 0.12 |
| 512 | censored 9/16 | 104.7M (4/16 cens.) | −35.9 % [−40.2, −28.7] | 0.22 |
| 1024 | censored 16/16 | censored 8/16 | −30.3 % [−32.3, −27.8] | 0.33 |

(b_mid: walker-steps N·k at the first save after which e_F ≤ 0.13 kJ/mol at every later save; B = 236.6M per run.)

## 2. What the numbers say

* **FR is a fixed-N accelerator here, not a better deployment.** With many short walkers (large N), ABF cannot
  establish the barrier population inside the budget (censored at N ≥ 512) and FR fixes that, as in the single-guest
  system. But at an equal walker-step budget ABF does best with few long walkers (N = 64 needs 3.5x fewer walker-steps
  than N = 256), and there each walker crosses the windows thousands of times on its own; FR's reallocation then has
  nothing left to establish and its cost shows.
* **The cost is genealogical.** Cloning copies the whole box — the tagged guest *and* its 15 interacting neighbours.
  At N = 64 the ancestor ESS falls to ~3 % of the walkers and in some seeds a single lineage takes over the
  population; the final error is then 42 % worse than ABF's. This is the risk anticipated for C60/water (cloned
  environments reduce the effective number of independent samples); here it is measured.
* **Compared with the single-guest A100 study** (`docs/nscc_a100/RESULTS.md`): there, FR's best deployment beat ABF's
  at 150 K in wall-clock (R_W 0.72 [0.64, 0.83]). At finite loading the fixed-N advantage is of the same size
  (−26 to −36 % at N = 256–512), but neither best-vs-best comparison resolves: walker-steps R_B 1.02 [0.81, 1.25],
  wall-clock R_W 0.93 [0.80, 1.10]. **Interacting guests remove the deployment-level advantage FR had for a single
  guest; they do not remove its fixed-N advantage.**

## 3. The system and its reference

* 16 TraPPE-UA ethanes in the 2x2x2 LTA box (2 per α-cage), CH3–CH3 108 K / 3.76 Å; one tagged guest carries the CV
  and the bias; the 15 others are kept equilibrated by reinsertion MC every step (sodalite cages blocked; acceptance
  0.47 %). Engine: `src/lta_loading/engine.py` (CuPy + CUDA kernels, float64), Stage I gates V1–V6 all pass
  (`results/lta_loading/validation/validation.json`).
* Reference (`results/lta_loading/reference/reference.json`): frozen-bias equilibrium, 8 x 1024 walkers x 1M steps.
  Barrier **17.09 kJ/mol = 13.70 kT** (single guest 13.59 kT). Jackknife SE rms 0.006 kJ/mol; mean-force vs density
  route rms 0.006; halves stationary (barrier 17.103 vs 17.070, Δ 0.033 ± 0.018). Loading barely changes the
  landscape (`figures/lta_loading/fig_reference`): any change in FR's behaviour comes from the interacting
  environment, not from a different barrier.

## 4. What went wrong on the way (and is part of the result)

* **v1 was scored against a non-stationary reference** (Amendment A4, `results/lta_loading/superseded_v1/`). Its
  halves disagreed by 0.31 kJ/mol in the barrier; the cause was the tagged guest's own position distribution spreading
  from the cage-centre starts through the crowded windows over ~5e5 steps, during which the per-bin mean force is
  biased (`results/lta_loading/validation/env_relax*.json`: uniform-x starts are stationary from 1e5 steps; the
  environment reinsertion rate changes nothing). The acceptance checks of the time (group jackknife, two routes)
  cannot see a drift common to all groups; the v2 reference adds a halves stationarity gate. The v1 ladder showed the
  same fixed-N ordering; its best-vs-best numbers are not used.
* Amendments A1–A2 hardened the start-up after the physical check caught walkers parked inside framework walls by
  the first reinsertion sweeps; no data were affected.

## 5. Timing [measured] (`results/lta_loading/bench/bench.json`)

Seconds per step of one seed on one A100 (reinsertion every step):

| N | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|
| ABF (us) | 59.4 | 66.9 | 78.7 | 106.3 | 175.7 |
| FR (us) | 74.2 | 82.8 | 97.8 | 129.8 | 208.8 |

FR's selection runs on the host here (one device-to-host copy per opportunity), +19–25 % per step: an upper bound
for a device-side FR.

## 6. Files

* Design and amendments: `docs/lta_loading/PREREGISTRATION.md`.
* Code: `src/lta_loading/engine.py`; `scripts/lta_loading/` (validate, reference, run_batch, bench, analyze, plot,
  tasks/, pbs/).
* Results: `results/lta_loading/{validation,reference,pilot,bench,ladder,confirm,confirm_wall}/*.json`; batch `.npz`
  files stay on scratch (deterministic per seed: rebuildable with the task scripts).
* Figures: `figures/lta_loading/fig_reference`, `fig_ladder`, `fig_confirm`.
