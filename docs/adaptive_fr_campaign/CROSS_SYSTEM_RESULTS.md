# Experiment C — cross-system validation of the frozen online stopping rule R1

*Preregistered: `PREREGISTRATION.md` §B (rule frozen) and §C (cells, analysis), commit c3ef3128, 2026-10-17 07:10 UTC.
Run 07:10–07:29 UTC. Analysis: the frozen `scripts/adaptive_fr/analyze_online.py`.
Figures: `figures/adaptive_fr/online_stop_times.png` and `cross_system_atlas.png`.*

Every cell compares two arms on 16 fresh paired seeds: histogram ABF, and the same ABF + FR stopped online by R1.
* The arms are identical apart from the controller, and bitwise equal before FR start (asserted).
* Force evaluations are equal (asserted).
* There are zero FR events after any online stop (asserted).

## 1. Primary outcome (restricted sustained time-to-accuracy speedup, per-seed geometric mean over the cell's ladder)

| cell | system | status | configuration | online stop time, t.u.: median [IQR] (range) | **S_geo [95 % CI]** | Bonferroni 99 % CI | FR faster | verdict (99 %) |
|---|---|---|---|---|---|---|---|---|
| C1 | ITE 300 K | in-sample | K1024 · T50 · rate 2.0 | 5.50 [5.50, 5.50] (5.5–6.0) | **1.67 [1.45, 1.98]** | [1.38, 1.99] | 16/16 | **SUPPORTED** |
| C2 | ITE 150 K | in-sample | K256 · T100 · rate 0.5 | 8.25 [8.00, 9.00] (7.0–11.5) | **1.64 [1.29, 1.90]** | [1.29, 2.17] | 15/16 | **SUPPORTED** |
| C3 | LTA 150 K | held out (positive control) | K1024 · T60 · rate 0.2 | 21.50 [21.00, 22.62] (20.0–24.5) | **2.52 [2.33, 2.60]** | [2.21, 2.67] | 16/16 | **SUPPORTED** |
| C4 | LTA 300 K | held out | K1024 · T60 · rate 0.2 | 16.00 [15.38, 17.00] (14.0–18.5) | **1.46 [1.37, 1.60]** | [1.34, 1.61] | 16/16 | **SUPPORTED** |
| C5 | ERI 600 K | held out (negative control) | K1024 · T40 · rate 2.0 | 15.25 [12.75, 20.50] (5.5–25.5) | **0.61 [0.53, 0.71]** | [0.49, 0.73] | 0/16 | **CONTRADICTED** |

**Prespecified rule-level questions:**
* **Q-ITE (in-sample cells, fresh seeds): YES.** R1 reproduces, and slightly exceeds, the fixed-schedule acceleration
  without being told the stop time.
* **Q-LTA150 (held-out positive control): YES.** R1 keeps FR on about 4× longer than in ITE (median 21.5 t.u.) and
  preserves the full benefit.
* **Q-ERI (held-out negative control): NO.** R1 does **not** avoid the harm. Its stops come late (median 15 t.u.,
  range 5.5–25.5), and FR slows ABF (S_geo 0.61; final error +55 %).
* **Overall.** Under the preregistered definition, **R1 does NOT generalise across the tested systems**: it fails the
  negative control. It works in all four establishment-limited cells.

## 2. Secondary outcomes (median per-seed relative change FR/ABF − 1 [95 % CI], FR better)

| cell | post-ramp Ī_F | early Ī_F [t_FR, 0.4 T] | e_F(T) | e_F′(T) | post-stop-only estimate |
|---|---|---|---|---|---|
| C1 ITE 300 K | -39 % [-42, -33] (16/16) | -38 % [-44, -36] (16/16) | -35 % [-40, -25] (16/16) | -23 % [-36, -13] (15/16) | -1 % [-17, +20] (8/16) |
| C2 ITE 150 K | -28 % [-33, -22] (15/16) | -26 % [-30, -20] (16/16) | -32 % [-36, -19] (13/16) | -22 % [-31, -7] (14/16) | -16 % [-29, +1] (11/16) |
| C3 LTA 150 K | -48 % [-51, -47] (16/16) | -40 % [-41, -37] (16/16) | -59 % [-65, -55] (16/16) | -49 % [-51, -45] (16/16) | -49 % [-60, -33] (14/16) |
| C4 LTA 300 K | -25 % [-28, -24] (16/16) | -24 % [-29, -22] (16/16) | -23 % [-30, -14] (16/16) | -15 % [-18, -5] (14/16) | -6 % [-22, +16] (10/16) |
| C5 ERI 600 K | +23 % [+12, +30] (1/16) | +6 % [-0, +13] (5/16) | +55 % [+29, +90] (0/16) | +50 % [+19, +73] (0/16) | +11 % [-17, +23] (6/16) |

## 3. The online schedule actually realised

| cell | never stopped | stop reasons | FR replacements per run | events after stop | min windowed ESS/K | max lineage | window establishment, ABF → FR (t.u.) | true translocations ABF / FR | wall s ABF / FR |
|---|---|---|---|---|---|---|---|---|---|
| C1 ITE 300 K | 0 | {'established'} | 611 | 0 | 0.320 | 0.019 | 4.25 → 2.50 | 1595 / 1630 | 744 / 744 |
| C2 ITE 150 K | 0 | {'established'} | 114 | 0 | 0.444 | 0.037 | 7.06 → 4.19 | 412 / 406 | 377 / 378 |
| C3 LTA 150 K | 0 | {'established'} | 400 | 0 | 0.600 | 0.008 | 16.80 → 9.30 | 328 / 386 | 269 / 267 |
| C4 LTA 300 K | 0 | {'established'} | 210 | 0 | 0.765 | 0.006 | 9.00 → 7.30 | 938 / 958 | 267 / 268 |
| C5 ERI 600 K | 0 | {'established'} | 1562 | 0 | 0.654 | 0.007 | 2.10 → 1.90 | 3617 / 3484 | 486 / 481 |

## 4. Context: the same systems with a fixed schedule or continuous FR (cross-block, other seeds, descriptive only)

| cell | this campaign (R1) | earlier result |
|---|---|---|
| C1 ITE 300 K | R1: 1.67 [1.45, 1.98]; e_F(T) -35 % | E1 fixed stop 10 (rate 2.0, other seeds): 1.59 [1.31, 1.85]; e_F(T) −32 % |
| C2 ITE 150 K | R1: 1.64 [1.29, 1.90]; e_F(T) -32 % | E2 fixed stop 15 (other seeds): 1.25 [1.11, 1.58]; e_F(T) −19 % |
| C3 LTA 150 K | R1: 2.52 [2.33, 2.60]; e_F(T) -59 % | continuous FR, equal-budget numba runs (32 other seeds): 2.42 [2.25, 2.51]; e_F(T) −51 % |
| C4 LTA 300 K | R1: 1.46 [1.37, 1.60]; e_F(T) -23 % | continuous FR, histogram campaign (other seeds; time-to-accuracy not on this ladder): –; e_F(T) −8 % (n.s.) |
| C5 ERI 600 K | R1: 0.61 [0.53, 0.71]; e_F(T) +55 % | continuous FR, ERI K1024 (other seeds): 0.51 [0.49, 0.58]; e_F(T) +119 % |

## 5. Why R1 fails on ERI (diagnosis from its own logs; no retuning)

Typical controller trajectories (first FR run of each cell; directedness j in fraction of K per t.u.):

| cell | peak j (at t) | j after the peak | falls below 0.25 × peak at t1 | stop (2 t1) |
|---|---|---|---|---|
| C1 ITE 300 K (rate 2.0) | 0.228 (2.0) | collapses within about 1 t.u. | 2.75 | 5.5 |
| C3 LTA 150 K (rate 0.2) | 0.039 (5.5) | stays at 65–100 % of the peak for about 4 t.u., then decays | 10.0 | 20.0 |
| C4 LTA 300 K (rate 0.2) | 0.020 (5.75) | 65–90 % of the peak for about 3 t.u., then decays | 8.5 | 17.0 |
| C5 ERI 600 K (rate 2.0) | about 0.08 (2.0) | **plateaus at 30–60 % of the peak for 5–10 t.u.** | 7–11 | 14–22.5 |

* **In the four successful cells the directed flux has a clear end.** It is a burst lasting a few t.u. while the
  bias establishes the barrier population, followed by a decay far below the peak. Its timing differs by system
  (about 3 t.u. in ITE 300 K, about 8–10 t.u. in LTA), and R1 tracks that difference without any system-specific
  setting.
* **In ERI there is no clear end.** ABF establishes the window on its own in about 2 t.u., so the burst is short and
  weak, and FR then sustains a directed flux at a sizeable fraction of that peak.
* **The likely cause (hypothesis).** That flux is consistent with the FR-sustained non-equilibrium state documented
  in ERI: the cage-η distortion and the biased late deposits (`MECHANISM.md`).
* **What the rule cannot do.** R1's relative criterion cannot tell this self-sustained work from establishment work,
  so it stops late (median 15 t.u.) while FR's useful window in ERI closes at about 5 t.u.
* **Failure condition.** The directed flux does not decay well below its establishment peak: there is no clear
  establishment burst.
* **A possible remedy, not tested** (it would need a new preregistered rule): a criterion on the *decay* of j, or a
  bound tying FR's duration to its own establishment time.
