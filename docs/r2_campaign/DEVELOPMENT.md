# R2 development record — offline replay only (2026-10-17, 11:20–11:40 UTC)

*Script: `scripts/r2_campaign/dev_replay_r2.py`. Output: `results/r2_campaign/dev/replay_r2.json` (every candidate,
every block, both arms) and `replay_r2_runs.csv` (per-run binding guard for the frozen rule). No new simulation was
run for any choice below. Everything in this file was fixed before `PREREGISTRATION.md` was committed.*

## 1. What R2 is

R1 (ONLINE_STOPPING_RULE.md) detects the end of FR's *directed birth-death phase* relative to its own peak. It
failed in ERI because that flux plateaus at 30–60 % of its peak and R1 stopped late (median 15.25 t.u.).

R2 replaces the trigger by the quantity the offline rule used: **the window-establishment time, measured online on
the run's own deposits**:

* `occ(t)` = the trailing-W mean share of the population in the **window bins** (the fine deposit bins whose centre
  lies in the engine's barrier window: `|phi| < window_phi` for ITE/ERI, 28 of 180 bins; `|z| < window_half` for LTA,
  46 of 180 bins), computed as `sum_window (C_j(t) − C_j(t−W)) / sum_all (C_j(t) − C_j(t−W))`. Every walker deposits
  every step, so this is exactly the time average over `(t−W, t]` of the instantaneous window fraction.
* `share` = (number of window bins) / n_grid = the deposit share of those bins under a flat bias (uniform marginal).
* `t_est` = the first check (grid `t_start + n × 0.25`) with `occ ≥ level × share` and every bin `C_j ≥ 200`
  (R1's support guard).
* **FR stops at `t_stop = 2 × t_est`** (absolute time, as R1 and the validated offline rule).
* R1's genealogy-safety backstop is unchanged and can only bring the stop earlier.

It is reference-free by the same AST test as R1 (the module imports only `math` and `numpy` and does no I/O). The
window mask is the engine's own region convention, passed at construction like K and `t_start`.

## 2. Development set and what was held out

| data | role |
|---|---|
| ITE 300 K K1024: C1 (R1 stop), E1 (stop 10), E1x (stops 5/20, 4 seeds), X1 (continuous) | development |
| ITE 300 K K256: EA_R1 (R1 stop, T200), A (stop 10, T200), M300_256 (continuous, rate 1), 9 dev monitor runs (0.25-t.u. records, exact replay) | development |
| ITE 150 K K256: C2 (R1 stop), E2 (stop 15), X2 (continuous, T200, 360 bins), M150_256 (continuous, rate 1) | development |
| ERI 600 K **K1024**: C5 (R1 stop), ERI_K1024 (continuous) | development (ERI was spent as a hold-out by R1) |
| LTA 150 K / 300 K K1024: C3, C4 (R1 stop) | **read-only check**: no element of R2 was chosen on them |
| ERI 600 K **K256** (ERI_K256, D2 re-runs) | **not loaded**: held out at the allocation level for the confirmatory block R2N256 |
| EB64, EB16 (saves every 2 / 8 t.u.) | not usable (the 1 t.u. window cannot be formed) |

Replay fidelity: the controller code was fed the saved cumulative accumulators at the save times, with
`check_every` = the file's save interval (0.125 t.u. for the ITE K1024/K256 T50–100 blocks, 0.1 for ERI and LTA,
0.5 for the T200 blocks, 0.25 exactly for the dev monitors). The production grid is 0.25, so a production `t_est`
is the first 0.25-grid check at or after the replayed value (≤ +0.25). For runs whose FR was actually stopped
(R1 or fixed), the replay is exact as long as the replayed decision precedes the actual stop; this held in **every**
run (`n_invalid` = 0 everywhere).

## 3. Candidates examined (all on the FR arms; stop = factor × t_est)

Grid: W ∈ {0.5, 1.0} t.u.; level ∈ {0.5, 0.6, 0.7, 0.75, 0.8, 0.9}; guards ∈ {none, support, support + stability};
persist ∈ {1, 2, 4}; factor ∈ {1.5, 2, 2.5, 3} (216 combinations per run; the factor only rescales `t_est`).

### 3.1 The brief's primary candidate (W 1, level 0.75, support + stability, factor 2) and the guard variants

Median `t_est` [min, max] over the FR arms (t.u.); the offline `establishment_time` of the same runs for comparison.

| block | offline est. ABF → FR | no guard | support | support + stability |
|---|---|---|---|---|
| C1 ITE 300 K K1024 | 4.25 → 2.50 | 2.50 [2.38, 2.62] | 2.50 [2.38, 2.62] | 2.50 [2.38, 2.62] |
| E1 ITE 300 K K1024 | 4.38 → 2.38 | 2.50 [2.38, 2.62] | 2.50 [2.38, 2.62] | 2.50 [2.38, 2.62] |
| X1 ITE 300 K K1024 (continuous) | 4.31 → 2.38 | 2.50 [2.38, 2.62] | 2.50 [2.38, 2.62] | 2.50 [2.38, 2.62] |
| E1x ITE 300 K K1024 | – → 2.44 | 2.50 [2.50, 2.62] | 2.50 [2.50, 2.62] | 2.50 [2.50, 2.62] |
| EA_R1 / A ITE 300 K K256 (saves 0.5) | – → 3.00 | 3.00 [3.00, 3.50] | 3.00 [3.00, 3.50] | 3.00 [3.00, 3.50] |
| M300_256 ITE 300 K K256 (rate 1) | 4.12 → 2.75 | 2.88 [2.62, 3.12] | 2.88 [2.62, 3.12] | 2.88 [2.62, 3.12] |
| dev monitors (ITE 300/150 K, rates 0.2–2) | – → 3.25 | 3.50 [2.75, 6.50] | 3.50 [2.75, 6.50] | 3.50 [2.75, 6.50] |
| C2 ITE 150 K K256 | 7.06 → 4.19 | 4.25 [3.62, 5.38] | 4.25 [3.62, 5.38] | 4.25 [3.62, 5.38] |
| E2 ITE 150 K K256 | 6.69 → 4.12 | 4.25 [3.62, 5.12] | 4.25 [3.62, 5.12] | 4.25 [3.62, 5.12] |
| X2 ITE 150 K K256 (saves 0.5) | 7.50 → 4.50 | 5.00 [4.00, 5.50] | 5.00 [4.00, 5.50] | 5.00 [4.00, 5.50] |
| M150_256 ITE 150 K K256 (rate 1) | 6.88 → 3.56 | 3.62 [3.12, 4.62] | 3.62 [3.12, 4.62] | 3.62 [3.12, 4.62] |
| **C5 ERI 600 K K1024** | 2.10 → 1.90 | **2.00 [2.00, 2.00]** | **2.00 [2.00, 2.00]** | **2.80 [2.50, 3.10]** |
| **ERI_K1024 (continuous)** | 2.10 → 1.90 | **2.00 [2.00, 2.00]** | **2.00 [2.00, 2.00]** | **2.80 [2.50, 3.00]** |
| C3 LTA 150 K (check) | 16.8 → 9.3 | 9.30 [8.80, 10.30] | 9.30 [8.80, 10.30] | 9.30 [8.80, 10.30] |
| C4 LTA 300 K (check) | 9.0 → 7.3 | 7.30 [6.70, 7.90] | 7.30 [6.70, 7.90] | 7.30 [6.70, 7.90] |

Readings.
* **The occupancy trigger reproduces the offline FR-arm establishment time** in every block (within the save
  resolution), and in the LTA check cells too (9.3 and 7.3, exactly the published values).
* **In ITE the guards never bind.** Per run (`replay_r2_runs.csv`), the support guard passes at 2.0–3.5 t.u. and the
  stability guard at 2.0–3.5 t.u., both before the occupancy crossing. `t_est` is the occupancy crossing in all 169
  ITE FR runs.
* **In ERI the stability guard binds and only there.** Occupancy crosses at the first closed window (2.0 t.u.; the
  trailing mean is already 0.81 × share there), support passes at 2.0, but the relative change of Γ over the trailing
  1 t.u. is 0.11 at 2.0 and only falls below 0.05 at 2.5–3.1 t.u. With the stability guard the ERI stop would be
  **5.6 t.u. [5.0, 6.2]**, at or past continuous FR's measured e_F crossing time in this cell (5.2 t.u., bootstrap
  4.9–5.6; ERI_RESULTS.md); without it the stop is **4.0 t.u.** in every ERI run.
* **Decision: the stability guard is logged but does not gate R2** (`stability_gate: false`; the statistic and a
  `stable` flag are written at every check). Reasons: it changes nothing in any ITE or LTA run; in ERI it converts
  R2 from "stops inside FR's useful window" into "stops at the crossing"; and R2's stop happens a further `t_est`
  after the decision, so the estimator is not read at the moment of the decision as it was under R1's logic. This
  departs from the brief's first-stated primary candidate and is recorded as such. The support guard (every bin
  `C_j ≥ 200`) stays: it is what makes a uniform-looking marginal with an unsupported estimator fail (tested).

### 3.2 Level (W 1, support, persist 1): median `t_est` [min, max]

| block | 0.5 | 0.6 | 0.7 | **0.75** | 0.8 | 0.9 |
|---|---|---|---|---|---|---|
| C1 ITE 300 K K1024 | 2.12 | 2.25 | 2.38 | **2.50 [2.38, 2.62]** | 2.62 | 2.88 |
| C2 ITE 150 K K256 | 3.38 | 3.69 | 4.00 | **4.25 [3.62, 5.38]** | 4.44 | 5.06 |
| M300_256 | 2.44 | 2.62 | 2.75 | **2.88** | 3.00 | 3.25 |
| C5 ERI K1024 | 2.00 | 2.00 | 2.00 | **2.00 [2.00, 2.00]** | 2.00 | 2.10 |
| C3 LTA 150 K (check) | 7.05 | 7.80 | 8.80 | **9.30** | 9.85 | 11.50 |
| C4 LTA 300 K (check) | 5.80 | 6.30 | 6.95 | **7.30** | 7.70 | 8.60 |

The level moves ITE's `t_est` by ±0.1–0.3 t.u. and ERI's not at all (ERI sits on the floor set by the first closed
window). **0.75 is kept**: it is the offline definition that produced every confirmed fixed-stop result, and it
reproduces the published establishment times.

### 3.3 Window (support, level 0.75): W 0.5 vs W 1.0

| block | W 0.5 | W 1.0 |
|---|---|---|
| C1 / E1 / X1 ITE 300 K K1024 | 2.12–2.25 | 2.50 |
| C2 / E2 ITE 150 K K256 | 3.94–4.06 [3.25, 5.00] | 4.25 [3.62, 5.38] |
| C5 / ERI_K1024 | 1.60 [1.60, 1.70] | 2.00 |
| C3 / C4 LTA (check) | 9.05 / 7.15 | 9.30 / 7.30 |

W 0.5 detects 0.2–0.4 t.u. earlier everywhere and would put the ITE 300 K K1024 stop at 4.25–4.5 t.u., below the
earliest fixed stop that has been tested (5 t.u., E1x). **W 1.0 is kept** (the offline statistic's smoothing, and
R1's window).

### 3.4 Persistence: persist ∈ {1, 2, 4} gives identical `t_est` in every block (the trailing mean does not fall back
below the level once it has crossed). **persist 1 is kept** (no delay of the decision).

### 3.5 Factor (W 1, level 0.75, support, persist 1): median stop time (t.u.)

| cell | t_est | × 1.5 | **× 2** | × 2.5 | × 3 | validated fixed / R1 stops in this cell |
|---|---|---|---|---|---|---|
| ITE 300 K K1024 | 2.50 | 3.75 | **5.00** | 6.25 | 7.50 | fixed 5 (E1x, S_geo 1.43) and 10 (E1, 1.59); R1 5.5 (1.67) |
| ITE 300 K K256 | 3.00 | 4.50 | **6.00** | 7.50 | 9.00 | R1 6.5 (1.31); fixed 10 (A, 1.28 unresolved) |
| ITE 150 K K256 | 4.25 | 6.4 | **8.50** | 10.6 | 12.75 | R1 8.25 (1.64); fixed 15 (E2, 1.25) |
| ERI 600 K K1024 | 2.00 | 3.0 | **4.00** | 5.0 | 6.0 | none below 5.5 ever run; crossing of continuous FR 5.2 [4.9, 5.6]; R1 stops 5.5 → S 0.96, 7.5 → 0.74, 9 → 0.73 |
| LTA 150 K (check) | 9.30 | 14.0 | **18.6** | 23.3 | 27.9 | R1 21.5 (2.52) |
| LTA 300 K (check) | 7.30 | 11.0 | **14.6** | 18.3 | 21.9 | R1 16.0 (1.46) |

× 1.5 would put ITE 300 K at 3.75, earlier than any tested stop; × 2.5 puts ERI at 5.0, at the crossing. **× 2 is
kept**: it is the validated offline factor, it lands inside the confirmed good range in every ITE cell, and it stops
ERI 1.2 t.u. before the crossing.

### 3.6 The secondary candidate (activation gate) — examined, not chosen

"Never start FR when the window establishes within δ of the ramp end" cannot be decided online before FR starts
(the ramp ends at `t_start`). Its online form is a piecewise factor: stop at `t_est` itself when `t_est − t_ramp ≤ δ`,
else at `2 t_est`. On the development data the delays `t_est − t_ramp` are ERI 1.0 (floor-bound by W), ITE 300 K
K1024 1.5, ITE 300 K K256 2.0, ITE 150 K 3.25, LTA 3.3 / 5.3. A δ would have to sit between 1.0 and 1.5, on top of
the ERI value and inside the ITE seed spread (C1 range 1.38–1.62). The factor rule already separates ERI (4.0) from
ITE 300 K (5.0) without a threshold. Rejected.

## 4. Frozen R2

```
R2 = dict(name="R2", trigger="occupancy", check_every=0.25, W=1.0, R=18, level=0.75, persist=1, stop_factor=2.0,
          support_count=200.0, stability_tol=0.05, stability_gate=False, essw_min=0.15, maxlin_min=0.10, maxlin_k=6.0)
```

Predicted stops (median over the replayed FR arms; production checks on the 0.25 grid may add ≤ 0.25 to `t_est`):

| cell | predicted stop (t.u.) | replayed range |
|---|---|---|
| ITE 300 K K1024 T50 | 5.0 | 4.75–5.25 |
| ITE 150 K K256 T100 | 8.5 | 7.25–10.75 |
| ERI 600 K K1024 T40 | 4.0 | 4.0 (all) |
| LTA 150 K K1024 T60 | 18.6 | 17.6–20.6 |
| LTA 300 K K1024 T60 | 14.6 | 13.4–15.8 |
| LTA 80 K (not replayed: no FR run of this cell exists in the data; historical histogram FR arm establishes at 16.2 t.u. on a 0.6-t.u. save grid) | about 32 | – |
| LTA 350 K (not replayed; historical FR arm establishes at 6.9) | about 14 | – |
| ERI 600 K K256 (not loaded) | about 4 | – |

## 5. Records of the guard signals in ERI (for the mechanism section of RESULTS.md)

Trailing-1 t.u. window occupancy relative to the flat share at the first closed window (2.0 t.u.), FR arm 0.81 vs
ABF arm 0.63 (seed 1340013); the stability statistic at 2.0 is 0.11 in both arms and reaches 0.05 at 2.5–3.1 t.u.
In ITE 300 K K1024 the occupancy at 2.0 is 0.43 × share (FR arm) and the stability statistic 0.22; in ITE 150 K
K256 0.01 × share and 0.68.
