# Energetic metastability toy: results of the preregistered confirmation

*2026-10-20 (UTC). Rules: `PREREGISTRATION.md` (commit ef903fc2, before any run). Frozen values: `FROZEN.json`
(commit c06859d2, before any confirmation run). Raw data (git-ignored): `results/urops_2026/metastability/` in the
main checkout (53 MB). CPU only, ≈ 6 core-hours in total.*

## Verdict: **FR_HARMFUL** (frozen rule, primary I_F)

32 fresh paired seeds 910000–910031, N = 1000, T = 200, h = 2×10⁻³, 120 bins, FR rate γ = 0.2. Paired per-seed
relative change (FR − ABF)/ABF, median [95 % bootstrap CI of the median], FR wins:

| statistic | ABF median | ABF + FR median | paired change | FR wins |
|---|---|---|---|---|
| **I_F** (trapezoid of e_F on [0, 200]) | 16.75 | 24.63 | **+51.9 % [+40.2, +60.7]** | 0/32 |
| **e_F(T)** | 0.0126 | 0.0554 | **+334 % [+291, +422]** | 0/32 |
| I_F′ | 50.53 | 50.59 | −1.0 % [−2.3, +2.4] | 20/32 |
| e_F′(T) | 0.148 | 0.168 | +13.9 % [+11.4, +18.2] | 0/32 |

* The I_F result is unchanged against a 4001-node y-quadrature reference (+51.9 %).
* The expectation written in the prereg ("ABF already performs well, so FR's effect may be small or neutral") is
  **not** what happened. FR is clearly harmful here. The result is reported as observed; nothing was retuned.
* FR does flatten the marginal, as designed. Panel (c): the instantaneous histogram reaches the finite-N floor
  (0.141) by t ≈ 40 under FR, against t ≈ 150 under ABF. The gate fraction (|x| < 0.5) is 0.165 under FR, close to
  the uniform 1/6, while ABF overshoots to 0.18.
* e_F′ is dominated by the piecewise-constant floor (0.139 at 120 bins), from the steep quartic walls. FR is
  slightly ahead of ABF up to t ≈ 90 and slightly behind afterwards.

### Where the extra error sits (diagnostic, not a tested mechanism)

* The final bin-mean-force bias is localised in the barrier bins x ≈ ±0.3, where the y-conditional is bimodal (upper
  sub-basin near (0, 1.54)). At x = −0.27 the median bias is +0.44 under FR against +0.15 under ABF.
* This leaves a free-energy step of about 0.11 (≈ 0.45 kT) across the barrier, against ≈ 0.04 under ABF.
* Share of gate walkers in the upper sub-basin (y > 0.9), last 40 t.u., median over seeds:
  * ABF 0.65;
  * FR 0.60;
  * exact 0.62 under a uniform x-law.
* Both arms are therefore biased in the hidden coordinate, in opposite directions:
  * ABF keeps an excess from the "mixed" start, which places half the walkers at y ∈ [1.1, 2.9];
  * FR's deaths at the over-filled gate are refilled by walkers that arrive from the y ≈ 0 wells.
* This matches the earlier project findings that FR's late bias follows the ancestor's birth region (D4/D5) and that
  FR harms a hidden conditional when active (pentane). It was not tested here.

## Stage 1: timestep gate (no ABF, no FR)

Flat-bias law (bias = exact F′), exact i.i.d. starts, 64 × 1000 walkers, 20 t.u. burn-in plus 200 t.u. of
accumulation. Each cell gives the value [95 % upper], jackknife over the groups.

Gates:

* V1: Var(y|x) relative error RMS ≤ 2 %;
* V2: D(F) by the mean-force route, upper bound ≤ 0.00457;
* V3: D(F′), upper bound ≤ 0.0239.

| chain | D(F′) | D(F), mean-force route | D(F), density route | Var(y\|x) rel. RMS (max) | pass |
|---|---|---|---|---|---|
| EM h 2e-3 (archived) | 0.0071 [0.0085] | 0.00344 [0.00389] | 0.0012 | 0.99 % (2.64 %) | yes |
| EM h 1e-3 | 0.0057 [0.0062] | 0.00282 [0.00314] | 0.0009 | 0.58 % (2.42 %) | yes |
| EM h 5e-4 | 0.0049 [0.0057] | 0.00339 [0.00391] | 0.0005 | 0.38 % (1.65 %) | yes |
| EM h 2.5e-4 | 0.0053 [0.0069] | 0.00369 [0.00422] | 0.0009 | 0.37 % (1.11 %) | yes |
| exact MALA h 1e-3 (acceptance 0.997) | 0.0057 [0.0068] | 0.00331 [0.00380] | 0.0010 | 0.39 % (1.35 %) | yes (V4) |

* **Selected h = 2×10⁻³**, the largest passing candidate (the archived timestep).
* Caveat: the D values are statistical-noise dominated. The exact MALA chain gives the same magnitudes. So the gate
  bounds the EM bias by its resolution (≈ 0.004 in F, ≈ 0.008 in F′), which is below ⅓ of the ABF error scale; it
  does not measure the bias itself.
* The only resolved EM effect is the 0.99 % Var(y|x) inflation at h = 2e-3 (MALA 0.39 %).

## Stage 2: histogram resolution (ABF only, pilot seeds 900000–900007)

| nb | median I_F | median e_F(T) | median e_F′(T) (floor) |
|---|---|---|---|
| 60 | 15.64 | 0.01202 | 0.279 (0.277) |
| **120** | 15.63 | **0.01110** | 0.144 (0.139) |
| 240 | 15.72 | 0.01138 | 0.081 (0.069) |

* The three resolutions tie on I_F (within 3 %).
* The frozen tie-break, the smallest median e_F(T), selects **nb = 120**.

## Stage 3: FR rate (safety only, the same pilot seeds; no accuracy read-out)

| γ | 0.05 | 0.1 | 0.2 | 0.5 | 1.0 | 2.0 |
|---|---|---|---|---|---|---|
| min windowed ESS/N (floor ≥ 0.30) | 0.858 | 0.770 | 0.706 | 0.561 | **0.296** | 0.180 |
| max windowed family share (floor ≤ 0.05) | 0.005 | 0.005 | 0.006 | 0.016 | 0.033 | 0.039 |
| safe | yes | yes | yes | yes | no | no |

* γ_safe = 0.5, so the production rate is the rung below: **γ = 0.2**.
* Confirmation genealogy (worst seed): min windowed ESS/N 0.655, max family share 0.011, ≤ 0.4 % of N per
  opportunity; median 2244 replacements per run (≈ 2.2 per walker).

## Comparison with the archived study (context only)

* The archived study used a kernel-type ABF, 5 exploratory seeds, and FR settings chosen on the same seeds.
* It reported a −6 % median integrated error for its best uniform-target configuration (14.97 vs 15.91).
* The histogram ABF here gives a similar ABF-only scale (I_F 15.6 to 16.8; e_F(T) 0.011 to 0.013).
* With a pre-specified, safety-chosen FR schedule and 32 fresh seeds, the uniform-FR effect is harmful.
* The archived positive is not reproduced. Its outcome-based selection is the likely reason, but this was not tested
  separately.
