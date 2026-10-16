# ERI campaign — results (dilute cell)

*2026-10-16. Ethane in rigid all-silica ERI (Gard & Tait 1973), PNAS-2005-type surrogate model, 600 K, infinite
dilution. Two adaptive arms are allowed (histogram ABF; ABF + uniform FR). **No FR production was licensed**, so this
document reports validated physics and the ABF-only classification.*

## 1. What is established (validated numerics)

| quantity | value | source |
|---|---|---|
| literature reproduction (single-bead xy barrier) | 5.96 ± 0.02 kBT, published 6.13; X-bead coordinate 6.13 | `results/eri_campaign/reference/framework_selection.json`, `gate1.json` |
| barrier on the CV (tagged COM along a1), exact MC | **33.28 kJ/mol = 6.67 kBT** | `gate1.json` |
| decomposition window vs cage | ΔU = +5.30 kJ/mol, −TΔS = +27.97 kJ/mol → **84 % entropic**; U has a local minimum *at* the window centre | `reference_profile.npz`, `figures/eri_campaign/reference/fig_reference_profiles.png` |
| reference precision | 0.007 kJ/mol RMS (16-group jackknife); every Gate 1 check passes | `gate1.json` |
| admissible dynamics | EM dt 2e-4: D vs MC 0.022 / 0.037 kJ/mol (density / MF), bond variance +19 % (first order) | `gate2_*.json`, `fig_gate2_timestep.png` |
| frozen histogram ABF | 180 bins, P0 floor 0.0034 kJ/mol, trust 200, ramp 1 t.u., pre-eq 5 t.u., T 40 t.u. | `PREREGISTRATION.md` A1 |

## 2. ABF-only regime screen (8 seeds per K; frozen classifier, Amendment 2; `screen_dilute/classification.json`)

| K | label | first translocation (t.u.) | window occupancy / Q\* (median block) | longest deficient run (t.u.) | window share of final MF error | conditional lag / floor (η, u_x²) | e_F(T) (kJ/mol) | Ī_F | e_F(T)/e_F(late) |
|---|---|---|---|---|---|---|---|---|---|
| 64 | **ABF-sufficient** | 2.06 | 0.99 | 0 | 0.07 | 0.71, 0.42 | 0.175 | 0.555 | 1.16 |
| 256 | **ABF-sufficient** | 1.29 | 1.00 | 1 | 0.07 | 0.65, 0.50 | 0.129 | 0.501 | 1.63 |
| 1024 | **ABF-sufficient** | 0.68 | 0.99 | 0 | 0.04 | 0.69, 0.51 | 0.124 | 0.446 | 2.06 |

The frozen A1 rule gives the same labels, but it had no power and this is documented (audit F1). Figure:
`figures/eri_campaign/screen/fig_screen_abf_only.png`.

## 3. Decision and reading

* **FR is not run for the dilute ERI cell** (the frozen gate says STOP). The 6.7 kBT, 84 % entropic window barrier is
  large, but ABF discovers the next cage within 0.7–2 t.u. It establishes the window population at its bias-aware
  target as soon as the 1 t.u. ramp ends, and keeps it there at every K. So FR has no establishment deficit to
  repair. This adds a negative control to the repo's lesson that a large entropic barrier does not imply an FR
  advantage.
* What limits ABF here is (i) at small K, sampling variance, and (ii) at large K, an early-deposit transient. The
  full-run error is 2.1× the error of the deposits made after 2 t.u. at K = 1024. That is a conditional-memory effect
  of the warm-up. Marginal reallocation does not target it, so FR is not expected to help.
* **Not established:** whether FR would be neutral, harmful or beneficial here. No FR arm was run. Do not cite this
  cell as "FR-neutral".

## 4. Not done (and why)

* Loading study (2 and 5 molecules/cage). It needs a GPU engine: CPU cost is about 6 core-h per N = 256 run at
  2/cage. The overnight instruction of 2026-10-16 gave priority to new molecular candidates; see
  `docs/overnight_molecular_extension/`.
* Equal-budget ladder: not licensed, because no eligible cell exists.
