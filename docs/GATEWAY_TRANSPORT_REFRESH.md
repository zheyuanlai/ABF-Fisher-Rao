# Gateway transport, follow-up: exact dose, and the oracle fibre refresh

**Date:** 2026-09-02. **Status:** preregistered, frozen by commit before each stage, CLOSED — outcome **R1a_repaired_competitive**, H5 confirmed.
**Prereg:** [`gateway_transport_refresh_prereg.json`](../configs/transport_campaign/gateway_transport_refresh_prereg.json) (commit a7e6fa8; alpha\*\* frozen at bb717d2).
**Data:** `results/transport_campaign/gateway_horizontal/{calibration_refine, production_refresh}/` (analysis.json, comparison.csv, figures/).
**Parent:** [GATEWAY_HORIZONTAL_TRANSPORT.md](GATEWAY_HORIZONTAL_TRANSPORT.md) (H1: FR beats horizontal OT by +65 % at dose 1.41).
**GPU:** 3 only; Stage A2 92 s, Stage B2 228 s.

## The two questions

(A) Does H1 survive an exact dose match? (B) Does transport fail *because* it moves the marginal without moving the
conditional law? On the gateway the conditional is `y | x ~ N(0, 1/(β ω(x)²))` exactly, so the intervention is
exact: after every move, redraw y from the conditional at the new x (`Method.refresh = 'oracle'`, acting last, on y
only, from its own RNG stream shared across the arms of a row; non-refresh arms bit-identical with or without it —
`tests/test_gateway_fibre_refresh.py`). The same refresh applied to ABF and to FR is the control: an oracle
equilibrator may help every arm, and "OT + refresh improved" attributes nothing without it.

## Stage A2 — exact dose (blind, seeds 480–487, same rows and noise as Stage A)

| arm | fr_uniform | α 0.0030 | **0.00325** | 0.0035 | 0.00375 | 0.0040 |
|---|---|---|---|---|---|---|
| J_KL (ratio to FR) | 0.489 | 0.551 (1.126) | **0.501 (1.023)** | 0.458 (0.936) | 0.419 (0.857) | 0.388 (0.793) |

**α\*\* = 0.00325**, ratio 1.023 (production rows: 1.068). No error metric was computed.

## Stage B2 — seven arms, seeds 540–571 × {left, one_right}, one batch per chunk

Cluster bootstrap by seed, 10 000 resamples. I_F and e_F(T) at h_read\* = 0.0175 unless stated; raw bins in the last column.

| tag | contrast | ΔI_F | Δe_F(T) at h\* | wins | ΔI_F / Δe_F(T) at raw bins |
|---|---|---|---|---|---|
| P1 | ot_exact vs fr_uniform | **+70.3 % [+67.1, +76.0]** | +197.9 % | 0/64 | +69.7 % / +217 % |
| P2 | ot_exact_refresh vs ot_exact | **−62.2 % [−62.9, −61.3]** | **−84.2 %** | 64/64 | −61.7 % / −86.4 % |
| C1 | abf_refresh vs abf | **−36.8 % [−38.3, −35.0]** | **−82.6 %** | 64/64 | −34.8 % / −82.5 % |
| C2 | fr_uniform_refresh vs fr_uniform | −23.1 % [−25.4, −21.0] | −55.2 % | 62/64 | −21.4 % / −56.8 % |
| A1 | ot_exact_refresh vs abf_refresh | **−32.4 % [−34.5, −30.2]** | +6.1 % [+4.9, +9.1] | 64/64 | −34.8 % / **−10.4 %** [−12.2, −8.7] |
| A2 | ot_exact_refresh vs fr_uniform_refresh | **−15.4 % [−18.0, −12.4]** | +1.9 % | 57/64 | −16.1 % / +1.6 % [−0.3, +2.5] |
| A3 | ot_full_refresh vs abf_refresh | **−86.6 % [−87.0, −86.2]** | +12.5 % [+10.5, +14.4] | 64/64 | −89.1 % / **−26.9 %** [−28.0, −25.0] |
| A4 | ot_full_refresh vs fr_uniform_refresh | −83.0 % [−83.6, −82.6] | +6.8 % | 64/64 | −85.9 % / −17.9 % |
| — | fr_uniform vs abf (positive control) | −33.7 % [−36.6, −31.3] | −59.8 % | 64/64 | −33.9 % / −64.1 % |
| — | ot_exact vs abf | +15.0 % [+12.4, +17.5] | +19.1 % | 4/64 | +12.6 % / +15.9 % |
| — | ot_exact_refresh vs abf | −57.7 % | −81.4 % | 64/64 | −58.3 % / −84.4 % |
| — | ot_full_refresh vs abf | **−91.6 % [−91.8, −91.5]** | −80.1 % | 64/64 | −93.0 % / **−87.3 %** |
| — | fr_uniform_refresh vs abf | −50.1 % | −81.8 % | 64/64 | −49.7 % / −84.5 % |

Frozen decisions: P1 **FR_better** (H1 replicates at exact dose, both inits); P2 **REPAIRED**; C1, C2 SAFE_ACCELERATOR;
A1 ACCELERATION_POSITIVE_WITH_REVERSAL at h\* (SAFE at raw bins, see below); A2 **OT_better**; A3 as A1; A4 OT_better.
Outcome **R1a_repaired_competitive**, **H5_refresh = true**. Both inits agree in kind on every decision.

Frozen-bias endpoint (learned bias re-scored with nothing adaptive): P2 −42.9 % [−44.9, −41.2]; C1 −26.5 %;
A1 +8.1 % [+5.0, +9.1]; A2 +0.7 % [−0.4, +2.1]; A3 −1.1 % [−3.2, +1.0]; vs abf: fr −6.4 %, ot_exact +39.9 %,
ot_exact_refresh −20.1 %, ot_full_refresh −25.5 %, fr_uniform_refresh −19.6 %. With exact fibres every arm's
learned bias is 20–25 % better than ABF's and they tie with each other.

Time to ABF's final accuracy on the median curve: abf 40, fr_uniform 11.2, ot_exact never, abf_refresh 4.4,
fr_uniform_refresh 3.0, ot_exact_refresh 2.6, **ot_full_refresh 0.4** (100×). To e₀/8: 4.0 / 2.8 / 8.0 / 4.0 / 2.8 / 2.4 / **0.4**.

## Mechanism

**The fibre carry-over is the mechanism.** Refreshing y from the exact conditional after every move repairs exact-dose
transport by 62 % (I_F) and 84 % (e_F(T)), on 64/64 rows, and cuts its learned bias by 43 %. The signed flank error
tells the same story on the profile (Fig. F): left-flank mean F′ error at T abf +0.031, fr +0.011, ot_exact +0.039 →
with refresh +0.006, +0.004, +0.004, and ot_full_refresh −0.001; barrier error +0.17 / +0.07 / +0.21 kT → +0.04 /
+0.04 / +0.04 / +0.02 kT.

**The control reframes the gateway itself.** The oracle refresh alone cuts plain ABF's final error by 83 %
(0.0051 → 0.0009): on this establishment-limited system, ABF's endpoint error is dominated by conditional
non-equilibrium at the entering flank, not by variance. That is the bias-dominated-endpoint theme of the whole
project, now with the bias *located*. It also says what uniform-FR was doing: with exact fibres FR's advantage over
ABF shrinks from −33.7 % / −59.8 % to −21.7 % [−24.4, −16.8] / +4.7 % at h\* (−10.6 % at raw bins; frozen bias +7 %)
(post hoc, not preregistered). About a third of FR's integrated gain and most of its final gain were the fibre
effect: FR populates the flank with copies of *resident* walkers, whose y has relaxed, instead of *entrants*
carrying basin y; transport does the opposite. The remaining, marginal-reallocation part of FR's gain is real but
smaller.

**With exact fibres, transport is the better reallocator.** ot_exact_refresh beats fr_uniform_refresh by 15 %
(I_F), ties at T and on the frozen bias; full transport + refresh is a 100× time-to-accuracy accelerator and has
the best raw-bin final error of every arm (0.00064 vs abf 0.00506). The closed campaign's reading of full OT — a
pinned stratified sampler whose only damage was the ramp-time scar — is confirmed: remove the scar (here by
refreshing the fibre) and the pinned sampler is the strongest arm.

**Read-out caveat (frozen rule, applied post hoc).** h\* = 0.0175 was fixed by step 1's 2 %-plateau rule on plain ABF
at e_F ≈ 0.005. At e_F ≈ 0.0006–0.0009 the same rule no longer puts 0.0175 on the plateau: abf_refresh 0.00087 vs
0.00084 at 0.00875 (4 % off), fr_uniform_refresh / ot_exact_refresh 21 % off, ot_full_refresh 0.00097 vs 0.00064 at
raw bins (52 % off); even fr_uniform is 6 % off. Every ΔI_F decision agrees across the three read-outs (I_F is
dominated by early, large errors), but the final-error "reversal" flags on A1 and A3 at h\* invert at raw bins
(−10.4 % and −26.9 %, no reversal). The frozen verdicts are reported as frozen; the raw-bin read-out is the valid one
at this accuracy, and any future arm at this level must re-derive h_read by the plateau rule (which now picks
0.00875 or raw).

## Prediction scorecard

Recorded: R1b with H5 true. Right: P1 FR_better (+70, top of [+40, +70]); C1 in [−40, −15] (−37); P2 ≤ −30 (−62);
A3 ≤ −40 (−87); A4 OT_better. Wrong: A2 was predicted FR_better/equivalent and is **OT_better** (−15 %); A1 was
predicted [−25, +5] and is −32 %; A3 predicted SAFE, is SAFE only at raw bins. Outcome R1a, not R1b: once the fibre
is exact, transport reallocates better than birth–death.

## What it means

Free-energy estimation by ABF needs allocation across fibres *and* correct sampling within fibres. Fisher–Rao
reallocation satisfies the second by construction (it never changes a configuration's fibre) and is therefore safe
but not optimal; horizontal transport is the better allocator and violates the second by construction; transport +
conditional equilibration is the strongest combination measured on this system. The deployable question is now
whether a **non-oracle** fibre equilibrator — constrained dynamics at fixed x, an Ornstein–Uhlenbeck process with
τ_y(x) = 1/ω(x)² here, solvent relaxation at fixed bond length on WCA — recovers enough of the oracle's effect at an
acceptable cost. That is the next prereg (τ_relax ∈ {0, 0.5, 1, 2, 5} τ_y, same controls); the WCA molecular version
is licensed conditional on it.

## Reproduce

```bash
CUDA_VISIBLE_DEVICES=3 python -u scripts/run_gateway_transport_refresh_calibration.py   # Stage A2 (92 s), blind
CUDA_VISIBLE_DEVICES=3 python -u scripts/run_gateway_transport_refresh.py               # Stage B2 (228 s)
python scripts/analyze_gateway_transport_refresh.py                                      # analysis.json, comparison.csv, figures B2, C2, F
CUDA_VISIBLE_DEVICES="" PYTHONPATH=src python -m pytest tests/test_gateway_fibre_refresh.py -q
```
