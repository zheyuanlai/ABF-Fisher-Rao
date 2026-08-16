# Gate C instrument calibration, measured on NaCl — consolidated for reuse

Written 2026-08-16 for the C60 session's methods section and for any later study running Gate C.
**Self-contained on purpose**: it is here rather than in a message because a promise to send
numbers in four days depends on a session surviving four days, and this one has already been
superseded once. Anything below can be recomputed from `results/nacl/screen_all/` with the
scripts named.

Every number is from **NaCl/water, 2 ions + 821 TIP3P, closed study** (`RESULT_STUDY.md`).
**None of them should be inherited as a value.** What transfers is the method and the shape of
the pathologies; the levels are system-specific and two of them are measured to be so.

---

## 1. The gate cannot detect the deficit it nominally tests

`scripts/nacl_gate_c_sensitivity.py` — plant a **stationary** deficit in the **real** occupancy
traces (synthetic samples answer a different question: the trace's correlation structure is the
quantity in play), redistribute the removed mass over the bias-aware target so each checkpoint's
total is preserved, run the unmodified gate.

| planted deficit | 45 % | 50 % | 55 % | 60 %+ |
|---|---|---|---|---|
| seeds firing, N=64 | 0/8 | 0/8 | **8/8** | 8/8 |
| seeds firing, N=32 | 0/8 | 0/8 | **8/8** | 8/8 |

**Zero seeds fire at the 50 % the gate is written to catch.** At a 50 % deficit the mean sits
exactly *on* the `0.5 Q*` threshold, so noise lifts about half the checkpoints above it and the
required contiguous run never forms. Methane measured ~60 % on its system. **The floor is set by
the span requirement, not by the system** — see §3.

**Therefore `lambda >= 16` does not mean what its derivation says.** That floor came from
`0.5 lambda >= 2 sqrt(lambda)`, the point where a 50 % deficit is 2 sigma **on one checkpoint** —
correct arithmetic about a quantity the gate does not compute. Keep the floor; discard the claim.

## 2. Consequence: report a measured band, not a gate output

Both closed studies' verdicts rest on direct measurement, not on the gate:

* NaCl: SSIP worst sliding-window band `[0.974, 0.991]`, excluding sustained deficits above
  **~3 %** — ~18x tighter than the gate could report. Minimum *instantaneous* `P/Q*` over **every**
  second-half state-checkpoint (632 at N=64, 1256 at N=32): **0.866** and **0.832**.
* Methane: worst occupancy ratio **0.731** over all second-half state-checkpoints.

**The windowed statistic** (`scripts/nacl_audit_cip_power.py`): the gate asks for a deficit
*sustained* over the required span, so average occupancy over a sliding window of exactly that
span. The error bar comes from the spread across **independent seeds** — no Poisson assumption,
no autocorrelation model. Report the band; call it INCONCLUSIVE when it straddles the threshold.
It calibrates **exclusion**; its ability to *fire* is validated only against planted deficits.

Carry a **positive control** (a state expected to track) through the same statistic, so a
systematic offset shows up as the well-behaved state missing 1.0.

## 3. The span frontier is system-specific and must not be inherited

Sweeping only the contiguity span, planted 50 % deficit, 32 state-seeds:

| span | NaCl false positives | NaCl detects | methane FP |
|---|---|---|---|
| 0.02 T | **9/32** | 31/32 | 0/24 |
| 0.05 T | 1/32 | 25/32 | — |
| 0.10 T | 0/32 | 11/32 | — |
| 0.20 T *(both prereg)* | 0/32 | 0/32 | 0/24 |

**At the span where methane sees zero spurious fires, NaCl sees nine in thirty-two.** The
frontier is set by each system's own occupancy autocorrelation. Measure yours; the
false-positive column alone finds the usable range and needs no planted deficits.

## 4. Parameterise the span in ABSOLUTE time, never as a fraction of `T`

`tau_occ` (integrated autocorrelation of basin occupancy, Sokal window) = **42.2 ps** median over
32 state-seeds, range 17-86, above the 10 ps checkpoint floor, and **N-independent** (41.6 at
N=64, 42.2 at N=32).

Sweeping the span in **absolute** time: frontier **120 ps** at N=64, **160 ps** at N=32 — where a
fixed-fraction rule predicts 120 vs 240. Subsampling walkers *within one cell at fixed `T`*
(multivariate hypergeometric on the integer histograms) gives **100 / 80 / 100 ps** for
N = 64/32/16 — **flat in `N`**. So the 120 -> 160 drift is not counting noise; it tracks the
**judged-window length** (790 vs 1570 ps), an extreme-value effect: more opportunities, longer
spurious runs. Working model:

    frontier ~ (physical floor set by tau_occ) x (weak ~log growth in window length)

plus a counting-noise term once `lambda` is small — methane sees it rise 8x from N=128 to N=32;
NaCl shows none over 64->16. Their `lambda` runs 128-224 where NaCl's runs 30-61, so the two may
be compatible rather than conflicting. **Untested; do not resolve it from two ladders.**

**The design defect this exposes.** With `N x T` fixed, `span = 0.20 T` scales as `1/N`:

| N | span | span / tau_occ | |
|---|---|---|---|
| 64 | 312 ps | 7.4 | — |
| 32 | 625 ps | 14.8 | 2x blunter |
| 16 | 1250 ps | 29.6 | 4x |
| 8 | 2500 ps | 59.2 | 8x |

**The cells of an N ladder are not compared on equal instrument terms**, and the bias hides
establishment failure worst at small `N` — the regime such campaigns exist to find. Worse, the
prescription grows *faster* than the requirement: 1.4x from N=64 to N=32 against a prescribed 2x,
and by N=8 a ~12x overshoot. **Fix: set the span in absolute time (a few `tau_occ`, from a short
pilot).** `N`-invariant by construction, and it removes the transferable-number problem instead
of solving it.

## 5. The power floor, stated correctly

`lambda_k = N Q*_k` (**minimum** over the judged window, never a mean — a well-populated stretch
otherwise masks one where the state cannot hold walkers). Targets partition, so `Q*_k <= 1`, and
strictly `< 1` with two or more basins — hence `lambda < N`, and **no cell with `N <= 16` can
reach a floor of 16 for any state, whatever the sampling.** Struck a priori, not "underpowered".

A *fire* below the floor is not evidence either: `P < 0.5 Q*` degenerates to "empty right now"
when `0.5 lambda < 1`, and autocorrelation gives a well-established state long empty runs. That
is `results/deca/screen_RETRACTED_no_min_count_guard/`. So the contiguity statistic is honest
only as a **one-sided detector, and only above the floor**.

## 6. Naming traps, both live in this repository

* `reference_report.json`: `preregistered_max_TV` = 1.000 is the verdict; the generic-looking
  `max_TV` = 0.9959 is the **superseded** transpose. `nacl_gates.py` read the generic key, so the
  check that can halt the study evaluated the wrong direction — harmless only because both clear
  0.30. Fixed; the missing-field case now raises rather than falling back.
* `gate_C.power`: `lambda_min` held the **policy constant** 16.0 while the *measured* statistic is
  itself a minimum, and the result pages used the same token for the measurement. Renamed
  `lambda_threshold` / `lambda_min_over_window`. Methane hit the word-order variant
  (`gateC_min_lambda` vs `gateC_lambda_min`).

**Two correct values for one quantity is a defect**, and neither derivation checks nor scope
audits find it — only asking *which key the prose actually read*.

## 7. Limits on everything above

* Planted deficits are **stationary**; a deficit decaying as the bias fills in fires less
  readily, so every detection floor here is a **floor**, not a characterisation.
* False-positive counts of 0 bound the rate by the rule of three on small denominators
  (<= 12 % at n=24), **not** at zero.
* `tau_occ` comes from 79-157 samples per state-seed (19-37 correlation times) with a 5x spread
  across state-seeds — a crude summary, not defensible to a factor of two.
* The frontier/`tau` coefficient spans **<= 0.17 (methane, censored) to 3.8 (NaCl)** — at least
  20x. "Order-one `tau`" is dead as a quantitative rule. What survives is that the floor is
  physical and `N`-invariant, which is what makes an absolute-time span the right *structure*.
* Redistribution-scheme agreement in the planting test is **structurally guaranteed** once the
  per-checkpoint total is preserved (`P_k -> f P_k` exactly), so it is not corroboration. The
  property worth asserting is the scaling itself.
