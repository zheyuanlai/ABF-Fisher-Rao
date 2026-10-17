# Preregistration — Experiment D5: a direct test of the arrival-law mechanism for continuous FR's late bias

*Status: EXPLORATORY. Written and committed on 2026-10-17 BEFORE the D5 runs were launched (the launch is logged in
`results/adaptive_fr_campaign/logs/D5.log` and `logs/launches.tsv`). Two arms only, ABF vs the same ABF + FR; no sham,
placebo or matched-turnover arm. Nothing here is used to choose an FR rate, a stop time or the online rule. Block
directories and job file use the label `D5`.*

## 0. Cost (stated first)

| item | CPU core-h |
|---|---|
| ERI 600 K, K256, T 40, seeds 1111000–015, both arms (D4 took 167 s per run at 64 concurrent processes) | ≈ 1.5 |
| ITE 300 K, K256, T 50, seeds 1121000–015, both arms (244 s per run) | ≈ 2.2 |
| diagnostic overhead (a few extra sums per deposit; D4's record cost 1.0 %), tests, analysis | ≈ 0.1 |
| **total** | **≈ 3.8** (campaign ≈ 48 → ≈ 52 of 100) |

Memory: the record is two arrays of 2 × 3 × 6 × 180 × 11 doubles (0.57 MB each) per run. No GPU.

## 1. The mechanism under test

**H_arr (MECHANISM.md §2.4.8; MECHANISM_SELECTION_PREREG.md §7.2).** Late in a continuous-FR run, FR clones in the
cage region and kills in the window region (ERI: births − deaths per t.u. per walker −0.0067 in the window, +0.0100
in the cage). The compensating return flow runs cage → window through the flank. A walker that arrives in a flank bin
carries the conditional (fibre) law of the region it came from until that law relaxes (ERI η: ≈ 2 t.u.). In
equilibrium sampling (ABF) arrivals from the two sides balance exactly, so the deposit-weighted conditional at the
bin is the equilibrium conditional. Under FR the arrivals are unbalanced toward the cage side, so the flank's
deposit-weighted conditional is displaced toward the cage's law and the mean force is biased. D4 showed the
*signature* (cage-born lineages carry the pooled pattern, window-born its mirror image; the cloned walkers are a
fibre-neutral draw); D5 measures the mechanism's two ingredients directly: per-provenance deviations and
per-provenance weights, in both arms.

**Alternative carried along (H_sub).** The provenance-dependent deviation could come from the sub-bin φ position
rather than from a fibre coordinate: a walker that has just entered a fine bin from one side sits on that side of the
bin, and the bin-averaged mean force shifts by Γ′ × (mean offset). The record stores the deposit-weighted sub-bin
offset per provenance, so this is tested too.

## 2. Instrumentation (engine `diag_arr`, default off; read-only; `tests/test_diag_arr.py`)

Per walker: the step of the last visit to the cage region (|φ| ≥ 2) and to the window region (|φ| < 0.5); the step
and side (from the lower / higher neighbouring coarse bin, or unknown) of the last entry into the current coarse bin.
Clones inherit the parent's record (the lineage's provenance). Every deposit is accumulated in two arrays, each
indexed by half (k ≤ n_steps/2 early, k > n_steps/2 late), a 3-way label, a 6-band time variable and the fine bin, with
11 quantities: count, f, η, u_x², bond r, u_z², cos / sin(2πY_com/L_y), cos / sin(2πZ_com/L_z) (transverse COM, L = the
orthorhombic unit-cell edges), and φ − bin centre:
* **P**: label = provenance (cage / window / none = neither visited yet; the more recent of the two), time = time since
  leaving that region (bands 0.01, 0.03, 0.1, 0.3, 1 t.u.);
* **E**: label = entry side into the current coarse bin, time = residence time in it (same bands).
Always-on whole-run sums `Scy, Ssy, Scz, Ssz` of the transverse phases are added next to `Seta, Sux2, Sbond, Suz2`.
The record draws no random number and writes no trajectory or estimator state: tests prove diag on vs off gives
identical trajectories, accumulators and RNG states in both arms (also with `diag_sel` on), the record partitions the
deposits exactly, cage / window deposits carry their own label with time 0, and chunking / checkpoint-resume are
bitwise. The default path still reproduces the trajectory frozen from engine 6716aefc (`tests/test_diag_sel.py`).

## 3. Runs

Both arms of the two D2/D4 blocks with `diag_arr = 1` (`configs/adaptive_fr/jobs_D5.json` from `make_jobs.py D5`):
ERI 600 K K256 rate 2, seeds 1111000–015; ITE 300 K K256 rate 1, seeds 1121000–015. **Gate**: every run bitwise
equal to its historical arm (sv_C, sv_M, sv_h, sv_i, sv_u, sv_cond, q_final); one mismatch stops the analysis.

## 4. Definitions

* Late half, regions, reference, wRMS, F RMS, seed-jackknife s.e.: as in MECHANISM_SELECTION_PREREG.md §4.
* **Per-provenance deviation** d_p(bin) = ⟨f⟩_p(bin) − Γ_ref(bin), per fine bin, from the late-half deposits with
  provenance p ∈ {cage, window}; coarse-bin versions use the class's own fine-bin counts as weights for both ⟨f⟩ and
  Γ_ref (exact at the coarse level). **Weight** w_p(bin) = share of the bin's late deposits with provenance p.
* **Arrival-law prediction** of the FR arm's late bias from ABF's deviations and FR's weights:
  b_pred(bin) = Σ_p w_p^FR(bin) d_p^ABF(bin) − Σ_p w_p^ABF(bin) d_p^ABF(bin) (the second term is ABF's own residual;
  under exact balance it is zero).
* **Sub-bin prediction** of a per-provenance deviation: s_p(bin) = Γ′_ref(bin) × ⟨φ − centre⟩_p(bin), Γ′_ref by
  centred differences of the reference mean force on the 180-bin grid.
* "Flank" statements are count-weighted over the flank bins; the "cage side" of a flank bin is its side away from
  φ = 0 (entry from the lower neighbour on the φ < 0 flank, from the higher neighbour on the φ > 0 flank).

## 5. Predictions (stated before the runs)

**P1 — provenance predicts the conditional and the mean force in the ABF arm (the premise).** In ABF's late half,
d_cage and d_window on the flank are opposite in sign and individually > 3 s.e. from zero (ERI: cage provenance
steepens the mean force on both flanks, i.e. d_cage > 0 on the φ < 0 flank and < 0 on the φ > 0 flank; window
provenance the reverse), and the flank η conditional (ERI) differs by provenance in the same sense (cage-provenance
walkers have lower η). Their count-weighted sum is ABF's own small residual (|·| ≤ 0.05 kJ/mol/rad on the flank). If
d_cage and d_window are both within noise of zero in ABF, the premise fails and the mechanism cannot be the
explanation.

**P2 — same deviations, different weights in FR (the mechanism).** (a) d_p^FR(bin) = d_p^ABF(bin) within jackknife
noise for both provenances on the flank (a count-weighted RMS difference not exceeding 2 s.e.); (b) w_cage^FR >
w_cage^ABF on the flank, with the excess largest near the cage and in the direction of the measured net transfer;
(c) the arrival-law prediction b_pred reproduces the FR arm's observed late bias on the flank (ERI coarse bins 3–6 /
11–14: +0.12…+0.27 / −0.16…−0.27 kJ/mol/rad; ITE coarse bins 2–3: +0.11 / +0.18): count-weighted RMS of
(b_obs − b_pred) over the flank smaller than half the RMS of b_obs, with the sign pattern reproduced. Under the null
(H_arr false), (b) or (c) fails: either the weights are the same in both arms, or the FR bias is not reproduced by
ABF's deviations with FR's weights (then the FR arm's per-provenance states themselves are altered, i.e. (a) fails).

**P3 — relaxation.** |d_cage(τ)| on the flank decays with time since leaving the cage: ERI by e over ≈ 1–3 t.u.
(the η relaxation of D4 §2.4.4), ITE faster. The same for the η conditional difference by provenance.

**P4 — carrier in ITE.** Among η, u_x², r, u_z² and the transverse COM phases, a coordinate whose flank conditional
differs by provenance in ABF (> 3 s.e., standardised by the ABF within-bin sd with the 1 %-of-range floor) and whose
FR − ABF pooled displacement has the sign of the cage law is the carrier. If none does, the carrier is not a recorded
coordinate.

**P5 — sub-bin alternative.** If H_sub carries the per-provenance deviation, s_p(bin) ≈ d_p(bin) on the flank (RMS
of d_p − s_p below half the RMS of d_p). If s_p is an order of magnitude smaller than d_p, the sub-bin position is not
the carrier.

**P6 — entry-side record (coarse-bin level, descriptive).** Flank deposits by entry side into the coarse bin: the
cage-side entries carry the cage-provenance deviation and the window-side entries the window-provenance deviation;
residence-time dependence as in P3 on the shorter (bin-crossing) scale.

## 6. Decision rule (per system)

* **SUPPORTED**: P1 (premise) and P2 (a), (b), (c) hold.
* **NOT SUPPORTED**: P1 fails (no provenance dependence in ABF), or P2 (b) fails (no weight shift in FR), or P2 (c)
  fails with (a) holding (the weights shift but do not reproduce the bias).
* **PARTLY SUPPORTED**: P1, P2 (b) and (c) hold but (a) fails (the weights shift *and* FR's per-provenance states are
  altered; the mechanism acts but is not the whole story).
* **UNRESOLVED**: anything else.
For ITE the verdict is given for the mean force alone (P1–P2 use f, which is recorded regardless of the carrier);
P4 says separately whether a carrier was found.

## 7. Reporting

MECHANISM.md gains §2.5 with the identity-check result, the P1/P2 tables (per-provenance deviations and weights in
both arms, flank; b_pred vs b_obs per coarse bin), P3 relaxation curves, the P4 carrier table, the P5 sub-bin check,
P6, and the verdict by §6. Script `scripts/adaptive_fr/arrival_split.py` → `results/adaptive_fr_campaign/mechanism/
arrival_split.json`; figure `figures/adaptive_fr/mechanism_arrival.png`. COMPUTE_LEDGER.md updated with the measured
core-h.

## 8. Post-launch analysis notes (dated; added after the runs, not part of the predictions)

*2026-10-17, 12:17–12:40 UTC.* Two passes of `arrival_split.py`. Changes after the launch, none touching §5–§6:
1. the **exact decomposition** of FR's flank excess over ABF into a weight term Σ_p (w_p^FR − w_p^ABF) d_p^ABF and a
   state term Σ_p w_p^FR (d_p^FR − d_p^ABF) (an identity; reported as post hoc in MECHANISM.md §2.5.2);
2. the **fixed-bin** version of P3 (cage-provenance deviation by time band at coarse bins 4–6 and 11–13), because the
   preregistered flank-pooled P3 statistic mixes the band's bin composition with the relaxation (§2.5.3);
3. the transverse COM phases are compared in natural units (no within-bin sd is recorded for them), a departure from
   the §5 P4 wording "standardised".
Final pass: ≈ 20 s single process; JSON `results/adaptive_fr_campaign/mechanism/arrival_split.json`.
