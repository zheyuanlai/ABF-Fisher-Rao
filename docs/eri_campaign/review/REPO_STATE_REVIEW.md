# Repository scientific-state review for the ERI campaign

*2026-10-16. Read-only review of `main` at `5f95131e`. Nothing was committed, nothing existing was edited, no GPU
was used. CPU use: a few seconds of numpy re-analysis of arrays already on disk (Appendix A).*

**Scope and sources.** Read in full: `docs/final_campaign/FINAL_SYNTHESIS.md`, `docs/final_campaign/EVIDENCE_LEDGER.md`,
`docs/equal_budget/FINAL_RESULTS.md`, `docs/numerical_validation/WCA_LANGEVIN_VALIDATION.md`, `.../WCA_FORCE_AUDIT.md`,
`.../LTA_VALIDATION.md`, `docs/HISTOGRAM_ABF_REPLICATION.md`, `docs/LTA_HISTOGRAM_REPLICATION.md`,
`results/lta_histogram/README.md`, `docs/final_campaign/PRIOR_ART_NOVELTY.md`, `.../PUBLICATION_READINESS.md`,
`docs/UNIFORM_FR_CAMPAIGN.md`, `docs/ZIF8_RESULT.md`, `configs/lta_histogram/campaign.json`,
`configs/numerical_validation/lta_validation_prereg.json`, `configs/uniform_campaign/lta_prereg.json`. Also read
because they define conventions the task asks for: `configs/uniform_campaign/{lta_sweep_prereg,zif8_prereg,cha_prereg}.json`,
`results/uniform_campaign/RESULTS.md` (LTA stages 4, 5, 7), `docs/equal_budget/{SCIENTIFIC_PLAN,CONFIRM_BEST_ALLOCATION,LTA_TIMESTEP_VALIDATION,SCIENTIFIC_INTERPRETATION}.md`,
`docs/final_campaign/{PREREGISTRATION,EXPB_WALLCLOCK}.md` (Exp B sections), `docs/V2_PREREGISTRATION.md` (§2, Amendments 7–10, §12.9, §12.13, §16.11),
`docs/numerical_validation/WCA_CONFIRMATORY.md`, `docs/ZIF8_CORRECTED_BASELINE.md`, `docs/BANDWIDTH_DEFECT_SCREEN.md`,
`docs/LTA_READOUT_SWEEP.md`, and the code that implements the conventions: `src/lta/core_lta.py`, `src/alkanes/core.py`
(FR functions), `scripts/analyze_uniform_lta.py`, `scripts/analyze_lta_histogram.py`, `scripts/equal_budget/eqb_metrics.py`,
`scripts/numerical_validation/analyze_lta_validation.py`, `scripts/run_lta_reference.py`, `scripts/calibrate_lta_fr.py`.

**What this file does not contain.** No literature facts. Everything below is a statement about this repository, with
file:line or section pointers. Literature numbers quoted in repo docs (e.g. Lelièvre–Rousset–Stoltz 2007) are reported as
"the repo says", not verified here. The ERI campaign prompt itself is not in the repository: `docs/eri_campaign/STATUS.md`
refers to its "§3.3 and §6" (STATUS.md:31-32) and to "Gate 0–3" stages (STATUS.md:13-16), but I found no copy (searched
`docs/eri_campaign/`, `configs/`, `scripts/`, `src/`). So §2.5 below lists **every** gate-power or vacuous-boundary defect the
repository records, plus the ones I found in the LTA analysis code. I cannot say which of them the prompt names.

---

## 0. New findings from this review (not recorded anywhere in the repo)

These change how some LTA statements should be read. Each was checked on data already on disk (Appendix A).

| # | finding | evidence | consequence |
|---|---|---|---|
| N1 | **The LTA engine's cage-to-cage crossing counter is inherited by clones.** At every FR (or sham) event, `has_left_cage` and `rep_crossings` of the dying slot are overwritten with the source's values (`src/lta/core_lta.py:558-559`). A clone born in the window carries `has_left_cage = True`, so its first descent into a cage is counted as a completed crossing although its own trajectory never left a cage. This is the defect ZIF-8 Amendment A1 found and fixed for ZIF-8 ("TRANSIT COUNTING ... inherited on a clone ... drifted UPWARD in the FR arm ONLY", `configs/uniform_campaign/zif8_prereg.json:201`). It was never ported to LTA. | Recount of the two movie seeds from their genealogy records at snapshot resolution (100 steps), Appendix A. Engine emulation reproduces the engine totals to 5–9 %. **150 K:** FR/ABF crossings 1.154 with inheritance (engine: 1.128), **1.083 counted per walker identity**. **300 K:** 1.017 with inheritance, **0.987 per identity**. | The mechanism sentences "FR raises the window traffic (1.09 vs 0.92 crossings per replica, +18 %)" (`docs/LTA_HISTOGRAM_REPLICATION.md` §4.4), "At 80 K FR doubles the window crossings; the sham does not touch them" (§5 item 2, §4.7) and "the uniform arm nearly DOUBLES the cage crossings (7081 vs 3840)" (`results/uniform_campaign/RESULTS.md` Stage 5 finding 3) are partly bookkeeping artefacts. On the 150 K movie seed about half of the excess disappears; at 300 K all of it does. The 80 K ratio has no genealogy record and is unverified. The F-error results are unaffected (the counter feeds no dynamics). The sham is immune by construction (random sources rarely carry `has_left_cage = True`), so "sham = ABF" is not evidence against the artefact. |
| N2 | **The LTA analyzers still use the pre-ZIF-8 verdict tree.** `verdict_from_rule` (`scripts/analyze_lta_histogram.py:478-491`, same tree in `scripts/analyze_uniform_lta.py:141-146`) has NEUTRAL = `|median ΔI_F| < 10 %` and median final change ≤ +5 % (point estimates, no CI), no HARMFUL label, and health gates only the SAFE label. ZIF-8 A1 ("VERDICT TREE", `zif8_prereg.json:202`) showed this tree lets a 9.9 %-worse arm with a CI excluding 0 read NEUTRAL, and replaced it with TOST NEUTRAL + explicit HARMFUL + INCONCLUSIVE (`zif8_prereg.json:139-145`). | code read | The LTA SAFE_ACCELERATOR verdicts are not affected (their CIs exclude 0 by a wide margin). The v1 300 K "NEGATIVE_OR_UNSAFE" label (ΔI_F −0.21 % [−2.93, +0.77], final +7.44 % [+1.89, +18.17], `RESULTS.md` scoreboard) would be INCONCLUSIVE under the corrected tree (CI inside ±10 % but final margin violated). ERI should use the corrected tree. |
| N3 | **The τ persistence window truncates at the end of the run.** `tau()` in `scripts/analyze_uniform_lta.py:38-48` uses `j = min(searchsorted(t_i + 0.2T), len−1)`, so a save near the end needs no persistence. ZIF-8 A1 ("PERSISTENCE WINDOWS", `zif8_prereg.json:197`) fixed exactly this; the LTA analyzers import the unfixed function (`scripts/analyze_lta_histogram.py:57-58`). | code read; `LTA_HISTOGRAM_REPLICATION.md` §4.4: "τ to ABF's final accuracy: 25.2 vs 60.0" | (a) τ at ε = ABF's own final median error can never be censored for ABF, and typically equals T (60.0 at 150 K): the "τ speed-up to ABF's final accuracy" (1.03×–5.26×, §5 table) is T/τ_FR with a denominator fixed by construction. (b) The screening rule's T_est ("time to 1.5× the final error", persistence 0.2 T) cannot be censored and is self-referential: a flat, never-converging curve gets a small T_est. The equal-budget τ (`eqb_metrics.py:517-533`) requires persistence to the last save; a single final save below ε still counts as uncensored τ = T. |
| N4 | **About 40 % of the LTA integrated error accrues before FR starts.** In the 16-seed histogram productions the median share of ABF's I_F from t ≤ 4 t.u. is 0.41 (300 K) and 0.40 (150 K); the ABF and FR PMFs agree to ≤ 3e-13 there. | Appendix A | Every LTA ΔI_F is diluted by that fixed segment. If the FR arm's pre-FR segment equals ABF's, the post-FR change is roughly ΔI_F/0.6, e.g. −14.1 % → about −24 % at 300 K (my arithmetic, not a measured endpoint). ZIF-8 A1 added a declared post-FR secondary for this reason (`zif8_prereg.json:205`); LTA never did. |
| N5 | **The reported-estimator restart is visible in the LTA error curves**, and it coincides with FR start. Production accumulators start at step 20 000 (`core_lta.py:428-432`) and the save at t = 4.2 t.u. reads them after 1 001 steps: median ABF e_F goes 0.387 → 0.849 kJ/mol (300 K) and 0.487 → 0.687 (150 K) between t = 3.6 and 4.2. | Appendix A | Same artefact ZIF-8 declared (`docs/ZIF8_RESULT.md` "Two artifacts" item 1). In LTA it sits in the FR-active window, not before it. It is common to both arms, but it inflates I_F and shapes early τ. |
| N6 | **The LTA FR-rate calibration ran at 120 000 steps, not the 300 000-step production horizon** (`scripts/calibrate_lta_fr.py:28`, `CAL_STEPS = 120_000`; the 350 K ladder also used 120 k steps, `LTA_HISTOGRAM_REPLICATION.md` §4.1). | code read | The ZIF-8 prereg records that a short ladder over-certifies, because run-long ancestor collapse grows with the horizon (`zif8_prereg.json:115`, the CHA 150 k → 400 k failure). The LTA 80 K rate 0.10 was certified this way and production then gave median min ESS/N 0.257 < 0.30 (sweep) / 0.262 (histogram) (`RESULTS.md` Stage 5; `LTA_HISTOGRAM_REPLICATION.md` §4.7). |
| N7 | **LTA ancestor ESS is run-long, never reset** (`core_lta.py:390`; only updated at events). | code read | A fixed floor on it is a cap on run length, the WCA "gate fails its own null control" trap (`docs/V2_PREREGISTRATION.md` Amendment 12; the sham at the same event count reads ESS 0.40–0.44 vs FR 0.26–0.44, `LTA_HISTOGRAM_REPLICATION.md` §5 item 5). The equal-budget numba engine adds a windowed genealogy over 4 t.u. (`docs/equal_budget/LTA_TIMESTEP_VALIDATION.md` §5). |
| N8 | **Minor deviations.** (a) The LTA validation prereg says the 95 % upper bound uses a group bootstrap of 2000 (`lta_validation_prereg.json:19`); the code and the stored summaries use 300 (`analyze_lta_validation.py:133`, `n_boot: 300` in `results/numerical_validation/lta/T*/summary.json`). (b) The 300 K umbrella reference uses κ = 300 kJ/mol/rad² (window sd 0.091 rad), not the sweep rule κ = kT/(0.07 rad)² = 509; it was reused from v1 (`lta_sweep_prereg.json:72-80`). (c) `kl_uniform` is a KDE (0.10 rad) KL to uniform, and no finite-N floor was computed for it (`core_lta.py:475-480`). | code / config read | None changes a verdict. (a) should be disclosed; (b) is harmless (G_LTA_reference was evaluated on that artefact); (c) matters for any ERI K-ladder at small N. |

---

## 1. Which conclusions are valid, and which are superseded or refuted

Status words follow `EVIDENCE_LEDGER.md:9-20` (VALIDATED, VALIDATED NEGATIVE, CORROBORATING NEGATIVE, NOT ESTABLISHED,
REFUTED, SUPERSEDED, HYPOTHESIS, OPEN). The four claim types must not be conflated (`EVIDENCE_LEDGER.md:82-86`): (1)
fixed-N, fixed physical time per replica; (2) equal force-evaluation budget; (3) wall-clock on stated hardware; (4)
correctness of the limiting estimate.

### 1.1 Currently valid (as of `FINAL_SYNTHESIS.md`, 2026-10-11, plus the 2026-10-15 short-T ladders)

| claim | type | status | pointer |
|---|---|---|---|
| C1: FR lowers the finite-time F error of ABF with many short replicas (gateway −24…−41 % at N 512–2048; LTA 300/150 K −13.7 / −28.5 % vs exact Gibbs, sham neutral) | 1 | VALIDATED (conditional) | `EVIDENCE_LEDGER.md:73`; `FINAL_SYNTHESIS.md` Q1 |
| C2: FR accelerates **establishment**, not discovery (first arrival −17 % … +9 %) | 1 | VALIDATED (N ≥ 16) | `EVIDENCE_LEDGER.md:74` |
| C3: a better CV marginal does not imply a better F | 1 | VALIDATED | `EVIDENCE_LEDGER.md:75`; `FINAL_SYNTHESIS.md` Q3 |
| C4: slow conditional relaxation can make FR harmful | 1 | effect VALIDATED (straight fibre, N 512); mechanism HYPOTHESIS; scope corrected in errata | `EVIDENCE_LEDGER.md:76`, errata `:121-126` |
| C5: no advantage over the best ABF allocation at equal force evaluations (gateway +15.9 % [−4.2, +26.6]; LTA 300 K +3.4 %; 150 K +5.2 %; fresh-seed confirmation: gateway ABF better +24.0 % [+1.1, +52.9], LTA ties) | 2 | VALIDATED | `EVIDENCE_LEDGER.md:77`; `docs/equal_budget/FINAL_RESULTS.md` §7; `CONFIRM_BEST_ALLOCATION.md` results |
| C6: at fixed N the gain carries over to wall-clock on 1–16 CPU cores | 3 | VALIDATED (fixed N) | `EVIDENCE_LEDGER.md:78`; `FINAL_SYNTHESIS.md` Q6 |
| C7: the gateway gain is not an integrator artefact (Ī_F −29.5 / −28.4 / −29.7 % at h 4e-4 / 1e-4 / 2.5e-5) | 1 | VALIDATED | `EVIDENCE_LEDGER.md:79` |
| C8: an entropic bottleneck is not necessary | 1 | VALIDATED | `EVIDENCE_LEDGER.md:80` |
| Exp A: transverse mobility rescues FR on the curved fibre (outcome A at both N), transport confound stated | 1 | VALIDATED | `FINAL_SYNTHESIS.md` Q4 |
| Exp B: best FR vs best ABF on 8 dedicated cores is **unresolved** (LTA 150 K confirmatory ρ_best 0.717 [0.536, 1.196]; gateway exploratory 0.763 [0.515, 1.109]) | 3 | null, underpowered for ≈ 25 % effects | `FINAL_SYNTHESIS.md` Q6; `EXPB_WALLCLOCK.md` §1–§3 |
| WCA at dt\* = 0.000125 vs exact MC: FR harmful (+9.9 % [+2.3, +13.2] Ī_F, final +131 %); sham neutral; stationary 0.017 kT tilt | 1, 4 | VALIDATED NEGATIVE | `EVIDENCE_LEDGER.md:51` (W8); `numerical_validation/WCA_CONFIRMATORY.md` |
| WCA dt 0.0005 ladder harmful at N 16–1024 vs exact Gibbs | 1 | CORROBORATING NEGATIVE | `EVIDENCE_LEDGER.md:50` (W7) |
| LTA EM dt 2e-4 samples the model's Gibbs F(φ) at 300 and 150 K | — | VALIDATED (G_LTA_dyn ADMISSIBLE at both T) | `LTA_VALIDATION.md` §3, §4, §5 |
| LTA histogram-ABF replication of the kernel temperature sweep: integrated gain within 0.0–3.8 points at 80/150/225/300 K, 16/16; 350 K −11.8 % [−15.2, −9.5]; sham neutral everywhere; FR beats sham directly 16/16 | 1 | valid **against the published umbrella reference**; 300/150 K re-validated vs MC; **80/225/350 K not re-validated** (`LTA_VALIDATION.md:204-206`) | `LTA_HISTOGRAM_REPLICATION.md` §5; `results/lta_histogram/README.md` |
| Entropy share does not predict the FR benefit (refuted at five temperatures); establishment starvation (ABF window traffic 0.23 → 3.23 crossings/replica maps onto −35 → −12 %) does in LTA | 1 | valid as a cross-T association; the crossing counts carry N1; temperature is a budget axis as much as a landscape axis (`RESULTS.md` Stage 5 caveat 4); the gateway seed-level starvation test has the opposite sign (ρ = +0.21, p = 0.019, `FINAL_RESULTS.md` §8) | `LTA_HISTOGRAM_REPLICATION.md` §4.5, §5 item 3 |
| LTA FR leaves no stationary bias at 300 K (FR limit vs ABF limit D = 0.000, upper 0.018) and reduces ABF's finite-time bias at 150 K (D vs MC 0.131 → 0.057) | 1, 4 | valid | `LTA_VALIDATION.md` §3.2, §4 |
| Equal-budget Results 1–3 and the conclusions of `FINAL_RESULTS.md` §10 (per-replica-time accelerator, not a force-evaluation accelerator; switch FR off at small N; σ_KDE ≥ 1 at N ≤ 16) | 1, 2 | valid | `FINAL_RESULTS.md` §1, §10 |
| Shortened-T ladders (T re-anchored at FR-converged T\*: gateway 12.6 of 40, LTA 300 K 51.3 of 60, 150 K 23.1 of 60): FR wins at large N, best-allocation ratio never significant (gateway 1.16 [0.76, 1.63], LTA 300 K 1.04 [0.69, 1.38], 150 K 1.06 [0.83, 1.46]) | 2 | valid (no new dynamics; truncation of stored runs) | commit `5f95131e` message; `results/equal_budget_v2/short_T/` |
| Histogram (P0) ABF is an acceptable estimator: EB replicates the integrated gain (−31 % vs −38 %), not the endpoint; gateway replicates (−30 % / −48 %) | 1 | valid for EB and gateway; **its WCA part is REFUTED** (row W3 below) | `HISTOGRAM_ABF_REPLICATION.md` §2.2, §5.2, §6 |
| The accepted EB (Case II) kernel ABF endpoint 0.21 is a kernel-bandwidth frontier stall, not starvation | — | valid | `HISTOGRAM_ABF_REPLICATION.md` §2.1, §2.3 |
| ZIF-8 300 K on a corrected ABF baseline (h_bias 0.10 Å): FR **NEUTRAL** (+0.80 % [−1.30, +2.81]) | 1 | valid; supersedes the HARMFUL verdict | `docs/ZIF8_CORRECTED_BASELINE.md` "Verdict" |
| ZIF-8 mechanism: F converged (5 % at 54 ps) before the marginal was established (T_marg 77 ps); ABF needs adequate, not equal, per-bin counts | — | valid as an observation; the HARMFUL magnitude was a baseline artefact | `docs/ZIF8_RESULT.md` "The mechanism" |
| Conditional-law identity: mean-field marginal selection leaves p(y \| ξ) invariant, for any ξ-only score | identity | valid, mean-field only (finite-N cloning can perturb it indirectly) | `V2_PREREGISTRATION.md` Amendment 8 |

### 1.2 Superseded, refuted, withdrawn or at risk

| statement | where it appears (not edited) | status | superseding evidence |
|---|---|---|---|
| Every WCA FR gain at dt 0.002 (Case IX v1/v2 −22.8 %/−17.97 %, Q1 five-arm, uniform-campaign Stage 2 −21.9 %, corrected-read-out R1 −18.3 %, FR-start ladder, OT/repair, phase diagram, OPES, atlas rows) | `CLOSURE_v1.md`, `report/`, `docs/UNIFORM_FR_CAMPAIGN.md` Stage 2 / `RESULTS.md` scoreboard row 2, `WCA_CORRECTED_CONFIRMATION.md`, … | NOT ESTABLISHED | `EVIDENCE_LEDGER.md:42-49` (W1a–W6). Root cause: EM unstable at dt 0.002 (dt·V″ = 2.05 at r = 0.95σ), non-gradient force clip, TI references built with the same integrator (0.282 / 0.288 kT RMS from exact) (`EVIDENCE_LEDGER.md` §1; `WCA_FORCE_AUDIT.md` §1) |
| Histogram-ABF replication of the WCA gain (−23 % / −51 %, 16/16) | `HISTOGRAM_ABF_REPLICATION.md` §2.5, §6 ("a clean replication") | **REFUTED**: same seeds vs exact Gibbs give +18.5 % [+17.9, +19.6], 0/16 | `EVIDENCE_LEDGER.md:46` (W3) |
| WCA dt 0.002 replica ladder (FR wins 16/16 at every N) | `docs/WCA_REPLICA_LADDER.md` §2 | **REFUTED** (+18…+29 %, 0/16) | `EVIDENCE_LEDGER.md:47` (W4) |
| "FR reduces the total computational cost of free-energy estimation" | — | never established | `EVIDENCE_LEDGER.md` §3; C5 |
| "FR beats the best ABF allocation on parallel hardware" | — | never established (Exp B unresolved) | `FINAL_SYNTHESIS.md` decision item 2 |
| Valine as a measured FR-neutral control | several docs | withdrawn: no FR arm was run (gate V3.2 FAIL, ABF-sufficient) | `EVIDENCE_LEDGER.md:118-120` |
| "Alanine neutral across N, T, γ, CV and dose" | `numerical_validation/OVERNIGHT_REPORT.md` §4; `EVIDENCE_LEDGER.md:73` (C1 counter-evidence) | corrected: neutral over N, T, start, histogram estimator; harmful at early high dose (kernel, +10.5 %) and on the φ-only CV (+18.9 %); no FR arm on the γ ladder | `EVIDENCE_LEDGER.md:113-117` (errata) |
| "The equal-force-evaluation ladder has not been run for LTA"; "the LTA benefit is a parallel wall-clock gain" | `LTA_VALIDATION.md` §5; `OVERNIGHT_REPORT.md` §4 | superseded: LTA ladders run (C5: no equal-budget advantage); wall-clock at best allocation unresolved (Exp B) | `FINAL_RESULTS.md`; `EXPB_WALLCLOCK.md` §1 |
| ZIF-8 300 K HARMFUL (+3.67 % [+1.97, +5.02]) | `docs/ZIF8_RESULT.md` | SUPERSEDED (baseline artefact: legacy h_bias 0.20 Å) | `docs/ZIF8_CORRECTED_BASELINE.md`: +0.80 % [−1.30, +2.81], ABF final e_F 0.2747 → 0.0838 kJ/mol |
| LTA 300 K v1 (kernel, fr_start 40 000 = t 8): null, final +7.44 % | `RESULTS.md` Stage 4 | stands as a measurement of that protocol; superseded as the system verdict by the v2 sweep (fr_start 20 000 = t 4), −14.84 % | `RESULTS.md` Stage 5 finding 1; `lta_sweep_prereg.json:4` |
| Gateway uniform-FR "transient, then late reversal" | `RESULTS.md` Stage 1 | the reversal was the legacy read-out's (h_read\* 0.0175 gives −31.9 %/−59.4 %); the historical h 4e-4 fails the transverse-variance gate (+26 % fibre variance), but the gain reproduces at h 2.5e-5 | `BANDWIDTH_DEFECT_SCREEN.md` §3; `FINAL_RESULTS.md` §2, Q8; C7 |
| EB Case II endpoint gain −50 % | report Case II | the kernel frontier stall; the histogram endpoint gain is −14 % | `HISTOGRAM_ABF_REPLICATION.md` §2.2–§2.3 |
| "FR generates the extra window traffic; the sham does not" (LTA 150 K +18 %, 80 K ×1.9; kernel 80 K 7081 vs 3840) | `LTA_HISTOGRAM_REPLICATION.md` §4.4, §4.7, §5 item 2; `RESULTS.md` Stage 5 finding 3 | **AT RISK** (this review, N1): the counter is inherited on clones | Appendix A |
| The LTA "τ speed-up to ABF's final accuracy" values (5.26× … 1.03×) | `LTA_HISTOGRAM_REPLICATION.md` §5 table | **qualified** (N3): ABF's τ at its own final error equals T by construction | `scripts/analyze_uniform_lta.py:38-48` |
| WCA mechanism ("fibre lag") | `WCA_CONFIRMATORY.md` | HYPOTHESIS (W9); dt → 0, N → ∞ OPEN (W10) | `EVIDENCE_LEDGER.md:52-53` |
| v1 report's variance-reduction mechanism attribution | `CLOSURE_v1.md` §5a | not to be cited | `EVIDENCE_LEDGER.md:97-98` |
| Information-adequacy clock T_info as a predictor | `docs/INFORMATION_CLOCK_AUDIT.md` | FAILS (Spearman −0.437, p 0.24, n 9; the endpoints are bias, not variance) | per that doc; project memory "information-clock-and-bandwidth" |
| Novelty of "FR accelerates ABF" | — | high novelty risk: Lelièvre–Rousset–Stoltz 2007, Minoukadeh 2010, Comer 2014/2015, Amber ABMD (as the repo summarises them; not verified here) | `PRIOR_ART_NOVELTY.md` §1, §2 |

---

## 2. Definitions and conventions

### 2.1 Free-energy error e_F(t), final error e_F(T), integrated error I_F / Ī_F

Several versions coexist. They are not interchangeable.

**(a) LTA uniform campaign, kernel sweep and histogram study** (`analyze_uniform_lta.py:51-65`, imported by
`analyze_lta_histogram.py:57-58, 368-381`; endpoint text `configs/lta_histogram/campaign.json:125`).
* Reference: `reference_T{T}.npz` (umbrella/WHAM, published), interpolated circularly by linear `np.interp` onto the
  engine's 180 cell centres (`circular_interp_ref`).
* e_F(t) per save and seed: `d = F̂ − F_ref`, `d ← d − mean_j(d)` (additive-constant alignment = subtract the mean
  difference over all 180 bins), `e_F = sqrt(mean_j d²)`, i.e. an **unweighted** RMS over the **full circle**, kJ/mol.
* F̂ = the reported estimator: production accumulators (steps ≥ 20 000) once they hold any count, else all-steps
  (`core_lta.py:466-469`); histogram arms are scored by their own exact P0 PMF, "no kernel re-scoring" (`campaign.json:125`).
* Time grid: saves every 3 000 steps = 0.6 t.u., t = 0 … 60, **101 saves including t = 0** (verified on the arrays).
  The t = 0 save holds one step of deposits (median e_F 10.15 kJ/mol at 300 K).
* **I_F = trapezoid of e_F(t) over the saves in physical time** (`np.trapezoid(err, t)`), units kJ/mol·t.u.
  (300 K ABF median 22.98). Includes the t ∈ [0, 4] pre-FR segment (≈ 40 % of I_F, N4) and the t = 0 point.
* e_F(T) = e_F at the last save (t = 60).
* Paired statistic: per-seed Δ = 100 (arm − base)/base for I_F and for e_F(T) (`analyze_lta_histogram.py:135-142`).

**(b) Equal-budget ladders** (`docs/equal_budget/SCIENTIFIC_PLAN.md` §4 lines 96-117; `eqb_metrics.py:320-341, 572-588`).
* LTA reference: the **exact Metropolis-MC** reference (`results/equal_budget_v2/references/lta_T{300,150}_reference.npz`),
  never the published umbrella reference (`SCIENTIFIC_PLAN.md:70-75`).
* e_F: RMS over the 180 bins of F̂ − F_ref after removing the mean difference (`lta_errors_from`, `eqb_metrics.py:334-341`).
  Gateway: RMS over the 151-node eval window [−1.5, 1.5] (the accepted scorer).
* **Ī_F = mean of e_F over the 200 uniform budget fractions u = k/200, k = 1…200** (excludes u = 0)
  (`FINAL_RESULTS.md:22`; `SCIENTIFIC_PLAN.md:112`). Each run is indexed on its own save grid (`uniform_indices`);
  the short-T pitfall was using another N's grid (project memory "short-t-ladders").
* Sign: G = (FR − ABF)/ABF per seed; negative = FR better (`FINAL_RESULTS.md:20-21`). Medians, not means — a
  preregistered deviation from the task text (`FINAL_RESULTS.md:23-35`).

**(c) Exp B wall-clock** (`docs/final_campaign/PREREGISTRATION.md:408-430`): W_mid = wall seconds to the first save of the
persistent window at the mid threshold; Ī_F^wall(W\*) = (1/W\*) ∫₀^{W\*} e_F(w) dw with a step-function e_F and W\* the
smallest median completion wall time.

**(d) Other systems, for contrast.** WCA: aligned RMS by trapezoid/width over z ∈ [−0.1, 1.1] (`HISTOGRAM_ABF_REPLICATION.md:43-45`).
EB: F gauged at x = 0 then aligned, mean over nodes in [−1.5, 1.5] (`:19-20`). ZIF-8: aligned RMS over the full circular
domain, "no mask to choose and no place to hide a gain outside one" (`zif8_prereg.json:124`).

### 2.2 Mean-force error e_F′

* LTA, equal-budget: `e_Fp` = RMS over the 180 bins of the raw histogram mean force Γ_j minus the MC conditional mean
  force γ_ref,j (**not aligned**: the mean force has no gauge); `e_Fp_proj` subtracts both circular means first
  (`eqb_metrics.py:334-341`; `SCIENTIFIC_PLAN.md:101-102`). Units kJ/mol/rad. The LTA mean-force reference is the MC
  per-bin conditional mean itself, so the LTA read-out floor is 0 (`FINAL_RESULTS.md:68`); its per-bin jackknife SE is
  0.0143 (300 K) / 0.0101 (150 K) kJ/mol/rad (`LTA_TIMESTEP_VALIDATION.md:62-66`).
* Gateway / EB / WCA histogram: function-space RMS of Γ(x) − F′_ref(x) by 8-node Gauss–Legendre per bin, never on a bin
  edge (`HISTOGRAM_ABF_REPLICATION.md:87-98`). Deterministic P0 floor at 180 gateway bins: e_F′ = 0.0323, which made the
  frozen strict threshold 0.012 unreachable (`FINAL_RESULTS.md:61-68`); e_F′_stat = sqrt(e² − floor²) is a labelled
  descriptive companion.
* ZIF-8: non-aligned RMS of F̂′ − F′_ref (`zif8_prereg.json:126`).

### 2.3 The D metric (debiased RMS), LTA validation

`lta_validation_prereg.json:19`; implementation `analyze_lta_validation.py:133-145`.
* d_j = F_A,j − F_B,j on the 180 production bins, centred (mean removed).
* noise² = mean_j (se_A,j² + se_B,j²), se from a leave-one-group-out jackknife over the 16 independent groups,
  se² = (G−1)/G Σ(loo − mean)².
* **D = sqrt(max(0, mean_j d_j² − noise²))**, kJ/mol.
* 95 % upper bound: 95th percentile over group-bootstrap resamples of sqrt(max(0, mean d_b² − 2·noise²)); **300 resamples**
  in code and summaries, 2000 in the prereg (N8a).
* A fixed comparator (the published reference) gets se = 0 ("treated as noiseless, conservative toward FAIL").
* The same statistic compares pooled long-run limits (sum of per-seed final accumulators) between arms
  (`LTA_VALIDATION.md` §3.2). The WCA campaign uses the same D in kT with tolerance 0.005 kT (`EVIDENCE_LEDGER.md:36`).
* Barrier scalar, two conventions: `analyze_lta_validation.py:148-153` (mean F over |z| < 0.4 Å minus mean over cage bins
  |z| > 4 Å) gives 26.40 / 16.55 kJ/mol; `run_lta_reference.py:75-80` ("within 0.4 Å of the cage centre") gives
  26.79 / 16.84 kJ/mol (`LTA_TIMESTEP_VALIDATION.md:55-60`, flagged as a documentation defect).

### 2.4 Marginal KL / TV and the finite-N floor

* **LTA engine diagnostic** `kl_uniform` (`core_lta.py:475-480`): p̂ = KDE marginal of the instantaneous walker φ
  (wrapped Gaussian kernel matrix, bandwidth `kde_bandwidth` = 0.10 rad, on the 180-bin grid, normalised); KL = Σ_j p̂_j
  log(p̂_j / (1/2π)) Δφ. Medians over seeds are probed at t = 4, 12, 60 (`analyze_lta_histogram.py:98, 584`).
  No finite-N floor is computed (N8c). Also stored: `pq_l2`, `kl_pq` against the target.
* **Equal-budget** (`SCIENTIFIC_PLAN.md:103-110`; `eqb_metrics.py:432-514`), on 18 coarse bins (10 production bins each):
  * TV_half(t): TV to uniform of the trailing-half visitation C_all(t) − C_all(≈ t/2) (nearest candidate save, ties to the
    earlier). Defined at every N, including N = 1. **Used for marginal τ** (threshold TV_half ≤ 0.10).
  * TV_inst(t): instantaneous walker histogram, reported with its **exact finite-N floor**
    E_N[TV] = (K/2) E|X/N − 1/K|, X ~ Bin(N, 1/K), K = 18 (`eqb_metrics.py:452-463`; at N = 1 it is 17/18). Never
    used for τ.
* ZIF-8: J(t) = TV(p_t^ξ, uniform) (`zif8_prereg.json:85`).
* Alanine: histogram KL(p ‖ U) on 9409 cells has the floor log(9409/N) (`docs/ALANINE_NLADDER.md:87`).

### 2.5 Discovery and establishment timing, thresholds, and the recorded defects

**Definitions in use.**

| definition | rule | where |
|---|---|---|
| LTA screening rule (proposed for choosing T\* from ABF-only data) | T_hit/T_run < 0.1 **and** 0.25 < T_est/T_run < 0.75; T_hit = median over seeds of the first save with any replica in the window (|z| < 1.5 Å); T_est = time the **median** ABF error curve comes within 1.5× its **final** value, persistence 0.2 T | `campaign.json:124`; `analyze_lta_histogram.py:554-604` |
| LTA observed | T_hit 1.2–3.0 t.u. (T_hit/T_run 0.02–0.05); T_est 34–43 of 60 (0.56–0.71) at every T; with τ(e₀/8) (1.8–4.2) every T would be "too easy" although FR gains 12–35 %, so "the threshold must be tied to the endpoint" | `LTA_HISTOGRAM_REPLICATION.md` §1 lines 41-56, §5 item 4 |
| Equal-budget first arrival | first save with a walker in the far state (LTA window, opposite cage; gateway right well) | `SCIENTIFIC_PLAN.md:139` |
| Equal-budget population establishment | first save after which the instantaneous far-state fraction stays ≥ ½ of its uniform-target value at **every later save**: LTA window |z| < 1.5 Å, target 0.252; gateway x > 0.5, target 0.361 | `SCIENTIFIC_PLAN.md:133-139`; `eqb_metrics.py:92-95` |
| …its null calibration | for an exactly uniform population, far count ~ Bin(N, target) independently across saves; the rule is UNRELIABLE when P(no failure in the second half) < 0.9 or N ≤ 2 → LTA N ≤ 64, gateway N ≤ 32; there cumulative establishment and first arrival are read instead | `FINAL_RESULTS.md:69-72`; `eqb_metrics.py:544-569` |
| Equal-budget marginal τ | persistent τ(TV_half ≤ 0.10) | `SCIENTIFIC_PLAN.md:107, 122-124` |
| ZIF-8 three clocks | T_cover: first save at which every grid bin has been visited; **T_marg**: with J_inf = median over the last 20 % and J_0 = first **post-warmup** value, first t with J ≤ J_inf + 0.2 (J_0 − J_inf), sustained 0.1 T (relative, not absolute); T_gate: same rule on a ξ-resolved JS divergence of the hidden-gate conditional against the reference | `zif8_prereg.json:84-86`, A1 `:194-198` |
| CHA (superseded rule) | T_marg = first save with TV < 0.10 sustained 0.1 T — "beta-naive", amended A1 | `cha_prereg.json:62-70, 102` |
| V2 Gate B (discovery) | T_hit,k < 0.1 T on ≥ 6 of 8 ABF-only seeds, else discovery-limited STOP | `V2_PREREGISTRATION.md` §2.3 |
| V2 Gate C (establishment) | under-established if occupancy < 0.5 Q\*_k(t) (bias-aware target, §2.1) for a contiguous span ≥ 0.20 T in the second half | `V2_PREREGISTRATION.md` §2.4 |
| V2 Gate D (clone decorrelation) | λ_rep τ_⊥ ≤ 0.1, τ_⊥ = first t with twin correlation C_Y(t) ≤ 1/e | `V2_PREREGISTRATION.md` §2.5 |

**Recorded defects in gate power or vacuous boundaries** (all from repo docs unless marked "this review").

1. *Exp B V-B4 equivalence band had no power.* The validated engine fails the frozen ABF-16 band [0.85, 1.18] against
   itself in 3/3 null controls (1.24 [0.90, 1.54] etc.), because block medians of 32 seeds vary ×1.24
   (`PREREGISTRATION.md` A3.2, lines 657-680). Lesson recorded: compute the null control at the planned n before
   freezing a band.
2. *Gate C (contiguity) does not fire at the deficit it nominally tests:* 0/8 seeds at a planted 50 % deficit, 8/8 at
   55–60 %; contiguity, not counting noise, sets the floor (`V2_PREREGISTRATION.md` §12.13, §16.11). Fix adopted for C60: a
   windowed band is the primary instrument, a contiguity non-fire carries no weight.
3. *A span frozen as a fraction of T silently varies the instrument along an N ladder:* with N·T fixed, 0.20 T scales as
   1/N, 8× blunter at N = 8 than N = 64, in the direction that hides failure; fix it in absolute time (multiples of a
   measured τ) (`V2_PREREGISTRATION.md` §12.13 continuation ≈ lines 1950-1965). Applies to every "0.2 T" persistence and
   to T_hit/T_run, T_est/T_run if ERI runs a ladder.
4. *Run-long ancestor ESS floor is a cap on run length* (the WCA mFR 0.166 vs sham 0.191 case), fixed by a windowed ESS
   (`V2_PREREGISTRATION.md` Amendment 12; project memory "wca-measurement-traps" trap 3). LTA still uses run-long ESS (N7).
5. *Short calibration over-certifies the FR rate* (CHA 150 k → production ESS 0.221) (`zif8_prereg.json:115`). LTA
   calibrations ran at 120 k of 300 k steps (N6).
6. *ZIF-8 A1 classifier defects* (`zif8_prereg.json:193-208`): `isfinite(inf)` routed a never-equilibrating gate into
   ABF-sufficient; J_0 read at t = 0 instead of post-warmup biased T_marg short (33 ps vs 148 ps on a test curve, a verdict
   flip); degrading curves met the relative threshold at i = 0; persistence windows truncated at the tail; a mixture
   statistic stood in for a conditional; `accept_gate_js` 0.02 passed by four orders of magnitude ("a gate that could not
   fire"); the coverage hold clause was a no-op; clone-inherited transit counts; the verdict tree (N2); pre-FR dilution of
   the primary.
7. *Gateway strict e_F′ threshold unreachable by construction* (P0 floor 0.0323 > 0.012) (`FINAL_RESULTS.md:61-68`).
8. *Instantaneous establishment rule fails by chance at small N* (F2 above).
9. *τ tied to the initial error is trivially easy* (e₀/8 at t = 1.8–4.2 because e₀ is a one-step estimator) — the LTA
   screening-rule lesson (§1 of `LTA_HISTOGRAM_REPLICATION.md`).
10. *Gate thresholds at unattainable precision, wrong-quantity gates, transposed specs* (NaCl `R_hi`; methane 0.1 kT/nm vs
    0.5–0.67 SEM; Gate A as TV(p(Y|ξ)) instead of TV(p(ξ|Y))) — "A gate that cannot fire reads as reassurance"
    (`V2_PREREGISTRATION.md` §12.9, quoted block ≈ line 1546).
11. *NaCl T_hit degenerated into the trace resolution* (0.5 ps = first recordable frame) (project memory
    "analysis-code-failure-classes"). In LTA T_hit is 2–5 save intervals (0.6 t.u. each) — resolved, but coarsely.
12. *(this review)* the LTA τ window still truncates (N3), so τ at ABF's own final accuracy and T_est cannot be censored.
13. *(this review)* the LTA crossing counter inherits on clones (N1).

### 2.6 Paired-median bootstrap CIs, wins, sign tests

* Per seed d = 100 (arm − base)/base; statistic = median over seeds; CI = 2.5 / 97.5 percentiles of 10 000 bootstrap
  medians (seeds resampled with replacement); wins = #seeds with d < 0.
  * LTA campaign/sweep/histogram: `default_rng(20260829)` for I_F, seed + 1 for e_F(T) (`analyze_uniform_lta.py:28-35`;
    `analyze_lta_histogram.py:446-449`).
  * Equal budget: `default_rng(20261010)`, non-finite values dropped (`eqb_metrics.py:893-916`).
* τ contrasts with censoring (equal budget): rank part over all common seeds (finite beats ∞, ∞ vs ∞ tie) with an exact
  two-sided sign test; ratio part only on seeds where both are finite (`eqb_metrics.py:919-948`).
* Exp B W quantities must **not** use these helpers (they drop non-finite values); ∞-aware medians (`quantile_inf`),
  ratios with finite/∞ = 0, ∞/finite = ∞, ∞/∞ undefined; > 2.5 % undefined resamples = unresolved
  (`PREREGISTRATION.md:432-441`).
* With n = 16 the bootstrap CI endpoints sit on a handful of order statistics; effects below 15–25 % are not resolvable at
  small N in the ladders (`FINAL_RESULTS.md:669-671`).

### 2.7 TOST / equivalence rules (they differ by study)

| study | rule | pointer |
|---|---|---|
| ZIF-8 (corrected tree) | NEUTRAL = TOST: whole 95 % CI inside ±10 %, health floors met, final margin held; HARMFUL = median ≥ +10 % or CI lower > 0; else INCONCLUSIVE | `zif8_prereg.json:139-145` |
| WCA confirmatory | FR_neutral = CI inside [−10 %, +10 %] ("TOST-style"); FR_harmful = CI lower > 0; FR_beneficial needs FR vs ABF **and** FR vs sham CIs < 0 | `wca_confirmatory_prereg.json` "decision" |
| Gateway transport preregs | equivalent = 90 % CI inside ±10 % (decision on the 95 % CI) | `configs/transport_campaign/gateway_horizontal_transport_prereg.json:48-52` |
| Exp A | C-equivalent = p_Holm ≥ 0.05, 95 % CI of R inside [0.90, 1.11], G_hi ≥ 0 | `PREREGISTRATION.md:218` |
| Exp B V-B4 | 95 % CI of the ratio of median Ī_F inside [0.85, 1.18] and G-difference CI within ±0.10, one pre-specified extension (failed its null control, §2.5 item 1) | `PREREGISTRATION.md:322-330` |
| Equal-budget engine equivalence | TOST ±5 % on I_F planned; "not shown" because per-seed I_F scatter is 8–9 % | `docs/equal_budget/audit/AUDIT_0A_lta.md:344`; `LTA_TIMESTEP_VALIDATION.md:96-97` |
| LTA campaign/sweep/histogram (old tree) | NEUTRAL = point estimates only (N2) — not TOST | `analyze_lta_histogram.py:478-491` |

A "whole 95 % CI inside ±m" rule is TOST at α = 0.025 per side; a 90 % CI rule is the textbook α = 0.05. ERI should state
which. The repo's standing rule is that equivalence is shown by TOST, "never by 'no CI excluded zero'"
(`V2_PREREGISTRATION.md:303`).

### 2.8 Seed as the unit; pairing

* The independent unit is the seed (one independent run of N walkers). Walkers within a run are never treated as
  independent (`PRIOR_ART_NOVELTY.md` §3 Grossfield row; `ORACLE_SHORT_BURST_STAGE_A_RESULT.md:134`). Validation
  statistics use independent chains (WCA: 64 MC chains; `WCA_LANGEVIN_VALIDATION.md` §5) or independent groups (LTA: 16
  groups × 40 windows; `lta_validation_prereg.json:8`).
* LTA GPU studies: the 16 seeds are 16 rows (R = 16) of one batch, all drawn from one generator seeded by the per-T
  `rng_seed`; the seed numbers are labels (`campaign.json:84`). ABF, FR and sham run in one process on the same initial
  conditions and noise stream; FR/sham use a separate generator (`rng_seed + 987654321`, `core_lta.py:347-348`). The
  histogram arms are noise-paired with the kernel sweep's arms by sharing `rng_seed`.
* Pairing evidence: ABF and FR PMFs agree to ≤ 3e-13 at every save before t = 4 in the histogram productions (this
  review, Appendix A); ZIF-8 asserts bit-identity for all 60 000 pre-FR steps (`zif8_prereg.json:176`); the equal-budget
  audit checks ABF/FR arms bitwise identical before FR's first event (`FINAL_RESULTS.md:76-80`).
* WCA GPU runs were **not** paired (nondeterministic float32 scatter-add in the estimator, chaotic amplification); the
  CPU numba path is bit-reproducible (project memory "wca-measurement-traps" item 4 and its 2026-09-02 correction).
* When one seed has two initial conditions (gateway left / one_right), the later preregs bootstrap by seed cluster
  (`gateway_horizontal_transport_prereg.json:48`); the histogram-campaign gateway block reported medians over 32 rows
  (`HISTOGRAM_ABF_REPLICATION.md` §5.2).

### 2.9 Holm and multiplicity

* Exp A: E1 (λ 16 vs λ 1) Holm over the two N (family size 2); E2 (λ 128) a separate family; MC1 Holm over 30 p at FWER
  5 %; V-B3 RNG tests one Holm family of 9 at FWER 1 % (`PREREGISTRATION.md:92-100, 148, 321`).
* Exp B: intersection–union over ABF arms with Bonferroni over FR arms (`PREREGISTRATION.md:442-446`).
* Equal-budget ladders and the LTA temperature studies report each N or T with its own unadjusted 95 % CI; there is no
  family-wise correction across the 11 N or 5 T. A K-ladder in ERI needs a declared family and correction.

### 2.10 Matched-turnover sham

* Definition (LTA): `fr_sham` replays its partner `fr_uniform`'s realised replacement count at every FR opportunity of
  the same seed (`event_counts` replay); dying replicas uniformly without replacement, clone sources uniformly **with
  replacement among the survivors**; no score, no target; same seeds, initial conditions and noise
  (`campaign.json:86`; `core_lta.py:294-319`, shape check `:353-359`). The FR multinomial can also repeat a source
  (`LTA_HISTOGRAM_REPLICATION.md:370-371`).
* Attribution rule: FR is attributed to its direction if the **direct** FR-vs-sham ΔI_F has median < 0 and CI upper < 0;
  sham-vs-ABF reported alongside (`campaign.json:123`). Same design on WCA (`wca_confirmatory_prereg.json` "arms").
* Results it delivered: LTA sham neutral (|ΔI_F| ≤ 1.2 %) at every T, FR beats sham by 93–98 % of its FR-vs-ABF margin
  (`LTA_HISTOGRAM_REPLICATION.md` §5 item 2); WCA sham neutral (−0.8 %), so the WCA tilt comes from score-directed
  selection, not cloning (`WCA_CONFIRMATORY.md`).
* **ERI:** `docs/eri_campaign/STATUS.md:31-34` records a user instruction of 2026-10-16: two arms only, no sham, attribution
  from the direct FR-vs-ABF contrast plus mechanism diagnostics. The repo's attribution statistic is therefore unavailable;
  see lesson L17.

### 2.11 Safety-only FR rate calibration

* Ladder {0.02, 0.05, 0.10, 0.20}; select the **largest** rate whose calibration run has min ancestor ESS/N ≥ 0.30 **and**
  max lineage share ≤ 0.05 over the FR-active saves; chosen without reading e_F, I_F or τ and without loading the reference
  (`lta_prereg.json:44-49`; `lta_sweep_prereg.json:84`; `campaign.json:85`). Per-opportunity turnover is capped
  structurally by `max_event_fraction`; events/replica reported, not gated.
* LTA outcomes: 0.20 at 150–350 K, 0.10 at 80 K (its 0.20 rung failed ESS) (`RESULTS.md` Stage 5).
* ZIF-8 version adds an event-fraction clause and requires the **full production horizon** and the production N
  (`zif8_prereg.json:102-116`; A1 raised calibration N to 384). LTA did not meet the horizon requirement (N6).
* Dose control: re-earning the rate under changed dynamics (ZIF-8 corrected baseline returned the same rate) is what made
  the baseline change attributable (`ZIF8_CORRECTED_BASELINE.md`).

### 2.12 ESS / ancestor diagnostics

* `_ancestor_stats` (`src/alkanes/core.py:163-175`): family fractions w_a from the ancestor labels; ESS = 1/Σ w_a²;
  number of unique ancestors; max lineage share = max w_a.
* LTA: labels set once at t = 0 (`core_lta.py:390`), so ESS is run-long. Health = median across seeds of each seed's min
  ESS/N over FR-active saves, and median of each seed's max share; worst seed alongside, never substituted
  (`analyze_lta_histogram.py:411-428`).
* Floors by system: gateway 0.30 / 0.05 (windowed, `ess_window_steps` 4000); WCA 0.10 / 0.05 (run-long, later a windowed
  `min_ancestor_ess_window`); LTA and ZIF-8 0.30 / 0.05; alanine 0.30 / 0.05 / cumulative events ≤ 5 %
  (`UNIFORM_FR_CAMPAIGN.md:108-110`; `HISTOGRAM_ABF_REPLICATION.md:272, 335`).
* Equal-budget numba engine: whole-run and windowed (4 t.u.) unique ancestors, ESS, largest family
  (`SCIENTIFIC_PLAN.md:91`; `LTA_TIMESTEP_VALIDATION.md` §5). Whole-run unique ancestors collapse to 1 at N ≤ 32 in LTA
  while the windowed ESS stays 0.5–0.8 (`FINAL_RESULTS.md:468`).
* LTA genealogy (movie, 150 K): 819 replacements in 1024 slots, families ≤ 13 members, ESS/N 0.31–0.45
  (`LTA_HISTOGRAM_REPLICATION.md` §4.9).

### 2.13 Regime classifier

**Four-way map** (`V2_PREREGISTRATION.md` Amendment 8 table; Amendment 10 order "Gate 0 → A → B → C", classification by the
first failing gate):

| regime | what fails | mFR expectation |
|---|---|---|
| ABF-sufficient | nothing important | neutral |
| discovery-limited | the state is never reached | cannot clone what does not exist |
| establishment-limited | reached, population relaxes slowly | the useful regime |
| conditional-equilibration-limited | p_t(q \| ξ) is wrong | marginal selection cannot directly repair it |

Favourable window: T_hit ≪ τ_⊥ ≪ T_est, τ_⊥ the relaxation time of hidden coordinates at fixed ξ (Amendment 10).

**Numeric thresholds where they exist.**
* Gate 0 (baseline validity): the span clause was retracted (Amendment 9); retained: no seed with > 0.90 of walkers in one
  tercile over the whole second half. No mean-force-agreement threshold was set; conditional-equilibration limitation is
  attributed by a controlled restrained-window test, not a screen (Amendment 9).
* Gate A (CV visibility): max over relevant pairs TV(p(ξ|Y=a), p(ξ|Y=b)) < 0.30 → STOP (§2.2).
* Gate B: T_hit < 0.1 T on ≥ 6/8 seeds (§2.3). Gate C: < 0.5 Q\* for ≥ 0.20 T contiguous in the second half (§2.4).
  Gate D: λ_rep τ_⊥ ≤ 0.1 (§2.5).
* ZIF-8 screen (`zif8_prereg.json:87-93`): discovery_limited if T_cover > 0.5 T or any bin never visited;
  conditional_limited if T_gate > 0.5 T with coverage; abf_sufficient if T_cover, T_marg, T_gate all < 0.25 T;
  establishment_limited if T_cover < 0.25 T and 0.25 T ≤ T_marg ≤ 0.8 T and T_gate < 0.25 T; otherwise intermediate.
  ZIF-8 300 K was classified establishment_limited (T_cover 0.08 T, T_gate 0.18 T, T_marg 0.26 T) and FR was not helpful
  (`ZIF8_RESULT.md`) — "the predictor needs a per-bin-adequacy test, not a uniformity test".
* CHA: abf_sufficient if T_marg < 0.25 T; establishment_limited if T_cover < 0.25 T and 0.25 T ≤ T_marg ≤ 0.8 T;
  discovery_limited if T_cover > 0.5 T or a scoring bin never visited (`cha_prereg.json:64-70`).
* LTA screening rule: §2.5 table. Equal budget: no formal classifier; "establishment-limited" is read as ABF needing
  ≥ 10 % of the budget to establish TV_half (`FINAL_RESULTS.md:591`).

### 2.14 Verdict trees

* Uniform/LTA tree (`UNIFORM_FR_CAMPAIGN.md:112-120`; `campaign.json:116-122`): ACCELERATION_POSITIVE if median ΔI_F ≤ −10 %
  and CI upper < 0; SAFE_ACCELERATOR if also median final change ≤ +5 % and health floors met; NEUTRAL by point estimates;
  NEGATIVE_OR_UNSAFE otherwise (N2).
* Corrected tree (ZIF-8 A1): §2.7 first row.

---

## 3. The LTA ethane model, its validation, and its reference

### 3.1 Model (`src/lta/core_lta.py:1-23, 76-228`; frozen in `configs/uniform_campaign/lta_prereg.json:4-16`)

| item | value | source |
|---|---|---|
| framework | IZA DLS76-optimised all-silica LTA, `cache/lta/LTA.cif` (Pm-3m pseudo-cell, a = 11.9190 Å); rigid; 2×2×2 pseudo-cells, box L = 2a = 23.838 Å, PBC; 576 framework atoms of which 384 O | `core_lta.py:5-8`; `cache/lta/framework.npz` (`a_pseudo` 11.919, `box` 23.838, `o_pos` (384, 3), note "alpha-cage at body centre, 8-ring windows at face centres"); built by `scripts/build_lta_framework.py` |
| interactions | guest–framework O only ("Si is screened"); no charges; no guest–guest (one ethane per replica) | `core_lta.py:6-8, 9-11` |
| guest | TraPPE-UA ethane, 2 CH3 beads, harmonic bond 0.5 k (r − r0)², k = 400 kJ/mol/Å², r0 = 1.54 Å ("BD-integrable stand-in for TraPPE's fixed bond") | `core_lta.py:83-84, 9-11` |
| LJ | CH3–O ε/k_B = 93.0 K (ε = 93.0 × 0.008314462618 kJ/mol), σ = 3.48 Å, labelled "TraPPE-zeolite family (Dubbeldam et al.)" with no specific citation in the repo | `core_lta.py:12-13, 80-81`; `lta_prereg.json:7-13` |
| cutoff / shift | rc = 10 Å < L/2 (asserted); energy truncated and **shifted** (V − V(rc)); the force is truncated, not shifted (jump at rc); minimum image | `core_lta.py:149-154, 156-193` |
| dynamics | overdamped Langevin, unit mobility: q ← q + dt (F + F_bias) + sqrt(2 dt/β) η, float64, no force clip in the dynamics | `core_lta.py:14-17, 383, 521-522`; `LTA_VALIDATION.md` §1 table |
| units | Å, kJ/mol, K; time in "t.u." of the unit-mobility update (dimensionally Å²·mol/kJ — my inference from the update rule); "the campaign budget axis is force evaluations, not physical ps" | `core_lta.py:14-17` |
| dt | 2 × 10⁻⁴ t.u. at every T; noise sd per step sqrt(2 dt kT) = 0.0316 Å at 300 K; stiffest mode the bond, dt·λ = 0.16 (λ = 2k) | `core_lta.py:94`; `AUDIT_0A_lta.md:60-67`; `LTA_TIMESTEP_VALIDATION.md:28` |
| CV | φ = wrap(2π x_COM / a) ∈ [−π, π); window plane at φ = 0, α-cage centres at φ = ±π; z = φ a/2π; linear CV, so f_loc = −(a/2π)(F₀ₓ + F₁ₓ) with no Jacobian term; the bond force cancels exactly in f_loc | `core_lta.py:18-23, 196-212`; `LTA_TIMESTEP_VALIDATION.md:37-40` |
| regions (diagnostics only) | window |z| < 1.5 Å, cage |z| > 4.0 Å, neck otherwise; crossing = cage → window → cage per slot (inherited on clones, N1) | `core_lta.py:116-117, 231-239, 456-463` |
| initial conditions | each molecule at a random α-cage centre of the 8, COM jitter 0.5 Å Gaussian per axis, random orientation, bond = r0 | `core_lta.py:214-228` |
| temperatures | 300 K (v1), 80/150/225/300 K (sweep), +350 K (histogram study); ladders at 300 and 150 K | configs above |
| N, n_steps, T | N = 1024, 300 000 steps → T = 60 t.u.; saves every 3 000 steps (0.6 t.u.), 101 saves | `campaign.json:19-23` |
| bins | 180 circular bins, Δφ = 0.0349 rad = 0.066 Å, cell-centred, bin = floor((φ + π)/Δφ) mod 180 | `core_lta.py:245-249`; `campaign.json:87-92` |
| ABF estimator (histogram study, ladders) | own-bin P0: Γ_j = M_j / C_j on populated bins, 0 on empty bins; bias force = clip(Γ_j(φ), ±60) × ramp × ∇φ; F̂ = exact integral of the piecewise-constant force, circular mean of Γ subtracted, evaluated at cell centres, centred; deposits clamped to ±480 (= 8 × bias clip) | `core_lta.py:252-269, 423-446`; `LTA_TIMESTEP_VALIDATION.md:78-80` |
| kernel estimator (sweep, v1) | `abf_bandwidth` 0.05 rad wrapped Gaussian (unused by the histogram branch) | `campaign.json:25, 35` |
| warmup / burn-in | bias ramp linear 0 → 1 over 20 000 steps (4 t.u.); the bias uses all-steps accumulators; the reported estimator uses production accumulators from step 20 000 | `core_lta.py:426-438, 466-469` |
| FR target | uniform density 1/(2π) on the circle | `core_lta.py:272-282` |
| FR schedule | start step 20 000 (t = 4; v1 used 40 000 = t 8), then every 5 steps (0.001 t.u.), after the dynamics update, on post-update positions | `campaign.json:30-31`; `core_lta.py:524-546` |
| FR score | p̂ = normalised KDE (bin counts smoothed by a wrapped Gaussian, `kde_bandwidth` 0.10 rad); raw S_i = log p̂(φ_i) − log q(φ_i) − KL(p̂‖q); recentre, clip to ±2, recentre, three times | `alkanes/core.py:102-121` |
| FR law | death probability for S > 0: 1 − exp(−γ S dt_eff), dt_eff = dt × fr_every = 1e-3 t.u.; sources multinomial ∝ max(−S, 0) with replacement; cap int(0.02 N) = 20 per opportunity at N = 1024 (random subset if more fire); **cap = 0 for N < 50 silently disables FR** (ladder uses max(1, ⌊0.02N⌋)) | `alkanes/core.py:124-160`; `FINAL_RESULTS.md:216-217`; `SCIENTIFIC_PLAN.md:54-62` |
| FR rate | 0.20 (150–350 K), 0.10 (80 K); max death probability per opportunity at S = 2: 4e-4 | `campaign.json:37-82`; `AUDIT_0A_lta.md:60-67` |
| FR dose realised | ≈ 0.64 deaths per walker at N = 1024 (300 K), 0.77 (150 K); the "events" in `LTA_HISTOGRAM_REPLICATION.md` tables are totals over 16 seeds (10 492 at 300 K = 656 per seed) | `FINAL_RESULTS.md:466, 516`; arrays (Appendix A) |

Entropic decomposition of the published references: ΔF‡ 10.8 kT at 300 K, −TΔS‡ 72 % (300 K), 61 % (150 K), 74 % (350 K)
(`LTA_HISTOGRAM_REPLICATION.md` §1, §4.1); computed as −TΔS = ΔF − ΔU with U(φ) the umbrella-pooled conditional ⟨V | φ⟩
(`run_lta_reference.py:113-121`; `core_lta.py:707-722`). The rule "the label 'entropic barrier' only if −TΔS ≥ 50 % of ΔF"
is frozen in `lta_prereg.json:21`.

### 3.2 Validation gates (`configs/numerical_validation/lta_validation_prereg.json`; `docs/numerical_validation/LTA_VALIDATION.md`)

* Independent re-implementation `src/lta_validation.py`, equal to `core_lta` to 1e-13; analytic force = central differences
  to < 1e-6 (`G_LTA_force`; `LTA_VALIDATION.md` header).
* Arbiter: **exact Metropolis MC** (energy only): single-bead ±0.08 Å and rigid translation ±0.3 Å moves; 16 groups × 16
  molecules per (group, window) × 40 windows, 200 000 steps, sample every 5; every chain starts from 5 000 exact MC steps so no
  clamp is used (`lta_validation_prereg.json:6-13`; `LTA_VALIDATION.md` §2).
* Candidates: production EM at dt 2e-4 / 1e-4 / 5e-5 and Leimkuhler–Matthews at 2e-4; 48 t.u. per molecule.
* Three read-outs: F_density (WHAM on 1800 fine bins, summed to 180), F_mid180 (180-bin WHAM, bias at midpoints = the
  published convention), F_MF (pooled conditional mean force, `histogram_pmf` convention).
* **G_LTA_MC_self**: MC F_density vs MC F_MF, D ≤ 0.03 kJ/mol. PASS (0.002 at 300 K, 0.003 at 150 K).
* **G_LTA_dyn**: EM 2e-4 ADMISSIBLE iff D ≤ 0.03 kJ/mol with upper95 ≤ 0.045 on **both** routes; MARGINAL if both ≤ 0.06;
  else FAIL. 0.03 kJ/mol ≈ ⅓ of the 300 K histogram-ABF median final error (0.09) and 0.012 kT
  (`lta_validation_prereg.json:23`). Result ADMISSIBLE at both T: 300 K density 0.000 [0.021], MF 0.014 [0.029]; 150 K
  0.000 [0.017], 0.011 [0.021]. EM bond variance +8.65 % / +8.57 % (first order, halves with dt; LM +0.02 %), barrier shift
  +0.03–0.05 kJ/mol.
* **G_LTA_reference**: published reference vs MC, same bounds, noise treated as 0: 300 K **MARGINAL** (D 0.034, upper 0.038,
  max abs 0.066; attributed to +0.049 kJ/mol first-order barrier shift plus its own sampling noise); 150 K ADMISSIBLE
  (0.017 [0.019]).
* Conclusion check: rescoring the published histogram productions against MC with the published arithmetic — FR gain
  SURVIVES at 300 K (−13.66 % [−14.79, −11.11]; vs sham −13.02 %) and 150 K (−28.54 % [−29.41, −26.56]; final −54.38 %)
  (`LTA_VALIDATION.md` §3.1, §4).
* Not validated: 80, 225, 350 K (`LTA_VALIDATION.md:204-206`).

### 3.3 How the references were built

**Published umbrella/WHAM reference** (`scripts/run_lta_reference.py:36-52, 100-150`; `core_lta.py:627-704`;
`lta_prereg.json:18-22`; `lta_sweep_prereg.json:33-43`).
* 40 windows at linspace(−π, π, 40, endpoint=False); harmonic bias ½ κ wrap(φ − c)².
* κ: 300 kJ/mol/rad² at 300 K (v1 artefact, reused); the sweep rule κ = kT/(0.07 rad)² elsewhere: 136 (80 K), 254 (150 K),
  382 (225 K), 594 (350 K).
* 256 molecules per window; 150 000 steps at 300 K (scaled 150 000 × 300/T), burn-in 30 000 (scaled), sample every 25 steps;
  the **same EM integrator and dt** as production; forces clamped to |F| ≤ 500 during the first 5 000 steps
  (`core_lta.py:647-655`).
* Periodic histogram WHAM on 180 bins with the bias at bin midpoints, 20 000 iterations or 1e-10 convergence; F centred
  (`core_lta.py:674-704`). Acceptance: split-half barrier within 0.2 kT; unbiased cross-check (16 384 molecules) RMS
  reported (0.21 kJ/mol at 300 K; 0.154 at 350 K).
* Barrier scalar: window |z| < 0.4 Å minus "within 0.4 Å of the cage centre" (§2.3).

**Exact MC reference used by the ladders and Exp B** (`docs/equal_budget/LTA_TIMESTEP_VALIDATION.md` §3; `SCIENTIFIC_PLAN.md:70-75`).
* Free energy: MC fine-bin WHAM over the 16 groups × 40 windows (noise 0.006 kJ/mol at 300 K, 0.0035 at 150 K).
* Mean force: γ_j = ΣM_f / ΣC per 180-bin, pooled over windows (bias-independent because the umbrella depends on φ only);
  jackknife SE 0.0143 / 0.0101 kJ/mol/rad; circular mean consistent with 0.
* Independent seeds (7000 + group); never ran ABF or FR. sha256 recorded in every summary.
* Zero-noise scorer floor (the reference's own mean-force vs density-route consistency): 0.0075 / 0.0056 kJ/mol
  (`FINAL_RESULTS.md:195-197`).

---

## 4. Equal-budget methodology, as implemented

* **Budget.** B = N × n_steps base walker-steps per arm per seed, identical at every N; n_steps(N) = B/N exactly;
  T_N = B h / N. LTA: B = 1024 × 300 000 = 3.072 × 10⁸, h = 2e-4, T_N = 60 t.u. at N 1024 … 61 440 at N 1. Gateway:
  B = 2048 × 1.6 × 10⁶ = 3.2768 × 10⁹, h = 2.5e-5, T_N = 40 … 81 920 (`SCIENTIFIC_PLAN.md:26-36`; inventory
  `FINAL_RESULTS.md` §3). `n_force_evals = N (n_steps + 1)` exactly (`LTA_TIMESTEP_VALIDATION.md:89`).
* **What B counts.** Force evaluations only; FR's KDE, score and resampling are excluded (measured +1.1 % at LTA N 1024,
  +13.9 % at N 2; gateway +0.3 % / +8.7 %), which can only lower FR at small N (`FINAL_RESULTS.md` §7, §9).
* **Ladder.** LTA N ∈ {1024, 512, …, 2, 1}, 16 seeds (300 K: 30000–30015; 150 K: 15000–15015); gateway N 2048 … 1, 32 seeds
  (8100–8131). ABF and FR at every N ≥ 2, ABF alone at N = 1. One process per (system, T, N, seed, method); shared initial
  conditions and noise slot by slot; independent FR stream (`SCIENTIFIC_PLAN.md:26-36, 164-171`).
* **Physical-time constants** (identical across N): warm-up 4 t.u., burn-in 4 t.u., FR start 4 t.u., FR every 0.001 t.u.,
  windowed genealogy 4 t.u. (`LTA_TIMESTEP_VALIDATION.md` §5). LTA cap max(1, ⌊0.02N⌋) at N < 50, flagged.
* **Save grid.** 200 uniform u ∈ (0, 1] + 24 log-spaced u ∈ [1e-4, 5e-3) + physical-time checkpoints; profile snapshots at
  u ∈ {0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 1.00}; raw accumulators (M_all, C_all, M_prod, C_prod), instantaneous
  histograms, region fractions, events, genealogy, cost at every save (`SCIENTIFIC_PLAN.md` §3).
* **Metrics.** Ī_F, Ī_F′ on the 200 uniform u (§2.1b); final e_F; persistent τ(ε) to the last save, censored → ∞
  (`eqb_metrics.py:517-533`).
* **Thresholds (frozen, same for every N)** (`SCIENTIFIC_PLAN.md:120-131`):

  | system | e_F strict / **mid** / loose | e_F′ strict / mid / loose | marginal |
  |---|---|---|---|
  | gateway | 0.002 / **0.0055** / 0.02 | 0.012 / 0.035 / 0.10 | TV_half ≤ 0.10 |
  | LTA 300 K | 0.03 / **0.094** / 0.5 kJ/mol | 0.04 / 0.12 / 1.0 kJ/mol/rad | TV_half ≤ 0.10 |
  | LTA 150 K | 0.03 / **0.13** / 0.5 | 0.04 / 0.15 / 1.0 | TV_half ≤ 0.10 |

  mid = the previously accepted high-N ABF endpoint accuracy at the anchor (LTA vs the MC reference); strict ≈ ⅓ of the anchor,
  above the noise floors; loose = early establishment.
* **Per-N contrast.** G_X(N) = (X_FR − X_ABF)/X_ABF per seed, median, 10 000-resample seed bootstrap, wins; absolute values
  alongside (`eqb_metrics.py:904-916`).
* **Best allocation** (`SCIENTIFIC_PLAN.md:146-148`; `eqb_metrics.py:951-1044`). Point estimate: min over N of the ABF seed
  median vs min over N ≥ 2 of the FR seed median. Bootstrap: for every N, seeds resampled with replacement (one index draw
  per N shared by both methods, independent across N); the minimising N is **re-selected in every resample**; ties share
  the credit (fix F5: earlier code gave N = 1 a spurious 0.54 share); reports the frequency of each best N, CIs of both best
  values, of best FR − best ABF and its relative form, and P(best FR better) = fraction of resamples. Outcomes A–E
  (`SCIENTIFIC_PLAN.md` §5).
* **Fresh-seed confirmation** at N\* selected for both arms (LTA 300 K N\* 16, 150 K N\* 2, gateway 16), 32 / 32 / 64 fresh
  seeds, same statistic; tie → weak A with the CI as bound; CI > 0 → strict A (`CONFIRM_BEST_ALLOCATION.md`). Result: LTA
  ties (FR advantage ≤ 1.2 % / ≤ 8.7 %), gateway ABF better.
* **Exp B ρ_best** = min_f median W_f / min_a median W_a, both sides re-selected in every resample; percentile CI and the
  fraction of resamples with ρ_best < 1 ("not a probability"); decision by IUT/Bonferroni (§2.9); resolvable only outside
  ≈ [0.82, 1.22] (LTA, 32 seeds) and ≈ [0.77, 1.3] (gateway, 64 seeds) (`PREREGISTRATION.md:442-454`). LTA 150 K: B =
  3.072 × 10⁸ per run, mid 0.13 kJ/mol, best FR N 64 vs best ABF N 32 (`EXPB_WALLCLOCK.md` §1).
* **Mechanism tests on the ladders** (`SCIENTIFIC_INTERPRETATION.md` §2): W (within-N seed-level Spearman of G(Ī_F) with the
  ABF arm's τ(TV_half), stratified permutation), T (temperature at equal N), M (marginal vs F). σ_KDE(N) =
  sqrt(L / (2√π N η)) is rank-collinear with N and with T_N, so cross-N correlations cannot separate mechanisms.

---

## 5. Design constraints and lessons for a new molecular system (ERI)

Each item says what went wrong before and where it is recorded.

**Numerics and reference**
* **L1. Validate the integrator against an energy-only exact arbiter before any ABF/FR run.** A stable, healthy-looking
  trajectory proved nothing on WCA (dt 0.002 was 0.3 kT wrong; `WCA_LANGEVIN_VALIDATION.md` §4 item a). Check force = −∇U
  by finite differences, forbid non-gradient clips in the dynamics (`WCA_FORCE_AUDIT.md` §1, §4), compute dt·λ_max for the
  stiffest mode against the EM limit 2 (LTA bond 0.16; WCA contacts 2.05), and gate on D at a tolerance tied to the planned
  effect size (LTA: 0.03 kJ/mol = ⅓ of the ABF final error).
* **L2. The reference must not share the production integrator's bias.** Every historical WCA TI reference used the clipped
  integrator and was 0.29 kT off (`EVIDENCE_LEDGER.md` §1); the LTA umbrella reference shares EM and is MARGINAL at 300 K.
  Re-scoring against exact Gibbs flipped the WCA sign (W3, W4). Use MC (energy only) for F and a per-bin conditional
  mean-force reference; report its jackknife noise; never score against the published umbrella reference when an MC one
  exists (`SCIENTIFIC_PLAN.md:75`).
* **L3. Fix one barrier convention** (two values, 26.40 vs 26.79 kJ/mol, for one quantity; `LTA_TIMESTEP_VALIDATION.md:55-60`).
* **L4. Time units.** LTA time is unit-mobility overdamped time, not ps. Any temperature axis also rescales diffusion, so it
  is a budget axis as much as a landscape axis (`RESULTS.md` Stage 5 caveat 4).
* **L5. CV geometry.** The LTA CV is linear (no Jacobian term) and periodic by a lattice translation; ZIF-8 needed a
  lattice-translation period, unwrapped coordinates and a radial confinement tube identical in all arms and the reference
  (`zif8_prereg.json:20-21`). For a new framework, check that the CV level set at the barrier is the window only.

**Estimator and read-out**
* **L6. Save raw accumulators** (M, C, all-steps and production, per save). Bandwidth and read-out can then be revisited
  offline; the convolution is not invertible (`LTA_READOUT_SWEEP.md` §1; project memory "information-clock-and-bandwidth").
* **L7. Read-out vs reference smoothing.** Compute the deterministic smoothing share before trusting an arm contrast
  (< 0.35 fine, > 0.8 fix the estimator first; ZIF-8 0.88, LTA 0.18, WCA 0.16) (`BANDWIDTH_DEFECT_SCREEN.md` §1, §6). The
  kernel ABF baseline was bandwidth-limited 25× on EB (frontier stall) and its ZIF-8 harm was a baseline artefact. The own-bin
  histogram with exact integration removes the read-out/dynamics mismatch; compute its deterministic P0 floor before
  setting any threshold (gateway strict e_F′ was unreachable).
* **L8. Bin-width rule.** The histogram campaign's clause (a) assumed a Monte Carlo residue the histogram does not have and
  never passed; the fallback decided every case (`HISTOGRAM_ABF_REPLICATION.md` §6 "Failure modes"). LTA: 180 bins, floor
  share 0.018–0.095, 360 bins worse by variance (§4.8).
* **L9. Estimator restart at burn-in** produces a visible e_F jump (ZIF-8 9.6 kJ/mol spike; LTA 0.39 → 0.85, N5). Either
  report the bias estimator or make the reported one continuous, and keep it out of the FR-start save.

**Pairing and determinism**
* **L10. Pair by construction and assert it.** Same initial conditions and Langevin stream for both arms, FR on its own
  stream, bit-identity (or ≤ 1e-12) before the first FR event asserted from the production files (`zif8_prereg.json:176`;
  `FINAL_RESULTS.md:76-80`). On GPU, set deterministic algorithms and replace float atomics in any accumulator that feeds the
  bias; otherwise arms silently unpair (WCA). Run all arms of a seed in one process; never compare absolute levels across
  processes (project memory "wca-measurement-traps", "zif8-determinism-and-pairing").

**Endpoints and statistics**
* **L11. Integrated endpoint window.** State the integration window, whether t = 0 is included, and report a post-FR
  secondary with the pre-FR share of I_F (ZIF-8 A1; LTA ≈ 40 %, N4).
* **L12. Persistence windows must fit inside the data or the value is censored** (ZIF-8 A1; LTA still truncates, N3). Do not
  define speed-ups against ABF's own final error without a censoring rule.
* **L13. Thresholds tied to the endpoint judged**, not to the initial error (LTA e₀/8 lesson); for an N or K ladder, fix
  persistence spans and time ratios in absolute time (§2.5 item 3).
* **L14. Use the corrected verdict tree** (TOST NEUTRAL, explicit HARMFUL, INCONCLUSIVE, health gating every label) and
  state the CI level of the TOST (§2.7). Equivalence by TOST, never by "no CI excluded zero".
* **L15. Null-control every gate and band at the planned n before freezing it**, and plant the defect a gate is meant to
  detect in real traces (V-B4; WCA run-long ESS; Gate C 0/8 at 50 %; `accept_gate_js` 0.02). State the detectable effect
  size; Exp B states its resolvable ρ band.
* **L16. Multiplicity.** Declare the family (e.g. the K ladder) and the correction (Holm as in Exp A) before data.

**FR dose, genealogy, attribution**
* **L17. Without a sham, attribution needs a substitute.** The sham was the instrument that attributed the LTA gains to the
  FR direction and showed that cloning per se does not cause the WCA tilt (§2.10). With the two-arm instruction
  (STATUS.md:31-34) the remaining evidence is the direct contrast plus mechanism read-outs; those read-outs must then be
  free of clone bookkeeping (L18).
* **L18. Count transits per walker identity; never copy cumulative per-walker counters on clones** (ZIF-8 A1; LTA N1).
  Record (dying slot, source slot) and persistent ids per event, as the LTA movie recorder does (`core_lta.py:45-50,
  562-577`), so traffic can be recounted.
* **L19. Calibrate the FR rate at the production horizon and production N**, safety-only (N6; `zif8_prereg.json:115`); use a
  windowed ESS plus max lineage share, not a run-long ESS floor (N7).
* **L20. Finite-N limits.** The historical cap int(0.02N) is 0 for N < 50 and silently disables FR; σ_KDE ≥ 1 at N ≤ 16
  makes the score noise; FR at N = 2 is nearly inert (median max |S| 0.005) yet realises 17–18.5 deaths per walker
  (`FINAL_RESULTS.md` §2b item 4, §9; `LTA_TIMESTEP_VALIDATION.md:102-105`). FR damaged the gateway marginal at N ≤ 8.
* **L21. FR must start before establishment is over.** The v1 LTA null was the late FR start (t 8 vs ABF e₀/8 at t 2.4)
  (`RESULTS.md` Stage 4–5); ZIF-8 set fr_start = warm-up end for this reason (`zif8_prereg.json:52`).

**Regime and claims**
* **L22. Classify before any FR run, from ABF-only data, with the four-way map and Gate 0 first** (§2.13), and add a
  per-bin-adequacy check: a cell can be marginal-establishment-limited with nothing left for FR to win (ZIF-8). A
  marginal gain does not imply an F gain (C3).
* **L23. Conditional relaxation.** FR cannot speed p(y | ξ) relaxation (identity) and slow orthogonal dynamics makes it
  harmful (C4); measure τ_⊥ (Gate D twin test) for the ERI window, where orientation and transverse position are
  constrained (in LTA ⟨|cos θ|⟩ 0.955–0.97 and transverse rms 0.18–0.22 Å in the window, `LTA_HISTOGRAM_REPLICATION.md`
  §4.0, §4.3).
* **L24. Keep the four claim types separate** and carry the equal-budget question with any fixed-N positive: no system has
  shown an equal-force-evaluation advantage, and the shortened-T ladders did not change that.
* **L25. Analysis-code failure classes to audit before the long run** (`V2_PREREGISTRATION.md` §12.9; project memories
  "analysis-code-failure-classes", "analysis-defect-classes", "adversarial-audit-before-the-long-run"): no data must not take
  the PASS branch (`isfinite(inf)`, nan in a `<`); no data must not read as a smaller number; partitions tested on sampled
  values, not only the grid; two writers to one resource; a declared bias is an assumption about the initial condition; a
  check that cannot entail its claim; a spec error faithfully implemented (two implementations agreeing on one ambiguous
  sentence is not evidence); scope widened in restatement; a deterministic constant in the reference making the endpoint
  arm-insensitive (feed the reference through the metric with no arm); ratios with incommensurable denominators; unequal
  tuning budgets; scope-blind diagnostics; testing the engine instead of the production pipeline; cumulative counters copied
  by cloning; persistence windows truncating at the tail. Require the auditor to demonstrate each defect with an
  adversarial input.
* **L26. Process.** Docs are git-ignored and need `git add -f` (STATUS.md:23-26); historical documents are never edited,
  errata are appended (`EVIDENCE_LEDGER.md` header); job state is a measurement, not a memory (project memory
  "reported-vs-performed-actions").

---

## 6. Open issues and things I could not determine

1. The ERI campaign prompt is not in the repository, so I cannot map §2.5's defect list onto the prompt's wording.
2. Whether the LTA 80 K "FR doubles the crossings" claim survives clean counting: there is no genealogy record at 80 K (N1).
   The 150 K and 300 K recounts are single seeds at 100-step snapshot resolution.
3. LTA 80, 225 and 350 K were never re-validated against exact MC (`LTA_VALIDATION.md:204-206`).
4. The LTA τ speed-ups and screening-rule T_est were not recomputed with a non-truncating persistence window (N3); I only
   identified the code path.
5. Post-FR ΔI_F for LTA was not computed as an endpoint; the "≈ −24 %" in N4 is arithmetic on the pre-FR share.
6. The LJ parameters' literature provenance: the repo labels them "TraPPE-zeolite family (Dubbeldam et al.)" without a
   specific reference (`lta_prereg.json:12`; `core_lta.py:12-13`). That belongs to the literature cross-check.
7. Starvation as the predictor: supported in LTA across T and within N at large N (ρ −0.35 / −0.29), contrary at seed level
   in the gateway (ρ +0.21) (`FINAL_RESULTS.md` §8). Unresolved.
8. No valid head-to-head of the FR rule against published selection rules exists on any system where FR helps; the
   Lelièvre–Rousset–Stoltz 2007 WCA positive is unreconciled (`PUBLICATION_READINESS.md` §5 items 1–2).
9. WCA FR bias as dt → 0 and N → ∞: OPEN (W10).

---

## Appendix A. Re-analyses run for this review (CPU, seconds; no file written outside the scratchpad)

All inputs are repository arrays on this host (git-ignored): `results/lta_histogram/production_T{150,300}/{abf,fr_uniform,fr_sham}.npz`,
`results/uniform_campaign/lta/reference/reference_T{150,300}.npz`, `results/lta_movie/T{150,300}/movie_data.npz`. Run
with `python3 -I`.

**A1. Pre-FR share and pairing (N4, N5).** e_F recomputed with the analyzer's arithmetic (circular `np.interp` of the
reference to the engine grid, mean-removed RMS over 180 bins). Results:

| T | median share of ABF I_F from t ≤ 4 | max \|PMF_ABF − PMF_FR\| over saves t < 4 | median ABF e_F at t = 3.6 → 4.2 (kJ/mol) |
|---|---|---|---|
| 300 K | 0.41 | 2.7e-13 | 0.387 → 0.849 |
| 150 K | 0.40 | 7.1e-15 | 0.487 → 0.687 |

The 300 K median e_F(T) recomputed this way is 0.0866 kJ/mol, matching the scoreboard (0.0866).

**A2. Crossing counts (N1).** Production totals of `n_cage_crossings` (inherited) and of `n_transitions` (every region
change of every slot per step; dominated by neck/boundary flicker, not a transit count) over 16 seeds:

| T | ABF crossings | FR crossings (×ABF) | sham (×ABF) | FR transitions ×ABF | FR events |
|---|---|---|---|---|---|
| 80 K | 3 705 | 7 087 (1.913) | 3 697 (0.998) | 1.074 | 10 605 |
| 150 K | 15 088 | 17 871 (1.184) | 15 153 (1.004) | 1.018 | 12 562 |
| 225 K | 29 010 | 30 516 (1.052) | 28 921 (0.997) | 1.004 | 11 040 |
| 300 K | 43 636 | 44 954 (1.030) | 43 699 (1.001) | 1.001 | 10 492 |
| 350 K | 52 933 | 53 903 (1.018) | 53 354 (1.008) | 1.001 | 10 291 |

Movie-seed recount from snapshots every 100 steps, applying each recorded event (dying slot, source slot) before the
snapshot it precedes; "inherited" copies the source's `has_left_cage` flag and count as the engine does; "per identity"
resets the flag for a clone so that only transits whose cage exit and cage re-entry happen under one walker identity count:

| seed | arm | engine total | snapshot, inherited | snapshot, per identity |
|---|---|---|---|---|
| 150 K | ABF | 919 | 852 | 852 |
| 150 K | FR (819 events) | 1 037 | 983 | 923 |
| 300 K | ABF | 2 721 | 2 475 | 2 475 |
| 300 K | FR (665 events) | 2 762 | 2 517 | 2 444 |

Script (scratchpad copy at `/tmp/claude-1008/-home-zheyuanlai-ABF-Fisher-Rao/d4317313-dc1c-4e55-8b95-fc34dea3264c/scratchpad/crossings_check.py`):

```python
import sys, math, numpy as np
z = np.load(sys.argv[1], allow_pickle=False); a = float(z['a_pseudo'])
for arm in ('abf', 'fr_uniform'):
    phi = z[f'{arm}/snap_phi'].astype(float); S, N = phi.shape
    zz = np.abs(phi) * a / (2 * math.pi)
    reg = np.ones_like(zz, dtype=np.int8); reg[zz < 1.5] = 2; reg[zz > 4.0] = 0
    slots = z[f'{arm}/snap_ev_slots']; die = z[f'{arm}/snap_ev_die']
    ev_snap = die[:, 0].astype(int) if die.size else np.zeros(0, int)
    left_inh = np.zeros(N, bool); cnt_inh = np.zeros(N, np.int64); left_cl = np.zeros(N, bool); clean = 0
    for s in range(S):
        if s > 0:
            for k in np.nonzero(ev_snap == s)[0]:
                d, src = int(slots[k, 0]), int(slots[k, 1])
                left_inh[d] = left_inh[src]; cnt_inh[d] = cnt_inh[src]; left_cl[d] = False
        in_cage = reg[s] == 0
        if s > 0:
            cnt_inh += (left_inh & in_cage); clean += int((left_cl & in_cage).sum())
        left_inh = np.where(in_cage, False, left_inh | (reg[s] == 2))
        left_cl = np.where(in_cage, False, left_cl | (reg[s] == 2))
    print(arm, int(np.asarray(z[f'{arm}/n_cage_crossings']).sum()), int(cnt_inh.sum()), clean)
```
