# Publication readiness: ABF + marginal Fisher–Rao birth–death

**2026-10-11.** Companion to `FINAL_SYNTHESIS.md` (answers Q1–Q7 and the five decision items), `PRIOR_ART_NOVELTY.md`
(verified literature), `EVIDENCE_LEDGER.md` (claim statuses, with errata), `docs/mechanism/EXPA_SHIFTED_FIBRE_MOBILITY_v3.md`,
`EXPB_PARALLEL_BACKEND.md` and `EXPB_WALLCLOCK.md`.

## 1. Verdict

**Not ready to submit. Publishable after three targeted additions, under a different framing than "FR accelerates ABF".**

* A methods paper claiming that marginal FR birth–death **accelerates** ABF is not supported, for two independent
  reasons:
  * *Evidence.* No advantage at equal force evaluations in any system (three systems, fresh-seed confirmation). No
    resolved wall-clock advantage on parallel hardware (LTA confirmatory 0.72 [0.54, 1.20]; gateway exploratory 0.76
    [0.52, 1.11]). Harmful on the one validated many-body solvated system (WCA). Neutral or harmful on alanine.
  * *Novelty.* Parallel ABF with birth–death selection on the replicas' CV marginal is published (Lelièvre, Rousset &
    Stoltz, J. Chem. Phys. 2007, on a WCA dimer), and so is multiple-walker ABF with selection (Minoukadeh, Chipot &
    Lelièvre 2010; Comer et al. 2014, in NAMD; Amber ABMD). The KDE log-ratio rate toward an explicit uniform target is
    an incremental change of rule, and no head-to-head against the published rules exists yet on any system where FR
    helps.
* A **cost-controlled analysis of when and why marginal birth–death helps or hurts adaptive biasing** is publishable.
  Its contributions would be:
  * the two necessary conditions (establishment-limited ABF; fast orthogonal dynamics at N ≤ 512);
  * the conditional-law statement and its measured corollary;
  * validated negatives that qualify a published positive (WCA);
  * the first equal-force-evaluation N-ladders with best allocation per arm;
  * the first best-vs-best wall-clock test on fixed hardware.

  These are well above the evidence standard of the prior selection literature (exact or dt-validated references,
  paired seeds, preregistration, shams, CIs).

## 2. What a paper can claim now (validated numerics, seed-bootstrap CIs)

| claim | type | evidence | strength |
|---|---|---|---|
| At fixed N, FR lowers the finite-time F error of many short replicas when ABF is establishment-limited and orthogonal dynamics are fast | 1 | gateway family (−24 to −41 %), LTA 150 K (−29 to −34 %; −28.5 % vs exact Gibbs, sham neutral), shifted fibre λ ≥ 16 (−23 to −34 %) | strong in toys; one atomistic positive |
| FR acts on establishment, not discovery | 1 | first arrival −17 % to +9 %; trailing marginal −63 to −92 % | strong |
| A better CV marginal does not imply a better free energy | 1 | every cell of Exp I/II/A; equal-budget outcome C | strong |
| Slow orthogonal dynamics makes FR harmful (N ≤ 512); the harm comes with a population-wide transverse force bias and an error floor | 1 | gateway λ 0.1; shifted fibre λ 1 / 4 (Exp A); clone-age deposit bias | strong effect; mechanism a hypothesis; transport confound in Exp A |
| Raising transverse mobility rescues FR on the curved fibre | 1 | Exp A outcome A at both N (R 0.65 / 0.37) | strong; attribution to mobility, not relaxation alone |
| No advantage over the best ABF allocation at equal force evaluations | 2 | equal-budget ladders + confirmation; Exp B force counts | strong |
| No resolved wall-clock advantage of best FR over best ABF on 8 dedicated cores | 3 | Exp B (LTA confirmatory, gateway exploratory) | a null; underpowered for ≈ 25 % effects |
| At fixed N the gain carries over to wall-clock on 1–16 cores | 3 | C6; Exp B N 512 pairs 60/64, 31/32 | strong |
| FR is harmful on WCA at a validated timestep, with a stationary 0.017 kT tilt | 1, 4 | dt\* vs exact Metropolis MC; sham neutral | strong |

## 3. What a paper must not claim

* "FR accelerates ABF" or "reduces the cost of free-energy estimation".
* "First to combine birth–death with ABF", "first CV-subspace birth–death", or "novel uniform-target selection".
* Any dt-0.002 WCA gain.
* Valine as a measured FR control (no FR arm was run).
* Alanine as uniformly neutral: FR was harmful in two cells, and the friction ladder had no FR arm.
* The gateway Exp B decision as confirmatory (frozen V-B4 FAIL; exploratory under Amendment 3).

## 4. Evaluation standards reviewers will apply (from `PRIOR_ART_NOVELTY.md` §3, checked against this project)

| standard | status |
|---|---|
| Equal aggregate cost against the cheapest alternative, including fewer, longer replicas | **met as a test**; the answer is "no advantage" |
| Each method at its best settings, on benchmarks with accurate answers | met for gateway, LTA, WCA (exact or dt-validated references) |
| Head-to-head against the published selection rules (Laplacian, count-based) | **missing**. The only head-to-head ran at the refuted WCA dt 0.002 |
| Reconcile with the published WCA positive (Lelièvre–Rousset–Stoltz 2007) | **missing** |
| Uncertainty quantification (independent runs as the unit, CIs, full procedure) | met (paired seeds, bootstrap, TOST, shams, preregistration) |
| Wall-clock credit for parallel methods (time-to-solution, overheads, scaling) | reported; unresolved; one hardware configuration; no strong- or weak-scaling curves beyond the benchmark |
| Separate exploration speed from converged error | partly met: FR gains are finite-time and early, never at the converged endpoint at equal budget |
| State the KDE rate form, edge handling and bandwidth vs N | **to be written** (Pampel et al. 2023 show the plain KDE rate under-samples barriers) |
| Why a uniform target rather than variance-optimal allocation | must be discussed (Aristoff & Zuckerman 2020; the project found that ABF needs adequate, not equal, counts) |
| Scope of the conditional-law statement (mean field vs finite N, genealogy) | must be stated precisely; it will be read as a lemma |

## 5. Work needed before submission (priority order; CPU-only estimates at measured rates)

1. **Head-to-head of selection rules.** Run FR (KDE log-ratio), the Lelièvre–Rousset–Stoltz Laplacian rule
   (S = c ∂_zz ψ/ψ) and a Comer-style count rule, plus a matched-turnover sham, at validated settings. Systems: gateway
   N 512 / 2048 and LTA 150 K N 256 / 512 (and the best-allocation N), 32 paired seeds each. About 30–50 core-h with
   the serial engines. This decides whether the rule matters, as Comer et al. 2015 found it did not.
2. **Reconcile Lelièvre–Rousset–Stoltz 2007 on WCA.** Reproduce their 2D overdamped WCA dimer with M = 2000 and their
   rule and metric (max-norm PMF error averaged over realisations, t ≤ 0.4), then score the same runs against an exact
   reference with the per-run integrated error used here. This explains whether their positive and our validated
   negative differ in rule, metric, horizon, reference or model. About 20–40 core-h plus an MC reference.
3. **Theory and methods text.** The conditional-law statement with its exact scope. The KDE rate form, edge treatment
   and bandwidth. The uniform-target choice against variance-optimal allocation.
4. *(Optional)* **A powered wall-clock study**, ≥ 128 seeds per arm for the two best arms per system, if a wall-time
   claim is wanted. About 15–25 core-h per system on 8-core allocations.
5. **Preprint promptly.** A funded group (TRR 146 project A10, Dünweg, Valsson, Hartung) has announced birth–death
   population control of CV-based multiple-walker sampling.

## 6. Reproducibility status

* **Design.** Preregistration frozen before any data (`c5a2264f`), with four dated amendments. Amendments 1–2 were
  written before any production run. Amendment 3 (after the gateway V-B4 FAIL) and Amendment 4 (resources) were written
  before any Experiment B production. All post-hoc decisions are disclosed.
* **Record.** Timestamped execution log; per-run records with sha256 of the science arrays; compute ledger (149 of the
  250 used core-h ceiling; 0 GPU-h); data manifest with sha256 of every artifact
  (`results/final_campaign/DATA_MANIFEST.json`).
* **Code and tests.** Validated engines unchanged (sha256 checked). New parallel backend bitwise-verified against them;
  test suites `tests/test_final_*.py`, `tests/test_parallel_*.py` (`results/final_campaign/final_tests.log`).
* **Gap: raw arrays.** The arrays (`*.npz`) are gitignored and live only on this host. A publication needs them
  deposited (e.g. Zenodo) with the manifest.
* **Audits.** Two adversarial audit rounds, both completed before any production data was scored, found and fixed
  eight analysis-code defects, plus report errors. Corrections are versioned (Exp A
  report v1 → v3) or appended as errata, never silently edited.

## 7. Risks a referee will raise

* Most positives are on constructed toys. There is one atomistic positive (ethane/LTA at 150 K), and WCA and alanine
  are negative.
* The Experiment A rescue confounds transport with relaxation, and the dependence on λ is not monotone (λ 4 is worse
  than λ 1).
* The Experiment B gateway arm is exploratory. Its preregistered equivalence gate failed. A post-hoc diagnostic showed
  the gate could not pass even for the validated engine against itself, and the law-insensitive hybrid projection gives
  the same answer.
* Experiment B was underpowered for the observed ≈ 25 % point effects.
* "Helps only at large N and short per-replica time" will be read as a transient effect, which it is. The paper should
  present it as a regime map with mechanisms, not as an accelerator.

## 8. Recommended framing and venue

* **Working title:** *When does marginal birth–death selection help adaptive biasing force? A cost-controlled study*.
  Alternative: *Walker selection in multiple-walker ABF revisited: necessary conditions, cost, and validated negatives*.
* **Venue:**
  * **J. Chem. Phys.**, the venue of Lelièvre–Rousset–Stoltz 2007. It takes mechanism-level analysis on model systems.
  * **J. Chem. Theory Comput.**, whose readers will demand item 1 above.
  * **Phys. Rev. E**, for a physics-of-sampling framing that engages the TRR 146 work.
  * SIAM journals only if the conditional-law result is proved rigorously.
