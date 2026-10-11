# Prior art and novelty (final campaign)

*2026-10-11. Produced by a literature workflow (wf_6debe950-87f): three independent web/arXiv sweeps (76 works), an independent citation-verification pass over the 38 non-background works (38/38 exist; overlap claims checked against abstracts or full text where accessible), and a synthesis restricted to verified works. Items marked [record] come from the project's own records and were cross-checked against `docs/final_campaign/EVIDENCE_LEDGER.md` and the memory index. Paywalled details are marked UNVERIFIED. Verification record: `results/final_campaign/prior_art_verified.json`.*


**Sources used.** This section uses only works that the independent citation check marked `exists=true`.
- Where the checks disagreed, I say so. Where a check found an overlap claim unsupported, I describe only what the check says the work does.
- The sweeps also found other works that were never citation-checked, so they are left out here. These include ABF long-time convergence theory, weighted-ensemble ergodic theory, most of the WFR/FR theory papers, flatPERM and go-with-the-winners, Voter's parallel replica, Rosta–Hummer, and Wang–Machta–Katzgraber. They need a check before they are cited.
- Project facts beyond the brief are marked **[record]**. They come from the project's memory index and commit log, and should be confirmed against the repository before going into a manuscript.

---

## 1. Closest prior work, and what is and is not new

### 1.1 Closest work: Lelièvre, Rousset & Stoltz, J. Chem. Phys. 126, 134111 (2007)

This paper already combines parallel ABF with birth–death on the replicas' marginal along the collective variable (CV). Two checks read the full text and confirmed the following.

- **Method.** M interacting replicas run ABF (or ABP). The mean force comes from the replicas' empirical distribution, on histogram bins of width dz = 0.05 with time regularisation.
- **Selection rule.** Each replica trajectory is weighted by exp(∫S ds), where S = c ∂_zz ψ/ψ and ψ is the replicas' instantaneous CV marginal. Births and deaths use SMC-style clocks. A killed replica is replaced by a copy of a random replica, so its orthogonal coordinates travel with the clone.
- **Implied target.** Selection vanishes when ψ is uniform, so the implied target is uniform.
  - The paper states without proof that in the overdamped mean field ∂_t ψ = (1/β + c) ∂_zz ψ. The rule therefore acts as extra diffusion along the CV.
  - S is positive where ψ is convex, so the rule responds to curvature, not to distance from uniform.
- **Stated limitation (asserted, not proved).** "The convergence rate is still limited by the relaxation in each submanifold ξ(q) = z."
- **Test system.** 2D WCA solvent, N = 16, density 0.25, bistable solute dimer (h = 20, w = 0.7, β = 5), overdamped, dt = 1e-4, M = 2000, K = 100 realisations.
  - The reference is not exact: an M = 5000 run averaged over t ∈ [5, 10].
  - This is the same model family as the project's WCA dimer. Whether the parameters match exactly was not checked.
- **Results.** The error metric is the max-norm PMF error averaged over realisations, which measures bias.
  - At t = 0.05, selection with c = 2 or 5 is worse than no selection (20.4 and 22.9 vs 9.51).
  - At t = 0.1–0.2 the gains are large (for example 0.035 vs 18.0).
  - At t = 0.4 selection is still better (0.020–0.034 vs 0.066).
  - Table III runs only c = 10 at M = 1000/2000/10000, with no c = 0 control at each M. So "selection works better with more replicas" is claimed but not isolated.
- **Cost accounting.** All comparisons are at fixed M and fixed physical time. There is no comparison at equal force evaluations against fewer, longer replicas, and no wall-clock comparison.

### 1.2 Other direct prior art

- **Minoukadeh, Chipot & Lelièvre, JCTC 6, 1008 (2010).**
  - Multiple-walker ABF on deca-alanine. "Far-reaching walkers are cloned, replacing less effective ones."
  - The exact rule is unverified (paywalled). One check reads it as a reward for distance travelled. The Amber 2022 manual credits its fitness function to this paper and implements the 2007 Laplacian weight. Treat the formula as unknown.
- **Comer, Phillips, Schulten & Chipot, JCTC 10, 5276 (2014).** The three checks had different access: one could not get the full text (marked not supported), one read the abstract, and one read the author PDF. The full-text check found:
  - **Rule.** Count-based weight 1/(n_t(bin) + 1) from cumulative bin counts, systematic resampling, triggered when (max − min)/max ≥ 0.4.
  - **Setup.** Deca-alanine in vacuum, 10 walkers.
  - **Cost.** Comparisons at "similar computational cost" (12 ns single walker vs 10 × 1.2 ns) and in real time. The real-time gain is multiple walkers vs one walker, not selection vs no selection.
  - **Findings.** Selection improves short-time convergence but has "adverse effects over longer simulation times". Multiple-walker ABF without selection is "the best performer in the long-time regime", and the authors recommend selection only early.
  - **Failure mode, stated explicitly.** The rules "do not take into account sampling along orthogonal degrees of freedom".
- **Comer, Gumbart, Hénin, Lelièvre, Pohorille & Chipot, J. Phys. Chem. B 119, 1129 (2015).**
  - Selection "eliminate[s] replicas ... already well sampled, while duplicating replicas in relatively unexplored regions, enforcing more uniform sampling". It "automatically vanishes at equilibrium".
  - Weight is the inverse of samples gathered in ±h bins since the last selection. Selection switches off at a 20% spread.
  - The review calls its weight simpler than the 2007 and 2010 rules "while obtaining similar results".
  - Deca-alanine at equal total simulated time (8 ns): single < independent < shared < shared + selection. This is qualitative, with no seeds or confidence intervals in the text read.
  - At 128 ns every strategy converges to the same profile.
- **Amber 16 Reference Manual, nfe chapter (same text in Amber 2022, Sec. 25.4).** A production multiple-walker selection scheme for ABMD, a metadynamics-like adaptive bias.
  - Weight w_n ∝ exp(∫ C ∇²ρ/ρ dt), where ρ is taken from the accumulated bias (roughly the CV histogram). Resampling keeps N fixed, and coordinates and velocities are copied. Selection stops on a weight-entropy criterion.
  - The motivation is walkers "bunched up" behind hidden orthogonal barriers. Selection is "most beneficial during the initial and middle parts"; late in a run, "proceed with the non interacting multiple walker algorithm".
  - The manual has no benchmarks. It is the Amber 16 manual, not Amber 12.
- **Chopin & Jacob, Bayesian Statistics 9 (2011).** The statistics-side precedent for adaptive biasing combined with resampling toward a uniform CV marginal.
  - SMC whose bias is estimated adaptively from the current particles by ABP or ABF. The targets have an approximately uniform ξ-marginal and the same conditional law given ξ.
  - Systematic resampling when ESS < 0.8N, and a final importance-sampling step that removes the bias.
  - It differs from the project in several ways: particles are weighted (unbiased), "time" is data tempering, and the tests are mixture posteriors with no MD.
- **Pampel, Holbach, Hartung & Valsson, Phys. Rev. E 107, 024141 (2023).**
  - Lu–Lu–Nolen birth–death in a known low-dimensional landscape. The target is π, not a uniform CV marginal, and there is no ABF.
  - The plain KDE rate Λ₀ = log(K∗ρ/π) − mean does not vanish at ρ = π, so it undersamples barriers. Their corrected rate replaces π by K∗π. The critical bandwidth depends on N.
  - The baseline is independent Langevin at the same N and step count. That is effectively equal force evaluations, with KDE cost ignored. There is no comparison against fewer, longer walkers.
  - The outlook proposes birth–death "only in some relevant subspace ... of a few collective variables", with the CV distribution estimated on the fly by CV-based enhanced sampling. **That is the project's design, proposed as future work.**
- **TRR 146 Project A10 (Dünweg, Valsson, Hartung) and APS March Meeting 2024 talk T18.2.**
  - A funded project aiming at birth–death population control of multiple-walker, CV-based enhanced sampling, with metadynamics named as the example.
  - It names two limitations: walkers in high free-energy regions get killed, and the initial walkers must cover the metastable states.
  - No paper was found as of Oct 2026. Whether it acts on the CV marginal or is coupled to ABF is UNVERIFIED. **This is likely competing work in progress.**
- **Lindsey, Weare & Zhang, SIAM/ASA JUQ 10, 860 (2022).**
  - Interacting moves on a sub-coordinate u. Walker i's u is replaced while it keeps its own v, so the scheme is exactly π-invariant.
  - It "cannot discover new modes any faster" than independent chains. This is a different design, because orthogonal coordinates are not copied.
- **Lu, Lu & Nolen, arXiv:1905.09863 (2019).** The origin of KDE birth–death as the WFR flow of KL, in the full state space. It notes that the support never grows.
- **Aristoff & Zuckerman, Multiscale Model. Simul. 18, 646 (2020).**
  - Weighted-ensemble allocation that minimises mutation variance, N(u) ∝ √(w(u) Σ w_i Var), shown to beat uniform allocation in a 1D toy.
  - The phrase "ad hoc rule" for uniform allocation appears only in arXiv v3, not in the journal version.

### 1.3 Side-by-side comparison

| Element | LRS 2007 | Comer 2014/2015; Amber ABMD | This project | New? |
|---|---|---|---|---|
| Adaptive bias | Parallel ABF/ABP, binned | Shared multiple-walker ABF; ABMD | Histogram ABF, N walkers | No |
| Selection driven by | Instantaneous replica CV marginal | Cumulative or recent bin counts; ABMD: bias-derived ρ | KDE of the instantaneous walker CV marginal | Partly (the estimator) |
| Rate form | c ∂_zz ψ/ψ (curvature; acts as extra diffusion) | Inverse counts; ABMD: C∇²ρ/ρ | Log density ratio to target (FR/KL score) | **Yes, but incremental** |
| Target | Implied uniform | "More uniform sampling" | Explicit uniform | No |
| Orthogonal coordinates | Copied with clone | Copied | Copied | No |
| Test system | 2D WCA bistable dimer, M = 1000–10000 | Deca-alanine, 10–16 walkers | Gateway toys, LTA/ethane, WCA, alanine, others | Partly (breadth) |
| Gain vs N, with a control at each N | Claimed, not isolated | — | Isolated (FR gain grows with N) [record] | Yes |
| Reference | Inexact (M = 5000, time-averaged) | Long-run profile | Exact or dt-validated references (gateway, LTA 300/150 K; WCA vs Metropolis MC) [record] | Yes (rigour) |
| Equal-cost comparison | None | Equal total simulated time across walker counts; qualitative | Equal force evaluations, best allocation per arm, validated dt, seeds and CIs | **Yes, in rigour** |
| Wall-clock | None | Real time, multiple vs single walker | Best FR vs best ABF on fixed cores, with CI | Yes (unresolved) |
| Failure modes | Unproved remark on submanifold relaxation | Early-only; blind to orthogonal coordinates; revert late | Measured necessary conditions; validated negatives | Quantitative, yes; qualitative, no |
| Controls | Realisation averages | Single runs | Preregistration, shams, paired seeds, TOST | Yes |

### 1.4 What not to claim, and what is new

**Not new (do not claim):**
1. ABF combined with kill/clone driven by the CV marginal. This is in LRS 2007, Minoukadeh 2010 and Comer 2014, in production in NAMD, and in Amber for an adaptive-bias potential. Chopin–Jacob is the SMC analogue.
2. A uniform CV marginal as the target, with selection vanishing there.
3. Copying orthogonal coordinates with the clone.
4. Testing on a WCA-solvated bistable dimer with thousands of replicas.
5. The qualitative failure picture. Selection is blind to orthogonal coordinates (Comer 2014; Amber), helps early and is neutral or harmful late (Comer 2014; Amber; Comer 2015 at 128 ns), and cannot beat the underlying dynamics at exploration (Tan & Lu 2025; Lindsey–Weare–Zhang).
6. The idea of CV-subspace KDE birth–death with an on-the-fly enhanced-sampling estimate (Pampel 2023 outlook; TRR 146 A10).
7. A √(w·Var) allocation, if proposed as the fix. It is Aristoff–Zuckerman's variance-optimal weighted-ensemble allocation. [record: the project lists n ∝ √(w Var(f|ξ)) as a next step.]
8. Comparing ABF walker strategies at equal total simulated time, in qualitative form (Comer 2014, 2015).

**New, or new in rigour:**
1. **A KDE log-ratio (FR) rate toward an explicit uniform target inside ABF.** I found no published version. It is an incremental change of rate, though: Comer 2015 found that rule details gave "similar results". [record] The one head-to-head the project ran was on WCA: count-balancing tied FR and the LRS Laplacian rule did nothing. That run used dt = 0.002, a step later shown unstable. So there is currently **no valid head-to-head** against the prior rules on any system.
2. **A statement about the conditional law.** [record] The project's result is d/dt p(y|ξ)|_FR = 0. Its corollary is that marginal selection cannot speed conditional relaxation, and that it slows a hidden coordinate in proportion to dose, which was measured on alanine with ψ as the only CV.
   - This formalises LRS 2007's unproved remark. The nearest rigorous results are design properties (Chopin–Jacob's conditional-preserving targets; Lindsey–Weare–Zhang's exact invariance), not theorems about this dynamics.
   - Reviewers will read the statement itself as a lemma. The measured corollary is the stronger contribution.
3. **The mechanism, as a controlled quantitative result.** [record] FR needs both establishment-limited ABF and fast conditional relaxation; entropy is not necessary. This comes from the matched-F gateway family and the mobility (λ) experiments. The direction was anticipated; the controlled separation is new.
4. **Gain vs N with an ABF control at every N.** This isolates what LRS 2007 Table III only claimed.
5. **Validated negatives.**
   - WCA at a validated timestep, against an exact Metropolis MC reference: FR is harmful. [record: +9.9% integrated error, with an intrinsic tilt of about 0.017 kT.]
   - Alanine is neutral or harmful across N, temperature, CV, estimator, dose and start time (no FR arm was run on the friction ladder; `EVIDENCE_LEDGER.md` errata).
   - The WCA result **qualifies a published positive on the same model family**, and the paper must reconcile the two. LRS ran at dt = 1e-4, inside the range the project validated [record: dt* = 1.25e-4]. So the timestep artefact that explained the project's own earlier WCA positive does not explain LRS's.
   - Candidate reasons for the difference: Laplacian rule (adds diffusion) vs log-ratio rule (reallocates walkers); realisation-averaged max-norm bias vs per-run error; horizon t ≤ 0.4; an inexact reference.
6. **Equal-force-evaluation N-ladders.** These use the best allocation per arm, a validated dt, exact references, and independent seeds with CIs. None of the prior selection papers do this. The result, that the best few-replica ABF is never beaten, is the clearest new and citable finding.
7. **Wall-clock comparison of best FR vs best ABF on fixed hardware, with a CI.** New as a comparison, but unresolved. [record] LTA 150 K: best FR at N = 64 vs best ABF at N = 32 gives a ratio of 0.717 [0.534, 1.196]. Gateway is pending.
8. **Process.** Preregistration, sham arms, paired seeds and equivalence tests (TOST). This is well above the evidence standard of the prior selection literature.

### 1.5 Corrections to the sweep summaries

- "No multiple-walker metadynamics work with CV-marginal reallocation": **wrong.** Amber has shipped the LRS Laplacian selection for multiple-walker ABMD since Amber 16.
- "No prior equal-cost comparison": **partly wrong.** Comer 2014 and 2015 compare at equal total simulated time across walker counts. What is missing is a tuned N-ladder, seeds with CIs, and exact references.
- LRS 2007 "especially early": at t = 0.05 the low-c runs are worse than no selection. The gains show at t = 0.1–0.4.
- Lambda-ABF's "only modest efficiency gains" is an unquantified practitioner opinion, not data.
- Christiansen et al. report a single scaling curve. It is labelled "weak scaling" but uses a fixed problem size.
- Lazić–Liu–Majka gives no rate and is not birth–death: its particles stay fixed and only their weights evolve. Zhu–Mielke's kernel approximation is kernel ridge regression on non-mass-conserving Hellinger flows, not a KDE. Neither supplies a finite-N, finite-bandwidth error bound for the project's rate.
- Lu–Slepčev–Wang: the plain KDE rate (Lu–Lu–Nolen form) is not the exact gradient flow of the kernelised energy. Kernelised minimisers sit O(ε) from π in W₂. No joint (N, ε) rate exists in the checked literature.

---

## 2. Novelty risk

| Framing | Risk | Reasons |
|---|---|---|
| **"FR accelerates ABF"** (methods paper) | **High** | The construction is LRS 2007 with a different rate, and selection in ABF is in NAMD and Amber. Practitioners already describe the gains as modest. The project's own evidence fails both accepted standards: best ABF is never beaten at equal force evaluations, and the wall-clock CI includes 1. There is no valid head-to-head against the Laplacian or count rules. A competing group (TRR 146) has announced the CV-subspace direction. A referee who knows Comer 2014 would likely reject this as a re-derivation with a weaker cost story. |
| **"When and why marginal birth–death helps or hurts adaptive biasing"** | **Medium** (low-to-medium after the fixes below) | The headline qualitative message is partly known: select early only, selection is blind to orthogonal coordinates, revert late. Novelty rests on the quantitative, controlled evidence: the two necessary conditions, the equal-cost null, the validated negatives, the conditional-law corollary and the wall-clock test. Risk falls if the paper (a) treats the rule family (Laplacian, count, KDE log-ratio) rather than FR alone, (b) reconciles the LRS 2007 WCA positive, and (c) reaches preprint before the TRR 146 group. |
| **Parity plus parallel scaling** (Christiansen-style) | **High / not licensed** | This needs a resolved wall-clock advantage and strong- and weak-scaling curves against the best competitor. The project has 8–16 cores and one unresolved ratio. |

---

## 3. Evaluation standards reviewers will apply

| Standard | Source | Project status | Verdict |
|---|---|---|---|
| Efficiency at equal aggregate cost against the cheapest alternative, including fewer, longer replicas | Zuckerman & Lyman 2006; Zuckerman & Chong 2017; Comer 2014/2015 practice | Validated-dt N-ladders with force-evaluation accounting; best ABF ties or beats best FR | **Meets as a test; fails as an acceleration claim** |
| Each method at its best (or typical-user) parameters, on a benchmark with a highly accurate answer | Hénin et al. 2022, Sec. 1.3 | Best-allocation ABF; bin, ramp and bandwidth audited; exact or dt-refined references on gateway, LTA and WCA [record] | **Meets** for those systems; state the alanine reference status explicitly |
| Head-to-head with the published selection rules | LRS 2007; Minoukadeh 2010; Comer 2014/2015; Amber | Only the WCA five-arm run, at the refuted dt [record] | **Fails (missing)** on every system where FR helps |
| 95% CIs, independent runs as the unit, care with correlated trajectories, full uncertainty procedure reported | Grossfield et al. 2018 | Seeds as the unit, paired arms, TOST, shams | **Meets**, if the manuscript reports the full procedure |
| Credit for parallel methods through wall-clock time-to-solution, with overheads and scaling | Raiteri 2006; Comer 2014; Christiansen 2019 | LTA 0.72 [0.53, 1.20]; gateway pending; no scaling curves | **Fails** for a speed-up claim; reportable as unresolved |
| Separate exploration speed from convergence of the free-energy error | Invernizzi & Parrinello 2022 | FR gains are in establishment and finite-time integrated error, never converged error at equal budget | **Partly meets**; report time-to-accuracy and endpoint separately |
| Transfer beyond simple systems; robustness vs optimality | Hénin et al. 2022 | Positives: toy gateway plus one atomistic case (LTA/ethane 150 K). Negatives: WCA, alanine | Thin for a methods claim; **adequate for a regime map** |
| Correctness of the KDE rate | Pampel 2023; Lu–Slepčev–Wang 2023 | Not stated in the brief: Λ₀ vs corrected form, edge treatment (K∗U ≠ U at a bounded CV's edges), bandwidth vs N | **Must be stated.** Expect questions on whether the WCA tilt relates to the rate form |
| Why a uniform target rather than variance-optimal allocation | Aristoff & Zuckerman 2020 | The project's own finding is that ABF needs adequate per-bin counts, not equal ones | Must be addressed; **cite as prior art** if a √(w·Var) rule is proposed |
| Theory claims precise about scope | LRS 2007 remark; Lindsey–Weare–Zhang; Tan & Lu 2025 | Conditional-invariance statement [record] | Must state mean-field vs finite-N scope and how clone genealogy enters; expect it to be read as a lemma |
| Replica-count gains confined to short horizons read as transient | Nymeyer 2008 (analogy, REMD) | FR helps only at large N and short per-replica time | A reviewer will raise this. Answer with the necessary-condition mechanism, not with an "acceleration" claim |

---

## 4. Recommended positioning and venues

**Positioning.** Present the work as a cost-controlled re-examination of walker selection in multiple-walker ABF. LRS 2007 and Comer 2014 are the starting point. The FR/KDE log-ratio rule is one member of a family of rules (Laplacian, count-based, log-ratio).

**Claims the evidence supports:**
1. Marginal selection preserves the conditional law, so it can only help when the error is limited by establishment of the CV marginal and conditional relaxation is fast. Both conditions are shown to be necessary.
2. At equal force evaluations, the best few-replica ABF allocation was never beaten.
3. When N is forced by hardware (Hénin's "robust to non-optimal circumstances" route), FR lowers the finite-time error in the regime above. Wall-clock benefit on 8–16 cores is unresolved.
4. Selection is harmful on a validated WCA dimer and neutral or harmful on alanine. The paper must explain why this differs from LRS 2007.

**Claims to avoid:** "accelerates ABF", "first to combine birth–death with ABF", "first CV-subspace birth–death", and "novel uniform-target selection".

**Before submission, in priority order:**
1. Run the LRS Laplacian and Comer count rules head-to-head on gateway and LTA 150 K at validated settings.
2. Reconcile LRS 2007 on WCA: same metric, horizon and rule.
3. Finish the gateway wall-clock and report both ratios with CIs.
4. State the KDE rate form and edge handling.
5. Post a preprint soon, given TRR 146 A10.

**Candidate venues:**
- **J. Chem. Phys.** The venue of LRS 2007. It takes mechanism-level method analysis on model systems, which suits a cost-controlled re-examination of parallel adaptive dynamics with selection.
- **J. Chem. Theory Comput.** Where ABF walker selection (Minoukadeh 2010, Comer 2014) and its critique (Lambda-ABF) appeared. Its readers will apply the equal-cost and head-to-head standards, so item 1 above is mandatory.
- **Phys. Rev. E.** Pampel 2023's venue, for the physics of birth–death sampling. It suits a regime map on gateway and WCA, and engages the TRR 146 group directly.
- **SIAM Multiscale Model. Simul. or SIAM/ASA JUQ.** Only if the conditional-invariance result and its corollary are proved rigorously, with the regime numerics as illustration. These are the venues of Aristoff–Zuckerman, Lindsey–Weare–Zhang and Tan–Lu.
- **Living J. Comput. Mol. Sci.** Possible if the work is recast as a comparative best-practice or lessons-learned study of walker-selection strategies. Check the current article types. The standards cited above (Grossfield 2018, Hénin 2022) were published there.
- **Not recommended:** letter-format, high-impact framing (PRL-style). It would need a demonstrated speed-up or a scaling advantage, and the project has neither.

---

## 5. References

1. T. Lelièvre, M. Rousset, G. Stoltz. Computation of free energy profiles with parallel adaptive dynamics. *J. Chem. Phys.* 126, 134111 (2007). doi:10.1063/1.2711185. https://arxiv.org/abs/cond-mat/0611276
2. K. Minoukadeh, C. Chipot, T. Lelièvre. Potential of mean force calculations: a multiple-walker adaptive biasing force approach. *J. Chem. Theory Comput.* 6(4), 1008–1017 (2010). https://doi.org/10.1021/ct900524t
3. J. Comer, J. C. Phillips, K. Schulten, C. Chipot. Multiple-replica strategies for free-energy calculations in NAMD: multiple-walker adaptive biasing force and walker selection rules. *J. Chem. Theory Comput.* 10(12), 5276–5285 (2014). https://doi.org/10.1021/ct500874p
4. J. Comer, J. C. Gumbart, J. Hénin, T. Lelièvre, A. Pohorille, C. Chipot. The adaptive biasing force method: everything you always wanted to know but were afraid to ask. *J. Phys. Chem. B* 119(3), 1129–1151 (2015). https://pmc.ncbi.nlm.nih.gov/articles/PMC4306294
5. D. A. Case et al. (eds; editors not separately verified). *Amber 2016 Reference Manual*, ch. 22 (nfe): adaptively biased MD with multiple-walker selection. https://ambermd.org/doc12/nfe.pdf (same text: *Amber 2022 Reference Manual*, Sec. 25.4, https://ambermd.org/doc12/Amber22.pdf)
6. T. Lelièvre, M. Rousset, G. Stoltz. *Free Energy Computations: A Mathematical Perspective.* Imperial College Press (2010). https://doi.org/10.1142/p579 (existence confirmed; selection-section content UNVERIFIED)
7. L. Lagardère, L. Maurin, O. Adjoua, K. El Hage, P. Monmarché, J.-P. Piquemal, J. Hénin. Lambda-ABF: simplified, portable, accurate, and cost-effective alchemical free-energy computation. *J. Chem. Theory Comput.* 20(11), 4481–4498 (2024). https://arxiv.org/abs/2307.08006
8. N. Chopin, P. E. Jacob. Free energy sequential Monte Carlo, application to mixture modelling. In *Bayesian Statistics 9*, OUP, 91–118 (2011). https://arxiv.org/abs/1006.3002
9. B. Pampel, S. Holbach, L. Hartung, O. Valsson. Sampling rare event energy landscapes via birth-death augmented dynamics. *Phys. Rev. E* 107, 024141 (2023). https://arxiv.org/abs/2209.00607
10. B. Dünweg, O. Valsson, L. Hartung. TRR 146 Project A10: Population control of multiple walker simulations via a birth/death process. https://trr146.uni-mainz.de/project-a10-n/ ; A. Gopakumar Remanidevi et al., "Enhanced Sampling using Birth-Death Algorithm", APS March Meeting 2024, T18.2. https://meetings-archive.aps.org/mar/2024/t18/2/
11. Y. Lu, J. Lu, J. Nolen. Accelerating Langevin sampling with birth-death. arXiv:1905.09863 (2019). https://arxiv.org/abs/1905.09863
12. Y. Lu, D. Slepčev, L. Wang. Birth-death dynamics for sampling: global convergence, approximations and their asymptotics. *Nonlinearity* 36(11), 5731–5772 (2023). https://arxiv.org/abs/2211.00450
13. M. Lindsey, J. Weare, A. Zhang. Ensemble Markov chain Monte Carlo with teleporting walkers. *SIAM/ASA J. Uncertain. Quantif.* 10(3), 860–885 (2022). https://arxiv.org/abs/2106.02686
14. L. Tan, J. Lu. Accelerate Langevin sampling with birth-death process and exploration component. *SIAM/ASA J. Uncertain. Quantif.* 13(3), 1265–1293 (2025). https://arxiv.org/abs/2305.05529
15. F. R. Crucinio, S. Pathiraja. An operator splitting analysis of Wasserstein–Fisher–Rao gradient flows. arXiv:2511.18060 (2025; v3 2026). https://arxiv.org/abs/2511.18060
16. J.-J. Zhu, A. Mielke. Kernel approximation of Fisher-Rao gradient flows. arXiv:2410.20622 (2024). https://arxiv.org/abs/2410.20622
17. P. Lazić, L. Liu, M. B. Majka. On propagation of chaos for the Fisher-Rao gradient flow in entropic mean-field optimization. AISTATS 2026 (PMLR 300). https://arxiv.org/abs/2602.15094
18. D. Aristoff, D. M. Zuckerman. Optimizing weighted ensemble sampling of steady states. *Multiscale Model. Simul.* 18(2), 646–673 (2020). https://arxiv.org/abs/1806.00860
19. D. M. Zuckerman, L. T. Chong. Weighted ensemble simulation: review of methodology, applications, and software. *Annu. Rev. Biophys.* 46, 43–57 (2017). https://pmc.ncbi.nlm.nih.gov/articles/PMC5896317
20. A. Grossfield, P. N. Patrone, D. R. Roe, A. J. Schultz, D. W. Siderius, D. M. Zuckerman. Best practices for quantification of uncertainty and sampling quality in molecular simulations. *Living J. Comput. Mol. Sci.* 1(1), 5067 (2018). https://pmc.ncbi.nlm.nih.gov/articles/PMC6286151
21. J. Hénin, T. Lelièvre, M. R. Shirts, O. Valsson, L. Delemotte. Enhanced sampling methods for molecular dynamics simulations. *Living J. Comput. Mol. Sci.* 4(1), 1583 (2022). https://arxiv.org/abs/2202.04164
22. D. M. Zuckerman, E. Lyman. A second look at canonical sampling of biomolecules using replica exchange simulation. *J. Chem. Theory Comput.* 2(4), 1200–1202 (2006). https://arxiv.org/abs/physics/0601145
23. H. Nymeyer. How efficient is replica exchange molecular dynamics? An analytic approach. *J. Chem. Theory Comput.* 4(4), 626–636 (2008). https://pubs.acs.org/doi/abs/10.1021/ct7003337
24. P. Raiteri, A. Laio, F. L. Gervasio, C. Micheletti, M. Parrinello. Efficient reconstruction of complex free energy landscapes by multiple walkers metadynamics. *J. Phys. Chem. B* 110(8), 3533–3539 (2006). https://doi.org/10.1021/jp054359r
25. H. Christiansen, M. Weigel, W. Janke. Accelerating molecular dynamics simulations with population annealing. *Phys. Rev. Lett.* 122, 060602 (2019). https://doi.org/10.1103/PhysRevLett.122.060602
26. M. Invernizzi, M. Parrinello. Exploration vs convergence speed in adaptive-bias enhanced sampling. *J. Chem. Theory Comput.* 18(6), 3988–3996 (2022). https://arxiv.org/abs/2201.09950