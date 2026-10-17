# Preregistration — Experiment D4: a direct test of the selection hypothesis (MECHANISM.md §2.3, reading 4)

*Status: EXPLORATORY. Written and committed on 2026-10-17 BEFORE the D4 runs were launched (the commit that adds
this file also adds the instrumentation and its tests; the runs are launched after that commit, and the launch is
logged in `results/adaptive_fr_campaign/logs/D3.log` and `logs/launches.tsv`). Nothing here is used to choose an FR
rate, a stop time or the online rule. Two arms only, ABF vs the same ABF + FR; no sham, placebo or matched-turnover
arm. Block directories and job file use the label `D3` (the next free label in `make_jobs.py`); this document calls
the experiment D4 to follow MECHANISM.md's numbering (D1–D3 exist).*

## 0. Cost (stated first)

| item | CPU core-h |
|---|---|
| ERI 600 K, K256, T 40, seeds 1111000–015, both arms (historical FR arm re-run took 118 s each; ABF the same) | ≈ 1.05 |
| ITE 300 K, K256, T 50, seeds 1121000–015, both arms (190 s each) | ≈ 1.70 |
| diagnostic overhead (measured 1.0 % at K256) and analysis | ≈ 0.05 |
| **total** | **≈ 2.8** |

Memory: the per-event accumulators are ≈ 16 kB per birth event; at ≈ 1 800 (ERI) / ≈ 1 100 (ITE) events per run
the run files are ≈ 30 / 20 MB each (≈ 0.8 GB for the 32 FR runs). No GPU.

## 1. The hypothesis under test

**H_sel (MECHANISM.md §2.3, reading 4).** Late in a continuous-FR run, a CV bin falls below the uniform target
because walkers flowed out of it. The walkers that remain in it are biased toward fibre states that resist leaving.
FR clones those walkers, which enriches those fibre states at that ξ. The population's conditional p(y | ξ) is
thereby displaced from equilibrium (ERI: η 4.6× and u_x² 2.6× ABF's deviation, §2.2), and the mean-force deposits
collected at that ξ are biased (late-half F RMS 0.132 vs ABF 0.038 in ERI; 0.050 vs 0.024 in ITE, §2.1).

What D2 established: the bias is population-level (dropping deposits of walkers cloned < 1 t.u. ago removes 24–27 %
of FR's excess late error). What D2 could not see: whether the walkers FR clones differ in fibre state from the
population they are drawn from, and whether their descendants carry the error.

## 2. Instrumentation (engine `diag_sel`, default off; read-only; `tests/test_diag_sel.py`)

* **Event record.** At every FR birth / death: step; parent (cloned) slot and killed slot; their fine CV bins at the
  event; their ancestor tags; the fibre state of the parent and of the killed walker at the event: η, u_x², bond
  length r, u_z² (u = unit bond vector; η = cos(4πZ_com/c), ERI: +1 at window heights; ITE: cos(2πZ/a), +1 on the
  channel axes).
* **Ancestor tag.** Per slot, the index of the most recent birth event that wrote into it (−1 = founder, never
  overwritten). The tag follows the slot content: a deposit "of event e" is made by the walker created at e until that
  slot is next overwritten. A clone made from a descendant gets its own event and its own birth state.
* **Per-event deposits.** C and M per fine bin, split into early (k ≤ n_steps/2) and late (k > n_steps/2) halves;
  per coarse bin (18) and time-since-birth band (edges 0.1, 0.3, 1, 3, 10 t.u.): count and sums of f, η, u_x², r, u_z².
  Founders' deposits go to separate arrays.
* **Fibre histograms** (both arms): per half, coordinate, fine bin: 40-bin histogram of the deposit's fibre value
  (η on [−1, 1], u² on [0, 1], r on r₀ ± 0.4 Å). The ABF arm's late-half histogram is the conditional law that
  defines the classes.
* **New whole-run accumulators** `Sbond`, `Suz2` next to `Seta`, `Sux2` (always on, read-only).
* **Inertness.** The diagnostic draws no random number and writes no trajectory or estimator state. Tests prove:
  (i) the engine's default path reproduces, bitwise, a two-arm trajectory frozen from the pre-change engine
  (`tests/fixtures/eri_pre_selection_fixture.npz`, engine 6716aefc); (ii) diag on vs off gives identical trajectories,
  accumulators and RNG states in both arms; (iii) the record partitions the deposits exactly and the event table is
  complete (n events = n replacements, no overflow).

## 3. Runs

The FR **and ABF** arms of the two D2 blocks, re-run with `diag_sel = 1` and otherwise identical configurations
(`configs/adaptive_fr/jobs_D3.json`, from `make_jobs.py D3`):

* ERI 600 K, K256, rate 2, seeds 1111000–015 (historical `results/convergence_fr_campaign/ERI_K256`);
* ITE 300 K, K256, rate 1, seeds 1121000–015 (historical `results/convergence_fr_campaign/M300_256_rate1`).

**Identity check (gate).** Every re-run must reproduce its historical arm bitwise in sv_C, sv_M, sv_h, sv_i, sv_u,
sv_cond and q_final. A single mismatch stops the analysis.

## 4. Definitions

* **Late half**: deposits at steps k > n_steps/2 (ERI t > 20, ITE t > 25), as in D2.
* **Reference**: the exact-MC references used by every confirmatory analysis (`results/eri_campaign/reference`,
  `results/overnight/ite/reference`); mean-force error = count-weighted RMS over the 180 bins of Γ − Γ_ref (wRMS);
  F RMS = centred RMS of histogram_pmf(Γ) − F_ref. Seed-jackknife s.e. on pooled estimates (drop one seed of 16).
* **Birth quantile** u_e of event e: the quantile of the parent's fibre value x_e within the ABF arm's late-half
  pooled (16 seeds) histogram at the birth bin j_e, mid-rank within the 40-bin histogram
  (u = [mass below the bin + ½ mass of the bin] / total). u'_e: the same within the FR arm's own pooled late-half
  histogram at j_e (the population the clone is drawn from). The same for the killed walker at its bin (death
  quantiles v_e, v'_e).
* **Ancestor class** of a deposit: founder, or the quartile of u_e of its tag's event in the **carrier coordinate**:
  Q1 (u < 0.25), Q2, Q3, Q4 (u ≥ 0.75); "tail" = Q1 ∪ Q4, "central" = Q2 ∪ Q3.
  * ERI carrier: η (the coordinate D2 found displaced).
  * ITE carrier: not known. Each of the four coordinates is analysed; the ITE carrier is declared (§5, P0) before the
    class analysis is read, by the late-half population displacement.
* **Regions**: window |φ| < 0.5, flank 0.5 ≤ |φ| < 2, cage |φ| ≥ 2 (rad), as in D2.
* **Variance-matched ABF control**: a class holds ≈ ¼ of the FR descendants' deposits, so its wRMS has more noise
  than the all-FR estimate. The sample-size effect is measured on the ABF arm by pooling disjoint quartets of seeds
  (4 × 4 seeds) and averaging their late-half wRMS. This is an analysis of the ABF arm, not an arm.

## 5. Predictions (stated before the runs)

**P0 — ITE carrier.** The late-half population conditional of each coordinate (all FR deposits) vs ABF's, per coarse
bin, standardised by the ABF within-bin sd (from the ABF histogram) and the pooled count: the coordinate with the
largest count-weighted RMS standardised displacement, if it exceeds 3 jackknife s.e. and ABF's own value, is the ITE
carrier. If no coordinate qualifies, the ITE carrier is outside {η, u_x², r, u_z²} and the ITE class analysis is
reported with η as a placeholder but cannot test H_sel (ITE verdict: unresolved). Expectation from physics: the bond
length relaxes in a few steps (overdamped relaxation time ≈ 1/(2k) ≈ 6 × 10⁻⁴ t.u.) and cannot carry a persistent
bias; the orientation relaxes in ≈ 0.1 t.u. and could; η/u_x² were undisplaced in D2.

**P1 — births are fibre-selected.** Under H_sel, the parents FR clones are not a random draw from the population at
their bin: the distribution of u' over births is non-uniform in the carrier coordinate (tail share
P(u' < 0.25 or u' ≥ 0.75) ≠ 0.5, or mean u' ≠ 0.5, by > 3 jackknife s.e.), and the distribution of u (vs ABF's law)
is displaced in the same direction as the population displacement D2 measured. Under the null, u' is uniform within
the histogram's resolution (tail share 0.5 ± noise) in every coordinate. The killed walkers' death quantiles v' are
reported alongside; H_sel's mirror image (deaths remove recently arrived, mobile fibre states) predicts a tail
enrichment of opposite sign; this part is descriptive.

**P2 — the late error is carried by tail-class descendants.** Under H_sel the late-half excess error over ABF
(wRMS and F RMS) is concentrated in Q1 ∪ Q4: excess(tail) ≥ 2 × excess(central), with non-overlapping jackknife
intervals, and the per-coarse-bin signed error of the central classes is closer to ABF's than the tail classes' is
(compare with figure 7b: ERI bins 3–6 positive, 11–13 negative; ITE bins 2–3 positive). Under the null all four
classes show the same signed per-bin pattern within noise, and their wRMS exceed ABF's by about what the
variance-matched ABF control shows.

**P3 — descendants' conditional relaxes toward ABF's with age.** Under H_sel the tail-class descendants' conditional
(per coarse bin, carrier coordinate) deviates from ABF's late-half conditional at small age (< 0.3 t.u.) by more
than the central classes' does, and the deviation falls with age toward ABF's on the conditional-relaxation scale
(within noise by the 3–10 t.u. band). Under the null the descendants' conditional equals ABF's within noise at every
age. A third outcome, deviation that does not fall with age, means the displacement is maintained by something other
than the birth state (see §7).

**P4 — locality (birth bin vs death bin).** Under H_sel the late error of a region is carried mainly by descendants
born in that region (deposits in the birth coarse bin and its neighbours), not by descendants born elsewhere; the
death region is irrelevant to where the error sits. Under the null the signed per-bin error is independent of the
ancestor's birth region. Reported: late wRMS and signed per-region error by birth region, by death region, and for
deposits inside vs outside the ancestor's birth coarse bin.

**P5 — age bands (D2 extended).** Late-half wRMS by time since birth, all classes and tail vs central. D2 predicts
the > 1 t.u. bands carry most of the excess; H_sel additionally predicts the excess within each band is larger for
tail than for central classes.

## 6. Decision rule (per system)

* **SUPPORTED**: P1 (births selected in the carrier, > 3 s.e.) AND P2 (tail excess ≥ 2× central, intervals
  disjoint) AND P3 (deviation at small age larger for tail, falling with age).
* **NOT SUPPORTED**: P1 null (|tail share − 0.5| < 2 s.e. in every coordinate) AND P2 null (class wRMS within
  2 s.e. of each other, all ≈ ABF + the variance-matched sample-size effect) AND P3 null (descendants' conditional
  within 2 s.e. of ABF's at every age). The distortion then comes from elsewhere (§7).
* **UNRESOLVED**: anything else — in particular P1 holds but P2 fails (FR selects fibre states, but the error is not
  carried by the selected lineages), or P0 finds no ITE carrier.

## 7. "Elsewhere": what is tested next if H_sel is not supported

1. **Death-side selection.** FR removes walkers from over-populated bins; the walkers it removes are those that
   arrived recently, i.e. in mobile fibre states. The population at the death bins is then depleted of mobile states
   without any birth-side selection. The event record contains the killed walkers' fibre states (v, v'), so this is
   read directly here (P1's death part); a dedicated test would split the late error by *death-bin* lineage history.
2. **Residence-time (arrival-law) distortion.** Birth–death changes the age-at-ξ distribution of the population at
   a bin. Where p(y | ξ) relaxes more slowly than a walker's residence time in the bin, the population's conditional
   reflects the arrival law rather than the equilibrium law, whichever walkers are cloned. The direct test is the
   per-bin residence-time distribution and the conditional as a function of residence time in both arms, which needs
   a per-walker entry-time record (not instrumented here).
3. **A transverse (y, z COM) carrier in ITE.** Only η (a Z phase), the orientation and the bond are measured. If P0
   finds no carrier, the next instrument is the molecule's COM position in the channel cross-section.

## 8. Reporting

MECHANISM.md gains §2.4 with: the identity-check result; P0 (ITE carrier table); P1 (birth / death quantile tables,
both references); P2 (class table: deposit share, wRMS ± s.e., F RMS; per-coarse-bin signed error figure vs 7b);
P3 (descendants' conditional vs age); P4 and P5 tables; the variance-matched ABF control; the verdict by the §6 rule;
sample-size caveats as in §2.1. Script: `scripts/adaptive_fr/selection_split.py` → `results/adaptive_fr_campaign/
mechanism/selection_split.json`; figure `figures/adaptive_fr/mechanism_selection.png`. COMPUTE_LEDGER.md is updated
with the measured core-h.
