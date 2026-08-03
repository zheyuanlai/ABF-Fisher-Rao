# v1 closure and freeze manifest — mFR–ABF

**Status: v1 experiments are CLOSED. No simulation is running. No new v1 experiment is
authorized after the freeze commit below.**

This file is the entry point for anyone picking the project up. It records what was frozen,
where the authoritative artifacts are, how to regenerate every number and the report, and
what was deliberately left for a future v2.

---

## 1. Freeze points

There are two, and they are different things.

| | commit | what it is |
|---|---|---|
| **Raw experimental freeze** | `a685d99` | The last commit that could still have changed a production result. Every `results/**` raw artifact quoted in the report predates it. Nothing after this commit runs a simulation. |
| **Report closure** | the commit carrying this file | Analysis, inventory, report and documentation only. It regenerates derived summaries and adds the closure layer; it does not add, remove, or alter a single raw run. |

Tagged `v1-regime-map-final`.

Everything between those two commits is post-hoc analysis on untouched raw data, run on CPU.
The specific derived artifacts regenerated during closure, and why, are listed in §6.

**Environment at freeze** — conda env `abffr`: Python 3.14.4, numpy 2.4.6, torch
2.12.0+cu130, matplotlib 3.10.9, pandas 3.0.3. Report built with tectonic 0.15.0.

**Validation at the closure commit** — 215 tests: **212 passed, 3 skipped, 0 failed**
(the three skips are GPU-parity checks, correctly skipped under `CUDA_VISIBLE_DEVICES=""`).
Report compiles clean: 89 pages, zero errors, zero undefined references, zero undefined
citations. `results/closure/v1_results_inventory.csv` is byte-reproducible from a fresh
`build_closure_inventory.py` run. No GPU was used and no simulation was run during closure.

---

## 2. Completed systems

Eleven benchmarks, in three regimes. The regime is assigned from discovery and establishment
evidence that does not reference the mFR outcome; see `results/closure/v1_regime_map.csv` and
`report/sections/07_synthesis.tex`.

| System | CV | Regime | Seeds | Raw artifacts |
|---|---|---|---|---|
| 2-D metastability model | `x` | ABF-sufficient | 0–4 | `results/two_dim_xi_x/production_gpu/` |
| Entropic bottleneck, β≤4 | `x` | ABF-sufficient | see config summary | `results/entropic_bottleneck/` |
| Entropic bottleneck, β=8 | `x` | establishment-limited | 20 seeds | `results/entropic_bottleneck/` |
| Butane torsion | `φ1` | ABF-sufficient | 1–16 (β=1), 1–12 (β=0.5, 2) | `results/alkanes/production/raw/` |
| Pentane torsion | `φ1` | ABF-sufficient | 1–16 | `results/alkanes/production/raw/` |
| Pentane end-to-end distance | `R15` | **discovery-limited** | 1–8 | `results/alkanes_cv_extension/r15_methods/raw/` |
| Pentane torsion torus | `(φ1,φ2)` | ABF-sufficient | 1–6 | `results/alkanes_cv_extension/2d_methods/raw/` |
| Alanine dipeptide | `(φ,ψ)` | ABF-sufficient | 0–3, 10–13, 20–23 | `results/alanine_oracle/{pilot,rate_ladder}/*/raw/` |
| Valine dipeptide | `(φ,χ1)` | ABF-sufficient | 0–15 | `results/valine/v3_screen/raw/` |
| Entropic gateway | `x` | **establishment-limited** | 0–15 (map/calibration), 100–131 (confirmatory) | `results/gateway_{phase,anchor}/*/raw.npz` |
| WCA dimer | bond coordinate | **establishment-limited** | 0–9 (production/phase), 400–415 (Case IX sham) | `results/wca_*/**/raw/` |

**Raw `*.npz` files are gitignored and therefore local-only.** Only the aggregated
`.csv`/`.json` summaries are version-controlled. A future collaborator without this machine
can read every conclusion from the tracked summaries but cannot re-derive them from raw
without the local `results/` tree.

### Reference free energies

| System | Reference | Path |
|---|---|---|
| Metastability | quadrature-exact | `results/two_dim_xi_x/reference/` |
| Entropic bottleneck / gateway | analytic (no reference error) | — |
| Butane | exact analytic `F(φ1)=V4(φ1)+C` | validated in `results/alkanes/references/reference_validation.json` |
| Pentane torsion / `R15` / torus | importance-sampling reference + independent uniform-proposal cross-check | `cache/alkanes/`, `cache/alkanes_cv/`; validated in `results/alkanes_cv_extension/references/cv_reference_validation.json` |
| Alanine | 24×24 periodic umbrella + MBAR, ff14SB vacuum | `results/alanine/reference/reference.npz` (ΔG = 3.419 ± 0.079 kT; declared systematic floor 0.25 kT) |
| Valine | 18×18 provisional pilot reference | `results/valine/pilot_reference/pilot_reference.npz` — **`meta.json` flags `IS_NOT_PUBLICATION_QUALITY`**; it supplies target populations for the establishment metric only |
| WCA dimer | cached thermodynamic integration | `cache/phase/` |

---

## 3. Authoritative results inventory

**One table, one sign convention, generated from artifacts:**

```bash
python scripts/build_closure_inventory.py
```

| Output | What |
|---|---|
| `results/closure/v1_results_inventory.csv` | 58 rows: every system, arm, comparator, endpoint, estimate, CI, seed count, regime, prereg reference, artifact path, caveat |
| `results/closure/v1_results_inventory.md` | the same, human-readable, grouped by family |
| `results/closure/v1_regime_map.csv` | one row per benchmark: discovery evidence, establishment evidence, mFR result, regime |
| `report/tables/{closure_inventory,synthesis,alkane_closure,dipeptide_closure,regime_map}.tex` | the report's closure tables |
| `report/tables/closure_numbers.tex` | `\def` macros for in-text numbers |

**Sign convention.** `rel_pct = 100 × (arm − comparator) / comparator` on the named endpoint.
**Negative means the arm is better.** The source artifacts use three different conventions —
`median_gain_pct_F` (positive = better), `pct`/`int_l2_f_pct` (negative = better), and
`rel_med` (a fraction, not a percentage) — and the inventory normalises all of them. Rows with
different `endpoint` values are not comparable and the report never mixes them silently.

The builder runs consistency checks and exits non-zero on failure: duplicate rows, missing or
nonfinite values, sign disagreement between `rel_pct` and the estimates, `favorable_seeds`
inconsistent with `n_seeds`, missing artifact paths, unrecognised regimes, regimes with no
stated basis, and any row sourced from a smoke or tuning stage. Two warnings are expected and
correct: two WCA phase-diagram cells report `|rel_pct| > 200%` because their ABF baseline error
is already tiny, so a small absolute change is a huge relative one. Quote those with the
absolute estimates alongside.

---

## 4. Headline results

All values reproduce from raw; the endpoint is the time-integrated `L²` free-energy error.

| Comparison | Δ | 95 % CI | seeds |
|---|---|---|---|
| **WCA dimer** (Case IX), practical mFR vs ABF | **−22.83 %** | [−25.42, −17.34] | **16/16** |
| **WCA dimer**, practical mFR vs its own matched sham | **−26.38 %** | [−28.89, −16.22] | **16/16** |
| WCA dimer, matched sham vs ABF | +2.60 % | [−0.65, +9.22] | 5/16 |
| **Gateway confirmatory v2** (quoted replicate), mFR vs ABF | **−12.48 %** | [−16.30, −9.84] | **31/32** |
| Gateway confirmatory v2, mFR vs its own sham | −14.95 % | [−16.22, −12.34] | 32/32 |
| Gateway confirmatory v1, mFR vs ABF | −12.12 % | [−14.50, −8.41] | 31/32 |
| Gateway confirmatory v1, mFR vs its own sham | −11.79 % | [−14.10, −8.19] | 31/32 |

Both gateway passes clear the preregistered **primary** rule. Amendment 2 of
`results/gateway_anchor/CONFIRMATORY_PREREGISTRATION.md` fixed in advance that when both pass,
the **replicate is quoted and both are cited** — which is why v2 is the headline.

---

## 5. Caveats that must travel with the headline

1. **The gateway sham equivalence (TOST) does not replicate.** It passes in v1 and misses the
   preregistered ±5 % margin in v2, where the 90 % CI upper bound is **+5.40 %** — a miss of
   **0.40 percentage points**. Both point estimates put the sham on the *harmful* side of ABF,
   so the failure cannot mean turnover explains the gain, but the rule as written did not
   distinguish the direction of failure. Attribution rests on the direct arm-vs-sham contrast,
   not on equivalence.
2. **The gateway direct sham contrast is post hoc for v1.** It was added during v1's analysis.
   It was *preregistered* for the WCA comparison, which is the stronger of the two claims.
3. **The WCA sham fails equivalence in the adverse direction** (+2.60 %, 90 % CI
   [+0.03, +8.88]). A control that makes the estimate worse cannot explain an arm that makes
   it better, so this is informative — but it is *not* a neutral procedural control, and it
   should not be described as "stronger than equivalence".
4. **WCA round trips are not unchanged, only nearly so.** Paired per seed, practical mFR
   changes barrier crossings by **+1.27 %** (95 % CI [+0.99, +1.50]) while changing the
   integrated error by −22.83 %. The establishment reading rests on that ~18× separation, not
   on the crossings being identical. It is empirical support for a representation mechanism
   *in this setup*, not a general theorem.
5. **Alanine was only ever run with the ORACLE target.** No practical estimated-target arm
   exists. The null is therefore the ideal-information control returning nothing, which is
   stronger than a practical null — but it is not a practical-arm result.
6. **Valine has no mFR arm at all.** Its establishment gate never fired, so none was
   justified. Its ABF-sufficient classification rests on the ABF-only screen.
7. **The valine reference is explicitly not publication quality** (see §2).
8. **The gateway β axis is a time-budget axis, not a landscape axis.** β·H is held fixed, so
   the dimensionless landscape is identical in every cell; `beta_scaling_audit.json` records
   `tau_est` varying 1.21× against an 8× change in β. Report it as a finite-budget
   establishment map, not a sweep over different equilibrium problems.
9. The metastability and WCA-production positives are **exploratory** — hyperparameters were
   selected on those same studies. Case IX re-tested WCA on fresh seeds with nothing retuned.

---

## 5a. UNRECONCILED: a parallel audit line that challenges the mechanism attribution

**Read this before quoting anything above as final.**

A second research line exists in this repository, on the local branch
**`claude/variance-aware-abf`** (tip `9642afb`, 50 commits not in `main`). It forked from the
same commit as this line — `b84de17`, 2026-07-21 — and was never merged in either direction.

**This branch is local-only — it does not exist on `origin` — and must not be deleted.** It is
the only reference keeping its last three commits (`9642afb`, `693e5f3`, `5f5ed46`, the last
of which is a "v2 novelty gate: DO NOT BUILD" decision) reachable; everything earlier on the
line is additionally held by the tags `v1-final-closure` (at `bae4534`) and
`v1-selection-evaluation` (at `f0c8d8c`). Two redundant pointers into the same history,
`claude/abf-fisher-rao-audit-fdbcfb` (`ee598d2`) and `mfr-mechanism-audit` (`733d2e8`), were
deleted during closure after confirming both are ancestors of `claude/variance-aware-abf`, so
no commit was lost; both hashes remain reachable from its tip.

Note that `v1-final-closure` — a tag that sounds like it names this closure — in fact points
into *that* line, not this one.

That line ran its own closure on 2026-07-28 and reached materially more conservative
conclusions. Its artifacts are `results/v1_closure/v1_results_table.csv`,
`docs/V1_FINAL_CLAIM_LEDGER.md` and `docs/ADVISOR_DECISION_BRIEF.md`, built by
`scripts/mfr_audit/build_v1_results_table.py`. Its principal findings:

1. **The cached TI reference inflates WCA improvements by roughly 2×.** Three independent
   high-precision constrained-TI replicates (960 k samples per `z`) put the cached reference
   0.264 rms from their mean — about 10× the arm effect. On `final L²(F′)` the WCA contrast
   reads **−4.75 % on the cached reference and −2.41 % on the high-precision one**. That line
   retires the phrase "5–6 % WCA gain".
2. **The Fisher–Rao score is not the active ingredient.** Plain count balancing matches it in
   WCA (`fr_uniform − count_bal` = +1.53 %, p = 0.17) and beats it in `R15`.
3. **Estimator-variance reduction is rejected as the mechanism** — variance is ~1 % of the WCA
   error budget and *rises* under count balancing.
4. **The novelty of adding selection to ABF is pre-empted** by Lelièvre–Rousset–Stoltz (JCP
   126:134111, 2007) and Comer et al. (JCTC 10:5276, 2014), the latter shipping a
   `1/(bin count+1)` balancing rule in NAMD.

**What is and is not in conflict.** That line's WCA numbers are a *different endpoint*
(`final L²(F′)`, mean force) on *different arms* (`count_balanced`, `random_turnover`,
`fr_uniform`) in a *different run tree* (`results/wca_replay/`) from Case IX here
(`integrated L²(F)`, `fr_estimated` vs `sham_practical`, `results/wca_sham/`). Its table
covers only WCA, the entropic bottleneck and pentane `R15`; it predates and therefore says
nothing about the entropic gateway, the Case IX matched sham, the alkane torsion cells, or
either dipeptide. So it does **not** directly refute any number in §4 above.

It does, however, put two claims this report still makes into doubt, and they are not
reconciled here:

* the report attributes the mechanism to **balanced-resampling variance reduction**
  (`report/sections/05_case_entropic_bottleneck.tex`,
  `report/sections/06_case_wca.tex`); that line rejects variance reduction outright;
* the Case IX headline uses the **cached TI reference**, which that line shows to be
  inflationary for a related WCA contrast. Whether the same inflation applies to
  `integrated L²(F)` at cell `b1_h2` has not been measured.

**This is an open decision for a human, not something closure should resolve silently.** The
options are to merge the audit line and rewrite the mechanism sections around its ledger, to
re-run Case IX against a high-precision reference, or to keep the two lines separate and
publish only the reference-robust contrasts. Nothing in this closure commits to any of them.

---

## 6. Derived artifacts regenerated during closure (and why)

No raw artifact was modified. These derived summaries were regenerated on CPU:

| Artifact | Why |
|---|---|
| `results/gateway_anchor/confirmatory_v2/confirmatory_summary.json`, `confirmatory_comparison.csv` | `baseline_noise_matched` was recorded **`false`** for the run that fixed exactly that defect. The flag was inferred from the ABF batch *count*, and Amendment 2's single shared batch (`group == "all"`) looks identical to the pre-Amendment case where one batch's baseline was discarded. Now distinguished by the tag. Every numeric value is unchanged; only the flag flipped. |
| `results/gateway_anchor/confirmatory/*` | Re-run with the same fix so both passes carry the provenance columns. v1's flag correctly stays `false` — it genuinely has the pairing defect for the oracle arms. All numbers unchanged. |
| `results/wca_sham/sham/sham_summary.json`, `sham_comparison.csv` | Added the **paired** round-trip statistic. The report's establishment claim rested on a ratio of medians with no interval attached. The new fields draw from an independent RNG stream so no existing interval moved: the regenerated file differs from its predecessor by additions only. |
| `results/alanine_oracle/pilot/analysis/*_N2048.*` | **These did not exist.** `analyze_alanine.py` wrote un-suffixed filenames and the stage-suffixed copies were made by hand, so each stage overwrote the previous one's analysis and the N2048 pilot's outputs were lost. The raw runs were intact, so it was re-derivable; the script now writes the stage-suffixed copy itself. Regenerating N4096 and N2048_refeq reproduced their stored values bit-for-bit, which is what makes the recovered N2048 value trustworthy. |
| `results/alanine_oracle/pilot/analysis/*_N4096.csv`, `*_N2048_refeq.csv` | Column rename only (`wmax_c7ax_max` → `wmax_rare_max`, `ess_age_c7ax_min` → `ess_age_rare_min`): the checked-in copies were stale snapshots from before the columns were made generic. Data identical. |
| `report/tables/*.tex` | Regenerated from the above by the asset scripts. |

The un-suffixed files in `results/alanine_oracle/*/analysis/` always belong to whichever stage
ran last. **Cite the stage-suffixed copies.** Nothing in the codebase reads the un-suffixed
ones.

---

## 7. Commands

```bash
conda activate abffr

# authoritative inventory + report tables (CPU, seconds)
python scripts/build_closure_inventory.py

# report figures that are not already checked in
python report/scripts/make_regime_map_figure.py

# report macros from the neutrality / gateway / sham artifacts
python scripts/make_neutrality_report_assets.py

# build the report
cd report && tectonic -X compile main.tex

# tests
CUDA_VISIBLE_DEVICES="" PYTHONPATH=src python -m pytest tests/ -q
```

Per-system analyzers, all CPU-only re-analysis over existing raw:

```bash
python scripts/analyze_wca_sham.py                       # Case IX, frozen rule
python scripts/analyze_gateway_confirm.py                # defaults to the quoted v2 replicate
python scripts/analyze_gateway_confirm.py --dir results/gateway_anchor/confirmatory   # v1
python scripts/analyze_alkanes.py --config configs/alkanes/production.yaml
python scripts/analyze_alkanes_cv_extension.py --config configs/alkanes_cv_extension/r15_methods.yaml
python scripts/analyze_alanine.py --root results/alanine_oracle/pilot --stage N4096 --window 20 100 --kind pilot
python scripts/close_valine.py
```

---

## 8. Deferred to v2 — documented, not started

None of these is scaffolded. No config, runner, or result directory exists for any of them,
and none should be created under the v1 freeze.

1. **Post-clone rejuvenation / decorrelation.** Every v1 failure at high rate is a diversity
   collapse: clones are exact copies and their orthogonal degrees of freedom re-equilibrate
   too slowly. A short decorrelation kernel after cloning is the obvious next lever, and v1
   deliberately did not test it.
2. **A factorial directed-vs-random × rejuvenation-vs-none study.** v1 has the matched sham
   (direction, holding turnover fixed) but never crossed it with rejuvenation, so the
   interaction is unmeasured.
3. **Target-estimator redesign.** The entropy-dominant stress test shows the *oracle* gain
   rising with the entropic share while the deployable estimated target captures essentially
   none of it. Wherever an oracle shows headroom, the online target is the binding constraint.
4. **New molecular benchmarks — e.g. chignolin.** v1's molecular systems are all
   ABF-sufficient or discovery-limited. A molecule with a genuinely establishment-limited
   coordinate is what would move the positive off a constructed model and a dimer.
5. **A different selection / population-balancing algorithm.** mFR is one choice of
   reallocation rule; v1 measured it against ABF and against matched random turnover, not
   against alternative balancing schemes.

---

## 9. Where things are

| | |
|---|---|
| Report source | `report/main.tex` (+ `report/sections/`, `report/tables/`, `report/figures/`) |
| Compiled report | `report/main.pdf` — 89 pages |
| Executive synthesis | `report/sections/00b_executive_synthesis.tex` (§1 of the PDF) |
| Regime map | `report/sections/07_synthesis.tex`, `\ref{tab:regime_map}`, `\ref{fig:regime_map}` |
| Authoritative inventory | `results/closure/v1_results_inventory.{csv,md}` |
| Per-case handoffs | `HANDOFF.md`, `ALKANES_HANDOFF.md`, `ALKANES_CV_EXTENSION_HANDOFF.md`, `ALANINE_*_HANDOFF.md`, `VALINE_*_HANDOFF.md`, `OPES_handoff.md` |
| Preregistrations | `results/gateway_phase/production/PREREGISTRATION.md`, `results/gateway_anchor/CONFIRMATORY_PREREGISTRATION.{md,json}`, `results/wca_sham/PREREGISTRATION.md`, `results/alanine_oracle/rate_ladder/PREREGISTRATION.json` |
| Frozen classification | `results/gateway_phase/production/phase_classification.frozen.json` (commit `61a8c1d`, before any FR arm ran) |

### Known non-authoritative leftovers

* `results/wca_followup/` contains only smoke-stage manifests. The real follow-up studies are
  `results/wca_representative/`, `results/wca_equal_compute/`, `results/wca_frozen_bias/`.
* `results/valine/v3_screen/raw/` holds a `.static.npz` sidecar for an abandoned
  `T=1000000` run with no matching full artifact, plus a `.partial.npz` checkpoint of the real
  run. `close_valine.py` excludes both by suffix; do not glob that directory naively.
* `results/valine/pilot_reference_v1_rejected/` is a superseded reference build, kept for
  provenance and named accordingly.
* `results/gateway_phase/pilot/` has raw + provenance but no analysis output; it is superseded
  by `production/`, which added the β axis.
