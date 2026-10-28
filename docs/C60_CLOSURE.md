# C60–C60 in water: CLOSED without verdict (2026-10-28)

*Closed by user directive on 2026-10-28, on the NSCC login node, after the assessment recorded below.
Companion to `docs/SPEC_c60_water.md` (frozen, Amendment 16; untouched) and
`docs/C60_EXECUTION_STATE.md` (banner added). This is a closure record, not an amendment: no clause of
the spec changes, and nothing here may be read as a gate verdict.*

**Claim labels.** *[measured]*: a number taken from a repo artifact. *[estimate]*: arithmetic from stated
inputs, not a measurement. *[directive]*: a decision by the user.

## 1. Status at closure

* **No verdict-bearing C60 datum was ever produced.** The study was suspended on 2026-08-16 and again on
  2026-08-30 with zero gate inputs; only the phase-A solvent anchors of reference build 1 existed, on a
  machine this repo no longer runs on. The spec's zero-data state and every amendment stand unchanged.
* The batched torch engine (`src/c60/`) was mid-parity at suspension. The NSCC conda environment that
  carried torch + OpenMM was found gutted by the scratch purge on 2026-10-27 *[measured, see the LTA
  notes]*; `results/c60/` does not exist on NSCC.
* `src/c60/omm.py` (commit `1a0a840`) is an **unvalidated, never-run** OpenMM-CUDA draft. It stays on the
  branch as code, not as an engine; nothing may cite it as validated.
* The study closes as **NOT RUN**. It is not ABF-sufficient, not discovery-limited, not
  conditional-equilibration-limited, not establishment-limited. It contributes no cell to the regime map.

## 2. Why it is closed *[directive, on the grounds below]*

1. **Cost of the frozen protocol in explicit water** *[estimate from frozen budgets]*. Reference ~857 ns,
   Gate 0 pools ~116 ns, screen 128 ns per cell, production (two arms, see §5) 2 x 16 seeds x 128 ns
   = 4.1 µs plus 25 % frozen-bias validation: about 5.3 µs of 5248-site TIP4P-Ew dynamics at 1 fs. The
   torch engine measured 175 ns/day aggregate at 1 fs on an idle H200 *[measured,
   `throughput_idle_reference.json`, recorded in the execution state]*; an A100 is slower. That is more
   than 30 GPU-days of compute before the NSCC queue limits (one running job per GPU queue, shared with
   the account's other projects) are counted.
2. **The prior is ABF-sufficient.** The solvent-induced barrier the spec expects to reproduce is ~6 kJ/mol
   at 300 K (Zangi 2014, Fig. 1d), about 2.4 kT. Methane (no barrier of note) and NaCl (5.34 kT) both
   closed ABF-sufficient with the same gate machinery. The LTA finite-loading preregistration already
   recorded this prior when it chose LTA over C60.
3. **The cloned-environment cost is now measured, and it is adverse.** Finite-loading ethane/LTA
   (`docs/lta_loading/RESULTS.md`, 2026-10-28) *[measured]*: at the equal-budget optimum N = 64, FR's
   ancestor ESS/N fell to 0.026, the final error was +41.9 % [+7.5, +57.6] worse than ABF, and the
   best-vs-best contrasts were unresolved (R_B 1.02 [0.81, 1.25]; modelled wall-clock R_W 0.93
   [0.80, 1.10]). The mechanism is that cloning copies the whole box, 15 interacting neighbours included.
   An explicit-water C60 walker clones 1282 waters. The question the spec posed (does collective
   interfacial water open an establishment gap?) would sit on top of a genealogical cost that is already
   measured as cancelling FR's deployment-level advantage.
4. **No cheaper variant keeps the question.** See §3.

## 3. Alternatives considered on 2026-10-28, and why none is adopted

| proposal | decision | reason |
|---|---|---|
| A. One-site (Choudhury-style) C60 + atomistic TIP4P-Ew water | rejected | saves ~1–2 % of the force work (§4); changes the model, so the frozen reproduction gate no longer applies; same cloned-water cost |
| B. MARTINI C60 + MARTINI water (Monticelli 2012) | not adopted | ~10^3 cheaper per ns, but the CG C60 beads were tuned against the atomistic C60–C60 PMF in water, so PMF agreement is a fit, not validation; the I2 confined-water mechanism that motivated C60 does not exist at 4:1 water mapping; needs a new walker-controlled engine (GROMACS/PLUMED on NSCC cannot clone walkers) |
| C. Atomistic conditional pilot at ~5 fixed distances | not run | this is exactly the frozen Stage II (Gate 0 pools + tau_perp), ~100 ns; it would most likely close C60 as conditional-limited or ABF-sufficient, and an establishment-limited verdict would still face the §2.1 production cost |
| NaCl 1-D rerun / reduced-N / shorter runs | rejected | closed ABF-sufficient 2026-08-15 with power (`results/nacl/RESULT_STUDY.md`); nothing to repair |
| NaCl 2-D CV `(r, n_NaO)` | rejected | the accepted NaCl reference measured hydration varying 14–83x more across `r` than at fixed `r` (the pre-committed orthogonality ratio), so `n` is a re-spelling of `r` for this system |
| NaCl as a negative-control production | rejected | the ABF-only screen is the control; a two-arm neutral confirmation costs A100-days for no new information |

## 4. Cost arithmetic for proposal A *[estimate]*

Cutoff 1.0 nm, bulk water ~33 nm^-3, so ~140 waters per cutoff sphere.

| term | pairs per step |
|---|---|
| water–water, 1282 waters x 140 / 2 water pairs, 10 site pairs each (3 x 3 Coulomb + O–O LJ) | ~0.9 x 10^6 |
| C60–water, 120 carbons x 140 oxygens (carbons uncharged; TIP4P-Ew H and M carry no LJ) | ~1.7 x 10^4 |
| PME reciprocal space, 3846 charged sites | unchanged by A |

Replacing 120 carbons by two spherical sites removes under 2 % of the direct-space work. The cost of this
benchmark is the solvent, and the solvent is the physics the benchmark was opened for.

## 5. Standing rules for any successor benchmark *[directive, 2026-10-28]*

* **Fisher–Rao is the only particle-reallocation method.** No matched-turnover sham, no count-balancing,
  no book-Laplacian arm. A production design is two arms: ABF and ABF + FR, scored against the reference.
  Where an existing frozen spec (C60 §9, the V2 preregistration arms) names such arms, that clause is
  superseded by this directive rather than executed.
* Compare at equal walker-step budget and at modelled wall-clock, not only at fixed N (the LTA lesson).
  A fixed-N gain that disappears at the budget optimum is reported as that, not as a positive.
* A system with a cloned environment of many interacting particles starts from the finite-loading LTA
  prior (adverse at the budget optimum). The one confirmed equal-budget FR win, single-guest ethane/LTA
  at 150 K (R_W 0.72 [0.64, 0.83], `docs/nscc_a100/RESULTS.md`), had a 13.5 kT CV barrier, a cheap
  force evaluation and nothing to clone but the tagged particle. That profile is the search direction;
  a solvated dimer is not.

## 6. What this closure does not claim

* It does not classify C60. Reopening it means running the frozen spec from Stage 0, on a validated
  engine, under a new execution-state file; the §12 rule that a successor is never a retuned C60 stands.
* It does not retract the literature motivation (Zangi's I2 waters). It records that the campaign cannot
  afford to test it in explicit water on NSCC, and that the campaign's own evidence now predicts the test
  would not produce an FR positive at equal budget.
