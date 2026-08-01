# ALANINE REFERENCE HANDOFF — corrected umbrella/MBAR FES, Stage 1

Branch `alanine-dipeptide`. Written 2026-08-01. **THE REFERENCE IS ACCEPTED: all 8 gates pass.**
Production ABF-vs-oracle-mFR is **NOT launched** and must not be until this handoff is reviewed.

Companions: `ALANINE_SPEC.md` (design), `ALANINE_EXECUTION_DECISION.md` (repository truth and the
A/B/C change classification). Where they disagree with this file about the *reference*, this file wins.

---

## 1. Commits and files changed

| commit | contents |
|---|---|
| `c6a6718` | design spec frozen; `poisson2d` Nyquist fix + regression tests |
| `f3149ff` | repo-truth audit, pentane Nyquist impact verdict, corrected two-stage seeding, `src/alanine/{system,forcefield}.py` |
| `1ccd9d5` | Category-A: IUPAC CV, full-state cloning, per-seed RNG, fail-fast, projection guarantee, seed gates |
| `ab6a3a3` | reference acceptance analysis + independent OPES cross-check |
| *(this)* | two analysis-bug fixes (§6), reference outputs, this handoff |

New under `src/alanine/`: `system.py`, `forcefield.py`, `cv2d.py`, `dynamics.py`, `projection.py`,
`reference.py`. New scripts: `run_alanine_reference.py`, `analyze_alanine_reference.py`,
`run_alanine_opes_crosscheck.py`, `check_alanine_dt_bias.py`, `audit_poisson_nyquist{,_impact}.py`.
Modified in `src/alkanes/`: `poisson2d.py` only (the Nyquist fix), plus its tests.
**No WCA or alkane birth-death code was touched.**

## 2. Tests

`CUDA_VISIBLE_DEVICES="" python -m pytest tests/ -q` → **109 passed, 3 skipped** (3 skips are the
CUDA-parity tests, skipped on CPU). Breakdown: 73 pre-existing, +8 Nyquist regression,
+10 alanine Stage-0, +21 alanine Category-A, minus reorganisation.

Category-A regression coverage: IUPAC values at C7eq and single-conversion through every consumer;
convention shift provably changes no geometry (mean force and Gram bit-identical, 0.0);
full-state cloning (position, cached force, genealogy copied; velocity fresh; parent untouched;
fixed population; no self-aliasing); Maxwell variance and exact output shape; per-seed RNG isolation
and isolated reproducibility; NaN/Inf containment with state dump and fail-fast; `gB == grad B` on
random fields at odd and even grids; online-vs-frozen field equality; magnitude clipping preserves
direction; even grids rejected; structural no-reference-leakage; BAOAB thermostat.

## 3. Reference run — configuration and cost

| item | value |
|---|---|
| system | Ace-Ala-Nme, 22 atoms, AMBER ff14SB, vacuum, `NoCutoff`, no constraints, **no HMR** |
| integrator | BAOAB, **dt = 1 fs**, γ = 1 ps⁻¹, T = 300 K, **float64** |
| CV | (φ, ψ) IUPAC, atoms (4,6,8,14) and (6,8,14,16); FES grid **n = 97 (odd)** |
| windows | **24 × 24 = 576**, cosine restraint `κ[(1−cos Δφ)+(1−cos Δψ)]`, **κ = 200 kJ/mol/rad²** (σ = 6.40°) |
| copies | **16 per window**, batch **B = 9216**, independently thermalised |
| schedule | 20 ps randomisation (γ = 20) + **100 ps discarded equilibration** + **1 ns production per copy** |
| aggregate | 9.216 µs; trajectory `(9216, 2000, 2)` |
| hardware | 1 × H200 NVL (**GPU 7**), shared with another tenant; ~127 steps/s at B = 9216 |
| wall time | **8782 s = 2.44 h** (1.12 M steps), MBAR 0.6 s |
| param hash | `6ffd00dc241f` |

## 4. Acceptance gates

| # | gate | criterion | measured | verdict |
|---|---|---|---|---|
| 1 | seed geometry+chirality | 100 % pass | Stage A **576/576** (CV err 2.5e−14°, max angle dev 3.57°); Stage B **576/576** | **PASS** |
| 2 | MBAR convergence | resid < 1e−8 | **9.26e−09** in **106** iterations (Anderson m=8), N = 172 800 (300/window) | **PASS** |
| 3 | min NN overlap inside Ω_eval(8 kT) | ≥ 0.03 | **0.0841** (0 of 231 pairs below 0.03; p1 0.0864, p5 0.0898) | **PASS** |
| 4 | median NN overlap | ≥ 0.05 | **0.1120** (all 1152 pairs) | **PASS** |
| 5 | global minimum near L-alanine C7eq | in (−120…−40, +20…+100)° | **(−74.2, +55.7)°** | **PASS** |
| 6 | bootstrap uncertainty + ESS | reported | ΔG = 3.419 ± **0.079** kT (200 copy-level bootstrap replicates); FES SE on Ω_eval median **0.608 kJ/mol = 0.244 kT** (p90 0.718, max 0.863); MBAR weight ESS fraction 0.068 | **PASS** |
| 7 | independent sampler agrees on ΔG | ≈ 0.3 kT | OPES (dispersed start) **3.657 ± 0.037 kT** vs reference **3.419 ± 0.079 kT**; **gap 0.238 kT** | **PASS** (see §7 — the gap is 2.7σ, a real small systematic, not noise) |
| 8 | reference/production physics hash identical | equal | `6ffd00dc241f` recorded in the run manifest; production must assert it | **PASS (by construction)** |

Over the whole torus 27 of 1152 NN pairs fall below 0.03 (min 4.2e−4), but **every one of them lies
outside Ω_eval**, in the sterically forbidden region. Inside the evaluation mask the worst pair is
0.0841, a 2.8× margin.

## 5. Recomputed physics — from the corrected reference only

Nothing below is inherited from the contaminated first attempt.

| quantity | corrected reference | first (contaminated) attempt |
|---|---|---|
| F range on T² | **92.60 kJ/mol = 37.1 kT** | 90.2 kJ/mol = 36.2 kT |
| global minimum (C7eq) | **(−74.2°, +55.7°)** | (−78.8°, +56.2°) |
| C5/β minimum | **(−152.2°, +155.9°) at 0.45 kT** | (−146.2°, +153.7°) at 0.97 kT |
| C7ax minimum | **(+63.1°, −48.2°) at 6.39 kJ/mol = 2.56 kT** | (+63.8°, −41.3°) at 2.44 kT |
| P(φ > 0) | **0.0317** | 0.0302 / 0.0321 / 0.0404 |
| ΔG(φ>0 vs φ≤0) | **+8.53 kJ/mol = +3.42 kT** | 3.17 / 3.41 / 3.47 kT |
| C7eq↔C7ax min-max barrier | **39.38 kJ/mol = 15.79 kT** | 15.50–15.75 kT |
| P(C7eq box) / P(αR) / P(C7ax box) | **0.6375 / 0.0909 / 0.0311** | — |
| **max internal barrier of F(ψ\|φ)** | **0.75 kT** (median 0.17, p90 0.55, over 43 populated φ columns) | 0.64 kT |

Ω_eval(8 kT) covers 2239 of 9409 cells and resolves into **5 periodic connected components**:
a dominant φ<0 megabasin (1708 cells), the C7ax island (347), a secondary lobe (164), and two
slivers (18 and 2 cells).

**The corrected reference confirms rather than overturns the earlier physics.** Every quantity lands
within the spread of the earlier independent estimators. This is an important negative result about
the seeding trap: it was a real defect and had to be fixed, but for *these* observables it did not
materially bias the answer. The reference is now clean by construction rather than by luck.

**ψ is NOT a hidden slow coordinate.** Max internal barrier **0.75 kT** across every populated φ
column. This is the load-bearing measurement behind the recommendation in §9 and it now rests on a
reference whose every seed passed geometry, chirality, steric-energy and CV gates.

## 6. Two bugs in the analysis, found and fixed before reporting

Both produced confidently wrong numbers that contradicted independent measurements, which is how
they were caught. Recorded because the same mistakes are easy to repeat.

**(a) Periodic box membership collapsed a full-range box to a single point.** The idiom
`lo, hi = lo % 360, hi % 360` maps (−180, 180) to `lo == hi == 180`, selecting one grid column
instead of all of them — measured: 1 of 360. Both full-range boxes were therefore ~empty, giving
`P(φ>0) = 0.0000` and `ΔG = +1723 kJ/mol = +691 kT`, against a C7ax box at 3.1 % and a C7ax minimum
2.56 kT up. Fixed by testing the wrapped distance from the box centre, with an explicit full-circle
branch.

**(b) The "internal barrier" of F(ψ|φ) was the global maximum of the column.** That is the height of
the sterically *forbidden* region, not a barrier between populated states. On a synthetic single-well
profile whose true internal barrier is 0, the old formula returned 6.0 kT. It reported a median
15.74 kT and concluded **"ψ IS a hidden slow coordinate"** — the exact reverse of the truth, and it
would have inverted the study's central conclusion. Replaced by a proper periodic min-max barrier:
locate the local minima, take the two deepest, take the smaller of the two arc maxima joining them,
and measure that above the shallower minimum. Corrected result: median 0.17, max 0.75 kT.

## 7. Gates 6 and 7 — closed, with one caveat carried forward

### Gate 6 — statistical uncertainty

Block bootstrap resampling **whole copies within each window** (200 replicates, MBAR re-solved on
every one; 197 s total). Copy-level rather than frame-level because resampling frames treats
correlated samples as independent and understates the error; the 16 copies were independently
thermalised precisely so this bootstrap can see the copy-to-copy component.

| quantity | value |
|---|---|
| ΔG(φ>0) | **3.419 ± 0.079 kT** |
| P(φ>0) | 0.03172 ± 0.00224 |
| FES SE on Ω_eval | median **0.608 kJ/mol = 0.244 kT**, p90 0.718, max 0.863 |
| MBAR weight ESS fraction | 0.068 |

**Headroom.** Against a ~15.8 % ABF residual on the thermal window (≈3.1 kJ/mol), the reference's
own statistical error is **5.1× smaller**. That is a real improvement on the earlier estimate of
3.8× and it clears the "reference must be ≥3× better than the effect" criterion.

*Stated limitation:* this bootstrap bounds **statistical** error only. It is structurally blind to
error common to all windows and all copies — a shared systematic in the force field, integrator or
CV definition. Gate 7 is what probes that.

### Gate 7 — independent sampler

| | ΔG(φ>0) |
|---|---|
| umbrella + MBAR (reference) | **3.419 ± 0.079 kT** |
| OPES, dispersed start (independent sampler) | **3.657 ± 0.037 kT** |
| gap | **0.238 kT** — inside the ≈0.3 kT tolerance ⇒ **PASS** |

**But the gap is 2.7σ on the combined error, so it is a genuine small systematic between the two
samplers, not noise.** Declare it: carry **≈0.25 kT as the systematic uncertainty on ΔG** for any
downstream claim. ΔG(C7ax) is a *secondary* endpoint of the mFR study, so this is tolerable — but it
must not be quoted as if it were 0.08 kT.

**Two failed attempts preceded this, both defects in the cross-check rather than the reference.**
Recorded because the failure modes are instructive:

*Attempt 1* (512 walkers × 400 ps): ΔG = 2.38 kT, gap 1.04 kT. The reweighted histogram accumulated
from 20 ps while the adaptive bias was still growing — `frac(φ>0)` ran 0.30→0.65→0.61→0.39→0.21→0.11
→0.09 — folding the entire bias transient into the estimate.

*Attempt 2* (16384 walkers × 300 ps, first 100 ps discarded): the block series **drifted
monotonically** 2.34 → 1.89 → **1.73** → 1.77 → 1.91 → 2.12 → 2.32 → 2.50 → 2.67 → 2.84 and never
plateaued (tail 2.49 ± 0.25). Discarding the bias transient was not enough because *every walker
started at C7eq*: the **biased** ensemble was still filling outward all run, and the OPES reweighting
identity `exp(-β(F+A))·exp(+βA) = exp(-βF)` holds only in stationarity.

Note the identity is *algebraically exact for any bias*, converged or not — so a stale bias cannot
produce a first-order systematic, only variance. The systematic came from **non-stationarity of the
sampled ensemble**, which is a different and easily-confused thing.

*Attempt 3* (this one) removes the transient by construction: 16384 walkers initialised from the
**576 validated 24×24 rigid-rotation umbrella seeds tiled over the torus** (all 576 pass the seed
gates; starting `frac(φ>0)` = 0.460, i.e. already at the torus-uniform distribution the flattening
bias drives toward). `neff` improved 0.125 → 0.442. The series rose 1.55 → 2.00 → 2.44 → 2.81 → 3.11
→ 3.36 → 3.50 → 3.58 → 3.63 → 3.68 → 3.64 → 3.67 → 3.69 → 3.70 → 3.67 with decaying increments and
then **flattened into a ±0.05 band over the last seven blocks** — a demonstrated plateau, not an
extrapolated one.

## 8. The thermostat question, and a gate of mine that was wrong

The reference ran at ⟨T_kin⟩ = 293–294 K, i.e. **2.2 % below** 300 K, marginally outside the V13
gate as written. Measured cause, on the frozen physics:

| dt | T_kin | **T_conf** |
|---|---|---|
| 0.5 fs | 298.28 K (−0.57 %) | **300.13 K (+0.04 %)** |
| **1.0 fs (frozen)** | **293.09 K (−2.30 %)** | **299.97 K (−0.01 %)** |
| 2.0 fs | 273.42 K (−8.86 %) | **300.69 K (+0.23 %)** |

The kinetic temperature degrades steeply with dt (−0.6 / −2.3 / −8.9 %) — the known BAOAB
kinetic-energy depression, and the 1 fs value reproduces the reference's 293–294 K exactly. The
**configurational** temperature, `kB T_conf = <|∇V|²>/<∇²V>` with an exact 66-pass Laplacian, stays
within **0.25 %** of 300 K across a 4× range of dt. Since the observable is a configurational free
energy, T_conf is the correct gate and it passes with a ~200× margin. **Recommendation: replace V13's
kinetic criterion with T_conf within 1 %, and report T_kin as a diagnostic only.**

**Redesigned TV check — passes.** With matched sample counts (N = 12 800 at every dt) and an
empirical split-half noise floor:

| dt | TV vs 0.5 fs | noise floor | verdict |
|---|---|---|---|
| 1.0 fs | 0.0602 | 0.0570 | consistent with noise |
| 2.0 fs | 0.0662 | 0.0558 | consistent with noise |

There is no detectable configurational difference across a 4x range of dt once the measurement's own
resolution is accounted for.

The first version of this check was **mis-designed**: it compared (φ,ψ) histograms
across the dt ladder against a fixed threshold of 0.02, but two *independent finite samples of the
same distribution* already differ by TV ≈ 0.050 at N = 12 800 over 1296 bins. Observed values (0.0554
at 1 fs, 0.0679 at 2 fs) sat at or below that pure-noise expectation, so the threshold was
unmeetable regardless of the physics. The ladder also had unequal sample counts (25 600 / 12 800 /
6 400) because it sampled every fixed number of *steps* rather than every fixed interval of *time*.
The rerun (above) samples at a matched physical interval and measures the noise floor empirically by
split-half resampling.

## 9. Is vacuum alanine still an expected neutrality control?

**Yes — and the corrected reference strengthens that classification rather than weakening it.**

The load-bearing reason is unchanged and now rests on clean data: **ψ carries at most a 0.75 kT
internal barrier at every populated φ**. There is no hidden slow coordinate for marginal reallocation
to repair, so 2-D ABF on (φ,ψ) biases the only slow directions that exist. The second basin is
genuinely populated (3.2 %) behind a genuine 15.8 kT barrier, so the system is metastable and *not*
discovery-limited — it is simply a system where ABF already resolves what matters.

Note this is a statement about the *system*, provable before any mFR code runs. It does not by itself
predict the mFR result; it predicts that the honest expected outcome is neutrality.

## 10. Is the repository ready for the N = 2048/4096 pilot?

**Two items block it, neither large.** (Gate closure is done.)

1. ~~Gates 6 and 7~~ — **both closed, §7. The reference is ACCEPTED.**
2. **There is no sampler.** `src/alanine/` currently has system, force field, CV, integrator,
   birth-death, projection and reference — but **no ABF+mFR driver**, no configs, no runner, no
   results layout. Category-B diagnostics (frozen-bias, ancestor ESS, family split-half accumulators,
   kernel-matched error, equal-cost reporting, first-hit/establishment times) are all unimplemented.
3. **Category-C must stay out of the baseline.** Exponential forgetting, weighted projection,
   clone-discounted counts, bandwidth 0.20→0.08, `fr_every` 5→500 and tempered targets are separate
   `mFR-ABF-v2` ablation arms. The pilot compares **corrected ABF** vs **corrected oracle mFR** only.

When those close, the next single experiment is the one already specified: ABF vs oracle mFR at
N = 2048 and 4096, all-dominant-basin initialisation as primary with reference-Boltzmann as a crossed
control, transient window 20–200 ps, and the pre-registered stop rule — **halt unless oracle mFR
improves both the integrated FES error over 20–200 ps and the mean-force error at 200 ps, with
ancestor ESS ≥ 0.30 N and no ancestor above 5 % in the rare basin.**
