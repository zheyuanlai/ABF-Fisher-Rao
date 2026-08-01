# VALINE SCREEN SPEC — Ace-Val-Nme, ABF-only screening phase

**Status: SCREEN ONLY. No oracle-mFR comparison, no production. Stop for review at the end.**
Written 2026-08-01, after `alanine-closed-v1`.

Prerequisite context: `ALANINE_ORACLE_PILOT_HANDOFF.md` (alanine closed EQUIVALENT),
`ALANINE_EXECUTION_DECISION.md` §7 (the four Val gates). A concurrent session has already
completed Val Stage 0; its findings are incorporated below and **not** re-derived here.

---

## 0. What Stage 0 already established (do not redo)

1. **χ1 barriers are 10–17 kT.** Far larger than alanine's ψ (≤0.75 kT).
2. **Therefore χ1 must be VISIBLE in the CV.** A hidden-χ1 design would be *discovery*-limited
   (`T_hit ≈ T_run`), which is the R15 regime where mFR provably cannot act — the mechanism
   cannot clone a basin containing no particles. This is the single most important Stage-0
   result and it reverses the naive "hide χ1 so mFR can fix it" intuition.
3. **Consequence for `ALANINE_EXECUTION_DECISION.md` §7 gate V2.** That gate asked whether χ1
   rotamers are *distinguishable in the selected CV*. With χ1 in the CV the question is
   tautological. Its replacement is **full-state distinguishability**: does the selected 2-D CV
   separate the metastable states of the full (φ,ψ,χ1) system?
4. **The CV stays 2-D**, so `BackboneCV2D(PHI_ATOMS, CHI1_ATOMS, n_atoms=28)` drops into the
   validated alanine sampler unchanged. **Do not build a 3-D CV**: it would require `fftn`,
   trilinear interpolation, rank-6 MBAR, a 97³ grid and 24³ = 13 824 umbrella windows.
5. **Standing prediction, recorded before measurement:** (ψ,χ1) should fail the mixing gate
   because it hides φ, which carries the ~15.8 kT backbone barrier; (φ,χ1) hides ψ, which
   alanine measured at ≤0.75 kT and is therefore the safer hide.

## 1. Questions this phase must answer, in order

| # | question | answered by |
|---|---|---|
| Q1 | What are the metastable states in (φ,ψ,χ1)? | S1 state map |
| Q2 | How large are the conditional χ1 barriers? | S1 (Stage 0 gives 10–17 kT; confirm per state) |
| Q3 | Which 2-D CV best separates them: (φ,χ1), (ψ,χ1), or a backbone-path coordinate with χ1? | S2 CV selection |
| Q4 | Does ordinary ABF discover every relevant state? | S3 ABF-only screen |
| Q5 | Does any discovered state remain persistently under-established? | S3 |
| Q6 | Is that deficit visible in the selected marginal? | S3 |

**The phase ends at Q6.** An oracle-mFR comparison is launched *only* if S3 finds a
discovered-but-under-established state that is distinguishable in the selected CV.

## 2. Frozen physical model (identical to alanine, so the sampler transfers unchanged)

Ace-Val-Nme, AMBER ff14SB, vacuum, `NoCutoff`, no constraints, **no HMR**, BAOAB, **dt = 1 fs**,
γ = 1 ps⁻¹, T = 300 K, float64, IUPAC dihedrals, **odd `n_grid = 97`**. Estimator frozen at the
alanine values: `abf_bandwidth 0.08`, `kde_bandwidth 0.15`, `abf_min_count 200`,
`abf_force_clip 200` kJ/mol/rad, `project_every 50`, `estimator_stride 1`.
The run must assert its force-field parameter hash against the Val reference, as alanine did.

χ1 for valine = N–CA–CB–CG1 (confirm indices against the built topology, and verify the
prochirality convention CG1 vs CG2 before anything else — a swap silently mirrors the χ1 axis).

## 3. Stages, numeric gates, and compute

Ordering note, inherited from the concurrent session and adopted: **run the ABF discovery screen
(S3) BEFORE any umbrella reference.** S3 is ~20× cheaper and is precisely the gate that killed
alanine; there is no point paying for a 576-window reference until S3 says a deficit exists.

### S1 — state map in (φ,ψ,χ1)

Method: 3-D histogram from a long high-temperature-seeded unbiased/lightly-biased ensemble, or
2-D umbrella on (φ,χ1) with ψ free, then cluster. This is *clustering, not sampling* — the CV
used for production stays 2-D.

| gate | criterion |
|---|---|
| S1.1 | ≥ 3 distinct metastable states resolved with pairwise min-max barriers ≥ 2 kT |
| S1.2 | every state's population ≥ 0.5 % (below this it cannot be established at N ≤ 4096) |
| S1.3 | conditional χ1 barrier reported per backbone state, with a stated uncertainty |

Compute: **~1.5 GPU-h** (B = 16384, 200 ps).

### S2 — CV selection

For each candidate ξ ∈ {(φ,χ1), (ψ,χ1), (s,χ1)} where `s` is a backbone-path coordinate:

| gate | criterion |
|---|---|
| S2.1 **state separation** | every S1 state maps to a distinct, connected region of ξ; no two states overlap by > 20 % of either's area |
| S2.2 **hidden-coordinate mixing** | the *hidden* coordinate's conditional barrier at fixed ξ must be **< 3 kT** — otherwise ξ is incomplete and ABF on it will be discovery-limited, exactly the alanine failure in reverse |
| S2.3 **Gram conditioning** | `frac(cond(G) > 100) = 0`, zero ridge activations, as alanine achieved (max cond 12.1) |

Selection rule: the candidate passing S2.1–S2.3 with the **smallest** hidden-coordinate barrier.
If none passes S2.2, **stop and report that no 2-D CV is adequate** — do not fall back to 3-D.

Compute: **~1 GPU-h** (analysis of S1 output plus a short conditioning scan).

### S3 — ABF-only screen (the decisive stage)

Plain 2-D ABF on the selected ξ. **No mFR arm.** N = 4096, 4 seeds, 200 ps, C7eq-analogue init
(all walkers in the dominant state), window 20–200 ps.

| gate | criterion | rationale |
|---|---|---|
| S3.1 **discovery** | every S1 state reached by `T_hit ≤ 10 %` of run in ≥ 3/4 seeds | if not, the system is discovery-limited (R15 regime) and mFR provably cannot act — **STOP** |
| S3.2 **under-establishment** | for ≥ 1 discovered state, `T_est − T_hit ≥ 25 %` of run, **or** time-averaged occupancy ≤ 0.5 × its reference population over the window | this is the deficit mFR is supposed to repair |
| S3.3 **marginal visibility** | the deficit in S3.2 must be visible in the *selected marginal* `p(ξ)` — i.e. the under-populated region is a resolved region of ξ, not an artifact of a hidden coordinate | mFR scores only `p(ξ)`; a deficit invisible there cannot be repaired |
| S3.4 **health** | clip fraction < 1e−4, zero non-finite, ⟨T⟩ within 2 % of 300 K by the configurational thermometer | same guards as alanine |

**Decision.** Proceed to an oracle-mFR comparison **only if S3.1 AND S3.2 AND S3.3 all pass.**
- S3.1 fails → discovery-limited → stop, report as an R15-type system.
- S3.2 fails → ABF already establishes everything → stop, report as a second neutrality control.
- S3.3 fails → deficit invisible in ξ → stop, or return to S2 for a different ξ (once only).

Compute: **~2.5 GPU-h** (measured alanine rate ~50 ms/step at B = 16384, flat in batch;
200 ps = 200 k steps).

### S4 — reference FES, **conditional on S3 passing**

Only if S3 passes. 24×24 periodic umbrella on the selected ξ, cosine restraint κ = 200
kJ/mol/rad², 16 **independently thermalised** copies per window, rigid-rotation seeding with the
two-stage gate (Stage A rotation fidelity, Stage B steric relief), MBAR with Anderson
acceleration, block bootstrap over copies, plus an **independent-sampler** cross-check from a
dispersed start. Acceptance: the eight gates of `ALANINE_REFERENCE_HANDOFF.md` §4.

Compute: **~3 GPU-h** (alanine's was 2.44 h at B = 9216 for 1.12 M steps).

**Total screening budget: ~5 GPU-h to the S3 decision; ~8 GPU-h if S4 is triggered.**
All on GPU 4–7, one device, `CUDA_VISIBLE_DEVICES` pinned to an absolute allowed index.

## 4. Code changes required before S3

1. **`rare_basin` is now a parameter** of `run_sampler_ala` (fixed in this commit). It previously
   hardcoded `cur == 2`, correct only for alanine's depth ordering C7eq/C5/C7ax. Val must pass
   `basins.index["<name>"]`.
2. **A sham arm is missing.** `METHODS = ("abf", "fr_oracle")` only. If a Val mFR comparison is
   ever launched, a sham arm — birth–death with a *shuffled* or *uniform* score, matched event
   count — is needed to separate "reallocation helped" from "any resampling at this rate helped".
   Not needed for the screen; required before any positive Val claim.
3. Val basin naming in `basins.py` `NAME_HINTS` (currently alanine-specific: C7eq/C5/αR/C7ax).
   Prominence merging itself is system-agnostic and transfers.

## 5. What this phase must NOT do

* No oracle-mFR or practical-mFR run until S3.1–S3.3 all pass.
* No 3-D CV (§0.4).
* No rate/bandwidth/target tuning — those are frozen at the alanine values for comparability.
* No revisiting alanine.
* No manuscript edits.

## 6. Stop point

Deliver: the S1 state map, the S2 CV selection with its three gates evaluated, the S3 screen with
its four gates evaluated, and a one-line verdict — *proceed to mFR comparison* / *discovery-limited*
/ *second neutrality control* / *no adequate 2-D CV*. **Then stop for review.**
