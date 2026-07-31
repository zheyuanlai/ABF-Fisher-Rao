# ALANINE EXECUTION DECISION — repository truth, change classification, and the next experiment

Companion to `ALANINE_SPEC.md`. Where the two disagree about **repository state**, this file wins;
where they disagree about **design**, the spec wins. Written 2026-07-31, branch `alanine-dipeptide`.

No production simulation has been run. Nothing under `results/` has been overwritten.

---

## 1. Repository truth

`ALANINE_SPEC.md` §0 states "DESIGN ONLY, FROZEN … no file under `src/`, `scripts/`, `configs/` or
`results/` has been created or modified", and its gate table marks V8 (Nyquist) and V15 (seed strain)
as "FAILS TODAY". Those statements were true **when the spec was written** and became stale within the
same session, when the Nyquist fix was committed. This is a genuine documentation defect; the table
below is the authority.

| Item | Spec/claim | Actual state | Verified by |
|---|---|---|---|
| Branch / HEAD | — | `alanine-dipeptide` @ `c6a6718` (parent `b84de17`) | `git rev-parse` |
| `src/alkanes/poisson2d.py` | "not modified" | **MODIFIED** (+7 lines, Nyquist zeroing) | `git diff b84de17..c6a6718` |
| Nyquist regression test | "FAILS TODAY" | **EXISTS AND PASSES** — `test_returned_gB_is_the_gradient_of_returned_B_on_random_input` (n ∈ {35,36,48,63,64,96,97}) and `test_nyquist_mode_is_not_applied_as_a_force` | fresh `pytest` |
| Alanine source code | "design only" | **TRUE as of `c6a6718`**; this commit adds `src/alanine/{__init__,system,forcefield}.py` (Stage-0 only: build, seed, parity — **no sampler, no driver**) | `ls src/alanine` |
| `configs/alanine`, `results/alanine`, `scripts/*alanine*` | — | **do not exist** | `ls` |
| Test suite | "78 passed" | **88 passed, 3 skipped** after this commit (78 + 10 new alanine Stage-0) | fresh `pytest` |
| Umbrella seeding | "blocking" | **WAS blocking; now FIXED and gated** — rigid rotation passes 576/576, NeRF rebuild passes 0/576 | `tests/test_alanine_stage0.py` |
| Pentane 2-D affected? | "unknown" | **NO — negligible**, see §3 | `results/poisson_nyquist_audit/` |

**Correction to the spec text.** Two claims in `ALANINE_SPEC.md` are now wrong and are superseded here:
V8 passes, and V15 passes. The spec's *design* content is unaffected.

---

## 2. Change classification

The spec bundles correctness fixes, diagnostics and algorithmic redesign together. Bundling them would
make any future positive result uninterpretable — a gain could come from a new ABF estimator rather
than from Fisher–Rao selection. They are separated here and **only A and B may enter the baseline**.

### A. Correctness fixes — implementation must match the intended method

| # | fix | status |
|---|---|---|
| A1 | Nyquist-safe Poisson projection (or odd grid) | **DONE** `c6a6718` |
| A2 | assert `gB == spectral_gradient(B)` after every projection | **DONE** as a test; add a runtime assert in the sampler |
| A3 | IUPAC dihedral convention for peptides (`cv2d` inherits `'rb'`) | pending — sampler not yet written |
| A4 | BAOAB clone copies the full dynamical state `(q, v, F_cached)`, child gets fresh Maxwell momenta | pending — blocking for any sampler |
| A5 | independent fixed-consumption RNG stream per seed | pending — **breaking**, see §2 note |
| A6 | reject non-finite mean-force samples immediately (not `clamp`, which passes NaN) | pending |
| A7 | online and frozen-bias runs apply the identical gradient field | partly — A1 removes the main source |
| A8 | rigid-rotation umbrella seeding + V15 gate | **DONE**, this commit |

*Note on A5:* changing the birth–death RNG consumption pattern invalidates every existing FR baseline
(they would have to be re-run to be comparable). Do it deliberately, in its own commit, and record it.

### B. Diagnostics — do not alter the dynamics, all mandatory

B1 frozen-bias validation · B2 ancestor ESS, max-ancestor fraction, rare-basin genealogy ·
B3 per-family / split-half force accumulators · B4 kernel-matched FES error ·
B5 equal-wall-clock **and** equal-force-evaluation cost reporting · B6 explicit first-hit and
establishment times. All pending; none implemented yet.

### C. Algorithmic changes — **separate ablation arms, never silent defaults**

Exponential forgetting in the ABF accumulator · weighted Poisson projection · clone-discounted
effective counts · `abf_bandwidth` 0.20 → 0.08 · `fr_every` 5 → 500 · lagged/tempered FR targets.

These change the estimator or the dynamics. The spec proposes several of them as if they were fixes;
they are not. Minimum arm set to retain:

> **corrected ABF** · **corrected original mFR-ABF** · **mFR-ABF-v2** (Category C bundled)

Without the middle arm, a gain cannot be attributed to Fisher–Rao selection rather than to a better
ABF estimator.

---

## 3. Pentane 2-D Nyquist impact — **NEGLIGIBLE, retain existing conclusions**

Artifacts: `results/poisson_nyquist_audit/{impact.csv, impact_verdict.txt, summary.csv,
field_diff_diagnostic.md}`. Scripts: `scripts/audit_poisson_nyquist_impact.py` (decisive),
`scripts/audit_poisson_nyquist.py` (field-level proxy diagnostic).

**The defect is real and the fix is necessary.** On an unsmoothed random field the returned `gB`
differs from `∇B` by ~12 % relative with `curl_norm(gB) = 1.71` at n = 48. Verified that the *minimal*
"zero only the self-conjugate modes" variant does **not** work (3.7e-1 residual): `fftfreq` assigns
`k = −n/2` at index `n/2` and the conjugate partner shares that index, so `i k` is not antisymmetric
there (measured Hermitian defect **81.3**). The Nyquist row has no representable derivative on the
grid, so zeroing the whole row is the correct and standard remedy.

**But its effect on the completed runs is nil**, because the production smoothing (`h = 0.20` rad on a
48-grid) suppresses `k = 24` by `exp(−k²h²/2) ≈ 1e-5`. Measured on the real saved potentials:

| quantity | measured |
|---|---|
| Nyquist power fraction of saved `final_pmf` | **1.9e-12 … 4.1e-07** |
| worst change in reported L2 | **0.0001 %** |
| ABF-vs-mFR ranking shift | **0.000 pp**, all signs preserved |

Per-arm, production stage: abf 5.1430 % → 5.1430 %, fr_estimated 5.1795 % → 5.1795 %,
fr_active 5.3654 % → 5.3654 %, opes 14.1570 % → 14.1570 %.

**Decision: document the bug, retain the pentane 2-D conclusions, no reruns.** Frozen-bias validation
also stands (`B` is unchanged to 1e-7 in power, so the re-differentiated frozen field matches the
online one). The fix remains mandatory going forward: the alanine spec's `h = 0.08` on a finer grid is
precisely the regime where the defect would bite.

**Honest caveat.** `core2d` does not persist `f1s/f2s/csum` or `g1f/g2f`, so the exact historical
applied field cannot be reconstructed. The decisive test above uses the real saved potentials, which is
what every reported number is computed from. The companion `field_diff_diagnostic.md` perturbs a
*reconstructed proxy* field and therefore overstates the effect (3.9 %); it is a diagnostic, not the
verdict, and an earlier claim that `B` was "unchanged exactly" is withdrawn — it is unchanged to
1e-7 in power, not exactly.

---

## 4. Physical model — contradiction resolved

The spec carries three incompatible settings (no-HMR/1 fs; HMR=3/2 fs in the kill-shot; HMR=4/2 fs in
the reference pilots). **For the first corrected alanine study, freeze the conservative option:**

```
Ace-Ala-Nme, vacuum, AMBER ff14SB, no constraints, NO HMR
BAOAB, dt = 1 fs, gamma = 1 ps^-1, T = 300 K, float64
CV = (phi, psi), IUPAC convention, atoms (4,6,8,14) and (6,8,14,16)
n_grid = 97  (ODD -- no Nyquist row exists, so A1 is belt-and-braces)
```

Rationale: it is the simplest physical model; it removes a mass-model variable; the spec's own
configurational-temperature check saw no bias at 1 fs but did at 2 fs; and reference and production can
then share an identical integrator, mass set, timestep and dtype — which is the only way to make the
`O((ω dt)²)` discretisation bias common-mode. HMR is a **later speed optimisation**, to be introduced
after a controlled `dt`-bias study, never mixed into the first scientific conclusion.

---

## 5. Umbrella initialisation — fixed and gated

`build_positions` (whole-molecule NeRF rebuild) is **removed from the seeding path** and retained only
to construct the single reference minimum. Every window is now reached by **rigid rotation about the
phi and psi bonds** from one minimised L-alanine C7eq structure, which preserves bond lengths, angles,
planarity and chirality by construction.

Gate V15, enforced in `tests/test_alanine_stage0.py`, requires per window: L chirality; harmonic-angle
energy < 50 kJ/mol; max bond-angle deviation < 15°; recovered (φ,ψ) within 1° of the requested centre.

| seeding path | 18×18 = 324 | 24×24 = 576 | median angle energy | max angle deviation |
|---|---|---|---|---|
| **rigid rotation (adopted)** | **324/324 = 100 %** | **576/576 = 100 %** | 5.5 kJ/mol | 3.57° |
| NeRF rebuild (rejected) | 0/324 = 0 % | 0/576 = 0 % | 2655.9 kJ/mol | 114.10° |

Reference minimum: φ = −74.95°, ψ = +51.50°, chirality volume +0.002 nm³ (> 0 ⇒ L). Parameter-set
content hash `6ffd00dc241f`. Forces present: HarmonicBond, PeriodicTorsion, Nonbonded, HarmonicAngle.

`test_nerf_builder_is_unfit_for_seeding` pins the rejected path so it cannot be silently reintroduced.

---

## 6. The next single experiment

> **Corrected alanine ABF versus oracle mFR — not Val production.**

Rationale: this tests whether the 2-D atomistic implementation and the Fisher–Rao mechanism work at
all, before introducing another molecule and another hidden coordinate.

**Arms (two only).** Corrected multi-replica 2-D ABF; corrected 2-D **oracle** mFR-ABF.
Do **not** tune the practical EMA target unless the oracle arm improves on ABF.

**Replicas.** `N = 2048` and `N = 4096` pilots, to diagnose 2-D KDE noise. The spec's occupancy
argument puts the requirement near `N ≈ 3500` for a log-density standard deviation of ~0.2 at
bandwidth 0.15, so 4096 is the natural main setting.

**Initialisation.** All-dominant-basin (100 % C7eq) as **primary**; reference-equilibrium as a crossed
control. The 95 %/5 % hand-seeded rare-basin protocol is **not** the headline — it hand-places
population in the basin whose establishment is the thing being measured.

**Measurement window: 20–200 ps** (transient), not a single long endpoint. Reporting a final-L2
comparison at 1 ns is guaranteed null by construction, because ABF has reached its estimator floor.

**Required outputs.** Kernel-matched integrated FES error; mean-force vector error; first-hit and
establishment times for reference-defined basins; frozen-bias reconstruction; ancestor ESS and
max-ancestor fraction; measured wall-clock and force-evaluation counts.

### Go / no-go

> **STOP** unless oracle mFR improves **both** `∫₂₀^₂₀₀ e_F(t) dt` **and** `e_∇F(200 ps)`, with
> ancestor ESS ≥ 0.30 N and no single ancestor above 5 % in the rare basin.

On a null, report alanine as: *a two-dimensional atomistic neutrality control in which ABF already
resolves the relevant backbone coordinates, leaving little room for marginal reallocation.* Do **not**
then spend budget tuning EMA, bandwidth or FR rate — the repo already holds that negative.

---

## 7. Val/Ile/Leu — screening required before any implementation

**The spec's pivot argument is incomplete and is corrected here.** It argues that Val has a slow hidden
χ1, therefore mFR acting on (φ,ψ) should help. **That does not follow.** The correction sees only
`p_t(φ,ψ)`. If two χ1 rotamers occupy the same (φ,ψ) region then

```
S(φ, ψ, χ1⁽¹⁾) ≈ S(φ, ψ, χ1⁽²⁾)
```

and the mechanism cannot repair a conditional mixing failure in `p(χ1 | φ, ψ)` that is invisible to the
selection coordinate. This is exactly the limitation that makes 1-D φ-mFR useless for alanine when two
states differ only in ψ. **A hidden coordinate helps only if it induces observable starvation in the
selected marginal.**

Screen before implementing, in order:

| gate | requirement |
|---|---|
| **V1** genuine hidden barrier | `F(χ1 \| φ,ψ)` barrier is several kT under *our* force field |
| **V2** distinguishability in the selected CV | χ1 rotamers occupy meaningfully different (φ,ψ) regions — quantify by `I(χ1; φ,ψ)` or by rotamer classification accuracy from (φ,ψ) alone. **If rotamers overlap strongly in (φ,ψ), marginal mFR cannot see the problem and the system is not a candidate.** |
| **V3** discovered but under-established | under plain 2-D ABF, `T_hit(rare) ≪ T_run` while `T_est − T_hit` is large — the reallocation-limited regime |
| **V4** oracle improvement | oracle mFR reduces the post-discovery establishment delay *and* improves FES or mean-force estimation |

Start with **Ace-Val-Nme** only: one principal side-chain dihedral, smallest code and interpretation
change, cleanest hidden-coordinate experiment. Ile and Leu are follow-up generalisation systems, not
parallel first attempts.

---

## 8. Execution order

```
1. Verify repository truth                                    DONE (§1)
2. Implement correctness fixes and diagnostics only           A1, A2, A8 done; A3-A7, B1-B6 pending
3. Audit material impact on pentane 2-D                       DONE — negligible (§3)
4. Rebuild the corrected alanine reference                    seeding fixed and gated (§5); reference not yet run
5. Run alanine ABF vs oracle mFR                              NEXT (§6)
6. Close alanine as positive or neutral
7. Screen Ace-Val-Nme for an mFR-repairable deficit           gated on §7 V1-V4
8. Run practical mFR only after oracle success
```

The immediate coding task is **not** "implement the 1945-line spec". It is steps 2 and 4: the remaining
Category-A fixes (A3–A7), the Category-B diagnostics, and the corrected reference run.
