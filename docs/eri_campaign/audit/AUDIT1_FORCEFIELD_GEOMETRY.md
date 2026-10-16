# AUDIT 1: force field, framework and geometry (`src/eri/model.py`, `src/eri/framework.py`, the CV)

*Independent adversarial audit, 2026-10-16, CPU only (`CUDA_VISIBLE_DEVICES=""`, `NUMBA_NUM_THREADS ≤ 2`, at most
two processes, each run under 10 min). I modified no existing file and committed nothing. No numba cache file in
the repository was written: my runs only loaded existing cache entries, and my own kernels use `cache=False`.*

* **Audited code:** commit `9792091e`.
* **Re-check:** the coordinator committed `0b15e228..ea885eb7` while I was working. I read that diff. The ERI path
  of `model.py` and `framework.py` is functionally unchanged: it adds `include_si` (true for ERI) and optional
  `P` / `x_window` keys that are absent from the ERI npz. I re-ran the neighbour-list stress test at `ea885eb7` and
  got identical numbers.
* **Scratch scripts:** `/tmp/claude-1008/-home-zheyuanlai-ABF-Fisher-Rao/d4317313-dc1c-4e55-8b95-fc34dea3264c/scratchpad/audit_ff_geom/`,
  files `a1_framework.py` through `a8_energy.py`.

## Bottom line

**I found no defect in the force field, the framework or the CV geometry that would distort any campaign
result.** Every item in my lens was checked numerically and independently of the code under audit. All of them
hold:

* the parameters;
* the CIF parsing, the symmetry expansion and the true-supercell property;
* the exactness of the voxel neighbour lists;
* forces against finite differences at extreme configurations;
* the sign of the CV mean force and the absence of a geometric term.

The strongest check is an **independent quadrature of F(φ)** for a rigid ethane: a direct sum over (y, z, 300
orientations) on planes of fixed X_com, sharing no code with the MC reference. It reproduces the Gate 1 Metropolis
reference:

* shape RMS 0.024 kJ/mol over the half period;
* barrier 33.30 kJ/mol after the flexible-bond correction, against MC 33.29;
* ΔU 5.298 against 5.30 kJ/mol;
* the 84 % entropic share.

**The φ = 0 dividing plane is clean.** It is open only at the 8-ring windows:

* All of the conditional mass p(y, z | X_com = a/4) lies within 1.25 Å of a window centre. Only 1.9e-9 lies beyond
  1.0 Å.
* Every rigid-ethane configuration on the plane whose COM is more than 1.5 Å from a window costs ≥ +305 kJ/mol
  (61 kT). The plane's minimum is −27.8 kJ/mol.
* The 6-rings that straddle φ = 0 (CAN ↔ ERI cage of the adjacent row) cost ≥ 11 666 kJ/mol for a single bead.

So no between-row pathway exists other than through a φ = 0 8-ring window.

What remains are five low or info items: one small precision and symmetry imperfection in the Gard & Tait build,
and four documentation or robustness points.

## 1. Force-field parameters vs `review/LIT_CROSSCHECK.md` §2

| item | master table | `model.PARAMS` → packed | status |
|---|---|---|---|
| ε(O–CH3) | 142 K | 142 × 0.008314462618 = 1.18065 kJ/mol; 4ε = 4.72261 | OK |
| σ(O–CH3) | 3.17 Å | σ² = 10.0489 | OK |
| ε(Si–CH3) | 82 K | 0.68179 kJ/mol; 4ε = 2.72714 | OK |
| σ(Si–CH3) | 2.12 Å | σ² = 4.4944 | OK |
| ε(CH3–CH3) | 104 K | 0.86470 kJ/mol; 4ε = 3.45882 | OK |
| σ(CH3–CH3) | 3.78 Å | σ² = 14.2884 | OK |
| pairs | O + Si host–guest; CH3–CH3 between different molecules only; no intramolecular LJ; no charges | exactly these loops (`config_energy_force`) | OK |
| truncation | energy-shifted at 12 Å, truncated force, no tail | V(rc) = −1.604e-3 / −8.29e-5 / −3.376e-3 kJ/mol subtracted per pair within rc (`r2 < rc2`) | OK |
| bond | ½ k (r − r0)², k/kB = 96 500 K/Å², r0 = 1.54 Å | k = 802.3456 kJ/mol/Å² | OK |
| T | 600 K | kT = 4.98868 kJ/mol | OK |

The campaign uses the K values that PNAS quotes, not Schüring's kJ/mol originals (141.92 / 82.15 / 104.16 K), as
the labelled surrogate requires. The potential is continuous at rc. The force jump there is below 1e-3
kJ/mol/Å, so overdamped Langevin and Metropolis MC target the same law.

## 2. Framework (`framework.py`)

* **Symmetry operators.** I checked the 24 parsed operators of each CIF myself:
  * they are distinct;
  * they are closed under composition;
  * they preserve the hexagonal metric;
  * Gard & Tait (RASPA2) and COD 2106432 give identical operator sets.

  The whitelist parser evaluates nothing.
* **Majority sites, Schlenker.** The parser merges Si+Al per position and keeps a position at occupancy ≥ 0.5:
  * kept: T1 1.000, T2 0.941, O1 0.942, O6 0.95, O8 0.95, O3/O4/O5 1.0;
  * dropped: the offretite-fault T3 (0.059) and O2/O7/O9 (0.058/0.05/0.05);
  * cations are dropped.

  This is correct.
* **Orbit multiplicities.** Both variants give 24 + 12 T and 24 + 12 + 12 + 12 + 6 + 6 O, so 36 Si and 72 O per
  cell. For Gard & Tait, DEDUP_TOL = 2e-3 merges two near-special orbits: O5, and also O2 (x, 2x, z) with 2x off by
  1e-4, which LIT_CROSSCHECK did not mention. The merged pairs are only 1.3 mÅ apart, and the tolerance merges
  nothing else. The closest distinct atoms are 1.62 Å apart.
* **Supercell.** My independent rebuild maps one-to-one onto the npz:
  * max distance 6.6e-4 Å (Gard & Tait; finding F1) and 9e-16 Å (Schlenker);
  * box 3a × 2√3a × 3c, with X along a1 and Y along a1 + 2a2 (a lattice vector);
  * translations by a1, a2, a1 + a2 and c map the box crystal onto itself to 1.6e-14 Å;
  * all 24 space-group operations do so to 2.4e-14 Å (Schlenker) and 1.3e-3 Å (Gard & Tait; F1).

  It is a true lattice supercell.
* **Special sites.** A single-bead scan along the B axis (1/3, 2/3, z) confirms the ERI cage at z ∈ (0.35, 1.15):
  * U is −9.7 kJ/mol at the centre z = 3/4 and −15.6 near the window heights;
  * the axial 6-ring is at z = 1/4, at +342 kJ/mol (Gard & Tait).

  The A column (CAN/D6R) is closed for ethane.

## 3. Exactness of the unit-cell voxel neighbour lists

I tested 820 bead positions per variant:

* uniform in the box;
* at voxel corners and faces offset by ±1e-15 … 1e-6 Å, in unit-cell images −3 to +3;
* exactly on cell or box faces (0, cell, box, −box, −1e-17, −1e-300);
* unwrapped by up to 10⁶ boxes.

Results:

| comparison | worst relative error |
|---|---|
| `bead_fw` vs an explicit crystal-image sum (images −1..1 of the box, no minimum image) | 1.6e-13 (Gard & Tait), 1.8e-13 (Schlenker) |
| `bead_fw` vs a direct minimum-image numpy sum (energy, force) | 2.8e-8 / 4.1e-8. This is my reference's own round-off at 10⁶-box unwrapping. |
| neighbour-list completeness at 3200 voxel corners (8 corners × 200 voxels × 2 types): atoms within rc of the corner vs atoms listed | 0 mismatches |
| unwrapping by 10² / 10⁴ / 10⁶ / 10⁸ boxes | ΔE 8e-13 / 7e-11 / 3e-8 / 1e-6 kJ/mol |
| forces vs central finite differences at close contacts (bead 2.2–2.9 Å from an O, 1.6–2.5 Å from a Si; \|F\| up to 2e11) | 2.7e-9 |

The exactness argument (rc + half-diagonal, unique images because box_min = 39.8 > 2(rc + hd) = 25.7 Å) is
correct, and no counterexample exists in practice.

## 4. CV geometry

* **Where things sit in φ.** φ = wrap(2π(X − a/4)/(a/2)):
  * the B and C cages of each row are at X = n·a/2, so φ = ±π;
  * the (1/2, 1/2, z) and (0, 1/2, z) windows are at X = a/4 mod a/2, so φ = 0. That is 144 of the 216 windows.
  * The (1/2, 0, z) family (72 windows) is at φ = ±π. These are within-row windows (F2).
* **Symmetry.** The mirror X → −X (the (x, 2x, z) mirror plane) gives F(φ) = F(−φ). The reference is symmetric to
  0.009 kJ/mol RMS, which is its noise.
* **Mean force.** ∇φ = (π/P) e_x on each bead, so ∇φ/|∇φ|² is constant and has no divergence term.
  f = −(P/2π)(F₀ₓ + F₁ₓ) = ∂U/∂φ along a rigid translation:
  * the Gate 0 test checks this;
  * Gate 1 check 2 (F_density = F_MF, D = 0.000) confirms it;
  * so does my quadrature.

  The ABF bias, +Γ·(π/P) on each bead's x component, has the correct sign and needs no |∇φ|² normalisation.
* **Dividing surface φ = 0.** I integrated over the plane X = a/4 with h = 0.1 Å and 300 orientations:

  | | Gard & Tait | Schlenker |
  |---|---|---|
  | conditional mass beyond 0.75 / 1.0 / 1.25 Å of a window centre | 6.1e-5 / 1.9e-9 / 1.6e-17 | 7.1e-5 / 3.2e-10 / 8.9e-17 |
  | RMS COM offset in the plane, (y, z) | (0.15, 0.20) Å | (0.17, 0.19) Å |
  | min U rigid ethane on the plane | −27.8 kJ/mol | −30.0 |
  | min U rigid ethane, COM > 1.5 / > 2.0 Å from every window | +305 / +1441 | +327 / +1229 |
  | min U single bead on the plane, > 1.5 / > 2.0 Å from every window | +308 / +721 | +327 / +793 |

  The plane cuts no cage interior: the ERI cage cross-section never reaches X = a/4 except through the window. I
  also checked the alternative routes:
  * The lateral 6-rings between a CAN (row n, X = 0) and the ERI cages of rows n ± 1 do straddle φ = 0. Their
    midpoints are at φ = 0.000. The straight-line maximum is 11 666 kJ/mol (Gard & Tait) and 30 490 (Schlenker)
    for a single bead.
  * The CAN and D6R are closed for ethane.
  * A single overdamped step cannot tunnel through a wall: the Gate 2 maximum step component at dt 2e-4 is 0.30 Å,
    and the wall is several Å of +10³ kJ/mol.

  **No physically accessible inter-row path avoids a φ = 0 8-ring window.**
* **Region thresholds** (A1.3), from the reference:
  * at |φ| = 2.007, F − F_min = 0.442 kBT (my quadrature gives 0.437);
  * in W = {|φ| < 0.5}, F ≥ 6.32 kBT.

## 5. Independent F(φ) by quadrature vs the Gate 1 MC reference (Gard & Tait)

I used the rigid ethane, 46 planes over φ ∈ [−π, 0], a (y, z) grid tiling the period exactly, and 300
orientations. The converged values come from window patches at h ≤ 0.05 and the row plane at h 0.05–0.1, with
orientations 300 vs 800 identical to 1e-4.

| quantity | quadrature | Gate 1 MC |
|---|---|---|
| F(0) − F(−π), rigid bond | 33.373 kJ/mol | – |
| flexible harmonic bond correction (5-point Gauss–Hermite in r, r² Jacobian) | −0.071 kJ/mol | – |
| barrier, flexible | **33.30 kJ/mol** | 33.29 (max − min), 33.28 (`barrier_stats`) |
| shape RMS over 45 bins (rigid; the residual grows toward the window, as the bond correction predicts) | 0.024 kJ/mol (max 0.058) | – |
| ΔU window − cage row | 5.298 kJ/mol | 5.30 |
| entropic share −TΔS/ΔF | 0.841 | 0.84 |

The reference's coarse-bin reweighting of the joint bead histograms (`bead_profile`, used for the framework
selection) is accurate: the φ-marginal it implies differs from the fine WHAM marginal by 0.0006 kBT RMS. The |s|
barriers reproduce at 5.962 (Gard & Tait) and 5.288 (Schlenker).

## 6. Findings

| id | sev | where | what |
|---|---|---|---|
| F1 | low | `framework.py` `majority_framework_sites` (l. 208–212 at HEAD), `expand_cell` (l. 227) | Positions are stored ROUNDED to 4 decimals (Gard & Tait Si1 0.00021 → 0.0002, 0.10446 → 0.1045). Near-coincident orbit images (O2, O5) keep the first image instead of being symmetrised. Result: the Gard & Tait crystal deviates from exact P6₃/mmc by up to 1.33e-3 Å. Effect on F is negligible: atoms move ≤ 1.3 mÅ, and quadrature and MC agree to 0.01 kJ/mol. The repository framework is therefore not bit-faithful to the CIF, and the X → −X mirror holds only to 1e-3 Å. Fix: round only the grouping key and store the CIF coordinates; average merged images, or project them onto the special position. |
| F2 | info | `model.py` docstring l. 17–21; PREREGISTRATION §1, A1.3 | "Cage rows at ±π" omits that the within-row (1/2, 0, z) windows (72 of 216) also sit at φ = ±π, inside the cage-core set C. The within-row hop is a hidden barrier orthogonal to φ, of about the same height. I checked that it is harmless: the B and C cages of a row are related by the x-preserving 2-fold axis along a1, so f and the conditional laws are identical. Each B–C pair in a row is joined by one within-row window and separated from the next pair by a closed A column (zero ethane density there). Fix: say so in the model description. |
| F3 | info | PREREGISTRATION A1.3 l. 146 | "Cage core C = {\|φ\| ≥ 2.0}, where F is within 0.4 kBT of its minimum". The reference gives 0.442 kBT at \|φ\| = 2.007, and my quadrature gives 0.437. The text only describes the region: the region is set by the φ threshold, so nothing changes. Fix: write "within 0.45 kBT". |
| F4 | info | `model.py` l. 26–27 vs l. 156 | The docstring says "fastmath 'reassoc' only", but `_type_sum` is compiled with `{"reassoc", "contract"}` (FMA contraction). Reproducibility is bitwise only per compile target, i.e. per CPU ISA and vector width. The abf/fr pairing claim therefore needs both arms on the same CPU model. That holds on the single host (AMD EPYC 9554), but the CPU is not recorded. Fix: correct the docstring, record the CPU model in the run metadata, or drop `contract`. |
| F5 | info | `reference.py` `window_stacks` (l. 52); new `framework.build_orthorhombic` (l. 165) | `window_stacks` hard-codes the (3, 2) supercell as `fw.box[:2] / [3, 2]` instead of using `fw.cell`. ERI is now guarded by variant name. The new generic `build_orthorhombic` (overnight extension, unused by ERI) is less safe than the ERI builder: it has no coordination check and no cross-orbit duplicate check; an odd `perm` builds the mirror-image framework; and `save()`/`special_sites()` are hexagonal-only, so they would write meaningless "cages"/"windows" for an orthorhombic framework. Fix: use `fw.cell`; add the coordination and duplicate checks, an even-permutation assert and a framework-specific `special_sites` before the extension is used. |
