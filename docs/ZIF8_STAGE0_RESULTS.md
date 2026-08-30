# Ethane/ZIF-8 — Stage 0 physical-model validation

**All Stage-0 numbers below were produced before any ABF-bias or Fisher–Rao run
on this system.** Preregistration: `configs/uniform_campaign/zif8_prereg.json`.
Raw outputs: `results/uniform_campaign/zif8/stage0/`.

## The model, and why each piece is trustworthy

| Piece | Source | How it is checked |
|---|---|---|
| Structure | Park et al. PNAS 2006 (FAWCEN) P1 cell, 276 atoms, a = 16.991 Å, via numat/RASPA2 | composition Zn₁₂N₄₈C₉₆H₁₂₀, coordination (Zn–4N, N–3, C1–3, C2–3, C3–4, H–1), charge neutrality to 1e-17 e |
| Force field | **Krokidas et al. JPCC 2015 — the force field of the 2024 anchor paper**, via the Anikeenko 2017 GROMACS adaptation | every parameter read from the `.itp`, none typed from memory |
| Topology | enumerated from connectivity by `scripts/build_zif8_framework.py` | **set-equality against the published 2×2×2 GROMACS topology: 2496 bonds, 4608 angles, 7296 propers, 6336 1-4 pairs — all four identical**, and every published term resolves to the same parameter type |

Three bugs the validation caught, each of which would have produced
plausible-looking wrong numbers:

1. **1-4 pair rule.** Naively "any 4-atom path" over-counted by exactly 120
   pairs per cell — the imidazolate 5-ring pairs that are *also* 1-2/1-3 by the
   other path. The shorter path wins. Found by set comparison, not by inspection.
2. **The CV must not be minimum-imaged.** The confinement tube is longer than
   the minimum-image cube, so a min-imaged ξ is wrong by up to **4.90 Å** near
   the tube ends. The CV is evaluated on unwrapped guest coordinates.
3. **Gate-observable min-image knife edge.** Opposite gate Zn are separated by
   exactly *a*/2 in each Cartesian component. Referencing the ring centroid to a
   *ring atom* put every other Zn exactly on the wrap boundary; a 0.1 Å thermal
   jitter flipped it and **A_gate jumped 2.85 → 5.03 Å**. It is now referenced to
   the fixed window centre. Regression-tested for continuity under jitter.

## 0A — equilibrium lattice constant (barostat-free)

The engine is NVT, so instead of a barostat the lattice constant is scanned and
⟨P⟩ measured from the **atomic virial** (affine-scaling finite difference of U,
exact for any potential). This matters: the anchor paper's SI shows an arbitrary
NVT cell can move the ethane barrier by tens of kJ/mol.

| a (Å) | 16.30 | 16.50 | 16.70 | 16.85 | 16.991 | 17.15 |
|---|---|---|---|---|---|---|
| ⟨P⟩ (bar) | +3129 | +100 | −2826 | −4834 | −6410 | −8244 |

**⟨P⟩ = 1 bar at a = 16.507 Å.** The force field wants a cell 2.8 % smaller than
the X-ray one — and this is corroborated externally: the anchor paper's own
force-field supercell is 33.19 Å, i.e. **a = 16.595 Å**, obtained with a real
barostat. Agreement **0.5 %**, via a completely independent route. Production
uses a = 16.5068 Å; the lattice was *not* tuned to reproduce any literature
barrier.

## 0B — flexible-gate sanity and framework stability (all gates PASS)

At a = 16.5068 Å, 300 K, 128 replicas, 50 ps:

- **Gate aperture A_gate = 2.796 ± 0.058 Å** (range 2.604–3.028), i.e. a free
  diameter of 2(2.796 − 1.10) = **3.39 Å against the literature 3.4 Å**. The gate
  is genuinely flexible — a frozen aperture would have meant the implementation
  was wrong for this study's purpose.
- Linker tilt θ_gate = 35.41 ± 1.59°.
- Framework stability: **Zn/N/C skeleton RMSD 0.263 Å with no drift**
  (first half 0.2615, second half 0.2630). The all-atom RMSD of 0.771 Å is
  *inflated by the free methyl rotors* — every H3–C3–C1–N torsion has k = 0 in
  this force field, so methyl H wander ~1.8 Å from any reference orientation no
  matter how stable the framework is. Measuring stability on all atoms would
  have reported rotor freedom as instability (and did, until it was fixed).
- T_kin 298.2 K; ⟨P⟩ −116 bar (≈ 0 as designed).
- **Gate autocorrelation τ[A_gate] = 49.6 fs, τ[θ_gate] = 44.7 fs** (fine
  stride 5 fs). In the *empty* framework the gate is a fast, small-amplitude
  coordinate. What the study actually needs is the gate distribution
  *conditioned on the guest being at the window* — the induced-fit
  configuration — which is what the umbrella reference and J_gate(t) measure.

## 0D — initial-condition pool

1024 configurations, one ethane inserted per cage centre, minimized then 30 ps
NVT. Guest ξ spans −11.20 to −1.66 Å around cage A at −7.15 Å; maximum radial
distance from the tube axis 4.74 Å against R_tube = 4.5 Å (soft wall,
≈1.2 kT there). The pool is shared by **every arm and the reference**, so the
two production arms start from the same distribution by construction.

## Geometry facts frozen into `cache/zif8/framework.npz`

- Cage–cage distance L = **14.7146 Å** along [111], equal to a√3/2 to four
  decimals: the vector is a **lattice translation**, so ξ is *exactly* periodic
  and the channel needs no axial walls. Both cages refine to the same free
  radius (7.001 Å) so ξ_A/ξ_B are symmetric.
- The 6-ring alternates: **three linkers present their ring C–H edge** to the
  window (those six H at radius 2.853 Å are the crystallographic bottleneck) and
  **three present their methyl** (methyl C at 4.448 Å). A_gate uses the six
  bottleneck H.
- Confinement tube R = 4.5 Å captures **98.9 %** of the guest-accessible cage
  volume; every other window out of the cage sits at radius 6.937 Å, i.e.
  2.4 Å outside the tube.
- Body-centring is exact for every atom type **except H3** (max 0.79 Å), because
  the CIF picks one ordered methyl rotamer per linker. The methyl is a free
  rotor, so F(φ) is exactly periodic by symmetry; only the *instantaneous*
  potential is not, by ≈0.001 kT (measured).

## Engine

`src/zif8/core_zif8.py`, 21 passing tests in `tests/test_zif8.py` covering
finite-difference forces, the LJ/DSF two-body form against a hand formula, the
½k bond and wall conventions isolated exactly, dihedral geometry, translation
and wrap invariances, framework-COM pinning, CV periodicity and gradient, the
bias force equalling ∇φ, equipartition, `fr_rate = 0` reproducing ABF
bit-for-bit, cloning invariants, the no-reference-leakage guard, circular ABF
inversion, circular WHAM against a known profile, and the two real-artifact
gates (crystal skeleton at the FF minimum; gate observable continuous).

Throughput on the H200 (f64, chunk 1024): **5551 ns/day aggregate**, 0.75 GiB
peak; saturated for B ≥ 1024, so one process per GPU is the right shape.
