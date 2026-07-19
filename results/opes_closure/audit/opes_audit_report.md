# OPES_METAD Audit Report (Part A) — closure v1

**Code audited:** `src/opes_core.py` (md5 `553e03ac3855`), driven as in `src/opes_jobs.py`.
**Suite:** `scripts/audit_opes.py` — 14 tests, each with a numeric tolerance that fails loudly.
**Result:** **14/14 PASS** on CPU in ~52 s. Machine JSON: `opes_audit_summary.json`.

Reproduce:
```bash
conda activate abffr
CUDA_VISIBLE_DEVICES="" python -u scripts/audit_opes.py            # CPU, ~1 min
python -u scripts/audit_opes.py --device cuda                     # optional GPU
```

## 1. Implemented equations (as verified, not as documented)

OPES_METAD on a 1-D CV grid, weighted-KDE form (Invernizzi & Parrinello, JPCL 2020):

- Online reweighted CV marginal: `p̃ₙ(z) = Σ_k w_k K_σ(z−z_k) / Σ_k w_k`, `w_k = exp(β·A_{n−1}(z_k))`.
- Normalized marginal `ρ(z)` (∫ρ=1); dimensionless `u = ρ·W`, `W = z_max−z_min`.
- Applied bias potential **`A_n(z) = (1−1/γ)·β⁻¹·log(u + ε)`**, anchored `A ← A − max A`.
- Regularization floor **`ε = exp(−β·BARRIER/(1−1/γ))`** (PLUMED `BARRIER` semantics).
- Applied biasing force **`−A′_n(z)`**, clipped to `±bias_force_clip`.
- Native FES: **`F̂(z) = −β⁻¹ log ρ(z)`**. Common estimator: mean-force reconstruction
  from OPES-biased samples (the bias depends only on z ⇒ conditional force unbiased).

## 2. BARRIER → γ mapping (specifically re-derived, not assumed)

`OPESConfig.effective_gamma()` with `gamma_from_barrier=True` returns **`γ = β·BARRIER`**,
which is PLUMED's documented default when `BIASFACTOR` is unset (γ = BARRIER/kT). Verified
numerically per system:

| system | β | BARRIER | γ = β·BARRIER | prefactor 1−1/γ | ε = exp(−β·BARRIER/pref) |
| --- | --- | --- | --- | --- | --- |
| WCA | 1 | 4 | 4.00 | 0.7500 | 4.83e−03 |
| metastability | 4 | 4 | 16.00 | 0.9375 | 3.87e−08 |
| entropic | 8 | 4 | 32.00 | 0.9688 | 4.51e−15 |

Flat ablation (`gamma_from_barrier=False`, `gamma=inf`) → prefactor 1 (verified T3).

**Methodological note (carried into tuning):** with `gamma_from_barrier=True` the single
`barrier` knob simultaneously sets γ (well-tempering aggressiveness) *and* ε (the bias cap).
At β=8 this yields γ=32 (near-flat WT) and ε≈0 (no cap). The closure tuning therefore treats
BARRIER as a first-class swept axis and additionally reports a fixed-γ control so the two
effects can be separated (see `configs/opes_closure/`).

## 3. The 14 tests (Gate 1 correctness + Gate 2 multi-walker normalization)

| # | test | what it proves | tol | value | verdict |
| --- | --- | --- | --- | --- | --- |
| T1 | exact_density_bias | injected ρ ⇒ code bias = analytic `(1−1/γ)β⁻¹log(uW+ε)` | 1e−4 | 2.5e−7 | PASS |
| T2 | wt_target | biased eq. `p_b ∝ P·e^{−βA} = P^{1/γ}` (WT target), not Boltzmann | 2e−2 | 0.014 | PASS |
| T3 | flat_limit_epsfloor_cap | γ=∞ ⇒ uniform; residual is the intended ε=exp(−β·BARRIER) cap (0.069→0.000 as BARRIER 6→12) | 2e−2 | 2e−4 | PASS |
| T4 | force_derivative | applied force = −dA/dz exactly (central diff) | 1e−6 | 0.0 | PASS |
| T5 | sign | bias force points OUTWARD from a density peak ⇒ flattens marginal (peak 4.99→2.23) | — | — | PASS |
| T6 | density_normalization | ∫ρ dz = 1 after a real deposit | 1e−3 | 0.0 | PASS |
| T7 | weight_scale_invariance | normalized ρ, bias invariant to a global weight rescale (×1000) | 1e−5 | 1.1e−7 | PASS |
| T8 | boundary | edge-concentrated density: mass=1, no force blowup (reflected KDE) | 1e−3 | 0.0 | PASS |
| T9 | compression/grid-consistency | grid = implicit kernel store, O(n_grid); bias shape stable across n_grid | 5e−2 | 7.5e−4 | PASS |
| T10 | no_leakage | F_ref guard fires on ref=True, silent on ref=False | — | — | PASS |
| T11 | native_estimator | native `F̂=−β⁻¹log ρ` recovers a 6 kT double well (dynamics) | 0.6 | 0.247 | PASS |
| T12 | common_meanforce | OPES-biased samples ⇒ unbiased conditional mean force integrates to F_true | 0.6 | 0.189 | PASS |
| **GATE2a** | multiwalker_exact | M points in 1 deposit ≡ 16 deposits (equal total samples, frozen weights): ρ, Σw identical | 1e−6 | 1.5e−7 | PASS |
| **GATE2b** | multiwalker_dynamics | equal-total-deposit runs across N∈{16,64,256} agree in native FES | 0.3 | 0.165 | PASS |

## 4. Bugs found and corrected

**In the implementation (`opes_core.py`): NONE.** The exact-math tests (T1, T4, T7, T8,
GATE2a) pass at 1e−6–1e−7 and the γ mapping is PLUMED-correct. The core is sound; the
pilot's finding that OPES underperforms ABF/mFR is therefore **not** an implementation
artifact of the OPES core (see §5 for the real caveats).

**In the audit harness (during construction, now fixed):** the first draft of T2/T3/T5
used the wrong sign for the biased marginal (`+βA` instead of `−βA`) — the engine ADDS the
bias potential A to V, so equilibrium is `p_b ∝ P·e^{−βA}`; and the first Gate-2 test
conflated normalization with mixing (unequal deposit counts, 20-step 1024-walker case).
Split into an exact normalization test (GATE2a) and a fair equal-deposit dynamics test
(GATE2b). These were test bugs, documented here for honesty; they did not touch the engine.

## 5. Category assessment and caveats

- **Core OPES algorithm: CORRECT (rules out Category D for the core).** Math, γ mapping,
  normalization, and multi-walker invariance all verified.
- **Real caveats that motivate the closure (NOT core bugs):** (i) the pilot tuned only
  BARRIER×PACE with σ fixed at 0.05 — σ is untuned and is the OPES kernel bandwidth, a
  first-order accuracy knob; (ii) `gamma_from_barrier` couples γ and ε through one knob;
  (iii) the pilot's WCA production used `sim.abf_force_clip=40` on the OPES force (a tight
  clip that can distort a sharp bias). The closure tuning sweeps BARRIER×PACE×SIGMA, adds a
  fixed-γ control, and records the clip fraction as a diagnostic. **This is why the pilot
  OPES numbers are labeled preliminary (v0) rather than being taken as the closed result.**

**Gate 1: PASS. Gate 2: PASS.**
