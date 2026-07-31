# SPEC — alanine dipeptide, 2-D torus CV, ABF + marginal Fisher–Rao

Status: **DESIGN ONLY, FROZEN.** No file under `src/`, `scripts/`, `configs/` or `results/`
has been created or modified. Written 2026-07-31 on branch `alanine-dipeptide`
(base commit `b84de17`). This file is the single normative source for the study; where it
disagrees with any design probe or scratch artifact, **this file wins**.

Reconciles five design probes (`system-hmr`, `integrator-bd`, `mean-force`, `reference-fes`,
`reuse-risk`) and three adversarial critiques (`physics`, `engineering`, `scientific-validity`).
Every disagreement is resolved explicitly in §14 with the reason and the measurement that
decided it.

House rules inherited from `docs/SPEC_score_free_controls.md`: every quoted number names the
artifact or script it came from; claims are stated at their true strength; hyperparameter
provenance is declared even when it is not guarded.

> **STATUS UPDATE (supersedes the header above).** Repository state has moved since this document
> was written. `ALANINE_EXECUTION_DECISION.md` is the authority on repository truth. Specifically:
> gate **V8 (Nyquist) now PASSES** — fixed in `c6a6718` with regression tests; gate **V15 (umbrella
> seed strain) now PASSES 576/576** — rigid-rotation seeding landed in `src/alanine/system.py`.
> `src/alanine/` now exists (Stage-0 only: build, seed, force field; no sampler, no driver), so
> "no file under `src/` has been modified" is no longer true. The *design* content below stands.
> Note also that §0's physical model is superseded for the FIRST study by the execution decision's
> conservative choice: **no HMR, dt = 1 fs, n_grid = 97 (odd)**.

---

## 0. DEVICE POLICY — READ BEFORE RUNNING ANYTHING

> ### GPUs 0, 1, 2 and 3 BELONG TO OTHER USERS. NEVER RUN ON THEM. NOT FOR ONE SECOND.
>
> **Only GPUs 4, 5, 6, 7 may be used, and only with `CUDA_VISIBLE_DEVICES` set explicitly.**

`torch`'s default CUDA device is GPU 0. Any `.to('cuda')` or `.cuda()` without
`CUDA_VISIBLE_DEVICES` lands on GPU 0 and violates this policy. Therefore:

```bash
source /home/zheyuanlai/miniconda3/etc/profile.d/conda.sh && conda activate abffr
CUDA_VISIBLE_DEVICES=7 python -u scripts/run_alanine_study.py --config … --require-single-gpu
```

* The indices are **absolute**. Never write `CUDA_VISIBLE_DEVICES=0` meaning "the first
  allowed GPU".
* Every launcher must call `--require-single-gpu` (precedent:
  `scripts/run_alkanes_cv_extension.py:7-11, 68-71`), which asserts
  `torch.cuda.device_count() == 1`.
* Every run's `.npz` records `cuda_visible_devices` (precedent: `alkanes/jobs.py` manifest).
  A post-hoc audit script must assert no artifact in `results/alanine/` carries a value
  intersecting `{0,1,2,3}`. A violation invalidates the artifact.
* A `results/alanine/GPU_ALLOCATION.txt` log records, per stage, which of 4–7 was used and
  why (precedent: `ALKANES_CV_EXTENSION_HANDOFF.md`).

**Measured device state at freeze time (`nvidia-smi`, 2026-07-31):** GPUs 0–3 idle (5 MiB,
0 %) — *idle is not permission*; GPU 4 105 023 MiB used / 99 %; GPU 5 105 023 MiB / 99 %;
GPU 6 **126 247 MiB / 92 % (only ≈17.5 GiB free)**; GPU 7 108 289 MiB / 70 %.

**Memory is the binding constraint, not throughput.** The launcher MUST check
`torch.cuda.mem_get_info()` against a computed requirement at startup and fail fast with the
required shape in the message (§3.9, §5 gate S0.12). GPU 6 will not hold the same job GPU 4
holds.

**CPU is a fully supported first-class fallback**, not a footnote. 256 logical cores
(2× AMD EPYC 9554), 1.5 TB RAM. Measured:

| device | dtype | shape | ms/step | source |
|---|---|---|---|---|
| CPU, `OMP_NUM_THREADS=64`, no compile | f32 | B = 1280 | 28.91 | brief |
| CPU, 64 thr, `torch.compile` | f64 | B = 9216 | **42.75** (jitter 2.9 %) | `rev_contend.py` |
| CPU, 64 thr, `torch.compile` | f64 | B = 2304 | **14.66** (jitter 0.4 %) | `rev_contend.py` |
| GPU 7 (70 % foreign), compiled | f64 | B = 9216 | **2.37** (synced) | `rev_bench.py` |
| GPU 7, **not** compiled | f32 | B = 9216 | 7.56 | `rev_bench.py` |

The CPU fit `t(B) = 17.59 + 9.436e-3·B` quoted by `reference-fes` is **pessimistic by 2.45×**
(it predicts 104.6 ms at B = 9216 against a measured 42.75). Stage 1 on CPU is 6.6 h in one
process, or **2.3 h wall** with 4 concurrent processes at B = 2304 — not the 16.3 h claimed.
Use CPU whenever GPU memory or contention makes the GPU path fragile.

**Cross-device throughput spread on the allowed GPUs** (identical workload, B = 9216, f64,
compiled, `rev_contend.py`): GPU 4 median 1.41 ms (min–max 1.41–2.55, **80 % jitter**),
GPU 5 2.28 ms (8.4 %), GPU 6 2.32 ms (4.8 %), GPU 7 2.37 ms. Plan against **2.4 ms**, the
slowest, never the mean. Do not use a wall-clock load balancer that assumes uniform devices.

---

## 1. Scientific question and pre-registered success criteria

### 1.1 The question

Does marginal-Fisher–Rao birth–death, layered on 2-D ABF, improve the estimate of the
free-energy surface `F(φ,ψ) = −β⁻¹ log p(φ,ψ)` of an **atomistic** molecule — Ace-Ala-Nme
in vacuum under AMBER ff14SB — relative to ABF alone at matched compute?

This is the project's first atomistic system. The three prior coordinate families gave:
WCA dimer **help** (−48 % L2 at `b1_h2`), entropic bottleneck **help** (−50 %), pentane 2-D
torsion **equivalent**, pentane R15 distance **harm** (+2 % to +32 %, 0/8 seed wins).

### 1.2 What the design probes already establish about this system — and what they do not

Established, and not to be re-litigated:

| fact | measurement | source |
|---|---|---|
| `F` range on `T²` | 90.2–97.6 kJ/mol = 36.2–39.1 kT | `fes_mbar48.npz`, `probe/ref36_F.npz` |
| C7eq global minimum | (−78.8°, +56.2°) [48² MBAR]; minimisation (−74.95°, +51.50°), E = **−91.27 kJ/mol** | `fes_mbar48.npz`, `s4_audit.py` |
| C5/β second minimum | (−146.2°, +153.7°) at **0.97 kT** | `fes_mbar48.npz` |
| C7ax minimum | (+63.8°, −41.3°) at **2.44 kT**; true `E_min` = −85.03 kJ/mol ⇒ ΔE = **6.23 kJ/mol = 2.50 kT** | `fes_mbar48.npz`, 18×18 OpenMM minimisation grid |
| `P(φ>0)` | 0.0302 / 0.0321 / 0.0404 across three independent references | §7.6 |
| ΔG(φ>0 vs φ≤0) | **3.17 / 3.41 / 3.47 kT** | same |
| C7eq↔C7ax saddle | 15.50–15.75 kT (2-D); 1-D φ-marginal barrier 15.1 kT (φ≈0) / 14.4 kT (±180° seam) | `fes_k200.npz`, `probe/analyze_*` |
| system **is** metastable | unbiased occupancy of φ∈[0°,90°) = **0.000000** over 102 ns aggregate (4 seeds × 512 replicas × 50 ps) vs reference 0.0298 | `probe/free_N512.npz` |
| ABF is **not** discovery-limited | median first entry into C7ax at **0.8–1.4 ps of 50 ps**; 512/512 replicas visit by 36 ps; 30.7 % of biased samples in the C7ax box | `probe/abf_N512.npz` |
| ψ is **not** a hidden slow coordinate | max internal barrier of `F(ψ|φ)` ≤ **0.64 kT** at every φ with `P(φ) > 1e-3`; 1-D φ-only ABF reproduces `p(ψ|φ)` at reference-weighted **TV = 0.080** with 100 % of φ bins supported | `cond_out.npz`, `probe/abf1d_N512.npz` |
| ABF residual, 50 ps, N = 512 | thermal-window norm L2 = **15.8 %**; final-quarter decrease 21 %; fitted `L2 ∝ t^-0.47` | `probe/abf_N512.npz` |
| starvation classifier verdict | **1 evidence family ⇒ `intermediate`**, extrapolating to `easy` (< 10 % floor) at ≈135 ps | `scripts/analyze_alkanes_cv_extension.py::starvation` applied to the probe |

**STRUCK — three claims that circulated in the design packet and are false.**

1. *"The positive-φ region sits ~19.5 kT (48.5 kJ/mol) above C7eq."* **FALSE.** It came from
   a minimisation started at (+80°, −80°) that relaxed into a spurious side minimum at
   (+60.9°, −179.5°). The true C7ax minimum is 6.23 kJ/mol = 2.50 kT above C7eq.
2. *"Entropy pays ~17 kT."* **FALSE**, and a corollary of (1): since ΔE = 2.50 kT and
   ΔG = 3.2–3.5 kT, `TΔS ≈ 0`.
3. *"This is structurally a starvation setup, the same regime as R15 where mFR failed 0/8."*
   **FALSE and backwards.** R15 was discovery-limited (support deficit, first entry ≫ run
   length). Here 512/512 replicas reach C7ax inside 3 % of a 50 ps run. Also the positive-φ
   minimum is **C7ax**, not α_L; `system-hmr`'s "α_L is a near-empty region" conflates the two.

### 1.3 The failure mode this system actually presents — named, because it drives everything

The ABF residual here is **transient-limited**, not support- or discovery-limited: it is the
un-forgotten early-time bias of a single shared, non-forgetting mean-force accumulator
(`core2d.py:202-204`: `f1s/f2s/csum +=`, one accumulator per seed shared by all N replicas).
It decays as `t^-0.47` and it is **N-independent by construction** — every replica feeds the
same wrong field.

Two consequences that the rest of this spec is built around:

* **The discriminating window is the transient, 20–300 ps — not 1 ns.** At 1 ns the
  extrapolated ABF residual is 3.9 %, against a reference systematic floor of ~2 % and a
  kernel-bandwidth floor (§4.5) that is method-independent. A final-L2-at-1-ns comparison is
  guaranteed null by construction. **The endpoint is therefore explicitly transient** (§9.1).
* **A residual with no sample-allocation component is not the same thing as a residual mFR
  cannot touch.** Verified on the repo's own artifact
  (`results/mfr_mechanism_audit/bias_variance/bias_variance_integrated.csv`, WCA `b1_h2`,
  the one cell where mFR demonstrably works, 10 seeds, `thermal_uniform_10kT`):

  | arm | `bias_fraction` | `int_var` | `int_bias2_debiased` | `int_mse_direct` |
  |---|---|---|---|---|
  | `abf` | **0.9905** | 0.0010264 | 0.107517 | 0.108544 |
  | `fr_estimated` | 0.9810 | 0.0013592 (**×1.324**) | 0.070166 (**×0.653**) | 0.071525 (**×0.659**) |

  **mFR's −34 % MSE on WCA is a reduction of the seed-common transient bias, while variance
  *increases* 32 %.** Reallocation changes per-bin sample counts, which changes the transient
  bias; it does not appear as across-seed variance. Any gate of the form "the variance
  fraction must exceed X" or "the residual must scale as `N^-1/2`" therefore predicts mFR
  cannot work on WCA, and is wrong.

### 1.4 Pre-registered success criteria (numeric, falsifiable, fixed before any production seed)

Primary endpoint: **kernel-matched, equilibrium-weighted, integrated L2 of `F`** over
`t ∈ [20 ps, 200 ps]`, on a common arm-independent support mask, paired within seed against
`abf` at **equal wall clock**. Defined exactly in §9.1–9.3.

An arm is declared **POSITIVE** only if ALL of the following hold (10 production seeds):

| # | criterion | threshold |
|---|---|---|
| P1 | paired median relative change, integrated L2 | ≤ **−15 %** |
| P2 | paired median relative change, L2 at `t = 200 ps` | ≤ **−15 %** |
| P3 | paired 95 % BCa bootstrap CI (10 000 resamples, seed 20260731) on P1 and P2 | upper bound **< 0** |
| P4 | matched-seed win rate | ≥ **8/10** (`p = 0.055` under the null; 3/4 would be `p = 0.3125`) |
| P5 | sign of P1 preserved under **all three** weightings: equilibrium, uniform-8 kT, uniform-10 kT | yes |
| P6 | named physical secondaries: |error in ΔG(C7eq→C7ax)| and |error in P(C7ax)| | both improve, or at worst worsen ≤ 10 % |
| P7 | frozen-bias reconstruction improvement / online improvement | ≥ **2/3** |
| P8 | age-aware ancestor ESS over a 6 ps lineage window (§9.5) | ≥ **0.30 N** |
| P9 | max ancestor fraction, same window | ≤ **0.05** |
| P10 | realised `fr_event_fraction` | < **0.05** |
| P11 | basin-occupancy TV computed from the **reconstructed** `F̂`, not the biased histogram | ≤ 1.10 × ABF |

**EQUIVALENT**: the 95 % CI of P1 lies entirely inside `[−0.10, +0.10]`.
**HARMFUL**: the CI lower bound of P1 exceeds `+0.10`.
**FALSE-IMPROVEMENT**: P1–P5 pass but any of P7–P11 fails. This is *not* a positive; it is
the R15 signature (support repaired while L2 worsened 34 % and ESS collapsed to 0.06 N) and
must be reported under that name.
**INCONCLUSIVE**: anything else.

Mirrors `scripts/analyze_alkanes_cv_extension.py::success` (lines 171-216) with three
deliberate strengthenings: win rate 8/10 not 3/4 (power, §14.7); frozen-bias promoted from a
separate confirm stage to a primary criterion (P7, because the accumulator artifact is the
leading alternative explanation here); and the equilibrium weighting promoted to primary
(P5, §9.3).

**What a POSITIVE would license:** *"On vacuum alanine dipeptide with ξ = (φ,ψ), at the
deployed replacement intensity and over the 20–200 ps transient, mFR birth–death reduces the
kernel-matched equilibrium-weighted L2 of the reconstructed FES by X % relative to ABF at
equal wall clock."* It would **not** license "mFR helps on atomistic systems", nor any claim
about the converged (t → ∞) regime, which this design cannot reach.

**What an EQUIVALENT would license:** a bounded atomistic neutrality result — the same role
pentane 2-D plays. This is the *expected* outcome (§12) and is a legitimate deliverable.

---

## 2. Physical model

### 2.1 Molecule and force field

* **Ace-Ala-Nme**, 22 atoms, capped alanine dipeptide, **L-alanine** (S at CA, confirmed
  against an unambiguous CIP reference construction).
* Atom order (frozen): `ACE(HH31 CH3 HH32 HH33 C O) ALA(N H CA HA CB HB1 HB2 HB3 C O)
  NME(N H CH3 HH31 HH32 HH33)`.
* **CV atom indices, frozen:** `PHI_ATOMS = (4, 6, 8, 14)` = C(ACE) N CA C(ALA);
  `PSI_ATOMS = (6, 8, 14, 16)` = N CA C(ALA) N(NME). Union `U = {4, 6, 8, 14, 16}`
  (5 atoms, 15 coordinates) — this is what the block-sparse estimator uses (§3.4).
* Force field: `amber14/protein.ff14SB.xml` as resolved by OpenMM 8.5.2.dev-36a30cb.
  **Identity is the sha256 of the resolved XML bytes**, `d9f9779c09d67cd5…`, 224 056 B — a
  version string is not sufficient provenance.
* Forces present, exactly and only: `HarmonicBondForce` (21), `HarmonicAngleForce` (36),
  `PeriodicTorsionForce` (42 = 38 proper + 4 improper), `NonbondedForce` (98 exceptions =
  57 fully excluded 1-2/1-3 + 41 scaled 1-4; 174 surviving pair interactions of 231).
  **No CMAP** — ff14SB is not CHARMM; state this explicitly, since alanine-dipeptide papers
  often assume one.
* 1-4 scale factors: **exactly** AMBER, `scee = 1/1.2 = 0.833333` (Coulomb), `scnb = 0.5` (LJ).
* Impropers use the AMBER convention (central atom **third**), all `n = 2`, `phase = 180°`,
  `k ∈ {43.932, 4.602}` kJ/mol. OpenMM stores them in the same `PeriodicTorsionForce` with an
  identical functional form, so no separate code path is needed.
* φ carries 2 explicit torsion terms (`k` = 1.757, 1.130); ψ carries 3 (2.301, 6.611, 1.883).
* Σm = **144.176000 amu** exactly (C₆H₁₂N₂O₂). Σq = **8.33e-17 e** — zero to float64 roundoff,
  not exactly 0.0 in bits; each residue separately neutral. Consequence is negligible; document
  it (the dipole becomes origin-dependent at 1.0e-15 e·nm per 7.3 nm shift).
* `ONE_4PI_EPS0 = 138.935456`.
* Vacuum dipole |μ| = 3.721 D at the relaxed C7eq minimum, 4.68–7.90 D over the torus.

### 2.2 OpenMM build recipe (frozen)

```python
from openmm import app, unit
import openmm as mm

ff  = app.ForceField('amber14/protein.ff14SB.xml')
top = <22-atom ACE-ALA-NME topology>            # src/alanine/build.py
sys = ff.createSystem(top,
                      nonbondedMethod = app.NoCutoff,
                      constraints     = None,        # ZERO constraints (asserted)
                      rigidWater      = False,
                      removeCMMotion  = False,       # deliberate, see below
                      hydrogenMass    = None)        # NO HMR, see §2.4
```

`removeCMMotion=False` is **correct and must be kept**, for three reasons:
(i) a `CMMotionRemover` constrains momenta to a 3N−3 subspace and fights the Langevin
thermostat; (ii) we remove the COM on *positions* every step, which is exact for all internal
observables (V is translation invariant to 1.9e-11 kJ/mol over a 3.7 nm shift; Σ_atoms F =
7.3e-12 kJ/mol/nm); (iii) decisively — `ff_torch.extract()` would **silently drop** a
`CMMotionRemover`, making the torch FF differ from the OpenMM system it was validated against.
Report `T_kin` over **3N = 66** degrees of freedom.

**`constraints = HBonds` is forbidden.** It removes 12 DOF and replaces `π` by
`∝ |Z_c|^{−1/2} e^{−βV} δ(σ(q))` — the classical rigid-vs-flexible discrepancy. Measured on
this molecule: the NME amide N–H length varies **100.73 → 101.63 pm (0.90 pm)** across the
torus, tracking the C7 H-bond distance `d(O···H) = 193 → 484 pm`. Freezing that bond freezes
out exactly this coupling. `getNumConstraints() == 0` is a hard startup gate.

### 2.3 Ensemble, integrator, timestep, friction — FROZEN

| knob | value | why |
|---|---|---|
| ensemble | NVT, canonical, `T = 300 K` | |
| `kT` | **2.4943 kJ/mol**, `β = 0.400914 mol/kJ` | |
| integrator | **BAOAB** underdamped Langevin (= OpenMM `LangevinMiddleIntegrator`) | §2.5 |
| `dt` | **1.0 fs = 1.0e-3 ps** | §2.4 |
| `γ` | **1.0 ps⁻¹** | pure efficiency knob; provably cannot bias `F` (§2.6 Step 5) |
| HMR | **NONE** (`hydrogenMass = None`) | §2.4 |
| constraints | none | §2.2 |
| dtype | **float64** everywhere | §2.7 |
| COM | remove COM **position** every step; do **not** remove COM velocity | §2.2 |
| units | nm, ps, amu, kJ/mol; `1 kJ/mol ≡ 1 amu·nm²/ps²` **exactly** | `F/m` needs **no** conversion factor |

> **TRAP, verified:** the `418.4` constant in the aborted scratch `baoab.py` is an AMBER
> kcal/Å factor and would have been a silent 418× error. In OpenMM units there is no factor.

BAOAB step, exactly **one** force evaluation per step (`F` cached across steps):

```
c1   = exp(-gamma*dt)                        # 0.99900050 at gamma=1/ps, dt=1e-3 ps
c2_i = sqrt((1 - c1**2) * kB*T / m_i)        # (22,1); 0.070331 nm/ps for H at 300 K
inv_m = 1/m_i                                # (22,1)

B:  v <- v + (dt/2) * F * inv_m              # F from the PREVIOUS step's final B
A:  q <- q + (dt/2) * v
O:  v <- c1*v + c2 * randn(v.shape, generator=gen_dyn)
A:  q <- q + (dt/2) * v
    q <- remove_com(q)                       # mass-weighted; rigid translation, exact
B:  F <- F_physical(q) + F_bias(q)           # SAME q for both
    v <- v + (dt/2) * F * inv_m
```

* The ABF bias force enters the **B** steps together with the physical force (it is a genuine
  conservative force `−∇[−B(ξ(q))]`) and must be evaluated at the **same `q`**.
* `gen_dyn` draws a **fixed-size** `(R, N, 22, 3)` block per step, independent of method, so
  `abf` and every `fr_*` arm are bit-matched outside birth–death. Preserve this exactly
  (precedent: `alkanes/core.py:211-212`).
* `remove_com` must be **mass-weighted**. `alkanes/geometry.remove_com` uses *unit* masses,
  which is not the physical COM for a molecule spanning H(1.008)–O(16.00). ADAPTED, §3.2.

### 2.4 HMR and timestep decision, with the invariance argument

**DECISION: no HMR, `dt = 1.0 fs`.** `system-hmr` recommended `hm = 3.0, dt = 2.0 fs`;
`integrator-bd` recommended no HMR, `dt = 1 fs`. **`integrator-bd` wins.** Reasons, in order:

1. **The only *configurational* accuracy measurement resolves 2 fs and does not resolve 1 fs.**
   Configurational temperature `kT_conf = ⟨|∇V|²⟩/⟨∇²V⟩` (exact for any Boltzmann sample,
   purely configurational, dominated by the stiff modes; resolution ±0.42 %, `e2_conftemp.py`):

   | scheme | dt | **T_conf error** | T_kin error |
   |---|---|---|---|
   | BAOAB γ=1/ps | 0.25 fs | +0.150 ± 0.42 % | −0.23 % |
   | BAOAB γ=1/ps | 0.50 fs | −0.355 ± 0.42 % | −1.13 % |
   | **BAOAB γ=1/ps** | **1 fs** | **+0.082 ± 0.42 %** | −2.00 % |
   | BAOAB γ=1/ps | 2 fs | **+0.688 ± 0.43 %** | −8.75 % |
   | overdamped | dt·k = 0.02 | +1.248 ± 0.43 % | — |

   `T_kin` at −8.75 % is **not** a valid rejection of 2 fs — it is a *momentum-marginal* error
   and BAOAB's entire point is that its configurational marginal is far more accurate
   (`system-hmr` says so itself). The valid rejection is `T_conf`: unresolvable at 1 fs,
   marginally resolvable at 2 fs. Our target `F(φ,ψ)` is a pure configurational marginal.
2. **HMR without constraints buys at most 1.494× in ω_max** — not the 2–2.5× of HMR+SHAKE
   workflows. A 1.5× speedup does not justify carrying a second mass set through the
   reference/production invariance argument (§7.1) when the reference and production **must**
   share `(M, dt)` exactly (§14.5).
3. **HMR is non-monotonic on this molecule and `hm = 4` is *worse* than `hm = 3`** — a footgun
   for anyone tuning later. `ω_max` = √λ_max(M^{−1/2}∇²V M^{−1/2}) at the C7eq minimum, exact
   autodiff Hessian, independently reproduced by finite-difference of OpenMM forces:

   | hydrogenMass | m(methyl C) | μ(C–H) | ω_max [ps⁻¹] | 2/ω_max [fs] | last stable dt |
   |---|---|---|---|---|---|
   | **none (1.008)** | 12.010 | 0.930 | **622.80** (622.72) | 3.211 | 2.5 fs |
   | 2.0 | 9.034 | 1.638 | 462.95 (462.75) | 4.320 | 3.5 fs |
   | 3.0 | 6.034 | 2.004 | **416.81** (416.75) | 4.798 | 4.0 fs |
   | 4.0 | 3.034 | 1.725 | 490.64 (490.63) | 4.076 | 3.0 fs |
   | 4.5 | 1.534 | 1.144 | 642.17 (642.16) | 3.114 | *worse than none* |

   Mechanism: three methyl groups carry 9 of the 12 H, so `m_C(h) = 15.034 − 3h` and the C–H
   reduced mass `μ(h) = h(15.034−3h)/(15.034−2h)` is non-monotonic. `dμ/dh = 0` gives
   `6h² − 90.204h + 226.021 = 0` ⇒ **`h* = 3.177`** (`system-hmr` wrote 225.02 / 3.158 — an
   arithmetic slip; `15.034² = 226.021`). The measured `ω_max` minimum is at `h = 3.0`
   (416.75) with `ω_max(3.177) = 420.64`, so the model and the measurement agree in *ranking*
   but the claimed "exact agreement" is withdrawn.
4. Margin at (none, 1 fs): `ω_max·dt = 0.623`, 2.5× to the measured blow-up at 2.5 fs; 20 ps
   γ=0 energy conservation `max|ΔE| = 0.997 kT` (vs 5.85 kT at 2 fs, NaN at 4 fs).

**Fallback, pre-registered:** if Stage-0 throughput proves binding, the *single* permitted
alternative is `(hm = 3.0, dt = 1.5 fs)`, `ω·dt = 0.625`. It must be applied **identically to
the reference and every method arm**, and gate S0.6 (§5) must be re-run and re-passed.

#### PROOF — HMR leaves `F(φ,ψ)` exactly invariant (spec-ready)

*Setup.* `Q = ℝ^{3A}`, A = 22. `V: Q → ℝ` is the ff14SB energy (harmonic bonds, harmonic
angles, periodic torsions incl. impropers, LJ, Coulomb). **No term of V contains a particle
mass.** `M = diag(m₁I₃,…,m_AI₃) ≻ 0`. `H_M(q,p) = ½pᵀM⁻¹p + V(q)`.

**Step 1 — Factorisation.** `e^{−βH_M(q,p)} = e^{−βpᵀM⁻¹p/2} · e^{−βV(q)}` exactly, for every
`M`. Hence the canonical measure is a **product measure** `μ_M = ν_M ⊗ π` with
`ν_M = N(0, M/β)`, `Z^kin_M = (2π/β)^{3A/2}(det M)^{1/2}`, and `π(dq) ∝ e^{−βV(q)}dq`.
All `M`-dependence is confined to `ν_M` and the constant `Z^kin_M`. **`π` does not depend on
`M`.** This is an exact algebraic identity, not an asymptotic statement.

**Step 2 — `F` is a functional of `π` alone.** `ξ(q) = (φ(q), ψ(q))` is a function of
**positions only** (a dihedral is determined by four position vectors; no momentum, no mass).
With `p(z) = ∫δ(ξ(q)−z)e^{−βV}dq / ∫e^{−βV}dq` and `F(z) = −β⁻¹log p(z)`, both `ξ` and `π` are
`M`-independent, therefore **`p` and `F` are `M`-independent**. ∎

**Step 3 — What HMR is.** For each hydrogen `h` with unique bonded heavy partner `a(h)`:
`m′_h = h_target`, `m′_a = m_a − Σ(h_target − m_h)`; all else unchanged. It changes `M` and
nothing else — not `V`, not the topology, not `ξ`, not `β`, not `Q`. *Verified:* bit-level
comparison of every frozen parameter array across `hydrogenMass ∈ {none,2,3,4}` — **the only
differing array is `masses_dynamic`**; `Σm′ = Σm = 144.176000` amu exactly;
`max|V_hm(q) − V_none(q)| = 0.000e+00` over thermal configurations (`s17_hmr_identity.py`).

**Step 4 — The estimator inherits the invariance.** With `g_a = ∇ξ_a`, `G_ab = g_a·g_b`,
duals `w_a = Σ_b (G⁻¹)_{ab} g_b` (so `w_a·g_c = δ_ac`):
`f_a = ∇V·w_a − β⁻¹ div w_a`, `⟨f_a | ξ=z⟩_π = ∂F/∂z_a`.
Every object (`V, ξ, G, w, div w`) is built from positions and the potential — **no mass
appears anywhere** — so the target is mass-independent term by term.

> **CORRECTED (adopted from critique `physics` #8).** `system-hmr` Step 4 stated the blue-moon
> correction as `−β⁻¹ d log det Z` and implied a mass-metric dual would need it. **Both parts
> are wrong.** (a) The correct Fixman potential is `U_F = +(2β)⁻¹ log det Z`, i.e. the quoted
> form had the wrong sign *and* was a factor 2 too large. (b) More importantly the conflation
> is wrong *in kind*: the **unconstrained** mass-metric den Otter dual of §4.2 has **no Fixman
> term at all** — `−β⁻¹ div w` already carries everything. The `|Z|^{−1/2}` weight is a
> property of the **constrained** (blue-moon) ensemble, not of a mass metric. Since this text
> was marked "spec-ready, verbatim" it would have been copied; it is replaced here.

**Step 5 — The sampler.** Underdamped Langevin `dq = M⁻¹p dt`,
`dp = −∇V dt − γp dt + √(2γM/β)dW` has `μ_M = ν_M ⊗ π` invariant for every `M ≻ 0` and every
`γ > 0`. So the configurational marginal of the exact continuous-time dynamics is `π`
independently of **both** `M` and `γ`. **`γ` therefore cannot change `F`.**

**Step 6 — The one honest caveat (finite `dt`).** BAOAB samples a perturbed `μ_M^{dt}`. Its
configurational averages converge at `O(dt²)` with an anomalously small coefficient relative
to other Langevin splittings, the leading term vanishing in the high-friction limit
(Leimkuhler & Matthews 2013). **That error does depend on `M`**, through `ω_max(M)·dt`. Hence:
*HMR leaves `F` exactly invariant for the continuous dynamics; at finite `dt` the residual
bias is `O((ω_max(M)dt)²)`, which is precisely why `(hydrogenMass, dt)` must be validated
empirically* (gate S0.6) *and why the reference and every method arm must share them* (§14.5).
Measured confirmation: the `T_kin` coefficient ratios `c(none)/c` = 1.00/1.92/2.14/1.54 for
`hm = none/2/3/4` agree with `(ω_none/ω)²` = 1.00/1.81/2.23/1.61 to 4–6 %.

**Step 7 — Domain technicality.** `V` is translation invariant so `∫_Q e^{−βV}dq = ∞`.
Factor `q = (R, Ω, q_int)`. `e^{−βV}` is constant in `(R,Ω)`, the `(R,Ω)` volume is genuinely
`z`-independent, and it cancels between numerator and denominator of `p(z)`; `ξ` depends on
`q_int` alone. *Verified:* `|E(x) − E(x+3.7nm)| = 1.9e-11`, `|E(x) − E(Rx)| = 1.4e-12` kJ/mol.

> **CORRECTED (adopted from `physics` #12).** `system-hmr` wrote `π = Leb(ℝ³) ⊗ Haar(SO(3)) ⊗
> π_int`. That is a **false factorisation** — the Cartesian → (COM, orientation, internal)
> change of variables carries a `q_int`-dependent Jacobian (`√det I(q_int)` × internal
> Jacobian) which does not factor out. The conclusion survives, because that Jacobian belongs
> *inside* `π_int` and both the den Otter estimator and the umbrella/MBAR reference sample the
> same flat Cartesian measure — but the identical slip in an internal-coordinate or
> constrained setting would be fatal, so the statement is corrected rather than kept.

**Step 8 — What HMR DOES change.** Momenta, velocities, kinetic temperature, vibrational
spectrum, diffusion, correlation times, transition rates, effective CV inertia. HMR changes
**how fast** `F` is sampled, never **what** `F` is. Any kinetic claim is outside this theorem.

*(Recorded but not acted on: HMR does not slow the CV — the mass metric `Z_ab` gives
`I_φ` 0.03636 → 0.03273 amu·nm² at `hm = 3`, i.e. **−10.0 %**, because φ's four atoms are all
heavy and HMR lightens N and CA. HMR would have been a pure win on both axes had reasons 2–3
above not applied.)*

### 2.5 Why BAOAB and not the repo's overdamped scheme

`alkanes/core*.py` all use `q += dt*(F+bias) + sqrt(2dt/β)ξ` — no momenta, no masses, unit
mobility. For ff14SB alanine this is **non-viable**, measured three independent ways:

* Max Cartesian Hessian eigenvalue **1.368e6 kJ/mol/nm²**; explicit-Euler overdamped stability
  needs `dt < 2/λ_max = 1.46e-6 ps`. Measured blow-up edges confirm `dt < 2γ/ω²_max` exactly:
  γ=1/ps → **0.005 fs**; γ=10 → 0.05 fs; γ=100 → 0.5 fs. At γ=1 that is **~620× worse** than
  BAOAB's stable step.
* Euler–Maruyama is **weak order 1**: `O(dt)` bias lands directly on `p(φ,ψ)`, the observable.
  BAOAB is weak order 2 with an anomalously small configurational coefficient — it puts the
  discretisation error where we do not measure it (measured: −2.00 % kinetic vs
  +0.08 ± 0.42 % configurational at 1 fs, **≥5× more accurate at the same `dt`**).
* At its own 1 %-accuracy step the overdamped scheme is **2.2× slower per step of φ
  exploration** and carries ≥3× more configurational bias; matching BAOAB's ≤0.4 %
  configurational bias needs `dt·k ≈ 0.006` ⇒ **≈7.5× more force evaluations at equal
  accuracy**, and the gap widens as `ε⁻¹` vs `ε⁻¹ᐟ²`.
* Unit mobility is physically wrong (H and O diffuse identically), it destroys kinetics
  (round-trips, mixing times, "ns of sampling" all become meaningless for an atomistic paper),
  and **there are no momenta**, so the momentum-aware birth–death rule cannot even be stated.

> **Do not quote the "1.5e9–2.4e10 steps/ns" figure** for the overdamped scheme. It is
> ill-defined: mapping `dt_phys = dt_repo·m·γ` spans 4.2e-5 fs (m=1.008) to 6.7e-4 fs (m=16),
> a 1 500–24 000× range, precisely because the repo scheme assigns one mobility to atoms whose
> masses span 16:1. It is not the overdamped limit of *any* single Langevin dynamics on this
> molecule. The accuracy-matched 7.5× above is the honest number.

BAOAB validation, all passed (`e1_validate.py`, `e1b_halfstep.py`, `s7_verlet_parity.py`):

| gate | measured |
|---|---|
| γ=0 ⇒ exact velocity Verlet vs OpenMM `VerletIntegrator`, 1000 steps @1 fs | `max|Δq| = 1.5e-9 nm`; `max|Δv| = 6.5e-8 nm/ps` *after* adding OpenMM's leapfrog half-step offset `+ (dt/2)F/m` (raw mismatch 0.491 nm/ps **is** that offset: the −h/2 correction gives 0.98, the +h/2 gives 6.5e-8) |
| γ=0, 2000 steps, `dt = 0.5 / 2 fs` | `max|Δx| = 3.5e-9 / 3.3e-8 nm`; NVE drift agrees to 5 decimals (+0.31224 vs +0.31223 kJ/mol) |
| γ=0 energy conservation, 20 ps | `max|ΔE| = 0.264 / 0.997 / 5.85 kT / NaN` at `dt = 0.5 / 1 / 2 / 4 fs` |
| equipartition, γ=1/ps | `T_kin/T = 0.99487 / 0.98042 / 0.91106 / NaN` |
| thermal cross-check vs OpenMM (24×20 ps) | `⟨V⟩ = −14.92 ± 0.68` (OpenMM) vs `−14.17 ± 0.02` kJ/mol (torch, B=1024), **1.1σ**; `T_kin` 300.11 vs 300.50 K |

### 2.6 The mean-force estimator is unbiased under this sampler

`F(z) = −β⁻¹ log ∫δ(ξ(q)−z)e^{−βV(q)}dq` — the **configurational delta-marginal**, no
Jacobian, no Fixman factor. For any smooth `w_a` with `w_a·∇φ_c = δ_ac`,

```
div_q( w_a δ(ξ−z) e^{−βV} )
   = (div w_a) δ e^{−βV} − d/dz_a[ δ e^{−βV} ] − β (∇V·w_a) δ e^{−βV}
```
Integrating over `q`, the LHS vanishes (decay / periodicity), giving the den Otter identity
`∂F/∂z_a = ⟨ ∇V·w_a − β⁻¹ div w_a ⟩_{ξ=z}`. **The only inputs are biorthogonality and
smoothness; `M` never appears.** Identity and mass duals are both exactly unbiased and differ
only in variance. Empirically the two conditional means agree to **max 0.472 kJ/mol/rad
(0.54 % of max |∇F|), RMS 0.095**, against a per-node SE of ~1.3.

Under BAOAB with any `γ > 0` and any `M ≻ 0`, the invariant density factorises and the
`q`-marginal is `e^{−βV}/Z` independent of `γ` and `M`. Momenta integrate out to a
`z`-independent constant because `M` is constant in Cartesian coordinates and `ξ` depends on
`q` only. Residual error is time-discretisation only, not estimator bias.

**Self-consistency with mFR:** `p̂` in `fr_score_2d` is the empirical *delta*-marginal of
`(φ,ψ)`, the same `F` the estimator targets, so `q ∝ exp(−β(F_target − F̂))` is self-consistent.

### 2.7 dtype

**float64 everywhere.** Nothing in the pipeline breaks in fp32 *except coordinate drift*: with
`removeCMMotion=False` and Langevin on all 3N DOF, `D_com = kT/(Mγ) = 0.01730 nm²/ps`, so
`rms|R_com|` = 3.2 nm @100 ps, **10.2 nm @1 ns**, 32 nm @10 ns. At 10 nm, fp32 forces are
already `1.4e-4` relative (vs `2.8e-6` at |x|~0) — a 50× degradation purely from cancellation.

Measured fp32-vs-fp64 errors (`s3_dtype.py`): energy ΔE 1.7e-4 kJ/mol; Hessian of φ rel 3.7e-7;
`div(w)` 9.9e-7 abs on rms 0.745; biorthogonality **6.7e-16 (fp64) vs 3.6e-7 (fp32)**; mean
force 4.1e-4 kJ/mol/rad = 1.3e-5 of `sd(f)`; FFT Poisson rel 4.5e-7…1.2e-6; ABF accumulators
at 1e6 counts/bin 1e-4 kJ/mol/rad.

fp64 costs **nothing at these shapes** (2.37 vs 2.44 ms/step at B = 9216, `rev_bench.py`).

> **Do not carry "float64 is free on the H200" forward as a general rule** (`engineering`,
> adopted). It is free *because the workload is kernel-launch-bound at 22 atoms*, not because
> H200 fp64 is fast — it is 2× slower than fp32 in peak FLOPs. The statement is false the
> moment the system grows.

If a CPU-fp32 path is ever used, **per-step mass-weighted COM recentering is mandatory** (it
keeps coordinates ~0.5 nm and the error at 2.8e-6).

> **Self-correction recorded (`system-hmr`):** an initial measurement showed a 5 % fp32
> accumulator bias. It was an artifact — `torch.rand(dtype=float32)` and `float64` draw
> *different RNG streams*, so the two runs sampled different configurations. With identical
> samples, fp32 accumulators are fine. Recorded because the same trap will recur in any
> dtype A/B test.

### 2.8 Parameter-export contract (provenance)

`src/alanine/params_io.py` writes a frozen pair, and **production reads only the frozen pair**
— never `openmm` at run time.

`ala2_params.npz` (float64, C-contiguous): `n_atoms`; `atom_name`, `residue_name`,
`residue_index`, `element`; `topology_bonds (21,2)`; `masses_physical`, `masses_dynamic (22)`;
`bonds_idx/r0/k (21)`; `angles_idx/theta0/k (36)`; `torsions_idx/n/phase/k (42)` +
`torsions_is_improper (42, bool)`; `nb_charge/sigma/epsilon (22)`; `exc_idx/qq/sigma/epsilon
(98)`; the **flattened pair list actually evaluated** `pair_idx (174,2)/pair_qq/pair_sigma/
pair_epsilon`; `one_4pi_eps0 = 138.935456`; `cv_phi_atoms`, `cv_psi_atoms`; `x0_min_nm`.

`ala2_params.json` (provenance): `schema_version`, `created_utc`, `hostname`, python/numpy/
torch/**openmm 8.5.2.dev-36a30cb**; repo commit/branch/dirty; **force-field identity = sha256
of the resolved XML bytes**; `create_system_kwargs`; **`system_xml_sha256`** of
`mm.XmlSerializer.serialize(system)` (catches anything the npz forgot); **`param_sha256`**;
units dict; CV atom indices + convention.

`param_sha256` is taken over sorted keys, feeding `key ‖ shape ‖ dtype-tag ‖ raw bytes`, with
floats canonicalised to `<f8` and ints to `<i8`, so a writer dtype change cannot move the hash
but any *value* change must. Measured for `hm = none`:
`0006a1261f267f74c1b7dc19ca8b39f2448d2e0373419ade510da441c20f5600`, reproducible across
rebuilds. **A production run records the hash; a mismatch is a hard startup error.**

**Hard gates in the exporter** (a real gap: `ff_torch.extract()` *silently ignores* any force
class it does not know):

* force-class set is exactly `{HarmonicBond, HarmonicAngle, PeriodicTorsion, Nonbonded}`, no
  duplicates → catches a future `GBSAOBCForce`, `CMAP`, `CMMotionRemover` or `CustomForce`;
* `getNumConstraints() == 0`; `getNonbondedMethod() == NoCutoff`; no virtual sites; 22 particles;
* pair-list drop count == fully-excluded-exception count (**57**), and `min ε > 0`, so the
  `(qq==0 & ε==0)` keep-mask cannot silently discard a real interaction.

**Validation, measured:** frozen npz → FF rebuilt with **no OpenMM at all** reproduces OpenMM
over 32 thermal configs at `max|ΔE| = 1.9e-6 kJ/mol`, `max ΔF/|F|max = 1.5e-9`
(`s8_export.py` + `s15_roundtrip.py`). Independently, torch-vs-OpenMM parity in float64 is
`max` relative energy error **1.0e-9**, force **3.0e-10**, over 24 thermally displaced configs
spanning `E ∈ [1923, 4823]` kJ/mol.

### 2.9 Chirality and the reflection symmetry

The classical FF is **exactly reflection invariant**: `E(mirror) = E(original)` to 6 decimals,
so `F_D(φ,ψ) = F_L(−φ,−ψ)` identically. A classical force field cannot distinguish
enantiomers. Consequences:

* Our map is **L-alanine only** and must be asymmetric; that it is, is a free internal
  consistency check.
* A chirality flip in a *subset* of umbrella windows would silently **symmetrise** `F_ref` and
  erase the C7eq/C7ax asymmetry. Chirality must therefore be asserted per window (§7.2).
* Correct statement, replacing a docstring bug that must be fixed on promotion:
  **L-alanine ⇒ `((C−CA)×(CB−CA))·(HA−CA) < 0`** (measured −2.024 at the minimum, +2.024 for
  the mirror image; uniformly −2.249…−1.977 across the torus at the builder default
  `cb_offset = −120`). `ala22.chirality()`'s docstring claims ">0 means L" — the geometry is
  correct, the comment is inverted.

### 2.10 CV convention

`geometry.signed_dihedral` supports `convention ∈ {"rb", "iupac"}`; `DihedralCV.value` and
`cv2d` hard-code `"rb"` (trans = 0). **Alanine uses IUPAC (trans = ±π).** Verified:
`φ_rb = wrap(φ_iupac + π)` to 4.4e-16, and **gradients and Hessians are bit-identical**
(max diff 0.0) because the conventions differ by an additive constant. So only `values()`
and the grid labelling are affected — but left unfixed, **every Ramachandran plot, basin
definition and literature comparison is shifted by 180°**. Confirmed live: at the C7eq
minimum (`φ_IUPAC = −75.05°, ψ = +53.95°`) the repo `cv2d.values()` reports ≈ (+105°, −126°).

**Hard assertion, once at startup:** the reference FES minimum must land within 10° of
(−78.8°, +56.2°). A mirrored build would put it at (+78.8°, −56.2°) and silently invert every
conclusion.

---

## 3. Code architecture — `src/alanine/`

New package, additive. **No file under `src/alkanes/` is edited except the two additive,
default-off changes in §3.10.** Follows `alkanes/jobs_cv.py` conventions (frozen dataclass →
md5 spec hash → deterministic `run_id` → one atomic `.npz` per job).

### 3.1 Module layout and public signatures

```
src/alanine/__init__.py          docstring + version constant CORE_VERSION = "alanine_v1"

src/alanine/build.py             # NEW
    PHI_ATOMS = (4, 6, 8, 14)
    PSI_ATOMS = (6, 8, 14, 16)
    CV_UNION  = (4, 6, 8, 14, 16)
    def build_topology() -> openmm.app.Topology
    def build_system(hydrogen_mass=None) -> (System, Topology)
    def minimized_reference(device, dtype) -> np.ndarray          # (22,3) nm, the ONE verified C7eq
    def seed_at(x0, phi, psi) -> np.ndarray                       # RIGID dihedral rotation of x0
    def chirality(x) -> float                                     # <0 == L-alanine (docstring FIXED)
    def assert_seed_ok(x, params, e_angle_max=50.0, dev_max_deg=15.0) -> None

src/alanine/ff_torch.py          # NEW (promoted from scratch, gates added)
    def extract(system) -> dict                                   # plain arrays; HARD gates of 2.8
    class TorchFF:
        def __init__(self, params: dict, device, dtype)
        def energy(self, x)  -> Tensor                            # (B,22,3) nm -> (B,) kJ/mol
        def forces(self, x)  -> Tensor                            # (B,22,3) kJ/mol/nm
        def energy_and_forces(self, x) -> (Tensor, Tensor)

src/alanine/params_io.py         # NEW
    def export_params(system, topology, out_stem) -> str          # returns param_sha256
    def load_params(stem, device, dtype) -> (dict, str)           # verifies param_sha256
    def assert_param_hash(expected: str, actual: str) -> None     # hard startup error

src/alanine/cv2d_fast.py         # NEW (block-sparse rewrite of alkanes/cv2d.py)
    class FastJointDihedral2D:
        def __init__(self, atoms_a, atoms_b, n_atoms, convention="iupac")   # NO defaults on atoms
        def values(self, q)       -> Tensor                       # (B,2) rad, IUPAC
        def grad_only(self, q)    -> (phi, gfull)                 # gfull (B,2,A,3)
        def geometry(self, q)     -> dict(phi,g,G,Ginv,div_v,cond,lam_min)
        def local_mean_force(self, q, F, beta) -> (f, phi, gfull, geo)
    def abf_bias_force_2d(gfull, bias_at) -> Tensor               # (B,A,3)

src/alanine/dynamics.py          # NEW
    class BAOAB:
        def __init__(self, ff, mass, dt, gamma, kT, gen_dyn, device, dtype)
        def init_state(self, x0, n) -> (q, v, F)
        def step(self, q, v, F, bias_force) -> (q, v, F)          # ONE force eval
    def maxwell(shape, mass, kT, gen, device, dtype) -> Tensor
    def remove_com(q, mass) -> Tensor                             # MASS-WEIGHTED

src/alanine/bd.py                # NEW (vectorised, per-seed streams, momenta+force aware)
    def birth_death(q, v, Fc, score, ancestors, family, sim, gens_fr) \
        -> (q, v, Fc, ancestors, family, n_repl, death_mask, birth_src)

src/alanine/projection.py        # NEW
    def weighted_projection(g1, g2, w, dz1, dz2, n_cg=24, tol=1e-10) -> (B, gB1, gB2, info)
    def masked_projection(g1, g2, den, dz1, dz2, min_count) -> (B, gB1, gB2, g1m, g2m)
    def assert_gradient_consistent(B, gB1, gB2, dz1, dz2, tol=1e-12) -> None

src/alanine/basins.py            # NEW
    RAMA_BOXES = (...)            # C7eq, C5/beta, PPII, C7ax, alphaL, other  (§9.4)
    def basin_index(phi, psi) -> LongTensor
    def basin_occupancy_from_F(F_hat, grid1, grid2, beta) -> Tensor

src/alanine/core2d_ala.py        # NEW (driver; estimator layer reused from alkanes)
    def run_sampler_ala(method, ff, sim: AlaSimConfig, seeds, cv, device, dtype,
                        initial_mode, oracle_free_energy=None, verbose=True) -> dict
    def run_frozen_bias_ala(B_frozen, ff, sim, seeds, cv, device, dtype,
                            init_ensemble, verbose=True) -> dict

src/alanine/umbrella.py          # NEW (promoted from stage1_ref/umbrella2.py)
    @dataclass class UmbrellaConfig                                # kappa, centers, copies, ...
    class Umbrella2:
        def seed(self, x0) -> Tensor                              # seed_at + gate + minimise
        def run(self, n_warm, n_equil, n_prod, save_every) -> dict # (n_traj, n_frames, 2)

src/alanine/mbar_torch.py        # NEW (promoted; Anderson acceleration added)
    def u_kn_rank5(centers, phi, psi, beta, kappa) -> (L, Rmat)   # NEVER materialise (K,N)
    def mbar_solve(L, Rmat, N_k, tol=1e-9, max_iter=400, anderson_m=8, chunk=None)
        -> (f, n_iter, dmax)                                      # RAISES if dmax > tol
    def log_weights(L, Rmat, N_k, f, chunk=None) -> Tensor
    def overlap_matrix(L, Rmat, N_k, f) -> Tensor                 # O = N_k[:,None]*(W@W.T)
    def histogram_fes(logW, phi, psi, n_grid, beta) -> (F, counts)
    def block_bootstrap(..., n_rep=100, n_blocks=40) -> (SE, reps)

src/alanine/metrics_ala.py       # NEW
    def kernel_matched_l2(F_hat, F_ref, h_abf, dz, mask, weight) -> float
    def eq_weight(F_ref, beta, mask) -> np.ndarray
    def common_mask(F_ref, beta, c_kT) -> np.ndarray              # ARM-INDEPENDENT
    def fes_secondaries(F_hat, F_ref, grid, beta) -> dict         # dG(C7ax), P(C7ax), ...

src/alanine/jobs_ala.py          # NEW
    @dataclass(frozen=True) class AlaRunSpec                       # §3.7
    def expand_stage(cfg, stage) -> list[AlaRunSpec]               # copied from jobs_cv
    def execute_run(spec, device, cache_dir="cache/alanine", verbose=False) -> dict
    def run_npz_path(raw_dir, spec) -> str

scripts/run_alanine_reference.py    scripts/run_alanine_study.py
scripts/run_alanine_frozen.py       scripts/analyze_alanine.py
scripts/run_alanine_killshot.py
configs/alanine/{killshot,stage0,reference,abf_characterise,fr_tuning,production,frozen}.yaml
```

### 3.2 REUSED UNCHANGED / ADAPTED / NEW — the explicit table

**A = reused verbatim, B = adapted (parameters or small change), C = new / rewritten.**

| `src/alkanes/` file | class | verdict and reason |
|---|---|---|
| `poisson2d.py` | **A** (+ one additive kwarg, §3.10) | Pure FFT Hodge on `T²`. Reused for the *masked* fallback path. **Requires the Nyquist fix or an odd grid** (§3.5). |
| `periodic.py` | **A** | `periodic_grid`, `wrapped_gaussian_kernel_matrix`, `bin_counts/bin_sum`, `circular_interp`, `mean_force_profile`, `free_energy_from_mean_force`, `circular_l2`, `marginal_{kl,tv,l2}` are CV-agnostic. |
| `density2d.py` | **A** | `torus_grid`, kernels, `scatter_counts/sum`, `smooth2`, `kde2`, `mean_force_fields`, `bilinear_interp2`, `kl2/tv2/l2_2d`, `entropy2`, `fr_score_2d` all generic. |
| `interval.py` | **A** | Unused for `T²`; reusable verbatim if a distance CV is added. |
| `distance_cv.py` | **A** | Index-based, molecule-agnostic. Unused here. |
| `metrics_cv.py` | **A** (`l2_2d_np`, `joint_profile_metrics`, `meanforce_vector_error`) / **B** (`reconstructed_fidelity`) | The barrier-partition in `reconstructed_fidelity` is pentane's 3-basin split → Ramachandran boxes. |
| `metrics.py` | **A** (`profile_metrics`, `marginal_metrics`, `_circ_l2`, `fr_event_metrics`, `conditional_from_joint`) / **B** (`_basins_1d`, `conditional_metrics`, `joint_basin_visits`) | Same barrier-partition issue. |
| `jobs.py` | **A** (`run_is_valid`, `save_run`, `save_failure` — atomic `*.tmp.npz` → `os.replace`) / **C** (`AlkaneRunSpec`, `build_*`, `physics_tag`) | IO helpers are generic; the spec is alkane physics. |
| `jobs_cv.py` | **A** (skeleton: `expand_stage`, `_method_knobs`, `load_yaml`, `_base_out`, `run_npz_path`, the `spec_hash`/`run_id` idempotency pattern) / **C** (bodies) | Copy the skeleton verbatim into `jobs_ala.py`; `build_2d_reference`, `make_init`, `execute_2d` are alkane-bound. |
| `core.py` | **A-in-spirit** (`_recentered_clipped_score`, `_ancestor_stats`, `assert_no_reference_leakage`, `FR_METHODS`/`ESTIMATED_TARGET_METHODS`/`ALL_METHODS`) / **C** (`_birth_death`, `run_sampler`, `run_frozen_bias`) | `_birth_death` clones positions only and couples seeds (§3.6); the drivers are overdamped and alkane-bound. |
| `core2d.py` | **A** (`_fr_target` — EMA → `q ∝ exp(−β(F_ema−B))`) / **B** (`_project_bias` structure) / **C** (`run_sampler_2d`, `run_frozen_bias_2d`, `_basin1d`, `_joint_basin`) | Drivers call `geom.place_chain`, `pot.forces`, `params.{n_atoms,n_dihedrals,beta,d0,theta0}` and are **overdamped**. `basin_barrier = 61.6°` is pentane's gauche/anti split, meaningless here. |
| `cv2d.py` | **C** (rewritten as `alanine/cv2d_fast.py`; the *original is verified correct* on 22 atoms and is the reference implementation for the equivalence test) | See §3.4. |
| `cv.py` | **B** | `DihedralCV` is index-based and works at 22 atoms; `_phi4`/`DihedralCV.value` hard-code `convention="rb"`. `abf_bias_force` is **A**. |
| `geometry.py` | **A** (`wrap_to_pi`, `circular_diff`, `signed_dihedral`) / **B** (`remove_com` — uses *unit* masses; must be mass-weighted) / **C** (`place_chain*`, `_place_next` — hard-wired to a linear chain, `n_dih = n_atoms−3`, one bond length, one angle) | |
| `potentials.py` | **C** | Reduced-unit united-atom alkane. No analogue; `decouple=True` has **no alanine counterpart**, so the repo's hard `B0/P0` "recover V4⊕V4" gate must be replaced (§5, gate S0.4). |
| `reference.py`, `reference_cv.py` | **C** (except `marginalize_joint_to_phi1`, `conditional_phi2_given_phi1`, which are **A**) | Rest rests on the linear-chain internal-coordinate factorisation with a φ-independent Jacobian `∏r²∏sinθ`. False for a branched molecule with 42 coupled torsions. |
| `opes_cv.py` | **A** (`BatchedTorusOPES`) / **B** (`TorusOPESConfig`: `barrier=8.0`, `bias_force_clip=60.0` are reduced units → 20–40 kJ/mol and ~400 kJ/mol/rad) / **C** (`run_opes_2d` driver) | `BatchedTorusOPES` consumes only `(phi1, phi2)` and returns `(f1, f2)`; its only imports are generic `density2d`/`poisson2d`. |
| `core_dist.py`, `opes.py`, `__init__.py` | **C** / not needed | |

> **CORRECTION to `reuse-risk` (a claimed defect that is not one).** `reuse-risk` reports an
> "import wart": *"`fr_score_2d` does `from .core import _recentered_clipped_score`, and `core`
> imports `potentials` at module level — so importing `density2d` transitively imports the
> alkane potential."* **Verified false at import time** — the import is *function-scoped*
> (`density2d.py:152`), so `import alkanes.density2d` leaves `alkanes.core` and
> `alkanes.potentials` absent from `sys.modules` (checked live; same for `poisson2d`,
> `periodic`, `cv2d`). Only *calling* `fr_score_2d` pulls them in. The coupling is still worth
> breaking (do it by passing the score function in, not by moving files), but it is not an
> import-time dependency and no refactor of `alkanes/` is required for it.

### 3.3 What must be built rather than reused, ranked by risk

1. `dynamics.py` (BAOAB) — the largest single rewrite; the repo has no momenta anywhere.
2. `bd.py` — birth–death with momenta, cached forces, per-seed streams, families.
3. `cv2d_fast.py` — block-sparse; a **prerequisite**, not an optimisation (§3.4).
4. `projection.py` — Nyquist consistency and the weighted solve (§3.5).
5. `umbrella.py` + `mbar_torch.py` — Stage 1; both largely promoted from validated scratch.

### 3.4 `cv2d_fast.py` — block-sparse den Otter, and why it is mandatory

`alkanes/cv2d.py` **is correct on the 22-atom layout** (measured, `s10_cvcheck.py`):
biorthogonality `5.6e-16`, analytic `div(w)` vs central FD `1.3e-10` relative, Gram never near
singular anywhere on the torus. It is kept as the reference implementation for the equivalence
test. It is **not** usable in production, for one reason: `_grad_hess_full` scatters each 12×12
Hessian into a dense `(B, 2, 3A, 3A)` tensor. At A = 22 that is 66×66 = 8712 doubles per
replica for physics that lives in 15 coordinates. Measured (`rev_mem.py`, GPU 7, f64):

| R | N | B | H tensor | **peak alloc** | `geometry` |
|---|---|---|---|---|---|
| 4 | 1024 | 4096 | 0.27 GiB | 2.35 GiB | 35.1 ms |
| 5 | 4096 | 20480 | 1.33 GiB | 11.59 GiB | 35.9 ms |
| 8 | 4096 | 32768 | 2.13 GiB | **18.54 GiB** | 40.2 ms |
| 8 | 8192 | 65536 | 4.25 GiB | **CUDA OOM** | — |

The peak is **8.7×** the H tensor itself (from `einsum("pbi,pcij,pdj->pbcd")` plus batched
`linalg.inv`/`eigvalsh`). At the production shape (10 seeds × 4096) the dense path does not fit
on GPU 6 (17.5 GiB free) and leaves ~10 GiB of headroom on GPU 7 against a co-tenant that grew
5 GiB during a single review session.

> `integrator-bd`'s conclusion *"16× more walkers is free — run N = 4096 or more"* was measured
> at **R = 5 only**. It is free in *time* (35.9 vs 35.1 ms) and costs **43× in memory**. Both
> halves must be quoted together.

Three exact, verified optimisations (`cv2d_fast.py`, `blocksparse.py`):

1. **Union-block Hessian.** Keep `H` in the 5-atom union `U = {4,6,8,14,16}` (15 coords, not
   66). All contractions (`G`, `tr H`, `T`) are invariant under restricting to any subspace
   containing both gradient and Hessian supports. **≈19–30× less memory traffic.**
2. **Closed-form 2×2 inverse and eigenvalues** (batched cuSOLVER on 2×2 is pathological):
   9.72 ms → **0.53 ms**; errors 1.7e-18 (inv), 4.5e-13 (eig).
3. **Single batched `autograd.grad`** instead of `vmap(grad(·))`: `grad_only` 22.6 → **3.5 ms
   (6.4×)**.

Equivalence, measured against the repo implementation on 64 thermal configs: `phi`/`g`/`G`
**bitwise identical**; `div_v` rel 5.7e-15; `Ginv` 3.6e-16; `cond` 3.5e-15; `local_mean_force`
9.1e-13 abs on a 2.34e3 scale. Independently, the block-sparse `div` reproduces the dense form
at `2.6e-12` (identity) / `3.4e-13` (mass), with `G` exact.

Net: `geometry` 35.5 → **27.3 ms**, `grad_only` 22.6 → **3.5 ms**, memory **30×** lower
(544 MiB → 18 MiB at B = 8192). The residual cost is `vmap(hessian)`; an analytic dihedral
Hessian would take it to ≈10 ms and is an optional later optimisation.

**Two constructor traps, both fixed by fiat:** `JointDihedralCV2D.__init__` defaults to
pentane's `atoms_a=(0,1,2,3)`, `atoms_b=(1,2,3,4)`, `n_atoms=5`; `self.n_atoms` is stored but
never read (`_grad_hess_full` takes `A` from `q.shape`), so a forgotten override does **not**
raise — it silently computes the wrong dihedrals on the right-shaped tensor.
**In `FastJointDihedral2D`, `atoms_a`, `atoms_b` and `n_atoms` are required positional
arguments with no defaults.** Also, `_grad_hess_full` rebuilds `torch.tensor([...],
device=q.device)` inside the per-CV loop on every call (a host→device copy on the hot path) —
cache it as a registered buffer.

**`estimator_stride = 1`** with the fast CV (removes a stride confound entirely). Fallback
`stride = 5` if throughput binds; either way the stride-equivalence test (gate S0.9) is
mandatory.

### 3.5 `projection.py` — the Nyquist defect and the weighted solve

#### 3.5.1 CRITICAL, independently verified: `poisson_projection` returns a `gB` that is not `∇B`

For even `n`, index `n/2` is the self-conjugate Nyquist mode. For real `g`, `ĝ(n/2)` is real
⇒ `divhat = ik·ĝ` is purely imaginary ⇒ `Bhat(n/2)` is purely imaginary ⇒ `.real` in
`B = ifft(Bhat).real` **kills it**; but `gBhat = ik·Bhat` is then *real* and **survives**.
So `gB` carries Nyquist content that `B` does not.

**Reproduced live in this session** (`scratchpad/nyq_check.py`, random `(g1,g2)`, f64):

| `n_grid` | `max|gB − spectral_gradient(B)|` | `rms|gB|` | residual `curl_norm(gB)` |
|---|---|---|---|---|
| 48 (even) | **3.929e-01** | 0.699 | **1.941** |
| 64 (even) | **2.942e-01** | 0.701 | **2.092** |
| 96 (even) | **2.566e-01** | 0.713 | **2.657** |
| **63 (odd)** | **2.109e-15** | 0.713 | **5.666e-15** |
| **97 (odd)** | **3.109e-15** | 0.705 | **1.517e-14** |

This is not a corner case: `mean_force_fields` returns `g = smooth(f_sum)/smooth(count)`, a
**Nadaraya–Watson ratio**. A ratio of two bandlimited fields is *not* bandlimited — division
reintroduces full-band content, so Gaussian smoothing gives no protection. A hard trust mask
reintroduces it too, and sharpens it.

**Three consequences, all load-bearing:**

1. The applied CV-space force is **not curl-free** (measured `curl_norm(gB) = 1.94` at n = 48
   against 5.7e-15 at n = 63). By §4.4 the Cartesian bias force is then not `−∇` of anything
   ⇒ no Gibbs stationary state ⇒ **the ABF fixed point is not `F`** and mFR's target
   `q ∝ exp(−β(F−B))` is the wrong target.
2. `final_pmf = B_raw` — which is saved, compared to the reference in L2, and used as the mFR
   `F_ema` — **is not the potential whose gradient was applied**.
3. `run_frozen_bias_2d` re-differentiates the saved `B` with `spectral_gradient`, so the frozen
   run applies a *different* field than the online run did. Measured online/frozen mismatch
   `max|∇B_saved − gB_online|`: 1.6e-3 (n=48, h=0.20), **0.77 (n=64, h=0.10)**, 0.071
   (n=96, h=0.08), **15.6 kJ/mol/rad = 11.4 % of |gB|max (n=64, unsmoothed)**. Small at
   smoothed settings only because the Gaussian suppresses `k = n/2` by 6e-4…6e-3 — and the
   trust mask, applied *after* smoothing (`core2d.py:93-95`), stamps a sharp 0/1 edge with
   full-spectrum content straight into the FFT input. **Raising `abf_min_count` from 5 to 200
   enlarges and sharpens that edge**, so the recommended estimator change makes this worse.

The design leans on the frozen-bias sign flip as independent confirmation of the starvation
classification (§10). A systematic online/frozen inconsistency is exactly the wrong thing to
have under that claim.

**FIX — belt and braces, both mandatory:**

* **`n_grid` is ODD.** Frozen at **97** (§4.5). Verified exactly consistent above. Zero code
  risk; it is a config choice.
* **`projection.assert_gradient_consistent(B, gB1, gB2, dz1, dz2, tol=1e-12)`** is called
  every time a bias field is built, in *both* the online and frozen paths. It is a ~0.1 ms FFT.
* Additive, default-off kwarg on `alkanes/poisson2d.poisson_projection` (§3.10) that zeroes
  the Nyquist row/column of `Bhat` for even `n`, so the alkane module is fixed for future users
  without changing any existing alkane result.
* Unit test on a **random** `g` (gate S0.10). The existing suite passes only because it feeds
  `g = spectral_gradient(F)`, which has already had its Nyquist mode zeroed.

#### 3.5.2 The trust mask must be a weight, not a hard mask

`core2d._project_bias` sets `g = 0` where `den < min_count` and *then* FFTs. The Hodge
projection is global, so zeros in unvisited cells propagate into the visited region. Measured
on an exactly bandlimited FES (unmasked round-trip 3.6e-14):

| mask | cells kept | rms(ΔB) on Ω_eval(8 kT) | max |
|---|---|---|---|
| drop `F−F_min > 25 kT` | 81.8 % | 0.133 kJ/mol (0.7 %) | 0.514 |
| drop `> 16 kT` | 56.2 % | 0.244 (1.2 %) | 1.319 |
| drop `> 10 kT` | 31.2 % | 0.290 (1.5 %) | 1.377 |

This is **occupancy-dependent, hence arm-dependent** — and mFR's entire purpose is to make the
occupancy differ from ABF's. 0.133–0.290 kJ/mol is comparable to the reference bootstrap SE
(0.162) and is 25–60 % of a 0.5 kJ/mol claimed effect. It enters the paired comparison as a
fake mFR effect.

**PRIMARY — weighted projection.** Solve
`min_B ∫ w(z) |∇B(z) − g(z)|² dz` with `w = den_eff` (§3.5.3), i.e. the normal equations
`∇·(w∇B) = ∇·(w g)`, by conjugate gradients with the constant-coefficient FFT Poisson solve as
preconditioner. 20–30 CG iterations, each two FFTs; measured FFT cost 0.84 ms per call at
n_grid 64, so ≈25–50 ms per projection — and projection runs every 50 steps, not every step.
An unvisited cell then contributes **nothing** instead of contributing a spurious "the gradient
is zero here" constraint, and there is no mask edge to leak spectrum.

**FALLBACK if the CG solve fails gate S0.11:** the masked FFT projection with a **smoothed**
mask (apply the trust weight *before* the Gaussian smoothing, so the projection input stays
bandlimited), at odd `n_grid`. The fallback must be recorded in the run spec.

#### 3.5.3 Clone-discounted effective counts, `den_eff`

`abf_min_count` calibrated as `(sd(f)/target_sd)²` **assumes independent samples**. Under
birth–death a clone is bit-identical to its parent at birth and needs **6–7 ps** to become an
independent sample of φ (§3.6), while `fr_every` events arrive every 0.5 ps. `den` therefore
over-counts by the clone multiplicity, the trust weight is systematically laxer in the mFR arm,
and that feeds straight back into §3.5.2.

**Fix, cheap and exact.** Ancestor labels are assigned at `t=0` as `arange(N)`; a clone inherits
its parent's label, hence its **parity**. Maintain **two** count accumulators keyed on ancestor
parity, `csum_even` and `csum_odd` (one extra `scatter_add`, ~0.1 ms), and define

```
den      = csum_even + csum_odd                      # raw, as today
den_eff  = 4 * csum_even * csum_odd / max(den, eps)   # clone-discounted effective count
```

For independent samples `csum_even ≈ csum_odd ≈ den/2` ⇒ `den_eff ≈ den`. For a fully cloned
cell (one family, one parity) `csum_odd = 0` ⇒ `den_eff = 0`. `den_eff` is the trust weight
`w` in §3.5.2 and the quantity `abf_min_count` is compared against.

The same two accumulators are the **ancestor-family split-half** that
`results/mfr_mechanism_audit/bias_variance/feasibility.json` records as a *blocking data gap*
("no engine persists per-particle ancestor labels or per-family mean-force accumulators").
Adding them here closes that gap for the third time of asking, at ~0.5 ms/step, and gives the
independent-halves bias/variance decomposition of §9.6 for free.

### 3.6 `bd.py` — birth–death with momenta, forces, families and per-seed streams

#### The corrected rule

```
per seed r, at a scheduled event, on synchronous full-step (q,v) — never inside the splitting:
  fire   ~ Bernoulli(1 - exp(-fr_rate * clamp(S,0) * dt_eff)),  capped at max_events
  src    ~ inverse-CDF draw from clamp(-S,0) over survivors
  q[fire]        <- q[src]
  Fc[fire]       <- Fc[src]                        # MANDATORY, see below
  ancestors[fire]<- ancestors[src]
  family[fire]   <- family[src]                    # parity label, §3.5.3
  v[fire]        <- maxwell(mass, kT, gen_fr[r])   # fresh, SAME mass matrix as the dynamics
  parent src: q, v, Fc all UNTOUCHED
```

**New failure mode that does not exist in the overdamped code:** the BAOAB step caches `F`
between steps. If the clone's cached force is not gathered along with `q`, the cloned replica
performs its next `B` step with the **dead** replica's force. This is silent, does not NaN, and
corrupts exactly the replicas the FR layer touches. `alkanes/core.py` never had to do this
because the overdamped loop recomputes `F` from `q` every step.

#### Fresh momenta: exactly unbiased, and honestly ~5 %

`μ(dq,dp) ∝ e^{−βV(q)}·e^{−½βpᵀM⁻¹p}` is a **product measure**. Assigning `p ~ N(0, M/β)`
**is** the BAOAB O step with `c1 = 0`, which is exactly `μ`-invariant for any `c1 ∈ [0,1]`. No
discretisation error, no order argument.

**Copying the parent's momenta is *also* unbiased** (`p_j ⊥ q_j` under `μ`, and the selection
event depends only on `q_j`). The case for fresh momenta is therefore **not** a bias argument;
it is that copying produces two replicas at the *identical phase-space point*, a pure ESS loss
(at γ=0 they would never separate). **Fresh momenta breaks the degeneracy at zero bias cost —
but it must not be advertised as the mechanism that restores diversity.** Measured
(`e3_decorr.py`, 1024 pairs, γ=1/ps, dt=1 fs):

| timescale | value |
|---|---|
| `τ_v` (mass-weighted atomic velocity) | **10.06 fs** (γ=1); 4.39 fs (γ=10) — essentially γ-independent for γ ≲ 100/ps |
| `τ(dφ/dt)`, `τ(dψ/dt)` | 5.78 / 5.40 fs |
| lineage decorrelation to 90 % of plateau | **6.1–7.2 ps ≈ 6 000–7 000 steps** |
| ratio `τ_p / τ_{φ,ψ}` | **≈ 1/600** |
| fresh-momenta head start | 0.07–0.29 ps (γ=1) / 0.02–0.06 ps (γ=10) = **≤5 % of decorrelation** |
| clone–parent velocity overlap at t = 10 fs, *fresh* momenta | **0.358, not 0** — sharing `q` means sharing `F`, so the common force re-correlates velocities within one vibrational period |

Resampling the *parent* as well is indistinguishable (E[Δφ²] ratios 1.008 / 1.090 at γ=1;
per-pair SE ≈ 4.4 %) and is **rejected** on a different ground: it perturbs a replica not
selected for death, breaking the repo's matched-seed design. With clone-only resampling,
replicas untouched by an event follow trajectories bitwise unaffected by the FR layer.
(`p_clone ← −p_parent` is also exactly `μ`-preserving and doubles the immediate velocity
separation at zero RNG cost, but given the ≤5 % result it is not worth the loss of independence.)

**Design consequence, adopted:** `fr_every` must be judged against **6 ps**, not against `τ_p`.
`fr_every = 5` steps (5 fs) means consecutive events act on an ensemble whose *positions* have
not moved. **`fr_every = 500` steps = 0.5 ps** is frozen (§4.6, §14.9). Ancestor ESS is an
**upper bound** on statistical diversity by a factor set by (time since event)/6 ps — hence the
age-aware ESS of §9.5.

#### Vectorisation — 129× on GPU, and it is 100 % host-sync cost

Measured at R = 5, N = 256, `n_grid = 64`, GPU 7 (`e5_bd.py`):

| implementation | GPU | CPU (16 thr) |
|---|---|---|
| `alkanes/core.py::_birth_death` (per-replica python loop) | **81.09 ms** | 0.211 ms |
| vectorised replacement | **0.630 ms** | 0.381 ms |

Diagnosis on the same GPU: `torch.cuda.synchronize()` = 4.66 ms; `nonzero(...).numel()` D2H =
3.21 ms; `multinomial(256→5)` = 0.234 ms; 200 async elementwise kernels = 5.6 ms total. The
loop performs ≈4 device→host syncs per replica (`birth_w[r].sum() <= EPS`,
`death_w[r].sum() <= EPS`, `torch.nonzero`, `int(di.numel())`) → 5 × 4 × ~4 ms ≈ 80 ms. On a
*shared* GPU every sync waits behind the co-tenant's kernels. Note the CPU crossover: on CPU
the loop is *faster*; production is on GPU, so use the vectorised form unconditionally.

The `core2d.py:299-308` post-event bookkeeping loop (`for r in range(R): … index_select`) must
be vectorised the same way — one `torch.gather` on `has_left_TT`, `rep_roundtrips`,
`prev_basin` using `idx`, and a masked `scatter_add_` over the flattened `(R·N)` axis for the
histograms. Keep `total_repl` on device; `.cpu()` only at `save_every`.

#### Per-seed independent RNG streams — a real defect in the existing code

Measured (`rev_rng.py`, CPU, exact):

```
R=4 batched vs R=1 alone, same score row, same gen_fr seed:  ancestors identical?  False
force seed#0 to 0 events (was 12):  n_repl [12,12,12,12] -> [0,12,12,12]
  seed#1 ancestors identical to baseline?  False
  seed#2 ancestors identical to baseline?  False
  seed#3 ancestors identical to baseline?  False
```

Cause: `core.py:145-159` consumes `gen_fr` **sequentially over `r`, by a data-dependent amount
`n`** (`randperm(n)`, `multinomial(birth_w[r], n)`). Changing anything about seed 0 re-rolls the
birth sources for seeds 1..R−1. Consequences: a seed cannot be reproduced in isolation for
debugging; any change to `R` re-rolls every seed; per-seed results in one batch are not the
per-seed results you would get any other way.

What *does* survive, at full strength: `gen_dyn` draws a **fixed-size** block per step
independent of method, so `abf` vs `fr_*` at identical `(R,N,A)` really are matched. The
coupling is strictly between **seeds**, not between **methods**, and the existing
`abf == fr_* with fr_start_steps=1e9` test is valid.

**Fix, mandatory:** one generator **per seed**, `gen_fr[r] = Generator(device).manual_seed(
rng_seed + 987654321 + 1000*r)`, and every draw **fixed-size**: one `rand(N)` for firing, one
`rand(N)` for inverse-CDF birth selection via `searchsorted` on the row cumsum of `birth_w`.
No `multinomial`, no `randperm`, no data-dependent consumption — and it vectorises at the same
time.

> **Two errors in `integrator-bd`'s vectorised sketch, corrected here.** (a) It omits
> `generator=gen_fr` on `torch.multinomial`, which would silently pull from the global RNG and
> destroy reproducibility outright. (b) It draws `torch.rand(R,N)` twice *and*
> `multinomial(·, N)` — a completely different consumption pattern that reproduces no existing
> FR result. That is acceptable **only because `src/alanine/` is a new namespace**: no alkane,
> WCA or EB artifact is touched. The handoff must state that the alanine birth–death is not
> bit-comparable to any prior FR run.

#### Non-finite guard — one bad replica destroys an entire seed, permanently

Measured (`rev_nyq.py`): injecting `nan` into **one replica's `f1` of one seed** at R=4, N=512,
`n_grid=64` gives non-finite grid cells in `B` per seed of `[0, 0, 4096, 0]` — **4096 of 4096
cells** of that seed. Separable smoothing and the FFT are global; `f1s/f2s/csum` are running
accumulators, so it is **permanent**. Other seeds are unaffected. The existing guard does not
help: `torch.clamp([1, nan, inf, -inf], -480, 480) = [1.0, nan, 480.0, -480.0]` — the
`core2d.py:196` clamp sanitises infinities and **passes NaN straight through**. The repo's only
other guard is `had_nan` in the final npz, i.e. you discover it after burning the run.

**Mandatory, before any accumulation:**

```python
bad   = ~torch.isfinite(f)                        # device-side, no sync
f     = torch.where(bad, torch.zeros_like(f), f)  # REJECT the sample (do not clamp)
n_bad += bad.sum()                                # read at save_every; abort with the step index
```

and separately **quarantine** the offending replica (resample `q, v` from a healthy sibling in
the same seed) rather than letting it keep emitting NaN every step. This matters more here than
for pentane: bond constants reach 476 976 kJ/mol/nm², and a strained seed explodes instantly.

### 3.7 `AlaRunSpec` — identity, resume safety, manifest

Frozen dataclass, own namespace, own hash. `spec_hash = md5(json.dumps(asdict(spec),
sort_keys=True))[:12]`; run id

```
ala__{stage}__{name}__{init_mode}__N{n_replicas}__T{n_steps}__ns{len(seeds)}__{spec_hash}
```

The `ala__` prefix cannot collide with any existing id (all existing ids begin with a stage
name: `production__`, `tuning__`, `smoke__`, `pilot__`, …).

Fields, at minimum: `stage, name, method, init_mode, seeds, rng_seed`;
physics `temperature_K=300.0, gamma_ps=1.0, dt_ps=1.0e-3, hydrogen_mass=None (str "none"),
forcefield="amber14/protein.ff14SB.xml", param_sha256, ff_xml_sha256`;
estimator `n_grid=97, abf_bandwidth=0.08, kde_bandwidth=0.15, abf_min_count=200.0,
abf_force_clip=200.0, projection_mode ("weighted"|"masked"), estimator_stride,
abf_warmup_steps, estimator_burn_in_steps, project_every`;
FR `fr_rate, fr_every=500, fr_start_steps, score_clip=2.0, target_ema_rate=0.005,
max_event_fraction, clone_momenta ("fresh"), ess_window_steps=6000`;
run `n_steps, n_replicas, save_every`; `core_version="alanine_v1"`.

**Manifest per `.npz`** (reproduce `alkanes/jobs._base_out` verbatim, plus): `run_id`,
`spec_hash`, `spec_json`, `kind="ala2d"`, `stage`, `name`, `method`, `init_mode`, physics
scalars, `n_steps`, `n_replicas`, `seeds`, `runtime_seconds`, `wall_seconds`,
**`ms_per_step_measured`** (required for the equal-wall-clock comparison, §9.2), `device`,
**`cuda_visible_devices`**, `had_nan`, `n_bad_samples`, `core_version`, `param_sha256`,
`git_commit`, `per_seed` (JSON list), `projection_mode`, `nyquist_consistency_max`.

Resume: `run_is_valid(path)` = file loads AND has `per_seed` AND `not had_nan` AND
`n_bad_samples == 0`. `save_run` writes `path.tmp.npz` then `os.replace` (atomic). Runner
supports `--dry-run`, `--max-runs`, `--only-method`, `--overwrite`, `--require-single-gpu`.

**Checkpointing** (`<raw_dir>/_ckpt/<run_id>.ckpt.npz`, ping-pong two generations so a torn
write cannot lose the run; deleted on successful completion). Cadence `checkpoint_every =
10 * save_every` plus a wall-clock floor (write if > 20 min since last). Capture the entire
state **before any RNG consumption in that step**.

| group | entries |
|---|---|
| clock | `step`, `spec_hash`, `run_id`, `torch_version`, `device_type`, `dtype` |
| dynamics | `q (R,N,22,3)`, **`v (R,N,22,3)`** f64. **Do not store `F`** — recompute on resume (bitwise identical on the same device/dtype), which also avoids storing a stale bias force. |
| ABF | `f1s, f2s, csum_even, csum_odd (R,n1,n2)`; recompute `B_raw, gB1, gB2` (cheaper and exact) |
| FR | `F_ema`, `p_ema`, `ancestors (R,N) int64`, `family (R,N) uint8`, `total_repl (R,)`, `birth_hist`, `death_hist`, `anc_reset_step` |
| bookkeeping | `joint_hist`, `prev_basin`, `has_left (R,N) bool`, `rep_roundtrips`, `trans_counts`, `trans_matrix`, `first_discovery` |
| diagnostics | `cond_sum_g`, `cond_max_g`, `lam_min_min_g`, `n_cfg`, `score_std_sum`, `score_absmax`, `n_score`, `n_bad_samples`, `n_clip_hits` |
| time series | every list in `diag`; on resume truncate to `steps <= ckpt_step` |
| **RNG** | `gen_dyn.get_state()`, **every** `gen_fr[r].get_state()`, **and** `torch.cuda.get_rng_state()` + `torch.get_rng_state()` |

> **`engineering` S5, adopted.** `Umbrella2._mk` uses `torch.randn_like(v)` — the **global**
> CUDA RNG, not any `torch.Generator`. A checkpoint listing only `gen_dyn`/`gen_fr` does not
> restore an umbrella run at all. Verified good news: `torch.Generator(device='cuda').
> get_state()` is 16 uint8 bytes and round-trips exactly through npz → `set_state`;
> `torch.cuda.get_rng_state()` is also 16 bytes and **does** reproduce a `torch.compile`'d
> `randn_like` step exactly (`torch.equal` over two chained compiled steps: True). So compile
> is not an obstacle either way. Store all four.

> **Determinism caveat, shortened deliberately.** `scatter_add_` on CUDA accumulates in
> nondeterministic order, so a resumed run is *statistically* but not *bitwise* identical.
> Measured at the production shape (R=8, N=4096, n_grid=96, f64, 6 identical calls): max
> pairwise diff **2.842e-14** on a field max of 295.6 ⇒ rel **9.61e-17**; projected drift over
> 500 k steps ≈ 2e-11 absolute — **11 orders below the 0.16 kJ/mol reference SE**.
> `torch.use_deterministic_algorithms(True)` costs 3.4× on this op (0.078 → 0.263 ms/call).
> **Leave it off.** One line in the docstring; do not let a reader mistake this for a real
> uncertainty.

### 3.8 No-reference-leakage — structural, plus a declared hyperparameter channel

`assert_no_reference_leakage(method, oracle)` is the **first statement** of the sampler: it
raises `AssertionError` if a non-oracle method receives a reference and `ValueError` if
`fr_oracle` gets none. OPES has **no oracle argument at all**.

> **`physics` #9, adopted as a declaration.** The structural guard checks the *method object*.
> It cannot see that `n_grid` was chosen from the measured spectral content of `F_ref`,
> `abf_bandwidth` from the measured basin curvature, `kde_bandwidth ≈ σ_φ`, `N` from the
> post-ABF density, and `abf_force_clip = 200` because `max|∇F_ref| = 88`. These are shared
> across arms, so the **paired** ABF-vs-mFR comparison is safe — but every *absolute* statement
> ("ABF is not starved", "L2 = 15.8 %") is oracle-informed and is not a deployable number.
> The handoff must carry a section distinguishing **structural leakage (guarded)** from
> **hyperparameter leakage (declared, not guarded)**, and list exactly the five knobs above.
> `abf_force_clip` is additionally neutralised by gate S0.8 (it must never bind, so its value
> carries no information).

### 3.9 Startup assertions (fail fast, before any dynamics)

1. `param_sha256` matches the frozen value; `ff_xml_sha256` matches.
2. `getNumConstraints() == 0`; force-class set exact; 22 particles; `NoCutoff`.
3. `n_grid` is odd.
4. `assert_gradient_consistent` on a random field, `< 1e-12`.
5. Chirality `< 0` for every initial configuration.
6. CV convention: reference minimum within 10° of (−78.8°, +56.2°).
7. `torch.cuda.mem_get_info()` free ≥ 1.5 × computed requirement, else abort with the
   required shape in the message (never OOM 40 minutes in).
8. `abf_warmup_steps < fr_start_steps` (see §3.10 latent bug 1).
9. `torch.backends.cuda.matmul.allow_tf32 is False` and
   `torch.get_float32_matmul_precision() == 'highest'` — verified currently true in this build,
   but **pin it explicitly**: a stray `set_float32_matmul_precision('high')` anywhere in the
   process turns the rank-5 MBAR factorisation from a 6e-5 kT nuisance into a real bug.

### 3.10 The only two touches to `src/alkanes/`, both additive and default-off

Precedent: `run_sampler_gpu(..., track_crossings=False)` — a default that keeps prior runs
byte-identical.

1. `poisson2d.poisson_projection(g1, g2, dz1, dz2, nyquist_safe: bool = False)`. When True and
   `n` is even, zero `Bhat[..., n1//2, :]` and `Bhat[..., :, n2//2]` after the zero-mode line.
   Default `False` preserves every existing alkane/WCA result bitwise. Alanine passes `True`
   *and* uses an odd grid.
2. `geometry.remove_com(q, mass=None)`. `mass=None` keeps the existing unit-mass behaviour
   exactly; alanine passes the physical mass vector.

Everything else in `src/alkanes/` is read-only.

**Two latent bugs in `core2d.py`, to be pre-empted in the new driver (not fixed in place):**

* `core2d.py:220` sets `B = abf_scale * B_raw` (ramped) while `F_ema` at `:208-211` accumulates
  the **un-ramped** `B_raw`. The mFR target `q ∝ exp(−β(F_ema − B))` therefore mixes ramped and
  un-ramped fields. Inert at the current defaults (`abf_warmup_steps = 5000 <
  fr_start_steps = 10000`) but silently wrong if anyone sets `fr_start_steps <
  abf_warmup_steps`. Assert the ordering (§3.9 item 8) and use the same scaling for both.
* `reg_threshold = 1e-8`, `ridge = 0.0`, `adaptive = 1e-6*eye` in `cv2d` are **dimensional**,
  hence silently unit-dependent. Alanine `λ_min ∈ [98, 2024] (rad/nm)²` — 10 orders above the
  threshold. The guard never fires and `reg_activations` will always report 0. Harmless here;
  make it **relative** (`λ_min < 1e-10 · λ_max`) in `cv2d_fast` before anyone switches to Å.

---

## 4. Mean-force estimator

### 4.1 Which den Otter dual — DECISION: the **identity metric**. Do not switch.

`w_a = Σ_b (G⁻¹)_{ab} g_b` with `G_ab = g_a·g_b` (unit metric), i.e. exactly what
`alkanes/cv2d.py` computes.

**Structural fact (checked, not assumed):** `∇φ_a` is supported on *exactly* the 4 atoms of
dihedral `a`, so **no hydrogen carries any weight at all**. The union `{4,6,8,14,16}` is
`C, N, CA, C, N` with masses `{12.01, 14.01, 12.01, 12.01, 14.01}`. `M⁻¹` therefore acts on the
relevant subspace as `≈(1/13)·I` plus a 17 % C/N contrast, and `w_M ≈ w_I`.

**Measured variance** (per-node conditional variance, 900 nodes × 2 components, 2.70e6 thermal
configs from the restrained scan):

| metric | pooled Var(f) | sd(f) | ratio (mass/identity) |
|---|---|---|---|
| identity | 2617 | **51.2** kJ/mol/rad | 1 |
| mass | 2686 | 51.8 kJ/mol/rad | **1.0264** |

Per-node/component ratio p5/p50/p95 = 0.9876 / 1.0312 / 1.0691. Autocorrelation-aware block
means: L=5 frames (50 fs) ratio **1.0040**; L=20 frames (200 fs) ratio **0.9522**;
`τ_int ≈ 2 frames = 20 fs` for both. Independently corroborated by an earlier OpenMM-Reference
24×24 pass: pooled ratio **1.0278**, p50 1.0338.

**Reason for the decision.** The mass metric is 2.6 % *worse* on instantaneous variance and
4.8 % better on 200-fs block means — a wash, well inside how you would tune anything else. It
costs an extra tensor multiply, adds a configuration knob, and **couples the estimator to the
mass matrix** (CV-atom masses `{C 12.01, N 14.01, CA 12.01}` without HMR vs
`{12.01, 11.018, 9.018}` at `hydrogenMass = 4` — a 17 % → 33 % contrast swing, so toggling HMR
would silently change the estimator). We choose no HMR (§2.4), which weakens the third
argument but not the first two. **If a mass metric is ever implemented, `M` must be an explicit
*physical*-mass constant decoupled from the dynamics mass matrix** — and, per §2.4 Step 4, it
needs **no Fixman term**.

The mass-metric derivation is recorded in §4.2 so that a future implementer does not re-derive
it, and because the *Fixman* correction to the record is itself a deliverable of this review.

### 4.2 The mass-metric dual, in implementable index notation (recorded, not used)

For any constant SPD **diagonal** `M` (per-coordinate mass `m_i`, `i = 3·atom + c`):

```
ĝ_a   := M⁻¹ g_a                          (ĝ_a)_i = (g_a)_i / m_i
G^M_ab := g_aᵀ M⁻¹ g_b = g_a·ĝ_b = ĝ_aᵀ M ĝ_b
w_a    := Σ_b (G_M⁻¹)_ab ĝ_b              ⇒  w_a·g_c = δ_ac    (exact)

div(w_a) = Σ_b (G_M⁻¹)_ab · Lap_M(φ_b)
         − Σ_{b,c,d} (G_M⁻¹)_ac (G_M⁻¹)_db [ T^M_bcd + T^M_bdc ]
Lap_M(φ_b) = tr(M⁻¹ H_b) = Σ_i (H_b)_ii / m_i
T^M_bcd    = ĝ_bᵀ H_c ĝ_d
∇V·w_a     = − Σ_b (G_M⁻¹)_ab (F·ĝ_b),     F = −∇V
f_a        = ∇V·w_a − β⁻¹ div(w_a)
```

`M = I` reproduces `cv2d.py` exactly; the code diff is four substitutions (identical einsum
strings with `ĝ` swapped in). Validated (`ala_cv2d.py`): biorthogonality `4.4e-16`; analytic
`div` vs central FD max rel **6.9e-9** (identity metric: `5.6e-16` / `1.2e-8`).
**No Fixman term** (§2.4 Step 4). The block-sparse form is
`T^M_bcd = (ĝ_b|atoms_c)ᵀ H_c^{12×12} (ĝ_d|atoms_c)`, verified equivalent at 2.6e-12 / 3.4e-13.

### 4.3 Conditioning — `G` is uniformly non-singular, provably

**Analytic result.** `φ` and `ψ` share `{6,8,14}` but atom **4 is private to φ** and atom **16
is private to ψ**, and a dihedral gradient is exactly supported on its own 4 atoms with
`|∇_i φ| = 1/(|r_j−r_i| sin∠(i,j,k)) > 0` for the first atom and `|∇_l ψ| = 1/(|r_l−r_k|
sin∠(j,k,l)) > 0` for the last. Restricting Cauchy–Schwarz to the *shared* block, with
`a = |∇_4 φ|²/|∇φ|²`, `b = |∇_16 ψ|²/|∇ψ|²`:

```
cos²α ≤ (1−a)(1−b)   ⇒   sin²α ≥ 1−(1−a)(1−b) ≥ max(a,b) > 0
det G = |u|²|v|² sin²α ,   λ_min ≥ det G / (|u|²+|v|²) ,
cond(G) ≤ (|u|²+|v|²)² / (|u|²|v|² sin²α)
```

**`∇φ` and `∇ψ` can never be parallel.** The bound is enforced by the ff14SB
`HarmonicAngleForce` on C–N–CA (121.9°) and CA–C–N (116.6°), which keeps the angles ~30° from
linear at all times (`σ_θ = √(kT/k) ≈ 4°`).

**Measured** (324 rigid nodes; 2.70e6 thermal configs at 300 K):

| quantity | rigid scan | thermal (identity) | thermal (mass) |
|---|---|---|---|
| `λ_min` min | 132.6 | **98.01** (rad/nm)² | **7.672** (rad/nm)²/amu |
| `λ_min` p0.01 / p50 | – | 108.5 / 236.2 | 8.445 / 18.35 |
| `det G` min | 5.54e4 | 4.448e4 | 272.5 |
| `cond` p50 / p99 / p99.99 / **max** | 1.09 / – / – / 7.00 | 2.452 / 6.970 / 9.964 / **12.079** | 2.500 / 7.071 / 10.011 / **12.125** |
| `frac(λ_min < 1e-8)` | 0 | **0** | **0** |
| `frac(cond > 100)` | – | **0** | **0** |

`sin²α` actual min **0.4395** vs analytic bound 0.2599 (bound verified ≤ actual everywhere).
`|∇φ| = 16.3–23.8`, `|∇ψ| = 15.9–22.2` rad/nm. Worst node-mean `cond` = 6.93 at (−6°, +6°),
the fully eclipsed cell at `F ≈ 97 kJ/mol`, never visited.

**There is no near-singular region.** `Ginv` is at worst a factor-12 anisotropic rescaling. The
ridge/`reg_threshold` machinery is **dead code for this system** and `reg_activations` will
always report 0 — report that fact rather than treating a zero as evidence of health.

Note `|G₁₂|/√(G₁₁G₂₂)` reaches **0.749** — φ and ψ share N–CA–C, so the **2-D vector
formulation is required**; two independent scalar ABFs would be wrong.

### 4.4 Conservativeness, and exactly what the Hodge projection buys

Applied Cartesian force `Φ_i(q) = Σ_a G_a(ξ(q)) (∇φ_a)_i`. Then

```
∂_j Φ_i − ∂_i Φ_j = (∂_2 G_1 − ∂_1 G_2) · (∇φ_2 ∧ ∇φ_1)_{ij}      [2 CVs]
```

The Hessian terms `Σ_a G_a (H_a)_{ij}` are symmetric in `(i,j)` and cancel identically. Since
`∇φ ∧ ∇ψ ≠ 0` everywhere (§4.3, `sin²α ≥ 0.26`), **the Cartesian force is conservative iff
`curl G = ∂_φ G_ψ − ∂_ψ G_φ = 0` on the torus.**

* The raw ABF field is *not* curl-free: node-wise `RMS curl = 25.95` against
  `RMS|g| = 41.75` kJ/mol/rad²; the projection removes a non-conservative residual of
  `2.07 = 5.0 %` of `|g|`. **Caveat (`physics` #2):** at even `n_grid` part of the reported
  `curl_norm` is residual the projection *fails* to remove (§3.5.1) — at odd `n_grid` the
  figure is clean.
* A spectral `∇B` kills **two** obstructions: the curl part *and* the harmonic (`H¹(T²)=ℝ²`)
  part, a net drift around either cycle. A true single-valued `F` has zero holonomy, so
  discarding the harmonic amplitude is correct, and its size is a free convergence diagnostic
  (report it).
* The biased dynamics then has stationary density `∝ e^{−β(V − B∘ξ)}` and CV marginal
  `∝ e^{−β(F−B)}` — flat at `B = F`. `abf_bias_force_2d` computes exactly
  `+Σ_a (∇B)_a ∇φ_a = −∇_q U_b`. Correct.
* Masking/weighting is applied *before* projection, so the applied force stays an exact
  gradient; only its interpretation changes.

### 4.5 Grid and bandwidths — FROZEN

```
n_grid          = 97        # ODD (§3.5.1); dz = 0.0648 rad = 3.71 deg
abf_bandwidth   = 0.08      # 4.6 deg;  h_eff = sqrt(h^2 + dz^2/12) = 0.0822  (+2.7%)
kde_bandwidth   = 0.15      # 8.6 deg ~ sigma_phi
abf_min_count   = 200.0     # on den_eff (§3.5.3), NOT on den
abf_force_clip  = 200.0     # kJ/mol/rad, applied to the 2-vector MAGNITUDE (§4.6)
n_replicas      = 4096
project_every   = 50
```

**Spectral content** (48-grid): `F` is essentially bandlimited — 99.16 % of power at `|k| ≤ 4`,
99.92 % at `|k| ≤ 6`. `p = e^{−βF}` is much broader: 89.9 % at `|k| ≤ 8`, 96.3 % at 10, 98.5 %
at 12, 99.79 % at 16. Nyquist alone asks only `n_grid ≥ 32`; **the binding constraint is the
cell/bandwidth ratio**, since the scatter step convolves with a top-hat of variance `dz²/12`.
`n_grid = 97` costs nothing (two 97×97 matmuls + one FFT).

**Basin widths**, fitted from measured mean forces at exact grid nodes (no `F` discretisation):
`F_φφ = 105–117` ⇒ **`σ_φ = 0.146–0.154 rad (8.4–8.8°)`**; `F_ψψ = 21–38` ⇒
`σ_ψ = 0.26–0.35 rad`. **The φ well is 2× sharper than pentane's `kde_bandwidth = 0.30`.**

#### The ABF bandwidth bias is deterministic, computable without running anything, and large

Because convolution commutes with `∇`, `K_h * ∇F = ∇(K_h * F)`, so the converged bias is
exactly `B = K_h * F`. **Measured in this session on the real 48² MBAR FES**
(`scratchpad/floor.py`; 8 kT window = 573 cells, range **19.935 kJ/mol**):

| `h` (rad) | rms(`K_h*F − F`) on Ω_eval(8 kT) | max | **% of 8 kT window** | rms on Ω_eval(10 kT) |
|---|---|---|---|---|
| 0.30 (pentane KDE) | 2.2621 | 4.8971 | **11.35 %** | 2.1823 |
| **0.20 (pentane ABF, and the kill-shot pilot)** | **1.1294** | 2.4283 | **5.67 %** | 1.0876 |
| 0.15 | 0.6667 | 1.4430 | 3.34 % | 0.6418 |
| 0.10 | 0.3129 | 0.7512 | 1.57 % | 0.3023 |
| **0.08 (FROZEN)** | **0.2070** | 0.5461 | **1.04 %** | 0.2008 |
| 0.06 | 0.1222 | 0.3537 | 0.61 % | 0.1193 |
| 0.04 | 0.0578 | 0.1797 | 0.29 % | 0.0571 |

(Independently reproduced: `physics` measured 1.141 kJ/mol / 5.7 % at `h = 0.20`. Two
implementations agree.)

**This is a method-independent floor that no amount of sampling and no reallocation can cross.**
At the kill-shot's `h = 0.20` it is **5.67 % of the window against a measured ABF residual of
15.8 %** — 36 % of the residual amplitude (12.9 % of the squared error). Gate G3 of the
`reuse-risk` kill-shot ("room to improve > 10 %") therefore fired partly on kernel bias.
**At `h = 0.08` the floor is 1.04 %**, comfortably below any effect worth claiming.

ABF *variance* is not the constraint at production scale (`sd(f) = 51.2`, `τ_int = 40 steps`;
a footprint accumulates > 2e4 independent samples within a few thousand steps), so pushing
`h_abf` down is nearly free.

#### The Nadaraya–Watson ratio is occupancy-dependent — the second reason `h = 0.08`

The claim *"the converged bias is exactly `B = K_h*F`"* holds **only when the biased occupancy
`ρ` is exactly uniform**. In general `g = K_h*(ρ∇F)/(K_h*ρ)`, and **mFR's entire purpose is to
make `ρ` differ from ABF's**. Measured arm-to-arm spread on the real 48² MBAR FES:

| `h` (rad) | rms(B−F), `ρ` uniform | `ρ ∝ e^{−β·0.5·(F−B)}` | **arm-to-arm spread** |
|---|---|---|---|
| 0.20 | 1.141 | 1.548 | **0.407 kJ/mol = 2.0 % of window** |
| 0.15 | 0.673 | 1.051 | 0.378 = 1.9 % |
| 0.10 | 0.310 | 0.551 | 0.241 = 1.2 % |
| **0.08** | 0.197 | 0.366 | **0.169 = 0.8 %** |

A nominal "15 % relative mFR improvement" on a 15.8 % residual is 2.4 % of the window ≈
0.5 kJ/mol. **At `h = 0.20` the confound is 0.407 kJ/mol — essentially the whole effect.** At
`h = 0.08` it is 0.169, i.e. ~34 % of a claimed effect, which is why §9.3 additionally reports
the **kernel-matched** metric `‖B_arm − K_h*F_ref‖` (exact, free) so the kernel is matched on
both sides of the comparison.

#### KDE occupancy and the minimum `N`

For a 2-D product Gaussian, `R(K) = ∫K² = 1/(4π) = 0.0796`, so
`sd(log p̂) = √(R(K)/(N h² p)) = 1/√n_eff` with `n_eff = 4π N h² p`. The correct `p` is the
**post-ABF near-uniform** density `p ≈ 1/(4π²) = 0.02533 rad⁻²`, not `p_max` — once ABF
flattens, every cell is equally starved. Then `n_eff = N h²/π`:

| criterion | `h = 0.10` | `h = 0.15` | `h = 0.20` |
|---|---|---|---|
| `sd(log p̂) ≤ 0.2` (`n_eff ≥ 25`) | N ≥ 7854 | **N ≥ 3491** | N ≥ 1963 |
| `sd(log p̂) ≤ 0.1` (`n_eff ≥ 100`) | N ≥ 31416 | N ≥ 13963 | N ≥ 7854 |

At `N = 4096, h = 0.15`: `n_eff = 29.3`, `sd(log p̂) = 0.185`. Pre-convergence in the C7ax basin
(`p ≈ 0.05 rad⁻²`, 3 % of population over ~40 cells): `n_eff = 58`, `sd = 0.13`. With
`score_clip = 2.0`, 0.185 of score noise is 9 % of the clip.

> **Minimum-N statement.** At `h_kde = 0.15` the 2-D KDE is trustworthy (`sd(log p̂) ≤ 0.2`)
> only for **N ≳ 3500**; use **N = 4096**. `N = 2048` is the hard floor and only at `h ≥ 0.20`.
> Below `N ≈ 1000` the log-ratio score at any `h ≤ 0.2` is dominated by KDE noise
> (`n_eff < 12`, `sd > 0.29`) and mFR birth–death becomes noise-driven.

Two nuances, recorded: (i) the KDE smoothing bias in the score is `≈(h²/2)(∇²log p + |∇log p|²)`
— **not** `(h²/2)∇²log p` as `mean-force` wrote — and both terms **vanish at the mFR fixed
point** (`B → F` ⇒ `q → uniform` ⇒ `p → uniform`), so it distorts only the transient;
(ii) `h_kde = 0.15 ≈ σ_φ` means mFR cannot resolve density structure finer than the unbiased
C7eq well, which is acceptable precisely because ABF has already broadened it.

### 4.6 Clipping and support floors — the pentane defaults are wrong here

**(a) `abf_force_clip`: 60.0 → 200.0, and clip the MAGNITUDE, not the components.**

Pentane runs in reduced units (`AlkaneParams: beta=1, epsilon=1, d0=1`) where `dF/dφ = O(5)`
and 60 is ~12× the natural scale, i.e. inert. Alanine is kJ/mol·rad:

* conditional mean `|∇F|`: **p50 16.45, p90 50.96, p99 76.78, max 88.05 (φ) / 59.84 (ψ)**;
* **`clip = 60` binds on 11.8 % of torus cells** (12.5 % on the 48-grid), 10.4 % of the
  thermally accessible `F < 25 kT` region. `clip = 80` → 1.1 %; `clip ≥ 100` → **0.00 %**.

A binding clip is a **correctness** bug, not a tuning issue, for three reasons:

1. **It destroys conservativeness.** `clamp` is applied *per component* to `(gB1, gB2)`
   (`core2d.py:222-223, 367-368`). Clipping one component and not the other injects
   `curl G ≠ 0` **after** the Poisson projection whose only purpose was to remove it — and by
   §4.4 the Cartesian force is then not `−∇` of anything ⇒ no Gibbs stationary state ⇒ the ABF
   fixed point is not `F` and mFR's target is wrong.
2. It caps the restoring bias exactly on the transition ridges, so replicas pile against the
   forbidden wall and the marginal never flattens.
3. The estimator pre-clip `clamp(f, ±8·clip)` (`core2d.py:196`) biases the conditional mean
   wherever the tails are asymmetric — and they are, near steric walls.

**FROZEN:** `abf_force_clip = 200.0`, applied as `v ← v · min(1, clip/|v|)` on the 2-vector so
direction is preserved and the field remains a scaled gradient along each streamline. The
pre-accumulation guard becomes a **finite filter that rejects the sample** (§3.6), not a clamp.
Measured instantaneous `|f_loc|`: p50 39.4, p99 153.7, p99.9 207.3, p99.99 308.0, max 2665
(sd 51.2); exceedance of `8·clip = 1600` is **7.4e-7** and of 3200 is 0 — so the blow-up guard
is a pure guard, which is its intent. **Gate S0.8 asserts the clip never binds**
(fraction of bias-force evaluations at the clip < 1e-4), which also neutralises the
hyperparameter-leakage objection of §3.8.

**(b) `abf_min_count`: 5.0 → 200.0, evaluated on `den_eff`.**
`den` is the *unnormalised* smoothed count ≈ the raw sample count in the kernel footprint.
With `sd(f) = 51.2`, 5 counts gives `sd(g) ≈ 23 kJ/mol/rad` — a quarter of the maximum true
mean force, injected straight into the projection. Scale-free criterion
`min_count ≥ (sd(f)/target_sd(g))²`; target `sd(g) ≤ 4` ⇒ **165 ⇒ use 200**. It is reached
within tens of steps in the explored region, so it costs nothing. It is applied to `den_eff`
(§3.5.3) and as a **weight**, not a hard mask (§3.5.2).

**(c) The geometric term is NOT negligible — keep the Hessian.**
Node-mean `|−β⁻¹ div w|`: p50 **1.016**, p99 3.297, max 3.643 kJ/mol/rad = **5.6 % of `|∇F|`
at the median node and > 100 % in the flat regions**. Its per-sample sd is only **1.48** against
**59.1** for `∇V·w`, so it costs essentially zero variance while carrying a real systematic.
Dropping it (the tempting "no-Hessian" shortcut) would bias the mean force by ~1 kJ/mol/rad
everywhere.

**(d) `fr_every`: 5 → 500 steps (0.5 ps).** Against the measured 6–7 ps lineage decorrelation
(§3.6). At `fr_every = 5` the effective resampling frequency is ~1200 events per decorrelation
time and consecutive events act on positionally frozen ensembles.

**(e) `basin_barrier = 61.6°`** (pentane's gauche/anti split) is meaningless here — replaced by
Ramachandran boxes (§9.4).

### 4.7 Bias/variance budget of the estimator at the frozen settings

| contribution | magnitude on Ω_eval(8 kT) | `N`-dependence | mFR-touchable? |
|---|---|---|---|
| kernel bandwidth, `h_eff = 0.0822` | **0.207 kJ/mol = 1.04 %** | none | **no** — subtract it (§9.3) |
| NW-ratio occupancy term | ≤ 0.169 kJ/mol = 0.8 % | none | **it IS the confound** — kernel-match |
| trust-weight / projection nonlocality | ≈0 with the weighted solve; 0.13–0.29 kJ/mol if masked | weak | partly |
| statistical (mean force) | `SE(g) ≈ 0.20` raw / `≈0.4` after `τ_int` ⇒ ≪ 0.1 kJ/mol on `F` | `∝ N^{-1/2} t^{-1/2}` | yes |
| **shared-accumulator transient** | **the dominant term**: 15.8 % at 50 ps, `∝ t^{-0.47}` | **none** | **yes, via per-bin counts (§1.3)** |

≳97 % of the *local* mean-force error at the kill-shot settings was deterministic
(29.5 % of `RMS|∇F| = 41.75` ⇒ ≈12 kJ/mol/rad, against a statistical floor of ≈0.4). That is
why the residual is `N`-flat, and why `N`-flatness is a statement about the accumulator rather
than about mFR (§14.1).

---

## 5. Stage 0 validation gates

Every PASS criterion is a number. Gates V1–V6 are **already measured and passing**; V7–V16 are
required before any Stage-1 or Stage-2 run. Mirror the alkane suite's structure
(`tests/test_alanine_*.py`, CPU-only via `CUDA_VISIBLE_DEVICES=""`).

| # | gate | PASS criterion | status |
|---|---|---|---|
| V1 | OpenMM ↔ torch **energy** parity, 24 thermal configs spanning E ∈ [1923, 4823] kJ/mol | rel < 1e−8 | **PASS 1.04e−9** |
| V2 | OpenMM ↔ torch **force** parity, same configs | rel (Frobenius) < 1e−8 | **PASS 3.03e−10** |
| V3 | dual biorthogonality `w_a·g_c = δ_ac` at 22 atoms | < 1e−12 | **PASS 4.4e−16** |
| V4 | analytic `div(w_a)` vs central finite differences | rel < 1e−8 | **PASS 1.8e−10** |
| V5 | Gram conditioning over 2.70e6 thermal configs | `frac(cond > 100) = 0`, **zero** ridge activations | **PASS** (λ_min ≥ 98.0, cond_max 12.08) |
| V6 | CV value vs independent numpy dihedral | < 1e−10 deg | **PASS 1.4e−14** |
| V7 | **chirality** — reference FES global minimum location | within one grid cell of **(−78.8°, +56.2°)**. A mirrored build gives (+78.8°, −56.2°) and silently inverts every conclusion (§2.9) | pending |
| V8 | **Nyquist consistency** — `max abs(gB − spectral_gradient(B))` on a **random** `g` | < 1e−12 | **PASS** — fixed `c6a6718`, measured 1.8e−15; regression tests on random `g` |
| V9 | matched seeds — `abf` vs `fr_*` with `fr_start_steps = 1e9` | agree to 1e−9 | **the single most valuable test**; mirror from alkanes |
| V10 | estimator stride 1 ≡ stride 5 | within seed noise | mirror |
| V11 | whole-configuration cloning, fixed `N`, no source/target aliasing | exact | mirror **and extend to velocities** — the current test would pass a buggy position-only clone (§3.6) |
| V12 | no-reference-leakage | non-oracle **raises** given a reference; `fr_oracle` **raises** without one; OPES has no oracle argument | mandatory; first statement of the sampler |
| V13 | thermostat | ⟨T⟩ within **2 %** of 300 K over ≥ 20 ps | measured 290–296 K; pre-register the mild BAOAB cooling rather than discovering it later |
| V14 | NeRF builder round-trip | `build_positions(φ,ψ)` → `signed_dihedral` recovers (φ,ψ) to 1e−10 deg | pending |
| V15 | **umbrella seed strain**, every window | `HarmonicAngleForce < 50 kJ/mol` **and** max angle deviation `< 15°` | **PASS 576/576** — rigid-rotation seeding, `src/alanine/system.py` (the 61.8 % failure was the rejected NeRF path) |
| V16 | rigid-rotor mean force | at a frozen non-CV geometry, den Otter `f_a` equals the analytic derivative of the ff14SB torsion sum along φ/ψ to 1e−8 | replaces the alkane `decouple` gate |
| V17 | **per-seed RNG isolation** | forcing seed 0 to zero birth–death events leaves seeds 1..R−1 bit-identical | **FAILS TODAY** (§3.6); breaking fix |
| V18 | **NaN containment** | injecting NaN into one replica of one seed leaves the other seeds' `B` finite **and** aborts the offending seed within one `save_every` | **FAILS TODAY** — currently contaminates 4096/4096 cells silently (§3.6) |

There is **no exactly-solvable alanine limit**, so the repo's B0/P0 "recover `V4 ⊕ V4`" gate has no
analogue. V1/V2 (OpenMM parity) plus V16 (rigid-rotor) replace it. State this explicitly in the
handoff rather than leaving the missing gate unremarked.

---

## 6. THE KILL-SHOT — runs FIRST, before any production spend

**Purpose.** Decide, for ≈1.5 GPU-hours per candidate system, whether that system is in a regime where
mFR could possibly help — before committing ~30 GPU-hours of production. This is the single most
valuable procedure produced by this design pass and it generalises to every future system.

### 6.1 Prerequisites — without these the gate measures the wrong thing

The first pass of this experiment returned a confident negative that **does not survive review**
(§14.1). Three of its four inputs were defective. Do not re-run it until all four hold.

1. **Reseed the reference.** `build_positions` must be deleted from the seeding path (§7.1) and every
   window must pass V15. The first pass had 61.8 % of windows in a ~436 kJ/mol strained trap.
2. **Fix the accumulator.** Add exponential forgetting (`f1s ← (1−λ) f1s + …`) or discard the first
   half, so `L2 → L2_stat` as `t → ∞`. Without this the N-ladder measures accumulator staleness.
3. **Subtract the analytic deterministic floor.** `L2_floor = ‖K_h * F_ref − F_ref‖`, computable with
   **no simulation**. Measured on the 48² MBAR FES at the first pass's settings: **1.14 kJ/mol = 5.7 %
   of the 19.9 kJ/mol 8 kT window** at `h = 0.20`, 6.0 % at `n_grid = 36`. Against the reported
   `L2 = 15.8 %`, **≈40 % of the "room to improve" was pure kernel bandwidth** that neither mFR nor
   more sampling can remove.
4. **Validate the instrument on a positive control.** Run the identical N-ladder on a WCA cell where
   mFR is known to help (`b1_h2`, −34 % MSE). If a *help* cell also returns N-flat, **G4 is refuted as
   an instrument** and no N-ladder null is admissible anywhere in this project (§14.1).

### 6.2 Jobs

Common physics: exactly §2 (ff14SB vacuum, BAOAB, `hm = 3.0`, `dt = 2 fs`, `γ = 1 ps⁻¹`, float64,
IUPAC convention). **The reference and every arm share `(forcefield, M, dt, γ, integrator, dtype)`** —
non-negotiable, see §7.2.

| job | settings | cost |
|---|---|---|
| **A** reference FES | §7 protocol, reduced to 24×24 × 4 copies for the screen | ~0.15 h |
| **B** unbiased control | 4 × 512, 50 ps, no bias | ~0.26 h |
| **C** 2-D ABF, large N | 4 × 512, 50 ps | ~0.27 h |
| **D** 2-D ABF, small N | 4 × 64, 50 ps — the N-ladder arm | ~0.28 h |
| **P** positive control | the same C/D ladder on WCA `b1_h2` | ~0.3 h |

### 6.3 Gates

| gate | PASS to proceed | first-pass result (alanine) |
|---|---|---|
| **G0** reference quality | ≥ 2 **genuinely independent** estimators agree to < 5 % of thermal-window range, **and** V15 passes for 100 % of windows | 4.2–5.2 % but on a contaminated reference ⇒ **VOID** |
| **G1** metastability | unbiased 2nd-basin occupancy < 0.5 × reference | **PASS** — 0.000000 vs 0.0298 over 102 ns aggregate |
| **G2** not a discovery problem | median first entry < 10 % of run | **PASS** — 0.8–1.4 ps of 50 ps; 512/512 replicas visit. **This is not R15.** Use basin-box occupancy, never `φ > 0`: `n_ever_φ>0 = 181/512` in the unbiased run is a false positive from transient ±180°-seam touches |
| **G3** room to improve | `(L2 − L2_floor) > 10 %` of range **and** final-quarter decrease > 10 % | 15.8 % and 21 % raw; **≈9.5 % after floor subtraction ⇒ MARGINAL/FAIL** |
| **G4** N-scaling | `L2_stat(N/8)/L2_stat(N) > 1.5` (an `N^-1/2` law gives 2.83) **after** §6.1 (1)–(4) | 0.86 measured — but **the instrument is unvalidated**; demoted from *decisive* to *informative* (§14.1) |
| **G5** reference headroom | ABF L2 > **3×** reference self-consistency on **both** `F` and `∇F` | 3.8× on `F` but only **1.5× on ∇F** ⇒ **FAIL**. Needs 48×48 windows, ≥ 100 ps/window, two spring constants (κ = 200 and 400) to bound stiff-spring bias |

**Verdict discipline.** A system proceeds only if G0, G1, G2, G3 and G5 pass. G4 informs but does not
decide until §6.1(4) validates it. **For vacuum alanine (φ,ψ) the study does not proceed** — but the
reason to state publicly is §8, not G4.

---

## 7. Stage 1 — the reference free-energy surface

### 7.1 Seeding — the blocking defect, and the fix

`ala22.build_positions` **must be removed from the seeding path**, not merely warned about.
`X[5] = nerf(X[8], X[6], X[4], 1.229, 122.9, 180.0)` places the ACE carbonyl O from the wrong
reference frame over a large part of the torus, inverting the sp² centre.

Measured on the 24×24 lattice with the exact first-pass recipe (NeRF init → restrained minimisation
at κ = 2000):

```
window E after restrained minimisation:  min −81.7   median 436.1   max 10512.3 kJ/mol
windows stuck in the strained trap (>Emin+200):  356/576 = 61.8 %
```

At a trapped window (−180°, −180°): `HarmonicAngle = 528.7` vs `15.8` at a good one. Worst offenders:
**ACE CH3–C–N at 163.7°** (θ₀ 116.6°, 197.7 kJ/mol) and **O–C–N at 79.4°** (θ₀ 122.9°, 192.8 kJ/mol) —
the carbonyl O and the methyl have swapped sides of the sp² plane. **Minimisation cannot repair an
inverted planar centre.**

The danger is that the strain is nearly (φ,ψ)-**independent** (median 436, p90 458), so it produces a
smooth, plausible-looking FES whose residual ±20 kJ/mol variation is **≈8 kT of spurious structure** —
the size of every feature being measured.

**Fix:** seed with `seed_at` — rigid dihedral rotation of the single verified minimum, which preserves
chirality and planarity by construction — then gate every seed on **V15** before any dynamics.
Chirality is safe at the default `cb_offset = −120` (measured uniformly −2.249 … −1.977), but the sign
is a free parameter of the builder and the force field is exactly reflection-symmetric, so a flip in a
subset of windows would silently **symmetrise `F_ref` and erase the C7eq/C7ax asymmetry**.

### 7.2 Protocol — FROZEN

| item | value |
|---|---|
| restraint | **`κ[(1 − cos(φ − c_φ)) + (1 − cos(ψ − c_ψ))]`** — smooth everywhere. The wrapped-harmonic `min()` form has a **2κπ force discontinuity** at the antipode and is rejected |
| effective stiffness | `1 − cos x ≈ x²/2` ⇒ `k_eff = κ` |
| grid, κ | **24×24 = 576 windows (15.0° spacing), κ = 200 kJ/mol/rad²** |
| von Mises widths | 12.96° / 9.11° / **6.42°** at κ = 50 / 100 / **200** |
| copies | **16 per window ⇒ B = 9216**, each seeded from a *different* structure (§7.5) |
| schedule | warmup 10 k @ 0.5 fs + equilibration 50 k @ 2 fs (100 ps) + production 500 k @ 2 fs (1 ns/copy) = **560 k steps**; 16 ns/window; 9.22 µs aggregate |
| cost | **0.37 h GPU** (2.37 ms/step at B = 9216, float64, compiled) / 6.6 h CPU single-process / **2.3 h CPU wall** at 4 concurrent processes |
| solver | MBAR, **Anderson/DIIS m = 8** (§5 of the engineering review): 80 iterations vs 1175, 4.1 s vs 58.1 s, agreeing to 1.1e−8 kT |
| `u_kn` | the restraint depends **only** on the CVs, so full coordinates are never stored. The rank-5 factorisation is fp32-safe (max 6.3e−5 kT at κ = 200) |
| eval mask | `Ω_eval = {F_ref − F_min ≤ 8 kT}` — splits into **exactly two** connected components (φ<0 megabasin, 484 cells; C7ax island, 86–89 cells) with **zero** interior holes |

**Overlap PASS criterion.** On the torus-periodic 4-neighbour window lattice, **every** nearest-neighbour
pair whose *both* centres lie in `Ω_eval` must have `O_kl ≥ 0.03`, and the median NN overlap over all
pairs must be ≥ 0.05. Report min, p1, p5, median and the count below 0.03. Both pilots passed on
`Ω_eval` (0 % of `Ω_eval` pairs below 0.03; min 0.073, p5 0.069); sub-0.03 pairs live at
`F − F_min > 25 kT`, outside the mask.

**Reference and method arms must share `(forcefield, M, dt, γ, integrator, dtype)`.** Otherwise the
measured "ABF error" contains the **difference of two O((ω_max dt)²) discretisation biases** — a
systematic invisible to the bootstrap and to every cross-check, sitting exactly at the size of the
claimed effect. The first-pass pilots ran at `hm = 4` (the second-worst setting in the ω_max table,
ω·dt = 0.98) and float32, while production is specified at `hm = 3`/float64. **Verify by a paired
common-random-number comparison of the reconstructed 2-D FES across a `dt` ladder {0.5, 1, 2 fs},
Richardson-extrapolated in `dt²`, tolerance 0.2 kJ/mol on `Ω_eval`.** The previously proposed pointwise
gate `|Δ⟨f⟩| ≤ 2 kJ/mol/rad ≈ 0.1 kT` is **arithmetically wrong by 20–50×**: a systematic mean-force
offset integrates, 2 × 2.48 rad = 4.96 kJ/mol = **2.0 kT** over the C7eq→C7ax φ span.

### 7.3 Why umbrella+MBAR and not an unbiased or single-walker method

C7ax needs **no special treatment**: umbrella windows tile the whole torus, so φ>0 is sampled at ~1/K
of total cost regardless of its Boltzmann weight. Its island is in fact *better* sampled per cell than
the global basin because it has little competing phase space. An unbiased run gives
**0.000000** occupancy over 102 ns aggregate — the barrier is 14.4–15.1 kT and genuinely rare
(≈0.06 expected crossings in 102 ns).

### 7.4 Backup solver

Binned 2-D periodic WHAM in torch: iterate `F_i ← −β⁻¹ log[ Σ_k n_ki / Σ_k N_k exp(β(f_k − u_ki)) ]`
and `f_k ← −β⁻¹ log Σ_i exp(−β(F_i + u_ki))` to self-consistency on the torus grid. Prefer it only if
MBAR's memory (§S8: chunk default allocates 13–26 GiB) or convergence becomes the bottleneck; with
Anderson acceleration it should not.

### 7.5 Uncertainty — the quoted SE is not usable as an acceptance gate

The block bootstrap resamples within each window independently and is therefore **structurally blind
to error common to all windows** — and the first-pass `seed_at` rigidly rotated **one** structure into
all 9216 rows. Measured: independent-setup RMS discrepancy **0.29 kJ/mol against a bootstrap SE of
0.16**. The systematic already dominates and the bootstrap cannot see the shared-seed part of it.

**Cheap fix, adopted:** seed the 16 copies of each window from *different* structures drawn from an
independent high-temperature run. This makes the shared-seed component visible to the bootstrap at no
extra sampling cost. **Do not use the 0.065 kT figure in any gate.**

### 7.6 Independent cross-checks — and their honest limits

Three estimators (umbrella integration at the window mean; Gaussian-node `kT Σ⁻¹(z − z̄)`; den-Otter
binned) gave `P(φ>0) = 0.0302 / 0.0321 / 0.0404` and `ΔG = 3.17 / 3.41 / 3.47 kT`. **This agreement is
weaker evidence than it looks:** all three run on **one** set of trajectories, one force field, one
propagator, one `dt`, one CV implementation, and two share the same FFT–Hodge integration. Every shared
systematic — the §7.1 seeding trap, incomplete equilibration, the `O(dt²)` bias, a CV convention error —
**cancels identically**.

A genuinely independent check requires a different *sampler*, not a different estimator. The repo's
`opes_cv.BatchedTorusOPES` is reusable verbatim (math is CV-agnostic); `TorusOPESConfig` needs
reduced→kJ/mol conversion (barrier ~20–40, clip ~400). **But** the previously proposed OPES acceptance
of **1 kT RMS on `Ω_eval` is the same size as the ABF residual being measured** (15.8 % × 19.9 kJ/mol =
3.2 kJ/mol = 1.3 kT) and therefore cannot validate the reference at the precision the study needs.
**Either tighten OPES acceptance to 0.3 kT or stop calling it a validator of `F_ref`.**

---

## 8. System selection — where the science goes

**Decision: demote vacuum alanine (φ,ψ) to the easy/neutrality control** — the role pentane 2-D already
plays — and move the scientific weight to a residue with a genuine hidden slow coordinate.

**The reason to state publicly is §1.2's ψ finding, not the N-ladder.** Vacuum alanine's `F(ψ|φ)` has a
maximum internal barrier of **0.64 kT**, and 1-D φ-only ABF reproduces `p(ψ|φ)` at reference-weighted
**TV = 0.080 with 100 % of φ bins supported**. There is no hidden slow coordinate for reallocation to
repair. This is measured, survives every critique, and is independent of both the contaminated
reference and the unvalidated G4 instrument.

Ranked alternatives:

**1. `Ace-Val-Nme` / `Ace-Ile-Nme` / `Ace-Leu-Nme`, ξ = (φ,ψ), χ1 deliberately EXCLUDED. [HEADLINE]**
χ1 rotamer barriers are 12–20 kJ/mol = **5–8 kT** — a genuine, chemically meaningful hidden slow
coordinate, precisely what vacuum alanine lacks. It reproduces the pentane `p(φ2|φ1)` structure the
project already understands, so `metrics.conditional_metrics` and the hidden-conditional narrative
transfer directly. **Zero new physics code**: same ff14SB, same `TorchFF`, same parity gate — only the
builder and the PHI/PSI/CHI atom indices change. Referee-proof, because "we chose a residue with a side
chain because that creates a hidden coordinate" is a *physical* justification, not a knob turned until
the answer appeared.

**2. Implicit solvent (OBC2 / GBn2), ξ = (φ,ψ), 300 K.** Most defensible in absolute terms, most
expensive. Alanine dipeptide *in solvent* is the canonical enhanced-sampling benchmark; vacuum is
widely regarded as a toy and a referee will say so. Requires a batched torch GB implementation plus a
new OpenMM parity gate. **Honest caveat: solvation flattens the FES** — it removes the vacuum
electrostatic over-structuring — so it likely moves the system *further from* starvation. Take it for
credibility, not for difficulty.

**3. Lower temperature (250 K, 200 K).** Zero new code, monotone difficulty knob, direct precedent in
the repo's β-ladder. But it deepens the *same* barrier that already lies along the biased CV, so ABF
still removes it; the expected effect is slower convergence, not starvation — exactly what happened to
pentane 2-D at β = 3 (still `intermediate`, still 9/9 basins). A temperature chosen after seeing the
result reads as post-hoc. Supporting axis, never the headline.

**4. Deliberately 1-D ξ = φ with ψ hidden. REFUTED BY MEASUREMENT**, see above. Retain only as a cheap
*strengthening* negative control — "we deliberately degraded the CV and it still did not starve" —
mirroring the pentane φ1 case.

**Plan.** Run the §6 kill-shot on option 1 (≈1.5 GPU-hours per candidate). If it clears G0/G1/G2/G3/G5,
production is ~30 GPU-hours (6 methods × ~2 h, plus a tuning ladder, plus frozen-bias, plus a
production-grade 48×48 reference). If it fails, the same argument kills it at the same low cost.

---

## 9. Metrics and diagnostics

### 9.1 Primary endpoint

Additive-constant-aligned L2 on a **common, arm-independent** support mask derived from `F_ref` alone:

`e_F(t) = min_c ‖F̂_t − F_ref − c‖_{L²(Ω_eval)}`, reported as a percentage of the thermal-window range,
plus the integrated `I_F = ∫₀^T e_F(t) dt` which rewards faster convergence.

**The mask must not be arm-dependent.** A mask derived from the arm's own occupancy structurally
flatters mFR. Report **uniform-8 kT and uniform-10 kT as sensitivities and require sign consistency
across all three.**

### 9.2 Mean-force error

`e_∇F(t) = ‖∇F̂_t − ∇F_ref‖_{L²(Ω_eval)}` via `metrics_cv.meanforce_vector_error` (reusable verbatim).
ABF estimates the mean force directly, so this is the more honest quantity — and note G5 currently
fails on it (1.5× headroom, not 3×).

### 9.3 Deterministic floor subtraction — mandatory

Report `e_F − L2_floor` alongside raw `e_F`, where `L2_floor = ‖K_h * F_ref − F_ref‖` is computed with
no simulation. At `h = 0.20` on the 48² MBAR FES this is **1.14 kJ/mol = 5.7 %** of the 8 kT window;
at the frozen `h_eff = 0.0822` it falls to **0.207 kJ/mol = 1.04 %**. Without this subtraction a large
fraction of any "room to improve" is kernel bandwidth that no sampling method can touch.

**Kernel-match both sides.** Because `g = K_h*(ρ∇F)/(K_h*ρ)` equals `K_h*∇F` **only when ρ is
uniform** — and mFR's entire purpose is to make ρ differ from ABF's — report
`‖B_arm − K_h * F_ref‖` so the kernel is matched on both sides of the comparison. Measured arm-to-arm
confound: **0.407 kJ/mol = 2.0 % of window at h = 0.20**, falling to **0.169 = 0.8 % at h = 0.08**.
Against a nominal 15 % improvement on a 15.8 % residual (≈2.4 % of window ≈ 0.5 kJ/mol), the confound
is **35–80 % of the claimed effect** at the pentane bandwidth. This is the single strongest reason the
frozen bandwidth is 0.08 and not 0.20.

### 9.4 Basin definitions — Ramachandran boxes, not `|φ| < barrier`

The alkane `basin_barrier = 61.6°` trans/gauche split is meaningless here. Define basins from **our own
`F_ref`** by watershed on `Ω_eval`, not from literature coordinates. The measured minima are
C7eq (−78.8°, +56.2°), C5/β (−146.2°, +153.7°) at 0.97 kT, C7ax (+63.8°, −41.3°) at 2.44 kT.

Report `T_hit`, `T_est`, basin entries, round trips, and `ΔF_{A,B}`. **Use basin-box occupancy, never
`φ > 0`** — the latter gives a 181/512 false positive from transient ±180°-seam touches.

Name `ΔG(C7ax)` and `P(C7ax)` errors as **secondary endpoints** in advance.

### 9.5 Marginal convergence and genealogy

`D_KL(p̂_t ‖ q_t)`, `TV(p̂_t, q_t)`, basin population errors; ancestor `ESS`, `n_unique`, `w_max`,
computed globally **and within the rare basin**.

**The ancestor-ESS guardrail is decorative as previously specified.** `fr_every = 5` steps against a
measured **6–7 ps lineage decorrelation** is ~1200 birth–death events per decorrelation time, acting on
positionally frozen ensembles. Require `fr_every ≥ 500` steps (0.5 ps) with explicit justification, and
guardrail on an **age-aware** statistic: reset ancestor labels every 6 ps and require `ESS ≥ 0.30 N`
over that window.

**The real diversity evidence is the family split-half.** Persist per-particle `ancestors` at every
save and maintain **two parity-split accumulators** (`f1s_even` / `f1s_odd`, keyed on ancestor-label
parity), cost ≈0.5 ms/step. Without it the clone-double-counting audit is impossible — as it has been
twice already.

### 9.6 Cost

Force evaluations, aggregate simulated time, wall-clock, FR/KDE overhead, **and measured ms/step per
arm logged into the npz**. Never compare iteration counts.

**"Equal compute" must be defined before the run.** Primary comparison at **equal wall-clock on
identical hardware** (ABF receives the extra steps); equal-force-evaluation as secondary. Report both.

---

## 10. Frozen-bias validation

After every adaptive production run: freeze the final bias, **discard the adaptive population**, start
fresh independent trajectories, simulate under the frozen bias with **no ABF update and no birth–death**,
and reconstruct `F = B − β⁻¹ log p_B + C`.

This is the instrument that separates a genuinely better learned bias from an online estimator that
merely counted cloned samples repeatedly — which is a **real, structural** risk here: `csum` counts
every duplicate as an independent sample, so `abf_min_count` is satisfiable by clones and the projected
field's effective sample size is overstated by the clone multiplicity.

**Design requirements, all binding:**
- Frozen runs start from a **common, method-independent, reference-Boltzmann-weighted** ensemble.
- Identical `(N, steps, seeds)` across arms; burn-in ≥ 7 ps discarded.
- L2 evaluated on the **intersection** of supported bins.
- **Pre-registered:** *the frozen-bias improvement must retain ≥ 2/3 of the online improvement.* If
  online improves 20 % and frozen 5 %, the result is an accumulator artefact and must be reported as
  such, not as a better bias.
- Fix §3.5 first. `run_frozen_bias_2d` re-differentiates the saved `B` with `spectral_gradient`, so
  until the Nyquist defect is fixed **the frozen run applies a different field than the online run
  did** — precisely the wrong property for the instrument the study leans on.

The repo's prior finding that **the sign of (recon − online) flipped exactly with the starvation
verdict** in all three earlier coordinates makes this an independent confirmation of the
classification. Reproduce it.

---

## 11. Risk register

| # | risk | discriminating diagnostic | status |
|---|---|---|---|
| R1 | ABF is not starved ⇒ mFR neutral | repo `starvation()` on ABF-only screens | **MEASURED: 1 family ⇒ `intermediate`, extrapolating to `easy` at ≈135 ps.** System property, provable before any mFR code runs |
| R2 | residual is N-independent ⇒ fixed-population redistribution cannot help | N-ladder on `L2_stat` after §6.1 | measured 0.86, but **instrument unvalidated** (§14.1) |
| R3 | discovery problem (the R15 mode) | first-entry time; equilibrium `P(basin)` | **MEASURED: does NOT apply.** 0.8–1.4 ps of 50 ps; 512/512 visit |
| R4 | reference not accurate enough to resolve a 5–15 % effect | ≥2 *genuinely independent* references (different sampler, not just different estimator) | **MARGINAL**: 3.8× headroom on `F`, **1.5× on ∇F** |
| R5 | velocity cloning wrong ⇒ mFR looks harmful for an implementation reason | `fr_*` with `fr_rate=0` bit-identical to `abf`; then `(q,v)`-cloned vs Maxwell-resampled agree within seed noise | fix + test (V11) |
| R6 | early-ABF transient heats the system | ⟨T⟩ at every save | **1-D sampler measured spiking to 356 K** — `core.py`'s 1-D ABF has **no `abf_min_count` gate** while `core2d` does. Port it before any 1-D run |
| R7 | system not metastable ⇒ nothing to enhance | unbiased 2nd-basin occupancy vs reference | **MEASURED: PASS** (0.000000 vs 0.0298) |
| R8 | target-quality confound (the EMA-sweep trap) | always run `fr_oracle` beside `fr_estimated` and `fr_uniform`. If oracle ≡ estimated ≡ uniform, the failure is the **mechanism**, not the target | do **not** spend budget on another EMA sweep — the repo already has that negative |
| R9 | genealogical collapse masquerading as harm | ESS/N, max-ancestor-fraction, round-trips/replica **and** support fraction, reported together | R15's signature: support repaired 0.223→0.082 while L2 worsened 34 % and ESS collapsed to 0.06 N |
| R10 | convention/chirality silent error | V7 — reference minimum at (−78.8°, +56.2°) | a mirrored build inverts every conclusion |
| R11 | **clone double-counting inflates the online estimate** | family split-half accumulators + the ≥2/3 frozen-retention rule (§10) | structural; instrument is mandatory |
| R12 | **occupancy-dependent estimator bias enters as a fake mFR effect** | kernel-matched L2 (§9.3); weighted projection instead of masking | 35–80 % of the claimed effect at `h = 0.20` |
| R13 | **underpowered** | ≥ 8 seeds, prefer 10, at N = 2048–4096 | the loop is dispatch-bound to RN ≈ 20480, so **there is no compute excuse for 4** |
| R14 | **initialisation does the work, not the method** | pin init as a **crossed factor**: primary 100 % in C7eq, control reference-Boltzmann. **Require the headline sign to hold under both** | the 95–99 %/1–5 % protocol hand-places population in the rare basin and is both indefensible and inert |
| R15 | **the measurable window is empty at the budget** | ABF reaches the estimator floor before 1 ns | either declare the endpoint explicitly transient (*"time to reach L2 = 10 % of range"*, a weaker but honest claim) or do not run a final-L2 comparison |
| R16 | **hyperparameter leakage** | `n_grid`, bandwidths, `N` and `abf_force_clip` were all chosen **from `F_ref`** | paired ABF-vs-mFR is safe; every **absolute** number is oracle-informed. Derive the clip from an online running p99.99 of `|f|`; declare the channel in the handoff (§3.8) |

---

## 12. Decision tree

```
V15 seed strain fails                   -> reference contaminated; NOTHING downstream admissible
V8 Nyquist fails                        -> applied field is not a gradient; ABF fixed point is not F
G0 fails after reseeding                -> fix the reference before any method claim
G4 positive control returns N-flat      -> G4 refuted as an instrument; establish any null directly
G2 fails (first entry > 10% of run)     -> R15 discovery-limited regime; mFR provably cannot help
G3 fails after floor subtraction        -> no measurable window; do not run a final-L2 comparison
G5 fails on grad-F                      -> reference cannot resolve the effect; no conclusion admissible
frozen retains < 2/3 of online gain     -> accumulator artefact, NOT a better bias
gain present but ESS<0.30N or wmax>0.05 -> false-improvement (repo's existing classifier)
oracle == estimated == uniform          -> mechanism failure, not target failure (do not tune the target)
gain survives frozen + both inits +
  all three masks + >= 8 seeds          -> POSITIVE
```

---

## 13. Open questions before production compute

1. **Does the G4 instrument survive its positive control?** Until a WCA *help* cell shows N-scaling, no
   N-ladder null is admissible anywhere in this project. This is the cheapest and highest-value
   outstanding experiment.
2. **Does the corrected reference change the measured physics?** Barrier heights, `P(C7ax)` and the
   ψ-conditional conclusion were all computed downstream of a 62 %-contaminated FES. The ψ conclusion
   is independently corroborated by the 1-D probe; the **barrier heights are not**.
3. **Does the `poisson2d` Nyquist defect require re-examining the published pentane 2-D results?**
   They used `n_grid = 48` (even), where the applied field carries ~12 % relative non-gradient content
   and `curl_norm(gB) = 1.71`. This is a scientific-integrity question for the alkane report, not only
   an engineering fix here.
4. **Which residue for option 1** — Val, Ile or Leu? Each needs its own kill-shot (~1.5 GPU-hours)
   before committing ~30 GPU-hours.
5. **Is the Nadaraya–Watson occupancy confound removable, or only boundable?** At `h = 0.08` it is
   0.8 % of window against a ~2.4 % target effect. If it cannot be pushed below ~1/3 of the effect,
   the headline claim needs a different endpoint.
6. **Is the per-seed RNG fix (§3.6) worth invalidating every existing FR baseline?** It is a breaking
   change requiring re-runs. Decide deliberately and record the decision.

---

## 14. Reconciliation — where the probes and critiques disagreed

### 14.1 The N-scaling gate G4 — **critique wins, decisively**

The `reuse-risk` probe ran the kill-shot and returned "vacuum alanine is not a viable mFR test bed"
on the strength of `L2(N/8)/L2(N) = 0.86`. **Adopted correction: this verdict is withdrawn.**

`core2d.run_sampler_2d` accumulates `f1s/f2s/csum` into **one non-forgetting running sum per seed,
shared by all N replicas**, so the error decomposes as

```
L2² ≈ (stale transient)² + (kernel bandwidth)² + (masked projection)² + (statistical variance)
```

and the first three are **N-independent by construction** — every replica feeds the same accumulator,
so at fixed step count the stale fraction and both deterministic biases are identical at N = 64 and
N = 512. A gate requiring `a ≈ 0.5` can fire only when the *variance* term dominates, which a
non-forgetting accumulator structurally prevents. **The gate as written would return FAIL on
essentially any system, including WCA where mFR demonstrably works.**

The arithmetic confirms it: with `sd(f) = 51.2 kJ/mol/rad`, 512 replicas × 10 000 accumulations,
36×36 grid, `h = 0.20`, the kernel footprint is ≈16.5 cells ⇒ ≈6.5e4 samples per smoothed cell ⇒
`SE(g) ≈ 0.20`, or ≈0.4 after `τ_int ≈ 20 fs`. The measured local `∇F` error is 29.5 % of
`RMS|∇F| = 41.75` ≈ **12 kJ/mol/rad — 30× the statistical floor**. So ≳97 % of the error is
deterministic, and the experiment **never measured the sampling-limited regime at all.** The probe
reported this N-flatness as corroboration; it is in fact proof the instrument was misapplied.

The decisive independent refutation: on WCA `b1_h2` (where mFR works), ABF `bias_fraction = 0.99` and
mFR `0.98`, yet MSE drops 0.1085 → 0.0715 (**−34 %**). mFR's gain on the one system where it works is
a reduction of **seed-common transient bias**, not of across-seed variance. Reallocation changes
per-bin sample *counts*, which changes the transient bias; it does not appear as across-seed variance.
Any variance-fraction gate therefore predicts mFR cannot work on WCA, and is wrong.
(Source: `results/mfr_mechanism_audit/bias_variance/bias_variance_integrated.csv`.)

**Resolution:** G4 is retained as *informative*, demoted from *decisive*, and blocked behind the
positive control in §6.1(4). **The demotion of vacuum alanine rests on §8's ψ finding instead.**

### 14.2 The 19.5 kT premise — **probe wins over the lead's initial measurement**

Struck in §1.2 with the correction: trapped minimisation; true ΔE = 2.50 kT, ΔG = 3.2–3.5 kT,
`TΔS ≈ 0`. Two probes and one critique independently agree.

### 14.3 Reference seeding — **critique wins**

`reuse-risk` used `build_positions` for job A; `reference-fes` derived the `seed_at` rigid-rotation
recipe. The engineering critique then measured 61.8 % strained windows under the former.
`seed_at` + gate V15 is adopted; `build_positions` is removed from the seeding path (§7.1).

### 14.4 Umbrella restraint form — **`reference-fes` wins**

Two probes shipped different restraints (`κ(1−cos)` vs wrapped harmonic `k = 400`). At κ = 400 the
antipode sits at ½kπ² = 1974 kJ/mol and is never reached at 300 K, so the practical damage was nil —
but the spec must ship **one** convention. `κ(1−cos)` with `k_eff = κ` is adopted (§7.2).

### 14.5 HMR setting — **`system-hmr` wins on the value, critique wins on the reasoning**

`hm = 3.0` is adopted. But the claimed "exact agreement" between the 1-D reduced-mass model and the
measurement is dropped: the corrected root is `h* = 3.177` (the quadratic constant is 226.021, not
225.02), while the measured optimum is 3.0. Keep the recommendation, drop the claim of agreement.

### 14.6 Mass-metric dual and the Fixman term — **critique wins**

The unconstrained mass-metric dual has **no Fixman term**; `−β⁻¹ div w` already carries everything.
Where a Fixman term does arise (holonomic constraints) it is `+(2β)⁻¹ d log det Z`, not
`−β⁻¹ d log det Z` — the original had the wrong sign *and* a factor 2. Corrected in §2.4/§4.2.

### 14.7 CPU cost — **critique wins**

The design's fit predicted 104.6 ms/step at B = 9216; measured with `torch.compile` and
`OMP_NUM_THREADS=64` it is **42.75 ms**, i.e. **2.45× better**. CPU is a first-class path, not a
footnote — especially given GPUs 4–7 are memory-constrained (§0). GPU remains the recommendation.

### 14.8 Batched umbrella windows — **probe wins, and should be stated at full strength**

The engineering critique tried to break it and could not: rows are independent (`ff.energy` has no
cross-row reduction, `autograd.grad(E.sum())` is row-local, O-step noise is i.i.d.), **NaN does not
propagate across batch rows**, `N_k` is identical by construction, **MBAR remains valid**.
`torch.compile` does not break RNG reproducibility. `scatter_add_` nondeterminism is 11 orders below
the bootstrap SE and should be a one-line caveat, not a stated uncertainty.

### 14.9 Seeds and initialisation — **critique wins**

4–5 seeds is underpowered given the loop is dispatch-bound; ≥ 8 (prefer 10) at N = 2048–4096 costs
~2× at the top. The 95–99 %/1–5 % initialisation is replaced by a crossed factor (§11 R14).
