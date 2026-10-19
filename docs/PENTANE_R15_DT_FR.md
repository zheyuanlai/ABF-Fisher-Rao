# Pentane R15: can a better ABF timestep rescue ABF + marginal Fisher–Rao?

*2026-10-18 (UTC), branch `research/pentane-r15-dt-fr`, CPU only (all four H200s were in use by other users).
Frozen designs: `configs/pentane_r15_dt_fr/prereg_stage_a.json` (commit d77a194e, before any dynamics),
`configs/pentane_r15_dt_fr/prereg_stages_bcd.json` (commit f243e914, before any ABF/FR run),
`configs/pentane_r15_dt_fr/addendum_exploratory_active_fr.json` (commit 5ef85a99, before any dt 2.5e-4 free-energy error was computed).
Only two algorithm families appear: ABF and ABF + uniform-marginal Fisher–Rao birth–death (mFR).*

## 1. Executive summary and verdict

**Verdict.** No category-A or category-B outcome. The earlier failure of mFR on pentane R15 was **not**
caused by the Langevin timestep. The same result holds at a validated timestep: uniform mFR at the
original gentle rate is **inactive** (C). At every more active rate tried it is **harmful** (D): the R
marginal becomes more uniform, the free energy gets worse and the replica genealogy collapses. ABF's own
error is dominated by **hidden torsional (conditional) sampling** (E), which marginal birth–death cannot
reach. Refining the timestep makes ABF *worse* at equal physical time. Pentane R15 should stay in the paper
as a **negative control** and as a **numerical-methodology cautionary case**, not as a positive mFR benchmark.

| question | answer (16 fresh paired seeds unless stated) |
|---|---|
| Is dt 5e-4 Gibbs-faithful? | Not in the stiff modes: four bond modes have λ·dt 0.95–1.44, bond-length variances are 2.1–2.3× too large, bend TV 0.10. But the R15 free energy is off by only 0.059 (U95 0.067) in the flattened ensemble ABF converges to, under the 0.10 tolerance. The frozen gate still FAILS dt 5e-4, on its clip-activity clause alone (1.7e-5 > 1e-5 clipped force evaluations); the paired no-clip control is identical to 2e-5 |
| Selected timestep (frozen rule: largest ADMISSIBLE) | **dt\* = 2.5e-4** (D_MF 0.041 [0.047], D_density 0.034 [0.039]); 1.25e-4 also ADMISSIBLE |
| ABF at dt\* vs dt 5e-4, T = 40 | **worse**: e_F(40) 1.820 vs 1.379 (+32 % [+26, +37]), I_F 113.3 vs 101.4 (+11.7 % [+9.8, +13.8]), e_F′(40) 8.61 vs 7.38. Pilot: e_F(40) 1.38 / 1.77 / 1.89 at dt 5e-4 / 2.5e-4 / 1.25e-4. Deposition frequency is not the cause (stride controls ≤ 0.4 %). Mechanism (Section 14): compensating invariant-measure error of dt 5e-4 in the deposited f_R samples, not kinetics |
| ABF at equal force evaluations (3.3e8, pilot) | dt 5e-4 for T = 160: e_F 0.99; dt 2.5e-4 for T = 80: 1.11; dt 1.25e-4 for T = 40: 1.89. More physical time is the lever, not a smaller dt |
| mFR (rate 0.02) vs ABF at dt\* | ΔI_F **+0.6 % [+0.5, +0.7]**, 0/16 wins; Δe_F(40) +2.0 % [+1.4, +2.4]; TOST-equivalent within ±10 %; ∫KL(p̂‖U) −3.8 % (16/16) |
| mFR (rate 0.02) vs ABF at dt 5e-4 | ΔI_F +0.7 % [+0.5, +0.8], 0/16; Δe_F(40) +2.7 % [+1.9, +3.0] (reproduces the 2026-09-06 result, +0.5 %) |
| FR activity at 0.02 | 0.10 N replacements in 13 600 events; ancestor ESS/N 0.81; max lineage share 0.6 % |
| Secondary rate (frozen rule) | **none**: 0.05 and 0.10 fail genealogy safety (ESS/N 0.37 / 0.07) before they become "active" (0.28 / 0.63 N replacements) |
| Exploratory active mFR at dt\* (rates 0.05 / 0.10, safety-failing) | ΔI_F **+2.0 % [+1.7, +2.3] / +5.6 % [+5.0, +5.9]**, Δe_F(40) +8.0 % / +24.0 %, 0/16 each. The marginal is far more uniform (∫KL −14 % / −41 %, under-supported bins −15 % / −83 %) while D_cond worsens +1.0 % / +4.4 % |
| Bottleneck | at compact R (2.17–2.36) the exact conditional is 99 % same-sign GG but 98.5 % of ABF's deposits are opposite-sign GG. At the TG/TT crossover (R ≈ 3.03–3.14) compressed-TT deposits drive a −20 to −38 mean-force bias. The raw deposit bias RMS is 9.2–9.3, against 0.6–0.8 for the integrator's own stationary mean-force bias |

Selected timestep 2.5e-4; ABF / mFR (0.02) at it: e_F(40) 1.820 / 1.860, I_F 113.3 / 114.3, relative
change +0.6 % [+0.5, +0.7]; genealogy safe (ESS/N 0.81) but the FR is inert. Total compute ≈ 25 CPU core-hours.

## 2. Exact model and simulation configuration

**Molecule.** United-atom pentane, five sites, reduced units (`src/alkanes/potentials.py`): harmonic bonds
k0 = 1000, d0 = 1; harmonic bends on the forward-bond angle, k_θ = 208, θ0 = 1.187 rad (interior angle 112°);
Ryckaert–Bellemans torsion (c1, c2, c3) = (1.18, −0.23, 2.64) on both dihedrals (trans at φ = 0); one
Lennard-Jones 1-5 pair, σ = 2.3, ε = 1, pair distance floored at 0.6σ. β = 2. Forces are −∇V by autograd,
then the per-atom norm is clipped at 200 (`force_clip`).

**CV and estimator** (`src/alkanes/distance_cv.py`, `src/alkanes/core_dist.py`, unchanged). R15 = |q5 − q1|.
Local mean force f_R = ½ e·(∇₅V − ∇₁V) − 2/(βR), each sample clipped to ±480. Grid 256 bins on [1.4, 3.7];
ABF mean force = Gaussian-kernel (h 0.04) Nadaraya–Watson over the deposits; applied bias force
+A′(R)∇R, |A′| ≤ 60, ramped linearly over the warm-up; soft walls at 1.45 / 3.65 (k 200). Reported free
energy = integral of the production mean force (deposits after the burn-in).

**FR** (`src/alkanes/core.py::_birth_death`, unchanged). Score s_i = log p̂(R_i) − log q(R_i) − KL, recentred,
clipped to ±2 three times; p̂ = reflected KDE (h 0.06) of the walker positions; q uniform on [1.4, 3.7]
(`fr_uniform`, `fr_domain = None`, as in the original project). Walker i with s_i > 0 dies with probability
1 − exp(−rate · s_i · dt · fr_every); each death is replaced by a whole-configuration copy of a walker drawn
∝ max(−s, 0); at most 1 % of N per event. With a uniform target the score is just the recentred log density
of the walker marginal, so the target's domain matters only for walkers outside it (none here).

**Production cell** (`configs/alkanes_cv_extension/r15_methods.yaml`, stage production): N = 1024 walkers,
all-trans start (φ1, φ2) = (0, 0) plus 1e-3 Gaussian noise, T = 40, rate 0.02, max event fraction 0.01.

**Physical-time schedule** (`src/alkanes/dt_schedule.py`). Every step count is physical time / dt:

| knob | physical value | dt 5e-4 | dt 2.5e-4 | dt 1.25e-4 |
|---|---|---|---|---|
| T | 40 | 80 000 | 160 000 | 320 000 |
| ABF warm-up (bias ramp) | 2.5 | 5 000 | 10 000 | 20 000 |
| estimator burn-in | 3.0 | 6 000 | 12 000 | 24 000 |
| FR start | 6.0 | 12 000 | 24 000 | 48 000 |
| FR event interval (`fr_every`) | 0.0025 | 5 | 10 | 20 |
| checkpoint | 1.0 | 2 000 | 4 000 | 8 000 |

dt · fr_every = 0.0025 exactly at every dt, so the per-event death probability for a given score and the
physical FR event times are identical across timesteps (tests: `test_birth_death_probability_identical_across_timesteps`,
`test_fr_event_times_identical_in_physical_time`).

**Reference** (evaluation only). `cache/alkanes_cv/ref_pentane_b2_R15_v2_meanforce.npz`
(sha256 7363c374…d9d9, checked on every load by `scripts/pentane_r15_dt_fr/r15metrics.py`): 200 M exact
internal-coordinate importance samples, F′ = binned conditional mean of f_R, window [2.0828, 3.5383]
(162 bins, n_eff ≥ 1466 per bin), F range 9.7, independent fine-KDE density route agrees to L2 0.042,
median SE(F′) 0.041. It is never read by ABF or FR dynamics: `run_arm.py` passes `oracle_free_energy=None` and
the engine's no-leakage assertion is unchanged. The only dynamics that see it are the Stage A integrator tests,
which use F′_ref as a fixed flattening bias (Section 4). Every error below is on this window with the additive constant
aligned on it.

**Seeds and pairing.** Pilot rng 20261101 (4 seeds), FR-rate calibration 20261102 (4 seeds), confirmatory
20261201–20261204 (16 fresh seeds). ABF and mFR with the same rng share initial states and the dynamics
noise stream; FR draws come from a separate generator; the two arms are bitwise identical before t = 6
(`test_abf_and_fr_arms_identical_before_fr_starts`).

## 3. Code-level numerical audit (`scripts/pentane_r15_dt_fr/audit_numerics.py`, `results/pentane_r15_dt_fr/audit/`)

| check | result |
|---|---|
| `potentials.forces` vs central finite differences (unclipped, 64 exact samples) | relative error 9.7e-11 |
| analytic vs autodiff ∇R and div v; closed form f_R | max error 2e-16 / 4e-16 / 5e-13; free-particle limit f = −2/(βR) exact |
| EM noise / β (1-D harmonic, λ = 1000, dt 5e-4) | Var·βλ = 1.337 vs theory 1/(1 − λdt/2) = 1.333 |
| per-atom physical clip (200) on exact Gibbs samples | active on 3e-27 of the Boltzmann weight (max raw force 1.5e4 occurs only at near-zero-weight LJ overlaps) |
| f_R sample clip (480) on exact Gibbs | never active |
| ABF bias clip (60) vs the reference | max \|F′_ref\| on the window 48.9 < 60 |
| sign / gauge | bias force +A′∇R with A′ the estimated F′ (flattens); F integrated from the left edge, centred; analysis aligns on the window |
| FR birth–death | population preserved, whole 5-atom configurations copied, ancestors propagated, dead walkers have s > 0 and parents s < 0 (`test_birth_death_preserves_population_and_clones_whole_configurations`) |

The three clips are distinct in the code: the physical clip acts on −∇V before both the dynamics and f_R
(so it enters the estimator too); the sample clip acts only on deposits; the bias clip only on the applied
bias.

**Hessian and EM stability** (exact Hessian by autograd). At the four torsional-family minima the largest
eigenvalues (bond-stretch dominated) are 2690–2889 (all-trans: 755, 1893, 2247, 2636, 2889); over 2000 exact
Gibbs samples λ_max has median 2863, 99th percentile 2940, max 2970. Stability indicator λ_max·dt:

| dt | median λ_max·dt | samples with λ·dt > 2 | harmonic Var_EM / Var_Gibbs of the stiffest mode |
|---|---|---|---|
| 5e-4 | 1.43 | 0 | 3.6 |
| 2.5e-4 | 0.72 | 0 | 1.57 |
| 1.25e-4 | 0.36 | 0 | 1.22 |
| 6.25e-5 | 0.18 | 0 | 1.10 |

The production dt is linearly stable but close to the edge: four bond modes have λ·dt between 0.95 and 1.44,
so their EM variance is inflated 1.9–3.6×. No correctness fix was needed anywhere; the production force
field, clips and estimator are unchanged.

**Engine changes** (additive, default off, byte-identical: `test_byte_identical_to_frozen_sampler` runs the
frozen `main` sampler and the new one on the same configuration for ABF and FR and compares every output
array exactly): `deposit_stride` (deposit every k-th step), `numerics_diag` (max per-atom step
displacement, non-finite walkers), `track_families` (torsional family / 9-basin first discovery, clone-corrected
family transitions, per-checkpoint occupancy and distinct-ancestor counts, per-R-bin family composition of the
deposits). Clip activity is recorded by a force wrapper whose arithmetic is bitwise the production force
(`test_clip_stats_force_wrapper_is_bitwise_production`).

## 4. Stage A — does the production integrator sample the intended Gibbs law?

Design (frozen before any dynamics, `prereg_stage_a.json`): the production update, verbatim, started from
**exact** samples so that slow torsional mixing cannot masquerade as timestep bias, with no adaptive state and
no deposits into any estimator. Two ensembles, 8 independent groups × 1024 walkers each, T_A = 10, deposits
from t = 0.25 (`scripts/pentane_r15_dt_fr/stage_a.py`, `src/alkanes/dt_validation.py`):

* **A2 flat (the gate)**: exact window-restricted Gibbs samples with a flat R marginal (per-bin weighted
  reservoir from a 200 M proposal pool, ≥ 51 per bin), run under the FIXED bias force +F′_ref(R)∇R (edge
  values held outside the window) plus the production walls. This is the ensemble ABF converges to. The
  reference enters this *integrator test* only as a fixed flattening bias; it never enters ABF or FR.
* **A1 Gibbs**: exact Gibbs samples (weighted reservoir from 20 M proposals, global ESS 0.877), unbiased
  production EM, no walls; read on the low-F sub-window [2.433, 3.475] (F_ref − min ≤ 2.5).

Every (ensemble, dt) was run twice on identical initial states and noise: clip 200 and no clip.
Statistic: debiased windowed RMS D of the group-mean profile minus F_ref (group noise and the reference's own
SE removed), 95 % upper bound by bootstrap over groups; tolerance 0.10 (= 0.2 kT; justification in the prereg).

| ensemble | dt | D_MF [U95] | D_density [U95] | D(MF vs density) | F′ bias RMS | clip frac | bond Var ratio | max bend TV | max step disp |
|---|---|---|---|---|---|---|---|---|---|
| A2 flat | 5e-4 | 0.059 [0.067] | 0.038 [0.042] | 0.057 | 0.82 | 1.7e-5 | 2.10–2.21 | 0.100 | 0.18 |
| A2 flat | 2.5e-4 | 0.041 [0.047] | 0.034 [0.039] | 0.024 | 0.56 | 2.4e-9 | 1.38–1.39 | 0.043 | 0.10 |
| A2 flat | 1.25e-4 | 0.037 [0.041] | 0.033 [0.037] | 0.012 | 0.45 | 0 | 1.20–1.22 | 0.031 | 0.07 |
| A1 Gibbs | 5e-4 | 0.233 [0.259] | 0.053 [0.066] | 0.245 | 2.54 | 2.6e-5 | 2.21–2.34 | 0.127 | 0.20 |
| A1 Gibbs | 2.5e-4 | 0.077 [0.104] | 0.021 [0.039] | 0.081 | 0.81 | 1.2e-9 | 1.35–1.36 | 0.040 | 0.11 |
| A1 Gibbs | 1.25e-4 | 0.037 [0.065] | 0.000 [0.033] | 0.037 | 0.36 | 3e-10 | 1.15 | 0.018 | 0.07 |

No walker became non-finite; the f_R sample clip never fired; the clipped and unclipped runs give the same
profiles to 2e-5 at every dt (the clipped force exceeds 200 by 10 on average when it fires and the stiff
modes contract the perturbation). Torsional-family populations of the exact Gibbs start (TT 0.6565, TG 0.3318,
same-sign GG 0.0117, opposite-sign GG < 1e-4) drift by ≤ 0.0016 ± 0.0012 over 10 time units at every dt.
Mean bond lengths in A1 are biased by +1.3e-4 / +4e-5 / +1.5e-5 at the three dts (Gibbs SD 0.022).

**Gate (frozen).**

| dt | verdict | reason |
|---|---|---|
| 5e-4 | **FAIL** | only the clip-activity clause: 1.7e-5 of per-atom force evaluations clipped (> 1e-5). D_MF 0.059 [0.067] and D_density 0.038 [0.042] are inside the tolerance, and the paired no-clip control is identical |
| 2.5e-4 | **ADMISSIBLE** | D_MF 0.041 [0.047], D_density 0.034 [0.039], clip 2.4e-9, paired no-clip identical |
| 1.25e-4 | **ADMISSIBLE** | D_MF 0.037 [0.041], D_density 0.033 [0.037], no clip |

**Selected timestep: dt\* = 2.5e-4** (largest ADMISSIBLE; from Stage A numerics only). The 6.25e-5 extension
was not needed.

Reading:

1. **The historical dt's discretisation error in F(R) is small.** In the flat ensemble ABF converges to, the
   free-energy error at dt 5e-4 is 0.059, against a dt-independent floor of ≈ 0.035: the same edge-shaped deviation (+0.13 at the compact edge,
   +0.05 to +0.10 at the extended edge) appears at all three dts and in both routes, so it comes from the
   reference's binned-F′ integration or the flat-ensemble construction, not from the integrator. The dt-attributable part is
   ≈ √(0.059² − 0.035²) ≈ 0.05, i.e. ~3–4 % of ABF's error at T = 40 (≈ 1.4).
2. **The stiff modes are not Gibbs-faithful at dt 5e-4.** Bond-length variances are inflated 2.1–2.3× and the
   bend distributions are off by TV 0.10–0.13 — as the Hessian predicts (λ·dt up to 1.44). These
   distortions largely average out of the CV free energy.
3. **The mean-force route is more sensitive than the density route, and its error depends on the applied
   bias.** Without bias (A1) the conditional mean force at dt 5e-4 is off by 2.5 RMS (F error 0.23 on the
   sub-window) and converges cleanly, ≈ 3× per halving (2.54 → 0.81 → 0.36). Under the flattening bias (A2)
   the same integrator's F′ error is 0.82 → 0.56 → 0.45. Either way it is an order of magnitude below ABF's own
   mean-force error at T = 40 (7–9 RMS, Section 5).
4. The clip is a non-issue for the intended Gibbs law: it is active at 1e-5 rates only at dt 5e-4 and has no
   measurable effect. Smaller timesteps validate the intended dynamics without any help from the clip.

**Supplementary kinetics check** (`scripts/pentane_r15_dt_fr/torsion_rates.py`; unbiased EM from exact Gibbs,
4 × 1024 walkers × 10 t.u. per dt, core-to-core counting with ±30° cores): trans→gauche
5e-4 ± 1e-4, 6e-4 ± 0.4e-4, 5e-4 ± 1e-4 per dihedral per unit time; gauche→trans 0.0028 ± 0.0004,
0.0028 ± 0.0002, 0.0020 ± 0.0003 at dt 5e-4 / 2.5e-4 / 1.25e-4 (31–44 events per dt). Unbiased torsional
kinetics show no resolvable dt dependence between 5e-4 and 2.5e-4. They are slow: a given trans dihedral
flips once per ~2000 time units, 50× the ABF horizon.

## 5. Stage B — ABF timestep convergence

ABF only, pilot rng 20261101, 4 seeds per configuration (`results/pentane_r15_dt_fr/pilot/REPORT.md`), plus the
16-seed confirmatory ABF arms at dt 5e-4 and 2.5e-4. Every time-dependent knob is in physical time; nothing in
the estimator was tuned.

**(1) Timestep at fixed physical time (T = 40).**

| dt | e_F(40) | I_F | e_F′(40) | under-supported | D_cond(40) | clip frac | max step disp | wall per seed |
|---|---|---|---|---|---|---|---|---|
| 5e-4 (pilot / confirmatory) | 1.375 / 1.379 | 102.8 / 101.4 | 7.37 / 7.38 | 0.157 / 0.148 | 0.369 / 0.368 | 1.1e-5 | 0.18 | 4.1 min |
| 2.5e-4 (pilot / confirmatory) | 1.771 / 1.820 | 111.7 / 113.3 | 8.45 / 8.61 | 0.133 / 0.136 | – / 0.371 | 1.5e-8 | 0.11 | 8.1 min |
| 1.25e-4 (pilot) | 1.893 | 113.5 | 8.85 | 0.136 | 0.373 | 9e-10 | 0.07 | 16.4 min |

Unpaired ratios to dt 5e-4: e_F(40) +28.8 % [+23.4, +34.0] (2.5e-4, pilot), +32.0 % [+26.4, +36.8]
(2.5e-4, confirmatory), +37.7 % [+31.5, +43.8] (1.25e-4, pilot). The curves coincide up to t ≈ 8–10 and separate
afterwards (Figure 2a). The change saturates (2.5e-4 → 1.25e-4: +7 %), which suggests the exact-dynamics ABF error at
T = 40 is ≈ 1.9–2.0. No run had a non-finite state.

**(2) Physical time at fixed dt, and equal force evaluations.** At dt 5e-4, e_F falls from 1.375 (T 40) to
0.989 (T 160); at dt 2.5e-4 from 1.771 (T 40) to 1.106 (T 80). At an equal budget of 320 000 steps × 1024
walkers: dt 5e-4 / T 160 → 0.989, dt 2.5e-4 / T 80 → 1.106, dt 1.25e-4 / T 40 → 1.893 (Figure 2c).
The ABF at dt 5e-4 and T 160 still has e_F′ 7.9 and D_cond 0.378: more time lowers the free-energy
error but not the hidden-conditional error.

**(3) Deposition frequency at fixed dt** (paired, same noise): depositing every 5e-4 time units instead of every
step changes I_F by −0.2 % [−0.8, 0.0] at dt 1.25e-4 and −0.4 % [−1.1, +1.0] at dt 2.5e-4, and e_F(40) by
≤ 0.1 %. The extra correlated deposits of a small dt are irrelevant. The dt effect is in the dynamics.

**Where the dt effect sits** (16 confirmatory seeds, Figure 4f, Figure 6). The final mean-force error is
identical in the compact half (R < 2.75: 8.7–10.8 RMS at both dts) and differs only at R ≥ 2.75 (7.8 / 7.1
at dt 2.5e-4 vs 4.3 / 4.1 at dt 5e-4). The difference comes from the TG/TT crossover, R ≈ 3.03–3.14, where F has a
barrier (F′ +13 → −20). Deposits there are sparse (0.5–0.7 ‰ per bin, the bins flagged as under-supported).
They mix stretched TG with compressed TT walkers. The TT share of the 2.93–3.12 conditional bin is 11.2 %
at dt 2.5e-4 vs 7.5 % at dt 5e-4 (exact: 0.4 %). The deposit mean-force bias peaks at −38 vs −26. The
biased estimate is self-reinforcing: an F̂′ that is too negative on the TG side adds a push toward smaller
R, so ABF *raises* the TG→TT barrier it should flatten. At t = 40 the walker marginal above R 3.1 is almost
empty (Figure 4a; TT holds 0.3 % of walkers) and ABF underestimates the barrier, leaving F̂ at the TT
minimum 1.4 (dt 5e-4) to 2.2 (dt\*) units too low (Figure 6). The coarse integrator damps this feedback; the accurate one does not.
Unbiased torsional rates do not explain it (Section 4, no resolvable dt dependence), so the effect arises
under the ABF bias. Section 14 (addendum) identifies the mechanism. It is not kinetic: dt 5e-4's invariant-measure
error shifts the f_R samples at the crossover against the feedback.

## 6. Stage C/D — ABF vs ABF + uniform mFR

**Pilot** (rng 20261101, 4 paired seeds, FR rate 0.02): ΔI_F +0.7 % [+0.5, +1.0] (dt 5e-4), +0.6 %
[+0.4, +1.0] (2.5e-4), +0.7 % [+0.5, +1.0] (1.25e-4), 0/4 wins at every dt. There were no broken runs, no
instability and no genealogy problem. FR was inactive at every dt (0.09–0.12 N replacements). This triggered
the pre-specified rate calibration.

**FR-rate calibration** (rng 20261102, 4 seeds, dt\*, turnover and genealogy only; no free-energy error was
computed before the rate was fixed; `results/pentane_r15_dt_fr/calib/`):

| rate | replacements / N | final ESS / N | max lineage share | safe | active |
|---|---|---|---|---|---|
| 0.02 | 0.106 | 0.776 | 0.009 | yes | no |
| 0.05 | 0.284 | 0.370 | 0.024 | **no** (ESS) | no |
| 0.10 | 0.633 | 0.073 | 0.052 | **no** (ESS, share) | no |

Frozen rule → **no secondary rate**. The genealogy collapses long before turnover reaches one population
equivalent, because births are drawn ∝ max(−s, 0): the few walkers in the sparsest R region are cloned
over and over.

**Confirmatory** (16 fresh paired seeds, rng 20261201–04; `results/pentane_r15_dt_fr/confirm/REPORT.md`):

| arm | e_F(40) | I_F | e_F′(40) | I_F′ | D_cond(40) | ∫KL(p̂‖U) | under-supported | repl./N | ESS/N | wall / seed |
|---|---|---|---|---|---|---|---|---|---|---|
| ABF, dt 2.5e-4 | 1.820 | 113.26 | 8.61 | 479.5 | 0.371 | 13.30 | 0.136 | 0 | – | 8.1 min |
| mFR 0.02, dt 2.5e-4 | 1.860 | 114.27 | 8.76 | 482.9 | 0.372 | 12.78 | 0.136 | 0.102 | 0.805 | 8.2 min |
| ABF, dt 5e-4 | 1.379 | 101.35 | 7.38 | 444.2 | 0.368 | 13.19 | 0.148 | 0 | – | 4.1 min |
| mFR 0.02, dt 5e-4 | 1.415 | 101.92 | 7.47 | 446.3 | 0.369 | 12.75 | 0.148 | 0.094 | 0.828 | 4.1 min |

Paired effects of mFR (median [95 % CI], wins / 16):

| contrast | ΔI_F | Δe_F(40) | ΔI_F′ | ΔD_cond(40) | Δ∫KL | Δ under-supported |
|---|---|---|---|---|---|---|
| dt\* 2.5e-4 | **+0.6 % [+0.5, +0.7] 0/16** | +2.0 % [+1.4, +2.4] 0/16 | +0.5 % [+0.4, +0.6] | +0.2 % [+0.1, +0.3] | −3.8 % [−5.7, −2.1] 16/16 | −4.9 % [−8.3, −2.4] |
| dt 5e-4 | **+0.7 % [+0.5, +0.8] 0/16** | +2.7 % [+1.9, +3.0] 0/16 | +0.5 % [+0.3, +0.6] | +0.1 % [+0.1, +0.2] | −2.9 % [−3.6, −1.8] 16/16 | +0.0 % [−4.0, 0.0] |

Equal physical time and equal force evaluations coincide within a timestep (FR needs no extra force calls;
it costs ≈ 2 % wall-clock). H4's premise, that a more uniform marginal helps, fails here:
the marginal is more uniform in every seed and the free energy is worse in every seed. The effect is the
same at both timesteps (+0.6 % vs +0.7 %), so reducing the discretisation error does not change the ABF–mFR
comparison.

**Exploratory active mFR** (rates 0.05 / 0.10 on the same 16 confirmatory seeds, dt\*; added by the addendum
after the rate rule returned "none"; both rates fail the safety rule; `results/pentane_r15_dt_fr/explore/`):

| rate | repl./N | ESS/N | max lineage | ΔI_F | Δe_F(40) | ΔD_cond | Δ∫KL | Δ under-supported | Δe_F compact / rest |
|---|---|---|---|---|---|---|---|---|---|
| 0.02 | 0.10 | 0.81 | 0.006 | +0.6 % [+0.5, +0.7] | +2.0 % | +0.2 % | −3.8 % | −4.9 % | +1.5 % / +2.2 % |
| 0.05 | 0.28 | 0.43 | 0.019 | +2.0 % [+1.7, +2.3] | +8.0 % [+6.3, +9.0] | +1.0 % | −14.4 % | −15.2 % | +5.7 % / +8.8 % |
| 0.10 | 0.63 | 0.07 | 0.069 | +5.6 % [+5.0, +5.9] | +24.0 % [+22.3, +25.3] | +4.4 % | −41.4 % | −83.4 % | +16.8 % / +26.3 % |

All 0/16 wins. The dose response is monotone: the more FR does, the more uniform the marginal and the worse
the free energy, mean force, hidden conditional and genealogy.

## 7. Marginal coverage vs hidden torsional mixing

**Discovery is not what fails.** Every confirmatory ABF seed visits all four torsional families: TG by a
median t 0.27, opposite-sign GG by 1.6–1.9, same-sign GG by 8–9 (latest 26; 20 % of the horizon). mFR does
not change these times. It never discovers anything, and in one of the 16 mFR 0.02 seeds at dt\* same-sign
GG was never visited.

**What fails is the family *within* an R bin** (deposit-weighted composition of the production deposits vs
the exact conditional, 16 seeds, dt\*; Figure 4e):

| R bin | exact p(family \| R) TT / TG / GG same / GG opp | ABF deposits | mFR 0.10 deposits |
|---|---|---|---|
| 2.17–2.36 | 0 / 0 / 0.992 / 0.007 | 0 / 0.013 / 0.002 / **0.985** | 0 / 0.013 / 0.003 / 0.984 |
| 2.36–2.55 | 0 / 0.006 / 0.994 / 0 | 0 / 0.683 / 0.013 / 0.305 | 0 / 0.684 / 0.015 / 0.301 |
| 2.93–3.12 | 0.004 / 0.996 / 0 / 0 | 0.112 / 0.888 / 0 / 0 | 0.223 / 0.777 / 0 / 0 |
| 3.12–3.32 | 0.997 / 0.003 / 0 / 0 | 0.999 / 0.001 / 0 / 0 | – |

The R-bias drives walkers to compact R through the opposite-sign GG ("pentane-effect") route, plausibly
because that flip shortens R most (energy minima: opposite-sign GG R 2.28, same-sign GG 2.50, TG 2.90). The family reached is the one the Gibbs conditional almost never uses there. The
correct same-sign GG family is found later and stays at 0.4 % of walkers. Once in the wrong family, a walker
stays there for the whole run: τ⊥ at fixed R is 10⁶–10⁷ steps (2026-09-06 P1/P2 measurement), and unbiased
trans→gauche flips occur at 5e-4 per dihedral per unit time (Section 4). So 46 % of all deposits fall in the
two compact conditional bins R < 2.36 (exact Boltzmann weight 0.0005) and carry the wrong family's mean force. The compact half
of the mean force is off by 8.7–10.8 RMS, and the raw deposit bias is 9.2–9.3 RMS. The integrator's own
stationary mean-force bias is 0.56 (dt\*) to 0.82 (dt 5e-4) RMS. Discretisation is a tenth of the problem,
and refining dt makes ABF worse rather than better (Section 5).

**Why marginal FR cannot repair this, and why it hurts.** FR acts on R only. Its deaths are almost all in
the over-dense compact region (1702 of 1714 deaths at rate 0.02, 10 069 of 10 561 at 0.10, all
opposite-sign GG). Its births are copies of walkers sitting in sparse R, mostly TT and TG (rate 0.10: 7076 TT
and 3030 TG births). Two consequences:

* The conditional composition of the compact bins is unchanged at every rate (0.985 → 0.984 opposite-sign
  GG). FR removes wrong-family walkers but cannot create right-family ones, because no same-sign GG walker
  sits in sparse R to be copied. This is the structural limit of marginal resampling (d/dt p(y | ξ)|_FR = 0).
* The copied TT walkers are not new information. At rate 0.10 the TT share of walkers rises from 0.3 % to
  16 %, but only 8 % of those TT walkers are distinct lineages (Figure 4b, c). Under the ABF bias they spill
  into the TG/TT crossover: the TT share of the 2.93–3.12 bin doubles, 0.112 → 0.223. That deepens the
  crossover bias and the error outside the compact region rises most (e_F "rest" +26 % at 0.10).

The more uniform marginal is therefore superficial coverage: duplicated configurations, not independent
exploration. The free energy, the mean force and the conditional all get worse in proportion.

## 8. Genealogical diversity

(Figure 5; `ancestor` labels propagate through every clone, ESS = 1/Σw² over original walkers.)

| arm | replacements / N | final ESS / N | min ESS / N | max lineage share | TT walkers that are distinct lineages |
|---|---|---|---|---|---|
| mFR 0.02, dt\* (16) | 0.102 | 0.805 | ≈ 0.80 | 0.006 | 61 % |
| mFR 0.02, dt 5e-4 (16) | 0.094 | 0.828 | ≈ 0.83 | 0.005 | – |
| mFR 0.05, dt\* (16, exploratory) | 0.277 | 0.431 | ≈ 0.43 | 0.019 | 21 % |
| mFR 0.10, dt\* (16, exploratory) | 0.629 | 0.071 | ≈ 0.07 | 0.069 | 8 % |

Frozen safety thresholds: ESS/N ≥ 0.5 and max lineage share ≤ 0.05. The gentle rate is safe and inert;
0.05 fails ESS after t ≈ 35; 0.10 fails both. Each birth is drawn from the handful of walkers in the sparsest
R region, so lineages collapse much faster than walkers are replaced: at rate 0.10, replacing 63 % of the
population once leaves 7 % effective lineages.

## 9. Outcome classification

Frozen rules (`prereg_stages_bcd.json`), applied by `analyze_runs.py confirm`:

| category | holds? | evidence |
|---|---|---|
| **A** numerical rescue of ABF | **no** | dt 5e-4 fails the Stage A gate (clip clause only; F error 0.059 < 0.10). ABF at the validated dt is *worse*: e_F(40) +32 % [+26, +37], I_F +11.7 % |
| **B** genuine FR acceleration | **no** | ΔI_F +0.6 % [+0.5, +0.7] at dt\* (rate 0.02); no safe secondary rate; the exploratory active rates are worse (+2.0 %, +5.6 %) |
| **C** FR inactive | **yes** (rate 0.02, both dts) | 0.10 / 0.09 N replacements over 13 600 events (< 0.5 N) |
| **D** FR harmful | **yes** | formally at 0.02 (ΔI_F CI > 0 with a more uniform marginal, 0/16, but within ±10 %: TOST-equivalent, practically negligible). Clearly at 0.05 / 0.10 (exploratory): e_F(40) +8 % / +24 %, ESS/N 0.43 / 0.07 while ∫KL falls 14 % / 41 % |
| **E** discovery- or conditional-limited | **yes** | sampling-dominated: ABF e_F(40) 1.82 ≫ 5 × the Stage A error 0.041. The frozen rule's sub-label is *discovery-limited*, because 13.6 % of window bins are under-supported at t = 40 (the TG/TT crossover R 3.07–3.16 and the extended tail R ≥ 3.43); the rule's *conditional-limited* branch requires support ≤ 0.10. Mechanistically (Section 7) the dominant error is conditional: every family is discovered (median t ≤ 9), but the compact bins hold the wrong family (98.5 % vs exact 0.7 %) and the crossover is contaminated by compressed TT. The under-support is itself produced by that biased estimate (ABF raises the TG→TT barrier) |
| **F** inconclusive | **no** | every CI is narrow (≤ 1 % wide for ΔI_F) |

Hypotheses: **H1** (timestep) is rejected as the cause. The discretisation error is real in the stiff modes
but small in F(R), and refining dt hurts ABF. **H2** (discovery) is minor: same-sign GG takes ~8 time
units, but discovery does not set the error. **H3** (conditional equilibration) is supported and dominant.
**H4** (FR establishment acceleration) is rejected: FR adds occupancy only by copying, and harms in
proportion to its activity.

## 10. Figures

All in `results/pentane_r15_dt_fr/figures/` (PNG + PDF), regenerated by committed scripts from the raw runs;
no hand-edited values.

* **Figure 1** `fig1_timestep_validity`: Stage A. Panels: (a) F_dt − F_ref (mean-force route, flat ensemble);
  (b) F′ deviation; (c) gate statistic vs dt with U95 and tolerance; (d) clip activity; (e) bond-variance
  inflation; (f) torsional-family drift from the exact start. Script: `analyze_stage_a.py`.
* **Figure 2** `fig2_abf_timestep_convergence`: ABF e_F(t), e_F′(t) at equal physical time for three dts (pilot
  plus confirmatory); e_F vs force evaluations; deposition-stride controls.
* **Figure 3** `fig3_abf_vs_mfr`: ABF vs ABF + mFR (0.02), e_F(t) median and IQR at dt\* and dt 5e-4
  (confirmatory). Per-seed paired ΔI_F with 95 % CI, including the exploratory 0.05 / 0.10 arms.
* **Figure 4** `fig4_regional_exploration`: final walker marginal with FR birth/death R histograms;
  per-family walker share; distinct lineages per family; under-supported fraction; per-R-bin family
  composition of the deposits vs exact; deposit mean-force bias vs R for ABF at both dts.
* **Figure 5** `fig5_fr_activity_genealogy`: cumulative replacements / N, ancestor ESS / N and the largest
  lineage share, for every rate and dt, with the frozen safety thresholds.
* **Figure 6** `fig6_profiles`: final F(R15) and F′(R15) of ABF and mFR vs the v2 reference at dt\* and dt 5e-4.
  The compact region is shaded; the TG/TT barrier at R ≈ 3.12 is the visible miss.
* Scripts for Figures 2–6: `make_figures.py`.
* **Figure 7** `fig7_crossover_mechanism` (addendum, Section 14): (a) adaptive drain of the all-trans start, every adaptive
  arm; (b) applied bias during the warm-up; (c) frozen-bias drains; (d) TT→TG hazard vs R; (e) per-family f_R under
  an identical frozen bias, with the EM-shadow arm; (f) adaptive crossover deposits by family; (g) e_F(t) including the
  dt-switch runs; (h) bin-8 occupancy by family. Script: `analyze_crossover.py`.

## 11. Reproducibility

Environment: Python 3.13.13, torch 2.12.1 (CPU, one thread per process, `CUDA_VISIBLE_DEVICES=""`), numpy 2.5.0;
host `atlas`. Every run file stores its full `DistSimConfig`, schedule, cell, rng, git commit, versions,
host and wall-clock (`meta_json`). Raw `.npz` files (≈ 120 MB) stay out of git (`*.npz` is ignored). Their
SHA-256 manifests are committed (`results/pentane_r15_dt_fr/*/SHA256SUMS.*.txt`) and the files live in this
worktree under `results/pentane_r15_dt_fr/*/raw/`. The v2 reference is checked by checksum at load. If it is
missing, rebuild it with `python scripts/build_pentane_r15_reference_v2.py` (200 M samples; ≈ 6 s on GPU,
minutes on CPU).

```bash
python -m pytest -q tests/test_pentane_r15_dt_fr.py
```
```bash
python scripts/pentane_r15_dt_fr/audit_numerics.py
```
```bash
python scripts/pentane_r15_dt_fr/stage_a.py init && python scripts/pentane_r15_dt_fr/stage_a.py launch --max-procs 24 && python scripts/pentane_r15_dt_fr/analyze_stage_a.py
```
```bash
python scripts/pentane_r15_dt_fr/launch_runs.py pilot --max-procs 9 && python scripts/pentane_r15_dt_fr/analyze_runs.py pilot
```
```bash
python scripts/pentane_r15_dt_fr/launch_runs.py calib --dt 2.5e-4 && python scripts/pentane_r15_dt_fr/analyze_runs.py calib
```
```bash
python scripts/pentane_r15_dt_fr/launch_runs.py confirm --dt 2.5e-4 --max-procs 16 && python scripts/pentane_r15_dt_fr/analyze_runs.py confirm
```
```bash
python scripts/pentane_r15_dt_fr/launch_runs.py explore --max-procs 8 && python scripts/pentane_r15_dt_fr/analyze_runs.py explore
```
```bash
for dt in 5e-4 2.5e-4 1.25e-4; do for g in 0 1 2 3; do python scripts/pentane_r15_dt_fr/torsion_rates.py --dt $dt --group $g; done; done
```
```bash
python scripts/pentane_r15_dt_fr/make_figures.py
```

Addendum (Section 14; ≈ 12.4 core-hours including the invalid uncapped and the stopped T 40 shadow runs):

```bash
python scripts/pentane_r15_dt_fr/crossover_mechanism.py adaptive --dt 5e-4 --T 4 && for dt in 5e-4 2.5e-4 1.25e-4; do python scripts/pentane_r15_dt_fr/crossover_mechanism.py adaptive --dt $dt; done
```
```bash
for rng in 20261301 20261302; do for dt in 5e-4 2.5e-4 1.25e-4; do python scripts/pentane_r15_dt_fr/crossover_mechanism.py frozen --bias early5 --dt $dt --T 12 --rng-seed $rng; done; python scripts/pentane_r15_dt_fr/crossover_mechanism.py frozen --bias early5 --dt 5e-4 --no-clip --T 12 --rng-seed $rng; python scripts/pentane_r15_dt_fr/crossover_mechanism.py frozen --bias early5 --dt 1.25e-4 --no-clip --shadow-dt0 5e-4 --T 12 --rng-seed $rng; done
```
```bash
for dt in 5e-4 2.5e-4 1.25e-4; do python scripts/pentane_r15_dt_fr/crossover_mechanism.py frozen --bias mid5 --dt $dt --T 12 --rng-seed 20261301; done; for b in final5 final25; do for dt in 5e-4 2.5e-4; do python scripts/pentane_r15_dt_fr/crossover_mechanism.py frozen --bias $b --dt $dt --T 20 --rng-seed 20261401; done; done
```
```bash
python scripts/pentane_r15_dt_fr/crossover_mechanism.py adaptive2 --dt 5e-4 --no-clip && python scripts/pentane_r15_dt_fr/crossover_mechanism.py adaptive2 --dt 5e-4 --dt2 2.5e-4 --t-switch 3 && python scripts/pentane_r15_dt_fr/crossover_mechanism.py adaptive2 --dt 2.5e-4 --dt2 5e-4 --t-switch 3
```
```bash
for rng in 20261211 20261212 20261213 20261214; do python scripts/pentane_r15_dt_fr/crossover_mechanism.py adaptive2 --dt 1.25e-4 --T 12 --no-clip --shadow-dt0 5e-4 --n-seeds 1 --rng-seed $rng; done; python scripts/pentane_r15_dt_fr/analyze_crossover.py
```

Compute used: Stage A 5.7, pilot 6.9, calibration 1.6, confirmatory 6.5, exploratory 4.3 core-hours
(≈ 25 core-hours plus < 1 for the audit and the rate check). Peak concurrency was ≈ 33 single-thread
processes on a 256-core host; no GPU was used.

## 12. Limitations, failed experiments, unresolved questions

* **One cell.** β 2, all-trans start, N 1024, T 40, as in the original study. The dispersed start, other N and
  other β were not re-run at the new timestep.
* **The Stage A gate was stricter than needed in one clause.** dt 5e-4 failed only on clip *activity*
  (1.7e-5) although the paired no-clip runs are identical. The rule was not changed after the fact. Had the
  clause been ignored, dt 5e-4 would have been selected and the ABF–mFR comparison would read the same
  (+0.7 % there).
* **Gate ensemble choice.** The gate used the flattened ensemble (A2). In the unbiased ensemble (A1) the
  mean-force route at dt 5e-4 is off by 0.23 on the low-F sub-window and dt 2.5e-4 would be MARGINAL
  (U95 0.104). An A1-based gate would have selected 1.25e-4, where ABF is worse still (e_F(40) 1.89) and
  mFR is again +0.7 % (pilot), so the conclusions do not depend on this choice.
* **Reference floor.** The A2 profiles share a dt-independent deviation of ≈ 0.035 RMS (edge-shaped, with
  a step at the TG/TT crossover), from the reference's binned-F′ integration or the flat-ensemble
  construction. It is not discretisation, but it bounds how small a timestep error Stage A can resolve.
* **The "smaller dt → worse ABF" mechanism** sits at the TG/TT crossover and shows as more compressed-TT
  contamination at small dt. *Resolved by the 2026-10-19 addendum (Section 14):* no kinetic property damps the
  feedback. EM's invariant-measure error at dt 5e-4 shifts the deposited f_R samples. During the warm-up the
  more negative compressed-TT samples finish the TT transient earlier, mostly inside the burn-in (≈ 40 % of the
  gap). Afterwards the more positive stretched-TG samples hold the crossover estimate up (≈ 60 %).
* **Exploratory arms.** The active-rate (0.05 / 0.10) results were added by an addendum written after the
  rate rule returned "none" and before any dt\* free-energy error was computed. They are not confirmatory and
  fail the safety rule.
* **Under-support vs conditional labels.** The frozen classification labels E as discovery-limited, because
  its conditional-limited branch required adequate support. The mechanism data show the under-support is a
  consequence of the conditional error. Both labels are reported.
* **Scope.** No hidden-coordinate method (bias on (φ1, φ2), rejuvenation, OT) was run, by instruction, so the
  obvious fix for H3 is untested here (the 2026-07-21 2-D torsion cell is the relevant prior evidence).
* **Failed experiments.** None: 0 of 96 Stage A jobs, 0 of 36 sampler jobs and 0 of 12 torsion-rate jobs failed, and no seed was dropped. Two
  analysis-script bugs (a keyword-argument syntax error and a glob that matched `.sha256` sidecars) were found
  and fixed before any number was reported.
* **Open questions.** (i) Does the wrong-family trap persist for T ≫ 40 at dt\* (ABF at dt 5e-4, T 160 still
  has D_cond 0.378)? (ii) Would a rate schedule that switches FR off in the compact region change the sign?
  Not tested: that would be a new algorithm.

## 13. Recommendation for the free-energy paper

Keep pentane R15 in the paper, in two roles:

1. **Negative control for marginal Fisher–Rao.** At a validated timestep, gentle uniform mFR is inert
   (ΔI_F +0.6 % [+0.5, +0.7], 16 seeds). Every rate active enough to matter makes the R marginal more uniform
   while making the free energy worse (+2.0 % / +5.6 % integrated, +8 % / +24 % final) and collapsing the
   genealogy (ESS/N 0.43 / 0.07). The mechanism is clean: the error is the hidden torsional family at fixed
   R, which marginal birth–death cannot change and amplifies by cloning.
2. **Numerical-methodology cautionary case.** (i) The production dt 5e-4 is near the EM stability edge for
   the bond modes and distorts bond and bend distributions 2–3.6×, yet the CV free energy is accurate to
   0.06. Validate the CV observable against exact Gibbs instead of inferring from the stiffest mode. (ii) A
   validated, more accurate integrator gave a *worse* ABF at equal physical time (+32 %). A small timestep is
   not automatically a better ABF baseline, and at equal force evaluations the long run at the coarse dt was
   best. (iii) Deposition frequency was irrelevant.

Do **not** present pentane R15 as a positive mFR benchmark. The historical R15 conclusions (ABF starved,
mFR inert/harmful) survive the timestep refinement. Re-scoring every R15 number on the v2 reference, as the
2026-09-06 report recommended, remains advisable.

## 14. Addendum (2026-10-19): what damps the crossover feedback at dt 5e-4

*ABF only, CPU only, ≈ 12.4 core-hours. Frozen design `configs/pentane_r15_dt_fr/addendum_crossover_mechanism.json`
(commit 561e316d, before any new dynamics), with four amendments committed before the runs they govern: an
exploratory follow-up (29d5d3c1), a cap on the shadow drift after the uncapped version blew up (0f186998), a dt-switch
experiment (9412cb55), and a T 12 limit for the adaptive shadow arm to stay inside the 20 core-hour cap (8e35cbbf). Runner `scripts/pentane_r15_dt_fr/crossover_mechanism.py`, analysis
`scripts/pentane_r15_dt_fr/analyze_crossover.py`, outputs `results/pentane_r15_dt_fr/crossover/` (REPORT.md, summary.json),
Figure 7.*

**Answer.** No property of the integrator's *kinetics* damps the feedback. Under any fixed bias, the dt 5e-4 drain of
compressed TT walkers is the same as at 2.5e-4 and 1.25e-4 (±1.5 %), and the clip and the walls play no part. What
differs is what dt 5e-4 *deposits*. Its invariant measure is distorted in the stiff bond and bend modes (Section 4), and
this shifts the f_R samples at the TG/TT crossover by a few units in the two directions that work against the
compressed-TT contamination:

* **(A) Warm-up (t < 3, weak ramping bias, all-trans walkers near equilibrium).** Compressed-TT samples are more
  negative. The estimate turns compressive earlier and the all-trans population drains earlier. More of the TT
  transient then falls inside the estimator burn-in, and fewer compressed-TT samples reach the production estimate.
* **(B) After the warm-up.** Stretched-TG samples at R 2.94–3.06 are more positive. This lifts the crossover estimate,
  so the bias pushes fewer TG walkers out of the crossover. More TG then dilutes the TT contamination that remains.

The dt-switch experiment attributes about 40 % of the e_F(40) gap to (A) and 60 % to (B). dt 5e-4 is not a more
faithful integrator here: its own mean-force error is three times larger (Section 4, A1). It wins at T = 40 because
that error partly cancels the sampling error, not because it samples the crossover better.

**Instrument.** *Tracked adaptive ABF* is the production sampler (`run_sampler_dist`, unchanged) with a force-function
hook whose arithmetic is bitwise the production force. Every step it records per-walker torsional state, R, the f_R
sample and clip activity. It also re-accumulates the ABF sums exactly as the engine does, which gives the bias actually
applied. The tracked dt 5e-4 and 2.5e-4 runs (rng 20261201, 4 seeds) reproduce the confirmatory files bitwise (PMF,
mean force, family counts, production accumulators). *Frozen bias* is the production update (EM, clip 200, bias
clipped ±60, walls, COM removal) under a FIXED applied-bias profile, from the all-trans start, with no estimator.
"Committed" family means the last ±30° core visited by each dihedral. τ_TT = ∫₀ᵀ (committed-TT fraction) dt. W8 is the
walker-time per walker in conditional bin 8 (R 2.933–3.125, the engine's 61.6° basin families).

### 14.1 The contamination is laid down during the drain of the all-trans start

(Confirmatory and pilot files, read before the design.) Almost all TT walker-time in bin 8 accrues during t ≈ 2–12,
while the all-trans population is driven into compression and converts to TG. That drain is faster at dt 5e-4: the TT
walker fraction at t = 4 is 0.36 / 0.49 / 0.54 at dt 5e-4 / 2.5e-4 / 1.25e-4 (16 / 16 / 4 seeds). Production bin-8
walker-time at t = 40 at dt 2.5e-4 vs 5e-4 is TT 0.224 vs 0.175 (+28 %) and TG 1.79 vs 2.15 (−17 %). The TT share
0.112 vs 0.075 thus has two parts, more TT and less TG dilution.

### 14.2 Under an identical frozen bias the integrator does not matter (preregistered H_int: not supported)

| frozen bias (all-trans start) | dt | τ_TT | W8_TT | W8_TG | TT→TG hazard, R 2.95–3.125 |
|---|---|---|---|---|---|
| t = 4 applied bias (drain phase), T 12, 8 seeds | 5e-4 | 1.735 | 0.246 | 0.045 | 0.69 |
| | 2.5e-4 | 1.736 (+0.1 % [−3.1, +6.3]) | 0.242 | 0.043 | 0.70 |
| | 1.25e-4 | 1.727 (−0.5 % [−3.6, +3.8]) | 0.239 | 0.042 | 0.75 |
| | 5e-4, no clip (paired) | 1.735 (0.0 % [−0.01, +0.04]) | 0.246 | 0.045 | 0.69 |
| t = 2 applied bias × 0.8 (inside the ramp), T 12, 4 seeds | 5e-4 / 2.5e-4 / 1.25e-4 | 7.86 / 7.93 / 7.98 | 0.361 / 0.347 / 0.349 | 0.31 / 0.32 / 0.32 | 0.89 / 0.96 / 1.01 |
| t = 40 applied bias of dt 5e-4, T 20, 4 seeds | 5e-4 / 2.5e-4 | 16.97 / 16.86 | 0.505 / 0.489 | 0.327 / 0.349 | 0.51 / 0.56 |
| t = 40 applied bias of dt 2.5e-4, T 20, 4 seeds | 5e-4 / 2.5e-4 | 14.80 / 14.89 | 0.692 / 0.696 | 0.446 / 0.465 | 0.60 / 0.62 |

The TT→TG hazard is steep in R: about 0 above R 3.06, 2 at 2.99, 20 at 2.94, 100–200 below 2.86 per unit TT time.
It is the same at every dt (Figure 7d). A compressed TT walker converts only once the bias has pushed it below
R ≈ 2.95. Under every fixed bias, dt changes τ_TT by ≤ 1.4 %, W8_TT by ≤ 4 % and W8_TG by ≤ 7 %. Where the difference
is resolved it goes *against* the adaptive pattern: dt 5e-4 has slightly *more* TT and *less* TG in bin 8. By the
frozen rule H_int is not supported, so the dt effect is a bias-history effect. H_static and H_clip as frozen were
therefore not interpreted; both are reported descriptively below. Changing the bias origin at fixed dt (t = 40 bias of
dt 2.5e-4 vs 5e-4) changes W8_TT and W8_TG by +37 % and +36 %, five to ten times the effect of the integrator under one bias.

**Clip and walls.** In tracked adaptive ABF at dt 5e-4 the clip fires on 5.5e-4 of the walker-steps of TT walkers
at R 2.95–3.35, ten times its global rate (≤ 5e-8 at the finer timesteps). Removing it (paired noise) changes
e_F(40) by −0.08 % [−0.53, +0.04] and the frozen drain by 0.0 %. The soft walls (R 1.45 / 3.65) never act at the
crossover; the global wall-step fraction is 0.07–0.10 % in every plain, no-clip and dt-switch adaptive arm.

### 14.3 What differs is the f_R sample, at fixed R and family

Per-family mean f_R sample under the identical frozen t = 4 bias (8 seeds per arm; Figure 7e):

| R | TT: dt 5e-4 / 2.5e-4 / 1.25e-4 / shadow | TG: dt 5e-4 / 2.5e-4 / 1.25e-4 / shadow |
|---|---|---|
| 2.94 | −37.3 / −33.6 / −32.4 / −38.5 | +5.4 / +2.7 / +2.0 / +6.4 |
| 2.96 | −39.2 / −36.5 / −35.8 / −39.7 | +8.3 / +4.2 / +3.0 / +10.0 |
| 2.99 | −39.8 / −38.4 / −38.0 / −39.9 | +10.3 / +4.6 / +2.7 / +12.8 |
| 3.01 | −39.4 / −38.6 / −38.4 / −39.2 | +10.6 / +2.7 / +0.3 / +13.3 |
| 3.04 | −37.9 / −37.5 / −37.5 / −37.8 | +7.7 / −0.7 / −3.2 / +10.4 |
| ≥ 3.06 | equal within 0.3 | (too few TG) |

Without a bias, the same sign appears in Stage A's unbiased ensemble (A1, exact start; Section 4): the conditional mean
force at R 3.15–3.2 (TT share 0.94) is −23.4 / −20.5 / −19.8 at dt 5e-4 / 2.5e-4 / 1.25e-4 (reference −19.5). At
R 3.0–3.1 (TG-dominated) it is +2.0 to +2.7 above the reference at dt 5e-4 and +0.4 to +0.8 at 2.5e-4. The shadow
column is dt 1.25e-4 dynamics on dt 5e-4's EM invariant measure: the first-order EM shadow drift b + (dt0/2) J_b b,
with dt0 = 5e-4 and J_b b by central differences of the full drift. It reproduces dt 5e-4's stiff-mode distortion
(committed-TT bond-length variance 1.75e-3 vs 1.64e-3 at dt 5e-4 and 1.09e-3 at 1.25e-4; bend variance 7.5e-3 vs
7.2e-3 vs 5.9e-3) and its per-family f_R, overshooting slightly, while its drain stays at τ_TT 1.737. The f_R shift is
therefore a property of EM's invariant measure, not of its kinetics.

### 14.4 How the shifted samples change ABF (tracked adaptive runs, rng 20261201, 4 seeds)

| | dt 5e-4 | dt 2.5e-4 | dt 1.25e-4 |
|---|---|---|---|
| e_F(40) | 1.374 | 1.841 (+34 % [+24, +45]) | 1.920 (+40 % [+37, +52]) |
| applied bias at R 3.05, t = 1 (all walkers TT) | −29.0 | −27.7 | −26.3 (−9 % [−18, −3]) |
| applied bias at R 3.0, t = 2 (16 / 16 / 4 confirmatory and pilot seeds) | −23.5 | −18.5 | −17.3 |
| committed-TT fraction at t = 4 | 0.37 | 0.51 | 0.55 |
| bin-8 TT walker-time, all steps / share before t = 3 | 0.290 / 39 % | 0.317 / 28 % | 0.325 / 24 % |
| bin-8 TT walker-time, production (t ≥ 3) | 0.176 | 0.229 | 0.248 |
| bin-8 TG walker-time, production | 2.19 | 1.78 | 1.71 |
| crossover deposit bias (mean over R 2.95–3.125 of deposit mean − F′_ref) | −7.8 | −15.0 | −16.3 |

Splitting the crossover deposit-mean difference (fine bins R 2.95–3.125) into a family-share part and a per-family-mean
part gives −7.2 = −6.1 (share) − 1.1 (per-family f_R) for 2.5e-4 vs 5e-4, and −8.9 = −7.5 − 1.4 for 1.25e-4. The direct
f_R shift is small; it acts mostly through the share it induces. A subtlety for (A): the total TT walker-time in bin 8
is only 9–12 % lower at dt 5e-4. dt 5e-4 starts compressing earlier, so 39 % of its TT time falls inside the
estimator burn-in (t < 3), against 28 % and 24 %. Its production TT deposits are therefore 23–29 % lower.

**Timestep switch at t = 3 (end of the burn-in; phase 1 bitwise the plain run; deposits weighted by dt / dt1).**
ψ = (X − X_2.5e-4) / (X_5e-4 − X_2.5e-4):

| run | e_F(40) | ψ(e_F(40)) | production TT, bin 8 | production TG, bin 8 | ψ(TT fraction at t = 4) | ψ(TG, bin 8) |
|---|---|---|---|---|---|---|
| dt 5e-4 until t 3, then 2.5e-4 | 1.673 | 0.36 [0.07, 0.55] | 0.176 | 1.83 | 0.95 | 0.07 |
| dt 2.5e-4 until t 3, then 5e-4 | 1.564 | 0.59 [0.39, 0.98] | 0.229 | 2.11 | 0.04 | 0.81 |

The two ψ add to ≈ 1. The amount of TT contamination is fixed entirely by the warm-up timestep (0.176 vs 0.229 in
either order); the TG dilution follows the later timestep. Under a fixed bias the later timestep does not change TG
occupancy (Section 14.2), so channel (B) runs through the estimate: the more positive TG samples (Section 14.3) lift
the crossover estimate, which feeds back on where TG walkers go.

**Adaptive shadow (exploratory, dt 1.25e-4 on the EM shadow of dt0 5e-4, T 12, 4 single-seed runs).** At t = 12,
compared with dt 5e-4 (no clip) and plain dt 1.25e-4: e_F(12) is 2.55 vs 3.01 vs 3.30; bin-8 TG walker-time 0.32 vs
0.17 vs 0.14; TT fraction at t = 4 0.47 vs 0.37 vs 0.55; applied bias at R 3.05, t = 1 −31.7 vs −29.0 vs −26.3. The
amendment's rule reads this as "static (invariant-measure)" (φ(e_F(12)) = 2.6 ≥ 0.7). But it is an overshoot, not a
quantitative match: the shadow's estimator effects are stronger than dt 5e-4's. Its population dynamics also differ,
with more TT walkers left after t ≈ 6 (0.08 vs 0.013 at t = 12) and ten times the wall activity (1.2 %). The
conclusion drawn from it is qualitative: EM's invariant-measure distortion with *accurate* kinetics is enough to
produce every estimator effect of dt 5e-4, in the same direction.

### 14.5 Reading and limits

* **Mechanism.** ABF at the crossover is self-reinforcing (Section 5): compressed-TT samples make F̂′ too negative,
  which pushes TG walkers away and preserves the contamination. The EM invariant measure at dt 5e-4 shifts the f_R
  samples against this loop, in the two ways above: (A) a stronger early compressive estimate, which finishes the TT
  transient earlier and largely inside the burn-in, and (B) more positive stretched-TG samples, which hold the
  crossover estimate up. Refining dt removes both compensating shifts, so the bare sampling error shows: e_F(40)
  1.37 → 1.84 → 1.92, saturating as the integrator error vanishes.
* **What it is not.** Not the torsional or crossover kinetics (frozen-bias drain, hazard and excursions are
  dt-independent within a few %). Not the clip, the walls or the deposition frequency (Section 5 (3)).
* **Consequences.** The +32 % ABF penalty at dt\* measures how much of dt 5e-4's apparent accuracy at T = 40 came
  from a lucky sign of its discretisation error. It is not a defect of the validated timestep. The conclusions of
  Sections 1–13 are unchanged: the mFR comparison is the same at both dts (Section 6), and the dominant ABF error is
  conditional (Section 7). The case strengthens the cautionary point of Section 13: an integrator error can lower a
  free-energy error by partly cancelling a sampling error.
* **Limits.** Four seeds per adaptive arm; the 16-seed confirmatory contrast is the established effect, and the tracked
  seeds reproduce it (+34 %). The 40/60 split rests on one switch time (t = 3) and four seeds. The shadow drift is
  first order, and its correction is capped at 0.9 |drift| on 1e-5 of the walker-steps, so it is a qualitative probe.
  The uncapped version was numerically invalid and is not used (`crossover/invalid_uncapped_shadow/`). Which stiff mode
  (bonds or bends) carries the f_R shift was not separated.
