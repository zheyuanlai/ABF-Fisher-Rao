# Ethane / flexible ZIF-8: Wasserstein lift + constrained gate repair — results

Prereg `docs/ZIF8_OT_REPAIR.md` (frozen 2026-09-06 12:40 UTC).  Operators `src/zif8/ot_repair_zif8.py`;
study `scripts/zif8_ot_z123.py` → `Z123/{z123.json, z123.npz, z123.log, figures/}` (29.9 min on the
shared GPU 1, 6 sites × 256 replicas, determinism flags off).  Gate-consistency follow-up
`scripts/zif8_ot_z1b_gate_consistency.py` → `Z1b*/`.

## Z1 — fixed-ξ constrained BAOAB vs the umbrella reference

| site | ξ interval (Å) | F′_ref | ⟨f_ξ⟩ − F′_ref (SE) | halves drift | ⟨A_gate⟩ (sd) | ref ⟨A_gate⟩ | TV(rec, ref) / TV(halves) | θ_gate | τ_A (fixed ξ) |
|---|---|---|---|---|---|---|---|---|---|
| cage minimum | [5.96, 6.11] | −0.0 | −0.02 (0.03) | +0.07 | 2.796 (0.058) | — | — / 0.006 | 35.4° | 100 steps = 50 fs |
| left half-height | [−1.04, −0.90] | +18.7 | +0.26 (0.03) | −0.11 | 2.886 (0.053) | — | — / 0.005 | 36.9° | 78 = 39 fs |
| left band | [−0.50, −0.25) | +16.1 | +0.14 (0.03) | +0.02 | 2.934 (0.051) | 2.865 | 0.395 / 0.003 | 37.2° | 74 = 37 fs |
| window plane | [−0.25, +0.25) | +10.0 | −0.04 (0.03) | +0.09 | 2.949 (0.052) | 2.863 | 0.468 / 0.004 | 37.1° | 78 = 39 fs |
| peak band | [+0.75, +1.00) | +0.3 | +0.03 (0.02) | +0.00 | 2.948 (0.053) | 2.847 | 0.547 / 0.005 | 36.7° | 82 = 41 fs |
| right half-height | [+2.29, +2.44] | −26.3 | −0.88 (0.06) | +0.23 | 2.818 (0.060) | — | — / 0.011 | 35.0° | 116 = 58 fs |

* **Mean force: PASS.**  The constrained operator reproduces F′_ref at every site to ≤ 0.3 kJ/mol/Å
  except the steep right flank (−0.88 on F′ = −26, 3 %, where the reference's own central
  difference is least accurate).  The fixed-ξ ensembles are stationary (TV between 10 ps halves
  0.003–0.011).
* **Gate conditional: FAILED the frozen 0.02 Å test, and the peak band touched the 0.10 Å hard
  stop.**  All three band sites show a wider gate than the reference by +0.07 to +0.10 Å (≈ 1.5 sd),
  a narrower distribution (sd 0.05 vs 0.09), and a more tilted linker (37.1° vs 36.0°).  The shift is
  uniform across replicas (within-replica sd = pooled sd), i.e. every replica sits in the same shifted
  state.  Resolution below (§Z1b).

## Z2 — single OT event (lift Δξ, then constrained repair at ξ′)

Injection = |b(0) − b_inf| against |F′_ref(ξ′)|; repair times in inner steps (0.5 fs each).

| site | Δξ = −2 bins (−0.30 Å): inj / \|F′\| | Δξ = +2 bins: inj / \|F′\| | b 20 %-time (−2 / +2) | b e-fold | ⟨A_gate⟩ lag after ±2 bins (sd units) | A 20 %-time |
|---|---|---|---|---|---|---|
| cage minimum | 6.6 / 0.4 | 4.8 / 0.1 | 173 / 187 | 126–128 | +0.10 / +0.15 | 181–189 |
| left half-height | 7.3 / 16.8 (0.43) | 7.6 / 18.7 (0.41) | 133 / 103 | 107 / 76 | +0.57 / −0.49 | 121 / 127 |
| left band | 5.0 / 18.6 (0.27) | 3.9 / 11.6 (0.34) | 119 / 85 | 92 / 67 | +0.56 / −0.15 | 140 / 56 |
| window plane | 3.5 / 14.7 (0.23) | 3.6 / 5.3 (0.67) | 83 / 87 | 69 / 67 | +0.28 / −0.00 | 72 / — |
| peak band | 2.6 / 2.3 (1.1) | 2.1 / 1.9 (1.1) | 73 / 95 | 61 / 79 | −0.04 / +0.09 | 40 / 284 |
| right half-height | 34.8 / 25.9 (1.34) | 10.2 / 20.0 (0.51) | 127 / 133 | 73 / 111 | −0.72 / +0.26 | 152 / 157 |

* **Injection is first order and large**: median over the flank/band sites at the 2-bin cap
  inj/|F′| = **0.47** (pentane: ≈ 0.03–0.1 per event; WCA: ≈ 2 at its cap).  It is asymmetric on the
  steep flank (moving *into* the barrier at the right flank injects −35 against F′ −26).  Injection
  is close to linear up to 2 bins (median slope 16 kJ/mol/Å per Å) and super-linear at 8 bins
  (1.2 Å), where the guest overlaps ring atoms (d_min 2.0–2.3 Å, ΔU up to +44 kJ/mol) — the
  2-bin cap is at the safe edge.
* **Repair works but is slow relative to the 5-step cadence**: 5 inner steps (2.5 fs) remove
  0–5 %; the mean-force lag decays with a **20 %-time of ~100 steps (50 fs; median 99) and an
  e-fold of ~75 steps (37 fs)**, with a guest-oscillation shoulder at ~100–150 fs; by 400 steps
  (200 fs) b(m) is within ±1 of b_inf at every site and |Δξ|, i.e. no irreparable residual.
* The gate itself lags by 0.1–0.7 sd after a 2-bin lift (1.5–2.5 sd after 8 bins) and re-adapts on
  the same 40–190-step scale; D_gate(m) returns to the 256-sample floor by m ≈ 100–400.

## Z3 — τ_gate(ξ) and the regime

Three consistent readings: τ_A at fixed ξ 74–116 steps (37–58 fs; the empty framework had 50 fs);
mean-force relaxation after a 2-bin lift 20 %-time 73–187 steps, e-fold 61–128; aperture
relaxation 40–190 steps.  **τ* ≈ 100 steps = 50 fs with inj ≈ 0.5 → Regime A (WCA-like) by the
frozen rule** — the gate lag is real, first order and repairable — **but not at the 5-step
cadence**: removing 80 % of a 2-bin injection costs ~100 inner steps per lifted walker (20×
the outer budget at an opportunity every 5 steps).  Affordable designs: an OT opportunity every
≥ 100 outer steps with ~100 inner steps (inner/outer ≤ 1), or repair only walkers moved ≥ 1 bin.
The 2-bin cap is the safe edge (8-bin lifts create host–guest overlaps).

## Z1b — resolving the gate-aperture inconsistency: a defect in the stored reference's gate conditional

Facts established in order (`Z1b/`, `Z1b_randompool/`, `Z1c_*.log`, scratch kernel/lateral checks):
1. The constrained ensembles are stationary and the shift is uniform across replicas (within-replica
   sd = pooled sd), so it is not a mixture of gate states nor an equilibration lag.
2. Preparation does not matter: fast pull, slow 20 ps pull, ξ-nearest or **random** pool frameworks all
   give ⟨A_gate⟩ = 2.949 ± 0.05 at the window plane; the framework's kinetic temperature is 297.5–299 K.
3. The constraint does not matter: the reference's own umbrella-spring protocol run by my loop gives
   2.948 in the band (random frameworks, 30 ps), and constraining those states afterwards leaves 2.937.
4. The kernel does not matter: compiled-deterministic vs compiled-nondeterministic forces differ only at
   f32 noise (5e-4 on 4e15) and both match eager to 1e-6; no engine commit since the corrected production;
   my cage-site aperture reproduces Stage 0B's empty-framework value (2.7957 ± 0.058) to four decimals.
5. The guest's lateral state does not matter: held at |ξ| < 0.25 it sits on the axis (ρ = 0.20 ± 0.02 Å),
   aligned within 12°, U_host–guest = −6.6 kJ/mol; A_gate is uncorrelated with ρ and orientation.
6. **The cause is in the reference and the production diagnostics, not in the operator.**  The CV is
   periodic on the circle, so a guest at the *periodic image* of the window (unwrapped ξ ≈ ±L) has φ ≈ 0
   and is counted "in band", but `gate_observables` measures the one indexed window — then empty.  The
   init pool has the guest anywhere in cage A (ξ ∈ [−11.2, −1.7]); **54.3 % of its frameworks are nearer
   the image window**, so the umbrella reference and every production in-band histogram are a 46/54
   mixture of the held-guest gate (2.949 ± 0.052) and the empty gate (2.796 ± 0.058): predicted mean
   2.866, sd 0.094; observed 2.857–2.867, sd 0.090–0.092.  My ensembles lattice-shift the guest into
   the indexed window and therefore see the true conditional.
7. Direct confirmations: (a) the library `run_umbrella` at one window from unshifted pool frameworks
   for 150 ps (`Z1c_umbrella`) — expected to reproduce the mixture; (b) the constrained ensemble for
   150 ps (`Z1c_constrained`) — 2.9488 flat so far; (c) a corrected gate reference built with the same
   umbrella protocol but the guest lattice-shifted into the indexed window and the gate binned by the
   unwrapped ξ (`scripts/zif8_build_gate_reference_v2.py` → `cache/zif8/gate_reference_v2_T300.npz`).
   Results appended below.

**Consequences.**  The Z1 conditional gate is re-scored against the corrected reference (below).  The
mean-force reference F(ξ) is unaffected (it uses φ, which is exactly periodic, as intended).  But the
legacy ZIF-8 hidden-gate diagnostics — p_ref(A_gate | band), J_gate(t), T_gate, the `conditional_limited`
classifier branch and the "gate divergence indistinguishable between arms" finding of
`docs/ZIF8_RESULT.md` — were all computed on the same mixture on both sides; they measure the fraction
of walkers at the indexed window as much as the gate state, and need re-examination (not done here).

**Z1c (150 ps, 256 replicas each, random pool frameworks).**  Library `run_umbrella` at one window centred
on ξ = 0 with UNSHIFTED pool frameworks: in-band ⟨A_gate⟩ = 2.861 (block sd 0.001) stationary
from 20 to 150 ps — the stored reference's 2.86 reproduced exactly by its own protocol.  Constrained
ensemble with the guest in the indexed window: 2.9488 / 2.9484 / 2.9485 at 15 / 50 / 100 ps, i.e. **the
held-guest gate is a true stationary state, not a transient**, and the two protocols differ only in which
window the guest occupies.  **Corrected gate reference** (`cache/zif8/gate_reference_v2_T300.npz`; same umbrella protocol, guest
lattice-shifted into the indexed window, gate binned by unwrapped ξ; 8 windows × 128 replicas × 90 ps,
split-half JS 4.3e-05, 0% frames at the image window): per-sub-bin ⟨A_gate⟩ =
2.894, 2.916, 2.934, 2.946, 2.952, 2.952, 2.950, 2.948 for ξ from −0.875 to +0.875 — a real induced-fit profile, the gate opening
by 0.15 Å in radius as the guest arrives (stored reference: 2.852, 2.861, 2.865, 2.865, 2.861, 2.856, 2.855, 2.847).

**Z1 conditional gate re-scored against the corrected reference (`Z123/z1_gate_rescored_v2.json`): PASS.**

| site | ⟨A⟩ Z1 constrained (sd) | ⟨A⟩ corrected ref (sd) | ⟨A⟩ stored ref | TV vs corrected | TV vs stored | TV halves |
|---|---|---|---|---|---|---|
| left_band | 2.9343 (0.051) | 2.9337 (0.051) | 2.8648 | 0.006 | 0.395 | 0.003 |
| window_plane | 2.9491 (0.052) | 2.9490 (0.052) | 2.8629 | 0.002 | 0.468 | 0.004 |
| peak_band | 2.9480 (0.053) | 2.9478 (0.053) | 2.8471 | 0.004 | 0.547 | 0.005 |

max |Δ⟨A⟩| = 0.0006 Å (threshold 0.02), max TV(rec, ref) − TV(halves) = 0.002 (threshold 0.05).
**The fixed-ξ constrained BAOAB operator reproduces both the mean force and the hidden-gate conditional of
the umbrella ensemble.  The operator is validated; the frozen hard stop was triggered by a defective
reference and is withdrawn on the corrected one.**

## Implications for Z4 / Z5 (design only; nothing launched)

* **Repair dose and cadence.**  With τ* ≈ 100 inner steps and a first-order injection of ≈ 0.5 |F′|
  per 2-bin move, the WCA/pentane recipe (5 inner steps every 5 outer steps) would repair nothing
  (0–5 %).  The affordable design keeps inner/outer ≤ 1: an OT opportunity every ≥ 100 outer steps
  (50 fs) with ≈ 100 inner steps for the moved walkers (≈ 80 % of the lag removed), or an
  opportunity every 5 steps with repair guarded to walkers moved ≥ 1 bin.  The per-opportunity cap
  stays at 2 bins (0.30 Å); 8-bin moves overlap ring atoms.
* **Headroom.**  The corrected 300 K baseline is already at e_F(T) = 0.084 kJ/mol against a 39 kJ/mol
  barrier, and uniform FR is neutral there (+0.8 %).  OT can only pay where ABF has an
  establishment deficit, so Z4 (ABF-only screen: lower temperature and/or smaller replica budget,
  per-bin adequacy, T_cover/T_marg/T_gate of the ZIF-8 prereg classifier) must pick the cell from
  ABF-only diagnostics before any OT arm exists.  A new temperature also needs a new umbrella
  reference (~2–3 GPU-h at the legacy protocol).
* **Cost.**  One corrected-baseline production arm (16 seeds × 384 replicas × 300 ps) took 9.5 h on
  the shared GPU 1; six arms with three repaired at inner/outer = 1 would be ≈ 85 GPU-h.  A reduced
  scale (8 seeds × 128 replicas × 150 ps, batch 1024 keeps the kernel saturated) brings a six-arm
  block to ≈ 10–12 GPU-h plus ≈ 4 h for the reference and screen — a one-day job, but a scale the
  user should confirm.

## Z4 — ABF-only budget ladder at 300 K (corrected baseline, 8 seeds × 150 ps; `Z4/`, `Z4/cell_choice.json`)

| cell | N | T_cover/T | T_marg/T | transits | verdict (T_cover/T_marg, amendment A2) |
|---|---|---|---|---|---|
| B1 | 64 | 0.18 | inf (censored by the 30 ps warm-up) | 953 | intermediate |
| B2 | 96 | 0.17 | inf (censored) | 1456 | intermediate |
| B3 | 128 | 0.18 | 0.81 | 1965 | intermediate |
| **B4** | **192** | 0.16 | **0.30** | 2960 | **establishment_limited → chosen** |
| control | 384 (300 ps) | 0.08 | 0.11 | 26 984 | abf-sufficient-like (gate legacy) |

Discovery is never limiting.  The T_gate clock could not be used for Z4 (amendment A2: the first
unwrapped-band implementation still produced a periodic-image mixture, confirmed by B1's in-band
⟨A_gate⟩ 2.845–2.870 against the corrected 2.89–2.95; fixed for Z5).

## Z5 calibration (N 192, 2 seeds × 75 ps; `Z5/calibration/alpha_star.json`)

J = ∫ KL(p̂ ∥ U) dt: A 3612, F 3114 (uniform FR at 0.05 flattens the marginal here), T 2509 / 1554 / 622
at α 0.03 / 0.10 / 0.30 → the [0.9, 1.1] band is unreachable, **α* = 0.03** by the frozen fallback
(mean move 0.0039 rad = 0.06 bins per 100-step opportunity, nothing capped).

## Z5 pilot — six arms, 8 seeds × 192 replicas × 150 ps (`Z5/pilot/REPORT.md`, closed 2026-09-07 05:08 UTC)

C* = 57.6 M walker-steps; repaired arms carry 46 M inner steps per seed (inner/outer = 1) and are read at C*.
e_F at h_read 0.05 Å vs the umbrella F; D_gate = mean JS of the corrected in-band gate law vs the
corrected reference.  ε_A = 0.1056 kJ/mol against a 39 kJ/mol barrier.

| arm | I_F^(C) | e_F(C*) | e_F(end) | D_gate(C*) | C(ε_A)/C* | transits |
|---|---|---|---|---|---|---|
| A | 1.098 | 0.1056 | 0.1056 | 0.0002 | 0.62 (6/8) | 2863 |
| F | 1.110 | 0.1273 | 0.1273 | 0.0002 | 0.83 (2/8) | 2240 |
| T | 1.120 | 0.1079 | 0.1079 | 0.0002 | 0.78 (5/8) | 2941 |
| R | 1.095 | 0.0829 | 0.0746 | 0.0003 | 0.52 (7/8) | 392 |
| F+R | 1.121 | 0.0879 | 0.0857 | 0.0003 | 0.79 (7/8) | 338 |
| T+R | 1.110 | 0.0785 | 0.0672 | 0.0003 | 0.62 (8/8) | 447 |

| contrast | ΔI_F^(C) [CI95] wins | Δe_F(C*) [CI95] wins |
|---|---|---|
| T vs A | +0.9 % [−5.0, +5.5] 4/8 | −0.4 % [−20.9, +21.4] |
| T vs F | −0.0 % [−2.1, +3.1] | −11.2 % [−30.2, +4.7] |
| F vs A | −0.0 % [−0.6, +2.3] | **+17.4 % [+5.7, +42.2] 1/8** |
| R vs A | +0.1 % [−3.0, +4.3] | −21.3 % [−43.5, +32.5] |
| F+R vs F | +1.0 % [−2.3, +3.8] | −27.0 % [−51.3, −15.3] 7/8 |
| T+R vs T | −0.0 % [−2.1, +3.4] | **−25.6 % [−32.3, −12.0] 8/8** |
| T+R vs R | −0.3 % [−0.9, +3.2] | −3.5 % [−38.2, +18.0] |
| T+R vs F+R | −1.3 % [−3.6, +3.3] | −15.9 % [−32.1, +30.0] |
| T+R vs A | +0.7 % [−1.8, +4.0] | **−28.2 % [−39.9, −8.3] 7/8** |

Genealogy floors met (ESS/N ≥ 0.41).  D_gate is at the finite-sample floor (2–3 × 10⁻⁴) for every arm.
**No contrast is positive by the prereg rule; go = False.**  The 16-fresh-seed block launched
speculatively on GPU 3 is therefore a supplementary replication, not a confirmation.

## Reading (ZIF-8, 300 K, 192 × 150 ps)

1. **Gentle OT is neutral on ZIF-8**: ΔI_F +0.9 % [−5.0, +5.5], final −0.4 %.  OT does flatten the walker
   marginal (final KL to uniform 0.027 vs ABF 0.031; T+R 0.013) but the free-energy error does not
   follow: on this cell the error is not marginal-limited (ABF's own final error is 0.11 kJ/mol on a
   39 kJ/mol barrier; Z4's establishment deficit is mild, T_marg = 0.30 T).  Uniform FR is integrated-
   neutral and **worse at the end (+17 %, 1/8)**, so OT ≥ FR again (T vs F final −11 %), consistent with
   pentane and WCA.
2. **Repair buys final accuracy at equal compute, through generic fibre decorrelation, not OT-specific
   repair**: T+R vs T −26 % (8/8), T+R vs A −28 % (7/8), F+R vs F −27 % (7/8), R vs A −21 % (noisy), while
   T+R vs R is −3.5 % and every integrated contrast is null.  The constrained segments decorrelate the
   framework between outer deposits, so each deposit carries more information; the price is half the
   outer time at fixed compute, which cancels the gain in the integrated error.  The same reading as
   WCA's "repair = relaxation", now on a flexible molecular gate, and the opposite sign from pentane,
   where the fibre could not be relaxed at all.
3. **No conditional damage anywhere**: with the corrected gate diagnostic, every arm reproduces the
   held-guest gate law to the sampling floor; at the calibrated dose (0.06 bins per 50 fs) the Z2
   injection scales to ≈ 1 % of |F′| per event and is invisible.
4. The three-system picture: OT accelerates where the marginal is the bottleneck (WCA, pentane) and is
   neutral where it is not (ZIF-8 at 300 K); repair pays where the fibre lags and is affordable (WCA),
   is pure cost where it is frozen (pentane), and on ZIF-8 acts as a final-accuracy decorrelator whose
   integrated value is zero at inner/outer = 1.  The Z2/Z3 mechanism numbers (0.47 |F′| injection per
   2-bin move, τ_gate ≈ 100 steps) stand, but at the dose the blind rule permits the injection is never
   large enough to need repairing.
