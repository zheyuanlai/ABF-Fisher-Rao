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

## Z1b — resolving the gate-aperture inconsistency

(appended when the random-pool run completes)
