# WCA: Wasserstein reallocation + constrained fibre repair — mechanism stage (preregistration)

**Frozen 2026-09-04 18:00 UTC, before any M2 run; M1 (single-event, no sampler, no error metric)
had already run and its numbers are quoted as the basis for the dose choice.**  GPU 3 only.
Design source: the user's OT + repair note (operator splitting `ABF -> OT -> lift -> repair -> ABF`,
eqs. (5), (9), (25), (36)-(38)) and the staged plan (A0 invariants, A1 lift safety, B fibre
timescale, C blind dose, D repair dose, then production).  Priority set by the user: **confirm
the mechanism**, not completeness.

## Operators (all existing, reference-consistent)

* z = (r − r0)/(2w); domain [−0.2, 1.2], grid 160, bin 0.0088.
* Lift `project_dimer_to_z`: midpoint and direction kept, bath untouched.
* Repair = the TI reference's own constrained scheme (`frozen_dimer_relax(scheme='projected')`:
  every particle moves one Euler–Maruyama step at the outer dt = 2e-3, dimer re-projected to z),
  validated in W1b to reproduce the reference at every diagnostic site; own RNG; nothing deposited;
  every inner step charged.  τ_f(z) from the W0 map (3–6 outer steps measured; frozen floor 10 dt).
* OT = 1-D rank-matched displacement interpolation toward the uniform quantiles on the sampler's
  z-domain, per-event cap |Δz| ≤ 2 bins = 0.0176, α the strength (`src/wca_ot_repair.py`).

## M1 — single-event mechanism (done 2026-09-04 17:50–18:00; `results/ot_repair_campaign/wca/M1/`)

1024 replicas per site, sites z ∈ {0, 0.2, 0.4, 0.6, 0.8, 1.0}, equilibrated 3000 + 1000 projected
steps (stationary mean force within ±0.1 of the reference at every site — the operator is
consistent), lifted by Δz ∈ ±{0.01, 0.02, 0.05, 0.1, 0.2}, then repaired at fixed z_dst with the
estimator's local-mean-force sample recorded at every step against F'_ref(z_dst).

Findings used below (the full table is in `M1/single_event.json`):
* the injected bias is **linear in the move**, ≈ ±500·Δz (≈ 5 at Δz = 0.01, 10 at 0.02, 30 at
  0.05, 90 at 0.1, 240 at 0.2) against |F'_ref| ≲ 9 — the bath's shell lags the dimer at first
  order, so the "O(α²) damage" hope does not hold for the mean force;
* projected repair removes 80–90 % of it within 5 steps (one τ_f) and ~95 % within 20, with a
  slow tail (still 3–10 after 20 steps for |Δz| ≥ 0.1 — 1–3× F'_ref);
* lift safety: for |Δz| ≤ 0.05 the post-lift max force and overlap fraction equal their
  equilibrium values; at |Δz| = 0.2 the energy jumps by +80 and the overlap fraction doubles.
  The 2-bin cap (0.0176) is inside the safe range.

## M2 — in-sampler arms (fresh runs; seeds 820–823 = the W1/W1b seeds, same initial conditions
and outer noise stream; OT consumes no RNG)

Accepted Case IX cell (β 1, h 2, w 2, n_dim 10, a 1.5), N 1024, 120 000 steps, save 2500, FR
schedule fr_start 20 000 / fr_every 5, read-out bandwidths {0.0125, 0.00625, 0} stored.

| arm | α | cap | repair c | inner steps per moved walker | notional cost ρ |
|---|---|---|---|---|---|
| `ot_a0.1_c0` (**T0**) | 0.1 | 2 bins | 0 | 0 | 0 |
| `ot_a0.1_c0.5` (**TR½**) | 0.1 | 2 bins | 0.5 | ceil(0.5·τ_f/dt) = 5 | ≈ 1 |
| `ot_a0.1_c1` (**TR1**) | 0.1 | 2 bins | 1 | 10 | ≈ 2 |

**Dose rule (blind, marginal-only):** α chosen so that the OT transport per opportunity on the
plain-ABF final marginal (Σ|Δz| over walkers) equals the uniform-FR arm's upper bound
(max_event_fraction 0.02 × N × a half-domain move ≈ 10 z-units): α = 0.1 gives 10.06 (1 % of
moves capped); α = 0.05 gives 5.0, α = 0.2 gives 15 with 69 % capped.  No F-error was consulted.

Comparators already on disk (same seeds): W1 `abf` (A), `fr_uniform` (F); W1b `abf_ptarg1` (R,
projected relaxation at ρ = 1) and `fr_ptarg1` (F + R).

## Endpoints

Read-out h\*\* = 0.00625 (the W1b intersection rule), error = interval L2 of the mean force /
free energy on the eval window vs the corrected TI reference, exactly as `analyze_wca_targeted_relax.py`.

1. **Mechanism (H1/H2), deposit-free:** per-bin mean of the first outer deposit after an OT
   event (`ot_Sf_post/ot_C_post`) minus F'_ref — the injected bias actually entering ABF in T0,
   and its post-repair residual in TR; plus TR's pre-repair sample (`ot_Sf_pre/ot_C_pre`).
   Statistic: occupancy-weighted RMS over bins with ≥ 200 samples; residual fraction
   = RMS_TR / RMS_T0.
2. **Error (H3/H4):** paired ΔI_F and Δe_F(T) (median over 4 seeds, descriptive CIs as in W1) for
   T0 vs A, TR vs T0, TR vs A, TR vs R, TR vs F, TR vs F+R.
3. **Compute:** C(ε) = N·steps + inner replica-steps (exact); C_TR(ε_A)/C_A(ε_A) with ε_A = A's
   final error, and vs F.
4. **Safety:** no NaN, |Δz| mean/max, capped fraction, moved fraction.

## Frozen reading

* H1 supported if T0's post-event deposit bias RMS ≥ 0.5·|F'_ref| RMS on the window (the lift
  contaminates ABF at first order) — predicted from M1: yes.
* H2 supported if TR½ removes ≥ 70 % and TR1 ≥ 85 % of T0's deposit bias (M1 predicts 80–90 %
  at 5 steps, ~95 % at 10–20).
* H3 supported if TR1 vs T0 ΔI_F ≤ −10 %, CI upper < 0.
* **Algorithmic gate:** TR beats A by ≥ 10 % on I_F at h\*\* AND reaches ε_A at ≤ 1× A's compute
  (inner steps charged).  Prediction recorded now: **fails** — with ρ = 1–2 and a first-order
  injection, repair buys back at most what it costs (the W1b lesson), so OT + generic repair on
  WCA is expected to be a correct but not compute-efficient allocator; the paper's claim then
  rests on the mechanism (H1/H2) and on FR's fibre-preserving nature.
* Every arm is reported regardless of sign; nothing is tuned after data are seen.
