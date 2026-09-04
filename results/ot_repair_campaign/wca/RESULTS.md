# WCA OT + repair — results

Preregistration: `docs/WCA_OT_REPAIR_MECHANISM.md` (frozen 18640f9 before any M2 run).
Code: `src/wca_ot_repair.py`, the `ot=` option of `wca_abffr_core.run_sampler_gpu` (commit d09afc5,
34 WCA tests pass, α = 0 reproduces plain ABF).  GPU 3 only.

## M1 — single-event lift-and-repair on conditionally equilibrated fibres (DONE, 9.7 min)

Setup: 1024 replicas per site, z_src ∈ {0, 0.2, 0.4, 0.6, 0.8, 1.0}, fibre equilibrated with the
TI reference's own projected constrained scheme (3000 + 1000 steps), lifted by Δz with
`project_dimer_to_z`, repaired at fixed z_dst; the estimator's local-mean-force sample recorded at
every repair step against F'_ref(z_dst).  Files: `M1/single_event.{json,npz}`, `M1/figures/`.

**Operator consistency.** Stationary mean force at every source site within ±0.10 of the reference
(−0.03, −0.05, +0.04, −0.09, −0.01, −0.10; SE ≈ 0.1) — the projected operator IS the reference's
law at dt = 2e-3 (W1b's lesson holds).

**H1 — the lift injects a first-order conditional error.**  Injected bias b(0) − b_∞ vs |Δz|
(median over sites; |F'_ref| ≈ 3.5–5.4 on these sites):

| \|Δz\| | 0.01 | 0.02 (≈ the 2-bin cap) | 0.05 | 0.10 | 0.20 |
|---|---|---|---|---|---|
| \|b(0)\| | 5.1 | 10.5 | 30.5 | 87.5 | 237 |
| per unit Δz | 512 | 525 | 609 | 875 | 1185 |

Linear in the move (≈ 500·Δz up to 0.05, steepening once the lift creates overlaps): the bath shell
lags the dimer at first order, so the "damage is O(α²)" hope does not hold for the mean force.  A
single capped move injects ≈ 2× the mean force itself.

**H2 — projected repair removes it, fast at first, then slowly.**  Fraction of the injection
remaining after m repair steps (median over sites; noise floor per snapshot ≈ 2.5 in absolute units,
so fractions for |Δz| ≤ 0.02 are unresolved below ~0.2):

| \|Δz\| | m = 1 | 2 | 5 (≈ τ_f) | 10 | 20 | 40 | 80 |
|---|---|---|---|---|---|---|---|
| 0.05 | 0.36 | 0.30 | 0.18 | 0.13 | 0.09 | 0.01 | 0.02 |
| 0.10 | 0.30 | 0.20 | 0.13 | 0.09 | 0.07 | 0.03 | 0.03 |
| 0.20 | 0.24 | 0.16 | 0.07 | 0.05 | 0.04 | 0.03 | 0.01 |

Absolute residual for |Δz| = 0.10 after 5 / 10 / 20 steps: 10.9 / 6.7 / 5.9 — i.e. 1–2× F'_ref
still present after 4 τ_f; for |Δz| = 0.05: 5.0 / 4.0 / 2.3.  The decay is fast (≈ 80 % gone in
one τ_f) but the tail is slow (the 0.3–0.7 t.u. ACF tail measured in W0), and what ABF sees is the
absolute residual, not the fraction.

**Lift safety.** For |Δz| ≤ 0.05 the post-lift max force (125–127) and overlap fraction (0.033)
equal their equilibrium values (125, 0.032); |Δz| = 0.10: ΔV +5.9, fmax 147; |Δz| = 0.20: ΔV +80,
fmax 393, overlaps doubled.  The 2-bin cap is safe by a wide margin.

**Reading.** The mechanism is confirmed and quantified: cross-fibre transport on WCA carries the
old bath along and deposits a mean-force error of ≈ 500 per unit z moved; constrained repair
recovers most of it within one τ_f but needs ≳ 20–40 inner steps per move to reach the noise
floor.  With one OT opportunity every 5 outer steps, that is a notional repair cost ρ ≈ 1 (5
steps) to 4–8 (20–40 steps) — the compute question M2 answers.

## M2 — in-sampler arms T0 / TR (DONE 2026-09-04 19:05 UTC; 4 seeds 820–823; `M2/analysis.json`, `M2/figures/`)

Arms: **T0** = ABF + OT (α 0.1, cap 2 bins, every 5 steps from step 20 000, no repair); **TR½** = the
same + 5 projected repair steps per moved walker per event (0.83× extra force evaluations,
charged).  The preregistered 10-step block (TR1, 1.67×) was cancelled after TR½ showed that the
repair does not touch the deposit bias (below); comparators A, F, R (= ABF + untargeted projected
relaxation, W1b ρ = 1, 0.83×), F+R on the same seeds.  Read-out h\*\* = 0.00625, window [−0.1, 1.1].

**Dose actually delivered.**  Every walker is moved at every event, but once OT has flattened the
marginal the moves are tiny: mean |Δz| = 0.0015 (a sixth of a bin), maximum = the cap, 0 % capped.
The lift's injection per event is therefore ≈ 500 × 0.0015 ≈ 0.75 per walker — a tenth of the
mean force — and the outer dynamics relax it within ≈ τ_f anyway.

**Mechanism in the sampler (deposit-free, first outer deposit after an event vs F'_ref, RMS over
136 bins).**  T0: 0.474 (signed mean −0.28) against |F'_ref| RMS 4.96 → **H1 not supported at
this dose** (ratio 0.10; the preregistered threshold was 0.5).  TR½: pre-repair 0.415, post-repair
0.430 → **the 5-step repair removes nothing of it** (H2 not supported): the residual is the
ordinary online-estimator offset every deposit carries, not the lift's damage.  What the repair
does do is what W1b's untargeted relaxation did — equilibrate the solvent shell of every walker
every 5 steps.

**Error at h\*\* (paired, 4 seeds, descriptive CIs).**

| contrast | ΔI_F | wins | Δe_F(T) | wins |
|---|---|---|---|---|
| T0 vs A | −15.9 % [−21.3, +6.2] | 3/4 | **−34.8 %** [−44.9, −25.8] | 4/4 |
| TR½ vs T0 | −12.0 % [−20.5, −3.6] | 4/4 | **−44.9 %** [−48.9, −37.4] | 4/4 |
| TR½ vs A | **−24.3 %** [−27.5, −15.5] | 4/4 | **−64.1 %** [−65.5, −62.0] | 4/4 |
| TR½ vs R | −13.9 % [−21.0, +1.6] | 3/4 | −44.5 % [−48.1, −40.3] | 4/4 |
| TR½ vs F | −8.3 % [−12.5, +4.6] | 3/4 | −38.2 % [−41.7, −24.7] | 4/4 |
| **TR½ vs F+R** | **+0.0 %** [−8.0, +10.2] | 2/4 | −2.1 % [−12.8, +5.7] | 3/4 |
| T0 vs F | +3.8 % [−8.6, +31.5] | 1/4 | +9.4 % [+2.7, +40.7] | 0/4 |

**Compute** (C = N·step + inner steps, exact; median curves): force evaluations to reach A's final
accuracy ε_A — F 0.30×, T0 0.33×, TR½ 0.35×, F+R 0.39×, R 0.44× of A's; TR½ reaches F's final
accuracy at 6.1e7 evaluations, F itself only at 1.2e8, F+R at 1.2e8.  Total compute: T0 1.00×,
TR½ 1.83× A.

**Reading against the preregistered rules.**
* H1/H2 (the lift injects ≥ 50 % of F' and repair removes it): **not supported** at the matched
  dose — the injection is ≈ 10 % of F' and the repair leaves it unchanged.  The M1 single-event
  experiment is where the mechanism lives (linear injection ≈ 500 per unit z, 80 % repaired in one
  τ_f, slow tail); in the sampler the capped, self-limiting OT never moves a walker far enough for
  it to matter.
* H3 (TR vs T0 ≤ −10 %): supported (−12 %, 4/4; final −45 %) — but by relaxation, not by repair of
  OT damage (the same −36 % endpoint gain W1b found for relaxed ABF and relaxed FR).
* H4 (TR vs R): −13.9 %, 3/4, final −44.5 % 4/4 — OT's allocation adds to relaxation alone.
* **Algorithmic gate** (TR beats A by ≥ 10 % on I_F and reaches ε_A at ≤ 1× A's compute):
  **passes** (−24 %, 0.35×) — the frozen prediction that it would fail was wrong, because the
  injection at this dose is negligible and the relaxation's endpoint gain is large.
* **Allocator contest at matched fibre treatment (TR½ vs F+R): a tie** (+0.0 % / −2.1 %).  Raw OT vs
  raw FR: FR slightly ahead at the end (T0 vs F final +9.4 %, 0/4), i.e. the fibre-preserving
  allocator carries a small, consistent endpoint advantage — but nothing like the gateway's +65 %.

**Bottom line for WCA.**  With a displacement cap and a gentle, self-limiting strength, Wasserstein
reallocation along z is a sound accelerator on its own (−35 % endpoint, −16 % integrated vs ABF at
1× compute) and matches uniform Fisher–Rao once both get the same solvent relaxation.  The
cross-fibre damage the theory worries about is real and first-order per unit moved (M1), but a
capped OT toward a nearly-flat marginal moves so little per event that the outer dynamics absorb
it; constrained "repair" in the sampler is then just generic solvent relaxation, whose value
(−45 % endpoint, ties F+R) is independent of the allocator.  The gateway's OT catastrophe does not
transfer to WCA.  Not tested here (out of the user's scope): larger α without a cap, C60 / NaCl.
