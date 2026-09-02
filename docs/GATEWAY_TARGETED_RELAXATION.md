# Gateway transport, fourth stage: targeted, budgeted fibre relaxation

**Date:** 2026-09-02. **Status:** preregistered, frozen by commit before each stage, CLOSED — D0 PASS, D1 ρ\* = 1 with the mechanism check passing, D2 **GATE_MET by the frozen definition** (transport vs ABF at matched treatment −15 %, compute to ABF's accuracy 0.67×), **with the secondary evidence pointing the other way for the practical decision**: FR with the same targeted relaxation is the strongest and cheapest arm found in the whole series, and transport at matched treatment ends worse than ABF.
**Prereg:** [`gateway_targeted_relax_prereg.json`](../configs/transport_campaign/gateway_targeted_relax_prereg.json) (commit 007d90a; ρ\* frozen at d288a5c).
**Data:** `results/transport_campaign/gateway_horizontal/targeted_{D0, D1, D2}/`. **Parents:** [GATEWAY_FIBRE_RELAXATION.md](GATEWAY_FIBRE_RELAXATION.md) and its parents.
**GPU:** 3 only; D0 86 s, D1 322 s, D2 549 s.

## The algorithm

Two landscape-free ingredients on top of the previous stage's exact constrained-OU relaxation:

- **Where.** A second-moment accumulator Sf2 = Σ f² is deposited with the ABF sums; the online sensitivity field is
  `v̂_t(z) = max(E[f² | z] − E[f | z]², 0)` with the online kernel (`gateway_core.vhat_from`). Analytically
  Var(f | x) = 2β⁻² (ω′/ω)², so the field is large exactly where the mean force depends on the fibre.
- **How much.** Per opportunity, `c_i = ½ [log(2 a_i / (λ τ_i))]₊` with a_i = v̂_t(x_i), τ_i = 1/ω(x_i)², and λ set by
  bisection so that Σ c_i τ_i = ρ × (outer steps per opportunity): the notional constrained-MD cost is ρ times the outer
  dynamics by construction (`budgeted_relaxation`). Zero budget is the identity bit for bit. The same policy is applied
  to ABF, FR and transport; a transport-specific variant weights a_i by |Δx_i| (T^move). The region |x| < 0.35 is used
  only as a diagnostic label. Eight invariants in `tests/test_gateway_targeted_relax.py`.

## D0 — the estimator (plain ABF, 16 rows)

corr(v̂_T, analytic) 0.861; corr with the resolution-matched reference s(v_ref C)/(s(C)+min_count) **0.932** (requirement
0.90, PASS); peak at x = −0.18 (analytic −0.20). The estimator carries a floor toward the basin edges (0.58 at x = ±1.5;
69 % of its mass within |x| < 0.35 against 99.8 % for the truth): the kernel window's own mean-force gradient,
(F″h)² ≈ 0.65 at x = 1.5 — not a fibre effect. Post hoc, a per-bin-variance-then-smooth estimator (v2, added as an
inert opt-in `sensitivity='binvar'`) has corr 0.998 / 0.939, floor 0.004 and 95 % of its mass on the flank. The
frozen estimator was kept for D1/D2 (D0 passed); the τ-weighting of the allocation turned out to suppress the floor
anyway (below).

## D1 — cost ladder (8 seeds × 2 inits; A/F/T/T^move × ρ ∈ {0.25, 0.5, 1, 2}, anchors, all-walker c = 0.5 references)

| ρ | flank budget share (A / F / T) | displacement share on flank (T) | active walkers | mean c (active) | retention A / T |
|---|---|---|---|---|---|
| 0.25 | 0.99 / 0.99 / 0.99 | 0.21 | 10–11 % | 1.4 | 0.51 / 0.38 |
| 0.5 | 0.98 / 0.98 / 0.99 | 0.19 | 12–13 % | 1.6 | 0.73 / 0.59 |
| 1 | 0.98 / 0.98 / 0.99 | 0.18 | 13–14 % | 1.7 | 0.88 / 0.79 |
| 2 | 0.97 / 0.98 / 0.98 | 0.18 | 15–16 % | 1.9 | 1.00 / 0.95 |

**The mechanism check passes at every budget:** 97–99 % of the relaxation goes to |x| < 0.35 while 18–22 % of the
transport displacement is there — the policy separates *where mass moves* from *where conditional equilibration
matters* without being told either, on 10–16 % of the walkers. Retention of the 106×-cost benefit (I_F): 0.79 for
transport and 0.88 for ABF at ρ = 1 (ChatGPT's 70–90 % prediction holds; my ≥ 0.8 is at the boundary), 0.95 / 1.00
at ρ = 2. Matched-treatment T vs A: +3.7 % (ρ 0.25), −4.6 %, **−11.0 % [−18.9, −0.05]** (ρ 1), −23.2 % (ρ 2); T vs F:
+49 %, +36 %, +19 %, −0.2 %. Compute to ABF's final accuracy relative to plain ABF: T 0.92 / 0.81 / **0.67** / 0.45,
A 0.61 / 0.46 / 0.25 / 0.38, plain FR 0.27. **ρ\* = 1** by the frozen rule (T vs A ≤ −10 % and C_T/C_A0 ≤ 0.8),
gate_D1 true (Fig. J).

## D2 — confirmatory at ρ = 1 (seeds 1300–1331 × 2 inits, 64 rows, raw bins, cluster bootstrap by seed)

| tag | contrast | ΔI_F | Δe_F(T) | wins | frozen bias | decision |
|---|---|---|---|---|---|---|
| **PRIMARY** | T_1 vs A_1 | **−15.4 % [−17.1, −13.0]** | **+16.4 % [+11.6, +24.1]** | 55/64 | +22.5 % | ACCELERATION_POSITIVE_WITH_REVERSAL; scientific success (frozen) = true |
| S1 | T_1 vs F_1 | **+17.2 % [+12.7, +24.5]** | +135 % | 6/64 | +16.1 % | **FR_better** |
| S2 | T^move_1 vs T_1 | +0.3 % [−0.2, +0.8] | +1.8 % | 27/64 | +0.4 % | no difference |
| S3 | F_1 vs A_1 | −25.2 % [−28.3, −22.8] | −50.2 % | 60/64 | +6.5 % | SAFE_ACCELERATOR |
| S4 | A_1 vs A_0 (relaxation alone) | −29.8 % [−32.2, −28.2] | −68.8 % | 64/64 | −27.0 % | SAFE_ACCELERATOR |
| S5 | T_1 vs T_0 (repair) | −46.0 % [−47.6, −44.9] | −67.3 % | 64/64 | −31.2 % | — |
| — | F_0 vs A_0 (positive control) | −35.3 % [−37.0, −32.8] | −66.1 % | 64/64 | −9.8 % | replicates |
| — | T_0 vs A_0 | +11.3 % [+7.9, +13.6] | +11.2 % | 4/64 | +35.7 % | NEGATIVE (as closed) |
| — | A_1 / F_1 / T_1 vs A_0 | −29.8 % / **−48.6 %** / −40.0 % | −68.8 % / **−84.9 %** / −63.8 % | 64/64 | −27 % / −21 % / −8 % | — |
| post hoc | F_1 vs F_0 | −25.7 % [−27.3, −21.1] | −55.2 % | 60/64 | −9.6 % | — |

Final errors: abf 0.00524, A_1 0.00156, F_0 0.00183, **F_1 0.00079**, T_0 0.00578, T_1 0.00188. Flank budget share
0.97–0.99 for every targeted arm (mechanism check true). Both inits agree in kind on every contrast.

**Compute to a fixed accuracy** (outer-time-equivalent, relaxation charged at ρ = 1):

| accuracy | ABF | plain FR | A_1 | **F_1** | T_1 |
|---|---|---|---|---|---|
| e₀/8 | 4.2 | **3.0** | 8.0 | 5.6 | 4.8 |
| ABF's final (0.00524) | 40 | 10.0 | 9.6 | **6.4** | 26.8 |
| FR's final (0.00183) | never | 40 | 64 | **14.4** | never |

Frozen deployability: C_{T_1}/C_{A_0} at ABF's final accuracy = 0.67 < 0.8 → true. **OUTCOME: GATE_MET** by the frozen
definition (scientific success AND deployability, both relative to plain ABF).

## Reading the result honestly

The gate was defined against plain ABF, as proposed, and it is met. Every secondary contrast, all preregistered,
says transport is nevertheless the wrong allocator to carry forward:

1. **At the same targeted treatment, FR beats transport** by 17 % integrated and 135 % at the end (6/64), and its
   learned bias is 16 % better. Transport at matched treatment ends *worse* than ABF (+16 %, resolved) with a 22 %
   worse learned bias: at ρ = 1 the targeted budget repairs 79 % of transport's damage, and the remainder is what the
   end-point sees.
2. **On the compute axis, transport is the most expensive accelerator.** To ABF's final accuracy: F_1 6.4, A_1 9.6,
   plain FR 10.0, T_1 26.8, ABF 40. At e₀/8 plain FR (3.0) beats every relaxed arm; T_1 needs 4.8. Transport plus
   relaxation is never the cheapest route to any threshold.
3. **FR plus targeted relaxation is the strongest deployable arm of the whole series:** −48.6 % / −84.9 % vs ABF,
   −25.7 % / −55.2 % vs plain FR, final error 0.00079 (the oracle-refresh level, 0.00078), ABF's final accuracy in 6.4
   outer-time units instead of 40 (6.3×), FR's own final accuracy in 14.4 instead of 40 (2.8×), at 1× extra cost spent
   on 13 % of the walkers where the mean force is fibre-sensitive.
4. The |Δx|-weighted importance changes nothing (T^move = T within 0.3 %): the walkers whose fibre matters are the
   flank residents, who barely move; what matters is sensitivity, not displacement — the same lesson the all-walker
   stage taught through τ_move.

**Recommendation (my judgment, separated from the frozen rule):** do not build the WCA *transport*. Build the WCA
version of **FR + online-targeted constrained solvent relaxation**: the sensitivity field is the conditional variance
of the local force from the ABF samples the code already has, the budget rule is the water-filling formula with
τ_i the local solvent relaxation time, and the cost is charged in full. Everything in that recipe transferred from
this stage; nothing in it needs the landscape.

## Prediction scorecard

Right: D0 passes; the mechanism check (budget > 80 % on the flank, displacement < 30 % there); retention 0.79 at ρ = 1
(boundary); T^move ≈ T; T vs F not required and indeed not obtained. Wrong: ρ\* = 1, not 0.5; T vs A −15 %, not
−25 to −35 %; compute ratio 0.67, not 0.1–0.2; retention 0.38 at ρ = 0.25 (predicted ≥ 0.5); D0 corr 0.93, not ≥ 0.95;
and the qualitative expectation that transport would be the allocator worth deploying.

## What the four stages established together

The marginal is not the ingredient (H1: FR beats transport by 65–70 % at matched dose); the fibre is (R1a: an
oracle fibre refresh repairs transport by 62 % and cuts ABF's own final error by 83 %); half a local relaxation time
per opportunity is enough (C1_gentle: 96 % recovery); and, targeted by an online sensitivity field at 1× cost, the
best combination is Fisher–Rao reallocation plus fibre relaxation where the mean force depends on the fibre. Transport
is a better allocator only when its fibre damage is fully repaired, and paying for that repair costs more than the
allocation gain is worth; FR never pays it because its copies stay on their fibres.

## Reproduce

```bash
CUDA_VISIBLE_DEVICES=3 python -u scripts/run_gateway_targeted_relax.py --stage D0 && python scripts/analyze_gateway_targeted_relax.py --stage D0
CUDA_VISIBLE_DEVICES=3 python -u scripts/run_gateway_targeted_relax.py --stage D1 && python scripts/analyze_gateway_targeted_relax.py --stage D1
CUDA_VISIBLE_DEVICES=3 python -u scripts/run_gateway_targeted_relax.py --stage D2 && python scripts/analyze_gateway_targeted_relax.py --stage D2
CUDA_VISIBLE_DEVICES="" PYTHONPATH=src:scripts python -m pytest tests/test_gateway_targeted_relax.py -q
```
