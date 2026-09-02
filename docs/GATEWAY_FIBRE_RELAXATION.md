# Gateway transport, third stage: finite fibre relaxation (exact constrained-OU propagation)

**Date:** 2026-09-02. **Status:** preregistered, frozen by commit before each stage, CLOSED — C1 bucket **C1_gentle** (c\* = 0.5),
C2 primary **SAFE_ACCELERATOR** (transport vs ABF at matched fibre treatment), frozen deployable-allocator clause **not met** (transport vs FR INCONCLUSIVE), cost verdict **not deployable as implemented**.
**Prereg:** [`gateway_fibre_relax_prereg.json`](../configs/transport_campaign/gateway_fibre_relax_prereg.json) (commit 7c11fe1; c\* and h_read\*\* frozen at bcdb395).
**Data:** `results/transport_campaign/gateway_horizontal/{relax_C1, relax_C2}/`. **Parents:** [GATEWAY_HORIZONTAL_TRANSPORT.md](GATEWAY_HORIZONTAL_TRANSPORT.md), [GATEWAY_TRANSPORT_REFRESH.md](GATEWAY_TRANSPORT_REFRESH.md).
**GPU:** 3 only; C1 275 s, C2 474 s. **This is a mechanism / relaxation-timescale experiment, not a compute-efficiency result.**

## The operator

At fixed x the transverse dynamics is an Ornstein–Uhlenbeck process with τ_y(x) = 1/ω(x)², so a constrained
relaxation of duration c τ_y(x) is propagated exactly: `y ← e^{−c} y + sqrt((1 − e^{−2c}) / (β ω(x)²)) z`
(`gateway_core.ou_relax`, `Method.refresh = 'ou'`). c = 0 is the identity bit for bit, c → ∞ is the oracle refresh
of the previous stage, and the old y survives through e^{−c}: this is the non-oracle version of the fibre refresh.
Applied to every walker at every opportunity, after any FR gather and any transport, from the same shared draw
stream. The notional cost of a non-analytic implementation, Σ c/(ω(x)² dt) constrained steps, and the
displacement-weighted relaxation time where the allocator moves mass, τ_move = Σ w_i τ_y(x_i⁺), are recorded.
Seven invariants in `tests/test_gateway_fibre_relax.py`, including the variance-contraction law e^{−2c} on the pure
function.

## Stage C1 — recovery block (seeds 580–587 × 2 inits; A/F/T/P × c ∈ {0, 0.5, 1, 2, 5, ∞}, 24 arms, one batch)

**Read-out first.** The plateau *intersection* rule over all 24 arms picks **h_read\*\* = raw bins**: at 0.0175 the
full-transport arms are 38–53 % off their plateau. Every number below is at raw bins.

**Recovery of the oracle's benefit,** R_X(c) = (I_F(X_0) − I_F(X_c)) / (I_F(X_0) − I_F(X_∞)):

| c | analytic 1 − e^{−2c} | A (abf) | T (ot_exact) | P (ot_full) | F (fr) |
|---|---|---|---|---|---|
| 0.5 | 0.632 | 0.957 [0.95, 1.00] | **0.959 [0.94, 0.97]** | 0.959 [0.96, 0.96] | 0.96 [0.75, 1.05] |
| 1 | 0.865 | 0.995 | 0.991 | 0.992 | 1.05 |
| 2 | 0.982 | 0.998 | 1.000 | 0.999 | 1.13 |
| 5 | 1.000 | 1.000 | 1.000 | 1.000 | 0.91 |

Half a local relaxation time per opportunity recovers 96 % of the oracle's integrated benefit for every allocator
(the FR curve is noisy: F arms across c are noise-paired, not birth-death-draw-paired). The transport-induced flank
excess E(c) = b_flank(T_c) − b_flank(A_c) is gone at c = 0.5 (−0.10 of E(0), then −0.25 ± noise), not tracking e^{−2c}.
The D_cond peak of full transport falls from 0.73 at c = 0 to the 0.026 floor at c = 0.5.

Why the single-move law is far too pessimistic: it describes one move followed by one relaxation. Here relaxation
is applied at every opportunity, while the *physical* fibre contracts by only e^{−0.004 ω²} per opportunity —
0.996 in the basins — so even c = 0.5 per opportunity accelerates fibre relaxation there by two orders of magnitude,
and the steady-state residual is e^{−2c}/(1 − e^{−2c}) of one event's injection instead of a whole transit's. The
dose that matters is relaxation per opportunity against injection per opportunity (|dx| ≈ 1.6e-4 at α\*\*), and
c = 0.5 already wins that race.

Matched-fibre contrasts in C1 (8 seeds, descriptive): T_c vs A_c −23 % at c = 0.5, −27 % at c ≥ 1; T_c vs F_c −14 %
at c ∈ {0.5, 1}; P_c vs A_c −71 % (c = 0.5) to −88 % (c ≥ 2). **c\* = 0.5** by the frozen rule (smallest c with
R_T ≥ 0.9 and T_c not worse than A_c) → bucket **C1_gentle**. Cost at c\*: the all-walker relaxation is **106×** the
outer cost; τ_move = 0.85, i.e. the allocator moves mass mostly in the basins (ω ≈ 1, τ_y ≈ 1), where relaxation is
slowest and, since ω′ ≈ 0 there, irrelevant to the mean force.

## Stage C2 — confirmatory (seeds 800–831 × 2 inits, 64 rows; A_0, F_0, T_0, A_0.5, F_0.5, T_0.5, P_0.5; raw bins)

| tag | contrast | ΔI_F | Δe_F(T) | wins | verdict / decision |
|---|---|---|---|---|---|
| **PRIMARY** | T_0.5 vs A_0.5 | **−33.9 % [−35.6, −31.5]** | −8.8 % [−10.1, −6.0] | 64/64 | **SAFE_ACCELERATOR** (also at 0.0175) |
| S1 | T_0.5 vs F_0.5 | −7.3 % [−13.4, −3.3] | +11.1 % [+8.3, +14.0] | 44/64 | **INCONCLUSIVE** (left −15.1 %, one_right −0.4 %) |
| S2 | F_0.5 vs A_0.5 | −29.4 % [−31.9, −26.2] | −17.6 % | 60/64 | SAFE_ACCELERATOR |
| S3 | A_0.5 vs A_0 (relaxation alone) | −33.9 % [−35.5, −32.6] | **−78.3 %** | 64/64 | SAFE_ACCELERATOR |
| S4 | T_0.5 vs T_0 (finite-c repair) | −58.6 % [−59.9, −58.0] | −82.0 % | 64/64 | — |
| S5 | P_0.5 vs A_0.5 | −74.8 % [−76.2, −73.9] | −12.7 % | 64/64 | SAFE_ACCELERATOR |
| S6 | P_0.5 vs F_0.5 | −65.5 % [−66.8, −64.4] | +8.5 % | 64/64 | OT_better |
| — | F_0 vs A_0 (positive control) | −36.9 % [−38.8, −34.0] | −63.2 % | 64/64 | replicates |
| — | T_0 vs A_0 | +8.4 % [+5.1, +11.2] | +11.4 % | 15/64 | NEGATIVE (as closed) |
| — | T_0.5 / F_0.5 / P_0.5 vs A_0 | −55.8 % / −53.0 % / −83.3 % | −79.8 % / −82.0 % / −80.8 % | 64/64 | — |

Frozen-bias endpoint: PRIMARY +6.8 % [+3.0, +10.5] (T's learned bias slightly worse than A's at c = 0.5); S1 +1.8 %;
S3 −25.6 %; S4 −41.7 %; F_0.5 / T_0.5 / P_0.5 vs A_0: −21.6 % / −20.1 % / −27.9 %. Mechanism at T: b_flank abf +0.041,
A_0.5 +0.020, fr +0.022, F_0.5 +0.017, ot_exact +0.048, **T_0.5 +0.019**, P_0.5 +0.017; barrier error 0.20 / 0.094 /
0.10 / 0.081 / 0.24 / 0.089 / 0.085 kT. Time to ABF's final accuracy: abf 40, F_0 10.6, A_0.5 4.6, F_0.5 3.0,
**T_0.5 2.8**, P_0.5 4.0 (P needs c ≥ 2 for its 0.4); on the notional cost axis t_eff = t (1 + 106) every relaxed arm
is in the hundreds.

**Frozen outcome.** Primary SAFE_ACCELERATOR; the deployable-allocator clause (primary accelerator AND T vs F
OT_better or equivalent) is **not met** because S1 is INCONCLUSIVE: −7.3 % integrated is inside the ±10 % margin,
and the end-point is resolved *worse* than FR (+11 %). The analyzer's label for this branch,
TRANSPORT_NOT_BETTER_AT_FINITE_C, is literally the clause's negation and should be read with the primary: at half a
relaxation time transport beats ABF at matched fibre treatment by a third and edges FR on the integral while losing
to it at the end. With the oracle (c = ∞) the same contrast was −15.4 % and a tie at the end; the 4 % of the oracle's
benefit that c = 0.5 leaves behind is where the transport-vs-FR contest is decided, and FR needs the fibre fix less
(its walkers are residents). At c ≥ 1 in C1 the contrast was −14 % (13/16); a confirmatory block at c = 1 or 2 would
likely reach OT_better, but that is a new prereg, not this one.

## Prediction scorecard

Recorded: c\* = 1 or 2, R_T(0.5) ∈ [0.40, 0.75], flank excess ∝ e^{−2c}, T vs A negative for c ≥ 1, T vs F OT_better
or equivalent, h_read\*\* = 0.00875, cost ≥ 100×, τ_move ≈ 0.01–0.1. Right: T vs A (−34 %, already at 0.5), cost (106×).
Wrong: c\* (0.5, gentler), R_T(0.5) (0.96 — the per-move law is not the repeated-application law), flank excess (gone at
0.5), T vs F (INCONCLUSIVE at 0.5), h_read\*\* (raw), τ_move (0.85: mass moves in the basins, not on the flanks).

## What it means

1. **The mechanism is closed and cheap in relaxation time.** A non-oracle fibre relaxation of half a local
   relaxation time per opportunity repairs transport (−59 % / −82 %), removes its flank excess, and recovers 96 % of
   the oracle's benefit for ABF, FR and transport alike. The fibre carry-over is the whole story of why horizontal
   transport lost, and it is not deep: it is a race between injection and relaxation per opportunity.
2. **Allocation and fibre are separable, and both matter.** At matched fibre treatment transport beats ABF by a
   third and full transport by three quarters; FR at the same treatment beats ABF by 29 %. Whether transport beats
   FR depends on how complete the fibre treatment is (oracle: yes by 15 %; c = 0.5: inconclusive).
3. **Not deployable as implemented.** Relaxing every walker for c τ_y(x) costs 106× the outer dynamics, because the
   allocator moves mass in the basins, where τ_y ≈ 1 and the mean force does not depend on y at all (ω′ ≈ 0). The
   right targeting is not "where mass moves" but "where the mean force depends on the fibre" (|ω ω′| large, the
   flank): a flank-only relaxation would cost O(1)× the outer dynamics. That design, and a WCA version whose solvent
   relaxation is charged in full, are the next preregs; the gateway now says the molecular algorithm needs about
   half a perpendicular correlation time per move where the mean force is fibre-sensitive.

## Reproduce

```bash
CUDA_VISIBLE_DEVICES=3 python -u scripts/run_gateway_fibre_relax.py --stage C1     # 275 s
python scripts/analyze_gateway_fibre_relax.py --stage C1                            # h_read**, recovery curves, c*
CUDA_VISIBLE_DEVICES=3 python -u scripts/run_gateway_fibre_relax.py --stage C2     # 474 s
python scripts/analyze_gateway_fibre_relax.py --stage C2
CUDA_VISIBLE_DEVICES="" PYTHONPATH=src python -m pytest tests/test_gateway_fibre_relax.py -q
```
