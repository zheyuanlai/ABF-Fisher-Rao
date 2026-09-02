# Gateway: Fisher–Rao birth–death vs Wasserstein transport toward the same uniform marginal

**Date:** 2026-09-02. **Status:** preregistered, frozen by commit before each stage, CLOSED — outcome **H1_FR_wins**. **Follow-up CLOSED the same day** ([GATEWAY_TRANSPORT_REFRESH.md](GATEWAY_TRANSPORT_REFRESH.md)): H1 replicates at exact dose (+70 %), an oracle fibre refresh REPAIRS transport (−62 %), and with exact fibres transport beats FR (R1a); the two follow-ups named below were run.
**Prereg:** [`gateway_horizontal_transport_prereg.json`](../configs/transport_campaign/gateway_horizontal_transport_prereg.json) (commit e4e8ca1; alpha\* frozen at 8139850).
**Data:** `results/transport_campaign/gateway_horizontal/{calibration, production}/` (analysis.json, comparison.csv, figures/).
**GPU:** 3 only; Stage A 95 s, Stage B 240 s.

## The question

The advisor's proposal: if the target is a uniform marginal, why clone and kill walkers (Fisher–Rao) instead of
simply *moving* them so the reaction-coordinate histogram becomes uniform (Wasserstein)? Both need no knowledge of
the landscape — the transport map uses the walker positions and the CV domain only — so this is a landscape-free
comparator, unlike count balancing. The gateway is where to ask: ξ = x, the transport is exact
(`(x, y) → (x', y)`), the conditional law `y | x ~ N(0, 1/(β ω(x)²))` is known in closed form, and uniform-FR is a
persistent 2.5× accelerator there on the corrected baseline ([GATEWAY_CORRECTED_BASELINE.md](GATEWAY_CORRECTED_BASELINE.md)).

**Operator** (`gateway_core.horizontal_ot_map`): sort the walkers, `u_i = XMIN + (i − ½)(XMAX − XMIN)/N`,
`X_(i)⁺ = (1 − α_t) X_(i)⁻ + α_t u_i`, `Y⁺ = Y⁻` exactly. In 1-D the rank matching is the W₂-optimal coupling to the
uniform quantile measure and the interpolation is its displacement geodesic. Same opportunities (every 10 steps) and
the same ramp as the FR rate, placed after the Langevin step so a moved walker deposits at the *next* step, like a
clone. No RNG, no reference, no bias, no histogram, N unchanged, every walker keeps its rank. Thirteen invariants
are pinned in `tests/test_gateway_horizontal_transport.py`, including bit-identity of every legacy path against a
fixture generated from the accepted engine before this code existed.

## Stage A — dose matching, blind (seeds 480–487 × {left, one_right})

α\* was chosen by **marginal action only**: `J_KL = ∫₄⁴⁰ KL(p̂ₜˣ ‖ U) dt`, median over 16 rows, matched to the accepted FR
arm (γ = 1.5). No error metric was computed, printed or stored.

| arm | abf | fr_uniform | α 0.0025 | 0.005 | 0.01 | 0.02 | 0.05 | 0.1 | 0.2 |
|---|---|---|---|---|---|---|---|---|---|
| J_KL | 5.35 | 0.477 | 0.676 (×1.42) | 0.295 (×0.62) | 0.133 | 0.063 | 0.024 | 0.012 | 0.006 |
| ∫D_cond dt | 0.771 | 0.762 | 0.783 | 0.805 | 0.836 | 0.861 | 0.884 | 0.893 | 0.921 |

The ladder brackets FR but no point lands inside ±10 %, so the frozen fallback applies: **α\* = 0.0025** (closest in
log ratio). The match is imperfect — matched OT flattens 42 % *less* than FR — and the prereg forbade interpolating.
Transport is a very efficient flattener per unit strength; conditional damage (∫D_cond) rises monotonically with it
(Figure E).

## Stage B — production (seeds 500–531 × {left, one_right}, 64 rows, four arms in one batch)

Every read-out below is computed offline from the same saved accumulators; the engine's own read-out is reproduced
to 1.5e-15. Cluster bootstrap by seed, 10 000 resamples.

| contrast at h_read\* = 0.0175 | ΔI_F | Δe_F(T) | wins |
|---|---|---|---|
| **ot_matched vs fr_uniform (PRIMARY)** | **+65.3 % [+58.2, +71.1]** | **+172.8 % [+156.2, +190.0]** | 0/64 |
| fr_uniform vs abf (positive control) | −32.1 % [−34.7, −30.3] | −59.3 % [−64.1, −55.1] | 64/64 |
| ot_matched vs abf | +14.4 % [+11.3, +15.9] | +10.0 % [+6.8, +14.4] | 4/64 |
| ot_full (α = 1) vs abf | +167.0 % [+155.6, +177.0] | +120.8 % [+110.9, +126.5] | 0/64 |
| ot_full vs fr_uniform | +305.7 % [+277.2, +316.3] | +464.8 % [+422.5, +516.8] | 0/64 |

Decision: **FR_better** (90 % CI [+59.4, +70.1]), on both inits (left +54.6 %, one_right +71.8 %) — not
heterogeneous. Verdicts vs abf: fr_uniform **SAFE_ACCELERATOR** (genealogy floors hold, min ESS/N 0.392, w_max 0.011),
ot_matched **NEGATIVE**, ot_full **NEGATIVE**. Raw bins agree (+61.7 % / +190.5 % primary; FR −33.2 % / −63.2 %), so
none of this is a read-out artefact; at the legacy 0.07 read-out the same ordering holds. The positive control
reproduces the corrected confirmation (−31.9 % / −59.4 % on seeds 400–415) to within a point on fresh seeds.

Secondary: I_F′ (mean-force RMS) is *better* than ABF for ot_matched (−13.7 % [−16.2, −12.4]) while I_F is worse
(+14.4 %) — the uniform occupancy lowers the variance of the flank estimate, but the free energy integrates the
*signed* flank error, which transport raises (below). Frozen-bias endpoint (learned bias re-scored with no adaptation,
no birth–death, no transport): fr_uniform −10.4 % [−13.7, −6.8] vs abf; ot_matched +31.2 %; ot_full +50.1 %. FR's
bias is better; both transport biases are worse. Time-to-accuracy on the median curve: FR reaches ABF's final
accuracy at t = 10.8 (3.70×); neither transport arm ever does.

## Mechanism

**Same marginal, opposite effect on the flank bias.** Fig. C: both flatteners drive KL(p̂ₜˣ‖U) two orders of magnitude
below ABF (ot_full to 1e-4 — exactly uniform), and after t ≈ 5 the empirical conditional diagnostic D_cond sits at its
finite-count floor for every arm. Yet the final mean-force error is localised on the *entering* (left) flank
`0.05 < |x| < 0.3` with a consistent sign, and the arms differ there by a factor 9:

| left-flank mean F′ error at T | abf | fr_uniform | ot_matched | ot_full |
|---|---|---|---|---|
| signed mean (F′_ref flank scale 1.43) | +0.032 | **+0.012** | +0.037 | **+0.104** |
| barrier error F(0) − F(−1), kT | +0.17 | **+0.07** | +0.21 | **+0.44** |

Birth–death never changes the fibre of a configuration when it reallocates it: a copy sits at its parent's x with
its parent's y, so it adds samples at the flank — where the mean force has the highest variance — without
introducing a cross-fibre inconsistency, and the bias goes down. (This is the weaker, sufficient property; FR does
not guarantee that the parent itself was conditionally equilibrated.) Transport carries every walker's y along as its x moves toward
higher ω, so the samples it feeds the flank have E[y²] systematically too large; the deposited `ω ω′ y²` is too
positive and the barrier is over-estimated. Per event the distortion is tiny at α\* (mean D_move 3e-6 nats, |dx|
1.5e-4), but it never stops, and ABF's accumulators never forget. Fig. A shows one event literally: at α\* the
displacement is 0.002; at α = 1 on a mid-run population it is 0.77 with D_move up to 508 nats at the gate centre
(= ½ (ω_in/ω_out)², the closed form).

**Full transport: a uniform marginal is not correct free-energy sampling.** D_cond spikes to 0.6 during the ramp
(30× the floor; the deposits made then are the corrupted ones), then returns to the floor by t ≈ 5: at steady state
the walkers are pinned near their quantiles (|dx| 3.5e-3 per event) and each one's y equilibrates locally, so the
arm behaves like a stratified constrained sampler — the H5 picture. But the scar is permanent: e_F falls as 1/t from
the ramp on (0.056 → 0.023 → 0.012 at t = 8, 20, 40; a factor 4.8 over a factor 5 in t), still 2.2× ABF at T and
+50 % worse on the frozen-bias endpoint. The recorded ping-pong prediction was right about the source (unrelaxed
walkers ejected from the flank, D_move in the hundreds of nats) and wrong about its persistence: it is a transient
whose damage persists only because the estimator is cumulative.

**Transient gain, then reversal — in the transport arm, on a corrected read-out.** Matched OT wins the discovery
race: it moves 27 % of the population into the right basin by t = 5 (ABF 3 %, FR 28 %), its error ratio to ABF dips to
0.53 at t = 2.8, and it reaches e₀/2 and e₀/4 *faster than FR* (τ 2.0 / 2.4 vs 2.2 / 2.6). It then loses the
establishment race: ABF's own error collapses at t ≈ 3–4 while OT's stalls (0.048 → 0.043 → 0.034 at t = 3, 4, 6), and
from t ≈ 3.5 the ratio stays above 1 (1.84 at t = 5, 1.10 at T). The project's recurring signature, here with raw
bins agreeing, so it belongs to the dynamics.

## What it means

The essential ingredient is **not** marginal flattening. Two operators that flatten the x-marginal by nearly the
same amount, with no knowledge of the landscape, give opposite results at the free-energy endpoint: Fisher–Rao
reallocation −32 % / −59 % (persistent, SAFE), reaction-coordinate transport +14 % / +10 % (NEGATIVE), and the
literal proposal — make the current marginal uniform — +167 % / +121 %. The difference is what happens to the fibre:
FR reallocates multiplicity and leaves every configuration on its own fibre (`(x, y) ↦ (x, y)` with altered
multiplicity); transport moves a configuration to another fibre carrying the old transverse coordinate
(`(x, y) ↦ (x′, y)`), which transports the marginal but not the conditional law. On a system whose mean force is
`ω ω′ y²`, that is the whole estimator: **marginal uniformity and conditional correctness are different
requirements, and ABF needs both.** This is the Fisher–Rao-vs-Wasserstein reading of the advisor's question, and it favours the reaction
geometry; the earlier fibre-horizon and clean-v2 closures (bias-dominated failures of *count* manipulation) and this
one (bias-dominated failure of *position* manipulation) point the same way.

Limitations, stated before anyone else does: the dose match is 1.41 not 1.0 (matched OT under-flattens relative to
FR; the neighbouring ladder point over-flattens by 0.62, and its F error was never read — a dose–response at the
error endpoint is a separate prereg); D_cond with 31 bins × 20 walkers is at its finite-count floor after t ≈ 5 and
cannot resolve the O(10⁻³) conditional lag that the signed flank error reveals; the frozen-bias endpoint keeps the
campaign's η = 0.10 KDE convention. Two follow-ups are licensed but NOT run: full transport with the ABF accumulators
started after the ramp (does the scar explain everything?), and transport + constrained fibre relaxation (the
Chapter-3 separation of normal and tangential motion). The WCA molecular version (move the two dimer atoms along
the bond axis, freeze the solvent) is only worth running if one of those repairs the gateway result.

## Reproduce

```bash
CUDA_VISIBLE_DEVICES=3 python -u scripts/run_gateway_horizontal_calibration.py   # Stage A (95 s), blind
CUDA_VISIBLE_DEVICES=3 python -u scripts/run_gateway_horizontal_transport.py     # Stage B (240 s)
python scripts/analyze_gateway_horizontal_transport.py                            # analysis.json, comparison.csv, figures B–E
CUDA_VISIBLE_DEVICES="" python scripts/plot_gateway_transport_event.py            # figure A
CUDA_VISIBLE_DEVICES="" PYTHONPATH=src python -m pytest tests/test_gateway_horizontal_transport.py -q
```
