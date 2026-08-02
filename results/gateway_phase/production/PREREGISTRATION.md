# Preregistration — entropic-gateway mFR comparison

**This document is committed before any Fisher–Rao arm has been run on this system.**
Its purpose is to make the anchor cell a consequence of a rule rather than of a result.
The frozen classification it refers to is
`results/gateway_phase/production/phase_classification.frozen.json`, whose `raw_sha256`
pins the ABF-only artifact it was derived from.

## 1. Why this system exists

Three molecular systems have now been screened and none of them can test mFR:

| system | regime | why mFR cannot be tested there |
|---|---|---|
| butane, pentane (φ₁) | ABF-sufficient | no support deficit forms |
| alanine dipeptide (φ, ψ) | ABF-sufficient | every basin established; oracle mFR equivalent |
| valine dipeptide (φ, χ₁) | ABF-sufficient | all 8 regions established within 52 ps of 300 ps |
| pentane R15 distance | discovery-limited | the state is never reached; there is nothing to clone |

The regime mFR is *supposed* to serve — a state **found early** but **populated slowly** —
has never been produced deliberately. Trying further peptides is example hunting: if the
slow coordinate is in the CV, ABF flattens it and the system is ABF-sufficient; if it is
omitted, the system is discovery-limited. This model creates the missing regime on purpose.

## 2. The model and the screen

Smooth entropic channel, analytic free energy, so there is no reference error to confound
anything:

    V(x,y) = H(x²−1)² + ½ ω(x)² y²,   ω(x) = ω_out + (ω_in − ω_out) e^{−x²/2s²},   ξ = x
    F(x)   = H(x²−1)² + β⁻¹ log ω(x) + C,        r = ω_in/ω_out

ABF only, no FR arm of any kind: 4 `s` × 4 `r` × 4 `β` × 16 seeds × 2 initialisations =
2048 runs, N = 2048 walkers, T = 40, dt = 4×10⁻⁴. `βH = 8 kT` is held fixed so the
*dimensionless* landscape `βF(x) = βH(x²−1)² + log ω(x)` is identical in every cell and the
total barrier is `8 + log r` kT throughout; β then varies only the transport speed, at an
unchanged sampling budget.

**β is on the map because the sizing scan showed `(s, r)` alone does not span the
regimes.** Reporting a single `(s, r)` slice would have hidden the axis that does the work.
This is not the forbidden "shorten the run until a deficit appears" move — that cuts
exploration *and* compute together, whereas this cuts exploration at fixed compute — but
the distinction is narrow enough to state plainly rather than bury.

## 3. Classification rule (frozen in `gateway_core`, applied in priority order)

With `T_hit` = first time `P̂_t(B₊) > 0` persistently, `T_est` = first time
`½Q*₊(t) ≤ P̂_t(B₊) ≤ 3⁄2 Q*₊(t)` persistently, both on a trailing window of 5 % of the run,
and `Q*₊(t) = ∫_{B₊} e^{−β(F−F̂_t)} / ∫ e^{−β(F−F̂_t)}` the **bias-aware** target:

1. **discovery-limited** — `T_hit ≥ 0.1 T` in ≥ 25 % of seeds;
2. **establishment-limited** — `T_est − T_hit > 0.25 T`, or below half target for ≥ 0.2 T;
3. **ABF-sufficient** — `T_est − T_hit < 0.1 T`;
4. **intermediate** — anything else.

Discovery is tested first because a state that is not reliably *found* cannot be judged on
how fast it fills.

## 4. The measured map (all 128 cells, reported in full)

| regime | cells |
|---|---|
| ABF-sufficient | 32 |
| intermediate | 42 |
| establishment-limited | 54 |
| discovery-limited | 0 |

β = 2 is ABF-sufficient everywhere; β = 4 intermediate everywhere; β = 8 mixed; β = 16
establishment-limited everywhere. Within a fixed β, `(s, r)` modulates the establishment
gap only weakly and monotonically in the expected direction (narrower and more severe
gateways are harder). **That is a result, not a nuisance, and it is reported as such:** the
regime is set by the ratio of transport time to run time, and the gateway geometry tunes it
at the margin.

No cell of this model is discovery-limited. The discovery-limited regime in this study is
represented by pentane's R15 distance CV, not here.

## 5. The anchor, selected by the rule below and by nothing else

    β = 16,  s = 0.10,  r = 32,  init = left

Selection rule, in order: the headline `left` arm; regime `establishment-limited`; **interior,
not knife-edge** — every neighbour along `s`, `r` and `β` that exists in the map must share
its regime; then the largest median integrated deficit among the survivors.

Measured at this cell (ABF only, 16 seeds): `T_hit/T = 0.040` with **0/16 seeds** late,
`T_est/T = 0.465`, gap `0.43`, below half its target for **42 %** of the run, and a final
occupancy of `0.331` against a bias-aware target of `0.399` — a state that is found in the
first 4 % of the run and is still 17 % short at the end. That is the deficit mFR claims to
repair.

The β = 8 cells were *not* chosen despite being closer to the regime boundary: their gap of
≈ 0.21 sits below the 0.25 threshold and they qualify only through the below-half criterion
at ≈ 0.20 against a 0.20 threshold. That is precisely the knife-edge the rule excludes.

## 6. The arms, declared now

| arm | what it is |
|---|---|
| `abf` | baseline |
| `sham` | **matched sham resampling**: same event times, and the *same realised* clone/delete counts as `fr_oracle`, copied from its partner row; identities drawn uniformly. Separates "mFR steered the population" from "any turnover of this magnitude would have done". |
| `fr_oracle` | FR target built from the analytic `F`; non-deployable, diagnostic |
| `fr_estimated` | FR target from the online EMA of the bias; deployable |

Plus the `one_right` initialisation as a **mechanism control**: one walker starts in the
right basin, so discovery is free and any acceleration measures population establishment
rather than first passage.

## 7. Reported metrics, declared now

`I_F = ∫₀ᵀ ‖F̂_t − F‖_{L²} dt`, `I_{F'} = ∫₀ᵀ ‖F̂'_t − F'‖_{L²} dt`, `T_hit`, `T_est`,
the integrated deficit `∫_{T_hit}^{T} [Q*₊(t) − P̂₊(t)]₊ dt`, and the health gates
`ESS_anc/N ≥ 0.30` and `w_max ≤ 0.05`.

Matched seeds throughout: every arm inside one `(config, seed)` row shares initial
conditions and Langevin noise, so arms are compared on the same trajectory realisation.

## 8. What a null result means

If mFR is equivalent to ABF **here** — in a regime built to order, with an analytic
reference, a state discovered in the first 4 % of the run, and a deficit that persists over
42 % of it — then the claim that mFR accelerates population establishment has failed its
most favourable available test. That outcome is to be reported as prominently as a positive
one, and the health gates are to be reported whether or not they pass.
