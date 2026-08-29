# Fibre-Horizon Information-Conversion Audit — preregistration

**Frozen 2026-08-29, before any run of this campaign.** Branch `information-allocation-fr`.
Fresh campaign, **not** an amendment to `INFORMATION_CONVERSION_AUDIT_PREREGISTRATION.md`.
That audit's verdict (`NO_FINITE_HORIZON_ALLOCATION_OPPORTUNITY`, zero FR pulses executed) stands
unchanged and is not reinterpreted here.

## Why this campaign exists

The previous audit stopped at its Stage-0D gate: median `G_ideal` = 0.052 (K2) / 0.067 (K3)
against a frozen 0.10 threshold. Its Amendment-1 sidecar then established that the planning
horizon had been set from `tau_max(eval)` = 0.096/0.108 — the **cell-mean-series** decorrelation
time, shortened by cell-population turnover — while the physically relevant quantity for a cloned
configuration to generate new conditional-force information is the **fixed-x fibre** decorrelation
time, measured at `tau_fib_max` = 5.27 (K2) / 4.65 (K3), i.e. `H_fib` = 2635 / 2325 steps, ~50×
longer. Re-running the same gate arithmetic across horizons gave median `G_ideal` ≈ 0.54 at
`H_fib`. Those numbers licensed nothing under the frozen rules; this campaign is what they license.

**The question this campaign answers, and the only one:** with a genuine ≈54 % oracle
finite-horizon allocation opportunity available, can one standard Fisher–Rao birth–death pulse
convert it into a real reduction in empirical estimator risk?

## What changes from the previous audit, and nothing else

| | previous | here |
|---|---|---|
| planning + cooldown horizon | `H` = 49 / 55 (cell-mean τ) | **`H_fib`** = 2635 / 2325 (fibre τ) |
| difficulty in the target | `V_j = σ_j² τ_j(cell-mean)` | **`V_j^fib = σ_j² τ_j^fib`** |
| Stage-0 gate | `G_ideal ≥ 0.10` | **`G_ideal ≥ 0.50`** |
| primary endpoint | free-energy time-to-accuracy | **empirical estimator risk `R_FR/R_ABF`** |
| ancestor-ESS ≥ 0.9 | hard gate | **reported, not a gate** (see §5) |

Everything else is inherited verbatim: potential, domain, β = 4, dt = 0.002, K = 256, burn-in to
t = 20 (10 000 steps), 32 allocation cells, geometric mask, `a_j` from the frozen leverage
operator, the estimator (`binned_smooth`, h = 0.05, `min_count` = 1), the chunk-keyed noise bank,
and — critically — the reallocation operator.

**The difficulty and the horizon must agree.** Using a fibre horizon with a cell-turnover-shortened
difficulty would mix two different notions of "how long until new information"; `V^fib` is
recomposed from the stored Stage-0A arrays as `V · (τ_fib / τ_cellmean)` per cell, so `σ²` is
exactly the validated one and only the time factor changes.

## Stage 0 — recompute the target at the fibre horizon (offline, no new dynamics)

Per seed, at the existing t = 20 checkpoint, with the stored counts `C_j`:

    pi* = argmin_pi  sum_j a_j V_j^fib / (C_j + M pi_j),   sum pi = 1,  pi_j >= 1/K,
    M = K * H_fib,      G_ideal = 1 - R(pi*) / R(uniform)

**Gate 0: median `G_ideal` ≥ 0.50 in at least one cell.** Below that in both, stop and report —
the fibre horizon did not deliver the opportunity the sidecar projected, and no pulse runs.

`pi*` is re-solved from scratch. The previous audit's `pi*` (70 % / 88 % of mass in one cell) was
a short-horizon solution and **may not be reused**; a larger budget is expected to spread the
water-filling across several high-value cells, and whatever it does is reported before Stage 1.

## Stage 1 — exactly one Fisher–Rao pulse

Reallocation is **only** `fr_v3.bd_standard`. No systematic resampling, no random turnover, no
transport, no jitter, no count balancing, no alternative birth/death.

    S_i = log( p_hat(z_i) / pi*(z_i) ) - (1/K) sum_k log( p_hat(z_k) / pi*(z_k) )
    P_i = 1 - exp( -|S_i| * dtau_FR )

Dose is standardised by the score's own 90th percentile `s90 = Q_0.9(|S_i|)`:

    dtau_FR = -log(1 - p90) / s90,     p90 in {0.02, 0.05, 0.10, 0.20}

so `p90 = 0.10` means "the particle at the 90th percentile of |S| has a 10 % event probability in
this pulse". This is a choice of FR reaction time — **not** a score clip and **not** an event cap;
neither is used.

Then **FR is switched off permanently** and plain ABF runs the full fibre cooldown `H_fib`
(2635 / 2325 steps). The cooldown is the hypothesis: a clone carries no new information at birth
(`q⁽¹⁾ = q⁽²⁾`); it must integrate independent Langevin noise along the fibre before it does.

## Stage 2 — primary endpoint: did FR create information?

Cellwise mean force at the end of cooldown, `f̂_{s,j}`, against the Stage-0A reference `f_j^ref`:

    R_s = sum_j a_j ( f̂_{s,j} - f_j^ref )^2

**Frozen pass condition, per dose:**

    E[R_FR] / E[R_ABF] <= 0.90    AND    paired 95 % CI upper bound < 1

i.e. FR must deliver at least a 10 % real reduction in estimator risk against the ≈54 % oracle
headroom. Paired by seed (arms share initial conditions and the noise bank), bootstrap 10 000
resamples, seed 20260829.

## Stage 3 — the causal chain, reported whatever the endpoint does

Reported for every dose, because a bare endpoint cannot distinguish the failure modes:

1. **Did the pulse move the population?** `KL(p_post‖pi*) / KL(p_pre‖pi*)` — below 1 or not.
2. **Did the new observations land where aimed?** `TV(r_future, pi*)` for FR vs ABF, where
   `r_future` counts only deposits made after the pulse.
3. **Did siblings decorrelate?** `rho_sibling(t) = Corr(f(q_t⁽¹⁾), f(q_t⁽²⁾))` over FR-born pairs,
   expected ≈1 at t = 0 and → 0 on the `tau_fib` scale. This is the mechanism made visible.

## Stage 4 — free-energy acceleration, only if Stage 2 passes

If and only if some dose passes Stage 2, continue **both** arms (no second pulse) to T = 100 and
report `S_eps = E[tau_eps^ABF] / E[tau_eps^FR]` with the inherited criterion: `S_eps2 >= 1.15`,
paired 95 % CI lower > 1, no additional censoring, `e_F^FR(T)/e_F^ABF(T) <= 1.05`.

## §5 — why ancestor ESS is demoted to a reported diagnostic

An ancestor label is permanent: two descendants of one FR birth remain co-labelled after ten
decorrelation times, when they are statistically independent. Under the repeated birth–death of
clean-v2 and q-r that was a good collapse diagnostic precisely because descendants were never
given time to decorrelate. Under **one pulse followed by a full fibre cooldown** it measures
history, not current effective sample size. It is reported alongside max-family-fraction and local
lineage diversity; the hard gate is the empirical risk `R_FR/R_ABF`, which measures the thing we
actually care about.

## Decision table

| Stage-3 KL ratio | Stage-2 risk ratio | conclusion |
|---|---|---|
| ≈ 1 (no movement) | — | **A**: the FR operator is too weak to realise this allocation |
| < 1 (moved) | ≈ 1 | **B**: representation ⇏ effective information — close sparse information-target FR-BD as an ABF accelerator |
| < 1 | ≤ 0.90, but `S_eps` < 1.15 | **C**: mechanism holds, one pulse insufficient → sparse receding-horizon adaptive Info-FR-ABF |
| < 1 | ≤ 0.90 and `S_eps` ≥ 1.15 | **D**: one FR information pulse → more effective conditional-force information → faster FEC |

## Prohibited

Reusing the short-horizon `pi*`; any operator other than `fr_v3.bd_standard`; score clipping or
event caps; changing the dose ladder, the gate, or the pass condition after seeing any result;
a second pulse anywhere; tuning against a stop.
