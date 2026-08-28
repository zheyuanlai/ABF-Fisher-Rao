# Information-Conversion Audit — Result

**Campaign**: `information-allocation-fr` · **Date**: 2026-08-28 (overnight, autonomous)
**Preregistration**: `docs/INFORMATION_CONVERSION_AUDIT_PREREGISTRATION.md`, frozen at
`fba1b4b`; Amendment 1 at `25c2279` (written after Stage 0A, before any Stage-0C/0D
computation). **Runner**: `6484574`. **Verdict data**: `results/information_conversion/`.

## Verdict

    NO_FINITE_HORIZON_ALLOCATION_OPPORTUNITY

The preregistered Stage-0D gate fired in **both** mirror cells before any Fisher–Rao
pulse ran: median `G_ideal` = **0.052** (K2, seed range 0.040–0.064) and **0.067**
(K3, range 0.055–0.089), against the frozen 0.10 threshold. **Zero FR pulses were
executed tonight.** Per the frozen decision table: the checkpoint/horizon combination
does not contain enough exploitable finite-horizon information heterogeneity, and FR
was not tuned against that stop.

## What the gate actually measured

At the t = 20 checkpoint (10,000 steps × 256 replicas = 2.56 M raw observations),
the oracle finite-horizon problem

    min_pi  sum_j a_j V_j / (C_j + M pi_j),   pi_j >= 1/K

was solved per seed with the validated reference difficulty `V_j` (Stage 0A rerun of
the q-r Stage-1B machinery: 200k-step A2 runs, seeds 5100–5103; V spread 154×/175×
across cells, mirror structure reproduced). Three structural facts:

1. **The asymptotic opportunity is large and misleading.** The Neyman comparator
   `1 − R_asym(opt)/R_asym(unif)` = **0.65** in both cells — the number that made
   information allocation look valuable in every earlier campaign. The *finite-horizon*
   opportunity at the frozen horizon is 0.05–0.07. The difference is the accumulated
   counts: the top risk cells already hold 56k–64k observations each, so one
   decorrelation-horizon of new effort (M = K·H ≈ 12.5k–14k observations *total*)
   barely moves any denominator, however well aimed.
2. **The optimal target is a near-delta.** The water-filling puts 70% (K2) / 88% (K3)
   of `pi*` into the single worst cell and floors almost everything else. Risk share
   of the top cell is 29%/35%, top-5 ≈ 90%. Had the frontier run, the one pulse would
   have been asked to move most of the population into one cell of width 0.19 — worth
   knowing before anyone proposes a repeated schedule toward such targets.
3. **Counts anti-correlate with difficulty only mildly.** Hard mid-domain cells hold
   ~0.7–0.8× the mean count at t = 20 — ABF has already made occupancy too flat for
   count deficit to be the story; the heterogeneity lives in `a_j V_j`, i.e. in the
   local asymptotic variance, exactly as the q-r campaign's decomposition says.

## Amendment 1: the stop is horizon-attributed (reported-only)

Stage 0A returned `tau_max(eval)` = 0.096/0.108 time units (H = 49/55 steps). That τ̂
is the A2 *cell-mean-series* decorrelation — validated for ranking Γ̂ = σ²τ̂, but
shortened by cell-population turnover. The Amendment-1 sidecar measured the fixed-x
**fibre** decorrelation at the cell centres with the frozen AR(1) estimator:

    tau_fib_max = 5.27 (K2) / 4.65 (K3)  →  H_fib = 2635 / 2325 steps

— ~50× the frozen horizon, and consistent with the one-off measurement recorded in
`information.py` (commit `a52e955`). Recomputing the same gate arithmetic on the same
saved checkpoints across horizons:

| median G_ideal | H = frozen | 250 | 600 | 1200 | ~H_fib | 6000 |
| --- | --- | --- | --- | --- | --- | --- |
| K2 | 0.052 | 0.179 | 0.310 | 0.427 | **0.540** | 0.617 |
| K3 | 0.067 | 0.197 | 0.330 | 0.443 | **0.539** | 0.628 |

So the verdict decomposes cleanly: **the checkpoint has large finite-horizon
opportunity (≈ 0.54) at the physically-correct fibre horizon; it has essentially none
within the frozen horizon, which Stage 0A revealed to be short by construction.**
Under the frozen rules and Amendment 1 these numbers licensed nothing tonight — no
pulse, no dose, no target was run or tuned.

## Audit trail

- 23 preregistered unit gates green before science, including: engine parity of the
  new runner against `simulation_torch.run_batch` (F̂, F̂′, trajectories to 1e-11 on
  K0 and K2 with identical injected noise); deposits-every-step units test (M = K·H);
  BD-event-deposits-zero test; exactly-one-pulse enforcement; FEC-blind dose
  selection; KKT/floor/scale-invariance tests of the water-filling solver.
- Deposition semantics inspected, not assumed: `abf.update_every = 10` is the
  grid-refresh cadence only; the accumulator deposits one observation per replica per
  step (`simulation_torch.py`, post-propagation path).
- Independent re-analysis: an SLSQP solve of the Stage-0C problem from the saved raw
  CSVs reproduces every seed's `G_ideal` to < 2e-4
  (`pilot/independent_check.json`).
- No NaNs, no failures; reference `tau` finite on all evaluation cells.
- Seeds: reference 5100–5103 (evaluation-only), pilot 8000–8007. Confirmation seeds
  8100–8131 were never consumed. GPU 2 only, conda `abffr`, torch 2.12.0+cu130.
- Figures: `results/information_conversion/figures/` — target/counts/π* panels,
  asymptotic-vs-finite opportunity scatter, G_ideal(H) sensitivity curve.

## What tonight settles, and what it does not

Settled: the missing-link question cannot be answered *at this checkpoint under the
frozen horizon* — not because FR fails, but because no allocation policy of any kind
(FR, bias, or oracle) has ≥10% of headroom there. The earlier asymptotic risk ratios
(e.g. `information_risk_ratio = 0.346` in the oracle-short-burst campaign) overstate
finite-horizon headroom by an order of magnitude at established checkpoints; any
future allocation claim should quote `G_ideal` at its actual horizon.

Not settled: the conversion question itself. The sidecar says the decisive version of
this experiment exists at `H = H_fib ≈ 2.3–2.6k` steps, where the ideal-allocation
ceiling is ≈ 54%. That experiment was deliberately **not** run tonight — rerunning
the gate at a longer horizon after this stop is exactly the move the preregistration
forbids. It needs a fresh preregistration (one line changes: the τ source for H
becomes the fixed-x fibre estimator), and it should anticipate the genealogy question
head-on: `pi*` concentrates 70–88% of the population into one cell, so the one-pulse
frontier at H_fib is simultaneously the cleanest test of
`FR_STRENGTH_GENEALOGY_CONFLICT` the project has ever had.
