# Fibre-Horizon Information-Conversion Audit — Result

**Campaign**: `fibre_horizon_audit` · **Date**: 2026-08-29
**Preregistration**: `docs/FIBRE_HORIZON_AUDIT_PREREGISTRATION.md`, frozen before any run.
**Data**: `results/fibre_horizon/`.

> **Provenance caveat, stated because this project's standards require it.** The
> preregistration was written at 05:24:24 UTC, the config at 05:26:47, and the pilot ran at
> **05:27:39** (independently timestamped in `results/fibre_horizon/pilot/receipt.json`,
> against base commit `4d2f400`); this result was written at 05:33:34. So the protocol was
> fixed roughly three minutes before the run — but `docs/` was blanket-ignored by
> `.gitignore`, so the prereg was **not committed before the run** and both documents enter
> git in the same commit. The ordering evidence is therefore file mtimes plus the run
> receipt, **not commit order**, which is weaker than the previous audit managed (prereg
> frozen at `fba1b4b`, runner at `6484574`, verdict later). Nothing in the protocol was
> changed after the run; the weakness is in what the repository can prove, not in what
> happened.

## Verdict

    OUTCOME B — REPRESENTATION WITHOUT INFORMATION

The opportunity was there, the pulse moved the population, the genealogy stayed healthy — and
the estimator risk **got worse at every dose in both cells**. Under the decision table frozen
before the run, this closes sparse information-target FR-BD as an ABF accelerator.

## Stage 0 — the fibre horizon delivered the opportunity, and more than projected

Re-solving `pi* = argmin sum_j a_j V_j^fib / (C_j + M pi_j)` at `H_fib` with the fibre-consistent
difficulty gives **median `G_ideal` = 0.621 (K2) / 0.606 (K3)** against the 0.50 gate — above the
0.54 the Amendment-1 sidecar projected, because `V^fib` reshapes the difficulty rather than merely
rescaling it (`V_fib/V` ranges 4.6–334× across cells; spread widens 61× → 195×). The runner's
independent Stage-0 reproduced the offline solve to 14 significant figures.

**The long-horizon target is a different object, exactly as predicted.** Water-filling now spreads:
`max_j pi*_j` = **0.368 / 0.365**, down from 0.826 / 0.879 at the short horizon; TV to the old
target 0.69 / 0.72. Top-5 mass is unchanged at 0.895 — so the mass moved from one cell into about
five, not into all thirty-two. 28 of 32 cells sit on the `1/K` floor.

## Stage 2 — the hard gate, failed in the wrong direction

| cell | dose | R_ABF | R_FR | ratio | 95 % CI | gate |
|---|---|---:|---:|---:|:--|:--|
| K2 | 0.02 | 0.01094 | 0.01104 | 1.009 | [1.005, 1.013] | fail |
| K2 | 0.05 | 0.01094 | 0.01111 | 1.016 | [1.008, 1.022] | fail |
| K2 | 0.10 | 0.01094 | 0.01125 | 1.029 | [1.020, 1.038] | fail |
| K2 | 0.20 | 0.01094 | 0.01122 | 1.026 | [1.013, 1.035] | fail |
| K3 | 0.02 | 0.01095 | 0.01104 | 1.008 | [1.005, 1.012] | fail |
| K3 | 0.05 | 0.01095 | 0.01109 | 1.013 | [1.006, 1.020] | fail |
| K3 | 0.10 | 0.01095 | 0.01114 | 1.018 | [1.012, 1.025] | fail |
| K3 | 0.20 | 0.01095 | 0.01136 | 1.038 | [1.025, 1.052] | fail |

Not merely "no improvement": every CI excludes 1 **from above**, and the damage is **monotone in
dose**. Against a required ≤ 0.90, FR delivered 1.008–1.038.

## Stage 3 — the causal chain says where it broke

**The pulse did move the population.** `KL_post/KL_pre` = 0.987 → 0.921 monotonically in dose, and
the future observations did land closer to the target: `TV(r_future, pi*)` 0.8328 → 0.8198 (K2),
0.8355 → 0.8171 (K3). So this is not Outcome A; the operator works.

**The genealogy stayed healthy.** Ancestor ESS 0.81–0.97, max family fraction 0.008–0.012. The old
`ESS_anc ≥ 0.9` gate would have failed the p = 0.2 arm — and would have been measuring the wrong
thing, exactly as §5 of the preregistration argued. This failure is not a lineage collapse.

**The clones never became independent.** `rho_sibling` decays from 0.999 to **0.412 (K2) / 0.507
(K3) after a full fibre cooldown** — and that is not a broken horizon estimate, it is the horizon
working as defined. Fitting `rho = exp(-t/tau_rho)`:

| cell | arm | `tau_rho` (steps) | `H_fib` (steps) | ratio | `rho` at `H_fib` |
|---|---|---:|---:|---:|---:|
| K2 | p0.1 | 2768 | 2635 | 1.05 | 0.416 |
| K2 | p0.2 | 3415 | 2635 | 1.30 | 0.412 |
| K3 | p0.1 | 2775 | 2325 | 1.19 | 0.487 |
| K3 | p0.2 | 3344 | 2325 | 1.44 | 0.507 |

**`tau_rho ≈ H_fib`, so one cooldown horizon leaves `rho ≈ e⁻¹ ≈ 0.37` by definition.** The fibre
horizon was measured correctly and the design was self-consistent; the trouble is that "one
decorrelation time" is not "independent". Reaching `rho < 0.05` needs ≈ 3 `tau_fib`.

## Why the sign is negative, quantitatively

Birth–death holds `K` fixed: it kills one particle and clones another. Before, those are two
independent trajectories; after, two copies with correlation `rho`, worth `2/(1+rho)` effective
particles. At the measured `rho` that is a real loss, and it is larger than the placement gain:

| cell | dose | events | `rho(H)` | effective particles lost | placement gain in TV |
|---|---|---:|---:|---:|---:|
| K2 | 0.10 | 14.5 | 0.416 | 8.5 of 256 (3.3 %) | 1.3 % |
| K2 | 0.20 | 31.0 | 0.412 | 18.1 of 256 (7.1 %) | 1.6 % |
| K3 | 0.10 | 14.5 | 0.487 | 9.5 of 256 (3.7 %) | 1.2 % |
| K3 | 0.20 | 31.5 | 0.507 | 21.2 of 256 (8.3 %) | 2.2 % |

**The operator spends 3–8 % of the effective population to buy a 1–2 % improvement in placement.**
That ratio is why the risk rises, why it rises monotonically with dose, and why no dose on the
ladder could have passed. It is a property of cloning under a finite decorrelation time, not of
this particular target.

## What is now closed, and what is not

**Closed** — under the frozen decision table, with the strongest conditions the project could
construct for it (oracle target, 62 % finite-horizon headroom, correct fibre horizon, healthy
genealogy, four doses, two mirror cells):

    sparse information-target Fisher-Rao birth-death as an ABF accelerator

**Not closed, and untouched by this result:**

* The *allocation* question. `G_ideal` = 0.62 is real: a sampler that could place effort at
  `pi*` without paying the clone-correlation cost would have that headroom available. What is
  refuted is birth–death as the mechanism for placing it.
* The bias-side account from the q-r mechanism campaign, which remains the live explanation for
  where the endpoint error actually lives.

**The arithmetic that would have to change for this to be revisited** is explicit: a reallocation
operator must deliver placement gain exceeding `n_events · (2 − 2/(1+rho))` in effective-particle
terms. Cloning at `rho ≈ 0.4` cannot; an operator that resamples *and decorrelates* — or one that
does not duplicate states at all — is a different proposition and this result says nothing
against it.
