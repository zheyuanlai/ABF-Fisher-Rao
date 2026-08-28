# Oracle conditional-force-variance information target

## Material Passport

- Artifact type: code-experiment protocol
- Status: implementation frozen before dose calibration
- Date frozen: 2026-08-28
- Branch: `unflattened-target`
- Version: `IT-FR-ABF-v1`
- Reallocation operator: only `fr_v3.bd_standard`

## Research question

After ordinary ABF burn-in, do three standard Fisher--Rao birth--death pulses
toward a frozen conditional-force-variance information target make ABF reach
the two existing free-energy accuracy thresholds faster, without endpoint
repayment or genealogy collapse?

This v1 campaign tests the target mechanism, not the full finite-horizon
allocation theory. It does not estimate an IAT, planning horizon, trigger,
cooldown, or genealogy-adjusted target.

## Target

The reaction-coordinate domain is divided into 32 equal allocation cells. The
free-energy reporting interval is fixed independently of `F_ref` as
`[-2.5, 2.5]`. Let `H` be cumulative trapezoidal integration from mean force to
free energy, `P` remove the quadrature-weighted additive constant on that
interval, `W` be its trapezoidal weights, and `B` expand a cell-wise force error
to the profile grid. The integration leverage is

```text
a_j = diag((P H B)' W (P H B))_j.
```

It uses no `F_ref`-defined R12 or thermal-scope mask. Stage A obtains the exact
conditional local-force variance by y-quadrature,

```text
sigma_j^2 = cell-average of E[(dV/dx)^2 | x] - E[dV/dx | x]^2.
```

The frozen target is the constrained optimizer

```text
q_j = max(1/K, c sqrt(a_j sigma_j^2)),   sum_j q_j = 1.
```

The `1/K` lower bound is a coverage constraint: at least one expected particle
per allocation cell. The code records the target masses and verifies that the
predicted risk `sum_j a_j sigma_j^2/q_j` is no larger than uniform allocation.

## Pulse operator and observation order

At a pulse, current reaction-coordinate density `p_hat` is estimated by the
existing KDE and the centered score is

```text
S_i = log p_hat(x_i) - log q(x_i)
      - mean_k[log p_hat(x_k) - log q(x_k)].
```

The only reallocation is `fr_v3.bd_standard`: positive score dies, negative
score reproduces, partners are uniform, population is fixed, and event
probability is `1-exp(-|S_i| dtau)`. There is no score clipping, event cap,
finite-time resampling, turnover, jitter, or alternative bias. Physical
propagation deposits the ordinary ABF observation before each pulse. The BD
event itself deposits no observation.

## Sequential stages

### Stage 0: mechanism-only dose calibration

- Seeds: 6000--6007.
- One pulse at step 10,000 (`t=20`).
- Dose grid: gamma in `{0.001, 0.002, 0.004, 0.008, 0.016, 0.032}`.
- Selection reads only event fraction, KL movement, ancestor ESS, numerical
  floors, and the target risk check. It cannot read free-energy outcomes.
- Eligibility is inherited from the earlier short-burst calibration: median
  event fraction in `[0.01,0.05]`, median KL ratio at most `0.99`, at least 5/8
  KL decreases, median one-pulse ESS/K at least `0.95`, and no density floor.
- The smallest eligible gamma is selected. No eligible dose stops the study.

### Stage A: oracle information target

Only an eligible Stage-0 receipt authorizes Stage A.

- Matched seeds: 7000--7031, 32 pairs.
- Baseline: plain ABF.
- Arm: oracle information target with standard BD.
- `K=256`, `beta=4`, `dt=0.002`, `T=100`.
- Exact pulse steps: 10,000, 10,500, 11,000 (`t=20,21,22`).
- Target is identical at all three pulses.
- FR is permanently off after the third pulse; the remainder is pure ABF.

## Decision rule

The primary endpoint remains the two frozen R12 free-energy hitting-time
thresholds from clean-v2. Both require speedup at least 1.15 and paired
bootstrap 95% lower bound above 1.0, with no intervention-inflated censoring.
The endpoint error ratio must be at most 1.05 and final ancestor ESS/K at least
0.70. Every seed must have exactly the three frozen pulses; median KL ratio must
be below one, at least half of pulse rows must reduce KL, density floors must
never bind, and the information-risk ratio must not exceed one.

- PASS: all identification, safety, acceleration and endpoint gates pass.
- FAIL: the experiment is identified and safe, but acceleration or endpoint
  fails.
- INCONCLUSIVE: mechanism, schedule, genealogy, target-risk, or censoring gate
  fails.

An identified oracle FAIL stops this target on the current benchmark. It does
not authorize adding IAT, triggers, or cooldowns. The next scientifically clean
step would be a deliberately variance-dominated benchmark. Oracle PASS permits
a separately frozen estimated-variance target using the already implemented
`C,S,Q` accumulators.
