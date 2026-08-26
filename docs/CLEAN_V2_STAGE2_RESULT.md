# Clean-v2 Stage 2: the pilot is negative, 0 of 9

- Protocol: `docs/CLEAN_V2_PREREGISTRATION.md` (frozen 2026-08-26, amendments 1-11)
- Executed: 2026-08-26, GPU 2, 80 runs in 255 s, 0 failed / 0 NaN
- Thresholds: `results/clean_v2/thresholds.json`, frozen from Stage 1 **before**
  any FR run existed
- Data: `results/clean_v2/stage2_pilot/pilot/`, analysis in `acceleration.csv`,
  figures in `figures/`

## Verdict

**Case C.** No schedule in the frozen 3 x 3 grid met the pre-declared
acceleration screen. `select_clean_v2_schedule.py` refused to write the Stage-3
and Stage-4 configs, so no confirmation run exists to start.

Under the frozen protocol the finding is stated as: *physical-target intermittent
Fisher--Rao birth--death does not accelerate this benchmark at this dose and
schedule.* No interpolation rescue, no new target, no grid extension.

## The grid

Primary scope R12, `eps_{F,1} = 0.096222` (0.4T), `eps_{F,2} = 0.060462` (0.6T).
`!` marks a threshold where the FR arm was censored more than the baseline.

| `gamma` | `L_FR` | `S^(T)_{F,1}` [95% CI] | `S^(T)_{F,2}` [95% CI] | hit `eps_2` ABF/FR | repl/pulse | `ESS_anc/K` (final) | `e_F(T)` ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.002 | 100 | 1.021 [0.99, 1.06] | **0.853**! [0.75, 0.96] | 0.88 / 0.75 | 0.007 | 0.179 | 1.30x |
| 0.002 | 500 | 0.980 [0.82, 1.12] | **0.821**! [0.65, 1.00] | 0.88 / 0.62 | 0.031 | 0.205 | 1.64x |
| 0.002 | 1000 | 0.974 [0.93, 1.02] | **0.828**! [0.71, 0.96] | 0.88 / 0.75 | 0.063 | 0.202 | 1.70x |
| 0.01 | 100 | 1.034! [0.87, 1.24] | **0.959**! [0.74, 1.27] | 0.88 / 0.62 | 0.021 | 0.057 | 3.04x |
| 0.01 | 500 | 0.865! [0.64, 1.20] | **0.777**! [0.63, 0.98] | 0.88 / 0.38 | 0.100 | 0.057 | 2.83x |
| 0.01 | 1000 | 0.717! [0.54, 0.97] | **0.609**! [0.50, 0.74] | 0.88 / 0.00 | 0.188 | 0.064 | 3.55x |
| 0.05 | 100 | 1.119 [0.99, 1.23] | **0.900**! [0.74, 1.11] | 0.88 / 0.75 | 0.053 | 0.025 | 1.54x |
| 0.05 | 500 | 0.702! [0.54, 0.97] | **0.813**! [0.68, 0.99] | 0.88 / 0.62 | 0.222 | 0.032 | 2.32x |
| 0.05 | 1000 | 0.625! [0.46, 0.88] | **0.609**! [0.50, 0.74] | 0.88 / 0.00 | 0.352 | 0.024 | 4.22x |

Eight of nine cells are **below 1** at the stringent threshold; the ninth is
0.959. No cell reaches the pre-declared 1.15 at either threshold.

## The censoring direction strengthens the negative

14 of 18 free-energy thresholds carry `!`: the FR arm failed to converge more
often than plain ABF. Restriction replaces those censored times by `T`, the
smallest value they could have had, so **`S^(T)` is inflated exactly there**.
The reported speedups are therefore *upper bounds* on the truth, and they are
already below 1.

The refusal rule that exists to block a false positive does not apply
symmetrically here, and it is worth being explicit about why: it protects
against censoring that flatters the arm. Here the same bias flatters the arm and
the arm still loses, so the negative is safe. The two cells at
`hit = 0.88 / 0.00` are the sharpest form of it -- **not one of eight FR seeds
ever reached the stringent threshold**, against seven of eight for plain ABF.

## What actually happened

The convergence curves (`figures/fig1_convergence_F.png`) show one shape in
every cell:

1. **A real but small transient gain.** Up to `t ~ 40` several FR cells sit at
   or below plain ABF; the best, `gamma = 0.01, L = 100`, is 18% lower on R12
   at `t = 40` (0.0761 against 0.0932) and still ahead at `t = 60`.
2. **Then it turns over, while FR is still on.** Every FR curve flattens near
   `t ~ 45-60` and rises, while plain ABF keeps descending.
3. **It does not recover after `t_off = 80`.** Final error is 1.30x to 4.22x
   plain ABF and still growing in the damaged cells, which trips the protocol's
   pathology criterion rather than the benign "curves converge again" picture.

This is the same two-phase signature the v2 audit recorded ("gain = transient
edge evacuation that fully relaxes back"), now reproduced with an **uncapped**
operator, **no** score clip, **no** EMA, and the correct time-to-accuracy
endpoint. Removing the three distortions did not change the sign.

## Mechanism diagnostics

Every pre-declared engineering expectation held at production scale:

- score span 66-78 nats per pulse (predicted `~80` from `beta` x the range of
  `F_ref`), `|S|_max ~ 60-64` -- so `score_clip = 5` had been discarding
  92% of the signal;
- `logp_floored_fraction = 0` in every pulse of every run: Gate D holds at
  production scale, not just at gate scale;
- `p_event_max` runs 0.024 at the weakest cell to 0.998 at the strongest, so
  the widened `gamma` grid did span unsaturated to saturated as intended.

**Ancestral ESS collapses in every cell**, to 1.5-21% of `K = 256`. The gentlest
cell replaces 0.7% of the population per pulse and still ends at
`ESS_anc/K = 0.179` after 300 pulses: it is cumulative turnover, not per-pulse
dose, that spends the genealogy. Final damage is not a monotone function of
`ESS_anc` alone (`gamma = 0.05, L = 100` has lower ESS than
`gamma = 0.01, L = 100` and less damage), so ESS is not by itself the whole
explanation, and with 8 seeds this is not the place to build one.

## A pre-declared prediction that was wrong

Section 9 of the protocol predicted that damage would concentrate **outside**
R12 -- the physical target evacuating the high-`F` wall strips -- so that R12
might improve while `full`/`legacy` degraded, giving a scope-limited
accelerator. **That is falsified.** Both scopes degrade together, and if
anything R12 is hit slightly harder (3.26x against 2.93x at
`gamma = 0.01, L = 100`). The damage is not a scope artifact and cannot be
reported as one.

## What the protocol permits from here

The frozen decision tree makes Case C terminal for this benchmark at this
dose/schedule. Two things are **not** licensed: extending the `gamma` grid
(the extension rule was deleted in amendment 10, precisely so a negative could
not be searched away), and inventing a new target.

One question the pilot cannot answer is whether the failure is the *estimated*
target or the *mechanism*, because the oracle arm lives in Stage 3 and Stage 3
was never authorised. Answering it would be a new preregistered diagnostic, not
a continuation of this one. Note that the v3 campaign already ran the equivalent
comparison and found that an oracle target does **not** fix finite-K genealogy
collapse.
