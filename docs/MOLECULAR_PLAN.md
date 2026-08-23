# Molecular campaign: does RC-WFR + constrained sampling survive the move to molecules?

Phase 0 (complete, frozen at commit 6398e3f) established, on an exactly-solvable
nonlinear-coordinate toy family:

| # | claim | toy evidence |
|---|---|---|
| P1 | a naive lift creates conditional-lift bias | e_F 0.6445 vs floor 0.0046 |
| P2 | minimum-norm horizontal transport does not fix it | 0.6157 (-4.5%) |
| P3 | conditionally correct transport does | 0.0088 (-98.6%) |
| P4 | the tradeoff `fast WFR <-> bad conditionals` is an artifact of a wrong lift | e_F vs kappa reverses sign |
| P5 | lift error in SLOW fiber modes dominates | 38.6 -> 2.3 floors as the spectator speeds up 256x |
| P6 | therefore: transport the slow fiber modes, relax the fast ones | motivates secondary-CV RC-WFR |

This campaign asks whether P1-P6 survive on real molecules, using
**constrained Brownian dynamics on Sigma(z) with the Chapter-3 geometry**
(mass-metric Gram matrix, Fixman correction, den Otter-Briels mean force).

## Systems

| tag | system | n coords | z | y (slow fiber mode) |
|---|---|---|---|---|
| `BUT` | united-atom butane (TraPPE) | 12 | central torsion phi | - |
| `PEN` | united-atom pentane (TraPPE) | 15 | phi1 | phi2 |
| `ALA` | alanine dipeptide, vacuum amber14 | 66 | phi | psi |

## Gates

* **Gate I (BUT)** constrained TI must reproduce the unbiased-MD reference
  `F(phi) = -beta^-1 log p(phi)` to <= 2 estimator floors; Fixman reweighting
  ESS > 0.9; constraint residual < 1e-9.  Failure = engine bug, not a result.
* **Gate II (PEN)** the ORACLE-y lift must materially beat the naive lift.
  If it does not, phi2 is not the damaging mode and no amount of learning helps.
* **Gate III (PEN)** learned-y must recover a substantial part of the oracle gain
  at matched force evaluations, on fresh seeds.

## Arms

| arm | lift | purpose |
|---|---|---|
| `ti_cold` / `ti_warm` | - | stratified constrained TI |
| `reti_cold` | - | + window exchange |
| `abf` | - | adaptive biasing force, multiple walkers |
| `wfr_naive` | SHAKE along M^-1 grad xi (= min-norm) | negative control |
| `wfr_yoracle` | CDF map on y from the reference conditional | mechanism ceiling |
| `wfr_ylearned` | CDF map on y from the run's own samples | the practical method |
| `wfr_nofr`, `wfr_now` | ablations | mechanism decomposition |

Cost currency: force evaluations, `N * n_steps`, identical for every arm;
replica-exchange energy evaluations charged explicitly.


---

## Outcomes (appended after the fact)

| # | preregistered claim | molecular outcome |
|---|---|---|
| P1 | a naive lift creates conditional-lift bias | **confirmed.** `D_cond` 0.194 naive vs 0.005 corrected; `e_F` -53.6% (pentane), -83.4% (alanine) |
| P2 | minimum-norm horizontal transport does not fix it | **confirmed and strengthened.** It is 2.1x WORSE than a plain rotation, at the same conditional error -- the extra damage is in the fast modes, and it grows 12.6x with transport rate |
| P3 | conditionally correct transport does fix it | **confirmed, with a condition the toy phase could not see.** Transport alone is not enough: an ORACLE uncorrected refresh is +361% on alanine. The move has to be Metropolis-corrected |
| P4 | the fast-WFR / bad-conditionals tradeoff is an artifact of a wrong lift | **confirmed in the strongest form.** For the corrected arm `e_F` and `D_cond` are flat over a 64-fold `kappa_W` sweep |
| P5 | lift error in SLOW fiber modes dominates | **refined.** Coupling, not timescale: hexane's `phi3` is 1.6x SLOWER than `phi2` and promoting it buys nothing, while `phi2` is worth -44.3%. The `S_k tau_k^2` diagnostic ranks them correctly in advance |
| P6 | transport the slow modes, relax the fast ones | **confirmed.** Promoting one coupled mode captures the whole benefit; promoting both is no better |

### Gates

| gate | requirement | outcome |
|---|---|---|
| I (butane) | constrained TI within 2 estimator floors of unbiased MD; Fixman ESS > 0.9; SHAKE residual < 1e-9 | **passed.** 0.0531 against a 0.0488 floor; ESS 0.980; residual 2.7e-15 |
| II (pentane) | the oracle-y lift must materially beat the naive lift | **passed.** -52% to -54% |
| III (pentane) | learned-y must recover a substantial part of the oracle gain on fresh seeds | **passed, once corrected.** +1.2% [-2.6, +15.3] from the oracle, i.e. indistinguishable. Uncorrected it fails by +53% (map) and +507% (refresh) |

### What was NOT done

* no solvated system (the plan's NaCl / hydration-coordinate stage);
* no replica-exchange TI baseline -- window exchange under a hard constraint is
  itself a lift problem, and building it fairly was out of scope for this pass;
* the two-CV complete-coordinate control (`z = (phi, psi)`) was implemented in
  the engine but not run;
* `wfr_qref` was intended as a conditional-oracle ceiling and does not work as
  one (see `MOLECULAR_RESULTS.md` section 11); `ti_warm` serves instead.
