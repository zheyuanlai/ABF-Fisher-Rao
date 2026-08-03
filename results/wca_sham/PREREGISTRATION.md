# Preregistration — matched-sham control at the accepted WCA-positive cell

**Committed before any of these runs start.** Nothing below may be changed afterwards.

## 1. What is being tested, and why now

The entropic gateway now has a sham-controlled positive result: at a preregistered anchor,
practical mFR improves the integrated free-energy error by $-12.1\%$ on 31/32 fresh seeds
while its matched sham is equivalent to ABF, and the two differ directly by $-11.8\%$.

The **molecular** positive does not have that control. The WCA dimer's mFR gain has only ever
been compared against ABF and against other *score-driven* arms (`fr_uniform`, `fr_oracle`),
every one of which shares the same replacement direction and differs only in how the target
is built. So the proposition "the molecular gain is Fisher–Rao steering rather than turnover
of that intensity" has never been tested at all.

Until it is, the study cannot claim the directional mechanism transfers to a many-body
molecular system. This run decides that.

## 2. Frozen configuration — nothing is retuned

Taken from the accepted phase-diagram production config, cell `b1_h2`. The runner reads the
YAML back and **asserts** every FR knob, so an edit to the config cannot silently move this
experiment.

| quantity | value |
|---|---|
| cell | `b1_h2_w2_n10_a1.5` — $\beta=1$, $h=2$, $w=2$, `n_dim`=10 ($M=100$ particles), $a=1.5$ |
| budget | 120 000 steps, $N = 1024$ replicas, `save_every` 2500 |
| FR knobs | `fr_rate` 0.10, `target_ema_rate` 0.005, `max_event_fraction` 0.02, `fr_every` 5, `fr_start_steps` 20 000, `score_clip` 2.0 |
| reference | cached TI reference for this cell, evaluation-only |
| diagnostics | `track_crossings = True` |
| seeds | **400–415 (16 fresh)**, verified unused anywhere in `results/` |

Reference effect at this cell from the accepted runs (4 seeds): ABF $L_2(F) = 0.0930$,
`fr_estimated` $0.0480$ ($-48.4\%$). A 10-seed replicate at the same cell gives ABF $0.0878$
and `fr_estimated` $0.0431$ ($-51.0\%$).

## 3. Arms

| arm | description |
|---|---|
| `abf` | baseline |
| `fr_estimated` | practical mFR — **the primary claim** |
| `sham_practical` | matched sham for `fr_estimated` |
| `fr_oracle` | oracle mFR — non-deployable diagnostic |
| `sham_oracle` | matched sham for `fr_oracle` |

The WCA sampler runs one method per process, so a sham replays the per-FR-opportunity
replacement count sequence its partner realised **on the same seed**, choosing which replicas
die and which are copied uniformly at random. The runner asserts that each sham's total
replacement count equals its partner's.

**Known limitation, stated in advance.** WCA draws Langevin noise and all birth–death
randomness from one global stream, so arms diverge in their noise realisation after the first
firing; matched-seed here means *matched initial conditions*, not matched noise. This applies
equally to every arm including the shams, so no arm is advantaged, but the paired differences
carry more variance than the gateway's. It is a property of the accepted sampler and is not
being changed for this run.

## 4. Endpoints

**Primary.** Paired relative change in integrated free-energy error
$I_F = \int_0^T \lVert \widehat F_t - F_{\mathrm{ref}} \rVert_{L^2}\,dt$ against `abf` on the
same seed. Negative is better. Median over seeds with a paired bootstrap CI (10 000
resamples, seed `20260803`).

**Attribution.** The **direct** paired contrast of each FR arm against **its own sham** on the
same seed — the statistic that holds the event schedule and count fixed by construction, so
only the selection direction differs.

**Secondary.** Final $L_2(F)$, $L_2(F')$, round trips, barrier crossings, occupancy of the
compact/transition/stretched regions, ancestor ESS, max ancestor fraction, replacement counts.

## 5. Decision rule — frozen

1. **`fr_estimated` beats ABF**: median $\le -10\%$, 95 % CI upper $< -5\%$, $\ge 12/16$ seeds.
2. **Attribution**: the direct `fr_estimated` vs `sham_practical` contrast has a 95 % CI that
   **excludes zero** and a median $\le -5\%$.
3. **Sham equivalence** (secondary, reported either way): TOST at $\alpha = 0.05$ against a
   $\pm 5\%$ margin, i.e. the sham's 90 % CI against ABF lies inside it.
4. Health: `min ancestor ESS / N ≥ 0.10` and `max ancestor fraction ≤ 0.05`. *The ESS floor is
   looser than the gateway's 0.30 because the accepted WCA positive already runs at ancestor
   ESS ≈ 166/1024 ≈ 0.16; applying the gateway's floor would fail the configuration this
   experiment exists to test, which would be changing the question.*

**Predeclared readings:**

| outcome | reading |
|---|---|
| 1 and 2 hold | the directional mechanism **transfers** to a many-body molecular system |
| 1 holds, 2 fails | the molecular gain may be generic turnover; the directional claim does **not** transfer, and the report must say so |
| 1 fails | the accepted WCA positive did not replicate on fresh seeds under this protocol; report as such |
| shams differ from ABF but mFR still beats them directly | report both: turnover contributes, direction contributes more, with the split quantified |

## 6. What this run does not do

No hyperparameter search, no new cells, no rate ladder, no change to the sampler's RNG
structure, no change to the accepted config. If the result is negative it is reported as
negative; the `possibly` in the study's regime table stays until criterion 2 is met.
