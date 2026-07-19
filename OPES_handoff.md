# OPES Closure — HANDOFF

Additive closure layer over the OPES pilot. Nothing in `results/opes_wca/`,
`results/opes_toys/`, or the pilot report numbers was deleted; the report was edited
in place only to (a) fix one factual error, (b) mark the pilot preliminary, and (c)
add the closure-status note (`sec:opes_closure_note`). All new code is
command-line runnable, idempotent, and writes per-run npz + a manifest.

Environment: `conda activate abffr`. **GPU policy: at most two GPUs, never more**
(same as the WCA follow-up). Check `nvidia-smi` and pass free GPUs explicitly.

---

## 1. Verification gates (all green, no production compute needed)

- **Gate 1 — OPES core correctness (14/14).** `scripts/audit_opes.py` →
  `results/opes_closure/audit/`. Exact-math tests pin the bias formula (L2=2.5e-7),
  applied force = -dA/dz (0.0), weight-scale invariance (1e-7), and the
  gamma = beta*BARRIER PLUMED mapping for all three systems. **Rules out an
  OPES-core bug as the cause of the pilot underperformance.** Also confirmed the
  eps-floor implements the PLUMED BARRIER cap (residual 0.069->0.000 as BARRIER
  deepens) and that `gamma_from_barrier` couples gamma and the eps-cap — motivating
  the fixed-gamma control method.
- **Gate 2 — multi-walker normalization.** Exact 1-batch == 16-batch (1.5e-7) plus
  a fair equal-deposit dynamics test.
- **Gate 3 — metric pipeline (13/13).** `src/closure_metrics.py` +
  `scripts/validate_metrics.py`. 61-column common schema validated against analytic
  KL/TV/entropy/tau and real pilot npz. **Native and common estimators are separate
  labelled columns** (`l2_f_native` vs `l2_f_common`) — fixes the pilot conflation.

## 2. Closure infrastructure (built + validated end-to-end on real data)

- `configs/opes_closure/wca_closure.yaml` — WCA closure. Adds the swept **sigma**
  axis (pilot fixed 0.05), a relaxed `abf_force_clip` (200 vs pilot 40), a
  fixed-gamma control method (`opes_fixedg`), and the flat ablation. `tune_r1` =
  4 barriers x 3 paces x 4 sigmas x 3 cells x 2 seeds = 288 runs.
- `configs/opes_closure/toys_closure.yaml` — meta + eb closure; sweeps
  barrier x pace x sigma (pilots swept only barrier). 96 round-1 runs per toy.
- `scripts/aggregate_closure.py` — npz -> `per_run.csv` (61-col) + `per_config.csv`
  (seed mean/std/sem/CI95 per hyperparameter cell). Validated on all 168 pilot npz.
- `scripts/tune_closure.py` — successive-halving orchestrator. `rank` prints/writes
  per-cell rankings; `emit` writes the next round's stage at higher budget carrying
  **explicit per-cell survivor tuples** (`stages.<round>.survivors`), so only the
  kept fraction advances — true SHA, not a re-expanded union grid. `src/opes_jobs.py`
  `expand()` honors `survivors` per cell (falls back to the global grid product when
  absent). Verified: tune_r1 288 runs -> tune_r2 144 runs at keep_frac=0.5.
- `scripts/run_closure_toy.py` — config-driven meta/eb runner; emits npz in the SAME
  schema so aggregate/tune drive all three systems identically. Round-trip verified.
- The WCA runner `scripts/run_opes.py` supports N-way sharding (its docstring says
  {4,5,7} but the code is general); **honour the 2-GPU policy at launch**.

## 3. Launch plan (respects the 2-GPU policy)

Check `nvidia-smi`, pick two free GPUs (call them $A $B). WCA round-1 (288 runs) is
~6.4 GPU-h => ~3.2 h on two GPUs. Per stage:

```bash
conda activate abffr && cd ~/ABF-Fisher-Rao
# --- WCA tuning round 1 (2 GPUs) ---
CUDA_VISIBLE_DEVICES=$A python scripts/run_opes.py --config configs/opes_closure/wca_closure.yaml \
      --stage tune_r1 --shard 0 --num-shards 2 > results/opes_closure/wca/logs/r1_s0.log 2>&1 &
CUDA_VISIBLE_DEVICES=$B python scripts/run_opes.py --config configs/opes_closure/wca_closure.yaml \
      --stage tune_r1 --shard 1 --num-shards 2 > results/opes_closure/wca/logs/r1_s1.log 2>&1 &
wait
# --- aggregate + rank + emit round 2 (@80k) then round 3 (@160k,+seeds) ---
python scripts/aggregate_closure.py --raw results/opes_closure/wca/raw --out results/opes_closure/wca/metrics --stage tune_r1
python scripts/tune_closure.py rank  --config configs/opes_closure/wca_closure.yaml --round tune_r1 --metrics results/opes_closure/wca/metrics
python scripts/tune_closure.py emit  --config configs/opes_closure/wca_closure.yaml --from-round tune_r1 --to-round tune_r2 --keep-frac 0.5 --n-steps 80000 --metrics results/opes_closure/wca/metrics
# ...run tune_r2 (2 GPUs), aggregate, emit tune_r3 @160k, run, then freeze winners into `representative`.
# Toys mirror this with scripts/run_closure_toy.py --toy {meta,eb}.
```

Successive-halving ladder: r1 48 cfg/cell @40k -> r2 top-50% @80k -> r3 top-25%
@160k, then freeze the per-cell winner (from `ranking_tune_r3.json`) into the
`representative` stage (opes / opes_fixedg / opes_flat x 6 cells x 10 seeds) and run
production. Finally the 32-seed validation on the anchor cells for CI width.

## 4. Reconciliation of pilot overclaims (done)

1. **Estimator conflation** — the pilot reported only mean-force as "OPES" and
   called the native reweight estimate "strictly worse". FALSE: reweight is
   comparable in the starved anchor and better in b1_h4 (0.162 vs 0.194; flat 0.125
   vs 0.197). Report edited (`06_case_wca.tex`) to report both; the closure metric
   schema keeps them as separate labelled columns.
2. **sigma untuned** — pilot fixed sigma (WCA 0.05, meta 0.12, eb 0.05); the "not a
   tuning artefact" claim was weakened to "along the well-tempering axis only", with
   an explicit pointer to the sigma-sweeping closure.
3. **Pilot vs final** — added `sec:opes_closure_note`; the single-sigma table is now
   flagged preliminary (a lower bound on OPES accuracy), qualitative ordering only.
4. **GPU policy** — closure launches are 2-GPU (section 3), matching the project
   policy; the pilot's implicit larger-fan-out is superseded.

Note: on-disk pilot data already has opes_flat for all 6/6 representative cells, so
the earlier "3-of-6" gap is resolved in the data; the report table is regenerated
from `opes_vs_baselines.csv` by `scripts/make_opes_report_assets.py`.

## 5. Live status / findings

- **WCA tune_r1 (@40k, 2 seeds, 288 runs): COMPLETE, 0 fail / 0 nan.** Aggregated
  -> 144 configs; ranked per cell (`results/opes_closure/wca/metrics/ranking_tune_r1.json`).
  Headline: **the pilot's fixed (barrier=4, pace=500, sigma=0.05) is not the per-cell
  optimum in any of the three cells** — direct support for marking the pilot numbers
  preliminary. Per-cell r1 winners (by l2_f_common, note wide CIs at 40k/2-seed):
  - b1_h2 (starved): barrier=3, pace=1000, sigma=0.05 (L2~0.188)
  - b2_h6 (intermediate): barrier=6, pace=250, sigma=**0.02** (L2~0.170; pilot used 0.05)
  - b4_h1 (easy): barrier=3, pace=250, sigma=0.035 (L2~0.085)
  sigma clearly matters (e.g. b2_h6 spans 0.17->0.59 across the sigma axis), which is
  exactly the axis the pilot never swept.
- **WCA tune_r2 (@80k, 144 runs): COMPLETE**, 0 fail/0 nan. CIs tightened to
  ±0.001-0.015. Emitted tune_r3.
- **WCA tune_r3 (@160k, 72 runs): COMPLETE**, 0 fail/0 nan. Converged per-cell winners
  (frozen into production):
  - b1_h2 (starved): barrier=4, pace=250, sigma=0.035 (L2=0.1147)
  - b2_h6 (intermediate): barrier=6, pace=250, sigma=0.02 (L2=0.0704)
  - b4_h1 (easy): barrier=3, pace=1000, sigma=0.05 (L2=0.0111)
  All three differ from the pilot's fixed (4,500,0.05) -> the pilot per-cell numbers
  are not the tuned optima. sigma matters and was never swept in the pilot.
- **SCOPE DECISION (important):** only 3 anchor cells were tuned (b1_h2,b2_h6,b4_h1).
  The 6-cell `representative` stage's other 3 cells (1_4,2_4,4_2) were NOT tuned, so
  production is deliberately restricted to the 3 tuned anchors. Applying anchor configs
  to untuned cells would reintroduce the exact untuned-hyperparameter flaw the closure
  exists to fix. To cover all 6, run a second SHA ladder on the remaining cells first.
- `tune_closure.py freeze` (new subcommand) writes the production stage from top-1/cell
  of a tuning round, restricted to tuned cells, all methods x 10 seeds. Reproducible;
  wrote `results/opes_closure/wca/metrics/freeze_tune_r3_to_production.json`.
- **WCA production (@120k, 3 cells x 3 methods x 10 seeds = 90 runs): RUNNING** on GPUs
  0,1. Methods share frozen (barrier,pace,sigma) per cell; differ in gamma treatment
  (opes=adaptive, opes_fixedg=10, opes_flat=inf/no well-tempering).
- Next: aggregate production -> method comparison table per cell -> then toys (meta+eb),
  tuning-only single-method sweep (see corrected section 6, NOT a WCA-style ladder).

## 6. Toy phase (meta + eb) — CORRECTED SCOPE (tuning-only, single method)

IMPORTANT correction to the "toys mirror WCA" language in sections 3/5. The toys are
NOT a 3-method ladder. Verified from code + config:
- `run_closure_toy.py` runs a SINGLE method (hardcoded name="opes", gamma/gfb from
  config). There is no opes_fixedg/opes_flat loop, and `toys_closure.yaml` has no
  `methods:` block at all (WCA config does). So the well-tempering-vs-flat comparison
  is a WCA-only result; toys are not built to reproduce it.
- Toys are single-physics (ONE cell each) and the config has only
  `[tune_r1, representative]`. `representative` is a single frozen (b,p,s) as singleton
  axis lists — the toy runner already consumes that format directly.

Therefore the toy closure job is just: does sweeping sigma (pilots swept only barrier)
change the tuned optimum vs the pilot's fixed sigma (meta 0.12, eb 0.05)? Flow:
  1. tune_r1 @40k, full 4x3x4 grid x 2 seeds = 96 runs/toy  [RUNNING on GPUs 0,1].
  2. aggregate -> rank the single cell by l2_f (NOT l2_f_common; toys emit l2_f only).
     Simplest to rank inline (sort per_config by l2_f_mean); tune_closure's WCA-shaped
     rank/emit/freeze are NOT needed and were NOT modified for toys.
  3. write the top (b,p,s) into `representative` (singleton axis lists), run its 5 seeds
     at the production budget (meta 100k, eb 40k). eb's tune budget already == prod, so
     its r1 winner is directly valid; meta tunes at 40k but reports at 100k, so confirm
     the r1 winner (and ideally runner-up) at 100k before freezing.
  4. compare tuned (b,p,s) vs pilot fixed-sigma to state whether sigma was a tuning
     artefact for the toys.

The two "fixes" a PRIOR version of this section listed (toy-runner survivors +
tune_closure --cell-keys/--primary flags) are NOT required — they were premised on the
mistaken "toys ladder like WCA" assumption. No toy-runner or tune_closure changes are
needed for the scope above. (If a full multi-round toy SHA is ever wanted, those two
changes would be the way — but it is out of scope for the pilot reconciliation.)

## 7. WCA production RESULT (@120k, 90 runs, 10 seeds/config) — COMPLETE, 0 fail/0 nan

Per-cell method comparison (L2(F) common-estimator mean ± CI95; table +
`results/opes_closure/wca/metrics/production_method_comparison.json`):

  cell               opes(adapt)     opes_fixedg(g=10)  opes_flat(g=inf)
  b1_h2 STARVED      0.1276±0.0029*  0.1295±0.0032      0.1380±0.0036
  b2_h6 INTERMED.    0.0892±0.0024   0.0851±0.0024*     0.1521±0.0065
  b4_h1 EASY         0.0143±0.0017   0.0129±0.0016*     0.0285±0.0105
  (* = best in cell)

Findings:
1. **Well-tempering is first-order.** opes_flat (no well-tempering) is worst in every
   cell — ~80% worse in b2_h6, ~2x worse in b4_h1 — and has the lowest neff_min
   throughout (least stable). This is the robust, publishable closure statement.
2. **Adaptive vs fixed gamma is second-order / a near-tie.** opes wins starved;
   opes_fixedg wins intermediate+easy; all margins within ~1-2 CI widths. The gamma
   SCHEDULE barely matters once well-tempering is present.
3. All 90 runs at 10 seeds, CIs ±0.002-0.006 (except flat b4_h1 ±0.0105). Solid.

Caveat to carry into any writeup: this compares the 3 methods at hyperparameters tuned
for `opes` (adaptive). fixedg/flat share the frozen (barrier,pace,sigma) per cell; they
were not independently tuned. That is the standard "tune the main method, compare
variants at matched settings" design, but state it explicitly.

## 8. Uncommitted changes at this point (ready to commit if desired)
- src/opes_jobs.py: expand() honors per-cell `survivors` (SHA fix; back-compat verified)
- scripts/tune_closure.py: emit writes explicit per-cell survivor tuples; new `freeze`
  subcommand; corrected emit print.
- configs/opes_closure/wca_closure.yaml: added tune_r2, tune_r3, production stages.
- OPES_handoff.md: sections 5-8 (findings, scope decision, toy TODO, this result).
- New metrics: ranking_tune_r{2,3}.json, emit_*.json, freeze_*.json,
  production_method_comparison.json, per_run/per_config rows for the new stages.

## 9. Toy eb RESULT (tune_r1 @40k + sigma-edge check) — sigma optimum found

eb tune_r1 (96 runs, 0 fail): ranked single cell by l2_f_common. Top-8 were ALL at
sigma=0.03 (the smallest r1 sigma), so the optimum sat at the grid edge -> ran a
`tune_sigma_edge` stage (b3/p50, sigma in {0.015,0.02,0.025}, 2 seeds) to probe below.

Full sigma trend at the winning (barrier=3, pace=50), l2_f_common mean ± CI95:
  sigma   L2(F)   ±CI95     (pilot eb sigma = 0.05)
  0.015   0.1800  0.0021
  0.020   0.1796  0.0099   <- interior optimum (flat vs 0.015)
  0.025   0.1910  0.0016
  0.030   0.2110  0.0010
  0.050   0.3005  0.0246   <- best config at pilot sigma
  0.080   0.4092  0.0051
  0.120   0.4692  0.0355

Finding: genuine INTERIOR sigma optimum ~0.02 (L2~0.180), NOT a runaway to zero
(0.015 no better than 0.02). Tuned sigma beats pilot sigma=0.05 by ~40% (0.180 vs 0.300).
Degeneracy ruled out: L2(F) and L2(Fp) both monotone-then-flat, CIs tight (±0.002-0.01),
two seeds agree ~1% at each sigma. neff_frac is low (3-4%) and declines smoothly with
sigma (no collapse; n_kernels 27-29) -- expected for narrow-kernel mean-force, whose
reliability is cross-seed consistency + per-bin sampling, not reweight-neff. Going below
0.02 buys nothing and declining neff says lower would eventually hurt -> 0.02 is defensible.
NOTE: aggregate_closure does NOT summarize the toy scalar `opes_neff_frac` (only WCA's
`opes_neff_frac_min` timeseries); pull neff from raw npz for toys (minor gap, not fixed).

Froze eb `representative` -> barrier=3, pace=50, sigma=0.02 (5 seeds @40k = production
budget; eb tune budget already == prod so r1 winner directly valid).
eb representative CONFIRMED (5 seeds): L2(F)=0.1839±0.0049, L2(Fp)=1.7443±0.0528 —
consistent with the 2-seed edge estimate (0.180), CI tight. eb arm COMPLETE:
tuned 0.184 vs pilot-sigma(0.05) best 0.30 => ~39% improvement from sigma tuning alone.

## 10. Toy meta RESULT (tune_r1 @40k) + 100k confirmation (RUNNING)

meta tune_r1 (96 runs, 0 fail): ranked single cell by l2_f_common. Winner
b8/p200/s0.06 -> L2=0.0595±0.0038. sigma dominates even more than eb (marginal best per
sigma at 40k): s0.06->0.0595, s0.09->0.0685, s0.12->0.2678, s0.18->0.8486. Pilot
sigma=0.12 best (b8/p200) = 0.2678 -> tuned sigma=0.06 is ~4.5x better.

Two honesty caveats being handled:
(a) r1 winner sits at THREE grid edges (barrier max 8, pace max 200, sigma min 0.06).
    sigma is the dominant axis (14x swing) vs barrier/pace (~10-15% near optimum), so
    sigma most needs probing below its edge.
(b) meta tunes @40k but PRODUCTION budget is 100k (only toy where tune!=prod). Must
    confirm ranking at 100k.
Both handled by stage `tune_confirm_100k` @100k: barriers{6,8} x paces{100,200} x
sigmas{0.03,0.045,0.06} x 2 seeds = 24 runs, sharded across GPU 0+1.

confirm_100k RESULT (both caveats resolved):
(a) ranking HELD at production budget — top-3 at 100k all b8/p200 (r1 winner cluster
    confirmed, no reshuffle 40k->100k). barrier=8/pace=200 stay edge-best but only ~10%
    over b6/p200.
(b) sigma INTERIOR optimum confirmed at b8/p200: s0.045 (0.0551±0.0011) ~= s0.03
    (0.0554±0.0043) > s0.06 (0.0580). Flat interior region ~0.045, NOT a runaway; neff
    healthy (~0.29-0.38, no degeneracy — unlike eb's tighter 3-4%). s0.045 wins on point
    estimate + tightest CI.
meta tuned winner @100k = b8/p200/s0.045 -> 0.0551±0.0011.

Froze meta `representative` = b8/p200/s0.045 (5 seeds @100k) AND a matched
`pilot_sigma_ref` = b8/p200/s0.12 (same b/p/budget/seeds, ONLY sigma differs = pilot
value) to isolate the pure sigma effect. Both RUNNING (GPU 0 + 1). Report on completion.

## 11. Toy meta FINAL + FULL CLOSURE SUMMARY

meta matched comparison (b8/p200, 5 seeds @100k, ONLY sigma differs):
  TUNED sigma=0.045: L2(F)=0.0571±0.0027   L2(Fp)=0.3107±0.0084
  PILOT sigma=0.12 : L2(F)=0.1169±0.0130   L2(Fp)=1.0218±0.3359
  => tuned sigma 2.05x better (51%% L2(F) reduction), pure sigma effect (matched b/p/budget/seeds).
  (5-seed tuned 0.0571 consistent w/ 2-seed confirm 0.0551; CIs overlap.)

=== CLOSURE COMPLETE — all three systems, one consistent finding ===
The pilot fixed sigma (never swept it) and it is FIRST-ORDER in every system; the tuned
optimum is well BELOW the pilot value in all three:
  system  pilot_sigma  tuned_sigma  effect
  WCA     0.05         0.02-0.035   b2_h6 wanted 0.02; sigma spans 0.17->0.59 across axis
  eb      0.05         ~0.02        tuned 0.184 vs pilot-sigma best 0.30  (~39%% better)
  meta    0.12         ~0.045       tuned 0.057 vs matched pilot-sigma 0.117 (51%% better)
Plus the WCA 3-method production result (section 7): well-tempering first-order,
adaptive-vs-fixed-gamma a near-tie. Together these fully support marking the pilot's
single-sigma numbers PRELIMINARY (a lower bound on OPES accuracy), which the report
edits (sec:opes_closure_note) already state.

Metrics artifacts: results/opes_closure/{wca,eb,meta}/metrics/per_{run,config}.csv +
production_method_comparison.json (wca). All runs 0 fail / 0 nan.

## 12. Uncommitted changes (nothing committed this session — left for owner)
- src/opes_jobs.py: expand() honors per-cell survivors (WCA SHA fix)
- scripts/tune_closure.py: emit survivor tuples + freeze subcommand
- configs/opes_closure/wca_closure.yaml: tune_r2/r3/production stages
- configs/opes_closure/toys_closure.yaml: eb tune_sigma_edge + frozen representatives;
  meta tune_confirm_100k + frozen representative + pilot_sigma_ref
- OPES_handoff.md: sections 5-12
- results/opes_closure/{wca,eb,meta}/ raw npz, logs, metrics for all stages above
Suggested commit grouping: (1) infra (opes_jobs+tune_closure), (2) configs, (3) results,
(4) handoff. Not staged — user runs git per repo norms.

## 13. 32-seed anchor validation — FORMALLY DESCOPED (decision, not oversight)

The launch plan (section 3) ended with "the 32-seed validation on the anchor cells
for CI width." It is intentionally NOT run. Justification from the 10-seed production
(production_method_comparison.json), per cell, l2_f_common:
  cell           flat-vs-best gap   combined_CI   resolved?   adapt-vs-fixedg   within_CI?
  b1_h2 starved  +0.0104            0.0066        YES         0.0019            yes
  b2_h6 inter    +0.0670            0.0090        YES         0.0040            yes
  b4_h1 easy     +0.0156            0.0121        YES         0.0014            yes
Both qualitative production claims are already decided at 10 seeds: (i) flat (no
well-tempering) is worst in every cell with gap > combined CI; (ii) adaptive vs
fixed-gamma is a near-tie with gap << combined CI. Going to 32 seeds shrinks CIs
around an already-negligible adaptive/fixedg gap (0.0014-0.0040) — it cannot create a
separation, only confirm the tie. No reported conclusion depends on the extra seeds,
so the cost is not justified. If a reviewer demands 32-seed CIs specifically, rerun
production with seeds 0-31 (config stage `production`, bump seeds list); infra is ready.

PLAN STATUS: all launch-plan items complete or explicitly decided. 6-cell extension =
descoped by owner (integrate 3 anchors). 32-seed validation = descoped here on the
numbers above. Report integrated + builds clean. Nothing else outstanding.
