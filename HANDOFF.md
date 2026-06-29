# WCA reviewer-proofing — HANDOFF

This documents the additive WCA follow-up layer built on top of the existing
phase-diagram study. Nothing in `results/wca_phase_diagram/production/` (the 224
completed runs) or the original notebooks was modified. All new code is
command-line runnable, resumable/idempotent, and writes a manifest per output dir.

Environment: `conda activate abffr` (sources `~/miniconda3/etc/profile.d/conda.sh`).
GPU policy: at most two GPUs, never more. The request was to default to GPU 0, but
GPU availability on this shared box changes — at launch time GPU 0 (and 4, 5) were
occupied by other users (3–108 GB), so the representative production was launched
on the genuinely-free **GPU 1,2** instead of 0. Always check `nvidia-smi` and pick
free GPUs; the launchers take the GPU list as an argument
(`run_wca_followup_h200.sh <config> <stage> <gpus_csv>`).

---

## 1. What changed (summary)

### Engine (`src/wca_abffr_core.py`) — additive, old methods byte-identical
- New method `fr_estimated_adaptive` (deployable, diversity-aware FR). Added to
  `FR_METHODS`; new `ESTIMATED_TARGET_METHODS` so it shares the online EMA target.
- New `adaptive_*` fields on `SimConfig` (defaults disable adaptivity; non-adaptive
  methods unaffected). `SimConfig.config_hash` is metadata only (not in any run id),
  so existing production runs still resume correctly.
- `adaptive_fr_rate(...)`: effective rate = base · support_gate · diversity_gate ·
  event_gate, clipped to [min,max]. Support gate default mode `marginal_mismatch`
  ramps on the EMA of L2(p_hat, q_target) — the only online signal that separates
  starved (~0.22) from accurate (~0.13–0.15) WCA cells. Diversity gate tapers in
  ancestor ESS fraction (full ≥0.25, off ≤0.10). Event gate backs off near the cap.
- Per-FR-event adaptive log + aggregate score statistics (`fr_score_std`,
  `fr_score_absmax`, `fr_score_clip_fraction`) stored in `diag`.
- `fixed_population_birth_death_torch(..., fr_rate_override=None)` so the adaptive
  schedule can pass a per-event rate (default path unchanged).
- `run_frozen_bias_gpu(...)`: fixed-bias dynamics (no ABF update, no FR) +
  reconstruction `F_recon = B(z) - beta^{-1} log p_B(z) + C`.
- No-oracle-leakage guard retained; the adaptive method never receives the TI ref.

### New modules / scripts
- `src/wca_followup_jobs.py` — `FollowupRunSpec` (superset of PhaseRunSpec with
  adaptive knobs + budget + sample/frozen mode), expansion, sample + frozen run
  executors, learned-bias loader. Does NOT touch `wca_phase_jobs.PhaseRunSpec`.
- `scripts/run_wca_followup.py` — resumable runner; `--dry-run`, `--overwrite`,
  `--max-runs`, `--shard/--num-shards`, `--seeds`, `--cells`, `--methods`,
  `--precompute-references`; writes `manifest_<stage>_shard*.json`.
- `scripts/run_wca_followup_h200.sh` — 1–2 GPU launcher (one process/GPU; refs
  precomputed for sample mode; frozen reuses cached refs).
- `scripts/analyze_wca_starvation.py` — Part B; pure re-analysis of existing runs.
- `scripts/analyze_wca_followup.py` — Parts C/D/E/F aggregation + LaTeX tables.
- `scripts/plot_wca_followup.py` — fixed-vs-adaptive, schedule, equal-compute,
  frozen figures (seed IQR bands), with `--report-figdir`.
- `scripts/make_followup_report_assets.py` — `report/tables/wca_followup_numbers.tex`
  macros (placeholders `--` until each study is aggregated, so the report compiles).

### New configs
- `configs/wca_representative.yaml` (Parts C+D), `configs/wca_equal_compute.yaml`
  (Part E), `configs/wca_frozen_bias.yaml` (Part F), `configs/wca_followup_smoke.yaml`
  (fast mechanical smoke for all modes).

### Report (`report/`)
- `sections/06_case_wca.tex`: new subsections — Starvation diagnostics, Adaptive
  diversity-aware FR, Equal-compute comparison, Frozen-bias validation; discussion
  updated. `main.tex` inputs `tables/wca_followup_numbers.tex`.
- Starvation figures copied to `report/figures/fig_wca_phase_starv_*`.
- Stub tables/figures created so the report compiles before GPU runs finish; they
  are overwritten by `analyze_wca_followup.py` / `plot_wca_followup.py`.

---

## 2. Status of results

### DONE (no GPU needed) — Part B starvation diagnostics
`results/wca_phase_diagram/production/starvation/`:
`wca_starvation_summary.csv` (224 rows), `wca_phase_diagram_augmented.csv`
(42 rows = 14 cells × 3 FR methods), 6 plots, `manifest.json`.
Key result: Spearman ρ(ABF error, mFR gain) = +0.80 vs ρ(βh, gain) = −0.57.
All 4 β=1 starved cells gain +47–51% (4/4 wins); easy cells lose, worst
β=4,h=1 at −489%. This is the centerpiece reviewer figure
(`fig_wca_phase_starv_01_abferr_vs_gain.png`).

### ADAPTIVE FR (Part D) — implemented, calibrated, validated, honestly bounded
The deployable `fr_estimated_adaptive` was validated at 120k steps (matched seeds):
- starved anchor β=1,h=2: ABF L2(F)=0.092 → adaptive 0.039 (**+58%**, gate
  saturated, ESS~180) — PRESERVES (even beats) the fixed-mFR gain.
- easy β=4,h=1: ABF L2(F)=0.0036 → adaptive 0.016 (−344%); fixed mFR was −489%, so
  adaptive REDUCES harm ~30% but does NOT eliminate it.
Key finding (5 calibration attempts): NO marginal-shape online signal separates the
harmful cold-accurate cells from the helpful warm-starved cells — cold cells have
genuinely sharp bimodal marginals as non-flat as starved cells. The only clean
separator is the measured ABF error (ρ=0.80, Part B), which is unobservable online.
The shipped gate uses the low-variance cumulative marginal non-flatness (exogenous
to FR) + a gate warm-up + diversity/event gates; it is a conservative first version
documented as such in the report. This is a finding, not a bug.

### COMPLETE (GPU) — all studies finished, 0 failures
- Representative cells (Parts C/D): 300/300 (6 cells × 5 methods × 10 seeds). Real
  10-seed result: starved b1h2/b1h4 +50%/+50% (10/10); intermediate b2h6 +25%;
  diagnostic b2h4 fixed -50% vs adaptive -2% vs oracle +24% (target-limited);
  easy b4h1/b4h2 net-harmful for all targets incl. oracle (no headroom).
- Equal-compute (Part E): 168/168 (equal_compute 144 + equal_compute_plus 24).
  Starved cell: mFR@base (0.041) beats ABF at every equal-budget shape AND at 2x
  budget (~0.09) — gain is the birth-death correction, not compute.
- Frozen-bias (Part F): 48/48. learned-bias L2 == online L2 (mFR genuinely ~2x
  better, not a transient); single-window reconstruction is mixing-limited
  (~0.21 at b1h2) and does not finely rank methods.
Runs were executed on **GPU 1,2** (GPU 0/4/5 were busy). All resumable: re-run the
launcher with any free GPUs to continue/extend.

---

## 3. Commands to reproduce / continue

```bash
conda activate abffr   # source ~/miniconda3/etc/profile.d/conda.sh first

# --- Part B (done; re-runnable any time, no GPU) ---
python scripts/analyze_wca_starvation.py \
  --config configs/wca_phase_diagram_production.yaml --stages production

# --- Part C/D representative cells (resumable; 1 or 2 GPUs) ---
python scripts/run_wca_followup.py --config configs/wca_representative.yaml \
  --stage representative --dry-run                      # workload summary
bash scripts/run_wca_followup_h200.sh configs/wca_representative.yaml representative 0,4
python scripts/analyze_wca_followup.py --config configs/wca_representative.yaml --stages representative
python scripts/plot_wca_followup.py --config configs/wca_representative.yaml \
  --stage representative --report-figdir report/figures

# --- Part E equal-compute ---
bash scripts/run_wca_followup_h200.sh configs/wca_equal_compute.yaml equal_compute 0,4
bash scripts/run_wca_followup_h200.sh configs/wca_equal_compute.yaml equal_compute_plus 0
python scripts/analyze_wca_followup.py --config configs/wca_equal_compute.yaml \
  --stages equal_compute equal_compute_plus
python scripts/plot_wca_followup.py --config configs/wca_equal_compute.yaml --stage equal_compute \
  --report-figdir report/figures

# --- Part F frozen-bias (after representative is done) ---
bash scripts/run_wca_followup_h200.sh configs/wca_frozen_bias.yaml frozen_bias 0,4
python scripts/analyze_wca_followup.py --config configs/wca_frozen_bias.yaml --stages frozen_bias \
  --online-summary results/wca_representative/summaries/representative_cells_summary.csv
python scripts/plot_wca_followup.py --config configs/wca_frozen_bias.yaml --stage frozen_bias \
  --report-figdir report/figures

# --- report assets + compile ---
python scripts/make_followup_report_assets.py \
  --starvation results/wca_phase_diagram/production/starvation/wca_phase_diagram_augmented.csv \
  --representative results/wca_representative/summaries/matched_seed_table.csv \
  --equal-compute results/wca_equal_compute/summaries/equal_compute_summary.csv \
  --frozen results/wca_frozen_bias/summaries/frozen_bias_summary.csv
cd report && tectonic -X compile main.tex
```

Optional crowding-success cell (computes a NEW TI ref ~5 min): stage
`representative_crowding` in `configs/wca_representative.yaml` (cell a=1.35).

---

## 4. Deliverables checklist (Part I) — ALL COMPLETE
- [x] HANDOFF.md (this file)
- [x] New scripts/configs/engine method (all CLI-runnable, resumable, manifested)
- [x] `wca_starvation_summary.csv`, `wca_phase_diagram_augmented.csv` (+ plots)
- [x] `representative_cells_summary.csv`, `matched_seed_table.csv`,
      `adaptive_fr_event_log.csv`
- [x] `equal_compute_summary.csv`
- [x] `frozen_bias_summary.csv`
- [x] Starvation scatter + augmented diagnostics plots
- [x] fixed-vs-adaptive, equal-compute, frozen-bias plots
- [x] Updated report source; compiles to `report/main.pdf`
Note: raw run outputs (`results/**/raw/*.npz`) and summary CSVs are git-ignored by
the repo's `.gitignore`; the manuscript figures/tables under `report/` are tracked.

---

## 5. Suspicious results / notes
- Adaptive support gate calibration (important): the first attempt used the
  p_hat-vs-FR-target mismatch, but that signal is *shrunk by the very FR firing it
  gates* (a feedback loop) and its starved/easy distributions overlap, so it could
  not shut FR off in the easy cell (b4h1 still ~ -400% at thresholds 0.18/0.22).
  The shipped default is `marginal_uniform`: ramp on L2(p_hat, uniform), the
  ABF-driven marginal non-flatness, which is exogenous to FR and separates starved
  β=1 cells (~0.12-0.14) from accurate cells (~0.086-0.10); ramp `[0.10, 0.13]`.
  Validated at 120k steps: starved b1h2 keeps firing (large gain), easy b4h1 turns
  the gate off (see `results/wca_representative/logs/recalib2_adaptive.log`).
- Genuinely ambiguous cells: high-β high-h cells (b4h6, b4h4) have a non-flat ABF
  marginal yet FR hurts; NO single online scalar separates them from the starved
  cells (only the unobservable ABF error does, ρ=0.80 in Part B). The adaptive
  schedule is deliberately conservative — it preserves the large β=1 gains and
  avoids the large clearly-easy harms, but will mis-fire on these ambiguous cells
  and may skip modest intermediate gains (e.g. β=2,h=6 +20%). This is a documented
  limitation, not a bug.
- Score statistics are stored only for the new follow-up runs; the original 224
  phase runs predate that logging, so those columns are blank in the starvation
  summary (honest, not an error).
