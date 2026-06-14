# Handoff: entropy-dominant bottleneck experiment

## Files added
- `src/edb_abffr_core.py` — m-dimensional ABF+FR engine (generalizes `src/eb_abffr_core.py`).
- `experiments/entropy_dominant_bottleneck/run_entropy_dominant_bottleneck.py` — runner.
- `experiments/entropy_dominant_bottleneck/analyze_entropy_dominant_bottleneck.py` — analysis + plots + this handoff.
- `experiments/entropy_dominant_bottleneck/configs/entropy_dominant_default.yaml` — config.
- Outputs under this directory.

## Exact commands
```bash
source /home/zheyuanlai/miniconda3/etc/profile.d/conda.sh && conda activate abffr
CFG=experiments/entropy_dominant_bottleneck/configs/entropy_dominant_default.yaml
R=experiments/entropy_dominant_bottleneck/run_entropy_dominant_bottleneck.py
A=experiments/entropy_dominant_bottleneck/analyze_entropy_dominant_bottleneck.py
CUDA_VISIBLE_DEVICES=<gpu> python -u $R --config $CFG --smoke-test --device cuda:0
CUDA_VISIBLE_DEVICES=<gpu> python -u $R --config $CFG --pilot      --device cuda:0
CUDA_VISIBLE_DEVICES=<gpu> python -u $R --config $CFG --production  --device cuda:0
python $A --sweep-dir results/entropy_dominant_bottleneck/sweep_20260614_015145
```

## Hardware / runtime
- Device: NVIDIA H200 NVL (cuda:0, CUDA_VISIBLE_DEVICES=7), host atlas.
- Mode: production; total runtime 222s (main 75s, rate 147s).
- Runs: 400 main + 400 rate.
- Settings: beta=4.0, m=2, B0=8.0, N=512, dt=0.0005, n_steps=80000, seeds=20, phis=[0.0, 0.25, 0.5, 0.75, 0.9].

## Status
- Smoke test: analytic sanity checks pass (finite-difference derivative, conditional variance, m=1 reduction).
- Production completed: True.
- No NaNs detected in aggregated runs (analyzer would surface them).

## Key numerical findings
**Headline: nuanced.** The achievable (oracle) FR gain rises strongly with the entropic share, but the deployable self-estimated target does not capture it.
- Deployable estimated-target gain vs phi: slope -8.3 %/unit, r=-0.28 -> does NOT increase with entropic share (U-shaped; hurts mid-range).
- Oracle-target gain vs phi: slope +64 %/unit, r=0.94 -> INCREASES strongly with entropic share (entropic-specific headroom is real).
  oracle gains per phi: ['-6', '-6', '+9', '+44', '+42'].
- estimated gains per phi: ['+13', '+5', '-10', '-8', '+13'] (median +4.7%).
- uniform gains per phi: ['+13', '+6', '-14', '-1', '+13'] (median +6.0%) -- ~identical to estimated => mechanism is balanced resampling, not shape-steering.
- transient (integrated-error) gain is positive everywhere: estimated median +21%, oracle median +38% -- FR accelerates convergence even where final error is unchanged.
- Conditional law Y|X preserved by all FR variants (matches analytic across 3 orders of magnitude).
- Rate sweep: at phi=0.75 estimated FR is net-harmful at every gamma (worse as gamma grows); at phi=0.9 modest gain best near gamma=15. No gamma recovers the oracle-sized gain.
- See `summary_by_phi_method.csv`, `paired_seed_stats.csv`, `rate_sweep_summary.csv`, `conditional_diagnostics.csv`.

## Plots generated
- `plots/barrier_decomposition.png`
- `plots/conditional_variance_phi_entropic.png`
- `plots/convergence_phi_energetic_vs_entropic.png`
- `plots/error_vs_phi.png`
- `plots/gain_vs_phi.png`
- `plots/rate_sweep_phi_075_090.png`
- `plots/x_marginal_selected.png`

## Recommended next steps
- **Close the target gap (highest value).** The oracle proves entropic-specific headroom exists; the limit is the noisy EMA-of-ABF-bias target. Try a better online free-energy estimator inside the FR target (e.g. smoother/slower EMA, eABF/CZAR-style estimate, or a variance-aware target) and re-run the phi sweep.
- Vary m (1,2,4,8) at fixed B0: higher transverse dimension sharpens the entropic force variance and may widen the oracle-vs-estimated gap.
- Probe a kinetically-matched control (equalize mean first-passage time across phi) to separate thermodynamic from kinetic barrier effects.
- Longer/short budget sweep: the deployable gain is budget-dependent (cf. the pilot at T=10 showed a rising trend that washed out by T=40).
