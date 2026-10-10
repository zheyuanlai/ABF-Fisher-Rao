# LTA numba-vs-production statistical-equivalence runs (copied for preservation)

These runs were produced during the 2026-10-10 engine workflow, in the session scratchpad
`/tmp/claude-1008/-home-zheyuanlai-ABF-Fisher-Rao/b25e8a63-d0d8-4935-af2b-7d50d5f3ea3b/scratchpad/engines/`. They
were copied here at about 09:45 UTC because that `/tmp` was 95–98 % full. The `.ckpt` checkpoint files were not
copied.

* `fullrun/`: 128 runs, numba engine with its own random draws. T ∈ {300, 150} K × {ABF, FR} × seeds 1–32, N = 1024,
  300 000 steps. Runner `scripts/run_one.py`; scorers `scripts/score_multi*.py` and `scripts/tost_fullrun.py`.
* `tdraw/`: 48 runs, numba engine driven by torch's own CPU random draws, 300 K ABF only. Runner
  `scripts/run_torchdraws.py`; scorers `scripts/score_tdraw.py` and `scripts/tost_tdraw.py`.
* The CUDA comparison side is `results/lta_histogram/production_T{300,150}/` (2026-10-03 campaign).

Scoring used the older `results/uniform_campaign/lta/reference/reference_T*.npz` reference. The numbers are
therefore comparable with the published CUDA CIs, but **not** with the equal-budget ladder's own Ī_F.

The scripts still contain their original hard-coded scratchpad paths (FULL / D). Edit those to point here before
re-running. File sizes and sha256 are in `MANIFEST.json`. Full context and regeneration commands:
`docs/equal_budget/DATA_LOCATIONS.md` §7.
