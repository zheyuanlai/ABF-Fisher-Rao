# C60 execution state machine

Read at the start of every session; updated after every stage.  Companion to
`docs/SPEC_c60_water.md` (frozen, Amendment 16) — this file records *where the study is*,
never a spec change.

## Current state (2026-08-14)

| item | state |
|---|---|
| pinned commit | see `results/c60/PINNED_COMMIT` (written at ladder time) |
| spec | FROZEN at `96bf011` (Amendment 16) |
| device | **GPU 3** (Amendment 16.4; NaCl keeps GPU 2) |
| NPT box | **FROZEN: Lx = Ly = 2.651139 nm, Lz = 5.673840 nm** (`results/c60/box/`) |
| PME | pinned: alpha 2.628260884878466 nm^-1, grid 24 x 24 x 48 (`pme_params.json`) |
| engine parity | in progress — `tests/test_c60_engine.py` |
| dt gate | not yet run |
| reference | not started (3 builds pending) |
| Gate 0 pools | not started |
| screen | **prohibited** until reference accepted + Gates 0/A pass |
| mFR | **prohibited** until an establishment-limited cell is frozen |

## Next permitted action

Run `bash scripts/c60_launch_ladder.sh` at a pinned commit, then (separate, reviewed)
launch reference builds 1–3 sequentially on GPU 3, one process each:

```
CUDA_VISIBLE_DEVICES=3 python scripts/c60_reference.py --build 1
```

## Forbidden actions

* anything on GPUs 0–2 (0/1 other group; 2 is NaCl's);
* torch import in an OpenMM-CUDA process (measured deadlock, NaCl);
* any mFR/FR code, calibration, or pilot — `results/c60/calibration/` and
  `results/c60/production/` must not exist;
* editing `docs/SPEC_c60_water.md` (amendment only);
* extending any run past its frozen budget after seeing a result.

## Measured facts worth keeping in view

* MC barostat scales massless carbons **individually** (cages are not molecules to it);
  the NPT script's per-ps cage projector is the remedy — do not reuse the barostat on
  fixed solutes without it.
* OpenMM State forces: parents redistributed, M row keeps the raw virtual-site force.
* The pip OpenMM wheel needs `openmm-cuda-13` plus `LD_LIBRARY_PATH` pointing at
  `site-packages/nvidia/cu13/lib` for the CUDA platform.
* Reference-platform eval at this box: 0.3 s — the parity oracle is affordable.
* Clash-inflated configurations (~1e9 kJ/mol) make FD tests ulp-limited; minimize first.
