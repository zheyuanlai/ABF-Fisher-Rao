# NaCl / water — status and handoff

**Study:** reversible Na⁺Cl⁻ ion pairing in explicit water, the published Talmazan et al. 2025
ABF tutorial system, asked as the campaign's four-way regime question.
**Spec (frozen):** [`docs/SPEC_nacl_water.md`](docs/SPEC_nacl_water.md) · **licence:**
Amendment 14 of `docs/V2_PREREGISTRATION.md` · **branch:** `v2-campaign`.

Runs concurrently with the methane study, which owns the shared periodic engine.

---

## Where the study is

| stage | state |
|---|---|
| 0A model extraction | **DONE** — `results/nacl/stage0/` (manifest, site params, published restart) |
| 0C OpenMM parity target | **DONE** — built from the published PSF/par files, parameters read back out |
| I engine equivalence | **PASSED 11/11** at `< 1e-6` — `tests/test_nacl_engine.py` |
| 1.3 NPT box | running / see `results/nacl/box/box_manifest.json` |
| 3.2 RDF + descriptor freeze | pending (needs the box) |
| 1.2 dt gate (2 vs 1 fs) | pending (needs the box) |
| II TI reference + Gate 0 + Gate A | pending (needs baths + dt) |
| III ABF-only regime map | pending (needs the reference to be interpreted) |
| IV mFR production | **not licensed** — requires every gate to pass first |

## The question, and what would end the study early

The hypothesis is the three-timescale window `T_hit << tau_perp << T_est` with hydration
reorganisation as the orthogonal physics. Gates run in the frozen order and the **first failure
is the verdict**:

```
  Gate 0  conditional mean force trustworthy at fixed r?   (cross-family spread)
  Gate A  hydration states visible through r?              (TV >= 0.30)
  Gate B  discovered?                                      (T_hit < 0.1 T on >= 6/8)
  Gate C  established?                                     (bias-aware deficit)
  Gate D  clones decorrelate faster than replaced?         (lambda_rep tau_perp <= 0.1)
```

Gate 0 leads because it is the cheapest way to end the study honestly: it comes out of the TI
reference the study needs anyway, and a failure there means marginal selection provably cannot
help (the Amendment 8 theorem `d/dt p_t(y|xi)|_FR = 0`).

## Execution order and why

```
  box(NPT) -> baths -> dt gate -> TI reference -> [Gate 0, Gate A] -> screen -> [Gate B, C]
                                                                   -> calibration -> production
```

The reference runs **before** the screen although the screen does not need it to *run*: Gate 0
and Gate A are decided from the reference, they lead the frozen order, and they can stop the
study for ~1 day of compute instead of ~4.

## Commands (one GPU per study, pinned, idleness recorded)

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/nacl_box_npt.py
CUDA_VISIBLE_DEVICES=2 python scripts/nacl_descriptor_freeze.py
CUDA_VISIBLE_DEVICES=2 python scripts/nacl_baths.py --per-r
CUDA_VISIBLE_DEVICES=2 python scripts/nacl_dynamics_gate.py
CUDA_VISIBLE_DEVICES=2 python scripts/nacl_benchmark.py --correctness --timing
CUDA_VISIBLE_DEVICES=2 python scripts/nacl_ti_torch.py
python scripts/nacl_ti_analyze.py
CUDA_VISIBLE_DEVICES=2 python scripts/nacl_screen.py --triton
python scripts/nacl_gates.py
```

`nacl_box_npt.py`, `nacl_descriptor_freeze.py` and `nacl_baths.py` are OpenMM-CUDA processes and
must run in the **`methane-cuda`** environment (`abffr`'s OpenMM has no CUDA platform); the
torch stages run in `abffr`. Nothing may import torch in an OpenMM-CUDA process — the two
runtimes deadlock on context creation on the same device, measured.

## Things that would otherwise have to be rediscovered

* **The SI archive is behind a proof-of-work cookie.** `cache/talmazan2025/solve_pow.py` solves
  it (SHA-256, 4 leading zeros) and downloads the 107 MB zip; hashes are in
  `cache/talmazan2025/nacl_file_hashes.sha256`.
* **The NaCl folder ships no CHARMM topology file** — only `par_all22_prot.inp`. The matching
  `top_all22_prot.inp` is elsewhere in the same archive (`Ethanol-Hydration/WTM-eABF/common/`).
* **OpenMM keeps one H–O–H angle term per rigid water**; NAMD's `rigidbonds all` runs the same
  water by SETTLE. With the triangle fully constrained the term is a constant on the manifold,
  so it is asserted inert and removed. Not doing this leaves a force the parity target has and
  the torch engine does not.
* **CHARMM TIP3P hydrogens carry LJ.** That invalidates the SPC/E-specific split-path assertion
  in `PairTerms` (now a `split_path_valid` flag) and means every one of the 2465 sites is both
  charged and LJ-active.
* **The published domain top sits at ~97 % of L/2**, which is the published setup's own
  finite-size exposure, inherited and gated (`R_hi <= 0.97 L/2`) rather than assumed away.
* **The published `abf.in` has no `hideJacobian`**, so the shipped `abf.pmf` is the book's
  `F(r)`, not the radial `W(r)`. Both are emitted everywhere and the identity
  `W' = F' + 2/(beta r)` is a hard test.

## Compute agreement with the methane session

Amendment 14.4: one device per study. Methane consolidates on GPU 3; **NaCl takes GPU 2** at the
seed-5004 boundary (~06:30 UTC 2026-08-13), recorded in
`results/methane/screen_N512/SCHEDULING.md`. GPUs 0 and 1 belong to another group and are not
touched. Every stage records its device and the device's idle state; timing numbers from a
contended device are never quoted as throughput.
