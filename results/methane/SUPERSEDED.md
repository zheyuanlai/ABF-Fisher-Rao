# Superseded drivers — retained, not live

Two scripts in `scripts/` are **no longer part of any pipeline**. They are kept because the
results they produced are cited, not because anything calls them. Recorded here because the
NaCl session found the sharper version of this hazard in its own tree — a library sampler that
nothing called, carrying a fix, with a test asserting against the dead copy while the live
driver went unchecked.

| script | superseded by | why |
|---|---|---|
| `methane_ti_reference.py` | `methane_ti_torch.py` | OpenMM one-context-at-a-time TI: 115 W of ~600, 465 ns/day. Replaced by the batched engine (Amendment 12.4). |
| `methane_ti_analyze.py` | `methane_reference.py` | reads the per-cell `build*_r*.npz` layout the OpenMM driver wrote; the batched driver writes `ti_final.npz`. |

**No test imports either** (checked), so neither can give false assurance about live code — which
was the failure mode in the NaCl tree. `methane_ti_reference.py` also predates the Gate C and
partition fixes and must not be revived without them.

The live pipeline is, in order:

```
methane_box.py -> methane_baths.py --per-r -> methane_ti_torch.py -> methane_reference.py
              -> methane_screen.py -> methane_gates.py        (+ methane_triton_bench.py)
```

Deliberately-dead *library* code is a different case and is fine: `PairTerms.energy_forces_split`
and `VerletList` are recorded performance negatives, never called in production, and
`tests/test_methane_engine.py::test_alternative_pair_paths_agree_with_the_parity_validated_one`
asserts they still match the live path — so a future fix that misses them fails a test rather
than rotting silently. That is the distinction: dead code with a *consistency* test against the
live path is safe; dead code with a *correctness* test standing in for the live path is not.
