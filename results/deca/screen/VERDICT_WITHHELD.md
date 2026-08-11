# No regime verdict — the ABF baseline failed Gate 0

`screen_verdict.json` in this directory reports `establishment-limited` with structural
corroboration. **That verdict is not issued and must not be cited.** It was computed before
Amendment 7 existed, and it fails the gate Amendment 7 adds.

## Why

`Q*_k(t)` and `Q*_y(t)` are both computed **from the applied bias `B_t`**. This run's bias is
20–53 % wrong, so both targets are wrong and any deficit measured against them is an artifact of
the baseline.

| | |
|---|---|
| `A_hat` span, 8 seeds | 86.7 – 110.3 kT against a **72.0 kT** reference |
| walkers above 2.80 nm, second half | 0.951 – **0.9996** |
| population trace, seed 0 | `[1.00, 0, 0]` at 0 ps → `[0, 0, 1.00]` by 100 ps, never returning |
| learned mean force vs `dF_ref/dR` | **61 %** relative error at up to 2e6 effective counts |

## What was ruled out

`results/v2_validity_audits/deca_mean_force/` ran the *same* `f_loc` estimator inside
umbrella-restrained windows with validated conditional sampling: **8.4 % relative error**, and
integrating `⟨f_loc⟩` gives 69.4 kT against the reference's 67.1 kT. The estimator, the CV
geometry, the reference and the integration are mutually consistent. **There is no bug in the
mean force.**

## What it means

ABF's *conditional* ensemble at fixed `xi` is not equilibrated at 16 walkers × 0.5 ns — the
hidden conformational degrees of freedom do not relax at fixed end-to-end distance. This is a
third failure mode, distinct from discovery-limited and establishment-limited, and **no marginal
reallocation rule can repair it** — not mFR, not book-Laplacian selection, not count balancing —
because they all move population *along* `xi` and the fault is *orthogonal* to `xi`.

The run itself is valid data and is retained. Only the regime verdict is withheld.
