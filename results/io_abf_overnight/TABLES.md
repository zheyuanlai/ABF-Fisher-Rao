## Headline

| System | role | R_Γ | A6b S(ε₂) | 95% CI | hit A6b/A0 | A6c S(ε₂) | mass ESS | final A6b/A0 | full-domain A6b/A0 | verdict |
|---|---|---:|---:|:--:|:--:|---:|---:|---:|---:|:--|
| Bottleneck beta=4 | control | 21.6 | **1.694** | [1.281, 2.136] | 1.00/0.84 | 0.810 | 0.500 | 0.652 | 1.399 | **NOT POSITIVE** |
| Bottleneck beta=8 | candidate | 12.4 | **1.395** | [1.313, 1.478] | 1.00/0.75 | 0.757 | 0.500 | 0.879 | 1.031 | **POSITIVE** |
| Entropic gateway | candidate | 123.7 | **1.366** | [1.158, 1.600] | 1.00/0.69 | 0.652 | 0.500 | 0.923 | 1.880 | **NOT POSITIVE** |

## Preregistered checks (A6b vs A0)

| System | S ≥ 1.15 | CI lower > 1 | censoring ok | final ≤ 1.10× | full ≤ 1.10× |
|---|:--:|:--:|:--:|:--:|:--:|
| Bottleneck beta=4 | PASS | PASS | PASS | PASS | **FAIL** |
| Bottleneck beta=8 | PASS | PASS | PASS | PASS | PASS |
| Entropic gateway | PASS | PASS | PASS | PASS | **FAIL** |

## Difficulty decomposition

| System | Q₁₀(σ²) | Q₉₀(σ²) | R_σ | Q₁₀(τ) | Q₉₀(τ) | R_τ | Q₁₀(Γ) | Q₉₀(Γ) | R_Γ | valid-τ | ρ_s(Γ early, late) | dominant |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--|
| Bottleneck beta=4 | 0.0745 | 5.49 | 73.7 | 0.000995 | 0.00367 | 3.7 | 0.000574 | 0.0118 | 20.6 | 0.999 | 0.981 | sigma2 |
| Bottleneck beta=8 | 0.0911 | 1.68 | 18.4 | 0.00175 | 0.00686 | 3.9 | 0.000603 | 0.00751 | 12.5 | 0.867 | 0.980 | sigma2 |
| Entropic gateway | 0.00143 | 0.93 | 648.5 | 0.00384 | 0.00731 | 1.9 | 3.93e-05 | 0.0049 | 124.7 | 0.998 | 0.992 | sigma2 |
| WCA dimer (A0 only) | 1.91e+03 | 3.94e+03 | 2.1 | 0.000392 | 0.000857 | 2.2 | 1.36 | 2.42 | 1.8 | 0.676 | — | both |

WCA row is a **diagnostic only** (16 A0 seeds): its reference gate failed, so no speedup is reported for it.
