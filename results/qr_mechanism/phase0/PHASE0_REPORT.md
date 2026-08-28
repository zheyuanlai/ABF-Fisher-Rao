# Phase 0 — the K-family re-audit

2026-08-28. Instrumented exact rerun of Stage 2: K0–K3 × {A0, A6a, A6b} × the original 32 seeds,
original configs verbatim, engine untouched. **Fidelity gate passes on all 12 configs** (median
relative deviation of final e_F from the archived `profiles.csv`: 1.8e-07 … 1.4e-03, all ≪ 1 %),
so what is decomposed below *is* Stage 2, to within the cross-process noise Stage 2's own H6
measured. Identity `E[e²] = b'Qb + tr(QΣ)` closes exactly (machine precision) with `Q` verified
against `metrics.l2_error_F` per profile.

## The verdict is neither preregistered box — the K-family is a mixed regime

η_bias runs **0.28–0.79** across cells and arms, against **0.93–0.999** on the four transfer
systems. Neither outcome box fires (survive required η_bias < 0.5 with a variance-carried margin;
withdraw required η_bias > 0.9 with a bias-carried margin). The honest statement has two halves,
and they point in opposite directions:

### 1. The variance term obeys the Neyman prediction everywhere — first direct measurement

| cell | tr(QΣ): A6b/A6a at T | A6b/A0 at T |
|---|---:|---:|
| K0 | 0.399 | 0.181 |
| K1 | 0.250 | 0.107 |
| K2 | 0.308 | 0.209 |
| K3 | 0.302 | 0.146 |

Following `r ∝ sqrt(aΓ̂)` cut the variance term **2.5–4× against the leverage-only arm and 5–9×
against plain ABF, in every cell, at both audit times.** Stage 2 never measured the variance term
at all — it measured time-to-accuracy. This is the first direct observation of the Γ channel doing
to `tr(QΣ)` exactly what the theory says it should.

### 2. But the measured margins were NOT carried by that term where they were biggest

Signed decomposition of the A6b − A6a margin, `ΔMSE = ΔBias + ΔVar`, at T:

| cell | ΔMSE | ΔBias | ΔVar | reading |
|---|---:|---:|---:|:--|
| K0 | +4.9e-05 | **+3.1e-04** | **−2.6e-04** | bias worsened, variance improved, near-cancellation |
| K1 | −4.3e-04 | +7.6e-04 | **−1.2e-03** | **the win is pure variance**, fighting a bias increase |
| K2 | −9.9e-03 | **−6.4e-03** | −3.6e-03 | 64 % of the win is bias |
| K3 | −1.1e-02 | **−8.5e-03** | −2.1e-03 | 80 % of the win is bias |

At 0.5T the bias share of the win is 59–86 % in K1–K3.

**So the claim "Stage 2 validated the Γ channel" must be narrowed, not withdrawn:** what Stage 2's
*margins* measured was a mixture, dominated in the strongest cells (K2, K3 — the ones the 1.55–1.87×
headline leaned on) by finite-time **bias** reduction. K1 is the one cell whose margin is genuinely
variance-carried. K0's near-tie at T is now explained: a bias increase and a variance decrease of
almost equal size cancelled — which also retires the old puzzle of K0's inconsistent sign between
Stage 0.5 and Stage 2.

## Why the two campaigns disagreed — one geometry, two regimes

On the K-family, extra exposure in a slow cell reduces **both** its variance (Neyman) and its
finite-time bias (more counts → smaller kernel/pseudocount bias exactly where τ made counts
effectively fewer). The two channels are *aligned*, which is why the kappa campaign looked so
clean. On the transfer systems the allocation was driven by the mask geometry (`a = 0` outside),
so the bias channel *opposed* the variance channel at the mask edges — and with η_bias at 0.93+,
the bias channel decided the outcome. One mechanism, two alignments. This is the single
explanation that covers the Stage-2 positives, the overnight inside/outside trade, and the WCA
harm at once.

## Consequences

* The Stage-2 measurements (1.55–1.87× vs A0, 1.27–1.56× vs A6a) stand. Their *attribution* is
  restated: **mixed mechanism; variance-carried in K1; bias-carried in K2/K3.**
* The Neyman theory's proper claim — about `tr(QΣ)` — now has direct supporting measurements, on
  all four cells. What it never had, and still needs, is a regime where that term carries the
  endpoint: Phase 5's η_bias < 0.1 gate is exactly that requirement.
* The finite-time bias model `b[r]` (Phase 1, running) is now the load-bearing object for both
  campaigns' interpretation.
