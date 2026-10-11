# Experiment A: frozen outcome classification (PREREGISTRATION.md 1.7)

Generated 2026-10-11T09:28:10Z by scripts/final_campaign/expA_outcome.py (final_expA_outcome/1) from cross_cell.json (sha256 c7145c941486) and mech.json (sha256 a8d77fea4bcf).  Complete: True.

Resolutions r1-r3 of cases the frozen table leaves open (analysis decisions after the freeze): recorded in the dated amendment docs/final_campaign/PREREGISTRATION.md line 586: "**A1.6 Outcome-table resolutions [EXPA-OUTCOME-RESOLUTIONS-r1-r3].** (r1) p_Holm < 0.05, R > 1 and G_hi < 0 meets both" (block sha256 79a31f068453).

## E1: lambda 16 vs lambda 1

| N | label | wording | R [95 % CI] | p Holm | G_hi | guards g1 g2 g3 g4 | failed | rule |
|---|---|---|---|---|---|---|---|---|
| 512 | **A** | faster mobility rescues FR | 0.6525 [0.6268, 0.7848] | 0.0004 | -0.299 | True True True True | -- | p_Holm < 0.05, R < 1, G_hi < 0, g1-g4 hold |
| 128 | **A** | faster mobility rescues FR | 0.3726 [0.2937, 0.4566] | 0.0004 | -0.071 | True True True True | -- | p_Holm < 0.05, R < 1, G_hi < 0, g1-g4 hold |

**Combined across N: rescue (A governs)** -- faster mobility rescues FR
(faster mobility (lambda 16 vs 1): A, A-qualified or B at both N (N = 512: A; N = 128: A); the weaker label A governs the wording)

Interpretation (frozen map):

* [identity] lambda changes gate transport (MC2) as well as conditional relaxation (MC1): any change in R is attributed to transverse mobility; an attribution to conditional relaxation alone is a [hypothesis].  Read with MC1 / MC2 (results/final_campaign/expA/validation), the ABF-only descriptors and the decomposition.
* [observed] N = 512: ABF improves strongly with lambda (ABF Ibar_F ratio 16/1 CI upper 0.141 < 1) while R changes: the change may come from ABF's own improved transport / establishment rather than from FR's clones (confound).
* [hypothesis] N = 512: rescue (A) with MC3 passing: consistent with the clone fibre-lag hypothesis.
* [observed] N = 128: ABF improves strongly with lambda (ABF Ibar_F ratio 16/1 CI upper 0.143 < 1) while R changes: the change may come from ABF's own improved transport / establishment rather than from FR's clones (confound).
* [hypothesis] N = 128: rescue (A) with MC3 passing: consistent with the clone fibre-lag hypothesis.
* (E1 vs E2) other family: rescue (A governs)

## E2: lambda 128 vs lambda 1

| N | label | wording | R [95 % CI] | p Holm | G_hi | guards g1 g2 g3 g4 | failed | rule |
|---|---|---|---|---|---|---|---|---|
| 512 | **A** | faster mobility rescues FR | 0.7797 [0.6629, 0.8184] | 0.0004 | -0.302 | True True True True | -- | p_Holm < 0.05, R < 1, G_hi < 0, g1-g4 hold |
| 128 | **A** | faster mobility rescues FR | 0.324 [0.2682, 0.3841] | 0.0004 | -0.233 | True True True True | -- | p_Holm < 0.05, R < 1, G_hi < 0, g1-g4 hold |

**Combined across N: rescue (A governs)** -- faster mobility rescues FR
(faster mobility (lambda 128 vs 1): A, A-qualified or B at both N (N = 512: A; N = 128: A); the weaker label A governs the wording)

Interpretation (frozen map):

* [identity] lambda changes gate transport (MC2) as well as conditional relaxation (MC1): any change in R is attributed to transverse mobility; an attribution to conditional relaxation alone is a [hypothesis].  Read with MC1 / MC2 (results/final_campaign/expA/validation), the ABF-only descriptors and the decomposition.
* [observed] N = 512: ABF improves strongly with lambda (ABF Ibar_F ratio 128/1 CI upper 0.0513 < 1) while R changes: the change may come from ABF's own improved transport / establishment rather than from FR's clones (confound).
* [hypothesis] N = 512: rescue (A) with MC3 passing: consistent with the clone fibre-lag hypothesis.
* [observed] N = 128: ABF improves strongly with lambda (ABF Ibar_F ratio 128/1 CI upper 0.0491 < 1) while R changes: the change may come from ABF's own improved transport / establishment rather than from FR's clones (confound).
* [hypothesis] N = 128: rescue (A) with MC3 passing: consistent with the clone fibre-lag hypothesis.
* (E1 vs E2) other family: rescue (A governs)

## Resolutions of cases the frozen table leaves open

* r1: p_Holm < 0.05, R > 1 and G_hi < 0 satisfy both the I and the C-reverse rule: labelled C-reverse (the contrast is significant; the significant-contrast rows decide on R, consistent with the combination rule's 'opposite significant signs -> N-dependent'), also_satisfies I, I wording appended; a top-down reading would give I
* r2: a guard that cannot be evaluated does not hold: an A case becomes A-qualified naming 'gX (not evaluable)'
* r3: n < 10 paired seeds -> 'no-claim'; non-finite inputs -> 'not-available' (outside the seven frozen labels); either at one N makes the combination 'no global claim (incomplete)'
