# ZIF-8 300 K on a corrected ABF baseline: mFR is NEUTRAL

**Closed 2026-09-02.** Preregistered in
`configs/information_campaign/corrected_baseline_prereg.json` before any run of
this experiment; outcomes R1–R4 frozen in advance.

## Verdict: R2_NEUTRAL

`h_bias = 0.10 Å`, `h_read = 0.05 Å`, `fr_rate = 0.05` (re-earned), 16 fresh
paired seeds (1000–1015), 300 ps, everything else identical to the legacy setup.

| reference | median ΔI_F | CI95 | seeds worse | verdict |
|---|---|---|---|---|
| full umbrella | +0.80 % | [−1.30, +2.81] | 11/16 | R2_NEUTRAL |
| split-half A | +0.88 % | [−0.75, +2.13] | 11/16 | R2_NEUTRAL |
| split-half B | +0.88 % | [−1.35, +3.05] | 10/16 | R2_NEUTRAL |

**The verdict is identical against all three references**, which is the
robustness rule frozen in advance — it matters because at these bandwidths the
reference is a large share of the residual error.

Validity gates, all passed: the arms are **bit-identical before the first FR
event** (asserted from the production files); FR genuinely acted (5,662 events,
0.92 per replica); genealogy healthy (min ESS/N 0.357, max lineage 0.0221);
transits comparable (26,984 vs 26,570).

## What fixing the baseline did

| | legacy h_bias = 0.20 | corrected h_bias = 0.10 |
|---|---|---|
| full-horizon ΔI_F | **+3.67 % [+1.97, +5.02]** — harmful | **+0.80 % [−1.30, +2.81]** — neutral |
| post-FR ΔI_F | +7.84 % [+4.08, +10.20] | +3.46 % [−5.11, +11.95] |
| ABF's own final e_F | 0.2747 kJ/mol | **0.0838 kJ/mol (10.7× MSE)** |

**Correcting ABF's bandwidth removed the harm.** The measured damage shrank
4.6× on the primary endpoint and the CI now spans zero.

Two controls make this attributable to the baseline and nothing else. First, the
FR **dose is unchanged**: the safety ladder re-run under the corrected dynamics
returned the same rate (0.05) with nearly identical genealogy (ESS/N 0.354 →
0.325, events per replica 0.98 → 0.90), so the intervention is the same size.
Second, the seeds are fresh and the pairing is bit-exact.

## The honest qualifier

The **declared secondary is inconclusive, not neutral**: post-FR ΔI_F is +3.46 %
with CI [−5.11, +11.95]. Restricting to the window where FR actually acts throws
away most of the horizon and the interval widens past the equivalence margin. So
the supported claim is:

> On the frozen primary endpoint, mFR is neutral on a corrected baseline.

and **not** "mFR is harmless in the window where it acts" — that question is
unresolved here, and its point estimate still leans positive (harmful).

## What this settles, and what it does not

**Settles:** the ZIF-8 harm was largely an artifact of comparing against a
mis-resolved ABF baseline. `h_bias = 0.20 Å` over-smoothed the bias force by 7 %,
under-resolving the barrier it was meant to flatten; mFR was then measured
against a baseline that had already given away most of its accuracy. Fix the
baseline and the harm mostly disappears — **and the fix is worth 10.7× in
absolute MSE, against mFR's ±1 %.**

**Does not settle:** whether mFR helps anywhere on a corrected baseline. This is
one system, and one that never had much sampling-limited error left. The WCA and
LTA positives (−21.9 %, −35.1 %) were measured at their own legacy bandwidths;
the p′/p audit showed the asymptotic kernel-bias channel explains only 1–2 % of
them, but that audit does not cover the *online* over-smoothing effect found
here, which is a different mechanism. Whether those gains survive a corrected
baseline is now the open question — and the natural next experiment.

**Closes:** the search for ZIF-8's microscopic harm mechanism. With the harm
itself reduced to a neutral result, the unexplained 91 % of the barrier
compression is no longer a phenomenon worth chasing.
