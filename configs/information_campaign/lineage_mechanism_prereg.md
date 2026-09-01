# Lineage mechanism experiment — frozen before any instrumented run

## The question

The same cloning operator gives ΔI_F ≈ −35 % on LTA 80 K and ≈ +4…+9 % on
ethane/ZIF-8 300 K. The asymptotic Nadaraya–Watson kernel bias cannot explain
either: its p′/p term accounts for 1–2 % of the measured effect in the strongest
positives, and for ZIF-8 it predicts the WRONG SIGN (+0.006 kJ/mol barrier
expansion against −0.088 observed compression). So the mechanism is
finite-particle and lives INSIDE a bin, where the asymptotic formula assumes
independent samples and cloning violates that.

    Does cloning create new conditional information faster than it destroys
    conditional diversity?

## The instrument (diagnostic only — never biases the dynamics)

Per (ancestor a, CV bin g) accumulate the local mean force and count, giving a
per-lineage conditional estimate `m_ag`. The ordinary ABF estimate weights
lineages by their descendant count,

    m_g      = sum_a (n_ag / N_g) m_ag,

so a lineage that was cloned many times is counted many times. Against it, the
**lineage-balanced** diagnostic weights each ancestral discovery once:

    m_g^lin  = (1 / A_g) sum_{a : n_ag > 0} m_ag,     A_g = #{a : n_ag > 0}.

Also recorded: instantaneous per-bin ancestor ESS
`ESS(g) = (sum_a n_ag)^2 / sum_a n_ag^2` and its normalised form `ESS(g)/N_g`
(global ESS is already known to be insufficient); and the local force residual
split by CLONE AGE in frozen buckets **< 0.5 ps, 0.5–5 ps, > 5 ps**, since ten
descendants of one parent may be fine once they have physically decorrelated
(measured gate autocorrelation is ~50 fs, so 0.5 ps is many correlation times
for the gate but short for the guest's CV motion).

## Frozen predictions

**ZIF-8 300 K (harmful).** FR acts after the conditional ensemble in each bin is
already adequate, so cloning overweights microstates without adding new ones.

1. `ESS(g)/N_g` LOWER in the FR arm than the ABF arm, most in the barrier bins.
2. The lineage-balanced estimator is LESS biased than the ordinary one:
   `|m_g^lin − F'_ref| < |m_g − F'_ref|` near the barrier, and the FR arm's
   ordinary-vs-balanced gap is LARGER than the ABF arm's.
3. Fresh clones (< 0.5 ps) carry a NEGATIVE mean force residual in barrier bins,
   relaxing toward zero with age — the origin of the barrier compression.

**LTA 80 K (strongest positive).** The target bins are genuinely starved, so
clones decorrelate and become new conditional information.

4. FR INCREASES the number of distinct ancestors contributing to the difficult
   bins relative to ABF.
5. The lineage-balanced estimator does NOT substantially outperform the ordinary
   one — i.e. descendant weighting is not the problem there.

**The discriminator.** Prediction 2 holding on ZIF-8 while 5 holds on LTA means
one mechanism with two signs, set by whether the bin's conditional ensemble was
already adequate. If instead the lineage-balanced estimator is better on BOTH,
the story is simply "descendant weighting is always harmful" — still useful, but
a different and weaker claim. If it is better on NEITHER, the lineage mechanism
is refuted and I stop rather than look for a sixth explanation.

## Scope, stated honestly

ZIF-8 is instrumented first because its engine is current and it is where the
harm is. LTA requires porting the instrumentation to a second engine and is a
follow-up, not a promise. A ZIF-8-only result establishes the mechanism of the
HARM but cannot by itself establish the two-sign claim.

## Held fixed

Physical model, reference, init pool, seeds, horizon, h_bias, h_read,
min_count, FR rate 0.05, and the frozen endpoints. The instrumentation is
additive and does not enter the bias force or the FR score. The per-(ancestor,
bin) accumulators use scatter-add, whose CUDA ordering is not bit-reproducible;
this is DIAGNOSTIC-only and cannot affect the trajectory, which remains
bit-paired between arms.
