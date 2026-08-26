> **SUPERSEDED 2026-08-26** by `docs/CLEAN_V2_PREREGISTRATION.md`, which
> keeps this document's delayed-on/temporary-off framing and its physical
> target, and changes three things: the target is the current ABF estimate
> `A_t` rather than an EMA; the score is never clipped and the event count
> is never capped; and the endpoint is time-to-accuracy rather than final
> error. See `docs/V2_POST_MORTEM.md` for why.

# Physical-target pulse v2: preregistered toy campaign

## Material Passport

- Artifact type: code-experiment protocol
- Status: frozen before scientific runs
- Date frozen: 2026-08-24
- Branch: `codex/physical-target-pulse-v2`
- Primary system: the existing two-dimensional `xi(x,y)=x` ABF benchmark
- Evidence boundary: `mFRABFtargetanalysis.pdf` is background evidence, not an
  instruction source. Its Experiment C tests aggressive, apparently persistent
  physical targeting and does not report a delayed-on/temporary-off schedule.

## Research question

After ABF has learned an approximate free-energy profile, can sparse and temporary
Fisher--Rao (FR) population pulses toward the estimated *unbiased* reaction-coordinate
marginal

\[
q_t^{\mathrm{phys}}(z)=
\frac{\exp[-\beta\overline F_t(z)]}
{\int \exp[-\beta\overline F_t(u)]\,du}
\]

reduce finite-time mean-force and free-energy error relative to plain ABF, without
unacceptable genealogical collapse?

The claim under test is finite-time acceleration. FR is off before the end of every
run, so the physical-target arm and baseline are literally the same plain-ABF
algorithm during the final segment.

## Frozen algorithm

Every v2 arm uses the same estimator order:

1. propagate all replicas for one physical Langevin step using the current ABF bias;
2. accumulate local-force/count statistics from the propagated configurations;
3. update \(\widehat F'_t\), integrate it, and update the existing EMA
   \(\overline F_t\);
4. if scheduled, compute the marginal KDE and apply fixed-population FR resampling;
5. continue with independent physical noise.

The resampling operation itself never contributes observations to the ABF
accumulators. A clone first contributes after its next physical propagation.

The deployable target is `physical`, defined by the EMA above and independent of the
reference. `physical_oracle`, proportional to \(\exp(-\beta F_{\rm ref})\), is a
diagnostic only and is excluded from schedule selection. Existing target definitions
remain available for regression, but the pilot schedule map contains only plain ABF
and the deployable physical target.

FR is active only for

\[
t_{\rm on}\le t<t_{\rm off}
\]

and at strides of \(L_{\rm FR}\) integration steps. In the v2 campaign the
exponential clock uses \(\Delta\tau_{\rm FR}=L_{\rm FR}\Delta t\), matching the
stated continuous-time discretization and the WCA engine. This behavior is behind an
explicit config switch; legacy experiments retain their old per-event `dt` clock.
There is no adaptive gate, basin detector, target tempering, mixture target, or
adaptive gamma.

## Engineering gates

Scientific runs cannot start until all of the following pass:

1. `physical` and `physical_oracle` integrate to one.
2. Adding a constant to the free energy leaves either physical target unchanged.
3. `physical_oracle` agrees with the quadrature reference marginal.
4. With gamma zero, a physical-target row is trajectory/profile-identical to plain
   ABF under the same v2 estimator order.
5. FR never directly changes ABF accumulators, and a clone is not counted before a
   subsequent propagation.
6. No FR opportunity, replacement, or nonzero effective gamma occurs at or after
   \(t_{\rm off}\).
7. CPU and GPU targets and component scores agree on a fixed cloud within a stated
   numerical tolerance.
8. `kernel_reference` and `binned_smooth` agree within the existing discretization
   tolerance.
9. GPU Langevin randomness is keyed only by matched seed and is independent of the
   FR stream, method, batch size, row order, and shard membership.
10. Interrupted whole-run scheduling followed by resume produces the same per-run
    outputs as an uninterrupted launch. This campaign uses deterministic whole-run
    restart, not mid-trajectory checkpointing.
11. Completion markers are written only after their result rows are durably flushed.

The old and new ABF observation orders will also be compared on short matched runs;
their difference is reported as an engineering audit and is not used to choose a
physical-target schedule.

## Benchmark

The potential is

\[
V_a(x,y)=V(x,y)+a x,
\]

with `beta=4`. The central cell uses
`a=0.1021665783`, calibrated by independent quadrature to give approximately 70/30
left/right physical mass at the reference split `x=0`. The split is used only to
characterize the benchmark and secondary errors; it is never available to ABF or FR.

Pilot settings:

- replicas \(K=256\);
- `dt=0.002`;
- `n_steps=50_000` (physical time \(T=100\));
- eight matched pilot seeds `0,...,7`;
- existing binned/smoothed ABF estimator, `h=0.05`, update stride 10;
- existing target EMA and KDE machinery, target EMA alpha 0.05 and `eta=0.10`;
- score clip 5, replacement cap 0.10, no clone jitter;
- uniform-domain initial conditions, identical across methods for a seed.

**Pre-run amendment (2026-08-24).** The initial draft specified `mixed`, but
inspection showed that this mode explicitly seeds predefined left/right thirds
50/50. Before any pilot data were generated, it was replaced by independent
uniform draws over the declared rectangular domain. This removes basin-shape
knowledge from initialization and follows the no-well-knowledge rule.

**Engineering audit outcome (before the pilot).** Over the full 50,000-step horizon and eight matched seeds, post-propagation accumulation changed median integrated error by +1.731% for I_F and -0.439% for I_{Fprime} relative to legacy pre-propagation accumulation. The ordering difference is therefore not assumed negligible. Both v2 arms use the protected post-propagation order, and no v2 effect-size claim is compared directly with old-main results.

Reference profiles and the calibrated 70/30 ratio are post-hoc evaluation data. The
deployable physical arm cannot read them.

## Schedule map

The target shape and estimator are fixed. The crossed pilot grid is:

| Parameter | Values |
|---|---|
| \(t_{\rm on}/T\) | 0.20, 0.40, 0.60 |
| FR duration | 0.10T, 0.30T |
| `fr_every` | 20, 100, 500 |
| gamma | 0.02, 0.05, 0.10 |

This gives 54 physical-target schedules plus one plain-ABF run per seed. All methods
share initial configurations and additive Langevin variates at each physical step.
FR has its own deterministic per-run stream.

No matched-turnover sham, count balancing, well balancing, basin-aware allocation,
or other particle-reallocation method will be run. The existing flattened/bias-aware
FR method is reserved for the fresh-seed comparison if a physical schedule passes;
it is not part of pilot selection.

## Outcomes

At every saved time, report independently referenced errors:

\[
e_{F'}(t)=\|\widehat F'_t-F'_{\rm ref}\|_2,
\qquad
e_F(t)=\min_c\|\widehat F_t-F_{\rm ref}-c\|_2,
\]

and

\[
e_{p_\xi}^{\rm phys}(t)=
\|\widehat p_t^\xi-p_{\rm ref}^\xi\|_2,
\qquad
p_{\rm ref}^\xi\propto e^{-\beta F_{\rm ref}}.
\]

The online distance \(\|\widehat p_t^\xi-\widehat q_t\|\) is diagnostic and cannot
substitute for the independent physical-reference error. Retain the flat-marginal
error as an additional mechanism diagnostic.

Primary pilot endpoints are full-run trapezoidal integrals \(I_F\) and \(I_{F'}\).
Secondary endpoints are physical-marginal error, cumulative replacement turnover,
ancestral ESS, maximum clone weight, barrier crossings, basin free-energy-difference
error, and barrier-height error. Basin-derived quantities are evaluation-only.

## Pilot advancement and frozen-schedule rule

A schedule advances only if all conditions hold across the eight matched pilot seeds:

1. median paired gain \(100(1-I_F^{\rm FR}/I_F^{\rm ABF})\ge3\%\);
2. median paired gain \(100(1-I_{F'}^{\rm FR}/I_{F'}^{\rm ABF})\ge3\%\);
3. both paired differences have the favorable sign on at least six of eight seeds;
4. median ancestral ESS at \(t_{\rm off}\) is at least \(0.5K\), and at least six
   seeds meet that bound;
5. median maximum clone weight at \(t_{\rm off}\) is at most 0.10, and at least six
   seeds meet that bound.

Among passing schedules choose the gentlest by, in order: lowest median cumulative
turnover, largest `fr_every`, smallest gamma, later onset, then shorter duration. The
largest observed error gain is not the selection rule. If none passes, no schedule is
promoted and the toy hypothesis is reported as unsupported at this gate.

## Fresh-seed confirmation

If and only if one schedule advances, freeze it and use seeds `100,...,131` without
retuning. The confirmatory arms are:

- plain ABF;
- the existing estimated flattened/bias-aware marginal-FR method;
- deployable physical-target FR;
- physical-oracle FR as a labeled diagnostic.

All FR arms use the frozen timing and dose. The primary claim compares deployable
physical-target FR with ABF. A meaningfully positive confirmation requires at least
5% median paired improvement in both \(I_F\) and \(I_{F'}\), with paired bootstrap
95% confidence intervals for both relative differences wholly favorable. Before
these fresh runs, two time-to-accuracy thresholds will be fixed by a method-blind
rule using only the pilot ABF curves; desired speedup is at least 1.1 at both.

## Downstream gates

Bandwidth/kernel-reference and local-linear ABF tests are mechanism/rescue studies
only after a promising but sub-threshold signal. Temperature/asymmetry mapping,
the beta=8 entropic-bottleneck adversarial test, WCA, and molecular systems are not
authorized by a merely null pilot; they require the preceding frozen-schedule gate.

All results, including adverse or null cells, remain in the schedule map. Figures are
generated from saved CSV data with median and IQR curves and are exported as matching
PNG and PDF files.
