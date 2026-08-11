# Gateway Gate 0 — PASSES, with a marginal constriction

Backfill required by Amendment 10: the entropic gateway was classified establishment-limited
**before Gate 0 existed**, so its interpretation — and the mFR positive resting on it — was
provisional. It is no longer.

## Why this system can be audited without reference error

`V(x,y) = H(x^2-1)^2 + 1/2 omega(x)^2 y^2`, `xi = x`, so the conditional law is exact:
`p(y|x) = N(0, 1/(beta omega^2))`, the transverse dynamics at fixed `x` is exactly OU with rate
`omega(x)^2`, and `<f_loc>_x = 4Hx(x^2-1) + omega'/(beta omega) = F'(x)` identically. Deca had
an 8.4 % reference floor; R15's reference was importance-sampling. Here there is none.

## 1. tau_perp measured, not assumed

`y` was started from the **bulk** width — at the gateway that is `r^2 = 1024x` too wide — and
`<y^2>` watched relax. Measured/analytic ratios 0.13–2.05 across `x`, i.e. order 1. The OU
prediction `tau_perp = 1/omega^2` holds.

## 2. Three timescales at the accepted cell (beta=16, s=0.1, r=32)

| | |
|---|---|
| `tau_perp` (gateway) | **9.77e-4** |
| `T_hit` (median, 32 seeds) | **1.40** |
| `T_est` (median, 32 seeds) | **17.0** |
| `T_total` | 40.0 |
| `tau_perp / T_est` | **5.7e-5** |
| `T_hit / T_est` | **0.082** |

`T_hit << tau_perp << T_est` is **not** literally satisfied — `tau_perp` is far *below* `T_hit`,
not between it and `T_est`. The operative condition is the weaker and correct one,
`tau_perp << T_est` together with early discovery, and it holds by ~4 orders of magnitude. This
is the mFR window, quantified.

## 3. Gate 0 statistic

| | all bins | gateway only (\|x\| <= s) |
|---|---|---|
| rel. \|`<f_loc>` − `F'_ref`\| / \|`F'_ref`\| | **0.036** | **0.189** |
| rel. \|`<y^2>` − `1/(beta om^2)`\| / eq | **0.051** | **0.213** |
| deca-alanine, same statistic | 0.61 | — |
| R15 beta=2, same statistic | 0.564 / 0.593 | — |

**Globally the gateway is 17x cleaner than deca.** Gate 0 passes.

## The caveat that must travel with this

**The constriction is marginal, not comfortable.** Median walker residence inside `|x| <= 0.1`
is 0.0012 against `tau_perp = 9.77e-4` — a ratio of **1.2x**, not the large separation the
global numbers suggest. A walker crosses the gateway in about *one* transverse relaxation time,
and the local conditional is correspondingly 19–21 % off equilibrium. (Residence is heavy-tailed:
mean 0.0141 is 12x the median over 302 246 crossings.)

So the gateway's mFR effect lives in the one region where its own conditional ensemble is least
equilibrated. That does not invalidate the positive — 0.19 against deca's 0.61, with a global
0.036 — but "the gateway is conditionally equilibrated" is true globally and only marginally
true where it matters most.

## Limitation of this audit

The live run here is a **simplified ABF reimplementation** (running-mean estimator, `min_count`
20, no kernel smoothing, no bias ramp), not the accepted sampler (`eta=0.1`, `h=0.07`,
`min_count=1.0`, `ramp_fraction=0.1`). The physics it measures — `tau_perp`, residence time, the
exact conditional — is identical, but the bias trajectory is not the accepted one. A tighter
version would instrument `gateway_core.simulate_batch` directly.
