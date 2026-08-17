# Convergence atlas — L2(F) against time, every study on one convention

Built 2026-08-17 from stored artifacts only. **No dynamics were re-run.**

```
python scripts/extract_convergence_atlas.py        # -> results/convergence_atlas/atlas.{npz,json}
python scripts/analyze_convergence_atlas.py        # -> speedup.json, scoreboard.csv
python scripts/audit_convergence_controls.py       # -> controls.json
python scripts/make_convergence_atlas_figures.py   # -> report/figures/fig_conv_0{1..4}.png
python scripts/make_convergence_report_assets.py   # -> report/tables/convergence_numbers.tex
cd report && tectonic main.tex                     # -> main.pdf, 133 pp
```

In the report this is §20, `sections/22_convergence_speed.tex`. Every number it states comes
from a `\Cv...` macro generated from the artifacts above; none is typed by hand.

## What the extraction is worth

Every core already scores `e_F(t) = || F̂_t − F_ref − c_t ||` online and stores it per seed, so
the atlas is a re-reading of the campaign, not a re-analysis of it. Two checks fix that:

* **The convention is one convention.** Each core removes the additive constant on its own
  evaluation window and reports an interior-window RMS. Verified equivalent up to endpoint
  weighting across `wca_abffr_core`, `gateway_core`, `eb_abffr_core`, `alkanes.metrics`,
  `alanine.metrics_ala`.
* **The extraction reproduces the shipped verdicts exactly.** Re-deriving the primary endpoint
  from the atlas gives mFR vs ABF = **−17.97 %** (WCA, stored −17.965), sham vs ABF = **−3.19 %**
  (stored −3.186), five-arm mFR **−17.62 %** / count-balancing **−18.23 %**, gateway **−12.48 %**.
  If the atlas had silently picked up a different window or reference, these would not land.

### The WCA reference trap, avoided

`results/wca_production/` was scored against the **superseded** reference and keeps no profile
time series, so its curves can be neither reused nor rescored — the two stored references differ
by up to **0.061 kT** inside the evaluation window. The flagship panels therefore read
`wca_caseix_hp/` and `wca_five_arm/`, whose runs store `pmf_t` and carry the corrected
high-precision reference in the file. Verified: stored `l2_f_t` reproduces **exactly** (max abs
diff 0.0 over 160 runs) from `pmf_t` + `reference_free_energy` on the [−0.1, 1.1] window.

Note for anyone reusing `SimConfig`: its dataclass default eval window is `[0.0, 1.0]`, but every
config file overrides it to `[-0.1, 1.1]`. Constructing a `SimConfig` without the YAML silently
scores on a different window — that is what the first reproduction attempt did, and it moved the
curve by up to 8 %.

## The scoreboard

Primary endpoint = time-integrated L2(F), paired by seed, median relative change vs ABF with a
95 % bootstrap CI. Regime assigned by the CI, not by eye: FASTER needs median ≤ −5 % and CI upper
< 0.

| verdict | study / cell | integrated | seeds won | final-time |
|---|---|---|---|---|
| **mFR faster** | entropic bottleneck, β=8 | **−40.5 %** | 10/10 | −51.7 % |
| **mFR faster** | entropic bottleneck, β=12 | **−33.8 %** | 10/10 | −32.1 % |
| **mFR faster** | WCA dimer, starved cell | **−18.0 %** | 16/16 | −45.3 % |
| **mFR faster** | entropic gateway (anchor) | **−12.5 %** | 31/32 | **+11.3 %** |
| no difference | 2-D metastability toy | −11.5 % | 3/5 | −7.7 % |
| no difference | entropic bottleneck, β=2 | −5.6 % | 7/10 | +14.5 % |
| no difference | alanine dipeptide (oracle) | −0.0 % | 3/4 | −0.0 % |
| no difference | butane, φ1 | +0.1 % | 2/16 | +0.1 % |
| no difference | pentane, φ1 | +0.4 % | 3/16 | +0.4 % |
| no difference | pentane, R15 | +0.7 % | 0/8 | +2.0 % |
| no difference | entropic bottleneck, β=4 | +3.2 % | 3/10 | +27.4 % |

**4 faster, 7 no difference, 0 significantly slower.** "Most experiments favour mFR" is not what
the plots show, and no way of drawing them will make it so. What they do show is a clean
conditional: mFR accelerates exactly where ABF develops a persistent population deficit, and does
nothing anywhere else.

Studies with **no mFR arm at all**, so no panel exists and none should be manufactured: methane,
NaCl, valine (all ABF-sufficient at a preregistered gate), deca-alanine (ABF baseline retracted),
C60 (suspended, zero data).

## Three things the panels say that the summary numbers do not

**1. The gateway's advantage is early, and it reverses.** mFR wins the integrated endpoint 31/32
— and loses the final-time endpoint by **+11.3 % on 32/32 seeds**. Its mFR/ABF error ratio bottoms
out at **0.37 at t = 3.0**, crosses back above 1.0 at **t = 16.6**, and ends at **1.11**. WCA is the
opposite: the ratio falls all the way to **0.54 at t = 220 of 240** and never returns above 1.
 So the two positive systems disagree about whether the
acceleration is a faster decay *rate* or a temporary head start, and only WCA supports the
stronger reading. `fig_conv_02_mechanism.png` panel (c) is that contrast.

**2. The acceleration is not Fisher-Rao-specific.** In the five-arm test, count-balancing — a
simpler non-FR reallocation rule — gives **−18.23 %** against mFR's **−17.62 %**, and the panel
shows the two curves lying on top of each other for the whole run. `book_laplacian` does nothing
(+0.7 %). The repo already sets `novelty_claim_licensed=false`; the convergence framing does not
change that.

**3. The 2-D toy is not evidence.** Each FR arm was tuned over **36 configurations**; ABF was run
at **one**. Its −11.5 % has a CI of [−22.5, +18.9] on 5 seeds. `fig_conv_04_toy_selection.png`
draws all 36 curves: the cloud straddles ABF, and the selected config sits at its favourable edge.

## Time to a prescribed accuracy

τ_ε = first time e_F stays ≤ ε for a persistence window Δ = 0.2 T; S_ε = τ_ε(ABF)/τ_ε(mFR).
Thresholds are fixed by a rule declared before looking at any arm: ε = f·e_0 for f ∈ {1/2, 1/4,
1/8}, where e_0 is the error at t = 0 — a point at which every arm carries an identically zero
bias, so the rule is a property of the system and cannot favour a method. A fourth threshold is
ABF's own final error, which asks the practitioner's question directly.

Where the effect is real it is large: to reach the accuracy ABF *finishes* with, mFR needs
**1/12.5** of the budget at β=8, **1/14.3** at β=12, and **1/3.7** on the WCA starved cell.
Where there is no effect, the same threshold is simply never reached — those cases are drawn as
✕ at the axis edge rather than dropped, because dropping them converts "never got there" into
"not in the average".

## The two controls (`scripts/audit_convergence_controls.py` → `controls.json`)

Both are re-readings of stored artifacts; no dynamics re-run.

**Does the advantage survive freezing the bias?** The atlas evaluates mFR while its population is
deliberately non-Boltzmann, so part of the online gain could be a property of the evaluation.

| system | online endpoint | frozen-bias endpoint |
|---|---|---|
| WCA starved | −18.0 % integrated, **−45.3 % final** (16/16) | **−4.0 %** (7/8), at the corrected reference |
| gateway | −12.5 % integrated, **+11.3 % final** (0/32) | **−10.2 %** (24/32) |

WCA keeps the sign and loses an order of magnitude. **The gateway's two endpoints disagree in
sign** — both claim to measure the accuracy at the end of the run. Caveats before leaning on
either: the WCA frozen tree is N=512 against the flagship's N=1024, and the frozen reconstruction
carries its own error floor of 0.21 against the online final error of 0.09, so it is a blunt
instrument. **This has to be settled before the convergence claim is written, not after.**

The WCA frozen-bias tree was scored against the superseded reference; it stores `F_recon`, so the
rescore above is free. Its raw directory also holds **two cells with overlapping seed numbers** —
keying by (arm, seed) without the cell silently merges them and produces a baseline belonging to
neither. That happened on the first pass here.

**Does the Fisher–Rao target matter?** Three independent tests now say the mechanism is
reallocation, not geometry:

* WCA five-arm: count-balancing **−18.23 %** vs mFR **−17.62 %** — a tie.
* Entropic bottleneck β=8: `fr_uniform` (reallocate toward uniform occupancy, no FR target at
  all) **−39.4 %** vs `fr_estimated` **−42.6 %** — captures ~92 % of the gain. And `fr_oracle`,
  which has the *exact* target, is **+10.4 % worse** on 4/5 seeds. If the geometry were the
  mechanism, the oracle would be the ceiling; here it is the worst arm.
* Gateway: oracle −10.0 % vs estimated −12.5 % — again the oracle is not the ceiling.
  (WCA starved is the one case where the oracle does lead, −19.5 % vs −18.0 %.)

**The one result that looks like a genuine mechanism.** The EB FR-rate ladder at β=8 is
**monotone over a 50× range** — −6.3 %, −8.8 %, −21.7 %, −30.3 %, −43.7 %, −53.3 %, −62.2 % at
γ = 1, 3, 5, 10, 15, 25, 50, with 10/10 seeds from γ=10 up. A clean dose–response is much harder
to explain away than any single contrast, and the headline γ=15 is *not* the sweep maximum, so
this cell is not a best-of-7 selection. Note this cuts against the alkanes, where raising the FR
rate harms — the boundary between the two behaviours is not characterised.

## Caveats that belong in any write-up

* **Equal compute is automatic here, not assumed.** Birth–death reallocates existing replicas; N
  and n_steps are identical across arms in every panel, so no arm buys its curve with extra force
  evaluations. (The separate `wca_equal_compute` study varies N×n_steps at fixed budget.)
* **Vertical scales are not comparable between panels** — different systems, CVs, windows. Nothing
  in the atlas pools across panels.
* **Pentane R15 has not converged** by the end of its run in either arm; its panel shows a null on
  a still-descending curve, which is a weaker null than butane's plateau.
* **Alanine's arms are oracle mFR at three FR rates.** The mechanism fires (up to 66 818 events)
  and the error does not move — that is the atomistic neutrality control, not a missing practical
  arm.
