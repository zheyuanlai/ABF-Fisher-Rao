# Experiment I: one free-energy landscape, four microscopic realisations

*2026-10-10. Preregistration: `SCIENTIFIC_PLAN.md` (44735ea) and Amendment 1 (bf56da4). Validation:
`EXP1_TIMESTEP_VALIDATION.md` (PASS, h = 2.5 × 10⁻⁵ common).*

**Data and figures**
* Raw runs: `results/mechanism/matched_free_energy/<variant>/N<N>/` (gitignored npz).
* Per-cell analysis: `.../analysis/{summary.json, tables.md, mech_diagnostics.json}`.
* Preregistered contrasts: `results/mechanism/synthesis/cross_cell.{json,md}`.
* Figures:
  * `figures/mechanism/matched_free_energy/<variant>/N<N>/`: A–F plus mech_\*;
  * `figures/mechanism/matched_free_energy/<variant>/synthesis/` (S1–S8);
  * `figures/mechanism/synthesis/mech_s{1,2,3,3b,4,5c,6}*`.

## 1. Design (as frozen)

**Models.** The same F\*(x): a barrier of 11.47 kT, 8 energetic + 3.47 entropic. It is realised four ways:

| variant | entropic part of the barrier | Var(f \| x) peak | transverse width | measured conditional relaxation of f at the flank (x = −0.21) |
|---|---|---|---|---|
| α = 1 (original gateway) | 3.47 kT (30 %) | 2.06 | narrows 32× at the gate | τ_f = 0.013 |
| α = 0.5 | 1.73 kT (15 %) | 0.52 | narrows 5.7× | 0.088 |
| α = 0 | 0 | 0 (deterministic force) | constant | — (f independent of y) |
| shifted fibre (κ = 1) | 0 | 2.06 (matched to α = 1) | constant, centre moves 1.2 (4.9 σ_y) across the gate | 0.38 (28× α = 1) |

**Runs and statistics**
* N ∈ {2048, 512, 128} at the same force budget B = 3.2768 × 10⁹ (T_N = 40, 160, 640).
* 32 seeds (8100–8131), paired across arms and coupled across variants (same initial normals and noise).
* Frozen gateway ABF and FR constants.
* **1 152 simulated runs, 0 failed.**
* The α = 1 cells were re-run with the new engine and are bitwise identical to the equal-budget files (192/192).

## 2. Results

G = (FR − ABF)/ABF, paired: the median over 32 seeds with a 95 % seed-bootstrap CI and FR wins. Ī_F is the integrated
free-energy error (reduced energy units); final = at the end of the budget.

| variant | N | ABF Ī_F | FR Ī_F | **G(Ī_F)** | G(final e_F) | G(Ī_F′_stat) | G(Ī_TV_half), marginal |
|---|---|---|---|---|---|---|---|
| α = 1 | 2048 | 0.0227 | 0.0156 | **−31.4 % [−34.7, −27.7]** 31/32 | −61.4 % [−65.1, −55.2] | −29.9 % [−34.6, −26.3] | −71.5 % |
| | 512 | 0.0083 | 0.0050 | **−41.0 % [−49.6, −33.4]** 32/32 | −50.5 % [−68.0, −36.4] | −32.6 % [−38.0, −28.7] | −72.7 % |
| | 128 | 0.0027 | 0.0018 | **−30.3 % [−37.4, −17.6]** 24/32 | +15.3 % [−25.3, +46.3] | −20.0 % [−29.5, −13.1] | −70.9 % |
| α = 0.5 | 2048 | 0.0189 | 0.0136 | **−26.2 % [−34.8, −21.7]** 32/32 | −63.8 % [−71.5, −56.6] | −25.8 % [−29.4, −19.8] | −72.2 % |
| | 512 | 0.0064 | 0.0039 | **−36.3 % [−43.2, −28.4]** 32/32 | −55.9 % [−71.9, −36.1] | −33.2 % [−40.3, −24.6] | −74.0 % |
| | 128 | 0.0019 | 0.0015 | **−21.4 % [−37.7, −15.2]** 26/32 | +4.0 % [−17.7, +142] | −17.5 % [−29.7, −8.7] | −72.6 % |
| α = 0 | 2048 | 0.0129 | 0.0099 | **−24.0 % [−29.4, −15.2]** 31/32 | −57.7 % [−61.1, −54.7] | −27.0 % [−32.7, −21.0] | −73.4 % |
| | 512 | 0.0040 | 0.0026 | **−31.7 % [−37.6, −20.2]** 30/32 | −36.6 % [−52.6, −27.6] | −36.9 % [−39.8, −21.7] | −76.4 % |
| | 128 | 0.00066 | 0.00063 | **−7.2 % [−19.4, −3.2]** 23/32 | −30.1 % [−48.8, −10.8] | −9.5 % [−38.9, −1.7] | −71.4 % |
| shifted fibre | 2048 | 0.156 | 0.125 | **−21.0 % [−22.0, −17.9]** 32/32 | −19.0 % [−21.5, −15.6] | −6.1 % [−8.3, −5.3] | −64.7 % |
| | 512 | 0.096 | 0.098 | **+2.0 % [−4.6, +7.8]** 15/32 | **+65.7 % [+55.5, +75.6]** 0/32 | +21.3 % [+11.4, +27.8] | −87.7 % |
| | 128 | 0.037 | 0.073 | **+105 % [+61, +128]** 0/32 | **+831 % [+585, +1036]** 0/32 | +131 % [+105, +156] | −91.6 % |

**Time to the mid threshold (e_F ≤ 0.0055), budget fraction u**
* α = 1: ABF 0.98 (14/32 censored) vs FR 0.32 at N = 2048 (32/0 wins); 0.27 vs 0.085 at 512; 0.08 vs 0.04 at 128.
* α = 0.5 and α = 0: FR is earlier at every N, by smaller margins at 128.
* Shifted fibre: **neither arm reaches it** at any N (censored in every seed but 11 ABF seeds at N = 128).

**Absolute baselines differ strongly although F\* is identical.**
* ABF's Ī_F is 0.0129 for α = 0 against 0.0227 for α = 1 at N = 2048. With a deterministic force, ABF's bin means are
  exact once visited; the final e_F is 2 × 10⁻⁵.
* For the shifted fibre ABF's Ī_F is 0.156, 7× worse than α = 1, and gate transitions are 2–4× rarer (508–1 316
  against 2 168–2 264 per run).

### 2.1 Preregistered contrasts (Holm over 12)

exp(Δ) is the ratio of the per-seed log-ratios; > 1 means the first cell's FR gain is smaller.

| contrast | N = 2048 | N = 512 | N = 128 |
|---|---|---|---|
| C1 α = 0 vs α = 1 | 1.12 [1.05, 1.22], Holm p 0.031: **α = 1 gains more** | 1.20 [1.08, 1.41], Holm p 0.007: **α = 1 gains more** | 1.17 [0.98, 1.65], inconclusive |
| C2 shifted vs α = 1 | 1.19 [1.11, 1.27], p 0.002: **α = 1 gains more** | 1.80 [1.43, 1.97], p 0.002 | 2.87 [2.15, 3.37], p 0.002 |
| C3 shifted vs α = 0 | 1.05 [0.97, 1.14], inconclusive | 1.52 [1.25, 1.65], p 0.002: **α = 0 gains more** | 2.30 [1.96, 3.16], p 0.002 |
| C4 α = 0.5 vs α = 1 | 1.05, inconclusive | 0.93, inconclusive | 1.06, inconclusive |

**Secondary (Ī_TV_half), the marginal.**
* The marginal gain is about the same across the α family (−71 to −76 %). C1 and C4 sit within the [0.90, 1.11]
  equivalence band, or slightly favour α = 0 at N = 512.
* The shifted fibre's marginal gain is the **largest** at N = 512 / 128 (−88 / −92 %; C2 ratio 0.45 / 0.30).

**Secondary (Ī_F′_stat).** C1 and C4 are equivalent or inconclusive. C2 and C3 show that the shifted fibre gains less
(ratios 1.3–2.9, all p 0.002).

## 3. Mechanistic diagnostics

**Recently cloned deposits** (D2, `mech_s5c_d2_matched_free_energy`; preregistered §4/§7 quantity, seed-bootstrap CIs):
* About 1.3–4.6 % of gate deposits come from walkers cloned in the last 0.1 t.u., similar in every variant.
* Their force samples deviate from the bin-averaged F\*′ by almost exactly what ABF's own deposits do in the α family:
  the RMS excess over ABF is 0.006 [0.002, 0.009] for α = 1 at N = 2048, and ≈ 0 elsewhere and for α = 0.
* In the shifted fibre the excess is **+0.17 [0.16, 0.18] (N 2048), +0.18 (512), +0.15 (128)**.
* The signed deviation is concentrated on the entry side of the gate (x < 0: +0.23 against α = 1 at N = 128).
* Cloned walkers there carry y values lagging behind the moving fibre centre m(x).

**Discovery is not what separates the variants.** First arrival in the right well (ABF, u) is 0.035 / 0.010 / 0.0034
at N 2048 / 512 / 128 for every variant, including the shifted fibre. FR shifts it by −17 % to +9 % (α = 1: 0.035 → 0.029; shifted fibre N = 512: 0.0093 → 0.0102).

**Convergence curves** (`mech_s2_convergence_matched_free_energy_N*_{u,t}`):
* In the α family FR's e_F curve lies below ABF's after establishment.
* At α = 0 both arms collapse to 10⁻⁵–10⁻⁶ once established, and FR merely moves that collapse earlier.
* In the shifted fibre both stall near 0.05–0.1. ABF keeps decreasing late, while FR flattens, so FR ends worse at
  N ≤ 512.

## 4. Interpretation, against the preregistered map (§7)

* **H1 (known): FR accelerates ABF on the original gateway.** Reproduced bitwise.
* **H2: does the gain change as entropy moves into energy?** Yes, but modestly, and only on the free energy.
  * C1 shows a significantly smaller relative gain at α = 0 at N 2048 and 512 (ratio 1.12–1.20).
  * The α = 0 gain remains large: −24 % / −32 % at N 2048 / 512, 31/32 and 30/32 seeds.
  * **An entropic bottleneck is not necessary for FR to help.**
  * The marginal gain does not depend on the entropic share at all.
* **H3: is the α-dependence explained by conditional-force variance?** Not by variance as such.
  * The variance-matched shifted fibre does not behave like α = 1: it is the worst case for FR, harmful at
    N ≤ 512.
  * The smaller relative gain at α = 0 coincides with a much better ABF baseline (deterministic force, exact bin means
    once visited), so FR has less finite-time error to remove.
  * This is an observed association; the experiment does not isolate it causally.
* **H4: what the shifted-fibre control separates.**
  * Under the frozen branch "C1 smaller gain at α = 0 → read C2", C2 shows the shifted fibre gains far less than
    α = 1, and C3 shows it gains even less than α = 0.
  * Neither branch of the map ("like α = 1" → variance; "like α = 0" → narrowing) applies.
  * What distinguishes the shifted fibre is its slow, curved conditional relaxation: τ_f at the flank is 28× α = 1's,
    and the fibre centre moves 4.9 σ across the gate.
  * D2 shows that FR's recent clones there deposit strongly biased forces.
  * So **matched conditional-force variance with slow, curved conditional relaxation makes FR harmful**, which is
    Experiment II's question. As §1.1 of the plan warned, this control does not isolate fibre narrowing from
    relaxation.

**Negative results preserved**
* FR is harmful in the shifted fibre at N = 512 (final +66 %, 0/32) and N = 128 (Ī_F +105 %, final +831 %, 0/32),
  while it gives the largest marginal improvement there.
* At α = 0, N = 128 its gain is small (−7 %).
