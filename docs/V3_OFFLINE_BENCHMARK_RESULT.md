# v3 offline discretization benchmark (Q-D / prediction P6)

Frozen protocol: `docs/V3_PREREGISTRATION.md` Appendix A.1–A.2, Amendment 3.
Executed 2026-08-25 on the 48 hashed K=1024 clouds (all 48 hashes re-verified).
2160 rows = 48 clouds × 5 subsample sizes × 3 doses × 3 operators × 100 FR seeds.

## Population accounting (read this before the tables)

The registered C_gene exclusion rule (KL drop < 0.01 nat) removed **1471/2160
cells**, and it did **not** remove them evenly across operators — bd_paired
0.654, ft 0.676, bd_standard 0.713 — with the three operators disagreeing about
exclusion in **51/720 cells (7.1 %)**. Pooling C_gene over each operator's own
surviving cells would therefore compare three different populations, which is
this project's standing ratio/population defect class.

All C_gene numbers below are **complete cases only**: the 203/720 cells where
all three operators cleared the floor. Two consequences are stated rather than
buried:

- **p_max = 0.02 contributes zero complete cells.** The correct reading is *at
  the smallest registered dose, one step produces no measurable contraction at
  the 0.01-nat resolution* — not "the methods fail at p_max = 0.02". It is
  absent from every C_gene table and retained in the KL-drop, damage,
  replacement and W1 tables, none of which involve a ratio and none of which
  apply the floor.
- The complete-case set is dominated by p_max = 0.10 (exclusion 0.10–0.44) with
  a minority from p_max = 0.05.

## Result 1 — C_gene: FT wins, essentially unanimously

Median genealogy cost per nat, **among the 203/720 complete cells in which all
three operators produce preregistered measurable contraction**, lower is better.
That is a selected regime and every C_gene figure below must be read with the
qualifier attached:

| K | bd_paired | bd_standard | **ft** |
|---|---|---|---|
| 64 | 3.100 | 6.500 | **2.915** |
| 128 | 3.033 | 6.214 | **2.832** |
| 256 | 3.184 | 6.603 | **2.927** |
| 512 | 3.147 | 6.200 | **2.887** |
| 1024 | 3.040 | 5.723 | **2.702** |

Paired within-cell over the 203 complete cells: FT beats bd_standard **203/203**,
FT beats bd_paired **196/203**, bd_paired beats bd_standard **203/203**. The
registered ordering FT < bd_paired < bd_standard holds.

## Result 2 — P6's variance clause is REFUTED

P6 predicted FT would dominate BD on variance. It does not; it is the **worst**
of the three. Median across-seed sd of post-step KL:

| dose | bd_paired | bd_standard | ft |
|---|---|---|---|
| 0.02 | 0.00304 | **0.00281** | 0.00562 |
| 0.05 | 0.00430 | **0.00418** | 0.00682 |
| 0.10 | 0.00545 | **0.00527** | 0.00600 |

Paired: FT has lower sd than bd_standard in only **97/720** cells and than
bd_paired in **109/720**.

The safe conclusion is exactly this and no more:

> FT is genealogy-efficient but **not** low-variance under this benchmark's
> measured stochastic output.

A single-random-offset story — systematic resampling draws one uniform per step,
so the whole comb shifts together and seeds move the cloud coherently, whereas
BD draws K near-independent Bernoullis — is a **post-hoc hypothesis, not a
finding**. Systematic resampling has good offspring-count properties but a
radically different dependence structure from independent event decisions;
whether that produces the observed coherence is untested here and is not needed
for any decision in this campaign.

## Result 3 — P6's "gaps shrink with K" is NOT OBSERVED

The bd_standard/FT C_gene ratio is flat: 2.230, 2.194, 2.256, 2.147, 2.118 for
K = 64…1024. A ~2.1× advantage that is stable over a 16× range of K, not a
finite-K artifact that anneals away.

## Result 4 — raw contraction favors bd_paired, not FT

Median KL drop (all cells, no exclusion), higher is better:

| K | **bd_paired** | bd_standard | ft |
|---|---|---|---|
| 256 | **0.00803** | 0.00724 | 0.00720 |
| 1024 | **0.00792** | 0.00723 | 0.00717 |

Paired: bd_paired beats bd_standard **637/720**; FT beats bd_standard 521/720
but loses to bd_paired 489/720.

**Which term of C_gene drives FT's win (corrected).** C_gene =
(1 − ESS_anc⁺/K) / (KL⁻ − KL⁺): the numerator is genealogical *damage*, the
denominator is *contraction*. On the complete cases:

| operator | numerator (damage) | denominator (contraction) | C_gene |
|---|---|---|---|
| ft | **0.04714** | 0.01463 | **2.878** |
| bd_paired | 0.05331 | **0.01572** | 3.135 |
| bd_standard | 0.09555 | 0.01405 | 6.227 |

FT's advantage is driven **entirely by the numerator — genealogy preservation —
not by contracting harder.** Against bd_standard it does 2.03× less damage at
essentially equal contraction (1.04×). Against bd_paired the point is sharper:
bd_paired contracts **1.07× harder**, which enlarges FT's ratio and works
*against* it, and FT still wins because it does 13 % less damage.

An earlier version of this report said the win came from the denominator. That
was backwards twice over: it is the numerator, and the denominator is the term
working against FT in the bd_paired comparison.

## Result 5 — the bandwidth-free companion sees no difference

Median W1 to q after the step, K = 1024: bd_paired 0.34314, bd_standard 0.34283,
ft 0.34375 — differences in the fourth decimal, against a KL-drop signal in the
third. Stated conservatively:

> The bandwidth-free W1 companion shows **no practically resolved separation**
> among the three discretizations at one-step scale. The large C_gene
> differences are therefore not accompanied by comparably large differences in
> this global marginal-distance metric.

This is deliberately weaker than "all three reach the same marginal", which W1
does not establish: W1 measures transport distance, so a one-step move can
change local log-density enough to move a KDE-based KL while barely moving W1.
What the pair of metrics does support is that the operators' *separation lives
in the ancestry they spend*, which is why C_gene is the right primary statistic
for Q-D.

## P6 verdict

| clause | verdict |
|---|---|
| FT best on C_gene | **strongly confirmed** (203/203 paired vs bd_standard) |
| bd_paired intermediate | **strongly confirmed** (196/203 vs FT; 203/203 vs bd_standard) |
| bd_standard worst | **strongly confirmed** (0/203 wins) |
| FT lower variance | **refuted** — FT is worst (97/720) |
| gaps shrink with K | **refuted** — flat 2.12–2.26× over a 16× range in K |
| all three realize the same flow | **supported** — matched-dose contraction and W1 agree closely |

The right summary is not "partially confirmed" but: **the registered efficiency
ordering is strongly confirmed, while two secondary mechanistic predictions are
refuted.** P6 is not amended after the fact.

The refuted K-clause is itself a stronger result than the prediction it
replaces: over the tested 16× range FT's genealogy-efficiency advantage looks
*structural* rather than a small-population artifact. No claim is made about
K → ∞.

## What this does NOT authorize

Q-D is a prediction, not a selection stage. **The online arms are frozen and are
not changed by this result**, and would not have been had FT lost. The
preregistration says the online P-BD/P-FT pair must not be read as a dose-matched
contrast; this benchmark is where the dose-matched question is answered, and it
answers it about discretizations, not about whether FR helps ABF.
