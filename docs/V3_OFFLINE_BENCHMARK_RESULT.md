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

- **p_max = 0.02 contributes zero complete cells.** The lowest registered dose
  produces no measurable contraction at this floor and is absent from every
  C_gene table. It is retained in the KL-drop and variance tables, which have no
  exclusion.
- The complete-case set is dominated by p_max = 0.10 (exclusion 0.10–0.44) with
  a minority from p_max = 0.05.

## Result 1 — C_gene: FT wins, essentially unanimously

Median genealogy cost per nat, complete cases, lower is better:

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
bd_paired in **109/720**. A plausible mechanism, offered as a hypothesis and not
a finding: systematic resampling draws a *single* uniform offset per step, so
the whole comb shifts together and different seeds move the cloud coherently,
whereas BD draws K near-independent Bernoullis. Lower offspring-count variance
and higher run-to-run cloud variance are not in conflict.

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
but loses to bd_paired 489/720. FT's C_gene win comes from the denominator of
that ratio — it spends far less ancestry (ESS_anc/K ≈ 0.975 vs 0.949) and makes
roughly half the replacements (6.6 vs 13.5 at p_max = 0.10) — not from
contracting harder.

## Result 5 — the bandwidth-free companion sees no difference

Median W1 to q after the step, K = 1024: bd_paired 0.34314, bd_standard 0.34283,
ft 0.34375 — differences in the fourth decimal, against a KL-drop signal in the
third. The companion metric exists precisely to catch a KDE-mediated artifact,
and its verdict is informative rather than null:

> All three discretizations reach essentially the same marginal. They differ in
> **what they charge in ancestry to get there.**

That is what makes C_gene the right primary statistic for Q-D, and it is the
cleanest statement this benchmark supports.

## P6 verdict

| clause | verdict |
|---|---|
| FT best on genealogy-per-nat | **confirmed** (203/203 paired vs bd_standard) |
| bd_paired between the two | **confirmed** (203/203 and 196/203) |
| FT dominates on variance | **refuted** — FT is worst (97/720) |
| gaps shrink with K | **not observed** — flat 2.1–2.3× over 16× in K |

Partially confirmed. Recorded as such; P6 is not amended after the fact.

## What this does NOT authorize

Q-D is a prediction, not a selection stage. **The online arms are frozen and are
not changed by this result**, and would not have been had FT lost. The
preregistration says the online P-BD/P-FT pair must not be read as a dose-matched
contrast; this benchmark is where the dose-matched question is answered, and it
answers it about discretizations, not about whether FR helps ABF.
