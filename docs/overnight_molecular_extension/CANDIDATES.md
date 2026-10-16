# Candidate assessment — overnight molecular extension

*2026-10-16, 17:10–17:58 UTC. Sources:*
* *`review/CAND_A_LITERATURE.md` (Candidate A);*
* *`review/CANDIDATES_B_E_SCAN.md` (Candidates B–E);*
* *`docs/eri_campaign/review/LIT_*` (force-field cross-checks reused).*

*All literature agents worked read-only. No candidate has been simulated before the decisions below.*

## Summary table

| | system (reference) | paper retrievable? | model recoverable? | entropic evidence | reaction coordinate | hidden-coordinate risk | cost vs ERI | similar model already tested? | verdict |
|---|---|---|---|---|---|---|---|---|---|
| **A** | ethane in Si-ITE / Si-IHW (Combariza, Sastre & Corma, JPCC 115, 875 (2011)) | **no**: ACS paywalled; RiuNet copy closed "forever"; SI 403 | **no**: fully flexible framework (BKS, charged, Ewald) + all-atom Oie 1981 guests (closed) | not stated (abstract) | ITE: linear along b, one window per period; IHW: x, period a/2 | ITE 1-D channels; IHW 2-D, 12-ring pairs | – | LTA (8-ring, ethane) | **NOT REPRODUCIBLE as published** |
| **A′** | **ethane in rigid Si-ITE (ITQ-3 structure) with the Dubbeldam et al. 2004 united-atom force field**: a labelled, independently documented model, not a reproduction of A | structure: IZA type material + RASPA2 `ITQ-3.cif`; force field: JPCB 108, 12301 Table 3 + PRL 93, 088302 Table I (both open, cross-checked) | **yes, every parameter** | to be measured (Gate 1 decomposition) | X = crystal b, linear, period b; walkers kept in one channel class | none known; the [001] / a-rings are closed in the rigid frame (to be verified) | ~0.5–0.6× ERI per step | LTA used the same O–CH3 pair (93 K / 3.48 Å) but with a soft bond and a 10 Å cut | **SELECTED (first)** |
| B | CO2 / CH4 in DDR (Jee & Sholl, JACS 131, 7896 (2009)) | no (paywalled, SI 403) | no (new force field, parameters unknown) | abstract only | through the 8-ring | CO2 orientation, electrostatics | CO2 needs Ewald + rigid-body BD | – | NOT REPRODUCIBLE. A labelled CH4/rigid-DDR/Dubbeldam substitute is possible (reserve) |
| C | CH4 in silicalite (López et al., PRE 72, 061111 (2005)) | no | no | site-preference switch, not a window barrier | channel / intersection, path-like | memory effects, multiple pathways | ~0.4× | methane closed elsewhere | NOT REPRODUCIBLE; a substitute is expected neutral |
| D | 2 CH4 in a water-filled cylindrical pore (Vaitheeswaran et al., JCP 130, 094502 (2009)) | yes (arXiv + group site) | mostly (wall-site ambiguity; hard walls → smooth wall; rigid TIP3P → rigid-body BD) | barrier ≤ ~3.2 kT, water-structure driven | CH4–CH4 distance (geometric term 2kT/r) | solvent (water) relaxation | 0.7–120× | CH4 / NaCl in water: both ABF-sufficient | FEASIBLE-WITH-CAVEATS, low priority (expected ABF-sufficient) |
| E | ethene / propene in flexible H-SAPO-34 (Cnudde et al., JACS 142, 6007 (2020)) | yes (OA + SI) | no (guest–framework from Oie 1981 / Combariza 2011, closed; acid-site positions; cell ambiguity) | strongest: 73 % entropic (propene, 450 K) | moving 8-ring plane (nonlinear) | flexible framework, acid sites | ≥ 10³× per unit time | CHA Stage 6 (rigid) neutral; ZIF-8 flexible neutral | NOT REPRODUCIBLE on the budget |

## Decision (frozen before any ITE computation)

1. **A is not reproducible as published.** The rule "do not silently substitute a rigid model and call it a
   reproduction" applies. It is recorded as a reproducibility obstacle.
2. **A′ is implemented as a separate, labelled model of the same physical system.** Every parameter comes from
   independently published, open sources:
   * structure: ITQ-3 experimental coordinates;
   * force field: Dubbeldam 2004.

   No parameter is invented, so this is not rescuing an irreproducible model. Every output names it "ethane in rigid
   Si-ITE (ITQ-3), Dubbeldam-2004 UA force field". It is never called Combariza 2011.
3. **Choices fixed now:**

   | choice | value | reason |
   |---|---|---|
   | temperature | **300 K** | room temperature; Dubbeldam 2004 was fitted on room-temperature isotherms; the repo's LTA precedent. Fixed before any barrier is computed |
   | loading | infinite dilution | – |
   | framework | ITQ-3 experimental (Cmcm; a 20.622, b 9.7242, c 19.623 Å) | the type material of ITE, available from two independent public sources (IZA type material and RASPA2). The IZA idealised cell is reported only as a sensitivity check |
   | force-field parameters | CH3–O 93.00 K / 3.48 Å, O only; CH3–CH3 108.00 K / 3.76 Å; cutoff 12 Å energy-shifted, no tail; harmonic bond k/kB 96 500 K/Å², r0 1.54 Å | all from Dubbeldam 2004 |
   | CV | tagged COM along crystal b (box X); every walker kept in one channel class (crystal z ≈ ¼), so the CV is linear and no mirror-image channel profile is mixed in | – |

   IHW is the fallback if ITE fails a gate.
4. **Order after A′:**
   * D only if A′ completes and resources remain; it is expected to be ABF-sufficient.
   * The labelled CH4/rigid-DDR substitute for B is a reserve.
   * C and E are ruled out (not reproducible).
