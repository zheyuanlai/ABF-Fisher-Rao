# LIT_PNAS2005 -- Dubbeldam, Beerdsen, Calero & Smit, "Molecular path control in zeolite membranes", PNAS 102 (2005) 12317-12320

Literature extraction for the ERI campaign: united-atom ethane in rigid all-silica ERI.
Written 2026-10-16. No simulations or GPU work were done. Nothing in the repository was
changed except this file and the new cache directory `cache/eri/literature/pnas2005/`.

## 0. How to read this file

* **Quotation policy.** I was asked for exact quotes. Copyright limits mean I can give only
  one short verbatim quote (Sec. 9). Everywhere else I give a paraphrase plus an exact
  location (page, column, section, paragraph, equation, figure). Numbers, parameter values
  and equations are copied exactly. They are facts, not prose. Anyone who needs the exact
  wording can open the cached PDFs (Sec. 1). Each location below was checked against the
  PDF text layer and the rendered page images.
* **Labels used below.**
  * **[TEXT]** means the paper states it.
  * **[FIG-DIGITIZED]** means I read the value off a figure image, either by eye or by
    pixel-level digitization of the embedded figure. Typical precision is about ±0.05 kBT
    for free energies and ±0.03×10⁻⁹ m²/s for diffusivities. The papers never state these
    numbers.
  * **[INFERENCE]** means my own deduction, with the reasoning given.
  * **[FLAG]** marks an ambiguity or inconsistency.
* **Page map of the PNAS PDF.** p. 12317 has the abstract, introduction and model section.
  p. 12318 has Eqs. 1-4 and Figs. 1-2. p. 12319 has Figs. 3-5 and the results text.
  p. 12320 has the discussion and references.

## 1. Provenance and retrieval log

All raw downloads are in their own directories under
`/home/zheyuanlai/ABF-Fisher-Rao/cache/eri/literature/pnas2005/`. No downloaded content
was executed. Text came from `pdftotext`, page images from `pdftoppm`/`pdfimages`, and
pixel digitization from my own scripts in the session scratchpad, run with `python -I`.

| What | Source URL / route | Local file | Result |
|---|---|---|---|
| PNAS full PDF (4 pp.) | Wayback snapshot 20250205130944 of `pmc.ncbi.nlm.nih.gov/articles/PMC1194924/pdf/pnas-0503908102.pdf` | `wayback_pmc_pdf/pnas-0503908102.pdf` | OK. sha256 `ff1d2e84eae6047a…`, 499,791 B. Matches PMC's stated 488.1 KB. |
| PNAS full PDF (independent copy) | EPFL Infoscience record 200810 (bitstream a093484c…) | `epfl_pnas2005/12317.full.pdf` | OK. **sha256 identical** to the Wayback copy. |
| PNAS full-text HTML (markdown) | PMC page through the Parallel web_fetch tool | `pmc_webfetch_tool/pmc1194924_fulltext.md` (+ raw JSON) | OK. Equations are images there; the PDF supplies them. |
| PMC metadata XML | NCBI efetch db=pmc id=1194924 | `eutils_efetch/pmc1194924.xml` | Front matter only. It records `pmc-prop-has-supplement = no`, so **no supporting information exists**. |
| Europe PMC metadata | EPMC REST search | `epmc_search/search.json` | isOpenAccess = N. Free to read at PMC/EPMC. |
| Europe PMC fullTextXML | EBI REST | `epmc_xml*/…HTTP500.json` | HTTP 500, because the article is not in the OA subset. |
| PMC HTML/PDF, PNAS PDF, Europe PMC PDF and figure images by direct curl | various | `*_try/…BLOCKED_challenge_page.html` | Blocked by reCAPTCHA or Cloudflare challenge pages. Kept and renamed so they are not mistaken for content. |
| Companion full paper: Dubbeldam, Beerdsen, Calero, Smit, *J. Phys. Chem. B* 110 (2006) 3164-3172, "Dynamically corrected TST calculations of self-diffusion in anisotropic nanoporous materials" | EPFL Infoscience (bitstream 4acc1ba2…) | `epfl_jpcb2006/3248743680dub061.pdf` | OK, 9 pp. **The same ERI/ethane/600 K system, with the methods written out in detail.** |
| PNAS ref. 25: Dubbeldam, Beerdsen, Vlugt, Smit, *J. Chem. Phys.* 122 (2005) 224712 | Utrecht Univ. DSpace bitstream 5a2e1f6d… | `uu_dspace_1083/1083.pdf` | OK, 18 pp. Covers LTL and LTA only, no ERI results. |
| PNAS ref. 24: Beerdsen, Smit, Dubbeldam, *PRL* 93 (2004) 248301 | EPFL (bitstream 94372c48…) | `epfl_prl2004_beerdsen/2667973892bee041.pdf` | OK. LTA only. |
| PNAS refs. 22, 21 (Dubbeldam force field, JPCB 108, 12301; PRL 93, 088302) | Utrecht DSpace | `uu_dspace_2003-007/`, `uu_dspace_force04/` | OK. **Not the force field the PNAS paper used** (see Sec. 4). No ERI content. |
| PNAS ref. 6 + Dubbeldam & Smit *JPCB* 107 (2003) 12138 (incommensurate diffusion, ERI at infinite dilution) | EPFL | `epfl_prl2003_incommensurate/`, `epfl_jpcb2003_window/` | OK. Different force field (Sec. 12). |
| Beerdsen, Dubbeldam, Smit *PRL* 96 (2006) 044501 and *JPCB* 110 (2006) 22754 (methane) | EPFL | `epfl_prl2006_beerdsen/`, `epfl_jpcb2006_beerdsen_methane/` | OK. Methane only, 300 K, but **Table 1 explains the PNAS box** (Sec. 3.3). |
| Smit 2007 and Beerdsen & Smit 2007 (IZC proceedings) | EPFL | `epfl_2007_*` | OK. Only passing ERI mentions. |
| RASPA2 `ERI_SI.cif` and `ERI_SI.block` (Dubbeldam, 2011) | raw.githubusercontent.com/iRASPA/RASPA2 | `raspa2_eri_cif/`, `raspa2_eri_block/` | OK. These files post-date 2005, so they are context only. |
| IZA ERI.cif (DLS-optimised SiO₂) | europe.iza-structure.org | `iza_eri_cif/ERI.cif` | OK. a = 13.0540, c = 15.1750 Å. |

**Not retrieved:**

* Schüring, Auerbach, Fritzsche, Haberlandt, *J. Chem. Phys.* 116 (2002) 10890 (PNAS
  ref. 19), the original source of the force field. I did not find a free copy in my
  search; the parallel force-field reviews may have more.
* Gard & Tait (1973), the ERI coordinates (PNAS ref. 14). It is in conference
  proceedings and has no online copy that I found.
* Dubbeldam's 2005 PhD thesis. The UvA lists it as "fully internal".
* Beerdsen's thesis.

## 2. Bibliographic facts [TEXT]

* PNAS 102(35):12317-12320, issue of 30 Aug 2005. The PMC epub date is 18 Aug 2005.
* doi 10.1073/pnas.0503908102, PMID 16109769, PMCID PMC1194924.
* Received 12 May 2005, approved 14 July 2005. Edited by David Chandler. Track II.
* Source: PDF p. 12317, header and footer.

## 3. System and structure

### 3.1 System [TEXT]

* PNAS p. 12317, right column, "Model and Computational Details", paragraph 1. Cations are
  neglected. The frameworks studied are **rigid, all-silica** ERI and CHA. The zeolite is
  modelled as a rigid network of **oxygen atoms**, following Bezus et al. (ref. 16). The
  stated justification is that the large O atoms shield the small Si atoms, and that
  lattice flexibility does not matter for small alkanes in all-silica zeolites (ref. 17).
* **[FLAG]** In the same paragraph the paper lists **Si-CH₃** LJ parameters (Sec. 4). The
  "oxygen-only" sentence and the parameter list contradict each other. The companion JPCB
  2006 paper (p. 3167, Sec. II.D) settles it: there the framework is a rigid network of
  **oxygen and silicon** atoms. So Si sites do interact, through Si-CH₃ LJ with σ = 2.12 Å
  and ε/kB = 82 K.

### 3.2 Structure source and unit cell [TEXT]

* PNAS p. 12317, right column, paragraph 1: the atom positions are taken from **ref. 14**,
  Gard & Tait (1973), *Proc. 3rd Int. Conf. Molecular Sieves*, pp. 94-99. ERI is named
  after erionite.
* PNAS Fig. 3 caption (p. 12319) gives the cell:
  * space group P6₃/mmc
  * **a = b = 1.327 nm, c = 1.505 nm**
  * α = β = 90°, γ = 120°
  * hopping-lattice distance **λ ≈ 0.75 nm** for the x, y and z directions.
* JPCB 2006 (p. 3165, Sec. II.A and the Fig. 1 caption) repeats the same cell and the same
  reference. It adds the following:
  * The erionite cages are elongated, roughly **1.3 × 0.63 nm cylinders**, joined by
    **0.36 × 0.51 nm** windows.
  * Only linear molecules can pass the 8-ring windows.
  * Each cage has **three windows at the top**, rotated 120° from one another, and **three
    at the bottom**, aligned with the top ones.
  * **Each ERI-type (hexagonal) unit cell contains two erionite cages.**
* JPCB 2006 (p. 3167, Sec. II.C) gives positions in fractional coordinates s of the
  hexagonal cell:
  * The **xy barrier (8-ring window) is at s_xy = (1/2, 1/2)**.
  * Cage A is selected by s_x + (1 - s_y) < 1 and cage B by s_x + (1 - s_y) > 1.
  * For the z profile of cage A, q = s_z. The free-energy minima qA and qB sit at
    s = (1/3, 2/3, 1/2) and (1/3, 2/3, 1), and the "barrier at the center of the cage" is
    at s = (1/3, 2/3, 3/4). See the [FLAG] in Sec. 8.2: that sentence conflicts with the
    paper's own figure and its later text.
* Context, not used by the paper:
  * RASPA2's `ERI_SI.cif` (author David Dubbeldam, 2011, citing Gard & Tait 1973) has
    a = 13.27, c = 15.05 Å and V = 2295.14 Å³. That matches the PNAS caption.
  * RASPA2's `ERI_SI.block` blocks four spheres of radius 2.5 Å at fractional (0, 0, 0),
    (0, 0, 0.25), (0, 0, 0.5) and (0, 0, 0.75). These are presumably the inaccessible
    cancrinite and double-6-ring column.
  * The IZA DLS cell (13.054 / 15.175 Å) differs from both cells discussed here.

### 3.3 Supercell: the PNAS box does not match the stated cell [TEXT + INFERENCE + FLAG]

* **PNAS (p. 12317, right column, paragraph 1)** gives a rectangular simulation box of
  **4.5906 × 3.9756 × 4.443 nm** for ERI (and 3.015 × 4.7814 × 2.7606 nm for CHA). Larger
  systems are said to show no significant finite-size effects. Periodic boundary conditions
  are used. **The paper does not give the number of unit cells, the number of cages, or N.**
* **These dimensions are not integer multiples of the stated cell** [INFERENCE, arithmetic].
  * With a = 1.327 and c = 1.505 nm, an orthorhombic supercell would measure
    2√3a = 4.5969, 3a = 3.981 and 3c = 4.515 nm.
  * The ratios the box actually gives are x/a = 3.459, y/a = 2.996 and z/c = **2.952**.
  * A periodic box 2.952 c long is impossible for a crystal.
* **Resolution found:**
  * Beerdsen, Dubbeldam & Smit, *JPCB* 110 (2006) 22754, Table 1 (p. 22758), list a
    **rectangular ERI cell of a = 22.953, b = 13.252, c = 14.810 Å**, orthorhombic, with
    simulation box **2×3×3**. The text on the same page says this rectangular cell
    **contains 4 cages**.
  * Dubbeldam & Smit, *JPCB* 107 (2003) 12138 (p. 12143, text near Fig. 6) state the same
    orthorhombic ERI cell, 2.2953 × 1.3252 × 1.481 nm, and the CHA cell
    1.5075 × 2.3907 × 1.3803 nm. That paper took atom positions from the IZA database,
    accessed May 2003.
  * 2 × 22.953 = 45.906 Å, 3 × 13.252 = 39.756 Å and 3 × 14.810 = 44.430 Å. **This is
    exactly the PNAS box.**
  * The PNAS CHA box is likewise exactly 2×2×2 of the orthorhombic CHA cell.
* **So the PNAS box is a 2×3×3 supercell of a 22.953 × 13.252 × 14.810 Å orthorhombic ERI
  cell.** That is 18 rectangular cells, equal to 36 hexagonal cells, holding **72 erionite
  cages**. My volume check gives 35.9998 hexagonal cells.
  * That cell corresponds to a hexagonal ERI with **a = 13.252 Å, c = 14.810 Å**. Its
    volume per cage is 1.1262 nm³, against 1.1476 nm³ for 13.27/15.05.
  * So it is about 1.9 % smaller in volume and **1.6 % shorter along c** than the cell
    printed in the PNAS caption.
  * I could not find an erionite refinement with a = 13.252 / c = 14.810 Å. Its origin is
    unknown.
* **The companion JPCB 2006 paper (p. 3167, Sec. II.D) gives a different box** for what is
  presented as the same simulations:
  * 3×3×3 hexagonal cells with **perpendicular widths 3.448 × 3.448 × 4.515 nm**.
  * Those widths are consistent with a = 1.327, c = 1.505 nm: 3a sin 120° = 3.4476 and
    3c = 4.515.
  * That is 27 cells and **54 cages**.
* **[FLAG, unresolved]** From the papers alone I cannot tell which cell and box produced
  the published ERI data. One candidate is 13.252/14.810 Å with 72 cages, which matches
  the PNAS box. The other is Gard & Tait 13.27/15.05 Å with 54 cages, which matches the
  PNAS caption and the JPCB 2006 box.
  * At loading L molecules/cage, N would be 72L or 54L respectively. At L = 7 that is 504
    or 378 ethanes.
  * The 1.6 % difference in c changes the cage length and the window geometry. This matters
    for barrier heights.
  * Recommendation: treat the PNAS box as most likely copied from the group's standard
    orthorhombic cells, and make cell choice an explicit sensitivity check. This is my
    judgement, not a literature fact.
* **Cages per unit cell:** 2 per hexagonal P6₃/mmc cell (JPCB 2006, p. 3165), or 4 per
  orthorhombic a×√3a×c cell (Beerdsen JPCB 2006, p. 22758).
* **Cage connectivity:**
  * Beerdsen JPCB 2006 (p. 22758): each elongated cage connects to **six** other cages,
    **three on each side**. The elliptical windows run from 3.6 Å (b direction) to 5.1 Å
    (c direction).
  * PNAS Fig. 3 / JPCB 2006 Fig. 2 draw a planar hexagonal (honeycomb-looking) hopping
    lattice in xy. In xz, the vertical lattice lines carry pink midpoints, the cage
    centres. Horizontal (xy) links occur at the ends of the vertical segments.

## 4. Force field

### 4.1 What PNAS states [TEXT, PNAS p. 12317, right column, paragraph 1]

* United-atom model (ref. 18, Ryckaert & Bellemans). Each CH₃ group is a single
  **chargeless** interaction centre. Adsorption in cation-free structures happens at sites
  with little or no electric field, so there is **no Coulomb interaction**.
* **Bond:** harmonic, U_bond = ½ k₁ (r - r₀)², with **k₁/kB = 96,500 K/Å²** and
  **r₀ = 1.54 Å**. Ethane has no bend or torsion.
* **Lennard-Jones parameters** for the non-intramolecular energy:

| pair | σ (Å) | ε/kB (K) |
|---|---|---|
| O-CH₃ | 3.17 | 142 |
| Si-CH₃ | 2.12 | 82 |
| CH₃-CH₃ | 3.78 | 104 |

* **Source: ref. 19, Schüring, Auerbach, Fritzsche & Haberlandt, *J. Chem. Phys.* 116
  (2002) 10890-10894.** It is *not* the Dubbeldam et al. 2004 force field.
  * Refs. 6 and 20-23 are cited only to show that the group's simulation techniques have
    been checked against experiment. Refs. 22 and 21 are Dubbeldam JPCB 108 12301 and PRL
    93 088302.
  * [FLAG] The PMC HTML rendering garbles this sentence ("ε CH3-CH33 / kB = 104 which K"
    and "σ Si-CH"). The PDF text gives the clean values in the table above.
* **Not stated in PNAS:**
  * cutoff
  * shifting
  * tail corrections
  * mixing rule (not needed: all cross pairs are given explicitly)
  * bead masses
  * whether inaccessible cancrinite pockets were blocked
  * any Si charge (none: chargeless)

### 4.2 Filled in by the companion and method papers

* **Cutoff 12 Å.** JPCB 2006, p. 3167, Sec. II.D: the guest-guest and host-guest LJ
  interactions use a cutoff radius of 12 Å. The same paragraph gives the parameters
  verbatim as in PNAS, again citing Schüring 2002 (ref. 48 there).
* **Shifting.** JCP 2005 (PNAS ref. 25), p. 224712-2, Sec. II.A: LJ potentials are
  **shifted and cut at 12 Å**, and analytical tail corrections are not used in zeolites.
  * [FLAG] That sentence belongs to JCP 2005's general description of its own (Dubbeldam)
    force field. PNAS refers readers to ref. 25 for "more details" of the MD (p. 12318,
    left column, top). JPCB 2006 says only "cutoff radius of 12 Å".
  * So "shifted at 12 Å" is the best-supported reading, but it is inherited rather than
    stated for this system.
* **The authors' own view of this parameter set:**
  * JPCB 2006 (p. 3167, Sec. II.D): the authors admit the size parameters are rather small.
    They chose them because ethane diffusion in ERI and CHA then stays feasible with
    conventional MD.
  * JCP 2005 (p. 224712-3, text near Table II): its Table II lists the **same** parameters
    (O 142 K/3.17 Å, Si 82 K/2.12 Å, CH₃ 104 K/3.78 Å, same bond), described as those used
    by Schüring et al. for ethane in cation-free LTA. JCP 2005 notes that the very small
    size parameters make ethane diffuse about **two orders of magnitude faster** in LTA
    than with the Dubbeldam parameters. It also says the authors consider them probably
    less realistic, but convenient for comparing MD with dcTST.
* For contrast, JCP 2005 Table I (Dubbeldam et al.) gives O-CH₃ ε/kB = 93.0 K and
  σ = 3.48 Å. The PNAS/Schüring O-CH₃ σ is 0.31 Å smaller and ε is 53 % larger.
* Masses are still not stated anywhere I read. United-atom CH₃ is conventionally about
  15.03 g/mol, but **that is my assumption, not a literature fact**.

## 5. Temperature, ensemble, MD protocol [TEXT]

* **Temperature: 600 K only** for every ERI result. This is in the captions of PNAS
  Figs. 1, 2, 3 and 5 and in the results text on p. 12318. **Verified.**
  * PNAS says path control could also be driven through temperature (abstract; Discussion,
    p. 12320), but **no other temperature was simulated for ERI**.
  * JPCB 2006 Fig. 12 also has CHA at 300 K.
* MD details, PNAS p. 12317 right column to p. 12318 left column:
  * velocity Verlet with a **0.5 fs** time step
  * relative energy drift below 10⁻⁴
  * **Nosé-Hoover chain** thermostat (Martyna et al., ref. 29)
* Initialisation:
  * random insertion without overlaps
  * an NVT MC stage to equilibrate the arrangement
  * Maxwell-Boltzmann velocities with zero total momentum
  * NVT-MD (NHC) equilibration
  * production **longer than 20 ns, NVT**. NVE runs gave equivalent results.
* JCP 2005 (p. 224712-5) adds that initial insertion uses only positions accessible from
  the main cages and channels. Neither PNAS nor JPCB 2006 says whether this, or pocket
  blocking, was applied to ERI. **[FLAG]**
* JCP 2005 (p. 224712-5, Fig. 2 caption) runs infinite-dilution MD with intermolecular
  forces switched off (non-interacting particles, NHC-NVT). PNAS and JPCB 2006 do not say
  how their loading-0 MD points were made. **[FLAG]**
* dcTST sampling, JPCB 2006 p. 3167, Sec. II.D:
  * transmission coefficients from **at least 50,000** independent configurations
  * configurations stored every 500 MC cycles
  * MC moves: translation, rotation and full regrow
  * free-energy profiles obtained from both MD and MC, with equivalent results.

## 6. Loadings and units [TEXT + FIG]

* Unit: **molecules per ERI-type cage**, on the axis labels of PNAS Figs. 1 and 2 and in
  the Fig. 3 caption. By the cage counts above, 1 molecule/cage = 2 molecules per
  hexagonal unit cell.
* Fig. 1 (D_S vs loading) covers **0 (infinite dilution) and 1, 2, 3, 4, 5, 6, 7**
  molecules/cage.
* Fig. 2 (ratio D_z/D_xy) covers **0-6** molecules/cage. There is no point at 7.
* Fig. 3 (free-energy profiles) covers infinite dilution and 1, 2, 3, 4, 5, 6, 7
  molecules/cage. In the legend, 1-6 appear as anonymous black lines.
* Fig. 5 (MSD) is at 600 K with **no loading stated** [FLAG]. The implied D values match
  the low-loading (0-2/cage) end of Fig. 1 (Sec. 11).
* JPCB 2006 Fig. 3 (MSD) is at **1 molecule/cage**. Its Fig. 5 (κ(t)) uses infinite
  dilution and 1, 2 and 5 /cage. Its Fig. 6 (k_z) uses 0-7 /cage.
* **No adsorption isotherm or pressure-loading mapping is given**, even though the abstract
  and discussion frame the control parameter as pressure. Saturation loading is not stated.
  JPCB 2006 only says diffusion decreases close to saturation loading. **[FLAG]**

## 7. How the free-energy profiles were computed

* **In PNAS itself:**
  * The paper says only that dcTST supplies free-energy profiles and lattice information
    (p. 12318, left column, paragraph 2).
  * dcTST splits diffusion into a static term (adsorption sites and the free-energy
    barriers between them) and a dynamic term that generally falls with loading and is
    tied to the inverse collision frequency (same paragraph).
  * The method is credited to Beerdsen et al. (refs. 24, 25).
  * **PNAS gives no estimator, sampling scheme, bin width or reaction-coordinate
    definition.**
* **From refs. 24/25 and JPCB 2006 [TEXT]:**
  * Rate (JPCB 2006 eqs. 3-4; JCP 2005 eqs. 27-30). The rate is
    k_A→B = κ · √(kBT/2πm) · e^(−βF(q*)) / ∫_cage A e^(−βF(q)) dq, where m is the mass of
    the segment(s) in the reaction coordinate and κ is the Bennett-Chandler transmission
    coefficient (plateau of κ(t)).
  * **Profile estimator.** JCP 2005, p. 224712-9, Sec. II.G: during a separate **NVT MC
    simulation at the desired loading**, F(q) comes from a **histogram of particle positions
    mapped onto q**, with βF(q) = −ln⟨P(q)⟩ (JCP 2005 eq. 41). All particles count as
    equivalent and all contribute.
    * When barriers are too high for plain histograms, one tagged particle is biased, with
      the Widom-insertion profile as the bias. Widom insertion is **not** used for F(q)
      directly at higher loading, because it is known to fail there (same page).
    * JPCB 2006 (p. 3167) says profiles come from both MD and MC with equivalent results.
  * **Other molecules present.** PRL 2004 (ref. 24), p. 248301-2:
    * The effective hopping rate is that of a single tagged particle at average loading c.
      The other N−1 particles act as an external field, kept up by a "background" NVT MC
      with translation, orientation and regrow moves.
    * Neighbouring-cage occupancy must be allowed to fluctuate. Closing off cages, or
      fixing an exact number per cage, changes F(q) and shifts D by up to 60 % (in LTA).
    * One shell of neighbouring cages is enough.
  * **Reaction coordinate.** JPCB 2006, p. 3166, Sec. II.C:
    * q is the position of **one of the two beads** of ethane, in fractional (S-space)
      coordinates of the hexagonal cell. Which bead is not stated; by symmetry it should
      not matter for ethane.
    * **xy profile:** the barrier sits at s_xy = (1/2, 1/2), and by symmetry all positions
      in the unit cell can be mapped onto the coordinate. **The exact mapping function for
      q in the xy plane is not written down.** **[FLAG]**
    * **z profile:** q = s_z, using only positions in cage A.
    * This choice of order parameter underestimates the true transition-state free energy,
      and κ compensates exactly (JPCB 2006 p. 3166; PRL 2004 p. 248301-2).
  * **xy rates:** Bennett-Chandler at the 8-ring window. κ(t) plateaus after about 1 ps at
    low loading and about 10 ps at 5/cage (JPCB 2006 Fig. 5 and text p. 3169).
  * **z rates:** the z profile has two barriers at higher loading. The authors therefore
    ran **importance-sampled ("biased") MD**, JPCB 2006 eq. 5 with
    w(q) = e^(aβF(q)), to get the total hop rate from qA (top) to qB (bottom) directly,
    with no separate κ (JPCB 2006 p. 3169).
* **[INFERENCE, relevant to ABF]** The published profiles are 1-D marginal free energies of
  one bead's projected position, sampled in equilibrium at fixed total N. Every other
  ethane is free to move, and cage occupancies fluctuate. They are not constrained-PMF
  profiles of a centre-of-mass coordinate.

## 8. Published profile features (PNAS Fig. 3 right panels = JPCB 2006 Fig. 4)

The two papers show the same curves. I digitized the cleaner JPCB 2006 Fig. 4 at its
native 500 × 812 px. Axis label: "Free energy barrier βF [-]". **Every curve is shifted so
its own minimum is near 0**, so only relative values are available. [FIG-DIGITIZED]
throughout.

### 8.1 xy profile (cage centre → 8-ring window → neighbouring cage centre)

Tick labels qA', qA, q*, qB, qB'. The caption defines qA as a cage centre and qB as the
neighbouring cage centre. **qA' and qB' are labelled but not defined anywhere** [FLAG].
They lie between qA (qB) and the steep outer walls.

| loading | βF at q* (8-ring window) | other features |
|---|---|---|
| infinite dilution (blue) | **6.13** | One broad minimum (≈0-0.2) from about qA towards the window side. At qA' it is about 0.8, on the rising wall. |
| 1-6 /cage (six black, unlabelled) | **6.05, 5.63, 4.24, 3.09, ~2.55, ~2.51** | As loading rises, a local maximum grows between qA and the window, and a secondary minimum appears just on the window side of it. |
| 7 /cage (red) | **2.61** | Global minima near qA' and qB' (≈0.1). βF(qA) ≈ 1.07, a local maximum ≈ 1.5 between qA and the window, and a local minimum ≈ 0.57 at the unlabelled grid line before q*. |

* Assigning the black curves to particular loadings by monotonic order (1→6/cage =
  6.05→2.51) is **my inference**. The figure does not label them.
  * It is self-consistent with Fig. 1. Between 2 and 3/cage the barrier drops by about
    1.4 kBT and D_xy rises about 3.7×. Between 3 and 4/cage it drops by about 1.15 kBT
    and D_xy rises about 3×.
  * At 5-7/cage the window barrier is about constant (2.5-2.6). D_xy still falls (Fig. 1),
    which fits the falling κ and denser cages.
* In absolute units, 6.13 kBT at 600 K is **≈30.6 kJ/mol** (my conversion).
* Text, PNAS p. 12319:
  * In-plane behaviour is "normal" for cage/window zeolites: diffusivity rises with loading
    because the barrier falls. The paper attributes this to finite cage volume, where added
    molecules mean more repulsion inside the cage (p. 12318, right column, bottom).
  * At low loading the only barriers are the xy barriers at the eight-ring windows between
    neighbouring cages (p. 12319, left column, top).

### 8.2 z profile (along c across one cage: qA top, qI middle, qB bottom)

| loading | βF at the two q* maxima | βF at qI (cage middle) |
|---|---|---|
| infinite dilution (blue) | **≈1.1** | ≈0.9. Nearly flat, a shallow ~0.2 kBT dip. |
| 1-6 /cage (black, unlabelled) | ≈1.2, ≈1.5, ≈2.45, ≈2.5, ≈2.7. Five resolved; the sixth overlaps. | 0.92, 0.65, 0.46, 0.42, 0.25, 0.18 |
| 7 /cage (red) | **≈4.4** (4.36-4.39) | **≈0.04**: a minimum as deep as the end minima (≈0.1) |

* **Positions [INFERENCE].** These rest on the JPCB fractional coordinates and on pixel
  positions measured on the 500 × 812 px Fig. 4b.
  * qA and qB are at s_z = 1/2 and 1, i.e. c/2 ≈ 0.75 nm apart; qI is at s_z = 3/4.
  * The two q* maxima lie about 46 % of the way from each end minimum to qI: s_z ≈ 0.615
    and 0.885, roughly ±0.2 nm from the cage middle.
  * The plotted range spans about s_z ≈ 0.34-1.14, about 1.2 nm in total, which fits a
    ~1.3 nm cage.
  * qA and qB are at the heights of the 8-ring windows. Neighbouring cages are offset by
    c/2, so the window to a +z-shifted neighbour sits at the top of the cage and vice
    versa. This is consistent with λ_z ≈ c/2 ≈ 0.75 nm, but **neither paper states where
    the windows sit along z**.
* **[FLAG] The wording about "barriers at cage centres" conflicts with the figure** (see the
  quote in Sec. 9).
  * The figure shows the cage **middle (qI) becoming a free-energy minimum** at high
    loading, with **two barriers** between the middle and the two ends.
  * JPCB 2006 contradicts itself:
    * Sec. II.C (p. 3167) puts "the barrier at the center of the cage" at s_z = 3/4, with
      minima at s_z = 1/2 and 1.
    * Sec. III.A (p. 3169) says a clear free-energy minimum forms in the cage centre with
      loading. It names qI as the intermediate minimum, and puts one barrier between qA
      and qI and another between qI and qB.
  * Read "barrier at the centres of the cages" as "intracage barriers, inside the cage,
    between ends and middle". Do not read it as a maximum at the geometric centre.

## 9. Diffusion pathways: low vs high loading

Lattice model, PNAS p. 12318, left column, Eqs. 1-2; JPCB 2006 eqs. 12-22. [TEXT]

* In the xy plane, diffusion runs isotropically on a hexagonal lattice:
  **D_xy = ¼ k_xy λ_xy²** (Eq. 1).
* In ERI every z hop is preceded by an xy hop, so diffusion is anisotropic:
  **D_z = ½ · k_xy k_z / (k_xy + k_z) · λ_z²** (Eq. 2).
* JPCB 2006 (p. 3167, eqs. 13-21) derives Eq. 2 from a mean-first-passage argument for
  hops in series (A→B→C), assuming symmetric barriers.
* λ ≈ 0.75 nm in all directions (Fig. 3 caption).
* [INFERENCE] If k_z ≫ k_xy, D_z → ½ k_xy λ² = 2 D_xy. This is the factor of 2 seen at
  low loading.
* **Low loading (0-2 /cage)** [TEXT, p. 12319, left column, top]:
  * There are no intracage barriers along z. The barriers come only from the xy (8-ring
    window) crossings.
  * D_z is about 2 D_xy for both self and collective diffusion (p. 12317, right column,
    introduction; p. 12318, right column, Results).
  * JPCB 2006 (p. 3168) reads D_z ≈ 2 D_xy as evidence of no significant intracage
    barriers at low loading.
  * Pathway: in-plane window hops are rate limiting. Movement along z comes from xy hops
    into neighbouring, c/2-offset cages followed by fast travel along the cage.
* **High loading (≥4 /cage)**:
  * The xy barriers fall and new intracage barriers appear. PNAS p. 12319, left column, in
    the paper's words: **"new barriers are formed at the centers of the cages"** (Dubbeldam
    et al., PNAS 2005, p. 12319).
  * Those barriers eventually dominate and reverse the anisotropy. D_z falls to below half
    of D_xy.
  * Pathway: in-plane hopping through the windows is now favoured, and travel along the
    cage axis is rate limiting.
* The JPCB 2006 abstract and conclusions put the same thing in plain terms:
  * At low loading the 8-ring windows (the exits from the cage) impede diffusion.
  * At higher loading the barrier is formed by the cage centres. See the [FLAG] in Sec. 8.2
    about where these barriers actually are.
* "Molecular path control" (PNAS abstract, introduction and Discussion) means one species
  takes different preferred pathways at different loadings. The paper contrasts this with
  molecular traffic control, a multicomponent effect (p. 12320). The authors say MPC comes
  from the anisotropy of the material: different channel types or elongated cages.
* **Control: CHA** (PNAS p. 12319, Figs. 4-5; JPCB 2006 Sec. III.B).
  * In CHA the cage is not elongated, there is a single hop rate k to every neighbouring
    cage, and diffusion is isotropic in practice. The lattice distortion (94.07° vs 90°)
    changes the components by less than 2 %.

## 10. Self- vs collective diffusion [TEXT + FIG]

* PNAS Eqs. 3-4 (p. 12318):
  * D_S^α = (1/2N) lim_{t→∞} d/dt ⟨Σ_i (r_iα(t) − r_iα(0))²⟩, using centre-of-mass
    coordinates.
  * D_C^α = (1/(2Nξ)) lim_{t→∞} d/dt ⟨(Σ_i (r_iα(t) − r_iα(0)))²⟩.
  * 1/ξ is the thermodynamic factor, obtained from the isotherm (not shown).
* D_C is identified with the transport diffusivity D_T. "Corrected" diffusivity is D_C with
  the thermodynamic factor removed. The **thermodynamic factor does not affect the ratio**
  D_z/D_xy (p. 12318, right column).
* Fig. 2 (MD, 600 K) shows ratios only. Absolute D_C values are not shown.

| loading /cage | D_z/D_xy self | D_z/D_xy collective |
|---|---|---|
| 0 | ≈1.97 | ≈1.97 (overlapping) |
| 1 | 1.79 | ≈1.79 (overlapping) |
| 2 | 1.72 | 1.54 |
| 3 | 0.83 | 1.03 |
| 4 | 0.42 | 0.40 |
| 5 | 0.44 | 0.28 |
| 6 | 0.53 | 0.38 |

The values are [FIG-DIGITIZED], ±~0.03; the error bars run about ±0.05-0.1. The
crossover to isotropy is near **3 molecules/cage**. At 7/cage, Fig. 1 implies a ratio of
roughly 0.03/0.52 ≈ 0.06 from dcTST, but Fig. 2 does not plot it.

## 11. Quoted and digitized numbers

* **PNAS text states no numerical diffusivities, barriers or rates.** Every number below is
  [FIG-DIGITIZED] unless marked otherwise.
* PNAS Fig. 1 / JPCB 2006 Fig. 7: self-diffusivity at 600 K, in units of 10⁻⁹ m²/s.

| loading /cage | dcTST xy | dcTST z | MD x≈y | MD z |
|---|---|---|---|---|
| 0 | 0.24 | 0.46 | ≈0.24 | ≈0.45 |
| 1 | 0.21 | 0.41 | ≈0.21 | ≈0.37 |
| 2 | 0.27 | 0.48 | ≈0.27 | ≈0.48 |
| 3 | 1.0 | 0.90 | ≈1.0 | 0.84 |
| 4 | **3.0** (maximum) | 1.14 | 2.93-2.95 | 1.22-1.25 |
| 5 | 2.87 | 1.04 | ≈2.8 | 1.23 |
| 6 | 1.72 | 0.77 | ≈1.84 | 0.96 |
| 7 | 0.52 | ≈0.02-0.03 | ≈0.5 | ≈0.07-0.1 |

* D_xy rises about 14× from 1 to 4 molecules/cage. D_z rises about 3× and peaks near
  4-5/cage. At 7/cage D_z collapses about 20-50× below D_xy.
* **JPCB 2006 text (p. 3168)** gives D_xy = 2.1 × 10⁻⁹ and D_z = 4.2 × 10⁻⁹ m²/s at
  1 molecule/cage and 600 K.
  * **[FLAG] That is 10× higher than the same authors' own figures.** JPCB Fig. 3 (MSD at
    1/cage) and Fig. 7, and PNAS Fig. 1, all give ≈0.21 and ≈0.41 × 10⁻⁹ m²/s.
  * PNAS Fig. 5 independently gives ≈0.25 and ≈0.47 × 10⁻⁹ (below).
  * This looks like an exponent typo (10⁻¹⁰ intended). **The text value should not be
    used.**
* **Transmission coefficients κ, xy window** (JPCB 2006 Fig. 5): plateaus ≈ **0.29**
  (infinite dilution), **0.24** (1/cage), **0.20** (2/cage) and **0.08** (5/cage).
* **Intracage z hop rate k_z** (JPCB 2006 Fig. 6, biased MD, plateau at 5-10 ps, events/s):

| loading /cage | k_z (events/s) |
|---|---|
| 0 | 1.1 × 10¹¹ |
| 1 | 5.8 × 10¹⁰ |
| 2 | 1.4 × 10¹⁰ |
| 3 | 6.0 × 10⁹ |
| 4 | 5.2 × 10⁹ |
| 5 | 4.4 × 10⁹ |
| 6 | 3.0 × 10⁹ |
| 7 | ≈4.6 × 10⁷ |

* **Consistency check [INFERENCE].** I took λ = 0.75 nm and k_xy = 4D_xy/λ², put the
  digitized k_z into Eq. 2, and got D_z of 0.47 (L = 0), 1.19 (L = 4) and 0.013 (L = 7),
  in units of 10⁻⁹ m²/s. Fig. 1 shows 0.46, 1.14 and ≈0.02-0.03. So the digitized values
  hang together, and **λ_z ≈ 0.75 nm and Eq. 2 are confirmed** as the authors' conversion.
  The implied k_xy is ≈1.7 × 10⁹ s⁻¹ at infinite dilution.
* **PNAS Fig. 5 (MSD, 600 K, loading not stated).**
  * ERI x, y: about 50 Å² at 1000 ps, so D ≈ 2.5 × 10⁻¹⁰ m²/s.
  * ERI z: an initial jump of about 18-20 Å² within a few ps (free travel along the cage),
    then a slope giving D ≈ 4.7 × 10⁻¹⁰ m²/s. The MSD reaches 112 Å² at 1000 ps.
  * CHA x, y, z: about 163-170 Å² at 1000 ps, so D ≈ 8.3 × 10⁻¹⁰ m²/s, isotropic.
* **JPCB 2006 Fig. 3 (MSD, 1/cage).**
  * The x and y MSD plateaus at about 1.5-2 Å² from roughly 0.5 to 50 ps (confined in the
    xy cross-section).
  * The z MSD plateaus at about 10-20 Å² (motion along the cage).
  * Both reach λ² ≈ 56 Å² at around 10²-10³ ps.
  * The text says the MSD becomes linear only after one cage crossing.

## 12. Other papers on ERI free energies (context)

* **Dubbeldam & Smit, *JPCB* 107 (2003) 12138, and PRL 90 (2003) 245901**
  (PNAS ref. 6; "incommensurate diffusion").
  * ERI free-energy profiles at 600 K **at infinite dilution only**, for C1-C5, C8 and C9
    (Figs. 2 and 16-17).
  * Reaction coordinate: the second-bead position, projected orthogonally onto the line
    perpendicular to the window (JPCB 2003, p. 12141, Fig. 1 caption and text).
  * **Different force field:** TraPPE-like, O-CH₃ ε/kB = 58 K, σ = 3.6 Å, Si implicit,
    r_cut = 13.8 Å with shift (JPCB 2003, p. 12139).
  * Structure from the IZA database, converted to the 22.953 × 13.252 × 14.81 Å
    orthorhombic cell (p. 12143). **Not comparable with the PNAS numbers.**
* **Beerdsen, Dubbeldam & Smit, PRL 96 (2006) 044501 and *JPCB* 110 (2006) 22754.**
  * Methane in ERI at 300 K with the Dubbeldam force field.
  * D_S peaks with loading, rising up to about 60× over infinite dilution in ERI. D_C rises
    nearly two orders of magnitude.
  * Profiles come from the MC histogram method.
  * **No ERI free-energy profiles are shown.** Table 1 is the source of the
    orthorhombic-cell/box resolution in Sec. 3.3.
* **JCP 2005 (PNAS ref. 25)** contains no ERI results. Its Fig. 18 compares the Schüring
  parameter set for ethane in rigid vs flexible LTA and finds significant,
  temperature-dependent differences. This is relevant to the PNAS "rigid lattice"
  assumption, but it concerns LTA, not ERI.

## 13. Figure-by-figure description

**PNAS (p. 12318-12319)**

* **Fig. 1:** D_S^α against loading, 0-7 molecules/cage, 600 K.
  * MD x, y and z appear as grey filled square, triangle and circle. dcTST appears as open
    squares (xy) and open circles (z), with spline guides.
  * D_z > D_xy for L ≤ 2, crossover near L ≈ 3, D_xy ≫ D_z for L ≥ 4. Both curves peak
    near L = 4-5 (values in Sec. 11).
* **Fig. 2:** D_z/D_xy against loading, 0-6 /cage. Filled squares are collective, open
  circles are self, and a solid line marks isotropy at 1. Values are in Sec. 10.
* **Fig. 3:** four panels.
  * The two left panels are renderings of the ERI pore space with the hopping lattice in
    blue: xy (honeycomb-looking hexagonal network) and xz (vertical cage-axis segments with
    magenta midpoints, linked horizontally at their ends).
  * The two right panels are the βF(q) profiles: upper is xy, lower is z (Sec. 8).
  * The caption gives the P6₃/mmc cell, a = 1.327 nm, c = 1.505 nm, λ ≈ 0.75 nm and the
    600 K profiles.
* **Fig. 4:** rendering of the CHA unit cell. R-3m (the caption calls it a "squashed
  cube"), a = 0.9459 nm, α = 94.07°, with its lattice.
* **Fig. 5:** MSD in Å² over 0-1000 ps at 600 K for CHA (x, y, z) and ERI (x, y, z).
  Loading is not stated. Values are in Sec. 11.

**JPCB 2006 companion (same data, more methods)**

* Fig. 1 shows the unit cells, with two cages per ERI cell and one per CHA cell.
* Fig. 2 shows the ERI hopping topology, with the pink points marking the cage centres.
* Fig. 3: MSD at 1/cage, log-log.
* Fig. 4: the profiles, identical to PNAS Fig. 3 right.
* Fig. 5: κ(t).
* Fig. 6: k_z(t) from biased MD.
* Fig. 7: D(L), identical to PNAS Fig. 1.
* Figs. 8-12: CHA (rectangular vs "squashed-cube" lattice, profiles, D at 300 and 600 K).

## 14. Ambiguities and open issues (consolidated)

1. **Simulation cell [major].** The PNAS box (4.5906 × 3.9756 × 4.443 nm) is exactly 2×3×3
   of a 22.953 × 13.252 × 14.810 Å orthorhombic ERI cell, which holds 72 cages.
   * That is **not** the Gard & Tait cell (13.27/15.05 Å) printed in the PNAS Fig. 3
     caption, which is also RASPA's `ERI_SI.cif`.
   * JPCB 2006 gives a 3×3×3 hexagonal box (3.448 × 3.448 × 4.515 nm, 54 cages)
     consistent with Gard & Tait.
   * Which box produced the data is unresolved. The origin of the 13.252/14.810 Å cell is
     unknown.
2. **Framework sites.** PNAS says O-only, but gives Si-CH₃ parameters. JPCB 2006 says O + Si.
   Use O + Si.
3. **Shifting and tail corrections** are not stated for this system. JPCB 2006 gives a 12 Å
   cutoff. Ref. 25 says "shifted and cut at 12 Å" in a general section.
4. **Masses, pocket blocking and initial-insertion accessibility** are not stated.
5. **The xy reaction-coordinate mapping** is not fully specified. qA' and qB' are
   undefined.
6. **"Barriers at the centres of the cages"** does not match the profile, which shows a
   minimum at the cage middle flanked by two barriers. JPCB 2006 contradicts itself here.
7. **The black profile curves (1-6 /cage) are unlabelled.** Assigning them by monotonic
   order is my inference, though it is consistent with Fig. 1.
8. **The JPCB 2006 text diffusivities at 1/cage are 10× its own figures.** Use the figures.
9. **The Fig. 5 loading** is not stated.
10. **There is no pressure/isotherm data.** "Pressure control" cannot be mapped onto
    loading from these papers.
11. **Barrier character is never discussed.** The word "entropic" does not appear in PNAS
    or JPCB 2006. Neither paper decomposes the ~6.1 kBT window barrier or the intracage
    barriers into energetic and entropic parts, and only one temperature (600 K) is
    reported for ERI.
    * For the campaign's "entropic barrier" premise, this paper alone does not establish
      that the ERI barriers are entropic. That needs a separate check, for example the
      temperature dependence of βF(q*) or ⟨U⟩(q). This is my assessment.
12. **The force field is deliberately unrealistic.** The authors themselves call the
    Schüring parameters small-σ and "probably less realistic", chosen so that MD stays
    feasible. Reproducing the published profiles requires these parameters. A realistic
    force field (Dubbeldam 2004) would give much slower diffusion and different barriers.
13. **The Schüring 2002 original was not retrieved.** The parameters rest on two consistent
    restatements by the same group (PNAS; JCP 2005 Table II; JPCB 2006).
