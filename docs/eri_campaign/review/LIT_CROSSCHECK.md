# LIT_CROSSCHECK: adversarial cross-check of the ERI / ethane literature extractions

Written 2026-10-16. Inputs checked: `LIT_FORCEFIELD_A.md` (FF-A), `LIT_FORCEFIELD_B.md` (FF-B),
`LIT_PNAS2005.md` (PNAS-X) and `LIT_ERI_STRUCTURE.md` (STRUCT-X), all in this directory. Nothing was
committed and no GPU was used. The only repository paths written are this file and the new download
directory `cache/eri/literature/crosscheck/`, with one new directory per download. Downloads were
treated as untrusted data. Python that read them ran with `python -I`, and nothing downloaded was
executed.

Status labels used throughout:

* **CONFIRMED**: a verified primary quote that I re-read myself, or two independent sources.
* **CONFLICT**: the sources disagree. I state both readings and which one is right, and why.
* **UNRECOVERABLE**: not stated in any source I could obtain. I say what I tried.
* **INFERRED**: not stated for this study, but carried over from the same group's method papers.

---

## 0. Bottom line

1. **The original force-field paper is now in hand. Both FF reports had listed it as unobtainable.**
   * The paper: Schüring, Auerbach, Fritzsche & Haberlandt, "On entropic barriers for diffusion in
     zeolites: A molecular dynamics study", J. Chem. Phys. 116, 10890 (2002) (PNAS ref. 19).
   * Where it came from: co-author Auerbach's reprint, captured by the Wayback Machine on 2003-03-10.
     It is the AIP-stamped publisher PDF, carrying the footer "Downloaded 10 Jun 2002 to
     128.119.39.33".
   * Local copy: `cache/eri/literature/crosscheck/wayback_auerbach_pap42_20030310/pap42.pdf`, sha256
     `0f9f8151…`.
   * What it settles:
     * The PNAS LJ values are Schüring's Table I values converted from kJ/mol to K and rounded.
     * The CH3–Si term is part of the original model.
     * CH3–CH3 comes from OPLS (Jorgensen, Madura & Swenson 1984). This upholds FF-A's inference.
     * Schüring's own model differs from PNAS in three ways: a **Morse** bond, a **flexible**
       framework, and **shifted-force LJ cut at 2.5σ**. PNAS 2005 therefore runs Schüring's pair
       parameters under the Dubbeldam group's conventions: harmonic 96 500 K/Å² bond, rigid
       framework, 12 Å cutoff.
2. **Every force-field number the four reports give for the PNAS model is CONFIRMED.**
   * The values: O–CH3 142 K / 3.17 Å, Si–CH3 82 K / 2.12 Å, CH3–CH3 104 K / 3.78 Å, harmonic bond
     k1/kB = 96 500 K/Å² with r0 = 1.54 Å, no charges, rigid all-silica framework, 12 Å cutoff, 600 K.
   * I found them in four documents: PNAS, JPCB 2006, JCP 2005 Table II, and Schüring 2002 Table I.
3. **Si is included. This is CONFIRMED and closes FF-A U2 and FF-B item 2.**
   * PNAS's sentence "rigid network of oxygen atoms" is contradicted by its own parameter list, by
     JPCB 2006 ("oxygen and silicon atoms"), and by Schüring's Table I.
   * The effect is small. On one CH3 bead, Si changes the window barrier along the window path by
     about −14 K, out of 565–579 K (Sec. 5).
4. **The truncation convention for the PNAS runs is still not stated** (INFERRED: energy-shifted at
   12 Å, no tail corrections). I computed its effect on the single-bead window barrier: energy-shifted
   at 12 Å, unshifted at 12 Å, and Schüring's 2.5σ shifted-force all agree to within 1 K (Sec. 5). So
   this gap has low impact on F(q), though it does move absolute energies.
5. **The framework remains the one first-order unresolved item.**
   * The PNAS rectangular box is exactly 2√3a × 3a × 3c for a = 13.252 Å, c = 14.81 Å. That is the
     cell of Schlenker et al. 1977.
   * The structure PNAS cites (Gard & Tait 1973: 13.27 / 15.05 Å) and the JPCB 2006 box (3×3×3
     hexagonal) do not fit that box.
   * I re-derived the arithmetic and STRUCT-X's single-bead barriers independently: 579 K (Gard &
     Tait) against 353 K (Schlenker), O-only.
   * New supporting evidence: the same group's 2003 offretite cell (2.3021 × 1.3291 × 0.7582 nm)
     equals the Gard & Tait 1972 offretite refinement cell (a = 13.291, c = 7.582 Å). So the group's
     2003 orthorhombic cells were built from experimental refinements, and 13.252 / 14.81 is a
     refinement cell (Schlenker 1977).
   * Even so, which coordinates sat in that cell cannot be recovered.
6. **Verdict.** The published ERI model **cannot be reproduced exactly**, only as a **labelled
   surrogate** ("PNAS-2005-type ERI model"). Reproduction is closest for the infinite-dilution
   profiles at 600 K, which are the relevant target for a one-ethane-per-replica ABF engine. A
   literature comparison must:
   * declare the framework and test the alternative;
   * use the single-bead fractional coordinate the papers used;
   * compare only barrier heights, because each published curve is shifted to its own minimum.

   Carrying the repo LTA engine's Dubbeldam-2004 parameters into ERI would give a different model.
   As an indication, its single-bead window barrier comes out about 5× higher (2748 K against 565 K,
   Gard & Tait; Sec. 5).

---

## 1. Sources I re-read myself, and new downloads

### 1.1 Primary documents I re-read (already cached by the other agents)

I identified each file by its text layer, not its file name. Byte-identical copies are noted.

| Document | Cached copy I read | sha256 (prefix) | Notes |
|---|---|---|---|
| PNAS 102, 12317 (2005), publisher PDF | `forcefield_B/pnas2005_epfl/12317.full.pdf` | ff1d2e84eae6047a | Byte-identical to `pnas2005/epfl_pnas2005/` and `pnas2005/wayback_pmc_pdf/`. I checked p. 12317 (force-field paragraph) and pp. 12318–12319 (Figs. 1, 2, 3, 5) on rendered pages. |
| Dubbeldam, Beerdsen, Calero, Smit, JPCB 110, 3164 (2006) | `pnas2005/epfl_jpcb2006/3248743680dub061.pdf` | dcf2e294110ed628 | Byte-identical to the FF-B copy. pp. 3168–3170 rendered; Fig. 4 digitized by me. |
| Dubbeldam, Beerdsen, Vlugt, Smit, JCP 122, 224712 (2005) | `forcefield_A/jcp2005_epfl_pdf/…` | 2e932d3d0bcb9302 | The UU copies (`forcefield_B/jcp2005_uu/`, `pnas2005/uu_dspace_1083/`) have identical text after the cover page. |
| Dubbeldam et al., JPCB 108, 12301 (2004) | `forcefield_B/jpcb2004_epfl/0689513732dub041.pdf` | b2a3a669d3addc5a | Byte-identical to the FF-A copy. |
| Dubbeldam et al., PRL 93, 088302 (2004) | three UU copies | different hashes | Text identical after the cover page (same md5 of the text body). |
| Dubbeldam & Smit, JPCB 107, 12138 (2003) | `pnas2005/epfl_jpcb2003_window/` | ef1bcbe18a3398d1 | Gives the orthorhombic cell definitions. |
| Beerdsen, Dubbeldam, Smit, JPCB 110, 22754 (2006) | `pnas2005/epfl_jpcb2006_beerdsen_methane/` | 567871d8d46f3302 | Table 1 (ERI orthorhombic cell). |
| Beerdsen, Smit, Dubbeldam, PRL 93, 248301 (2004) | `pnas2005/epfl_prl2004_beerdsen/` | 90217f71629e5342 | Neighbouring-cage fluctuation statements. |
| RASPA2 `ERI_SI.cif`; archived IZA record 55 (2006-01-03); COD 2106432 | `structure/…` | per STRUCT-X | Framework coordinates compared site by site. |

### 1.2 New downloads (all under `cache/eri/literature/crosscheck/`)

| Directory / file | What | sha256 |
|---|---|---|
| `wayback_auerbach_pap42_20030310/pap42.pdf` | **Schüring et al., JCP 116, 10890 (2002)**, 5 pp., AIP-stamped PDF (Creator XPP, StampPDF, CreationDate 2002-05-28). URL `http://web.archive.org/web/20030310124515id_/http://samson.chem.umass.edu:80/~auerbach/pub_pdf/pap42.pdf`. | 0f9f815159fb1776cec8c461d1587278a43e7ef36f09f20203a7db50f0f63621 |
| `wayback_auerbach_pap53_20041028/pap53.pdf` | Schüring, Auerbach, Fritzsche, Haberlandt, "Random walk treatment of dumb-bell molecules in an LTA zeolite and in chabazite", Proc. 14th IZC (2004) (JCP 2005 ref. 43). Wayback 2004-10-28. | 63914d23d79cc617d54d72f4d608af77949056ac6d1c10917c9fcb765ab59dc5 |
| `auerbach_cv_html/cvweb.html` | Auerbach CV page (samson.chem.umass.edu/html/cvweb.html). The publication-to-PDF link (entry 42 → `pub_pdf/pap42.pdf`) was read through the web-fetch tool. | ece7dda2724d962d… |
| `auerbach_pap42_live_404/pap42_404_notfound.html` | The live URL now returns HTTP 404. Kept and relabelled for provenance. | 9118201933d4e479… |
| `crossref_gard_tait/q.json` | Crossref query for Gard & Tait erionite papers. | 65ecb6e96d6edaf0… |

Search route, so that others can repeat it: a web search for Auerbach's publication list found
samson.chem.umass.edu/html/cvweb.html, entry 148/42: "Andreas Schuring, Scott M. Auerbach, Siegfried
Fritzsche and Reinhold Haberlandt, 'On Entropic Barriers for Diffusion in Zeolites: A Molecular
Dynamics Study,' J. Chem. Phys. 116, 10890-10894 (2002). [Abstract] [PDF]". The live PDF is gone
(404). The Wayback CDX lists six captures between 2003 and 2008.

---

## 2. Reconciled master parameter table: the PNAS 2005 ERI/ethane model

Unit conversions use kB = 0.008314462618 kJ mol⁻¹ K⁻¹.

| # | Parameter | Value | Units | Source (exact location) | Status |
|---|---|---|---|---|---|
| 1 | Source of the non-bonded set | Schüring, Auerbach, Fritzsche, Haberlandt 2002 | – | PNAS p. 12317: "…which were taken from ref. 19". JPCB 2006 p. 3167: "…were taken from ref 48". JCP 2005 Table II: "Force field LJ parameters used by Schüring et al. for ethane in cation-free LTA-type zeolite (Ref. 24)" | CONFIRMED |
| 2 | ε(O–CH3)/kB | 142 (original 1.18 kJ/mol = 141.92 K) | K | PNAS p. 12317 "εO–CH3/kB = 142 K"; JPCB 2006 p. 3167; JCP 2005 Table II; Schüring 2002 Table I, p. 10891 ("CH3–O 3.17 1.18") | CONFIRMED |
| 3 | σ(O–CH3) | 3.17 | Å | same four sources | CONFIRMED |
| 4 | ε(Si–CH3)/kB | 82 (original 0.683 kJ/mol = 82.15 K) | K | PNAS; JPCB 2006; JCP 2005 Table II; Schüring Table I ("CH3–Si 2.12 0.683") | CONFIRMED |
| 5 | σ(Si–CH3) | 2.12 | Å | same (the PMC HTML misprint "σSi–CH" is absent from the publisher PDF, checked on the rendered page) | CONFIRMED |
| 6 | Does Si interact? | **Yes**, O + Si | – | JPCB 2006 p. 3167 "modeled as a rigid network of oxygen and silicon atoms"; Schüring Table I lists CH3–Si; JCP 2005 Table II fills the Si column. Against: PNAS p. 12317 "modeled as a rigid network of oxygen atoms". | CONFLICT, resolved in favour of O + Si (3 sources against 1; the PNAS sentence is the group's stock Bezus/Kiselev wording, and PNAS itself lists the Si term) |
| 7 | ε(CH3–CH3)/kB | 104 (original 0.866 kJ/mol = 104.16 K) | K | PNAS; JPCB 2006; JCP 2005 Table II; Schüring Table I | CONFIRMED (enters only at finite loading) |
| 8 | σ(CH3–CH3) | 3.78 | Å | same | CONFIRMED |
| 9 | Origin of CH3–CH3 | OPLS-UA ethane CH3 (Jorgensen, Madura, Swenson, JACS 106, 6638 (1984)) | – | Schüring 2002 p. 10890: "The intermolecular potential parameters were taken from Ref. 7"; ref. 7 = Jorgensen et al. 1984. FF-A's TraPPE-4 table gives OPLS-UA ethane CH3 as 3.775 Å / 104.1 K. | CONFIRMED (FF-A's inference upheld) |
| 10 | Mixing rule in the PNAS model | none needed: all three pairs are given explicitly | – | PNAS p. 12317 | CONFIRMED |
| 11 | How Schüring built the host–guest pairs | Lorentz–Berthelot from O (and Si) self-parameters | – | Schüring 2002 p. 10891: "The Lennard-Jones parameters for oxygen taken from Ref. 12 were combined with the above-mentioned intermolecular parameters using the rules of Lorentz-Berthelot" (ref. 12 = Fritzsche et al., Chem. Phys. 174, 229 (1993)). IZC 2004: "taken from [11] for CH4-O and transfered to CH3-O using the combination rules of Lorentz-Berthelot. The same has been done for CH3–Si" ([11] = Bezus et al. 1978). | CONFIRMED that LB was used. The two Schüring papers name different upstream sources (Fritzsche 1993 vs Bezus 1978), which were not retrieved. The implied self-terms are my back-calculation: σ_O = 2.56 Å, ε_O/kB ≈ 193 K; σ_Si = 0.46 Å, ε_Si/kB ≈ 65 K. |
| 12 | Charges | none | – | PNAS p. 12317 "single, chargeless interaction centers"; JPCB 2006 p. 3167 | CONFIRMED. Note: RASPA2 `ERI_SI.cif` carries charges Si 2.05 / O −1.025, which must be ignored. |
| 13 | Intramolecular LJ | none (2-bead molecule) | – | PNAS "non-intramolecular energy is described with a Lennard–Jones potential" | CONFIRMED |
| 14 | Bond functional form | harmonic, U = ½ k1 (r − r0)² | – | PNAS p. 12317 (rendered); JPCB 2006 p. 3167; JCP 2005 Table II; JPCB 2004 Table 3 | CONFIRMED |
| 15 | k1/kB | 96 500 (= 802.35 kJ mol⁻¹ Å⁻²) | K/Å² | same | CONFIRMED |
| 16 | r0 | 1.54 | Å | same | CONFIRMED |
| 17 | Whose bond it is | Dubbeldam 2004 (from Nath, Escobedo, de Pablo 1998), **not** Schüring | – | Schüring 2002 p. 10891: "A Morse potential … with D = 83.9 kJ/mol, β = 1.84 Å⁻¹, and r0 = 1.54 Å as in Ref. 8" (ref. 8 = Demontis et al. 1996). Its curvature 2Dβ² = 568.1 kJ mol⁻¹ Å⁻² (68 330 K/Å²) is my arithmetic. | CONFIRMED. Resolves FF-A U1(d) and FF-B U1: PNAS is a hybrid of Schüring LJ and the Dubbeldam bond. |
| 18 | LJ cutoff | 12 | Å | JPCB 2006 p. 3167 "a Lennard-Jones potential with a cutoff radius of 12 Å" (same study); also JCP 2005 p. 224712-2, JPCB 2004 p. 12302, PRL 2004 p. 088302-2. PNAS is silent. | CONFIRMED (study-specific primary quote) |
| 19 | Truncation / shift / tail | energy-shifted at rc, truncated force, no tail corrections | – | Not stated in PNAS or JPCB 2006. JCP 2005 p. 224712-2, in the paragraph that introduces both Table I and Table II: "The Lennard-Jones potentials are shifted and cut at 12 Å. Analytical tail corrections are not applicable in zeolites." JPCB 2004 Eq. 14 defines "shifted" as U − E_cut. | INFERRED, low impact: the single-bead barrier changes by < 1 K across conventions (Sec. 5). Schüring's own convention differs: "The shifted force simulation technique has been employed for all Lennard-Jones interactions. The cutoff radii were chosen to be 2.5σ of the respective interaction." (p. 10891). |
| 20 | Framework flexibility | rigid | – | PNAS p. 12317 "rigid, all-silica versions"; JPCB 2006 p. 3167 | CONFIRMED. Schüring 2002 used Demontis's flexible lattice, and JCP 2005 p. 224712-16 says "the differences between flexible and rigid LTA-type zeolites for ethane are significant and temperature dependent" (LTA, Schüring parameters). |
| 21 | Cations / composition | none; all-silica | – | PNAS p. 12317 "we neglect cations" | CONFIRMED |
| 22 | CH3 mass | not stated | g/mol | grep of PNAS, JPCB 2006 and JCP 2005 text layers: no value | UNRECOVERABLE. Irrelevant to equilibrium F(q); matters only for dynamics. |
| 23 | Temperature (all ERI results) | 600 | K | PNAS captions of Figs. 1, 2, 3, 5 and Results p. 12318 ("at 600 K as a function of loading"); JPCB 2006 Figs. 3–7. The only other temperature in JPCB 2006 is CHA at 300 K (Fig. 12). | CONFIRMED |
| 24 | Loading unit | molecules per ERI-type cage | – | PNAS Fig. 3 caption "(infinite dilution, 1, 2, 3, 4, 5, 6, and 7 molecules per ERI-type cage)"; Fig. 1/2 axes "Loading [molecules/cage]" | CONFIRMED |
| 25 | Loadings simulated | profiles: ∞-dilution and 1–7; D_S: 0–7; D_z/D_xy: 0–6 | /cage | PNAS Figs. 1–3 (rendered) | CONFIRMED |
| 26 | Cages per cell | 2 per hexagonal cell (4 per a√3 × a × c orthorhombic cell) | – | JPCB 2006 p. 3165 "Each ERI-type unit cell contains two erionite cages"; Beerdsen 2006 p. 22758 "one unit cell of erionite contains 4 cages" (rectangular) | CONFIRMED |
| 27 | N in the box at loading L | 72 L (PNAS box) or 54 L (JPCB 2006 box) | molecules | arithmetic on the two stated boxes | UNRECOVERABLE (not stated) |
| 28 | Structure source cited | Gard & Tait 1973, Proc. 3rd Int. Conf. Molecular Sieves, pp. 94–99; P6₃/mmc, a = 13.27 Å, c = 15.05 Å | Å | PNAS ref. 14 and Fig. 3 caption "a = b = 1.327 nm, c = 1.505 nm"; JPCB 2006 ref. 30 | CONFIRMED as the citation. The proceedings themselves were not obtained (UNRECOVERABLE). |
| 29 | Gard & Tait coordinates | T1 (0.00021, 0.2335, 0.10446), T2 (0.0935, 0.4245, 0.25), O1–O6 as in STRUCT-X | fractional | RASPA2 `ERI_SI.cif` = archived IZA record 55 (I compared every site; the IZA values are listed at 0.75/0.25 Si/Al occupancy on the same positions) | CONFIRMED (two independent copies) |
| 30 | Simulation box | **PNAS:** "rectangular simulation box sizes … 4.5906 × 3.9756 × 4.443 nm". **JPCB 2006:** "3 × 3 × 3 (perpendicular widths are 3.448 × 3.448 × 4.515 nm)" | nm | PNAS p. 12317; JPCB 2006 p. 3167 | CONFLICT, UNRECOVERABLE. PNAS = 2√3a × 3a × 3c with a = 13.252, c = 14.81 Å (exact). JPCB 2006 = 3×3×3 Gard & Tait (34.48 / 45.15 Å). |
| 31 | Framework actually used | Gard & Tait (13.27 / 15.05), Schlenker 1977 (13.252 / 14.81, COD 2106432), or Gard & Tait fractional coordinates in the Schlenker cell | – | Sec. 3.5 | UNRECOVERABLE; first-order for the barrier |
| 32 | Blocking of CAN / D6R pockets | not mentioned | – | No "block", "pocket" or "inaccessible" in the PNAS or JPCB 2006 text layers. RASPA2 `ERI_SI.block` (4 spheres of 2.5 Å) dates from 2011/2015. | UNRECOVERABLE whether used. Immaterial for ethane started in an ERI cage: the pockets are isolated, and rigid ethane in them sits at ≥ +24 900 K (STRUCT-X; I reproduced the single-bead values). |
| 33 | Cage / window size (literature) | cages ≈ 1.3 × 0.63 nm cylinders; windows 0.36 × 0.51 nm; 3 windows at the top and 3 at the bottom | nm | JPCB 2006 p. 3165; Beerdsen 2006 p. 22758 "3.6 Å in the b direction to 5.1 Å in the c direction" | CONFIRMED |
| 34 | Cage / window geometry (computed, Gard & Tait) | window-centre d_O 3.134 Å (short free aperture 3.57 Å); cage centre ↔ window 5.369 Å; cage ↔ cage 10.739 Å, Δz = 7.525 Å | Å | STRUCT-X, reproduced by my own script | CONFIRMED (independent recomputation) |
| 35 | Hopping length λ | ≈ 0.75 | nm | PNAS Fig. 3 caption "λ ≈ 0.75 nm for x, y, and z directions"; computed a/√3 = 7.661 Å, c/2 = 7.525 Å | CONFIRMED |
| 36 | Profile coordinate | position of **one bead**, in fractional (S-space) coordinates of the hexagonal cell. xy: "along the center-to-center line in the hexagonal xy plane", barrier at s_xy = (½, ½). z: q = s_z within cage A, qA = ½, q at ¾, qB = 1. | – | JPCB 2006 p. 3166 "We can choose q as the position of one of the two beads of ethane"; p. 3167 Eqs. 10–11; p. 3168 | CONFIRMED in outline. The exact xy folding and the definitions of qA′/qB′ are UNRECOVERABLE. |
| 37 | Profile estimator | βF(q) = −ln⟨P(q)⟩ from position histograms in NVT MC at the loading (all particles counted) | – | JCP 2005 p. 224712-9 Eq. 41 "histogram sampling (HS)"; JPCB 2006 p. 3167 "The free-energy profiles are obtained using MD and MC; both give equivalent results" | CONFIRMED for the method. Bins and run lengths for ERI: UNRECOVERABLE. |
| 38 | Integrator (dynamics only) | velocity Verlet, 0.5 fs; Nosé–Hoover chains; production > 20 ns NVT | – | PNAS pp. 12317–12318; JPCB 2006 p. 3167 | CONFIRMED |
| 39 | Box vs cutoff | both stated boxes have minimum widths (39.756 Å; 34.48 Å) > 2 × 12 Å | Å | JPCB 2006 p. 3166 "The smallest perpendicular width of the unit cell has to be larger than twice the spherical cutoff" | CONFIRMED (arithmetic) |

### 2.1 Comparison models (context only)

| Model | Values | Status |
|---|---|---|
| Dubbeldam 2004 (JPCB 108 Table 3; PRL Table I) | O–CH3 93.00 K / 3.48 Å; O only; CH3–CH3 108.00 K / 3.76 Å; 12 Å energy-shifted, no tail; harmonic bond 96 500 K/Å², r0 = 1.54 Å | CONFIRMED (re-read Table 3, p. 12302, and Eq. 14 at "rcut ) 12.0 Å") |
| TraPPE-UA ethane | CH3 98 K / 3.75 Å; fixed 1.54 Å bond; 14 Å + tail corrections; LB | CONFIRMED (two independent sources: FF-A from the TraPPE-4 preprint, FF-B from the TraPPE database) |
| Dubbeldam & Smit 2003 ERI | O–CH3 58 K / 3.6 Å, Si implicit, rcut 13.8 Å with shift | CONFIRMED (JPCB 107 p. 12139: "rcut ) 13.8 Å", "/kB ) 58 K for O-CH3"). Not comparable with PNAS. |
| Repo LTA engine (`src/lta/core_lta.py`) | eps_go = 93.0·KB, sigma_go = 3.48, rc = 10.0, k_bond = 400 kJ mol⁻¹ Å⁻² (lines 80–84); O only; "TraPPE-zeolite family" label (line 12) | CONFIRMED as both FF reports describe it. Deviations from Dubbeldam 2004: rc 10 vs 12 Å; k 400 vs 802.35. Labels wrong. |

---

## 3. Adjudication item by item (A vs B vs PNAS-X vs STRUCT-X vs primary)

### 3.1 Points where all four reports agree, and the primary sources confirm them

* LJ values, bond, no charges, rigid framework, 600 K, and the PNAS box dimensions. I re-read the
  publisher PDF on a rendered page (PNAS p. 12317, right column) and confirm the parameter sentence
  verbatim, as quoted in row 2 of the table.
* PNAS cites Dubbeldam 2004 (refs. 21, 22) only for "The accuracy of the simulation techniques have
  been verified in several publications (6, 20–23)". It is not the parameter source.
* JCP 2005 Table II and its comments. Two exact quotes (p. 224712-3):
  * "their very small size parameters enhance diffusion by two orders of magnitude for ethane in
    LTA-type zeolite compared to the parameters of Dubbeldam et al."
  * "Although the parameters used by Schüring et al. are probably less realistic in our opinion".

### 3.2 Points where the reports disagreed or flagged gaps

| Item | FF-A | FF-B | PNAS-X / STRUCT-X | My finding |
|---|---|---|---|---|
| Schüring 2002 original | not obtained (U1) | not obtained (item 1) | not obtained | **Obtained** (Sec. 1.2). Table I: σ / ε(kJ/mol) = 3.78 / 0.866 (CH3–CH3), 3.17 / 1.18 (CH3–O), 2.12 / 0.683 (CH3–Si). |
| Si active? | unresolved (U2) | "medium confidence" yes | PNAS-X: yes (from JPCB 2006) | **CONFIRMED yes.** Schüring includes CH3–Si, and JPCB 2006 says O + Si. |
| Whose bond | open (U1d) | "Dubbeldam's harmonic" (stated as fact, unverified) | – | **CONFIRMED Dubbeldam's.** Schüring used Morse (D 83.9 kJ/mol, β 1.84 Å⁻¹). |
| Cutoff for PNAS | "L (inferred)" | 12 Å "high" (JPCB 2006) | 12 Å (JPCB 2006) | **CONFIRMED 12 Å** (JPCB 2006 quote). FF-A missed the JPCB 2006 sentence because it did not read that paper. |
| Shift / tail | inferred | inferred, medium | inferred | **INFERRED** and still not stated. The barrier is insensitive to it (Sec. 5). Schüring's original was shifted-force at 2.5σ, so "Schüring's convention" must not be imported either. |
| Origin of 3.78 Å / 104 K | inferred OPLS-UA | – | – | **CONFIRMED OPLS** (Schüring ref. 7 = Jorgensen et al. 1984). |
| Origin of 142 K / 3.17 Å and 82 K / 2.12 Å | unknown (U4) | unknown | – | **Partly resolved.** They are LB combinations with O/Si self-parameters from Fritzsche et al. 1993 (JCP 2002), or a CH4–O pair from Bezus 1978 transferred by LB (IZC 2004). The upstream papers were not retrieved. |
| Dubbeldam 2004 guest–guest mixing | "geometric in both ε and σ" (§4) | "numerically Lorentz–Berthelot" (§3) | – | **CONFLICT → FF-A is right.** Table 3 gives σ(CH3–C) = 4.90 Å: √(3.76·6.38) = 4.898, but (3.76+6.38)/2 = 5.07. Likewise σ(CH–C) = 5.46 (geometric 5.458, arithmetic 5.525) and σ(CH4–CH) = 4.17 (geometric 4.168, arithmetic 4.195). FF-B checked only pairs with near-equal σ, where the two rules coincide to 0.01 Å. Irrelevant for pure ethane. |
| JCP 2005 Table I, CH3–CH2 ε | – | printed "77.7" | – | FF-B is right. The text layer has "77.7" in the CH3 row and "77.77" in the CH2 row: a print asymmetry, irrelevant. |
| PNAS supporting information | – | – | none (PMC flag) | Not re-checked. No SI is cited in the PDF. |

### 3.3 Structure

| Item | Finding | Status |
|---|---|---|
| RASPA2 `ERI_SI.cif` = Gard & Tait | All 8 sites agree digit for digit with archived IZA record 55. That record reads: "Reference: Gard, J.A. and Tait, J.M. Proc. 3rd Int. Conf. Molecular Sieves, , pp. 94-99, (1973)", "X-ray single crystal refinement, R=0.15", a = 13.27, c = 15.05 Å. | CONFIRMED |
| Schlenker 1977 cell | COD 2106432: "_cell_length_a 13.252", "_cell_length_c 14.81", Acta Cryst. B 33, 3265–3268. | CONFIRMED |
| PNAS box arithmetic | Schlenker: 2√3 × 13.252 = 45.9063, 3 × 13.252 = 39.756, 3 × 14.81 = 44.430 Å, which equals the PNAS box exactly. Gard & Tait: 45.9686 × 39.810 × 45.150 Å, off by 1.6 % in z. | CONFIRMED (my arithmetic) |
| Group's orthorhombic cell | JPCB 2003 p. 12143: "all structures are converted to orthorhombic periodic unit cells with dimensions … a ) 2.2953 nm, b ) 1.3252 nm, and c ) 1.481 nm for ERI". Positions are "taken from ref 22" (IZA, "accessed May 2003"). Beerdsen 2006 Table 1: ERI 22.953 / 13.252 / 14.810, "orthorhombic", "2×3×3". | CONFIRMED |
| New evidence on where the 2003 cells came from | The same 2003 sentence gives OFF as 2.3021 × 1.3291 × 0.7582 nm = √3·13.291 × 13.291 × 7.582 Å. That is the Gard & Tait (1972, Acta Cryst. B28, 825) offretite cell; the cell values are from Mindat (secondary), the citation from Crossref. So the group's orthorhombic cells were built from experimental refinements, which makes Schlenker 1977 the likely origin of 13.252 / 14.81. The PNAS CHA box (2 × 15.075, 2 × 23.907, 2 × 13.803 Å) likewise matches the 2003 CHA cell, not the cited Calligaris cell (rhombohedral 9.459 Å / 94.07° → hexagonal 13.847 / 15.176 Å by my conversion). | INFERRED (supporting, not decisive) |
| JPCB 2006 box | "3 × 3 × 3 (perpendicular widths are 3.448 × 3.448 × 4.515 nm)". This is 3 × 13.27 × sin 60° = 34.476 and 3 × 15.05 = 45.15 Å, i.e. Gard & Tait. Its CHA box (2.81 nm) is consistent with Calligaris (my estimate 2.82 nm). So JPCB 2006 matches the cited cells for both zeolites, and PNAS matches the 2003 cells for both. | CONFIRMED (arithmetic). The conflict itself is unresolvable. One possibility (my hypothesis): MD in the rectangular 2003-cell box, dcTST profiles in the crystallographic cell, which JPCB 2006 Sec. II.C says is "often very convenient" for profiles. |
| "Gard & Tait, Acta Cryst B29, 2007" (from the structure task text) | Crossref lists Gard & Tait, Acta Cryst. B28, 825 (1972) (offretite) and Adv. Chem. 101, 230 (1971/74) ("Structural Studies on Erionite and Offretite"). There is no Acta Cryst. erionite refinement. | CONFIRMED STRUCT-X: the reference does not exist |
| Windows, cages, connectivity | Window d_O = 3.134 Å (Gard & Tait) and 3.256 Å (Schlenker); cage–window 5.369 / 5.324 Å; cage–cage 10.739 / 10.648 Å with Δz = −7.525 / −7.405 Å. Recomputed by me; identical to STRUCT-X. | CONFIRMED |
| Pocket energies (single bead, O-only, 12 Å, unshifted) | CAN 1556 K and D6R 506 K (Gard & Tait); CAN 9828 K and D6R 1215 K (Schlenker). Identical to STRUCT-X. | CONFIRMED |

---

## 4. Claims in the PNAS extraction about loadings, temperature, pathways and profiles

| PNAS-X claim | Direct support | Verdict |
|---|---|---|
| 600 K only for ERI | Captions of PNAS Figs. 1, 2, 3, 5; "at 600 K as a function of loading" (p. 12318); JPCB 2006 Figs. 3–7. The abstract says paths "may be controlled by changing the loading, either through a change in pressure or temperature", but no other ERI temperature is reported. | CONFIRMED |
| Loading unit and ranges (Fig. 1: 0–7, Fig. 2: 0–6, Fig. 3: ∞ + 1–7) | Captions and axes (rendered) | CONFIRMED |
| No isotherm or pressure–loading map | PNAS says only that the thermodynamic factor "can easily be evaluated from the adsorption isotherm (30)"; no isotherm is shown. JPCB 2006: diffusion "decreases only close to saturation loading" (p. 3170), with no value. | CONFIRMED |
| Low loading: window-limited, D_z ≈ 2 D_xy | PNAS p. 12318: "at low loading the diffusivity in the z direction is two times faster than in the xy direction for both the self and collective diffusivity, whereas for higher loadings this changes into a z diffusivity that is more than two times slower". PNAS pp. 12318–12319: "Initially, at low loadings, there are no intracage barriers, and therefore the barriers to diffusion are formed by the xy barriers, i.e., the eight-ring windows between the adjacent cages." | CONFIRMED. The numerical bands "0–2 /cage" and "≥ 4 /cage" are PNAS-X's reading of Fig. 2 (crossover near 3). That reading is correct, but it is a figure reading, not text. |
| High loading: intracage barriers dominate | "At higher loadings, the xy barriers decrease and new barriers are formed at the centers of the cages. Eventually, the barriers at the centers of the cages dominate the diffusion mechanism, thereby reversing the anisotropy of the diffusion." (p. 12319) | CONFIRMED as text |
| "Barrier at cage centre" conflicts with the figure | JPCB 2006 p. 3167: "The barrier at the center of the cage is located at s = {1/3, 2/3, 3/4}". Against it, p. 3169: "with increasing loading a clear free-energy minimum is formed in the center of the cage" and "There are two barriers, one separating qA and qI, and another one between qI and qB". The rendered PNAS Fig. 3 (lower right) and JPCB 2006 Fig. 4b show qI as a local **minimum** at every loading. | CONFLICT, text against figure. The figure is right. The papers' "barriers at the centers of the cages" means intracage barriers between the cage ends and the middle. |
| Profile features | My own pixel digitization of JPCB 2006 Fig. 4 (500 × 812 px; axis calibrated on frame and grid lines, 43.14 px per kBT in panel a and 55 px per kBT in panel b; values from the top edge of a line about 2 px wide): xy ∞-dilution peak 6.14; 7/cage peak 2.64, side bumps 1.53–1.55; z ∞-dilution maxima 1.11–1.13 with qI at 0.89–0.91; z 7/cage maxima 4.36–4.40 with qI at 0.04–0.09. | CONFIRMED against PNAS-X (6.13, 2.61, ≈1.1, ≈0.9, 4.36–4.39, ≈0.04): two independent digitizations agree to about 0.03 kBT. These remain figure values, and every curve is plotted relative to its own minimum. |
| 1–6 /cage curves unlabelled | Legend "1-6 molecules/cage", six black lines | CONFIRMED that they are unlabelled. Assigning them by monotonic order is an inference. |
| JPCB 2006 text diffusivities are 10× its figures | Text p. 3168: "The diffusion coefficients at one molecule per cage 600 K are Dxy = 2.1 × 10-9 m2/s and Dz = 4.2 × 10-9 m2/s". The rendered PNAS Fig. 1 and JPCB Fig. 7 at 1 /cage show about 0.2 × 10⁻⁹ (xy) and about 0.4 × 10⁻⁹ (z). | CONFLICT. The figures are right: three figures agree (PNAS Fig. 1, JPCB Fig. 7, and PNAS Fig. 5 MSD slopes), and the k_z/κ consistency check of PNAS-X supports them. |
| No "entropic" in PNAS / JPCB 2006 | grep of both text layers: no match for "entrop" | CONFIRMED |
| Profiles are single-bead marginals at fixed N, with neighbouring-cage occupancy fluctuating | JPCB 2006 p. 3166 (one bead); JCP 2005 Eq. 41; Beerdsen PRL 2004 p. 248301-2: "It is essential to average over fluctuations in the number of particles in the neighboring cages" and "deviate up to 60%" | CONFIRMED |
| Provenance table: `uu_dspace_2003-007/` is "OK" (JPCB 108 12301) | Its text layer, after the UU cover page, is *Surface Science 529 (2003) 197–203*, "Hot carrier luminescence during porous etching of GaP". | **ERROR in PNAS-X Sec. 1.** It is the same wrong-file defect FF-A and FF-B flagged. No PNAS-X fact depends on it. |
| "Exact xy mapping not written down" | JPCB 2006 p. 3168 adds "along the center-to-center line in the hexagonal xy plane", and p. 3167 "by symmetry all positions in the unit cell can be mapped on the reaction coordinate" | Partly too strong: the line is specified, but the folding is not. Minor. |

### 4.1 The literature on whether the barrier is entropic (new and relevant to the campaign premise)

* **Schüring 2002 is about entropic barriers for this exact parameter set, in LTA.**
  * Abstract: "The diffusivity and the cage-to-cage rate constant both decrease with temperature
    because heating the system moves ethane away from eight-ring windows, on average, which increases
    the entropic barrier for cage-to-cage motion."
  * p. 10893: "At 300 K the entropic barrier is 15 kJ/mol."
  * p. 10893: "At 80 K, the profile of the local Helmholtz free energy A(x) is similar to the profile
    of the internal energy U(x) with a local minimum at the position of the window."
  * The study is LTA with a flexible lattice, Morse bond and 2.5σ shifted force. It is not ERI.
* JCP 2005 p. 224712-13 restates this for the same parameter set in rigid LTA: "Heating the system,
  ethane moves away from the windows, which increases the entropic barrier for cage-to-cage motion."
  It adds that the Dubbeldam set "does not show a decrease with increasing temperature".
* **No source decomposes the ERI barrier.** An indicative comparison (an inference, not a
  free-energy calculation):
  * The 600 K infinite-dilution window barrier, 6.13 kBT, is about 3680 K.
  * The single-bead potential-energy barrier along the window path is 565–579 K (Gard & Tait).
  * The best rigid-ethane window energy minus the best cage energy is about 506 K (STRUCT-X).
  * So the minimum-energy barrier is roughly 0.85–1 kBT of the 6.1 kBT. This suggests the ERI
    window barrier is predominantly entropic under the PNAS parameters. It still has to be checked
    with U(q) and −TS(q), or with the temperature dependence.

---

## 5. Independent computations (CPU, numpy, `python -I`; framework sites typed from the CIF text)

Setup:

* Single CH3 bead.
* Path P2: horizontal at window height (s_z = ½), from the B-column axis (⅓, ⅔) through the window
  (½, ½) to the C-column axis (⅔, ⅓).
* Barrier = U(window centre) − min U along P2.
* Images to ±2 cells; LJ cut at 12 Å unless stated.

| Framework | Convention | U(window) K | min U(P2) K | barrier K | U(cage centre) K |
|---|---|---|---|---|---|
| Gard & Tait | O only, unshifted (STRUCT-X's setting) | −1325 | −1904 | **579** (= STRUCT-X) | −1190 |
| Gard & Tait | O only, energy-shifted | −1279 | −1858 | 579 | −1145 |
| Gard & Tait | **O + Si, energy-shifted (PNAS reading)** | −1345 | −1910 | **565** | −1170 |
| Gard & Tait | O + Si, Schüring shifted-force at 2.5σ (7.93 Å / 5.30 Å) | −882 | −1448 | 566 | – |
| Gard & Tait | Dubbeldam 2004 set (93 K / 3.48 Å, O only, shifted) | **+1110** | −1637 | **2748** | – |
| Schlenker 1977 | O only, unshifted | −1566 | −1919 | **353** (= STRUCT-X) | −1150 |
| Schlenker 1977 | **O + Si, energy-shifted** | −1586 | −1926 | **340** | −1130 |
| Schlenker 1977 | O + Si, shifted-force 2.5σ | −1115 | −1456 | 342 | – |
| Schlenker 1977 | Dubbeldam 2004 set | +446 | −1648 | 2094 | – |

What the table shows:

* **Si term.** On its own it contributes −66 K (window), −51 to −53 K (P2 minimum) and −25 K (cage
  centre). It lowers the barrier by 13–14 K. This is needed for fidelity but is second order.
* **Truncation convention.** It changes the barrier by ≤ 1 K, though absolute energies move by about
  460 K. It does not block an F(q)-shape comparison.
* **Framework choice.** The barrier changes by 225 K (565 vs 340 K), i.e. 0.37 kBT per bead at
  600 K. This is first order.
* **Parameter set.** The barrier changes about 5× and the window becomes repulsive for a single bead.
  Do not substitute the LTA engine's force field.
* Expanding the asymmetric unit initially gave 78 O per cell for Gard & Tait. The cause is that O5
  (0.2293, 0.4585) sits off the 6h special position y = 2x by 1e-4. A 2e-3 de-duplication tolerance
  restores 72 O. **Anyone building the Gard & Tait framework must de-duplicate with a tolerance of
  about 1e-3; a 1e-4 tolerance silently duplicates 6 O atoms per cell.** That duplication shifted the
  single-bead barrier by about 6 K in my first run.

Scripts: `xcheck_eri.py`, `xcheck_pockets.py` and `xcheck_trunc.py` in this session's scratchpad. The
scratchpad is not persistent; every number needed is in this section.

---

## 6. Errors and over-statements found in the four reports

1. **FF-A U1 and FF-B item 1 ("Schüring 2002 unobtainable")** are superseded: the paper is obtained,
   and it answers both reports' Schüring sub-questions (Sec. 3.2).
2. **FF-B §3, Dubbeldam 2004 mixing** "numerically Lorentz–Berthelot" is wrong for pairs with unequal
   σ: the table is geometric in σ. FF-A is correct.
3. **FF-B §0.1** calls the bond "Dubbeldam's harmonic" without evidence. Schüring 2002 now confirms
   it.
4. **FF-A §8a, cutoff "L (inferred)"** understates the evidence: JPCB 2006 states 12 Å for this very
   study.
5. **PNAS-X Sec. 1** labels `pnas2005/uu_dspace_2003-007/2003-007.pdf` as JPCB 2004 "OK". It is the
   Surface Science GaP paper.
6. **PNAS-X Sec. 7 and Sec. 14 item 5**, "xy mapping not written down", is partly too strong (see
   Sec. 4).
7. **STRUCT-X:** no numerical error found. Every number I recomputed (window and cage distances, the
   579/353 K barriers, the four pocket energies) reproduces exactly. Its energies are O-only (Si adds
   about −14 K to the barrier) and unshifted (barrier-neutral).
8. No report mentions that Schüring's original model used a **Morse bond, a flexible lattice and 2.5σ
   shifted-force LJ**. Any statement that the campaign uses "the Schüring et al. model" would be
   inaccurate. The correct label is "Schüring et al. LJ parameters as used by Dubbeldam et al. 2005".

---

## 7. Unresolved points that block an exact reproduction

Ordered by expected impact on the infinite-dilution F(q) at 600 K, the target for a
one-ethane-per-replica engine.

1. **B1. Framework identity (first order).**
   * Gard & Tait 13.27 / 15.05 (cited; JPCB 2006 box) vs Schlenker 1977 13.252 / 14.81 (PNAS box;
     2003 and 2006 group cells) vs Gard & Tait fractional coordinates in the Schlenker cell.
   * The single-bead window barrier spans 340–565 K (O + Si); STRUCT-X gives the hybrid as about
     666 K, O-only.
   * Not recoverable from any source I could obtain. I tried the PNAS / JPCB 2006 / JPCB 2003 /
     Beerdsen 2006 text, the archived IZA records (2005–2006; no pre-2005 ERI capture per STRUCT-X),
     COD and Crossref.
   * The Dubbeldam thesis (UvA "fully internal") might hold the answer.
2. **B2. xy reaction-coordinate details.**
   * Which bead is used (immaterial by symmetry in equilibrium if both beads are histogrammed).
   * How the three window directions are folded onto one line.
   * What qA′ / qB′ are.
   * Bin widths.
   * Affects only the comparison of profile shapes; the barrier height at q* is robust.
3. **B3. Truncation convention** (energy-shifted assumed; not stated).
   * Barrier-neutral to ≤ 1 K for a single bead.
   * It does change absolute adsorption energies, so it matters for anything that depends on the
     number of molecules adsorbed: loading assignments, isotherms.
4. **B4. Loaded cases only.**
   * The box, and hence N = 72 L or 54 L.
   * Assigning the six unlabelled 1–6 /cage curves.
   * No isotherm, so no pressure ↔ loading map.
5. **B5. Pocket blocking** is unstated. It is immaterial for ethane in MD/ABF started inside an ERI
   cage, but relevant if MC insertion or a smaller probe is used.
6. **B6. Infinite-dilution protocol** for the profiles (non-interacting MD as in JCP 2005, or one
   molecule) is unstated. Immaterial in exact equilibrium.
7. **B7. CH3 mass, thermostat and integrator.** Immaterial to equilibrium F(q). They matter only for
   any dynamic comparison (D, κ, k_z). For those, also note that the JPCB 2006 text D values are 10×
   wrong.
8. **B8. Gard & Tait 1973 primary** not obtained. The coordinates rest on two consistent copies
   (archived IZA record 55, RASPA2).
9. **B9. Upstream sources of Schüring's O and Si self-parameters** (Fritzsche et al. 1993; Bezus
   et al. 1978) were not retrieved. This does not block anything: the PNAS pair values are explicit.
10. **B10. Premise risk, not a reproduction issue.** The literature never shows that the ERI barriers
    are entropic (Sec. 4.1). For LTA with this parameter set it does (Schüring 2002).

---

## 8. Recommendation

**The PNAS 2005 ERI model cannot be reproduced exactly. It can be reproduced as a labelled
surrogate, and the label must name the framework choice.**

Recommended specification: "PNAS-2005-type ERI model", every item from Sec. 2.

* **Force field and conditions:**
  * Schüring LJ pairs as stated in PNAS: O–CH3 142 K / 3.17 Å, Si–CH3 82 K / 2.12 Å, CH3–CH3
    104 K / 3.78 Å.
  * O + Si interacting; no charges; rigid all-silica framework.
  * Harmonic bond ½·(96 500 K/Å²)·(r − 1.54 Å)².
  * LJ energy-shifted at 12 Å, no tail corrections.
  * T = 600 K for any literature comparison.
* **Framework, run both variants:**
  * (a) Gard & Tait coordinates (RASPA2 `ERI_SI.cif` with charges removed, de-duplication tolerance
    about 1e-3) in a box whose perpendicular widths are all > 24 Å. Either 3×3×3 hexagonal as in
    JPCB 2006, or 2√3a × 3a × 3c = 45.97 × 39.81 × 45.15 Å.
  * (b) Schlenker 1977 (COD 2106432, majority framework sites, all T as Si) in the PNAS box
    45.906 × 39.756 × 44.43 Å.
  * Optionally (c), the hybrid.
* **Labelling:** do not call either variant "the PNAS model". Agreement of one variant with the
  digitized 6.13 kBT infinite-dilution xy barrier counts as evidence for that variant. It does not
  prove it.
* **Literature comparison:**
  * Use histograms of a single bead's fractional position, as in JPCB 2006 Eqs. 10–11. The campaign's
    own ABF CV may differ (e.g. COM), but it must then not be compared directly.
  * Compare only barrier heights relative to each curve's minimum: 6.13 kBT xy and ≈ 1.1 kBT z at
    infinite dilution.
* **Do not** carry the repo LTA engine's Dubbeldam-2004 parameters, 10 Å cutoff or 400 kJ mol⁻¹ Å⁻²
  bond into ERI under any PNAS label. As a single-bead indication, they give a ≈ 2.7 × 10³ K
  repulsive window instead of ≈ 0.57 × 10³ K.
* **Before relying on the "entropic barrier" premise in ERI,** decompose the barrier: compute U(q)
  along the profile and −TS(q) = F − U, or the temperature dependence of βF(q*).
