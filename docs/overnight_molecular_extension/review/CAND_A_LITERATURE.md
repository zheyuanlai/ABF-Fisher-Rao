# Candidate A: ethane in Si-ITE / Si-IHW. Literature and physical-model audit

Written 2026-10-16. Literature retrieval, structure download and CPU-only geometry only: no simulation was run, no existing file was modified, and nothing was committed.

**Bottom line**
1. The primary paper cannot be reproduced with a rigid model. Its abstract says the framework was **fully flexible (BKS)** and the guest was the **flexible Oie et al. potential**, which has electrostatic terms (all-atom). A rigid-framework, united-atom ethane model would therefore be a *new model inspired by* the paper, not a reproduction.
2. The full text and SI of the paper could not be obtained (see section 0). Every numerical setting except the framework model (host–guest parameters, cutoff, temperatures, loadings, timestep, thermostat, structure source, reported D values) is **unrecoverable** here.
3. Geometry was computed on the IZA DLS-idealised and the IZA type-material (experimental) structures:
   - **ITE**: for any ethane-sized guest, ITE is a set of independent 1D cage–window channels along **b**. There is one 8-ring per period of b = 9.72 Å (exp) / 9.80 Å (DLS). Free diameter of the [010] 8-ring: 3.79 Å (exp) vs 4.16 Å (DLS).
   - **IHW**: IHW is a 2D layer of 12-ring-bridged cage *dimers* linked by 8-rings (free D 3.55 Å exp / 3.61 Å DLS). An x-projection has period a/2 = 6.85 Å (exp) / 6.87 Å (DLS) and one 8-ring per period.
4. Recommendation: **Si-ITE built from the ITQ-3 type-material coordinates (Camblor et al. 1997)**, with ξ = signed Cartesian y. The caveats are in section 6.

---

## 0. Retrieval log: what was tried and what came back

All downloads were saved under `cache/overnight/literature/candA/<one new dir per source>/` and were read only with `python -I`. sha256 prefixes are given in section 8.

| Source tried | Result |
|---|---|
| ACS landing page `pubs.acs.org/doi/10.1021/jp102262n` (web extractor) | Gave the abstract, author list, history and subject tags. Full text is paywalled ("Available to Purchase"). |
| ACS `/doi/full`, `/doi/pdf` | Abstract only. |
| ACS SI `.../suppl/10.1021/jp102262n/suppl_file/jp102262n_si_001.pdf` and `/doi/suppl/...` | HTTP 403 (Cloudflare) from both curl and the extractor. **Whether an SI exists at all could not be determined.** The saved `acs_si/*.bin` files are the 403 HTML pages, not SI. |
| UPV repository RiuNet, record hdl 10251/28708 (entity 76fd290c-…) | Metadata and abstract were obtained. The PDF bitstream `J_Phys_Chem_C115(4)_875_884.pdf` (544 KB) is listed with access rights "Cerrado" and embargo "forever". The download link is "request-a-copy" and leads to a UPV login. curl hits an Anubis proof-of-work bot wall; I did not attempt to defeat it. |
| Unpaywall API | Reports "green" OA, but the only location is the closed RiuNet record (`url_for_pdf: null`). |
| Semantic Scholar API | `openAccessPdf` points to the same closed RiuNet handle. The citation contexts were fetched (36 citing papers); none quotes ethane numbers for ITE/IHW. |
| Crossref API | Full **reference list (46 refs)** obtained (`crossref/crossref.json`) and used in section 1. |
| ResearchGate | The paper page failed (fetch error). A figure page of the authors' 2013 Chem. Soc. Rev. review (Combariza, Gomez, Sastre, Chem. Soc. Rev. 42, 114, doi 10.1039/c2cs35243e) did return *context excerpts*; these are used in section 4. |
| Review full text (RiuNet hdl 10251/70227) | Bot wall / no PDF. |
| Combariza PhD thesis (UPV, about 2012) | Not found in any index I searched. |
| Internet Archive Scholar, fatcat, COD | IA Scholar returned a bot-verification page. fatcat and COD connections failed (HTTP 000). |
| Chrome extension (institutional access) | No browser connected. |
| Related OA papers from the same group | Ghysels et al. JPCC 2015 (UGent preprint, read in full); Alaithan, Harrison, Sastre JPCC 2021 (methods via the extractor). Used only as context on the group's methods; they are **not** the 2011 setup. |
| IZA-SC database | DLS framework CIFs for ITE, IHW and LTA came from the europe, america and www mirrors, all byte-identical. The asia mirror timed out. Type-material CIFs (ITQ-3 calcined; ITQ-32) via `mat_cif.php`. Framework pages give cell, Di/Df and accessible volume. |
| IZA Atlas 6th ed. PDF | curl returned 404. The extractor returned the ITE/IHW/LTA channel lines, which are used below. |
| RASPA2 (github iRASPA/RASPA2, master) | `structures/zeolites/cif/ITQ-3.cif` and `block/ITQ-3.block`, plus `LTA_SI.cif` and `LTA_SI.block`. **There is no IHW or ITQ-32 file in RASPA2** (I searched the whole tree). The `share/raspa/structures/...` paths in the task do not exist in the repo (404). |

**Quotation policy note.** The task asked for exact quotes. The copyright rule I operate under allows only one short verbatim quote, so everything else is paraphrased with a precise location; each location can be checked directly.

---

## 1. Paper identity, systems, method, what is reported

- **Citation:** Combariza, A. F.; Sastre, G.; Corma, A. "Molecular Dynamics Simulations of the Diffusion of Small Chain Hydrocarbons in 8-Ring Zeolites", *J. Phys. Chem. C* **2011**, 115(4), 875–884, doi 10.1021/jp102262n.
  - Affiliation: ITQ (UPV–CSIC), Valencia.
  - History: received 12 Mar 2010, revised 23 Aug 2010, online 23 Sep 2010, issue 3 Feb 2011.
  - Part of the "Alfons Baiker Festschrift".
  - Source: ACS landing page, author and history block.
- **Guests:** methane, ethane, propene, propane. **Frameworks:** pure-silica Si-LTA, Si-IHW, Si-ITE, all with 8-ring windows. Source: abstract, sentence 1.
- **Model:** both framework and guests are fully flexible. The framework uses the BKS model and the guests the potential of Oie et al. The one verbatim quote is from abstract sentence 2: *"simulated allowing full flexibility, using the well-known BKS model"*.
- **What is reported** (abstract, sentences 3–6):
  - Intra-cage and inter-cage dynamics of methane and ethane are obtained over statistically meaningful times "in the temperature range studied". For the larger guests, inter-cage motion is analysed only at higher temperatures.
  - **Self-diffusion coefficients** of methane, ethane and propene decrease as the guest kinetic diameter grows and as the critical window shrinks, consistent with experimental sieving.
  - Propane in Si-ITE diffuses somewhat anomalously, which the authors relate to the levitation effect.
  - The dimensionality of the diffusion path is the dominant topological influence.
- **Probably also reported, inferred from metadata only (low confidence):**
  - ACS subject tags include "Activation energy", so Arrhenius activation energies of D are probably given.
  - Ref. 30 (Shell, Panagiotopoulos, Pohorille, "Methods based on probability distributions and histograms") suggests free-energy profiles from position histograms.
  - Refs 16 and 46 (Beerdsen, Dubbeldam, Smit, PRL 2004 and 2005) suggest a free-energy-barrier, TST-type discussion of cage/window zeolites.
  - Refs 39–43 (Derouane; Derycke) are the levitation / "floating molecule" literature.
  - **None of these numbers was retrievable.** I do not know whether barriers are given in kJ/mol or kBT, nor whether hopping rates or dcTST were computed.
- **Temperatures, loadings, ensemble, timestep, thermostat, run length:** **unrecoverable.**
  - The abstract mentions only "the temperature range studied" and higher temperatures for the larger guests.
  - The MD code is probably DL_POLY: ref. 26 is Smith et al., Mol. Simul. 28, 385 (2002), and the group's homepage says it uses DL_POLY.
  - Ewald summation is probably used: ref. 31 is Ewald 1921.
  - Context only, not transferable: the group's later work used DL_POLY NVT with Nosé–Hoover, 1 fs, SPME, cutoff 11.7–12 Å (Ghysels 2015, section "D. Molecular dynamics"). Alaithan 2021 used NVE, 0.25 fs, 9 Å cutoff and Ewald. The 2013 review's own window-aperture MD was NVE at 300 K with 5.5e5 steps. These are different studies.
- **Structure-related references** in the reference list:
  - ref. 8: Camblor et al., Angew. Chem. 1997, 36, 2659 (ITQ-3, i.e. ITE)
  - ref. 9: Cantín et al., JACS 2005, 127, 11560 (ITQ-32, i.e. IHW)
  - ref. 24: Baerlocher et al. 2007 (Atlas, 6th ed.)
  - ref. 6: Olson et al., MMM 2004, 67, 27 (light-hydrocarbon sorption in Si-CHA and ITQ-3, experimental context)
  - Which coordinates were actually simulated (DLS vs experimental; BKS-relaxed or not) is **unrecoverable**.
- **LTA context:** Si-LTA (ITQ-29) is the 3D reference system: α-cages joined by circular 8-rings, sodalite cages inaccessible.

## 2. Force field (what can and cannot be established)

| Item | Finding | Basis |
|---|---|---|
| Framework rigid or flexible | **Flexible** in the reported results | Abstract, sentence 2 |
| Framework FF | **BKS** (van Beest, Kramer, van Santen, PRL 64, 1955 (1990)) = Buckingham + Coulomb, charges q_Si = +2.4 e, q_O = −1.2 e | Abstract plus ref. 32. Numbers verified in a secondary source (arXiv:1109.2852, Table I) |
| BKS short-range modification (r⁻²⁴ wall, splines) | Unknown | Not in abstract |
| Guest FF | **Oie, Maggiora, Christoffersen, Duchamp, IJQC QBS 8, 1 (1981)** | Abstract plus ref. 33 |
| Oie functional form | Flexible. Intramolecular: harmonic bonds and angles, out-of-plane and torsional terms. Intermolecular: nonbonded, hydrogen-bond and **electrostatic** terms. Atom-based, so C and H are explicit (all-atom), **not united-atom** | OSTI abstract of Oie 1981. The same group applies Oie to explicit C/H ethene, propene and propylene (Ghysels 2015, Alaithan 2021) |
| Oie numeric parameters for ethane (charges, vdW, bonds) | **Unrecoverable** (the Oie paper is paywalled; the group's DL_POLY FIELD files in the 2021 SI return 403) | |
| Host–guest LJ/Buckingham parameters; which framework atoms carry them (O only or also Si) | **Unrecoverable.** Ref. 34 (Catlow et al., J. Chem. Soc. Faraday Trans. 87, 1947 (1991), hydrocarbon MD in zeolites) and ref. 35 (Llopis, Sastre, Corma, J. Catal. 2004) are plausible sources, but this is *unverified* | |
| Host–guest electrostatics | Likely present (BKS framework charges and Oie partial charges, Ewald cited), but unverified | |
| Mixing rules, cutoff, shifting, tail corrections | **Unrecoverable** | |
| Rigid-framework comparison in the paper | Not stated in the abstract; unknown. The group's 2013 review argues that window "breathing" matters for C3 molecules in 8-ring silicas and that a rigid framework with temperature-correct geometry is adequate only for guests clearly smaller than the window (review section on pore size / framework dynamics, via ResearchGate context excerpts) | |

**Consequence.** A rigid IZA framework with a united-atom ethane (TraPPE/Dubbeldam-type) is **not a reproduction** of this paper. At best its diffusivities can be compared *qualitatively* with the paper's trends (ITE vs IHW vs LTA ordering). Flexibility is known to raise D for tight 8-ring windows; for example, Gutiérrez-Sevillano et al. 2016 find about a 2x effect for methane in Si-LTA.

## 3. Framework structures

I took the cell parameters and asymmetric units below directly from the downloaded CIFs. In every case my own expansion with the symmetry operators reproduces the expected cell contents: **ITE 64 Si + 128 O; IHW 112 Si + 224 O; LTA 24 Si + 48 O**. Each Si has 4 O at 1.51–1.67 Å, and no fifth O lies closer than 3.18 Å.

| | ITE DLS (IZA) | ITE type material ITQ-3 calcined | IHW DLS (IZA) | IHW type material ITQ-32 |
|---|---|---|---|---|
| Space group | Cmcm (63) | Cmcm (63), Hall −C 2c 2 | Cmce (64) ('C m c a') | Cmca (64), Hall −C 2bc 2 |
| a, b, c (Å) | 20.7530, 9.8040, 20.0090 | 20.622, 9.7242, 19.623 | 13.7470, 24.0690, 18.3290 | 13.6988, 24.0665, 18.1968 |
| V (Å³) | 4071.1 | 3935.0 | 6064.6 | 5999.2 |
| FD (T/1000 Å³) | 15.72 (IZA 15.7) | 16.26 (Atlas 16.3) | 18.47 (IZA 18.5) | 18.67 |
| Si–O (Å), min/max/mean | 1.626/1.629/1.628 | 1.510/1.665/1.603 | 1.608/1.613/1.610 | 1.568/1.645/1.595 |
| Si–O–Si (deg) | 137.8–156.4 | 141.1–158.9 | 131.9–168.3 | 134.2–173.3 |
| Source file | `iza_ITE_europe/ITE.cif` | `iza_typematerial_ITE/mat_cif.bin` (identical coordinates to RASPA2 `ITQ-3.cif`, which has Si1 z = 0.17011 vs 0.1701) | `iza_IHW_europe/IHW.cif` | `iza_typematerial_IHW/mat_cif.bin` |

The type-material CIFs contain no symmetry loop. I combined them with the IZA operator list for the same standard setting. The derived files are `candA_scratch/ITQ3_tm_with_IZAops.cif` and `ITQ32_tm_with_IZAops.cif`, and their cell contents and Si–O distances are correct.

**Asymmetric units (fractional).**

ITE DLS (IZA): O1 0.2614 0.8673 0.1438; O2 0.3314 0.9350 0.2500; O3 0.2941 0.1262 0.1640; O4 0.3833 0.9457 0.1288; O5 0.0987 0.5000 0.0000; O6 0.0000 0.4222 0.0758; O7 0.0920 0.2447 0.0436; O8 0.3270 0.2763 0.0588; O9 0.2118 0.1581 0.0637; O10 0.1278 0.0000 0.0000; T1 0.3175 0.9685 0.1717; T2 0.0767 0.4032 0.0620; T3 0.2677 0.2322 0.1076; T4 0.1512 0.1565 0.0121.

ITE type material (Camblor 1997): Si1 0.32018 −0.043 0.1701; Si2 0.07561 0.4067 0.0561; Si3 0.2641 0.2255 0.1121; Si4 0.15302 0.1551 0.0036; O1 0.3427 −0.0581 0.25; O2 0.2675 −0.1485 0.1491; O3 0.0996 0.5 0; O4 0.0978 0.2516 0.0287; O5 0 0.4105 0.0718; O6 0.2148 0.1587 0.0569; O7 0.387 −0.0628 0.1288; O8 0.2913 0.1093 0.1625; O9 0.3275 0.2871 0.0695; O10 0.1308 0 0.

IHW DLS (IZA): O1 0.6473 0.5000 0.5000; O2 0.5000 0.4547 0.4308; O3 0.6320 0.5099 0.3563; O4 0.6751 0.4139 0.4133; O5 0.7500 0.5425 0.2500; O6 0.7415 0.5975 0.3749; O7 0.5958 0.6011 0.2831; O8 0.7738 0.6504 0.4989; O9 0.7427 0.7074 0.3786; O10 0.9046 0.6503 0.3970; O11 0.7356 0.6845 0.6320; O12 0.5955 0.6767 0.5334; O13 0.7500 0.7545 0.2500; O14 0.5962 0.7685 0.3366; O15 0.5000 0.7535 0.4602; O16 0.5000 0.7543 0.6037; O17 0.5000 0.6763 0.3607; O18 0.5000 0.6816 0.2194; T1 0.6137 0.4701 0.4256; T2 0.6807 0.5624 0.3158; T3 0.7893 0.6516 0.4119; T4 0.6947 0.6497 0.5636; T5 0.7128 0.7618 0.3326; T6 0.5000 0.7153 0.5324; T7 0.5000 0.7426 0.3735; T8 0.5000 0.6393 0.2874; T9 0.0000 0.6823 0.3684.

IHW type material ITQ-32 (Cantín 2005): Si1 0.618 0.4716 0.4246; Si2 0.6828 0.565 0.3182; Si3 0.7896 0.6549 0.4144; Si4 0.6939 0.6486 0.5655; Si5 0.7148 0.7619 0.3319; Si6 0.5 0.7135 0.5365; Si7 0.5 0.7412 0.3736; Si8 0.5 0.6377 0.2844; Si9 0 0.6805 0.3713; O1 0.6433 0.5 0.5; O2 0.5 0.459 0.4262; O3 0.6423 0.5109 0.3568; O4 0.6805 0.4145 0.4134; O5 0.75 0.5495 0.25; O6 0.7423 0.6056 0.3702; O7 0.5919 0.5983 0.2863; O8 0.7669 0.6526 0.4991; O9 0.746 0.7105 0.3832; O10 0.907 0.6497 0.4025; O11 0.7371 0.6812 0.6358; O12 0.5936 0.676 0.5384; O13 0.75 0.7581 0.25; O14 0.5971 0.7675 0.3396; O15 0.5 0.7494 0.4614; O16 0.5 0.756 0.605; O17 0.5 0.6762 0.3588; O18 0.5 0.6797 0.2166.

**IZA / Atlas pore descriptors.** From the IZA framework pages, data last updated 1 Jul 2007; Atlas 6th ed. type-material pages.
- **ITE**:
  - Atlas channels: [010] 8-ring 3.8 × 4.3, interconnected with [001] 8-ring 2.7 × 5.8 (2D).
  - IZA: included sphere 8.3 Å; diffusing sphere a 2.07, b 4.21, c 2.8 Å; accessible volume 16.45 %; CBU rth.
- **IHW**:
  - Atlas: [100] 8-ring 3.5 × 4.3, marked 2D. The Atlas note says there are 12-ring connections between the [100] channels but no continuous 12-ring channel.
  - IZA: included 6.67 Å; diffusing a 3.67, b 1.58, c 3.67 Å; accessible volume 9.70 %; CBUs cas, non, ton.
- **LTA**: <100> 8-ring 4.1 × 4.1 (3D); IZA included 11.05 Å, diffusing 4.21 Å.

**RASPA2 ITQ-3 extras.**
- The RASPA2 CIF carries charges Si +2.05 / O −1.025 (a CO2-type convention, irrelevant for alkanes).
- `ITQ-3.block` lists 8 spheres of radius 2.0 Å at fractional (0.2083|0.7917, 0.5, 0|0.5) and (0.2917|0.7083, 0, 0|0.5).
- I computed the free radius at these centres: r_free = 0.89 Å, nearest O at 2.24 Å. They are tiny voids (free D ≈ 1.8–2.2 Å), not cages (section 4).

## 4. Pore geometry (own computation)

**Method.** Script: `/tmp/claude-1008/-home-zheyuanlai-ABF-Fisher-Rao/candA_scratch/cifgeom.py`.
- The symmetry-operator strings are tokenised term by term, with no eval.
- Free-radius field r_free(x) = min_O |x − O| − 1.35 Å (the IZA convention), computed on an origin-aligned 0.2 Å periodic grid.
- A union-find merge tree carrying lattice offsets gives cage peaks, saddle (window) values and the percolation threshold per lattice direction.
- Each window is then refined by maximising r_free in the plane normal to the hop.

**Validation on LTA:** included sphere 10.99 Å (IZA 11.05) and 8-ring 4.15 Å (IZA 4.21). The sodalite cage (D 6.18 Å) is correctly found to be isolated, which is the standard blocking case. My diffusing diameters run about 0.05–0.25 Å below IZA's; the refined window values below are the reliable ones.

### 4.1 ITE

**Cages.** Four cages per conventional cell.
- DLS: D = 8.24 Å, at (0, 0.0653, ¾), (0, 0.9347, ¼), (½, 0.4347, ¼), (½, 0.5653, ¾).
- Exp: D = 8.08 Å, at (0, 0.0112, ¾), (0, 0.9888, ¼), (½, 0.4888, ¼), (½, 0.5112, ¾).
- Each cage sits on the mirror planes x = 0 or ½ and z = ¼ or ¾.

**Window types.** Free D is the refined maximin diameter; the aperture column is O–O minus 2.7 Å.

| Window | DLS free D (Å) | Exp free D (Å) | Aperture | Connects |
|---|---|---|---|---|
| [010] 8-ring, planar (rms 0.10–0.13 Å), normal ∥ b | **4.16** | **3.79** | 4.15 × 4.30 DLS / 3.79 × 4.29 exp (Atlas 3.8 × 4.3) | Cage → same cage translated by ±b |
| [001] 8-ring, puckered (rms 0.55–0.58 Å) | 2.74 | 2.83 | Atlas 2.7 × 5.8 (my cross-distance metric under-reads the long axis of puckered rings) | Cage at z = ¼ ↔ cage at z = ¾ in the same x-plane; Δy = ∓1.23 Å (DLS) / ∓0.41 Å (exp) |
| Ring for a-direction transport | 1.9–2.0 | 1.87 | — | Cages at x = 0 ↔ x = ½. Hop vector (a/2, −0.375b, c/2) in DLS; (a/2, b/2, 0) in exp |

**Connectivity and dimensionality.**
- At free-diameter thresholds D > 3.0 Å, the void space breaks into **four disconnected 1D channels per cell, each running along b**. The channels sit at (x, z) = (0, ¼), (0, ¾), (½, ¼), (½, ¾).
- Only at D < 2.7–2.8 Å do the [001] windows join them into 2D b–c sheets, which is the "2D" of the Atlas.
- For ethane in a rigid framework, the [001] window is closed. With D ≈ 2.7–2.8 Å, a CH3 site in the ring centre sits about 2.7 Å from two O atoms, deep in repulsion. **ITE is therefore pseudo-1D along b**, and the **rate-limiting and only productive hop is through the [010] 8-ring.**
- The authors' own group says the same thing in its 2013 Chem. Soc. Rev. review (section on the structures studied, ITQ-3 paragraph, via the ResearchGate context excerpt): the window sizes in ITQ-3 let hydrocarbons diffuse only along b, so the cages form a pseudo-1D arrangement.

**Blocking.** At the D > 2.0 Å threshold, apart from the 1D channels there are only **16 tiny isolated voids per cell**:
- Peak free D is 2.29 Å (DLS) / 2.23 Å (exp), with ≤ 0.25 Å³ above the threshold.
- In the exp structure they lie in pairs 0.87 Å from each of the 8 RASPA2 block centres, so the RASPA2 2.0 Å blocks are exactly these voids.
- Ethane cannot reach or fit them. Blocking is unnecessary for MD started in a cage, but advisable if insertion or MC moves are used.

**Channel profile along b**, with D(y) = max r_free over the channel cross-section (`profiles.py`):
- DLS: cage (D 8.21) and window (D 4.16) are about 4.8–5.0 Å apart (window at y = 0.4477, cage at y = 0.9347 in channel (0, ¼)).
- Exp: the separations are 4.36 / 5.36 Å.
- **The profile within a channel is not mirror-symmetric in y.** No space-group operation maps a channel onto itself with y → −y. The operations that reverse y map the z = ¼ channel onto the z = ¾ channel.

### 4.2 IHW

**Cages.** Eight cages per cell, D = 6.61 Å (DLS) / 6.48 Å (exp), arranged as **4 dimers per cell**.
- Each dimer is two cages joined through a **12-ring opening of free D 5.87 Å (DLS) / 5.73 Å (exp)**, centred at (0, ½, ½), (½, ½, 0), (0, 0, 0) and (½, 0, ½).
- DLS example: cages (0, 0.4648, 0.3877) and (0, 0.5351, 0.6121), with the 12-ring at (0, ½, ½).
- This matches the 2013 review's description of IHW: unidirectional 8-ring channels along a, crossed by a 12-ring cavity about 6.3 Å by 16.2 Å that joins neighbouring 8-ring channels along c.

**Windows.**
- **8-ring, free D 3.61 Å (DLS) / 3.55 Å (exp)**. Aperture 3.63 × 4.23 (DLS) / 3.54 × 4.31 (exp), matching the Atlas 3.5 × 4.3. Puckered (rms 0.49–0.54 Å), with the normal tilted in the a–c plane (±0.93, 0, ±0.36).
- It sits at a general position near x ≈ a/4 and links a cage to a cage of the neighbouring dimer. The hop vector is (±a/2, 0, ±4.96–5.00 Å).
- A third ring (free D 1.52–1.54 Å) joins different layers. It is closed.

**Connectivity.**
- For 3.6 Å > D > ~1.6 Å the network is **two independent 2D layers per cell, perpendicular to b** (at y ≈ 0 and y ≈ ½), percolating in a and c.
- Above about 3.6 Å the dimers are isolated. Below about 1.5 Å the layers would join.
- **The rate-limiting hop is the inter-dimer 8-ring.** The intra-dimer 12-ring hop is essentially barrier-free for ethane.

**Blocking.** At D > 2.0 Å the isolated pockets per cell are:
- 8 with free D 2.64 Å (DLS) / 2.72 Å (exp), about 1.6–2.1 Å³ each above the threshold;
- 16 (DLS) to 24 (exp) more with D ≈ 2.15 Å.

None is ethane-sized (a CH3 site needs free D of about 3.5 Å or more), so blocking is only cosmetic. It would matter only for MC insertion of smaller probes.

**Layer profile along a** (max r_free over the layer cross-section):
- D(x) = 6.53, 5.52, 4.13, 3.58, 3.53, 3.58, 4.13, 5.52, 6.53 Å at x/a = 0, 1/16, …, 1/2 (DLS). The exp structure is almost identical.
- The profile is **symmetric, with period a/2**.

### 4.3 Proposed reaction coordinates

**ITE: ξ = s·(y − y₀), the Cartesian projection on b, period b** (9.7242 Å exp; 9.804 Å DLS).
- s = +1 in the channels (0, ¼) and (½, ¼); s = −1 in the channels (0, ¾) and (½, ¾).
- y₀ is the cage-centre y of that channel: 0.9888b, 0.4888b, 0.0112b and 0.5112b respectively (exp).
- The sign makes the four channels share one F(ξ), because they are related by y-reversing operations and by the C-centring shift of ½b in y.

Why one window per step, and no hidden window type:
- (i) Along a channel there is exactly one cage and one [010] 8-ring per period b, so every unit change of ⌊ξ/b⌋ crosses exactly one [010] ring.
- (ii) The only other windows an ethane could conceivably cross are the [001] rings (D ≈ 2.7–2.8 Å). Crossing one moves the molecule to the mirror channel with |Δy| ≤ 1.23 Å and **zero net lattice translation along b**: successive [001] hops zig-zag between the z = ¼ and z = ¾ channels. So they cannot produce progress in ξ.
- The a-direction rings (D ≈ 1.9–2.0 Å) are closed.
- Progress along ξ therefore requires [010] crossings and nothing else.
- **Caveat.** If a model ever lets ethane pass the [001] ring, the hidden variable "which channel" becomes dynamic. The unsigned y-profile would then mix F(y) with its mirror image. The signed ξ handles this only if the channel label is tracked. Verify with an energy scan that the [001] ring is closed before production.

**IHW: ξ = x, the Cartesian projection on a; the F(ξ) period is a/2** (6.8494 Å exp; 6.8735 Å DLS).
- The glide (x + ½, −y, z + ½) maps each layer onto itself with x → x + a/2, and the C-centring maps one layer onto the other. So both layers share one symmetric F(x).
- Every 8-ring hop changes x by exactly ±a/2.
- The 12-ring intra-dimer hop has Δx = 0, and the inter-layer ring is closed.
- So each half-period step in x crosses exactly one 8-ring, and no other window type can make progress in x.
- **Mild orthogonal issue.** At x ≈ a/4 the conditional distribution covers two mirror-related 8-rings (at x/a ≈ 0.234 and 0.266) that lead to different neighbouring dimers. They mix only through the adjacent dimer, which is fast because the 12-ring is open.

## 5. Feasibility for the CPU engine (rigid framework, single ethane)

Script: `feas.py`. "Pairs per bead" is the mean number of framework atoms within r_c of 400 random pore points with r_free > 1 Å, counted with explicit periodic images.

| System | atoms/cell | r_c (Å) | Supercell for all widths > 2r_c | Framework atoms | O within r_c / bead | Si within r_c / bead |
|---|---|---|---|---|---|---|
| ITE exp | 192 | 10 | 1×3×2 | 1152 | 144 | 73 |
| ITE exp | 192 | 12 | 2×3×2 (41.2×29.2×39.2 Å) | 2304 | 235 | 117 |
| ITE exp | 192 | 14 | 2×3×2 | 2304 | 364 | 181 |
| ITE DLS | 192 | 12 | 2×3×2 | 2304 | 228 | 114 |
| IHW exp | 336 | 12 | 2×1×2 (27.4×24.07×36.4 Å; b only just > 24) | 1344 | ≈274 (DLS value) | ≈137 |
| IHW DLS | 336 | 12.8 | 2×2×2 | 2688 | 326 | 163 |
| IHW DLS | 336 | 14 | 3×2×2 | 4032 | 419 | 209 |
| LTA DLS (context) | 72 | 12 | 3×3×3 | 1944 | 209 | 105 |

**Cost of each model.**
- **United-atom ethane, O-only host–guest:** about 2 × 230 ≈ 460 (ITE) or 2 × 275 ≈ 550 (IHW) LJ pairs per step at r_c = 12 Å.
- **Rigid framework with tabulation:** the whole host–guest energy can be pre-tabulated per bead type on a 0.1–0.2 Å grid of the unit cell (ITE about 4.1e6 points at 0.1 Å). The cost then drops to two interpolations per step, with no dependence on r_c or supercell.
- **The paper's model (flexible BKS + all-atom Oie + Ewald)** needs MD of the whole charged framework (about 2300 atoms at r_c = 12 Å) at a timestep of order 1 fs. That is out of scope and not useful for an ABF/FR benchmark.

**Is infinite dilution representative?**
- It cannot be judged: the paper's loadings are unrecoverable.
- Even at low loading, a rigid infinite-dilution ethane gives the zero-loading free-energy profile and hop rate *of a different (rigid, UA) model*. It is not comparable to the flexible-BKS diffusivities except in trend.
- For the paper's flexible model, the window apertures fluctuate. My static apertures show how sensitive ITE is to this: free D is 3.79 Å (exp) vs 4.16 Å (DLS), a 0.37 Å difference. IHW differs by only 0.06 Å.

## 6. Recommendation

**Si-ITE, built from the ITQ-3 type-material coordinates (Camblor et al. 1997).** The IZA `mat_cif.php?ID=133` file matches RASPA2 `ITQ-3.cif`. Use ξ = signed y, with period b = 9.7242 Å.

The choice rests on model reproducibility and feasibility only:
- (a) The transport topology is strictly 1D for ethane in a rigid framework: one window type and one cage per period. So a single linear CV captures all long-range transport, and the authors' group describes ITE in the same pseudo-1D terms.
- (b) The structure exists in two independent public sources (IZA type material and RASPA2) with identical coordinates, plus a RASPA2 blocking file, which allows a third-party cross-check.
- (c) The cell is smaller (192 atoms) and needs fewer pairs per bead than IHW (about 230 vs 275 O at 12 Å).

**IHW is the fallback.** It is preferable if structure-robustness of the window matters more than 1D simplicity: its 8-ring free D agrees within 0.06 Å between DLS and experiment, and its F(x) is symmetric and identical for all layers. However, it has 2D transport, the conditional distribution at x ≈ a/4 mixes two windows, and it has no RASPA2 file.

**ITE caveats that must be pre-registered:**
1. State which coordinates are used. Exp (3.79 Å) and DLS (4.16 Å) windows will give very different ethane barriers.
2. Verify by an energy scan that the [001] 2.7–2.8 Å ring is closed, so that the channel label is conserved.
3. Use the signed ξ, or keep all walkers in one channel class.
4. **Do not call the result a reproduction** of Combariza et al. 2011.

**Unrecoverable items**
- full text and SI, including whether an SI exists
- reported D values, activation energies and any barriers or free-energy profiles for ethane in ITE, IHW and LTA
- temperatures and loadings
- MD ensemble, timestep, thermostat, run lengths, box sizes
- the structure source actually simulated, and whether it was BKS-relaxed
- Oie ethane parameters (charges, vdW, intramolecular terms)
- host–guest parameters and which framework atoms carry them
- mixing rules
- cutoff, shifting and tail corrections; Ewald settings
- BKS short-range modification
- whether any rigid-framework comparison was made

## 7. Parameter table

| Parameter | Value | Units | Exact source | Confidence |
|---|---|---|---|---|
| Authors / journal | Combariza, Sastre, Corma; JPCC 115(4) 875–884 (2011) | — | ACS landing page; Crossref | High |
| Guests | CH4, C2H6, C3H6, C3H8 | — | Abstract s.1 | High |
| Frameworks | Si-LTA, Si-IHW, Si-ITE | — | Abstract s.1 | High |
| Framework flexibility in reported results | Fully flexible | — | Abstract s.2 | High |
| Framework FF | BKS (PRL 64, 1955) | — | Abstract s.2; ref. 32 | High |
| BKS q_Si / q_O | +2.4 / −1.2 | e | BKS; verified arXiv:1109.2852 Tab. I | High (BKS); Medium (used unmodified) |
| BKS O–O A, b, C | 1388.7730, 2.76000, 175.0000 | eV, Å⁻¹, eV Å⁶ | as above | High (BKS) |
| BKS Si–O A, b, C | 18003.7572, 4.87318, 133.5381 | eV, Å⁻¹, eV Å⁶ | as above | High (BKS) |
| BKS Si–Si short range | none (Coulomb only) | — | as above | High (BKS) |
| Guest FF | Oie et al. 1981, flexible, with electrostatics | — | Abstract s.2; ref. 33; OSTI abstract | High |
| Guest representation | All-atom (explicit C, H) | — | Oie form; group usage 2015/2021 | Medium-high |
| Oie numeric parameters (ethane) | not recovered | — | — | Unrecoverable |
| Host–guest parameters / atoms / mixing | not recovered | — | candidates refs 34, 35 (unverified) | Unrecoverable |
| Cutoff / tail / shift | not recovered | Å | — | Unrecoverable |
| Electrostatics | Ewald (cited) | — | ref. 31 | Medium |
| MD code | DL_POLY (cited) | — | ref. 26 | Medium |
| Ensemble / timestep / thermostat | not recovered | — | — | Unrecoverable |
| Temperatures / loadings | not recovered | K / molec per cell | abstract mentions only a "temperature range" | Unrecoverable |
| Reported observables | self-diffusion coefficients (certain); activation energies, histogram free-energy profiles (inferred) | — | Abstract; ACS subject tags; refs 16, 30, 46 | High / Low |
| ITE type-material cell | Cmcm, 20.622, 9.7242, 19.623 | Å | IZA mat_cif ID=133; Atlas; RASPA2 | High |
| ITE DLS cell | Cmcm, 20.7530, 9.8040, 20.0090 | Å | IZA ITE.cif | High |
| IHW type-material cell | Cmca, 13.6988, 24.0665, 18.1968 | Å | IZA mat_cif ID=124 | High |
| IHW DLS cell | Cmce, 13.7470, 24.0690, 18.3290 | Å | IZA IHW.cif | High |
| Atlas channels ITE | [010] 8: 3.8 × 4.3 ↔ [001] 8: 2.7 × 5.8 | Å | Atlas 6th ed. ITE page | High |
| Atlas channels IHW | [100] 8: 3.5 × 4.3 (2D via 12-ring connections) | Å | Atlas 6th ed. IHW page | High |
| ITE [010] 8-ring free D | 3.79 (exp) / 4.16 (DLS) | Å (r_O = 1.35) | own: cifgeom.py | High (method validated on LTA within 0.06 Å) |
| ITE [001] 8-ring free D | 2.83 (exp) / 2.74 (DLS) | Å | own | High |
| ITE cage included D | 8.08 (exp) / 8.24 (DLS; IZA 8.3) | Å | own; IZA | High |
| ITE CV / period | signed y / 9.7242 (exp), 9.804 (DLS) | Å | own | High (geometry) |
| IHW 8-ring free D | 3.55 (exp) / 3.61 (DLS) | Å | own | High |
| IHW 12-ring (intra-dimer) free D | 5.73 (exp) / 5.87 (DLS) | Å | own | High |
| IHW cage included D | 6.48 (exp) / 6.61 (DLS; IZA 6.67) | Å | own; IZA | High |
| IHW CV / F period | x / a/2 = 6.8494 (exp), 6.8735 (DLS) | Å | own | High (geometry) |
| ITE box, r_c = 12 Å | 2×3×2 cells, 2304 atoms | — | own feas.py | High |
| IHW box, r_c = 12 Å | 2×1×2 cells, 1344 atoms | — | own feas.py | High |
| O within 12 Å per bead | ITE ≈ 230, IHW ≈ 274 | — | own feas.py | High |
| RASPA2 ITQ-3 blocks | 8 spheres, r = 2.0 Å, in voids with free D ≈ 1.8–2.2 Å | Å | RASPA2 ITQ-3.block; own check | High |

## 8. Files

Downloads are untrusted and stored one per directory under `cache/overnight/literature/candA/`. sha256 prefixes:
- `iza_ITE_*/ITE.cif` 346a283c5990f6b3
- `iza_IHW_*/IHW.cif` c3e461c7a765036f
- `iza_LTA_*/LTA.cif` fe090b64a2101c21
- `iza_typematerial_ITE/mat_cif.bin` 1621e77093bf1b8d
- `iza_typematerial_IHW/mat_cif.bin` 611d2cf213df12fb
- `raspa2_cif_ITQ_3_cif/ITQ-3.cif` 9184abe93cdf75a8
- `raspa2_block_ITQ_3_block/ITQ-3.block` 1c7eb7458a8d573e
- `raspa2_cif_LTA_SI_cif/LTA_SI.cif` a3a9183ef6cbe8a2
- `raspa2_block_LTA_SI_block/LTA_SI.block` 0fe1e28e8c006b35
- `crossref/crossref.json` cd13d494b5726837
- `ghysels2015_ugent/paper.pdf` 21971330d2fb8da4
- `unpaywall/unpaywall.json`
- `iza_page_*/*.html`

Saved bot or error pages that are not the named content: `acs_si/*.bin` (403), `alaithan2021_csic/paper.pdf` (HTML bot page), `iza_atlas_6ed/Atlas_6ed.pdf` (404 HTML), `riunet_record/record.html` (Anubis page).

Scripts and outputs are in `/tmp/claude-1008/-home-zheyuanlai-ABF-Fisher-Rao/candA_scratch/`:
- `cifgeom.py`, `show.py`, `profiles.py`, `feas.py`
- `*_geom.json`, `feas.json`
- `*_rfree.npy` and `*_atoms.npy`
- derived CIFs `ITQ3_tm_with_IZAops.cif` and `ITQ32_tm_with_IZAops.cif`
