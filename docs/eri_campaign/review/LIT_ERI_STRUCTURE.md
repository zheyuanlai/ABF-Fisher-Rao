# LIT_ERI_STRUCTURE: ERI framework structure and geometry for ethane / all-silica ERI

Written 2026-10-16 for the ERI campaign (united-atom ethane in rigid all-silica ERI, after
Dubbeldam, Beerdsen, Calero & Smit, PNAS 102, 12317 (2005)). No GPU was used and nothing was
committed. The only repository paths written are this file and the new cache directory
`cache/eri/literature/structure/`. Every geometric number below comes from my own numpy
scripts, run with `python -I` on the raw CIFs (Sec. 11). Every literature statement has a
short verbatim quote and its location.

Labels: **[LIT]** = stated in a source (quoted). **[COMP]** = computed by my scripts.
**[INF]** = my inference, with the reasoning given. **[FLAG]** = unresolved or inconsistent.

---

## 0. Bottom line

1. **Which coordinates the PNAS paper names** [LIT]. These are Gard & Tait (1973), *Proc. 3rd
   Int. Conf. Molecular Sieves*, pp. 94-99, with a = 13.27 Å, c = 15.05 Å, P6₃/mmc. The
   RASPA2 file `structures/zeolites/cif/ERI_SI.cif` matches them **to every digit** (the
   cell and all 8 framework sites). The match is against the Gard & Tait erionite record of
   the IZA database (archived 2006-01-03; Sec. 2.4). So RASPA2 `ERI_SI.cif` is the Gard &
   Tait framework.
2. **Which cell the PNAS box implies** [LIT + COMP]. The box is 45.906 × 39.756 × 44.43 Å.
   It equals 2√3a × 3a × 3c **exactly** for a = 13.252 Å, c = 14.81 Å. That is the cell of
   **Schlenker, Pluth & Smith, Acta Cryst. B33, 3265 (1977)**, "Dehydrated natural erionite
   with stacking faults of the offretite type" (COD 2106432). Two checks:
   * The Schlenker framework tiles that box perfectly: 3888 atoms, every Si 4-coordinated,
     every O 2-coordinated.
   * The Gard & Tait cell does **not** fit it. In the PNAS box, 36 Si and 144 O end up
     wrongly coordinated, and the shortest interatomic distance is 0.94 Å.

   The group's other papers use the same 13.252/14.81 cell while citing Gard & Tait (or the
   IZA database). Dubbeldam & Smit JPCB 2003 and Beerdsen et al. JPCB 2006 Table 1 both
   tabulate the orthorhombic cell 22.953 × 13.252 × 14.810 Å. **[FLAG]** The ERI ethane
   companion paper (JPCB 110, 3164 (2006)) gives a different box: 3×3×3 hexagonal cells
   with perpendicular widths 3.448 × 3.448 × 4.515 nm. That box matches Gard & Tait.
   **Which framework produced the published PNAS data cannot be settled from the
   literature.**
3. **The cell choice changes the window** [COMP]. With O radius 1.35 Å, the 8-ring free
   aperture is:

   | Structure | Free aperture (Å) |
   |---|---|
   | Gard & Tait / RASPA2 | 3.57 × 5.04 |
   | Schlenker 1977 | 3.81 × 4.99 |
   | IZA idealised SiO₂ (DLS) | 3.29 × 5.08 |

   An indicative single-CH₃-bead window barrier (O-only LJ, PNAS O-CH₃ parameters) is
   579 K for Gard & Tait, 353 K for Schlenker, 951 K for IZA, and 666 K for the hybrid
   "Gard & Tait fractional coordinates in the Schlenker cell". **Structure choice is
   therefore a first-order sensitivity for any barrier, and must be chosen explicitly.**
4. **Topology** [COMP, all structures].
   * Counts: 36 T + 72 O per hexagonal cell.
   * Rings per cell: 24 four-rings, 12 six-rings and 6 eight-ring windows.
   * Coordination sequences match the IZA values exactly.
   * Cages and windows (fractional coordinates):

     | Unit | Position | Per cell |
     |---|---|---|
     | ERI cage centre | (1/3, 2/3, 3/4) and (2/3, 1/3, 1/4) | 2 |
     | CAN cage | (0, 0, 1/4) and (0, 0, 3/4) | 2 |
     | D6R | (0, 0, 0) and (0, 0, 1/2) | 2 |
     | 8-ring window | 6g sites (1/2,0,0), (0,1/2,0), (1/2,1/2,0) and the same at z = 1/2 | 6 |

5. **Connectivity** [COMP]. Each window joins two ERI cages: one in the B column (1/3, 2/3)
   and one in the C column (2/3, 1/3). The two cage centres differ by a/√3 = 7.661 Å in xy
   and by ±c/2 = 7.525 Å in z (Gard & Tait). Their separation is 10.739 Å, and the window
   centre is exactly the midpoint (an inversion centre). Each cage has 6 windows: 3 at
   z_cage + c/4 and 3 at z_cage − c/4, in the same three xy directions 120° apart.
   * Along c, cages in the same column share a 6-ring with free diameter 2.61 Å. It is
     closed to ethane, so there is no straight c-channel.
   * No window links two cages that are offset purely in-plane, or purely along c.
6. **The PNAS rectangular box** holds:
   * 36 hexagonal cells, i.e. 18 orthorhombic cells of a√3 × a × c, each holding 4 cages;
   * 72 ERI cages, 216 eight-ring windows, 72 CAN cages and 72 D6R;
   * 1296 Si and 2592 O.
7. **Blocking.**
   * The RASPA2 file `ERI_SI.block` [LIT] defines 4 spheres per hexagonal cell. Each has
     radius 2.5 Å, at fractional (0,0,0), (0,0,0.25), (0,0,0.5) and (0,0,0.75). These are
     the D6R and CAN centres.
   * Neither PNAS 2005 nor JPCB 2006 (ERI) mentions blocking. JPCB 2003 says in general
     terms that inaccessible pockets must be blocked.
   * For ethane under the PNAS O-CH₃ LJ the question is moot [COMP]. The lowest rigid-ethane
     energy in a CAN cage is +24 900 K (Gard & Tait), and in a D6R it is +26 150 K. The ring
     separating the CAN and ERI cages is a 6-ring with free diameter ≤ 2.96 Å.
8. **The pathway in the xy-plane** (plain statement in Sec. 9) is the window passage itself.
   It is the horizontal segment at the window height (z = 0 or c/2). It runs along the
   horizontal B→C direction, from the B column axis through the window centre to the C
   column axis, and is a/√3 = 7.66 Å long. This is PNAS's λ_xy ≈ 0.75 nm.
   * The straight line between the two cage centres also passes through the window centre,
     but it is tilted 44.5° out of the xy-plane. About 1.5 Å before and after the window
     plane it comes within 2.92 Å of framework O, so it clips the ring rim.
   * The dcTST papers' coordinates are fractional projections of one ethane bead (Sec. 9.3).

---

## 1. Sources and local copies

Every download sits in its own new directory under
`/home/zheyuanlai/ABF-Fisher-Rao/cache/eri/literature/structure/`. Nothing downloaded was
executed. Python that read the files was run with `python -I`.

| Item | URL / route | Local path (under `structure/`) | sha256 (prefix) |
|---|---|---|---|
| IZA ERI.cif (DLS-idealised SiO₂) | https://europe.iza-structure.org/IZA-SC/cif/ERI.cif | `iza_europe/ERI.cif` | `cfb382bb…` |
| IZA Atlas pages, ERI (pp. 128-129) | https://europe.iza-structure.org/IZA-SC/Atlas_pdf/ERI.pdf | `iza_atlas_pdf_ERI/ERI.pdf` | `cd61c36c…` |
| RASPA2 ERI_SI.cif (commit 6498ab1e) | raw.githubusercontent.com/iRASPA/RASPA2/6498ab1e…/structures/zeolites/cif/ERI_SI.cif | `raspa2_ERI_SI_cif/ERI_SI.cif` | `55b69953…` |
| RASPA2 ERI_SI.block | …/structures/zeolites/block/ERI_SI.block | `raspa2_ERI_SI_block/ERI_SI.block` | `e7b5d138…` |
| RASPA2 src/framework.c (block reader) | …/src/framework.c | `raspa2_src_framework_c/framework.c` | `51ce4e9e…` |
| RASPA2 src/input.c (BlockPockets keywords) | …/src/input.c | `raspa2_src_input_c/input.c` | `5656b3b8…` |
| RASPA2 example 9, hexane in ERI (input + output) | …/examples/Basic/9_HenryCoefficient_of_Hexane_in_ERI_mono/ | `raspa2_ex9_hexane_ERI_input/`, `raspa2_ex9_hexane_ERI_output/` | `e81a7391…`, `3a98810f…` |
| RASPA2 example, methane density in ERI (input) | …/examples/Visualization/UnitCell/Density/ERI/RASPA/simulation.input | `raspa2_vis_density_ERI_input/` | `79e4ed4b…` |
| RASPA2 CHA_SI.cif (side check only) | …/structures/zeolites/cif/CHA_SI.cif | `raspa2_CHA_SI_cif/` | `980a6f07…` |
| COD 2106432, Schlenker, Pluth & Smith 1977 | https://www.crystallography.net/cod/2106432.cif | `cod_2106432/2106432.cif` | `23bfe87d…` |
| COD 9013676, Ballirano et al. 2009, erionite-K | …/cod/9013676.cif | `cod_9013676/` | `9ffc837f…` |
| COD 9001941, Gualtieri et al. 1998, erionite-Ca | …/cod/9001941.cif | `cod_9001941/` | `0f1bbce6…` |
| COD 9011786, Staples & Gard 1959 | …/cod/9011786.cif | `cod_9011786/` | `e9b615aa…` |
| IZA crystal-data record 55 (Erionite, Gard & Tait), Wayback 2006-01-03 | web.archive.org/web/20060103105754id_/http://topaz.ethz.ch:80/IZA-SC/FMPro?-db=crystal%5fdata.fp5&…&STC=ERI&-recid=55&-find= | `wayback_iza_crystaldata_ERI_recid55_20060103/` | `113b4d2d…` |
| IZA crystal-data record 56 (AlPO-17), Wayback 2006-01-03 | …-recid=56… | `wayback_iza_crystaldata_ERI_recid56_20060103/` | `e91fcccf…` |
| IZA crystal-data list for ERI, Wayback 2005-03-14 | …20050314014207… | `wayback_iza_crystaldata_ERI_20050314/` | `d1ec9db7…` |
| IZA type-material page for ERI, Wayback 2006-01-03 | …20060103083348… | `wayback_iza_TMdisplay_ERI_20060103/` | `7c424882…` |
| IZA framework page for ERI, Wayback 2005-03-10 | …20050310220716… | `wayback_iza_FWtopology_ERI_20050310/` | `3aa32083…` |
| IZA natural-zeolite datasheet, Erionite | https://www.iza-online.org/natural/Datasheets/Erionite/Erionite.html | `iza_online_natural_erionite_datasheet/` | `a01d6254…` |
| PNAS 2005 full text (PMC page as markdown, via the web_fetch tool) | https://pmc.ncbi.nlm.nih.gov/articles/PMC1194924/ | `pnas2005_pmc_webfetch_text/PMC1194924_fulltext.md` | `f09dc230…` |

**Read but not downloaded by me.** These PDFs were cached by the parallel PNAS review in
`cache/eri/literature/pnas2005/`. I converted them with `pdftotext` (into my scratch
directory) and checked every statement I use from them.
* Dubbeldam & Smit, *JPCB* 107, 12138 (2003): `epfl_jpcb2003_window/2098800132dub033.pdf`.
* Dubbeldam, Beerdsen, Calero & Smit, *JPCB* 110, 3164 (2006): `epfl_jpcb2006/3248743680dub061.pdf`.
* Beerdsen, Dubbeldam & Smit, *JPCB* 110, 22754 (2006): `epfl_jpcb2006_beerdsen_methane/3987880708bee063.pdf`.

**Tried and failed:**
* `asia.iza-structure.org` CIF: the connection timed out.
* IZA `framework.php?STC=ERI` (all mirrors, curl and both fetch tools): the server returned
  "Input error ERROR 79".
* Europe PMC fullTextXML: HTTP 500.
* NCBI OAI: "cannotDisseminateFormat".
* PMC figure and equation images: a reCAPTCHA page. The one Fig. 3 JPEG I received was
  overwritten in a retry loop and not kept, and later requests were challenged.
* Wayback CDX found no ERI CIF capture before 2005.

The failed-download directories were deleted.

---

## 2. What the literature says about the structure (quotes)

### 2.1 PNAS 2005 (PMC1194924; section names from the PMC page)

* *Model and Computational Details*: "we neglect cations and study rigid, all-silica versions
  of the ERI- and CHA-type zeolites"; "The positions of the atoms are taken from refs. 14
  and 15, respectively." Ref. 14 is "Gard, J. A. & Tait, J. M. (1973) Proceedings of the
  Third International Conference on Molecular Sieves, ed. Uytterhoeven, J. B. (Leuven Univ.
  Press, Zurich), pp. 94–99."
* Same section: "The rectangular simulation box sizes we used are 4.5906 × 3.9756 × 4.443
  nm for ERI-type zeolite and 3.015 × 4.7814 × 2.7606 nm for CHA-type zeolite."
* Same section: "the zeolites are modeled as a rigid network of oxygen atoms". The same
  paragraph also lists "σ Si–CH = 2.12 Å, ε Si–CH3 / kB = 82 K" (an internal
  inconsistency; JPCB 2006 settles it, Sec. 2.3).
* Same section: "Molecules were inserted into the framework at random positions as long as
  no overlaps occurred with the framework or other particles." The page contains no word
  "block", "pocket" or "cancrinite" (grep of the cached text).
* Fig. 3 caption: "The ERI-type silica structure crystallizes in the hexagonal dipyramidal
  space group P6₃/mmc with a = b = 1.327 nm, c = 1.505 nm, and α = β = 90° and γ = 120°."
* Fig. 3 caption: "The lattice is drawn in blue dots connected by blue lines of lattice
  distance λ ≈ 0.75 nm for x, y, and z directions."
* Fig. 3 caption: the xy profile runs "with q A the center of a cage, and q B the center of
  a neighboring cage"; the z profile runs "across a cage with q A the top of the cage, q I
  the middle of the cage, and q B the bottom of the cage."
* *Model and Computational Details*: "In ERI-type zeolite, each hop in the z direction is
  preceded by a hop in xy direction".
* *Results*: "the barriers to diffusion are formed by the xy barriers, i.e., the eight-ring
  windows between the adjacent cages."

### 2.2 Dubbeldam & Smit, JPCB 107, 12138 (2003)

* p. 12139: "The positions of the atoms are taken from ref 22." Ref. 22 is "IZA Structure
  Commission. Database of Zeolite Structures. http://www.iza-structure.org/databases/
  (accessed May 2003)."
* p. 12143: "all structures are converted to orthorhombic periodic unit cells with
  dimensions … a = 2.2953 nm, b = 1.3252 nm, and c = 1.481 nm for ERI".
* p. 12143: "To obtain correct results, it is necessary to artificially block the
  inaccessible pockets for adsorbing molecules." The only example it gives is the LTA
  sodalite cage.

### 2.3 Dubbeldam, Beerdsen, Calero & Smit, JPCB 110, 3164 (2006): the ERI ethane study

* p. 3165, Sec. II.A: the cell is "a = b = 1.327 nm, c = 1.505 nm". The structure reference
  is ref. 30, which is Gard & Tait (1973).
* p. 3165, Sec. II.A: "The elongated erionite cages approximate the shape of 1.3 × 0.63 nm
  cylinders connected by 0.36 × 0.51 nm windows."
* p. 3165, Sec. II.A: "There are three windows at the top of the cage rotated 120° with
  respect to one another. At the bottom of the cage there are also three windows rotated
  120° with respect to one another. The top three windows are aligned with the windows at
  the bottom. Each ERI-type unit cell contains two erionite cages".
* p. 3167, Sec. II.D: "the zeolites are modeled as a rigid network of oxygen and silicon
  atoms".
* p. 3167, Sec. II.D: "The simulation box sizes we used are 3 × 3 × 3 (perpendicular widths
  are 3.448 × 3.448 × 4.515 nm) for ERI-type zeolite".
* p. 3166, Sec. II.C: "We can choose q as the position of one of the two beads of ethane."
* Eq. 10 (pp. 3166-3167): cage A is selected by "s x + (1 − s y) < 1" and cage B by
  "s x + (1 − s y) > 1".
* pp. 3166-3167: "For xy computations the barrier is located at s xy = {1/2, 1/2}".
* p. 3167: "For the z computation we select only cage A and use q = s z" (Eq. 11). Also:
  "The barrier at the center of the cage is located at s = {1/3, 2/3, 3/4}, while the
  free-energy minima qA and qB are located at s = {1/3, 2/3, 1/2} and s = {1/3, 2/3, 1}".

### 2.4 IZA database, current and archived

* Current CIF header (`iza_europe/ERI.cif`): "The atom coordinates and the cell parameters
  were optimized with DLS76 assuming a pure SiO2 composition." Cell 13.0540 / 15.1750 Å,
  'P 63/m m c'.
* The 2005-03-10 framework page (Wayback) already gives the same DLS cell: "a = 13.054Å …
  c = 15.175 Å" and "R DLS = 0.0023".
* Atlas p. 129, *Type Material Data*: "hexagonal, P6₃/mmc, a = 13.27Å, c = 15.05Å", citing
  ref. 3, Gard & Tait 1973. It also gives "Framework density: 15.7 T/1000Å³" and
  "Channels: ⊥ [001] 8 3.6 x 5.1***".
* Atlas p. 128: "Framework description: AABAAC sequence of 6-rings"; "Composite building
  units: d6r can". Coordination sequences: "T1 (24,1) 4 9 17 30 50 75 98 118 144 185" and
  "T2 (12,m) 4 10 20 32 46 64 90 126 164 196"; vertex symbols "4·4·4·6·6·8" and
  "4·8·4·8·6·6".
* **Archived IZA crystal-data record 55 (2006-01-03)** reads "Material name: Erionite";
  "Reference: Gard, J.A. and Tait, J.M. Proc. 3rd Int. Conf. Molecular Sieves, , pp. 94-99,
  (1973)"; "X-ray single crystal refinement, R=0.15"; a = 13.27, c = 15.05 Å. Its
  coordinates include SI1 0.00021 0.23350 0.10446, SI2 0.0935 0.4245 0.2500, O1 0.0267
  0.3493 0.1610, O2 0.0979 0.1957 0.1264, O3 0.1258 0.2516 0.6339, O4 0.2638 0.0 0.0, O5
  0.2293 0.4585 0.25 and O6 0.4603 0.9206 0.25. **These are identical to RASPA2
  `ERI_SI.cif`.** The record also lists cations and waters, including "K1 K 0.0 0.0 0.25
  1.0", i.e. K at the CAN-cage centre.
* The 2005-03-14 crystal-data list offers ERI coordinates for only two materials:
  "Erionite" (the Gard & Tait record) and "Piperidine AlPO-17". Record 56, AlPO-17 (Pluth,
  Smith & Bennett 1986), has a = 13.2371 and c = 14.7708 Å, which is not 13.252/14.81.
* IZA natural-zeolite datasheet: "This stacking arrangement produces columns of cancrinite
  cages alternating with double 6-ring (D6R) cages (formed by the A 6-rings) and of
  erionite cavities between the B or C 6-rings." Also: "K-cations … occur in the cancrinite
  cages coordinated with 6 of the framework oxygen anions at distances of 2.92 Å", and
  "With two cavities per unit cell". My computed CAN-centre-to-O distance for Gard & Tait
  is 2.919 Å, which agrees.

### 2.5 RASPA2 files

* `ERI_SI.cif` header: "_audit_creation_date 2011-2-19", "_audit_author_name 'David
  Dubbeldam'", "_citation_author_name 'J.A. Gard and J.M. Tait'", "_citation_title
  'Refinement of the crystal structure of erionite'", "_cell_length_a 13.27",
  "_cell_length_c 15.05", "_cell_volume 2295.14". The charges are Si 2.05 and O −1.025.
  Git history: one commit, "fd2398c807 2015-02-03 Initial Release RASPA-2.0".
* `ERI_SI.block`, verbatim:
  ```
  4
  0.0         0.0        0.0        2.5
  0.0         0.0        0.25       2.5
  0.0         0.0        0.5        2.5
  0.0         0.0        0.75       2.5
  ```
* Its meaning, from `src/framework.c` (`ReadBlockingPockets`, around l. 6179):
  `fscanf(FilePtr,"%d\n",&…NumberOfBlockCenters…)`, then per line
  `fscanf(FilePtr,"%lf %lf %lf %lf\n",&tempr.x,&tempr.y,&tempr.z,&temp)`. So the format is
  a count, then fractional (unit-cell) x y z and a radius. The centres are replicated over
  all unit cells and converted with `ConvertFromABCtoXYZ`.
* `BlockedPocket()` (around l. 6252) compares a Cartesian distance r against the radius:
  "if inside block-pocket, then block (return 'true')". So the radius is 2.5 Å. Blocking is
  per component and opt-in, through `BlockPockets yes` and `BlockPocketsFileName` (input.c
  around ll. 2666-2691).
* RASPA2 example 9 (hexane Widom insertion in ERI, 573 K) has `UnitCells 3 3 3`. Its output
  says "Number of pockets blocked in a unitcell: 0" and "Pockets are NOT blocked for this
  component" (output ll. 663-664). It also prints "Unit cell size: 13.270000 13.270000
  15.050000", "perpendicular cell widths: 34.47647 34.47647 45.15000" and "Number of
  framework atoms in the unit cell: 108". Those widths are the JPCB 2006 ERI box.

### 2.6 Schlenker, Pluth & Smith (1977), COD 2106432

* `_publ_section_title`: "Dehydrated natural erionite with stacking faults of the
  offretite type". Acta Crystallographica B 33, 3265-3268.
* "_cell_length_a 13.252", "_cell_length_c 14.81", "_cell_volume 2252.416", 'P 63/m m c'.
* Semantic Scholar abstract snippet (search excerpt): "A single crystal of erionite, …
  from Mazé, Japan, was dehydrated in high vacuum at 300°C for 1 d, and the structure
  refined in P63/mmc [a = 13.252 (4), c = …".
* The CIF carries offretite-type fault sites at about 5% occupancy: O2 0.058, O7 0.05, O9
  0.05, and Al3/Si3 at 0.059 in total. It also has cations, e.g. "Ca1 Ca+2 0 0 0.25 1"
  (CAN centre) and "K1 K+1 0.5 0 0 0.2667" (an 8-ring window site).

---

## 3. The candidate structures: cells and framework coordinates

The space group is P6₃/mmc (No. 194) in all cases. There are 24 symmetry operators, in the
setting with the 6₃ axis through the origin, and γ = 120°. Each "majority framework" below
keeps the Si/Al (T) and framework-O sites with occupancy ≥ 0.5. Cations, water and the
~5% offretite-fault sites are dropped, and all T are treated as Si.

| Structure | a (Å) | c (Å) | V (Å³) | FD (T/1000 Å³) |
|---|---|---|---|---|
| IZA DLS SiO₂ (idealised) | 13.0540 | 15.1750 | 2239.48 | 16.08 |
| Gard & Tait 1973 = RASPA2 ERI_SI = IZA type material | 13.27 | 15.05 | 2295.14 | 15.69 |
| Schlenker et al. 1977, dehydrated (COD 2106432) | 13.252 | 14.81 | 2252.42 | 15.98 |
| Ballirano et al. 2009, erionite-K, hydrated (COD 9013676; cross-check only) | 13.22794 | 15.06946 | 2283.56 | 15.76 |
| Gualtieri et al. 1998, Shourdo (COD 9001941; cell only) | 13.264 | 15.067 | – | – |
| Staples & Gard 1959 (COD 9011786; cell only) | 13.26 | 15.12 | – | – |

**Asymmetric-unit framework coordinates.** Columns are x, y, z (fractional); Wyckoff
letters and multiplicities come from my orbit expansion.

| Site (Wyckoff, mult.) | IZA DLS | Gard & Tait / RASPA2 | Schlenker 1977 (label, occ.) |
|---|---|---|---|
| T1 (24l, 24) | 0.0006 0.2345 0.1014 | Si1 0.00021 0.2335 0.10446 | Si1/Al1 0 0.2304 0.1071 (1.0) |
| T2 (12j, 12) | 0.0961 0.4261 0.2500 | Si2 0.0935 0.4245 0.25 | Al2/Si2 0.0938 0.4256 0.25 (0.941) |
| O, T1–T1 in the T1 6-rings, set 1, alternates with set 2 (12k, 12) | O3 0.0957 0.1914 0.1139 | O2 0.0979 0.1957 0.1264 | O3 0.093 0.186 0.1346 (1.0) |
| O, T1–T2 (24l, 24) | O2 0.0360 0.3474 0.1634 | O1 0.0267 0.3493 0.161 | O1 0.0338 0.3501 0.1593 (0.942) |
| O, T1–T1 in the T1 6-rings, set 2 (12k, 12) | O1 0.8711 0.1289 0.1281 | O3 0.1258 0.2516 0.6339 | O4 0.1258 0.2516 0.641 (1.0) |
| O, D6R vertical T1–T1 (12i, 12) | O4 0.0000 0.2705 0.0000 | O4 0.2638 0 0 | O5 0.2523 0 0 (1.0) |
| O, T2–T2 in B/C 6-ring (6h, 6) | O5 0.2356 0.4712 0.2500 | O5 0.2293 0.4585 0.25 | O6 0.2363 0.4726 0.25 (0.95) |
| O, T2–T2 in 8-ring (6h, 6) | O6 0.0757 0.5378 0.2500 | O6 0.4603 0.9206 0.25 | O8 0.4552 0.9104 0.25 (0.95) |

Rows are matched by orbit [COMP, `match_sites.py`]. For each Gard & Tait O site, the
nearest symmetry image of the matched site in the other structure lies 0.04-0.20 Å away.
The bridging T pairs were also checked: O1 bridges T1–T2; O2, O3 and O4 bridge T1–T1;
O5 and O6 bridge T2–T2 (Gard & Tait labels). The IZA and RASPA2 labels differ, and
different symmetry-equivalent representatives are listed; for example, IZA O6
(0.0757, 0.5378) is the 6h orbit (−2x, −x) with x = −0.5378. Ballirano 2009 has the
same orbits with T1 0.23397 −0.00016 0.10440 and T2 0.09314 0.42355 0.25 (see its JSON).

---

## 4. Expansion checks [COMP]

All four frameworks expand to **36 T and 72 O per hexagonal cell**: T1 has 24 and T2 has 12;
the O orbits have 24 + 12 + 12 + 12 + 6 + 6. RASPA2 itself prints "Number of framework
atoms in the unit cell: 108". Every T is 4-coordinated and every O is 2-coordinated
(T–O < 2.0 Å).

| Structure | T–O min–max (mean), Å | O–T–O (°) | T–O–T (°) | min O–O (Å) |
|---|---|---|---|---|
| IZA DLS | 1.6079–1.6101 (1.6097) | 109.26–109.78 | 142.1–163.5 (mean 148.5) | 2.624 |
| Gard & Tait / RASPA2 | 1.6229–1.6406 (1.6317) | 106.85–111.78 | 141.1–168.6 (mean 147.1) | 2.623 |
| Schlenker 1977 | 1.6082–1.6667 (1.6291) | 102.83–111.83 | 133.7–176.6 (mean 146.8) | 2.560 |
| Ballirano 2009 | 1.6116–1.6520 (1.6328) | 106.36–113.53 | 141.5–170.9 (mean 146.3) | 2.604 |
| Hybrid: G&T fractional in the 13.252/14.81 cell | 1.5987–1.6374 (1.6216) | 106.75–111.78 | 141.3–168.6 | 2.587 |

The T–O distances are 1.60-1.67 Å in every structure, consistent with the expected ~1.6 Å.
The Schlenker values are Al/Si averages, Si:Al ≈ 27:9.

**Topology.**
* The coordination sequences of the expanded net (10 shells) are T1 = 4 9 17 30 50 75 98 118
  144 185 and T2 = 4 10 20 32 46 64 90 126 164 196 for IZA, RASPA2 and Schlenker. They are
  identical to the IZA Atlas values.
* Primitive rings per cell: 24 four-rings and 12 six-rings. These agree with the vertex
  symbols: (24·3 + 12·2)/4 = 24 and (24·2 + 12·2)/6 = 12.
* There are 18 primitive 8-cycles per cell. Only **6 are windows**: each holds 4 T1 + 4 T2
  and sits at a 6g site, as the vertex symbols require: (24·1 + 12·2)/8 = 6. The other 12
  are 8-cycles of T1 only that wind around a D6R; their centroid is the D6R centre. They
  are not strong rings and not windows.

---

## 5. The orthorhombic supercell and the PNAS box [COMP + LIT]

**Construction** [COMP]. The rectangular cell is spanned by A' = −(a + 2b) (length a√3),
a (length a) and c. The supercell is 2 × A', 3 × a and 3 × c, which gives
Lx = 2√3a, Ly = 3a, Lz = 3c. It is a lattice supercell. It contains:
* 36 hexagonal cells (area ratio 6√3a² / (√3a²/2) = 12 per layer, times 3 layers);
* 2 × 36 = **72 ERI cages**, 6 × 36 = **216 8-ring windows**, **72 CAN cages** and
  **72 D6R**;
* 1296 Si + 2592 O = 3888 atoms.

Built frame: Cartesian x is along −(a + 2b), and y is along the hexagonal a. The hexagonal
vectors are a = (0, a, 0), b = (−a√3/2, −a/2, 0) and c = (0, 0, c). This is a proper +90°
rotation of the standard a ∥ x frame. It is the orientation implied by the group's
orthorhombic cell "a = 2.2953 nm, b = 1.3252 nm" (JPCB 2003), where 22.953 = √3·13.252.

| Cell used | 2√3a × 3a × 3c (Å) | PBC check in that box | Same atoms placed in the PNAS box 45.906 × 39.756 × 44.43 |
|---|---|---|---|
| IZA DLS 13.054/15.175 | 45.2204 × 39.1620 × 45.5250 | OK (all Si 4-coordinated, all O 2-coordinated, d_min 1.608) | broken: 96 Si and 318 O wrongly coordinated, d_min 0.59 Å |
| Gard & Tait 13.27/15.05 | **45.9686 × 39.8100 × 45.1500** | OK (d_min 1.623) | broken: 36 Si, 144 O, d_min 0.94 Å |
| Schlenker 13.252/14.81 | **45.9063 × 39.7560 × 44.4300** | OK (d_min 1.608) | **perfect: 3888 atoms, d_min 1.608** |
| Ballirano 13.228/15.069 | 45.8229 × 39.6838 × 45.2084 | OK | broken: d_min 0.07 Å |

* Inverting the PNAS box [COMP]: 45.906/(2√3) = 13.2519 Å, 39.756/3 = 13.2520 Å and
  44.43/3 = 14.8100 Å.
* So the box is consistent with x = 2√3a, y = 3a, z = 3c **only for a = 13.252 Å and
  c = 14.81 Å**. That is the Schlenker 1977 cell, to every digit COD gives.
* With the cell printed in the PNAS caption (13.27/15.05), z/c = 44.43/15.05 = 2.952. That
  is not an integer, so the box cannot be periodic.
* Beerdsen et al. JPCB 110, 22754 (2006), Table 1 (p. 22758), ERI row: "22.953 13.252
  14.810 orthorhombic 2×3×3". The text says "we used a rectangular version of the unit
  cell, where one unit cell of erionite contains 4 cages". Their ERI coordinates are cited
  to Gard & Tait (ref. 59).
* 2 × 22.953 = 45.906, 3 × 13.252 = 39.756 and 3 × 14.810 = 44.43: **the PNAS box is
  exactly this 2×3×3 orthorhombic box**, i.e. 18 orthorhombic cells = 36 hexagonal cells =
  72 cages.
* **[FLAG]** In the papers I could check, the 13.252/14.81 cell always comes with a Gard &
  Tait or IZA citation. The archived IZA ERI records (2005-2006) contain no
  13.252/14.81 entry: they have Gard & Tait 13.27/15.05, DLS 13.054/15.175 and AlPO-17
  13.2371/14.7708. The only source of that cell I found is Schlenker et al. 1977.
* So the group's ERI file **may** have been one of two things, and I cannot tell which:
  * the Schlenker framework, mislabelled; or
  * Gard & Tait fractional coordinates placed in Schlenker's cell. That "hybrid" is still
    chemically sane: T–O 1.599–1.637 Å.
* JPCB 2006 (the ERI ethane paper) states a box consistent with Gard & Tait instead
  (3.448 × 3.448 × 4.515 nm perpendicular widths = 3 × 3 × 3 hexagonal cells, 54 cages).
* Side observation, not pursued: the PNAS CHA box (30.15 × 47.814 × 27.606 Å) inverts to a
  hexagonal CHA with a = 13.803 and c = 15.075 Å. RASPA2's Calligaris `CHA_SI.cif`
  (rhombohedral 9.459 Å, 94.07°) converts to a = 13.844 and c = 15.176 Å. So the PNAS CHA
  box does not match the RASPA2 CHA file either. It does match the Beerdsen Table 1 CHA row
  (15.075 × 23.907 × 13.803).

Supercell files I generated (for reuse; scratch only):
* `/tmp/claude-1008/-home-zheyuanlai-ABF-Fisher-Rao/eri_struct_scratch/out/<NAME>_supercell.xyz`
  (extended XYZ with the `Lattice=` box).
* `<NAME>_supercell_sites.json` holds all 72 cage centres, 216 window centres, 72 CAN and
  72 D6R positions in the box frame.
* NAME is one of `RASPA2_ERI_SI`, `Schlenker1977`, `IZA_DLS`, `Ballirano2009` or
  `HYBRID_GTfrac_SchlenkerCell`.

---

## 6. Cages and void geometry [COMP]

d_O is the distance to the nearest framework O centre. The free diameter is 2(d_O − 1.35 Å).
The table has the special points for Gard & Tait; the Schlenker and IZA values follow in
brackets.

| Point (fractional) | What it is | d_O (Å) | free diam. (Å) |
|---|---|---|---|
| (1/3, 2/3, 3/4), (2/3, 1/3, 1/4) [Wyckoff 2d, 2c] | ERI cage centres | 4.743 [4.854; 4.624] | 6.79 [7.01; 6.55] |
| (1/3, 2/3, 1/2), (1/3, 2/3, 1) | cage axis at window height, the "top/bottom" of the cage | 4.459 [4.321; 4.390] | 6.22 |
| on the B axis, z ≈ 0.54 and 0.96 | widest point of the cage (largest included sphere) | 4.968 [4.951; 4.834] | 7.24 |
| (1/2, 1/2, 1/2) and the other 6g sites | 8-ring window centres | 3.134 [3.256; 2.996] | 3.57 [3.81; 3.29] |
| (0, 0, 1/4), (0, 0, 3/4) | CAN (cancrinite, ε) cage centres | 2.919 [2.735; 2.991] | 3.14 [2.77; 3.28] |
| (0, 0, 0), (0, 0, 1/2) | D6R centres | 2.946 [2.921; 2.769] | 3.19 |
| (1/3, 2/3, 1/4), (2/3, 1/3, 3/4) | 6-ring cap shared by consecutive ERI cages in one column | 2.392 [2.227; 2.210] | 2.08 |

**ERI cage shape** (Gard & Tait). d_O along the cage axis (1/3, 2/3, z) for z = 0.50 → 1.00:
4.459, 4.780, 4.934, 4.851, 4.765, 4.707, 4.678, 4.680, 4.712, 4.758, 4.743 (the centre),
then symmetric.
* The cage spans the full c from one 6-ring cap to the next, which is 15.05 Å between ring
  planes. It is a nearly cylindrical void of free diameter about 6.2-7.2 Å.
* It is widest just inside each window height. There is a shallow waist at z ≈ 0.66 / 0.84
  and a weak local maximum at the centre.
* This agrees with JPCB 2006's "1.3 × 0.63 nm cylinders" [LIT].

**Void connectivity** (grid 66 × 66 × 75 per cell; connected components of {d_O > t}):
* For t = 2.6 and 2.8 Å, one component holds all ERI cages and percolates in x, y and z.
  The CAN and D6R pockets are separate closed components smaller than 1 Å³.
* For Gard & Tait the ERI network still percolates at t = 3.0 Å. For IZA DLS it splits at
  3.0 Å, because its window centre has d_O = 2.996 Å.
* So the percolation threshold is set by the 8-ring window. The CAN and D6R pockets are
  never connected to the ERI network.

---

## 7. Rings, windows and connectivity [COMP]

### 7.1 The 8-ring window, for Gard & Tait (Schlenker; IZA)

* **Position.** The window centres are the 6g sites. At z = 0: (1/2, 0, 0), (0, 1/2, 0) and
  (1/2, 1/2, 0). At z = 1/2: (1/2, 0, 1/2), (0, 1/2, 1/2) and (1/2, 1/2, 1/2). So there are
  6 per cell, at the heights of the D6R centres.
* **Composition.** 4 T1 + 4 T2, with O = 4 × O(T1–T2) + 2 × O4 (D6R vertical bridge) +
  2 × O6 (T2–T2 at z = 1/4 or 3/4).
* **O–O diametric distances** (Å): 6.269, 6.544, 6.544, 7.743 (Schlenker 6.512, 6.512,
  6.565, 7.685; IZA 5.992, 6.715, 6.715, 7.778). T–T diametric: 7.743, 7.743, 8.472, 8.472.
* **Free aperture** (O–O − 2.70 Å): **3.57 × 5.04 Å** (Schlenker 3.81 × 4.99; IZA
  3.29 × 5.08; Ballirano 3.47 × 5.06; hybrid 3.56 × 4.93).
  * The largest free sphere in the ring plane is 3.58 Å (3.81; 3.36), centred on the ring
    centre.
  * IZA quotes 3.6 × 5.1 Å for the type material. Gard & Tait reproduces 3.57 ≈ 3.6, but
    the long axis comes out 5.04, not 5.1 Å **[FLAG, minor]**. It may come from a different
    O pair or rounding in IZA's method.
* **Orientation.**
  * Ring plane: least-squares plane of the 8 O (RMS out-of-plane 0.14 Å).
  * The normal is **tilted 10.8° from horizontal** (Schlenker 12.8°; IZA 11.2°).
  * Its horizontal part points along the B→C xy direction.
  * The ring long axis is 13.6° from c (15.5°; 12.7°), so the long axis is near-vertical and
    the short axis horizontal.
  * This agrees with Beerdsen 2006, p. 22758 [LIT]: "their diameter varies from 3.6 Å in
    the b direction to 5.1 Å in the c direction".
  * The normal leans toward the lower of the two connected cages, so it is 33.7° from the
    cage-to-cage vector (31.2°; 34.0°) rather than 44.5°.

### 7.2 Which cages each window joins (all 216 windows in the box checked)

* Every window lies 5.369 Å (Schlenker 5.324; IZA 5.347) from exactly two cage centres. The
  third-nearest cage is 10.81 Å away.
* The two cages are on opposite sides of the ring plane, one in the B column (1/3, 2/3) and
  one in the C column (2/3, 1/3). The window is their exact midpoint, which is an inversion
  centre of P6₃/mmc.

Cage-to-cage vector through one window:

| Structure | \|Δ\| (Å) | Δxy = a/√3 (Å) | Δz = ±c/2 (Å) | angle of Δ to the xy-plane |
|---|---|---|---|---|
| Gard & Tait | 10.739 | 7.661 | ±7.525 | 44.5° |
| Schlenker | 10.648 | 7.651 | ±7.405 | 44.1° |
| IZA DLS | 10.695 | 7.537 | ±7.5875 | 45.2° |

* The cage graph is bipartite (B ↔ C) with degree 6: 216 edges for 72 cages.
* Take a B cage at z₀. Its 3 windows at z₀ + c/4 lead to C cages at z₀ + c/2. Its 3
  windows at z₀ − c/4 lead to C cages at z₀ − c/2.
* Both sets use the same three xy directions. In the box frame these are 0°, +120° and
  −120° for one sublattice, and 180° and ±60° for the other.
* Projected on xy, the cage centres form a honeycomb of spacing a/√3. This is PNAS's
  "hexagonal lattice" with λ_xy ≈ 0.75 nm; the vertical λ_z = c/2 = 7.525 Å (Gard & Tait).

**Other neighbour shells** (cage-centre distances from one cage, Gard & Tait): 10.739, 13.27,
15.05, 17.07 and 20.07 Å.
* **13.27 Å (a).** These are cages of the same sublattice at the same z. They share no
  ring; reaching one takes two window hops (B → C → B).
* **15.05 Å (c).** This is the next cage in the same column, behind the shared T2 6-ring
  cap. The cap's free diameter is 2.61 Å (Schlenker 2.32; IZA 2.42). Its centre has
  d_O = 2.39 Å, and a single CH₃ bead there has LJ energy +41 000 K. It is closed to ethane.

**Answer to "which windows connect cages in the ab-plane vs along c".**
* All 8-ring windows open in near-horizontal directions; the IZA "⊥ [001]" channels.
* Every window hop moves a molecule by a/√3 in xy **and** by ±c/2 in z between cage
  centres.
* No window connects cages purely in-plane, and none purely along c.
* Net c-transport is a zigzag B → C → B through alternating upper and lower windows. In
  the papers' half-cage picture it splits into horizontal window hops plus vertical
  intra-cage moves of c/2 (Sec. 9).

### 7.3 The 6-rings (12 per cell, three kinds)

| Kind | Where | Normal | O–O diameters (Å) | Free (Å) |
|---|---|---|---|---|
| D6R / CAN axial (T1 × 6), 4 per cell | (0, 0, ±0.13), (0, 0, 0.37/0.63) | ∥ c | 5.142 × 3 | 2.44 [Schl. 2.32; IZA 2.38] |
| CAN equatorial (4 T1 + 2 T2), 6 per cell, CAN ↔ ERI | e.g. (0.8505, 0.1495, 1/4) | ⊥ c | 3.495, 5.663, 5.663 | 0.79-2.96 [0.53-3.05; 1.00-2.95] |
| ERI cage cap (T2 × 6), 2 per cell, ERI ↔ ERI along c | (1/3, 2/3, 1/4), (2/3, 1/3, 3/4) | ∥ c | 5.311 × 3 | 2.61 [2.32; 2.42] |

These counts are consistent with the cage symbols. The CAN cage has 2 axial + 3 equatorial
6-rings, i.e. [4⁶6⁵]. The ERI cage has 2 caps + 3 rings shared with CAN cages, i.e. 6⁵,
together with its six 8-rings.

---

## 8. Inaccessible CAN cages and D6R, and blocking

* **[LIT]**
  * RASPA2 blocks four spheres of radius 2.5 Å per cell, at (0,0,0), (0,0,1/4), (0,0,1/2)
    and (0,0,3/4). My cage assignment shows these are exactly the 2 D6R centres and the
    2 CAN centres; quoted in Sec. 2.5.
  * The file dates from the RASPA-2.0 release (2015). The CIF header carries
    "_audit_creation_date 2011-2-19", so it post-dates the 2005 work.
  * PNAS 2005 and JPCB 2006 say nothing about blocking.
  * JPCB 2003 (p. 12143) says that in general inaccessible pockets must be artificially
    blocked; its example is LTA.
  * RASPA's own ERI hexane example runs with "Pockets are NOT blocked for this component".
* **Geometry [COMP].** The CAN and D6R pockets can be reached only through 6-rings of free
  diameter ≤ 2.96 Å. The CAN equatorial ring is 0.79 × 2.96 Å; the axial ring is 2.44 Å.
  The pockets are isolated void components (Sec. 6).
* **Energetics, indicative [COMP].** The model is a single CH₃ bead, or rigid two-bead
  ethane (1.54 Å), with O-only LJ, σ = 3.17 Å, ε/kB = 142 K, a 12 Å cut and no shift. These
  are the PNAS O-CH₃ values; the PNAS Si-CH₃ term is omitted.

  | | Gard & Tait | Schlenker | IZA DLS |
  |---|---|---|---|
  | bead at CAN centre (lowest within 2.5 Å) | +1 556 K | +9 828 K | −217 K |
  | bead at D6R centre | +506 K | +1 215 K | +7 116 K |
  | rigid ethane, best COM within 1 Å of CAN centre, all orientations | +24 889 K | +66 141 K | +12 466 K |
  | rigid ethane, best in D6R | +26 150 K | +31 419 K | +69 864 K |
  | rigid ethane, best in ERI cage (B column, r < 4 Å) | −3 792 K | −3 789 K | −3 918 K |

* **[INF]** Ethane cannot be inserted into a CAN cage or D6R with finite Boltzmann weight
  under this force field, so blocking changes nothing for ethane in Gard & Tait or
  Schlenker. A single bead (methane-like) can sit at the IZA-DLS CAN centre (−217 K), so a
  campaign that uses the IZA-DLS framework with a smaller probe should block these pockets.
  If blocking is wanted for safety, the RASPA2 spheres (2.5 Å at the four column points
  above) are the documented choice. They do not cut into the accessible void [COMP,
  `block_clearance.py`; grid 80 × 80 × 90]. Every grid point with d_O > 2.6 Å outside the
  pocket interiors lies at least 3.20 Å from a block centre (Gard & Tait 3.26; Schlenker
  3.21; IZA 3.20). With d_O > 3.0 Å the minimum is at least 5.37 Å.

---

## 9. Pathways and the coordinate between two neighbouring cages

### 9.1 Plain statement

**No cage-to-cage connection lies in the xy-plane between cage centres.** The two cages that
share a window have centres offset by Δz = ±c/2 (7.525 Å, Gard & Tait).

**The pathway that does lie in the xy-plane is the window passage itself.** It is the
horizontal straight segment at the window height (z = 0 or c/2). It runs along the
horizontal B→C direction n̂_xy, from the B-column axis point (1/3, 2/3, z_w) through the
window centre (1/2, 1/2, z_w) to the C-column axis point (2/3, 1/3, z_w). Its length is
a/√3 = 7.661 Å (Schlenker 7.651), i.e. ±3.83 Å about the window. This is PNAS's "xy hop"
(λ_xy ≈ 0.75 nm). The axis points at window height are the cages' "top" and "bottom" in
JPCB 2006's z coordinate (q_A at s_z = 1/2, q_B at s_z = 1).

**Connecting the two cage centres through one window:**
* **Straight line, P1.** From (1/3, 2/3, 3/4) to (2/3, 1/3, 1/4), 10.739 Å, tilted 44.5°
  to the xy-plane. It passes exactly through the window centre, but it is not a good path.
  About 1.5 Å before and after the ring plane (3.9 Å from each cage centre) it comes within
  d_O = 2.918 Å of the ring rim. A single bead sees −296 K there, against −1325 K at the
  window centre.
* **Curved / bent path, P3, "axis–window–axis".** Go down (or up) the cage axis by c/4
  (3.76 Å) to the window height. Cross horizontally through the window (7.66 Å). Then go
  along the neighbour's axis by c/4 to its centre. The total is 15.19 Å. Along this path
  the minimum clearance is the window centre itself (d_O 3.134 Å), and the bead energy
  never rises above −1165 K.
* **The papers' own coordinates** [LIT, JPCB 2006]:
  * Both are fractional projections of one ethane bead: "q as the position of one of the
    two beads". The papers' Eqs. 10-11 and pp. 3166-3167 (Sec. 2.3) put the xy barrier at
    s_xy = (1/2, 1/2) and use q = s_z within cage A for z.
  * In my frame, cage A of their Eq. 10 (s_x + 1 − s_y < 1) is the B column (1/3, 2/3).
    Its z coordinate runs from the window height s_z = 1/2, through the cage centre at 3/4,
    to the window height at 1. That is exactly the cage I computed.

### 9.2 Clearance and indicative single-bead energy along candidate paths (Gard & Tait)

| Path | Length (Å) | min d_O (Å), where | bead LJ: start / min / max (K) |
|---|---|---|---|
| P1 straight, cage centre → cage centre | 10.739 | 2.918, 3.89 Å from the start (rim clip) | −1190 / −1501 / **−296** |
| P2 horizontal at window height, B axis → C axis | 7.661 | 3.134, at the window centre | −1391 / −1904 / −1325 (at the window) |
| P3 bent: axis → window height → window → axis → centre | 15.186 | 3.134, at the window | −1190 / −1904 / −1165 |
| P4 straight along the tilted ring normal, same xy span | 7.799 | 3.134, at the window | −1221 / −1883 / −1221 (ends) |

Along P2 the bead sees two adsorption minima (−1904 K), each 2.3 Å from a column axis, i.e.
about 1.5 Å inside each cage. The window centre is a local maximum (−1325 K), and the
window-plane minimum of the bead energy is exactly the ring centre. The window barrier from
the P2 minimum is:

| Structure | Window barrier (K) |
|---|---|
| Gard & Tait | 579 |
| Schlenker | 353 |
| IZA DLS | 951 |
| Ballirano | 732 |
| Hybrid | 666 |

**Rigid ethane with its COM at the window centre** [COMP, indicative]:

| | Gard & Tait | Schlenker | IZA DLS |
|---|---|---|---|
| best energy | −3286 K | −3569 K | −2921 K |
| ERI-cage minimum | −3792 K | −3789 K | −3918 K |

The best orientation is within about 10° of the ring normal, at the 300-direction grid
resolution of about 7°. Only about 6% of orientations lie within 600 K of that best value,
i.e. an orientational (entropic) restriction at the window. These are potential energies,
**not free energies**, with O-only LJ. They are given only to compare candidate paths and
structures.

### 9.3 Geometric facts that matter for a collective-variable design (no recommendation)

* **The window plane.** Its horizontal normal n̂_xy is shared by a whole stack of windows:
  at fixed xy (1/2, 1/2) there are windows every c/2 in z, alternately leading to the B
  cage above and the B cage below. The vertical plane through the window also contains the
  A columns (CAN/D6R) at (0, 0) and (1, 1), which are inaccessible.
* **Consequence for a projection CV.** A projection ξ = n̂_xy·(r − r_w) does not tell the
  "up" window from the "down" one. By symmetry they are equivalent, but a hop-specific CV
  needs a local cage-pair frame plus wrapping.
* **The literature fractional definitions.** The window lies at s_xy = (1/2, 1/2), cage A
  satisfies s_x + 1 − s_y < 1, and the z coordinate within cage A is q = s_z, with window
  heights at 1/2 and 1 and the cage centre at 3/4. All of these hold in the standard CIF
  setting used here: the IZA, RASPA2 and COD files share the same origin, with the 6₃ axis
  and D6R at (0, 0).

---

## 10. Open issues and what was not recovered

1. **[FLAG, major] Which ERI framework the PNAS data used is not determinable.** The text,
   the Fig. 3 caption and JPCB 2006 point to Gard & Tait (13.27/15.05; 54 cages in a
   3×3×3 hexagonal box). The PNAS box, JPCB 2003 and Beerdsen 2006 point to a 13.252/14.81
   cell (72 cages in 45.906 × 39.756 × 44.43 Å).
   * The 13.252/14.81 cell is exactly Schlenker et al. 1977. Whether the group used
     Schlenker's coordinates, or Gard & Tait's fractional coordinates in that cell, is
     unknown.
   * The window aperture and the indicative barrier differ between these options (Secs. 7.1,
     9.2). The single-bead barrier is 353 K (Schlenker), 579 K (Gard & Tait) and 666 K
     (hybrid), a spread of about 1.9×.
2. **Gard & Tait 1973 itself was not obtained.** It is conference proceedings and I found no
   online copy. My only verification is the archived IZA record (R = 0.15), which matches
   RASPA2 digit for digit.
3. The task text mentioned a "Gard & Tait 1973 Acta Cryst B29, 2007" paper. **I found no
   such erionite paper.** Every source (IZA Atlas, the archived IZA record, PNAS, JPCB
   2006, Beerdsen 2006, RASPA2) cites *Proc. 3rd Int. Conf. Molecular Sieves*, pp. 94-99.
   Alberti et al. (1997), *Zeolites* 19, 349, is cited by the IZA datasheet but was not
   retrieved.
4. The IZA free diameter 5.1 Å versus my 5.04 Å for the same coordinates is unexplained
   (Sec. 7.1).
5. PNAS uses "O-only", yet its parameter list includes Si-CH₃, and JPCB 2006 says O + Si.
   My indicative energies are O-only. Any production calculation must take the force field
   from the force-field reviews.
6. The PNAS equations (images) and Fig. 3 were not retrieved by me, because of a reCAPTCHA.
   The JPCB 2006 text (Sec. 2.3) gives the coordinate definitions instead.
7. Blocking: no source states that blocking was used for ERI in 2005. The RASPA2 block file
   is later (2011/2015).
8. All energies here are single-configuration potentials on a rigid framework with a
   simplified LJ model. They are meant for comparison only, and are not free-energy
   barriers.

---

## 11. Scripts and reproduction

All scripts are in `/tmp/claude-1008/-home-zheyuanlai-ABF-Fisher-Rao/eri_struct_scratch/`.
They use numpy only, run on CPU, and are run with `python -I`.

* `eri_geometry.py` is the main script. It parses the CIF, expands the symmetry, checks
  T–O distances, enumerates rings (primitive 4-, 6- and 8-cycles), measures the windows,
  builds the void grid and components, computes connectivity and paths, gives indicative
  LJ energies, and builds and checks the 2√3a × 3a × 3c supercell and the PNAS-box test.
* `summarize.py` prints the comparison tables (output in `summary.txt`).
* `coordseq.py` gives the coordination sequences.
* `check_pockets.py` gives the rigid-ethane energies in the CAN, D6R, ERI and window
  regions (output in `check_pockets.log`).
* `supercell_table.py` gives the cage–window graph in the box frame.
* `block_clearance.py` gives the distance from the RASPA2 block centres to the accessible
  void.
* `match_sites.py` matches O/T orbits between structures and lists the T neighbours of
  each O.
* `derived/HYBRID_GT_frac_in_Schlenker_cell.cif` is the RASPA2 CIF with only the cell
  lengths changed to 13.252/14.81.
* Outputs: `out/*_report.json`, `out/*_supercell.xyz` and `out/*_supercell_sites.json`;
  logs in `out_run.log` and `out_run_hybrid.log`.

To reproduce, with `S` the scratch directory and `B` the cache directory:

```
OMP_NUM_THREADS=4 python -I $S/eri_geometry.py $S/out \
  IZA_DLS=$B/iza_europe/ERI.cif RASPA2_ERI_SI=$B/raspa2_ERI_SI_cif/ERI_SI.cif \
  Schlenker1977=$B/cod_2106432/2106432.cif Ballirano2009=$B/cod_9013676/9013676.cif
python -I $S/summarize.py $S/out
python -I $S/coordseq.py NAME=CIF ...
python -I $S/check_pockets.py NAME=CIF ...     # about 9 min on CPU for 3 structures
```

The scratch directory is outside the repository and may be cleaned. The key numbers are
reproduced in this file.

Related review: `docs/eri_campaign/review/LIT_PNAS2005.md` (parallel PNAS extraction). It
reaches the same box arithmetic but did not find the origin of the 13.252/14.81 cell. This
file identifies it as Schlenker et al. 1977 and confirms that RASPA2 = Gard & Tait.
