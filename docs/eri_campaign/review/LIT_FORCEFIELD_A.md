# LIT_FORCEFIELD_A: the force field behind Dubbeldam et al., PNAS 102, 12317 (2005), ethane in ERI

Independent extraction A, 2026-10-16. I worked only from primary sources and the authors' own repositories. Every number below comes with a quote and a location. Where I inferred something, I label it **inference**. No GPU was used, and no downloaded file was executed.

---

## 0. Bottom line

1. **The PNAS 2005 ethane/ERI study did not use the Dubbeldam 2004 (JPCB/PRL) alkane–zeolite force field.** The paper names its non-bonded parameters explicitly and attributes them to Schüring, Auerbach, Fritzsche & Haberlandt, J. Chem. Phys. 116, 10890 (2002) (PNAS ref. 19):
   σ(O–CH3) = 3.17 Å, ε(O–CH3)/kB = 142 K; σ(Si–CH3) = 2.12 Å, ε(Si–CH3)/kB = 82 K; σ(CH3–CH3) = 3.78 Å, ε(CH3–CH3)/kB = 104 K.
   The bond is harmonic, with k1/kB = 96,500 K/Å² and r0 = 1.54 Å.
   PNAS cites the Dubbeldam 2004 papers (refs 21, 22) only to support the accuracy of the simulation techniques, not as the source of the parameters.
2. A second Dubbeldam paper confirms this. Dubbeldam, Beerdsen, Vlugt & Smit, J. Chem. Phys. 122, 224712 (2005) is PNAS ref. 25 ("More details can be found in ref. 25"). Its Table II reproduces the same set, labelled "Force field LJ parameters used by Schüring et al. for ethane in cation-free LTA-type zeolite". The authors also comment on it: "their very small size parameters enhance diffusion by two orders of magnitude for ethane in LTA-type zeolite compared to the parameters of Dubbeldam et al. Although the parameters used by Schüring et al. are probably less realistic in our opinion, they are convenient to compare …".
3. The Dubbeldam 2004 model (JPCB 108, 12301, Table 3; PRL 93, 088302, Table I) is:
   - CH3–O: 93.00 K, 3.48 Å
   - CH3–CH3: 108.00 K, 3.76 Å
   - oxygen-only host–guest interaction (no Si term)
   - LJ "shifted and cut at 12 Å", with no tail corrections
   - harmonic bond, k1/kB = 96500 K/Å², r0 = 1.54 Å
   - chargeless beads, rigid framework
4. The repo's LTA engine (`src/lta/core_lta.py`) implements the **Dubbeldam 2004** cross term (93.0 K, 3.48 Å, O only), not the PNAS/Schüring set. It deviates from Dubbeldam 2004 in two places:
   - cutoff 10 Å instead of 12 Å
   - bond constant 400 kJ mol⁻¹ Å⁻² instead of 96500 K/Å² = 802.35 kJ mol⁻¹ Å⁻²
   It also carries two wrong labels ("TraPPE-zeolite family", "TraPPE-UA ethane"). See §7.
5. Consequence for the campaign: an ERI engine that copies the LTA engine's parameters is a *Dubbeldam-2004 ERI model*. It is **not** a reproduction of PNAS 2005, so any literature-agreement claim against PNAS Fig. 3 would be invalid. Reproducing PNAS requires the Schüring set, and that raises the open questions in §8 (the Si term and the cutoff convention).

---

## 1. Sources obtained (all under `cache/eri/literature/forcefield_A/`)

| # | What | Local path | sha256 (first 16) | Status |
|---|---|---|---|---|
| S1 | Dubbeldam, Beerdsen, Calero, Smit, PNAS 102, 12317–12320 (2005), doi 10.1073/pnas.0503908102 | not saved (see note) | – | Full text read through the web_fetch tool from the PMC HTML (https://pmc.ncbi.nlm.nih.gov/articles/PMC1194924/) and the PMC PDF (…/pdf/pnas-0503908102.pdf). The two renderings agree. |
| S2 | Dubbeldam, Calero, Vlugt, Krishna, Maesen, Smit, J. Phys. Chem. B 108, 12301–12313 (2004), doi 10.1021/jp0376727 (publisher PDF, 13 pp.) | `jpcb2004_epfl_pdf/0689513732dub041.pdf` | b2a3a669d3addc5a | Open access, EPFL Infoscience record 200694. MD5 759f80b09efc1c25171ee99e3f5f0785 matches the checksum the repository lists. |
| S3 | Dubbeldam et al., PRL 93, 088302 (2004), doi 10.1103/PhysRevLett.93.088302 (publisher version) | `prl2004_uu_dspace_pdf/prl2004_dubbeldam.pdf` | b3a2970b7000e2ec | Utrecht University DSpace handle 1874/15925. |
| S4 | Dubbeldam, Beerdsen, Vlugt, Smit, J. Chem. Phys. 122, 224712 (2005), doi 10.1063/1.1924548 (= PNAS ref. 25) | `jcp2005_epfl_pdf/Dubbeldam-2005-Molecular_simulation.pdf` | 2e932d3d0bcb9302 | EPFL Infoscience record 200675. MD5 31ea9c015fffa98c700e1c792f6116d3 matches the repository checksum. |
| S5 | Wick, Martin, Siepmann, "TraPPE 4" preprint (OSTI 759881; published as J. Phys. Chem. B 104, 8008 (2000)) | `trappe4_osti/osti_759881.pdf` | d8e304ede4536bf6 | Scanned preprint. I read Tables 1–2 and p. 6 from page renders, not from OCR. |
| S6 | RASPA2 at commit 6498ab1eec9c8e0f063dcd0d71dd7add372c529b (2024-07-13): `forcefield/ExampleZeolitesForceField/{force_field_mixing_rules.def, pseudo_atoms.def}`, `molecules/ExampleDefinitions/{ethane,propane,methane,butane}.def`, `Docs/{Potentials/potentials.tex, InputFiles/input_files.tex, Examples/examples.tex}` | `raspa2_ExampleZeolitesForceField/`, `raspa2_ExampleDefinitions/`, `raspa2_Docs/` | see `sha256sum` | raw.githubusercontent.com, pinned to the commit. |
| S7 | RASPA3 at commit 564b89329c1fd302b12c382995ad08d25d2c41ba (2026-10-06): example `force_field.json` files and `ethane.json` | `raspa3_examples/` | see `sha256sum` | Pinned to the commit. The tree listings were saved in `raspa2_listings/raspa3_*.json` (misnamed directory, same content). |
| S8 | Schüring, Fritzsche, Auerbach, Diffusion Fundamentals 2, 104 (2005) | `schuring2005_diffusion_fundamentals/df2_290.pdf` | 2416e478726cc543 | Two-page abstract. It contains **no** potential parameters. |
| – | Failed or irrelevant downloads, kept for provenance | `pnas2005_europepmc/` (NCBI efetch: metadata only, "The publisher of this article does not allow downloading of the full text in XML form."), `pnas2005_pdf_europepmc/` (Cloudflare challenge page, not a PDF), `jpcb2004_epfl_landing/` (empty or 404), `jcp2005_uu_dspace_landing/` (no file attached) | | |
| – | **Wrong file in a repository** | `jpcb2004_uu_dspace_pdf/jpcb2004_dubbeldam.pdf` | 402a23fe6dbf83df | UU DSpace handle 1874/16710 is titled for the JPCB 2004 paper, but the file attached is *Surface Science 529 (2003) 197–203* (van Driel et al., "Hot carrier luminescence during porous etching of GaP"). Do not cite it as JPCB 2004. |

Not obtained: Schüring, Auerbach, Fritzsche, Haberlandt, J. Chem. Phys. 116, 10890 (2002), doi 10.1063/1.1480011 (DOI from Crossref). OpenAlex lists it as closed access ("oa_status: closed", "any_repository_has_fulltext: False"), and I found no author or repository copy. Martin & Siepmann, J. Phys. Chem. B 102, 2569 (1998) is also closed access in OpenAlex, so I used S5 for the TraPPE-UA values.

---

## 2. PNAS 2005 (S1): exact statements about the model

All quotes are from "Model and Computational Details", p. 12317. For symbols I used the PMC HTML rendering, which keeps σ and ε. The PMC PDF text drops the Greek letters and "=" signs but has the same numbers and the same "ref. 19".

- Framework: "In our simulations, we neglect cations and study rigid, all-silica versions of the ERI- and CHA-type zeolites. … The positions of the atoms are taken from refs. 14 and 15, respectively. Following the work of Bezus *et al.* (16), the zeolites are modeled as a rigid network of oxygen atoms. This is a very common approximation because the large oxygen atoms essentially shield the much smaller silicon atoms, and lattice flexibility is not important for small alkanes in all-silica zeolites (17)." (ref. 14 = Gard & Tait 1973; ref. 16 = Bezus, Kiselev, Lopatkin, Du 1978.)
- Box: "The rectangular simulation box sizes we used are 4.5906 × 3.9756 × 4.443 nm for ERI-type zeolite and 3.015 × 4.7814 × 2.7606 nm for CHA-type zeolite."
- Guest: "For these reasons the united atom model (18) is a straightforward choice. We consider the CH₃ groups as single, chargeless interaction centers with their own effective potentials."
- Bond: "The beads of ethane are connected by a harmonic bonding potential *U*bond = ½ *k*₁(*r* − *r*₀)², with *k*₁/*k*B = 96,500 K/Å² and *r*₀ = 1.54 Å." The formula is an inline graphic in the HTML. The PDF text reads "2 k1(r r0)2, with k ¹ kB 96,500 KÅ ² and r0 1.54 Å".
- **Non-bonded**: "The non-intramolecular energy is described with a Lennard–Jones potential by using parameters σO–CH3 = 3.17 Å, εO–CH3/*k*B = 142 K, σSi–CH3 = 2.12 Å, εSi–CH3/*k*B = 82 K, and σCH3–CH3 = 3.78 Å, εCH3–CH3/*k*B = 104 K, which were taken from ref. 19."
  - The PMC HTML has typesetting errors: "σ Si–CH = 2.12 Å" and "ε CH3–CH33 / kB = 104 which K, were taken". The PDF reads "SiCH3 2.12 Å … CH3CH3 kB 104 K, which were taken from ref. 19", so the version above is the correct one.
- Ref. 19: "Schüring, A., Auerbach, S. M., Fritzsche, S. & Haberlandt, R. (2002) J. Chem. Phys 116, 10890–10894."
- Validation sentence: "The accuracy of the simulation techniques have been verified in several publications (6, 20–23) …". Refs 21 and 22 are the Dubbeldam 2004 PRL and JPCB papers. They are cited for technique validation, **not** as the parameter source.
- MD details: "velocity Verlet integration scheme with a time step of 0.5 fs"; "Nosé–Hoover chain (NHC) method"; "production run of more than 20 ns"; "More details can be found in ref. 25." (ref. 25 = S4.)
- Conditions: Fig. 1–3 captions refer to "600 K". The Fig. 3 caption gives "Free energy profiles βF(q) at 600 K of ethane in ERI at various loadings (infinite dilution, 1, 2, …, 7 molecules per ERI-type cage)".
- **Not stated anywhere in PNAS:** the LJ cutoff radius, whether the potential is shifted or truncated, tail corrections, and mixing rules (all pair parameters are given explicitly).

## 3. Same parameter set as reproduced in Dubbeldam et al. JCP 2005 (S4)

- p. 224712-3, Table II (exact caption): "Force field LJ parameters used by Schüring et al. for ethane in cation-free LTA-type zeolite (Ref. 24). Lennard-Jones parameters, ε/kB [K] in the top of each field, σ [Å] in the bottom of each field, bond and bend parameters."

  | | O | Si | CH3 |
  |---|---|---|---|
  | CH3 | 142 / 3.17 | 82 / 2.12 | 104 / 3.78 |

  Bond row: "Ubond = ½ k1(r − r0)², k1/kB = 96 500 K/Å², r0 = 1.54 Å".
- p. 224712-3, text: "The parameters listed in Table II are used mainly for comparison with the simulations of Schüring et al.,24,43 and because their very small size parameters enhance diffusion by two orders of magnitude for ethane in LTA-type zeolite compared to the parameters of Dubbeldam et al. Although the parameters used by Schüring et al. are probably less realistic in our opinion, they are convenient to compare the diffusion of ethane from a simulation-method point of view by MD and dcTST in LTA-type zeolite."
- p. 224712-2 to -3, §II.A, which introduces both parameter tables: "The Lennard-Jones potentials are shifted and cut at 12 Å. Analytical tail corrections are not applicable in zeolites.35 A truncated and shifted potential is equally suitable to Monte Carlo and molecular dynamics."
- Table I of S4 (the Dubbeldam set) has an Si column filled with "—" (no Si interaction). Table II (the Schüring set) has an Si column with numbers. **Inference:** the Schüring set as Dubbeldam used it includes an explicit Si–CH3 term.
- p. 224712-13, §III.A.3 "Ethane in LTA-type zeolite": "Figure 11 shows that the behavior found by Schüring et al.24 is strongly dependent on the parameter set (Tables I and II). The parameter set of Dubbeldam et al.39 does not show a decrease with increasing temperature … The size parameter used by Schüring et al. is so small that ethane at low temperature is found in the windows itself, and heating shifts the adsorption sites to just in front of the windows. In contrast, the set of Dubbeldam et al. has a larger size parameter for ethane and also at low temperature the adsorption sites are always in front of the window, reducing the behavior found by Schüring et al."
  This matters for an entropic-barrier campaign: whether the window acts as an energetic or entropic feature depends on the parameter set.

## 4. Dubbeldam 2004 model (S2 JPCB, S3 PRL)

- JPCB p. 12302, §II.A: "We consider the CHx groups as single, chargeless interaction centers with their own effective potentials. The beads in the chain are connected by harmonic bonding potentials. A harmonic cosine bending potential models the bond bending between three neighboring beads, and a Ryckaert-Bellemans potential controls the torsional angle. The beads in a chain separated by more than three bonds interact with each other through a Lennard-Jones potential. The Lennard-Jones potentials are shifted and cut at 12 Å. Analytical tail-corrections do not apply in zeolites.5 A truncated and shifted potential is equally suitable to Monte Carlo and molecular dynamics. Flexibility of the framework is not an issue for adsorption of linear and branched alkanes.11 The interactions between the rigid framework and the guest molecules are assumed to be dominated by the oxygen atoms.12"
  - ref 5 = Macedonia & Maginn, Mol. Phys. 96, 1375 (1999)
  - ref 11 = Vlugt & Schenk, J. Phys. Chem. B 106, 12757 (2002)
  - ref 12 = Bezus et al. 1978
- JPCB p. 12305, Table 3 ("Force Field Guest-Host and Guest-Guest Interactions of Hydrocarbons in Charge Neutral Nanoporous Materials"). Values are ε/kB [K] above σ [Å]. I checked the table against a page render.

  | | O | CH4 | CH3 | CH2 | CH | C |
  |---|---|---|---|---|---|---|
  | CH4 | 115.00 / 3.47 | 158.50 / 3.72 | 130.84 / 3.74 | 94.21 / 3.84 | 51.91 / 4.17 | 11.26 / 4.87 |
  | CH3 | **93.00 / 3.48** | 130.84 / 3.74 | **108.00 / 3.76** | 77.77 / 3.86 | 42.85 / 4.19 | 9.30 / 4.90 |
  | CH2 | 60.50 / 3.58 | 94.21 / 3.84 | 77.77 / 3.86 | 56.00 / 3.96 | 30.85 / 4.30 | 6.69 / 5.03 |
  | CH | 40.00 / 3.92 | 51.91 / 4.17 | 42.85 / 4.19 | 30.85 / 4.30 | 17.00 / 4.67 | 3.69 / 5.46 |
  | C | 10.00 / 4.56 | 11.26 / 4.87 | 9.30 / 4.90 | 6.69 / 5.03 | 3.69 / 5.46 | 0.80 / 6.38 |

  - Bond row: "Ubond = 1/2 k1(r − r0)², k1/kB = 96500 K/Å², r0 = 1.54 Å".
  - Bend row: "Ubend = 1/2 k2(cosθ − cosθ0)², k2/kB = 62500 K/rad², θeq = 114°".
  - Footnote: "Some of the alkane-alkane interactions are taken from ref 66 and optimized to reproduce vapor-liquid coexistence curves of the phase diagrams, the internal bond from ref 67, the internal bend from ref 41, and the torsion from T. J. H. Vlugt and M. Frash.68"
    - ref 66 = Martin, Thompson, Nenoff, J. Chem. Phys. 114, 7174 (2001)
    - ref 67 = Nath, Escobedo, de Pablo, J. Chem. Phys. 108, 9905 (1998)
    - ref 41 = Martin & Siepmann, J. Phys. Chem. B 103, 4508 (1999)
- PRL p. 088302-2, Table I has the identical matrix (CH3: O 93.00/3.48, CH3 108.00/3.76, …). Its text says: "The Lennard-Jones potentials are shifted and cut at 12 Å. The interactions between the rigid framework and the guest molecules are assumed to be dominated by the oxygen atoms [7]. Some of the alkane-alkane interactions are taken from Ref. [8]." ([7] = Bezus et al.; [8] = Martin, Thompson & Nenoff 2001.) The Table I caption adds: "The parameters may be combined with any reasonable internal interaction model."
- JPCB p. 12311, Appendix c, Eq. 14: U = Σ 4ε_ij[(σ_ij/r_ij)¹² − (σ_ij/r_ij)⁶] − E_cut, "where rij is the distance between site i and site j, rcut = 12.0 Å, the cutoff radius, Ecut the energy at the cutoff radius, and U = 0 when rij > rcut." This is an **energy shift**: the potential is continuous at rc and the force is not shifted. The same page: "where k1/kB = 96500 K/Å2 is the bond energy constant, r0 = 1.54 Å the reference bond length".
- JPCB p. 12310, on silicon: "In most nanoporous framework structures, the large oxygen atoms shield the much smaller silicon, aluminum, and phosphorus atoms. Therefore, the model only needs to consider interactions between the adsorbate and the oxygen atoms, provided there is no net negative electrical charge on the framework.12"
- JPCB p. 12304, on fitting CH3–O: "The εO-CH3 and σO-CH3 are uniquely obtainable from the ethane isotherm." Also from p. 12305: "The εO-CH3 and σO-CH3 parameters are uniquely fixed with a precision better than 0.01 Å for σ and better than 1 K for ε/kB." There is **one** CH3 type, fitted on ethane, and it is used for every alkane.
- JPCB p. 12306, §II.G, on sensitivity to the bond model: "We modified the harmonic bond potential to a fixed distance of 1.54 Å and recomputed the isotherm of ethane. … The results for the three test cases shown in Figure 7 suggest a minimal dependency of the adsorption results on the internal interaction parameters and that the adsorbent-adsorbate interaction parameters may be combined with any other physically reasonable internal interaction model." This statement was tested on MFI isotherms, not on window free-energy barriers.
- Mixing (**inference**, my arithmetic): in Table 3 every guest–guest off-diagonal ε equals √(ε_iε_j). Every off-diagonal σ equals **√(σ_iσ_j)** to the printed precision, not (σ_i + σ_j)/2. Examples: CH3–C 4.90 = √(3.76·6.38) = 4.898, while the arithmetic mean is 5.07; CH4–C 4.87 = 4.872, arithmetic 5.05. So the table is geometric for both ε and σ, not Lorentz–Berthelot. This does not matter for pure ethane (CH3–CH3 is the diagonal 108.00/3.76). It does matter if someone rebuilds mixtures from the RASPA "Lorentz-Berthelot" files (§6). Host–guest O–CHx values are fitted directly; no O–O parameter exists.

## 5. CH3 in ethane vs CH3 in longer alkanes

- **Dubbeldam 2004:** a single CH3 type, fitted on ethane and used for all alkanes (§4). There is no ethane-specific CH3.
- **TraPPE-UA:** also no distinction. S5 (TraPPE-4 preprint), Table 1, p. 27 ("Comparison of the Lennard-Jones Parameters for the OPLS-UA, SET, and TraPPE-UA Force Fields"), read from the page render:
  - CH3 (sp3), ethane: OPLS-UA 3.775 Å / 104.1 K; SET 3.64 / 112.2; TraPPE-UA **3.75 / 98.0**
  - CH3 (sp3), n-alkane: OPLS-UA 3.905 / 88.1; SET 3.94 / 49.32; TraPPE-UA **3.75 / 98.0**
  - CH2 (sp3), n-alkane: TraPPE-UA 3.95 / 46.0
- **OPLS-UA does have an ethane-specific CH3**, 3.775 Å / 104.1 K (Jorgensen, Madura, Swenson, JACS 106, 6638 (1984), as cited in S5 Table 1).
- **Inference:** the PNAS/Schüring CH3–CH3 pair, 3.78 Å / 104 K, equals OPLS-UA ethane CH3 rounded. PNAS does not say this, and I could not check it in Schüring 2002.

## 6. RASPA2 / RASPA3 force-field files (S6, S7)

- **RASPA2 has no "Dubbeldam2004" zeolite force field.** At commit 6498ab1, `forcefield/` contains only: CoreShellSchroderSauer, Dubbeldam2007FlexibleIRMOF-1/-10/-16, ExampleMOFsForceField, ExampleMoleculeForceField, ExampleZeolitesForceField, Nicholas. `molecules/` contains only ExampleDefinitions.
- `forcefield/ExampleZeolitesForceField/force_field_mixing_rules.def`, verbatim:
  - `shifted` (general rule)
  - tail corrections `no`
  - `O_ lennard-jones 53.0 3.30 // P. Bai, M. Tsapatsis, J. I. Siepmann, J. Phys. Chem. C 2013, 117, 24375-24387.`
  - `Si_ lennard-Jones 22.0 2.30 // idem`
  - `CH3 lennard-jones 108.0 3.76 // D. Dubbeldam et al., J. Phys. Chem. B, 108(33), 12301-12313`
  - `CH2 … 56.0 3.96`, `CH … 17.0 4.67`, `C … 0.8 6.38` (all `// idem`)
  - `CH4 158.5 3.72 // M. G. Martin et al., J. Chem. Phys. 2001`
  - mixing rule `Lorentz-Berthelot`
  - There is **no `force_field.def`** in this directory, so no explicit O–CH3 = 93 K / 3.48 Å override exists.
  - Host–guest is therefore **TraPPE-zeo (O 53 K / 3.30 Å, Si 22 K / 2.30 Å) mixed by Lorentz–Berthelot with the Dubbeldam guest diagonals**. By my arithmetic O–CH3 comes out at √(53·108) = 75.7 K and (3.30 + 3.76)/2 = 3.53 Å. This is **not** the Dubbeldam 2004 host–guest model.
  - `Docs/Examples/examples.tex` says: "This example uses a generic zeolite force field based on TraPPE-zeo \cite{Bai2013}".
- `molecules/ExampleDefinitions/ethane.def`, verbatim: atoms `0 CH3`, `1 CH3`; `# Bond stretch: atom n1-n2, type, parameters` / `0 1 HARMONIC_BOND 96500 1.54`.
- RASPA2 manual, `Docs/Potentials/potentials.tex`: "HARMONIC\_BOND … U=\frac{1}{2} p_0 (r-p_1)^2 … 2 arguments: $p_0/k_B$ in units of K/\AA$^2$, $p_1$ in \AA." So the ethane definition encodes ½·(96500 K/Å²)·(r − 1.54 Å)², the same as JPCB Table 3.
- RASPA3 at commit 564b893 likewise has no Dubbeldam-2004 zeolite file.
  - `examples/basic/15_md_butane_isobutane_in_fau/force_field.json` (identical to examples 9 and non_basic 8) uses `"MixingRule": "Lorentz-Berthelot"`, `"TruncationMethod": "shifted"`, `"TailCorrections": false`, `"CutOffVDW": 12.0`, O = [53.0, 3.30] and Si = [22.0, 2.30] (Bai 2013, charges O −1.025 and Si +2.05), and CH3 = [108.0, 3.76] "D. Dubbeldam et al., J. Phys. Chem. B 2004".
  - `examples/basic/4_mc_binary_mixture_propane_butane_in_box/ethane.json` has `"Bonds": [[["CH3","CH3"], "FIXED", [1.54]]]`. Its `force_field.json` is TraPPE (CH3 [98.0, 3.75], CH2 [46.0, 3.95], `"TruncationMethod": "truncated"`, `"TailCorrections": true`, `"CutOff": 12.0`).

## 7. TraPPE-UA (S5) conventions

- p. 6: "The LJ parameters for the interactions of unlike pseudo-atoms are calculated using the standard Lorentz-Berthelot combining rules15,16". Then: "A spherical potential truncation for pairs of pseudo-atoms separated by more than 14 Å was enforced, and analytic tail corrections (for energies, pressures, and chemical potentials) were employed.17,18"
- p. 6 also says the model uses "fixed bond lengths for neighboring pseudo-atoms".
- Table 2, p. 28: "CHx − CHy 1.54" Å. The TraPPE website (trappe.oit.umn.edu) says "TraPPE uses fixed bond lengths".
- TraPPE-UA ethane is therefore CH3 98.0 K / 3.75 Å, a fixed 1.54 Å bond, and 14 Å truncation with tail corrections. Neither PNAS 2005 nor Dubbeldam 2004 uses this convention.

---

## 8. Parameter table

Confidence scale: **H** means verbatim in a primary source I read; **M** means stated in a primary source with a typesetting defect, or confirmed in a second source by the same authors; **L** means inference.

### 8a. Model actually used in PNAS 2005 (ethane in ERI)

| Parameter | Literature value | Units | Exact source + quote/location | Confidence |
|---|---|---|---|---|
| σ O–CH3 | 3.17 | Å | PNAS p. 12317: "σO–CH3 = 3.17 Å … taken from ref. 19". Also JCP 2005 Table II. | H |
| ε O–CH3/kB | 142 | K (= 1.18065 kJ/mol) | PNAS p. 12317: "εO–CH3/kB = 142 K". Also JCP 2005 Table II. | H |
| σ Si–CH3 | 2.12 | Å | PNAS p. 12317 (HTML typo "σ Si–CH"; PDF "SiCH3 2.12 Å"). Also JCP 2005 Table II. | M (typo) |
| ε Si–CH3/kB | 82 | K (= 0.68179 kJ/mol) | PNAS p. 12317. Also JCP 2005 Table II. | H |
| Si actually interacting? | Parameters are given, but the text says "modeled as a rigid network of oxygen atoms" | – | PNAS p. 12317. Contradiction; see U2. | **unresolved** |
| σ CH3–CH3 | 3.78 | Å | PNAS p. 12317. Also JCP 2005 Table II. | H |
| ε CH3–CH3/kB | 104 | K (= 0.8647 kJ/mol) | PNAS p. 12317 (HTML typo "CH33 … 104 which K"; PDF clean). Also JCP 2005 Table II. | M (typo) |
| Mixing rules | none needed; every pair given explicitly | – | PNAS p. 12317 | H |
| Ethane-specific CH3? | 3.78/104 equals OPLS-UA ethane CH3 (3.775/104.1) | – | S5 Table 1 + my comparison | L |
| Bond model | harmonic, U = ½k1(r − r0)² | – | PNAS p. 12317 | H |
| k1/kB | 96,500 | K/Å² (= 802.35 kJ mol⁻¹ Å⁻²) | PNAS p. 12317. Also JCP 2005 Table II. | H |
| r0 | 1.54 | Å | PNAS p. 12317 | H |
| Charges | none ("chargeless interaction centers") | – | PNAS p. 12317 | H |
| Framework | rigid, all-silica, cation-free | – | PNAS p. 12317 | H |
| LJ cutoff | not stated in PNAS. JCP 2005 (PNAS ref. 25) states "shifted and cut at 12 Å" in its force-field section | Å | JCP 2005 p. 224712-2 | L (inferred carry-over) |
| Shift / tail | not stated in PNAS. JCP 2005: shifted, "Analytical tail corrections are not applicable in zeolites" | – | JCP 2005 p. 224712-2 | L |
| Temperature (ERI results) | 600 | K | PNAS Fig. 1–3 captions | H |
| MD integrator / dt | velocity Verlet, 0.5 fs, NHC thermostat | – | PNAS p. 12317 | H |
| ERI box | 4.5906 × 3.9756 × 4.443 | nm | PNAS p. 12317 | H |

### 8b. Dubbeldam 2004 united-atom alkane–zeolite model (JPCB 108, 12301 / PRL 93, 088302)

| Parameter | Literature value | Units | Exact source + quote/location | Confidence |
|---|---|---|---|---|
| ε O–CH3/kB | 93.00 | K (= 0.77325 kJ/mol) | JPCB Table 3, p. 12305; PRL Table I, p. 088302-2 | H |
| σ O–CH3 | 3.48 | Å | JPCB Table 3; JPCB p. 12304 "σO-CH3 = 3.48 Å"; PRL Table I | H |
| Si (and Al) interactions | none ("only needs to consider interactions between the adsorbate and the oxygen atoms") | – | JPCB p. 12310; p. 12302 "dominated by the oxygen atoms"; JCP 2005 Table I Si column "—" | H |
| ε CH3–CH3/kB | 108.00 | K | JPCB Table 3; PRL Table I | H |
| σ CH3–CH3 | 3.76 | Å | JPCB Table 3; PRL Table I | H |
| Pair specification | explicit pair matrix. Guest–guest off-diagonals are geometric in both ε and σ (my check); host–guest O–CHx fitted | – | JPCB Table 3 + my arithmetic | H (explicit) / L (rule) |
| Ethane-specific CH3? | **No.** One CH3 type, fitted on the ethane isotherm in MFI | – | JPCB p. 12304 "uniquely obtainable from the ethane isotherm" | H |
| Cutoff | 12.0 | Å | JPCB p. 12302 "shifted and cut at 12 Å"; Eq. 14, p. 12311 "rcut = 12.0 Å"; PRL p. 088302-2 | H |
| Truncation | energy shifted to 0 at rc (subtract E_cut); force not shifted | – | JPCB Eq. 14, p. 12311 | H |
| Tail corrections | none ("Analytical tail-corrections do not apply in zeolites") | – | JPCB p. 12302 | H |
| Bond | harmonic, k1/kB = 96500 K/Å², r0 = 1.54 Å, U = ½k1(r − r0)² (source: Nath, Escobedo, de Pablo 1998) | K/Å², Å | JPCB Table 3 + p. 12311 | H |
| Bond sensitivity | fixed 1.54 Å vs harmonic made minimal difference to the ethane isotherm (MFI) | – | JPCB p. 12306 | H |
| Charges | none ("single, chargeless interaction centers") | – | JPCB p. 12302 | H |
| Framework | rigid ("Flexibility of the framework is not an issue …") | – | JPCB p. 12302 | H |

### 8c. TraPPE-UA ethane (comparison only)

| Parameter | Value | Units | Source | Confidence |
|---|---|---|---|---|
| CH3 σ, ε/kB | 3.75, 98.0 | Å, K | S5 Table 1, p. 27 (page render) | H (secondary to Martin & Siepmann 1998) |
| Bond | fixed 1.54 | Å | S5 Table 2, p. 28; TraPPE site | H |
| Cutoff / tail | 14 Å truncation + analytic tail corrections; Lorentz–Berthelot | – | S5 p. 6 | H |

---

## 9. Repo LTA engine vs the literature

Engine facts (`src/lta/core_lta.py`):
- Docstring lines 7–13: "Only the 384 framework O atoms interact with the guest (standard for alkane-zeolite force fields; Si is screened)"; "one TraPPE-UA ethane per replica: two CH3 pseudo-atoms, harmonic bond r0 = 1.54 A (stiff k, BD-integrable stand-in for TraPPE's fixed bond), no charges"; "cross LJ CH3-O: eps/kB = 93.0 K, sigma = 3.48 A (TraPPE-zeolite family, Dubbeldam et al.), truncated+shifted at rc = 10 A < L/2."
- `LTAParams` lines 80–84: `eps_go = 93.0 * KB`, `sigma_go = 3.48`, `rc = 10.0`, `r0_bond = 1.54`, `k_bond = 400.0  # kJ/mol/A^2  (0.5 k (r-r0)^2)`.
- Line 149: `assert self.p.rc < self.L / 2`.
- Lines 152–154 and 183–192: the energy is shifted by `v_rc`, the force is not shifted, and there is no tail correction.
- `configs/uniform_campaign/lta_prereg.json`: `system.cross_lj.family = "TraPPE-zeolite (Dubbeldam et al.)"`; `system.guest = "TraPPE-UA ethane: 2 CH3 beads, harmonic bond r0=1.54 A k=400 kJ/mol/A^2"`.

### 9a. Where the engine matches Dubbeldam 2004
- ε O–CH3 = 93.0 K and σ = 3.48 Å match exactly.
- Host–guest is O only (no Si), no charges, rigid framework.
- LJ energy shift at the cutoff with no tail correction is the same convention as JPCB Eq. 14.
- Harmonic form ½k(r − r0)² with r0 = 1.54 Å.
- CH3–CH3 non-bonded and intramolecular LJ do not apply: there is one ethane per replica, and ethane has no bead pairs more than 3 bonds apart. The CH3–CH3 parameter therefore never enters, so its value (108 K Dubbeldam, 98 K TraPPE, 104 K Schüring) does not matter for this engine.

### 9b. Where the engine deviates from Dubbeldam 2004
1. **Cutoff 10 Å vs 12.0 Å.**
   - The 2×2×2 pseudo-cell box has L/2 = 11.919 Å < 12 Å, so the minimum-image engine cannot use 12 Å without a larger box or explicit image sums.
   - Indicative size (my computation, CPU only, script in Appendix A): for one CH3 bead on the cage-centre → window → cage-centre line of `cache/lta/framework.npz`, U(rc = 10, shifted) − U(rc = 12, shifted) varies by **7.9 K (0.066 kJ/mol)** along the line.
   - The single-bead window-minus-cage-centre energy is 66.1 K at rc = 10 and 72.7 K at rc = 12 (73.6 K for unshifted 12 Å truncation). That is about 0.02 kBT per bead at 300 K.
   - This is not a free-energy calculation. It suggests a small but systematic effect that a reference calculation should quantify.
2. **Bond constant.** 400 kJ mol⁻¹ Å⁻² vs 96500 K/Å² × kB = 802.35 kJ mol⁻¹ Å⁻² (ratio 0.4985).
   - Bond-length standard deviation at 300 K: 0.079 Å (engine) vs 0.056 Å (literature); at 600 K: 0.112 vs 0.079 Å.
   - JPCB §II.G says adsorption depends minimally on the internal model, but that was tested on MFI isotherms. The effect on a window free-energy barrier is untested.
3. **Labels (documentation, not physics).**
   - "TraPPE-zeolite family" is incorrect. TraPPE-zeo is Bai, Tsapatsis & Siepmann 2013 (O 53 K / 3.30 Å, Si 22 K / 2.30 Å, Lorentz–Berthelot, partial charges, per S6/S7). Dubbeldam 2004 is a separate, Kiselev-type oxygen-only model.
   - "TraPPE-UA ethane" is also inaccurate for a Dubbeldam-2004 model: there CH3–CH3 is 108/3.76 and the bond is harmonic 96500 K/Å², while TraPPE uses 98/3.75 and a fixed bond. Only the r0 = 1.54 Å value is common to both.
4. **Dynamics and temperature.** The engine uses overdamped Langevin rather than MD; equilibrium properties are unaffected. Its LTA runs are at 300/150 K, while the PNAS ERI results are at 600 K.

### 9c. Where the engine deviates from the PNAS 2005 (Schüring) set
- ε 93.0 vs 142 K; σ 3.48 vs 3.17 Å.
- No Si–CH3 term (PNAS lists 82 K / 2.12 Å).
- Cutoff 10 Å vs an unstated value (12 Å inferred).
- A sanity check with the same script on the LTA framework: the Schüring O–CH3 makes the 8-ring window strongly attractive for a single bead (−1670 K at the window vs −381 K at the cage centre, O only). With Dubbeldam 2004 the window is repulsive relative to the cage centre (−368 vs −434 K). This matches the JCP 2005 statement that with the Schüring set "ethane at low temperature is found in the windows itself". The two sets give qualitatively different window physics.

---

## 10. Unresolved items

- **U1. Schüring et al., JCP 116, 10890 (2002) was not obtained** (closed access, no repository copy found).
  - Tried: OpenAlex (closed), Crossref (DOI only), AIP page (fetch returned an unrelated page), web searches for author or thesis copies, and Diffusion Fundamentals 2 (2005) 104 (abstract only, no parameters).
  - Cannot verify: (a) the original parameter values, their stated origin and precision; (b) whether Schüring included Si; (c) Schüring's cutoff, shift and tail convention; (d) whether the 96500 K/Å² bond belongs to Schüring or to Dubbeldam. Schüring may have used a different bond model; JCP 2005 Table II lists 96500 K/Å² with the Schüring set without saying whose bond it is.
  - Institutional access (e.g., NUS library) to doi 10.1063/1.1480011 would close this.
- **U2. Si in PNAS.** The text says the zeolite is "modeled as a rigid network of oxygen atoms" (Bezus/Kiselev) but lists σSi–CH3 = 2.12 Å and εSi–CH3/kB = 82 K. JCP 2005 Table II has a populated Si column for this set, which suggests the Si term was used, but neither paper says so explicitly. An ERI reproduction must decide this and declare the choice. The Si contribution is not negligible a priori (ε = 82 K); it can be quantified once the ERI framework exists.
- **U3. Cutoff/shift for the PNAS runs.** Not stated in PNAS. "Shifted and cut at 12 Å" is stated in Dubbeldam's own papers (JPCB 2004, PRL 2004, JCP 2005 §II.A), so it is most likely, but still an inference. The PNAS ERI box (smallest width 39.756 Å) is compatible with 12 Å.
- **U4. Origin of the Schüring values.** CH3–CH3 matches OPLS-UA ethane CH3 (inference). For O–CH3 142 K / 3.17 Å and Si–CH3 82 K / 2.12 Å I found no origin.
- **U5. Which set should the campaign use?** This is a decision for the campaign, not a literature fact.
  - "Ethane in ERI as in PNAS 2005" requires the Schüring set; anything else must not claim agreement with PNAS Fig. 3.
  - Dubbeldam himself calls the Schüring set "probably less realistic" and favours the 2004 set (JCP 2005 p. 224712-3).
  - The two sets differ by about two orders of magnitude in ethane diffusivity in LTA and in where the window minimum sits.
- **U6. Mixing rules.** JPCB Table 3 guest–guest cross terms follow geometric σ (my arithmetic), whereas the RASPA example files declare Lorentz–Berthelot. This is irrelevant for single-component ethane and relevant only for future mixtures.
- **U7. TraPPE primary.** Martin & Siepmann 1998 was not accessed (closed). The TraPPE values come from the Siepmann-group TraPPE-4 preprint (OSTI 759881) and the TraPPE website.
- **U8. Repository defect.** UU DSpace 1874/16710 serves the wrong PDF (§1). Use the EPFL copy (MD5 verified).
- **U9. Out of scope here:** ERI atomic positions (PNAS ref. 14, Gard & Tait 1973), and the ERI box and cell conversion.

## Appendix A: script for the §9 numbers (CPU only, reads the repo's own `cache/lta/framework.npz`)

Run as `python3 -I cutoff_effect.py cache/lta/framework.npz`. It was kept in the session scratchpad, which does not persist. The listing below matches the run, except that the formatted print statements are condensed into one.

```python
import sys, numpy as np
KB = 0.008314462618
z = np.load(sys.argv[1], allow_pickle=True)
O = np.asarray(z["o_pos"], float); L = float(z["box"]); a = float(z["a_pseudo"])
shifts = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)], float) * L
Oimg = (O[None, :, :] + shifts[:, None, :]).reshape(-1, 3)
def U(p, eps, sig, rc, shift=True):
    d = np.linalg.norm(Oimg - p, axis=1); d = d[d < rc]
    sr6 = (sig / d) ** 6; v = 4 * eps * (sr6 * sr6 - sr6)
    if shift:
        s6 = (sig / rc) ** 6; v = v - 4 * eps * (s6 * s6 - s6)
    return v.sum()
xs = np.linspace(-a / 2, a / 2, 61)   # window plane x=0 (mod a), cage centres x=+-a/2 (engine convention)
line = [np.array([x % L, 0.5 * a, 0.5 * a]) for x in xs]
for label, eps, sig in [("Dubbeldam2004", 93.0, 3.48), ("Schuering/PNAS", 142.0, 3.17)]:
    u10 = np.array([U(p, eps, sig, 10.0) for p in line]); u12 = np.array([U(p, eps, sig, 12.0) for p in line])
    u12t = np.array([U(p, eps, sig, 12.0, shift=False) for p in line])
    d = u10 - u12; i0 = np.argmin(abs(xs)); ic = 0      # i0 = window centre, ic = cage centre
    print(label, u10[i0], u12[i0], u10[ic], u12[ic], d.max() - d.min(),
          u10[i0] - u10[ic], u12[i0] - u12[ic], u12t[i0] - u12t[ic])
```

Output:
- nO = 384, L = 23.838 Å.
- Dubbeldam2004: U(window) = −367.5 K (rc 10) and −431.0 K (rc 12); U(cage centre) = −433.6 K and −503.8 K. The rc10 − rc12 spread along the line is 7.89 K (0.066 kJ/mol). Window − cage: 66.11 K (rc 10), 72.73 K (rc 12), 73.62 K (rc 12, unshifted).
- Schüring: U(window) = −1670.5 K (rc 10); U(cage centre) = −381.2 K (rc 10). The rc10 − rc12 spread is 6.89 K.
