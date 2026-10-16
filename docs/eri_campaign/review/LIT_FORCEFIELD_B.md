# LIT_FORCEFIELD_B: force field used for ethane in all-silica ERI (Dubbeldam et al., PNAS 2005)

*Independent extraction B, 2026-10-16. Read-only with respect to the source tree: the only files written are this
report and the downloads under `cache/eri/literature/forcefield_B/`. I read every quoted value from the primary PDF.
Where pdftotext dropped Greek letters, I rendered the PDF page and checked the value by eye. Downloads were treated
as untrusted data. Nothing from them was executed. Python that read them ran with `python -I`.*

## 0. Headline (read this first)

1. **PNAS 2005 (ethane in ERI) does not use the Dubbeldam 2004 united-atom force field.** It uses the
   Lennard-Jones parameter set of **Schüring, Auerbach, Fritzsche & Haberlandt, J. Chem. Phys. 116, 10890 (2002)**
   (PNAS ref. 19): O–CH3 142 K / 3.17 Å, **Si–CH3 82 K / 2.12 Å**, CH3–CH3 104 K / 3.78 Å. The bond is Dubbeldam's
   harmonic ½k1(r−r0)² with k1/kB = 96 500 K/Å² and r0 = 1.54 Å.
   The companion full paper on the same ERI/CHA ethane study (Dubbeldam, Beerdsen, Calero, Smit, J. Phys. Chem. B
   110, 3164 (2006)) gives the same numbers. It adds "rigid network of oxygen **and silicon** atoms" and "cutoff
   radius of 12 Å".
2. The Dubbeldam 2004 model (O–CH3 93 K / 3.48 Å, oxygen only, CH3–CH3 108 K / 3.76 Å, shifted and cut at 12 Å, no
   tail corrections, harmonic bond 96 500 K/Å²) is a **different model**. The same group says the Schüring set's
   "very small size parameters enhance diffusion by two orders of magnitude for ethane in LTA-type zeolite compared
   to the parameters of Dubbeldam et al." (JCP 122, 224712 (2005)). The Dubbeldam set also **removes** the
   Schüring-type entropic anomaly (diffusivity falling with temperature) in LTA (same paper, Sec. III.A.3).
3. The repo LTA engine (`src/lta/core_lta.py`) matches **Dubbeldam 2004** for O–CH3 ε and σ and for "O only". It
   deviates in the cutoff (10 Å, not 12 Å) and in the bond stiffness (400 kJ/mol/Å² is 0.4985× the literature
   802.35 kJ/mol/Å²). It is mislabelled "TraPPE-zeolite family" in both the docstring and `lta_prereg.json`.
   **If that force field is carried into ERI, the result is not the PNAS 2005 model.**
4. I could not obtain the primary Schüring et al. 2002 paper (closed access, no repository copy found). Its
   parameters are known here only through three restatements by the Dubbeldam group (§2).

## 1. Sources obtained (all under `cache/eri/literature/forcefield_B/`)

| dir / file | what it is | provenance | sha256 (first 16) |
|---|---|---|---|
| `pnas2005_epfl/12317.full.pdf` | **PNAS 102, 12317 (2005), publisher PDF**, 4 pp | EPFL Infoscience bitstream a093484c…, item b66fa0f1… | ff1d2e84eae6047a |
| `pnas2005_pmc/pmc1194924.html`, `M1.gif` | PMC HTML of the same article; M1.gif is the inline bond formula | pmc.ncbi.nlm.nih.gov/articles/PMC1194924/ | 39e3bc69e3e14ca5 / e6ae768f95cf0301 |
| `jpcb2004_epfl/0689513732dub041.pdf` | **J. Phys. Chem. B 108, 12301 (2004), publisher PDF**, 13 pp (complete, 12301–12313) | EPFL Infoscience bitstream 3ed7ee91… | b2a3a669d3addc5a |
| `jpcb2004_uu/2003-007.pdf` | Utrecht repository deposit labelled as JPCB 2004; **the PDF inside is a different paper** (van Driel et al., Surf. Sci. 529, 197 (2003), GaP electroluminescence). Not used. | dspace.library.uu.nl handle 1874/16710 | 3937d6c42769f5b8 |
| `prl2004_uu/dubbeldam_04_force.pdf` | **PRL 93, 088302 (2004), publisher version** | Utrecht DSpace handle 1874/15925 | 9540c5f9f9bc087b |
| `jcp2005_uu/1083.pdf` | **J. Chem. Phys. 122, 224712 (2005), publisher PDF** (PNAS ref. 25, "More details can be found in ref. 25") | Utrecht DSpace handle 1874/14776 | b164e7d6aa5e7b8a |
| `jpcb2006_epfl/3248743680dub061.pdf` | **J. Phys. Chem. B 110, 3164 (2006), publisher PDF**: dcTST + MD of ethane in ERI and CHA (cites PNAS as ref. 32) | EPFL Infoscience bitstream 4acc1ba2… | dcf2e294110ed628 |
| `raspa2_files/*` | RASPA2 HEAD (commit 6498ab1, 2024-07-13) `forcefield/ExampleZeolitesForceField/*`, `molecules/ExampleDefinitions/{ethane,methane,propane}.def` | raw.githubusercontent.com/iRASPA/RASPA2/master | see §6 |
| `raspa2_clone/RASPA2`, `raspa2_clone/RASPA2_hist` | shallow clone of HEAD (source used for the HARMONIC_BOND convention) and a blobless history clone | github.com/iRASPA/RASPA2 | n/a |
| `raspa2_hist_files_fd2398c/*` | files extracted from the **RASPA-2.0 initial release** commit fd2398c (2015-02-03): `GarciaPerez2006`, `GenericZeolites`, `TraPPE`, `CastilloVlugtCalero2009` force fields; `molecules/TraPPE{,-UA}/ethane.def` | git show fd2398c:… | 2c0c14fd40882ccd (GarciaPerez2006 force_field.def) |
| `raspa3_clone/RASPA3` | shallow clone of RASPA3 HEAD (564b893, 2026-10-06) | github.com/iRASPA/RASPA3 | n/a |
| `trappe_site/parameters_ethane_id2.json`, `publications.json` | TraPPE database record for ethane (molecule_id 2), via the site's own `scripts/parameters.php` endpoint | trappe.oit.umn.edu | e0d2f8ce8fb0a3c3 |
| `openalex/*.json`, `schuring2002_meta/*.json`, `uu_dspace_landing/*`, `epfl_api/*` | metadata used to find open copies (OpenAlex, Crossref, Semantic Scholar, DSpace REST) | APIs | n/a |

Not obtained: Schüring et al., JCP 116, 10890 (2002) (OpenAlex `oa_status: closed`; Semantic Scholar
`openAccessPdf.status: CLOSED`; I found no repository copy); Martin & Siepmann, JPCB 102, 2569 (1998) (ACS
paywall; I used the TraPPE database record from the authors' group instead); Bai, Tsapatsis & Siepmann, JPCC 117,
24375 (2013) (TraPPE-zeo; only cited here through RASPA files). PMC also refused its own PDF (reCAPTCHA page,
kept as `pnas2005_pmc/pnas-0503908102.pdf`, which is HTML). Europe PMC refused too (Cloudflare and HTTP 500).

## 2. What PNAS 2005 actually states (publisher PDF, p. 12317, right column, "Model and Computational Details")

Verbatim (checked on the rendered page):

> "In our simulations, we neglect cations and study rigid, all-silica versions of the ERI- and CHA-type zeolites.
> … The positions of the atoms are taken from refs. 14 and 15, respectively. Following the work of Bezus et al. (16),
> the zeolites are modeled as a rigid network of oxygen atoms. This is a very common approximation because the large
> oxygen atoms essentially shield the much smaller silicon atoms, and lattice flexibility is not important for small
> alkanes in all-silica zeolites (17). The rectangular simulation box sizes we used are 4.5906 × 3.9756 × 4.443 nm for
> ERI-type zeolite … Periodic boundary conditions were used. … For these reasons the united atom model (18) is a
> straightforward choice. We consider the CH3 groups as single, chargeless interaction centers with their own
> effective potentials. The beads of ethane are connected by a harmonic bonding potential U^bond = ½k1(r − r0)², with
> k1/kB = 96,500 K/Å² and r0 = 1.54 Å. The non-intramolecular energy is described with a Lennard–Jones potential by
> using parameters σO−CH3 = 3.17 Å, εO−CH3/kB = 142 K, σSi−CH3 = 2.12 Å, εSi−CH3/kB = 82 K, and σCH3−CH3 = 3.78 Å,
> εCH3−CH3/kB = 104 K, which were taken from ref. 19."

* Ref. 19 (PNAS reference list): "Schüring, A., Auerbach, S. M., Fritzsche, S. & Haberlandt, R. (2002) J. Chem.
  Phys 116, 10890–10894." Ref. 22 is Dubbeldam et al. JPCB 108, 12301 (2004). PNAS cites ref. 22 only among works
  where "the accuracy of the simulation techniques have been verified (6, 20–23)", **not** as the parameter source.
* Same page: "We used the velocity Verlet integration scheme with a time step of 0.5 fs." and "More details can be
  found in ref. 25" (= JCP 122, 224712 (2005)).
* No cutoff, shift or tail-correction statement appears anywhere in PNAS.
* Fig. 3 caption: "Free energy profiles βF(q) at 600 K of ethane in ERI at various loadings (infinite dilution, 1,
  2, 3, 4, 5, 6, and 7 molecules per ERI-type cage)".
* **Internal inconsistency:** the text says the framework is "a rigid network of oxygen atoms" with Si shielded,
  yet it lists Si–CH3 LJ parameters.
* **PMC HTML transcription errors** (do not copy from PMC): it renders "σSi–CH = 2.12 Å" and "εCH3–CH33/kB = 104
  which K, were taken from ref. 19". The publisher PDF reads correctly (as quoted above).

### 2a. Companion paper on the same system: JPCB 110, 3164 (2006), p. 3167, Sec. II.D "Force Field Potentials and Simulation Details" (checked on the rendered page)

> "We neglect cations and study rigid, all-silica versions of the ERI- and CHA-type zeolites. … Following the work
> of Bezus et al.,44 the zeolites are modeled as a rigid network of oxygen **and silicon** atoms. … The simulation box
> sizes we used are 3 × 3 × 3 (perpendicular widths are 3.448 × 3.448 × 4.515 nm) for ERI-type zeolite … The beads
> of ethane are connected by an harmonic bonding potential U^bond = 1/2 k1(r − r0)² with k1/kB = 96500 K/Å² and
> r0 = 1.54 Å. The extramolecular energy U^ext consists of a guest−guest intermolecular energy U^gg, a host−guest
> interaction U^hg, modeled with a Lennard-Jones potential with a **cutoff radius of 12 Å**. The parameters
> σO−CH3 = 3.17 Å, εO−CH3/kB = 142 K, σSi−CH3 = 2.12 Å, εSi−CH3/kB = 82 K, and σCH3−CH3 = 3.78 Å,
> εCH3−CH3/kB = 104 K were taken from ref 48. Although the size parameters are rather small, for this study we prefer
> to use these parameters because then diffusion of ethane in ERI- and CHA-type zeolite is still feasible using
> conventional MD."

Ref. 48 = "Schüring, A.; Auerbach, S. M.; Fritzsche, S.; Haberlandt, R. J. Chem. Phys. 2002, 116, 10890-10894." The
same paper (p. 3166) also says: "We can choose q as the position of one of the two beads of ethane.6,7" (the dcTST
reaction coordinate; relevant to CV design, outside this extraction).

### 2b. JCP 122, 224712 (2005) ("ref. 25" of PNAS), Sec. II.A p. 224712-2 and Tables I and II p. 224712-3 (Table II checked on the rendered page)

* Sec. II.A: "The beads in the chain are connected by harmonic bonding potentials. … **The Lennard-Jones potentials
  are shifted and cut at 12 Å. Analytical tail corrections are not applicable in zeolites.35 A truncated and shifted
  potential is equally suitable to Monte Carlo and molecular dynamics.** Flexibility of the framework is not an issue
  for adsorption of linear and branched alkanes.36 … The interactions between the rigid framework and the guest
  molecules are assumed to be dominated by the oxygen atoms.38"
* "TABLE II. Force field LJ parameters used by Schüring et al. for ethane in cation-free LTA-type zeolite (Ref. 24).
  Lennard-Jones parameters, ε/kB [K] in the top of each field, σ [Å] in the bottom of each field, bond and bend
  parameters." Row CH3: O **142 / 3.17**; Si **82 / 2.12**; CH3 **104 / 3.78**; "Bond U^bond = ½k1(r−r0)²,
  k1/kB = 96 500 K/Å², r0 = 1.54 Å".
* "TABLE I. Force field of Dubbeldam et al. (Refs. 39 and 40) …". Row CH3: O 93.00 / 3.48; **Si "—"** (no
  interaction); CH4 130.84 / 3.74; CH3 108.00 / 3.76; CH2 77.7 / 3.86 (printed "77.7"; JPCB 2004 Table 3 has 77.77).
* p. 224712-3: "The parameters listed in Table II are used mainly for comparison with the simulations of Schüring et
  al.,24,43 and because their very small size parameters enhance diffusion by two orders of magnitude for ethane in
  LTA-type zeolite compared to the parameters of Dubbeldam et al. Although the parameters used by Schüring et al. are
  probably less realistic in our opinion, they are convenient to compare the diffusion of ethane from a
  simulation-method point of view by MD and dcTST in LTA-type zeolite."
* Sec. III.A.3 "Ethane in LTA-type zeolite", p. 224712-13: "Figure 11 shows that the behavior found by Schüring et al.24 is strongly
  dependent on the parameter set (Tables I and II). The parameter set of Dubbeldam et al.39 does not show a decrease
  with increasing temperature although for both parameter sets the local activation energy depends on temperature.
  The size parameter used by Schüring et al. is so small that ethane at low temperature is found in the windows
  itself, and heating shifts the adsorption sites to just in front of the windows. In contrast, the set of Dubbeldam
  et al. has a larger size parameter for ethane and also at low temperature the adsorption sites are always in front
  of the window, reducing the behavior found by Schüring et al."
* Ref. 43 = Schüring et al., "Random Walk Treatment of Dumb-Bell Molecules in an LTA Zeolite and in Chabazite", Proc.
  14th Int. Zeolite Conf. (a second Schüring source for the set; not obtained).

## 3. Dubbeldam 2004 model (JPCB 108, 12301; PRL 93, 088302)

* JPCB 2004 Sec. II.A, p. 12302: "We consider the CHx groups as single, chargeless interaction centers with
  their own effective potentials. The beads in the chain are connected by harmonic bonding potentials. … The beads
  in a chain separated by more than three bonds interact with each other through a Lennard-Jones potential. **The
  Lennard-Jones potentials are shifted and cut at 12 Å. Analytical tail-corrections do not apply in zeolites.5** A
  truncated and shifted potential is equally suitable to Monte Carlo and molecular dynamics. Flexibility of the
  framework is not an issue for adsorption of linear and branched alkanes.11 **The interactions between the rigid
  framework and the guest molecules are assumed to be dominated by the oxygen atoms.12**" (ref. 12 = Bezus et al.,
  J. Chem. Soc. Faraday Trans. 2 74, 367 (1978)).
* JPCB 2004 Table 3, p. 12305 (checked on the rendered page): "Force Field Guest-Host and Guest-Guest Interactions of
  Hydrocarbons in Charge Neutral Nanoporous Materials". Row CH3: **O 93.00 / 3.48**; CH4 130.84 / 3.74;
  **CH3 108.00 / 3.76**; CH2 77.77 / 3.86; CH 42.85 / 4.19; C 9.30 / 4.90. "bond U^bond = 1/2 k1(r − r0)²;
  k1/kB = 96500 K/Å², r0 = 1.54 Å". Footnote a: "Lennard-Jones parameters, ε/kB [K] in top, σ [Å] in bottom of each
  field … Some of the alkane-alkane interactions are taken from ref 66 and optimized to reproduce vapor-liquid
  coexistence curves of the phase diagrams, the internal bond from ref 67, the internal bend from ref 41, and the
  torsion from T. J. H. Vlugt and M. Frash.68" (ref. 66 = Martin, Thompson, Nenoff, JCP 114, 7174 (2001); ref. 67 =
  Nath, Escobedo, de Pablo, JCP 108, 9905 (1998); ref. 41 = Martin & Siepmann, JPCB 103, 4508 (1999)).
* Appendix c, p. 12311, eqs. 10 and 14: U_bond = Σ ½k1(r − r0)², "where k1/kB = 96500 K/Å² is the bond energy constant,
  r0 = 1.54 Å the reference bond length". U_ij^{gg,hg,intra} = Σ 4ε_ij[(σ_ij/r_ij)^12 − (σ_ij/r_ij)^6] − E_cut,
  "where rij is the distance between site i and site j, **rcut = 12.0 Å**, the cutoff radius, **Ecut the energy at the
  cutoff radius**, and U_ij = 0 when rij > rcut." So the potential is energy-shifted with truncated forces. It is not
  force-shifted.
* Mixing: guest–host O–CHx pairs are **explicit fitted pairs** (O has no self-parameter). Guest–guest cross terms are
  tabulated explicitly. They agree numerically with Lorentz–Berthelot (√(108×158.5) = 130.84; (3.76+3.72)/2 = 3.74;
  √(108×56) = 77.77). For ethane only CH3–CH3 = 108.00 K / 3.76 Å matters.
* Ethane-specific CH3: **there is none.** Table 3 has a single CH3 type. The CH3 guest–host pair was itself fitted
  on ethane: "The O-CH3 and σO-CH3 are uniquely obtainable from the ethane isotherm" (p. 12304). "The O-CH3 and σO-CH3
  parameters are uniquely fixed with a precision better than 0.01 Å for σ and better than 1 K for ε/kB" (p. 12305;
  the ε symbol is lost in the text layer). The PRL quotes a looser overall bound: "the errors in the size and
  strength-parameters are limited to no more than 0.01 Å and 5 K, respectively."
* Bond-model sensitivity (Sec. G, p. 12306): "We modified the harmonic bond potential to a fixed distance of 1.54 Å and
  recomputed the isotherm of ethane. … The results … suggest a minimal dependency of the adsorption results on the
  internal interaction parameters". (This is about adsorption isotherms, not about free-energy barriers.)
* Si (Discussion, p. 12310): "In most nanoporous framework structures, the large oxygen atoms shield the much smaller
  silicon, aluminum, and phosphorus atoms. Therefore, the model only needs to consider interactions between the
  adsorbate and the oxygen atoms, provided there is no net negative electrical charge on the framework.12"
* PRL 2004 p. 088302-2 gives the same Table I values and the same convention: "The Lennard-Jones potentials are
  shifted and cut at 12 Å. The interactions between the rigid framework and the guest molecules are assumed to be
  dominated by the oxygen atoms [7]." Also: "The parameters may be combined with any reasonable internal interaction
  model."

## 4. Parameter table

Model P is the PNAS 2005 ERI model (the Schüring set as used by Dubbeldam et al.). Model D is Dubbeldam 2004.
Model T is TraPPE-UA. Units: K, Å. The kJ/mol values use kB = 0.008314462618 kJ/mol/K.

| parameter | literature value | units | exact source + quote/location | confidence |
|---|---|---|---|---|
| **P** ε(O–CH3)/kB | 142 (1.18065 kJ/mol) | K | PNAS p.12317 "εO−CH3/kB = 142 K"; JPCB 2006 p.3167; JCP 2005 Table II | high (3 Dubbeldam-group sources agree; primary Schüring unseen) |
| **P** σ(O–CH3) | 3.17 | Å | PNAS p.12317 "σO−CH3 = 3.17 Å"; JPCB 2006; JCP 2005 Table II | high (same caveat) |
| **P** ε(Si–CH3)/kB | 82 (0.68179 kJ/mol) | K | PNAS p.12317 "εSi−CH3/kB = 82 K"; JPCB 2006; JCP 2005 Table II | high for the value |
| **P** σ(Si–CH3) | 2.12 | Å | PNAS p.12317 "σSi−CH3 = 2.12 Å" (PMC HTML misprints "Si–CH") | high for the value |
| **P** Si actually interacting? | yes (O and Si network) | n/a | JPCB 2006 p.3167 "rigid network of oxygen and silicon atoms"; PNAS text says "rigid network of oxygen atoms" yet lists Si params | **medium** (PNAS self-contradictory) |
| **P** ε(CH3–CH3)/kB | 104 (0.86470 kJ/mol) | K | PNAS p.12317 "εCH3−CH3/kB = 104 K"; JPCB 2006; JCP 2005 Table II | high; irrelevant at infinite dilution |
| **P** σ(CH3–CH3) | 3.78 | Å | same | high |
| **P** mixing | none needed (all three pairs explicit) | n/a | PNAS/JCP 2005 Table II list explicit pairs | high |
| **P** cutoff | 12 | Å | JPCB 2006 p.3167 "Lennard-Jones potential with a cutoff radius of 12 Å"; PNAS silent | high (companion paper, same study) |
| **P** truncation convention | shifted (energy-shifted, truncated forces), no tail corrections | n/a | JCP 2005 Sec. II.A "shifted and cut at 12 Å. Analytical tail corrections are not applicable in zeolites"; not restated in PNAS/JPCB 2006 | **medium** (inferred from same group, same code, same period) |
| **P** bond | U = ½k1(r−r0)², k1/kB = 96 500 K/Å² (802.35 kJ/mol/Å²), r0 = 1.54 Å | K/Å², Å | PNAS p.12317 (formula image M1 = "U^bond = ½ k1 (r − r0)²"); JPCB 2006; JCP 2005 Table II | high |
| **P** charges | none ("single, chargeless interaction centers") | n/a | PNAS p.12317 | high |
| **P** framework | rigid, all-silica, cations neglected | n/a | PNAS p.12317 | high |
| **P** intramolecular LJ | none (2-bead molecule) | n/a | "non-intramolecular energy is described with a Lennard–Jones potential" | high |
| **D** ε(O–CH3)/kB | 93.00 (0.77325 kJ/mol) | K | JPCB 2004 Table 3 p.12305; PRL 2004 Table I; JCP 2005 Table I | high |
| **D** σ(O–CH3) | 3.48 | Å | same | high |
| **D** Si | no interaction (O only) | n/a | JPCB 2004 p.12302 "dominated by the oxygen atoms"; JCP 2005 Table I Si column "—" | high |
| **D** ε(CH3–CH3)/kB | 108.00 (0.89796 kJ/mol) | K | JPCB 2004 Table 3 | high |
| **D** σ(CH3–CH3) | 3.76 | Å | JPCB 2004 Table 3 | high |
| **D** guest–guest mixing | explicit table, numerically Lorentz–Berthelot | n/a | Table 3 (checked arithmetic) | high |
| **D** ethane-specific CH3 | none (single CH3 type; fitted on ethane in MFI) | n/a | Table 3; p.12304 | high |
| **D** cutoff / shift / tail | 12.0 Å; energy-shifted ("− Ecut"); no tail corrections | Å | JPCB 2004 p.12302 and eq. 14 p.12311; PRL 2004 p.088302-2 | high |
| **D** bond | ½k1(r−r0)², k1/kB = 96 500 K/Å², r0 = 1.54 Å (from Nath et al. 1998) | K/Å², Å | Table 3 + footnote; eq. 10 | high |
| **T** ε(CH3)/kB, σ(CH3) | 98, 3.75 (atomtype "[CH3]-CHx", q = 0) | K, Å | TraPPE database record molecule_id 2, ref_ID 33 = Martin & Siepmann JPCB 102, 2569 (1998) | high (database), paper itself not opened |
| **T** bond | fixed 1.54 | Å | TraPPE record "stretchtype CHx-CHy, distance 1.54" | high |
| **T** cutoff | 14 Å spherical cutoff + analytic LJ tail corrections; LB for unlike pairs | Å | TraPPE record message: "Simulations using this force field should be performed with a spherical cutoff of 14 Å and analytic tail corrections for the Lennard-Jones interactions" | high |

## 5. Comparison with the repo LTA engine (`src/lta/core_lta.py`, `configs/uniform_campaign/lta_prereg.json`)

The engine's `LTAParams` are `eps_go = 93.0*KB` (kJ/mol), `sigma_go = 3.48`, `rc = 10.0`, `r0_bond = 1.54`,
`k_bond = 400.0` "(0.5 k (r-r0)^2)", oxygen only ("Only the 384 framework O atoms interact with the guest"), energy
shift `v_rc` subtracted, forces plain truncated LJ, and one ethane per replica.

| item | LTA engine | Dubbeldam 2004 (D) | PNAS ERI (P) | verdict |
|---|---|---|---|---|
| ε(O–CH3)/kB | 93.0 K | 93.00 K | 142 K | = D; ≠ P |
| σ(O–CH3) | 3.48 Å | 3.48 Å | 3.17 Å | = D; ≠ P |
| Si–CH3 | absent | absent | 82 K / 2.12 Å | = D; ≠ P |
| cutoff | **10 Å** | 12 Å | 12 Å | **deviates from both.** It is forced by the LTA box: `assert rc < L/2` with L = 2a = 23.838 Å gives L/2 = 11.919 Å < 12 Å. Effect on F(z) not quantified. |
| truncation | energy-shifted, truncated force, no tail correction | same ("− Ecut") | same (medium conf.) | form matches |
| bond | harmonic, ½k(r−r0)², **k = 400 kJ/mol/Å²**, r0 = 1.54 Å | k1 = 96 500 K/Å² = **802.35 kJ/mol/Å²** | same as D | **k is 0.4985× the literature value.** rms bond fluctuation √(kBT/k) at 300 K: 0.079 Å (engine) vs 0.056 Å (literature) |
| CH3–CH3 LJ | not implemented | 108 K / 3.76 Å | 104 K / 3.78 Å | irrelevant at one molecule per replica; needed for the stage-8 loadings |
| charges | none | none | none | = |
| labels | docstring: "(TraPPE-zeolite family, Dubbeldam et al.)", "one TraPPE-UA ethane"; prereg `"family": "TraPPE-zeolite (Dubbeldam et al.)"` | n/a | n/a | **mislabel.** TraPPE-zeo is Bai, Tsapatsis & Siepmann 2013 (O 53.0 K / 3.30 Å, Si 22.0 K / 2.30 Å with LB mixing, as cited in RASPA2/RASPA3 files). The bead is not TraPPE-UA either: TraPPE uses a fixed bond, 98 K / 3.75 Å, and a 14 Å cutoff with tail corrections. |

## 6. RASPA2 / RASPA3 force-field files (secondary; Dubbeldam is the RASPA author)

* **No `Dubbeldam2004` force-field directory exists** anywhere in the RASPA2 git history. I listed every historical
  `forcefield/*` directory (26 of them, e.g. GarciaPerez2006, GenericZeolites, TraPPE, CastilloVlugtCalero2009,
  Pascual2004, ExampleZeolitesForceField). Current RASPA2 HEAD `forcefield/` has only 8 entries.
* **Only historical file with the explicit Dubbeldam-2004 guest–host pair:** RASPA-2.0 initial release (fd2398c,
  2015-02-03) `forcefield/GarciaPerez2006/force_field.def`: "CH3_sp3 O lennard-jones 93.00 3.48" (and the same for
  Oa; CH4_sp3 115.00/3.47, CH2_sp3 60.50/3.58, …). The matching `force_field_mixing_rules.def` reads "shifted" /
  tailcorrections "no" / "CH3_sp3 LENNARD_JONES 108.0 3.76" / "Si HARD_SPHERE 1.0" / "Lorentz-Berthelot". (How RASPA
  mixes a HARD_SPHERE type with LJ was not checked.)
* **Current RASPA2 HEAD `forcefield/ExampleZeolitesForceField/force_field_mixing_rules.def`** is a different
  guest–host model: "shifted", tailcorrections "no", "O_ lennard-jones 53.0 3.30 // P. Bai, M. Tsapatsis, J. I.
  Siepmann, J. Phys. Chem. C 2013, 117, 24375-24387.", "Si_ lennard-Jones 22.0 2.30 // idem", "CH3 lennard-jones
  108.0 3.76 // D. Dubbeldam et al., J. Phys. Chem. B, 108(33), 12301-12313", "Lorentz-Berthelot". The implied
  CH3–O pair is √(53×108) = 75.66 K, (3.30+3.76)/2 = 3.53 Å, and Si interacts. RASPA3 HEAD
  `examples/non_basic/8_mc_adsorption_c6_isomers_in_mfi/force_field.json` is the same: "MixingRule":
  "Lorentz-Berthelot", "TruncationMethod": "shifted", "TailCorrections": false, "CutOffVDW": 12.0, O [53.0, 3.30],
  Si [22.0, 2.30], CH3 [108.0, 3.76].
* Ethane definitions: RASPA2 HEAD `molecules/ExampleDefinitions/ethane.def` "0 1 HARMONIC_BOND 96500 1.54" (types
  CH3/CH3). RASPA-2.0 `molecules/TraPPE/ethane.def` is the same with CH3_sp3. `molecules/TraPPE-UA/ethane.def` has
  "0 1 FIXED_BOND 1.54". RASPA3 example `ethane.json` has `[["CH3","CH3"], "FIXED", [1.54]]`.
* RASPA HARMONIC_BOND convention (RASPA2 `src/internal_energy.c` lines 390–395): "// 0.5*p0*SQR(r-p1); … // p_0/k_B
  [K/A^2] force constant // p_1 [A] reference bond distance" and "U=0.5*parms[0]*SQR(r-parms[1]);". This matches the
  ½k1 convention of the papers.
* The Schüring set (142/3.17, 82/2.12) appears in no RASPA2 or RASPA3 HEAD `.def`/`.json` file (grep), and in none of
  the historical directories I extracted.

## 7. Unresolved items

1. **Schüring et al. 2002 primary not read.** It is closed access, and Crossref returns only the abstract ("The
   self-diffusion of ethane in cation-free Linde type A zeolite has been studied by molecular dynamics simulations
   for various temperatures…"). So these remain unverified: the original cutoff and shift convention, whether
   Schüring used a rigid or a harmonic bond (JCP 2005 Table II pairs their LJ with Dubbeldam's harmonic bond), and the
   origin of the set. Schüring's reference list includes Fritzsche/Haberlandt-group LTA papers (Chem. Phys. 174, 229
   (1993); CPL 198, 283 (1992); JPC 97, 13778 (1993); CPL 223, 363 (1994)); none was obtained.
2. **Si in the PNAS runs.** JPCB 2006 says yes; the PNAS text says O-only but lists Si parameters. Medium confidence
   that Si–CH3 82 K / 2.12 Å was active.
3. **Shift in the PNAS/JPCB 2006 runs** is not stated in either paper; it is inferred from the JCP 2005 methods
   (medium confidence). Tail corrections: none (same inference).
4. **Exact vs surrogate decision.** The "PNAS model" means Schüring-set LJ (with Si) + 96 500 K/Å² bond + 12 Å cutoff.
   Reusing the LTA engine force field (Dubbeldam 2004 values, 10 Å cutoff, k = 400 kJ/mol/Å²) in ERI would be a
   surrogate. The literature gives no ERI free-energy barriers for the Dubbeldam set, but per JCP 2005 the barriers in
   LTA are much higher (diffusion about 100× slower).
5. **Cutoff vs box.** Both models need rc = 12 Å, so every perpendicular box width must exceed 24 Å. JPCB 2006
   states: "The smallest perpendicular width of the unit cell has to be larger than twice the spherical cutoff in R
   space."
6. **Incidental structure flag (outside this extraction, for the structure review):** the PNAS ERI box
   4.5906 × 3.9756 × 4.443 nm implies a = 13.252 Å (3a = 39.756; 2√3·a = 45.906) and c = 14.81 Å (3c = 44.43). PNAS
   itself states a = 1.327 nm and c = 1.505 nm. JPCB 2006's widths 3.448 × 3.448 × 4.515 nm match a = 13.27 Å and
   c = 15.05 Å (3a·sin60° = 34.48, 3c = 45.15). The discrepancy in the PNAS box is unexplained.
7. TraPPE-zeo values (53 K/3.30 Å, 22 K/2.30 Å) are cited here only through RASPA files; Bai et al. 2013 was not
   opened.
8. The existing LTA "entropic barrier" premise is parameter-dependent in the literature: JCP 2005 reports that the
   Dubbeldam set, which the repo LTA engine uses, does not reproduce the Schüring entropic anomaly. This is relevant
   context, not a defect in this extraction.
