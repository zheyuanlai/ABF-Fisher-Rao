# Candidates B–E: literature and feasibility scan

*Read-only feasibility assessment for the overnight molecular extension, 2026-10-16. No implementation and no
production runs. No GPU was used. The only computation was a CPU symmetry expansion of two CIFs (< 1 s), which
reused the whitelist parser `src/eri/framework.py` from a scratch script. No repository file was modified apart from
the creation of this report.*

**Question.** Can each candidate become a two-arm (histogram ABF vs histogram ABF + uniform marginal FR) test system
on the repo's CPU numba / overdamped-Langevin machinery? The ERI dilute cell (united-atom ethane, rigid ERI, 600 K,
6.67 kBT barrier, 84 % entropic) is used as the cost unit throughout.

**Reading rules used here.**
* A parameter is reported only if it was read in a retrieved document; the location is given.
* Any inference is labelled **(inference)**.
* For copyright reasons, prose is paraphrased with exact locations rather than quoted at length, and there is one
  short verbatim quote in the whole report. Parameter values are reproduced exactly; they are facts.
* All downloads are under `cache/overnight/literature/candBE/<one directory per download>/`. They were read with
  `pdftotext` and `python -I` only, and nothing downloaded was executed.

---

## 0. Retrieval ledger

| item | obtained? | route / file | note |
|---|---|---|---|
| B: Jee & Sholl, JACS 131, 7896 (2009), main text | **no** | ACS: Cloudflare 403; Unpaywall `is_oa: false, closed` (`unpaywall_json/10.1021_ja901483e.json`); no figshare deposit; no OA thesis or preprint found | only the abstract (PubMed 19422215) and the Crossref reference list (`crossref_refs/10.1021_ja901483e.json`) |
| B: SI `ja901483e_si_001.pdf` | **no** | ACS 403 (also through the fetch proxy) | — |
| C: López, Pérez, Ruette & Medina, PRE 72, 061111 (2005) | **no** | APS: Cloudflare challenge (`C_aps_pdf_try/BLOCKED_cloudflare_challenge_page.html`); Unpaywall closed; no arXiv version | only the abstract (APS / PubMed 16485935) and the Crossref reference list (20 refs, 16 resolved: `C_crossref_ref_titles/`) |
| D: Vaitheeswaran, Reddy & Thirumalai, JCP 130, 094502 (2009) | **yes** | arXiv:0901.2891v1 (`D_arxiv_0901.2891_pdf/paper.pdf`, sha256 `fbbbd140…`); published version from the Thirumalai group site (`D_published_utexas_pdf/paper.pdf`, `250c8956…`) | the methods, Table I and Table II agree between the two versions |
| E: Cnudde et al., JACS 142, 6007 (2020), main text | **yes** | UGent Biblio OA copy (ACS AuthorChoice) `E_ugent_biblio_pdf/paper.pdf` (`f589fa5b…`) | publisher version, 11 pp. |
| E: SI `ja9b10249_si_001.pdf` | **yes** | ACS figshare deposit 12009042 (`E_si_figshare/si.pdf`; md5 `2ff65ed5…` matches figshare) | 25 pp. |
| E: force-field source, Ghysels et al., JPCC 119, 23721 (2015) | **yes** | UGent accepted manuscript `E_ghysels2015_ugent_pdf/paper.pdf`; SI pdf + CIF zip from figshare 2121370 / 2121364 (`E_ghysels2015_si1_figshare/`, `E_ghysels2015_si2_figshare/`, md5s match) | zip **listed, not extracted** |
| E: Correction, JPCC 121, 5854 (2017) | text only | read through the fetch proxy (it is free to read) | adds the O–O LJ row that is missing from Table S3 |
| E: hydrocarbon force field, Oie et al., Int. J. Quantum Chem. 20(S8), 1 (1981), doi 10.1002/qua.560200703 | **no** | Unpaywall closed | — |
| E: Combariza, Sastre & Corma, JPCC 115, 875 (2011), doi 10.1021/jp102262n | **no** | RiuNet copy is "request a copy" only | — |
| DDR / MFI structures | **yes** | RASPA2 commit `6498ab1e` `DDR.cif` (`raspa2_DDR_cif/`, sha256 `ac2e315e…`), `DDR.block` (`raspa2_DDR_block/`) and `MFI_SI.cif` (`raspa2_MFI_SI_cif/`, `65ffb4fe…`); IZA Atlas PDFs `iza_atlas_pdf_{DDR,MFI}/` | expanded here: DDR **Si120 O240**; MFI **Si96 O192** |

No browser extension was connected, so no institutional route was available. B and C are paywalled with no open
copy that could be found.

---

## B. Jee & Sholl 2009 — CO2 (and CH4) in DDR

### (i) Physical model, as far as it can be established

* **The paper introduces a new force field.** The abstract says the simulations use a force field that, for the
  first time, is consistent with the experimental single-component adsorption and diffusion data. It also says
  earlier simulations had greatly overestimated CH4 (and, less so, CO2) transport.
  * Because CH4 diffuses extremely slowly in DDR, CH4 diffusion was obtained from a TST-based kinetic Monte Carlo
    scheme (abstract, PubMed 19422215).
  * None of the model's numbers could be read: the guest models, charges, framework charges, mixing rule, cutoff,
    Ewald settings and rigid/flexible framework are all **unknown**.
* **Reference list (Crossref).** Reading only titles, the list points to:
  * DDR structure: Gies, Z. Kristallogr. 175, 93 (1986) (ref. 5);
  * guest–zeolite models: Goodbody et al. 1991, CH4/butane in silicalite (ref. 21); Makrodimitris et al. 2001, CO2/N2
    in silicalite (ref. 22); Selassie et al. 2008, CO2/N2 in silica zeolites (ref. 37); Skoulidas, Sholl & Johnson
    2006, CO2/N2 in nanotubes (ref. 34); Dubbeldam et al. 2004, UA alkane force field (ref. 19);
  * a silica/zeolite framework potential: Kramer et al. PRB 43, 5068 (1991) and van Beest et al. PRL 64, 1955
    (1990), the "BKS" family (refs. 32–33);
  * DDR experiments: Hedin et al. 2008 PFG-NMR (ref. 31); Tomita et al. 2004 (ref. 11);
  * TST/hopping methods: Ford (ref. 24); Tunca & Ford (refs. 25–27); Dubbeldam/Beerdsen (refs. 28, 29, 36, 42).
  * **(inference)** The BKS citations suggest a BKS-type framework potential or BKS-type framework charges were used.
    Whether the framework was flexible **cannot be verified**.
* **Framework structure (verified independently of the paper).**
  * IZA Atlas (`iza_atlas_pdf_DDR/DDR.pdf` p. 2): type material deca-dodecasil 3R, [Si120O240], trigonal R-3m,
    a = 13.860 Å, c = 40.891 Å; channels listed as 8-ring 3.6 × 4.4 Å; the double asterisk marks a 2-D channel
    system.
  * RASPA2 `DDR.cif` cites Gies 1986 (the same source as Jee & Sholl ref. 5) and has the same cell. It carries
    charges Si +2.05, O −1.025; their provenance is not stated in the file.
  * Symmetry expansion here: 120 Si + 240 O per cell, V = 6802.75 Å³.

### (ii) Are all parameters recoverable?

**No.** Every model parameter is in a paywalled paper and SI that could not be obtained. The cited sources are a
dozen papers whose roles are inferred from titles only. A labelled surrogate is possible:

| surrogate component | source | status |
|---|---|---|
| CH4 | Dubbeldam 2004 UA O–CH4 115.00 K / 3.47 Å | already verified in the repo (`docs/eri_campaign/review/LIT_FORCEFIELD_A.md` l. 89) |
| CO2 | García-Sánchez et al. 2009, as transcribed in RASPA2 `ExampleZeolitesForceField` | not checked against the paper |
| framework | RASPA2 / IZA DDR | verified (see (i)) |

The RASPA2 CO2 values are C +0.6512 e, O −0.3256 e, LJ O_co2 85.671 K / 3.017 Å, C_co2 29.933 K / 2.745 Å, with
Lorentz–Berthelot mixing and shifted potentials. Such a surrogate would be "a DDR model", not Jee & Sholl.

### (iii) Evidence of an entropic barrier

Not verifiable. The abstract establishes that CH4 hopping through the 8-ring is a rare event, slow enough to need
TST/kMC. A TST rate needs a free-energy profile across the window, so the paper presumably contains one, but its
values and any energy/entropy split could not be read. There is no temperature series in the abstract.

### (iv) Reaction coordinate

**(inference)** The natural CV is the COM projection onto the normal of one 8-ring window between two large
DDR cages (2-D 8-ring net). It is linear in Cartesian coordinates for a rigid framework, so there is no
geometric term. As in `src/cha`, the window planes and cage centres would have to be located numerically. CO2 adds
nothing to the CV, which stays the COM projection.

### (v) Hidden slow coordinates

* **CO2 orientation.** The linear molecule must align with the window normal to cross. Orientational entropy enters
  the barrier, and the orientation is fast in the cage but slaved in the window. (inference)
* **Framework flexibility for CH4.** Awati, Ravikovitch & Sholl, JPCC 117, 13462 (2013, abstract) report that for
  molecules with kinetic diameters comparable to or larger than the window, the rigid-framework approximation can
  change diffusivities by orders of magnitude (LTA, CHA, ERI, BIK). CH4 in a 3.6 × 4.4 Å window falls in that class.
  A rigid model stays self-consistent for the equilibrium F(ξ), but it is physically mis-specified.
* **Inaccessible cages.** (inference) DDR has small cages behind 5- and 6-rings. They are unreachable by dynamics
  from the large cages, but initial placement has to avoid them. RASPA2 (commit `6498ab1e`) ships `DDR.block`, which
  defines 15 blocking spheres (radii 3.0–4.5 Å) per cell (`raspa2_DDR_block/`, sha256 `c6e8021b…`). It can be used
  for placement and reference sampling, after its own check.

### (vi) Cost per molecule-step (rigid framework, CPU numba)

* **CH4, 1 site.** About 255 O (+128 Si) within 12 Å, so 255–383 LJ terms: **0.4–0.6× ERI** (ERI ≈ 680 terms per
  step for 2 beads). No Ewald.
* **CO2, 3 charged sites.**
  * About 1150 LJ + Coulomb terms: **~1.7× ERI**, plus long-range electrostatics. That means either Ewald with a
    reciprocal sum per site per step, or (at infinite dilution in a rigid framework) a precomputed potential grid,
    which must pass its own accuracy gate.
  * A damped-shifted-force definition, as the ZIF-8 stage used, is also possible as a declared model definition.
  * **Integrator.** Rigid linear CO2 needs rigid-body (translational + rotational) Brownian dynamics. A collinear
    3-bond triangle is degenerate (soft bending), so stiff bonds are not a clean substitute. This is a new
    integrator; none exists in the repo.

### (vii) Already tested in the repo?

* No DDR, MFI or CO2 model exists in `src/`, `docs/`, `configs/`, `scripts/` or `tests/` (grep).
* Closest relatives:
  * `src/eri`: rigid all-silica 8-ring cages, UA ethane, PNAS-2005-type. Dilute cell ABF-sufficient at
    K 64/256/1024.
  * `src/lta`: rigid LTA ethane. 300→80 K sweep positive (−15 … −35 % ΔI_F), with benefit tracking starvation,
    not entropy share.
  * `src/cha`: rigid CHA, TraPPE olefins. NEUTRAL ×3.
* No charged guest has been run in a zeolite engine.

### (viii) Verdict: **NOT-REPRODUCIBLE** as published

* **Main risk:** the model cannot be verified at all.
* **Feasible fallback:** a *labelled* CH4-in-rigid-DDR surrogate is FEASIBLE-WITH-CAVEATS. With the C surrogate it is
  one of the two cheapest systems in this scan, and it drops into the `src/eri` machinery with only a new framework and CV. Its own risks are:
  * the barrier height and entropic share are unknown, and could be discovery-limited at accessible T;
  * the rigid-framework approximation is known to be poor for CH4-sized guests in 8-rings.
* **Cheap check before any commitment:** a < 5 min Gate-0-style single-bead profile through one window with the
  repo-verified Dubbeldam 2004 parameters. Not run here (this scan is not an implementation).

---

## C. López et al. 2005 — methane in silicalite (MFI)

### (i) Physical model

* **Unknown beyond the abstract.** The abstract (APS / PubMed) says:
  * MD at infinite dilution or small loading;
  * an Arrhenius regime above 250 K with deviations below it;
  * memory (geometric correlation) effects assessed with an earlier probabilistic model, stronger at low T;
  * a discrete, hysteretic switch between channel-centre preference (low T) and intersection preference (high T);
  * an accessible-volume-versus-energy model that reproduces the switch.
* The methane model, force field, framework structure and rigid/flexible framework are **not stated in any accessible
  text**.
* **Reference list (Crossref, 16 of 20 resolved).**
  * Methane/silicalite MD and force-field papers: June, Bell & Theodorou 1990; Nicholas et al. 1993; Demontis et al.
    1990; Goodbody et al. 1991. Their lattice treatments were not checked here.
  * Flexibility-related studies: Fritzsche et al. 2003 (degrees of freedom in methane/silicalite-1 MD); Leroy et al.
    2004 (rigid vs flexible frameworks).
  * Entropy: Schüring et al. 2002 (entropic barriers, the ERI force-field source); Krishna, Smit & Calero 2002;
    Schenk 2003.
  * Memory effects: Fritzsche & Kärger 2003; Jousse et al. 2000.
  * NMR: Caro et al. 1985; Heink et al. 1992.
  * **(inference)** One of the June / Nicholas / Demontis / Goodbody families is the likely force field. Which one,
    and whether the lattice was flexible, cannot be determined.
* **Framework (verified, generic).** IZA Atlas MFI: Pnma, 10-ring channels 5.1 × 5.5 Å [100] and 5.3 × 5.6 Å [010],
  3-D (triple asterisk). RASPA2 `MFI_SI.cif`: a 20.022, b 19.899, c 13.383 Å, expanded to Si96O192.

### (ii) Recoverable?

**No** for the paper's model. A fully open surrogate exists:
* Dubbeldam 2004 UA CH4 in rigid MFI: O–CH4 115.00 K / 3.47 Å, repo-verified (`LIT_FORCEFIELD_A.md` rows 87–89).
  That review records the force field's CH3 term as fitted on MFI isotherms (l. 186);
* Beerdsen, Dubbeldam & Smit, JPCB 110, 22754 (2006), cached at
  `cache/eri/literature/pnas2005/epfl_jpcb2006_beerdsen_methane/`, which gives rigid-framework methane/MFI
  free-energy profiles at 300 K for comparison.

### (iii) Entropic contribution

* **Paper.** The entropic effect is a temperature-driven change of the preferred site (intersection versus channel
  centre), not an entropic window barrier.
* **Open surrogate literature.** Barriers are low. Beerdsen et al. 2006 (p. 22771) state that internal barriers are
  relatively small for methane in MFI. The y-direction profile has its highest barrier at site "B" and minima at the
  intersections and zig-zag segments (pp. 22766–22767).
* **(inference)** These are expected to lie well below the ERI cell's 6.7 kBT, which was itself ABF-sufficient. No MFI
  barrier value was read.

### (iv) Reaction coordinate

COM position along the straight channel, y [010]: linear, periodic, no geometric term. The zig-zag channels map onto
the same y (Beerdsen 2006).

### (v) Hidden slow coordinates

* **x/z placement.** Straight versus zig-zag channel occupancy is folded into y.
* **Memory effects.** These are dynamical (correlated successive jumps). They affect sampling efficiency, not F(y).
* **Hysteresis.** The reported hysteresis of the site preference implies a slow mode in their MD, or a sampling
  artefact.
* **Framework flexibility**, if their model was flexible.

### (vi) Cost

* 1 LJ site × about 261 O within 12 Å: **~0.4× ERI**. No Ewald. This is the cheapest system in the scan.
* A flexible MFI lattice would raise this by at least two orders of magnitude, with framework atoms per replica.

### (vii) Already tested?

* No MFI engine exists in the repo.
* UA methane has only been run in water (`src/methane`, closed ABF-sufficient 8/8).
* The `src/eri` kernel would serve: one bead, a new framework, a periodic y CV.

### (viii) Verdict: **NOT-REPRODUCIBLE** as published

* **Main risk:** the paper's model is unverifiable, and the physics is unpromising either way. Low barriers and a
  site-preference crossover are what the repo's predictor (marginal establishment starvation) flags as
  ABF-sufficient.
* The open surrogate (CH4 / rigid MFI / Dubbeldam 2004) is FEASIBLE but expected NEUTRAL. It would need a low-T cell
  (below 250 K, where the paper reports non-Arrhenius behaviour) to have any chance of establishment starvation.

---

## D. Vaitheeswaran, Reddy & Thirumalai 2009 — methane pair in water-filled cylindrical pores

Paper retrieved; the arXiv and published versions agree on Methods and Tables I–II.

### (i) Physical model (published version, Sec. II, p. 094502-2; arXiv v1 pp. 3–4, 14)

* **Water and solutes.**
  * Water: TIP3P (ref. 13, Jorgensen et al. 1983).
  * Methane: one united-atom LJ sphere.
  * Model ions M^q±: charges of magnitude 0.4e or e added to the methane spheres (ref. 15, Wallqvist, Covell &
    Thirumalai 1998).
* **LJ parameters (Table I).** V = 4ε[(σ/r)¹² − (σ/r)⁶].

  | pair | ε (kJ/mol) | σ (nm) | source |
  |---|---|---|---|
  | methane–methane | 1.23 | 0.373 | Kalra, Hummer & Garde 2004 |
  | O–O | 0.64 | 0.315 | TIP3P |
  | methane–O | 0.887 | 0.344 | Lorentz–Berthelot (Table I footnote) |

* **Ensemble.** Metropolis MC, NVT, T = 298 K. Effective volume V_eff = V − N_s V_s with N_s = 2 and
  V_s = (4/3)π σ_MO³. Pore length from V_eff = π(D/2)²L and ρ_w = 997 kg/m³ (eqs. 1–3).
* **Pore.**
  * Walls: hard, U₁(ξ) = 10¹² kJ/mol for radial ξ > D/2 and U₂(z) = 10¹² kJ/mol for |z| > L/2 (eqs. 4–5).
  * **No periodic boundaries.**
  * Electrostatics and LJ are evaluated **without a cutoff**: direct all-pairs, no Ewald.
* **Sampling.**
  * MC moves: translations of solutes and waters, rotations of waters, plus collective translations and collective
    z → −z flips of the two methanes.
  * Umbrella sampling on r = |r₁ − r₂|, harmonic k = 5 kcal/(mol Å²), windows 0.1 nm apart, WHAM (Kumar et al.;
    Crouzy code).
  * −2kT ln r is *not* subtracted.
* **Pores (Table II).**

  | Nw | D (nm) | L (nm) |
  |---|---|---|
  | 10 | 0.8 | 1.27 |
  | 17 | 0.8 | 1.69 |
  | 26 | 0.8 | 2.23 |
  | 32 | 0.8 | 2.59 |
  | 128 | 0.8 | 8.31 |
  | 128 | 1.2 | 3.69 |
  | 128 | 1.6 | 2.08 |
  | 128 | 2.0 | 1.33 |

### (ii) Recoverable?

**Yes, with three declared ambiguities.** TIP3P geometry and charges come from the standard Jorgensen 1983 model
(ref. 13). Note that this is not the CHARMM-modified TIP3P (H carries LJ) that `src/nacl` uses.
1. **The wall acts on unspecified sites.** The paper does not say whether U₁, U₂ act on the water O, every atom or the
   molecular centre, or whether they act on the solute centres.
2. **The walls are hard (10¹² kJ/mol steps).** They are non-differentiable, so force-based ABF/BD needs a smooth
   surrogate wall, which is a declared deviation.
3. **No run lengths or MC acceptance data are given.** This does not affect the equilibrium target.

### (iii) Entropic contribution

* **Analytic entropy term.** The paper attributes the broad curvature of the PMF to solute translational entropy at
  the pore surface. It computes a purely geometric −TΔS_A(r) = −kT ln l_c(r) (Appendix p. 094502-7, eqs. 6–16; black curves in
  Fig. 5, discussed on p. 094502-5) that qualitatively tracks the simulated PMF for the wider pores.
* **The barrier itself is solvent-driven.** For D = 0.8 nm, Nw = 10–32 (Fig. 4, p. 094502-3):
  * contact minimum at r ≈ 0.4 nm;
  * distant minimum at r ≈ L, about 3 kJ/mol (1.2 kT) above contact;
  * the barrier rises from about 3.2 to about 8 kJ/mol (1.3 → 3.2 kT) as L grows. The paper ascribes this to the cost
    of disrupting the more ordered water column.
* For Nw = 128, D = 1.2–2.0 nm (Fig. 5): a distant minimum at r ≈ max{D, L} lies 0.5–0.7 kJ/mol *below* contact,
  and a second one sits near min{D, L}.
* **No temperature series and no energy/entropy decomposition of the simulated PMF.**

### (iv) Reaction coordinate

* The CV is the solute–solute distance r. It is not a linear projection.
* With both solutes mobile under unit mobility, the ABF instantaneous force needs the divergence term (2kT/r in 3-D).
* The paper reports −kT ln P(r) without removing the r² Jacobian. That is exactly the marginal free energy that ABF
  reconstructs, so the target is consistent.

### (v) Hidden slow coordinates

* **Water-column structure.**
  * Concentric layering for D ≤ 1.2 nm (Fig. 1).
  * Dipoles aligned up or down along the pore axis, more strongly in long narrow pores, with zero net dipole (Fig. 2).
  * Hydrogen-bond rearrangement as the solutes separate.
* **Solute radial position.** Methanes sit at the pore periphery (Fig. 3).
* **Symmetric end-cap states.** The authors needed collective z-flips of the methanes to sample, which indicates slow
  exchange between symmetric end-cap configurations at fixed r.
* **(inference)** The barrier lives in the solvent (the conditional of the orthogonal coordinates). That is the
  conditional-equilibration regime where marginal FR provably cannot act (the repo's d/dt p(y|ξ)|_FR = 0 result).

### (vi) Cost per configuration-step (non-periodic, all-pairs, no Ewald)

| Nw | site-pair terms per step | × ERI |
|---|---|---|
| 10 | ≈ 470 | 0.7 |
| 26 | ≈ 3,300 | 4.8 |
| 32 | ≈ 5,000 | 7.4 |
| 128 | ≈ 81,500 | 120 |

Each water–water pair costs 9 Coulomb terms + 1 O–O LJ. Ions add 6 Coulomb terms per water.
* **Integrator.** Rigid TIP3P needs rigid-body Brownian dynamics (rotations) or a declared flexible-water surrogate.
  Neither exists in the CPU engines (the water studies ran in torch/OpenMM).
* **Feasible cells.** The small D = 0.8 nm pores are cheap. The Nw = 128 pores cost about 100× ERI.

### (vii) Already tested?

Explicit-water solute-pair PMFs have been run twice, both ABF-sufficient:
* methane pair in SPC/E (`src/methane`; 0.67 kT barrier, 8/8 seeds, preregistered null);
* NaCl in CHARMM TIP3P (`src/nacl`);
* C60/TIP4P-Ew was suspended for complexity.

No confined-water or non-periodic engine exists.

### (viii) Verdict: **FEASIBLE-WITH-CAVEATS**

The parameters are recoverable and the small pores are cheap.
* **Main risk:** the regime. The barriers are ≤ 3.2 kT and live in the water structure, which is
  conditional-equilibration-limited. It is the same family that closed ABF-sufficient twice, so the prior is
  ABF-sufficient / NEUTRAL.
* **Secondary risks:**
  * the smooth-wall and wall-site ambiguity changes the model;
  * a new rigid-water BD integrator plus its validation gate (MC reference) is required.

---

## E. Cnudde et al. 2020 — ethene/propene through the 8-ring of flexible H-SAPO-34

Paper and SI retrieved, plus the force-field source paper (Ghysels 2015) with its SI and the 2017 correction.

### (i) Physical model

* **Two levels of theory** (main text Sec. 2.2–2.3, p. 6009–6010; SI S3, p. S6–S7).
  * *FF-MTD*: classical, flexible, all-atom, 2×2×2 supercell.
  * *AI-US*: revPBE-D3 CP2K AIMD, 1×2×1 supercell, 49 windows ξ ∈ [−6, 6] Å, k = 25 kJ/mol/Å², at least 5 ps per
    window (SI S3.5), WHAM. The unbiased AIMD runs were 100 ps after 10 ps equilibration.
  * The paper itself says the DFT barriers are 5–10 kJ/mol higher than the FF ones (p. 6011; Table S1). The loading,
    acid-site and hydrocarbon-pool results rest on AI-US only.
* **Framework** (p. 6009).
  * CHA topology: cages 11 × 6.7 Å, 8-rings 3.8 × 3.8 Å; unit cell (AlPO4)18 with 36 T sites.
  * Brønsted sites are made by P→Si substitution plus a proton, about 2 BAS per cage.
  * Unit cells come from 20 ps AIMD NpT at 300/450/600 K and 1 bar (Table S2: e.g. 1 BAS, 450 K: a 14.07, b 14.10,
    c 14.98 Å, α 90.09, β 90.08, γ 119.95°, V 2574.68 Å³).
  * Ring types 0/1/2 carry 0/1/2 BAS on the crossed ring. For type 1 the authors chose the acid-proton position arbitrarily (p. 6009).
* **Framework force field.** Cnudde SI S3.2 says only that the FFs were fitted to structural data of many zeotypes,
  with thermal expansion via ZSISA in GULP (refs. 12–14 = Gale 1998, **Ghysels et al. 2015**, GULP). The parameters
  are in Ghysels 2015 SI Table S3 (pp. 10–11), plus the O–O row restored by the 2017 correction:
  * **Charges (e):** Si 2.100, Al 1.575, P 2.625, O1 (Si–O(H)–Al) −0.725, O2 (all other O) −1.050, H1 0.200.
  * **E_vdW = A/r¹² − B/r⁶,** with (A in eV Å¹², B in eV Å⁶):

    | pair | A | B |
    |---|---|---|
    | Si–Si | 0.5601 | 0.0004 |
    | Al–Si | 1.0153 | 0.0005 |
    | Al–Al | 1.8405 | 0.0006 |
    | P–Si | 0.5403 | 0.0003 |
    | P–Al | 0.9794 | 0.0004 |
    | P–P | 0.5211 | 0.0003 |
    | Si–O1 | 218.1689 | 0.9583 |
    | Si–O2 | 172.6992 | 0.1086 |
    | Al–O1 | 259.9127 | 0.0786 |
    | Al–O2 | 342.4165 | 0.0786 |
    | P–O1, P–O2 | 118.3529 | 0.0930 |
    | O–O, all O types (2017 correction) | 26877.9664 | 29.8306 |

  * **Three-body harmonic** (k in eV Å⁻², θ₀): O–Si–O 1.4944 / 109.47°; Si–O–Si 1.5509 / 142.71°; Si–O–Al
    1.7875 / 140.83°; P–O–Al 2.0468 / 139.69°. There is no O–Al–O term (Ghysels 2015 accepted manuscript p. 10).
  * **Coulomb-subtracted Morse O1–H1:** D_e 7.052 eV, α 2.1986, r₀ 0.9476 Å.
  * There are no bonds. This is a rigid-ion nonbonded + three-body model.
* **Guest force field.**
  * Ethene and propene take the all-atom FF of Oie et al. 1981 (Ghysels 2015 accepted manuscript p. 11, ref. 63), previously used by
    Combariza et al. 2011 (ref. 64).
  * Host–guest coupling is by C–O and H–O LJ terms (Ghysels SI Fig. S2 caption). **Their values appear nowhere in
    the retrieved texts.**
* **MD settings** (Cnudde SI S3.2, S3.4).
  * DL_POLY Classic 1.9, NVT, velocity Verlet, 1 fs, Nosé–Hoover τ = 1 ps, 25 ns × 10 independent runs.
  * Ewald with real-space cutoff 11.8 Å and SPME tolerance 10⁻⁵. Ghysels 2015 used SPME with an 11.7–12.0 Å cutoff
    and a 0.5 fs step for acid systems at 450/600 K.
  * Metadynamics through PLUMED: hills 0.5 kJ/mol, width 0.5 Å, pace 2.5 ps.
  * Walls keep ξ ∈ [−9, 9] Å. Spectator molecules are held within a maximum distance of the cage centre.
* **CV** (p. 6009): ξ = (r_M − r_C)·n_plane, the guest COM projected on the normal of the crossed ring, measured from
  the ring centre.

### (ii) Recoverable?

**Partially.** The framework force field, the framework topology CIFs (Ghysels zip, listed only) and the
MTD/US/MD protocol are recoverable. The rest is missing, closed or ambiguous:
1. **Guest intramolecular and host–guest parameters** are in closed-access sources (Oie 1981; Combariza 2011,
   restricted copy). Not recoverable tonight.
2. **The BAS positions elsewhere in the supercell** (1 or 2 BAS per unit cell even for a "type 0" ring) are not
   specified.
3. **Which cell the FF runs used is ambiguous.** The main text says AIMD-NpT cells were used for the production runs;
   SI S3.2 says FF thermal expansion via ZSISA/GULP.
4. **The spectator-restraint radius is inconsistent:** 8 Å in the main text (Sec. 2.2) versus 5 Å in SI S3.4.
5. **The AI-US level (revPBE-D3)** cannot be reproduced with a classical engine at all.

### (iii) Entropic contribution: strongest evidence of all candidates

Sec. 3.1, pp. 6010–6011; Table S5, p. S13:
* **Ethene, type-0 ring, 450 K (FF-MTD).** ΔF‡ = 21.2 kJ/mol (5.7 kT). The average nonbonded interaction energy is
  almost flat across the ring: the authors call it a "purely entropic barrier" (p. 6011).
* **Propene, type 0 (FF-MTD).**

  | T (K) | ΔF‡ (kJ/mol) | ΔE‡ (kJ/mol) | −TΔS‡ (kJ/mol) | −TΔS‡ (kT) |
  |---|---|---|---|---|
  | 300 | 27.8 | 8.7 | 19.1 | 7.7 |
  | 450 | 40.9 | 11.0 | 29.9 | 8.0 |
  | 600 | 49.4 | 11.6 | 37.7 | 7.6 |

  The propene barrier is 73 % entropic at 450 K.
* **AI-US** gives higher barriers: propene 32.0 / 42.6 / 56.4 kJ/mol at 300/450/600 K; ethene 31.1 kJ/mol at
  450 K (Table S1).
* **Acid sites.** π-complexation with BAS lowers the barriers (ethene type-1 6.9 kJ/mol A→B, AI-US). The FF does not
  reproduce π-complexation (p. 6013; Fig. S4).

### (iv) Reaction coordinate and geometric term

* In the flexible framework, r_C and n_plane are computed from the instantaneous ring atoms. ξ is therefore a
  function of about 16 ring atoms as well as the guest, and it is **nonlinear** in the framework coordinates (a
  rotating normal).
* ABF would need the full Jacobian and the divergence term, or a redefinition with a fixed reference plane, which is
  a different CV.
* In the repo's rigid CHA reduction the plane is fixed and the CV is linear.

### (v) Hidden slow coordinates

* **8-ring shape.** Breathing, ellipticity and puckering: Ghysels 2015 correlates crossings with d_min, puckering and
  ring area. Ring expansion is larger for propene (Fig. S7).
* **Guest orientation.** Propene's long axis must align with the ring normal.
* **BAS π-complex formation and proton mobility** (AIMD only, SI S9).
* **Loading of spectators.**
* **Unit-cell and temperature dependence of the ring area** (Table S3).
* **Every replica carries its own flexible framework,** as in the ZIF-8 stage.

### (vi) Cost per molecule-step, FF level

* Framework 2×2×2 = 864 atoms (+ BAS protons), all charged, so full SPME every step.
* Real-space pairs within 11.8 Å ≈ 864 × 289 / 2 ≈ 1.25 × 10⁵, plus SPME on about a 28 × 28 × 30 Å cell, plus about
  10³ three-body terms. Guest: about 2.6 × 10³ terms.
* Together that is **~180–200× ERI per step**. The stiff rigid-ion framework also forces a far smaller BD step, so
  the cost is **≳10³× ERI per unit physical time** (inference). The 1.5 core-h ERI dilute screen would become
  ≳1,500 core-h, against a 200 core-h ceiling.
* The AI level is out of reach.

### (vii) Already tested?

**Yes, in reduced form, twice.**
* **`src/cha` (Stage 6).** Rigid all-silica CHA (type-0 ring environment), TraPPE-UA ethene/propene, a fixed-plane
  version of Cnudde's CV, 450/600 K. Its own docstring and prereg say it borrows Cnudde's system, CV and temperature
  logic.
  * References: 12.2–17.0 kT barriers, 47–56 % entropic; rigid barriers are higher than Cnudde's flexible 5.7–10.9 kT.
  * Two-arm result: **NEUTRAL ×3**. ΔI_F −5.7 to −6.0 %, 16/16 wins but below the −10 % bar; final −12 to −27 %;
    median min ESS/N 0.22–0.30.
  * Source: `results/uniform_campaign/RESULTS.md` Stage 6.
* **`src/zif8`.** The flexible-framework-per-replica design (Krokidas FF, DSF Coulomb): closed harmful (+3.67 %),
  then neutral after the bandwidth fix.

### (viii) Verdict: **NOT-REPRODUCIBLE** within the CPU budget

* The AI level is impossible.
* The FF level is only partially recoverable: guest and host–guest terms are in closed sources, the BAS placement and
  the cell are unspecified, and the restraint radius is inconsistent.
* **Main risk:** cost (≳10³× ERI) plus repeating a closed negative. The feasible reduction is exactly the closed CHA
  Stage 6. The one new ingredient, framework flexibility, was already tested on ZIF-8 without a gain.

---

## Summary table

| | B: CO2/CH4 in DDR (Jee & Sholl 2009) | C: CH4 in MFI (López 2005) | D: CH4 pair in water-filled pore (Vaitheeswaran 2009) | E: olefins in flexible H-SAPO-34 (Cnudde 2020) |
|---|---|---|---|---|
| full text / SI | **no / no** (paywall; no OA copy) | **no** (paywall) | **yes** (arXiv + published) | **yes / yes** (+ Ghysels 2015 FF + 2017 correction) |
| guest model | unknown ("new force field") | unknown | UA methane (Kalra 2004), TIP3P | all-atom, Oie 1981 (closed) |
| framework | DDR (Gies 1986, inferred); rigid/flexible unknown | MFI; rigid/flexible unknown | hard cylinder walls, no PBC | flexible SAPO-34, rigid-ion + 3-body + Morse, all charged |
| electrostatics | unknown (CO2 needs Ewald/grid/DSF) | none expected for UA CH4 | direct all-pairs, no cutoff | SPME, 11.8 Å |
| all parameters recoverable? | **no** | **no** | **yes**, with 3 declared ambiguities (wall sites, hard → smooth wall, rigid water) | **partial** (framework yes; guest, host–guest, BAS sites, cell no) |
| entropic-barrier evidence | not readable | site-preference switch, low barriers (Beerdsen 2006) | analytic geometric −TΔS shapes the PMF; barrier 1.3–3.2 kT, solvent-driven | **strong**: ethene "purely entropic" 5.7 kT; propene 73 % entropic, T-series 300–600 K |
| CV / geometric term | COM ⟂ window, linear (rigid) | COM y, linear, periodic | distance r, 2kT/r term | COM ⟂ moving ring plane: nonlinear in the framework |
| hidden slow coordinates | CO2 orientation; framework flexibility for CH4 | channel-type occupancy; hysteresis; dynamics-only memory | water-column order / dipoles; radial and end-cap states | ring breathing; orientation; BAS π-complex; spectators |
| cost / molecule-step vs ERI | CH4 0.4–0.6×; CO2 ~1.7× + long-range + rigid-body BD | ~0.4× (rigid) | 0.7× (Nw 10) … 5–7× (Nw 26–32) … 120× (Nw 128) + rigid-water BD | ~180–200× per step, ≳10³× per unit time; per-replica framework |
| tested in repo? | no (closest: ERI, LTA, CHA) | no (UA CH4 only in water) | same family closed ABF-sufficient twice (methane/SPC/E, NaCl/TIP3P) | **yes**, reduced: CHA Stage 6 NEUTRAL ×3; flexible framework on ZIF-8, no gain |
| verdict | **NOT-REPRODUCIBLE** (surrogate CH4/rigid DDR FEASIBLE-WITH-CAVEATS) | **NOT-REPRODUCIBLE** (surrogate FEASIBLE, expected NEUTRAL) | **FEASIBLE-WITH-CAVEATS** | **NOT-REPRODUCIBLE** (CPU budget) |
| main risk | unverifiable model | unverifiable model; ABF-sufficient physics | conditional (solvent) regime ⇒ ABF-sufficient prior; new integrator | cost; repeats a closed negative; AI level impossible |

## Bottom line for the coordinator

1. **None of B–E is reproducible as published *and* likely establishment-starved.**
   * D is the only fully reproducible paper. Its physics sits in the solvent-conditional regime that marginal FR
     cannot repair, and in a family that closed ABF-sufficient twice.
   * E has the best entropic-barrier evidence, but it cannot be run as published on CPU, and its feasible reduction
     (rigid CHA) already closed NEUTRAL.
2. **The only cheap, new geometry in this scan** is a *labelled* CH4-in-rigid-DDR surrogate (B-derived). It uses the
   IZA/RASPA2 DDR structure (verified Si120O240) and the repo-verified Dubbeldam 2004 O–CH4 parameters on the
   `src/eri` kernel, and costs 0.4–0.6× ERI.
   * It is not Jee & Sholl and must not be described as such.
   * Its barrier height and entropic share are unknown. A < 5 min single-bead profile through one 8-ring would decide
     whether it is worth a Gate-0 → screen sequence, or whether it is discovery-limited.
3. **C's surrogate (CH4/MFI)** is the cheapest of all but is expected ABF-sufficient (low barriers). It is not
   recommended.
