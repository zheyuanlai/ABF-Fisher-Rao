# ZIF-8 flexible force field — literature retrieval findings (2026-08-30)

All statements below are backed by files saved in this directory; see `provenance_forcefield.json`
for exact URLs. Nothing is reconstructed from memory. (`provenance_structure.json` and the
`zif8_*.cif` files belong to a parallel structure-retrieval task.)

## Which force field the 2024 anchor paper uses

The anchor paper (Schmidt, Cnudde, Van Speybroeck, Vanduyfhuys, *J. Phys. Chem. C* 2024,
128, 18509, DOI 10.1021/acs.jpcc.4c04790 — full accepted manuscript saved as
`jpcc2024_ethane_zif8_accepted_manuscript.docx`, open access via biblio.ugent.be) states in its
Methods: simulations use **the force field of Krokidas et al.** ("based on AMBER and TraPPE and
improved using DFT calculations", main-text ref 60 = Krokidas 2015, DOI 10.1021/acs.jpcc.5b08554),
run with **YAFF coupled to LAMMPS** on a 2×2×2 supercell (2208 atoms), NVT/NPT, 12.0 Å
truncated+switched cutoffs, 0.5 fs timestep. The paper's own SI cites the same FF via the group's
2018 mixed-linker paper (DOI 10.1021/acsami.8b12605).

## Force fields secured COMPLETELY

1. **Krokidas et al. flexible ZIF-8 FF** (the anchor paper's FF) — **complete, from two primary
   SIs plus a machine-readable implementation**:
   - `krokidas2018_zif78_si.pdf` (SI of DOI 10.1021/acsami.8b12605, via ACS figshare):
     **Table S1** (p. S-2) bonds (l0 in Å, k_l in kJ/mol/nm²) + angles (θ0 in deg, k_θ in
     kJ/mol/rad²); **Table S2** (p. S-3) torsions (φ0 deg, multiplicity m, k_φ kJ/mol, "AMBER");
     **Table S3** (p. S-3) partial charges (e) + LJ ε/σ per atom type (σ in nm; the ε column
     header is misprinted "kJ/mol/nm2" — values are kJ/mol); **Figure S1** atom-type map
     (Zn, N, C1 = ring C–H, C2 = N–C–N carbon, C3 = methyl C, H1 = ring H, H2 = methyl H).
     **No improper terms are defined for ZIF-8 in this FF** (that is the FF design, not a gap).
   - `krokidas2017_zif8_zif67_si.pdf` (SI of DOI 10.1021/acs.jpcc.7b05700): Tables S1–S2
     bond/angle/torsion values **identical** to the 2018 SI (independent cross-check), Table S5
     charges; LJ only by reference to Hertäg et al. (so this file alone is incomplete).
   - `zif8_gromacs_aanik_atomtypes_ZIF-8_Krokidas.itp` (GitHub a-anik/zif-8_md, commit 615ea23,
     2019-01-14; adaptation published with Sheveleva et al. JPCC 2017, DOI 10.1021/acs.jpcc.7b06884):
     complete GROMACS parameter file whose numbers match the 2018 SI to its printed precision
     (the itp carries extra digits consistent with exact kcal→kJ conversion, e.g. 296.2272 vs
     printed 296.23); it also pins the
     functional-form conventions (harmonic bond/angle GROMACS func 1, periodic proper dihedrals
     func 1, Lorentz–Berthelot combining, fudgeLJ 0.5 / fudgeQQ 0.8333) and documents its own
     deviations in the header (Zn charge 1.3426 vs published 1.3429 for exact neutrality; atom
     names remapped to Zheng convention; C2-C2-N angle 108.7° from an author-provided topology
     vs 107.9° published; a few unlisted dihedrals added from the author topology).
   - Companion `zif8_gromacs_aanik_zif8_2x2x2_periodic.itp` enumerates every bonded interaction
     of the 2×2×2 supercell (15,360 typed bonded terms) and `zif8_gromacs_aanik_conf_2x2x2.pdb`
     is the matching Park-2006-derived starting structure.

2. **Zheng, Sant, Demontis, Suffritti 2012 FF** (DOI 10.1021/jp209463a) — **complete**, from:
   - `mmm2022_zif8_ff_evaluation_si.pdf` (UvA Pure; SI of Acuna-Yeomans et al., Micropor.
     Mesopor. Mater. 2023, 348, 112406, CC-BY main text saved as
     `mmm2022_zif8_ff_evaluation_main.pdf`): FF1 columns of Tables S1 (non-bonded: ε kcal/mol,
     σ Å, q e), S2 (bonds), S3 (angles), S5 (impropers), S6–S7 (dihedrals), S8 (cutoffs +
     1-4 scaling), **with explicit functional-form equations** — the most rigorous open
     reproduction found.
   - `zif8_gromacs_aanik_atomtypes_ZIF-8_Zheng.itp`: machine-readable GROMACS version
     (nonzero harmonic impropers 16.736 kJ/mol; header notes Zn charge +0.7326 vs published
     +0.7362, declared a misprint fix — reconcile before use).
   - The paper's own SI (`zheng2012_zif8_ff_si.pdf`, via ACS figshare) contains **no** parameter
     tables (they are in the paywalled main text); it does give the 6MR window radius (3.33 Å at
     a = 16.99 Å; 3.25 Å at 16.79 Å) and CO2 D_s sensitivity numbers.

3. **Wu, Huang, Cai, Jaroniec 2014 FF** (RSC Adv., DOI 10.1039/C4RA00664J) — **complete**, from
   the same two sources: FF4 columns of `mmm2022_zif8_ff_evaluation_si.pdf` and
   `zif8_gromacs_aanik_atomtypes_ZIF-8_Wu.itp` (fudgeQQ 0.5 for this FF).

4. **Bonus (complete, same 2022 SI):** Jiang 2012 (FF2), Zhang 2013 (FF3), and
   Weng–Schmidt 2019 (FF5, includes Urey–Bradley terms) — full tables in
   `mmm2022_zif8_ff_evaluation_si.pdf`.

## Secured partially / not secured

- **Krokidas 2015 SI itself** (jp5b08554_si_001.pdf): not obtainable without a browser (ACS 403;
  not on figshare, no OA mirror). Mitigated: the group's 2017 and 2018 SIs (saved) publish the
  full ZIF-8 tables with values identical to each other and to the GROMACS implementation.
- **Verploegh/Sholl 2015** (`verploegh2015_jacs_zif8_si.pdf`): guest-molecule FF complete
  (Tables S1–S2) and ethane free-energy barriers vs T (Table S3), but the framework FF it uses
  (Zhang 2013) is only cited there — Zhang 2013 tables are instead in the 2022 evaluation SI.
- **Krokidas 2020 SI** (`krokidas2020_modified_zifs_si.pdf`): ZIF-7-8/Co analogues, context only.

## Anchor-paper physical numbers (all from the saved accepted manuscript / its SI)

- **Ethane cage-to-cage phenomenological free-energy barrier ΔF‡/ΔG‡** (Table 1 of the docx;
  reproduced in `jpcc2024_ethane_zif8_accepted_manuscript_extracted.txt` as "TABLE 0"):
  Krokidas FF **NPT 300 K, 1 bar: 24.2 ± 2.6 kJ/mol** (NVT: 23.6 ± 2.7); DFT revPBE:
  24.9 ± 2.6 (NPT) / 23.3 ± 2.7 (NVT); literature comparison Verploegh et al.: 31.3 (NVT) /
  26.6 (NPT) kJ/mol at 308.15 K.
- **D_self (ethane, 300 K, NPT, Krokidas FF)**: 1.50×10⁻¹⁰ m²/s (CI 1.35–1.66); experimental
  (Chmelik et al., IR microscopy, Table 1 footnote) 0.1×10⁻¹⁰ m²/s; NMR value quoted in the
  intro: 1×10⁻¹¹ m²/s.
- **Temperature trend** (Table 2 / "TABLE 1"): barrier **rises** with T — 20.1 ± 2.1 (250 K),
  24.2 ± 2.6 (300 K), 28.4 ± 3.4 (350 K), 32.5 ± 4.0 kJ/mol (400 K) — the paper's headline
  evidence that the barrier is **entropic** (covalent gate-opening cost canceled by host–guest
  non-bonded stabilization at the TS); gate size also increases with T (mostly thermal expansion).
- **Loading trend** (Table 3 / "TABLE 2"): symmetric high loading raises the forward barrier to
  30.3 ± 2.6 kJ/mol (6-5 loading) and cuts D_self ~10× (0.13×10⁻¹⁰ m²/s); asymmetric loading
  of one cage leaves the barrier ~unchanged (23.3–25.2 kJ/mol).
- **Geometry** (main text): sodalite cage diameter 11.6 Å; 6MR gate ≈ 3.4 Å (8 per cage);
  4MR aperture 0.8 Å (non-diffusive); 2×2×2 cell length 33.19 Å; symmetric-loading cell volume
  grows 42.83 → 42.93 nm³ from low to high loading.

## Concerns / traps for the extraction step

1. **Unit misprint in the primary SI**: Krokidas 2017/2018 Table S1/S3 headers write k_l in
   "kJ/mol/nm2" (bonds — correct) but also label LJ ε as "kJ/mol/nm2" (should be kJ/mol);
   angle k_θ is kJ/mol/rad². The GROMACS itp resolves every unit unambiguously.
2. **Energy-constant convention**: the SI tables do not print the functional forms; the GROMACS
   adaptation uses the printed constants directly as GROMACS func-1 constants (E = ½k(b−b0)² etc.)
   and was built with an author-provided topology, but anyone re-deriving for LAMMPS/AMBER-style
   E = K(x−x0)² must check the ½ convention against a published observable first.
3. **Atom-name remapping**: the 2018 SI's C1/C2/H1/H2 differ from the Zheng-convention names used
   in the GROMACS files (paper C2 = file C1 = N–C–N carbon; paper C1 = file C2 = ring CH;
   paper H1 = file H2, paper H2 = file H3). The mapping is stated in the itp header.
4. **Charge neutrality tweaks**: published charge sets are very slightly non-neutral; the itp
   headers document per-FF fixes (Krokidas Zn 1.3429→1.3426; Zheng Zn 0.7362→0.7326 "misprint";
   Wu C3/H3 rounding). Decide deliberately which to adopt.
5. **pdftotext quirks in `mmm2022_zif8_ff_evaluation_si.pdf`**: the ε row labels of Table S1
   disappear in plain-text extraction (rows cycle ε/σ/q per atom), and FF4's H3 carries two
   charge values (0.1325, 0.1306) with a group-neutrality note on p. S8 — extract that table
   with layout mode and cross-check against the itp files.
6. **A poisoned secondary source was rejected**: the UCL thesis (discovery.ucl.ac.uk/10069803)
   appendix "Krokidas intramolecular Force Field" mixes Krokidas force constants with Zheng
   equilibrium values (e.g. Zn-N l0 2.011 Å instead of 2.048 Å). It was NOT saved. Do not use
   theses' restated tables without cross-checking a primary SI.
7. The anchor docx tables extracted cleanly, but superscripts flatten in plain text
   (e.g. "1.38x1011" in one intro sentence should read 1.38×10⁻¹¹) — the companion
   `*_extracted.txt` flags this; use the docx for load-bearing values.
