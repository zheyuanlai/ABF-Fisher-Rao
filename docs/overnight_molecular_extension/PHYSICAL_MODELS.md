# Physical models — overnight molecular extension

Both models use the same engine (`src/eri/`):
* united-atom ethane: two CH3 beads with a harmonic bond;
* a rigid framework with exact unit-cell voxel neighbour lists;
* energy-shifted Lennard-Jones;
* overdamped Langevin with unit mobility (time unit t.u. = Å² mol/kJ, so D = kT);
* a linear CV equal to the tagged centre of mass along one box axis, mapped to the circle (no geometric term in the
  local mean force).

## 1. Ethane / ERI (600 K): PNAS-2005-type surrogate

Details and sources: `docs/eri_campaign/review/LIT_CROSSCHECK.md`, `docs/eri_campaign/RESULTS.md`.

| item | value |
|---|---|
| framework | Gard & Tait 1973 erionite (RASPA2 `ERI_SI.cif`), rigid, all-silica; box 3a × 2√3a × 3c = 39.81 × 45.97 × 45.15 Å; 1296 Si, 2592 O |
| host–guest | O–CH3 142 K / 3.17 Å; Si–CH3 82 K / 2.12 Å (Schüring et al. 2002, as quoted by PNAS 2005) |
| guest–guest | CH3–CH3 104 K / 3.78 Å |
| bond | ½ · 96 500 K/Å² · (r − 1.54 Å)² |
| cutoff | 12 Å, energy-shifted |
| CV | φ = 2π (X − a/4)/(a/2), X along hexagonal a1 |
| deviations from the paper | Framework identity unrecoverable (Gard & Tait chosen by the frozen literature rule). Overdamped Langevin replaces the paper's MD (same equilibrium) |

## 2. Ethane / Si-ITE (ITQ-3), 300 K and 150 K — labelled model, NOT Combariza et al. 2011

| item | value | source |
|---|---|---|
| framework | ITQ-3 calcined, Cmcm, a 20.622, b 9.7242, c 19.623 Å; Camblor et al., Angew. Chem. Int. Ed. 36, 2659 (1997) | RASPA2 `ITQ-3.cif` (= IZA type material); sha256 in `cache/eri/framework_ite_itq3.json` |
| box | X = 3b, Y = 2c, Z = 2a = 29.17 × 39.25 × 41.24 Å (perm (b, c, a), proper rotation); 768 Si, 1536 O; coordination check passes (Si–O 1.51–1.67 Å) | `scripts/overnight/save_ite.py` |
| host–guest | CH3–O 93.00 K / 3.48 Å, O only (Si does not interact) | Dubbeldam et al., J. Phys. Chem. B 108, 12301 (2004), Table 3; PRL 93, 088302 Table I |
| guest–guest | CH3–CH3 108.00 K / 3.76 Å (loaded cases only) | same |
| bond | ½ · 96 500 K/Å² · (r − 1.54 Å)² | same (Nath, Escobedo & de Pablo 1998) |
| cutoff | 12 Å, energy-shifted, no tail | JPCB 2004 Eq. 14 |
| channels | 1-D 8-ring channels along b. One [010] 8-ring per period: 8 O atoms at 3.24–3.50 Å from the axis, free aperture ≈ 3.8 Å. Single-bead percolation along b at 2–5 kJ/mol; along c only above 100 kJ/mol; never along a | `scripts/overnight/ite_percolation.py`, `ite_channels.py` |
| channel families | four per unit cell. C-centring shifts the window by b/2 between families; the c-glide maps y → −y. **Only family A is used** (axis Y = c/4 + m c, Z = n a): every molecule starts there and physically cannot leave (Gate 0 test) | `save_ite.py` |
| CV | φ = 2π (X_com − x_w)/b, x_w = 4.1979 Å (mean X of the 8-ring O atoms). Windows at φ = 0, cage centres at ±π | – |
| hidden-variable diagnostics | η = cos(2π Z_com/a) (lateral off-axis position); u_x² (alignment with the channel) | – |
| differences from Combariza 2011 | The paper used a flexible BKS framework (charged, Ewald) with all-atom Oie 1981 ethane; its parameters, temperatures and loadings are unrecoverable. This model shares only the physical system | `CANDIDATES.md` |

## 3. Reaction-coordinate validity

* **ERI.** The φ = 0 plane is open only at the 144 between-row windows; no other pathway exists (audit
  `AUDIT1_FORCEFIELD_GEOMETRY.md`). The within-row windows sit at φ = ±π and are symmetry-equivalent. No hidden
  metastability enters F(φ).
* **ITE.** Within family A every period has exactly one window. The perpendicular rings are closed. The CV is exactly
  linear, so f = −(b/2π)(F_0x + F_1x).
