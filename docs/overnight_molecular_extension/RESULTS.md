# Results — overnight molecular extension (ABF vs ABF + FR only)

*Last updated 2026-10-16 ~19:00 UTC. The only adaptive algorithms are histogram ABF (A) and the identical histogram
ABF + uniform marginal FR (B). FR is run only for cells that the frozen ABF-only classifier labels
ESTABLISHMENT-LIMITED.*

## Headline

| cell | model status | barrier on the CV (exact MC) | entropic share | ABF-only classification (K 64 / 256 / 1024) | FR run? |
|---|---|---|---|---|---|
| ethane / ERI, 600 K, infinite dilution | labelled surrogate of PNAS 2005 (Gard & Tait ERI, Schüring LJ); literature barrier reproduced (5.96 vs 6.13 kBT) | 33.3 kJ/mol = 6.67 kBT | **84 %** | **ABF-sufficient** at all K | no (not licensed) |
| ethane / Si-ITE (ITQ-3), 300 K, infinite dilution | labelled model (Dubbeldam 2004 UA FF); NOT a reproduction of Combariza 2011 | 39.7 kJ/mol = 15.9 kBT | **48 %** | **ABF-sufficient** at all K | no (not licensed) |

**No cell so far has met the preregistered condition for an ABF vs ABF + FR comparison.** No FR production result
exists in this extension, so no "FR helps" or "FR does not help" statement about these systems is supported. What
is supported is narrower:

* In both realistic 8-ring systems, standard histogram ABF discovers the neighbouring cage within a few time units.
  This holds even with a 16 kBT barrier.
* ABF then establishes the window population at its bias-aware equilibrium target as soon as the bias ramp ends. The
  population deficit FR is designed to repair is short-lived (ERI: none after 2 t.u.; ITE: 2–4 t.u. of a 50 t.u.
  horizon). It also does not carry the free-energy error: window bins hold 1–7 % of the final squared mean-force
  error.

## ERI (600 K) — ABF-only screen (8 seeds per K)

| K | first translocation | longest deficient run | window error share | e_F(T) | Ī_F |
|---|---|---|---|---|---|
| 64 | 2.06 | 0 | 0.07 | 0.175 | 0.555 |
| 256 | 1.29 | 1 | 0.07 | 0.129 | 0.501 |
| 1024 | 0.68 | 0 | 0.04 | 0.124 | 0.446 |

Times in t.u., errors in kJ/mol; T = 40 t.u. Details: `docs/eri_campaign/RESULTS.md`.

## ITE (300 K) — ABF-only screen (8 seeds per K)

| K | first translocation | min block occupancy / Q\* | longest deficient run (thr. 10) | window deposit share | window error share | conditional lag / floor (η, u_x²) | e_F(T) | Ī_F | e_F(T) / e_F(late) |
|---|---|---|---|---|---|---|---|---|---|
| 64 | 3.96 | 0.40 | 2 | 1.00 | 0.019 | 0.71, 0.56 | 0.124 | 0.523 | 1.15 |
| 256 | 2.95 | 0.41 | 4 | 0.99 | 0.010 | 0.85, 0.64 | 0.083 | 0.348 | 1.23 |
| 1024 | 2.63 | 0.39 | 4 | 1.00 | 0.009 | 0.87, 0.78 | 0.072 | 0.315 | 1.43 |

Times in t.u., errors in kJ/mol; T = 50 t.u.

* The window is ~60 % under-populated relative to Q\* only in the first 2–4 t.u. after the ramp, while the bias is
  still being built. The frozen rule needs ≥ 10 t.u.
* All file paths: `results/overnight/ite/screen/classification.json`.
* Figures:
  * `figures/overnight/ite/fig_screen_abf_only.png`: e_F(t), occupancy/Q\* in 2 t.u. blocks, translocations;
  * `figures/overnight/ite/fig_ite_reference.png`: F, U, −TS; hidden coordinates; channel geometry.

## Physical findings (validated, independent of FR)

* **ERI.**
  * The window barrier is overwhelmingly entropic: U has a local minimum at the window centre.
  * The single-bead literature profile is reproduced with the cited Gard & Tait framework and not with the Schlenker
    1977 cell.
* **ITE.**
  * Ethane is confined to 1-D channels along b. Moving sideways costs more than 100 kJ/mol per bead.
  * Neighbouring channel classes are related by C-centring (a b/2 shift of the window) or by a c-glide (y reversed),
    so a single linear CV is valid only within one translation family.
  * The barrier is half energetic and half entropic.
  * Crossing the window aligns the molecule with the channel: ⟨u_x²⟩ goes from 0.2–0.4 in the cage to 0.98 at the
    window.
* **Numerics.**
  * EM at dt 2e-4 is admissible for ERI at 600 K (D 0.037 kJ/mol against a 0.05 tolerance). It is NOT admissible for
    ITE at 300 K (D 0.036 against 0.03), so ITE uses dt 1e-4.
  * The bond variance error is first order in dt in both systems (+19 / +8.8 / +4.2 % at dt 2e-4 / 1e-4 / 5e-5).
    LM removes it.

## Not done / ruled out

See `CANDIDATES.md`:
* Candidate A as published (Combariza 2011: flexible charged framework, all-atom guests): not reproducible.
* B (Jee & Sholl CO2/DDR) and C (López CH4/MFI): paywalled with no recoverable parameters.
* E (H-SAPO-34): parameters closed, and over 10³× the budget.
* D (CH4 pair in a water-filled pore): feasible with model changes (smooth walls, a rigid-water integrator); the
  barrier is ≤ 3 kT and solvent-controlled, so it is expected to be ABF-sufficient. Deprioritised.
