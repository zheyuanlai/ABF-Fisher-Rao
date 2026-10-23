## Numerical audit notes (manual checks; complement the automated table above)

**Reproduction of the published statistics.** Every exporter calls the repository's own scorer and recomputes the
original paired statistic from the exported per-seed curves.

| experiment | statistic | reproduced | published / frozen record | agreement |
|---|---|---|---|---|
| gateway (v2 anchor, seeds 8100–8131) | Ī_F | −31.4 % [−34.7, −27.7] 31/32 | eqb summary.json −31.4 % [−34.7, −27.7] 31/32 | per-seed identical (max diff 0) |
| gateway (dt study, seeds 7100–7131) | Ī_F at h 2.5e-5 | −29.67 % 30/32 | OVERNIGHT_REPORT −29.7 % 30/32 | identical |
| LTA 150 K | I_F | −28.54 % [−29.4, −26.6] 16/16 | LTA_VALIDATION −28.54 % [−29.41, −26.56] | identical; atlas curves deviation 0 |
| LTA 300 K | I_F | −13.66 % [−14.8, −11.1] 16/16 | LTA_VALIDATION −13.66 % [−14.79, −11.11] | identical; atlas curves deviation 0 |
| ITE 300 K E1 | S_geo; post-ramp Ī_F; e_F(T) | 1.586 [1.311, 1.852] 16/16; −37.7 %; −32.3 % | E1.json 1.586 [1.311, 1.852]; −37.7 %; −32.3 % | identical; E1_curves deviation 0 |
| alanine | dI_F on [20, 100] ps; e_F(T) | +0.04 % [−1.36, +0.64]; −0.60 % | scoreboard +0.04 % [−1.36, +0.64]; −0.60 % | identical |
| WCA | I_F; e_F(T); e_F′(T) | +9.9 % [+2.3, +13.2] 4/16; +131 %; +10.5 % | WCA_CONFIRMATORY +9.9 % [+2.3, +13.2] (4/16); +131 %; +10.5 % | identical; median curves deviation 0 |
| metastability (NEW) | I_F; e_F(T) | +51.9 % [+40.2, +60.7] 0/32; +334 % | no prior record (new preregistered experiment) | — |

**Marginal panel (c) checks.**

* Every histogram row sums to N: gateway 2048, ITE 1024, WCA 256, metastability 1000.
* The alanine `marg_hist` mass is 1 to within 1e-4 (float32 storage).
* The LTA `p_hat` integrates to 1 to within 1e-9. For the FR arm the exported e_ξ equals the engine's own ‖p̂ − q‖
  (`pq_l2`, q uniform) to 1e-16.
* The densities use the full RC domain: [−1.8, 1.8], [−π, π), [−0.2, 1.2], [−3, 3] and the torus.

**Known deviations, reported rather than hidden.**

1. *Alanine pairing is statistical, not bitwise.*
   * The two stages ran on CUDA, where float64 `scatter_add` is order-nondeterministic
     (`docs/FR_START_TIMING.md` l. 103–107).
   * The arms are identical through t = 2 ps and drift apart from about 4 ps, when the ABF bias starts acting
     (median relative difference ≈ 0.5 % before FR starts at 20 ps).
   * The other five existing experiments, and the new one, are bitwise identical before FR acts (asserted).
2. *WCA has only 49 checkpoints* (every 5 t.u.). No denser grid was saved and none was interpolated.
3. *WCA panel (c) is edge-dominated.*
   * About 16 % of the replicas sit in the soft-wall tails, which `hist_bin` clamps into the two 0.00875-wide edge
     bins.
   * They contribute ≈ 1.2 of the ≈ 1.35 total.
   * The interior-only diagnostic (`figs/data_supplementary/wca_exi_interior.csv`) sits at the finite-N floor in
     both arms.
4. *Alanine panel (c) is at the finite-N floor.*
   * N = 2048 walkers on 97² cells: the raw-histogram floor is 0.341, reached by about 20 ps in both arms.
   * The steric floor (0.025 for the reference's finite support, 97.6 % of the torus) is not resolvable at this N,
     and neither is it with the FR's own KDE: 0.042 against a KDE noise floor of 0.0416. Supplementary
     `alanine_exi_kde.csv`.
5. *Alanine e_F′ is reference-noise limited.* The endpoints (≈ 8.25 kJ/mol/rad) are of the size of the reference's
   own cell noise propagated through the central difference (8.66, independent-cell estimate; indicative only).
6. *Piecewise-constant floors of e_F′.* These are hard floors of the frozen scorers: gateway 0.0323; metastability
   0.139 at 120 bins.
7. *Metastability timestep gate.* All candidates pass. The measured discrepancies are statistical-noise dominated
   (the exact MALA chain gives the same values), so the gate bounds the EM bias by its resolution rather than
   measuring it.

**Visual inspection of the seven previews.**

* No missing segments, spikes from processing, clipped axes or wrong units.
* Genuine features that are kept:
  * the gateway initial plateau;
  * the LTA kink at t = 4, where the reported estimator switches to the production accumulators;
  * the alanine e_F rise during the 5 ps ABF ramp;
  * the WCA e_F / e_F′ spike at t = 20 (estimator burn-in reset);
  * the WCA plateau of FR above ABF after t ≈ 50.
