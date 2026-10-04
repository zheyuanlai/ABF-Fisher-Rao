# Ethane in LTA at 150 K, histogram estimator: ABF vs ABF + Fisher-Rao birth-death

`lta_abf_vs_fr_histogram.mp4` (1920x1080, 30 fps, 35 s) is one seed of the ethane/LTA system at 150 K,
the starved point of the temperature sweep where ABF alone feeds the window slowly (0.9 cage-to-cage
crossings per replica in the whole run against 2.7 at 300 K). Same construction as `../T300/`:
textbook online histogram (P0) ABF estimator, frozen knobs of `configs/lta_histogram/campaign.json`,
both arms in ONE process from the same initial ensemble and noise stream, recorded every 100 steps,
drawn three snapshots per frame. Made 2026-10-03 (docs/LTA_HISTOGRAM_REPLICATION.md).

| item | value |
|---|---|
| system / CV / sampler / estimator | as `../T300/README.md`; T = 150 K (kT = 1.247 kJ/mol) |
| reference | umbrella/WHAM `results/uniform_campaign/lta/reference/reference_T150.npz`: dF‡ = 13.5 kT, -TdS‡ = 8.2 kT (61 % entropic), dU‡ = 5.3 kT |
| FR arm | `fr_uniform`: rate 0.20 (the sweep's safety-frozen 150 K rate), every 5 steps from step 20 000 (t = 4), score clip 2, cap 0.02, marginal KDE 0.10 |
| seed | label 1200, rng 20261150; featured replica: slot 0 (never replaced) |
| this seed | e_F(T): ABF 0.1765 kJ/mol, FR 0.0951 (-46.2 %); I_F 31.9 vs 22.2 (-30.3 %); 819 replacements; min ancestor ESS/N 0.308, max lineage share 0.013; cage-to-cage crossings 919 (ABF) vs 1 037 (FR, +13 %) |
| kernel sweep medians at 150 K (16 seeds, results/uniform_campaign/lta/summary_T150.json) | -31.9 % integrated [-33.4, -28.3], -56.3 % final, 16/16; the 16-seed histogram replication is `results/lta_histogram/` |
| time to ABF's final accuracy (this seed) | FR holds e_F <= 0.1765 from t = 21.6; ABF reaches it only at the end (2.8x) |
| pairing | identical until the first FR opportunity at t = 4.02; paired in distribution afterwards. The record was re-simulated on 2026-10-04 with the genealogy keys (same seed; CUDA scatter-add order makes a re-run differ slightly after t ~ 10: the first 254 events are identical, the endpoint moved from -49 % to -46 %); the first record is archived in `raw_pre_genealogy/` |

Panels as in `../T300/README.md`.

`lta_abf_vs_fr_histogram_t0-20_clip.mp4` (11 s, 8.8 MB) is the first 20 time units of the same movie cut
out for talks: the starved phase, where every death is in a cage and every birth lands in the neck or the
window while the ABF column is still filling the window.

## What to point at while it plays

1. t < 4: both columns identical; first window visit at t = 1.3, 2.8 % of replicas in the window at
   t = 4, zero complete crossings in 1024 replicas. The window cloud in (b) is a dot inside the 8-ring.
2. t = 4 to 10 - the starved phase, where this movie differs from 300 K. 254 replacements, and
   251 of the 254 deaths are in the CAGE while 249 of the 254 births go to the neck / window: selection
   moves population through the bottleneck that ABF alone is still struggling to feed. By t = 10 the
   window holds 22.6 % of the FR replicas against 13.3 % under ABF (uniform level 25 %), the crossing
   count is 33 vs 17, and e_F is already 0.56x.
3. t = 10 to 20: FR's marginal reaches the uniform level (0.257 at t = 20) while ABF is still at 0.226;
   crossings 201 vs 106; e_F ratio bottoms at 0.41 at t = 20. Deaths start to include the window
   (41 of 146) once it is populated: FR now REDISTRIBUTES rather than pushes.
   Genealogy (new record): at t = 6, 40 % of the replicas in the window are Fisher-Rao COPIES and 0 % of
   those in the cages; at t = 8, 52 % vs 4 %. 904 of the 1024 initial families are still alive at t = 6,
   516 at the end; the largest family ever has 13 members - the mechanism is hundreds of small families,
   not a few lineages taking over, which is why the ancestor ESS stays above 0.30.
4. t > 20: the two curves fall in parallel at a ratio of 0.44-0.54; FR holds ABF's final accuracy from
   t = 21.6. Over the whole run FR generated 13 % more window crossings than ABF - the mechanism
   signature of the sweep (at 80 K the kernel sweep found the uniform arm nearly doubling the traffic,
   at 300 K the two arms are identical).
5. Compare with `../T300/`: there the first FR phase is short (ABF has the window populated by t ~ 10)
   and the gain saturates at 0.7-0.9x; here the ABF marginal needs until t ~ 30 and FR halves the error.

Provenance: `scripts/run_lta_histogram.py movie --temperature 150`, `scripts/lta_movie_render.py --out results/lta_movie/T150`.

## The ensemble movie: `lta_ensemble_abf_vs_fr.mp4` (≈34 s; `scripts/lta_ensemble_movie_render.py`)

Same record, same seed, but the object on screen is the EMPIRICAL MEASURE of the 1024 replicas, not the
molecules. Three acts:

* **Act I (0–6 s), the physical problem.** One replica (slot 652, the ensemble's first window
  discoverer at t = 1.3) moves through the framework slab while a cursor follows its z_CV on the
  F / U / −TS decomposition; at the end all 1024 COMs fade in and the caption switches to "one dot = one
  replica, not one atom".
* **Act II (6–22 s), the ensemble.** Two columns, ABF | ABF + Fisher–Rao, same scales. Top: the replica
  strip — every replica a bead stacked along z_CV (96 bins), originals grey, Fisher–Rao COPIES (ids ≥ N)
  orange, the three largest families (initial walkers 728, 595, 75) in blue / purple / green; deaths (×)
  and births (○) flash where they happen. Under it the heat strip log₂ p̂_t(z)/q(z) (blue = deficit →
  births, red = excess → deaths). Then the meter R_W(t) = p̂_t(W)/q(W) as one large number ("0.49×" vs
  "0.88×" at t = 10) with the window occupancy, the running counts of deaths and births, "copies among
  window / cage replicas" (49 % / 6 % at t = 10) and "families alive" (775 of 1024), and the learned F.
  Bottom: e_F(t) and R_W(t) of both arms. The clock is slow between t = 4 and 16 (the bloom) and fast
  afterwards.
* **Act III (22–32 s), the memory of the movie.** The two density kymographs (z vs t) revealed left to
  right, with the lineage trees of the three families drawn on top (single lines under ABF, branching
  trees under FR; the largest family in full colour), deaths and births on the FR panel, a panel with the
  fraction of copies among window and cage replicas vs t, and R_W(t) for both arms. Closing line from the
  data: at t = 8, 52 % of the replicas in the window are Fisher–Rao copies (ABF: 0 %); largest family 13
  of 1024; e_F(T) 0.14 vs 0.08 kT (−46 %).

What to point at: the orange copies appear in the window and neck first (t = 4–8) and only later in the
cages; the ABF window stays blue in the heat strip until t ≈ 20 while the FR strip is neutral by t ≈ 10;
and the family trees show that the "bloom" is hundreds of small families (largest 13), not a takeover —
which is why the ancestor ESS stays above 0.30.
