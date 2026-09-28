# Entropic-gateway movie, histogram estimator: ABF vs ABF + Fisher-Rao birth-death

`gateway_abf_vs_fr_histogram.mp4` (1920x1080, 30 fps, 43.7 s, 54 MB) is the companion of
`results/gateway_movie/gateway_abf_vs_fr.mp4` with ONE change: the online mean-force estimator is the
textbook ABF histogram (P0) of the replication campaign (docs/HISTOGRAM_ABF_REPLICATION.md) instead of
the Gaussian kernel.  Same frozen cell, same seed, same batch seed, so the Langevin noise is the SAME
as in the kernel movie as well as between the two columns; recorded every 40 steps, two snapshots per frame.

| item | value |
|---|---|
| cell | beta 16, s 0.10, r 32, beta*H 8 kT (barrier 11.5 kT = 8 energetic + 3.5 entropic) |
| sampler | N 2048, dt 4e-4, 100 000 steps (T = 40) |
| estimator | histogram, 45 bins on [-1.8, 1.8] (Delta xi = 0.08, the campaign's frozen width `configs/histogram_abf/selected_bins.json`); bias force = own-bin M_j / (C_j + 1); reported F = exact piecewise-linear integral of the bins (own read-out, no kernel rescoring) |
| FR arm | `fr_uniform`, gamma 1.5, rate ramped over the first 10 % of the run, fr_every 10 (unchanged: the FR marginal KDE, target, score and schedule are the kernel movie's) |
| seed / init | 400, `left` (every walker starts in B_-), batch seed 41 000 |
| this seed | e_F(T): ABF 0.00715, FR 0.00319 (-55 %); integrated -26 %; 3 386 deaths, 2 960 births; min ESS/N 0.421 |
| same seed under the kernel (results/gateway_movie) | e_F(T) at h_read*: ABF 0.00536, FR 0.00177 (-67 %); integrated -33 %; 5 782 deaths, 4 900 births |
| confirmation medians (32 pairs, results/histogram_abf/gateway/confirmation/summary.json) | histogram FR -47.7 % final, -30.0 % integrated (32/32, 31/32); kernel at h_read* -58.8 % / -32.9 % |
| P0 discretisation floor at Delta 0.08 (results/histogram_abf/gateway/floor) | e_F 0.0030, i.e. ~40 % of the ABF endpoint: the endpoint gain is smaller under the histogram partly because ABF is already near this floor; a finer width lowers it (0.0017 at 0.06, 0.0008 at 0.04) |

Provenance: prereg cell and sampler from `results/gateway_anchor/CONFIRMATORY_PREREGISTRATION.json`,
rate from `configs/information_campaign/gateway_corrected_confirmation_prereg.json`, width from
`configs/histogram_abf/selected_bins.json`, engine `src/gateway_core.py` (`GatewayConfig.estimator =
'histogram'`, shared `eb_abffr_core.HistogramABFEstimator`) with the report-only `store_snapshots`
record (bit-inert: `tests/test_gateway_snapshots.py`).  Data in `movie_data.npz`; the drawn / scored
free energy per frame is the engine's own `snap_F`, checked against the engine's `final_l2_f` and
its `l2_f_t` at every save that is also a snapshot (`scripts/make_gateway_movie.py simulate`).
The kernel read-out at h_read* = 0.0175 of the fine-grid accumulators is stored alongside
(`snap_eF_star`: ABF 0.00671, FR 0.00224) but is not what the movie shows.

## What each panel shows

Identical to the kernel movie's panels (see `results/gateway_movie/README.md`), except:

* **Third row** - the learned free energy is the histogram's own exact piecewise-linear PMF; the
  coloured tick marks along the bottom edge are the 45 bin edges.  There is no read-out bandwidth.
* The title carries "(histogram mean-force estimator)".

## What to point at while it plays

1. t < 1.5: both arms sit in B_-; the first walker crosses the gateway at t = 1.57 (ABF) and 1.47 (FR),
   the same noise in both columns.
2. t ~ 1.5-6: the FR crosses cluster in the crowded left basin, the rings appear at the gateway exit
   and in B_+; the FR histogram is flat and its B_+ fraction reaches 90 % of the uniform target at
   t = 5.7.  The ABF arm needs until t = 38.7 for the same.
3. t ~ 5-20: the FR free-energy error falls to 0.70x ABF's at t = 5, 0.56x at t = 10 and 0.48x at
   t = 20; the ABF learned F still bulges around the gateway where its bins have few samples.
4. t > 20: the ratio settles at ~0.45 (final -55 %).  Compared with the kernel movie of the same seed
   the FR arm fires ~40 % fewer birth-death events (3 386 vs 5 782 deaths): the own-bin bias is
   sharper, so the ABF marginal flattens faster on its own and the score has less to act on.

## Reproduce

```bash
CUDA_VISIBLE_DEVICES=3 python scripts/make_gateway_movie.py simulate --estimator histogram   # 93 s on one H200
python scripts/make_gateway_movie.py render --estimator histogram --workers 24                 # ~1 min, 24 CPUs
python scripts/make_gateway_movie.py render --estimator histogram --only 300 --frames-dir /tmp/x   # one frame
```
`--n-bins` overrides the frozen width; other seeds / the `one_right` init as in the kernel movie.
