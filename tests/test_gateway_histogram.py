"""The histogram online estimator: textbook ABF bins, own-bin bias force, exact offline read-out;
the kernel path is untouched by its presence."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import gateway_core as gw  # noqa: E402


def _cfg(**kw):
    base = dict(beta=16.0, H=0.5, r=32.0, s=0.1, N=256, dt=4e-4, n_steps=1500, save_every=500,
                init="left", h=0.07, gamma=15.0)
    base.update(kw)
    return gw.GatewayConfig(**base)


def _run(cfg, seed=7, methods=(gw.ABF, gw.FR_UNIFORM)):
    spec = gw.BatchSpec(configs=[cfg, cfg], seeds=[seed, seed + 1], methods=list(methods), batch_seed=5)
    return gw.simulate_batch(spec, device="cpu", dtype=torch.float64, store_profiles=True,
                             store_accumulators=True)


def test_kernel_path_is_unchanged_by_the_new_fields():
    a = _run(_cfg())
    b = _run(_cfg(estimator="kernel", n_bins=0))
    for r in range(4):
        for k in ("F_hat", "Fp_hat", "l2_f_t", "l2_fp_t", "n_die", "n_clone"):
            assert np.array_equal(a[r][k], b[r][k]), (r, k)
    assert "Mh_t" not in a[0] and "hist_edges" not in a[0]


def test_histogram_estimator_is_the_textbook_bin_average():
    recs = _run(_cfg(estimator="histogram", n_bins=60))
    for r in recs:
        edges = r["hist_edges"]
        assert len(edges) == 61 and edges[0] == gw.XMIN and edges[-1] == gw.XMAX
        Mh, Ch = r["Mh_t"][-1], r["Ch_t"][-1]
        assert np.allclose(Ch.sum(), 256 * 1500) and np.array_equal(Ch, np.round(Ch))
        # the last fine-grid accumulator holds the same deposits (fine grid still fed for read-out)
        assert np.allclose(r["C_t"][-1].sum(), 256 * 1500)
        # the engine's final bin profile is M / (C + min_count)
        assert np.allclose(r["Fp_bins"], Mh / (Ch + 1.0), atol=1e-12)
        # the grid-valued profile the metrics read is that profile, piecewise constant per bin
        x = r["x_grid"]
        g2h = np.clip(np.floor((x - gw.XMIN) / (edges[1] - edges[0])).astype(int), 0, 59)
        # grid points that sit ON a bin edge may round to either neighbour (float64 linspace vs
        # division); everywhere else the mapping is unambiguous and must match exactly
        on_edge = np.abs(x[:, None] - edges[None, :]).min(axis=1) < 1e-9
        assert np.allclose(r["Fp_hat"][~on_edge], r["Fp_bins"][g2h][~on_edge], atol=1e-12)
        for i in np.where(on_edge)[0]:
            assert any(np.isclose(r["Fp_hat"][i], r["Fp_bins"][j], atol=1e-12)
                       for j in (max(g2h[i] - 1, 0), g2h[i], min(g2h[i] + 1, 59)))
        # bins visited at least once carry a finite, force-scale estimate
        vis = Ch > 0
        assert vis.sum() > 5 and np.all(np.isfinite(r["Fp_bins"][vis]))


def test_histogram_and_kernel_arms_differ_but_fr_acts_the_same_way():
    k = _run(_cfg())
    h = _run(_cfg(estimator="histogram", n_bins=180))
    assert not np.allclose(k[0]["Fp_hat"], h[0]["Fp_hat"])
    # both FR arms fire events; the estimator does not switch the birth-death step off
    assert h[1]["n_die"] + h[1]["n_clone"] > 0 and k[1]["n_die"] + k[1]["n_clone"] > 0


def test_n_bins_must_be_uniform_across_a_batch():
    a, b = _cfg(estimator="histogram", n_bins=60), _cfg(estimator="histogram", n_bins=30)
    spec = gw.BatchSpec(configs=[a, b], seeds=[1, 2], methods=[gw.ABF], batch_seed=5)
    try:
        gw.simulate_batch(spec, device="cpu", dtype=torch.float64)
    except AssertionError as e:
        assert "n_bins" in str(e)
    else:
        raise AssertionError("mixed n_bins accepted")
