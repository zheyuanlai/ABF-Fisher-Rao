"""ZIF-8 sampler ``ot=`` option (docs/ZIF8_OT_Z4Z5.md): byte-identity of the default, lift/repair
accounting, unwrapped gate band is diagnostic-only.  Synthetic framework, CPU."""
import importlib.util, os, sys
import numpy as np, torch
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path.insert(0, os.path.join(ROOT, "src"))
spec = importlib.util.spec_from_file_location("tz", os.path.join(os.path.dirname(__file__), "test_zif8.py")); tz = importlib.util.module_from_spec(spec); spec.loader.exec_module(tz)
from zif8.core_zif8 import ZIF8OTConfig, ZIF8SimConfig, run_sampler   # noqa: E402

KW = dict(dt=0.0005, gamma=1.0, n_steps=400, n_replicas=12, save_every=100, n_grid=24, abf_warmup_steps=50,
          estimator_burn_in_steps=50, fr_start_steps=100, fr_every=5, gate_every=10, rng_seed=3)


def _run(tmp_path, method="abf", ot=None, **extra):
    s = tz.make_system(tmp_path); pool = tz.make_pool(tmp_path, s)
    return run_sampler(method, s, ZIF8SimConfig(**KW, **extra), seeds=[0], init_pool=pool, verbose=False, ot=ot)


def test_ot_default_is_byte_identical(tmp_path):
    a = _run(tmp_path); b = _run(tmp_path, ot=ZIF8OTConfig(alpha=0.0, m_repair=0, every=100))
    assert np.array_equal(np.asarray(a["pmf"]), np.asarray(b["pmf"]))
    assert int(b["inner_steps_total"]) == 0 and int(b["ot_n_opportunities"]) == 4
    c = _run(tmp_path, gate_band_unwrapped=True)                      # diagnostic only: trajectory unchanged
    assert np.array_equal(np.asarray(a["pmf"]), np.asarray(c["pmf"]))


def test_lift_and_repair_accounting(tmp_path):
    N = KW["n_replicas"]
    t = _run(tmp_path, ot=ZIF8OTConfig(alpha=0.5, cap_bins=2.0, every=100, m_repair=0))
    assert int(t["ot_n_opportunities"]) == 4 and int(t["inner_steps_total"]) == 4 * N          # one lift force evaluation per opportunity
    assert float(t["ot_moved_frac"].mean()) > 0.5 and float(t["ot_absdphi_max"].max()) <= 2.0 * (2 * np.pi / KW["n_grid"]) + 1e-12
    assert float(t["ot_C_pre"].sum()) == float(t["ot_moved_frac"].sum()) * 4 * N and float(t["ot_C_post"].sum()) == 0.0
    assert np.isfinite(np.asarray(t["pmf"])).all()
    tr = _run(tmp_path, ot=ZIF8OTConfig(alpha=0.5, cap_bins=2.0, every=100, m_repair=7))
    assert int(tr["inner_steps_total"]) == 4 * (N + 7 * N)
    assert abs(float(tr["ot_C_post"].sum()) - float(tr["ot_C_pre"].sum())) < 1e-9                 # every moved walker sampled after repair
    assert np.isfinite(np.asarray(tr["pmf"])).all() and len(tr["series_inner_steps"]) == len(tr["steps"])
    r = _run(tmp_path, ot=ZIF8OTConfig(alpha=0.0, every=100, m_repair=7))                       # R arm: repair only
    assert int(r["inner_steps_total"]) == 4 * 7 * N and float(r["ot_C_pre"].sum()) == 0.0
    fr = _run(tmp_path, method="fr_uniform", ot=ZIF8OTConfig(alpha=0.0, every=100, m_repair=7), fr_rate=5.0, max_event_fraction=0.2)
    assert int(fr["inner_steps_total"]) == 4 * 7 * N and np.isfinite(np.asarray(fr["pmf"])).all()


def test_unwrapped_band_excludes_image_window(tmp_path):
    """A guest sitting at the periodic IMAGE of the window (unwrapped xi = L) has phi ~ 0 but must not be
    counted in the unwrapped band; a guest at the indexed window must."""
    from zif8.core_zif8 import gate_hist, TWO_PI
    s = tz.make_system(tmp_path); sim = ZIF8SimConfig(**KW, gate_band_unwrapped=True)
    q = tz.rand_config(s, B=4, jitter=0.0)
    xi = torch.tensor([0.0, s.period, -s.period, 2 * s.period], dtype=torch.float64)       # indexed, image, image, true image
    q = q.clone(); q[:, s.n_frame:] += ((xi - s.xi_value(q))[:, None] * s.normal[None, :])[:, None, :]
    xi_true = s.xi_value(q).reshape(1, 4) * s.k_phi
    phi_band = torch.remainder(xi_true + TWO_PI, 2 * TWO_PI) - TWO_PI
    a = torch.full((1, 4), 3.0, dtype=torch.float64)
    gh, _ = gate_hist(a, phi_band, sim, s.k_phi, torch.device("cpu"), torch.float64)
    assert float(gh.sum()) == 2.0                                                        # xi = 0 and xi = 2L only
    ghw, _ = gate_hist(a, s.cv_value(q).reshape(1, 4), sim, s.k_phi, torch.device("cpu"), torch.float64)
    assert float(ghw.sum()) == 4.0                                                        # the wrapped (legacy) band counts all four
