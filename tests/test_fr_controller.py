"""Online FR stopping controller (src/eri/fr_controller.py) and its engine integration (engine.fr_controller).
Synthetic tests of the decision logic first, then molecular-engine tests (ERI surrogate, small K).

    NUMBA_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" /home/zheyuanlai/miniconda3/envs/abffr/bin/python -m pytest -q tests/test_fr_controller.py
"""
import inspect
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from eri import engine as E  # noqa: E402
from eri import fr_controller as FC  # noqa: E402
from eri.model import Model  # noqa: E402

NG, K = 180, 512


def synthetic(schedule, support=True, G_noise=0.0, essw=1.0, seed=0, T=12.0, spec=FC.R1, t_start=1.0):
    """Feed cumulative birth/death histograms built from a per-interval schedule.  schedule(t) -> (n_events,
    directed fraction).  Directed events move walkers from the 'cage' arc (regions 0-3) into the 'window' arc
    (regions 8-9); the rest are symmetric noise events."""
    rng = np.random.default_rng(seed)
    ctl = FC.OnlineStop(spec, K, t_start)
    BH = np.zeros(NG)
    DH = np.zeros(NG)
    C = np.full(NG, 1e6 if support else 10.0)
    if not support:
        C[: NG // 2] = 1e6
    M = 5.0 * np.sin(np.linspace(-np.pi, np.pi, NG, endpoint=False)) * C
    dt = spec["check_every"]
    base = 5.0 * np.sin(np.linspace(-np.pi, np.pi, NG, endpoint=False))
    t = t_start
    first = True
    while t < T - 1e-9:
        if first:                       # observation at t_start: nothing has happened yet
            first = False
            if ctl.observe(t, BH, DH, C, base * C, essw, 0.01) is not None:
                break
        t += dt
        n, frac = schedule(t)
        nd = rng.binomial(n, frac)
        for _ in range(nd):
            DH[rng.integers(0, 40)] += 1
            BH[rng.integers(80, 100)] += 1
        for _ in range(n - nd):
            a, b = rng.integers(0, NG, 2)
            DH[a] += 1
            BH[b] += 1
        inc = np.where(C >= 1e5, 1000.0, 1.0)          # poorly supported bins stay below the support count
        C = C + inc
        M = C * (base + G_noise * rng.standard_normal(NG))   # G_t = base + noise_t: unstable when G_noise is large
        ts = ctl.observe(t, BH, DH, C, M, essw, 0.01)
        if ts is not None:
            break
    return ctl


def early_then_noise(t):
    return (60, 0.8) if t <= 3.0 else (25, 0.0)


def test_directed_phase_then_noise_stops_at_twice_t1():
    c = synthetic(early_then_noise)
    assert c.reason == "established"
    assert 3.0 < c.t1 <= 4.25                       # first window that no longer contains directed events
    assert c.t_stop == pytest.approx(2.0 * c.t1)
    assert c.t_decision == pytest.approx(c.t1 + 3 * FC.R1["check_every"])


def test_transient_dip_does_not_stop():
    """A dip of 3 checks (< persist = 4) inside a long directed phase never triggers the irreversible stop."""
    def sched(t):
        if 4.0 < t <= 4.75:
            return (25, 0.0)
        return (60, 0.8) if t <= 9.0 else (25, 0.0)
    c = synthetic(sched, T=14.0)
    assert c.t1 is not None and c.t1 > 9.0


def test_poorly_supported_estimator_never_passes():
    """A flat (non-directed) FR activity with half the bins below the support count never stops (uniform marginal
    does not certify the mean-force estimator)."""
    c = synthetic(lambda t: (25, 0.0), support=False)
    assert c.t_stop is None


def test_unstable_mean_force_blocks_the_stop():
    c = synthetic(lambda t: (25, 0.0), G_noise=50.0)
    assert c.t_stop is None


def test_safety_trigger_stops_immediately():
    c = synthetic(lambda t: (60, 0.8), essw=0.10, T=3.0)
    assert c.reason == "safety" and c.t_stop == pytest.approx(1.0)                # the first check


def test_no_directed_work_at_all_stops_early():
    c = synthetic(lambda t: (25, 0.0))
    assert c.reason == "established" and c.t1 == pytest.approx(2.0) and c.t_stop == pytest.approx(4.0)


def test_state_roundtrip_continues_identically():
    a = FC.OnlineStop(FC.R1, K, 1.0)
    b = None
    rng = np.random.default_rng(3)
    BH = np.zeros(NG); DH = np.zeros(NG); C = np.full(NG, 1e6); M = np.zeros(NG)
    t = 1.0
    for i in range(30):
        t += 0.25
        BH += rng.poisson(0.3, NG); DH += rng.poisson(0.3, NG); C += 100; M += rng.standard_normal(NG)
        ra = a.observe(t, BH, DH, C, M, 1.0, 0.01)
        if i == 12:
            b = FC.OnlineStop.from_state(json.loads(json.dumps(a.state())))
        elif b is not None:
            rb = b.observe(t, BH, DH, C, M, 1.0, 0.01)
            assert ra == rb and a.log[-1] == b.log[-1]


def test_controller_is_reference_free_by_construction():
    """The controller module performs no I/O and imports nothing but math/numpy: it cannot read a reference, a
    trajectory file or another run.  (Docstrings stripped; names checked on the syntax tree.)"""
    import ast
    tree = ast.parse(inspect.getsource(FC))
    imports = {a.name for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
    assert imports <= {"math", "numpy"}, imports
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    for forbidden in ("open", "load", "loadtxt", "fromfile", "glob", "listdir", "F_ref", "G_ref", "reference_for_grid"):
        assert forbidden not in names, forbidden


# ---------------------------------------------------------------------------------------------- engine level
@pytest.fixture(scope="module")
def model():
    return Model("gard_tait")


FORCE = dict(FC.R1, theta=1.0, persist=1, support_count=0.0, stability_tol=1e9, check_every=0.05, W=0.05,
             essw_min=0.0, maxlin_min=1.0)          # decides at the first full window -> exercises the stop path


def _cfg(**k):
    base = dict(N=64, n_steps=3000, save_every=100, warmup_steps=500, fr_start_steps=500, fr_every=10, seed=17,
                fr_rate=20.0, trust_count=200.0, dt=1e-4)
    base.update(k)
    return E.make_cfg(**base)


def _run(c, method, model, chunk=400, ckpt_at=None, tmp=None):
    r = E.Run(c, method, model=model)
    while not r.done:
        nxt = r.k + chunk
        if ckpt_at is not None and r.k < ckpt_at <= nxt:
            r.advance_controlled(ckpt_at)
            p = str(tmp / "ck.npz")
            r.checkpoint(p)
            r = E.Run(c, method, model=model)
            r.restore(p)
            ckpt_at = None
            continue
        r.advance_controlled(nxt)
    return r


KEYS = ("sv_C", "sv_M", "sv_h", "sv_i", "sv_u", "sv_cond", "q", "anc", "BH", "DH")


def test_controlled_stop_equals_the_fixed_stop_and_is_deterministic(model):
    c = _cfg(fr_controller=FORCE)
    a = _run(c, "fr", model)
    b = _run(c, "fr", model, chunk=137)
    assert a.ctl.t_stop is not None and a.ctl.log == b.ctl.log
    for k in KEYS:
        assert np.array_equal(getattr(a, k), getattr(b, k)), k
    stop = int(a.ip[8])
    f = _run(_cfg(fr_stop_steps=stop), "fr", model)      # the validated fixed-stop path
    for k in KEYS:
        assert np.array_equal(getattr(a, k), getattr(f, k)), k
    assert a.ist[E.S_REPL] > 0 and a.ist[E.S_NOPP] == (stop - 1 - c["fr_start_steps"]) // c["fr_every"] + 1


def test_checkpoint_resume_reproduces_every_decision(model, tmp_path):
    c = _cfg(fr_controller=FORCE)
    a = _run(c, "fr", model)
    for at in (700, 1700):                                # before and after the decision
        b = _run(c, "fr", model, ckpt_at=at, tmp=tmp_path)
        assert a.ctl.log == b.ctl.log and a.ctl.t_stop == b.ctl.t_stop
        for k in KEYS + ("ist",):
            assert np.array_equal(getattr(a, k), getattr(b, k)), (k, at)


def test_after_the_controlled_stop_the_run_is_plain_abf(model):
    c = _cfg(fr_controller=FORCE)
    s = _run(c, "fr", model)
    stop = int(s.ip[8])
    s2 = E.Run(c, "fr", model=model)
    s2.advance_controlled(stop)
    b = E.Run(_cfg(), "abf", model=model)
    for k in E.Run._STATE:
        setattr(b, k, np.array(getattr(s2, k)))
    b.k, b.isave = s2.k, s2.isave
    b.g_noise.bit_generator.state = s2.g_noise.bit_generator.state
    fr_state = dict(s2.g_fr.bit_generator.state)
    s2.advance_controlled(c["n_steps"] + 1)
    b.advance(c["n_steps"] + 1)
    for k in KEYS:
        assert np.array_equal(getattr(s2, k), getattr(b, k)), k
    assert s2.g_fr.bit_generator.state == fr_state                 # no FR draw after the stop
    assert np.array_equal(s2.sv_C.sum(1), c["N"] * (s2.save_steps + 1.0))   # constant population, all deposit


def test_controller_is_inert_for_abf_and_absent_by_default(model):
    c = _cfg()
    a = _run(c, "abf", model)
    b = _run(_cfg(fr_controller=FC.R1), "abf", model)
    assert b.ctl is None
    for k in KEYS:
        assert np.array_equal(getattr(a, k), getattr(b, k)), k
    with pytest.raises(ValueError):
        E.Run(_cfg(fr_controller=FC.R1, fr_stop_steps=900), "fr", model=model)


def test_safety_after_the_decision_can_only_bring_the_stop_earlier():
    c = synthetic(early_then_noise, T=4.5)            # decided (t_stop = 2 t1 > decision) but not yet stopped
    assert c.reason == "established" and c.t_stop > c.t_decision
    ts = c.observe(c.t_decision + 0.25, np.zeros(NG), np.zeros(NG), np.ones(NG), np.zeros(NG), 0.05, 0.0)
    assert ts == pytest.approx(c.t_decision + 0.25) and c.reason == "established+safety"
    assert c.observe(c.t_decision + 0.5, np.zeros(NG), np.zeros(NG), np.ones(NG), np.zeros(NG), 1.0, 0.0) == ts


def test_lineage_limit_scales_with_K_and_grid_is_checked():
    assert FC.OnlineStop(FC.R1, 16, 1.0).safety_limit() == pytest.approx(6.0 / 16)
    assert FC.OnlineStop(FC.R1, 1024, 1.0).safety_limit() == pytest.approx(0.10)
    with pytest.raises(ValueError):
        FC.check_grid(FC.R1, 3e-4)                    # 0.25 is not a multiple of 3e-4
    assert FC.check_grid(FC.R1, 1e-4) == 2500


def test_direct_advance_is_refused_with_a_controller(model):
    r = E.Run(_cfg(fr_controller=FORCE), "fr", model=model)
    with pytest.raises(RuntimeError):
        r.advance(100)
