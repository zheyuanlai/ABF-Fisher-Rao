"""Tests of scripts/equal_budget/fig_legibility.py (the save-time legibility check of plot_config.py / plot_synthesis.py).

Every check must be able to FIRE on a figure that has the defect and stay silent on the corrected figure (a gate that
cannot fire reads as reassurance), including the defect that motivated it: the two S8 bootstrap-frequency labels at
the same N printed on top of each other (plot_synthesis.draw_best_freq before the fix).

  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 python -m pytest -p no:cacheprovider <this file> -q
"""
import os
import sys

import numpy as np
import pytest

REPO = "/home/zheyuanlai/ABF-Fisher-Rao"
sys.path.insert(0, os.path.join(REPO, "scripts", "equal_budget"))
sys.path.insert(0, os.path.join(REPO, "src"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import fig_legibility as LG  # noqa: E402


@pytest.fixture(autouse=True)
def _close():
    with matplotlib.rc_context({"legend.frameon": False}):
        yield
    plt.close("all")


def sub(*a, **k):
    return plt.subplots(*a, layout="constrained", **k)      # default margins cut small figures' labels off


def checks(res):
    return sorted({f["check"] for f in res["failures"]})


def clean_axes(ax, label=True):
    ax.plot([1, 2, 3], [1, 4, 9], label="a" if label else None)
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def test_clean_figure_passes():
    fig, ax = sub(figsize=(4, 3))
    clean_axes(ax)
    ax.plot([1, 2, 3], [2, 3, 4], label="b")
    ax.legend(loc="upper left")
    ax.set_title("t", loc="left")
    res = LG.check_figure(fig)
    assert res["status"] == "pass", res["failures"]
    assert res["n_data_axes"] == 1 and res["n_texts"] > 5


def test_missing_axis_labels_and_shared_partner():
    fig, (a, b) = sub(2, 1, sharex=True, figsize=(4, 5))
    for ax in (a, b):
        ax.plot([1, 2], [1, 2])
        ax.set_ylabel("y")
    b.set_xlabel("x")                                 # top panel takes its x label from the shared bottom panel
    assert LG.check_figure(fig)["status"] == "pass"
    fig2, (c, d) = sub(1, 2, figsize=(6, 3))  # not shared: both need their own labels
    for ax in (c, d):
        ax.plot([1, 2], [1, 2])
    c.set_xlabel("x")
    c.set_ylabel("y")
    res = LG.check_figure(fig2)
    assert checks(res) == ["axis_label_x", "axis_label_y"] and all("axes 1" in f["where"] for f in res["failures"])


def test_legend_rule_strict_shared_key_and_figure_legend():
    fig, (a, b) = sub(1, 2, figsize=(6, 3))
    for ax in (a, b):
        clean_axes(ax)
        ax.plot([1, 2, 3], [3, 2, 1], label="b")
    res = LG.check_figure(fig)
    assert checks(res) == ["legend"] and len(res["failures"]) == 2    # nobody explains 'a' / 'b'
    a.legend(loc="upper left")
    res = LG.check_figure(fig)                        # b's labels are entries of a's legend: a shared key
    assert res["status"] == "pass", res["failures"]
    assert any("legend on axes [0]" in n for n in res["notes"])
    assert checks(LG.check_figure(fig, allow_shared_key=False)) == ["legend"]
    a.get_legend().remove()
    fig.legend(loc="outside upper center", ncol=2)
    assert LG.check_figure(fig, allow_shared_key=False)["status"] == "pass"


def test_text_overlap_fires_and_rotated_ticks_do_not():
    fig, ax = sub(figsize=(4, 3))
    clean_axes(ax, label=False)
    ax.text(2, 5, "0.40", fontsize=8)
    ax.text(2.02, 5.05, "0.39", fontsize=8)
    res = LG.check_figure(fig)
    assert checks(res) == ["text_overlap"] and "'0.40'" in res["failures"][0]["reason"]
    # 45-degree tick labels whose axis-aligned boxes intersect but whose glyph boxes do not
    fig2, ax2 = plt.subplots(figsize=(4, 3))
    clean_axes(ax2, label=False)
    xs = np.arange(1, 9)
    ax2.set_xticks(xs, [f"label{k:04d}" for k in xs])
    for t in ax2.get_xticklabels():
        t.set_rotation(45)
        t.set_ha("right")
        t.set_rotation_mode("anchor")
    ax2.set_xlim(0.5, 8.5)
    fig2.subplots_adjust(bottom=0.35, left=0.2)
    r = fig2.canvas.get_renderer()
    fig2.canvas.draw()
    bbs = [t.get_window_extent(r) for t in ax2.get_xticklabels()]
    assert any(bbs[i].overlaps(bbs[i + 1]) for i in range(len(bbs) - 1))      # the AABBs DO intersect
    assert LG.check_figure(fig2)["status"] == "pass"
    # the same labels horizontal are crowded: a real overlap
    for t in ax2.get_xticklabels():
        t.set_rotation(0)
        t.set_ha("center")
    assert "text_overlap" in checks(LG.check_figure(fig2))


def test_text_outside_canvas():
    fig, ax = plt.subplots(figsize=(4, 3))            # fixed margins (constrained layout would make room)
    fig.subplots_adjust(left=0.2, bottom=0.2)
    clean_axes(ax, label=False)
    ax.text(1.02, 0.5, "a long label right of the axes", transform=ax.transAxes, clip_on=False)
    res = LG.check_figure(fig)
    assert checks(res) == ["text_outside"]


def test_colorbar_label():
    fig, ax = sub(figsize=(4, 3))
    m = ax.pcolormesh(np.random.default_rng(0).random((5, 6)))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    assert checks(LG.check_figure(fig)) == ["colorbar"]
    cb = fig.colorbar(m, ax=ax)
    assert checks(LG.check_figure(fig)) == ["colorbar"]           # a colourbar without a label is not a key
    cb.set_label("density")
    assert LG.check_figure(fig)["status"] == "pass"


def test_legend_over_data_and_reference_lines():
    x = np.linspace(0, 1, 200)
    fig, ax = sub(figsize=(4, 3))
    clean_axes(ax, label=False)
    ax.plot(x * 3, 9 * x, label="rising")
    ax.plot(x * 3, 9 - 9 * x, label="falling")
    ax.legend(loc="center")                           # the two curves cross the centre
    res = LG.check_figure(fig)
    assert checks(res) == ["legend_over_data"]
    fig2, ax2 = sub(figsize=(4, 3))
    clean_axes(ax2, label=False)
    ax2.plot(x + 1, 1 + 0 * x, label="flat")
    ax2.plot(x + 1, 1.5 + 0 * x, label="flat 2")
    ax2.axhline(9.0, color="0.5")                     # reference line through the upper-left corner
    ax2.legend(loc="upper left", frameon=False)
    ax2.set_ylim(0, 10)
    res = LG.check_figure(fig2)
    assert checks(res) == ["legend_over_data"] and "reference line" in res["failures"][0]["reason"]
    ax2.legend(loc="upper left", frameon=True, framealpha=0.85, facecolor="white", edgecolor="none")
    assert LG.check_figure(fig2)["status"] == "pass"  # an opaque legend hides a reference line


def test_waivers_and_unused_waiver():
    fig, (c, d) = sub(1, 2, figsize=(6, 3))
    for ax in (c, d):
        ax.plot([1, 2], [1, 2])
        ax.set_xlabel("x")
    c.set_ylabel("y")
    d.set_title("row 2 panel", loc="left")
    w = [dict(check="axis_label_y", where="[row 2 panel]", why="test"), dict(check="legend", where="nothing", why="x")]
    res = LG.check_figure(fig, waivers=w)
    assert res["status"] == "pass" and res["n_waived"] == 1 and res["waived"][0]["why"] == "test"
    assert any("unused waiver" in n for n in res["notes"])


def test_segments_hit_rect():
    r = (0, 0, 1, 1)
    assert LG.segments_hit_rect([[-1, 0.5], [2, 0.5]], r) == 1                  # crosses
    assert LG.segments_hit_rect([[-1, 2], [2, 2]], r) == 0                      # passes above
    assert LG.segments_hit_rect([[-1, 0.5], [np.nan, np.nan], [2, 0.5]], r) == 0  # gap: nothing drawn
    assert LG.segments_hit_rect([[-1, -1], [2, 2]], r) == 1                     # diagonal through
    assert LG.segments_hit_rect([[1.5, -1], [3, 0.5]], r) == 0                  # misses the corner


class _StubD:
    """What draw_best_freq needs from SysData."""
    Ns = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    T = {n: 1.0 for n in Ns}

    def status_text(self, N, methods=("abf", "fr")):
        return None, True


def _best(fa, ff):
    return {"abf": {"boot_best_N_freq": {str(k): v for k, v in fa.items()}, "boot_frac_censored": 0},
            "fr": {"boot_best_N_freq": {str(k): v for k, v in ff.items()}, "boot_frac_censored": 0}}


def _s8_axes():
    import plot_synthesis as PS
    PS.setup_style()
    fig = plt.figure(figsize=(3.5 + 1.0, 2.0))
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.2, bottom=0.3)
    return PS, fig, ax


def test_s8_value_labels_no_longer_overlap():
    """gateway S8 Ibar_F: ABF 0.40 and FR 0.39 modal at N = 16 printed on top of each other before the fix."""
    PS, fig, ax = _s8_axes()
    b = _best({1: 0.17, 2: 0.17, 4: 0.13, 8: 0.12, 16: 0.40}, {2: 0.11, 4: 0.08, 8: 0.18, 16: 0.39, 32: 0.22})
    PS.draw_best_freq(ax, _StubD(), b)
    res = LG.check_figure(fig)
    assert res["status"] == "pass", res["failures"]
    labs = [t for t in ax.texts if t.get_text() in ("0.40", "0.39")]
    assert len(labs) == 2 and {matplotlib.colors.to_hex(t.get_color()) for t in labs} == {PS.C_ABF, PS.C_FR}
    # the pre-fix drawing (both labels at their own bar top, same offset, centred on bars 0.22 octaves apart) fires
    PS2, fig2, ax2 = _s8_axes()
    PS2.draw_best_freq(ax2, _StubD(), b)
    for t in list(ax2.texts):
        if t.get_text() in ("0.40", "0.39"):
            t.remove()
    for m, f in (("abf", 0.40), ("fr", 0.39)):
        ax2.annotate(f"{f:.2f}", xy=(16 * 2 ** (-0.11 if m == "abf" else 0.11), f), xytext=(0, 1.5),
                     textcoords="offset points", ha="center", va="bottom", fontsize=5.6)
    res2 = LG.check_figure(fig2)
    assert checks(res2) == ["text_overlap"] and "'0.40'" in res2["failures"][0]["reason"]


def test_s8_label_clears_taller_neighbour_bar():
    """A label of the shorter bar sits above the taller bar at the same N (it used to print over the blue bar)."""
    PS, fig, ax = _s8_axes()
    PS.draw_best_freq(ax, _StubD(), _best({16: 0.65, 1: 0.22}, {16: 0.48, 4: 0.22}))
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    t65 = next(t for t in ax.texts if t.get_text() == "0.65")
    t48 = next(t for t in ax.texts if t.get_text() == "0.48")
    top65 = ax.transData.transform((16, 0.65))[1]
    assert t65.get_window_extent(r).y0 >= top65 - 0.5 and t48.get_window_extent(r).y0 >= t65.get_window_extent(r).y1 - 0.5


def test_legend_outside_axes_ignores_clipped_data():
    """Data clipped to the axes is not drawn under a key placed below the axes (log axis, values beyond the limits)."""
    x = np.logspace(-3, 0, 100)
    fig, ax = sub(figsize=(4, 3.5))
    ax.plot(x, x, label="rising")
    ax.plot(x, 5e-5 * np.ones_like(x), label="below the axis")      # clipped away (it runs under the key)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), frameon=False)
    assert LG.check_figure(fig)["status"] == "pass"
    ax.get_lines()[1].set_clip_on(False)                              # now it IS drawn across the key
    assert checks(LG.check_figure(fig)) == ["legend_over_data"]


def test_s8_labels_at_adjacent_N_with_censored_column():
    """ABF modal at N, FR modal at N / 2 (bars 0.78 octave apart) at the same height, with the 'all cens.' column
    widening the axis: the two labels printed as one ('0.400.40') when only same-N labels were stacked (review)."""
    PS, fig, ax = _s8_axes()
    fig.set_size_inches(11.4, 3.0)
    ax.set_position([0.075, 0.25, 0.25, 0.6])          # the real S8 bottom-panel width (~2.85 in)
    b = _best({16: 0.40, 8: 0.30}, {8: 0.40, 16: 0.30})
    b["abf"]["boot_frac_censored"] = b["fr"]["boot_frac_censored"] = 0.1
    PS.draw_best_freq(ax, _StubD(), b)
    res = LG.check_figure(fig)
    assert res["status"] == "pass", res["failures"]
    assert sorted(t.get_text() for t in ax.texts if t.get_text() == "0.40") == ["0.40", "0.40"]


def test_safe_check_and_summary_of_unchecked_entries():
    class Broken:
        canvas = None
    res = LG.check_figure_safe(Broken())
    assert res["status"] == "error" and res["failures"][0]["check"] == "checker_error"
    s = LG.summarize({"a": res, "b": dict(status="not checked"), "c": dict(status="pass", n_failures=0)})
    assert s["n_fail"] == 2 and s["failing"] == {"a": 1, "b": None} and s["n_pass"] == 1
    assert "ERROR" in LG.one_line("a", res) and "NOT CHECKED" in LG.one_line("b", dict(status="not checked"))
