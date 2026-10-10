"""Legibility check of a finished matplotlib figure, run at save time by plot_config.py and plot_synthesis.py.

    res = check_figure(fig, waivers=())        # after the layout is final (call it right after savefig)

Returns a JSON-safe dict that the plot scripts store as the 'legibility' field of the figure's manifest entry:
  status       'pass' | 'fail'
  failures     [{check, where, reason}]          (at most MAX_LISTED; n_failures is the full count)
  waived       [{check, where, reason, why}]     failures matched by a documented, intentional waiver
  notes        informational lines (axes that pass the legend rule through a key on another panel; unused waivers)
  n_axes, n_data_axes, n_texts                 what was inspected
Checks.  A DATA axes is a visible axes, switched on, that is not a colourbar and holds at least one visible line,
collection, image or patch (the axes frame / background is not a patch here).  Checks 1-4 run on data axes;
5-6 on every visible text of the figure.
  1 axis_label_x   a non-empty x label, or a visible shared-x partner that has one (a column sharing x labels the
                   bottom panel only)
  2 axis_label_y   a non-empty y label, or a visible shared-y partner that has one, or -- colour-mapped data only --
                   a labelled colourbar for it
  3 colorbar       colour-mapped data (image, mesh, scatter with a value array) is covered by a colourbar with a
                   non-empty label (the same mappable, the same norm object, or the same cmap and limits)
  4 legend         an axes with more than one labelled artist has a legend of its own, or the figure has a
                   figure-level legend, or every one of its labels is an entry of a legend on another panel of the
                   same figure (one key shared by the panels: recorded in notes, not a failure)
  5 text_overlap   no two visible texts overlap by more than tol_pt (default 0.25 pt: boxes may touch, not
                   intersect -- two separate labels whose boxes meet already read as one paragraph): texts placed in
                   the axes, title, axis labels,
                   tick labels, secondary axes and the legend box of ONE axes ('text_overlap'), and texts of
                   different axes / figure-level texts ('text_overlap_across').  Oriented boxes (separating-axis
                   test), so rotated tick labels whose axis-aligned boxes intersect are not false positives; a text
                   clipped away entirely (outside its clip box) is not drawn and not checked
  6 text_outside   no visible text or legend (box less its border pad) extends beyond the canvas by more than
                   tol_pt (the scripts save without a tight bounding box, so that text is cut off in the PNG / PDF)
  7 legend_over_data  an in-axes legend (box less its border pad) is crossed by a drawn DATA line of its axes
                   (drawn segments, NaN gaps respected) or covers scatter points: the data is hidden or the legend text
                   struck through.  A reference line (axhline / axvline: blended axes/data transform) only counts
                   when the legend is transparent (no frame, or framealpha < 0.75): an opaque legend hides it
A waiver is a dict(check=<name>, where=<substring of 'where' or 'reason'>, why=<reason it is intentional>); a matched
failure moves to 'waived' (status can then pass), a waiver that matches nothing is reported in notes (stale).
"""
from __future__ import annotations

import math

import numpy as np
import matplotlib.text as mtext
from matplotlib.collections import PathCollection
from matplotlib.transforms import BlendedAffine2D, BlendedGenericTransform

MAX_LISTED = 60
CHECKS = ("axis_label_x", "axis_label_y", "colorbar", "legend", "text_overlap", "text_overlap_across", "text_outside",
          "legend_over_data")


# ------------------------------------------------------------------------------------------------ helpers
def _visible(a):
    try:
        al = a.get_alpha()
        return bool(a.get_visible()) and (al is None or al > 0)
    except Exception:  # noqa: BLE001
        return False


def is_colorbar_axes(ax):
    return getattr(ax, "_colorbar", None) is not None


def _snip(s, n=40):
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[:n - 1] + "…"


def axes_name(ax, i):
    t = ""
    for loc in ("left", "center", "right"):
        try:
            t = ax.get_title(loc=loc)
        except Exception:  # noqa: BLE001
            t = ""
        if t.strip():
            break
    if not t.strip():
        t = ax.get_ylabel() or ax.get_xlabel() or ""
    return f"axes {i}" + (f" [{_snip(t)}]" if t.strip() else "")


def data_artists(ax):
    out = [ln for ln in ax.lines if _visible(ln) and len(np.atleast_1d(ln.get_xdata(orig=False))) > 0]
    for c in ax.collections:
        if not _visible(c):
            continue
        if isinstance(c, PathCollection):            # scatter: data only when it has points
            ok = len(c.get_offsets()) > 0
        else:
            try:
                ok = len(c.get_paths()) > 0
            except Exception:  # noqa: BLE001
                ok = True
        if ok:
            out.append(c)
    out += [im for im in ax.images if _visible(im)]
    out += [p for p in ax.patches if _visible(p)]
    return out


def mappables(ax):
    out = [im for im in ax.images if _visible(im)]
    for c in ax.collections:
        if not _visible(c):
            continue
        try:
            arr = c.get_array()
        except Exception:  # noqa: BLE001
            arr = None
        if arr is not None and np.size(arr) > 0:
            out.append(c)
    return out


def _cb_label(cb):
    ax = cb.ax
    return (ax.get_ylabel() if getattr(cb, "orientation", "vertical") == "vertical" else ax.get_xlabel()).strip()


def _cb_covers(cb, m):
    cm = cb.mappable
    if cm is m:
        return True
    try:
        if cm.norm is m.norm:
            return True
        return (cm.get_cmap().name == m.get_cmap().name and float(cm.norm.vmin) == float(m.norm.vmin)
                and float(cm.norm.vmax) == float(m.norm.vmax))
    except Exception:  # noqa: BLE001
        return False


def _shared_label(ax, which):
    grp = ax.get_shared_x_axes() if which == "x" else ax.get_shared_y_axes()
    for p in grp.get_siblings(ax):
        if p is ax or not p.get_visible():
            continue
        if (p.get_xlabel() if which == "x" else p.get_ylabel()).strip():
            return p
    return None


def _label_ok(lab):
    return bool(lab) and not str(lab).startswith("_")


def labelled_artists(ax):
    out = []
    for a in list(ax.lines) + list(ax.collections) + list(ax.patches) + list(ax.images):
        if _visible(a) and _label_ok(a.get_label()):
            out.append(a)
    out += [c for c in ax.containers if _label_ok(c.get_label())]
    return out


def legend_entries(leg):
    return {t.get_text() for t in leg.get_texts()}


# ------------------------------------------------------------------------------------------------ text geometry
def _clip_hides(t, bb):
    """True when the text is clipped away entirely (its box misses its clip box / clip path)."""
    if not t.get_clip_on():
        return False
    regions = []
    if t.get_clip_box() is not None:
        regions.append(t.get_clip_box())
    cp = t.get_clip_path()
    if cp is not None:
        try:
            regions.append(cp.get_fully_transformed_path().get_extents())
        except Exception:  # noqa: BLE001
            pass
    return any(not bb.overlaps(r) for r in regions)


def text_item(t, renderer, kind, group, where):
    """dict(kind, group, where, text, aabb=(x0, y0, x1, y1), obb=(cx, cy, hw, hh, theta)) or None if not drawn."""
    if not isinstance(t, mtext.Text) or not _visible(t):
        return None
    s = t.get_text()
    if not s or not s.strip():
        return None
    if isinstance(t, mtext.Annotation):
        try:
            t.update_positions(renderer)
            if not t._check_xy(renderer):
                return None
        except Exception:  # noqa: BLE001
            pass
    bb = mtext.Text.get_window_extent(t, renderer)
    if not (bb.width > 0 and bb.height > 0 and np.all(np.isfinite(bb.extents))):
        return None
    if _clip_hides(t, bb):
        return None
    rot = float(t.get_rotation()) % 180.0
    if min(rot, 180.0 - rot) < 1e-6:
        w, h, th = bb.width, bb.height, 0.0
    else:                                        # unrotated size of the same (multi-line) text block
        r0 = t.get_rotation()
        t.set_rotation(0)
        b0 = mtext.Text.get_window_extent(t, renderer)
        t.set_rotation(r0)
        w, h, th = b0.width, b0.height, math.radians(rot)
    cx, cy = 0.5 * (bb.x0 + bb.x1), 0.5 * (bb.y0 + bb.y1)    # the AABB of a rotated box is centred on it
    return dict(kind=kind, group=group, where=where, text=_snip(s, 32), aabb=tuple(bb.extents),
                obb=(cx, cy, 0.5 * w, 0.5 * h, th))


def box_item(bb, kind, group, where, text, pad=0.0):
    x0, y0, x1, y1 = bb.x0 + pad, bb.y0 + pad, bb.x1 - pad, bb.y1 - pad
    if not (x1 > x0 and y1 > y0):
        return None
    return dict(kind=kind, group=group, where=where, text=text, aabb=(x0, y0, x1, y1),
                obb=(0.5 * (x0 + x1), 0.5 * (y0 + y1), 0.5 * (x1 - x0), 0.5 * (y1 - y0), 0.0))


def _corners(o, s):
    cx, cy, hw, hh, th = o
    hw, hh = hw - s, hh - s
    if hw <= 0 or hh <= 0:
        return None, None
    c, si = math.cos(th), math.sin(th)
    u, v = np.array([c, si]) * hw, np.array([-si, c]) * hh
    ctr = np.array([cx, cy])
    return np.array([ctr + u + v, ctr + u - v, ctr - u - v, ctr - u + v]), (np.array([c, si]), np.array([-si, c]))


def obb_overlap(o1, o2, s):
    """Oriented boxes o1, o2 (each shrunk by s pixels on every side) intersect (separating-axis test)."""
    p1, a1 = _corners(o1, s)
    p2, a2 = _corners(o2, s)
    if p1 is None or p2 is None:
        return False
    for ax in (*a1, *a2):
        q1, q2 = p1 @ ax, p2 @ ax
        if q1.max() <= q2.min() or q2.max() <= q1.min():
            return False
    return True


def axis_texts(ax, renderer, group, name):
    items = []
    for axis, nm in ((ax.xaxis, "x"), (ax.yaxis, "y")):
        if not axis.get_visible():
            continue
        try:
            ticks = axis._update_ticks()            # the ticks that are drawn (inside the view interval)
        except Exception:  # noqa: BLE001
            ticks = axis.get_major_ticks()
        for tk in ticks:
            for lab in (tk.label1, tk.label2):
                it = text_item(lab, renderer, f"{nm} tick label", group, name)
                if it:
                    items.append(it)
        it = text_item(axis.label, renderer, f"{nm} label", group, name)
        if it:
            items.append(it)
    return items


def collect_texts(fig, renderer):
    items = []
    dpi = fig.dpi
    for i, ax in enumerate(fig.axes):
        if not ax.get_visible():
            continue
        group, name = f"axes {i}", axes_name(ax, i)
        for t in ax.texts:
            it = text_item(t, renderer, "text", group, name)
            if it:
                items.append(it)
        for t in (ax.title, getattr(ax, "_left_title", None), getattr(ax, "_right_title", None)):
            if t is not None:
                it = text_item(t, renderer, "title", group, name)
                if it:
                    items.append(it)
        if ax.axison:
            items += axis_texts(ax, renderer, group, name)
        for ch in getattr(ax, "child_axes", []):    # secondary axes belong to their parent
            if ch.get_visible():
                items += axis_texts(ch, renderer, group, name + " (secondary axis)")
                for t in ch.texts:
                    it = text_item(t, renderer, "text", group, name + " (secondary axis)")
                    if it:
                        items.append(it)
        leg = ax.get_legend()
        if leg is not None and _visible(leg):
            pad = leg.borderpad * leg._fontsize * dpi / 72.0
            it = box_item(leg.get_window_extent(renderer), "legend", group, name, "legend", pad=pad)
            if it:
                items.append(it)
    for t in fig.texts:
        it = text_item(t, renderer, "figure text", "figure", "figure")
        if it:
            items.append(it)
    for k, leg in enumerate(fig.legends):
        if _visible(leg):
            pad = leg.borderpad * leg._fontsize * dpi / 72.0
            it = box_item(leg.get_window_extent(renderer), "figure legend", f"figure legend {k}", "figure",
                          "figure legend", pad=pad)
            if it:
                items.append(it)
    return items


def segments_hit_rect(xy, rect):
    """Number of drawn segments of the polyline xy (n, 2; NaN = gap) that intersect rect (x0, y0, x1, y1)
    (Liang-Barsky clipping, vectorised)."""
    xy = np.asarray(xy, float)
    if xy.ndim != 2 or len(xy) == 0:
        return 0
    x0, y0, x1, y1 = rect
    if len(xy) == 1:
        p = xy[0]
        return int(np.all(np.isfinite(p)) and x0 <= p[0] <= x1 and y0 <= p[1] <= y1)
    P, Q = xy[:-1], xy[1:]
    ok = np.all(np.isfinite(P), axis=1) & np.all(np.isfinite(Q), axis=1)
    P, Q = P[ok], Q[ok]
    if len(P) == 0:
        return 0
    d = Q - P
    t0, t1 = np.zeros(len(P)), np.ones(len(P))
    alive = np.ones(len(P), bool)
    for pk, qk in ((-d[:, 0], P[:, 0] - x0), (d[:, 0], x1 - P[:, 0]), (-d[:, 1], P[:, 1] - y0), (d[:, 1], y1 - P[:, 1])):
        par = pk == 0
        alive &= ~(par & (qk < 0))
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(par, 0.0, qk / np.where(par, 1.0, pk))
        t0 = np.where(~par & (pk < 0), np.maximum(t0, t), t0)
        t1 = np.where(~par & (pk > 0), np.minimum(t1, t), t1)
    return int(np.sum(alive & (t0 <= t1)))


def is_reference_line(ln):
    """axhline / axvline (and any line drawn in a blended axes / data transform)."""
    return isinstance(ln.get_transform(), (BlendedGenericTransform, BlendedAffine2D))


def legend_opaque(leg):
    fr = leg.get_frame()
    try:
        fa = fr.get_facecolor()[3]                  # RGBA already carries the patch alpha (framealpha)
    except Exception:  # noqa: BLE001
        fa = 0.0
    return bool(leg.get_frame_on()) and fa >= 0.75


def legend_over_data(ax, leg, renderer, dpi):
    """[(artist description, count)] of the lines crossing / scatter points inside the in-axes legend's box."""
    opaque = legend_opaque(leg)
    pad = leg.borderpad * leg._fontsize * dpi / 72.0
    bb = leg.get_window_extent(renderer)
    rect = (bb.x0 + pad, bb.y0 + pad, bb.x1 - pad, bb.y1 - pad)
    if not (rect[2] > rect[0] and rect[3] > rect[1]):
        return []
    # data clipped to the axes is not drawn outside it: a key placed outside the axes only meets unclipped artists
    ab = ax.bbox
    inner = (max(rect[0], ab.x0), max(rect[1], ab.y0), min(rect[2], ab.x1), min(rect[3], ab.y1))
    inner = inner if (inner[2] > inner[0] and inner[3] > inner[1]) else None
    hits = []
    for k, ln in enumerate(ax.lines):
        if not _visible(ln) or ln.get_linestyle() in ("None", "none", "", " ") and ln.get_marker() in (None, "None", "none", ""):
            continue
        ref = is_reference_line(ln)
        if ref and opaque:
            continue
        r_ = inner if ln.get_clip_on() else rect
        if r_ is None:
            continue
        try:
            tp = ln.get_transform().transform(ln.get_path().vertices)
        except Exception:  # noqa: BLE001
            continue
        if ln.get_linestyle() in ("None", "none", "", " "):        # markers only: points inside the box
            n = int(np.sum(np.all(np.isfinite(tp), 1) & (tp[:, 0] >= r_[0]) & (tp[:, 0] <= r_[2])
                           & (tp[:, 1] >= r_[1]) & (tp[:, 1] <= r_[3])))
        else:
            n = segments_hit_rect(tp, r_)
        if n:
            lab = ln.get_label()
            hits.append(((f"line '{lab}'" if _label_ok(lab) else f"line #{k}")
                         + (" (reference line through a transparent legend)" if ref else ""), n))
    for k, c in enumerate(ax.collections):
        if not (_visible(c) and isinstance(c, PathCollection)) or len(c.get_offsets()) == 0:
            continue
        r_ = inner if c.get_clip_on() else rect
        if r_ is None:
            continue
        try:
            tp = c.get_offset_transform().transform(c.get_offsets())
        except Exception:  # noqa: BLE001
            continue
        n = int(np.sum(np.all(np.isfinite(tp), 1) & (tp[:, 0] >= r_[0]) & (tp[:, 0] <= r_[2])
                       & (tp[:, 1] >= r_[1]) & (tp[:, 1] <= r_[3])))
        if n:
            lab = c.get_label()
            hits.append((f"scatter '{lab}'" if _label_ok(lab) else f"scatter #{k}", n))
    return hits


# ------------------------------------------------------------------------------------------------ the check
def check_figure(fig, waivers=(), tol_pt=0.25, draw=True, allow_shared_key=True):
    """Run checks 1-7 on a finished figure (see the module docstring); draw=True renders the canvas once first so
    that tick labels, constrained layout and annotation offsets are those of the saved file.  allow_shared_key=False
    applies the strict legend rule (a legend on another panel does not count)."""
    if draw:
        fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fails, notes = [], []

    def fail(check, where, reason):
        fails.append(dict(check=check, where=where, reason=reason))

    colorbars = [a._colorbar for a in fig.axes if is_colorbar_axes(a)]
    legends = [(i, a.get_legend()) for i, a in enumerate(fig.axes) if a.get_legend() is not None]
    n_data = 0
    for i, ax in enumerate(fig.axes):
        if not ax.get_visible() or not ax.axison or is_colorbar_axes(ax):
            continue
        if not data_artists(ax):
            continue
        n_data += 1
        name = axes_name(ax, i)
        maps = mappables(ax)
        cb_lab = []
        for m in maps:
            cov = [cb for cb in colorbars if _cb_covers(cb, m)]
            lab = [_cb_label(cb) for cb in cov if _cb_label(cb)]
            cb_lab += lab
            if not lab:
                fail("colorbar", name, f"colour-mapped {type(m).__name__} without a labelled colourbar"
                     + (" (its colourbar has an empty label)" if cov else ""))
        if not ax.get_xlabel().strip() and _shared_label(ax, "x") is None:
            fail("axis_label_x", name, "empty x label and no shared-x partner with one")
        if not ax.get_ylabel().strip() and _shared_label(ax, "y") is None and not cb_lab:
            fail("axis_label_y", name, "empty y label, no shared-y partner with one"
                 + (", no labelled colourbar" if maps else ""))
        lab_art = labelled_artists(ax)
        if len(lab_art) > 1 and ax.get_legend() is None and not fig.legends and not allow_shared_key:
            fail("legend", name, f"{len(lab_art)} labelled artists and no legend on this axes or the figure")
        elif len(lab_art) > 1 and ax.get_legend() is None and not fig.legends:
            labs = sorted({str(a.get_label()) for a in lab_art})
            other = [(j, lg) for j, lg in legends if j != i]
            covered = set().union(*[legend_entries(lg) for _, lg in other]) if other else set()
            missing = [lb for lb in labs if lb not in covered]
            if missing:
                fail("legend", name, f"{len(lab_art)} labelled artists, no legend on this axes or the figure; labels "
                     f"not in any legend of the figure: {missing[:6]}")
            else:
                src = [j for j, lg in other if set(labs) & legend_entries(lg)]
                notes.append(f"legend: {name} has no legend of its own; its labels {labs[:6]} are entries of the "
                             f"legend on axes {src}")
    # texts: overlaps and canvas clipping
    items = collect_texts(fig, renderer)
    s = 0.5 * tol_pt * fig.dpi / 72.0               # shrink each box by half the tolerance on every side
    W, H = fig.bbox.width, fig.bbox.height
    tol_px = tol_pt * fig.dpi / 72.0
    for it in items:
        x0, y0, x1, y1 = it["aabb"]
        out = max(-x0, -y0, x1 - W, y1 - H)
        if out > tol_px:
            fail("text_outside", it["where"], f"{it['kind']} '{it['text']}' extends {out * 72.0 / fig.dpi:.1f} pt "
                 f"beyond the canvas (cut off in the saved file)")
    if len(items) > 1:
        A = np.array([it["aabb"] for it in items], float)
        cand = ((A[:, None, 0] < A[None, :, 2] - 2 * s) & (A[None, :, 0] < A[:, None, 2] - 2 * s)
                & (A[:, None, 1] < A[None, :, 3] - 2 * s) & (A[None, :, 1] < A[:, None, 3] - 2 * s))
        iu, ju = np.nonzero(np.triu(cand, 1))
        for a, b in zip(iu, ju):
            p, q = items[a], items[b]
            if not obb_overlap(p["obb"], q["obb"], s):
                continue
            same = p["group"] == q["group"]
            where = p["where"] if same else f"{p['where']} / {q['where']}"
            fail("text_overlap" if same else "text_overlap_across", where,
                 f"{p['kind']} '{p['text']}' overlaps {q['kind']} '{q['text']}'")
    # legends struck through by data
    for i, ax in enumerate(fig.axes):
        leg = ax.get_legend()
        if leg is None or not _visible(leg) or not ax.get_visible():
            continue
        for what, n in legend_over_data(ax, leg, renderer, fig.dpi):
            fail("legend_over_data", axes_name(ax, i), f"the legend box is crossed by {what} ({n} drawn segment(s) / "
                 "point(s) inside it)")
    # waivers
    waived, used = [], set()
    kept = []
    for f in fails:
        hit = None
        for k, w in enumerate(waivers or ()):
            if w.get("check") == f["check"] and str(w.get("where", "")) in (f["where"] + " " + f["reason"]):
                hit = k
                break
        if hit is None:
            kept.append(f)
        else:
            used.add(hit)
            waived.append(dict(f, why=waivers[hit].get("why", "")))
    for k, w in enumerate(waivers or ()):
        if k not in used:
            notes.append(f"unused waiver (nothing to waive): {w}")
    return dict(status="pass" if not kept else "fail", checks=list(CHECKS), tol_pt=tol_pt,
                n_axes=len(fig.axes), n_data_axes=n_data, n_texts=len(items),
                n_failures=len(kept), failures=kept[:MAX_LISTED], n_waived=len(waived), waived=waived[:MAX_LISTED],
                notes=notes[:MAX_LISTED])


def check_figure_safe(fig, **kw):
    """check_figure that never raises (the plot scripts call it after the figure is saved: a checker error must not
    abort the remaining figures and the manifest).  An error is recorded as status 'error' (not a pass)."""
    try:
        return check_figure(fig, **kw)
    except Exception as e:  # noqa: BLE001
        return dict(status="error", checks=list(CHECKS), n_failures=1, n_waived=0, waived=[], notes=[],
                    failures=[dict(check="checker_error", where="figure", reason=f"{type(e).__name__}: {e}")])


def summarize(results):
    """{figure id: result} -> manifest-level summary (a result without a count, e.g. 'not checked', fails with None)."""
    bad = {k: v.get("n_failures") for k, v in results.items() if v.get("status") != "pass"}
    return dict(n_figures=len(results), n_pass=len(results) - len(bad), n_fail=len(bad), failing=bad,
                n_waived=sum(int(v.get("n_waived", 0)) for v in results.values()))


def one_line(fid, res):
    f = res.get("failures") or []
    first = f"{f[0]['check']}: {f[0]['where']}: {f[0]['reason']}" if f else ""
    return (f"LEGIBILITY {str(res.get('status')).upper()} {fid}: {res.get('n_failures')} failure(s)"
            + (f"; first {first}" if f else ""))
