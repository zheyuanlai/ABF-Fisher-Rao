"""Shared curve-end label placement for the presentation figures.

Placing the two labels by shifting them a fixed number of DECADES fails whenever the arms
nearly coincide: on a panel whose whole y-range is 0.06 decades (alanine's mean force) a
0.08-decade shift puts both labels outside the axes.  Separation is a typographic quantity,
so it is applied in display points, after the layout is final.
"""
from __future__ import annotations


def draw_end_labels(fig, entries, min_sep_pt=17.0, sep_pt=9.0, dx_pt=6.0):
    """entries: list of (ax, x_end, [(y, colour, text), ...]) -- annotate each curve end.

    Labels that would overlap are pushed apart symmetrically in points, preserving which
    curve sits above the other.
    """
    fig.canvas.draw()
    for ax, x_end, items in entries:
        ys_disp = [ax.transData.transform((x_end, y))[1] for y, _, _ in items]
        offs = [0.0] * len(items)
        if len(items) == 2 and abs(ys_disp[0] - ys_disp[1]) < min_sep_pt:
            hi = 0 if ys_disp[0] >= ys_disp[1] else 1
            offs[hi], offs[1 - hi] = sep_pt, -sep_pt
        for (y, colour, text), off in zip(items, offs):
            ax.annotate(text, xy=(x_end, y), xytext=(dx_pt, off), textcoords="offset points",
                        va="center", ha="left", color=colour, fontsize=13, fontweight="bold",
                        annotation_clip=False)
