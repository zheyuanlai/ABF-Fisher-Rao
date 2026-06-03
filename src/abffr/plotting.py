"""Matplotlib plotting helpers (no seaborn) for the ABF--FR study.

Scripts compose figures from these low-level helpers so that styling stays
consistent across the reference, tuning and eval figures.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # headless / unattended-safe backend
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Stable colours per method/target so figures are comparable across scripts.
METHOD_COLORS = {
    "abf_only": "#222222",
    "abf_fr_estimated": "#1f77b4",
    "abf_fr_uniform": "#ff7f0e",
    "abf_fr_oracle": "#2ca02c",
    "abf_fr_self": "#9467bd",
}
TARGET_COLORS = {
    "none": "#222222",
    "estimated": "#1f77b4",
    "uniform": "#ff7f0e",
    "oracle": "#2ca02c",
    "self": "#9467bd",
}
METHOD_LABELS = {
    "abf_only": "ABF only",
    "abf_fr_estimated": "ABF+FR (estimated)",
    "abf_fr_uniform": "ABF+FR (uniform)",
    "abf_fr_oracle": "ABF+FR (oracle)",
    "abf_fr_self": "ABF+FR (self)",
}


def set_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "lines.linewidth": 1.8,
    })


def save_fig(fig, path: str) -> str:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def method_color(method: str) -> str:
    return METHOD_COLORS.get(method, "#666666")


def method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


# --------------------------------------------------------------------------- #
# Reference geometry
# --------------------------------------------------------------------------- #
def potential_contour(ax, x_grid, y_grid, V_grid, title=r"Potential $V(x,y)$"):
    cf = ax.contourf(x_grid, y_grid, V_grid, levels=40, cmap="RdYlBu_r")
    ax.contour(x_grid, y_grid, V_grid, levels=12, colors="k",
               linewidths=0.4, alpha=0.4)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_title(title)
    ax.grid(False)
    return cf


def density_contour(ax, x_grid, y_grid, rho_grid,
                    title=r"Boltzmann density $\propto e^{-\beta V}$"):
    cf = ax.contourf(x_grid, y_grid, rho_grid, levels=40, cmap="viridis")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_title(title)
    ax.grid(False)
    return cf


def reference_profile(ax, x, y, ylabel, title, color="k"):
    ax.plot(x, y, color=color, lw=2)
    ax.axhline(0.0, color="gray", lw=0.7, ls="--")
    ax.set_xlabel("x"); ax.set_ylabel(ylabel); ax.set_title(title)
    return ax


# --------------------------------------------------------------------------- #
# Study plots
# --------------------------------------------------------------------------- #
def time_curve(ax, t, y, label, color, ls="-", logy=False, alpha=0.95):
    ax.plot(t, y, color=color, ls=ls, label=label, alpha=alpha)
    if logy:
        ax.set_yscale("log")
    return ax


def profile_compare(ax, x, est, ref, est_label, ref_label="reference",
                    color="#1f77b4"):
    ax.plot(x, ref, "k--", lw=1.6, label=ref_label)
    ax.plot(x, est, color=color, lw=2.0, label=est_label)
    ax.set_xlabel("x")
    ax.legend()
    return ax


def heatmap(ax, pivot, xlabel, ylabel, title, cbar_label=None, cmap="viridis_r"):
    """Render a 2-D pivot table (``pandas.DataFrame``) as a labelled heatmap."""
    values = pivot.values.astype(float)
    im = ax.imshow(values, origin="lower", aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c:g}" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{r:g}" for r in pivot.index])
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(False)
    # Annotate cells (skip if grid is large).
    if values.size <= 64:
        finite = values[np.isfinite(values)]
        vmid = np.nanmedian(finite) if finite.size else 0.0
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                v = values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.3g}", ha="center", va="center",
                            color="white" if v < vmid else "black", fontsize=8)
    cbar = ax.figure.colorbar(im, ax=ax)
    if cbar_label:
        cbar.set_label(cbar_label)
    return im


def boxplot_by_group(ax, groups: Dict[str, Sequence[float]], ylabel: str,
                     title: str, colors: Optional[Dict[str, str]] = None):
    """Boxplot one series per group; falls back to a scatter for tiny samples."""
    labels = list(groups.keys())
    data = [np.asarray(groups[k], dtype=float) for k in labels]
    data = [d[np.isfinite(d)] for d in data]
    max_n = max((len(d) for d in data), default=0)
    if max_n >= 3:
        bp = ax.boxplot(data, labels=labels, showmeans=True, patch_artist=True)
        if colors:
            for patch, lab in zip(bp["boxes"], labels):
                patch.set_facecolor(colors.get(lab, "#cccccc"))
                patch.set_alpha(0.6)
    else:
        # Too few seeds for a boxplot: show individual points.
        for i, (lab, d) in enumerate(zip(labels, data)):
            ax.scatter(np.full_like(d, i + 1), d, s=30,
                       color=(colors or {}).get(lab, "#1f77b4"))
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel); ax.set_title(title)
    for tick in ax.get_xticklabels():
        tick.set_rotation(20)
        tick.set_ha("right")
    return ax
