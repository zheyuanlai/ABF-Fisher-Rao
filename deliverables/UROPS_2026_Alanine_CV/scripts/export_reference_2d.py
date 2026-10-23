#!/usr/bin/env python
"""Export the project's 2-D alanine reference F(phi, psi) (umbrella sampling + MBAR, 97 x 97) as plain CSV.

Source: results/alanine/reference/reference.npz (git-ignored; key 'F', kJ/mol) and its copy-level bootstrap
results/alanine/reference/bootstrap.npz (key 'F_se').  Outputs in data/reference/:
  reference_F2_kJ.csv      F(phi, psi) in kJ/mol; ROWS = phi, COLUMNS = psi, both on the grid
                           -pi + (i + 1/2) 2 pi / 97 (IUPAC dihedrals); NaN = never-visited cell (+inf in the npz).
                           Exactly the requested np.savetxt(..., delimiter=',') of np.where(isfinite(F), F, nan)
                           (default '%.18e', so float64 round-trips exactly).  The stored surface is shifted so that
                           its global minimum is 0.
  reference_F2_se_kJ.csv   bootstrap standard error per cell (kJ/mol; NaN where the bootstrap is undefined)
  reference_F2_minima.csv  local minima of the cell grid with prominence > 2 x the median cell s.e. (8-neighbour,
                           periodic): basin label, phi, psi, F - F_min, prominence, cell s.e.
  reference_F2_grid_deg.csv the 97 grid values in degrees (same for phi and psi)
"""
import heapq
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common_ala as CA  # noqa: E402

OUT = os.path.join(CA.PKG, "data", "reference")
LABELS = {(-74.2, 55.7): "C7eq", (-152.2, 155.9): "C5", (63.1, -48.2): "C7ax"}   # the repo's basin centres (deg)


def prominence(Fi, a0, b0):
    """Minimax barrier from (a0, b0) to any strictly lower cell (8-neighbour, periodic); inf for the global minimum."""
    n = Fi.shape[0]
    v0 = Fi[a0, b0]
    best = {(a0, b0): v0}
    h = [(v0, a0, b0)]
    nbr = [(da, db) for da in (-1, 0, 1) for db in (-1, 0, 1) if (da, db) != (0, 0)]
    while h:
        m, a, b = heapq.heappop(h)
        if Fi[a, b] < v0:
            return m - v0
        for da, db in nbr:
            c = ((a + da) % n, (b + db) % n)
            x = Fi[c]
            if not np.isfinite(x):
                continue
            mm = max(m, x)
            if mm < best.get(c, np.inf):
                best[c] = mm
                heapq.heappush(h, (mm,) + c)
    return math.inf


def main():
    os.makedirs(OUT, exist_ok=True)
    F = np.asarray(np.load(CA.dpath(CA.REF), allow_pickle=True)["F"], float)
    np.savetxt(os.path.join(OUT, "reference_F2_kJ.csv"), np.where(np.isfinite(F), F, np.nan), delimiter=",")
    se = np.asarray(np.load(CA.dpath("results/alanine/reference/bootstrap.npz"))["F_se"], float)
    np.savetxt(os.path.join(OUT, "reference_F2_se_kJ.csv"), np.where(np.isfinite(se), se, np.nan), delimiter=",")
    n = F.shape[0]
    g = np.degrees(-np.pi + (np.arange(n) + 0.5) * 2 * np.pi / n)
    np.savetxt(os.path.join(OUT, "reference_F2_grid_deg.csv"), g[:, None], delimiter=",", header="angle_deg", comments="")
    Fi = np.where(np.isfinite(F), F, np.inf)
    Fi = Fi - Fi[np.isfinite(Fi)].min()
    thr = 2.0 * float(np.nanmedian(se[np.isfinite(se) & np.isfinite(F) & (Fi <= 8 * 2.494)]))
    nbr = [(da, db) for da in (-1, 0, 1) for db in (-1, 0, 1) if (da, db) != (0, 0)]
    rows = []
    for a in range(n):
        for b in range(n):
            if np.isfinite(Fi[a, b]) and all(Fi[a, b] < Fi[(a + da) % n, (b + db) % n] for da, db in nbr):
                pr = prominence(Fi, a, b)
                if pr > thr:
                    lab = next((v for (p, q), v in LABELS.items() if abs(p - g[a]) < 0.1 and abs(q - g[b]) < 0.1), "")
                    if not lab:
                        lab = "unresolved (unsampled region; no bootstrap s.e.)" if not np.isfinite(se[a, b]) else "unlabelled"
                    rows.append((lab, g[a], g[b], Fi[a, b], pr, se[a, b]))
    rows.sort(key=lambda r: r[3])
    with open(os.path.join(OUT, "reference_F2_minima.csv"), "w") as fh:
        fh.write("label,phi_deg,psi_deg,F_minus_Fmin_kJ,prominence_kJ,cell_se_kJ\n")
        for lab, p, q, f, pr, s in rows:
            fh.write(f"{lab},{p:.4f},{q:.4f},{f:.4f},{'inf' if not math.isfinite(pr) else f'{pr:.4f}'},{'nan' if not np.isfinite(s) else f'{s:.4f}'}\n")
    # the alpha_R region: the cell-level local minimum nearest (-60, -40) and its prominence
    cand = [(Fi[a, b], g[a], g[b], prominence(Fi, a, b)) for a in range(n) for b in range(n)
            if -90 < g[a] < -40 and -60 < g[b] < 0 and np.isfinite(Fi[a, b])
            and all(Fi[a, b] < Fi[(a + da) % n, (b + db) % n] for da, db in nbr)]
    info = dict(source=CA.REF, sha256_npz=CA.sha256_file(CA.dpath(CA.REF)), shape=list(F.shape), rows="phi", columns="psi",
                grid="-pi + (i + 1/2) 2 pi / 97, i = 0..96 (radians); degrees in reference_F2_grid_deg.csv",
                units="kJ/mol (kT = 2.494 kJ/mol at 300 K)", nan_cells=int((~np.isfinite(F)).sum()),
                global_min_phi_psi_deg=[float(g[np.unravel_index(np.argmin(Fi), Fi.shape)[0]]), float(g[np.unravel_index(np.argmin(Fi), Fi.shape)[1]])],
                prominence_threshold_kJ=thr, resolved_minima=[dict(label=r[0], phi=r[1], psi=r[2], F=r[3], prominence=r[4]) for r in rows],
                alpha_R_region_cell_minima=[dict(phi=p, psi=q, F=f, prominence=pr) for f, p, q, pr in sorted(cand)],
                csv_sha256={f: CA.sha256_file(os.path.join(OUT, f)) for f in ("reference_F2_kJ.csv", "reference_F2_se_kJ.csv",
                                                                             "reference_F2_minima.csv", "reference_F2_grid_deg.csv")})
    CA.jdump(info, os.path.join(OUT, "reference_F2_info.json"))
    print(json.dumps({k: info[k] for k in ("shape", "nan_cells", "global_min_phi_psi_deg", "prominence_threshold_kJ")}))
    for r in rows:
        print("  minimum:", r)
    print("  alpha_R-region cell minima (F, phi, psi, prominence):", [(round(f, 2), round(p, 1), round(q, 1), round(pr, 2)) for f, p, q, pr in sorted(cand)])


if __name__ == "__main__":
    main()
