"""PREREGISTRATION.md section 2: framework selection from the single-bead folded |s| barrier (Gate 1 groups 0-3)."""
import json
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from eri import refanalysis as ra  # noqa: E402
from eri.model import KB  # noqa: E402

TARGET, BAND = 6.13, 0.35
kT = KB * 600.0
res = {}
for v in ("gard_tait", "schlenker"):
    acc = ra.load(os.path.join(ROOT, "results", "eri_campaign", "reference", v, "MC"), groups=[0, 1, 2, 3])
    sel = np.arange(4)
    bFs = ra.bead_profile(acc, sel, kT, "Hs")
    bFx = ra.bead_profile(acc, sel, kT, "Hx")
    F, f, p = ra.F_density(acc, sel, kT)
    U = ra.cond_mean(acc, sel, "MU")
    bs = ra.barrier_stats(F, U, kT)
    # jackknife over the 4 groups for the |s| barrier
    loo = [ra.bead_profile(acc, np.delete(sel, g), kT, "Hs")[0] for g in range(4)]
    se = float(np.sqrt(3 / 4 * ((np.array(loo) - np.mean(loo)) ** 2).sum()))
    res[v] = dict(bead_s_barrier_kT=float(bFs[0]), bead_s_barrier_se=se, bead_s_profile=bFs.tolist(),
                  bead_x_barrier_kT=float(bFx[0] if np.isfinite(bFx[0]) else np.nan),
                  bead_x_profile=bFx.tolist(),
                  com_phi=bs, in_band=bool(abs(bFs[0] - TARGET) <= BAND),
                  min_fine_count=float(acc["C_fine"][sel].sum((0, 1)).min()))
    print(v, f"|s| bead barrier {bFs[0]:.3f} +- {se:.3f} kT (target {TARGET} +- {BAND}); X-bead profile at 0: "
             f"{bFx[0]:.3f}; COM-phi barrier {bs['F_barrier_kT']:.3f} kT, dU {bs['U_barrier']:.2f} kJ/mol, "
             f"-TdS {bs['mTS_barrier']:.2f} kJ/mol; min fine count {res[v]['min_fine_count']:.0f}")
inb = [v for v in res if res[v]["in_band"]]
if len(inb) == 1:
    primary, why = inb[0], "the only variant inside the band"
elif len(inb) == 2:
    primary, why = "gard_tait", "both inside the band -> the cited structure"
else:
    primary = min(res, key=lambda v: abs(res[v]["bead_s_barrier_kT"] - TARGET))
    why = "neither inside the band -> the closer one; the published barrier is NOT reproduced"
res["decision"] = dict(primary=primary, rule=why, target=TARGET, band=BAND)
print("PRIMARY:", primary, "--", why)
od = os.path.join(ROOT, "results", "eri_campaign", "reference")
json.dump(res, open(os.path.join(od, "framework_selection.json"), "w"), indent=1)
