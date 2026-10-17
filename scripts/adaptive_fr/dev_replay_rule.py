"""DEVELOPMENT ONLY: replay the online rule on dev_monitor records (continuous-FR runs) -> where it would stop."""
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from eri.fr_controller import R1, OnlineStop  # noqa: E402


def replay(path, spec=R1):
    z = np.load(path)
    c = json.loads(str(z["cfg"]))
    t0 = c["fr_start_steps"] * c["dt"]
    ctl = OnlineStop(spec, c["N"], t0)
    for i, t in enumerate(z["t"]):
        if t < t0 - 1e-9:
            continue
        ts = ctl.observe(t, z["BH"][i], z["DH"][i], z["C"][i], z["M"][i], z["essw"][i], z["maxfw"][i])
        if ts is not None:
            break
    return ctl


if __name__ == "__main__":
    for f in sorted(glob.glob(os.path.join(ROOT, "results/adaptive_fr_campaign/dev/*.npz"))):
        c = replay(f)
        print(f"{os.path.basename(f):40s} t1 {c.t1} decision {c.t_decision} t_stop {c.t_stop} ({c.reason}); "
              f"j_max {c.j_max:.3f}")
