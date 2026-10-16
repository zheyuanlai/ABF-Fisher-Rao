"""Build and check both ERI framework variants -> cache/eri/framework_{variant}.npz (+ JSON summary)."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from eri import framework as fwm  # noqa: E402

out = {}
for v in ("gard_tait", "schlenker"):
    fw = fwm.build(v)
    ok, dmin, (lo, hi) = fwm.coordination_check(fw)
    s = fwm.special_sites(fw)
    path = os.path.join(fwm.ROOT, "cache", "eri", f"framework_{v}.npz")
    fwm.save(fw, path)
    out[v] = {"a": fw["a"], "c": fw["c"], "box": fw["box"].tolist(), "n_si": int((fw["types"] == 1).sum()),
              "n_o": int((fw["types"] == 0).sum()), "coordination_ok": ok, "d_min_any": dmin,
              "si_o_range": [lo, hi], "n_cages": len(s["cage"]), "n_windows": len(s["window"]),
              "asym": [list(x) for x in fw["asym"]], "source": fw["source"], "source_sha256": fw["source_sha256"],
              "npz_sha256": fwm.sha256_file(path)}
    print(v, json.dumps({k: out[v][k] for k in out[v] if k != "asym"}))
json.dump(out, open(os.path.join(fwm.ROOT, "cache", "eri", "frameworks.json"), "w"), indent=1)
