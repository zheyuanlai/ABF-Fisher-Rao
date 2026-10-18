#!/usr/bin/env python
"""Pipeline-fault check of production / confirmation records (prereg section 5 'smoke'): NO accuracy outcome is
printed -- only pass / fail of the record contract, the save grid, the force-evaluation accounting, the science
hashes, finite scores, endpoint computability, the contention verdicts, the ledger marks and the timing model
(completion wall / (B x benchmark c_ss on the route)).

    python scripts/hybrid_short_T/smoke_check.py [--kind prod] [--max-seed-index 2]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

import hst_common as H  # noqa: E402
import expB_common as C  # noqa: E402
import benchmark as BM  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--kind", default="prod")
    a = ap.parse_args(argv)
    summ = C.read_json(os.path.join(BM.BENCH_DIR, "summary.json"))["configs"]
    probs, n, tm, by_backend = [], 0, [], {}
    for s in H.SYSTEMS:
        spec = H.system_spec(s)
        sc = H.Scorer(spec)
        root = os.path.join(H.OUT_ROOT, a.kind, s)
        if not os.path.isdir(root):
            continue
        for arm in sorted(os.listdir(root)):
            d = os.path.join(root, arm)
            for f in sorted(os.listdir(d)):
                if not f.endswith(".json") or ".failed" in f:
                    continue
                rec = C.read_json(os.path.join(d, f))
                j = rec.get("job", {})
                tag = f"{s}/{arm}/{f}"
                if rec.get("status") != "complete":
                    probs.append(f"{tag}: status {rec.get('status')}")
                    continue
                n += 1
                try:
                    res = dict(H.load_science(j))
                    st, pm, um = H.save_grid(spec, j["N"])
                    if not np.array_equal(np.asarray(res["save_step"]), st):
                        probs.append(f"{tag}: save grid differs")
                    w = np.asarray(res["wall_s_at_save"], float)
                    if w.shape != st.shape or not np.all(np.isfinite(w)) or np.any(np.diff(w) < 0):
                        probs.append(f"{tag}: wall stamps not finite / monotone")
                    nfe = np.asarray(res["n_force_evals_at_save"], np.int64)
                    want = j["N"] * st if spec["kind"] == "gateway" else j["N"] * (st + 1)
                    if not np.array_equal(nfe, want):
                        probs.append(f"{tag}: force-evaluation accounting")
                    if j["replicate"] == 0 or "M_all" in C.load_run(H.job_paths(j)["npz"]):
                        sha = {k: C.sha256_array(res[k]) for k in C.SCORE_KEYS[spec["kind"]] + ("save_step",)}
                        if sha != rec.get("science_sha256"):
                            probs.append(f"{tag}: science sha256 differs from the record")
                    e = sc.e_F(res)
                    if not np.all(np.isfinite(e)):
                        probs.append(f"{tag}: non-finite e_F")
                    res["wall_total_s"] = float(rec["wall_total_s"])
                    H.run_endpoints(res, spec, sc, j, rec=rec, err=e)
                    if "contention_endpoints" not in rec or "contention_whole" not in rec:
                        probs.append(f"{tag}: contention verdicts missing")
                    if not os.path.exists(H.job_paths(j)["ledger_mark"]):
                        probs.append(f"{tag}: no ledger mark")
                    if rec.get("backend") != H.backend_label(j["impl"], j["threads"]):
                        probs.append(f"{tag}: backend label")
                    if rec.get("precision") != "float64" or rec.get("B") != spec["B"]:
                        probs.append(f"{tag}: precision / B")
                    route = BM.load_routing()["routes"][s][str(j["N"])]["common"]
                    if (route["impl"], route["threads"]) != (j["impl"], j["threads"]):
                        probs.append(f"{tag}: not on the frozen route")
                    c = BM.entry(summ, s, j["method"], j["N"], j["impl"], j["threads"])["c_ss_median"]
                    r = float(rec["wall_total_s"]) / (spec["B"] * c)
                    tm.append(r)
                    b = by_backend.setdefault(rec["backend"], dict(n=0, contended=0, tm=[]))
                    b["n"] += 1
                    b["contended"] += int(bool(rec.get("contended")))
                    b["tm"].append(r)
                except Exception as ex:  # noqa: BLE001
                    probs.append(f"{tag}: {type(ex).__name__}: {ex}")
    out = dict(n_records=n, problems=probs,
               timing_model_ratio=dict(median=float(np.median(tm)) if tm else None,
                                       min=float(np.min(tm)) if tm else None, max=float(np.max(tm)) if tm else None),
               by_backend={k: dict(n=v["n"], contended=v["contended"], timing_model_median=float(np.median(v["tm"])))
                           for k, v in sorted(by_backend.items())})
    print(json.dumps(out, indent=1))
    return 1 if probs else 0


if __name__ == "__main__":
    sys.exit(main())
