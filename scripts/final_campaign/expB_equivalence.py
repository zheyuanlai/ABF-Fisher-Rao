#!/usr/bin/env python
"""V-B4 statistical equivalence of the counter-mode parallel engine with the validated engine (prereg section 2.3).

    python scripts/final_campaign/expB_equivalence.py --system gateway [--extension-done]

Arms (per system): the core small-N ABF arm (gateway ABF 16, LTA ABF 2), ABF 512 and FR 512.
  counter    : expB_run.py equiv runs (counter seeds; + extension counter seeds), per-seed Ibar_F from their records;
  validated  : gateway = the 32 equal-budget production runs 8100-8131 (per-seed Ibar_F of
               results/equal_budget_v2/gateway/analysis/summary.json; + extension validated runs 8700-8731);
               LTA 150 K = the 16 production runs 15000-15015 (summary) + the 16 new validated runs 15340-15355
               (+ extension validated 15460-15491).
Rules (frozen):
  * ratio of median Ibar_F (counter / validated): 95 % percentile CI of a 10 000-resample bootstrap
    (default_rng(20261010); the two samples resampled independently, counter first) must lie within [0.85, 1.18]
    for every arm;
  * G = (FR - ABF) / ABF per seed within each engine (pairs = seeds with both N = 512 arms); the 95 % CI of the
    difference of median G (counter - validated; independent bootstraps of the two engines' pair sets) must lie
    within +-0.10;
  * extension: if a CI exceeds its band while its point estimate lies in [0.93, 1.075] (ratio) / [-0.05, 0.05] (G),
    ONE pre-specified extension adds 32 seeds per side (for the failing arms; G -> both N = 512 arms) and the test is
    re-evaluated once (--extension-done).  A failing contrast with its point outside the window fails outright.
  * R1-R5 of scripts/mechanism/analyze_parallel_benchmark.consistency_tests on the N = 512 pair, reported as a
    descriptive non-rejection check, never cited as equivalence.
  * the counter records must all come from ONE engine (identical engine source hashes in their signatures); records
    of more than one engine make the verdict INCOMPLETE (equivalence of a mixture validates no engine).  The hash set
    is reported (``engine_sources``) and decision.json / production are tied to it (expB_bench.decide).
Output: results/final_campaign/expB/equiv/<system>/vb4.json.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

import expB_common as C  # noqa: E402

EQUIV_VERSION = "expB_equivalence/1"
RATIO_BAND = (0.85, 1.18)
RATIO_EXT_WINDOW = (0.93, 1.075)
G_BAND = 0.10
G_EXT_WINDOW = 0.05
N_BOOT = 10_000
BOOT_SEED = 20261010


def equiv_arm_names(spec):
    small = min(n for m, n in spec["core_arms"] if m == "abf")
    return [C.arm_name("abf", small), "abf512", "fr512"]


# ================================================================================================ data
def load_counter(out_root, system, arm, seeds, kind="counter"):
    out = {}
    for s in seeds:
        p = os.path.join(out_root, "equiv", system, kind, arm, f"s{s}.json")
        if not os.path.exists(p):
            continue
        r = C.read_json(p)
        if r.get("status") == "complete" and r.get("Ibar_F") is not None:
            out[int(s)] = C.num(r["Ibar_F"])
    return out


def load_production_ibar(spec, arm, seeds):
    m, N = C.parse_arm(arm)
    d = C.read_json(spec["eqb_summary"])
    ps = d["per_N"][str(N)][m]["Ibar_F"]["per_seed"]
    return {int(s): C.num(ps[str(s)]) for s in seeds if str(s) in ps}


def gather(spec, out_root, extension=False):
    es = spec["equivalence_seeds"]
    data = {}
    for arm in equiv_arm_names(spec):
        cseeds = list(es["counter"]) + (list(es["extension_counter"]) if extension else [])
        counter = load_counter(out_root, spec["system"], arm, cseeds)
        validated = load_production_ibar(spec, arm, es["reference_validated"])
        newv = list(es.get("new_validated", [])) + (list(es["extension_validated"]) if extension else [])
        validated.update(load_counter(out_root, spec["system"], arm, newv, kind="validated"))
        data[arm] = dict(counter=counter, validated=validated)
    return data


# ================================================================================================ statistics
def ratio_of_medians(c, v, n_boot=N_BOOT, seed=BOOT_SEED):
    """median(c) / median(v) and its 95 % percentile CI (independent resampling: counter first)."""
    c = np.array(list(c.values()) if isinstance(c, dict) else c, dtype=float)
    v = np.array(list(v.values()) if isinstance(v, dict) else v, dtype=float)
    if c.size == 0 or v.size == 0:
        return dict(point=math.nan, ci95=[math.nan, math.nan], n_counter=int(c.size), n_validated=int(v.size))
    rng = np.random.default_rng(seed)
    bc = np.median(c[rng.integers(0, c.size, size=(n_boot, c.size))], axis=1)
    bv = np.median(v[rng.integers(0, v.size, size=(n_boot, v.size))], axis=1)
    r = bc / bv
    return dict(point=float(np.median(c) / np.median(v)), ci95=[float(np.percentile(r, 2.5)),
                                                               float(np.percentile(r, 97.5))],
                median_counter=float(np.median(c)), median_validated=float(np.median(v)), n_counter=int(c.size),
                n_validated=int(v.size))


def paired_G(abf, fr):
    common = sorted(set(abf) & set(fr))
    return {s: (fr[s] - abf[s]) / abf[s] for s in common if abf[s] != 0}


def g_difference(c_abf, c_fr, v_abf, v_fr, n_boot=N_BOOT, seed=BOOT_SEED):
    """median G (counter) - median G (validated), 95 % CI (independent bootstraps; counter first)."""
    gc = np.array(list(paired_G(c_abf, c_fr).values()), dtype=float)
    gv = np.array(list(paired_G(v_abf, v_fr).values()), dtype=float)
    if gc.size == 0 or gv.size == 0:
        return dict(point=math.nan, ci95=[math.nan, math.nan], n_counter=int(gc.size), n_validated=int(gv.size))
    rng = np.random.default_rng(seed)
    bc = np.median(gc[rng.integers(0, gc.size, size=(n_boot, gc.size))], axis=1)
    bv = np.median(gv[rng.integers(0, gv.size, size=(n_boot, gv.size))], axis=1)
    d = bc - bv
    return dict(point=float(np.median(gc) - np.median(gv)), ci95=[float(np.percentile(d, 2.5)),
                                                                 float(np.percentile(d, 97.5))],
                median_G_counter=float(np.median(gc)), median_G_validated=float(np.median(gv)),
                n_counter=int(gc.size), n_validated=int(gv.size))


def classify(ci, point, band, window):
    """PASS (CI inside band) / EXTEND (CI outside, point in window) / FAIL."""
    lo, hi = ci
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return "INCOMPLETE"
    if band[0] <= lo and hi <= band[1]:
        return "PASS"
    return "EXTEND" if window[0] <= point <= window[1] else "FAIL"


def evaluate(data, extension_done=False, n_boot=N_BOOT, seed=BOOT_SEED, min_n=32):
    """The V-B4 verdict from {arm: {counter: {seed: Ibar_F}, validated: {...}}}."""
    arms = {}
    for arm, d in data.items():
        r = ratio_of_medians(d["counter"], d["validated"], n_boot, seed)
        st = classify(r["ci95"], r["point"], RATIO_BAND, RATIO_EXT_WINDOW)
        if r["n_counter"] < min_n or r["n_validated"] < min_n:
            st = "INCOMPLETE"
        arms[arm] = dict(r, status=st, band=list(RATIO_BAND), extension_window=list(RATIO_EXT_WINDOW))
    G = None
    if "abf512" in data and "fr512" in data:
        G = g_difference(data["abf512"]["counter"], data["fr512"]["counter"], data["abf512"]["validated"],
                         data["fr512"]["validated"], n_boot, seed)
        st = classify(G["ci95"], G["point"], (-G_BAND, G_BAND), (-G_EXT_WINDOW, G_EXT_WINDOW))
        if G["n_counter"] < min_n or G["n_validated"] < min_n:
            st = "INCOMPLETE"
        G.update(status=st, band=[-G_BAND, G_BAND], extension_window=[-G_EXT_WINDOW, G_EXT_WINDOW])
    sts = [a["status"] for a in arms.values()] + ([G["status"]] if G else [])
    ext_arms = sorted({a for a, v in arms.items() if v["status"] == "EXTEND"} |
                      ({"abf512", "fr512"} if G and G["status"] == "EXTEND" else set()))
    if "INCOMPLETE" in sts:
        verdict = "INCOMPLETE"
    elif all(s == "PASS" for s in sts):
        verdict = "PASS"
    elif "FAIL" in sts:
        verdict = "FAIL"
    elif extension_done:
        verdict = "FAIL"                     # still outside the band after the one extension
    else:
        verdict = "EXTENSION REQUIRED"
    return dict(verdict=verdict, arms=arms, G_difference=G, extension_done=bool(extension_done),
                extension_arms=(ext_arms if verdict == "EXTENSION REQUIRED" else []), n_boot=n_boot, seed=seed)


def descriptive_R(data):
    """R1-R5 (scripts/mechanism/analyze_parallel_benchmark.consistency_tests) on the N = 512 pair, and the R1-type
    permutation test of the small-N arm.  Descriptive only."""
    try:
        import analyze_parallel_benchmark as APB         # noqa: E402  (imports matplotlib; analysis process)
    except Exception as e:  # noqa: BLE001
        return dict(status=f"not computed ({type(e).__name__}: {e})")
    out = {}
    if "abf512" in data and "fr512" in data:
        back = {"abf": data["abf512"]["counter"], "fr": data["fr512"]["counter"]}
        ref = {"abf": data["abf512"]["validated"], "fr": data["fr512"]["validated"]}
        out["N512"] = APB.consistency_tests(back, ref)
    for arm in data:
        if arm not in ("abf512", "fr512"):
            out[arm] = APB.perm_median_test(list(data[arm]["counter"].values()), list(data[arm]["validated"].values()))
    out["note"] = "descriptive non-rejection battery (R1-R5); never cited as equivalence"
    return out


def counter_engine_sources(out_root, system):
    """Distinct engine source hash sets over the counter-mode records of this system."""
    import glob
    sets = {}
    for f in glob.glob(os.path.join(out_root, "equiv", system, "counter", "*", "s*.json")):
        try:
            r = C.read_json(f)
        except (OSError, ValueError):
            continue
        es = (r.get("signature") or {}).get("engine_sources")
        if es and r.get("status") == "complete":
            sets[json.dumps(es, sort_keys=True)] = es
    return list(sets.values())


def run(system, out_root=C.OUT_ROOT, extension_done=False, write=True, spec=None):
    spec = spec or C.system_spec(system)
    data = gather(spec, out_root, extension=extension_done)
    res = evaluate(data, extension_done)
    eng = counter_engine_sources(out_root, system)
    res["engine_sources"] = eng[0] if len(eng) == 1 else eng
    if len(eng) > 1:
        res.update(verdict="INCOMPLETE", extension_arms=[],
                   reason=f"the counter records come from {len(eng)} different engines (source hashes differ)")
    res.update(version=EQUIV_VERSION, system=system, descriptive_R=descriptive_R(data),
               per_seed={a: {k: {str(s): v for s, v in d[k].items()} for k in d} for a, d in data.items()},
               rule=("ratio of median Ibar_F (counter / validated) 95 % CI within [0.85, 1.18] for every arm, and 95 % "
                     "CI of the difference of median paired G at N 512 within +-0.10; one extension (+32 seeds per side) "
                     "if a CI exceeds its band with the point in [0.93, 1.075] (+-0.05)"))
    if write:
        C.write_json(os.path.join(out_root, "equiv", system, "vb4.json"), res)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description="V-B4 equivalence")
    ap.add_argument("--system", required=True, choices=sorted(C.SYSTEMS))
    ap.add_argument("--extension-done", action="store_true")
    ap.add_argument("--out-root", default=C.OUT_ROOT)
    a = ap.parse_args(argv)
    r = run(a.system, a.out_root, a.extension_done)
    print(json.dumps(dict(verdict=r["verdict"], extension_arms=r["extension_arms"],
                          arms={k: (v["point"], v["ci95"], v["status"]) for k, v in r["arms"].items()},
                          G=(r["G_difference"] or {}).get("ci95")), indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
