"""Production gate of Experiment A (docs/final_campaign/PREREGISTRATION.md section 1.3: "Production starts only after
the gate G-A1/G-A2 and all nine checks below pass").  Importable; used by scripts/final_campaign/run_cells.py before
any production job is dispatched (and printed by its --dry-run).

    python scripts/final_campaign/expA_gate.py            # print the gate status (exit 0 iff satisfied)

``production_gate(P)`` reads only records; it runs nothing.  Items (each must hold):

  G-A1/G-A2 (and checks 3, 4a, 5)  results/final_campaign/expA/validation/summary.json (expA_validation.py analyze):
                not a smoke summary; selection.verdict == "PASS" exactly (PROVISIONAL / REFINE_REQUIRED / UNRESOLVED /
                HARNESS_FAIL / INCOMPLETE are not); selection.h_A == the frozen config's h (a refined h_A needs a dated
                amendment and a new config -- the driver refuses the frozen h then);
  checks 1, 7   checks/check1_7_tests.json (expA_record_tests.py): verdict PASS and input signature current;
  check 2       checks/check2_references.json (expA_check_references.py): verdict PASS and input signature current;
  check 4b      checks/check4b_frozen_x.json (h 2.5e-5; check4b_frozen_x_h<h>.json otherwise; expA_frozen_x.py):
                verdict PASS, status complete, signature.h_A == h, provenance.engine_sha256 == the frozen engine;
  check 8       checks/check8_old_bitwise.json (expA_check_old_bitwise.py): verdict PASS (8a/8b and 8c) and its 8c part
                is the CURRENT harness-inertness record checks/check8c_harness_inertness.json (verdict PASS, input
                signature current, same sha256 as recorded in check 8);
  check 9, 6    checks/check9_smoke.json and checks/check6_pairing_smoke.json (expA_check_smoke.py /
                expA_check_pairing.py --smoke): verdict PASS; run in smoke mode at the smoke horizon (default 4) with
                replay (and, for check 9, the pipeline clause) on; the expected numbers of files / pairs; root = the smoke
                root; input signature current over the smoke files now on disk (expA_check_pairing.record_is_current);
                an event-rule amendment named in a record must still occur in the prereg's amendment section.

Returns dict(ok, items=[dict(item, ok, detail, record)], h, smoke_root).
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import expA_run_common as C  # noqa: E402

ROOT = C.ROOT
H_FIRST = 2.5e-5


def _rel(p):
    return os.path.relpath(p, ROOT) if p and os.path.abspath(p).startswith(ROOT + os.sep) else p


def _load(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception as e:  # noqa: BLE001
        return {"_unreadable": f"{type(e).__name__}: {e}"}


def check4b_name(h):
    return "check4b_frozen_x.json" if float(h) == H_FIRST else f"check4b_frozen_x_h{float(h):g}.json"


def production_gate(P, out_root=None, smoke_T=4.0, prereg=None):
    root = C.prod_root(out_root)
    checks = os.path.join(root, "checks")
    h = float(P["engine_cfg"]["h"])
    items = []

    def add(item, probs, ok_detail, path):
        items.append(dict(item=item, ok=not probs, detail="; ".join(probs) if probs else ok_detail, record=_rel(path)))

    def need(rec, path):
        if rec is None:
            return [f"record missing ({_rel(path)})"]
        if "_unreadable" in rec:
            return [f"record unreadable ({rec['_unreadable']})"]
        return []

    # G-A1 / G-A2 selection
    p = os.path.join(root, "validation", "summary.json")
    S = _load(p)
    probs = need(S, p)
    if not probs:
        sel = S.get("selection") or {}
        if S.get("smoke"):
            probs.append("a SMOKE summary (not a gate result)")
        if sel.get("verdict") != "PASS":
            probs.append(f"selection verdict {sel.get('verdict')!r} (must be exactly 'PASS')")
        hA = sel.get("h_A")
        if hA is None or float(hA) != h:
            probs.append(f"selection h_A {hA} != the frozen config h {h:g} (a refined h needs a dated amendment and a "
                         f"new config)")
    add("G-A1/G-A2 timestep gate (covers checks 3, 4a, 5)", probs, f"selection PASS, h_A = {h:g}", p)

    # checks 1 / 7 and 8c (test records)
    import expA_record_tests as RT
    for key, label in (("check1_7", "checks 1 and 7 (test record)"),
                       ("check8c", "check 8c harness inertness (test record)")):
        spec = RT.SUITES[key]
        p = os.path.join(checks, spec["out"])
        rec = _load(p)
        probs = need(rec, p)
        if not probs:
            if rec.get("verdict") != "PASS":
                probs.append(f"verdict {rec.get('verdict')!r}")
            cur, why = RT.record_is_current(rec, spec)
            if not cur:
                probs.append(f"stale: {why}")
        add(label, probs, f"PASS ({(rec or {}).get('counts')})", p)

    # check 2
    import expA_check_references as CR
    p = os.path.join(checks, "check2_references.json")
    rec = _load(p)
    probs = need(rec, p)
    if not probs:
        if rec.get("verdict") != "PASS":
            probs.append(f"verdict {rec.get('verdict')!r}")
        cur, why = CR.record_is_current(rec)
        if not cur:
            probs.append(f"stale: {why} (re-run expA_check_references.py)")
    add("check 2 references", probs, "PASS, current", p)

    # check 4b
    p = os.path.join(checks, check4b_name(h))
    rec = _load(p)
    probs = need(rec, p)
    if not probs:
        if rec.get("verdict") != "PASS" or rec.get("status") != "complete":
            probs.append(f"verdict {rec.get('verdict')!r}, status {rec.get('status')!r}")
        hA = (rec.get("signature") or {}).get("h_A")
        if hA is None or float(hA) != h:
            probs.append(f"signature h_A {hA} != {h:g}")
        es = (rec.get("provenance") or {}).get("engine_sha256")
        if es != C.EXPECTED_ENGINE_SHA256:
            probs.append(f"engine sha256 {es} != the frozen engine")
    add("check 4b frozen-x AR(1)", probs, f"PASS at h_A {h:g}", p)

    # check 8 (8a/8b + 8c)
    import expA_check_old_bitwise as C8
    p = os.path.join(checks, "check8_old_bitwise.json")
    rec = _load(p)
    probs = need(rec, p)
    if not probs:
        if rec.get("verdict") != "PASS":
            probs.append(f"verdict {rec.get('verdict')!r}" + (f" ({rec.get('verdict_note')})" if rec.get("verdict_note")
                                                              else ""))
        p8 = rec.get("part_8c") or {}
        now = C8.part_8c(os.path.join(checks, RT.SUITES["check8c"]["out"]))
        if not (now["present"] and now["current"] and now["verdict"] == "PASS"):
            probs.append(f"8c harness-inertness record now: present {now['present']}, verdict {now['verdict']}, "
                         f"{now['reason']}")
        elif p8.get("sha256") != now["sha256"]:
            probs.append("check 8 was written against a different 8c record (re-run expA_check_old_bitwise.py)")
        if (rec.get("engine") or {}).get("expected") != C.EXPECTED_ENGINE_SHA256:
            probs.append("check 8 engine expectation differs from the frozen engine")
    add("check 8 old configurations (8a, 8b, 8c)", probs, "PASS", p)

    # checks 9 and 6 on the smoke runs
    import expA_check_pairing as CP
    sroot = C.smoke_root(out_root, smoke_T)
    cells = [v["name"] for v in P["variants"]]
    files = [q for v in CP.discover(sroot, cells).values() for q in v.values()]
    n_pairs = len(P["N"]) * len(cells) * len(P["smoke_seeds"])
    for fn, label, n_key, n_want, need_pipe in (
            ("check9_smoke.json", "check 9 smoke", "n_files_expected", 2 * n_pairs, True),
            ("check6_pairing_smoke.json", "check 6 pairing (smoke)", "n_expected_pairs", n_pairs, False)):
        p = os.path.join(checks, fn)
        rec = _load(p)
        probs = need(rec, p)
        amend = None
        if not probs:
            if rec.get("verdict") != "PASS":
                probs.append(f"verdict {rec.get('verdict')!r}" + (f" ({rec.get('verdict_note')})"
                                                                  if rec.get("verdict_note") else ""))
            o = rec.get("signature_options") or {}
            if o.get("mode") != "smoke" or o.get("smoke_T") is None or float(o.get("smoke_T")) != float(smoke_T):
                probs.append(f"not the smoke-mode check at smoke_T {smoke_T:g} (options {o})")
            if not o.get("replay"):
                probs.append("run without the replay")
            if need_pipe and not o.get("pipeline"):
                probs.append("run without the pipeline clause")
            if rec.get(n_key) != n_want:
                probs.append(f"{n_key} {rec.get(n_key)} != {n_want}")
            if os.path.realpath(str(rec.get("root"))) != os.path.realpath(sroot):
                probs.append(f"root {rec.get('root')} != {sroot}")
            amend = o.get("event_rule_amendment")
            if amend is not None and not C.amendment_ref_found(amend, prereg):
                probs.append(f"event-rule amendment {amend!r} not found in the prereg amendment section")
            cur, why = CP.record_is_current(rec, files)
            if not cur:
                probs.append(f"stale: {why}")
        add(label, probs, "PASS" + (f" (event rule licensed by amendment {amend!r})" if amend else ""), p)

    return dict(ok=all(i["ok"] for i in items), items=items, h=h, smoke_root=sroot)


def format_gate(G):
    lines = [f"production gate (prereg 1.3): {'SATISFIED' if G['ok'] else 'NOT SATISFIED'}"]
    for i in G["items"]:
        lines.append(f"  [{'ok' if i['ok'] else 'NO'}] {i['item']}: {i['detail']}")
    return "\n".join(lines)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    G = production_gate(C.load_config(), a.out_root)
    print(format_gate(G))
    return 0 if G["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
