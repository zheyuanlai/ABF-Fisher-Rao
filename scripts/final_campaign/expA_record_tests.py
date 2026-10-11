#!/usr/bin/env python
"""Test-based pre-production checks of Experiment A as RECORDS (docs/final_campaign/PREREGISTRATION.md section 1.3):

    python scripts/final_campaign/expA_record_tests.py --dry-run                      # nodes, record status; no run
    taskset -c <1 core> python scripts/final_campaign/expA_record_tests.py [--suite check1_7|check8c|all]

Checks 1 and 7 and the harness-inertness part of check 8 are decided by pytest tests.  This tool runs exactly the listed
test nodes (pytest, one process, junit XML in a temp dir) and writes one JSON record per suite under
results/final_campaign/expA/checks/, which the production gate of scripts/final_campaign/run_cells.py and check 8
(expA_check_old_bitwise.py) read:

  check1_7            check1_7_tests.json            check 1 (analytic forces vs finite differences; kernel deposit
                      force = grad_V at every lam) and check 7 (independent Langevin MT19937 / FR PCG64
                      SeedSequence([seed, N, 1]) streams; the FR arm's Langevin stream equals the ABF arm's): the
                      tests/test_final_expA_checks.py C1 / C7 tests and the existing engine tests F1 and F6
                      (tests/test_gateway_family_numba.py), re-run as the prereg asks;
  check8c             check8c_harness_inertness.json the harness-inertness test of the prereg (before any G-A1 run):
                      tests/test_final_expA_validation.py (a) old dynamics bitwise identical to 189c2408 and the raw
                      kernel tuple appended at the end, (b) settings / seeds / traces / ACF plans identical and the
                      runner reproducing run_validation bitwise, (c) the unmodified analyzer reproducing the frozen
                      summary.json, (d) run_validation --dry-run unchanged.

Verdict PASS iff pytest exits 0 and every listed node produced >= 1 test case and every test case PASSED (a skipped
test is NOT a pass: it did not run).  Each record holds the node list, every test case's outcome and duration, the
pytest exit code and output tail, the wall time, provenance (git commit, dirty flag, versions, host, CPU affinity) and
an input signature: sha256 over the test files, the code under test and this tool.  Idempotent: an existing record
with the same signature is kept ('up to date'; not re-run); one with a different signature is never overwritten
without --force (exit 2).  ``record_is_current`` recomputes the signature (the production gate uses it), so editing a
test or the code under test makes the record stale.  Each run appends one row to results/final_campaign/compute_ledger.csv
(stage expA-tests, threads 1).  Exit: 0 every requested suite PASS, 1 a FAIL, 2 refused.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import expA_run_common as C  # noqa: E402

ROOT = C.ROOT
CHECKS = os.path.join(C.PROD_ROOT, "checks")
T_CHK = "tests/test_final_expA_checks.py"
T_ENG = "tests/test_gateway_family_numba.py"
T_VAL = "tests/test_final_expA_validation.py"
SUITES = {
    "check1_7": dict(
        out="check1_7_tests.json",
        title="checks 1 and 7 (PREREGISTRATION.md 1.3): analytic forces vs finite differences and the kernel deposit "
              "force; independent Langevin / FR streams, FR arm's Langevin stream = ABF arm's",
        nodes=[f"{T_CHK}::test_c1_grad_V_matches_central_differences",
               f"{T_CHK}::test_c1_kernel_deposit_force_equals_grad_V",
               f"{T_CHK}::test_c7_langevin_and_fr_streams",
               f"{T_CHK}::test_c7_streams_and_initial_state_independent_of_lam",
               f"{T_CHK}::test_c7_fr_arm_paired_with_abf_before_first_event",
               f"{T_ENG}::test_f1_gradient_matches_finite_differences",
               f"{T_ENG}::test_f1_kernel_force_and_one_em_step",
               f"{T_ENG}::test_f6_fr_gamma0_equals_abf",
               f"{T_ENG}::test_f6_stream_separation_and_chunk_independence",
               f"{T_ENG}::test_f6_fr_helpers_equal_original"],
        code=["src/gateway_family_numba.py", "src/gateway_ladder_numba.py"]),
    "check8c": dict(
        out="check8c_harness_inertness.json",
        title="check 8, harness-inertness part (PREREGISTRATION.md 1.3 G-A1 'Inertness test (before any G-A1 run)')",
        nodes=[f"{T_VAL}::test_a_old_dynamics_bitwise_identical_to_189c2408",
               f"{T_VAL}::test_a_raw_kernel_tuple_appended_at_the_end",
               f"{T_VAL}::test_b_settings_seeds_traces_plans_identical_for_old_dynamics",
               f"{T_VAL}::test_b_runner_reproduces_run_validation_bitwise_for_old_dynamics",
               f"{T_VAL}::test_c_unmodified_analyzer_reproduces_frozen_summary",
               f"{T_VAL}::test_d_run_validation_dry_run_unchanged"],
        code=["src/gateway_family_validation.py", "scripts/mechanism/run_validation.py",
              "scripts/mechanism/analyze_validation.py", "scripts/final_campaign/expA_validation.py",
              "results/mechanism/validation/summary.json"]),
}


def _abs(p):
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def suite_files(spec):
    """Test files (from the node ids) and code under test, absolute, sorted, unique."""
    tests = sorted({_abs(n.split("::")[0]) for n in spec["nodes"]})
    return tests, sorted({_abs(p) for p in spec["code"]})


def signature(spec):
    """Input signature of a suite record: node list, sha256 of test files, code under test and this tool."""
    tests, code = suite_files(spec)
    h = hashlib.sha256()
    h.update(json.dumps(list(spec["nodes"])).encode())
    for p in tests + code + [os.path.abspath(__file__)]:
        h.update((os.path.relpath(p, ROOT) if p.startswith(ROOT + os.sep) else p).encode())
        h.update((C.sha256_file(p) if os.path.exists(p) else "absent").encode())
    return h.hexdigest()


def record_is_current(rec, spec):
    if not isinstance(rec, dict) or not rec.get("input_signature"):
        return False, "record has no input signature"
    if rec["input_signature"] != signature(spec):
        return False, "input signature differs from the current tests / code under test / tool"
    return True, "current"


def parse_junit(path, nodes):
    """Test cases of a junit XML file -> list of dicts (node, name, outcome, time_s) and per-node counts."""
    cases = []
    tree = ET.parse(path)
    for tc in tree.iter("testcase"):
        cls, name = tc.get("classname", ""), tc.get("name", "")
        outcome = "passed"
        for child in tc:
            if child.tag in ("failure", "error", "skipped"):
                outcome = {"failure": "failed", "error": "error", "skipped": "skipped"}[child.tag]
                break
        base = name.split("[")[0]
        mod = cls.replace(".", "/") + ".py"
        node = next((n for n in nodes if n.split("::")[1] == base and n.split("::")[0] == mod), None)
        if node is None:
            node = next((n for n in nodes if n.split("::")[1] == base), None)
        cases.append(dict(node=node, classname=cls, name=name, outcome=outcome, time_s=float(tc.get("time", 0) or 0)))
    per_node = {n: sum(c["node"] == n for c in cases) for n in nodes}
    return cases, per_node


def run_suite(name, spec, timeout=7200):
    t0 = time.time()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", NUMBA_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1")
    env.setdefault("NUMBA_CACHE_DIR", os.path.expanduser("~/.cache/numba_final"))
    env["PYTHONPATH"] = os.pathsep.join([os.path.join(ROOT, "src")] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH")
                                                                        else []))
    with tempfile.TemporaryDirectory(prefix="expA_record_tests_") as td:
        xml = os.path.join(td, "junit.xml")
        cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--junitxml={xml}",
               f"--basetemp={os.path.join(td, 'bt')}", "-o", "junit_family=xunit1"] + list(spec["nodes"])
        try:
            p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout)
            code, out = p.returncode, (p.stdout + p.stderr)
        except subprocess.TimeoutExpired as e:
            code, out = "timeout", f"pytest timed out after {timeout} s: {e}"
        cases, per_node = (parse_junit(xml, spec["nodes"]) if os.path.exists(xml) else ([], {n: 0 for n in
                                                                                          spec["nodes"]}))
    counts = {k: sum(c["outcome"] == k for c in cases) for k in ("passed", "failed", "error", "skipped")}
    unmatched = [n for n, k in per_node.items() if k == 0]
    stray = [c["name"] for c in cases if c["node"] is None]
    problems = []
    if code != 0:
        problems.append(f"pytest exit code {code}")
    if unmatched:
        problems.append(f"nodes without a test case (not collected / renamed?): {unmatched}")
    if stray:
        problems.append(f"test cases not mapped to a listed node: {stray[:5]}")
    for k in ("failed", "error", "skipped"):
        if counts[k]:
            problems.append(f"{counts[k]} test case(s) {k}" + (" (a skipped test did not run: not a pass)"
                                                              if k == "skipped" else ""))
    if not cases:
        problems.append("no test case ran")
    verdict = "PASS" if not problems else "FAIL"
    tests, code_files = suite_files(spec)
    return dict(check=name, title=spec["title"], verdict=verdict, problems=problems, nodes=list(spec["nodes"]),
                counts=counts, n_cases=len(cases), per_node=per_node, cases=cases, pytest_exit_code=code,
                pytest_output_tail=out[-4000:], command=" ".join(cmd[1:]), wall_s=time.time() - t0, start_epoch=t0,
                files_sha256={(os.path.relpath(p, ROOT) if p.startswith(ROOT + os.sep) else p):
                              (C.sha256_file(p) if os.path.exists(p) else None) for p in tests + code_files},
                input_signature=signature(spec), generated_utc=C.utc(),
                tool=os.path.relpath(os.path.abspath(__file__), ROOT), provenance=C.provenance())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", choices=list(SUITES) + ["all"], default="all")
    ap.add_argument("--out-dir", default=CHECKS)
    ap.add_argument("--force", action="store_true", help="replace a record with a different input signature, or re-run "
                                                         "a suite whose record is up to date")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--timeout-s", type=float, default=7200.0)
    ap.add_argument("--compute-ledger", default=C.COMPUTE_LEDGER, help=argparse.SUPPRESS)   # tests only
    ap.add_argument("--nodes", nargs="*", default=None, help=argparse.SUPPRESS)              # tests only
    ap.add_argument("--code", nargs="*", default=None, help=argparse.SUPPRESS)               # tests only
    a = ap.parse_args(argv)
    names = list(SUITES) if a.suite == "all" else [a.suite]
    rc = 0
    for name in names:
        spec = dict(SUITES[name])
        if a.nodes is not None:
            spec.update(nodes=list(a.nodes), code=list(a.code or []))
        out = os.path.join(a.out_dir, spec["out"])
        old = None
        if os.path.exists(out):
            try:
                old = json.load(open(out))
            except Exception:  # noqa: BLE001
                old = {}
        cur, why = record_is_current(old, spec) if old is not None else (False, "no record")
        if a.dry_run:
            print(f"{name}: {len(spec['nodes'])} nodes -> {out}; record: "
                  + ("absent" if old is None else f"verdict {old.get('verdict')}, {why}"), flush=True)
            for n in spec["nodes"]:
                print(f"    {n}", flush=True)
            continue
        if old is not None and cur and not a.force:
            print(f"{name}: record up to date (verdict {old.get('verdict')}), not re-run: {out}", flush=True)
            rc = max(rc, 0 if old.get("verdict") == "PASS" else 1)
            continue
        if old is not None and not cur and not a.force:
            print(f"{name}: REFUSED -- {out} exists with a different input signature ({why}; verdict "
                  f"{old.get('verdict')}); use --force to replace it", flush=True)
            rc = 2
            continue
        rec = run_suite(name, spec, timeout=a.timeout_s)
        if old is not None:
            rec["supersedes"] = dict(generated_utc=old.get("generated_utc"), verdict=old.get("verdict"),
                                     input_signature=old.get("input_signature"))
        os.makedirs(a.out_dir, exist_ok=True)
        tmp = out + f".tmp{os.getpid()}"
        with open(tmp, "w") as fh:
            json.dump({k: v for k, v in rec.items() if k != "start_epoch"}, fh, indent=1)
        os.replace(tmp, out)
        C.append_csv(a.compute_ledger, C.COMPUTE_FIELDS,
                     [C.compute_row("expA-tests", f"{name}: {len(spec['nodes'])} pytest nodes", rec["wall_s"],
                                    rec["start_epoch"], note=rec["verdict"])])
        print(f"{name}: {rec['verdict']} ({rec['counts']}, {rec['n_cases']} cases, {rec['wall_s']:.0f} s)"
              + ("" if not rec["problems"] else f"  {rec['problems']}") + f" -> {out}", flush=True)
        rc = max(rc, 0 if rec["verdict"] == "PASS" else 1)
    return rc


if __name__ == "__main__":
    sys.exit(main())
