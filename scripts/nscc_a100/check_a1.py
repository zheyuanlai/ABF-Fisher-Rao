"""Amendment A1 check: law-check jobs re-run on the amended engines must reproduce the frozen-code records bitwise.

    python scripts/nscc_a100/check_a1.py --out-root <scratch dir>     (GPU node)
"""
import argparse, glob, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nscc_common as NC  # noqa: E402

PICK = [("gateway", "fr64_G1", 619000), ("gateway", "fr512_G1", 619001), ("lta150", "fr64_G1", 639000),
        ("lta150", "fr256_G1", 639001), ("lta300", "fr64_G1", 629000), ("lta300", "fr256_G1", 629001),
        ("lta150", "abf64_G1", 639002)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    a = ap.parse_args()
    rows, ok_all = [], True
    for system, cfgname, seed in PICK:
        ref = json.load(open(os.path.join(NC.OUT_ROOT, "law", system, cfgname, f"s{seed}.json")))
        job = ref["job"]
        r = subprocess.run([sys.executable, os.path.join(HERE, "child.py"), json.dumps(job), "--out-root", a.out_root],
                           capture_output=True, text=True)
        new = json.load(open(NC.job_paths(job, a.out_root)["json"]))
        same = new["science_sha256"] == ref["science_sha256"]
        ok_all &= same
        rows.append(dict(system=system, config=cfgname, seed=seed, identical=same,
                         wall_ref=ref["wall_total_s"], wall_new=new["wall_total_s"]))
        print(f"A1 {system} {cfgname} s{seed}: science identical {same}; wall {ref['wall_total_s']:.2f} -> "
              f"{new['wall_total_s']:.2f} s", flush=True)
    out = os.path.join(NC.OUT_ROOT, "validation", "amendment1_check.json")
    json.dump(dict(rows=rows, passed=ok_all), open(out, "w"), indent=1)
    print("A1_VERDICT", "PASS" if ok_all else "FAIL", out)
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
