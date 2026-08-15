"""Preflight for an expensive NaCl launch (Amendment 15.3), including the check 15.3 lacked.

15.3's preflight verified five FACTS: pinned commit matches, code tree clean, tests pass, the
target GPU is idle, launch manifest written. On 2026-08-15 that preflight would have passed
while the launch was forbidden: GPU 3 was genuinely idle and genuinely ours to see, and
Amendment 16.4 had reassigned it to C60. `nvidia-smi` answers "is this device free"; it cannot
answer "may I use this device", and no measurement distinguishes those two questions because a
rule is not part of the state of the world.

So this script re-derives the GOVERNING COMPUTE CLAUSE from the preregistration at launch time
and asserts the target device against it. It deliberately does NOT cache the answer, and there
is deliberately no second machine-readable copy of the allocation to drift away from the
amendment: the amendment text is the only source of truth, and it is re-read every launch.

Fails loud and refuses to launch when it cannot find a grant. "I could not parse the rule" must
never be quieter than "the rule forbids this" -- the whole failure being guarded against is a
check that stayed silent when it did not apply.

    python scripts/nacl_preflight.py --device 2 --stage screen_N32 --out results/nacl/screen_N32
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREREG = os.path.join(REPO, "docs", "V2_PREREGISTRATION.md")
CODE_PATHS = ["src", "scripts", "tests", "docs"]

# "NaCl keeps GPU 2", "NaCl may use GPU 3", "NaCl stays on GPU 2", "NaCl continues on GPU 2".
# The character classes must admit NEWLINES: the clause this guard exists for wraps as
# "C60 takes GPU 3; NaCl\nkeeps GPU 2", and a [^.\n] class silently fails to parse it. They must
# still exclude '.' so a grant cannot be assembled across a sentence boundary from two clauses.
GRANT = re.compile(
    r"NaCl\b[^.]{0,60}?\b(?:keeps|may use|stays on|runs on|continues on|owns|takes)\b[^.]{0,40}?GPU\s*(\d)",
    re.I)
DENY = re.compile(r"(\w[\w/\-]*)\s+takes\s+GPU\s*(\d)", re.I)


def governing_compute_clause(text=None):
    """The highest-numbered amendment subsection that allocates GPUs. Highest number wins because
    amendments supersede; 15.4 is not evidence about the present once 16.4 exists."""
    text = open(PREREG).read() if text is None else text
    # subsections look like: #### 16.4 Compute: GPU 3 reassigned ...
    parts = re.split(r"^#### (\d+)\.(\d+)\s+(.*)$", text, flags=re.M)
    best = None
    for i in range(1, len(parts), 4):
        major, minor, title, body = (int(parts[i]), int(parts[i + 1]), parts[i + 2], parts[i + 3])
        if not re.search(r"GPU\s*\d", body):
            continue
        key = (major, minor)
        if best is None or key > best[0]:
            best = (key, f"{major}.{minor}", title.strip(), body.strip())
    return best


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{': ' + detail if detail else ''}", flush=True)
    return bool(ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, required=True)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pin-file", default="results/nacl/PINNED_COMMIT")
    ap.add_argument("--worktree", default="/home/zheyuanlai/ABF-Fisher-Rao-nacl-run")
    ap.add_argument("--skip-tests", action="store_true")
    args = ap.parse_args()

    ok = True
    print(f"NaCl preflight -- stage {args.stage}, target GPU {args.device}\n")

    # ---- 0. THE RULE. Re-read every launch; never cached, never remembered. ----------------
    print("[rule] governing compute clause, re-derived from the preregistration now:")
    got = governing_compute_clause()
    if got is None:
        ok = check("a GPU-allocating amendment exists", False, "none found -- REFUSING")
    else:
        (_, num, title, body) = got
        print(f"\n  --- Amendment {num} {title} ---")
        for ln in body.splitlines()[:12]:
            print(f"  | {ln}")
        print(f"  --- end {num} ---\n")
        grants = GRANT.findall(body)
        if not grants:
            ok = check(f"clause {num} grants NaCl a device", False,
                       "no NaCl grant parsed -- REFUSING (read it yourself; do not assume)")
        else:
            granted = sorted({int(g) for g in grants})
            ok &= check(f"clause {num} grants NaCl GPU {args.device}", args.device in granted,
                        f"NaCl is granted GPU {granted}; requested {args.device}")
        for who, dev in DENY.findall(body):
            if int(dev) == args.device and who.lower() not in ("nacl",):
                ok = check(f"GPU {args.device} not assigned elsewhere", False,
                           f"clause {num} assigns GPU {dev} to {who}")

    # ---- 1..4. the facts 15.3 already checked ---------------------------------------------
    pin = open(os.path.join(REPO, args.pin_file)).read().strip()
    wt = subprocess.run(["git", "-C", args.worktree, "rev-parse", "HEAD"],
                        capture_output=True, text=True).stdout.strip()
    ok &= check("pinned worktree at PINNED_COMMIT", wt.startswith(pin[:12]) or pin.startswith(wt[:12]),
                f"worktree {wt[:12]} vs pin {pin[:12]}")

    dirty = subprocess.run(["git", "-C", args.worktree, "status", "--porcelain"] + CODE_PATHS,
                           capture_output=True, text=True).stdout
    dirty = [l for l in dirty.splitlines() if not l.startswith("??")]
    ok &= check("worktree code paths clean", not dirty, "; ".join(dirty[:3]))

    if not args.skip_tests:
        r = subprocess.run([sys.executable, "-m", "pytest", "-q",
                            os.path.join(REPO, "tests")] + ["-k", "nacl"],
                           capture_output=True, text=True, cwd=REPO)
        ok &= check("NaCl test suite", r.returncode == 0, r.stdout.strip().splitlines()[-1][:80]
                    if r.stdout.strip() else "")

    q = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_bus_id,pid",
                        "--format=csv,noheader"], capture_output=True, text=True).stdout
    busid = subprocess.run(["nvidia-smi", "-i", str(args.device),
                            "--query-gpu=pci.bus_id", "--format=csv,noheader"],
                           capture_output=True, text=True).stdout.strip()
    users = [l for l in q.splitlines() if busid and busid in l]
    ok &= check(f"GPU {args.device} idle", not users, "; ".join(users[:2]))

    if not ok:
        print("\nPREFLIGHT FAILED -- not launching.")
        return 1

    os.makedirs(os.path.join(REPO, args.out), exist_ok=True)
    man = dict(stage=args.stage, device=args.device, pin=pin, worktree_head=wt,
               governing_compute_clause=f"Amendment {got[1]} {got[2]}" if got else None,
               rule_rechecked_at_launch=True,
               utc=subprocess.run(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"],
                                  capture_output=True, text=True).stdout.strip())
    with open(os.path.join(REPO, args.out, "launch_manifest.json"), "w") as fh:
        json.dump(man, fh, indent=2)
    print(f"\nPREFLIGHT PASSED -- manifest -> {args.out}/launch_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
