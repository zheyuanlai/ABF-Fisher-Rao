"""Run a list of arms in ONE process.

torch.compile warm-up is ~2-3 minutes per distinct code path and is paid per
PROCESS, so a shell loop over 19 single-arm jobs spends more wall-clock
compiling than sampling.  This driver keeps the system, the reference tables
and the compiled graphs alive across arms.
"""
from __future__ import annotations

import argparse, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mol_campaign import run_one


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, help="JSON list of run_one kwargs")
    a = ap.parse_args()
    spec = json.load(open(a.spec)) if os.path.exists(a.spec) else json.loads(a.spec)
    t0 = time.time()
    for i, kw in enumerate(spec):
        print(f"[{i+1}/{len(spec)}] {kw.get('arm')} {kw.get('tag','')}", flush=True)
        try:
            run_one(**kw)
        except Exception as e:
            import traceback; traceback.print_exc()
    print(f"ALL_DONE {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
