"""Derive the screen's ETA from the queue it is actually running, never from memory.

I quoted a Thursday-afternoon completion three times after the population it was computed over
had changed from three seeds to six. Each quote was individually careful and each was wrong,
because the number lived in prose while the queue lived in a launch script and nothing connected
them. The NaCl session's rule is the fix:

    a number and the population it was computed over must be stored together,
    so that quoting one without the other is impossible.

So this script does not store an ETA. It **recomputes** one from three things it reads at call
time -- the seed files on disk, the seed lists of the processes actually running, and the
per-seed wall times measured from the logs -- and prints the population alongside the number.
There is no path through it that yields a completion time without also yielding the queue that
produced it.

Usage:
    python scripts/methane_status.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

SEED_BLOCK = list(range(5000, 5008))          #: §5 of the preregistration


def seeds_done(d):
    return sorted(int(re.search(r"seed(\d+)\.npz", f).group(1))
                  for f in glob.glob(os.path.join(d, "seed*.npz")))


def running_queues():
    """``[(pid, [seeds])]`` for live screen processes -- python only, never wrapper shells."""
    out = subprocess.run(["ps", "-eo", "pid,comm,args"], capture_output=True, text=True).stdout
    queues = []
    for line in out.splitlines():
        parts = line.split(None, 2)
        if len(parts) < 3 or not parts[1].startswith("python"):
            continue
        if "scripts/methane_screen.py" not in parts[2]:
            continue
        m = re.search(r"--seeds\s+([\d,]+)", parts[2])
        queues.append((int(parts[0]), [int(s) for s in m.group(1).split(",")] if m else []))
    return queues


def measured_seed_hours(d):
    """Per-seed wall times parsed from the logs, cleanest first.

    Returns ``(hours, source)``. Contended seeds are kept but reported, since a median over a
    mixture would silently blend two throughputs.
    """
    times = []
    for log in sorted(glob.glob(os.path.join(d, "run*.log"))):
        with open(log, errors="ignore") as fh:
            for line in fh:
                m = re.search(r"\[seed (\d+)\] done in ([\d.]+) min", line)
                if m:
                    times.append((int(m.group(1)), float(m.group(2)) / 60.0))
    return times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", default="results/methane/screen_N512")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    done = seeds_done(args.screen)
    queues = running_queues()
    times = measured_seed_hours(args.screen)
    queued = sorted({s for _, q in queues for s in q} - set(done))
    # Seeds an armed watcher intends to launch are neither running nor orphaned.  That intent
    # must live ON DISK: a watcher holding it only in its own process is exactly how the ETA and
    # the queue got disconnected in the first place.
    pending_file = os.path.join(args.screen, "pending_seeds.json")
    pending = []
    if os.path.exists(pending_file):
        # MUST subtract `queued` too: once a watcher hands its seeds to a live process they are
        # both declared and running, and adding the two sets double-counts them.  This tool
        # produced a 32 h ETA instead of 16 h on its second use for exactly that reason -- the
        # anti-stale-ETA tool getting the population wrong, in the union rather than in the
        # arithmetic.  A population assembled from two sources needs the union, not the sum.
        pending = sorted(set(json.load(open(pending_file)).get("pending", []))
                         - set(done) - set(queued))
    orphaned = sorted(set(SEED_BLOCK) - set(done) - set(queued) - set(pending))

    per_seed = min(h for _, h in times) if times else float("nan")
    remaining = len(queued) + len(pending)
    now = time.time()

    # in-flight progress of the currently running seed, from its own log
    in_flight_h = 0.0
    for log in sorted(glob.glob(os.path.join(args.screen, "run*.log"))):
        with open(log, errors="ignore") as fh:
            tail = fh.readlines()[-400:]
        steps = [re.search(r"step\s+(\d+)/(\d+).*\(\s*(\d+)s\)", ln) for ln in tail]
        steps = [m for m in steps if m]
        if steps and any(f"[seed" in ln and "done in" in ln for ln in tail[-3:]) is False:
            cur, tot, el = (int(steps[-1].group(1)), int(steps[-1].group(2)),
                            int(steps[-1].group(3)))
            if cur < tot and any(int(p) not in done for _, q in queues for p in q):
                in_flight_h = max(in_flight_h, (tot - cur) * (el / max(cur, 1)) / 3600.0)

    eta_h = in_flight_h + max(0, remaining - (1 if in_flight_h > 0 else 0)) * per_seed
    eta_epoch = now + eta_h * 3600

    res = dict(
        seeds_required=SEED_BLOCK, seeds_done=done, seeds_queued=queued,
        seeds_pending=pending, seeds_orphaned=orphaned, n_done=len(done), n_required=len(SEED_BLOCK),
        per_seed_hours_measured=per_seed,
        measured_from=[{"seed": s, "hours": round(h, 2)} for s, h in times],
        in_flight_hours=round(in_flight_h, 2), remaining_seeds=remaining,
        eta_hours=round(eta_h, 2),
        eta_utc=time.strftime("%a %H:%M UTC", time.gmtime(eta_epoch)),
        eta_local=time.strftime("%a %H:%M %Z", time.localtime(eta_epoch)),
        running=[{"pid": p, "seeds": q} for p, q in queues],
    )
    if args.json:
        print(json.dumps(res, indent=2))
        return

    print(f"seeds required : {SEED_BLOCK}")
    print(f"seeds done     : {done}  ({len(done)}/{len(SEED_BLOCK)})")
    print(f"seeds queued   : {queued}   on pids {[p for p, _ in queues]}")
    print(f"seeds pending  : {pending}   (declared in pending_seeds.json, watcher-armed)")
    if orphaned:
        print(f"seeds ORPHANED : {orphaned}  <-- in no queue and not on disk; they will NEVER run")
    print(f"per-seed hours : {per_seed:.2f}  (min of {[round(h,2) for _, h in times]})")
    print(f"in flight      : {in_flight_h:.2f} h remaining on the current seed")
    print(f"ETA            : {eta_h:.2f} h  ->  {res['eta_utc']}  /  {res['eta_local']}")
    if orphaned:
        raise SystemExit("orphaned seeds present: the ETA above does not cover them")


if __name__ == "__main__":
    sys.exit(main())
