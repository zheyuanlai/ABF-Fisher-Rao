# Prereg exposure note (written during the run, 2026-09-02)

`configs/information_campaign/wca_baseline_audit_prereg.json` prohibits "reading any
e_F of the 0.0125 / 0.00625 arms before Stage 1 has fixed h_read_star on the legacy
arm". `scripts/run_wca_bandwidth_audit.py` nevertheless prints `L2(F)=...` for EVERY
completed run to `run.log`, so those values were visible while the job was running.

Why this does not compromise the selection: Stage 1 is a mechanical rule over the
legacy arm ONLY -- ladder frozen in the prereg, plateau = within 2% of the ladder
minimum, h_read* = legacy if on the plateau else the largest plateau point -- and it
is executed by `scripts/analyze_wca_bandwidth_audit.py`, committed at 723bc6e BEFORE
any data existed. No human choice enters between seeing a number and fixing h_read*.

What it does compromise: the clause's *guarantee*. "Could not have been influenced"
is now an argument from the code rather than a property of the procedure. A future
version of this runner should print the endpoint only for the arm Stage 1 scores, or
print nothing and leave scoring entirely to the analyzer.

Recorded while the run was in flight, not after seeing the outcome.
