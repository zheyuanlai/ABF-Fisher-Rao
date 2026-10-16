# Overnight molecular extension — STATUS

*Started 2026-10-16 17:12 UTC. Two adaptive arms only: histogram ABF and histogram ABF + uniform marginal FR. No sham,
oracle, OPES or any other algorithmic arm.*

## Current stage

**Stage A (literature assessment of candidates A–E) is running.** Two independent literature agents are working,
read-only:
* Candidate A: ethane in Si-ITE / Si-IHW;
* a feasibility scan of candidates B–E.

## Where things stand

| item | status |
|---|---|
| ERI dilute (ethane, Gard & Tait ERI, 600 K) | **done.** All gates pass. ABF-only screen: **ABF-sufficient at K 64/256/1024** → FR not licensed (`docs/eri_campaign/RESULTS.md`) |
| ERI loading study | not run. It needs a GPU engine; deprioritised under the overnight instruction |
| Candidate A (ITE/IHW) | literature in progress |
| Candidates B–E | feasibility scan in progress |

## Active jobs

None. Nothing runs on any GPU.

## Next automatic action

Read the Candidate A report. If it is reproducible, implement it on the validated `src/eri` machinery: the same
united-atom ethane engine, a new framework and a new CV. Then run Gates 0–3 → ABF-only screen → (if eligible) two-arm
production.

## Resource use

See `COMPUTE_LEDGER.md`: 19.2 of 200 core-h and 0 of 24 GPU-h used.
