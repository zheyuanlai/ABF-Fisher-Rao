# Compute ledger — overnight molecular extension

The ceiling is **aggregate** across the ERI campaign and this extension: ≤ 200 CPU core-h and ≤ 24 GPU device-h in
total (2026-10-16 instruction: "Preserve the existing aggregate compute allocation"). GPUs 0/1 belong to another user
and are never touched. GPU 3 is the project device. CPU work is single-threaded processes, at most 32 at a time.

| date (UTC) | item | CPU core-h | GPU h | cumulative CPU | cumulative GPU |
|---|---|---|---|---|---|
| 2026-10-16 ≤ 17:10 | ERI campaign, everything up to the dilute screen (see `docs/eri_campaign/STATUS.md`) | 18.9 | 0 | 18.9 | 0 |
| 2026-10-16 17:10–17:30 | ERI classification, audit verification checks (small CPU), tests | 0.3 | 0 | 19.2 | 0 |

**Remaining authorisation at 17:30:** about 180.8 CPU core-h and 24.0 GPU-h.
