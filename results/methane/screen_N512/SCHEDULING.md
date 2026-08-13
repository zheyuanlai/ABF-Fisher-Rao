# Device scheduling (Amendment 12.4 / 13.1 / 14.4)

2026-08-13 01:55 UTC — NaCl session (Amendment 14) requested one device per study.
Agreed: **GPU 2 hands over to NaCl at seed 5004's boundary** (~06:30 UTC Aug 13); no methane
work is requeued on it. Methane consolidates on **GPU 3**.

Consequences for this screen:
- seeds 5000 (GPU 3) and 5004 (GPU 2) complete on the pre-fix engine as started;
- seeds 5001–5003, 5005–5007 run serially on GPU 3;
- the engine for those six seeds is decided at the seed-5000 boundary idle window
  (~03:30 UTC): the Triton pair kernel (5/5 correctness gates, commit 9292b89) is deployed
  **only if** it additionally passes the same 8 ps equipartition trajectory gate the sync-free
  solver passed, on an idle device; otherwise the sync-free tensor engine runs.
  Either way ENGINE_VERSIONS.md records the build per seed.
- the seed-5003 tripwire is obsolete (GPU 3 now continues past 5003 by design).

Device idle-state at decision time: GPUs 2 and 3 both ~89 % / ~320 W (both screens running);
GPUs 0/1 other user (yesom), untouched.
