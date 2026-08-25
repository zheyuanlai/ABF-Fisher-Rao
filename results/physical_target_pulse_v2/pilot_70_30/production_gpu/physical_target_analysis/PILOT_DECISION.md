# Physical-target pulse v2 pilot decision

- Status: no_schedule_passed
- Observed runs: 440 / 440
- Schedules passing the accuracy-gain gate: 3 / 54
- Schedules passing the genealogy gate: 9 / 54
- Schedules passing both gates: 0 / 54
- Schedules passing post-stop integrity: 54 / 54
- Diagnostic best-gain schedule (not selected): abf_fr_physical|tt=physical|g=0.1|eta=0.1|bi=0.2|sf=0.3|fe=20
- Median paired gain in I_F: 9.006%
- Median paired gain in I_Fprime: 5.751%
- Favorable seeds: I_F=7, I_Fprime=6
- Median ancestral ESS/K: 0.185
- Seeds with ancestral ESS/K >= 0.5: 0 / 8
- Median maximum clone weight: 0.0586
- Median cumulative replacements: 362.0
- Post-stop integrity: True

A no-schedule-passed status is a completed negative pilot, not a software failure. No downstream campaign is authorized in that case.
The reported schedule is diagnostic and was selected after viewing all 54 cells; its apparent gain must not be presented as confirmatory.

## Strongest genealogy-safe schedule

- Config: abf_fr_physical|tt=physical|g=0.02|eta=0.1|bi=0.6|sf=0.7|fe=100
- Median paired gain in I_F: 0.170%
- Median paired gain in I_Fprime: 0.172%
- Median ancestral ESS/K: 0.510
