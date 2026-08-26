# Shared physical backbone (clean-v2)

Every stage below uses the identical physical model, estimator and evaluation
grid; only the seed block, the arms and the FR schedule differ.  The backbone is
inherited verbatim from the v2 `pilot_70_30` benchmark so that the clean-v2
result is comparable with the campaign it replaces.

| Quantity              | Value            |
| --------------------- | ---------------- |
| beta                  | 4.0              |
| K (replicas)          | 256              |
| dt                    | 0.002            |
| steps                 | 50000            |
| physical T            | 100              |
| ABF bandwidth h       | 0.05             |
| ABF update stride     | 10               |
| FR KDE eta            | 0.10             |
| evaluation stride     | 500              |
| initialization        | uniform          |
| x_tilt                | 0.1021665783     |

Seed blocks are disjoint by stage, so "fresh seeds" in Stage 3 is literally
true rather than approximately true:

| Stage | Purpose                     | Seeds       |
| ----- | --------------------------- | ----------- |
| 0     | engineering gates           | 0-1 (tiny)  |
| 1     | ABF-only threshold freezing | 1000-1015   |
| 2     | 9-schedule pilot            | 2000-2007   |
| 3     | fresh-seed confirmation     | 3000-3031   |
| 4     | long-horizon sanity (2T)    | 4000-4007   |

## The two scientific knobs

| Knob | Values | Meaning |
| --- | --- | --- |
| `L_FR` (`fr_every`) | 100, 500, 1000 | frequent-and-weak vs sparse-and-strong |
| `gamma` | 0.002, 0.01, 0.05 | FR dose |

The FR clock is interval-scaled, `dtau_FR = L_FR dt`, so the **integrated hazard**
over the active window is `gamma * T_FR * |S| = 60 gamma |S|` and does not depend
on `L_FR`.  That is what makes `fr_every` a fair axis instead of a disguised dose
axis.  It is *not* the expected number of events: a replica fires at most once per
opportunity, so

    E[N_events] = N_opp * (1 - exp(-gamma |S| L_FR dt)),   N_opp = 300 / 60 / 30

for `L_FR` = 100 / 500 / 1000.  Expected events per replica over the window:

| `gamma` | R12 (`\|S\| ~ 3`) L = 100 / 500 / 1000 | walls (`\|S\| ~ 40`) L = 100 / 500 / 1000 |
| --- | --- | --- |
| 0.002 | 0.36 / 0.36 / 0.36 | 4.8 / 4.6 / 4.4 |
| 0.01 | 1.79 / 1.77 / 1.75 | 23.1 / 19.8 / 16.5 |
| 0.05 | 8.87 / 8.36 / 7.78 | 98.9 / 51.9 / 29.5 |

The matched-dose approximation is excellent inside R12 (<= 12% spread across
`L_FR`) and breaks in the wall strips, where per-pulse probabilities saturate.
So `L_FR` is a clean frequent-weak/sparse-strong axis where the primary endpoint
lives, and additionally a realised-turnover axis outside it.
