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

The FR clock is interval-scaled, `dtau_FR = L_FR dt`, so the integrated dose over
the active window is `gamma * T_FR = 60 gamma` and does **not** depend on `L_FR`.
That is what makes `fr_every` a fair axis instead of a disguised dose axis.
Expected events per replica over the whole window:

| `gamma` | dose | in R12 (`\|S\| ~ 3`) | in the wall strips (`\|S\| ~ 40`) |
| --- | --- | --- | --- |
| 0.002 | 0.12 | 0.36 | 4.8 |
| 0.01 | 0.60 | 1.8 | 24 |
| 0.05 | 3.00 | 9.0 | 120 |
