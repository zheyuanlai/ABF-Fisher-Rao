| device | system | phys. force (us) | FR KDE+score (us) | kernel estimator (us/step) | histogram estimator (us/step) | estimator speedup | kernel total (us/step) | histogram total (us/step) | total speedup | peak mem kernel / hist (MB) |
|---|---|---|---|---|---|---|---|---|---|---|
| cuda | Entropic bottleneck | 163 | 435 | 272 | 120 | 2.28x | 878 | 617 | 1.42x | 0 / 1 |
| cuda | WCA dimer | 767 | 558 | 632 | 167 | 3.78x | 1552 | 1126 | 1.38x | 340 / 340 |
| cpu | Entropic bottleneck | 54 | 228 | 194 | 55 | 3.51x | 506 | 300 | 1.69x | n/a |
| cpu | WCA dimer | 236693 | 1092 | 1446 | 101 | 14.26x | 200357 | 167045 | 1.20x | n/a |

Campaign end-to-end wall time:

```
{
 "entropic_bottleneck": {
  "kernel_wall_s": 33.28638315759599,
  "hist_wall_s": 24.77001104131341,
  "speedup": 1.3438178570884887,
  "R": 40,
  "note": "one batch of 20 seeds x [abf, fr_uniform], 40000 steps, end to end"
 },
 "gateway": {
  "kernel_wall_s": 92.36267609801143,
  "hist_wall_s": 87.01815226580948,
  "speedup": 1.0614184936480417,
  "R": 64,
  "note": "one batch of 32 (init, seed) rows x [abf, fr_uniform], 100000 steps, end to end"
 },
 "wca": {
  "kernel_abf": 236.56022593006492,
  "kernel_fr_uniform": 258.93992113741115,
  "hist_abf": 142.24472783552483,
  "hist_fr_uniform": 166.02210817718878,
  "note": "median sampler seconds per 120000-step run; kernel arms carry the read-out bank (accepted scoring), histogram arms none"
 }
}
```
