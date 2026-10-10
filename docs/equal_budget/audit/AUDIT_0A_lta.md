# Porting spec: ethane/LTA ABF(+FR) histogram sampler, torch (`src/lta/core_lta.py`) to a numba CPU engine

All facts below come from reading the code at HEAD 7bbbe23 and from CPU checks (float64, no GPU). Anything not read or measured directly is marked **INFERENCE**. No repo file was modified. Scratch files are in `/tmp/claude-1008/-home-zheyuanlai-ABF-Fisher-Rao/b25e8a63-d0d8-4935-af2b-7d50d5f3ea3b/scratchpad/audit/`:
- `replay_port.py`: an independent numpy port written from this spec, plus a recorder for the torch engine's random draws.
- `rng_checks.py`, `rng_checks2.py`: probes of torch RNG behaviour.

## 0. Verification of this spec

I implemented the pseudo-code below in numpy (`replay_port.py`). The recorder monkeypatches `torch.randn/rand/randint/randperm/multinomial` and captures every draw `core_lta.run_sampler` makes. The port then consumes that same stream. Setup: CPU float64, T = 300 K, R = 2, N = 64 (cap = 1, so the randperm path runs), 1000 steps, warmup/burn-in 100, fr_start 150, fr_rate 20, histogram estimator.

| arm | discrete outputs (events, event_counts, n_transitions, n_cage_crossings, birth/death hists, ancestor ESS, frac_window) | max abs diff, floats |
|---|---|---|
| abf | all identical | pmf 7.1e-15, mean_force 7.5e-14, u_of_z 6.0e-14, fsum_prod 2.2e-11 (scale 5e3), p_hat 6.7e-16 |
| fr_uniform (65 events) | all identical | pmf 5.4e-14, mean_force 5.5e-13, fsum_prod 1.9e-11 |
| fr_sham (65 events) | all identical | pmf 4.4e-14, mean_force 5.2e-13, fsum_prod 5.1e-11 |

Gap: 1000 steps from cage starts gave 0 cage crossings. The crossing branch was therefore not exercised (see §8, test A3).

Draw counts per 1000 steps:
- abf: `randint` 1 + `randn` 1002.
- fr_uniform: adds `rand` 171 (= n_opps), `multinomial` 65, `randperm` 1.
- fr_sham: `randint` 66, `randperm` 65, no `rand`.

From the production npz I reproduced the published 300 K numbers: e_F(T) medians 0.0866 / 0.0654 / 0.0838, I_F 22.98 / 19.85 / 23.36, ΔI_F −14.10 %.

## 1. Constants and configuration

**Physics** (`core_lta.py:76-89`; framework from `cache/lta/framework.npz`):
- a = 11.919 Å (`a_pseudo`), L = 23.838 Å (`box`), O = `o_pos`, shape (384, 3).
- eps = 93.0·KB with KB = 0.008314462618 (so eps = 0.773245 kJ/mol); σ = 3.48 Å; rc = 10 Å.
- r0 = 1.54 Å; k_bond = 400 kJ/mol/Å² (energy ½k(r−r0)²).
- β = 1/(KB·T).
- v_rc = 4·eps·(sr6c² − sr6c), with sr6c = (σ/rc)**6 (`:153-154`). Used only in the potential.
- Production runs in float64 on CUDA: `LTASystem` defaults to float64 (`:143`) and the runner does not override it (`run_lta_histogram.py:86-91`). All five production logs say "device cuda", GPU 3.

**Sampler** (`configs/lta_histogram/campaign.json:18-36`, passed by `build_sim`, `run_lta_histogram.py:73-83`):

| field | value | note |
|---|---|---|
| n_replicas N | 1024 | |
| n_steps | 300000 | loop runs step = 0..300000; 300000 moves |
| dt | 2e-4 | |
| save_every | 3000 | 101 saves |
| n_grid | 180 | dphi = 2π/180 = 0.0349066 rad = 0.0662 Å |
| abf_estimator | "histogram" | |
| abf_warmup_steps | 20000 | ramp |
| abf_force_clip | 60.0 | bias read clip ±60; deposit clip ±480 |
| estimator_burn_in_steps | 20000 | production accumulators |
| fr_start_steps | 20000 | the `LTASimConfig` default is 40000 (`:110`); always pass 20000 |
| fr_every | 5 | |
| score_clip | 2.0 | |
| max_event_fraction | 0.02 | cap = int(0.02·N) = 20 |
| kde_bandwidth | 0.10 rad | FR marginal KDE |
| abf_bandwidth / target_ema_rate | 0.05 / 0.005 | unused by abf / fr_uniform / fr_sham with histogram |
| abf_bias_scale | 1.0 | default |
| window_half / cage_min | 1.5 / 4.0 Å | defaults (`:116-117`), not in the campaign file |

**Per T** (`campaign.json:37-82`):

| T (K) | rng_seed | seed labels | fr_rate | kT (kJ/mol) | noise sd √(2·dt·kT) (Å/step) | max death prob per opportunity, S = 2 |
|---|---|---|---|---|---|---|
| 80 | 20260901 | 1000-1015 | 0.10 | 0.66516 | 0.016311 | 2.0e-4 |
| 150 | 20260902 | 1020-1035 | 0.20 | 1.24717 | 0.022335 | 4.0e-4 |
| 225 | 20260903 | 1040-1055 | 0.20 | 1.87075 | 0.027355 | 4.0e-4 |
| 300 | 20260904 | 1060-1075 | 0.20 | 2.49434 | 0.031587 | 4.0e-4 |
| 350 | 20260905 | 1080-1095 | 0.20 | 2.91006 | 0.034118 | 4.0e-4 |

**Derived constants** (keep these exact evaluation forms):
```
PI = math.pi; TWO_PI = 2.0*PI; EPS = 1e-12
dphi = TWO_PI / ng
grid[j] = -PI + (j + 0.5)*dphi                                     # periodic.py:25-29
c_phi = 2.0*PI / a          # phi = c_phi * x_COM                    core_lta.py:198
c_f   = -(a / (2.0*PI))     # f_loc = c_f*(F0x+F1x)                  :209
gpx   = PI / a              # d(phi)/d(x_bead)                       :211
ns    = math.sqrt(2.0*dt/beta)                                      # :383
cap   = int(max_event_fraction * N)                                 # alkanes/core.py:131
dt_eff = dt * max(int(fr_every), 1)       # = 1e-3                  core.py:130
K_kde[i,j] = exp(-0.5*(d/0.10)**2),  d = (grid[i]-grid[j]) - TWO_PI*rint((grid[i]-grid[j])/TWO_PI)
             # periodic.py:32-41: UNNORMALISED, single minimum-image Gaussian (no sum over images); rint = half-to-even like torch.round
```

## 2. Kernels (written in the order the arithmetic is evaluated)

**Force on one molecule** (`core_lta.py:164-180`). Coordinates are unwrapped. The bond uses no minimum image. Each bead–O displacement uses the minimum image.
```
def mol_force(q, O, F):                      # q (2,3) -> F (2,3)
    d = q[0]-q[1]; r = max(sqrt(dx*dx+dy*dy+dz*dz), EPS)   # torch .norm; internal order INFERENCE
    c = (-kb)*(r - r0)                        # "-p.k_bond * (r - r0) * dr / r" parses as ((-k)*(r-r0))*dr / r
    fb = (c*d)/r ;  F[0] = fb ;  F[1] = -fb    # F starts at 0: 0+fb, 0-fb
    for b in 0,1:
        acc = 0
        for k in range(384):
            dv = q[b]-O[k]; dv = dv - L*rint(dv / L)      # divide by L; do NOT multiply by 1/L
            r2 = dvx*dvx+dvy*dvy+dvz*dvz
            if r2 < rc*rc:                                # strict
                inv = 1.0/max(r2, EPS)
                s = (sig*sig)*inv ;  sr6 = s*s*s          # torch pow(.,3) = x*x*x (INFERENCE on the ATen fast path)
                coef = ((24.0*eps) * ((2.0*sr6)*sr6 - sr6)) * inv
                acc += coef*dv
        F[b] += acc          # torch sums all 384 entries (out-of-cutoff ones are exact 0) in its own order -> ulp differences
```

**Potential energy.** Needed only for `u_of_z`, every step after burn-in (`:182-193`):
`0.5*kb*(r-r0)**2` (no clamp on r) `+ Σ_{in cutoff} ((4.0*eps)*(sr6*sr6 - sr6) - v_rc)`.

**CV** (`:196-199`). x = (q[0,0] + q[1,0])/2, which is torch `mean` and bit-equal to 0.5·(x0 + x1). Then:
```
t = c_phi*x + PI;  m = math.fmod(t, TWO_PI);  if m != 0.0 and m < 0.0: m += TWO_PI;  phi = m - PI
```
This form is bit-identical to `torch.remainder` on 2,000,010 values, edge cases included (0 mismatches). The floor-based form used in `src/lta_validation.py:108-111` differs on 377,312 of those values, by up to 7.1e-15.

**Bin index** (`core_lta.py:245-249`, identical to `periodic.py:46-47,55-56`): `j = floor((phi + PI) / dphi) % ng` (floor-mod). This gives index 0 if phi + π rounds to 2π. Do not use `(phi+π)/(2π)·ng` with clipping, which is `lta_validation.pbin` at `:127-134`. It matched on 2e6 samples, but nothing guarantees it (**INFERENCE**).

**Local mean force** (`:209`, `:423-424`): `f = clip(c_f*(F[0,0]+F[1,0]), -480.0, 480.0)` (clip = abf_force_clip·8). The physical force F used in the move is never clipped.

**Histogram PMF** (`:257-269`):
```
g0 = G - mean(G);  Fr = cumsum(g0*dphi);  Fc = Fr - (0.5*g0)*dphi;  F = Fc - mean(Fc)
```

**Periodic interpolation** (`periodic.py:80-95`):
```
x = (phi - grid[0]) / dphi;  i0 = floor(x);  fr = x - i0
v = (1-fr)*P[i0 % ng] + fr*P[(i0+1) % ng]        # i0 in {-1..179}; -1 wraps to 179
```

**Density normalisation** (`periodic.py:67-71`): `p = max(p, 0); p / max(sum(p)*dphi, EPS)`.

## 3. Per-run state and initial conditions

**Per-run state:**
- q[N,2,3] (unwrapped; never wrapped).
- fsum, csum (all steps); fsp, csp, usp (step ≥ burn-in). Each has ng entries.
- anc[N] = arange(N) for fr_uniform and fr_sham.
- has_left_cage[N] = False, rep_crossings[N] = 0, prev_reg[N] (unset at step 0), trans = 0.
- birth_hist[ng], death_hist[ng], total_repl.
- score_std_sum, score_absmax, n_score; event_counts list.
- References: `core_lta.py:384-399`.

**Initial conditions** (`:214-228`). Drawn from `gen_dyn` once for the whole batch:
```
cages = a*[[i+.5, j+.5, k+.5] for i in 0,1 for j in 0,1 for k in 0,1]   # i outermost; 8 alpha-cage centres; phi = -pi there
pick = randint(0, 8, (R*N,))           # draw 1
com  = cages[pick] + 0.5*randn(R*N,3)  # draw 2
u    = randn(R*N,3);  u = u / max(|u|, EPS)                             # draw 3
q    = stack([com + 0.77*u, com - 0.77*u], dim=1).reshape(R, N, 2, 3)   # 0.77 = 0.5*r0; run r = flat rows r*N .. r*N+N-1
```

## 4. Main loop, exact order (`core_lta.py:418-577`; methods abf, fr_uniform, fr_sham; histogram estimator)

```
for step in range(n_steps + 1):
  # (1) forces, CV, local mean force at the PRE-move positions                 :419-424
  for i: F[i] = mol_force(q[i]); phi[i] = cv(q[i]); f[i] = clip(c_f*(F[i,0,0]+F[i,1,0]), ±480); j[i] = bin(phi[i])
  # (2) deposits: ALL N replicas, before any bias is read                      :426-432
  for i: fsum[j[i]] += f[i]; csum[j[i]] += 1.0
  if step >= burn_in (20000):
      for i: fsp[j[i]] += f[i]; csp[j[i]] += 1.0; usp[j[i]] += U(q[i])
  # (3) bias read from the ALL-STEPS accumulators, this step's deposit included  :434-446
  ramp = min(1.0, step / max(warmup, 1))          # Python float; 0 at step 0, 1 from step 20000
  for i: c = csum[j[i]]                           # always >= 1 here (own deposit), so the "0 if empty" branch never fires for the bias
         G = fsum[j[i]] / max(c, 1.0) if c > 0 else 0.0
         bx[i] = ((1.0*ramp) * clip(G, -60, 60)) * gpx      # x-component, same on both beads
  #   (A_hat = histogram_pmf(Gamma) and B_n are computed every step in torch (:435-438) but feed only
  #    fr_oracle/fr_estimated targets and diagnostics: skip them for these three arms)
  # (4) region bookkeeping at pre-move positions                              :456-463, region_index :231-239
  for i: z = (abs(phi[i])*a)/(2.0*PI)             # distance from the window plane, Å
         reg = 2 if z < 1.5 else (0 if z > 4.0 else 1)   # 2 = window, 1 = neck, 0 = cage (|z| > 4; cage centre at a/2 = 5.96 Å)
         if step > 0: trans += (reg != prev_reg[i]);  rep_crossings[i] += (has_left_cage[i] and reg == 0)  # uses the OLD flag
         has_left_cage[i] = False if reg == 0 else (has_left_cage[i] or reg == 2)
         prev_reg[i] = reg
  # (5) save: state at the START of this step; accumulators include this step   :465-501
  if step % save_every == 0 or step == n_steps: SAVE()        # §6
  if step == n_steps: break
  # (6) Euler-Maruyama move                                                   :521-522
  xi = randn((R,N,2,3), gen_dyn)                  # one call per step, whole batch
  q = (q + dt*(F + bias)) + ns*xi                 # bias = (bx, 0, 0) per bead; y,z: F + 0.0
  # (7) FR at POST-move positions                                             :524-577
  nxt = step + 1
  if is_fr and nxt >= fr_start and (nxt - fr_start) % fr_every == 0:          # nxt = 20000, 20005, ..., 300000
      phin = cv(q)                                # post-move, pre-replacement
      events = SHAM(...) if sham else FR_BIRTH_DEATH(score(phin))            # §5
      event_counts.append(n);  total_repl += n    # appended at EVERY opportunity, including n = 0
      for k in range(n): d = deaths[k]; s = sources[k]
          birth_hist[bin(phin[s])] += 1; death_hist[bin(phin[d])] += 1         # binned at phin
          has_left_cage[d] = has_left_cage[s]; rep_crossings[d] = rep_crossings[s]; prev_reg[d] = prev_reg[s]
```

Counts: n_opps = (300000 − 20000)//5 + 1 = 56001, matching the npz `event_counts` shape (56001, 16).

Timing relationships:
- The first FR opportunity follows the move of step 19999.
- Every save at step ≥ 21000 sees a state in which an FR opportunity has just been applied.
- `csum_prod.sum() > 0` is a global check over R (`:466`). It is equivalent to step ≥ 20000. Saves up to t = 3.6 (step 18000) report the all-steps estimator; saves from step 21000 report the production estimator.

## 5. Fisher–Rao and sham (`alkanes/core.py:102-160`, `core_lta.py:272-319`)

**Target** for fr_uniform (and for the sham's diagnostics only): `qv = 1.0/max(ng*dphi, EPS)` in every bin (`normalize_density(ones)`, `core_lta.py:279-282`).

**Score** (`core.py:110-121`, `:102-107`). Computed once per run per opportunity on phin:
```
cnt = bincount(bin(phin), ng)
p   = normd(K_kde @ cnt)            # torch: counts @ K.T (BLAS); bandwidth 0.10 rad
p_at[i] = interp(p, phin[i]);  q_at[i] = interp(qv-array, phin[i])     # q_at is qv to within an ulp
kl  = sum_j p[j]*(log(max(p[j],EPS)) - log(max(qv,EPS))) * dphi
raw[i] = log(max(p_at[i],EPS)) - log(max(q_at[i],EPS)) - kl
s = raw - mean(raw)
repeat 3: s = clip(s, -2.0, 2.0); s = s - mean(s)     # the final |s| can slightly exceed 2; the clip binds in every production run (absmax 2.0)
diagnostics: score_std_sum += std(s, ddof=0); score_absmax = max(., max|s|); n_score += 1
```

**Birth–death** (`core.py:124-160`):
```
if cap < 1 or fr_rate <= 0: no events, NO RNG draw            # :137-138. N < 50 => cap = int(0.02N) = 0 => FR silently OFF
dw = max(s,0); bw = max(-s,0)
dp[i] = 1.0 - exp(((-fr_rate)*dw[i])*dt_eff) if dw[i] > 0 else 0.0      # :141
u = rand((R,N), gen_fr)       # ALWAYS drawn for the whole batch once the guard passes   :143
fire = u < dp                                                              # :144
for r in range(R):            # runs in order; per-run draws come from the shared gen_fr
    if sum(bw[r]) <= EPS or sum(dw[r]) <= EPS: continue                    # :146, after the shared rand
    di = ascending indices with fire;  n = len(di);  if n == 0: continue
    if n > cap: di = di[randperm(n, gen_fr)[:cap]]; n = cap                # di now in perm order
    src = multinomial(bw[r], n, replacement=True, gen_fr)                  # iid categorical, P(i) ∝ bw[i]
    q[r, di] = q_old[r, src]   (whole molecule, both beads, unwrapped image);  anc[r, di] = anc_old[r, src]
```
- Deaths need s > 0 and sources need s < 0, so the two sets are disjoint. In-place copying is therefore equal to the torch copy from the pre-event state. The bookkeeping copy (`index_select` then assign) has the same property.
- The cap never bound in production: the maximum events per opportunity per run was 3, against cap 20, at every T. Histogram at 300 K: {0: 885604, 1: 10332, 2: 80}.

**Sham** (`core_lta.py:294-319`). It consumes the partner fr_uniform's `event_counts[k_opp]`:
```
for r: n = min(int(shadow[k_opp, r]), N-1); if n < 1: continue (no draw)
       perm = randperm(N, gen_fr); di = perm[:n]; surv = perm[n:]
       src = surv[randint(0, N-n, (n,), gen_fr)]     # uniform among survivors, with replacement
       copy as above
```
The sham draws no `rand(R,N)` and computes no score. Its `fr_score_std` stays 0.

**RNG streams:**
- `gen_dyn = Generator(device).manual_seed(rng_seed)`; `gen_fr = …manual_seed(rng_seed + 987654321)` (`:347-348`).
- The ABF, FR and sham arms share the IC and the noise stream. The FR selection stream is separate.
- CPU torch multinomial semantics, measured on torch 2.12:
  - n = 1 matches `argmax(w / Exp(1))` 200/200, where `exponential_` uses the same generator state as `rand(N)` and is within 8.9e-16 of −log1p(−u).
  - n ≥ 2 matches 200/200 against n `rand` uniforms with a left `searchsorted` on the normalised cumsum.
  - The CUDA multinomial and Philox are different algorithms (**INFERENCE**).

## 6. Saves, outputs, and how the analysis reads them

**SAVE()** (`core_lta.py:465-501`):

| key | content | shape in npz |
|---|---|---|
| steps, times | step, step·dt | (101,) |
| mean_force | Γ = where(ce>0, fe/max(ce,1), 0) from (fsp, csp) if step ≥ burn-in, else (fsum, csum) | (101,R,180) |
| pmf | histogram_pmf(that Γ) | (101,R,180) |
| eff_counts | copy of csum (all steps) | (101,R,180) |
| p_hat | normd(K_kde @ bincount(phi pre-move)) | (101,R,180) |
| kl_uniform | Σ p·(log max(p,EPS) − log(1/(2π)))·dphi | (101,R) |
| q_target / pq_l2 / kl_pq | uniform qv; √(Σ(p−q)²·dphi); Σ p·(log p − log q)·dphi, with clamps. NaN for abf | (101,R,180) / (101,R) |
| ancestor_ess, n_unique_ancestor, max_ancestor_frac | from bincount(anc): 1/Σw², #>0, max/sum (`core.py:163-175`). abf gets NaN, N, NaN | (101,R) |
| repl_cumulative | total_repl | (101,R) int64 |
| frac_cage / neck / window | mean(reg==0/1/2) at pre-move phi | (101,R) float32 |

**Final outputs** (`:579-597`):
- `total_replacement_events`, `n_transitions`, `n_cage_crossings` (= Σ_slots rep_crossings): (R,) int64.
- `birth_hist`, `death_hist`, `u_of_z` = usp/max(csp,1), `u_counts` = csp, `final_eff_counts` = csum, `fsum_prod`: (R,180).
- `fr_score_std` = score_std_sum/max(n_score,1); `fr_score_absmax`: (R,).
- `event_counts`: (56001,R) int32; (0,R) for abf.
- Scalars: `method`, `grid`, `dphi`, `a_pseudo`, `box`, `abf_estimator`, `runtime_seconds`.
- `F_target_ema` is None, so `write_npz` drops it (`run_lta_histogram.py:94-99`).
- The runner adds `seeds`, `config_hash` (300 K: b9978052868b) and `meta` JSON with method, seeds, rng_seed, fr_rate, abf_estimator, n_grid, temperature_K, shadow_of, … (`:140-147`).
- The analyzer requires equal `times` and `seeds` across arms, `meta.temperature_K`, and `abf_estimator == "histogram"` (`analyze_lta_histogram.py:334-366`).

**Analysis.** Reproduced from the npz; the numbers in §0 match the doc.
- e_F per (save, seed): d = pmf − F_ref, where F_ref is the reference interpolated onto the engine grid with `np.interp` and periodic extension (`analyze_uniform_lta.py:51-57`; identical grids, so the error is 1e-14). Then d −= mean(d) and e_F = √mean(d²) (`:60-64`).
- I_F = `np.trapezoid(err, times)` over the 101 saves, t = 0..60 (`analyze_lta_histogram.py:383`). e_F(T) = err[−1].
- Contrasts: paired 100·(arm − base)/base, median, bootstrap CI (10000 resamples, seed 20260829 for I_F, +1 for the final), wins (`:135-142`, `:446-449`).
- Crossings per replica = Σ n_cage_crossings / (R·N) (`:394-398`).
- Health: per seed, min over saves with step ≥ fr_start of ESS/N and max of wmax; then the median across seeds (`:409-428`).
- First window visit: first save with frac_window > 0 (`:560-569`).
- A "cage crossing" is a cage → window → cage return to |z| > 4. It can be back into the same cage, so it is not necessarily a translocation.
- Reference npz `results/uniform_campaign/lta/reference/reference_T{T}.npz` has keys F, grid_phi (180), U, TS, kT, dF_barrier, ….

## 7. Replicas R and N, and initial-state reproducibility

- `seeds` are labels only. `run_sampler` uses only `len(seeds)` (`:342`) and, for the movie record, `seeds[r0]` (`:602`).
- The 16 runs at one T are rows of a single batch drawn from one generator.
- Runs are independent apart from the RNG: each has its own fsum/csum and its own N-replica estimator.
- Run r's state depends on R and on its position r in the batch:
  - IC are flat rows r·N..r·N+N−1 of draws of size R·N. The two `randn` calls follow `randint(R*N)`. Measured: run 0 of R = 1 ≠ run 0 of R = 2, max diff 2.63 Å.
  - Noise is slice [r] of each per-step `randn((R,N,2,3))`.
  - FR selection draws for run r depend on what runs < r consumed at the same opportunity.
- **Reproducing a production run (rng_seed, r) is not practical:**
  - The generators were CUDA (Philox). CPU torch uses MT19937, so a CPU torch generator cannot reproduce them.
  - The element-to-counter mapping of torch's CUDA distribution kernels depends on launch configuration, i.e. GPU model and torch version (**INFERENCE**).
  - Production itself is not bitwise reproducible: CUDA float64 `scatter_add_` atomics change fsum/fsum_prod/usp bits. The doc records that a CUDA re-run diverges after t ≈ 4 (`docs/LTA_HISTOGRAM_REPLICATION.md:400-404`).
  - Production ICs could be recovered by running `initial_conditions(R, N, cuda_gen)` once on the same GPU type and torch version, then passing q0 to numba. The noise stream (R·N·6 doubles × 3e5 steps) cannot reasonably be shipped.
- **Recommendation:** equivalence with production is in law. The numba engine should take externally generated random arrays, e.g. PCG64 from `SeedSequence([rng_seed, r])`, so each run is independent of R. The same kernel can then take torch-recorded arrays in replay mode.

## 8. Hazards for exact equivalence, and the proposed test

**Hazards:**
1. **Reduction order.** The LJ sum over 384 O, means over N in the score, histogram_pmf means and the BLAS matmul in the KDE all produce ulp-level differences. Observed effect: about 1e-13 relative after 1000 steps (§0). Long horizons may amplify this chaotically (**INFERENCE**).
2. **phi wrap** must be fmod-based; **bin index** must use division by dphi; **min image** must use `rint(d / L)`. `src/lta_validation.py` differs on all three (`:64-68` multiplies by invL, `:108-111` floor-wrap, `:127-134` pbin). It also uses different force forms (`s2/r2` at `:71`; `c = -kb(r-r0)/r` at `:97`). Its test asserts only 1e-10 relative on forces (`tests/test_lta_validation.py:33-36`), although `LTA_VALIDATION.md:22` says 1e-13. Reuse it as a physics oracle, not as a bit-level template.
3. **Torch-on-CPU bug** in the reference (histogram branch only). `support_of(csum)` returns `csum` itself, and `.cpu().numpy()` on a CPU tensor shares memory (`core_lta.py:373-374`, `:473`). As a result, every saved `eff_counts` entry equals the final csum. Measured: [976, 976, 976, 976] where [16, 336, 656, 976] was expected. CUDA production is correct (1024, 3.07e6, …). The numba engine should store copies. In a CPU replay, compare only `eff_counts[-1]`.
4. **cap = int(0.02·N) is 0 for N < 50.** FR is then silently disabled, with zero events and no draws (`core.py:131,137`). This matters for any N ladder. The sham has no cap (n ≤ N−1).
5. **`n_cage_crossings` follows lineage.** A clone overwrites the dying slot's count with the source's count (`core_lta.py:558-559`). The output is a sum over the surviving slot histories, not a count of crossing events. Replicate this for equivalence. If true event counts are wanted, add a separate counter.
6. **Ordering traps:**
   - deposit before read (§4 step 2 before step 3);
   - the bias uses all-steps accumulators, the report uses production accumulators;
   - FR runs on post-move positions;
   - the FR `rand(R,N)` is drawn even for runs that are skipped;
   - with the cap active, deaths follow perm order.
7. **Config defaults.** `LTASimConfig` defaults are not the campaign values (fr_start 40000, fr_rate 0.10, `abf_estimator` "kernel"). `config_hash` includes `fr_rate` and `rng_seed` even for abf.

**Equivalence test:**
- **A. Exact replay on CPU float64.** This is the harness in `replay_port.py`, ready to reuse.
  - Run `core_lta.run_sampler` with `LTASystem(..., torch.device("cpu"), torch.float64)` while monkeypatching the five `torch.*` random functions to record their outputs in call order. Feed the recording to numba in replay mode:
    - IC from (randint, randn, randn);
    - one `randn` (R,N,2,3) per step;
    - per FR opportunity: `rand` (R,N), then for each run with n > 0, an optional `randperm(n)` and the `multinomial` index output (replay the indices; torch's internal exponentials are not visible);
    - sham: `randperm(N)` and `randint`.
  - Use R = 1, or R = 2 in lockstep.
  - Cases:
    - A1: T = 300 and T = 80, N = 64 (cap 1, randperm path), 5000 steps, warmup/burn-in 200, fr_start 300, fr_rate 20; all three arms, with the sham fed the recorded fr_uniform counts.
    - A2: N = 1024, R = 1, 1500 steps.
    - A3: monkeypatch `system.initial_conditions` to return a q0 with molecules at |z| < 1.5, so window/crossing bookkeeping runs.
  - Pass criteria:
    - Exact: csum, csp, event_counts, death and source index lists, anc, n_transitions, n_cage_crossings, birth_hist/death_hist, frac_*.
    - Floats: max |Δq| ≤ 1e-9 Å, pmf ≤ 1e-10 kJ/mol, mean_force ≤ 1e-9, fsum/fsp relative ≤ 1e-12, p_hat ≤ 1e-13.
    - Measured so far at 1000 steps: pmf 5.4e-14, mean_force 5.5e-13, p_hat 4.4e-16.
- **B. Law tests of numba's own RNG paths** with fixed score vectors, chi-square at p > 0.01:
  - firing frequency equals 1 − exp(−rate·S⁺·dt_eff);
  - sources ∝ S⁻ (with replacement);
  - the cap subset is uniform;
  - sham deaths are uniform without replacement and sources uniform among survivors;
  - noise moments.
- **C. Statistical equivalence with the CUDA production.** Run the full campaign configuration at 300, 150 and 80 K, 16 independent seeds per arm. Compare against the production npz (pairing is impossible because the noise differs):
  - medians of e_F(T) and I_F;
  - total events (300 K: 10492; 150 K: 12562; 80 K: 10605);
  - crossings per replica (2.66 / 0.92 / 0.226 for abf);
  - median min ESS/N.
  
  Use bootstrap CIs of the median ratio with TOST margins: I_F ±5 %, events ±5 %, crossings ±5 %, e_F(T) ±15 % (the per-seed endpoint is noisy). In addition, the within-engine paired ΔI_F should fall inside the production CIs: 300 K −14.10 % [−15.10, −10.98], 150 K −28.07 % [−29.21, −26.25], 80 K −35.14 % [−36.23, −33.35]. The sham vs abf contrast should be neutral, |ΔI_F| ≤ 1.2 %.