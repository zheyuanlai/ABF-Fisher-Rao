# Experiment log — adaptive-FR campaign

Times are UTC (`date -u` or git).

## 2026-10-17

* **06:19 — inspection.**
  * main = origin/main = 0f225325. No job of ours running.
  * GPUs 0–3 belong to other users (2/3 at 98 %), so the campaign is CPU only.
  * Load about 10 on 256 cores. 165 GB disk free.
* **06:20 — worktree.** Created `/home/zheyuanlai/ABF-Fisher-Rao-adaptive` on the new branch
  `research/adaptive-fr-equal-budget`, and linked the shared inputs (`scripts/convergence_fr/link_data.sh`).
* **06:20 — EB baseline verified.** 16/16 complete.
  * Configs equal `jobs_EB.json`.
  * Engine sha256 equals the current import (`a165f0a9…`).
* **06:21 — Gate F pilot** (seed 1140000, `pilot_pairing.py`).
  * The FR configuration is bitwise equal to the historical file before FR start.
  * An ABF re-run with `fr_rate = 1.0` is bitwise equal through step 15 000.
  * 14 engine tests pass, including the new inertness test.
* **06:22:39 — PREREGISTRATION §A frozen** (bc78e255); block A launched (16 runs).
* **06:25 — A analysis code committed** (2aa9c026) before any A result. Re-analysing E1 with the extended script
  reproduces its JSON exactly.
* **06:26–06:32 — Exp B development runs:** continuous FR, ITE only, designated exploratory seeds.
  * ITE 300 K K1024 rate 2 (1200000/1), 150 K K256 rate 0.5 (1200100/1), and 300 K K256 rate 1 (1200200/1).
  * Online observables recorded every 0.25 t.u.
* **06:35:41 — A done** (3.415 core-h).
* **06:35 — blind genealogy check of the FR arm:** PASS (ESSw/K ≥ 0.535; lineage ≤ 0.029; 26 replacements per t.u.;
  0 events after the stop).
* **06:35:54 — A frozen analysis** (`analysis/A.json`). All asserts pass, with the ABF arm's `fr_rate` inert.
  * **PRIMARY S_geo 1.281 [0.904, 1.426], FR faster 11/16 → UNRESOLVED.**
  * H1 −25.5 % [−29.9, −16.3], 16/16 → SUPPORTED.
  * H2 e_F(T) −0.9 % [−14.1, +14.0] → inconclusive.
  * H4 +8.3 % [−9.1, +21.4] → inconclusive.
  * Post-ramp Ī_F −15.3 % [−22.4, −4.1].
* **06:36 — A.5 equal budget** (`analysis/A_equal_budget4.json`).
  * The best allocation is K256 for both algorithms.
  * Best FR / best ABF median geometric-mean τ_B = 0.913 [0.646, 1.086] → **Question B UNRESOLVED**; all-in 0.925
    [0.691, 1.075].
* **06:38–07:08 — Exp B development.**
  * Controller R1 implemented, then integrated in both engines with 33 tests. The default paths are unchanged: old
    suites pass, EB and X1 files re-run bitwise, and the LTA suite gives 39 passed.
  * Independent review: no blocker; design fixes applied before the freeze.
  * Weak-rate dev runs, ITE rate 0.2 (seeds 1200300/400/500).
  * Development replay: stop at 5.5–6.5 t.u. at 300 K and 10–10.5 at 150 K.
* **07:10:05 — PREREGISTRATION B (R1 frozen) + C** committed as c3ef3128.
* **07:10:16 — wave 1 launched:** C3, C4 (LTA engine, `launch_lta.sh`) and C5 (ERI).
* **07:12 — D1 bias–variance on existing blocks** (`mechanism/bias_variance.json`).
  * Continuous FR's late-only deposits are biased in every cell: |bias| 0.035–0.13 vs ABF 0.01–0.03.
  * Stopped FR's post-stop deposits are not (0.009–0.015).
* **07:15 — C3 and C4 done**; wave 2 (C1, C2) launched at 07:15.
* **07:15 — C3 LTA 150 K: SUPPORTED**, S_geo 2.52 [2.33, 2.60]; 99 % [2.21, 2.67]; 16/16.
  * R1 stops at a median 21.5 t.u. (20–24.5).
  * e_F(T) −58.6 %.
* **07:15 — C4 LTA 300 K: SUPPORTED**, S_geo 1.46 [1.37, 1.60]; 16/16.
  * R1 stops at a median 16 t.u. (14–18.5).
  * e_F(T) −22.6 %.
* **07:20 — C5 ERI 600 K: CONTRADICTED**, S_geo 0.61 [0.53, 0.71]; 99 % [0.49, 0.73]; 0/16.
  * R1 stops late, at a median 15.25 t.u. (5.5–25.5).
  * e_F(T) +55 %.
  * Controller logs: the residual directed flux after the t ≈ 2 peak stays at 30–60 % of the peak, above the 25 %
    threshold, for 5–10 t.u.
