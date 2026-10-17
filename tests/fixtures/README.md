`gateway_pre_transport_fixture.npz` — outputs of the ACCEPTED gateway engine (commit abcfaaf, the
engine that produced the gateway/WCA corrected-baseline confirmations) on CPU, generated BEFORE the
horizontal-transport code existed: a two-arm batch (abf, fr_uniform) and the five-arm confirmatory
batch (abf, fr_oracle, fr_estimated, sham_oracle, sham_practical), N=256, 2500 steps, seed 0,
batch_seed 12345.  tests/test_gateway_horizontal_transport.py asserts every legacy path reproduces it
bit for bit.  Regenerate only from that commit.

`wca_pre_relax_fixture.npz` — outputs of the ACCEPTED WCA engine (commit d3fc93e) on CPU float64,
generated BEFORE the targeted-solvent-relaxation code existed: the tiny dimer of
tests/test_wca_sham.py (n_dim 4, N 64, 3000 steps, seed 7), arms abf and fr_uniform with a
read-out bank at (0.025, raw).  tests/test_wca_targeted_relax.py asserts the legacy paths reproduce
it bit for bit.  Regenerate only from that commit.

`eb_pre_histogram_fixture.npz` — outputs of the ACCEPTED entropic-bottleneck engine (commit 64567ea) on
CPU float64, generated BEFORE the histogram ABF estimator existed: a two-seed batch (seeds 3, 4,
batch_seed 5) of (abf, fr_estimated, fr_uniform), N=64, 400 steps, save every 100.
tests/test_histogram_abf.py asserts the kernel path (default and `abf_estimator="kernel"`) reproduces
it bit for bit.  Regenerate only from that commit.

`histogram_abf_head_ids.json` — the legacy WCA `run_id` / `spec_hash` / `config_hash` strings at commit
64567ea (the accepted Case IX confirmation spec and the tiny test SimConfig), so adding the
`abf_estimator` / `abf_n_bins` fields can be shown not to orphan any completed run.

## `lta_pre_histogram_fixture.npz`

Generated 2026-10-03 from `src/lta/core_lta.py` as of `56ab774` (BEFORE the histogram-estimator / sham /
snapshot additions): CPU float64, `LTASimConfig(n_steps=1500, n_replicas=64, save_every=500,
abf_warmup_steps=200, estimator_burn_in_steps=200, fr_start_steps=300, fr_rate=20.0, rng_seed=11)`,
arms `abf` and `fr_uniform` (84 replacement events), seeds [0, 1]; keys `<arm>/pmf, mean_force, p_hat,
eff_counts, total_replacement_events, n_cage_crossings, ancestor_ess, kl_uniform, u_of_z, birth_hist,
death_hist`, plus `sim_json` and `config_hash`. `tests/test_lta_histogram.py` asserts the current kernel
engine reproduces every array bitwise and keeps the hash (`git add -f`, like the other fixtures).

## `eri_pre_selection_fixture.npz` + `eri_pre_selection_hashes.json`

Generated 2026-10-17 from `src/eri/engine.py` as of `6716aefc` (BEFORE the `Sbond` / `Suz2` fibre accumulators and
the read-only selection diagnostic `diag_sel` were added): CPU, `make_cfg(N=64, n_steps=3000, save_every=500,
warmup_steps=500, fr_start_steps=500, fr_every=10, seed=23, fr_rate=20.0, trust_count=200.0)`, variant `gard_tait`,
arms `abf` and `fr` (45 replacement events); keys `<arm>_{sv_C, sv_M, sv_h, sv_i, sv_u, sv_cond, q, anc, BH, DH, Seta,
Sux2, born, ist}`. The json holds the sha256 of every array's bytes. `tests/test_diag_sel.py` asserts the current
engine's default path reproduces every array bitwise (`git add -f`, like the other fixtures). Regenerate only from
that commit.
