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
