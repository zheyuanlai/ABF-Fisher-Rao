`gateway_pre_transport_fixture.npz` — outputs of the ACCEPTED gateway engine (commit abcfaaf, the
engine that produced the gateway/WCA corrected-baseline confirmations) on CPU, generated BEFORE the
horizontal-transport code existed: a two-arm batch (abf, fr_uniform) and the five-arm confirmatory
batch (abf, fr_oracle, fr_estimated, sham_oracle, sham_practical), N=256, 2500 steps, seed 0,
batch_seed 12345.  tests/test_gateway_horizontal_transport.py asserts every legacy path reproduces it
bit for bit.  Regenerate only from that commit.
