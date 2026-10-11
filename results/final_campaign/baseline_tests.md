# Baseline test state at HEAD 189c2408, 2026-10-11T05:45:03Z, before any change of this campaign
## test_eqb_metrics
24 passed in 7.70s
## test_gateway_family_numba
1 failed, 73 passed, 9 skipped in 29.42s
FAILED tests/test_gateway_family_numba.py::test_f12_driver_reuse_policy - ass...
## test_gateway_family_validation
57 passed in 45.89s
## test_gateway_ladder_numba
37 passed, 1 skipped in 16.05s
## test_lta_ladder_numba
39 passed, 1 skipped in 132.54s (0:02:12)
## test_mech_analysis
33 passed in 38.33s
## test_mech_cross_cell
18 passed in 0.66s
## test_mech_pipeline
ERROR conda.cli.main_run:execute(148): `conda run python -m pytest -q --no-header -p no:cacheprovider tests/test_mech_pipeline.py` failed. (See above for error)
FAILED tests/test_mech_pipeline.py::test_cell_configs_and_references_are_current
FAILED tests/test_mech_pipeline.py::test_reuse_cell_scores_real_files_like_the_equal_budget_analysis[matched_free_energy-alpha1]
FAILED tests/test_mech_pipeline.py::test_reuse_cell_scores_real_files_like_the_equal_budget_analysis[conditional_relaxation-lam1]
FAILED tests/test_mech_pipeline.py::test_engine_and_model_checks - eqb_metric...
FAILED tests/test_mech_pipeline.py::test_guard_and_cell_locations - Assertion...
FAILED tests/test_mech_pipeline.py::test_cell_ledger_layout_and_filter - Asse...
FAILED tests/test_mech_pipeline.py::test_reuse_gate_record_rules - TypeError:...
## test_parallel_benchmark
ERROR conda.cli.main_run:execute(148): `conda run python -m pytest -q --no-header -p no:cacheprovider tests/test_parallel_benchmark.py` failed. (See above for error)
FAILED tests/test_parallel_benchmark.py::test_skip_refuse_and_gpu_guard - Run...
FAILED tests/test_parallel_benchmark.py::test_campaign_resource_guard_and_contended_refusal
