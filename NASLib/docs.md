`run_nb201_comparison.py`: Patch utilities present in the codebase and compare the BANANAS surrogate-assisted NAS algorithm mainly in two different modes: using a default Path encoding to convert NAS-Bench-201 neural architectures into numerical representations, versus using COLE (our process) to convert neural architectures into numerical representations. Other configuration changes are also supported, such as the regressor model used, NAS search parameters, and a non-surrogate-assisted baseline.
  - This file can also run multiple trials at once. Each experiment is started with a different random seed. The seed used for the Nth trial is equal to N * (original seed).

`naslib/predictors/llm_enhanced_201.py`: Provides functionality to retrieve pre-computed COLE for any NAS-Bench-201 architecture. Acts as a wrapper to other predictors (MLP, Xgboost, etc).

`results_paper.py`: Provides visualization and statistical testing capability for executed NAS trials.
