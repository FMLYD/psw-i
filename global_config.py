"""
Global configs for running code of time-series imputation survey.
"""



# random seeds for five rounds of experiments
RANDOM_SEEDS = [2024, 2025, 2026, 2027, 2028]
# number of threads for pytorch
TORCH_N_THREADS = 1

# whether to apply lazy-loading strategy (help save memory) to read datasets
LAZY_LOAD_DATA = False
