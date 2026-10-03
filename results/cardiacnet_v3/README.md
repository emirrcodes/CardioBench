# CardiacNet v3 (Kaggle kernel version 3)

Same as v2 (fixed reader; consecutive vs uniform sampling; full and noPHI test), plus `probe-clean` / `probe-meanstd-clean`: probes trained with the 42 `PHI*` controls removed from train, val and test (`--exclude_regex /PHI`). Compare them with `probe` / `probe-meanstd` on `eval_noPHI/`.
