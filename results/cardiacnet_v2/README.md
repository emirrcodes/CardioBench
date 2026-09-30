# CardiacNet v2 (Kaggle kernel ahmetemirarslan/cardiobench-cardiacnet, version 2)

Fixed NIfTI reader (RGB `PHI*` clips read as `(H, W, T, 3)`), two frame-sampling settings:

- `consecutive/` — first 16 frames (repo default)
- `uniform/` — 16 frames spread over the whole clip

Each run is scored on the full test split (`eval/`) and without the 42 `PHI*` control clips
(`eval_noPHI/`), which come from a different source and are all negatives.
`paired_bootstrap_stats.json` holds AUROC CIs, the all→noPHI drop per model, and paired
uniform−consecutive differences (B = 1000, seed 42, two-sided bootstrap p).
The v1 run (original reader, first 16 frames) is in `../cardiacnet_v1/`.
