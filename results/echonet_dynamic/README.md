# EchoNet-Dynamic EF — frame coverage × temporal heads (2026-10-03)

Frozen encoders (EchoCLIP, BiomedCLIP, SigLIP2; later PanEcho frames), embedded once on Colab T4 with
`--sampling both --max_frames 32`; heads trained with `src.linear_probe.temporal_probe`
(official TRAIN / VAL / TEST = 7465 / 1288 / 1277 videos). Trained heads (attn, transformer)
are a 3-seed ensemble; per-seed predictions are in `predictions/*__seedK.csv`.

- `<model>/summary.csv` — test MAE / RMSE / R² with bootstrap CIs, paired vs `consecutive16/mean`.
- `paired_comparisons.csv` — head effect (vs mean at the same frames) and coverage effect
  (uniform vs consecutive at the same head), paired bootstrap B = 10,000, Holm over all 80 tests
  (`panecho_frames`, added 2026-10-04: PanEcho ConvNeXt-T frame encoder, same sampling).
- `seed_spread.csv` — test MAE across seeds for the trained heads.

Reproduce the analysis: `python scripts/echonet_analysis.py --results results/echonet_dynamic --filelist <FileList.csv> --B 10000`.
