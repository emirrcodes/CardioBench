# EchoNet-Dynamic EF — cardiac-cycle-anchored frames (2026-10-04)

ED / ES per video from the traced LV contours in `VolumeTracings.csv` (larger polygon area =
end-diastole); 10,024 of 10,030 videos have a valid pair (ES − ED median 16 frames, ED median at
frame 57, so the standard "first 16 frames" usually contain neither). Embedded on Colab T4 with
`--sampling cycle --keyframes_csv` and merged with the `--sampling both` embeddings of
`results/echonet_dynamic`; TEST = 1,276 videos (the one without valid ED/ES is dropped from every view).

Views: `consecutive16`, `uniform16` (as before); `edes` (the ED and ES frames only),
`halfcycle16` (16 frames from ED to ES), `ed16` (16 frames starting at ED).

- `<model>/summary.csv` — test MAE / RMSE / R² with bootstrap CIs, paired vs `consecutive16/mean`.
- `paired_comparisons.csv` — head effect, coverage effect, and cycle effect (cycle views vs
  `consecutive16`, and `halfcycle16` vs `uniform16`, same head); paired bootstrap B = 10,000, Holm over all rows.

Reproduce: `python scripts/echonet_analysis.py --results results/echonet_cycle --filelist <FileList.csv> --B 10000`.
