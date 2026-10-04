# CardiacNet — temporal heads (Kaggle `cardiobench-cardiacnet-temporal` v2, code `7b0ddcb`)

Same design as `results/echonet_dynamic/`, on classification: EchoCLIP / BiomedCLIP / SigLIP2 embedded once
with `--sampling both --max_frames 32`; `temporal_probe --task classification` compares 4 frame views x 4
heads; the 42 `PHI*` controls are removed from train, val and test. Trained heads = 5-seed ensembles.
Test: ASD 42, PAH 102 videos (PHI removed).

- `summary.csv` — test AUROC with bootstrap CIs per task / model / config.
- `<task>/paired_comparisons.csv` — head and coverage effects, paired bootstrap B = 10,000, Holm over 60 tests per task.
- `<task>/seed_spread.csv` — test AUROC across seeds for the trained heads.

Result: no head effect (ASD mean dAUROC +0.017, PAH -0.016; 0/48 significant after Holm). Uniform frames
help EchoCLIP on PAH (+0.05 to +0.09, uncorrected p < 0.05 in 5 comparisons), none survive Holm.
Contrast with EchoNet-Dynamic EF (7,465 training videos), where the temporal heads won 24/24.
