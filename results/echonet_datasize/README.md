# EchoNet-Dynamic — does the temporal-head gain survive small training sets? (2026-10-04)

`scripts/echonet_datasize.py` on the cached EchoNet embeddings (`.work/echonet/embeddings/`): TRAIN subsampled
to 150 / 300 / 1,000 / 3,000 videos (3 random subsets each), VAL shrunk to 16 % of that (as in CardiacNet),
TEST = full 1,277. Views consecutive16 and uniform16; heads mean (ridge), attention, transformer (3-seed ensembles).

- `datasize_runs.csv` — test MAE of every run.
- `datasize_summary.csv` — per model / size / view / head: mean dMAE vs mean pooling, wins over subsets.

Result: the temporal heads beat mean pooling in 125 of 144 (head, view, subset) runs. EchoCLIP wins 12/12 at
every size (-0.2 MAE at 150 videos, -0.5 at 3,000); BiomedCLIP 10/12 at 150; SigLIP2 is mixed at 150 and 1,000.
So small data shrinks the gain but does not remove it: CardiacNet's null result (same sizes) is not explained by
size alone — the task (EF vs ASD/PAH classification) matters.
