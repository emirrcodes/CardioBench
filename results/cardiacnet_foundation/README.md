# CardiacNet — PanEcho and EchoPrime (Kaggle `cardiobench-cardiacnet-foundation` v1, code `6a725db`)

PHI* controls removed from train, val and test (test: ASD 42, PAH 102).

- `panecho_frames`: PanEcho's ConvNeXt-T frame encoder through the per-frame pipeline
  (`--sampling both --max_frames 32`), 4 frame views x 4 heads, as for the CLIP models.
- `panecho-video__<view>` / `echoprime-video__<view>`: one embedding per 16-frame clip from PanEcho's video
  backbone (its own temporal transformer) and EchoPrime's MViT-v2-S encoder; logistic-regression probe.
  Clip views: `consecutive16` (frames 0-15), `stride2` (every 2nd of the first 32, EchoPrime's sampling),
  `uniform16`.

Observed: PanEcho's video embedding probes below mean-pooled PanEcho frame features (ASD -0.164, PAH -0.100
AUROC, consecutive16), but not significantly on these small test sets (paired bootstrap p = 0.16 / 0.18).
PanEcho frame features are on par with the CLIP models. Preprocessing follows PanEcho's own (16 consecutive
frames, 224 px, ImageNet normalisation); we crop with a 10 % zoom and resize to 224, PanEcho resizes to 256
and centre-crops 224.
