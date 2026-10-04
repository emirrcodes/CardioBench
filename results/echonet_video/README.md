# EchoNet-Dynamic EF — video-level PanEcho and EchoPrime embeddings (2026-10-04)

One embedding per 16-frame clip from each encoder's own video model (`src.video_embeddings`):
PanEcho backbone (768-d, its own temporal transformer) and EchoPrime (MViT-v2-S, 512-d).
Clip views: `consecutive16` (frames 0–15), `stride2` (every 2nd of the first 32, EchoPrime's own
sampling), `uniform16` (16 frames over the whole clip). Probe: ridge on the clip embedding
(`temporal_probe --views all --heads mean`), alpha chosen on VAL; TEST = 1,277 videos.

Note: PanEcho's training tasks include EF (on its own Yale data), so its embedding is tuned to EF.
EchoPrime comes from the EchoNet group (Cedars-Sinai); whether its pretraining data overlaps EchoNet-Dynamic was not checked.

`paired_vs_best_frame.txt`: paired bootstrap (B = 10,000) of the video embeddings against frame-level
setups. On the same 16 frames, PanEcho's own temporal transformer equals a small transformer head
trained on PanEcho's frame features (−0.04 MAE, p = 0.76); both beat the frame mean by ~1.4 MAE.
