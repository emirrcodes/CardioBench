"""Video-level embeddings from PanEcho / EchoPrime (one vector per 16-frame clip).

For every video and every clip view (``consecutive16``, ``stride2``, ``uniform16``; see
``video_encoders.clip_indices``) this writes ``<out_root>/<view>/<split>/<video_id>.pt`` in the
format ``src.linear_probe.temporal_probe`` reads (one "frame" = the clip embedding), so a probe is
``temporal_probe --emb_root <out_root>/<view> --views all --heads mean``.

    python -m src.video_embeddings --dataset Dynamic --root /content/echonet --split test \\
        --model echoprime --out_root runs/video_emb/echoprime
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from .datasets import DatasetLoader
from .embeddings import _select_video_id
from .video import read_clip
from .video_encoders import CLIP_VIEWS, VIDEO_ENCODERS, clip_indices, load_video_encoder


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--split_csv", default=None)
    ap.add_argument("--model", required=True, choices=VIDEO_ENCODERS)
    ap.add_argument("--out_root", required=True, type=Path)
    ap.add_argument("--views", nargs="+", default=list(CLIP_VIEWS), choices=CLIP_VIEWS)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no_pretrained", action="store_true", help="Random weights (smoke tests only)")
    args = ap.parse_args()

    items = DatasetLoader().load(args.dataset, root=args.root, split=args.split, split_csv=args.split_csv)
    pretrained = not (args.no_pretrained or os.environ.get("CARDIOBENCH_NO_PRETRAINED"))
    model, prep = load_video_encoder(args.model, args.device, pretrained=pretrained)
    use_amp = args.device.startswith("cuda")
    for v in args.views:
        (args.out_root / v / args.split).mkdir(parents=True, exist_ok=True)

    for k, item in enumerate(items, 1):
        vid = _select_video_id(args.dataset, item)
        outs = {v: args.out_root / v / args.split / f"{vid}.pt" for v in args.views}
        if all(p.exists() for p in outs.values()):
            continue
        try:
            sel = lambda n: sorted({i for v in args.views for i in clip_indices(n, v)})
            frames, n_raw, idx = read_clip(item.path, res=(224, 224), selector=sel)
            where = {f: i for i, f in enumerate(idx)}
            bgr = item.path.suffix.lower() in (".avi", ".mp4")  # cv2 decodes BGR
            clips = torch.stack([prep(frames[[where[f] for f in clip_indices(n_raw, v)]], bgr)
                                 for v in args.views]).to(args.device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                emb = model(clips).float().cpu()
            for v, e in zip(args.views, emb):
                torch.save({"video_id": vid, "path": str(item.path), "embedding_per_frame": e[None].half(),
                            "embedding_pooled": e.half(), "frame_indices": [0], "n_frames_raw": int(n_raw),
                            "clip_view": v, "model": args.model, "metadata": item.metadata}, outs[v])
            if k % 100 == 0 or k == len(items):
                print(f"[{k}/{len(items)}] {vid} n_raw={n_raw}", flush=True)
        except Exception as exc:
            print(f"[{k}/{len(items)}] ERROR {vid}: {exc}", flush=True)
    print(f"Done: {args.model} {args.split} -> {args.out_root}")


if __name__ == "__main__":
    main()
