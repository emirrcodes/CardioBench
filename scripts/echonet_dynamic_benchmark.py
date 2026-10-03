"""EchoNet-Dynamic EF: frame coverage x temporal heads, for several frozen encoders.

For each model, every split is embedded once with ``--sampling both`` (the first K frames
plus K frames spread over the clip), then ``src.linear_probe.temporal_probe`` compares
frame views (consecutive16, uniform16, consecutive32, uniform32) and heads (mean, meanstd,
attention pooling, a one-layer transformer, plus zero-shot EF) on the official TEST split.

Designed for a Colab/Kaggle GPU; completed embeddings are reused, so a run can resume.

    python scripts/echonet_dynamic_benchmark.py --data_root /content/echonet \\
        --out_dir /content/runs/echonet --models echo_clip biomed_clip siglip2
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SPLITS = {"train": "TRAIN", "val": "VAL", "test": "TEST"}


def sh(cmd):
    print("$", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=REPO)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_root", required=True, type=Path, help="Folder with FileList.csv and Videos/")
    p.add_argument("--out_dir", required=True, type=Path)
    p.add_argument("--models", nargs="+", default=["echo_clip", "biomed_clip", "siglip2"])
    p.add_argument("--stored_max", type=int, default=32)
    p.add_argument("--batch_size", default="128")
    p.add_argument("--precision", default="fp16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--heads", nargs="+", default=["mean", "meanstd", "attn", "transformer"])
    p.add_argument("--views", nargs="+", default=["consecutive16", "uniform16", "consecutive32", "uniform32"])
    p.add_argument("--seeds", nargs="+", default=["0", "1", "2"])
    p.add_argument("--skip_probe", action="store_true", help="Only compute embeddings")
    args = p.parse_args()

    filelist = pd.read_csv(args.data_root / "FileList.csv")
    expected = filelist["Split"].str.upper().value_counts()
    for model in args.models:
        emb = args.out_dir / "embeddings" / model
        for split, tag in SPLITS.items():
            done = len(list((emb / split).glob("*.pt")))
            if done >= expected.get(tag, 0):
                print(f"[{model}/{split}] {done} embeddings cached, skipping", flush=True)
                continue
            sh([sys.executable, "-m", "src.embeddings", "--dataset", "Dynamic",
                "--root", args.data_root, "--split", split, "--out_dir", emb / split,
                "--model", model, "--device", args.device, "--precision", args.precision,
                "--max_frames", args.stored_max, "--sampling", "both",
                "--batch_size", args.batch_size])
        if args.skip_probe:
            continue
        sh([sys.executable, "-m", "src.linear_probe.temporal_probe", "--emb_root", emb,
            "--labels_csv", args.data_root / "FileList.csv", "--id_col", "FileName",
            "--label_col", "EF", "--pred_col", "EF_pred", "--stored_max", args.stored_max,
            "--views", *args.views, "--heads", *args.heads, "--seeds", *args.seeds,
            "--zeroshot_model", model, "--out_dir", args.out_dir / "probe" / model])

    rows = []
    for model in args.models:
        f = args.out_dir / "probe" / model / "summary.csv"
        if f.exists():
            rows.append(pd.read_csv(f).assign(model=model))
    if rows:
        summary = pd.concat(rows, ignore_index=True)
        summary.to_csv(args.out_dir / "summary.csv", index=False)
        print(summary.round(3).to_string(index=False))
    print(f"Results in {args.out_dir}")


if __name__ == "__main__":
    main()
