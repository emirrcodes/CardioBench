"""Temporal heads on CardiacNet (ASD, PAH): does the EchoNet-Dynamic result carry over to
classification?

Per model, each task's splits are embedded once with ``--sampling both --max_frames 32``; then
``src.linear_probe.temporal_probe`` (task classification) compares frame views x heads with the
42 ``PHI*`` controls removed from train, val and test (a separate, all-negative source; see the
CardiacNet v1-v3 results). Paired AUROC comparisons per task: ``scripts/echonet_analysis.py``.

    python scripts/cardiacnet_temporal.py --data_root /kaggle/input --out_dir /kaggle/working/ct
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cardiacnet_benchmark import REPO, TASKS, embed, find_cardiacnet_parent, localize_split, sh  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_root", required=True, type=Path)
    p.add_argument("--out_dir", required=True, type=Path)
    p.add_argument("--models", nargs="+", default=["echo_clip", "biomed_clip", "siglip2"])
    p.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    p.add_argument("--device", default="cuda")
    p.add_argument("--precision", default="fp16")
    p.add_argument("--batch_size", default="64")
    p.add_argument("--stored_max", default="32")
    p.add_argument("--views", nargs="+", default=["consecutive16", "uniform16", "consecutive32", "uniform32"])
    p.add_argument("--heads", nargs="+", default=["mean", "meanstd", "attn", "transformer"])
    p.add_argument("--seeds", nargs="+", default=["0", "1", "2", "3", "4"])
    p.add_argument("--bootstrap", default="2000")
    p.add_argument("--video_models", nargs="*", default=[], choices=["panecho", "echoprime"],
                   help="Video-level encoders: one embedding per 16-frame clip view, linear probe on it")
    args = p.parse_args()
    args.sampling, args.max_frames = "both", args.stored_max  # what embed() reads

    out_dir = args.out_dir.resolve()
    parent = find_cardiacnet_parent(args.data_root.resolve())
    for task in args.tasks:
        split_csv = localize_split(task, parent, out_dir)
        t = task.lower()
        for model in args.models:
            tag = model.rstrip("/").split("/")[-1].replace(":", "_")  # aliases stay as-is
            emb_dir = out_dir / "embeddings" / tag / t
            embed(model, task, split_csv, emb_dir, args)
            sh([sys.executable, "-m", "src.linear_probe.temporal_probe", "--emb_root", emb_dir,
                "--labels_csv", split_csv, "--id_col", "unique_id", "--label_col", TASKS[task][1],
                "--pred_col", f"prob_{t}", "--task", "classification", "--stored_max", args.stored_max,
                "--views", *args.views, "--heads", *args.heads, "--seeds", *args.seeds,
                "--exclude_col", "path", "--exclude_regex", "/PHI", "--bootstrap", args.bootstrap,
                "--out_dir", out_dir / "probe" / task / tag])
        for vm in args.video_models:
            vroot = out_dir / "video_embeddings" / vm / t
            for split in ("train", "val", "test"):
                sh([sys.executable, "-m", "src.video_embeddings", "--dataset", f"{t}_csv", "--root", "/",
                    "--split_csv", split_csv, "--split", split, "--model", vm, "--out_root", vroot,
                    "--device", args.device])
            for view in ("consecutive16", "stride2", "uniform16"):
                sh([sys.executable, "-m", "src.linear_probe.temporal_probe", "--emb_root", vroot / view,
                    "--labels_csv", split_csv, "--id_col", "unique_id", "--label_col", TASKS[task][1],
                    "--pred_col", f"prob_{t}", "--task", "classification", "--views", "all",
                    "--heads", "mean", "--reference", "all/mean", "--exclude_col", "path",
                    "--exclude_regex", "/PHI", "--bootstrap", args.bootstrap,
                    "--out_dir", out_dir / "probe" / task / f"{vm}-video__{view}"])
        sh([sys.executable, "scripts/echonet_analysis.py", "--results", out_dir / "probe" / task,
            "--metric", "auroc", "--filelist", REPO / TASKS[task][0], "--id_col", "unique_id",
            "--label_col", TASKS[task][1], "--pred_col", f"prob_{t}", "--B", "10000"])

    rows = []
    for task in args.tasks:
        for f in sorted((out_dir / "probe" / task).glob("*/summary.csv")):
            rows.append(pd.read_csv(f).assign(task=task, model=f.parent.name))
    if rows:
        summary = pd.concat(rows, ignore_index=True)
        summary.to_csv(out_dir / "summary.csv", index=False)
        print(summary[["task", "model", "config", "AUROC", "AUROC_lo", "AUROC_hi"]].round(3).to_string(index=False))
    print(f"Results in {out_dir}")


if __name__ == "__main__":
    main()
