"""End-to-end CardioBench run on CardiacNet (ASD + PAH).

For every model: embed train/val/test once, then score the test split with
  * zero-shot   -- prompt similarity (src.classification.binary),
  * probe       -- logistic regression on mean-pooled frame embeddings,
  * probe-meanstd -- same, on [mean, std] over frames (a cheap temporal signal),
and a label-efficiency curve for both probes. Predictions go through the official
evaluation/cardiacnet.py so the numbers match the benchmark's bootstrap protocol.

Designed for a Kaggle GPU notebook (see notebooks/kaggle_cardiacnet.ipynb) but runs
anywhere; completed embeddings are reused, so an interrupted run can be resumed.

    python scripts/cardiacnet_benchmark.py --data_root /kaggle/input/abnormcardiacechovideos \\
        --out_dir /kaggle/working/cardiacnet --models echo_clip biomed_clip siglip2
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
TASKS = {
    # task: (split csv, label column, zero-shot prompt keys)
    "ASD": ("data/splits/cardiacnet/cardiacnet_asd_split.csv", "ASD", ("asd_present", "asd_absent")),
    "PAH": ("data/splits/cardiacnet/cardiacnet_pah_split.csv", "PAH", ("pah_present", "pah_absent")),
}
SPLITS = ("train", "val", "test")


def sh(cmd: list[str], **env) -> None:
    print("$", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=REPO, env={**os.environ, **env})


def find_cardiacnet_parent(data_root: Path) -> Path:
    """Return the directory that contains ``CardiacNet/`` (Kaggle mount layouts vary)."""
    for cand in [data_root, *(c for pat in ("*", "*/*", "*/*/*") for c in sorted(data_root.glob(pat)))]:
        if (cand / "CardiacNet" / "CardiacNet-ASD").is_dir():
            return cand
    raise FileNotFoundError(f"No CardiacNet/CardiacNet-ASD folder under {data_root}")


def localize_split(task: str, parent: Path, out_dir: Path) -> Path:
    """Rewrite the published split CSV so ``path`` points at the local copy."""
    csv_rel, _, _ = TASKS[task]
    df = pd.read_csv(REPO / csv_rel)
    df["path"] = [str(parent / p.replace("data/raw/", "", 1)) for p in df["path"]]
    exists = df["path"].map(os.path.exists)
    print(f"[{task}] {exists.sum()}/{len(df)} videos found locally")
    if not exists.all():
        print(df.loc[~exists, "path"].head().to_string())
    out = out_dir / "splits" / f"{task.lower()}_split.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out


def embed(model: str, task: str, split_csv: Path, emb_dir: Path, args) -> None:
    n_expected = pd.read_csv(split_csv)["split"].value_counts()
    for split in SPLITS:
        done = len(list((emb_dir / split).glob("*.pt")))
        if done >= n_expected.get(split, 0):
            print(f"[{model}/{task}/{split}] {done} embeddings cached, skipping")
            continue
        cmd = [sys.executable, "-m", "src.embeddings", "--dataset", f"{task.lower()}_csv",
               "--root", "/", "--split_csv", split_csv, "--split", split,
               "--out_dir", emb_dir / split, "--model", model, "--device", args.device,
               "--precision", args.precision, "--max_frames", args.max_frames,
               "--batch_size", args.batch_size]
        if args.device == "cpu":
            cmd += ["--no_pin_memory", "--no_channels_last"]
        sh(cmd)


def zero_shot(model: str, task: str, emb_dir: Path, pred_path: Path, args) -> None:
    pos_key, neg_key = TASKS[task][2]
    csv_name = f"zeroshot_{task.lower()}.csv"
    sh([sys.executable, "-m", "src.classification.binary", "--emb_dir", emb_dir / "test",
        "--csv_name", csv_name, "--mode", "argmax", "--pos_key", pos_key, "--neg_key", neg_key,
        "--model_id", model, "--device", args.device, "--precision", args.precision])
    raw = pd.read_csv(emb_dir / "test" / csv_name)
    t = task.lower()
    out = pd.DataFrame({
        "unique_id": raw["video_id"],
        f"{t}_pred": (raw["pred_class"] == "positive").astype(int),
        f"prob_{t}": raw["prob_positive"],
    })
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(pred_path, index=False)


def probe(model: str, task: str, split_csv: Path, emb_dir: Path, pooling: str,
          pred_path: Path, curve_path: Path, args) -> None:
    t = task.lower()
    sh([sys.executable, "-m", "src.linear_probe.embedding_probe", "--emb_root", emb_dir,
        "--labels_csv", split_csv, "--label_col", TASKS[task][1], "--pooling", pooling,
        "--pred_col", f"{t}_pred", "--prob_col", f"prob_{t}", "--out_csv", pred_path,
        "--train_fractions", *args.fractions, "--seeds", *args.seeds,
        "--curve_csv", curve_path])


def summarize(out_dir: Path) -> pd.DataFrame:
    rows = []
    for task in TASKS:
        path = out_dir / "eval" / f"{task}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df[["model", "protocol"]] = df["model"].str.split("__", n=1, expand=True)
        df.insert(0, "task", task)
        rows.append(df)
    summary = pd.concat(rows, ignore_index=True)
    summary.to_csv(out_dir / "summary.csv", index=False)
    return summary


def plot(out_dir: Path, summary: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    # 1) AUROC with bootstrap 95% CI, per task / model / protocol.
    fig, axes = plt.subplots(1, len(TASKS), figsize=(6 * len(TASKS), 4), sharey=True)
    for ax, task in zip(axes, TASKS):
        d = summary[summary["task"] == task]
        models = sorted(d["model"].unique())
        protocols = sorted(d["protocol"].unique())
        width = 0.8 / len(protocols)
        for j, protocol in enumerate(protocols):
            g = d[d["protocol"] == protocol].set_index("model").reindex(models)
            x = [i + (j - (len(protocols) - 1) / 2) * width for i in range(len(models))]
            auc = g["AUROC(%)"].to_numpy()
            err = [auc - g["AUROC(%)_ci_lo"].to_numpy(), g["AUROC(%)_ci_hi"].to_numpy() - auc]
            ax.bar(x, auc, width, yerr=err, capsize=3, label=protocol)
        ax.set_xticks(range(len(models)), models)
        ax.axhline(50, color="grey", lw=0.8, ls="--")
        ax.legend(fontsize=8, loc="lower right")
        ax.set_title(f"CardiacNet-{task} (test)")
        ax.set_ylabel("AUROC (%)")
        ax.set_ylim(30, 100)
    fig.tight_layout()
    fig.savefig(fig_dir / "auroc_by_protocol.png", dpi=150)

    # 2) Label efficiency: test AUROC vs. share of labelled training videos.
    curves = []
    for path in (out_dir / "curves").glob("*.csv"):
        model, task, protocol = path.stem.split("__")
        c = pd.read_csv(path).assign(model=model, task=task, protocol=protocol)
        curves.append(c)
    if curves:
        curves = pd.concat(curves)
        fig, axes = plt.subplots(1, len(TASKS), figsize=(6 * len(TASKS), 4), sharey=True)
        for ax, task in zip(axes, TASKS):
            d = curves[curves["task"] == task]
            for (model, protocol), g in d.groupby(["model", "protocol"]):
                agg = g.groupby("fraction")["auroc"].agg(["mean", "std"]).fillna(0) * 100
                ax.errorbar(agg.index, agg["mean"], yerr=agg["std"], marker="o", capsize=3,
                            label=f"{model} / {protocol}")
            ax.set_xscale("log")
            ax.set_xlabel("fraction of labelled training videos")
            ax.set_ylabel("test AUROC (%)")
            ax.set_title(f"Label efficiency — CardiacNet-{task}")
            ax.grid(alpha=0.3)
        axes[-1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / "label_efficiency.png", dpi=150)
        curves.to_csv(out_dir / "label_efficiency.csv", index=False)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_root", required=True, type=Path)
    p.add_argument("--out_dir", required=True, type=Path)
    p.add_argument("--models", nargs="+", default=["echo_clip", "biomed_clip", "siglip2"])
    p.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    p.add_argument("--device", default="cuda")
    p.add_argument("--precision", default="fp16", help="fp16 on Kaggle T4/P100 (no bf16)")
    p.add_argument("--max_frames", default="16")
    p.add_argument("--batch_size", default="64")
    p.add_argument("--fractions", nargs="+", default=["0.05", "0.1", "0.25", "0.5", "1.0"])
    p.add_argument("--seeds", nargs="+", default=["0", "1", "2", "3", "4"])
    p.add_argument("--skip_zero_shot", action="store_true")
    args = p.parse_args()

    out_dir = args.out_dir.resolve()
    parent = find_cardiacnet_parent(args.data_root)
    pred_root = out_dir / "predictions"

    for task in args.tasks:
        split_csv = localize_split(task, parent, out_dir)
        for model in args.models:
            tag = model.rstrip("/").split("/")[-1].replace(":", "_")  # aliases stay as-is
            emb_dir = out_dir / "embeddings" / tag / task.lower()
            embed(model, task, split_csv, emb_dir, args)
            if not args.skip_zero_shot:
                zero_shot(model, task, emb_dir, pred_root / task / f"{tag}__zeroshot.csv", args)
            for pooling, name in (("mean", "probe"), ("meanstd", "probe-meanstd")):
                probe(model, task, split_csv, emb_dir, pooling,
                      pred_root / task / f"{tag}__{name}.csv",
                      out_dir / "curves" / f"{tag}__{task}__{name}.csv", args)

    sh([sys.executable, "evaluation/cardiacnet.py"],
       CARDIACNET_PRED_ROOT=str(pred_root), CARDIACNET_OUT_DIR=str(out_dir / "eval"))
    summary = summarize(out_dir)
    plot(out_dir, summary)
    cols = ["task", "model", "protocol", "n", "AUROC(%)", "AUROC(%)_ci_lo", "AUROC(%)_ci_hi",
            "Balanced_Accuracy(%)", "F1(%)"]
    print(summary[cols].sort_values(["task", "AUROC(%)"], ascending=[True, False]).to_string(index=False))
    print(f"\nResults in {out_dir}")


if __name__ == "__main__":
    main()
