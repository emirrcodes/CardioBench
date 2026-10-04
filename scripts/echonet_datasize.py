"""Does the temporal-head gain on EchoNet-Dynamic EF survive small training sets?

On CardiacNet (150-300 training videos) attention / transformer heads did not beat mean pooling,
while on EchoNet-Dynamic (7,465) they won every comparison. This subsamples EchoNet's TRAIN split to
CardiacNet-like sizes (val shrunk in proportion, as CardiacNet's val is ~16 % of train) and reruns the
heads on the cached embeddings, so data size and task can be told apart. TEST stays the full 1,277.

    python scripts/echonet_datasize.py --emb_root .work/echonet/embeddings \\
        --filelist .work/echonet/FileList.csv --out results/echonet_datasize
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.linear_probe.temporal_probe import (  # noqa: E402
    build_view, fit_sklearn_head, load_split, pooled, train_torch_head,
)


def mae(y, p):
    return float(np.mean(np.abs(y - p)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emb_root", type=Path, required=True)
    ap.add_argument("--filelist", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--models", nargs="+", default=["echo_clip", "biomed_clip", "siglip2"])
    ap.add_argument("--sizes", type=int, nargs="+", default=[150, 300, 1000, 3000])
    ap.add_argument("--subsamples", type=int, default=3)
    ap.add_argument("--views", nargs="+", default=["consecutive16", "uniform16"])
    ap.add_argument("--heads", nargs="+", default=["mean", "attn", "transformer"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--val_frac", type=float, default=0.16)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    fl = pd.read_csv(args.filelist)
    ef = fl.assign(FileName=fl.FileName.astype(str).str.replace(r"\.avi$", "", regex=True)).set_index("FileName")["EF"]
    rows_out = []
    for model in args.models:
        rows = {s: [r for r in load_split(args.emb_root / model / s) if r["id"] in ef.index] for s in ("train", "val", "test")}
        y = {s: ef.loc[[r["id"] for r in rows[s]]].to_numpy(float) for s in rows}
        V = {v: {s: build_view(rows[s], v, 32) for s in rows} for v in args.views}
        for n in args.sizes:
            for sub in range(args.subsamples):
                rng = np.random.default_rng(1000 * n + sub)
                tr = rng.choice(len(y["train"]), n, replace=False)
                va = rng.choice(len(y["val"]), max(20, round(args.val_frac * n)), replace=False)
                for v in args.views:
                    (Xtr, Mtr), (Xva, Mva), (Xte, Mte) = (V[v]["train"][0][tr], V[v]["train"][1][tr]), \
                        (V[v]["val"][0][va], V[v]["val"][1][va]), V[v]["test"]
                    data = ((Xtr, Mtr, y["train"][tr]), (Xva, Mva, y["val"][va]), (Xte, Mte, y["test"]))
                    for h in args.heads:
                        if h in ("mean", "meanstd"):
                            st = h == "meanstd"
                            p = fit_sklearn_head("regression", pooled(Xtr, Mtr, st), y["train"][tr],
                                                 pooled(Xva, Mva, st), y["val"][va], pooled(Xte, Mte, st))
                        else:
                            p = np.mean([train_torch_head(h, data, "regression", s, device) for s in args.seeds], axis=0)
                        rows_out.append({"model": model, "n_train": n, "subsample": sub, "view": v, "head": h,
                                         "n_val": len(va), "MAE": mae(y["test"], p)})
                        print(f"[datasize] {model} n={n} sub={sub} {v}/{h}: MAE={rows_out[-1]['MAE']:.3f}", flush=True)
                pd.DataFrame(rows_out).to_csv(args.out / "datasize_runs.csv", index=False)

    d = pd.DataFrame(rows_out)
    ref = d[d["head"] == "mean"][["model", "n_train", "subsample", "view", "MAE"]].rename(columns={"MAE": "MAE_mean"})
    g = d[d["head"] != "mean"].merge(ref, on=["model", "n_train", "subsample", "view"])
    g["dMAE"] = g["MAE"] - g["MAE_mean"]
    summ = (g.groupby(["model", "n_train", "view", "head"])
            .agg(dMAE=("dMAE", "mean"), dMAE_sd=("dMAE", "std"), wins=("dMAE", lambda x: int((x < 0).sum())),
                 runs=("dMAE", "size"), MAE=("MAE", "mean"), MAE_mean=("MAE_mean", "mean")).reset_index())
    summ.to_csv(args.out / "datasize_summary.csv", index=False)
    print(summ.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
