"""Paired comparisons across the EchoNet-Dynamic temporal-probe runs.

Reads ``<results>/<model>/predictions/<view>__<head>.csv`` (written by temporal_probe) and asks
two questions per model, each with a paired bootstrap on the same resampled TEST videos and a
Holm correction over every comparison in the table:

* head effect     -- attn / transformer / meanstd vs mean, at a fixed frame view;
* coverage effect -- uniformN vs consecutiveN, at a fixed head.

Also reports the seed spread of the trained heads (per-seed MAE, from ``*__seedK.csv``).

    python scripts/echonet_analysis.py --results results/echonet_dynamic \\
        --filelist .work/echonet/FileList.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HEADS = ("mean", "meanstd", "attn", "transformer")
VIEWS = ("consecutive16", "uniform16", "consecutive32", "uniform32")


def mae(y, p):
    return float(np.mean(np.abs(y - p)))


def holm(pvals):
    order = np.argsort(pvals)
    adj = np.empty(len(pvals))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=Path("results/echonet_dynamic"))
    ap.add_argument("--filelist", type=Path, required=True)
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    fl = pd.read_csv(args.filelist)
    fl["FileName"] = fl["FileName"].astype(str).str.replace(r"\.avi$", "", regex=True)
    ef = fl.set_index("FileName")["EF"]

    rows, seeds = [], []
    for model_dir in sorted(p for p in args.results.iterdir() if (p / "predictions").is_dir()):
        model = model_dir.name
        preds = {}
        for view in VIEWS:
            for head in HEADS:
                f = model_dir / "predictions" / f"{view}__{head}.csv"
                if f.exists():
                    d = pd.read_csv(f)
                    preds[(view, head)] = d.set_index("FileName")["EF_pred"]
                for s in range(10):
                    fs = model_dir / "predictions" / f"{view}__{head}__seed{s}.csv"
                    if fs.exists():
                        d = pd.read_csv(fs).set_index("FileName")["EF_pred"]
                        seeds.append({"model": model, "view": view, "head": head, "seed": s,
                                      "MAE": mae(ef.loc[d.index].to_numpy(), d.to_numpy())})
        ids = sorted(set.intersection(*(set(p.index) for p in preds.values())))
        y = ef.loc[ids].to_numpy()
        P = {k: v.loc[ids].to_numpy() for k, v in preds.items()}
        rng = np.random.default_rng(args.seed)
        idx = [rng.integers(0, len(y), len(y)) for _ in range(args.B)]

        def compare(kind, a, b):
            d = np.array([mae(y[i], P[a][i]) - mae(y[i], P[b][i]) for i in idx])
            rows.append({
                "model": model, "effect": kind, "config": f"{a[0]}/{a[1]}", "vs": f"{b[0]}/{b[1]}",
                "MAE": mae(y, P[a]), "MAE_vs": mae(y, P[b]), "dMAE": mae(y, P[a]) - mae(y, P[b]),
                "dMAE_lo": np.quantile(d, 0.025), "dMAE_hi": np.quantile(d, 0.975),
                "p": max(1.0 / args.B, min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))),
            })

        for view in VIEWS:
            for head in ("meanstd", "attn", "transformer"):
                if (view, head) in P and (view, "mean") in P:
                    compare("head", (view, head), (view, "mean"))
        for n in ("16", "32"):
            for head in HEADS:
                a, b = (f"uniform{n}", head), (f"consecutive{n}", head)
                if a in P and b in P:
                    compare("coverage", a, b)

    table = pd.DataFrame(rows)
    table["p_holm"] = holm(table["p"].to_numpy())
    table.to_csv(args.results / "paired_comparisons.csv", index=False)
    seed_tab = (pd.DataFrame(seeds).groupby(["model", "view", "head"])["MAE"]
                .agg(["mean", "std", "min", "max"]).reset_index())
    seed_tab.to_csv(args.results / "seed_spread.csv", index=False)
    pd.set_option("display.width", 200)
    print(table.round(3).to_string(index=False))
    print("\nSeed spread of trained heads (test MAE across seeds):")
    print(seed_tab.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
