"""Paired comparisons across temporal-probe runs (EchoNet-Dynamic EF, CardiacNet ASD/PAH).

Reads ``<results>/<model>/predictions/<view>__<head>.csv`` (written by temporal_probe) and asks
two questions per model, each with a paired bootstrap on the same resampled TEST videos and a
Holm correction over every comparison in the table:

* head effect     -- attn / transformer / meanstd vs mean, at a fixed frame view;
* coverage effect -- uniformN vs consecutiveN, at a fixed head;
* cycle effect    -- ED/ES-anchored views (edes, halfcycle16, ed16) vs consecutive16, and
  halfcycle16 vs uniform16, at a fixed head (only when cycle views were probed).

Also reports the seed spread of the trained heads (per-seed MAE, from ``*__seedK.csv``).

    python scripts/echonet_analysis.py --results results/echonet_dynamic \\
        --filelist .work/echonet/FileList.csv
    python scripts/echonet_analysis.py --results results/cardiacnet_temporal/ASD --metric auroc \\
        --filelist data/splits/cardiacnet/cardiacnet_asd_split.csv --id_col unique_id \\
        --label_col ASD --pred_col prob_asd
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HEADS = ("mean", "meanstd", "attn", "transformer")
VIEWS = ("consecutive16", "uniform16", "consecutive32", "uniform32", "edes", "halfcycle16", "ed16")
CYCLE_PAIRS = (("edes", "consecutive16"), ("halfcycle16", "consecutive16"), ("ed16", "consecutive16"),
               ("halfcycle16", "uniform16"))


def mae(y, p):
    return float(np.mean(np.abs(y - p)))


def auroc(y, p):
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan


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
    ap.add_argument("--filelist", type=Path, required=True, help="Labels CSV")
    ap.add_argument("--id_col", default="FileName")
    ap.add_argument("--label_col", default="EF")
    ap.add_argument("--pred_col", default="EF_pred")
    ap.add_argument("--metric", choices=["mae", "auroc"], default="mae")
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    fl = pd.read_csv(args.filelist)
    fl[args.id_col] = fl[args.id_col].astype(str).str.replace(r"\.avi$", "", regex=True)
    ef = fl.drop_duplicates(args.id_col).set_index(args.id_col)[args.label_col]
    score = mae if args.metric == "mae" else auroc
    M = "MAE" if args.metric == "mae" else "AUROC"

    rows, seeds = [], []
    for model_dir in sorted(p for p in args.results.iterdir() if (p / "predictions").is_dir()):
        model = model_dir.name
        preds = {}
        for view in VIEWS:
            for head in HEADS:
                f = model_dir / "predictions" / f"{view}__{head}.csv"
                if f.exists():
                    d = pd.read_csv(f)
                    preds[(view, head)] = d.set_index(args.id_col)[args.pred_col]
                for s in range(10):
                    fs = model_dir / "predictions" / f"{view}__{head}__seed{s}.csv"
                    if fs.exists():
                        d = pd.read_csv(fs).set_index(args.id_col)[args.pred_col]
                        seeds.append({"model": model, "view": view, "head": head, "seed": s,
                                      M: score(ef.loc[d.index].to_numpy(), d.to_numpy())})
        if len(preds) < 2:  # e.g. video-level encoders (one clip embedding, no frame views)
            continue
        ids = sorted(set.intersection(*(set(p.index) for p in preds.values())))
        y = ef.loc[ids].to_numpy()
        P = {k: v.loc[ids].to_numpy() for k, v in preds.items()}
        rng = np.random.default_rng(args.seed)
        idx = [rng.integers(0, len(y), len(y)) for _ in range(args.B)]

        def compare(kind, a, b):
            d = np.array([score(y[i], P[a][i]) - score(y[i], P[b][i]) for i in idx])
            d = d[~np.isnan(d)]  # AUROC: resamples with a single class
            rows.append({
                "model": model, "effect": kind, "config": f"{a[0]}/{a[1]}", "vs": f"{b[0]}/{b[1]}",
                M: score(y, P[a]), f"{M}_vs": score(y, P[b]), f"d{M}": score(y, P[a]) - score(y, P[b]),
                f"d{M}_lo": np.quantile(d, 0.025), f"d{M}_hi": np.quantile(d, 0.975),
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
        for va, vb in CYCLE_PAIRS:
            for head in HEADS:
                if (va, head) in P and (vb, head) in P:
                    compare("cycle", (va, head), (vb, head))

    table = pd.DataFrame(rows)
    if table.empty:
        print("no paired comparisons to make"); return
    table["p_holm"] = holm(table["p"].to_numpy())
    table.to_csv(args.results / "paired_comparisons.csv", index=False)
    if not seeds:
        print(table.round(3).to_string(index=False)); return
    seed_tab = (pd.DataFrame(seeds).groupby(["model", "view", "head"])[M]
                .agg(["mean", "std", "min", "max"]).reset_index())
    seed_tab.to_csv(args.results / "seed_spread.csv", index=False)
    pd.set_option("display.width", 200)
    print(table.round(3).to_string(index=False))
    print(f"\nSeed spread of trained heads (test {M} across seeds):")
    print(seed_tab.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
