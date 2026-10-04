"""Temporal heads on cached per-frame embeddings.

Embeddings written by ``src.embeddings --sampling both --max_frames K`` hold the first K
frames of each clip *and* K frames spread over the whole clip. From that single pass this
module compares, for every (frame view, head) pair:

* frame views -- ``consecutiveN`` (the first N frames, CardioBench's default) and
  ``uniformN`` (N frames spread over the clip), see ``video.frame_view_positions``;
* heads -- ``mean`` and ``meanstd`` (ridge / logistic regression on pooled features, alpha
  or C picked on val), ``attn`` (learned attention pooling) and ``transformer`` (one
  encoder layer over the frame sequence), both trained on GPU with early stopping on val;
* ``zeroshot`` -- prompt-based EF from ``src.regression`` on the same frames (optional).

Every configuration is scored on the test split with bootstrap CIs, and compared with a
reference configuration (default ``consecutive16/mean``) by a paired bootstrap on the
same resampled test videos.

    python -m src.linear_probe.temporal_probe --emb_root emb/echo_clip \\
        --labels_csv FileList.csv --id_col FileName --label_col EF --split_col Split \\
        --views consecutive16 uniform16 consecutive32 uniform32 \\
        --heads mean meanstd attn transformer --out_dir runs/echonet/echo_clip
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..video import frame_view_positions

ALPHA_GRID = (1e-1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
C_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
SPLITS = ("train", "val", "test")


# --------------------------------------------------------------------------- data


def load_split(split_dir: Path) -> List[dict]:
    rows = []
    for f in sorted(split_dir.glob("*.pt")):
        p = torch.load(f, map_location="cpu")
        rows.append({
            "id": str(p.get("video_id", f.stem)),
            "frames": p["embedding_per_frame"].float().detach(),
            "idx": [int(i) for i in p["frame_indices"]],
            "n_raw": int(p.get("n_frames_raw", -1)),
        })
    if not rows:
        raise FileNotFoundError(f"No embeddings in {split_dir}")
    return rows


def build_view(rows: List[dict], view: str, stored_max: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stack one frame view as a padded (N, T, D) tensor plus a (N, T) validity mask."""
    seqs = []
    for r in rows:
        n_raw = r["n_raw"] if r["n_raw"] > 0 else max(r["idx"]) + 1
        pos = frame_view_positions(r["idx"], n_raw, view, stored_max=stored_max)
        seqs.append(r["frames"][pos] if pos else r["frames"][:1])
    T = max(s.shape[0] for s in seqs)
    D = seqs[0].shape[1]
    X = torch.zeros(len(seqs), T, D)
    M = torch.zeros(len(seqs), T, dtype=torch.bool)
    for i, s in enumerate(seqs):
        X[i, : s.shape[0]] = s
        M[i, : s.shape[0]] = True
    return X, M


def pooled(X: torch.Tensor, M: torch.Tensor, with_std: bool) -> np.ndarray:
    m = M.unsqueeze(-1).float()
    n = m.sum(1).clamp(min=1)
    mean = (X * m).sum(1) / n
    if not with_std:
        return mean.numpy()
    var = (((X - mean.unsqueeze(1)) * m) ** 2).sum(1) / n
    return torch.cat([mean, var.sqrt()], dim=1).numpy()


# --------------------------------------------------------------------------- heads


class AttnPool(nn.Module):
    """Project frames, add a learned position embedding, attention-pool, predict."""

    def __init__(self, d_in: int, T: int, hidden: int = 256, out: int = 1, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout))
        self.pos = nn.Parameter(torch.zeros(1, T, hidden))
        self.score = nn.Linear(hidden, 1)
        self.head = nn.Linear(hidden, out)

    def forward(self, x, mask):
        h = self.proj(x) + self.pos[:, : x.shape[1]]
        s = self.score(h).squeeze(-1).masked_fill(~mask, float("-inf"))
        a = torch.softmax(s, dim=1).unsqueeze(-1)
        return self.head((a * h).sum(1)).squeeze(-1)


class TinyTransformer(nn.Module):
    """One transformer encoder layer over the frame sequence, then masked mean pooling."""

    def __init__(self, d_in: int, T: int, hidden: int = 256, out: int = 1, layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, hidden))
        self.pos = nn.Parameter(torch.zeros(1, T, hidden))
        enc = nn.TransformerEncoderLayer(hidden, nhead=4, dim_feedforward=2 * hidden, dropout=dropout,
                                         batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, num_layers=layers)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, out))

    def forward(self, x, mask):
        h = self.enc(self.proj(x) + self.pos[:, : x.shape[1]], src_key_padding_mask=~mask)
        m = mask.unsqueeze(-1).float()
        return self.head((h * m).sum(1) / m.sum(1).clamp(min=1)).squeeze(-1)


def train_torch_head(kind, data, task, seed, device, epochs=200, patience=15, lr=1e-3, wd=1e-2, bs=128):
    """Fit one torch head; returns test predictions (de-standardised for regression)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    (Xtr, Mtr, ytr), (Xva, Mva, yva), (Xte, Mte, _) = data
    D, T = Xtr.shape[-1], Xtr.shape[1]
    model = (AttnPool if kind == "attn" else TinyTransformer)(D, T).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if task == "regression":
        mu, sd = float(ytr.mean()), float(ytr.std() + 1e-8)
        t_tr = torch.tensor((ytr - mu) / sd, dtype=torch.float32)
        loss_fn = F.smooth_l1_loss
    else:
        mu, sd = 0.0, 1.0
        t_tr = torch.tensor(ytr, dtype=torch.float32)
        loss_fn = F.binary_cross_entropy_with_logits

    def predict(X, M):
        model.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, len(X), 512):
                outs.append(model(X[i:i + 512].to(device), M[i:i + 512].to(device)).float().cpu())
        out = torch.cat(outs).numpy()
        return out * sd + mu if task == "regression" else 1 / (1 + np.exp(-out))

    def val_score(p):  # lower is better
        if task == "regression":
            return mean_absolute_error(yva, p)
        return -roc_auc_score(yva, p) if len(np.unique(yva)) > 1 else log_loss(yva, p, labels=[0, 1])

    best, best_state, bad = np.inf, None, 0
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr), generator=g)
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            out = model(Xtr[b].to(device), Mtr[b].to(device))
            loss = loss_fn(out, t_tr[b].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
        s = val_score(predict(Xva, Mva))
        if s < best - 1e-6:
            best, bad = s, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return predict(Xte, Mte)


def fit_sklearn_head(task, Ftr, ytr, Fva, yva, Fte):
    best, best_s = None, np.inf
    grid = ALPHA_GRID if task == "regression" else C_GRID
    for g in grid:
        if task == "regression":
            m = make_pipeline(StandardScaler(), Ridge(alpha=g)).fit(Ftr, ytr)
            s = mean_absolute_error(yva, m.predict(Fva))
        else:
            m = make_pipeline(StandardScaler(), LogisticRegression(C=g, class_weight="balanced", max_iter=5000)).fit(Ftr, ytr)
            # a tiny val split can hold one class; fall back to log-loss so C can still be chosen
            pv = m.predict_proba(Fva)[:, 1]
            s = -roc_auc_score(yva, pv) if len(np.unique(yva)) > 1 else log_loss(yva, pv, labels=[0, 1])
        if s < best_s:
            best, best_s = m, s
    return best.predict(Fte) if task == "regression" else best.predict_proba(Fte)[:, 1]


# --------------------------------------------------------------------------- scoring


def metric_fns(task):
    if task == "regression":
        return {
            "MAE": mean_absolute_error,
            "RMSE": lambda y, p: float(np.sqrt(mean_squared_error(y, p))),
            "R2": r2_score,
        }
    return {"AUROC": lambda y, p: roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan}


def bootstrap(task, y, preds: Dict[str, np.ndarray], ref: str, B=1000, seed=42):
    fns = metric_fns(task)
    key = "MAE" if task == "regression" else "AUROC"
    rng = np.random.default_rng(seed)
    idx = [rng.integers(0, len(y), len(y)) for _ in range(B)]
    rows = []
    for name, p in preds.items():
        row = {"config": name}
        for m, fn in fns.items():
            vals = np.array([fn(y[i], p[i]) for i in idx])
            row[m] = float(fn(y, p))
            row[f"{m}_lo"], row[f"{m}_hi"] = np.nanquantile(vals, [0.025, 0.975])
        if name != ref and ref in preds:
            d = np.array([fns[key](y[i], p[i]) - fns[key](y[i], preds[ref][i]) for i in idx])
            d = d[~np.isnan(d)]
            row[f"d{key}_vs_ref"] = float(fns[key](y, p) - fns[key](y, preds[ref]))
            row[f"d{key}_lo"], row[f"d{key}_hi"] = np.quantile(d, [0.025, 0.975])
            row["p_vs_ref"] = float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- main


def run(args) -> pd.DataFrame:
    out_dir = Path(args.out_dir)
    (out_dir / "predictions").mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    lab = pd.read_csv(args.labels_csv)
    lab[args.id_col] = lab[args.id_col].astype(str).str.replace(r"\.avi$", "", regex=True)
    if args.exclude_regex:  # e.g. CardiacNet's PHI* controls: dropped from train, val and test
        drop = lab[args.exclude_col].astype(str).str.contains(args.exclude_regex, regex=True)
        print(f"[temporal] excluding {int(drop.sum())} rows where {args.exclude_col} ~ {args.exclude_regex!r}")
        lab = lab[~drop]
    labels = lab.drop_duplicates(args.id_col).set_index(args.id_col)[args.label_col]

    data_rows = {}
    for s in SPLITS:
        rows = [r for r in load_split(Path(args.emb_root) / s) if r["id"] in labels.index]
        data_rows[s] = rows
        print(f"[temporal] {s}: {len(rows)} labelled videos")
    y = {s: labels.loc[[r["id"] for r in data_rows[s]]].to_numpy(dtype=float) for s in SPLITS}
    test_ids = [r["id"] for r in data_rows["test"]]

    preds: Dict[str, np.ndarray] = {}
    for view in args.views:
        V = {s: build_view(data_rows[s], view, args.stored_max) for s in SPLITS}
        for head in args.heads:
            name = f"{view}/{head}"
            if head in ("mean", "meanstd"):
                Fs = {s: pooled(*V[s], with_std=head == "meanstd") for s in SPLITS}
                p = fit_sklearn_head(args.task, Fs["train"], y["train"], Fs["val"], y["val"], Fs["test"])
            elif head in ("attn", "transformer"):
                data = tuple((V[s][0], V[s][1], y[s]) for s in SPLITS)
                runs = [train_torch_head(head, data, args.task, sd, device) for sd in args.seeds]
                for sd, r in zip(args.seeds, runs):  # per-seed predictions, for seed variance
                    preds_seed = pd.DataFrame({args.id_col: test_ids, args.pred_col: r})
                    preds_seed.to_csv(out_dir / "predictions" / f"{view}__{head}__seed{sd}.csv", index=False)
                p = np.mean(runs, axis=0)  # seed ensemble
            else:
                raise ValueError(f"Unknown head {head}")
            preds[name] = p
            pd.DataFrame({args.id_col: test_ids, args.pred_col: p}).to_csv(
                out_dir / "predictions" / f"{view}__{head}.csv", index=False)
            m = metric_fns(args.task)
            print(f"[temporal] {name}: " + " ".join(f"{k}={fn(y['test'], p):.3f}" for k, fn in m.items()), flush=True)

        if args.zeroshot_model and args.task == "regression":
            p = zeroshot_ef(V["test"], args.zeroshot_model, device)
            preds[f"{view}/zeroshot"] = p
            pd.DataFrame({args.id_col: test_ids, args.pred_col: p}).to_csv(
                out_dir / "predictions" / f"{view}__zeroshot.csv", index=False)
            print(f"[temporal] {view}/zeroshot: MAE={mean_absolute_error(y['test'], p):.3f}", flush=True)

    summary = bootstrap(args.task, y["test"], preds, args.reference, B=args.bootstrap)
    summary.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=1, default=str))
    print(summary.round(3).to_string(index=False))
    return summary


def zeroshot_ef(view_test, model_id, device) -> np.ndarray:
    """CardioBench's prompt-based EF (src.regression) restricted to the view's frames."""
    from ..prompts import zero_shot_prompts
    from ..regression import EjectionFractionConfig, _compute_regression_metric, _expand_integer_prompts
    from ..text import encode_text_prompts

    cfg = EjectionFractionConfig()
    prompts, values = _expand_integer_prompts(zero_shot_prompts["ejection_fraction"], cfg.min_value, cfg.max_value)
    P = encode_text_prompts(prompts, model_id=model_id, device=device,
                            precision="fp16" if device == "cuda" else "fp32").float()
    X, M = view_test
    out = []
    for x, m in zip(X, M):
        v = F.normalize(x[m].to(device), dim=-1).unsqueeze(0)
        out.append(float(_compute_regression_metric(v, P, values)))
    return np.array(out)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--emb_root", required=True, help="Directory with train/ val/ test/ embedding folders")
    p.add_argument("--labels_csv", required=True)
    p.add_argument("--id_col", default="FileName")
    p.add_argument("--label_col", default="EF")
    p.add_argument("--pred_col", default="EF_pred")
    p.add_argument("--task", choices=["regression", "classification"], default="regression")
    p.add_argument("--stored_max", type=int, default=32, help="--max_frames used with --sampling both")
    p.add_argument("--views", nargs="+", default=["consecutive16", "uniform16", "consecutive32", "uniform32"])
    p.add_argument("--heads", nargs="+", default=["mean", "meanstd", "attn", "transformer"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--reference", default="consecutive16/mean")
    p.add_argument("--zeroshot_model", default=None, help="Model alias for zero-shot EF (regression only)")
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--exclude_col", default="path", help="labels_csv column tested by --exclude_regex")
    p.add_argument("--exclude_regex", default=None, help="Drop label rows whose --exclude_col matches")
    p.add_argument("--out_dir", required=True)
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
