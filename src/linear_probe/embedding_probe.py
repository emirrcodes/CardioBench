"""Linear probes on cached embeddings.

``src.linear_probe.train`` re-encodes every video through the frozen backbone on every
epoch. For a frozen encoder that work is identical each time, so this module instead fits
the probe on the ``.pt`` files written by ``src.embeddings`` (one per video, per split).
A probe then trains in seconds on CPU, which also makes it cheap to sweep:

* the regularisation strength (selected on the val split, never on test);
* the fraction of labelled training data (a label-efficiency curve);
* several random seeds for the subsampled fractions.

Example::

    python -m src.linear_probe.embedding_probe \\
        --emb_root embeddings/echo_clip/asd \\
        --labels_csv data/splits/cardiacnet/cardiacnet_asd_split.csv \\
        --label_col ASD --pred_col asd_pred --prob_col prob_asd \\
        --out_csv predictions/ASD/echo_clip_probe.csv \\
        --train_fractions 0.05 0.1 0.25 0.5 1.0 --seeds 0 1 2 3 4 \\
        --curve_csv results/asd_echo_clip_curve.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

C_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
ALPHA_GRID = (1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)


def load_split_embeddings(split_dir: str | Path, pooling: str = "mean") -> Tuple[List[str], np.ndarray]:
    """Load every ``<video_id>.pt`` in ``split_dir`` as one feature row."""
    files = sorted(Path(split_dir).glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"No embeddings (.pt) found in {split_dir}")
    ids: List[str] = []
    rows: List[np.ndarray] = []
    for fpath in files:
        payload = torch.load(fpath, map_location="cpu")
        if pooling == "mean":
            vec = payload["embedding_pooled"].float()
        elif pooling == "meanstd":
            # Frame-to-frame variation is a crude but free temporal signal on top of the mean.
            frames = payload["embedding_per_frame"].float()
            std = frames.std(dim=0) if frames.shape[0] > 1 else torch.zeros(frames.shape[1])
            vec = torch.cat([frames.mean(dim=0), std])
        else:
            raise ValueError(f"Unknown pooling '{pooling}' (expected mean|meanstd)")
        ids.append(str(payload.get("video_id", fpath.stem)))
        rows.append(vec.detach().numpy())  # older caches were saved with grad
    return ids, np.stack(rows)


def attach_labels(
    ids: Sequence[str], feats: np.ndarray, labels: pd.Series
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Keep the rows that have a label, in embedding order."""
    keep = [i for i, vid in enumerate(ids) if vid in labels.index and pd.notna(labels[vid])]
    missing = len(ids) - len(keep)
    if missing:
        print(f"[probe][WARN] {missing} embeddings have no label and are ignored")
    return [ids[i] for i in keep], feats[keep], labels.loc[[ids[i] for i in keep]].to_numpy()


def subsample(
    X: np.ndarray, y: np.ndarray, fraction: float, seed: int, task: str
) -> Tuple[np.ndarray, np.ndarray]:
    if fraction >= 1.0:
        return X, y
    n = max(2, int(round(fraction * len(y))))
    stratify = y if task == "classification" else None
    if stratify is not None:
        n = max(n, len(np.unique(y)))
    X_sub, _, y_sub, _ = train_test_split(X, y, train_size=n, random_state=seed, stratify=stratify)
    return X_sub, y_sub


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    binary = probs.shape[1] == 2
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="binary" if binary else "macro"),
    }
    if len(np.unique(y_true)) > 1:
        out["auroc"] = (
            roc_auc_score(y_true, probs[:, 1])
            if binary
            else roc_auc_score(y_true, probs, multi_class="ovr")
        )
    return out


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": r2_score(y_true, y_pred),
    }


class Probe:
    """A fitted sklearn pipeline plus, for binary tasks, a decision threshold."""

    def __init__(self, model, threshold: Optional[float] = None):
        self.model = model
        self.threshold = threshold
        self.classes_ = getattr(model, "classes_", None)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.threshold is None:
            return self.model.predict(X)
        return np.where(self.predict_proba(X)[:, 1] >= self.threshold, self.classes_[1], self.classes_[0])


def _val_threshold(y_val: np.ndarray, p_val: np.ndarray, classes: np.ndarray) -> float:
    """Threshold on P(positive) that maximises balanced accuracy on val (Youden's J)."""
    best_thr, best = 0.5, -np.inf
    for thr in np.unique(p_val):
        pred = np.where(p_val >= thr, classes[1], classes[0])
        score = balanced_accuracy_score(y_val, pred)
        if score > best:
            best_thr, best = float(thr), score
    return best_thr


def fit_probe(
    task: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int = 0,
) -> Tuple[Probe, float, float]:
    """Fit one probe per grid value and keep the one that scores best on val.

    Classification is selected on val AUROC (balanced accuracy for multi-class) because
    heavily regularised logistic regressions squeeze probabilities around 0.5, where a
    fixed 0.5 cut-off can put every sample in one class. Binary probes then get their
    threshold from val too, so the test split is never used for any choice.
    """
    best, best_score, best_param = None, -np.inf, None
    if task == "classification":
        binary = len(np.unique(y_train)) == 2
        for C in C_GRID:
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=C, class_weight="balanced", max_iter=5000, random_state=seed),
            )
            model.fit(X_train, y_train)
            if binary and len(np.unique(y_val)) == 2:
                score = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
            else:
                score = balanced_accuracy_score(y_val, model.predict(X_val))
            if score > best_score:
                best, best_score, best_param = model, score, C
        threshold = None
        if binary:
            threshold = _val_threshold(y_val, best.predict_proba(X_val)[:, 1], best.classes_)
        return Probe(best, threshold), best_param, best_score

    for alpha in ALPHA_GRID:
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(X_train, y_train)
        score = -mean_absolute_error(y_val, model.predict(X_val))
        if score > best_score:
            best, best_score, best_param = model, score, alpha
    return Probe(best), best_param, best_score


def run(args: argparse.Namespace) -> pd.DataFrame:
    labels_df = pd.read_csv(args.labels_csv)
    labels_df[args.id_col] = labels_df[args.id_col].astype(str).str.strip()
    if args.exclude_regex:
        drop = labels_df[args.exclude_col].astype(str).str.contains(args.exclude_regex, regex=True)
        print(f"[probe] excluding {int(drop.sum())} rows where {args.exclude_col} ~ {args.exclude_regex!r}")
        labels_df = labels_df[~drop]
    labels = labels_df.drop_duplicates(args.id_col).set_index(args.id_col)[args.label_col]
    if args.task == "classification":
        labels = labels.dropna().astype(int)

    emb_root = Path(args.emb_root)
    data = {}
    for split in (args.train_split, args.val_split, args.test_split):
        ids, feats = load_split_embeddings(emb_root / split, pooling=args.pooling)
        data[split] = attach_labels(ids, feats, labels)
        print(f"[probe] {split}: {len(data[split][0])} labelled videos, dim={feats.shape[1]}")

    _, X_tr_full, y_tr_full = data[args.train_split]
    _, X_val, y_val = data[args.val_split]
    test_ids, X_te, y_te = data[args.test_split]

    curve_rows = []
    final_model = None
    for fraction in sorted(set(args.train_fractions)):
        seeds = [args.seeds[0]] if fraction >= 1.0 else args.seeds
        for seed in seeds:
            X_tr, y_tr = subsample(X_tr_full, y_tr_full, fraction, seed, args.task)
            if args.task == "classification" and len(np.unique(y_tr)) < 2:
                print(f"[probe][WARN] fraction={fraction} seed={seed}: single class, skipped")
                continue
            model, param, val_score = fit_probe(args.task, X_tr, y_tr, X_val, y_val, seed=seed)
            if args.task == "classification":
                probs = model.predict_proba(X_te)
                metrics = _classification_metrics(y_te, model.predict(X_te), probs)
            else:
                metrics = _regression_metrics(y_te, model.predict(X_te))
            curve_rows.append(
                {"fraction": fraction, "seed": seed, "n_train": len(y_tr), "param": param,
                 "threshold": model.threshold, "val_score": val_score, **metrics}
            )
            print(f"[probe] fraction={fraction:.3f} seed={seed} n={len(y_tr)} param={param} "
                  + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
            if fraction >= 1.0:
                final_model = model

    curve = pd.DataFrame(curve_rows)
    if args.curve_csv:
        Path(args.curve_csv).parent.mkdir(parents=True, exist_ok=True)
        curve.to_csv(args.curve_csv, index=False)
        print(f"[probe] wrote {args.curve_csv}")

    if args.out_csv:
        if final_model is None:
            raise ValueError("--out_csv needs 1.0 in --train_fractions (the full-data probe)")
        out = pd.DataFrame({args.id_col: test_ids})
        out[args.pred_col] = final_model.predict(X_te)
        if args.task == "classification":
            probs = final_model.predict_proba(X_te)
            classes = list(final_model.classes_)
            if len(classes) == 2:
                out[args.prob_col or f"prob_{args.label_col}"] = probs[:, 1]
            else:
                for j, cls in enumerate(classes):
                    out[f"prob_{cls}"] = probs[:, j]
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.out_csv, index=False)
        print(f"[probe] wrote {args.out_csv}")
    return curve


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Linear probe on cached CardioBench embeddings")
    p.add_argument("--emb_root", required=True, help="Directory with <split>/<video_id>.pt files")
    p.add_argument("--labels_csv", required=True, help="Split CSV holding the labels")
    p.add_argument("--label_col", required=True)
    p.add_argument("--id_col", default="unique_id")
    p.add_argument("--task", choices=["classification", "regression"], default="classification")
    p.add_argument("--pooling", choices=["mean", "meanstd"], default="mean")
    p.add_argument("--train_split", default="train")
    p.add_argument("--val_split", default="val")
    p.add_argument("--test_split", default="test")
    p.add_argument("--train_fractions", type=float, nargs="+", default=[1.0])
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--out_csv", default=None, help="Test predictions in the evaluation/ format")
    p.add_argument("--pred_col", default="pred")
    p.add_argument("--prob_col", default=None)
    p.add_argument("--curve_csv", default=None, help="Per-(fraction, seed) test metrics")
    p.add_argument("--exclude_col", default="path", help="labels_csv column tested by --exclude_regex")
    p.add_argument("--exclude_regex", default=None,
                   help="Drop label rows whose --exclude_col matches (from train, val and test)")
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
