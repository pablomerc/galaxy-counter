"""
Downstream evaluation for AION on Legacy ProvaBGS (v2): train an MLP to predict
galaxy/physics parameters from mean AION embeddings produced by
aion_legacy/prepare_aion_legacy.py (downstream_aion_legacy_train.h5).

Uses same target columns as predict_aion_legacyprovabgs (physics_provabgs).
Default dataset: aion_legacy/downstream_aion_legacy_train.h5.

Run:
  python aion_benchmark/predict_aion_legacy_v2.py
  python aion_benchmark/predict_aion_legacy_v2.py --dataset path/to/downstream_aion_legacy_train.h5 --targets desi_Z LOG_MSTAR
"""
import argparse
import csv
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

_here = Path(__file__).resolve().parent

# Default path (from aion_legacy/prepare_aion_legacy.py)
DATASET_PATH = _here / "downstream_aion_legacy_train.h5"
R2_OUT_DIR = _here / "r2_comparisons_aion_legacy_v2"

# Legacy ProvaBGS targets (same as predict_aion_legacyprovabgs)
physics_provabgs = ["desi_Z", "LOG_MSTAR", "TAGE_MW", "LOG_Z_MW", "sSFR", "hsc_g_extendedness_value", "DEC"]
DEFAULT_TARGETS = physics_provabgs

# Embedding keys (from prepare_aion_legacy.py: embeddings_mean, embeddings_mean_legacy, embeddings_mean_hsc)
EMBEDDING_KEYS = [
    ("mean", "embeddings_mean"),
    ("mean_legacy", "embeddings_mean_legacy"),
    ("mean_hsc", "embeddings_mean_hsc"),
]
EMBEDDING_LABELS = {
    "mean": "Mean (Legacy+HSC)",
    "mean_legacy": "Mean (Legacy only)",
    "mean_hsc": "Mean (HSC only)",
}


def load_h5_aion_legacy_v2(path, verbose=True):
    """
    Load mean AION embeddings and numeric labels from downstream_aion_legacy_train.h5
    (produced by aion_legacy/prepare_aion_legacy.py).
    Returns embeddings_dict (name -> (N, dim)), metadata (N, num_targets), param_names.
    Only keeps numeric label columns.
    """
    with h5py.File(path, "r") as f:
        n = None
        embeddings_dict = {}
        for name, key in EMBEDDING_KEYS:
            if key not in f:
                if name == "mean":
                    raise ValueError(f"No '{key}' in {path}")
                continue
            arr = np.array(f[key][:])
            if n is None:
                n = arr.shape[0]
            elif arr.shape[0] != n:
                raise ValueError(f"Length mismatch: {key} has {arr.shape[0]}, expected {n}")
            embeddings_dict[name] = arr
        if not embeddings_dict:
            raise ValueError(f"No embedding keys found in {path}")

        raw_cols = f.attrs.get("label_columns", [])
        label_columns = []
        for c in raw_cols if isinstance(raw_cols, (list, tuple)) else list(raw_cols):
            label_columns.append(c.decode("utf-8") if isinstance(c, bytes) else c)
        if not label_columns:
            label_columns = [k.replace("labels/", "") for k in f.keys() if k.startswith("labels/")]

        meta_list = []
        param_names = []
        dropped = []
        for col in label_columns:
            key = "labels/" + col
            if key not in f:
                dropped.append((col, None, "key not in file"))
                continue
            arr = np.array(f[key][:])
            if verbose:
                print(f"  {col}: shape={arr.shape}, dtype={arr.dtype}")
            if arr.dtype.kind not in "fiu":
                try:
                    arr = arr.astype(np.float64)
                except Exception:
                    dropped.append((col, arr.shape, "non-numeric dtype"))
                    continue
            if arr.shape[0] != n:
                dropped.append((col, arr.shape, f"length {arr.shape[0]} != n_samples {n}"))
                continue
            if arr.ndim == 1:
                meta_list.append(arr.astype(np.float64))
                param_names.append(col)
            elif arr.ndim == 2:
                for j in range(arr.shape[1]):
                    meta_list.append(arr[:, j].astype(np.float64))
                    param_names.append(f"{col}_{j}")
                if verbose:
                    print(f"    -> split into {arr.shape[1]} columns: {col}_0 .. {col}_{arr.shape[1]-1}")
            else:
                dropped.append((col, arr.shape, "ndim > 2"))

        if verbose and dropped:
            print("  Dropped:")
            for col, sh, reason in dropped:
                print(f"    - {col}: shape={sh}, reason={reason}")
            print(f"  Kept {len(param_names)} label columns.")

        if not meta_list:
            raise ValueError("No numeric label columns found in H5.")

        metadata = np.stack(meta_list, axis=1)

    return embeddings_dict, metadata, param_names


def _standardize_with_stats(data, mean, std):
    std = np.where(std == 0, 1.0, std)
    return (np.asarray(data, dtype=np.float64) - mean) / (std + 1e-8)


MLP_HIDDEN = (512, 256, 128)
MLP_SUFFIX = "-".join(map(str, MLP_HIDDEN))


class MLPRegressor(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=None, dropout=0.2):
        if hidden is None:
            hidden = MLP_HIDDEN
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [
                nn.Linear(prev, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class LitRegressorAion(pl.LightningModule):
    def __init__(self, in_dim, out_dim, hidden=None, dropout=0.2, lr=1e-3, weight_decay=1e-2):
        if hidden is None:
            hidden = MLP_HIDDEN
        super().__init__()
        self.save_hyperparameters()
        self.model = MLPRegressor(in_dim=in_dim, out_dim=out_dim, hidden=hidden, dropout=dropout)
        self.loss_fn = nn.SmoothL1Loss(beta=1.0)

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, stage):
        x, y = batch
        y_hat = self(x)
        y_hat = torch.nan_to_num(y_hat, nan=0.0)
        loss = self.loss_fn(y_hat, y)
        self.log(f"{stage}/loss", loss, on_epoch=True, prog_bar=True, batch_size=x.size(0))
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def configure_optimizers(self):
        opt = AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.trainer.max_epochs)
        return {"optimizer": opt, "lr_scheduler": sched}

    def on_before_optimizer_step(self, optimizer):
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)


def evaluate_per_target(model, loader, param_names, device):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            all_preds.append(y_hat.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    results = []
    for i in range(all_targets.shape[1]):
        y_true = all_targets[:, i]
        y_pred = all_preds[:, i]
        valid = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_t = y_true[valid]
        y_p = y_pred[valid]
        if len(y_t) == 0:
            r2, mae, rmse = np.nan, np.nan, np.nan
        else:
            r2 = np.nan if np.std(y_t) < 1e-6 else r2_score(y_t, y_p)
            mae = mean_absolute_error(y_t, y_p)
            rmse = np.sqrt(mean_squared_error(y_t, y_p))
        name = param_names[i] if i < len(param_names) else f"target_{i}"
        results.append({"target": name, "r2": r2, "mae": mae, "rmse": rmse, "n_valid": len(y_t)})
    return results


def train_and_eval(train_loader, val_loader, param_names, emb_dim, out_dim, use_gpu, precision):
    model = LitRegressorAion(
        in_dim=emb_dim,
        out_dim=out_dim,
        hidden=MLP_HIDDEN,
        dropout=0.2,
        lr=1e-3,
        weight_decay=1e-2,
    )
    ckpt = ModelCheckpoint(
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        filename="aion_legacy_v2-{epoch:02d}-{val_loss:.4f}",
    )
    trainer = Trainer(
        max_epochs=100,
        accelerator="gpu" if use_gpu else "cpu",
        devices=1,
        precision=precision,
        callbacks=[
            ckpt,
            EarlyStopping(monitor="val/loss", mode="min", patience=20),
            LearningRateMonitor(logging_interval="epoch"),
        ],
        log_every_n_steps=20,
    )
    trainer.fit(model, train_loader, val_loader)
    best = (
        LitRegressorAion.load_from_checkpoint(ckpt.best_model_path)
        if ckpt.best_model_path
        else model
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return evaluate_per_target(best.to(device), val_loader, param_names, device)


def save_results_csv(results_by_experiment, param_names, filepath):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    names = list(results_by_experiment.keys())
    fieldnames = ["target"] + [f"r2_{n}" for n in names] + [f"mae_{n}" for n in names]
    with open(filepath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for t in param_names:
            row = {"target": t}
            for n in names:
                r = next((x for x in results_by_experiment[n] if x["target"] == t), None)
                if r is not None:
                    row[f"r2_{n}"] = r["r2"] if not np.isnan(r["r2"]) else ""
                    row[f"mae_{n}"] = r["mae"] if not np.isnan(r["mae"]) else ""
                else:
                    row[f"r2_{n}"] = row[f"mae_{n}"] = ""
            w.writerow(row)
    print(f"Results CSV saved: {filepath}")


def save_plot(results_by_experiment, param_names, filepath):
    names = list(results_by_experiment.keys())
    labels = [EMBEDDING_LABELS.get(n, n) for n in names]
    plot_data = []
    for t in param_names:
        r2s = []
        for n in names:
            r = next((x for x in results_by_experiment[n] if x["target"] == t), None)
            r2s.append(r["r2"] if r is not None and not np.isnan(r["r2"]) else 0.0)
        if any(v != 0 for v in r2s):
            plot_data.append({"target": t, "r2s": r2s})
    if not plot_data:
        return
    targets = [d["target"] for d in plot_data]
    n_t = len(targets)
    x = np.arange(n_t)
    width = 0.35
    offsets = [-(width / 2), (width / 2)] if len(names) == 2 else np.linspace(-width * (len(names) - 1) / 2, width * (len(names) - 1) / 2, len(names))
    if len(names) != 2:
        offsets = list(offsets)
    colors = ["#2E86AB", "#E07A5F", "#81B29A"][: len(names)]
    fig, ax = plt.subplots(figsize=(max(12, n_t * 0.4), 6))
    for i, (n, lbl) in enumerate(zip(names, labels)):
        vals = [d["r2s"][i] for d in plot_data]
        off = offsets[i] if i < len(offsets) else offsets[-1]
        ax.bar(x + off, vals, width, label=lbl, color=colors[i % len(colors)], alpha=0.8)
    ax.set_xlabel("Target")
    ax.set_ylabel("R²")
    ax.set_title("AION mean embeddings → Legacy ProvaBGS parameters (MLP, v2)")
    ax.set_xticks(x)
    ax.set_xticklabels(targets, rotation=45, ha="right", fontsize=9)
    ax.legend()
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    print(f"Plot saved: {filepath}")
    plt.close()


def main():
    p = argparse.ArgumentParser(
        description="Train MLP to predict Legacy ProvaBGS parameters from mean AION embeddings (aion_legacy/prepare_aion_legacy.py data)"
    )
    p.add_argument(
        "--dataset",
        type=str,
        default=str(DATASET_PATH),
        help="Path to downstream_aion_legacy_train.h5",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(R2_OUT_DIR),
        help="Output directory for CSV and plot",
    )
    p.add_argument(
        "--targets",
        type=str,
        nargs="*",
        default=None,
        help="Target columns (default: physics_provabgs); empty = all numeric labels",
    )
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    target_columns = None if (args.targets is not None and len(args.targets) == 0) else (args.targets if args.targets is not None else DEFAULT_TARGETS)

    use_gpu = torch.cuda.is_available()
    precision = "16-mixed" if use_gpu else "32-true"
    print(f"GPU: {'yes' if use_gpu else 'no'}")

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        return

    print(f"Loading {dataset_path}...")
    embeddings_dict, meta, param_names = load_h5_aion_legacy_v2(str(dataset_path))
    for name, arr in embeddings_dict.items():
        print(f"  {name}: {arr.shape}")
    print(f"  labels: {meta.shape}, {len(param_names)} params")

    if target_columns is not None:
        col_set = set(target_columns)
        indices = [i for i, n in enumerate(param_names) if n in col_set]
        missing = col_set - {param_names[i] for i in indices}
        if missing:
            print(f"  Warning: targets not in H5 (skipped): {missing}")
        if not indices:
            raise ValueError("No requested target columns found in H5.")
        meta = meta[:, indices]
        param_names = [param_names[i] for i in indices]
    print(f"  Using {len(param_names)} targets")

    finite = np.isfinite(meta).all(axis=1)
    for name in embeddings_dict:
        embeddings_dict[name] = np.nan_to_num(embeddings_dict[name][finite], nan=0.0, posinf=0.0, neginf=0.0)
    meta = meta[finite]
    n = len(meta)

    rng = np.random.default_rng(args.seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(0.9 * n)
    tr_idx, va_idx = idx[:n_train], idx[n_train:]

    mean = meta[tr_idx].mean(axis=0)
    std = meta[tr_idx].std(axis=0)
    meta = _standardize_with_stats(meta, mean, std)

    out_dim = meta.shape[1]
    results_by_experiment = {}

    for exp_name, emb in embeddings_dict.items():
        print(f"\nTraining MLP for {EMBEDDING_LABELS.get(exp_name, exp_name)}...")
        train_ds = TensorDataset(
            torch.tensor(emb[tr_idx], dtype=torch.float32),
            torch.tensor(meta[tr_idx], dtype=torch.float32),
        )
        val_ds = TensorDataset(
            torch.tensor(emb[va_idx], dtype=torch.float32),
            torch.tensor(meta[va_idx], dtype=torch.float32),
        )
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)
        emb_dim = emb.shape[1]
        results_by_experiment[exp_name] = train_and_eval(
            train_loader, val_loader, param_names, emb_dim, out_dim, use_gpu, precision
        )

    names = list(results_by_experiment.keys())
    fmt = lambda v: "    NaN" if np.isnan(v) else f"{v:>14.4f}"
    col_w = max(14, max(len(EMBEDDING_LABELS.get(n, n)) for n in names))
    print("\n" + "=" * (32 + col_w * len(names)))
    print("R² — AION mean embeddings → Legacy ProvaBGS parameters (v2)")
    print("=" * (32 + col_w * len(names)))
    header = f"{'Target':<32} " + " ".join(f"{EMBEDDING_LABELS.get(n, n):>{col_w}}" for n in names)
    print(header)
    print("-" * (32 + col_w * len(names)))
    for t in param_names:
        r2s = [next((r["r2"] for r in results_by_experiment[n] if r["target"] == t), np.nan) for n in names]
        print(f"{t:<32} " + " ".join(fmt(r) for r in r2s))
    avgs = [np.nanmean([next((r["r2"] for r in results_by_experiment[n] if r["target"] == t), np.nan) for t in param_names]) for n in names]
    print("-" * (32 + col_w * len(names)))
    print(f"{'Average':<32} " + " ".join(fmt(a) for a in avgs))
    print("=" * (32 + col_w * len(names)))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = dataset_path.stem
    save_results_csv(results_by_experiment, param_names, out_dir / f"{stem}_{MLP_SUFFIX}.csv")
    save_plot(results_by_experiment, param_names, out_dir / f"{stem}_{MLP_SUFFIX}.png")


if __name__ == "__main__":
    main()
