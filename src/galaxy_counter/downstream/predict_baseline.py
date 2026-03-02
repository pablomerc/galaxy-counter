"""
Downstream prediction for baseline (mean/std) features from baseline_all.py.

Loads downstream_neighbors_baseline.h5 (features/hsc, features/legacy, features/hsc_legacy + labels),
trains MLP regressors for the same neighbor tasks as predict_all.py:
  - instrument_neighbors_legacy
  - instrument_neighbors_hsc (+ derived hsc_*_psf_fwhm)

Uses the concatenated hsc_legacy (16-dim) features. Outputs CSV with task, target,
r2_baseline, mae_baseline, r2_mean, mae_mean.

Run from galaxy_model/ or downstream_evaluation/:
  python downstream_evaluation/final/predict_baseline.py
  python downstream_evaluation/final/predict_baseline.py --output-dir downstream_evaluation/final
"""
import argparse
import csv
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

from galaxy_counter.downstream import prepare_all as pa

# Baseline H5 has a single feature set: use hsc_legacy (16-dim)
FEATURE_KEY = "features/hsc_legacy"

# Same neighbor tasks as predict_all.py (neighbors only)
TASKS = [
    ("instrument_neighbors_legacy", pa.instrument_neighbors_legacy),
    ("instrument_neighbors_hsc", pa.instrument_neighbors_hsc + pa.instrument_neighbors_hsc_psf_fwhm),
]

MLP_HIDDEN = (256, 128)
BATCH_SIZE = 64
TRAIN_FRAC = 0.9


def compute_hsc_psf_seeing(shape11, shape22):
    """HSC PSF FWHM in arcsec from shape moments (pixel scale 0.168). Same as predict_all.py."""
    pixel_scale_hsc = 0.168
    return 2.355 * np.sqrt((shape11 + shape22) / 2) * pixel_scale_hsc


def load_baseline_h5(path, add_derived=True, verbose=False):
    """
    Load features and labels from downstream_neighbors_baseline.h5.
    Returns (features, metadata, param_names). metadata is (n, n_targets), param_names list.
    If add_derived=True, appends hsc_*_psf_fwhm from shape11/22 like predict_all.
    """
    with h5py.File(path, "r") as f:
        if FEATURE_KEY not in f:
            raise ValueError(f"Missing '{FEATURE_KEY}' in {path}. Run baseline_all.py first.")
        features = np.array(f[FEATURE_KEY][:])

        raw_cols = f.attrs.get("label_columns", [])
        label_columns = [c.decode("utf-8") if isinstance(c, bytes) else c for c in (raw_cols if isinstance(raw_cols, (list, tuple)) else list(raw_cols))]
        if not label_columns:
            label_columns = [k.replace("labels/", "") for k in f.keys() if k.startswith("labels/")]

        meta_list = []
        param_names = []
        for col in label_columns:
            key = "labels/" + col
            if key not in f:
                continue
            arr = np.array(f[key][:])
            if arr.dtype.kind not in "fiu":
                try:
                    arr = arr.astype(np.float64)
                except Exception:
                    continue
            if arr.ndim == 1:
                meta_list.append(arr)
                param_names.append(col)
            elif arr.ndim == 2:
                for j in range(arr.shape[1]):
                    meta_list.append(arr[:, j].astype(np.float64))
                    param_names.append(f"{col}_{j}")
            else:
                continue

        if not meta_list:
            raise ValueError(f"No numeric label columns in {path}")
        n = meta_list[0].shape[0]

        if add_derived:
            for band in ("g", "i", "r", "z"):
                name_11 = f"hsc_{band}_sdssshape_psf_shape11"
                name_22 = f"hsc_{band}_sdssshape_psf_shape22"
                if name_11 in param_names and name_22 in param_names:
                    idx11 = param_names.index(name_11)
                    idx22 = param_names.index(name_22)
                    shape11 = meta_list[idx11]
                    shape22 = meta_list[idx22]
                    psf_fwhm = compute_hsc_psf_seeing(shape11, shape22)
                    meta_list.append(psf_fwhm.astype(np.float64))
                    param_names.append(f"hsc_{band}_psf_fwhm")
                    if verbose:
                        print(f"  Derived: hsc_{band}_psf_fwhm")

        metadata = np.stack(meta_list, axis=1)
    return features, metadata, param_names


def _standardize_with_stats(data, mean, std):
    std = np.where(std == 0, 1.0, std)
    return (np.asarray(data, dtype=np.float64) - mean) / (std + 1e-8)


class MLPRegressor(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=None, dropout=0.2):
        super().__init__()
        hidden = hidden or MLP_HIDDEN
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class LitRegressor(pl.LightningModule):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = MLPRegressor(in_dim=in_dim, out_dim=out_dim, **kwargs)
        self.loss_fn = nn.SmoothL1Loss(beta=1.0)

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, stage):
        x, y = batch
        y_hat = torch.nan_to_num(self(x), nan=0.0)
        loss = self.loss_fn(y_hat, y)
        self.log(f"{stage}/loss", loss, on_epoch=True, prog_bar=True, batch_size=x.size(0))
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def configure_optimizers(self):
        opt = AdamW(self.parameters(), lr=1e-3, weight_decay=1e-2)
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
        y_t, y_p = y_true[valid], y_pred[valid]
        if len(y_t) == 0:
            r2, mae, rmse = np.nan, np.nan, np.nan
        else:
            r2 = np.nan if np.std(y_t) < 1e-6 else r2_score(y_t, y_p)
            mae = mean_absolute_error(y_t, y_p)
            rmse = np.sqrt(mean_squared_error(y_t, y_p))
        name = param_names[i] if i < len(param_names) else f"target_{i}"
        results.append({"target": name, "r2": r2, "mae": mae, "rmse": rmse, "n_valid": len(y_t)})
    return results


def evaluate_mean_baseline(meta_tr, meta_val, param_names, tr_idx, va_idx):
    """Predict training mean for validation set."""
    mean_tr = meta_tr[tr_idx].mean(axis=0)
    meta_val_subset = meta_val[va_idx]
    results = []
    for i in range(meta_val_subset.shape[1]):
        y_true = meta_val_subset[:, i]
        y_pred = np.full_like(y_true, mean_tr[i])
        valid = ~np.isnan(y_true)
        y_t, y_p = y_true[valid], y_pred[valid]
        if len(y_t) == 0:
            r2, mae, rmse = np.nan, np.nan, np.nan
        else:
            r2 = np.nan if np.std(y_t) < 1e-6 else r2_score(y_t, y_p)
            mae = mean_absolute_error(y_t, y_p)
            rmse = np.sqrt(mean_squared_error(y_t, y_p))
        name = param_names[i] if i < len(param_names) else f"target_{i}"
        results.append({"target": name, "r2": r2, "mae": mae, "rmse": rmse, "n_valid": len(y_t)})
    return results


def train_and_eval_one(features, meta, param_names, tr_idx, va_idx, use_gpu, seed):
    mean = meta[tr_idx].mean(axis=0)
    std = meta[tr_idx].std(axis=0)
    meta_s = _standardize_with_stats(meta, mean, std)
    feats = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    x = torch.tensor(feats, dtype=torch.float32)
    m = torch.tensor(meta_s, dtype=torch.float32)
    train_ds = TensorDataset(x[tr_idx], m[tr_idx])
    val_ds = TensorDataset(x[va_idx], m[va_idx])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    in_dim = features.shape[1]
    out_dim = meta_s.shape[1]

    model = LitRegressor(in_dim=in_dim, out_dim=out_dim)
    ckpt = ModelCheckpoint(monitor="val/loss", mode="min", save_top_k=1, save_last=False)
    trainer = Trainer(
        max_epochs=100,
        accelerator="gpu" if use_gpu else "cpu",
        devices=1,
        callbacks=[ckpt, EarlyStopping(monitor="val/loss", mode="min", patience=20), LearningRateMonitor(logging_interval="epoch")],
        log_every_n_steps=20,
        enable_progress_bar=True,
    )
    trainer.fit(model, train_loader, val_loader)
    best = LitRegressor.load_from_checkpoint(ckpt.best_model_path) if ckpt.best_model_path else model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return evaluate_per_target(best.to(device), val_loader, param_names, device)


def run_task(task_name, target_list, h5_path, seed, use_gpu):
    features, metadata, param_names = load_baseline_h5(
        h5_path, add_derived=True, verbose=(task_name == "instrument_neighbors_hsc")
    )

    col_set = set(target_list)
    indices = [i for i, n in enumerate(param_names) if n in col_set]
    missing = col_set - {param_names[i] for i in indices}
    if missing:
        print(f"  Warning {task_name}: targets not in H5 (skipped): {missing}")
    if not indices:
        raise ValueError(f"No requested target columns for {task_name} in H5.")
    param_names = [param_names[i] for i in indices]
    meta = metadata[:, indices].astype(np.float64)

    finite = np.isfinite(meta).all(axis=1)
    n = np.sum(finite)
    if n == 0:
        raise ValueError(f"No finite targets for {task_name}")
    meta = meta[finite]
    feats = features[finite]
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(TRAIN_FRAC * n)
    tr_idx, va_idx = idx[:n_train], idx[n_train:]

    res_baseline = train_and_eval_one(feats, meta, param_names, tr_idx, va_idx, use_gpu, seed)
    res_mean = evaluate_mean_baseline(meta, meta, param_names, tr_idx, va_idx)
    return {"baseline": res_baseline, "mean": res_mean}, param_names


def save_results_csv(all_results, output_path):
    """all_results: list of (task_name, results_dict, param_names). results_dict has 'baseline' and 'mean'."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["task", "target", "r2_baseline", "mae_baseline", "r2_mean", "mae_mean"]
    with open(output_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for task_name, results_per_variant, param_names in all_results:
            for t in param_names:
                row = {"task": task_name, "target": t}
                r = next((x for x in results_per_variant["baseline"] if x["target"] == t), None)
                if r is not None:
                    row["r2_baseline"] = r["r2"] if not np.isnan(r["r2"]) else ""
                    row["mae_baseline"] = r["mae"] if not np.isnan(r["mae"]) else ""
                else:
                    row["r2_baseline"] = row["mae_baseline"] = ""
                r = next((x for x in results_per_variant["mean"] if x["target"] == t), None)
                if r is not None:
                    row["r2_mean"] = r["r2"] if not np.isnan(r["r2"]) else ""
                    row["mae_mean"] = r["mae"] if not np.isnan(r["mae"]) else ""
                else:
                    row["r2_mean"] = row["mae_mean"] = ""
                w.writerow(row)
    print(f"Results CSV saved: {output_path}")


def main():
    p = argparse.ArgumentParser(description="Predict downstream targets from baseline (mean/std) features")
    p.add_argument("--output-dir", type=Path, default=_here, help="Directory containing downstream_neighbors_baseline.h5 and where to write CSV")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-gpu", action="store_true", help="Disable GPU")
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    h5_path = output_dir / "downstream_neighbors_baseline.h5"
    if not h5_path.exists():
        raise FileNotFoundError(f"Run baseline_all.py first. Missing: {h5_path}")

    use_gpu = not args.no_gpu and torch.cuda.is_available()
    print(f"GPU: {'yes' if use_gpu else 'no'}")

    all_results = []
    for task_name, target_list in TASKS:
        if not target_list:
            continue
        print(f"\n--- Task: {task_name} ({len(target_list)} targets) ---")
        try:
            results_per_variant, param_names = run_task(task_name, target_list, h5_path, args.seed, use_gpu)
            all_results.append((task_name, results_per_variant, param_names))
            for t in param_names:
                r2b = next((r["r2"] for r in results_per_variant["baseline"] if r["target"] == t), np.nan)
                r2m = next((r["r2"] for r in results_per_variant["mean"] if r["target"] == t), np.nan)
                print(f"  {t}: R2 baseline={r2b:.4f} mean={r2m:.4f}")
        except Exception as e:
            print(f"  Error: {e}")
            raise

    csv_path = output_dir / "predict_baseline.csv"
    save_results_csv(all_results, csv_path)
    print("\nDone.")


if __name__ == "__main__":
    main()
