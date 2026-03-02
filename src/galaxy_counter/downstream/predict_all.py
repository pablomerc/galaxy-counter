"""
Downstream prediction for all targets from prepare_all.py H5s.

Uses one set of H5s per suffix (downstream_mmu_{suffix}.h5, downstream_legacy_provabgs_{suffix}.h5,
downstream_neighbors_{suffix}.h5), each with real / untrained embeddings.

Trains 5 separate model groups:
  1. physics_mmu          — MMU H5, encoder1 (physics), targets SHAPE_E1, SHAPE_E2, SHAPE_R
  2. instrument_mmu       — MMU H5, encoder2 (instrument), targets EBV + a_g,a_i,a_r,a_y,a_z
  3. physics_provabgs     — Legacy ProvaBGS H5, encoder1, targets desi_Z, LOG_MSTAR, ...
  4. instrument_neighbors_legacy — Neighbors H5, encoder2, targets legacy_PSFSIZE_*, ...
  5. instrument_neighbors_hsc    — Neighbors H5, encoder2, targets hsc_*_variance_value + derived hsc_*_psf_fwhm

For each task we train on real embeddings with BOTH encoder1 (physics) and encoder2 (instrument), plus
untrained. Also computes mean baseline (predicting training mean). Saves one CSV with columns: task, target,
r2_physics, r2_instrument, r2_untrained, r2_mean, mae_physics, mae_instrument, mae_untrained, mae_mean.

Run from galaxy_model/ or downstream_evaluation/:
  python downstream_evaluation/final/predict_all.py --suffix zdim16_nogeom_neighbors
  python downstream_evaluation/final/predict_all.py --suffix zdim16_nogeom_neighbors --output-dir downstream_evaluation/final
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

# Target lists and embedding keys (must match prepare_all.py and predict_neighbors.py)
from galaxy_counter.downstream import prepare_all as pa

# Which embedding pair to use for each dataset (legacy_hsc = HSC+Legacy concatenated)
KEY1 = "hsc_legacy_encoder1"
KEY2 = "hsc_legacy_encoder2"
EMBEDDING_VARIANTS = ("real", "untrained", "random")  # suffix: "", "_untrained", "_random"

# Tasks: (name, h5_stem, use_embedding, target_list)
# use_embedding 1 = encoder1 (physics), 2 = encoder2 (instrument)
TASKS = [
    ("physics_mmu", "mmu", 1, pa.physics_mmu),
    ("instrument_mmu", "mmu", 2, pa.instrument_mmu_legacy + pa.instrument_mmu_hsc),
    ("physics_provabgs", "legacy_provabgs", 1, pa.physics_provabgs),
    ("instrument_neighbors_legacy", "neighbors", 2, pa.instrument_neighbors_legacy),
    ("instrument_neighbors_hsc", "neighbors", 2, pa.instrument_neighbors_hsc + pa.instrument_neighbors_hsc_psf_fwhm),
]

MLP_HIDDEN = (256, 128)
BATCH_SIZE = 64
TRAIN_FRAC = 0.9


def compute_hsc_psf_seeing(shape11, shape22):
    """HSC PSF FWHM in arcsec from shape moments (pixel scale 0.168). Same as predict_neighbors.py."""
    pixel_scale_hsc = 0.168
    return 2.355 * np.sqrt((shape11 + shape22) / 2) * pixel_scale_hsc


def load_h5_variants(path, key1, key2, add_neighbors_derived=False, verbose=False):
    """
    Load real and untrained embedding pairs + labels from an H5 produced by prepare_all.py.
    Returns dict with keys "real", "untrained"; each value is (emb1, emb2, metadata, param_names).
    For neighbors H5, if add_neighbors_derived=True, adds hsc_*_psf_fwhm from shape11/22.
    """
    with h5py.File(path, "r") as f:
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

        # Add derived HSC PSF FWHM for neighbors (same as predict_neighbors load_h5)
        if add_neighbors_derived:
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

        out = {}
        for variant in EMBEDDING_VARIANTS:
            suf = "" if variant == "real" else "_untrained"
            k1, k2 = key1 + suf, key2 + suf
            if k1 not in f or k2 not in f:
                raise ValueError(f"Missing {k1} or {k2} in {path}")
            emb1 = np.array(f[k1][:])
            emb2 = np.array(f[k2][:])
            if emb1.shape[0] != n or emb2.shape[0] != n:
                raise ValueError(f"Length mismatch: {k1} {emb1.shape[0]} vs metadata {n}")
            out[variant] = (emb1, emb2, metadata.copy(), list(param_names))
    return out


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
    def __init__(self, in_dim, out_dim, use_embedding=1, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = MLPRegressor(in_dim=in_dim, out_dim=out_dim, **kwargs)
        self.loss_fn = nn.SmoothL1Loss(beta=1.0)

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, stage):
        emb1, emb2, y = batch
        x = emb1 if self.hparams.use_embedding == 1 else emb2
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


def evaluate_per_target(model, loader, param_names, device, use_embedding=1):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for emb1, emb2, y in loader:
            x = emb1 if use_embedding == 1 else emb2
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
    """Evaluate mean baseline: predict training mean for validation set."""
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


def train_and_eval_one(emb, meta_tr, param_names, tr_idx, va_idx, use_embedding, use_gpu, seed):
    mean = meta_tr[tr_idx].mean(axis=0)
    std = meta_tr[tr_idx].std(axis=0)
    meta_s = _standardize_with_stats(meta_tr, mean, std)
    emb1, emb2 = emb[0], emb[1]
    e1 = torch.tensor(emb1, dtype=torch.float32)
    e2 = torch.tensor(emb2, dtype=torch.float32)
    m = torch.tensor(meta_s, dtype=torch.float32)
    train_ds = TensorDataset(e1[tr_idx], e2[tr_idx], m[tr_idx])
    val_ds = TensorDataset(e1[va_idx], e2[va_idx], m[va_idx])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    emb_dim = emb1.shape[1]
    out_dim = meta_s.shape[1]

    model = LitRegressor(in_dim=emb_dim, out_dim=out_dim, use_embedding=use_embedding)
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
    return evaluate_per_target(best.to(device), val_loader, param_names, device, use_embedding)


def run_task(task_name, h5_stem, use_embedding, target_list, h5_path, seed, use_gpu):
    add_derived = h5_stem == "neighbors"
    data = load_h5_variants(h5_path, KEY1, KEY2, add_neighbors_derived=add_derived, verbose=(task_name == "instrument_neighbors_hsc"))

    # Use real variant to get param_names and build target index
    _, _, metadata, param_names = data["real"]
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
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(TRAIN_FRAC * n)
    tr_idx, va_idx = idx[:n_train], idx[n_train:]

    results_per_variant = {}
    for variant in EMBEDDING_VARIANTS:
        emb1, emb2, _, _ = data[variant]
        emb1 = np.nan_to_num(emb1[finite], nan=0.0, posinf=0.0, neginf=0.0)
        emb2 = np.nan_to_num(emb2[finite], nan=0.0, posinf=0.0, neginf=0.0)
        emb = (emb1, emb2)
        if variant == "real":
            # Evaluate with BOTH physics (encoder1) and instrument (encoder2) so we store both R² per target
            res_physics = train_and_eval_one(emb, meta, param_names, tr_idx, va_idx, use_embedding=1, use_gpu=use_gpu, seed=seed)
            res_instrument = train_and_eval_one(emb, meta, param_names, tr_idx, va_idx, use_embedding=2, use_gpu=use_gpu, seed=seed)
            results_per_variant["real_physics"] = res_physics
            results_per_variant["real_instrument"] = res_instrument
        else:
            res = train_and_eval_one(emb, meta, param_names, tr_idx, va_idx, use_embedding, use_gpu=use_gpu, seed=seed)
            results_per_variant[variant] = res

    # Compute mean baseline
    res_mean = evaluate_mean_baseline(meta, meta, param_names, tr_idx, va_idx)
    results_per_variant["mean"] = res_mean

    return results_per_variant, param_names


def save_results_csv(all_results, output_path):
    """all_results: list of (task_name, results_per_variant, param_names).
    results_per_variant has real_physics, real_instrument, untrained, mean."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task", "target",
        "r2_physics", "r2_instrument", "r2_untrained", "r2_mean",
        "mae_physics", "mae_instrument", "mae_untrained", "mae_mean",
    ]
    with open(output_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for task_name, results_per_variant, param_names in all_results:
            for t in param_names:
                row = {"task": task_name, "target": t}
                for key, r2_col, mae_col in [
                    ("real_physics", "r2_physics", "mae_physics"),
                    ("real_instrument", "r2_instrument", "mae_instrument"),
                ]:
                    r = next((x for x in results_per_variant.get(key, []) if x["target"] == t), None)
                    if r is not None:
                        row[r2_col] = r["r2"] if not np.isnan(r["r2"]) else ""
                        row[mae_col] = r["mae"] if not np.isnan(r["mae"]) else ""
                    else:
                        row[r2_col] = row[mae_col] = ""
                for v in ("untrained", "mean"):
                    r = next((x for x in results_per_variant.get(v, []) if x["target"] == t), None)
                    if r is not None:
                        row[f"r2_{v}"] = r["r2"] if not np.isnan(r["r2"]) else ""
                        row[f"mae_{v}"] = r["mae"] if not np.isnan(r["mae"]) else ""
                    else:
                        row[f"r2_{v}"] = row[f"mae_{v}"] = ""
                w.writerow(row)
    print(f"Results CSV saved: {output_path}")


def main():
    p = argparse.ArgumentParser(description="Predict all downstream targets (real / untrained embeddings + mean baseline)")
    p.add_argument("--suffix", required=True, help="Suffix used in prepare_all (e.g. zdim16_nogeom_neighbors)")
    p.add_argument("--output-dir", type=Path, default=_here, help="Directory containing the 3 H5s and where to write CSV")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-gpu", action="store_true", help="Disable GPU")
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    h5_paths = {
        "mmu": output_dir / f"downstream_mmu_{args.suffix}.h5",
        "legacy_provabgs": output_dir / f"downstream_legacy_provabgs_{args.suffix}.h5",
        "neighbors": output_dir / f"downstream_neighbors_{args.suffix}.h5",
    }
    for k, path in h5_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Run prepare_all first. Missing: {path}")

    use_gpu = not args.no_gpu and torch.cuda.is_available()
    print(f"GPU: {'yes' if use_gpu else 'no'}")

    all_results = []
    for task_name, h5_stem, use_embedding, target_list in TASKS:
        if not target_list:
            continue
        h5_path = h5_paths["mmu" if h5_stem == "mmu" else "legacy_provabgs" if h5_stem == "legacy_provabgs" else "neighbors"]
        print(f"\n--- Task: {task_name} ({len(target_list)} targets) ---")
        try:
            results_per_variant, param_names = run_task(task_name, h5_stem, use_embedding, target_list, h5_path, args.seed, use_gpu)
            all_results.append((task_name, results_per_variant, param_names))
            for t in param_names:
                r2p = next((r["r2"] for r in results_per_variant["real_physics"] if r["target"] == t), np.nan)
                r2i = next((r["r2"] for r in results_per_variant["real_instrument"] if r["target"] == t), np.nan)
                r2u = next((r["r2"] for r in results_per_variant["untrained"] if r["target"] == t), np.nan)
                r2mean = next((r["r2"] for r in results_per_variant["mean"] if r["target"] == t), np.nan)
                print(f"  {t}: R2 physics={r2p:.4f} instrument={r2i:.4f} untrained={r2u:.4f} mean={r2mean:.4f}")
        except Exception as e:
            print(f"  Error: {e}")
            raise

    csv_path = output_dir / f"predict_all_{args.suffix}.csv"
    save_results_csv(all_results, csv_path)
    print("\nDone. Use a separate script to plot from the CSV if desired.")


if __name__ == "__main__":
    main()
