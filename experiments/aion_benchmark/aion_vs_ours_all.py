"""
Compare AION vs Ours (physics + instrument) vs Untrained on downstream R².

Reads:
  - Ours: downstream_evaluation/final/predict_all_zdim16_nogeom_neighbors_table.csv (or .csv long format)
  - AION: r2_comparisons_aion/downstream_aion_512-256-128.csv
  - AION legacy: aion_legacy/r2_comparisons_aion_legacy_v2/downstream_aion_legacy_train_512-256-128.csv
  - AION MMU: aion_mmu/r2_comparisons_aion_mmu_v2/downstream_aion_mmu_matched_512-256-128.csv — merged into same AION-1 column (MMU fills where available).
  - Baseline: downstream_evaluation/final/predict_baseline.csv (r2_baseline).

Produces:
  1. Single grouped bar plot: Physics latents | Instrument latents | AION-1 | Untrained ResNet-18 | Baseline.
  2. CSV with all columns merged on target.

Run from galaxy_model:
  python aion_benchmark/aion_vs_ours_all.py
  python downstream_evaluation/final/aion_vs_ours_all.py   # script can live in either place
  python .../aion_vs_ours_all.py --baseline   # include baseline series
"""
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_script_dir = Path(__file__).resolve().parent
# Support script in aion_benchmark/ or in downstream_evaluation/final/
if _script_dir.name == "final" and _script_dir.parent.name == "downstream_evaluation":
    _galaxy_model = _script_dir.parent.parent
    _aion_benchmark = _galaxy_model / "aion_benchmark"
    _final_dir = _script_dir
else:
    _galaxy_model = _script_dir.parent
    _aion_benchmark = _script_dir
    _final_dir = _galaxy_model / "downstream_evaluation" / "final"

# Ours: prefer table (one row per target), fallback to long format
OURS_TABLE_CSV = _final_dir / "predict_all_zdim16_nogeom_neighbors_table.csv"
OURS_LONG_CSV = _final_dir / "predict_all_zdim16_nogeom_neighbors.csv"

# AION sources (under aion_benchmark)
AION_CSV = _aion_benchmark / "r2_comparisons_aion" / "downstream_aion_512-256-128.csv"
AION_LEGACY_CSV = _aion_benchmark / "aion_legacy" / "r2_comparisons_aion_legacy_v2" / "downstream_aion_legacy_train_512-256-128.csv"
AION_MMU_CSV = _aion_benchmark / "aion_mmu" / "r2_comparisons_aion_mmu_v2" / "downstream_aion_mmu_matched_512-256-128.csv"
BASELINE_CSV = _final_dir / "predict_baseline.csv"

# Outputs go to aion_benchmark/r2_comparisons_aion so they stay in one place
OUT_DIR = _aion_benchmark / "r2_comparisons_aion"
OUT_CSV = OUT_DIR / "aion_vs_ours_all_merged.csv"
OUT_PNG = OUT_DIR / "aion_vs_ours_all_512-256-128.png"

# Averaging patterns (same logic as downstream_evaluation/final/makeplot_v2.py): regex -> averaged target name
AVERAGE_PATTERNS = {
    r"^a_": "a",
    r"^legacy_GALDEPTH_": "legacy_GALDEPTH",
    r"^legacy_NOBS_": "legacy_NOBS",
    r"^legacy_PSFSIZE_": "legacy_PSFSIZE",
    r"^legacy_PSFDEPTH_": "legacy_PSFDEPTH",
    r"^hsc_.*_variance_value$": "hsc_variance_value",
    r"^hsc_.*_psf_fwhm$": "hsc_psf_fwhm",
}

# Bar colors (order: Physics latents, Instrument latents, AION-1, Untrained ResNet-18, Baseline)
BAR_COLORS = ["#2E86AB", "#CC546D", "#6CA75D", "#9473C6", "#BF823B"]

# Groups and background shading (same as makeplot_v2)
GROUP_PHYSICS = "Physics"
GROUP_LEGACY = "Legacy Prop."
GROUP_HSC = "HSC Prop."
GAP_SIZE = 0.5
# More visible block colors (same hue as before, more saturated)
BG_COLORS = {
    GROUP_PHYSICS: "#d8d8d8",
    GROUP_LEGACY: "#b8d4f0",
    GROUP_HSC: "#f5dcc4",
}
# Group label text: darker shade of each block so labels match but remain readable
GROUP_LABEL_COLORS = {
    GROUP_PHYSICS: "#888888",
    GROUP_LEGACY: "#5a9fd4",
    GROUP_HSC: "#c49b6a",
}
LABEL_MAPPING = {
    "a": "HSC Extinction (a)",
    "hsc_variance_value": "HSC Variance",
    "hsc_psf_fwhm": "HSC PSF FWHM",
    "legacy_GALDEPTH": "Legacy Depth",
    "legacy_NOBS": "Legacy # Obs",
    "legacy_PSFSIZE": "Legacy PSF Size",
    "legacy_PSFDEPTH": "Legacy PSF Depth",
    "EBV": "E(B-V)",
    "Z": "Redshift",
    "logMstar": "Stellar Mass",
}


def _get_group(target_name):
    """Assign group for background shading (same logic as makeplot_v2)."""
    t = str(target_name)
    if t in ["a", "hsc_variance_value", "hsc_psf_fwhm"] or t.startswith("hsc_"):
        return GROUP_HSC
    if t.lower() == "ebv" or t.startswith("legacy_"):
        return GROUP_LEGACY
    return GROUP_PHYSICS


def average_by_patterns(df):
    """Apply same averaging logic as makeplot_v2; exclude hsc_z_psf_fwhm from hsc_psf_fwhm average."""
    df = df[df["target"] != "DEC"].copy()

    numeric_cols = [c for c in ["Physics_latents", "Instrument_latents", "Untrained", "AION", "Baseline"] if c in df.columns]
    indices_to_drop = []
    new_rows = []

    for pattern, new_name in AVERAGE_PATTERNS.items():
        mask = df["target"].astype(str).str.contains(pattern, regex=True)
        subset = df.loc[mask].copy()
        # Leave out hsc_z_psf_fwhm from the hsc_psf_fwhm average
        if new_name == "hsc_psf_fwhm":
            subset = subset[subset["target"] != "hsc_z_psf_fwhm"]

        if not subset.empty:
            indices_to_drop.extend(subset.index.tolist())
            averaged = subset[numeric_cols].mean(numeric_only=True).to_dict()
            averaged["target"] = new_name
            if "group" in df.columns:
                averaged["group"] = subset.iloc[0]["group"]
            new_rows.append(averaged)

    df = df.drop(index=indices_to_drop)
    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    # Assign group for every row (for plotting)
    df["group"] = df["target"].apply(_get_group)
    return df


def load_ours():
    """Load our predictions: table CSV (preferred) or long CSV. Returns df with target, Physics_latents, Instrument_latents, Untrained, and optionally group, Random."""
    if OURS_TABLE_CSV.exists():
        df = pd.read_csv(OURS_TABLE_CSV)
        # Drop placeholder rows (e.g. "legacy only" with no values)
        df = df.dropna(subset=["Physics latents", "Instrument latents"], how="all")
        df = df[df["target"].astype(str).str.strip().astype(bool)]
        out = pd.DataFrame({
            "target": df["target"].values,
            "Physics_latents": df["Physics latents"].values,
            "Instrument_latents": df["Instrument latents"].values,
            "Untrained": df["Untrained"].values,
        })
        if "group" in df.columns:
            out["group"] = df["group"].values
        if "Random" in df.columns:
            out["Random"] = df["Random"].values
        return out
    if OURS_LONG_CSV.exists():
        df = pd.read_csv(OURS_LONG_CSV)
        # One row per target; keep first occurrence if any duplicate
        df = df.drop_duplicates(subset=["target"], keep="first")
        out = pd.DataFrame({
            "target": df["target"].values,
            "Physics_latents": df["r2_physics"].values,
            "Instrument_latents": df["r2_instrument"].values,
            "Untrained": df["r2_untrained"].values,
        })
        if "r2_random" in df.columns:
            out["Random"] = df["r2_random"].values
        return out
    raise FileNotFoundError(f"Ours CSV not found at {OURS_TABLE_CSV} or {OURS_LONG_CSV}")


def load_baseline():
    """Load baseline R² from predict_baseline.csv (instrument targets only)."""
    if not BASELINE_CSV.exists():
        return None
    df = pd.read_csv(BASELINE_CSV)[["target", "r2_baseline"]].rename(columns={"r2_baseline": "Baseline"})
    return df


def load_aion_combined():
    """Load all AION sources (instrument, legacy, MMU) into one AION-1 series. MMU overwrites where same target exists."""
    aion = pd.read_csv(AION_CSV)[["target", "r2_mean"]].rename(columns={"r2_mean": "AION"})
    legacy = pd.read_csv(AION_LEGACY_CSV)[["target", "r2_mean"]].rename(columns={"r2_mean": "AION"})
    parts = [aion, legacy]
    if AION_MMU_CSV.exists():
        mmu = pd.read_csv(AION_MMU_CSV)[["target", "r2_mean"]].rename(columns={"r2_mean": "AION"})
        parts = [mmu, aion, legacy]  # MMU first so it overwrites for overlapping targets
    combined = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["target"], keep="first")
    return combined


def main():
    p = argparse.ArgumentParser(description="AION vs Ours downstream R² comparison")
    p.add_argument("--baseline", action="store_true", help="Include baseline series from predict_baseline.csv")
    args = p.parse_args()

    ours = load_ours()
    aion = load_aion_combined()
    baseline = load_baseline() if args.baseline else None

    # Merge: ours + AION + baseline (left join so we keep all ours targets)
    merged = ours.merge(aion, on="target", how="left")
    if baseline is not None:
        merged = merged.merge(baseline, on="target", how="left")

    # Average multi-band targets using same logic as makeplot_v2
    merged = average_by_patterns(merged)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Sort by group (Physics -> Legacy -> HSC), then target
    group_order = [GROUP_PHYSICS, GROUP_LEGACY, GROUP_HSC]
    merged["group_rank"] = merged["group"].map({g: i for i, g in enumerate(group_order)})
    merged = merged.sort_values(by=["group_rank", "target"]).reset_index(drop=True)

    # Exclude hsc_z_psf_fwhm from plot (and CSV)
    merged = merged[merged["target"] != "hsc_z_psf_fwhm"].reset_index(drop=True)
    merged.to_csv(OUT_CSV, index=False)
    print(f"Merged CSV saved: {OUT_CSV}")

    targets = merged["target"].tolist()
    display_labels = [LABEL_MAPPING.get(t, t) for t in targets]
    n_t = len(targets)

    # X positions with gaps between groups (same as makeplot_v2)
    x_positions = []
    current_x = 0.0
    last_group = None
    group_bounds = {}
    for _, row in merged.iterrows():
        group = row["group"]
        if last_group is not None and group != last_group:
            current_x += GAP_SIZE
        if group not in group_bounds:
            group_bounds[group] = {"min": current_x, "max": current_x}
        else:
            group_bounds[group]["max"] = current_x
        x_positions.append(current_x)
        last_group = group
        current_x += 1.0
    x = np.array(x_positions)
    total_width = current_x

    n_series = 5 if baseline is not None and "Baseline" in merged.columns else 4
    width = 0.18 if n_series == 5 else 0.22
    offsets = np.linspace(-(n_series - 1) / 2, (n_series - 1) / 2, n_series) * width

    fig_width = max(10, total_width * 0.6)
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    series = [
        ("Physics latents", merged["Physics_latents"].values, BAR_COLORS[0]),
        ("Instrument latents", merged["Instrument_latents"].values, BAR_COLORS[1]),
        ("AION-1", merged["AION"].values, BAR_COLORS[2]),
        ("Untrained ResNet-18", merged["Untrained"].values, BAR_COLORS[3]),
    ]
    if n_series == 5:
        series.append(("Baseline", merged["Baseline"].values, BAR_COLORS[4]))
    for i, (label, vals, color) in enumerate(series):
        vals = np.where(np.isnan(vals), 0.0, vals)
        ax.bar(x + offsets[i], vals, width, label=label, color=color, alpha=0.85, edgecolor="gray", linewidth=0.5, zorder=3)

    # Background shading and group labels (label color matches block)
    for group in group_order:
        if group in group_bounds:
            start = group_bounds[group]["min"] - 0.5
            end = group_bounds[group]["max"] + 0.5
            ax.axvspan(start, end, facecolor=BG_COLORS[group], alpha=1.0, zorder=0)
            mid = (start + end) / 2
            ax.text(mid, 1.05, group, ha="center", va="bottom", fontsize=12, fontweight="bold", color=GROUP_LABEL_COLORS[group], transform=ax.get_xaxis_transform())

    ax.set_ylabel("R²")
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, rotation=45, ha="right", fontsize=9)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(-0.15, 1.05)
    ax.grid(axis="y", alpha=0.3, linestyle="--", zorder=1)
    ax.axhline(0, color="gray", linewidth=0.7, zorder=2)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
