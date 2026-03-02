"""
makeplot_v2_contrastive.py

Reads predict_all_contrastive CSV and plots R² bar chart.

Run:
  python makeplot_v2_contrastive.py --suffix contrastive_latest
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots

_here = Path(__file__).resolve().parent

BAR_LABELS = ["Physics latents", "Instrument latents", "Untrained"]
BAR_KEYS = ["r2_physics", "r2_instrument", "r2_untrained"]
COLORS = ["#8AC3EE", "#E5254E", "#B19221"]

GAP_SIZE = 0.5

LABEL_MAPPING = {
    "a": "HSC Extinction (a)",
    "hsc_variance_value": "HSC Variance",
    "hsc_psf_fwhm": "HSC PSF FWHM",
    "legacy_GALDEPTH": "Legacy Depth",
    "legacy_NOBS": "Legacy # Obs",
    "legacy_PSFSIZE": "Legacy PSF Size",
    "legacy_PSFDEPTH": "Legacy PSF Depth",
    "EBV": "E(B-V)",
    "ebv": "E(B-V)",
    "Z": "Redshift",
    "logMstar": "Stellar Mass",
}

AVERAGE_PATTERNS = {
    r"^a_": "a",
    r"^legacy_GALDEPTH_": "legacy_GALDEPTH",
    r"^legacy_NOBS_": "legacy_NOBS",
    r"^legacy_PSFSIZE_": "legacy_PSFSIZE",
    r"^legacy_PSFDEPTH_": "legacy_PSFDEPTH",
    r"^hsc_.*_variance_value$": "hsc_variance_value",
    r"^hsc_.*_psf_fwhm$": "hsc_psf_fwhm",
}

GROUP_PHYSICS = "Physics"
GROUP_LEGACY = "Legacy Prop."
GROUP_HSC = "HSC Prop."

BG_COLORS = {
    GROUP_PHYSICS: "#f0f0f0",
    GROUP_LEGACY: "#e6f2ff",
    GROUP_HSC: "#fff0e6",
}


def _get_group(target_name):
    if target_name in ["a", "hsc_variance_value", "hsc_psf_fwhm"] or str(target_name).startswith("hsc_"):
        return GROUP_HSC
    if str(target_name).lower() == "ebv" or str(target_name).startswith("legacy_"):
        return GROUP_LEGACY
    return GROUP_PHYSICS


def _average_by_patterns(df):
    indices_to_drop = []
    new_rows = []
    for pattern, new_name in AVERAGE_PATTERNS.items():
        mask = df["target"].astype(str).str.contains(pattern, regex=True)
        subset = df[mask]
        if subset.empty:
            continue
        indices_to_drop.extend(subset.index.tolist())
        numeric_cols = [c for c in BAR_KEYS if c in df.columns]
        averaged_vals = subset[numeric_cols].mean(numeric_only=True)
        new_row = averaged_vals.to_dict()
        new_row["target"] = new_name
        new_rows.append(new_row)
    df = df.drop(index=indices_to_drop)
    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    return df


def load_and_process(csv_path, include_hsc_provabgs=False):
    df = pd.read_csv(csv_path)
    if "objective" in df.columns and not include_hsc_provabgs:
        df = df[df["objective"] != "hsc_provabgs"].copy()

    # Keep one row per target if objectives overlap.
    numeric_cols = [c for c in BAR_KEYS if c in df.columns]
    if "target" in df.columns:
        df = df.groupby("target", as_index=False)[numeric_cols].mean(numeric_only=True)

    df = df[df["target"] != "DEC"].copy()
    df = _average_by_patterns(df)
    df["group"] = df["target"].apply(_get_group)
    return df


def make_plot(df, out_path, suffix):
    plt.style.use(["science", "no-latex"])

    group_order = [GROUP_PHYSICS, GROUP_LEGACY, GROUP_HSC]
    df["group_rank"] = df["group"].map({g: i for i, g in enumerate(group_order)})
    df = df.sort_values(by=["group_rank", "target"]).reset_index(drop=True)

    raw_targets = df["target"].tolist()
    display_labels = [LABEL_MAPPING.get(t, t) for t in raw_targets]
    n_targets = len(raw_targets)
    if n_targets == 0:
        print("No targets to plot.")
        return

    x_positions = []
    current_x = 0.0
    last_group = None
    group_bounds = {}
    for _, row in df.iterrows():
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
    fig_width = max(10, total_width * 0.6)
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    width = 0.25
    offsets = [-width, 0, width]
    for i, (key, label) in enumerate(zip(BAR_KEYS, BAR_LABELS)):
        if key not in df.columns:
            continue
        vals = pd.to_numeric(df[key], errors="coerce").to_numpy()
        vals_clean = np.where(np.isfinite(vals), vals, 0.0)
        ax.bar(
            x + offsets[i],
            vals_clean,
            width,
            label=label,
            color=COLORS[i],
            edgecolor="gray",
            linewidth=0.5,
            zorder=3,
        )

    for group in group_order:
        if group not in group_bounds:
            continue
        start = group_bounds[group]["min"] - 0.5
        end = group_bounds[group]["max"] + 0.5
        ax.axvspan(start, end, facecolor=BG_COLORS[group], alpha=1.0, zorder=0)
        mid_point = (start + end) / 2
        ax.text(
            mid_point,
            1.05,
            group,
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color="#333333",
            transform=ax.get_xaxis_transform(),
        )

    ax.set_ylabel("R²", fontsize=12)
    ax.set_title(f"Downstream Performance (contrastive: {suffix})", fontsize=14, y=1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, rotation=45, ha="right", fontsize=10)
    ax.set_ylim(-0.05, 1.1)
    ax.grid(axis="y", alpha=0.3, linestyle="--", zorder=1)
    ax.axhline(0, color="black", linewidth=0.8, zorder=2)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=10)

    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved: {out_path}")
    plt.close()


def main():
    p = argparse.ArgumentParser(description="Plot contrastive downstream CSV (v2)")
    p.add_argument("--suffix", default="contrastive_latest", help="Suffix used in predict_all_contrastive output")
    p.add_argument("--output-dir", type=Path, default=_here, help="Directory with CSV and where to write output")
    p.add_argument("--include-hsc-provabgs", action="store_true", help="Also include hsc_provabgs objective rows")
    args = p.parse_args()

    csv_path = args.output_dir / f"predict_all_contrastive_{args.suffix}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Run predict_all_contrastive first. Missing: {csv_path}")

    df = load_and_process(csv_path, include_hsc_provabgs=args.include_hsc_provabgs)
    plot_path = args.output_dir / f"predict_all_contrastive_{args.suffix}_plot_v2.png"
    make_plot(df, plot_path, args.suffix)


if __name__ == "__main__":
    main()
