"""
make_plot_v2.py

Reads predict_all CSV and plots R² bar chart.

Changes:
1) Single long horizontal plot.
2) Background colors indicate Physics vs Legacy vs HSC.
3) Removes 'DEC' and 'Random' bar.
4) Moves 'a_*' targets to HSC group.
5) Averages multi-band targets using REGEX.
6) Maps raw target names to clean labels.
7) Adds BLANK SPACE between groups for better separation.

Run:
  python make_plot_v2.py --suffix zdim16_nogeom_neighbors
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots

_here = Path(__file__).resolve().parent

# --- Configuration ---

# 1. BARS & COLORS
BAR_LABELS = ["Physics latents", "Instrument latents", "Untrained"]
BAR_KEYS   = ["r2_physics", "r2_instrument", "r2_untrained"]

# User Defined Colors:
COLORS = ["#8AC3EE", "#E5254E", "#B19221"]

# 2. GAP SETTING
GAP_SIZE = 0.5  # Amount of whitespace between groups

# 3. MAPPING DICTIONARY (Raw Name -> Plot Label)
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

# 4. AVERAGING PATTERNS (Regex -> New Name)
AVERAGE_PATTERNS = {
    r"^a_": "a",
    r"^legacy_GALDEPTH_": "legacy_GALDEPTH",
    r"^legacy_NOBS_": "legacy_NOBS",
    r"^legacy_PSFSIZE_": "legacy_PSFSIZE",
    r"^legacy_PSFDEPTH_": "legacy_PSFDEPTH",
    r"^hsc_.*_variance_value$": "hsc_variance_value",
    r"^hsc_.*_psf_fwhm$": "hsc_psf_fwhm",
}

# Define Groups
GROUP_PHYSICS = "Physics"
GROUP_LEGACY = "Legacy Prop."
GROUP_HSC = "HSC Prop."

# Background colors
BG_COLORS = {
    GROUP_PHYSICS: "#f0f0f0",      # Light Grey
    GROUP_LEGACY:  "#e6f2ff",      # Light Blue tint
    GROUP_HSC:     "#fff0e6",      # Light Orange tint
}

def load_and_process(csv_path):
    """Load CSV, filter, average bands (via regex), and assign groups."""
    df = pd.read_csv(csv_path)

    # 1. Remove 'DEC'
    df = df[df["target"] != "DEC"].copy()

    # 2. Average specific bands using Regex
    indices_to_drop = []
    new_rows = []

    for pattern, new_name in AVERAGE_PATTERNS.items():
        mask = df["target"].str.contains(pattern, regex=True)
        subset = df[mask]

        if not subset.empty:
            indices_to_drop.extend(subset.index.tolist())

            # Average the numeric columns
            numeric_cols = [c for c in BAR_KEYS if c in df.columns]
            averaged_vals = subset[numeric_cols].mean(numeric_only=True)

            # Create new row
            new_row = averaged_vals.to_dict()
            new_row["target"] = new_name
            new_row["task"] = subset.iloc[0]["task"] if "task" in subset.columns else "averaged"
            new_rows.append(new_row)

    # Remove the individual band rows and append averages
    df = df.drop(index=indices_to_drop)
    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

    # 3. Assign Groups
    def get_group(target_name):
        # HSC specific
        if target_name in ["a", "hsc_variance_value", "hsc_psf_fwhm"]:
            return GROUP_HSC
        if target_name.startswith("hsc_"):
            return GROUP_HSC

        # Legacy specific
        if target_name.lower() == "ebv":
            return GROUP_LEGACY
        if target_name.startswith("legacy_"):
            return GROUP_LEGACY

        # Physics (Default)
        return GROUP_PHYSICS

    df["group"] = df["target"].apply(get_group)

    return df

def make_plot(df, out_path, suffix):
    """Generate single horizontal plot with shaded backgrounds and gaps."""
    plt.style.use(["science", "no-latex"])  # publication style; use "nature" or "ieee" for other journals

    # Define order: Physics -> Legacy -> HSC
    group_order = [GROUP_PHYSICS, GROUP_LEGACY, GROUP_HSC]

    # Sort dataframe by group order, then by target name
    df["group_rank"] = df["group"].map({g: i for i, g in enumerate(group_order)})
    df = df.sort_values(by=["group_rank", "target"]).reset_index(drop=True)

    raw_targets = df["target"].tolist()
    display_labels = [LABEL_MAPPING.get(t, t) for t in raw_targets]
    n_targets = len(raw_targets)

    if n_targets == 0:
        print("No targets to plot.")
        return

    # --- Calculate X Positions with Gaps ---
    x_positions = []
    current_x = 0.0
    last_group = None

    # Store min/max x for each group to draw background later
    group_bounds = {}

    for i, row in df.iterrows():
        group = row["group"]

        # Add gap if group changes (but not for the very first item)
        if last_group is not None and group != last_group:
            current_x += GAP_SIZE

        # Track bounds
        if group not in group_bounds:
            group_bounds[group] = {"min": current_x, "max": current_x}
        else:
            group_bounds[group]["max"] = current_x

        x_positions.append(current_x)
        last_group = group
        current_x += 1.0 # Standard step between bars within a group

    x = np.array(x_positions)

    # Setup Plot
    # Adjust width dynamically: basic width + extra for gaps
    total_width = current_x
    fig_width = max(10, total_width * 0.6)
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    width = 0.25
    offsets = [-width, 0, width]

    # Plot Bars
    for i, (key, label) in enumerate(zip(BAR_KEYS, BAR_LABELS)):
        if key not in df.columns:
            continue

        vals = df[key].values.copy()
        vals_clean = np.where(np.isfinite(vals), vals, 0.0)

        ax.bar(x + offsets[i], vals_clean, width, label=label,
               color=COLORS[i], edgecolor="gray", linewidth=0.5, zorder=3)

    # --- Background Shading with Gaps ---
    for group in group_order:
        if group in group_bounds:
            # Start a bit before the first bar, end a bit after the last bar of the group
            start = group_bounds[group]["min"] - 0.5
            end = group_bounds[group]["max"] + 0.5

            # Draw background
            ax.axvspan(start, end, facecolor=BG_COLORS[group], alpha=1.0, zorder=0)

            # Add Group Label
            mid_point = (start + end) / 2
            ax.text(mid_point, 1.05, group, ha='center', va='bottom',
                    fontsize=12, fontweight='bold', color='#333333',
                    transform=ax.get_xaxis_transform())

    # Formatting
    ax.set_ylabel("R²", fontsize=12)
    ax.set_title(f"Downstream Performance ({suffix})", fontsize=14, y=1.1)

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
    p = argparse.ArgumentParser(description="Plot predict_all CSV (v2)")
    p.add_argument("--suffix", default="zdim16_nogeom_neighbors", help="Suffix used in predict_all output")
    p.add_argument("--output-dir", type=Path, default=_here, help="Directory with CSV and where to write output")
    args = p.parse_args()

    csv_path = args.output_dir / f"predict_all_{args.suffix}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Run predict_all first. Missing: {csv_path}")

    df = load_and_process(csv_path)
    plot_path = args.output_dir / f"predict_all_{args.suffix}_plot_v2.png"
    make_plot(df, plot_path, args.suffix)

if __name__ == "__main__":
    main()
