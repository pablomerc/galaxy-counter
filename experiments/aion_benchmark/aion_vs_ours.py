"""
Compare AION vs Ours (physics + instrument) vs Untrained on downstream R².

Reads:
  - r2_comparisons_aion/downstream_aion_512-256-128.csv  (AION: r2_mean, r2_mean_legacy)
  - downstream_evaluation/r2_comparisons_neighbors/downstream_neighbors_*_legacy_hsc_512-256-128.csv
    (Ours: r2_trained_emb1=physics, r2_trained_emb2=instrument; r2_untrained_emb1/emb2)

Produces two grouped bar plots:
  1. R²: AION | Ours (Physics) | Ours (Instrument) | Untrained.
  2. Delta R²: AION, Ours (Physics), Ours (Instrument) each minus Untrained.

Run from galaxy_model:
  python aion_benchmark/aion_vs_ours.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_here = Path(__file__).resolve().parent
_galaxy_model = _here.parent

# Paths to the two CSVs
AION_CSV = _here / "r2_comparisons_aion" / "downstream_aion_512-256-128.csv"
OURS_CSV = _galaxy_model / "downstream_evaluation" / "r2_comparisons_neighbors" / "downstream_neighbors_zdim16_nogeom_neighbors_legacy_hsc_512-256-128.csv"
OUT_PNG = _here / "r2_comparisons_aion" / "aion_vs_ours_512-256-128.png"
OUT_PNG_DELTA = _here / "r2_comparisons_aion" / "aion_vs_ours_delta_512-256-128.png"


def load_and_merge():
    """Load both CSVs and merge on target."""
    aion = pd.read_csv(AION_CSV)
    ours = pd.read_csv(OURS_CSV)

    # Merge on target (keep order from aion)
    merged = aion[["target", "r2_mean"]].merge(
        ours[["target", "r2_trained_emb1", "r2_trained_emb2", "r2_untrained_emb1", "r2_untrained_emb2"]],
        on="target",
        how="inner",
    )

    targets = merged["target"].values
    r2_aion = merged["r2_mean"].values
    r2_ours_physics = merged["r2_trained_emb1"].values
    r2_ours_instrument = merged["r2_trained_emb2"].values
    r2_untrained = (merged["r2_untrained_emb1"].values + merged["r2_untrained_emb2"].values) / 2

    return targets, r2_aion, r2_ours_physics, r2_ours_instrument, r2_untrained


def main():
    targets, r2_aion, r2_ours_physics, r2_ours_instrument, r2_untrained = load_and_merge()
    n_t = len(targets)
    # Space targets apart so bars don't run together; 1.0 = no gap
    spacing = 1.4
    x = np.arange(n_t) * spacing

    # Four series: AION, Ours (Physics), Ours (Instrument), Untrained
    width = 0.22
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * width
    colors = ["#2E86AB", "#81B29A", "#E07A5F", "#6C757D"]  # AION, Ours Physics, Ours Instrument, Untrained
    labels = ["AION-1", "Ours (Physics)", "Ours (Instrument)", "Untrained ResNet-18"]

    fig, ax = plt.subplots(figsize=(max(14, n_t * 0.55), 6))
    for i, (vals, lbl, c) in enumerate(
        zip([r2_aion, r2_ours_physics, r2_ours_instrument, r2_untrained], labels, colors)
    ):
        vals = np.where(np.isnan(vals), 0.0, vals)
        ax.bar(x + offsets[i], vals, width, label=lbl, color=c, alpha=0.85)

    # Dashed vertical lines between targets
    for i in range(1, n_t):
        ax.axvline(x=(i - 0.5) * spacing, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)

    ax.set_xlabel("Target")
    ax.set_ylabel("R²")
    ax.set_title("AION vs Ours (Physics / Instrument) vs Untrained — downstream R²")
    ax.set_xticks(x)
    ax.set_xticklabels(targets, rotation=45, ha="right", fontsize=9)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(-0.1, 0.95)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.axhline(0, color="gray", linewidth=0.7)
    plt.tight_layout()

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {OUT_PNG}")

    # --- Delta R² plot: each method minus untrained ---
    delta_aion = r2_aion - r2_untrained
    delta_ours_physics = r2_ours_physics - r2_untrained
    delta_ours_instrument = r2_ours_instrument - r2_untrained

    fig2, ax2 = plt.subplots(figsize=(max(14, n_t * 0.55), 6))
    width_delta = 0.24
    offsets_delta = np.array([-1, 0, 1]) * width_delta  # three bars: AION, Ours Physics, Ours Instrument
    colors_delta = ["#2E86AB", "#81B29A", "#E07A5F"]
    labels_delta = ["AION", "Ours (Physics)", "Ours (Instrument)"]
    for i, (vals, lbl, c) in enumerate(
        zip([delta_aion, delta_ours_physics, delta_ours_instrument], labels_delta, colors_delta)
    ):
        vals = np.where(np.isnan(vals), 0.0, vals)
        ax2.bar(x + offsets_delta[i], vals, width_delta, label=lbl, color=c, alpha=0.85)

    for i in range(1, n_t):
        ax2.axvline(x=(i - 0.5) * spacing, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)

    ax2.set_xlabel("Target")
    ax2.set_ylabel("Δ R² (vs Untrained)")
    ax2.set_title("Delta R²: AION vs Ours (Physics / Instrument) over Untrained")
    ax2.set_xticks(x)
    ax2.set_xticklabels(targets, rotation=45, ha="right", fontsize=9)
    ax2.legend(loc="upper right", fontsize=9)
    ax2.set_ylim(-0.1, 0.6)
    ax2.grid(axis="y", alpha=0.3, linestyle="--")
    ax2.axhline(0, color="gray", linewidth=0.7)
    plt.tight_layout()
    plt.savefig(OUT_PNG_DELTA, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {OUT_PNG_DELTA}")


if __name__ == "__main__":
    main()
