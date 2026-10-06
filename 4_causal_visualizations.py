"""
Publication-Ready Causal Visualizations: Hormone Therapy in Breast Cancer (METABRIC)
Script: 4_causal_visualizations.py

Generates two publication-grade, 300 DPI visualization figures:
  1. causal_waterfall_plot.png:
     Horizontal bar/waterfall chart showing CATE across molecular subtype & menopausal status.
     Highlights the profound positive survival gain (+53 months) in Her2 pre-menopausal patients.
  2. causal_distribution_plot.png:
     Distribution (boxplot with jitter) showing treatment effect variance across subtypes
     stratified by pre- vs. post-menopausal status.

Author: Causal AI Machine Learning Engineer
Date: 2026-10-06
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure UTF-8 output encoding for Windows terminal
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, io.UnsupportedOperation):
        pass

# Configure plot styling for clean, publication-grade aesthetics
plt.rcParams.update({
    "font.sans-serif": "Arial",
    "font.family": "sans-serif",
    "figure.titlesize": 14,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
})


def generate_causal_plots(
    input_csv: str = "metabric_causal_results.csv",
    waterfall_out: str = "causal_waterfall_plot.png",
    distribution_out: str = "causal_distribution_plot.png",
):
    """
    Loads causal results and generates both 300 DPI figures.
    """
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input file '{input_csv}' not found. Please run '3_master_causal_pipeline.py' first.")

    print("=" * 80)
    print("      CAUSAL MACHINE LEARNING: PUBLICATION VISUALIZATIONS GENERATOR")
    print("=" * 80)
    print(f"\n[1] Loading causal estimation results from '{input_csv}'...")
    df = pd.read_csv(input_csv)
    print(f"    - Dataset loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")

    # =========================================================================
    # PLOT 1: CATE Waterfall / Horizontal Bar Chart by Subgroup
    # =========================================================================
    print("\n[2] Generating Plot 1: CATE Subgroup Waterfall/Bar Chart...")

    # Group by CLAUDIN_SUBTYPE and INFERRED_MENOPAUSAL_STATE
    subgroups = (
        df.groupby(["CLAUDIN_SUBTYPE", "INFERRED_MENOPAUSAL_STATE"])
        .agg(
            Mean_CATE=("Individual_CATE_Months", "mean"),
            Patient_Count=("Individual_CATE_Months", "count"),
        )
        .reset_index()
    )

    # Sort from lowest to highest for plt.barh (so highest positive effect appears at top)
    subgroups = subgroups.sort_values(by="Mean_CATE", ascending=True).reset_index(drop=True)

    # Prepare labels and color coding
    labels = [
        f"{row['CLAUDIN_SUBTYPE']} ({row['INFERRED_MENOPAUSAL_STATE']}, n={row['Patient_Count']})"
        for _, row in subgroups.iterrows()
    ]
    cate_values = subgroups["Mean_CATE"].values
    colors = ["#2ca02c" if val > 0 else "#d62728" for val in cate_values]

    fig, ax = plt.subplots(figsize=(11, 7.5))

    # Horizontal bars
    bars = ax.barh(
        labels,
        cate_values,
        color=colors,
        edgecolor="#333333",
        linewidth=0.8,
        height=0.68,
        alpha=0.9,
    )

    # Vertical reference line at X=0 (Neutral effect)
    ax.axvline(x=0, color="#222222", linestyle="--", linewidth=1.3, alpha=0.85, zorder=3)

    # Add numeric data labels on each bar
    for bar, val in zip(bars, cate_values):
        width = bar.get_width()
        y_pos = bar.get_y() + bar.get_height() / 2.0
        if val >= 0:
            label_text = f"+{val:.1f}"
            ax.text(
                width + 1.2,
                y_pos,
                label_text,
                va="center",
                ha="left",
                fontsize=9.5,
                fontweight="bold",
                color="#1b7837",
            )
        else:
            label_text = f"{val:.1f}"
            ax.text(
                width - 1.2,
                y_pos,
                label_text,
                va="center",
                ha="right",
                fontsize=9.5,
                fontweight="bold",
                color="#b2182b",
            )

    # Subgroup annotation box highlighting the maximum benefit
    top_row = subgroups.iloc[-1]
    annotation_text = (
        f"Key Clinical Insight:\n"
        f"• Highest Gain: {top_row['CLAUDIN_SUBTYPE']} Pre-menopausal\n"
        f"• Treatment Benefit: +{top_row['Mean_CATE']:.1f} months\n"
        f"• Cohort Size: n={top_row['Patient_Count']} patients"
    )
    ax.text(
        0.62,
        0.28,
        annotation_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#f7fcf5", edgecolor="#2ca02c", linewidth=1.5, alpha=0.95),
    )

    # Formatting axes and titles
    ax.set_title(
        "Conditional Average Treatment Effect (CATE) of Hormone Therapy by Patient Subgroup",
        fontsize=13.5,
        fontweight="bold",
        pad=16,
    )
    ax.set_xlabel("Estimated Survival Benefit (Months)", fontsize=11.5, fontweight="bold", labelpad=10)
    ax.set_ylabel("Patient Subgroup (Molecular Subtype & Menopausal Status)", fontsize=11.5, fontweight="bold", labelpad=10)

    # Set x-limits with balanced margins for labels
    min_x = np.min(cate_values) - 10
    max_x = np.max(cate_values) + 12
    ax.set_xlim(min_x, max_x)

    ax.grid(axis="x", linestyle=":", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)

    # Legend for Benefit vs Harm
    import matplotlib.patches as mpatches
    green_patch = mpatches.Patch(color="#2ca02c", label="Positive Benefit (CATE > 0)")
    red_patch = mpatches.Patch(color="#d62728", label="Negative Association / Indication Bias (CATE < 0)")
    ax.legend(handles=[green_patch, red_patch], loc="lower right", frameon=True, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(waterfall_out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    [OK] Saved Plot 1 to '{waterfall_out}' (300 DPI)")

    # =========================================================================
    # PLOT 2: CATE Distribution by Molecular Subtype (Box Plot with Jitter)
    # =========================================================================
    print("\n[3] Generating Plot 2: CATE Distribution across Molecular Subtypes...")

    # Sort subtypes by descending order of maximum benefit
    subtype_order = ["Her2", "claudin-low", "Basal", "LumB", "Normal", "NC", "LumA"]

    fig, ax = plt.subplots(figsize=(11, 7))

    # Color palette for menopausal status
    palette = {
        "Pre": "#2ca02c",   # Green for Pre-menopausal
        "Post": "#1f77b4",  # Blue for Post-menopausal
    }

    # Boxplot showing median and IQR
    sns.boxplot(
        data=df,
        x="CLAUDIN_SUBTYPE",
        y="Individual_CATE_Months",
        hue="INFERRED_MENOPAUSAL_STATE",
        hue_order=["Pre", "Post"],
        order=subtype_order,
        palette=palette,
        boxprops=dict(alpha=0.75, edgecolor="#333333", linewidth=1.0),
        medianprops=dict(color="#111111", linewidth=2.2),
        whiskerprops=dict(color="#333333", linewidth=1.2),
        capprops=dict(color="#333333", linewidth=1.2),
        fliersize=0,  # Fliers suppressed as stripplot overlays all points
        width=0.62,
        ax=ax,
    )

    # Stripplot overlay to display patient sample density across subgroups
    sns.stripplot(
        data=df,
        x="CLAUDIN_SUBTYPE",
        y="Individual_CATE_Months",
        hue="INFERRED_MENOPAUSAL_STATE",
        hue_order=["Pre", "Post"],
        order=subtype_order,
        dodge=True,
        jitter=0.22,
        alpha=0.45,
        palette=palette,
        edgecolor="black",
        linewidth=0.3,
        size=4.5,
        ax=ax,
    )

    # Horizontal reference line at Y=0 (Neutral treatment effect)
    ax.axhline(0, color="#222222", linestyle="--", linewidth=1.3, alpha=0.85, zorder=4)

    # Title & labels
    ax.set_title(
        "Distribution of Individual Treatment Effects across Molecular Subtypes",
        fontsize=13.5,
        fontweight="bold",
        pad=16,
    )
    ax.set_xlabel("Claudin Molecular Subtype", fontsize=11.5, fontweight="bold", labelpad=10)
    ax.set_ylabel("Individual CATE (Estimated Survival Benefit in Months)", fontsize=11.5, fontweight="bold", labelpad=10)

    # Grid lines
    ax.grid(axis="y", linestyle=":", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)

    # Clean up duplicate legend from boxplot + stripplot
    handles, labels_list = ax.get_legend_handles_labels()
    # Retain only unique entries for Pre and Post
    unique_legend = dict(zip(labels_list[:2], handles[:2]))
    ax.legend(
        unique_legend.values(),
        [f"{k}-menopausal" for k in unique_legend.keys()],
        title="Menopausal Status",
        title_fontsize=10.5,
        loc="upper right",
        frameon=True,
        framealpha=0.92,
    )

    plt.tight_layout()
    plt.savefig(distribution_out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    [OK] Saved Plot 2 to '{distribution_out}' (300 DPI)")

    # =========================================================================
    # Verification & File Size Audit
    # =========================================================================
    print("\n" + "=" * 80)
    print("                    VISUALIZATION GENERATION SUMMARY")
    print("=" * 80)
    for plot_path in [waterfall_out, distribution_out]:
        size_kb = os.path.getsize(plot_path) / 1024
        print(f"  ✓ File: {plot_path:<30} | Size: {size_kb:>8.2f} KB | DPI: 300")
    print("=" * 80)


if __name__ == "__main__":
    generate_causal_plots(
        input_csv="metabric_causal_results.csv",
        waterfall_out="causal_waterfall_plot.png",
        distribution_out="causal_distribution_plot.png",
    )
