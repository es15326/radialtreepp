import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.lines import Line2D
import os
import sys

from radiatreepp import radialTreee, RadialTreeConfig
from radiatreepp.utils import compute_radialtree_linkage

def run_example(csv_path: str = "log_feature_importance.csv", out_dir: str = "out_figs"):
    # === Check input CSV ===
    if not os.path.isfile(csv_path):
        print(f"❌ Error: File '{csv_path}' not found.")
        return

    # === Ensure output directory exists ===
    os.makedirs(out_dir, exist_ok=True)

    # === Load Data ===
    df = pd.read_csv(csv_path)

    # === Data Pre-processing ===
    df["XGBoost_log"] = np.log1p(df["XGBoost"])
    df["Feature_with_importance"] = df.apply(
        lambda row: f"{row['Feature']} ({row['XGBoost']:.3f})", axis=1
    )

    # === Color Setup ===
    categories = df["Category"].unique()
    cmap = colormaps.get_cmap("tab20")
    category_to_color_index = {cat: i for i, cat in enumerate(categories)}
    category_rgb = np.array([cmap(category_to_color_index[cat] % cmap.N) for cat in df["Category"]])
    colors_dict = {"Category": category_rgb}

    # === Legend Data ===
    legend_colors = [cmap(i % cmap.N) for i in range(len(categories))]
    legend_labels = list(categories)

    # === Dendrogram Data ===
    Z_xgb = compute_radialtree_linkage(df, "XGBoost_log", "Feature_with_importance")

    # === Plotting ===
    fig_main, ax_main = plt.subplots(figsize=(12, 12))
    config = RadialTreeConfig(
        colorlabels=colors_dict,
        gradient_colors=["black", "blue"],
        radial_labels=False,
        fontsize=8,
        label_radius=1.2,
        node_display_mode='inner',
        node_label_display_mode='top_3',
        node_size=5,
        node_label_fontsize=7
    )
    radialTreee(Z_xgb, ax=ax_main, config=config)
    ax_main.set_title("XGBoost Radial Dendrogram", fontsize=16, pad=20)
    
    dendrogram_path = os.path.join(out_dir, "radial_dendrogram_xgboost.png")
    fig_main.savefig(dendrogram_path, bbox_inches="tight", dpi=300)

    # === Legend ===
    fig_legend = plt.figure(figsize=(3, len(legend_labels) * 0.35))
    ax_legend = fig_legend.add_subplot(111)
    ax_legend.axis("off")
    legend_elements = [
        Line2D([0], [0], color=c, lw=6, label=l)
        for c, l in zip(legend_colors, legend_labels)
    ]
    ax_legend.legend(handles=legend_elements, loc="center", frameon=False, title="Category")
    fig_legend.tight_layout()

    legend_path = os.path.join(out_dir, "radial_dendrogram_legend_xgboost.png")
    fig_legend.savefig(legend_path, bbox_inches="tight", dpi=300)

    print("✅ Radial dendrogram complete.")
    print(f"📁 Figures saved to: {out_dir}/")

if __name__ == "__main__":
    # Use command-line CSV argument if given
    input_csv = sys.argv[1] if len(sys.argv) > 1 else "log_feature_importance.csv"
    run_example(csv_path=input_csv)
