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
    if not os.path.isfile(csv_path):
        print(f"❌ Error: File '{csv_path}' not found.")
        return

    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    # === Data Pre-processing ===
    df["TabNet_log"] = np.log1p(df["TabNet"])
    df["Feature_with_importance"] = df.apply(
        lambda row: f"{row['Feature']} ({row['TabNet']:.3f})", axis=1
    )

    # === Color Setup ===
    categories = df["Category"].unique()
    cmap = colormaps.get_cmap("Set3")
    category_to_color_index = {cat: i for i, cat in enumerate(categories)}
    category_rgb = np.array([cmap(category_to_color_index[cat] % cmap.N) for cat in df["Category"]])
    colors_dict = {"Category": category_rgb}

    # === Legend Data ===
    legend_colors = [cmap(i % cmap.N) for i in range(len(categories))]
    legend_labels = list(categories)

    # === Dendrogram Data ===
    Z_tabnet = compute_radialtree_linkage(df, "TabNet_log", "Feature_with_importance")

    # === Plotting ===
    fig_main, ax_main = plt.subplots(figsize=(12, 12))
    config = RadialTreeConfig(
        colorlabels=colors_dict,
        gradient_colors=["black", "green"],
        radial_labels=False,
        fontsize=8,
        label_radius=1.2,
        node_display_mode='inner',
        node_label_display_mode='top_3',
        node_size=5,
        node_label_fontsize=7
    )
    radialTreee(Z_tabnet, ax=ax_main, config=config)
    ax_main.set_title("TabNet Radial Dendrogram", fontsize=16, pad=20)

    dendrogram_path = os.path.join(out_dir, "radial_dendrogram_tabnet.png")
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

    legend_path = os.path.join(out_dir, "radial_dendrogram_tabnet_legend.png")
    fig_legend.savefig(legend_path, bbox_inches="tight", dpi=300)

    print("✅ TabNet radial dendrogram complete.")
    print(f"📁 Figures saved to: {out_dir}/")

if __name__ == "__main__":
    input_csv = sys.argv[1] if len(sys.argv) > 1 else "log_feature_importance.csv"
    run_example(csv_path=input_csv)
