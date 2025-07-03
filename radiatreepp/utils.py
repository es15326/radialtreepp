import numpy as np
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from .core import radialTreee
from .config import RadialTreeConfig


def compute_radialtree_linkage(df, model_col, label_col):
    scores = df[model_col].values.reshape(-1, 1)
    if not np.all(np.isfinite(scores)):
        raise ValueError(f"Non-finite values in '{model_col}'.")
    dist_matrix = sch.distance.pdist(scores, metric="euclidean")
    linkage_matrix = sch.linkage(dist_matrix, method="ward")
    dendro = sch.dendrogram(
        linkage_matrix,
        labels=df[label_col].tolist(),
        no_plot=True
    )
    return dendro


def plot_and_add_legend(Z, title, config, colors_dict, legend_colors, legend_labels):
    fig_main, ax_main = plt.subplots(figsize=(12, 12))
    radialTreee(Z, ax=ax_main, config=config)
    ax_main.set_title(title, fontsize=16, pad=20)
    fig_main.show()

    # Manual legend
    fig_legend = plt.figure(figsize=(3, len(legend_labels) * 0.35))
    ax_legend = fig_legend.add_subplot(111)
    ax_legend.axis("off")
    legend_elements = [plt.Line2D([0], [0], color=c, lw=6, label=l) for c, l in zip(legend_colors, legend_labels)]
    ax_legend.legend(handles=legend_elements, loc="center", frameon=False, title="Category")
    fig_legend.tight_layout()
    fig_legend.show()
