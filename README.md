# 📊 radiatreepp

**RadialTree++** is a Python package for generating **radial dendrograms** from hierarchical clustering output, with rich visual overlays for **SHAP-style feature importances**, **semantic rings**, and **custom annotations**.

---

## 🚀 Features

- 📐 **Radial dendrograms** for hierarchical clustering (via SciPy)
- 🎨 **Gradient edge coloring** (e.g., by average depth or SHAP value)
- 🧠 **Semantic ring overlays** to display feature groups or categories
- 🔤 **Flexible label layout** (radial or horizontal)
- 🔘 **Node highlighting options** (e.g., only inner merges, or top N)
- 🧩 Easy to integrate with any feature importance method (XGBoost, TabNet, etc.)

---

## 📦 Installation

```bash
pip install radiatreepp
```

Or clone locally and install in editable mode:

```bash
git clone git@github.com:es15326/radialtreepp.git
cd radiatreepp
pip install -e .
```

---

## 🧪 Minimal Working Example

Want to quickly test how `radiatreepp` works? Here's a complete, minimal example using synthetic data. It demonstrates how to compute a dendrogram from feature importances and visualize it with category-colored outer rings and SHAP-style node labels.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from radiatreepp import radialTreee, RadialTreeConfig
from radiatreepp.utils import compute_radialtree_linkage

# Generate synthetic data
np.random.seed(42)
features = [f"F{i}" for i in range(10)]
importances = np.random.rand(10) * 10
categories = ["A", "B"] * 5

df = pd.DataFrame({
    "Feature": features,
    "XGBoost": importances,
    "Category": categories
})
df["XGBoost_log"] = np.log1p(df["XGBoost"])
df["Feature_with_importance"] = df.apply(
    lambda row: f"{row['Feature']} ({row['XGBoost']:.2f})", axis=1
)

# Create color mapping
unique_cats = df["Category"].unique()
cat_colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(unique_cats)))
cat_map = {cat: cat_colors[i] for i, cat in enumerate(unique_cats)}
color_array = np.array([cat_map[c] for c in df["Category"]])
colors_dict = {"Category": color_array}

# Generate dendrogram and plot
Z = compute_radialtree_linkage(df, "XGBoost_log", "Feature_with_importance")
fig, ax = plt.subplots(figsize=(8, 8))
config = RadialTreeConfig(
    colorlabels=colors_dict,
    gradient_colors=["black", "blue"],
    radial_labels=True,
    node_display_mode='inner',
    node_label_display_mode='top_3',
)
radialTreee(Z, ax=ax, config=config)
ax.set_title("Test RadialTree", fontsize=16)
plt.show()
```

---

## 🧪 Demo Examples

To generate the plots from the included synthetic dataset:

```bash
python -m radiatreepp.examples.xgboost_demo log_feature_importance_synthetic.csv
python -m radiatreepp.examples.tabnet_demo log_feature_importance_synthetic.csv
```

Each command generates:

- `out_figs/radial_dendrogram_<model>.png`
- `out_figs/radial_dendrogram_<model>_legend.png`

---

## 🔧 Customization

You can fully control the look and behavior via the `RadialTreeConfig` class:

```python
from radiatreepp import RadialTreeConfig

config = RadialTreeConfig(
    fontsize=8,
    radial_labels=False,
    label_radius=1.2,
    node_display_mode='inner',         # all, inner, none
    node_label_display_mode='top_3',   # all, inner, top_3, none
    node_size=5,
    node_label_fontsize=7,
    gradient_colors=["black", "blue"], # colormap for edge gradient
    colorlabels={"Category": color_array},  # optional outer rings
)
```

---

## 📁 File Structure

```
radiatreepp/
├── core.py          # Main plotting logic
├── config.py        # Dataclass for RadialTreeConfig
├── utils.py         # Helper for dendrogram computation
├── __init__.py

examples/
├── xgboost_demo.py
├── tabnet_demo.py

log_feature_importance_synthetic.csv   # Safe-to-publish example CSV
```

---

## 📘 Input CSV Format

Your input file should have at least:

- `Feature`: Feature names
- `XGBoost`, `TabNet`, ...: Importance values
- `Category`: For outer ring color labels

Example:

```csv
Feature,Category,XGBoost,TabNet
Feature_0,Category_1,0.134,0.112
Feature_1,Category_2,0.984,0.803
...
```

---

## 🧠 Applications

- Interpreting SHAP values from models like XGBoost, TabNet, LightGBM
- Explaining hierarchical clusters with group semantics
- Publishing visualizations for papers, dashboards, and reports

---

## ✍️ Author

**Elham Soltani Kazemi**  
University of Missouri  
[GitHub Profile](https://github.com/YOUR_USERNAME)

---

## 📜 License

MIT License — free for personal, academic, and commercial use.

---

## ❤️ Acknowledgments

Inspired by SHAP visualization, SciPy’s dendrograms, and feature importance research in explainable AI.
