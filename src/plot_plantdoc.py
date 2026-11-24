"""Figures 7-8: PlantVillage vs PlantDoc (cross-domain and fine-tuned), per-class recall on PlantDoc. python src/plot_plantdoc.py"""
import os, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); R = os.path.join(ROOT, "results"); OUT = os.path.join(ROOT, "figures")
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#a3a29c", "#e6e5e0", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "legend.frameon": False, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlecolor": INK})
C = {"mobilenetv3_small_100": "#2a78d6", "mobilevit_xxs": "#eb6834"}; NAME = {"mobilenetv3_small_100": "MobileNetV3-Small", "mobilevit_xxs": "MobileViT-XXS"}

summ = pd.read_csv(os.path.join(R, "summary.csv")); zs = pd.read_csv(os.path.join(R, "plantdoc", "plantdoc_crossdomain.csv"))
ft = pd.read_csv(os.path.join(R, "plantdoc", "plantdoc_finetune.csv"))
groups = ["PlantVillage test\n(trained on PlantVillage)", "PlantDoc test\nno PlantDoc training", "PlantDoc test\nfine-tuned from ImageNet",
          "PlantDoc test\nfine-tuned from PlantVillage"]
def vals(m):
    return [summ[summ.model == m].test_acc.values,
            zs[(zs.model == m) & (zs.subset == "PlantDoc test split") & zs.precision.str.startswith("fp32")].acc.values,
            ft[(ft.model == m) & (ft.init == "imagenet")].test_acc.values, ft[(ft.model == m) & (ft.init == "plantvillage")].test_acc.values]
fig, ax = plt.subplots(figsize=(9.5, 4.6)); x = np.arange(len(groups)); w = 0.36
for j, m in enumerate(C):
    for i, v in enumerate(vals(m)):
        xi = x[i] + (j - 0.5) * (w + 0.03)
        ax.bar(xi, 100 * v.mean(), w, color=C[m], label=NAME[m] if i == 0 else None)
        ax.scatter(np.full(len(v), xi), 100 * v, s=14, color=INK, zorder=3, alpha=0.7)
        ax.text(xi, 100 * v.max() + 2.2, f"{100 * v.mean():.1f}", ha="center", fontsize=9, color=INK)
ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=9); ax.set_ylim(0, 108); ax.set_ylabel("Test accuracy (%)"); ax.grid(axis="x", visible=False)
ax.set_title("Lab accuracy does not transfer to field images (bars = mean of 3 seeds, dots = seeds)", loc="left"); ax.legend(loc="upper right")
fig.savefig(os.path.join(OUT, "7_plantvillage_vs_plantdoc.png"), dpi=200, bbox_inches="tight"); plt.close(fig)

pc = pd.read_csv(os.path.join(R, "plantdoc", "plantdoc_crossdomain_per_class_recall.csv"), index_col=0)
mean = pd.DataFrame({m: pc[[c for c in pc.columns if c.startswith(m)]].mean(axis=1) for m in C})
mean = mean.drop(index="Tomato two spotted spider mites leaf").sort_values("mobilevit_xxs")  # only 2 images
fig, ax = plt.subplots(figsize=(7.5, 8)); y = np.arange(len(mean))
ax.hlines(y, mean.min(axis=1) * 100, mean.max(axis=1) * 100, color=GRID, lw=3, zorder=1)
for m in C:
    ax.scatter(mean[m] * 100, y, s=46, color=C[m], edgecolor=SURF, linewidth=1.5, zorder=3, label=NAME[m])
ax.set_yticks(y); ax.set_yticklabels(mean.index, fontsize=8.5); ax.set_xlabel("Recall on PlantDoc (%), mean of 3 seeds, no PlantDoc training")
ax.set_xlim(-2, 100); ax.grid(axis="y", visible=False); ax.legend(loc="lower right")
ax.set_title("Which field classes the PlantVillage models still recognise", loc="left")
fig.savefig(os.path.join(OUT, "8_plantdoc_per_class_recall.png"), dpi=200, bbox_inches="tight"); plt.close(fig)
print("saved figures 7-8")
