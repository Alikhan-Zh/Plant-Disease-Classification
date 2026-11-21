"""Figures 5-6: final training curves and test confusions. python src/plot_final.py"""
import json, os, matplotlib
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
F = os.path.join(ROOT, "results", "final"); OUT = os.path.join(ROOT, "figures") + "/"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#a3a29c", "#e6e5e0", "#fcfcfb"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.edgecolor":MUTED,"axes.labelcolor":INK2,
 "xtick.color":INK2,"ytick.color":INK2,"axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,
 "grid.color":GRID,"axes.axisbelow":True,"figure.facecolor":SURF,"axes.facecolor":SURF,"savefig.facecolor":SURF,
 "legend.frameon":False,"axes.titlesize":11,"axes.titleweight":"bold","axes.titlecolor":INK})
M={"MobileNetV3-Small":dict(c="#2a78d6",note="label smoothing 0.1",js=[f"{F}/mobilenetv3_small_100_final_seed{s}.json" for s in ("0", "1_full", "2")],
      zs=[f"{F}/mobilenetv3_small_100_final_seed{s}_test_preds.npz" for s in ("0", "1_full", "2")]),
   "MobileViT-XXS":dict(c="#eb6834",note="no label smoothing",js=[f"{F}/mobilevit_xxs_final_seed{s}.json" for s in range(3)],
      zs=[f"{F}/mobilevit_xxs_final_seed{s}_test_preds.npz" for s in range(3)])}
fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
for row,(name,m) in zip(axes, M.items()):
    axl, axf = row
    for j in m["js"]:
        r=json.load(open(j)); h=r["history"]; e=[x["epoch"] for x in h]; b=r["best_epoch"]
        axl.plot(e,[x["train_loss"] for x in h],color=m["c"],lw=1.4,ls="--",alpha=0.75)
        axl.plot(e,[x["val_loss"] for x in h],color=m["c"],lw=1.8)
        axl.scatter(b,h[b-1]["val_loss"],s=40,color=m["c"],edgecolor=SURF,linewidth=1.5,zorder=3)
        axf.plot(e,[x["val_macro_f1"] for x in h],color=m["c"],lw=1.8)
        axf.scatter(b,h[b-1]["val_macro_f1"],s=40,color=m["c"],edgecolor=SURF,linewidth=1.5,zorder=3)
    axl.set_yscale("log"); axl.set_ylabel("Cross-entropy loss (log scale)")
    axl.set_title(f"{name}: loss ({m['note']})",loc="left")
    axf.set_ylim(0.90,1.0); axf.set_ylabel("Validation macro-F1"); axf.set_title(f"{name}: validation macro-F1",loc="left")
    axl.legend(handles=[Line2D([],[],color=m["c"],ls="--",lw=1.4,label="Train loss"),
                        Line2D([],[],color=m["c"],lw=1.8,label="Validation loss"),
                        Line2D([],[],ls="",marker="o",ms=7,color=m["c"],label="Selected checkpoint")],loc="upper right",fontsize=9)
for ax in axes[1]: ax.set_xlabel("Epoch"); ax.set_xticks(range(1,21,1))
fig.suptitle("Final training, 3 seeds per model", x=0.02, ha="left", fontweight="bold", color=INK)
fig.text(0.02, -0.02, "Train loss includes augmentation, dropout and (MobileNet) label smoothing; validation loss is plain cross-entropy, "
         "so train loss can sit above validation loss.\nLosses are not comparable between the two models: label smoothing keeps "
         "MobileNet's loss near 0.1 by design. Compare models on macro-F1. MobileViT's epoch-1 F1 (~0.7) is below the plotted range.",
         fontsize=8.5, color=INK2)
fig.tight_layout(); fig.savefig(OUT+"5_final_training_curves.png",dpi=200,bbox_inches="tight"); plt.close(fig)

# 6. Confusion sub-matrix for the confusable corn + tomato classes, summed over seeds
pick=["Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot","Corn_(maize)___Northern_Leaf_Blight","Corn_(maize)___Common_rust_",
      "Tomato___Early_blight","Tomato___Late_blight","Tomato___Target_Spot","Tomato___healthy"]
short=["Corn: Cercospora","Corn: N. Leaf Blight","Corn: Common rust","Tomato: Early blight","Tomato: Late blight","Tomato: Target spot","Tomato: healthy"]
fig, axes = plt.subplots(1,2,figsize=(12,5.2))
for ax,(name,m) in zip(axes,M.items()):
    C=None
    for z in m["zs"]:
        d=np.load(z,allow_pickle=True); cl=list(d["classes"]); C=d["confusion"] if C is None else C+d["confusion"]
    idx=[cl.index(p) for p in pick]; S=C[np.ix_(idx,idx)].astype(float)
    off=S.copy(); np.fill_diagonal(off,np.nan)
    ax.imshow(np.where(np.isnan(off),0,off),cmap="Oranges" if "ViT" in name else "Blues",vmin=0,vmax=10)
    for i in range(len(idx)):
        for j in range(len(idx)):
            v=int(S[i,j]); ax.text(j,i,str(v),ha="center",va="center",fontsize=8.5,
                color=MUTED if i==j else (INK if v<6 else SURF),fontweight="normal" if i==j else "bold")
    ax.set_xticks(range(len(idx))); ax.set_xticklabels(short,rotation=40,ha="right",fontsize=8.5)
    ax.set_yticks(range(len(idx))); ax.set_yticklabels(short,fontsize=8.5); ax.grid(False)
    ax.set_title(f"{name} (summed over 3 seeds)",loc="left"); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
fig.suptitle("Where the test errors are: confusable corn and tomato classes (diagonal = correct, grey)",
             x=0.02,ha="left",fontweight="bold",color=INK)
fig.tight_layout(); fig.savefig(OUT+"6_confusion_confusable_classes.png",dpi=200,bbox_inches="tight"); plt.close(fig)
