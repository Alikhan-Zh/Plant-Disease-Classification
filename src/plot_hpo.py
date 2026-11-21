"""Figures 1-4: HPO results. Run from anywhere: python src/plot_hpo.py"""
import ast, os, warnings
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, optuna
warnings.filterwarnings("ignore"); optuna.logging.set_verbosity(optuna.logging.ERROR)

A = os.path.join(ROOT, "results", "hpo") + "/"
OUT = os.path.join(ROOT, "figures") + "/"
M = {
    "MobileNetV3-Small": dict(csv="hpo_mobilenetv3_small_100_trials.csv",
                              db="hpo_mobilenetv3_small_100.db", study="mobilenetv3_small_100", c="#2a78d6"),
    "MobileViT-XXS":     dict(csv="hpo_mobilevit_xxs_trials.csv",
                              db="hpo_mobilevit_xxs.db", study="mobilevit_xxs", c="#eb6834"),
}
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#a3a29c", "#e6e5e0", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "figure.facecolor": SURF, "axes.facecolor": SURF,
    "savefig.facecolor": SURF, "legend.frameon": False, "axes.titlesize": 11, "axes.titleweight": "bold"})

D = {}
for name, m in M.items():
    df = pd.read_csv(A + m["csv"])
    df["curve"] = df["user_attrs_history"].apply(ast.literal_eval)
    D[name] = df

def save(fig, fn):
    fig.savefig(OUT + fn, dpi=200, bbox_inches="tight"); plt.close(fig)

# 1. Convergence: best-so-far vs trial
fig, ax = plt.subplots(figsize=(7, 4))
for name, df in D.items():
    c = M[name]["c"]
    v = df["value"].where(df["state"] == "COMPLETE")
    best = v.fillna(-1).cummax()
    ax.step(df["number"], best, where="post", color=c, lw=2, label=name)
    done = df[df.state == "COMPLETE"]
    ax.scatter(done["number"], done["value"], s=36, color=c, edgecolor=SURF, linewidth=1.5, zorder=3)
    ax.annotate(f"{best.iloc[-1]:.3f}", (df["number"].iloc[-1], best.iloc[-1]), xytext=(6, 0),
                textcoords="offset points", va="center", color=INK, fontsize=9)
    ax.annotate(f"paper config {df.value[0]:.3f}", (0, df.value[0]), xytext=(8, 6 if "ViT" in name else -14),
                textcoords="offset points", color=INK2, fontsize=8.5)
ax.set_ylim(0.80, 1.0); ax.set_xlim(-0.5, 21.5); ax.set_xticks(range(0, 21, 2))
ax.set_xlabel("Trial number"); ax.set_ylabel("Validation macro-F1")
ax.set_title("HPO convergence: best macro-F1 so far", loc="left", pad=22)
ax.text(0, 1.02, "Dots = completed trials (pruned and trials below 0.80 not shown). Trial 0 = paper configuration.",
        transform=ax.transAxes, fontsize=8.5, color=INK2)
ax.legend(loc="lower right")
save(fig, "1_hpo_convergence.png")

# 2. Learning curves: small multiples
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, (name, df) in zip(axes, D.items()):
    c = M[name]["c"]; best_n = df.loc[df.value[df.state == "COMPLETE"].idxmax(), "number"]
    for _, r in df.iterrows():
        e = [h["epoch"] for h in r.curve]; f = [h["val_macro_f1"] for h in r.curve]
        if r.number == 0:
            ax.plot(e, f, color=INK, lw=2, ls="--", zorder=4, label="Paper config (trial 0)")
        elif r.number == best_n:
            ax.plot(e, f, color=c, lw=2.5, marker="o", ms=5, zorder=5, label=f"Best (trial {best_n})")
        elif r.state == "PRUNED":
            ax.plot(e, f, color=MUTED, lw=1.2, marker="x", ms=5, alpha=0.8, zorder=2)
        else:
            ax.plot(e, f, color=c, lw=1.2, alpha=0.45, zorder=3)
    ax.plot([], [], color=c, lw=1.2, alpha=0.45, label="Other completed")
    ax.plot([], [], color=MUTED, lw=1.2, marker="x", label="Pruned")
    ax.set_title(name, loc="left"); ax.set_xticks(range(1, 6)); ax.set_xlabel("Epoch"); ax.set_ylim(0, 1.0)
    ax.legend(loc="lower right", fontsize=8.5)
axes[0].set_ylabel("Validation macro-F1")
fig.suptitle("Per-epoch validation macro-F1 of every HPO trial", x=0.06, ha="left", fontweight="bold", color=INK)
save(fig, "2_hpo_learning_curves.png")

# 3. Parameter importance (fANOVA) - grouped horizontal bars
imp = {}
for name, m in M.items():
    s = optuna.load_study(study_name=m["study"], storage=f"sqlite:///{A}{m['db']}")
    imp[name] = optuna.importance.get_param_importances(s, evaluator=optuna.importance.FanovaImportanceEvaluator(seed=0))
labels = {"lr": "Learning rate", "weight_decay": "Weight decay", "label_smoothing": "Label smoothing",
          "schedule": "LR schedule", "dropout": "Head dropout", "batch_size": "Batch size"}
order = sorted(labels, key=lambda k: -max(imp[n].get(k, 0) for n in imp))
fig, ax = plt.subplots(figsize=(7, 3.8)); y = np.arange(len(order)); h = 0.36
for i, (name, vals) in enumerate(imp.items()):
    v = [vals.get(k, 0) for k in order]
    ax.barh(y + (i - 0.5) * (h + 0.02), v, height=h, color=M[name]["c"], label=name)
    for yy, vv in zip(y + (i - 0.5) * (h + 0.02), v):
        ax.text(vv + 0.008, yy, f"{vv:.2f}", va="center", fontsize=8.5, color=INK2)
ax.set_yticks(y); ax.set_yticklabels([labels[k] for k in order]); ax.invert_yaxis()
ax.set_xlabel("Importance (fANOVA, share of variance in macro-F1)"); ax.grid(axis="y", visible=False)
ax.set_xlim(0, max(max(v.values()) for v in imp.values()) * 1.18)
ax.set_title("Which hyperparameters mattered", loc="left"); ax.legend(loc="lower right")
save(fig, "3_hpo_param_importance.png")

# 4. LR vs macro-F1, small multiples, schedule as filled/hollow
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, (name, df) in zip(axes, D.items()):
    c = M[name]["c"]
    for sched, fill in [("cosine", True), ("constant", False)]:
        for state in ["COMPLETE", "PRUNED"]:
            d = df[(df.params_schedule == sched) & (df.state == state)]
            col = c if state == "COMPLETE" else MUTED
            ax.scatter(d.params_lr, d.value, s=60, marker="o",
                       facecolor=col if fill else SURF, edgecolor=col, linewidth=1.8, zorder=3)
    p = df.iloc[0]; ax.scatter(p.params_lr, p.value, s=140, marker="*", color=INK, zorder=4)
    ax.annotate("paper", (p.params_lr, p.value), xytext=(7, -3), textcoords="offset points", fontsize=8.5, color=INK2)
    ax.set_xscale("log"); ax.set_xlim(8e-6, 1.2e-2); ax.set_title(name, loc="left"); ax.set_xlabel("Learning rate (log scale)")
axes[0].set_ylabel("Validation macro-F1 (best epoch;\npruned = score when stopped)")
from matplotlib.lines import Line2D
h = [Line2D([], [], ls="", marker="o", ms=8, mfc=INK2, mec=INK2, label="Cosine schedule"),
     Line2D([], [], ls="", marker="o", ms=8, mfc=SURF, mec=INK2, mew=1.8, label="Constant LR"),
     Line2D([], [], ls="", marker="o", ms=8, mfc=MUTED, mec=MUTED, label="Pruned (grey)"),
     Line2D([], [], ls="", marker="*", ms=12, color=INK, label="Paper configuration")]
fig.legend(handles=h, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.13), fontsize=9)
fig.suptitle("Learning rate vs. validation macro-F1", x=0.06, ha="left", fontweight="bold", color=INK)
save(fig, "4_hpo_lr_vs_score.png")
print('HPO importance:', {n: {k: round(v, 3) for k, v in d.items()} for n, d in imp.items()})
