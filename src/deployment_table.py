"""Builds results/deployment_summary.csv and figure 9 (Raspberry Pi 5 latency, size, accuracy). python src/deployment_table.py"""
import os, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); R = os.path.join(ROOT, "results")
q = pd.read_csv(f"{R}/quantization/qat_results.csv"); p = pd.read_csv(f"{R}/quantization/ptq_results.csv")
zs = pd.read_csv(f"{R}/plantdoc/plantdoc_crossdomain.csv"); pi = pd.read_csv(f"{R}/pi/pi_results.csv")
VAR = [  # label, accuracy source (df, weights, variant), plantdoc precision label, pi file suffix
    ("FP32", (q, "ptq", "fp32"), "fp32 (PyTorch)", "fp32"),
    ("FP16", (p, None, "fp16"), "fp16", "fp16"),
    ("INT8 full (PTQ)", (q, "ptq", "int8_full"), "ptq_int8_full", "ptq_int8_full"),
    ("INT8 mixed (PTQ)", (q, "ptq", "int8_mixed"), "ptq_int8_mixed", "ptq_int8_mixed"),
    ("INT8 mixed (QAT)", (q, "qat", "int8_mixed"), "qat_int8_mixed", "qat_int8_mixed")]
rows = []
for m, name in [("mobilenetv3_small_100", "MobileNetV3-Small"), ("mobilevit_xxs", "MobileViT-XXS")]:
    for label, (df, w, v), pdl, pif in VAR:
        a = df[(df.model == m) & (df.variant == v) & ((df.weights == w) if w else True)]
        z = zs[(zs.model == m) & (zs.precision == pdl) & (zs.subset == "all PlantDoc")]
        r = pi[pi.file == f"{m}_seed0_{pif}.onnx"]
        if r.empty:
            continue
        rows.append({"model": name, "variant": label, "fp32_convs_kept": a["fp32_convs_kept"].iloc[0] if "fp32_convs_kept" in a and v == "int8_mixed" else None,
                     "size_MB": a.size_MB.mean(), "plantvillage_acc_mean": a.test_acc.mean(), "plantvillage_acc_std": a.test_acc.std(),
                     "plantdoc_all_acc_mean": z.acc.mean(), "pi_ms_1thread": r.median_ms_1t.iloc[0], "pi_ms_4threads": r.median_ms_4t.iloc[0],
                     "pi_p95_ms_4threads": r.p95_ms_4t.iloc[0], "pi_peak_rss_MB_incl_data": r.peak_rss_MB.iloc[0], "pi_subset_acc": r.subset_acc_pi.iloc[0]})
d = pd.DataFrame(rows); d.to_csv(f"{R}/deployment_summary.csv", index=False)
print(d.round(4).to_string(index=False))

INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#a3a29c", "#e6e5e0", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "legend.frameon": False, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlecolor": INK})
C = {"MobileNetV3-Small": "#2a78d6", "MobileViT-XXS": "#eb6834"}
d = d.iloc[::-1].reset_index(drop=True); y = np.arange(len(d)); labels = [f"{m}  ·  {v}" for m, v in zip(d.model, d.variant)]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True, gridspec_kw={"width_ratios": [1.6, 1]})
cols = [C[m] for m in d.model]
a1.barh(y, d.pi_ms_4threads, color=cols, height=0.6, label="4 threads (bar)")
a1.scatter(d.pi_ms_1thread, y, marker="|", s=260, color=INK, linewidth=2, zorder=3, label="1 thread (tick)")
for yi, (t4, acc) in enumerate(zip(d.pi_ms_4threads, d.plantvillage_acc_mean)):
    a1.text(max(t4, d.pi_ms_1thread[yi]) + 1.5, yi, f"{t4:.1f} ms   |   PlantVillage acc {100 * acc:.1f}%", va="center", fontsize=8.5,
            color=INK if acc > 0.9 else "#c0392b")
a1.axvline(33.3, color=MUTED, ls="--", lw=1); a1.text(33.8, len(d) - 0.45, "30 fps budget", fontsize=8, color=INK2)
a1.set_yticks(y); a1.set_yticklabels(labels, fontsize=9); a1.set_xlabel("Median latency per image on Raspberry Pi 5 (ms)")
a1.set_xlim(0, 135); a1.grid(axis="y", visible=False); a1.legend(loc="lower right", fontsize=8.5)
a1.set_title("Raspberry Pi 5 latency (ONNX Runtime, CPU)", loc="left")
a2.barh(y, d.size_MB, color=cols, height=0.6)
for yi, s in enumerate(d.size_MB):
    a2.text(s + 0.08, yi, f"{s:.2f} MB", va="center", fontsize=8.5, color=INK)
a2.set_xlabel("Model file size (MB)"); a2.set_xlim(0, 7.2); a2.grid(axis="y", visible=False); a2.set_title("Model size", loc="left")
fig.text(0.01, -0.03, "Accuracy in red = broken (full INT8). INT8 mixed keeps the most quantization-sensitive convolutions in FP32 "
         "(MobileNet: 32 of 53 for PTQ, 12 for QAT; MobileViT: 8 of 35).", fontsize=8.5, color=INK2)
fig.tight_layout(); fig.savefig(os.path.join(ROOT, "figures", "9_deployment_pi5.png"), dpi=200, bbox_inches="tight")
