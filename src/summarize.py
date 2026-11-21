"""Builds results/summary.csv and prints the model comparison (mean +- std over seeds, McNemar test).
python src/summarize.py"""
import json, os
import numpy as np, pandas as pd
from scipy.stats import binomtest, ttest_ind

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F = os.path.join(ROOT, "results", "final")
RUNS = {"mobilenetv3_small_100": ["0", "1_full", "2"],  # seed 1 = full-schedule rerun (see README)
        "mobilevit_xxs": ["0", "1", "2"]}

rows, correct = [], {}
for model, seeds in RUNS.items():
    for s in seeds:
        r = json.load(open(f"{F}/{model}_final_seed{s}.json"))
        p = np.load(f"{F}/{model}_final_seed{s}_test_preds.npz", allow_pickle=True)
        correct[(model, s[0])] = p["preds"] == p["labels"]
        rows.append({"model": model, "seed": int(s[0]), "params_M": r["params_M"], "epochs_run": r["epochs_run"],
                     "best_epoch": r["best_epoch"], "train_min": r["train_time_s"] / 60,
                     "val_acc": r["val"]["acc"], "val_macro_f1": r["val"]["macro_f1"],
                     "test_acc": r["test"]["acc"], "test_macro_f1": r["test"]["macro_f1"],
                     "test_errors": int((~correct[(model, s[0])]).sum())})
df = pd.DataFrame(rows)
df.to_csv(os.path.join(ROOT, "results", "summary.csv"), index=False)
print(df.round(4).to_string(index=False), "\n")
agg = df.groupby("model")[["test_acc", "test_macro_f1", "test_errors"]].agg(["mean", "std"])
print(agg.round(4), "\n")

b = c = 0
for s in "012":
    a, v = correct[("mobilenetv3_small_100", s)], correct[("mobilevit_xxs", s)]
    b += int((a & ~v).sum()); c += int((~a & v).sum())
print(f"McNemar (pooled over seed pairs): only MobileNet correct={b}, only MobileViT correct={c}, "
      f"exact p={binomtest(b, b + c, 0.5).pvalue:.3f}  (same images reused across seeds - indicative only)")
t = ttest_ind(df[df.model == "mobilenetv3_small_100"].test_acc, df[df.model == "mobilevit_xxs"].test_acc, equal_var=False)
print(f"Welch t-test on per-seed test accuracy: p={t.pvalue:.3f}")
