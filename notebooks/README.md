# Notebooks

All experiments ran on Kaggle (2× Tesla T4, "Save & Run All"). Every notebook needs the dataset `abdallahalidev/plantvillage-dataset` and a dataset containing `splits.csv` attached as inputs.

| Notebook | Phase | Extra inputs | Main outputs |
|---|---|---|---|
| `hpo_mobilenet.ipynb` | HPO, MobileNetV3-Small | – | `hpo_mobilenetv3_small_100.db`, `_trials.csv`, `_top3.json` |
| `hpo_mobilevit.ipynb` | HPO, MobileViT-XXS | – | `hpo_mobilevit_xxs.db`, `_trials.csv`, `_top3.json` |
| `phase2_mobilenet.ipynb` | Top-3 check + final training | output of `hpo_mobilenet` | `*_top3_rank*.json`, `*_final_seed*.json/.pt/_test_preds.npz` |
| `phase2_mobilevit.ipynb` | Top-3 check + final training | output of `hpo_mobilevit` | same |
| `rerun_mobilenet_seed1.ipynb` | MobileNet seed 1, full 20 epochs | – | `mobilenetv3_small_100_final_seed1_full.*` |

Each notebook's first cell writes the training script to `/kaggle/working/` with `%%writefile`; the following cells launch one worker process per GPU. Replace these files with the executed Kaggle versions (with outputs) to keep the logs alongside the code.
