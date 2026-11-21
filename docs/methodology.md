# Methodology

This document records how every result in the repository was produced, which decisions were made and why, and where we deviate from the reference paper (Bahaa et al., 2026).

## 1. Data

- **Dataset:** PlantVillage, Kaggle version `abdallahalidev/plantvillage-dataset`, `color` subset only (the dataset also contains grayscale and segmented versions). 54,305 images, 38 classes (14 crops, healthy and diseased). Images are 256×256 lab photographs of single leaves on a plain background.
- **Not used:** the "New Plant Diseases Dataset" (~87k images). It contains offline-augmented copies of the same leaves, so a random split leaks near-duplicates into the test set.
- **Split:** stratified 70 / 20 / 10 (train / validation / test), seed 42, created once by `src/prepare_splits.py` and shared by every experiment: 38,013 / 10,861 / 5,431 images. The ratios follow the paper.
- **HPO subset:** a stratified 30 % of the training split (11,403 images), flagged in `splits.csv`. The smallest class has 32 images in this subset.
- **Class imbalance:** large (from ~150 to ~5,500 images per class). We therefore report macro-F1 alongside accuracy and use validation macro-F1 as the HPO objective. Some test classes are very small (Potato healthy has 15 test images), so a single error can move that class's F1 by ~0.03.
- **Why no k-fold cross-validation:** with ~10.9k validation images a single hold-out estimate is stable, and 5-fold CV would multiply the GPU cost five times. Training variability, the dominant source of randomness, is captured by repeating final training with three seeds.

## 2. Models

| Model | timm name | Pretrained weights | Params (38 classes) | Input normalisation |
|---|---|---|---|---|
| MobileNetV3-Small | `mobilenetv3_small_100` | ImageNet | 1.56 M | ImageNet mean/std |
| MobileViT-XXS | `mobilevit_xxs` | ImageNet (`cvnets_in1k`, same checkpoint as the paper) | 0.96 M | scaled to [0, 1], no mean/std |

**Why this pair:** MobileViT is built on MobileNetV2-style inverted residual blocks with added self-attention blocks, so at similar size the comparison isolates the effect of the transformer component. Both are the standard reference models of their families, have pretrained weights, and export to mobile runtimes (needed for the quantization step). Mamba-style models were rejected because of immature export and quantization support.

All layers are fine-tuned. timm's `drop_rate` adds dropout before the new 38-class head.

## 3. Training setup shared by all runs

From the paper:
- 224×224 input (resize), ImageNet-pretrained weights
- Training augmentation: random horizontal and vertical flips, rotation ±10°, brightness factor 0.5–1.5. Validation/test: resize and normalise only.
- Cross-entropy loss, up to 20 epochs, checkpoint selection on validation loss.

Our changes (applied identically to both models):
- **One framework.** The paper used Keras for MobileNet and PyTorch/Hugging Face for MobileViT; we use PyTorch + timm for both, so framework differences cannot explain model differences.
- **AdamW** instead of Adam. Adam is AdamW with zero weight decay, so the paper's setting is inside our search space (weight decay 1e-6 approximates it).
- **Softmax cross-entropy.** The paper mentions a sigmoid classifier; softmax is the correct choice for single-label multiclass classification.
- **Per-model normalisation.** The paper states ImageNet normalisation for both models, but MobileViT's pretrained weights expect [0, 1] inputs; each model uses its own pretraining statistics.
- **Mixed precision (AMP)** for speed.
- **LR schedule** is a tuned hyperparameter: constant, or cosine decay with one warmup epoch.

## 4. Phase 1: hyperparameter optimisation

- **Algorithm:** Optuna Tree-structured Parzen Estimator (Bayesian optimisation), multivariate, 5 random start-up trials, then model-guided sampling.
- **Pruning:** median pruner. From epoch 2 onwards, a trial is stopped if its best validation macro-F1 so far is below the median of completed trials at the same epoch; no pruning until 5 trials have completed.
- **Fidelity:** 5 epochs on the 30 % HPO subset, scored on the full validation set (multi-fidelity HPO).
- **Objective:** best validation macro-F1 over the 5 epochs.
- **Budget:** 20 trials per model, the same search space for both. Trial 0 is the paper's configuration for that model.
- **Hardware:** two worker processes, one per T4 GPU, sharing one SQLite study.

**Search space**

| Hyperparameter | Range | Paper (MobileNet / MobileViT) |
|---|---|---|
| Learning rate | log-uniform 1e-5 – 1e-2 | 1e-3 / 1e-4 |
| Weight decay | log-uniform 1e-6 – 1e-1 | none |
| Batch size | {32, 64} | 64 / 32 |
| Head dropout | {0, 0.2, 0.5} | not stated |
| Label smoothing | {0, 0.1} | 0 |
| LR schedule | {constant, cosine + 1 warmup epoch} | constant |

**Results**

| | Paper config (trial 0) | Best trial | Best configuration | Complete / pruned |
|---|---|---|---|---|
| MobileNetV3-Small | 0.852 | 0.987 (trial 8) | lr 5.8e-4, wd 0.063, bs 64, dropout 0.5, LS 0.1, cosine | 10 / 11 |
| MobileViT-XXS | 0.866 | 0.972 (trial 17) | lr 3.9e-3, wd 5e-5, bs 32, dropout 0.5, LS 0.1, cosine | 11 / 10 |

Observations:
- The paper's configurations fail in opposite ways: MobileNet at lr 1e-3 (constant) oscillates between epochs; MobileViT at lr 1e-4 is stable but still far from converged after 5 epochs.
- Learning rate explains most of the variance in macro-F1 for both models (fANOVA importance 0.85 and 0.94). With ~10 completed trials per model the exact importance values are noisy; the dominance of the learning rate is robust.
- MobileViT's top-5 trials all use cosine decay; high constant learning rates diverge (trial 4: validation loss 155). MobileNet's top trials mix constant and cosine.
- Best values are not at the edges of the search space.

**Known issues**
- *Duplicate trials.* Both workers used the same sampler seed, so their random start-up trials coincided (trials 1 = 2 and 3 = 4 in both studies). Each study therefore evaluated 19 unique configurations (21 rather than 20 trials because two workers finishing simultaneously can overshoot by one). The duplicates give a free noise estimate: the same configuration scored 0.960 vs. 0.963 (MobileNet) and 0.929 vs. 0.930 (MobileViT), so differences below ~0.003 are noise.
- *Pruning bias.* Median pruning at epoch 2 saved ~40 % of HPO time but favours configurations that start fast (high learning rate, no warmup). Some trials were pruned by margins below the noise level (e.g. MobileNet trial 13: 0.932 vs. a median of 0.9335). The top-3 check (Section 5) partly compensates.
- The `lr_scheduler.step()` warning in the logs comes from AMP skipping the first optimizer step and has no effect.

## 5. Top-3 check

The three best HPO configurations of each model were retrained on the full training set for 10 epochs (seed 0, cosine schedule length 10), and the winner was chosen by validation macro-F1. The test set was not used.

| Model | HPO rank | Configuration | Full-data val macro-F1 |
|---|---|---|---|
| MobileNetV3-Small | 1 (trial 8) | lr 5.8e-4, cosine, dropout 0.5 | 0.99673 |
| | **2 (trial 18)** | lr 2.0e-4, cosine, dropout 0.2, LS 0.1, wd 3.9e-4, bs 64 | **0.99676** (winner) |
| | 3 (trial 15) | lr 1.1e-4, constant | 0.98775 |
| MobileViT-XXS | 1 (trial 17) | lr 3.9e-3, LS 0.1 | 0.99191 |
| | 2 (trial 14) | lr 1.2e-3 | 0.99567 |
| | **3 (trial 12)** | lr 5.0e-4, cosine, dropout 0.5, LS 0, wd 6.1e-4, bs 32 | **0.99584** (winner) |

- MobileNet: ranks 1 and 2 are tied within noise; rank 2 has a smoother validation curve. The only constant-LR configuration falls clearly behind.
- MobileViT: the HPO winner, which had the highest learning rate, is last on full data (its validation macro-F1 dropped to 0.60 at epoch 3). This is the short-run bias described above.

## 6. Phase 2: final training

- Winner configuration, full training split, 20 epochs, seeds 0, 1 and 2.
- The checkpoint with the lowest validation loss is evaluated once on the test set.

**Protocol change (early stopping).** Phase 2 originally used the paper's early stopping (patience 5 on validation loss). MobileNet seed 1 stopped at epoch 11 while the cosine schedule was still at about half its initial learning rate; the other runs improved most in the low-LR tail (best epochs 16–20). Patience-based stopping is inconsistent with a cosine schedule that is designed to run to the end, so we adopted one rule for all runs: train the full 20 epochs and select the checkpoint with the lowest validation loss. Every other run already satisfied this rule (none stopped early), so only MobileNet seed 1 was rerun. The decision was based on the validation curves and the schedule; both versions are kept in `results/final/`:

| MobileNet seed 1 | Epochs run (best) | Test accuracy | Test macro-F1 | Test errors |
|---|---|---|---|---|
| Early-stopped (patience 5) | 11 (6) | 99.34 % | 0.9904 | 36 |
| Full schedule (used) | 20 (16) | 99.83 % | 0.9969 | 9 |

**Final results**

| Model | Seed | Best epoch | Val acc | Test acc | Test macro-F1 | Test errors |
|---|---|---|---|---|---|---|
| MobileNetV3-Small | 0 | 20 | 99.84 % | 99.80 % | 0.9963 | 11 |
| | 1 | 16 | 99.80 % | 99.83 % | 0.9969 | 9 |
| | 2 | 19 | 99.89 % | 99.78 % | 0.9968 | 12 |
| MobileViT-XXS | 0 | 17 | 99.83 % | 99.65 % | 0.9954 | 19 |
| | 1 | 19 | 99.81 % | 99.71 % | 0.9955 | 16 |
| | 2 | 19 | 99.84 % | 99.76 % | 0.9966 | 13 |

Mean ± std: MobileNet 99.80 ± 0.03 % accuracy, 0.9967 ± 0.0003 macro-F1; MobileViT 99.71 ± 0.06 %, 0.9958 ± 0.0006.

**Comparison.** On the same test images, MobileNet was correct on 28 images where MobileViT was wrong, and MobileViT on 12 where MobileNet was wrong (pooled over the three seed pairs; exact McNemar p = 0.017, indicative only because the same images are reused across seeds). Across seeds the accuracy difference is not significant (Welch t-test p = 0.07, n = 3 per model). In each seed, 6–7 test images are misclassified by both models.

**Error analysis.** The dominant confusion for both models is corn Northern Leaf Blight ↔ Cercospora (Gray) leaf spot (15 errors over three seeds for each model), followed by tomato Late blight → healthy and tomato Target spot ↔ Early blight. MobileViT additionally confuses corn Common rust → Northern Leaf Blight.

**Training curves.** MobileNet reaches high validation F1 after one epoch (0.94–0.96) but fluctuates in the middle epochs; MobileViT starts low (0.68–0.74) and climbs steadily. Both converge to ≈0.997 in the low-LR tail of the cosine schedule. Losses are not comparable between models because MobileNet's winning configuration uses label smoothing. MobileNet's training loss stays above its validation loss because it is measured with augmentation, dropout and label smoothing. MobileViT shows very mild overfitting in the last 3–4 epochs (training loss keeps falling while validation loss plateaus), which checkpoint selection handles.

**Compute.** HPO took ~62 min (MobileNet) and ~75 min (MobileViT) wall time on 2× T4. Final runs took 44–54 min (MobileNet) and 46–75 min (MobileViT) each, two runs in parallel. MobileViT has fewer parameters but trains slower per epoch (~138 s vs. ~105–115 s when running alone on a GPU). The first epoch of each Kaggle session is slow (up to ~10 min) while images are read from disk for the first time.

## 7. Limitations

- PlantVillage consists of lab images with uniform backgrounds; near-ceiling accuracy does not imply field performance. A cross-domain test on PlantDoc is planned.
- Three seeds give limited statistical power for differences of ~0.1 percentage points.
- The HPO budget (20 trials) is small, and median pruning and 5-epoch runs bias the search toward fast-starting configurations.
- Some paper details are not stated (layer freezing, dropout, LR schedule) and are assumptions in our reconstruction; the paper's code is not public.
