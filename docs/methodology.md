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

## 7. Compression (FP16, INT8)

**Setup.** Each final checkpoint was exported to ONNX (opset 17; logits match PyTorch within ~1e-5). Variants:
- **FP16**: ONNX Runtime's float16 converter (the `onnxconverter-common` converter produced an invalid MobileViT graph because of Cast nodes in the attention blocks).
- **INT8 static**: QDQ format, per-channel int8 weights, per-tensor uint8 activations, MinMax calibration on 266–532 training images.
- **INT8 dynamic**: weights int8, activations quantized at run time.
All accuracies are on the full PlantVillage test set, three seeds.

**FP16 is lossless:** MobileNet 99.80 % (3.00 MB, was 5.96 MB), MobileViT 99.71 % (2.11 MB, was 3.88 MB), 99.98–99.99 % identical predictions.

**INT8 failed for both models** (5–18 % accuracy) in every configuration: static and dynamic, U8S8/U8U8/S8S8, `reduce_range`, depthwise convolutions excluded, ONNX Runtime 1.29 and 1.32, x86 with and without VNNI, and ARM (Raspberry Pi). Diagnosis with the real models:
- The graph pre-processing step is not the cause (pre-processed FP32 model unchanged).
- Quantizing only MatMul/Gemm (attention, classifier) costs nothing; quantizing only convolutions collapses the model.
- Quantizing one convolution at a time: MobileNet's stem convolution alone drops accuracy to 22 %, its first depthwise convolution to 51 %; two early MobileViT depthwise convolutions alone drop it to 62 % and 24 %.
- Activation statistics: inputs to several convolutions contain a few channels with 25–100× the range of the typical channel (e.g. first MobileNet depthwise input: max 35.5 vs. median channel max 0.43), and BatchNorm folding produces depthwise weights up to |355|. Per-tensor 8-bit activations round the small channels to zero and later layers amplify the error. This is the failure mode described by Nagel et al. (2019) for MobileNetV2.

**Mixed precision.** Convolutions were ranked by the accuracy cost of quantizing each one alone (on a 1,000-image validation subset), and the k most sensitive were kept in FP32, with k chosen as the smallest value within 1 point of FP32 validation accuracy (k ∈ {0, 2, 4, 8, 12, 16, 24, 32}). Only Conv, Gemm and MatMul were quantized; quantizing Add/Mul/activations as well made results worse.

**Quantization-aware training (QAT).** 3 epochs of fine-tuning (AdamW, lr 5e-5, cosine, BatchNorm statistics frozen) with fake quantization at conv inputs, BatchNorm outputs before the activation, bias-convolution outputs and per-channel weights; activation ranges were min/max values reset every 200 steps.

| Model | Variant | Size | Test accuracy (3 seeds) |
|---|---|---|---|
| MobileNetV3-Small | FP32 | 5.96 MB | 99.81 ± 0.03 % |
| | INT8 full (PTQ) | 1.79 MB | 13.1 ± 8.8 % |
| | INT8 mixed (PTQ), 32 of 53 convs FP32 | 3.38 MB | 92.5 ± 7.6 % (84.1 / 98.8 / 94.5) |
| | FP32 after QAT | 5.96 MB | 92.8 ± 1.6 % |
| | INT8 mixed (QAT), 12 convs FP32 | 1.98 MB | 94.3 ± 0.7 % |
| MobileViT-XXS | FP32 | 3.88 MB | 99.71 ± 0.06 % |
| | INT8 full (PTQ) | 1.44 MB | 9.4 ± 4.7 % |
| | **INT8 mixed (PTQ), 8 of 35 convs FP32** | **1.43 MB** | **99.53 ± 0.09 %** |
| | FP32 after QAT | 3.88 MB | 59.1 ± 2.6 % |
| | INT8 mixed (QAT), 2 convs FP32 | 1.44 MB | 66.2 ± 2.9 % |

- MobileViT tolerates INT8 well once 8 sensitive convolutions stay in FP32. MobileNet needs more than 32 FP32 convolutions and is unstable across seeds; its mixed model is barely smaller than FP16.
- Our QAT degraded FP32 accuracy (MobileNet −7 points, MobileViT −41 points). The likely cause is the simplified scheme: narrow activation ranges (reset every 200 steps) clip activations during training, the model learns to depend on that clipping, and behaves differently without it; MobileViT's unbounded SiLU activations are affected most. Proper QAT (framework observers, longer schedule, BatchNorm folding during training) is left as future work.

## 8. PlantDoc (second dataset)

- **Data:** PlantDoc cropped-leaf classification images (Singh et al., 2020) from the authors' GitHub repository: 2,578 images, 28 classes, original split 2,342 train / 236 test. Every PlantDoc class maps one-to-one onto a PlantVillage class (e.g. "Corn leaf blight" → `Corn_(maize)___Northern_Leaf_Blight`, "Tomato leaf" → `Tomato___healthy`). Images were resized to 224×224 exactly like PlantVillage. For fine-tuning, 10 % of PlantDoc train (stratified, 234 images) was held out as validation.
- **Cross-domain test:** PlantVillage checkpoints evaluated without any PlantDoc training; predictions restricted to the 28 PlantDoc classes (38-way accuracy also reported).
- **Fine-tuning:** 30 epochs on PlantDoc train with each model's winning PlantVillage configuration; best epoch chosen by validation macro-F1; one test evaluation. Two starting points: ImageNet weights, or our PlantVillage checkpoint with the 28-class head initialised from the matching PlantVillage head rows. 3 seeds each.

| Setting (PlantDoc test split, 236 images) | MobileNetV3-Small | MobileViT-XXS |
|---|---|---|
| Trained on PlantVillage only | 23.0 ± 0.5 % (macro-F1 0.19) | **29.5 ± 0.9 %** (0.26) |
| Fine-tuned on PlantDoc from ImageNet | 56.4 ± 3.7 % (0.56) | **64.0 ± 2.7 %** (0.63) |
| Fine-tuned on PlantDoc from our PlantVillage model | 55.4 ± 1.4 % (0.55) | **60.0 ± 0.9 %** (0.59) |

- On all 2,578 PlantDoc images the PlantVillage-only models reach 17.0 % (MobileNet) and 19.3 % (MobileViT), and identify the correct crop in only 29–37 % of images.
- MobileViT is ahead in every setting and every seed.
- PlantVillage pretraining did not help compared with ImageNet (equal for MobileNet, −4 points for MobileViT), although it reduced seed variance. Lab-image features appear to over-specialise.
- Classes that still transfer without PlantDoc training have distinctive, high-contrast symptoms (tomato late blight, corn gray leaf spot, squash powdery mildew: 40–70 % recall); healthy leaves and subtle diseases (bacterial spot, mosaic virus, soybean, cherry) are near 0 %.
- Compression costs more out of domain: FP16 is unchanged, MobileViT INT8-mixed loses ~1 point (19.3 → 18.3 %, vs. 0.2 points on PlantVillage), MobileNet INT8-mixed loses ~4 points (17.0 → 13.3 %).

## 9. Raspberry Pi 5 benchmark

- **Device:** Raspberry Pi 5, 64-bit Raspberry Pi OS, ONNX Runtime (CPU execution provider), Python virtual environment.
- **Protocol (`src/pi_benchmark.py`):** each model in a fresh process; 30 warm-up runs, then 200 single-image runs with 1 thread and with 4 threads (median and 95th percentile); accuracy on 380 test images (10 per class); peak process memory; CPU temperature before/after and throttling flags. CPU temperature stayed at 49–61 °C and no throttling occurred (`throttled=0x0`).
- Accuracy on the Pi matched Kaggle for every file, so the ARM CPU reproduces the x86 results (including the broken full-INT8 models).

| Model | Variant | Size | 1 thread | 4 threads |
|---|---|---|---|---|
| MobileNetV3-Small | FP32 | 5.96 MB | 13.2 ms | **6.1 ms** |
| | FP16 | 3.00 MB | 13.4 ms | 10.2 ms |
| | INT8 full (broken accuracy) | 1.79 MB | 7.4 ms | 5.3 ms |
| | INT8 mixed (PTQ) | 3.38 MB | 10.9 ms | 5.7 ms |
| | INT8 mixed (QAT) | 1.98 MB | 9.2 ms | 5.4 ms |
| MobileViT-XXS | FP32 | 3.88 MB | 42.9 ms | **23.5 ms** |
| | FP16 | 2.11 MB | 79.9 ms | 50.9 ms |
| | INT8 full (broken accuracy) | 1.44 MB | 44.5 ms | 32.8 ms |
| | INT8 mixed (PTQ) | 1.43 MB | 44.5 ms | 32.3 ms |

- MobileNet is ~3–4× faster than MobileViT despite having more parameters; attention, unfold/fold reshapes and SiLU are costlier on the CPU than their parameter count suggests.
- FP16 is a storage format only on this CPU: ONNX Runtime has no optimised FP16 kernels here, so FP16 models run slower than FP32.
- INT8 speeds up MobileNet's single-thread inference (13.2 → 7.4–10.9 ms) but gains little with 4 threads; for MobileViT INT8 is slower than FP32, because many operations stay in float and the inserted quantize/dequantize conversions cost more than the integer convolutions save.
- Peak memory (485–790 MB) includes the 380 test images held as float arrays (~290 MB) and the Python runtime, so it is reported only as process memory; MobileViT used ~220–300 MB more than MobileNet.
- All FP32 models are well inside a 30 fps (33 ms) camera budget with 4 threads.

## 10. Limitations

- PlantVillage consists of lab images with uniform backgrounds; near-ceiling accuracy does not imply field performance (Section 8). PlantDoc's test set is small (236 images, 4–12 per class), and crops from the same photograph may appear in both its train and test splits.
- Three seeds give limited statistical power for differences of ~0.1 percentage points.
- The HPO budget (20 trials) is small, and median pruning and 5-epoch runs bias the search toward fast-starting configurations.
- Some paper details are not stated (layer freezing, dropout, LR schedule) and are assumptions in our reconstruction; the paper's code is not public.
- The QAT implementation was simplified and did not work; INT8 conclusions are for post-training quantization with ONNX Runtime.
- Latency was measured on one device and runtime (CPU, ONNX Runtime). Mobile NPUs/GPUs or other runtimes (TFLite/XNNPACK) could change absolute numbers and the INT8/FP16 speed results. Energy consumption was not measured.
- The explanation of MobileViT's better field accuracy (global context from self-attention) is a hypothesis; it was not tested directly (e.g. with Grad-CAM).
