# Reference Model Analysis: AnemiaVision AI vs EyeHealthAI

**Document Date**: 2026-10-11  
**Reference Repository**: `https://github.com/RAHULPATEL2002/anemia-detection`  
**Reference Demo**: `https://rahulpatel2002-anemia-detection.hf.space/`  
**Reference Paper**: `https://arxiv.org/abs/2604.22964`  
**Inspection Workspace**: Temporary scratch workspace (`.../scratch/anemia-detection`)  
**Licence Status**: Unspecified / "License TBD" (Clean-room technical analysis; zero proprietary weights or verbatim code copied)

---

## 1. Architectural & Methodological Comparison

| Dimension | AnemiaVision AI (Reference) | EyeHealthAI (Our System) | Clinical & Engineering Implications |
| :--- | :--- | :--- | :--- |
| **Clinical Target Formulation** | **Binary Classification** (`anemic` vs `non-anemic`) with heuristic Hb band lookup (e.g., `>85% confidence -> "7-9 g/dL"`). | **Continuous Regression** directly predicting verified laboratory hemoglobin in g/dL ($Hb \in [7.0, 17.4]$). | EyeHealthAI solves the physiologically genuine task. Heuristic lookup tables fail to capture borderline anemia, mild anemia, or polycythemia. |
| **Model Architecture** | EfficientNet-B0 (fast) or EfficientNet-B3 (full) with 2-layer MLP head (`Linear(1536->512) -> BatchNorm -> GELU -> Dropout(0.45) -> Linear(512->256) -> BatchNorm -> GELU -> Dropout(0.35) -> Linear(256->2)`). | ResNet-50 CNN Regressor with linear projection (`Linear(2048->1)`) + MobileNetV3-Small binary classifiers for conjunctival injection / icterus. | The reference multi-layer head design with GELU and dropout provides strong regularization against overfitting on small medical datasets. |
| **Loss Function** | Cross-Entropy Loss / Mixup-blended CE (`mixup_criterion`). | Smooth L1 Loss / Huber Loss / MSE Loss for regression; BCEWithLogitsLoss for binary classification. | Classification loss cannot be applied to continuous Hb regression without artificial quantization. |
| **Evaluation Metrics** | Accuracy, Sensitivity, Specificity, Precision, F1-score, ROC-AUC. | Patient-level and View-level MAE (g/dL), RMSE (g/dL), $R^2$, Pearson correlation vs Mean/Median Baselines. | EyeHealthAI measures true error against laboratory venous blood draws. |
| **Data Partitioning** | Folder-level directory split (`train/`, `valid/`, `test/`). | Patient-level grouped splits stratified across Hb quartiles (zero subject overlap). | Subject-level grouping is non-negotiable in EyeHealthAI to prevent multi-view data leakage. |
| **Data Augmentation** | `TrivialAugmentWide` + `RandomErasing(value="random")` + `ColorJitter` + `Mixup(alpha=0.2)`. | Color-preserving spatial augmentations (RandomHorizontalFlip, subtle RandomAffine, subtle scale/crop). **No aggressive hue shifts, solarization, or inversions**. | `TrivialAugmentWide` applies random solarize, invert, posterize, and equalize, which destroy conjunctival microvascular erythema. Incompatible with clinical color safety. |
| **Learning Rate Schedule** | Warmup + Cosine Annealing (`SequentialLR` with `LinearLR` and `CosineAnnealingLR`). | StepLR or plateau-based scheduling. | Warmup + Cosine Annealing is a state-of-the-art optimization technique highly suitable for fine-tuning our vision backbones. |
| **Gradient Regularization** | Gradient norm clipping (`max_norm=5.0`). | Unclipped standard SGD / Adam. | Gradient clipping prevents exploding gradients during fine-tuning on noisy medical views. Highly reusable. |
| **Image Preprocessing & Validation** | PIL RGB loading, Laplacian blur score threshold, brightness bounds ([40, 240]), resize + center crop. | PIL RGB loading with truncated image tolerance, Laplacian variance sharpness, luminance check ([30, 230]), 224x224 crop/resize. | Both employ discrete Laplacian blur and luminance checks. EyeHealthAI additionally verifies ocular framing. |
| **Explainability** | Grad-CAM targeting last convolutional feature maps (`features[-1]`). | Layer 4 feature map inspection / planned Grad-CAM overlay. | Grad-CAM implementation is reusable for visualizing whether the model attends to the palpebral conjunctiva vs sclera or eyelids. |

---

## 2. In-Depth Technical Audit of Reference Components

### 2.1 Training Loop & Optimizer Mechanics
- **Warmup and Cosine Annealing**: The reference project combines `LinearLR` (warmup over initial epochs) followed by `CosineAnnealingLR` down to `min_lr = 1e-6`. This stabilizes transfer learning when transitioning from frozen ImageNet weights to fine-tuning.
- **Differential Learning Rates**: Backbone parameters receive $10\times$ lower learning rate ($1\times 10^{-5}$) than the newly initialized classification head ($1\times 10^{-4}$), mitigating catastrophic forgetting.
- **Mixed Precision (`torch.amp.autocast`)**: Employs automatic mixed precision with `GradScaler` for GPU training.
- **Checkpoint Payload**: Checkpoints store rich metadata: `epoch`, `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `best_score`, `metrics`, `architecture`, and `history`.

### 2.2 Why Specific Reference Techniques Cannot Be Blindly Merged
1. **`TrivialAugmentWide` is Clinically Dangerous for Pallor Analysis**:
   - `TrivialAugmentWide` randomly samples image transformations including `Invert`, `Solarize`, `Posterize`, and `AutoContrast`.
   - Erythema reduction (pallor) is a fine chromatic gradient in the red/green ratio of conjunctival capillary loops. Inverting or solarizing colors destroys the physical biomarker, teaching the neural network to rely on spurious non-biological artifacts.
2. **Standard Mixup on Continuous Targets**:
   - The reference mixes categorical one-hot targets $y_a, y_b$ using cross-entropy.
   - For continuous regression, mixing targets $y = \lambda y_a + (1 - \lambda) y_b$ is theoretically possible, but mixing two distinct eye crops visually blends distinct anatomical boundaries (cornea, palpebral conjunctiva, eyelashes), generating unnatural synthetic images that do not resemble human ocular anatomy.
3. **Accuracy-Based Early Stopping**:
   - Early stopping on validation classification accuracy does not apply to continuous regression. Model A requires early stopping based on **validation MAE** or **validation MSE**.
4. **Heuristic Hemoglobin Lookup Table**:
   - In `predict.py` lines 101–116, the reference model maps classification confidence to arbitrary text bands:
     ```python
     if prediction == "Anemic":
         if confidence >= 0.85: return "7-9 g/dL (estimated)"
     ```
   - This creates an illusion of laboratory quantification. EyeHealthAI rejects this heuristic and strictly retains direct regression against verified laboratory measurements from `India.xlsx` and `Italy.xlsx`.

---

## 3. What is Reusable in EyeHealthAI

The following concepts from the reference project are technically sound and will be adapted independently:
1. **Two-Stage Multi-Layer Projection Head**:
   - Replacing simple linear projections with a normalized, regularized MLP (`AdaptiveAvgPool2d -> Flatten -> Linear -> BatchNorm1d / LayerNorm -> GELU -> Dropout -> Linear`) reduces overfitting on small cohorts.
2. **Warmup + Cosine Annealing Learning Rate Scheduling**:
   - Implementing a smooth cosine decay schedule after linear warmup improves convergence stability.
3. **Gradient Norm Clipping**:
   - Clipping gradients at `max_norm=1.0` or `5.0` stabilizes training when backpropagating through deep backbones.
4. **Enhanced Checkpoint Schema**:
   - Saving complete training metadata, optimization states, architecture identifiers, and metric histories into `.pth` checkpoints enables exact training reproducibility and seamless resumption.
5. **Deterministic Inference Preprocessing Contract**:
   - Ensuring that inference preprocessing matches validation preprocessing exactly (RGB order, Bicubic/Bilinear resizing, ImageNet normalization, deterministic center crop) prevents domain shift.

---

## 4. Summary & Action Plan for EyeHealthAI

- **Model A (Hb Regression)**: Retain continuous regression as the primary objective. Test an EfficientNet or MobileNet backbone alongside ResNet-50 using cosine annealing, differential learning rates, and our color-preserving pipeline. Compare all models against mean/median baselines.
- **Model B (Conjunctival Injection)**: Evaluate clinical datasets (e.g. SLID slit-lamp data) to replace the confounded resolution-shortcut dataset.
- **Model C (Jaundice / Scleral Icterus)**: Evaluate legally permissible, verified scleral icterus data for a normal-vs-jaundiced classifier, preserving scleral yellowing signals.
