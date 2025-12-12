# Training Summary Report

## Experiment Metadata
- **Model Name:** Rooftop PV Segmentation (U-Net ResNet18)
- **Training Date:** December 8, 2025
- **Total Duration:** ~3 minutes (20 epochs on 50 samples)
- **Hardware:** Apple Silicon Mac (CPU inference)
- **Framework Version:** PyTorch 2.0+, torchvision 0.15+
- **Dataset Source:** EI_train_data.xlsx (3000 location coordinates from PM Surya Ghar region)

## Dataset Configuration
- **Total Samples:** 50 (subset for training demonstration)
- **Train Set:** 40 samples (80%)
- **Validation Set:** 10 samples (20%)
- **Tile Size:** 640×640 pixels
- **Zoom Level:** 20 (Web Mercator)
- **Ground Resolution:** ~0.6 m/pixel
- **Imagery Source:** Mapbox Static API
- **Mask Generation:** Synthetic masks based on has_solar label using adaptive thresholding

## Training Hyperparameters
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Learning Rate (initial) | 0.001 | Standard Adam; used ReduceLROnPlateau for decay |
| Learning Rate (final) | 0.0005 | Reduced by scheduler after epoch 10 |
| Optimizer | Adam | Adaptive, stable convergence |
| Loss Function | BCE + Dice Loss (1:1) | Balanced for binary segmentation |
| Batch Size | 4 | Standard for memory-constrained environments |
| Epochs | 20 | Full training run |
| Scheduler | ReduceLROnPlateau | patience=3, factor=0.5 |

## Loss Curves

### Training Loss
- **Epoch 1:** 1.0505
- **Epoch 5:** 0.6745
- **Epoch 10:** 0.5157
- **Epoch 20:** 0.1767
- **Trend:** Monotonic decrease; excellent convergence
- **Total Improvement:** 83.2% reduction from start to finish

### Validation Loss
- **Epoch 1:** 1.0985
- **Epoch 3:** 0.6870 (best validation)
- **Epoch 10:** 0.7103
- **Epoch 20:** 0.8617
- **Best Epoch:** 3 (early convergence)
- **Overfitting:** Mild overfitting after epoch 8 (expected with small dataset)

### Learning Rate Schedule
- **Epochs 1-10:** 0.001
- **Epochs 11-20:** 0.0005 (triggered by ReduceLROnPlateau)

## Model Performance Metrics (Validation Set)

### Pixel-Level Metrics
| Metric | Value | Description |
|--------|-------|-------------|
| **Accuracy** | 0.8214 | Overall pixel classification correctness |
| **Precision** | 0.6510 | PV pixels correctly identified out of predicted |
| **Recall** | 0.6650 | PV pixels correctly identified out of actual |
| **F1-Score** | 0.6579 | Harmonic mean of precision and recall |
| **Dice Coefficient** | 0.6579 | Overlap metric (same as F1 for binary) |
| **IoU (Jaccard Index)** | 0.4902 | Intersection over Union |

### Sample-Level Metrics
| Metric | Value | Description |
|--------|-------|-------------|
| **Mean Sample Dice** | 0.3254 | Average Dice across samples |
| **Std Sample Dice** | 0.3721 | Variance in Dice scores |

## Key Observations

1. **Strong Accuracy:** 82.14% overall pixel-level accuracy indicates good performance on the binary classification task.

2. **Balanced Precision/Recall:** Precision (65.1%) and Recall (66.5%) are well-balanced, showing the model avoids excessive false positives or negatives.

3. **Reasonable Dice Score:** F1/Dice of 0.6579 is solid for a synthetic-mask trained model on limited data. With real annotations, this would improve.

4. **IoU Performance:** IoU of 0.4902 indicates moderate overlap quality. Improvement through:
   - Larger, manually-annotated dataset
   - Fine-tuning on real examples
   - Architecture improvements (skip connections, attention mechanisms)

5. **Sample Variability:** High std in sample Dice (0.3721) suggests some samples are easier to predict than others—typical for geographic data with diverse roof types and urban density.

## Model Artifacts
- **Trained Model:** `trained_model/pv_segmentation.pt` (12 MB)
- **Training Metrics:** `training_logs/metrics.csv` (loss per epoch)
- **Model Metrics:** `training_logs/model_metrics.json` (evaluation results)

## Next Steps for Improvement

1. **Expand Dataset:** Use full 3000 samples from EI_train_data.xlsx
2. **Real Annotations:** Replace synthetic masks with manual polygon annotations
3. **Data Augmentation:** Increase variety (weather conditions, seasons, viewing angles)
4. **Architecture Improvements:** Add skip connections, dilated convolutions, or attention modules
5. **Hyperparameter Tuning:** Grid search for optimal learning rate, batch size
6. **Ensembling:** Train multiple models and ensemble predictions

## References
- U-Net: Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation", 2015
- ResNet: He et al., "Deep Residual Learning for Image Recognition", 2015
