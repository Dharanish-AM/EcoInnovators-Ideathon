# Training Checkpoint - Full Dataset (3000 rows)

## Status: ⏸️ PAUSED

**Paused at:** 2025-12-12 (Epoch 3/20)  
**Training progress:** 3 epochs completed, 17 remaining  
**Elapsed time:** ~6 minutes on CPU

## Progress Summary

| Epoch | Train Loss | Val Loss | Status |
|-------|-----------|----------|--------|
| 1 | 1.1400 | 1.1232 | ✅ Saved |
| 2 | 0.9777 | 1.0508 | ✅ Saved |
| 3 | 0.9559 | 1.0156 | ✅ Saved (Best) |

**Best Model:** `trained_model/pv_segmentation.pt` (epoch 3, val_loss=1.0156)

## Dataset Info

- **Training set:** 2,422 samples
- **Validation set:** 644 samples
- **Total:** 3,000 samples from EI_train_data.xlsx
- **Location:** `data/training_dataset/`

## To Resume Training

Simply run the same command:
```bash
source .venv/bin/activate
python -m pipeline_code.train_model \
  --dataset_dir data/training_dataset \
  --output_model trained_model/pv_segmentation.pt \
  --epochs 20 \
  --batch_size 8 \
  --lr 0.001
```

**The model will:**
1. Load the best checkpoint from epoch 3
2. Continue training from epoch 4
3. NOT retrain epochs 1–3
4. Complete remaining 17 epochs
5. Update `trained_model/pv_segmentation.pt` as it improves

## Logs

- **Training log:** `logs_train_full.txt` (contains full epoch history)
- **Checkpoint metadata:** `training_logs/checkpoint_full_dataset.json`

---
**Next steps:** Run evaluation after training completes to compute accuracy, precision, recall, F1, Dice, IoU metrics.
