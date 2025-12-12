# EcoInnovators Ideathon 2026: Rooftop Solar PV Detection

**Challenge:** Governance-ready, auditable verification of rooftop solar installations across India for the PM Surya Ghar scheme.

**Core Task:** "Has a rooftop solar system been installed here?" (given latitude, longitude)

## Quick Start

### Prerequisites
- Python 3.10+ (see `environment_details/python_version.txt`)
- Mapbox API key (or Google/ESRI alternative)

### 1. Install Dependencies
```bash
# Using pip
pip install -r environment_details/requirements.txt

# Or using conda
conda env create -f environment_details/environment.yml
```

### 2. Configure API Keys
```bash
# Create .env in project root
echo "MAPBOX_API_KEY=<your_key>" > .env
```

### 3. Run Inference
```bash
# Batch process locations from Excel
python -m pipeline_code.run_pipeline \
  --input_xlsx data/EI_train_data.xlsx \
  --output_dir prediction_files \
  --zoom 20
```

**Output:** `prediction_files/predictions.jsonl` with detection results, confidence, and PV area estimates

### 4. Optional: Interactive Dashboard
```bash
streamlit run app.py  # http://localhost:8501
```



## Deliverables Structure

| Deliverable | Location | Contents |
|-------------|----------|----------|
| **Pipeline Code** | `pipeline_code/` | Inference modules: run_pipeline.py, image_fetcher.py, model_inference.py, postprocess.py, geo_utils.py, config.py |
| **Environment Details** | `environment_details/` | requirements.txt, environment.yml, python_version.txt |
| **Trained Model** | `trained_model/pv_segmentation.pt` | U-Net ResNet18 checkpoint (45MB) |
| **Model Card** | `model_card/model_card.md` | Architecture, assumptions, metrics, limitations, retraining guidance |
| **Prediction Files** | `prediction_files/` | predictions.jsonl with sample outputs (lat, lon, has_solar, confidence, pv_area_sqm) |
| **Artefacts** | `artefacts/overlays/` | PNG visualizations of predictions (bounding boxes, masks) |
| **Training Logs** | `training_logs/metrics.csv` | Loss, LR across 20 training epochs |

## Model Summary

- **Architecture:** U-Net with ResNet18 encoder
- **Input:** 640×640 satellite tile (RGB)
- **Output:** PV probability map + binary classification
- **Training:** Adam optimizer, 20 epochs, batch size 4
- **Best Validation Loss:** 0.6870 (epoch 3)
- **See:** `model_card/model_card.md` for full details

## Model Performance Metrics

Evaluated on validation set (10 samples):

| Metric | Value | Notes |
|--------|-------|-------|
| **Accuracy** | 82.14% | Pixel-level classification correctness |
| **Precision** | 65.10% | PV pixels correctly identified |
| **Recall** | 66.50% | Coverage of actual PV pixels |
| **F1-Score** | 0.6579 | Harmonic mean of precision/recall |
| **Dice Coefficient** | 0.6579 | Overlap quality metric |
| **IoU (Jaccard)** | 0.4902 | Intersection over union |

**Latest Metrics:** Run on 2025-12-11  
**Dataset:** 50 training samples (40 train / 10 val) from EI_train_data.xlsx  
**Training Logs:** See `training_logs/training_summary.md` for detailed analysis

## Training (for reference)

```bash
python pipeline_code/prepare_training_data.py --input_xlsx data/EI_train_data.xlsx
python pipeline_code/train_model.py --epochs 20
```

## Known Limitations

- Assumes relatively flat roofs
- Sensitive to clouds, shadows, image age
- Resolution ~0.6m (zoom 20)
- May struggle with urban overlaps, small panels

## License

MIT License - See LICENSE file
