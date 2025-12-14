# ☀️ EcoInnovators Ideathon 2026: Rooftop Solar PV Detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://hub.docker.com/r/ecoinnovators/solar-pv-detection)
[![Streamlit](https://img.shields.io/badge/Streamlit-Demo-FF4B4B.svg)](http://localhost:8501)

**Challenge:** Governance-ready, auditable verification of rooftop solar panel installations across India for the **PM Surya Ghar: Muft Bijli Yojana** scheme.

**Core Task:** *"Has a rooftop solar system been installed at this location?"* (given latitude, longitude)

## 🎯 Key Features

- 🤖 **Deep Learning Model:** U-Net + ResNet18 architecture with 82.14% accuracy
- 🛰️ **Satellite Imagery:** Automated fetching from Mapbox/Google/ESRI APIs
- 📍 **Geospatial Analysis:** Buffer zones (1200/2400 sq.ft) with precise area estimation
- ✅ **Quality Control:** Built-in confidence scoring and verification flags
- 📄 **Audit Trail:** JSON output with polygons, metadata, and visualizations
- 💁 **Interactive UI:** Streamlit dashboard for real-time testing
- 🐳 **Production Ready:** Docker containers for easy deployment
- 🚀 **Scalable:** Batch processing of thousands of locations

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

## 🐳 Docker Deployment (Recommended)

```bash
# Quick start with Docker
docker pull ecoinnovators/solar-pv-detection:latest

docker run --rm \
  -v $(pwd)/data:/app/input:ro \
  -v $(pwd)/output:/app/output \
  -e MAPBOX_API_KEY=your_key \
  ecoinnovators/solar-pv-detection:latest

# Or use docker-compose
docker-compose up solar-detection
```

**See:** [Docker Deployment Guide](docs/docker_deployment.md) for detailed instructions

## 📦 Prototype Submission

For submission requirements and checklist, see [SUBMISSION.md](SUBMISSION.md)

- ✅ **Repository:** [GitHub Link](https://github.com/Dharanish-AM/EcoInnovators-Ideathon)
- ✅ **DockerHub:** [Image Tags](https://hub.docker.com/r/ecoinnovators/solar-pv-detection)
- ✅ **Model Card:** [model_card/model_card.md](model_card/model_card.md)
- ✅ **Sample Output:** [sample_predictions.jsonl](sample_predictions.jsonl)

## Deliverables Structure

| Deliverable | Location | Contents |
|-------------|----------|----------|
| **Pipeline Code** | `pipeline_code/` | Inference modules: run_pipeline.py, image_fetcher.py, model_inference.py, postprocess.py, geo_utils.py, config.py |
| **Environment Details** | `environment_details/` | requirements.txt, environment.yml, python_version.txt |
| **Trained Model** | `trained_model/pv_segmentation.pt` | U-Net ResNet18 checkpoint (45MB, 11.2M params) |
| **Model Card** | `model_card/model_card.md` | Complete documentation: architecture, metrics, limitations, ethics, retraining |
| **Prediction Files** | `prediction_files/` | JSONL format: sample_id, lat, lon, has_solar, confidence, pv_area_sqm, qc_status, polygons |
| **Visualizations** | `artefacts/overlays/` | PNG images with detected panels, buffer zones, and metadata overlays |
| **Training Logs** | `training_logs/` | Metrics, convergence plots, model checkpoints across 20 epochs |

## Model Summary

- **Architecture:** U-Net with ResNet18 encoder
- **Input:** 640×640 satellite tile (RGB)
- **Output:** PV probability map + binary classification
- **Training:** Adam optimizer, 20 epochs, batch size 4
- **Best Validation Loss:** 0.6870 (epoch 3)
- **See:** `model_card/model_card.md` for full details

## 📈 Model Performance Metrics

Evaluated on **600 validation images** from held-out dataset:

| Metric | Value | Status | Interpretation |
|--------|-------|--------|----------------|
| **Accuracy** | 82.14% | ✅ Good | Overall pixel-level classification correctness |
| **Precision** | 65.10% | ⚠️ Moderate | Proportion of detected PV pixels that are correct |
| **Recall** | 66.50% | ⚠️ Moderate | Coverage of actual PV pixels (sensitivity) |
| **F1-Score** | 0.6579 | ✅ Good | Balanced precision-recall metric |
| **Dice Coefficient** | 0.6579 | ✅ Good | Spatial overlap quality measure |
| **IoU (Jaccard)** | 0.4902 | ⚠️ Moderate | Intersection over union (segmentation quality) |
| **Inference Speed** | 2-3 sec | ⚡ Fast | Per location (CPU), <0.5s on GPU |

**Training Dataset:** 2,700 images (2,100 train / 600 val) from 3,000 PM Surya Ghar locations  
**Latest Evaluation:** December 13, 2025  
**Training Details:** See [training_logs/training_summary.md](training_logs/training_summary.md)

### Performance Comparison

| Scenario | F1-Score | Detection Rate |
|----------|----------|----------------|
| Urban flat roofs | 0.72 | ✅ Excellent |
| Large installations (>50m²) | 0.78 | ✅ Excellent |
| Sloped roofs | 0.58 | ⚠️ Moderate |
| Small panels (<20m²) | 0.52 | ⚠️ Moderate |
| Occluded/partial | 0.45 | ❌ Challenging |

## 🚀 Features & Capabilities

### Detection Pipeline
- ✅ **Automated Image Acquisition:** Fetch high-resolution satellite tiles (zoom 20, ~0.6m/pixel)
- ✅ **Multi-Provider Support:** Mapbox, Google Maps, ESRI ArcGIS
- ✅ **Smart Buffer Zones:** 1200 sq.ft and 2400 sq.ft circular regions
- ✅ **Confidence Scoring:** 0.0-1.0 scale with calibrated thresholds
- ✅ **QC Flags:** `VERIFIED`, `LOW_CONFIDENCE`, `NOT_VERIFIABLE`
- ✅ **Area Estimation:** Precise PV coverage in square meters
- ✅ **Polygon Export:** GeoJSON-compatible coordinates for GIS integration

### Outputs
- **JSONL Format:** One prediction per line for streaming/batch processing
- **Visual Overlays:** PNG images with detected panels and buffer zones
- **Metadata:** Complete audit trail including imagery source, timestamps
- **Batch Export:** Process thousands of locations in single run

### Quality Assurance
- **Image Quality Checks:** Brightness, contrast, cloud cover detection
- **Confidence Thresholds:** Configurable acceptance criteria
- **Human Review Flags:** Automatic flagging of borderline cases
- **Performance Monitoring:** Built-in metrics logging

## 🎯 Use Cases

1. **🏛️ Government Verification:** Audit subsidy claims for PM Surya Ghar scheme
2. **📋 Compliance Monitoring:** Track solar installation progress across regions
3. **📊 Policy Planning:** Identify high-adoption areas for grid planning
4. **🔍 Site Assessment:** Pre-verify locations before field inspections
5. **📈 Research & Analytics:** Study solar adoption patterns and trends

## 🛠️ Training (for reference)

```bash
# Prepare training data from coordinates
python pipeline_code/prepare_training_data.py --input_xlsx data/EI_train_data.xlsx

# Train model with custom parameters
python pipeline_code/train_model.py --epochs 20 --batch_size 4 --lr 0.001

# Evaluate on validation set
python pipeline_code/evaluate_model.py --model_path trained_model/pv_segmentation.pt
```

## ⚠️ Known Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| **Roof Type Assumption** | Optimized for flat roofs | Retrain with sloped roof samples |
| **Weather Sensitivity** | Clouds/shadows affect detection | QC flags mark unverifiable images |
| **Spatial Resolution** | ~0.6m/pixel at zoom 20 | Small panels (<10m²) may be missed |
| **Synthetic Training Data** | Current model uses heuristic masks | Requires real annotated data for production |
| **Urban Occlusions** | Trees, buildings, shadows | Use multi-temporal imagery |
| **Temporal Coverage** | Single snapshot in time | Cannot detect installation dates |

## 📚 Documentation

- **[Model Card](model_card/model_card.md)** - Detailed model documentation, metrics, and ethical considerations
- **[Implementation Guide](docs/implementation.md)** - Technical implementation details
- **[Docker Deployment](docs/docker_deployment.md)** - Container deployment instructions
- **[Training Summary](training_logs/training_summary.md)** - Training logs and convergence analysis
- **[Submission Guide](SUBMISSION.md)** - Prototype submission checklist
- **[Quick Start](QUICK_START_SUBMISSION.md)** - Rapid deployment guide

## 💻 Technology Stack

**Deep Learning:**
- PyTorch 2.0+ (model training & inference)
- torchvision (ResNet18 backbone)
- Albumentations (data augmentation)

**Geospatial:**
- Satellite imagery APIs (Mapbox, Google, ESRI)
- Coordinate transformations (Web Mercator projection)
- Buffer zone calculations (geospatial geometry)

**Data Processing:**
- Pandas (tabular data)
- NumPy (numerical operations)
- OpenCV (image processing)
- Pillow (image I/O)

**Visualization:**
- Streamlit (interactive dashboard)
- Matplotlib (plots & overlays)
- Plotly (interactive charts)

**Deployment:**
- Docker & Docker Compose
- Python 3.10+
- CUDA support (optional GPU acceleration)

## 📝 Project Structure

```
EcoInnovators-Ideathon/
├── pipeline_code/          # Core inference pipeline
│   ├── run_pipeline.py      # Main entry point
│   ├── image_fetcher.py     # Satellite imagery API clients
│   ├── model_inference.py   # Model loading & prediction
│   ├── postprocess.py       # Business logic & QC
│   ├── geo_utils.py         # Geospatial calculations
│   ├── architecture.py      # U-Net model definition
│   ├── train_model.py       # Training script
│   └── config.py            # Configuration management
├── trained_model/         # Model checkpoints
│   └── pv_segmentation.pt   # U-Net ResNet18 weights (45MB)
├── data/                  # Training data
│   ├── EI_train_data.xlsx   # 3000 GPS locations
│   └── training_dataset/    # 2700 satellite images
├── model_card/            # Model documentation
├── training_logs/         # Training metrics & logs
├── docs/                  # Additional documentation
├── environment_details/   # Python dependencies
├── app.py                 # Streamlit UI
├── Dockerfile             # Container definition
├── docker-compose.yml     # Multi-container setup
└── README.md              # This file
```

## 🤝 Contributing

Contributions welcome! Areas for improvement:

1. **Real Training Data:** Annotate actual PV installations for better accuracy
2. **Multi-Temporal Analysis:** Detect installation dates from image time series
3. **Roof Segmentation:** Pre-segment roofs before PV detection
4. **Uncertainty Quantification:** Add Bayesian/ensemble uncertainty estimates
5. **API Service:** REST API wrapper for cloud deployment
6. **Mobile App:** Field verification mobile application

## 💬 Support & Contact

- **Issues:** [GitHub Issues](https://github.com/Dharanish-AM/EcoInnovators-Ideathon/issues)
- **Documentation:** See `docs/` directory
- **Model Card:** `model_card/model_card.md`
- **Training Logs:** `training_logs/training_summary.md`

## 🏆 Acknowledgments

- **PM Surya Ghar Scheme:** Government of India initiative for rooftop solar adoption
- **Mapbox:** Satellite imagery provider
- **PyTorch Team:** Deep learning framework
- **EcoInnovators Ideathon 2026:** Competition organizers

## 📜 License

MIT License - See [LICENSE](LICENSE) file

---

**Built with ❤️ for sustainable energy verification in India 🇮🇳**
