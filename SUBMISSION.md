# Prototype Submission Checklist

## Submission Requirements

### ✅ 1. Repository Link

**GitHub Repository:** https://github.com/Dharanish-AM/EcoInnovators-Ideathon

**Repository Contents:**
- ✓ Complete pipeline code (`pipeline_code/`)
- ✓ Trained model (`trained_model/pv_segmentation.pt`)
- ✓ Model card documentation (`model_card/model_card.md`)
- ✓ Training logs and metrics (`training_logs/`)
- ✓ Environment specifications (`environment_details/`)
- ✓ Docker deployment files (Dockerfile, docker-compose.yml)
- ✓ Interactive UI (`app.py`)
- ✓ Sample data structure (`data/`)
- ✓ Comprehensive README

### ✅ 2. DockerHub Link & Tags

**DockerHub Repository:** https://hub.docker.com/r/ecoinnovators/solar-pv-detection

**Available Tags:**

| Tag | Description | Platform | Size |
|-----|-------------|----------|------|
| `ecoinnovators/solar-pv-detection:latest` | Latest inference pipeline | linux/amd64, linux/arm64 | ~2.5GB |
| `ecoinnovators/solar-pv-detection:v1.0` | Version 1.0 release | linux/amd64, linux/arm64 | ~2.5GB |
| `ecoinnovators/solar-pv-detection:streamlit` | Interactive UI | linux/amd64, linux/arm64 | ~2.6GB |
| `ecoinnovators/solar-pv-detection:streamlit-v1.0` | UI version 1.0 | linux/amd64, linux/arm64 | ~2.6GB |

**Quick Run:**
```bash
docker pull ecoinnovators/solar-pv-detection:latest
docker run --rm \
  -v $(pwd)/data:/app/input:ro \
  -v $(pwd)/output:/app/output \
  -e MAPBOX_API_KEY=your_key \
  ecoinnovators/solar-pv-detection:latest \
  --input_xlsx /app/input/locations.xlsx \
  --output_dir /app/output
```

### ✅ 3. Model Card File

**Location:** `model_card/model_card.md`

**Contents:**
- ✓ Model architecture (U-Net + ResNet18)
- ✓ Training configuration and hyperparameters
- ✓ Performance metrics (Accuracy: 82.14%, F1: 0.6579, IoU: 0.4902)
- ✓ Training dataset details (50 samples, 40 train/10 val)
- ✓ Limitations and known failure modes
- ✓ Intended use cases (PM Surya Ghar verification)
- ✓ Geographic/demographic biases
- ✓ Ethical considerations
- ✓ Retraining guidance

**Key Metrics:**
```json
{
  "accuracy": 0.8214,
  "precision": 0.6510,
  "recall": 0.6650,
  "f1_score": 0.6579,
  "dice_coefficient": 0.6579,
  "iou": 0.4902
}
```

### ✅ 4. JSON Output

**Format:** JSONL (JSON Lines) - One JSON object per line

**Output File:** `prediction_files/predictions.jsonl`

**Sample Record Structure:**
```json
{
  "sample_id": "sample_001",
  "lat": 28.6139,
  "lon": 77.2090,
  "has_solar": true,
  "pv_area_sqm_est": 45.3,
  "buffer_radius_sqft": 1200,
  "confidence": 0.87,
  "qc_status": "VERIFIED",
  "polygons": [[[100, 150], [200, 150], [200, 250], [100, 250]]],
  "image_metadata": {
    "provider": "mapbox",
    "zoom": 20,
    "tile_size": 640,
    "timestamp": "2025-12-13T10:30:00Z"
  },
  "overlay_path": "overlays/sample_001_overlay.png"
}
```

**Field Definitions:**

| Field | Type | Description |
|-------|------|-------------|
| `sample_id` | string | Unique identifier from input data |
| `lat` | float | Latitude coordinate |
| `lon` | float | Longitude coordinate |
| `has_solar` | boolean | Solar PV detected (true/false) |
| `pv_area_sqm_est` | float | Estimated PV area in square meters |
| `buffer_radius_sqft` | int | Buffer zone used (1200 or 2400 sq.ft) |
| `confidence` | float | Prediction confidence (0.0-1.0) |
| `qc_status` | string | Quality control flag (VERIFIED/NOT_VERIFIABLE/LOW_CONFIDENCE) |
| `polygons` | array | PV panel polygon coordinates [[x,y], ...] |
| `image_metadata` | object | Imagery source and parameters |
| `overlay_path` | string | Path to visualization image |

**QC Status Values:**
- `VERIFIED`: High confidence detection (confidence ≥ 0.7)
- `LOW_CONFIDENCE`: Detection but low confidence (0.5 ≤ confidence < 0.7)
- `NOT_VERIFIABLE`: Image quality issues (clouds, shadows, missing data)

---

## Deployment Instructions

### Step 1: Build Docker Images

```bash
# Build inference pipeline
docker build -t ecoinnovators/solar-pv-detection:latest .
docker build -t ecoinnovators/solar-pv-detection:v1.0 .

# Build Streamlit UI
docker build -f Dockerfile.streamlit -t ecoinnovators/solar-pv-detection:streamlit .
```

### Step 2: Test Locally

```bash
# Test with sample data
docker run --rm \
  -v $(pwd)/data:/app/input:ro \
  -v $(pwd)/test_output:/app/output \
  -e MAPBOX_API_KEY=$MAPBOX_API_KEY \
  ecoinnovators/solar-pv-detection:latest \
  --input_xlsx /app/input/EI_train_data.xlsx \
  --output_dir /app/output

# Verify output
cat test_output/predictions.jsonl | jq
```

### Step 3: Push to DockerHub

```bash
# Login
docker login

# Tag images (replace with your DockerHub username)
docker tag ecoinnovators/solar-pv-detection:latest <username>/solar-pv-detection:latest
docker tag ecoinnovators/solar-pv-detection:v1.0 <username>/solar-pv-detection:v1.0
docker tag ecoinnovators/solar-pv-detection:streamlit <username>/solar-pv-detection:streamlit

# Push
docker push <username>/solar-pv-detection:latest
docker push <username>/solar-pv-detection:v1.0
docker push <username>/solar-pv-detection:streamlit
```

### Step 4: Create Sample Output

```bash
# Generate sample predictions for submission
python -m pipeline_code.run_pipeline \
  --input_xlsx data/EI_train_data.xlsx \
  --output_dir submission_files \
  --image_source mapbox \
  --zoom 20

# The output will be in submission_files/predictions.jsonl
```

---

## Submission Package

Submit the following:

### 1. **README with Links**

Create a `SUBMISSION.md` with:

```markdown
# EcoInnovators Solar PV Detection - Prototype Submission

## 1. Repository Link
https://github.com/Dharanish-AM/EcoInnovators-Ideathon

## 2. DockerHub Links
- Inference Pipeline: https://hub.docker.com/r/<username>/solar-pv-detection
  - Tag: `latest` (recommended)
  - Tag: `v1.0` (stable release)
- Streamlit UI: https://hub.docker.com/r/<username>/solar-pv-detection
  - Tag: `streamlit`

## 3. Model Card
Located at: `model_card/model_card.md`
Direct link: https://github.com/Dharanish-AM/EcoInnovators-Ideathon/blob/main/model_card/model_card.md

## 4. Sample JSON Output
Located at: `submission_files/predictions.jsonl`
Format: JSONL (JSON Lines)

## Quick Test
```bash
docker pull <username>/solar-pv-detection:latest
docker run --rm -v $(pwd)/data:/app/input:ro -v $(pwd)/output:/app/output \
  -e MAPBOX_API_KEY=your_key <username>/solar-pv-detection:latest
```

## Performance Summary
- Accuracy: 82.14%
- F1-Score: 0.6579
- IoU: 0.4902
- Inference Speed: ~3-5 seconds/location (CPU)
```

### 2. **Docker Verification**

```bash
# Verify image works
docker pull <username>/solar-pv-detection:latest
docker run --rm <username>/solar-pv-detection:latest --help

# Output should show usage instructions
```

### 3. **Sample Output Files**

Package these files:
- `submission_files/predictions.jsonl` (sample predictions)
- `submission_files/overlays/*.png` (visualization samples)
- `model_card/model_card.md` (model documentation)
- `SUBMISSION.md` (submission summary)

---

## Verification Checklist

Before submission, verify:

- [ ] GitHub repository is public and accessible
- [ ] All code runs without errors
- [ ] Docker images are pushed to DockerHub
- [ ] Docker images are public (not private)
- [ ] Model card is complete and up-to-date
- [ ] Sample JSON output follows specified format
- [ ] README has clear deployment instructions
- [ ] Environment variables are documented
- [ ] License file is included (MIT)
- [ ] Sample visualizations are included
- [ ] API requirements are documented

---

## Post-Submission

After submission:

1. **Monitor Issues:** Check GitHub for evaluator questions
2. **Keep Images Updated:** Don't delete DockerHub images during review
3. **Maintain Availability:** Ensure repository remains public
4. **Document Updates:** Track any post-submission changes

---

## Contact

For questions or support:
- GitHub Issues: https://github.com/Dharanish-AM/EcoInnovators-Ideathon/issues
- Email: [Your email]
- DockerHub: [Your DockerHub profile]

---

**Last Updated:** December 13, 2025
**Version:** 1.0
**Status:** Ready for Submission ✅
