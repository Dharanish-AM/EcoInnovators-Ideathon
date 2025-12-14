# Quick Submission Guide

## Prerequisites
- [x] Trained model exists (`trained_model/pv_segmentation.pt`)
- [x] Model card complete (`model_card/model_card.md`)
- [x] Docker files created
- [ ] DockerHub account ready
- [ ] GitHub repository updated

## Step-by-Step Submission Process

### Step 1: Build and Test Docker Images (5-10 minutes)

**Windows PowerShell:**
```powershell
# Navigate to project directory
cd d:\Github_Repository\EcoInnovators-Ideathon

# Build images
.\build_and_push.ps1
```

**Linux/Mac:**
```bash
cd /path/to/EcoInnovators-Ideathon
chmod +x build_and_push.sh
./build_and_push.sh
```

**Manual Build:**
```bash
docker build -t yourusername/solar-pv-detection:latest .
docker build -t yourusername/solar-pv-detection:v1.0 .
docker build -f Dockerfile.streamlit -t yourusername/solar-pv-detection:streamlit .
```

### Step 2: Test Locally (2-3 minutes)

```powershell
# Create test input
mkdir test_output

# Test inference (replace YOUR_API_KEY)
docker run --rm `
  -v ${PWD}/data:/app/input:ro `
  -v ${PWD}/test_output:/app/output `
  -e MAPBOX_API_KEY=YOUR_API_KEY `
  yourusername/solar-pv-detection:latest `
  --input_xlsx /app/input/EI_train_data.xlsx `
  --output_dir /app/output

# Verify output
cat test_output/predictions.jsonl | ConvertFrom-Json
```

### Step 3: Push to DockerHub (3-5 minutes)

```bash
# Login to DockerHub
docker login

# Push images
docker push yourusername/solar-pv-detection:latest
docker push yourusername/solar-pv-detection:v1.0
docker push yourusername/solar-pv-detection:streamlit
```

### Step 4: Update Documentation (2 minutes)

Update `SUBMISSION.md`:
1. Replace `<username>` with your DockerHub username
2. Update DockerHub links
3. Verify all links work

### Step 5: Generate Sample Predictions (Optional, 5 minutes)

```powershell
# Generate real predictions for submission
python -m pipeline_code.run_pipeline `
  --input_xlsx data/EI_train_data.xlsx `
  --output_dir submission_output `
  --image_source mapbox `
  --zoom 20

# Copy to submission folder
Copy-Item submission_output/predictions.jsonl -Destination sample_predictions.jsonl
```

### Step 6: Commit and Push to GitHub (2 minutes)

```bash
git add .
git commit -m "feat: add Docker deployment and submission materials"
git push origin main
```

### Step 7: Prepare Submission Package

Create a submission document with:

```markdown
# Prototype Submission: EcoInnovators Solar PV Detection

## 1. Repository Link
https://github.com/Dharanish-AM/EcoInnovators-Ideathon

## 2. DockerHub Links & Tags

**Repository:** https://hub.docker.com/r/yourusername/solar-pv-detection

**Tags:**
- `latest` - Latest inference pipeline (recommended)
- `v1.0` - Stable release version 1.0
- `streamlit` - Interactive UI version

**Quick Run:**
```bash
docker pull yourusername/solar-pv-detection:latest
docker run --rm -e MAPBOX_API_KEY=key yourusername/solar-pv-detection:latest --help
```

## 3. Model Card File

**Location:** `model_card/model_card.md`
**Direct Link:** https://github.com/Dharanish-AM/EcoInnovators-Ideathon/blob/main/model_card/model_card.md

**Key Metrics:**
- Accuracy: 82.14%
- F1-Score: 0.6579
- IoU: 0.4902

## 4. JSON Output

**Sample File:** `sample_predictions.jsonl`
**Direct Link:** https://github.com/Dharanish-AM/EcoInnovators-Ideathon/blob/main/sample_predictions.jsonl

**Format:** JSONL (JSON Lines) - One JSON object per line

**Schema:**
```json
{
  "sample_id": "string",
  "lat": float,
  "lon": float,
  "has_solar": boolean,
  "pv_area_sqm_est": float,
  "buffer_radius_sqft": int,
  "confidence": float,
  "qc_status": "VERIFIED|LOW_CONFIDENCE|NOT_VERIFIABLE",
  "polygons": [[[x, y], ...]],
  "image_metadata": {...},
  "overlay_path": "string"
}
```

## 5. Additional Documentation

- **Implementation Guide:** `docs/implementation.md`
- **Docker Deployment:** `docs/docker_deployment.md`
- **Training Logs:** `training_logs/training_summary.md`
- **Model Metrics:** `training_logs/model_metrics.json`

## 6. Quick Verification

Test the submission:

```bash
# Pull and test
docker pull yourusername/solar-pv-detection:latest
docker run --rm yourusername/solar-pv-detection:latest --help

# Should output usage instructions without errors
```

## 7. Contact

- **GitHub Issues:** https://github.com/Dharanish-AM/EcoInnovators-Ideathon/issues
- **Email:** [Your email]

---

**Submission Date:** December 13, 2025
**Status:** ✅ Ready for Review
```

## Verification Checklist

Before submitting, verify:

### GitHub Repository
- [ ] Repository is public
- [ ] All code is committed and pushed
- [ ] README is updated with Docker instructions
- [ ] Model card is complete
- [ ] Sample predictions.jsonl is included
- [ ] LICENSE file exists
- [ ] .gitignore is properly configured

### DockerHub
- [ ] Images are successfully built
- [ ] Images are pushed to DockerHub
- [ ] Images are public (not private)
- [ ] Tags are properly named (latest, v1.0, streamlit)
- [ ] Images can be pulled successfully
- [ ] Images run without errors

### Model Card
- [ ] Architecture documented
- [ ] Training configuration included
- [ ] Performance metrics listed
- [ ] Limitations clearly stated
- [ ] Use cases described
- [ ] Retraining guidance provided

### JSON Output
- [ ] Follows JSONL format (one JSON per line)
- [ ] All required fields present
- [ ] Sample predictions included
- [ ] Schema documented
- [ ] Example records provided

### Documentation
- [ ] README has clear instructions
- [ ] Docker deployment guide exists
- [ ] SUBMISSION.md is complete
- [ ] All links are valid
- [ ] Environment variables documented

## Common Issues and Solutions

### Issue: Docker build fails
**Solution:** Check that all files are in place:
```powershell
ls pipeline_code/
ls trained_model/
ls environment_details/requirements.txt
```

### Issue: Image too large (>3GB)
**Solution:** Optimize Dockerfile:
- Use multi-stage builds
- Clean up apt cache
- Use .dockerignore properly

### Issue: Model file not found
**Solution:** Ensure model is not in .gitignore and is committed:
```bash
git add -f trained_model/pv_segmentation.pt
git commit -m "add trained model"
```

### Issue: API key not working in container
**Solution:** Pass environment variable correctly:
```powershell
docker run -e MAPBOX_API_KEY=$env:MAPBOX_API_KEY ...
```

### Issue: Volume mount not working (Windows)
**Solution:** Use absolute paths:
```powershell
docker run -v C:/Users/Name/project/data:/app/input ...
```

## Estimated Timeline

| Task | Time | Status |
|------|------|--------|
| Build Docker images | 5-10 min | ⏳ |
| Test locally | 2-3 min | ⏳ |
| Push to DockerHub | 3-5 min | ⏳ |
| Update documentation | 2 min | ⏳ |
| Generate samples | 5 min | ⏳ |
| Git commit & push | 2 min | ⏳ |
| **Total** | **~20-30 min** | ⏳ |

## Next Steps

1. Run `build_and_push.ps1` (or `.sh`)
2. Update SUBMISSION.md with your DockerHub username
3. Test Docker image locally
4. Push to GitHub
5. Submit prototype with all four deliverables!

---

**Good luck with your submission! 🚀**
