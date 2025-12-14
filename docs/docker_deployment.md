# Docker Deployment Guide

## Quick Start with Docker

### Prerequisites
- Docker Desktop installed (Windows/Mac) or Docker Engine (Linux)
- Mapbox API key (or alternative imagery provider)

### Option 1: Run Pre-built Image

```bash
# Pull the image from DockerHub
docker pull ecoinnovators/solar-pv-detection:latest

# Run inference on your data
docker run --rm \
  -v /path/to/your/data:/app/input:ro \
  -v /path/to/output:/app/output \
  -e MAPBOX_API_KEY=your_api_key_here \
  ecoinnovators/solar-pv-detection:latest \
  --input_xlsx /app/input/locations.xlsx \
  --output_dir /app/output \
  --zoom 20
```

**Windows PowerShell:**
```powershell
docker run --rm `
  -v ${PWD}/data:/app/input:ro `
  -v ${PWD}/prediction_files:/app/output `
  -e MAPBOX_API_KEY=$env:MAPBOX_API_KEY `
  ecoinnovators/solar-pv-detection:latest `
  --input_xlsx /app/input/EI_train_data.xlsx `
  --output_dir /app/output `
  --zoom 20
```

### Option 2: Build from Source

```bash
# Clone repository
git clone https://github.com/Dharanish-AM/EcoInnovators-Ideathon.git
cd EcoInnovators-Ideathon

# Build the Docker image
docker build -t ecoinnovators/solar-pv-detection:latest .

# Run with your data
docker run --rm \
  -v $(pwd)/data:/app/input:ro \
  -v $(pwd)/prediction_files:/app/output \
  -e MAPBOX_API_KEY=your_api_key \
  ecoinnovators/solar-pv-detection:latest
```

### Option 3: Docker Compose (Recommended)

```bash
# 1. Create .env file with your API keys
echo "MAPBOX_API_KEY=your_key_here" > .env

# 2. Run inference pipeline
docker-compose up solar-detection

# 3. Optional: Launch Streamlit UI
docker-compose --profile ui up streamlit-ui
# Access UI at http://localhost:8501
```

## Building and Pushing to DockerHub

### Build Multi-Platform Image

```bash
# Set up buildx (one-time setup)
docker buildx create --name multiplatform --use
docker buildx inspect --bootstrap

# Build for multiple platforms
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t ecoinnovators/solar-pv-detection:latest \
  -t ecoinnovators/solar-pv-detection:v1.0 \
  --push \
  .

# Build Streamlit UI image
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -f Dockerfile.streamlit \
  -t ecoinnovators/solar-pv-detection:streamlit \
  -t ecoinnovators/solar-pv-detection:streamlit-v1.0 \
  --push \
  .
```

### Push to DockerHub

```bash
# Login to DockerHub
docker login

# Tag your images
docker tag ecoinnovators/solar-pv-detection:latest <your-dockerhub-username>/solar-pv-detection:latest

# Push to registry
docker push <your-dockerhub-username>/solar-pv-detection:latest
docker push <your-dockerhub-username>/solar-pv-detection:v1.0
docker push <your-dockerhub-username>/solar-pv-detection:streamlit
```

## Image Tags

| Tag | Description | Size |
|-----|-------------|------|
| `latest` | Latest stable release | ~2.5GB |
| `v1.0` | Version 1.0 release | ~2.5GB |
| `streamlit` | Streamlit UI version | ~2.6GB |
| `streamlit-v1.0` | UI version 1.0 | ~2.6GB |

## Usage Examples

### Process Single Location

```bash
# Create input Excel with one row
echo "sample_id,latitude,longitude" > single.xlsx
echo "test_1,28.6139,77.2090" >> single.xlsx

docker run --rm \
  -v $(pwd):/app/input \
  -v $(pwd)/output:/app/output \
  -e MAPBOX_API_KEY=your_key \
  ecoinnovators/solar-pv-detection:latest \
  --input_xlsx /app/input/single.xlsx \
  --output_dir /app/output
```

### Batch Processing

```bash
# Process entire dataset
docker run --rm \
  -v $(pwd)/data:/app/input:ro \
  -v $(pwd)/prediction_files:/app/output \
  -e MAPBOX_API_KEY=$MAPBOX_API_KEY \
  ecoinnovators/solar-pv-detection:latest \
  --input_xlsx /app/input/EI_train_data.xlsx \
  --output_dir /app/output \
  --image_source mapbox \
  --zoom 20
```

### Custom Configuration

```bash
docker run --rm \
  -v $(pwd)/data:/app/input:ro \
  -v $(pwd)/output:/app/output \
  -e GOOGLE_MAPS_API_KEY=your_google_key \
  ecoinnovators/solar-pv-detection:latest \
  --input_xlsx /app/input/locations.xlsx \
  --output_dir /app/output \
  --image_source google \
  --zoom 19 \
  --tile_size 512
```

## Output Files

After running the container, you'll find:

```
output/
├── predictions.jsonl          # JSON Lines format with all predictions
├── overlays/                  # Visualization images
│   ├── sample_1_overlay.png
│   ├── sample_2_overlay.png
│   └── ...
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `MAPBOX_API_KEY` | Yes* | Mapbox Static Maps API key |
| `GOOGLE_MAPS_API_KEY` | Yes* | Google Maps Static API key |
| `ESRI_API_KEY` | Yes* | ESRI ArcGIS API key |

*At least one imagery provider key required

## Troubleshooting

### Permission Denied (Linux/Mac)

```bash
# Add current user to docker group
sudo usermod -aG docker $USER
newgrp docker
```

### Volume Mount Issues (Windows)

```powershell
# Use absolute paths
docker run --rm `
  -v C:/Users/YourName/project/data:/app/input:ro `
  -v C:/Users/YourName/project/output:/app/output `
  ecoinnovators/solar-pv-detection:latest
```

### API Key Not Found

```bash
# Verify environment variable is set
docker run --rm ecoinnovators/solar-pv-detection:latest env | grep API_KEY

# Pass explicitly
docker run --rm -e MAPBOX_API_KEY=sk.xxx... ecoinnovators/solar-pv-detection:latest
```

## Performance Notes

- **CPU Mode:** ~2-5 seconds per location
- **GPU Mode:** Requires nvidia-docker and CUDA-enabled image
- **Memory:** ~2GB minimum, 4GB recommended
- **Storage:** ~3GB for image + model weights

## Health Checks

```bash
# Check container logs
docker logs solar-pv-detector

# Monitor resource usage
docker stats solar-pv-detector

# Verify outputs
docker exec solar-pv-detector ls -la /app/output
```

## Production Deployment

For production use, consider:

1. **Use specific version tags** (not `latest`)
2. **Set resource limits** (`--memory=4g --cpus=2`)
3. **Enable logging** (`--log-driver json-file --log-opt max-size=10m`)
4. **Health monitoring** (integrate with Prometheus/Grafana)
5. **Scale horizontally** (Kubernetes/ECS for parallel processing)

```bash
# Production-ready example
docker run -d \
  --name solar-pv-prod \
  --memory=4g \
  --cpus=2 \
  --restart=unless-stopped \
  --log-driver json-file \
  --log-opt max-size=10m \
  --log-opt max-file=3 \
  -v /data/input:/app/input:ro \
  -v /data/output:/app/output \
  -e MAPBOX_API_KEY=$MAPBOX_API_KEY \
  ecoinnovators/solar-pv-detection:v1.0
```

## Support

- **Issues:** https://github.com/Dharanish-AM/EcoInnovators-Ideathon/issues
- **Documentation:** See `model_card/model_card.md`
- **Contact:** [Your contact information]
