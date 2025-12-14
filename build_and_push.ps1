# Build and Push Docker Images Script
# Run this to prepare Docker images for submission

# Exit on error
$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Building EcoInnovators Docker Images" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Get DockerHub username
$DOCKER_USERNAME = Read-Host "Enter your DockerHub username"

if ([string]::IsNullOrWhiteSpace($DOCKER_USERNAME)) {
    Write-Host "Error: DockerHub username is required!" -ForegroundColor Red
    exit 1
}

Write-Host "`nBuilding inference pipeline image..." -ForegroundColor Yellow
docker build -t ${DOCKER_USERNAME}/solar-pv-detection:latest .
docker build -t ${DOCKER_USERNAME}/solar-pv-detection:v1.0 .

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Failed to build inference image" -ForegroundColor Red
    exit 1
}

Write-Host "`nBuilding Streamlit UI image..." -ForegroundColor Yellow
docker build -f Dockerfile.streamlit -t ${DOCKER_USERNAME}/solar-pv-detection:streamlit .
docker build -f Dockerfile.streamlit -t ${DOCKER_USERNAME}/solar-pv-detection:streamlit-v1.0 .

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Failed to build UI image" -ForegroundColor Red
    exit 1
}

Write-Host "`n========================================" -ForegroundColor Green
Write-Host "Build Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

Write-Host "`nBuilt images:" -ForegroundColor Cyan
docker images ${DOCKER_USERNAME}/solar-pv-detection

# Ask to push
$PUSH = Read-Host "`nDo you want to push images to DockerHub? (y/n)"

if ($PUSH -eq "y" -or $PUSH -eq "Y") {
    Write-Host "`nLogging into DockerHub..." -ForegroundColor Yellow
    docker login
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error: Docker login failed" -ForegroundColor Red
        exit 1
    }
    
    Write-Host "`nPushing images to DockerHub..." -ForegroundColor Yellow
    
    docker push ${DOCKER_USERNAME}/solar-pv-detection:latest
    docker push ${DOCKER_USERNAME}/solar-pv-detection:v1.0
    docker push ${DOCKER_USERNAME}/solar-pv-detection:streamlit
    docker push ${DOCKER_USERNAME}/solar-pv-detection:streamlit-v1.0
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error: Failed to push images" -ForegroundColor Red
        exit 1
    }
    
    Write-Host "`n========================================" -ForegroundColor Green
    Write-Host "Successfully pushed to DockerHub!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    
    Write-Host "`nYour images are available at:" -ForegroundColor Cyan
    Write-Host "  - https://hub.docker.com/r/${DOCKER_USERNAME}/solar-pv-detection" -ForegroundColor White
    
    Write-Host "`nUpdate SUBMISSION.md with your DockerHub username:" -ForegroundColor Yellow
    Write-Host "  Replace '<username>' with '${DOCKER_USERNAME}'" -ForegroundColor White
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Next Steps:" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "1. Update SUBMISSION.md with your DockerHub username"
Write-Host "2. Test the Docker image locally"
Write-Host "3. Generate sample predictions"
Write-Host "4. Push code to GitHub"
Write-Host "5. Submit the prototype!"
Write-Host ""
