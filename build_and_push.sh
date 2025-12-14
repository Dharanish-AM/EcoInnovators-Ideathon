#!/bin/bash
# Build and Push Docker Images Script (Linux/Mac)
# Run this to prepare Docker images for submission

set -e

echo "========================================"
echo "Building EcoInnovators Docker Images"
echo "========================================"

# Get DockerHub username
read -p "Enter your DockerHub username: " DOCKER_USERNAME

if [ -z "$DOCKER_USERNAME" ]; then
    echo "Error: DockerHub username is required!"
    exit 1
fi

echo ""
echo "Building inference pipeline image..."
docker build -t ${DOCKER_USERNAME}/solar-pv-detection:latest .
docker build -t ${DOCKER_USERNAME}/solar-pv-detection:v1.0 .

echo ""
echo "Building Streamlit UI image..."
docker build -f Dockerfile.streamlit -t ${DOCKER_USERNAME}/solar-pv-detection:streamlit .
docker build -f Dockerfile.streamlit -t ${DOCKER_USERNAME}/solar-pv-detection:streamlit-v1.0 .

echo ""
echo "========================================"
echo "Build Complete!"
echo "========================================"

echo ""
echo "Built images:"
docker images ${DOCKER_USERNAME}/solar-pv-detection

# Ask to push
read -p "Do you want to push images to DockerHub? (y/n): " PUSH

if [ "$PUSH" = "y" ] || [ "$PUSH" = "Y" ]; then
    echo ""
    echo "Logging into DockerHub..."
    docker login
    
    echo ""
    echo "Pushing images to DockerHub..."
    
    docker push ${DOCKER_USERNAME}/solar-pv-detection:latest
    docker push ${DOCKER_USERNAME}/solar-pv-detection:v1.0
    docker push ${DOCKER_USERNAME}/solar-pv-detection:streamlit
    docker push ${DOCKER_USERNAME}/solar-pv-detection:streamlit-v1.0
    
    echo ""
    echo "========================================"
    echo "Successfully pushed to DockerHub!"
    echo "========================================"
    
    echo ""
    echo "Your images are available at:"
    echo "  - https://hub.docker.com/r/${DOCKER_USERNAME}/solar-pv-detection"
    
    echo ""
    echo "Update SUBMISSION.md with your DockerHub username:"
    echo "  Replace '<username>' with '${DOCKER_USERNAME}'"
fi

echo ""
echo "========================================"
echo "Next Steps:"
echo "========================================"
echo "1. Update SUBMISSION.md with your DockerHub username"
echo "2. Test the Docker image locally"
echo "3. Generate sample predictions"
echo "4. Push code to GitHub"
echo "5. Submit the prototype!"
echo ""
