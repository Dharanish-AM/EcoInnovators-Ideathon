# EcoInnovators Solar PV Detection - Production Docker Image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY environment_details/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY pipeline_code/ ./pipeline_code/
COPY trained_model/ ./trained_model/
COPY model_card/ ./model_card/
COPY LICENSE .
COPY README.md .

# Create output directories
RUN mkdir -p /app/output/overlays /app/input

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Expose port for potential API service
EXPOSE 8000

# Default command runs inference pipeline
# Users can override with custom input file
ENTRYPOINT ["python", "-m", "pipeline_code.run_pipeline"]
CMD ["--input_xlsx", "/app/input/locations.xlsx", \
     "--output_dir", "/app/output", \
     "--zoom", "20"]
