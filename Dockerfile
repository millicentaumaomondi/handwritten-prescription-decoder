FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p frontend/weights frontend/data

# Copy application code
COPY backend/ backend/
COPY frontend/app.py frontend/
COPY frontend/utils.py frontend/
COPY frontend/inference/ frontend/inference/
COPY frontend/data/training_labels.csv frontend/data/

# Download model weights
COPY download_weights.py .
RUN python download_weights.py

# Set environment variables
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"] 