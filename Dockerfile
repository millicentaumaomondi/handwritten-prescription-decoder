FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create directories for weights and data
RUN mkdir -p frontend/weights frontend/data

# Copy only necessary files first
COPY backend/ backend/
COPY frontend/app.py frontend/
COPY frontend/utils.py frontend/
COPY frontend/inference/ frontend/inference/
COPY frontend/data/training_labels.csv frontend/data/
COPY download_weights.py .

# Download model weights during build
RUN python download_weights.py

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"] 