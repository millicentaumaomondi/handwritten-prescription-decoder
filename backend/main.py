import os
import sys
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
from PIL import Image
import io

# Add the project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from frontend.inference.model_loader import load_models
from frontend.inference.predict import predict_image

app = FastAPI(
    title="RxVision API",
    description="API for handwritten prescription decoding",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize models
try:
    model, vocab, char_to_idx, idx_to_char, ngram_model = load_models()
    print("Models loaded successfully")
except Exception as e:
    print(f"Error loading models: {str(e)}")
    model = None
    vocab = None
    char_to_idx = None
    idx_to_char = None
    ngram_model = None

@app.get("/")
async def root():
    return {"message": "Welcome to RxVision API"}

@app.get("/health")
async def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    return {"status": "healthy", "models_loaded": True}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Read and validate image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Get prediction
        prediction = predict_image(image, model, vocab, char_to_idx, idx_to_char, ngram_model)
        
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 