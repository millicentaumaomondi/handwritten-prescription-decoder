import os
import sys
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
from PIL import Image
import io
import gdown

# Add the project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from frontend.inference.model_loader import load_all_models
from frontend.inference.predict import predict_image

def download_model_weights():
    """Download the model weights file from Google Drive."""
    weights_dir = os.path.join(project_root, "frontend", "weights")
    os.makedirs(weights_dir, exist_ok=True)
    
    model_path = os.path.join(weights_dir, "best_model_weights.pth")
    if not os.path.exists(model_path):
        print("Downloading model weights...")
        file_id = "1nWWGSed3cJv5H6UKewvAkLyKcTawtBzs"
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, model_path, quiet=False)
        print("Model weights downloaded successfully")

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
    download_model_weights()  # Download model weights if needed
    model, tokenizer, gpt_lm, ngram_lm, vocab, idx_to_char = load_all_models(
        model_path=os.path.join(project_root, "frontend", "weights", "best_model_weights.pth"),
        ngram_path=os.path.join(project_root, "frontend", "weights", "ngram.pkl"),
        label_file=os.path.join(project_root, "frontend", "data", "training_labels.csv")
    )
    print("Models loaded successfully")
except Exception as e:
    print(f"Error loading models: {str(e)}")
    model = None
    tokenizer = None
    gpt_lm = None
    ngram_lm = None
    vocab = None
    idx_to_char = None

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
        prediction = predict_image(
            image=image,
            model=model,
            idx_to_char=idx_to_char,
            ngram_lm=ngram_lm,
            gpt_lm=gpt_lm,
            tokenizer=tokenizer
        )
        
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 