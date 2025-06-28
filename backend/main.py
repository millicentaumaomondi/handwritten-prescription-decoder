import os
import sys
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
from PIL import Image
import io
import gdown
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Add the backend directory to Python path
backend_dir = str(Path(__file__).parent)
sys.path.append(backend_dir)

from inference.model_loader import load_all_models
from inference.predict import predict_image

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

def download_model_weights():
    """Download the model weights file from Google Drive."""
    weights_dir = os.path.join(backend_dir, "weights")
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

# Add rate limiter to app
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

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
        model_path=os.path.join(backend_dir, "weights", "best_model_weights.pth"),
        ngram_path=os.path.join(backend_dir, "weights", "ngram.pkl"),
        label_file=os.path.join(backend_dir, "data", "training_labels.csv")
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
@limiter.limit("5/minute")  # 5 requests per minute
async def root(request: Request):
    return {"message": "Welcome to RxVision API"}

@app.get("/health")
@limiter.limit("10/minute")  # 10 requests per minute
async def health_check(request: Request):
    if model is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    return {"status": "healthy", "models_loaded": True}

@app.post("/predict")
@limiter.limit("3/minute")  # 3 requests per minute
async def predict(request: Request, file: UploadFile = File(...)):
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
            vocab=vocab,
            ngram_lm=ngram_lm,
            gpt_lm=gpt_lm,
            tokenizer=tokenizer
        )
        
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 