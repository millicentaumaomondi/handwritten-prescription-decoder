from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io
import os
import sys

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frontend.inference.model_loader import load_all_models
from frontend.inference.predict import predict_image

app = FastAPI(
    title="RxVision API",
    description="API for handwritten prescription decoding",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your Streamlit app URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
def load_models():
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, "frontend", "weights", "best_model_weights.pth")
        ngram_path = os.path.join(base_dir, "frontend", "weights", "ngram.pkl")
        label_file = os.path.join(base_dir, "frontend", "data", "training_labels.csv")

        # Check if files exist
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        if not os.path.exists(label_file):
            raise FileNotFoundError(f"Labels file not found at {label_file}")

        return load_all_models(
            model_path=model_path,
            ngram_path=ngram_path,
            label_file=label_file
        )
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        raise

# Initialize models
try:
    model, tokenizer, gpt_lm, ngram_lm, vocab, idx_to_char = load_models()
except Exception as e:
    print(f"Failed to initialize models: {str(e)}")
    raise

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Read and process the image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Make prediction
        prediction = predict_image(image, model, idx_to_char, ngram_lm, gpt_lm, tokenizer)
        
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "version": "1.0.0"
    }

# Add root endpoint
@app.get("/")
async def root():
    return {
        "message": "Welcome to RxVision API",
        "docs_url": "/docs",
        "health_check": "/health"
    } 