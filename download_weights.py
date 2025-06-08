import gdown
import os

def download_weights():
    # Create weights directory if it doesn't exist
    os.makedirs("frontend/weights", exist_ok=True)
    
    # Download the model weights
    gdown.download(
        "https://drive.google.com/uc?id=1nWWGSed3cJv5H6UKewvAkLyKcTawtBzs",
        "frontend/weights/best_model_weights.pth",
        quiet=False
    )
    print("Model weights downloaded successfully!")

if __name__ == "__main__":
    download_weights() 