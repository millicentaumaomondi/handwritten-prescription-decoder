# RxVision: Handwritten Prescription Decoder

A deep learning-powered application that decodes handwritten prescriptions and identifies medicine names, helping to reduce medication errors and improve healthcare efficiency.

## Overview

RxVision is a full-stack application that combines computer vision and natural language processing to decode handwritten prescriptions. It addresses a critical challenge in healthcare: the interpretation of handwritten medical prescriptions, which can be prone to errors and misinterpretation.

### Why RxVision?

- **Reduces Medication Errors**: By accurately decoding handwritten prescriptions, RxVision helps prevent medication errors that could harm patients.
- **Improves Efficiency**: Automates the time-consuming process of manually interpreting handwritten prescriptions.
- **Enhances Accessibility**: Makes prescription information more accessible and readable for patients and healthcare providers.
- **Educational Value**: Serves as a learning tool for understanding the intersection of healthcare and AI.

##  Architecture

The application consists of two main components:

### Backend (FastAPI)
- Hosted on Fly.io
- Handles model inference and prediction
- Provides RESTful API endpoints
- Manages model loading and image processing

### Frontend (Streamlit)
- Hosted on Streamlit Cloud
- Provides user-friendly interface
- Handles image upload and display
- Manages user sessions and history

##  Models Used

### 1. CRNN with EfficientNet-B4 Backbone
- **Purpose**: Handwritten text recognition
- **Architecture**:
  - EfficientNet-B4 for feature extraction
  - Bidirectional LSTM for sequence modeling
  - CTC (Connectionist Temporal Classification) for text decoding
- **Features**:
  - Positional encoding for sequence awareness
  - CBAM (Convolutional Block Attention Module) for better feature focus
  - Dropout for regularization

### 2. Language Models
- **N-gram Language Model**
  - Trained on medicine names corpus
  - Helps correct and validate predictions
  - Improves accuracy for common medicine names

- **GPT-2 Model**
  - Provides contextual understanding
  - Helps with spelling and grammar correction
  - Enhances prediction quality

##  Deployment Guide

### Backend Deployment (Fly.io)

1. **Prerequisites**
   ```bash
   # Install Fly.io CLI
   curl -L https://fly.io/install.sh | sh
   
   # Login to Fly.io
   flyctl auth login
   ```

2. **Project Structure**
   ```
   backend/
   ├── main.py           # FastAPI application
   ├── requirements.txt  # Python dependencies
   └── Dockerfile       # Container configuration
   ```

3. **Deployment Steps**
   ```bash
   # Initialize Fly.io app
   flyctl launch
   
   # Deploy the application
   flyctl deploy
   ```

### Frontend Deployment (Streamlit)

1. **Prerequisites**
   - GitHub account
   - Streamlit Cloud account

2. **Project Structure**
   ```
   frontend/
   ├── app.py           # Streamlit application
   ├── requirements.txt # Python dependencies
   ├── Procfile        # Streamlit configuration
   └── setup.py        # Package configuration
   ```

3. **Deployment Steps**
   - Push code to GitHub repository
   - Go to [share.streamlit.io](https://share.streamlit.io/)
   - Create new app
   - Select repository and main file (`frontend/app.py`)
   - Configure settings:
     - Python version: 3.9
     - Working directory: `frontend`
     - Environment variables:
       - `PYTHONPATH=/app/frontend`

## 💻 Local Development

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/handwritten-prescription-decoder.git
   cd handwritten-prescription-decoder
   ```

2. **Set Up Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   # Backend
   pip install -r requirements.txt
   
   # Frontend
   cd frontend
   pip install -r requirements.txt
   ```

4. **Run Locally**
   ```bash
   # Backend
   uvicorn backend.main:app --reload
   
   # Frontend
   streamlit run frontend/app.py
   ```

##  Model Training

The models are trained on a dataset of handwritten prescriptions with the following characteristics:
- Character-level vocabulary
- Medicine name annotations
- Various handwriting styles
- Multiple samples per medicine

Training process:
1. Data preprocessing and augmentation
2. Model training with CTC loss
3. Language model integration
4. Validation and testing

##  Security

- User authentication for frontend access
- API rate limiting
- Secure file handling
- Input validation
- Error handling

##  License

This project is licensed under the MIT License - see the LICENSE file for details.

##  Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

##  Contact

For questions or support, please open an issue in the GitHub repository.

## API Usage Guidelines

### Rate Limits
The API implements rate limiting to ensure fair usage and system stability:

| Endpoint | Rate Limit | Purpose |
|----------|------------|----------|
| `/` | 5 requests/minute | General API information |
| `/health` | 10 requests/minute | System health monitoring |
| `/predict` | 3 requests/minute | Prescription decoding |

When rate limits are exceeded, the API will return a `429 Too Many Requests` response:
```json
{
    "detail": "Rate limit exceeded: X per 1 minute"
}
```

### Best Practices
1. Implement exponential backoff when receiving 429 responses
2. Cache health check results locally
3. Batch predictions when possible
4. Monitor your API usage to stay within limits 
