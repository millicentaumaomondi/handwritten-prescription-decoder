# Deployment Guide

## Backend (Fly.io) ✅ DEPLOYED
- **URL**: https://handwriting-decoder.fly.dev/
- **Status**: ✅ Running and healthy
- **Models**: ✅ Loaded successfully

## Frontend (Streamlit Cloud)

### Prerequisites
- GitHub repository with your code
- Streamlit Cloud account

### Deployment Steps

1. **Go to [share.streamlit.io](https://share.streamlit.io)**
2. **Sign in with GitHub**
3. **Click "New app"**
4. **Configure:**
   - Repository: `millicentaumaomondi/handwritten-prescription-decoder`
   - Branch: `main`
   - Main file path: `frontend/app.py`
   - App URL: (leave default or choose custom)
5. **Click "Deploy!"**

### Configuration Files

The following files are already configured for deployment:

- `frontend/requirements.txt` - Frontend dependencies
- `frontend/app.py` - Main Streamlit app (points to deployed backend)
- `backend/main.py` - FastAPI backend
- `Dockerfile` - Backend container configuration
- `fly.toml` - Fly.io configuration

### Testing

1. **Backend Health Check:**
   ```bash
   curl https://handwriting-decoder.fly.dev/health
   ```
   Expected: `{"status":"healthy","models_loaded":true}`

2. **Frontend Test:**
   - Visit your Streamlit Cloud URL
   - Login with: `admin` / `password`
   - Upload a prescription image
   - Verify prediction works

### Troubleshooting

- **Backend not responding**: Check Fly.io logs with `fly logs`
- **Frontend can't connect to backend**: Verify API_URL in `frontend/app.py`
- **Models not loading**: Check if `training_labels.csv` is included in Docker build 