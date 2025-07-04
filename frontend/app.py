# RxVision: Handwritten Prescription Decoder
# Note: Backend API implements rate limiting (3 requests/minute for predictions)

import streamlit as st
import os
import io
import requests
from PIL import Image
from utils import export_history_as_csv, export_history_as_pdf
import time

# Set Streamlit page config FIRST - Use wide layout for better screen utilization
st.set_page_config(
    page_title="📋 RxVision: Handwritten Prescription Decoder 💊", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better typography and layout
st.markdown("""
<style>
    /* Main title styling */
    .main-title {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        color: #1f77b4 !important;
        text-align: center !important;
        margin-bottom: 2rem !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1) !important;
    }
    
    /* Subtitle styling */
    .subtitle {
        font-size: 1.3rem !important;
        font-weight: 600 !important;
        color: #2c3e50 !important;
        margin-bottom: 1.5rem !important;
    }
    
    /* Welcome message styling - darker and bolder */
    .welcome-text {
        font-size: 1.1rem !important;
        line-height: 1.6 !important;
        color: #1f77b4 !important;
        font-weight: 700 !important;
        margin-bottom: 1.5rem !important;
    }
    
    /* Instructions styling - better color blending with blue theme */
    .instructions {
        font-size: 1rem !important;
        line-height: 1.7 !important;
        color: #ffffff !important;
        background: linear-gradient(135deg, #1f77b4 0%, #2c5aa0 100%) !important;
        padding: 1.2rem !important;
        border-radius: 8px !important;
        border-left: 4px solid #ffffff !important;
        margin-bottom: 1.5rem !important;
        box-shadow: 0 4px 8px rgba(31, 119, 180, 0.3) !important;
    }
    
    /* Container to limit width and center content */
    .main-container {
        max-width: 1200px !important;
        margin: 0 auto !important;
        padding: 0 2rem !important;
    }
    
    /* Card styling for better visual separation */
    .card {
        background-color: white !important;
        padding: 1.2rem !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
        margin-bottom: 1rem !important;
        border: 1px solid #e9ecef !important;
    }
    
    /* Button styling */
    .stButton > button {
        font-size: 1rem !important;
        padding: 0.5rem 1.5rem !important;
        border-radius: 20px !important;
        font-weight: 600 !important;
    }
    
    /* Specific styling for Predict Medicine button */
    .stButton > button:contains("🔍 Predict Medicine"),
    .stButton > button {
        background-color: #28a745 !important;
        border-color: #28a745 !important;
        color: white !important;
    }
    
    .stButton > button:hover {
        background-color: #218838 !important;
        border-color: #1e7e34 !important;
    }
    
    /* Override any existing button styles */
    .stButton > button[style*="background"] {
        background-color: #28a745 !important;
        border-color: #28a745 !important;
        color: white !important;
    }
    
    /* Tab styling - fix text visibility */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 45px !important;
        white-space: pre-wrap !important;
        background-color: #f8f9fa !important;
        border-radius: 6px 6px 0px 0px !important;
        gap: 0.5rem !important;
        padding-top: 8px !important;
        padding-bottom: 8px !important;
        color: #2c3e50 !important;
        font-weight: 600 !important;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4 !important;
        color: white !important;
    }
    
    .stTabs [aria-selected="false"] {
        color: #2c3e50 !important;
        background-color: #e9ecef !important;
    }
    
    /* Success message styling */
    .success-msg {
        font-size: 1.2rem !important;
        font-weight: 600 !important;
        color: #28a745 !important;
        background-color: #d4edda !important;
        padding: 0.8rem !important;
        border-radius: 6px !important;
        border: 1px solid #c3e6cb !important;
    }
    
    /* History item styling */
    .history-item {
        font-size: 1rem !important;
        padding: 0.8rem !important;
        background-color: #ffffff !important;
        border-radius: 6px !important;
        margin-bottom: 0.5rem !important;
        border-left: 3px solid #1f77b4 !important;
        color: #2c3e50 !important;
        border: 1px solid #e9ecef !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
    }
    
    /* Responsive text sizing */
    @media (max-width: 768px) {
        .main-title {
            font-size: 2rem !important;
        }
        .subtitle {
            font-size: 1.1rem !important;
        }
        .welcome-text {
            font-size: 1rem !important;
        }
        .main-container {
            padding: 0 1rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# API endpoint
API_URL = "https://handwriting-decoder.fly.dev"  # Deployed backend

def check_backend_health():
    """Check if the backend is healthy and ready"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get('models_loaded', False)
        return False
    except:
        return False

# --- Login page ---
def login():
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.markdown('<h1 class="main-title">📋 RxVision: Handwritten Prescription Decoder 💊</h1>', unsafe_allow_html=True)
    
    # Welcome message
    st.markdown('<p class="welcome-text">👋 Welcome to RxVision! Your AI-powered assistant for decoding handwritten prescriptions.</p>', unsafe_allow_html=True)
    
    # User guide
    st.markdown('<div class="instructions"><h4>📝 How to use RxVision:</h4><ol><li><strong>Upload Image:</strong> Upload a prescription image or choose from our samples</li><li><strong>Get Prediction:</strong> Our AI will decode the handwritten text</li><li><strong>Download History:</strong> Access and download your decoded prescriptions anytime</li><li><strong>Sign Out:</strong> Securely log out when you\'re done</li></ol></div>', unsafe_allow_html=True)
    
    # Display demo credentials
    st.info("""
    **Demo Credentials:**
    - Username: `admin`
    - Password: `password`
    """)
    
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "password":
            st.session_state.logged_in = True
            st.success("✅ Login successful")
            st.rerun()
        else:
            st.error("❌ Invalid username or password")
    st.markdown('</div>', unsafe_allow_html=True)

# --- Dashboard page ---
def dashboard():
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.markdown('<h1 class="main-title">📋 RxVision: Handwritten Prescription Decoder 💊</h1>', unsafe_allow_html=True)

    # Initialize history
    if "history" not in st.session_state:
        st.session_state.history = []

    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["📤 Upload Image", "📥 Download", "🚪 Sign Out"])

    with tab1:
        st.subheader("Upload or Select Sample Image")
        col1, col2 = st.columns(2)

        with col1:
            uploaded_file = st.file_uploader("Upload Prescription Image", type=["png", "jpg", "jpeg"])

        with col2:
            # Get the path to the samples directory
            base_dir = os.path.dirname(os.path.abspath(__file__))
            samples_dir = os.path.join(base_dir, "assets", "samples")
            
            # Get list of sample images if the directory exists
            sample_images = []
            if os.path.exists(samples_dir):
                sample_images = [f for f in os.listdir(samples_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            sample_choice = st.selectbox("Or choose a sample", [""] + sample_images)
            if sample_choice:
                sample_path = os.path.join(samples_dir, sample_choice)
                if os.path.exists(sample_path):
                    uploaded_file = open(sample_path, "rb")
                    display_filename = sample_choice
                else:
                    st.error(f"Sample image not found: {sample_path}")

        if uploaded_file:
            try:
                # Handle both BytesIO (uploaded files) and regular files
                if isinstance(uploaded_file, bytes):
                    image = Image.open(io.BytesIO(uploaded_file))
                else:
                    image = Image.open(uploaded_file)
                
                # Display image with better aspect ratio control
                st.image(image, caption="Selected Image", width=400)

                # Custom green prediction button
                st.markdown("""
                <style>
                .green-button {
                    background-color: #28a745 !important;
                    border-color: #28a745 !important;
                    color: white !important;
                    font-size: 1rem !important;
                    padding: 0.5rem 1.5rem !important;
                    border-radius: 20px !important;
                    font-weight: 600 !important;
                    border: none !important;
                    cursor: pointer !important;
                    width: 100% !important;
                }
                .green-button:hover {
                    background-color: #218838 !important;
                    border-color: #1e7e34 !important;
                }
                </style>
                """, unsafe_allow_html=True)
                
                # Use custom HTML button
                if st.button("🔍 Predict Medicine", use_container_width=True):
                    # Check backend health first
                    if not check_backend_health():
                        st.warning("⚠️ Backend is starting up. This may take 30-60 seconds on first request. Please wait...")
                        with st.spinner("🔄 Waiting for backend to be ready..."):
                            # Wait for backend to be ready
                            for i in range(30):  # Wait up to 30 seconds
                                if check_backend_health():
                                    st.success("✅ Backend is ready!")
                                    break
                                time.sleep(1)
                            else:
                                st.error("❌ Backend is taking too long to start. Please try again in a few minutes.")
                                st.stop()
                    
                    with st.spinner("🤖 AI is analyzing your prescription..."):
                        # Prepare the image for API
                        img_byte_arr = io.BytesIO()
                        image.save(img_byte_arr, format='PNG')
                        img_byte_arr = img_byte_arr.getvalue()

                        # Send to API with retry logic
                        max_retries = 3
                        for attempt in range(max_retries):
                            try:
                                files = {'file': ('image.png', img_byte_arr, 'image/png')}
                                response = requests.post(f"{API_URL}/predict", files=files, timeout=30)
                                
                                if response.status_code == 200:
                                    prediction = response.json()['prediction']
                                    st.markdown(f'<div class="success-msg">🧾 Predicted Medicine: <strong>{prediction}</strong></div>', unsafe_allow_html=True)
                                    # Use the stored display filename
                                    filename = display_filename if 'display_filename' in locals() else os.path.basename(uploaded_file.name)
                                    st.session_state.history.append({
                                        "filename": filename,
                                        "prediction": prediction
                                    })
                                    break  # Success, exit retry loop
                                else:
                                    if attempt < max_retries - 1:
                                        st.warning(f"Attempt {attempt + 1} failed. Retrying... (Backend might be starting up)")
                                        time.sleep(2)  # Wait 2 seconds before retry
                                    else:
                                        st.error(f"Error from API after {max_retries} attempts: {response.text}")
                            except requests.exceptions.Timeout:
                                if attempt < max_retries - 1:
                                    st.warning(f"Request timed out (attempt {attempt + 1}). Retrying... (Backend is starting up)")
                                    time.sleep(3)  # Wait 3 seconds before retry
                                else:
                                    st.error("Request timed out after multiple attempts. The backend might be starting up. Please try again in a few seconds.")
                            except requests.exceptions.ConnectionError:
                                if attempt < max_retries - 1:
                                    st.warning(f"Connection error (attempt {attempt + 1}). Retrying... (Backend is starting up)")
                                    time.sleep(3)
                                else:
                                    st.error("Cannot connect to backend. The service might be starting up. Please try again in a few seconds.")
                            except Exception as e:
                                st.error(f"Unexpected error: {str(e)}")
                                break
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")

    with tab2:
        st.subheader("Your Prescription Journey 📋")
        st.write("✨ View and download your decoded prescriptions history below")
        if st.session_state.history:
            for record in st.session_state.history:
                st.markdown(f'<div class="history-item"><strong>{record["filename"]}</strong> → <code>{record["prediction"]}</code></div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                export_history_as_csv(st.session_state.history)
            with col2:
                export_history_as_pdf(st.session_state.history)
        else:
            st.info("No predictions made yet. Upload an image to get started! 🚀")

    with tab3:
        st.write("Thank you for using RxVision Decoder 😊")
        if st.button("🚪 Sign Out"):
            st.session_state.logged_in = False
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# --- App Logic ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if st.session_state.logged_in:
    dashboard()
else:
    login()