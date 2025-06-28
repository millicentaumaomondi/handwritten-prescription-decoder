# RxVision: Handwritten Prescription Decoder
# Note: Backend API implements rate limiting (3 requests/minute for predictions)

import streamlit as st
import os
import io
import requests
from PIL import Image
from utils import export_history_as_csv, export_history_as_pdf

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
        font-size: 3rem !important;
        font-weight: 700 !important;
        color: #1f77b4 !important;
        text-align: center !important;
        margin-bottom: 2rem !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1) !important;
    }
    
    /* Subtitle styling */
    .subtitle {
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        color: #2c3e50 !important;
        margin-bottom: 1.5rem !important;
    }
    
    /* Welcome message styling */
    .welcome-text {
        font-size: 1.2rem !important;
        line-height: 1.6 !important;
        color: #34495e !important;
        margin-bottom: 2rem !important;
    }
    
    /* Instructions styling */
    .instructions {
        font-size: 1.1rem !important;
        line-height: 1.8 !important;
        color: #555 !important;
        background-color: #f8f9fa !important;
        padding: 1.5rem !important;
        border-radius: 10px !important;
        border-left: 4px solid #1f77b4 !important;
        margin-bottom: 2rem !important;
    }
    
    /* Card styling for better visual separation */
    .card {
        background-color: white !important;
        padding: 1.5rem !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
        margin-bottom: 1.5rem !important;
        border: 1px solid #e9ecef !important;
    }
    
    /* Button styling */
    .stButton > button {
        font-size: 1.1rem !important;
        padding: 0.75rem 2rem !important;
        border-radius: 25px !important;
        font-weight: 600 !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px !important;
        white-space: pre-wrap !important;
        background-color: #f8f9fa !important;
        border-radius: 8px 8px 0px 0px !important;
        gap: 1rem !important;
        padding-top: 10px !important;
        padding-bottom: 10px !important;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4 !important;
        color: white !important;
    }
    
    /* File uploader styling */
    .stFileUploader {
        border: 2px dashed #1f77b4 !important;
        border-radius: 10px !important;
        padding: 2rem !important;
    }
    
    /* Success message styling */
    .success-msg {
        font-size: 1.3rem !important;
        font-weight: 600 !important;
        color: #28a745 !important;
        background-color: #d4edda !important;
        padding: 1rem !important;
        border-radius: 8px !important;
        border: 1px solid #c3e6cb !important;
    }
    
    /* History item styling */
    .history-item {
        font-size: 1.1rem !important;
        padding: 1rem !important;
        background-color: #f8f9fa !important;
        border-radius: 8px !important;
        margin-bottom: 0.5rem !important;
        border-left: 4px solid #1f77b4 !important;
    }
    
    /* Responsive text sizing */
    @media (max-width: 768px) {
        .main-title {
            font-size: 2rem !important;
        }
        .subtitle {
            font-size: 1.2rem !important;
        }
        .welcome-text {
            font-size: 1rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# API endpoint
API_URL = "https://handwriting-decoder.fly.dev"  # Deployed backend

# --- Login page ---
def login():
    st.markdown('<h1 class="main-title">📋 RxVision: Handwritten Prescription Decoder 💊</h1>', unsafe_allow_html=True)
    
    # Create two columns for better layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<p class="welcome-text">👋 Welcome to RxVision! Your AI-powered assistant for decoding handwritten prescriptions with advanced machine learning technology.</p>', unsafe_allow_html=True)
        
        st.markdown('<div class="instructions"><h4>📝 How to use RxVision:</h4><ol><li><strong>Upload Image:</strong> Upload a prescription image or choose from our samples</li><li><strong>Get Prediction:</strong> Our AI will decode the handwritten text</li><li><strong>Download History:</strong> Access and download your decoded prescriptions anytime</li><li><strong>Sign Out:</strong> Securely log out when you\'re done</li></ol></div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h4 class="subtitle">🔐 Login</h4>', unsafe_allow_html=True)
        
        # Display demo credentials
        st.info("""
        **Demo Credentials:**
        - Username: `admin`
        - Password: `password`
        """)
        
        username = st.text_input("Username", placeholder="Enter username")
        password = st.text_input("Password", type="password", placeholder="Enter password")

        if st.button("🚀 Login", use_container_width=True):
            if username == "admin" and password == "password":
                st.session_state.logged_in = True
                st.success("✅ Login successful")
                st.rerun()
            else:
                st.error("❌ Invalid username or password")
        st.markdown('</div>', unsafe_allow_html=True)

# --- Dashboard page ---
def dashboard():
    st.markdown('<h1 class="main-title">📋 RxVision: Handwritten Prescription Decoder 💊</h1>', unsafe_allow_html=True)

    # Initialize history
    if "history" not in st.session_state:
        st.session_state.history = []

    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["📤 Upload Image", "📥 Download History", "🚪 Sign Out"])

    with tab1:
        st.markdown('<h2 class="subtitle">Upload or Select Sample Image</h2>', unsafe_allow_html=True)
        
        # Use three columns for better layout
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<h4>📁 Upload Your Image</h4>', unsafe_allow_html=True)
            uploaded_file = st.file_uploader("Choose a prescription image", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<h4>🎯 Try Samples</h4>', unsafe_allow_html=True)
            # Get the path to the samples directory
            base_dir = os.path.dirname(os.path.abspath(__file__))
            samples_dir = os.path.join(base_dir, "assets", "samples")
            
            # Get list of sample images if the directory exists
            sample_images = []
            if os.path.exists(samples_dir):
                sample_images = [f for f in os.listdir(samples_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            sample_choice = st.selectbox("Select a sample image", [""] + sample_images, label_visibility="collapsed")
            if sample_choice:
                sample_path = os.path.join(samples_dir, sample_choice)
                if os.path.exists(sample_path):
                    uploaded_file = open(sample_path, "rb")
                    display_filename = sample_choice
                else:
                    st.error(f"Sample image not found: {sample_path}")
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<h4>⚡ Quick Actions</h4>', unsafe_allow_html=True)
            if uploaded_file:
                if st.button("🔍 Analyze Prescription", use_container_width=True, type="primary"):
                    # Prediction logic will be handled below
                    pass
            else:
                st.info("Upload an image or select a sample to get started!")
            st.markdown('</div>', unsafe_allow_html=True)

        # Display image and handle prediction
        if uploaded_file:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<h4>📸 Selected Image</h4>', unsafe_allow_html=True)
            
            try:
                # Handle both BytesIO (uploaded files) and regular files
                if isinstance(uploaded_file, bytes):
                    image = Image.open(io.BytesIO(uploaded_file))
                else:
                    image = Image.open(uploaded_file)
                
                # Display image in a larger format
                st.image(image, caption="Prescription Image", use_container_width=True)

                # Prediction button
                if st.button("🔍 Predict Medicine", use_container_width=True, type="primary"):
                    with st.spinner("🤖 AI is analyzing your prescription..."):
                        # Prepare the image for API
                        img_byte_arr = io.BytesIO()
                        image.save(img_byte_arr, format='PNG')
                        img_byte_arr = img_byte_arr.getvalue()

                        # Send to API
                        files = {'file': ('image.png', img_byte_arr, 'image/png')}
                        response = requests.post(f"{API_URL}/predict", files=files)
                        
                        if response.status_code == 200:
                            prediction = response.json()['prediction']
                            st.markdown(f'<div class="success-msg">🧾 Predicted Medicine: <strong>{prediction}</strong></div>', unsafe_allow_html=True)
                            # Use the stored display filename
                            filename = display_filename if 'display_filename' in locals() else os.path.basename(uploaded_file.name)
                            st.session_state.history.append({
                                "filename": filename,
                                "prediction": prediction
                            })
                        else:
                            st.error(f"Error from API: {response.text}")
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
            st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<h2 class="subtitle">Your Prescription Journey 📋</h2>', unsafe_allow_html=True)
        st.markdown('<p class="welcome-text">✨ View and download your decoded prescriptions history below</p>', unsafe_allow_html=True)
        
        if st.session_state.history:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            for i, record in enumerate(st.session_state.history, 1):
                st.markdown(f'<div class="history-item"><strong>{i}.</strong> <strong>{record["filename"]}</strong> → <code>{record["prediction"]}</code></div>', unsafe_allow_html=True)
            
            st.markdown('<h4>📥 Export Options</h4>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                export_history_as_csv(st.session_state.history)
            with col2:
                export_history_as_pdf(st.session_state.history)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.info("📝 No predictions made yet. Upload an image to get started! 🚀")
            st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h2 class="subtitle">👋 Thank you for using RxVision!</h2>', unsafe_allow_html=True)
        st.markdown('<p class="welcome-text">We hope RxVision helped you decode your prescriptions efficiently. Your data is secure and you can always come back to access your history.</p>', unsafe_allow_html=True)
        
        if st.button("🚪 Sign Out", use_container_width=True, type="secondary"):
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