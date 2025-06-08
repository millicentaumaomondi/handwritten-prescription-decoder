import streamlit as st
import os
import io
import requests
from PIL import Image
from utils import export_history_as_csv, export_history_as_pdf

# Set Streamlit page config FIRST
st.set_page_config(page_title="📋 RxVision: Handwritten Prescription Decoder 💊", layout="centered")

# API endpoint
API_URL = "https://handwriting-decoder.fly.dev"  # Updated to use Fly.io backend

# --- Login page ---
def login():
    st.title("📋 RxVision: Handwritten Prescription Decoder 💊")
    
    # Welcome message
    st.markdown("""
    ### 👋 Welcome to RxVision!
    Your AI-powered assistant for decoding handwritten prescriptions.
    """)
    
    # User guide
    st.markdown("""
    #### 📝 How to use RxVision:
    1. **Upload Image**: Upload a prescription image or choose from our samples
    2. **Get Prediction**: Our AI will decode the handwritten text
    3. **Download History**: Access and download your decoded prescriptions anytime
    4. **Sign Out**: Securely log out when you're done
    """)
    
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

# --- Dashboard page ---
def dashboard():
    st.title("📋 RxVision: Handwritten Prescription Decoder 💊")

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
                
                st.image(image, caption="Selected Image", use_container_width=True)

                if st.button("🔍 Predict"):
                    with st.spinner("Analyzing..."):
                        # Prepare the image for API
                        img_byte_arr = io.BytesIO()
                        image.save(img_byte_arr, format='PNG')
                        img_byte_arr = img_byte_arr.getvalue()

                        # Send to API
                        files = {'file': ('image.png', img_byte_arr, 'image/png')}
                        response = requests.post(f"{API_URL}/predict", files=files)
                        
                        if response.status_code == 200:
                            prediction = response.json()['prediction']
                            st.success(f"🧾 Predicted Medicine: **{prediction}**")
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

    with tab2:
        st.subheader("Your Prescription Journey 📋")
        st.write("✨ View and download your decoded prescriptions history below")
        if st.session_state.history:
            for record in st.session_state.history:
                st.markdown(f"• **{record['filename']}** → `{record['prediction']}`")
            
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

# --- App Logic ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if st.session_state.logged_in:
    dashboard()
else:
    login()