# RxVision: Handwritten Prescription Decoder

A Streamlit application that uses deep learning to decode handwritten prescriptions and identify medicine names.

## Features

- Upload handwritten prescription images
- Real-time prediction of medicine names
- Export prediction history as CSV or PDF
- User authentication system
- Sample images for testing

## Setup

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run frontend/app.py
   ```

## Usage

1. Login with the demo credentials:
   - Username: `admin`
   - Password: `password`
2. Upload a handwritten prescription image or select a sample image
3. Click "Predict" to get the medicine name
4. View prediction history and export as needed

## Model Information

The application uses a combination of:
- CNN for image processing
- N-gram language model for text prediction
- GPT-based language model for context understanding

## License

This project is licensed under the MIT License - see the LICENSE file for details. 