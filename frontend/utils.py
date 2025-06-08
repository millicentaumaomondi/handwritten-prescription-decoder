import pandas as pd
import streamlit as st
from fpdf import FPDF
import io
import tempfile
import os

def export_history_as_csv(history_records):
    df = pd.DataFrame(history_records)
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download CSV", csv, "predictions.csv", "text/csv")

def export_history_as_pdf(history_records):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Handwriting Recognition Predictions", ln=True, align="C")
    pdf.ln(10)

    for record in history_records:
        pdf.cell(200, 10, txt=f"{record['filename']} -> {record['prediction']}", ln=True)

    # Create a temporary file to store the PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
        pdf.output(tmp.name)
        with open(tmp.name, 'rb') as f:
            pdf_bytes = f.read()
        os.unlink(tmp.name)  # Delete the temporary file
    
    st.download_button("📥 Download PDF", data=pdf_bytes, file_name="predictions.pdf", mime="application/pdf")
