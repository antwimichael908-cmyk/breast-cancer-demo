# app.py - Modern & Catchy Breast Ultrasound Cancer Classifier
# Deployed version with improved UI/UX

import streamlit as st
import pandas as pd
import joblib
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from PIL import Image
import io

# ────────────────────────────────────────────────
# Page config - modern look
# ────────────────────────────────────────────────
st.set_page_config(
    page_title="BreastCare AI – Ultrasound Classifier",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ────────────────────────────────────────────────
# Custom CSS for nicer visuals
# ────────────────────────────────────────────────
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: bold;
    }
    .stButton>button:hover { background-color: #45a049; }
    .result-box {
        padding: 20px;
        border-radius: 12px;
        margin: 20px 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        font-size: 1.3rem;
        text-align: center;
    }
    .success { background-color: #e8f5e9; border-left: 6px solid #4CAF50; }
    .danger  { background-color: #ffebee; border-left: 6px solid #f44336; }
    .info    { background-color: #e3f2fd; border-left: 6px solid #2196f3; }
    h1 { color: #1e3a8a; text-align: center; }
    .disclaimer { font-size: 0.85rem; color: #6b7280; text-align: center; margin-top: 40px; }
    </style>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────
# Load model with error handling
# ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model = joblib.load("best_tuned_rf_breast_ultrasound.pkl")  # ← change if in subfolder
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'best_tuned_rf_breast_ultrasound.pkl' is in the repo root.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

model = load_model()

# ────────────────────────────────────────────────
# Feature extraction function (same as training)
# ────────────────────────────────────────────────
def extract_features(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, (224, 224))
    
    mean_intensity = np.mean(img_resized)
    std_intensity  = np.std(img_resized)
    
    distances = [1, 2, 3]
    angles    = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(img_resized, distances, angles, levels=256, symmetric=True, normed=True)
    
    texture_feats = {}
    for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']:
        vals = graycoprops(glcm, prop)
        for i, d in enumerate(distances):
            for j, a in enumerate(angles):
                texture_feats[f"{prop}_d{d}_a{j:.2f}"] = vals[i, j]
    
    # Shape
    _, thresh = cv2.threshold(img_resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = perimeter = circularity = 0.0
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
    
    features = {
        'mean_intensity': mean_intensity,
        'std_intensity': std_intensity,
        'area': area,
        'perimeter': perimeter,
        'circularity': circularity,
        **texture_feats
    }
    return pd.DataFrame([features])

# ────────────────────────────────────────────────
# Sidebar – Instructions & Info
# ────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/breast-cancer-ribbon.png", width=80)
    st.title("BreastCare AI")
    st.markdown("### Early Detection Tool")
    st.info("""
    Upload a breast ultrasound image (PNG/JPG) to get an AI-assisted prediction:
    - **Benign** (non-cancerous)
    - **Malignant** (cancerous)
    - **Normal**
    
    **Note**: This is a research prototype — always consult a qualified radiologist or doctor for medical decisions.
    """)
    st.markdown("---")
    st.caption("Powered by Random Forest • Trained on BUSI dataset")

# ────────────────────────────────────────────────
# Main content – Catchy header
# ────────────────────────────────────────────────
st.title("🩺 BreastCare AI – Ultrasound Cancer Classifier")
st.markdown("**Upload your breast ultrasound image and get an instant prediction**")

col1, col2 = st.columns([3, 2])

with col1:
    uploaded_file = st.file_uploader(
        "Drag & drop or click to upload ultrasound image...",
        type=["png", "jpg", "jpeg"],
        help="Best results with clear, grayscale or lightly colored ultrasound scans."
    )

if uploaded_file is not None:
    # Read & display image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Show nice preview
    st.image(img, channels="BGR", caption="Uploaded Ultrasound Image", use_column_width=True)
    
    if st.button("Analyze Image → Get Prediction", type="primary", use_container_width=True):
        with st.spinner("Extracting texture & shape features..."):
            try:
                features_df = extract_features(img)
                
                prediction = model.predict(features_df)[0]
                probabilities = model.predict_proba(features_df)[0]
                
                labels = {0: "Benign", 1: "Malignant", 2: "Normal"}
                pred_label = labels[prediction]
                confidence = probabilities[prediction]
                
                # Result box – color coded
                if pred_label == "Malignant":
                    css_class = "danger"
                    emoji = "⚠️"
                    message = "Potential malignancy detected – please consult a specialist urgently."
                elif pred_label == "Benign":
                    css_class = "success"
                    emoji = "✅"
                    message = "Likely benign lesion – routine follow-up recommended."
                else:
                    css_class = "success"
                    emoji = "🟢"
                    message = "Appears normal – no significant findings."
                
                st.markdown(f"""
                <div class="result-box {css_class}">
                    {emoji} <strong>{pred_label}</strong><br>
                    Confidence: <strong>{confidence:.1%}</strong>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"**{message}**")
                
                # Probability bar chart
                probs_df = pd.DataFrame({
                    "Class": list(labels.values()),
                    "Probability": probabilities * 100
                })
                
                st.subheader("Prediction Confidence Breakdown")
                st.bar_chart(probs_df.set_index("Class"), color="#1e88e5", height=300)
                
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
                st.info("Try another image or ensure the upload is a valid ultrasound scan.")

# Footer disclaimer
st.markdown("---")
st.markdown(
    '<div class="disclaimer">'
    'This tool is for educational/research purposes only and is NOT a substitute for professional medical diagnosis. '
    'Always seek advice from a qualified healthcare provider. '
    'Model trained on BUSI dataset | © 2026 BreastCare AI Prototype'
    '</div>',
    unsafe_allow_html=True
)
