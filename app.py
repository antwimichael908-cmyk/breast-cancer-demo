# app.py - Lively version with background image, overlay & micro-animations

import streamlit as st
import pandas as pd
import joblib
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
import base64
from pathlib import Path
import base64
from pathlib import Path

# ────────────────────────────────────────────────
# Page config + favicon
# ────────────────────────────────────────────────
st.set_page_config(
    page_title="BreastCare AI – Smart Ultrasound Classifier",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ────────────────────────────────────────────────
# Helper: Load local image as base64 for CSS background
# ────────────────────────────────────────────────


# ────────────────────────────────────────────────
# Load local background image as base64
# ────────────────────────────────────────────────
def get_base64_of_file(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# Use this filename (must match exactly what you saved)
BG_FILENAME = "background.jpg"

try:
    bg_base64 = get_base64_of_file(BG_FILENAME)
    bg_data_url = f"data:image/jpeg;base64,{bg_base64}"
except FileNotFoundError:
    st.warning("Background image not found — using fallback color")
    bg_data_url = "linear-gradient(135deg, #e0f2fe, #bfdbfe)"  # light blue fallback

# ────────────────────────────────────────────────
# Apply background with overlay for readability
# ────────────────────────────────────────────────
st.markdown(f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: 
            linear-gradient(rgba(245, 245, 250, 0.84), rgba(245, 245, 250, 0.88)),
            url("{bg_data_url}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}

    [data-testid="stHeader"] {{
        background: rgba(0,0,0,0);
    }}

    /* Optional: make sidebar pop more */
    section[data-testid="stSidebar"] > div:first-child {{
        background: linear-gradient(135deg, rgba(30,58,138,0.82), rgba(59,130,246,0.65));
        backdrop-filter: blur(8px);
    }}
    </style>
""", unsafe_allow_html=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Path to your background image (put it in same folder as app.py)
BG_IMAGE_PATH = "background.jpg"  # ← CHANGE TO YOUR FILE NAME

try:
    bg_base64 = get_base64_of_bin_file(BG_IMAGE_PATH)
    bg_url = f"data:image/jpeg;base64,{bg_base64}"
except FileNotFoundError:
    bg_url = "https://images.unsplash.com/photo-1557683316-973673baf926?auto=format&fit=crop&q=80&w=2000"  # fallback subtle gradient
    st.warning("Background image not found → using fallback online image")

# ────────────────────────────────────────────────
# Modern & lively CSS with background + overlay + animations
# ────────────────────────────────────────────────
# Use this if image is in repo root
BG_IMAGE = "bg-ultrasound.jpg"          # your saved filename

# or use direct link
# BG_IMAGE = "https://i.postimg.cc/your-real-link.jpg"

st.markdown(f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background: linear-gradient(rgba(248, 250, 255, 0.86), rgba(240, 248, 255, 0.90)),
                    url('{BG_IMAGE if BG_IMAGE.startswith("http") else f"data:image/jpeg;base64,{get_base64_image(BG_IMAGE)}"}');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    </style>
""", unsafe_allow_html=True)
    

/* Header / top bar transparent */
    [data-testid="stHeader"] {{
        background: rgba(0,0,0,0);
    }}

    /* Sidebar lively touch */
    [data-testid="stSidebar"] > div:first-child {{
        background: linear-gradient(135deg, rgba(30, 58, 138, 0.88), rgba(59, 130, 246, 0.75));
        backdrop-filter: blur(8px);
        border-right: 1px solid rgba(255,255,255,0.15);
    }}

    /* Card-like result box with subtle animation */
    .result-card {{
        padding: 28px;
        border-radius: 16px;
        margin: 24px 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.12);
        background: rgba(255, 255, 255, 0.92);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.3);
        animation: fadeInUp 0.6s ease-out;
    }}

    @keyframes fadeInUp {{
        from {{ opacity: 0; transform: translateY(20px); }}
        to   {{ opacity: 1; transform: translateY(0); }}
    }}

    .malignant {{ border-left: 6px solid #ef4444; }}
    .benign    {{ border-left: 6px solid #10b981; }}
    .normal    {{ border-left: 6px solid #3b82f6; }}

    /* Buttons – lively hover */
    .stButton > button {{
        background: linear-gradient(90deg, #3b82f6, #60a5fa);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 14px 32px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(59,130,246,0.3);
    }}
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(59,130,246,0.5);
        background: linear-gradient(90deg, #2563eb, #3b82f6);
    }}

    /* Typography */
    h1 {{ color: #1e40af; text-align: center; font-weight: 700; }}
    .disclaimer {{ font-size: 0.82rem; color: #6b7280; text-align: center; margin-top: 50px; }}
    </style>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────
# Load model (with cache)
# ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        return joblib.load("best_tuned_rf_breast_ultrasound.pkl")
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        st.stop()

model = load_model()

# ────────────────────────────────────────────────
# Feature extraction (unchanged)
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
    
    _, thresh = cv2.threshold(img_resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = perimeter = circularity = 0.0
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
    
    return pd.DataFrame([{
        'mean_intensity': mean_intensity,
        'std_intensity': std_intensity,
        'area': area,
        'perimeter': perimeter,
        'circularity': circularity,
        **texture_feats
    }])

# ────────────────────────────────────────────────
# Sidebar – enhanced
# ────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/breast-cancer-ribbon.png", width=90)
    st.title("BreastCare AI")
    st.markdown("#### Intelligent Ultrasound Assistant")
    st.info("Upload a breast ultrasound image to receive an AI-supported classification.\n\n**Disclaimer**: Research prototype — not for clinical diagnosis.")
    st.markdown("---")
    st.caption("Trained on BUSI • Random Forest • 2026")

# ────────────────────────────────────────────────
# Main content
# ────────────────────────────────────────────────
st.title("🩺 BreastCare AI Classifier")
st.markdown("**Upload your ultrasound image and discover insights in seconds**")

uploaded_file = st.file_uploader(
    "Choose or drag & drop an ultrasound image (PNG / JPG)",
    type=["png", "jpg", "jpeg"],
    help="For best results use clear, high-contrast ultrasound scans"
)

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    st.image(img, channels="BGR", caption="Your Uploaded Image", use_column_width=True)
    
    if st.button("Analyze & Predict", type="primary", use_container_width=True):
        with st.spinner("Analyzing texture, shape & patterns..."):
            try:
                features = extract_features(img)
                pred = model.predict(features)[0]
                probs = model.predict_proba(features)[0]
                
                labels = {0: "Benign", 1: "Malignant", 2: "Normal"}
                label = labels[pred]
                conf = probs[pred]
                
                # Dynamic card style
                if label == "Malignant":
                    cls = "malignant"
                    emoji = "🔴"
                    msg = "Potential concern detected — urgent professional review recommended."
                elif label == "Benign":
                    cls = "benign"
                    emoji = "🟢"
                    msg = "Likely benign finding — continue routine monitoring."
                else:
                    cls = "normal"
                    emoji = "✅"
                    msg = "No significant abnormalities observed."
                
                st.markdown(f"""
                <div class="result-card {cls}">
                    <h2 style="margin:0;">{emoji} {label}</h2>
                    <p style="font-size:1.4rem; margin:12px 0;">Confidence: <strong>{conf:.1%}</strong></p>
                    <p>{msg}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Confidence visualization
                probs_df = pd.DataFrame({
                    "Class": ["Benign", "Malignant", "Normal"],
                    "Confidence (%)": probs * 100
                }).set_index("Class")
                
                st.subheader("Detailed Confidence Distribution")
                st.bar_chart(probs_df, color="#3b82f6", height=280)
                
            except Exception as ex:
                st.error(f"Prediction failed: {ex}")
                st.info("Try a different image or check file format.")

# Footer
st.markdown("---")
st.markdown(
    '<div class="disclaimer">'
    'This is an educational/research demonstration only. '
    'It is **not** a medical device and should never replace professional radiological or clinical evaluation. '
    'Always consult a qualified healthcare provider.'
    '</div>',
    unsafe_allow_html=True
)
