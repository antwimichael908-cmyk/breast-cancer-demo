# app.py - BreastCare AI • Clean, Simple & Catchy Design
import streamlit as st
import pandas as pd
import joblib
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

# ────────────────────────────────────────────────
# Page config
# ────────────────────────────────────────────────
st.set_page_config(
    page_title="BreastCare AI",
    page_icon="🩺",
    layout="wide"
)

# ────────────────────────────────────────────────
# Simple & catchy styling
# ────────────────────────────────────────────────
st.markdown("""
    <style>
    /* Clean dark theme */
    [data-testid="stAppViewContainer"] {
        background-color: #0f172a;
        color: #e2e8f0;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    /* Nice header */
    .header {
        background: linear-gradient(90deg, #1e40af, #3b82f6);
        padding: 24px;
        border-radius: 0 0 16px 16px;
        margin: -16px -32px 32px -32px;
        text-align: center;
        color: white;
    }

    .header h1 {
        margin: 0;
        font-size: 2.6rem;
        font-weight: 700;
    }

    .header p {
        margin: 8px 0 0 0;
        font-size: 1.15rem;
        opacity: 0.95;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #1e293b;
        border-right: 1px solid #334155;
    }

    .sidebar-title {
        color: #60a5fa;
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 20px;
        text-align: center;
    }

    /* Upload area */
    .upload-box {
        padding: 48px 32px;
        border: 2px dashed #60a5fa;
        border-radius: 16px;
        text-align: center;
        background: #1e293b;
        margin: 24px 0 32px 0;
        transition: all 0.3s ease;
    }

    .upload-box:hover {
        border-color: #93c5fd;
        background: #253549;
    }

    .upload-box h3 {
        margin: 0 0 16px 0;
        color: #93c5fd;
        font-size: 1.6rem;
    }

    .upload-box p {
        margin: 0 0 12px 0;
        color: #cbd5e1;
    }

    .upload-box small {
        color: #94a3b8;
    }

    /* Result card */
    .result-card {
        padding: 28px;
        border-radius: 12px;
        background: #1e293b;
        border: 1px solid #334155;
        margin: 32px 0;
    }

    .malignant { border-left: 6px solid #f87171; }
    .benign    { border-left: 6px solid #34d399; }
    .normal    { border-left: 6px solid #60a5fa; }

    /* Button */
    .stButton > button {
        background: #3b82f6;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 32px;
        font-weight: 600;
        transition: all 0.3s;
    }

    .stButton > button:hover {
        background: #60a5fa;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 24px;
        color: #94a3b8;
        font-size: 0.9rem;
        border-top: 1px solid #334155;
        margin-top: 60px;
    }

    .footer strong {
        color: #93c5fd;
    }
    </style>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────
# Clean header
# ────────────────────────────────────────────────
st.markdown("""
    <div class="header">
        <h1>🩺 BreastCare AI</h1>
        <p>Breast Ultrasound Classifier</p>
    </div>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────
# Sidebar – clean & simple
# ────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">BreastCare AI</div>', unsafe_allow_html=True)
    
    st.image("https://img.icons8.com/fluency/96/000000/breast-cancer-ribbon.png", width=80)
    
    st.markdown("**Purpose**")
    st.caption("AI-supported classification of breast ultrasound images")
    
    st.markdown("**Model**")
    st.caption("Random Forest • Trained on BUSI dataset")
    
    st.info("**Research prototype**\n\nNot for clinical diagnosis.\nAlways consult a doctor.", icon="⚠️")

# ────────────────────────────────────────────────
# Load model
# ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        return joblib.load("best_tuned_rf_breast_ultrasound.pkl")
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        st.stop()

model = load_model()

# Feature extraction (unchanged)
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
# Main content – simple & attractive upload
# ────────────────────────────────────────────────
st.markdown("### Upload Breast Ultrasound Image")

st.markdown("""
<div class="upload-box">
    <h3>Drag & Drop Ultrasound Image</h3>
    <p>or click Browse files</p>
    <small>PNG • JPG • JPEG • Max 200 MB</small>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    label="",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=False,
    label_visibility="collapsed",
    key="simple_uploader"
)

if uploaded_file is not None:
    st.markdown("---")
    
    col1, col2 = st.columns([5, 2])
    
    with col1:
        st.image(uploaded_file, caption=f"{uploaded_file.name} • {uploaded_file.size / 1024:.1f} KB", use_column_width=True)
    
    with col2:
        st.markdown("<br>"*3, unsafe_allow_html=True)
        if st.button("Analyze Image", type="primary", use_container_width=True):
            with st.spinner("Analyzing..."):
                img_array = np.frombuffer(uploaded_file.getvalue(), np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                
                features = extract_features(img)
                pred = model.predict(features)[0]
                probs = model.predict_proba(features)[0]
                
                labels = {0: "Benign", 1: "Malignant", 2: "Normal"}
                label = labels[pred]
                conf = probs[pred]
                
                if label == "Malignant":
                    cls = "malignant"
                    emoji = "🔴"
                    msg = "Potential concern detected – please consult a specialist urgently."
                elif label == "Benign":
                    cls = "benign"
                    emoji = "🟢"
                    msg = "Likely benign – routine follow-up recommended."
                else:
                    cls = "normal"
                    emoji = "✅"
                    msg = "Appears normal – no significant findings."
                
                st.markdown(f"""
                <div class="result-card {cls}">
                    <h2>{emoji} {label}</h2>
                    <p style="font-size:1.6rem; margin:12px 0;">Confidence: <strong>{conf:.1%}</strong></p>
                    <p>{msg}</p>
                </div>
                """, unsafe_allow_html=True)
                
                probs_df = pd.DataFrame({
                    "Class": ["Benign", "Malignant", "Normal"],
                    "Confidence (%)": probs * 100
                }).set_index("Class")
                
                st.bar_chart(probs_df, color="#3b82f6", height=300)

# ────────────────────────────────────────────────
# Simple footer
# ────────────────────────────────────────────────
st.markdown("""
    <div class="footer">
        <strong>BreastCare AI</strong> • Research Prototype 2026<br>
        Trained on BUSI dataset • Not for clinical use
    </div>
""", unsafe_allow_html=True)
