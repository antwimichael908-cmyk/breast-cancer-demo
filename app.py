# app.py - BreastCare AI • Dark Theme + Attractive Neon Cyan Drag & Drop
import streamlit as st
import pandas as pd
import joblib
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

# ────────────────────────────────────────────────
# Page configuration
# ────────────────────────────────────────────────
st.set_page_config(
    page_title="BreastCare AI – Ultrasound Classifier",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ────────────────────────────────────────────────
# Dark theme + attractive neon cyan drop zone
# ────────────────────────────────────────────────
st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(-45deg, #0b0f1a, #111827, #0b0f1a);
        background-size: 400% 400%;
        animation: slowShift 30s ease infinite;
        color: #e2e8f0;
    }

    @keyframes slowShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    [data-testid="stHeader"] { background: rgba(0,0,0,0); }

    h1, h2, h3 { color: #22d3ee !important; }

    /* Attractive neon cyan dashed drop zone */
    .drop-zone {
        padding: 100px 60px;
        border: 3px dashed #22d3ee;
        border-radius: 20px;
        text-align: center;
        background: rgba(15, 23, 42, 0.55);
        backdrop-filter: blur(12px);
        margin: 40px auto;
        max-width: 900px;
        box-shadow: 0 0 35px rgba(34, 211, 238, 0.3);
        transition: all 0.4s ease;
        position: relative;
    }

    .drop-zone:hover {
        border-color: #67e8f9;
        background: rgba(15, 23, 42, 0.75);
        box-shadow: 0 0 60px rgba(103, 232, 249, 0.5);
        transform: translateY(-8px);
    }

    .drop-zone::after {
        content: "";
        position: absolute;
        inset: -4px;
        border: 3px dashed #22d3ee;
        border-radius: 24px;
        opacity: 0.5;
        animation: neonGlow 3s infinite alternate;
        pointer-events: none;
    }

    @keyframes neonGlow {
        0% { opacity: 0.4; transform: scale(1); }
        100% { opacity: 0.85; transform: scale(1.04); }
    }

    .drop-zone h2 {
        margin: 0 0 24px 0;
        font-size: 2.4rem;
        color: #67e8f9;
        text-shadow: 0 0 15px rgba(103, 232, 249, 0.6);
    }

    .drop-zone .subtitle {
        font-size: 1.35rem;
        color: #a5f3fc;
        margin: 0 0 16px 0;
    }

    .drop-zone .hint {
        font-size: 1.05rem;
        color: #94a3b8;
        margin: 8px 0 0 0;
    }

    .drop-zone .icon {
        font-size: 4.5rem;
        margin-bottom: 20px;
        color: #67e8f9;
        text-shadow: 0 0 20px rgba(103, 232, 249, 0.5);
    }

    /* Result card */
    .result-card {
        padding: 32px;
        border-radius: 16px;
        background: rgba(30, 41, 59, 0.85);
        backdrop-filter: blur(14px);
        border: 1px solid #334155;
        box-shadow: 0 10px 40px rgba(0,0,0,0.6);
        animation: fadeInUp 0.7s ease-out;
        margin: 40px 0;
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(40px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    .malignant { border-left: 6px solid #f87171; }
    .benign    { border-left: 6px solid #34d399; }
    .normal    { border-left: 6px solid #22d3ee; }

    .stButton > button {
        background: linear-gradient(90deg, #0891b2, #22d3ee);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 16px 40px;
        font-weight: 600;
        box-shadow: 0 6px 25px rgba(34,211,238,0.4);
        transition: all 0.35s;
    }

    .stButton > button:hover {
        transform: translateY(-4px);
        box-shadow: 0 14px 40px rgba(34,211,238,0.6);
    }

    .disclaimer {
        font-size: 0.9rem;
        color: #94a3b8;
        text-align: center;
        margin-top: 80px;
        padding: 20px;
        background: rgba(15,23,42,0.7);
        border-radius: 12px;
    }
    </style>
""", unsafe_allow_html=True)

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
# Sidebar
# ────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/breast-cancer-ribbon.png", width=90)
    st.title("BreastCare AI")
    st.markdown("#### Intelligent Ultrasound Assistant")
    st.info("Upload a breast ultrasound image to receive an AI-supported classification.\n\n**Disclaimer**: Research prototype — not for clinical diagnosis.")
    st.markdown("---")
    st.caption("Trained on BUSI • Random Forest • 2026")

# ────────────────────────────────────────────────
# Main content – Attractive neon cyan drag & drop
# ────────────────────────────────────────────────
st.title("🩺 BreastCare AI Classifier")
st.markdown("**Upload your ultrasound image and get instant insights**")

# Main attractive drag & drop zone
st.markdown("""
<div class="drop-zone">
    <div class="icon">☁️</div>
    <h2>Drag & Drop Ultrasound Image</h2>
    <p class="subtitle">or click Browse files</p>
    <small class="hint">PNG • JPG • JPEG • Limit 200 MB per file</small>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    label="",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=False,
    label_visibility="collapsed",
    key="neon_cyan_ultrasound_drop"
)

if uploaded_file is not None:
    st.markdown("---")
    
    st.subheader("Uploaded Image Preview")
    st.image(uploaded_file, caption=f"{uploaded_file.name} • {uploaded_file.size / 1024:.1f} KB", use_column_width=True)
    
    st.markdown("<div style='height: 32px;'></div>", unsafe_allow_html=True)
    
    if st.button("Analyze This Image", type="primary", use_container_width=True):
        with st.status("Processing your ultrasound image...", expanded=True) as status:
            status.update(label="Loading image...", state="running")
            img_array = np.frombuffer(uploaded_file.getvalue(), np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            status.update(label="Extracting features...", state="running")
            features = extract_features(img)
            
            status.update(label="Running prediction...", state="running")
            pred = model.predict(features)[0]
            probs = model.predict_proba(features)[0]
            
            status.update(label="Complete", state="complete", expanded=False)
            
            labels = {0: "Benign", 1: "Malignant", 2: "Normal"}
            label = labels[pred]
            conf = probs[pred]
            
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
                <h2 style="margin:0 0 12px 0;">{emoji} {label}</h2>
                <p style="font-size:1.7rem; margin:10px 0;">Confidence: <strong>{conf:.1%}</strong></p>
                <p style="color:#cbd5e1;">{msg}</p>
            </div>
            """, unsafe_allow_html=True)
            
            probs_df = pd.DataFrame({
                "Class": ["Benign", "Malignant", "Normal"],
                "Confidence (%)": probs * 100
            }).set_index("Class")
            
            st.subheader("Confidence Breakdown")
            st.bar_chart(probs_df, color="#22d3ee", height=340)

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
