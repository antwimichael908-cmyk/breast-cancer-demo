# app.py - BreastCare AI • Blue-Black Theme + Improved Upload & Output
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
    page_title="BreastCare AI – Ultrasound Classifier",
    page_icon="🩺",
    layout="wide"
)

# ────────────────────────────────────────────────
# Blue-black theme + improved upload & result styling
# ────────────────────────────────────────────────
st.markdown("""
    <style>
    /* Blue-black background */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #e2e8f0;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    /* Header */
    .header {
        background: linear-gradient(90deg, #1e40af, #3b82f6);
        padding: 24px;
        border-radius: 0 0 16px 16px;
        margin: -16px -32px 40px -32px;
        text-align: center;
        color: white;
    }

    .header h1 {
        margin: 0;
        font-size: 2.8rem;
        font-weight: 700;
    }

    .header p {
        margin: 8px 0 0 0;
        font-size: 1.2rem;
    }

    /* Improved upload area */
    .upload-container {
        padding: 60px 40px;
        border: 2.5px dashed #60a5fa;
        border-radius: 16px;
        text-align: center;
        background: rgba(30, 41, 59, 0.5);
        margin: 32px auto;
        max-width: 900px;
        transition: all 0.3s ease;
    }

    .upload-container:hover {
        border-color: #93c5fd;
        background: rgba(30, 41, 59, 0.7);
        box-shadow: 0 0 25px rgba(96,165,250,0.3);
    }

    .upload-container h3 {
        margin: 0 0 16px 0;
        color: #93c5fd;
        font-size: 1.9rem;
    }

    .upload-container p {
        margin: 0 0 12px 0;
        font-size: 1.15rem;
        color: #cbd5e1;
    }

    .upload-container small {
        color: #94a3b8;
        font-size: 1rem;
    }

    /* Result card */
    .result-card {
        padding: 32px;
        border-radius: 14px;
        background: #1e293b;
        border: 1px solid #334155;
        box-shadow: 0 8px 25px rgba(0,0,0,0.4);
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
        padding: 14px 40px;
        font-weight: 600;
        transition: all 0.3s;
        width: 100%;
        max-width: 400px;
    }

    .stButton > button:hover {
        background: #60a5fa;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #1e293b;
        border-right: 1px solid #334155;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 24px 0;
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
# Header
# ────────────────────────────────────────────────
st.markdown("""
    <div class="header">
        <h1>🩺 BreastCare AI</h1>
        <p>Breast Ultrasound Image Classifier</p>
    </div>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────
# Sidebar
# ────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/breast-cancer-ribbon.png", width=80)
    st.title("BreastCare AI")
    
    st.markdown("**About the tool**")
    st.caption("AI-assisted classification of breast ultrasound images")
    
    st.markdown("**Model**")
    st.caption("Random Forest • Trained on BUSI dataset")
    
    st.info("**Important**\n\nThis is a research prototype only.\nNot for medical diagnosis.", icon="⚠️")

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
# Main content – improved upload area
# ────────────────────────────────────────────────


st.markdown("""
<style>
.upload-box {
    padding: 60px 40px;
    border: 3px dashed #22d3ee;
    border-radius: 16px;
    text-align: center;
    background: rgba(30, 41, 59, 0.6);
    backdrop-filter: blur(8px);
    margin: 32px auto;
    max-width: 800px;
    box-shadow: 0 4px 25px rgba(34, 211, 238, 0.15);
    transition: all 0.35s ease;
    position: relative;
    overflow: hidden;
}

 .upload-box:hover {
    border-color: #67e8f9;
    box-shadow: 0 8px 35px rgba(103, 232, 249, 0.35);
    transform: translateY(-4px);
}

.upload-box::after {
    content: "";
    position: absolute;
    inset: -3px;
    border: 3px dashed #22d3ee;
    border-radius: 18px;
    opacity: 0.5;
    animation: softPulse 4s infinite alternate;
    pointer-events: none;
}

@keyframes softPulse {
    0% { opacity: 0.35; transform: scale(1); }
    100% { opacity: 0.75; transform: scale(1.03); }
}

.upload-box .icon {
    font-size: 3.8rem;
    margin-bottom: 20px;
    color: #67e8f9;
}

.upload-box h3 {
    margin: 0 0 16px 0;
    color: #a5f3fc;
    font-size: 2rem;
    font-weight: 600;
    text-shadow: 0 0 10px rgba(167, 243, 252, 0.4);
}

.upload-box .or-text {
    font-size: 1.25rem;
    color: #cbd5e1;
    margin: 12px 0;
    font-weight: 500;
}

.upload-box .hint {
    font-size: 1rem;
    color: #94a3b8;
    line-height: 1.5;
}
</style>

<div class="upload-box">
    <div class="icon">🩻</div>
    <h3>Drag & Drop Ultrasound Image</h3>
    <p class="or-text">or click Browse files</p>
    <div class="hint">
        Supported formats: PNG • JPG • JPEG<br>
        Recommended: Clear, high-contrast scans<br>
        Limit: 200 MB per file
    </div>
</div>
""", unsafe_allow_html=True)

# The actual uploader (invisible label, full width)
uploaded_file = st.file_uploader(
    label="",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=False,
    label_visibility="collapsed",
    key="styled_ultrasound_uploader"
)
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    label="",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=False,
    label_visibility="collapsed",
    key="clean_uploader"
)

if uploaded_file is not None:
    st.markdown("---")
    
    st.subheader("Image Preview")
    st.image(uploaded_file, caption=f"{uploaded_file.name} • {uploaded_file.size / 1024:.1f} KB", use_column_width=True)
    
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
                msg = "Potential concern detected – urgent specialist review recommended."
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
                <h2 style="margin:0 0 12px 0;">{emoji} {label}</h2>
                <p style="font-size:1.6rem; margin:12px 0;">Confidence: <strong>{conf:.1%}</strong></p>
                <p>{msg}</p>
            </div>
            """, unsafe_allow_html=True)
            
            probs_df = pd.DataFrame({
                "Class": ["Benign", "Malignant", "Normal"],
                "Confidence (%)": probs * 100
            }).set_index("Class")
            
            st.subheader("Confidence Breakdown")
            st.bar_chart(probs_df, color="#3b82f6", height=300)

# ────────────────────────────────────────────────
# Footer
# ────────────────────────────────────────────────
st.markdown("""
    <div class="footer">
        <strong>BreastCare AI</strong> • Research Prototype • 2026<br>
        Trained on BUSI dataset • Not for clinical diagnosis
    </div>
""", unsafe_allow_html=True)
