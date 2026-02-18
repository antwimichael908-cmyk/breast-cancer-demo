# app.py - BreastCare AI • Reset to original core + Improved lively UI (neon accents, sidebar, header, footer)
import streamlit as st
import pandas as pd
import joblib
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

# ────────────────────────────────────────────────
# Page config + modern lively look
# ────────────────────────────────────────────────
st.set_page_config(
    page_title="BreastCare AI – Ultrasound Classifier",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ────────────────────────────────────────────────
# Lively UI – header, neon sidebar, animated footer
# ────────────────────────────────────────────────
st.markdown("""
    <style>
    /* Global dark + neon cyan vibe */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0f172a 0%, #111827 100%);
        color: #e2e8f0;
    }

    /* Fancy header */
    .main-header {
        background: linear-gradient(90deg, #0ea5e9, #22d3ee);
        padding: 28px 40px;
        border-radius: 0 0 20px 20px;
        box-shadow: 0 10px 30px rgba(34,211,238,0.3);
        text-align: center;
        margin: -16px -32px 32px -32px;
        color: white;
        text-shadow: 0 2px 10px rgba(0,0,0,0.5);
    }

    .main-header h1 {
        margin: 0;
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: -1px;
    }

    .main-header p {
        margin: 8px 0 0 0;
        font-size: 1.25rem;
        opacity: 0.95;
    }

    /* Neon sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        border-right: 1px solid #334155;
    }

    section[data-testid="stSidebar"] > div:first-child {
        background: rgba(15,23,42,0.85);
        backdrop-filter: blur(12px);
        padding: 20px 16px;
    }

    .sidebar-title {
        color: #22d3ee;
        font-size: 1.6rem;
        margin-bottom: 20px;
        text-shadow: 0 0 10px rgba(34,211,238,0.6);
        text-align: center;
    }

    .sidebar-info {
        background: rgba(30,41,59,0.6);
        border-radius: 12px;
        padding: 16px;
        margin: 16px 0;
        border-left: 4px solid #67e8f9;
    }

    /* Result card with neon glow */
    .result-card {
        padding: 32px;
        border-radius: 16px;
        background: rgba(30,41,59,0.85);
        backdrop-filter: blur(12px);
        border: 1px solid #334155;
        box-shadow: 0 0 30px rgba(34,211,238,0.25);
        margin: 32px 0;
        animation: glowPulse 3s infinite alternate;
    }

    @keyframes glowPulse {
        0% { box-shadow: 0 0 20px rgba(34,211,238,0.2); }
        100% { box-shadow: 0 0 45px rgba(34,211,238,0.45); }
    }

    .malignant { border-left: 8px solid #f87171; }
    .benign    { border-left: 8px solid #34d399; }
    .normal    { border-left: 8px solid #22d3ee; }

    /* Cool button */
    .stButton > button {
        background: linear-gradient(90deg, #0891b2, #22d3ee);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 14px 36px;
        font-weight: bold;
        box-shadow: 0 6px 20px rgba(34,211,238,0.4);
        transition: all 0.35s;
        width: 100%;
    }

    .stButton > button:hover {
        transform: translateY(-4px) scale(1.02);
        box-shadow: 0 12px 35px rgba(34,211,238,0.6);
    }

    /* Animated footer */
    .cool-footer {
        background: rgba(15,23,42,0.85);
        backdrop-filter: blur(10px);
        border-top: 1px solid #334155;
        padding: 28px;
        margin-top: 60px;
        text-align: center;
        border-radius: 16px 16px 0 0;
        font-size: 0.95rem;
        color: #94a3b8;
    }

    .cool-footer strong {
        color: #67e8f9;
    }
    </style>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────
# Fancy Header
# ────────────────────────────────────────────────
st.markdown("""
    <div class="main-header">
        <h1>🩺 BreastCare AI</h1>
        <p>Intelligent Ultrasound Breast Cancer Classifier</p>
    </div>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────
# Sidebar – improved layout
# ────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">BreastCare AI</div>', unsafe_allow_html=True)
    
    st.image("https://img.icons8.com/fluency/96/000000/breast-cancer-ribbon.png", width=100)
    
    st.markdown('<div class="sidebar-info">'
                '<strong>Powered by:</strong><br>'
                'Random Forest Classifier<br>'
                'Trained on BUSI dataset<br>'
                'Research prototype – 2026'
                '</div>', unsafe_allow_html=True)
    
    st.info("**Important**\n\nThis tool is for educational/research purposes only. "
            "It is **not** a substitute for professional medical diagnosis.", icon="⚠️")

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

# Feature extraction function (unchanged)
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
# Main content – simplified & attractive
# ────────────────────────────────────────────────
st.markdown("### Upload Breast Ultrasound Image")

uploaded_file = st.file_uploader(
    label="Drag & drop or click to select image",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=False,
    label_visibility="visible",
    help="PNG • JPG • JPEG • Max 200 MB • Clear, high-contrast scans work best",
    key="main_ultrasound_uploader"
)

if uploaded_file is not None:
    st.success(f"Image uploaded: {uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)")
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        st.image(uploaded_file, caption="Uploaded image preview", use_column_width=True)
    
    with col2:
        st.markdown("<br>"*2, unsafe_allow_html=True)
        if st.button("Analyze Image", type="primary", use_container_width=True):
            with st.spinner("Analyzing ultrasound image..."):
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
                    msg = "Potential malignancy detected – urgent specialist review recommended."
                elif label == "Benign":
                    cls = "benign"
                    emoji = "🟢"
                    msg = "Likely benign lesion – routine follow-up advised."
                else:
                    cls = "normal"
                    emoji = "✅"
                    msg = "Appears normal – no significant findings."
                
                st.markdown(f"""
                <div class="result-card {cls}">
                    <h2 style="margin:0;">{emoji} {label}</h2>
                    <p style="font-size:1.8rem; margin:16px 0;">Confidence: <strong>{conf:.1%}</strong></p>
                    <p>{msg}</p>
                </div>
                """, unsafe_allow_html=True)
                
                probs_df = pd.DataFrame({
                    "Class": ["Benign", "Malignant", "Normal"],
                    "Confidence (%)": probs * 100
                }).set_index("Class")
                
                st.subheader("Confidence Breakdown")
                st.bar_chart(probs_df, color="#22d3ee", height=340)

# ────────────────────────────────────────────────
# Cool animated footer
# ────────────────────────────────────────────────
st.markdown("""
    <div class="cool-footer">
        <strong>BreastCare AI</strong> • Research Prototype • 2026<br>
        Trained on BUSI dataset • Not for clinical use • Always consult a radiologist or oncologist
    </div>
""", unsafe_allow_html=True)
