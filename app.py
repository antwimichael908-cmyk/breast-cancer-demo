# app.py - BreastCare AI with polished upload section (Option 3)
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
# Lively animated background + styling
# ────────────────────────────────────────────────
st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(-45deg, #a1c4fd, #c2e9fb, #89d4cf, #a1c4fd, #c2e9fb);
        background-size: 400% 400%;
        animation: gradientFlow 18s ease infinite;
    }

    @keyframes gradientFlow {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    [data-testid="stHeader"] { background: rgba(0,0,0,0); }

    .upload-card {
        padding: 32px 24px;
        border-radius: 16px;
        background: linear-gradient(135deg, #f0f9ff, #e0f2fe);
        border: 2px dashed #60a5fa;
        text-align: center;
        margin: 16px 0 24px 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }

    .result-card {
        padding: 28px;
        border-radius: 16px;
        margin: 24px 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.12);
        background: rgba(255, 255, 255, 0.94);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.3);
        animation: fadeInUp 0.7s ease-out;
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(30px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    .malignant { border-left: 6px solid #ef4444; }
    .benign    { border-left: 6px solid #10b981; }
    .normal    { border-left: 6px solid #3b82f6; }

    .stButton > button {
        background: linear-gradient(90deg, #3b82f6, #60a5fa);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 14px 32px;
        font-weight: 600;
        transition: all 0.3s;
        box-shadow: 0 4px 15px rgba(59,130,246,0.35);
    }

    .stButton > button:hover {
        transform: translateY(-3px) scale(1.03);
        box-shadow: 0 12px 30px rgba(59,130,246,0.55);
    }

    h1 { color: #1e3a8a; text-align: center; font-weight: 700; }
    .disclaimer { font-size: 0.82rem; color: #6b7280; text-align: center; margin-top: 50px; }
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

# ────────────────────────────────────────────────
# Feature extraction
# ────────────────────────────────────────────────
def extract_features(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, (224, 224))
    
    mean_intensity = np.mean(img_resized)
    std_intensity = np.std(img_resized)
    
    distances = [1, 2, 3]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
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
# Main content – Improved upload section
# ────────────────────────────────────────────────
st.title("🩺 BreastCare AI Classifier")
st.markdown("**Upload your ultrasound image and discover insights in seconds**")

with st.container():
    st.markdown("### Step 1: Upload Breast Ultrasound Image")

    # Beautiful upload card
    st.markdown("""
    <div class="upload-card">
        <h3 style="margin: 0 0 12px 0; color: #1e40af;">Drop your image here</h3>
        <p style="color: #475569; margin: 0 0 20px 0;">
            or click to browse • PNG, JPG, JPEG supported
        </p>
        <small style="color: #64748b;">
            Best results with focused lesion views • Max ~10 MB
        </small>
    </div>
    """, unsafe_allow_html=True)

    col_upload, col_info = st.columns([5, 2])

    with col_upload:
        uploaded_file = st.file_uploader(
            label="",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=False,
            label_visibility="collapsed",
            key="breast_ultrasound_uploader_v3"
        )

    with col_info:
        st.markdown("**Quick tips**")
        st.info("Crop to region of interest if possible", icon="💡")
        st.info("Avoid heavy text overlays or markers", icon="🔍")
        st.info("Clear, high-contrast images work best", icon="📸")

    if uploaded_file is not None:
        st.markdown("---")
        
        preview_col, status_col = st.columns([3, 1])
        
        with preview_col:
            st.image(uploaded_file, caption=f"Preview: {uploaded_file.name}", use_column_width=True)
        
        with status_col:
            st.metric("File size", f"{uploaded_file.size / 1024:.1f} KB")
            
            if st.button("Analyze Image", type="primary", use_container_width=True):
                with st.status("Processing image...", expanded=True) as status:
                    status.update(label="Reading & preprocessing image...", state="running")
                    img = cv2.imdecode(np.frombuffer(uploaded_file.getvalue(), np.uint8), cv2.IMREAD_COLOR)
                    
                    status.update(label="Extracting texture & shape features...", state="running")
                    features = extract_features(img)
                    
                    status.update(label="Running Random Forest prediction...", state="running")
                    pred = model.predict(features)[0]
                    probs = model.predict_proba(features)[0]
                    
                    status.update(label="Analysis complete!", state="complete", expanded=False)
                    
                    # Show result
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
                        <h2 style="margin:0;">{emoji} {label}</h2>
                        <p style="font-size:1.4rem; margin:12px 0;">Confidence: <strong>{conf:.1%}</strong></p>
                        <p>{msg}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    probs_df = pd.DataFrame({
                        "Class": ["Benign", "Malignant", "Normal"],
                        "Confidence (%)": probs * 100
                    }).set_index("Class")
                    
                    st.subheader("Detailed Confidence Distribution")
                    st.bar_chart(probs_df, color="#3b82f6", height=280)

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
