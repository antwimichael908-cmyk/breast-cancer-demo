# app.py - BreastCare AI • Dark Blue-Black Theme + Improved Upload Section
import streamlit as st
import pandas as pd
import joblib
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

# ────────────────────────────────────────────────
# Page config + dark mode base
# ────────────────────────────────────────────────
st.set_page_config(
    page_title="BreastCare AI – Ultrasound Classifier",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ────────────────────────────────────────────────
# Dark blue-black theme with subtle motion
# ────────────────────────────────────────────────
st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(-45deg, #0f172a, #1e293b, #0f172a);
        background-size: 400% 400%;
        animation: slowShift 25s ease infinite;
        color: #e2e8f0;
    }

    @keyframes slowShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    [data-testid="stHeader"] { background: rgba(0,0,0,0); }

    h1, h2, h3 { color: #60a5fa !important; }

    .stButton > button {
        background: linear-gradient(90deg, #3b82f6, #60a5fa);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 28px;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(59,130,246,0.4);
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(59,130,246,0.6);
    }

    .upload-card {
        padding: 44px 36px;
        border-radius: 20px;
        background: rgba(30, 41, 59, 0.75);
        border: 2px dashed #60a5fa88;
        text-align: center;
        margin: 28px 0 40px 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
        backdrop-filter: blur(10px);
        border-image: linear-gradient(45deg, #3b82f6, #60a5fa) 1;
    }

    .result-card {
        padding: 32px;
        border-radius: 16px;
        background: rgba(30, 41, 59, 0.85);
        backdrop-filter: blur(12px);
        border: 1px solid #475569;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        animation: fadeInUp 0.6s ease-out;
        margin: 24px 0;
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(30px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    .malignant { border-left: 6px solid #f87171; }
    .benign    { border-left: 6px solid #34d399; }
    .normal    { border-left: 6px solid #60a5fa; }

    .disclaimer { 
        font-size: 0.85rem; 
        color: #94a3b8; 
        text-align: center; 
        margin-top: 60px; 
        padding: 16px;
        background: rgba(15,23,42,0.6);
        border-radius: 12px;
    }

    .stTabs [data-baseweb="tab-list"] {
        background: rgba(30,41,59,0.6);
        border-radius: 12px;
    }

    .stTabs [data-baseweb="tab"] {
        color: #cbd5e1;
    }

    .stTabs [aria-selected="true"] {
        color: #60a5fa !important;
        background: rgba(96,165,250,0.15);
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
# Sidebar (dark theme friendly)
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

# Improved upload card
st.markdown("""
<div class="upload-card">
    <h2 style="margin: 0 0 16px 0;">Upload Breast Ultrasound Image</h2>
    <p style="color: #cbd5e1; margin: 0 0 20px 0; font-size: 1.1rem;">
        Drag & drop or click to select • PNG / JPG / JPEG
    </p>
    <small style="color: #94a3b8;">
        Best results with clear, high-contrast scans focused on region of interest
    </small>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([6, 3])

with col1:
    uploaded_file = st.file_uploader(
        label="",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=False,
        label_visibility="collapsed",
        key="improved_uploader_2026"
    )

with col2:
    st.markdown("**Quick checklist**")
    st.markdown("• High resolution preferred")
    st.markdown("• Focused on area of interest")
    st.markdown("• Avoid heavy annotations")
    st.markdown("• Max ~10 MB recommended")

if uploaded_file is not None:
    st.markdown("---")
    
    # Preview & controls
    preview_col, info_col = st.columns([4, 2])
    
    with preview_col:
        st.image(uploaded_file, caption=f"Uploaded: {uploaded_file.name}", use_column_width=True)
    
    with info_col:
        st.metric("File size", f"{uploaded_file.size / 1024:.1f} KB")
        st.metric("Format", uploaded_file.type.split('/')[-1].upper())
        
        st.markdown("**Ready?**")
        analyze = st.button("Analyze Image Now", type="primary", use_container_width=True)

    if analyze:
        with st.status("Processing your ultrasound image...", expanded=True) as status:
            status.update(label="Loading and preprocessing image...", state="running")
            img_array = np.frombuffer(uploaded_file.getvalue(), np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            status.update(label="Extracting texture and shape features...", state="running")
            features = extract_features(img)
            
            status.update(label="Running prediction model...", state="running")
            pred = model.predict(features)[0]
            probs = model.predict_proba(features)[0]
            
            status.update(label="Analysis complete", state="complete", expanded=False)
            
            # Result display
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
                <p style="font-size:1.5rem; margin:8px 0;">Confidence: <strong>{conf:.1%}</strong></p>
                <p style="margin:16px 0 0 0; color:#cbd5e1;">{msg}</p>
            </div>
            """, unsafe_allow_html=True)
            
            probs_df = pd.DataFrame({
                "Class": ["Benign", "Malignant", "Normal"],
                "Confidence (%)": probs * 100
            }).set_index("Class")
            
            st.subheader("Prediction Confidence Breakdown")
            st.bar_chart(probs_df, color="#60a5fa", height=300)

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
