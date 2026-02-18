# app.py - BreastCare AI with lively animated gradient background
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
# Lively animated background + modern styling
# ────────────────────────────────────────────────
st.markdown("""
    <style>
    /* Animated gradient background */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(-45deg, #a1c4fd, #c2e9fb, #89d4cf, #a1c4fd, #c2e9fb);
        background-size: 400% 400%;
        animation: gradientFlow 18s ease infinite;
        min-height: 100vh;
    }

    @keyframes gradientFlow {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Transparent header */
    [data-testid="stHeader"] {
        background: rgba(0,0,0,0);
    }

    /* Floating subtle particles / bubbles effect */
    .floating-particle {
        position: absolute;
        background: rgba(255, 255, 255, 0.25);
        border-radius: 50%;
        pointer-events: none;
        animation: floatUp linear infinite;
        z-index: -1;
    }

    @keyframes floatUp {
        0% { transform: translateY(100vh) scale(0.5); opacity: 0; }
        20% { opacity: 0.8; }
        80% { opacity: 0.6; }
        100% { transform: translateY(-20vh) scale(1.2); opacity: 0; }
    }

    /* Sidebar glass effect */
    section[data-testid="stSidebar"] > div:first-child {
        background: linear-gradient(135deg, rgba(30,58,138,0.75), rgba(59,130,246,0.55));
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255,255,255,0.18);
        box-shadow: 0 4px 30px rgba(0,0,0,0.1);
    }

    /* Result card with animation */
    .result-card {
        padding: 28px;
        border-radius: 16px;
        margin: 24px 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        background: rgba(255, 255, 255, 0.94);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.4);
        animation: fadeInUp 0.7s ease-out;
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(30px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    .malignant { border-left: 6px solid #ef4444; }
    .benign    { border-left: 6px solid #10b981; }
    .normal    { border-left: 6px solid #3b82f6; }

    /* Catchy button */
    .stButton > button {
        background: linear-gradient(90deg, #3b82f6, #60a5fa);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 14px 32px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(59,130,246,0.35);
    }

    .stButton > button:hover {
        transform: translateY(-3px) scale(1.03);
        box-shadow: 0 12px 30px rgba(59,130,246,0.55);
        background: linear-gradient(90deg, #2563eb, #3b82f6);
    }

    h1 { color: #1e3a8a; text-align: center; font-weight: 700; }
    .disclaimer { font-size: 0.82rem; color: #6b7280; text-align: center; margin-top: 50px; }
    </style>

    <!-- Floating particles (added via JS) -->
    <script>
    function createParticle() {
        const particle = document.createElement('div');
        particle.className = 'floating-particle';
        particle.style.width = particle.style.height = Math.random() * 12 + 6 + 'px';
        particle.style.left = Math.random() * 100 + 'vw';
        particle.style.animationDuration = Math.random() * 12 + 10 + 's';
        particle.style.animationDelay = Math.random() * 5 + 's';
        document.body.appendChild(particle);
        
        setTimeout(() => particle.remove(), 20000);
    }

    setInterval(createParticle, 800);
    </script>
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
