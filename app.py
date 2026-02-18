# app.py - BreastCare AI • Clean Blue-Black Theme
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
# Simple blue-black styling
# ────────────────────────────────────────────────
st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0f172a, #1e293b);
        color: #e2e8f0;
    }
    [data-testid="stHeader"] {
        background: transparent;
    }
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
        font-size: 2.6rem;
        font-weight: 700;
    }
    .header p {
        margin: 8px 0 0 0;
        font-size: 1.15rem;
    }
    .result-card {
        padding: 28px;
        border-radius: 12px;
        background: #1e293b;
        border: 1px solid #334155;
        box-shadow: 0 6px 20px rgba(0,0,0,0.4);
        margin: 32px 0;
    }
    .malignant { border-left: 6px solid #f87171; }
    .benign { border-left: 6px solid #34d399; }
    .normal { border-left: 6px solid #60a5fa; }
    .stButton > button {
        background: #3b82f6;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 32px;
        font-weight: 600;
    }
    .stButton > button:hover {
        background: #60a5fa;
    }
    section[data-testid="stSidebar"] {
        background: #1e293b;
        border-right: 1px solid #334155;
    }
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
  
    st.markdown("**Purpose**")
    st.caption("AI-supported classification of breast ultrasound images")
  
    st.markdown("**Model**")
    st.caption("Random Forest • Trained on BUSI dataset")
  
    st.info("**Research prototype only**\n\nNot for clinical diagnosis.", icon="⚠️")

# ────────────────────────────────────────────────
# Load model
# ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model = joblib.load("best_tuned_rf_breast_ultrasound.pkl")
        if hasattr(model, 'n_features_in_'):
            st.session_state.expected_features = model.n_features_in_
        return model
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        st.stop()

model = load_model()

# ────────────────────────────────────────────────
# Fixed feature extraction – flattened 128×128 grayscale
# ────────────────────────────────────────────────
def extract_features(img):
    try:
        # Convert to grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Resize to 128×128 → exactly 16384 pixels
        img_resized = cv2.resize(img_gray, (128, 128))
        
        # Safety check
        if img_resized.shape != (128, 128):
            raise ValueError(f"Resized image has wrong shape: {img_resized.shape}")
        
        # Flatten → 16384 values
        features_flat = img_resized.flatten().astype(np.float32)
        
        # Optional: normalize to [0,1] range (uncomment if your training data was normalized)
        # features_flat = features_flat / 255.0
        
        # Build dictionary with 16384 keys
        feature_dict = {f"pixel_{i}": float(v) for i, v in enumerate(features_flat)}
        
        # Create DataFrame – 1 row, 16384 columns
        df = pd.DataFrame([feature_dict])
        
        # Final safety check
        if df.shape[1] != 16384:
            raise ValueError(f"Generated wrong number of features: {df.shape[1]} (expected 16384)")
        
        return df
    
    except Exception as e:
        st.error(f"Feature extraction failed: {str(e)}")
        return None

# ────────────────────────────────────────────────
# Main content – attractive drag-and-drop uploader
# ────────────────────────────────────────────────
st.markdown("### Upload Breast Ultrasound Image")
uploaded_file = st.file_uploader(
    label="**Drag and drop ultrasound image here** or **click Browse files**",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=False,
    help="""Supported formats: PNG, JPG, JPEG
    • Drag image directly onto this area
    • Or click 'Browse files' button
    • Max size: 200 MB
    • Best results with clear, high-contrast scans""",
    key="main_ultrasound_uploader"
)

if uploaded_file is not None:
    st.markdown("---")
  
    st.subheader("Image Preview")
    st.image(uploaded_file, caption=f"{uploaded_file.name} • {uploaded_file.size / 1024:.1f} KB", use_column_width=True)
  
    if st.button("Analyze Image", type="primary", use_container_width=True):
        with st.spinner("Analyzing image..."):
            # Read image
            img_array = np.frombuffer(uploaded_file.getvalue(), np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if img is None:
                st.error("Failed to decode the uploaded image. Please try a different file.")
            else:
                features = extract_features(img)
                
                if features is None:
                    st.error("Could not extract features from this image.")
                else:
                    try:
                        raw_pred = model.predict(features)[0]
                        probs = model.predict_proba(features)[0]

                        # Handle both integer index and string label
                        if isinstance(raw_pred, (int, np.integer)):
                            pred_idx = int(raw_pred)
                            label_map = {0: "Benign", 1: "Malignant", 2: "Normal"}
                            label = label_map.get(pred_idx, f"Class {pred_idx}")
                            conf = float(probs[pred_idx]) if 0 <= pred_idx < len(probs) else 0.0
                        elif isinstance(raw_pred, str):
                            label = raw_pred.strip().title()
                            try:
                                class_names = model.classes_
                                idx = np.where(class_names == raw_pred)[0][0]
                                conf = float(probs[idx])
                            except:
                                conf = float(probs.max())  # fallback
                        else:
                            raise ValueError(f"Unexpected prediction type: {type(raw_pred)}")

                        # Display result
                        if "malignant" in label.lower():
                            cls = "malignant"
                            emoji = "🔴"
                            msg = "Potential concern detected – urgent specialist review recommended."
                        elif "benign" in label.lower():
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
                            <p style="font-size:1.6rem; margin:12px 0;">
                                Confidence: <strong>{conf:.1%}</strong>
                            </p>
                            <p>{msg}</p>
                        </div>
                        """, unsafe_allow_html=True)
                      
                        probs_df = pd.DataFrame({
                            "Class": ["Benign", "Malignant", "Normal"],
                            "Confidence (%)": probs * 100
                        }).set_index("Class")
                      
                        st.subheader("Confidence Breakdown")
                        st.bar_chart(probs_df, color="#3b82f6", height=300)
                    
                    except Exception as ve:
                        st.error(f"Prediction error:\n{str(ve)}")
                        # Uncomment for debugging
                        # st.write("Raw prediction:", raw_pred)
                        # st.write("Model classes:", getattr(model, 'classes_', 'Not available'))
                        # st.write("Features shape:", features.shape)

# ────────────────────────────────────────────────
# Footer
# ────────────────────────────────────────────────
st.markdown("""
    <div class="footer">
        <strong>BreastCare AI</strong> • Research Prototype • 2026<br>
        Trained on BUSI dataset • Not for clinical diagnosis
    </div>
""", unsafe_allow_html=True)
