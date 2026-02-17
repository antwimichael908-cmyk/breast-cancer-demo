# app.py - Breast Cancer Classifier with Training & Professional UI

import streamlit as st
import os
import zipfile
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from tqdm import tqdm
from PIL import Image

# ────────────────────────────────────────────────
# Page Config & Professional Theme
# ────────────────────────────────────────────────
st.set_page_config(
    page_title="Breast Imaging Classifier",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    .stButton>button {
        background-color: #0d6efd;
        color: white;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
    }
    .stButton>button:hover {background-color: #0b5ed7;}
    .result-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin: 1rem 0;
    }
    .malignant {border-left: 6px solid #dc3545;}
    .benign    {border-left: 6px solid #198754;}
    h1, h2, h3 {color: #0d6efd;}
    .disclaimer {font-size: 0.9rem; color: #6c757d; margin-top: 2rem;}
    </style>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────
# CONFIGURATION (change these if needed)
# ────────────────────────────────────────────────
SAVE_DIR = "saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)

MODEL_PATH     = os.path.join(SAVE_DIR, "rf_breast_model.pkl")
LE_PATH        = os.path.join(SAVE_DIR, "label_encoder.pkl")
IMG_SIZE       = (224, 224)
BATCH_SIZE     = 32

# ────────────────────────────────────────────────
# Load / Create Extractor (recreated – no .h5 needed)
# ────────────────────────────────────────────────
@st.cache_resource
def get_feature_extractor():
    st.info("Initializing ResNet50 feature extractor... (one-time)")
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    extractor = Model(inputs=base_model.input, outputs=base_model.output)
    return extractor

extractor = get_feature_extractor()

# ────────────────────────────────────────────────
# Load trained model & encoder
# ────────────────────────────────────────────────
@st.cache_resource
def load_trained_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LE_PATH):
        st.warning("Trained model files not found in 'saved_models/'. Please train first.")
        return None, None
    
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(LE_PATH, 'rb') as f:
        le = pickle.load(f)
    
    st.success("Trained model loaded successfully")
    return model, le

model, le = load_trained_model()

# ────────────────────────────────────────────────
# Prediction Function
# ────────────────────────────────────────────────
def predict_image(pil_image):
    if model is None or le is None:
        return "Model not loaded", 0.0
    
    img = pil_image.resize(IMG_SIZE)
    arr = img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    
    feats = extractor.predict(arr, verbose=0)
    pooled = tf.keras.layers.GlobalAveragePooling2D()(feats)
    features = pooled.numpy()
    
    pred_class = model.predict(features)[0]
    pred_proba = model.predict_proba(features)[0]
    
    label = le.inverse_transform([pred_class])[0]
    malignant_prob = pred_proba[1] if len(pred_proba) > 1 else pred_proba[0]
    
    return label, malignant_prob

# ────────────────────────────────────────────────
# SIDEBAR
# ────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/medical-doctor.png", width=80)
    st.title("Breast Imaging AI")
    st.markdown("**Research Prototype v1.0**")
    st.divider()
    st.subheader("About")
    st.markdown("""
    • Deep features from ResNet50  
    • Classifier: Random Forest  
    • Trained on ultrasound + mammogram images  
    • For research/educational use only
    """)
    st.divider()
    st.caption("© 2026 Research Project")

# ────────────────────────────────────────────────
# MAIN INTERFACE
# ────────────────────────────────────────────────
st.title("Breast Imaging Cancer Classifier")
st.markdown("Upload an ultrasound or mammogram image for AI-assisted prediction.")

tab1, tab2 = st.tabs(["📤 Predict", "🛠 Train / Retrain"])

# ─── PREDICT TAB ─────────────────────────────────────
with tab1:
    uploaded_file = st.file_uploader(
        "Upload image (PNG/JPG/JPEG)",
        type=["png", "jpg", "jpeg"],
        help="Recommended: clear region-of-interest images"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded image", use_column_width=True)

        if st.button("Analyze Image", type="primary", use_container_width=True):
            if model is None:
                st.error("No trained model found. Please go to 'Train / Retrain' tab first.")
            else:
                with st.spinner("Analyzing... (5–30 seconds)"):
                    label, prob = predict_image(image)

                st.markdown("### Result")
                card_class = "malignant" if label.lower() == "malignant" else "benign"

                st.markdown(f'<div class="result-card {card_class}">', unsafe_allow_html=True)
                
                if label.lower() == "malignant":
                    st.markdown("### Malignant")
                    st.markdown("**Suggestive of malignancy**")
                    st.metric("Malignant Probability", f"{prob:.1%}", delta_color="inverse")
                else:
                    st.markdown("### Benign")
                    st.markdown("**Likely benign**")
                    st.metric("Malignant Probability", f"{prob:.1%}")

                st.markdown('</div>', unsafe_allow_html=True)

                st.info("**Note:** This is a research prototype. Always confirm with clinical evaluation.")

# ─── TRAIN / RETRAIN TAB ───────────────────────────────
with tab2:
    st.header("Train or Retrain Model")
    st.warning("Training on cloud is slow and resource-limited. Use local machine for full training.")
    
    if st.button("Start Training (Demo Mode)", disabled=True):
        st.info("Full training disabled on cloud for performance reasons.")
    
    st.markdown("**Recommended workflow:**")
    st.markdown("1. Train locally on your computer using the original script")
    st.markdown("2. Copy saved_models/ folder to deployment folder")
    st.markdown("3. Push to GitHub → redeploy app")

# ─── FOOTER DISCLAIMER ────────────────────────────────
st.markdown("""
<div class="disclaimer" style="margin-top: 3rem;">
    <strong>Disclaimer:</strong> This is a research/educational prototype only.  
    It is not a medical device and should never replace professional clinical diagnosis.
</div>
""", unsafe_allow_html=True)
