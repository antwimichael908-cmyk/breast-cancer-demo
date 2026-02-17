# app.py - Professional Breast Cancer Image Classifier

import streamlit as st
import numpy as np
from PIL import Image
import pickle
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import MobileNetV2
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
from tensorflow.keras.models import Model

# ────────────────────────────────────────────────
# Page Config & Theme
# ────────────────────────────────────────────────
st.set_page_config(
    page_title="Breast Imaging Classifier",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    .stButton>button {
        background-color: #0d6efd;
        color: white;
        border: none;
        padding: 0.6rem 1.5rem;
        border-radius: 8px;
        font-weight: 500;
    }
    .stButton>button:hover {
        background-color: #0b5ed7;
    }
    .result-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin: 1rem 0;
    }
    .malignant {border-left: 6px solid #dc3545;}
    .benign    {border-left: 6px solid #198754;}
    h1 {color: #0d6efd; font-weight: 600;}
    .disclaimer {font-size: 0.9rem; color: #6c757d; margin-top: 2rem;}
    </style>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────
# Load model & encoder
# ────────────────────────────────────────────────
@st.cache_resource
def load_model_and_encoder():
    try:
        with open("saved_models/rf_breast_model.pkl", 'rb') as f:
            model = pickle.load(f)
        with open("saved_models/label_encoder.pkl", 'rb') as f:
            le = pickle.load(f)
        return model, le
    except FileNotFoundError:
        st.error("Model files not found in 'saved_models/' folder.")
        st.stop()

# ────────────────────────────────────────────────
# Feature extractor (recreated, no .h5 file)
# ────────────────────────────────────────────────
@st.cache_resource
def get_feature_extractor():
    with st.spinner("Initializing deep feature extractor..."):
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        extractor = Model(inputs=base_model.input, outputs=base_model.output)
    return extractor

model, le = load_model_and_encoder()
extractor = get_feature_extractor()

# ────────────────────────────────────────────────
# Prediction function
# ────────────────────────────────────────────────
def predict_image(pil_image):
    img = pil_image.resize((224, 224))
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
    st.image("https://img.icons8.com/color/96/000000/medical-doctor.png", width=80)
    st.title("Breast Imaging AI")
    st.markdown("**Version 1.0** – Research Prototype")
    st.divider()
    st.subheader("About this tool")
    st.markdown("""
    • Uses ResNet50 deep features + Random Forest  
    • Trained on ultrasound & mammogram images  
    • Best used as second opinion support  
    """)
    st.divider()
    st.caption("For research & educational purposes only")

# ────────────────────────────────────────────────
# MAIN CONTENT
# ────────────────────────────────────────────────
st.title("Breast Imaging Cancer Classifier")
st.markdown("Upload an ultrasound or mammogram image to receive an AI-assisted prediction.")

col_upload, col_info = st.columns([3, 1])

with col_upload:
    uploaded_file = st.file_uploader(
        "Select breast imaging file",
        type=["png", "jpg", "jpeg"],
        help="Accepted formats: PNG, JPG, JPEG. Max recommended size: 10 MB"
    )

with col_info:
    st.info("**Important**  \nThis tool is **not** a substitute for professional medical diagnosis.")

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        
        # Show image
        st.image(image, caption="Uploaded image", use_column_width=True)

        # Analyze button
        if st.button("Analyze Image", type="primary", use_container_width=True):
            with st.spinner("Processing image... (may take 5–30 seconds on first use)"):
                label, prob = predict_image(image)

            st.markdown("### Prediction Result")

            card_class = "malignant" if label.lower() == "malignant" else "benign"

            with st.container():
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

            st.info("""
            **Interpretation guidance**  
            • High malignant probability → further clinical evaluation recommended  
            • Low probability → still requires radiologist review  
            This is an AI research prototype with known limitations.
            """)

    except Exception as e:
        st.error("Error during image processing")
        st.write("Details:", str(e))

# Footer disclaimer
st.markdown("""
<div class="disclaimer">
    <strong>Disclaimer:</strong> This application is for research and educational purposes only. 
    It is not a certified medical device and should never be used as the sole basis for clinical decisions.
</div>
""", unsafe_allow_html=True)
