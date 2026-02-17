import streamlit as st
import numpy as np
from PIL import Image
import pickle
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model

# ────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────
MODEL_PATH = "saved_models/rf_breast_model.pkl"
LE_PATH    = "saved_models/label_encoder.pkl"

IMG_SIZE   = (224, 224)

st.set_page_config(
    page_title="Breast Cancer Image Classifier",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ────────────────────────────────────────────────
# Load model & encoder (cached – loads only once)
# ────────────────────────────────────────────────
@st.cache_resource
def load_model_and_encoder():
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(LE_PATH, 'rb') as f:
            le = pickle.load(f)
        return model, le
    except FileNotFoundError:
        st.error("Model files not found in 'saved_models/' folder.")
        st.stop()

# ────────────────────────────────────────────────
# Create ResNet50 feature extractor (no saved .h5 needed)
# ────────────────────────────────────────────────
@st.cache_resource
def get_feature_extractor():
    st.info("Initializing ResNet50 feature extractor... (one-time operation)")
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    extractor = Model(inputs=base_model.input, outputs=base_model.output)
    return extractor

model, le = load_model_and_encoder()
extractor = get_feature_extractor()

# ────────────────────────────────────────────────
# Prediction function
# ────────────────────────────────────────────────
def predict_image(pil_image):
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
# USER INTERFACE
# ────────────────────────────────────────────────
st.title("🩺 Breast Ultrasound & Mammogram Cancer Classifier")
st.markdown("""
This tool classifies breast ultrasound or mammogram images as **benign** or **malignant**.  
**Important disclaimer:**  
This is a **research prototype** — NOT a medical device.  
Always consult a qualified radiologist or oncologist for diagnosis.
""")

with st.sidebar:
    st.header("About")
    st.markdown("""
    - Model: ResNet50 features + Random Forest  
    - Trained on: BUSI + CBIS-DDSM (ultrasound + mammogram)  
    - Prediction time: 5–30 seconds (first time slower)  
    """)

uploaded_file = st.file_uploader(
    "Upload breast ultrasound or mammogram image",
    type=["png", "jpg", "jpeg"],
    help="Supported: PNG, JPG, JPEG. Max size ~10 MB recommended."
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded image", use_column_width=True)

        with st.spinner("Analyzing image..."):
            label, prob = predict_image(image)

        st.subheader("Prediction Result")

        col1, col2 = st.columns(2)
        with col1:
            if label.lower() == "malignant":
                st.error("**Malignant** (suggestive of cancer)")
            else:
                st.success(f"**{label.capitalize()}** (likely benign)")

        with col2:
            st.metric("Malignant Confidence", f"{prob:.1%}")

        st.info("This confidence score is the model's probability output. False results are possible.")

    except Exception as e:
        st.error("Error during analysis")
        st.write("Details:", str(e))

st.markdown("---")
st.caption("Prototype built for research/educational purposes | February 2026")
