import streamlit as st
import numpy as np
from PIL import Image
import pickle
import os
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array

# ────────────────────────────────────────────────
# CONFIG – paths relative to the deployed app
# ────────────────────────────────────────────────
MODEL_PATH = "saved_models/rf_breast_model.pkl"
LE_PATH    = "saved_models/label_encoder.pkl"

IMG_SIZE   = (224, 224)

# ────────────────────────────────────────────────
# Load trained Random Forest + LabelEncoder (cached)
# ────────────────────────────────────────────────
@st.cache_resource
def load_model_and_encoder():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found: {MODEL_PATH}")
        st.stop()
    if not os.path.exists(LE_PATH):
        st.error(f"Label encoder not found: {LE_PATH}")
        st.stop()

    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(LE_PATH, 'rb') as f:
        le = pickle.load(f)
    
    return model, le

# ────────────────────────────────────────────────
# Create feature extractor on-the-fly (no .h5 file needed)
# ────────────────────────────────────────────────
@st.cache_resource
def get_feature_extractor():
    st.info("Creating ResNet50 feature extractor... (first time only)")
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

    # Extract features
    feats = extractor.predict(arr, verbose=0)
    pooled = tf.keras.layers.GlobalAveragePooling2D()(feats)
    features = pooled.numpy()

    # Predict
    pred_class = model.predict(features)[0]
    pred_proba = model.predict_proba(features)[0]

    label = le.inverse_transform([pred_class])[0]
    malignant_prob = pred_proba[1] if len(pred_proba) > 1 else pred_proba[0]

    return label, malignant_prob

# ────────────────────────────────────────────────
# Streamlit UI
# ────────────────────────────────────────────────
st.set_page_config(page_title="Breast Cancer Classifier", layout="centered")

st.title("Breast Ultrasound & Mammogram Cancer Classifier")
st.markdown("""
Upload an ultrasound or mammogram image to get a prediction.  
**Note:** This is a research prototype — not for clinical use.
""")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_column_width=True)

    with st.spinner("Analyzing image... (may take 5–30 seconds on first run)"):
        try:
            label, prob = predict_image(image)
            
            if label.lower() == "malignant":
                st.error(f"**Prediction: MALIGNANT**")
                st.markdown(f"Malignant confidence: **{prob:.1%}**")
            else:
                st.success(f"**Prediction: {label.upper()}**")
                st.markdown(f"Malignant confidence: **{prob:.1%}**")
                
            st.info("Confidence is based on the model's probability output.")
        except Exception as e:
            st.error("Error during prediction")
            st.write(str(e))

st.markdown("---")
st.caption("Built with ResNet50 features + Random Forest | Prototype only")
