import streamlit as st
import numpy as np
from PIL import Image
import pickle
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# ─── Config ───
st.set_page_config(page_title="Breast Cancer Classifier", layout="wide")

MODEL_PATH = "saved_models/rf_breast_model.pkl"
LE_PATH    = "saved_models/label_encoder.pkl"
EXTRACTOR_PATH = "saved_models/feature_extractor.h5"

IMG_SIZE = (224, 224)

# ─── Load model & objects (cached so only loads once) ───
@st.cache_resource
def load_model_and_tools():
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(LE_PATH, 'rb') as f:
        le = pickle.load(f)
    
    extractor = None
    if os.path.exists(EXTRACTOR_PATH):
        extractor = tf.keras.models.load_model(EXTRACTOR_PATH, compile=False)
    else:
        # Fallback: recreate extractor (slower first time)
        from tensorflow.keras.applications import ResNet50
        from tensorflow.keras.models import Model
        base = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
        extractor = Model(inputs=base.input, outputs=base.output)
    
    return model, le, extractor

model, le, extractor = load_model_and_tools()

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

# ─── UI ───
st.title("Breast Ultrasound & Mammogram Cancer Classifier")
st.markdown("Upload an image (ultrasound or mammogram) to get a prediction.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_column_width=True)

    with st.spinner("Analyzing... (may take 5–20 seconds)"):
        label, prob = predict_image(image)

    if label.lower() == "malignant":
        st.error(f"**Prediction: MALIGNANT**\nMalignant confidence: **{prob:.1%}**")
    else:
        st.success(f"**Prediction: {label.upper()}**\nMalignant confidence: **{prob:.1%}**")

    st.info("Note: This is a research prototype — not for clinical diagnosis.")
