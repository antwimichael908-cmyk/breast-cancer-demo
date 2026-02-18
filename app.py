# app.py
import streamlit as st
import pandas as pd
import joblib
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

# ── Load your best model ───────────────────────────────
MODEL_PATH = "best_tuned_rf_breast_ultrasound.pkl"

try:
    model = joblib.load(MODEL_PATH)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ── Reuse your feature extraction function ─────────────
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
    
    # Shape features (simple)
    _, thresh = cv2.threshold(img_resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = perimeter = circularity = 0.0
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
    
    features = {
        'mean_intensity': mean_intensity,
        'std_intensity': std_intensity,
        'area': area,
        'perimeter': perimeter,
        'circularity': circularity,
        **texture_feats
    }
    return pd.DataFrame([features])

# ── Streamlit UI ───────────────────────────────────────
st.title("Breast Ultrasound Cancer Classifier")
st.write("Upload an ultrasound image (PNG/JPG) to predict: Benign, Malignant, or Normal")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    st.image(img, channels="BGR", caption="Uploaded Image", use_column_width=True)
    
    if st.button("Predict"):
        with st.spinner("Extracting features and predicting..."):
            features_df = extract_features(img)
            
            # Make sure columns match training
            # (you can print model.feature_names_in_ if needed)
            prediction = model.predict(features_df)[0]
            probabilities = model.predict_proba(features_df)[0]
            
            labels = {0: "Benign", 1: "Malignant", 2: "Normal"}
            pred_label = labels[prediction]
            
            st.subheader(f"Prediction: **{pred_label}**")
            st.write(f"Confidence: {probabilities[prediction]:.1%}")
            
            probs_df = pd.DataFrame({
                "Class": list(labels.values()),
                "Probability": probabilities
            })
            st.bar_chart(probs_df.set_index("Class"))
