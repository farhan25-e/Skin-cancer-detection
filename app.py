import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Skin Cancer Detection", page_icon="🩺", layout="centered")
st.title("🩺 Skin Cancer Detection")
st.warning("Educational demo only. This is NOT a medical diagnosis. Please consult a dermatologist.")

@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('skin_cancer_model.keras', compile=False)
    with open('class_names.json') as f:
        names = json.load(f)
    return model, names

try:
    model, class_names = load_assets()
    height, width = model.input_shape[1:3]

    uploaded = st.file_uploader("Upload a skin lesion image", type=["jpg", "jpeg", "png"])

    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)

        x = np.array(image.resize((width, height)), dtype="float32")[None, ...]
        probs = model.predict(x, verbose=0)[0]
        top = int(np.argmax(probs))

        st.subheader(f"Prediction: {class_names[top].upper()}")
        st.metric("Confidence", f"{probs[top] * 100:.1f}%")
        st.bar_chart({class_names[i]: float(p) for i, p in enumerate(probs)})

except Exception as e:
    st.error(f"Error loading model: {e}")