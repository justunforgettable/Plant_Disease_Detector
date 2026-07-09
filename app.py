import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import json
import cv2
import os
import urllib.request

# ────────────────────────────────────────────────────────────
# PAGE CONFIG
# ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tomato Leaf Disease Detector",
    page_icon="🍅",
    layout="wide"
)

IMG_SIZE = 128
LAST_CONV_LAYER = "conv2d_2"   # same layer name used in your Colab notebook

# ────────────────────────────────────────────────────────────
# HUGGING FACE MODEL URLS — replace YOUR_USERNAME below
# ────────────────────────────────────────────────────────────
HF_USERNAME = "justunforgettable"         
HF_REPO = "Plant_Disease_Detector"       

MODEL_URL = f"co/justunforgettable/Plant_Disease_Detector/resolve/main/plant_disease_model.h5"
CLASS_INDICES_URL = f"https://huggingface.co/justunforgettable/Plant_Disease_Detector/resolve/main/class_indices.json"

MODEL_PATH = "plant_disease_model.h5"
CLASS_INDICES_PATH = "class_indices.json"

def download_if_missing(url, path):
    if not os.path.exists(path):
        with st.spinner(f"Downloading {path} (first time only)..."):
            urllib.request.urlretrieve(url, path)

# ────────────────────────────────────────────────────────────
# DISEASE INFO (fixed: Leaf_Mold instead of Leaf_Miner bug)
# ────────────────────────────────────────────────────────────
DISEASE_INFO = {
    "Tomato_healthy": {
        "display_name": "Healthy Tomato",
        "description": "The tomato plant is healthy with no signs of disease.",
        "first_aid": "No treatment needed. Continue regular watering and fertilizing.",
        "prevention": "Maintain proper watering schedule. Ensure good air circulation.",
        "severity": "None"
    },
    "Tomato_Early_blight": {
        "display_name": "Early Blight",
        "description": "Caused by Alternaria solani fungus. Shows dark brown spots with yellow rings.",
        "first_aid": "1. Remove infected leaves immediately.\n2. Apply copper-based fungicide.\n3. Avoid overhead watering.",
        "prevention": "Rotate crops. Use resistant varieties. Apply mulch.",
        "severity": "Moderate"
    },
    "Tomato_Late_blight": {
        "display_name": "Late Blight",
        "description": "Caused by Phytophthora infestans. Shows water-soaked lesions turning brown.",
        "first_aid": "1. Remove and destroy all infected plant parts.\n2. Apply chlorothalonil or mancozeb fungicide.\n3. Improve drainage around plants.",
        "prevention": "Plant in well-drained soil. Use certified disease-free seeds.",
        "severity": "High — spreads rapidly"
    },
    "Tomato_Leaf_Mold": {
        "display_name": "Leaf Mold",
        "description": "Caused by Passalora fulva fungus. Shows pale yellow spots on top of leaves and olive-green/grey mold underneath, common in humid conditions.",
        "first_aid": "1. Remove and destroy infected leaves.\n2. Improve greenhouse/room ventilation.\n3. Apply chlorothalonil or copper-based fungicide.",
        "prevention": "Reduce humidity. Avoid overhead watering. Increase plant spacing for airflow.",
        "severity": "Moderate"
    },
    "Tomato_Septoria_leaf_spot": {
        "display_name": "Septoria Leaf Spot",
        "description": "Caused by Septoria lycopersici. Shows small circular spots with dark borders.",
        "first_aid": "1. Remove infected lower leaves first.\n2. Apply mancozeb or copper fungicide.\n3. Water at the base, not on leaves.",
        "prevention": "Avoid overhead irrigation. Crop rotation every 2-3 years.",
        "severity": "Moderate"
    },
}

# ────────────────────────────────────────────────────────────
# LOAD MODEL + CLASS INDICES (cached so it loads only once)
# ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model_and_classes():
    download_if_missing(MODEL_URL, MODEL_PATH)
    download_if_missing(CLASS_INDICES_URL, CLASS_INDICES_PATH)

    model = keras.models.load_model(MODEL_PATH)
    with open(CLASS_INDICES_PATH, "r") as f:
        index_to_class = json.load(f)   # {"0": "Tomato_Early_blight", ...}
    # Build once so Grad-CAM graph works properly (important for Sequential models)
    dummy = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    _ = model(dummy)
    return model, index_to_class

model, index_to_class = load_model_and_classes()
class_names = [index_to_class[str(i)] for i in range(len(index_to_class))]
short_names = [c.replace("Tomato__", "").replace("Tomato_", "") for c in class_names]

# ────────────────────────────────────────────────────────────
# PREDICTION FUNCTION
# ────────────────────────────────────────────────────────────
def predict_disease(pil_img):
    img_resized = pil_img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img_resized).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)
    predicted_index = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][predicted_index]) * 100
    predicted_class = index_to_class[str(predicted_index)]

    return predicted_class, confidence, predictions[0], img_array

# ────────────────────────────────────────────────────────────
# GRAD-CAM FUNCTIONS
# ────────────────────────────────────────────────────────────
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.outputs[0]]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array, training=False)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def overlay_gradcam(pil_img, heatmap):
    img = np.array(pil_img.resize((IMG_SIZE, IMG_SIZE)))
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    superimposed = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)
    return superimposed

# ────────────────────────────────────────────────────────────
# UI
# ────────────────────────────────────────────────────────────
st.title("🍅 Tomato Leaf Disease Detector")
st.markdown("Upload a photo of a tomato leaf and the model will detect the disease, "
            "show its confidence, and explain **which part of the leaf** it focused on (Grad-CAM).")

col_left, col_right = st.columns([1, 1])

with col_left:
    uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    pil_img = Image.open(uploaded_file).convert("RGB")

    with col_left:
        st.image(pil_img, caption="Uploaded Image", use_container_width=True)

    predicted_class, confidence, all_probs, img_array = predict_disease(pil_img)
    disease_info = DISEASE_INFO.get(predicted_class, {})

    with col_right:
        st.subheader(f"🩺 Detected: {disease_info.get('display_name', predicted_class)}")
        st.metric("Confidence", f"{confidence:.1f}%")
        st.write(f"**Severity:** {disease_info.get('severity', 'Unknown')}")

        st.write("**Description:**")
        st.write(disease_info.get("description", "-"))

        st.write("**First Aid / Treatment:**")
        st.write(disease_info.get("first_aid", "-"))

        st.write("**Prevention:**")
        st.write(disease_info.get("prevention", "-"))

    st.divider()

    # Confidence bar chart
    st.subheader("📊 Model Confidence per Class")
    chart_data = {short_names[i]: float(all_probs[i]) * 100 for i in range(len(short_names))}
    st.bar_chart(chart_data)

    # Grad-CAM visualization
    st.subheader("🔍 Grad-CAM: Where the model looked")
    heatmap = make_gradcam_heatmap(img_array, model, LAST_CONV_LAYER)
    gradcam_img = overlay_gradcam(pil_img, heatmap)

    gc1, gc2 = st.columns(2)
    with gc1:
        st.image(pil_img.resize((IMG_SIZE, IMG_SIZE)), caption="Original", use_container_width=True)
    with gc2:
        st.image(gradcam_img, caption="Grad-CAM Heatmap", use_container_width=True)

else:
    st.info("👆 Upload a tomato leaf image to get started.")

st.divider()
st.caption("Built with TensorFlow + Streamlit | CNN trained on PlantVillage dataset")
