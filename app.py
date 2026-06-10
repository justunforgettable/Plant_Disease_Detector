import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# =========================
# Page Settings
# =========================
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="centered"
)

# =========================
# Load Model
# =========================
model = tf.keras.models.load_model("plant_disease_model.h5")

# =========================
# Load Class Names
# =========================
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

# =========================
# Disease Information
# =========================
DISEASE_INFO = {
    "Tomato_Early_blight": {
        "severity": "Moderate",
        "description": "A fungal disease that causes brown spots with concentric rings on tomato leaves.",
        "treatment": "Remove infected leaves and apply copper-based fungicide.",
        "prevention": "Practice crop rotation, proper spacing, and avoid overhead watering."
    },

    "Tomato_Late_blight": {
        "severity": "High",
        "description": "A severe disease causing dark lesions and rapid damage to leaves and fruits.",
        "treatment": "Remove infected plant parts immediately and apply fungicide.",
        "prevention": "Use healthy seedlings and avoid excessive moisture."
    },

    "Tomato_Leaf_Mold": {
        "severity": "Moderate",
        "description": "A fungal infection that produces yellow patches and mold growth on leaves.",
        "treatment": "Improve ventilation and use recommended fungicide.",
        "prevention": "Reduce humidity and ensure proper airflow."
    },

    "Tomato_Septoria_leaf_spot": {
        "severity": "Moderate",
        "description": "Characterized by small circular spots with dark borders on leaves.",
        "treatment": "Remove infected leaves and apply fungicide.",
        "prevention": "Avoid splashing water on leaves and rotate crops."
    },

    "Tomato_healthy": {
        "severity": "None",
        "description": "The tomato plant appears healthy and disease-free.",
        "treatment": "No treatment required.",
        "prevention": "Continue proper watering, nutrition, and routine care."
    }
}

IMG_SIZE = 128

# =========================
# Title
# =========================
st.title("🌿 Plant Disease Detection System")

st.markdown("""
Upload a tomato leaf image and the AI model will detect the disease,
show confidence score, severity level, treatment suggestions,
and prevention methods.
""")

# =========================
# Sidebar
# =========================
st.sidebar.title("Project Information")

st.sidebar.info("""
Model: CNN (Convolutional Neural Network)

Accuracy: 89.5%

Classes:
- Tomato Early Blight
- Tomato Late Blight
- Tomato Leaf Mold
- Tomato Septoria Leaf Spot
- Healthy Tomato Leaf
""")

# =========================
# File Upload
# =========================
uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png"]
)

# =========================
# Prediction
# =========================
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Leaf Image",
        use_container_width=True
    )

    # Resize
    img = image.resize((IMG_SIZE, IMG_SIZE))

    # Convert to array
    img_array = np.array(img)

    # Normalize
    img_array = img_array / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array, verbose=0)

    predicted_index = np.argmax(prediction)

    confidence = float(
        prediction[0][predicted_index] * 100
    )

    predicted_class = class_indices[
        str(int(predicted_index))
    ]

    disease = DISEASE_INFO[predicted_class]

    # =========================
    # Results
    # =========================
    st.success(
        f"Detected Disease: {predicted_class}"
    )

    st.info(
        f"Confidence: {confidence:.2f}%"
    )

    st.progress(confidence / 100)

    st.subheader("Disease Details")

    st.write("### Severity")
    st.write(disease["severity"])

    st.write("### Description")
    st.write(disease["description"])

    st.write("### Treatment")
    st.write(disease["treatment"])

    st.write("### Prevention")
    st.write(disease["prevention"])

# =========================
# Footer
# =========================
st.markdown("---")
st.caption(
    "Plant Disease Detection using Deep Learning (CNN)"
)