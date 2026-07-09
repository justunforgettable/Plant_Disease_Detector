
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import json
import os
import urllib.request
import pandas as pd
import plotly.express as px

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Tomato Leaf Disease Detector",
    page_icon="🍅",
    layout="wide"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #f8fff8, #eef8f1);
}

.hero {
    padding: 2rem;
    border-radius: 20px;
    background: linear-gradient(135deg, #1B5E20, #43A047);
    color: white;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0px 10px 30px rgba(0,0,0,0.15);
}

.info-card {
    background: white;
    padding: 20px;
    border-radius: 18px;
    box-shadow: 0px 5px 20px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.title("🍅 Plant AI")

    st.markdown("""
    ### Features
    ✅ Disease Detection  
    ✅ Confidence Score  
    ✅ Treatment Suggestions  
    ✅ Prevention Tips
    """)

    st.markdown("---")

    st.caption("Dataset: PlantVillage")
    st.caption("Framework: TensorFlow + Streamlit")
    st.caption("Model: CNN")

# ============================================================
# CONSTANTS
# ============================================================
IMG_SIZE = 128

HF_USERNAME = "justunforgettable"
HF_REPO = "Plant_Disease_Detector"

MODEL_URL = (
    f"https://huggingface.co/"
    f"{HF_USERNAME}/"
    f"{HF_REPO}/"
    f"resolve/main/plant_disease_model.h5"
)

CLASS_INDICES_URL = (
    f"https://huggingface.co/"
    f"{HF_USERNAME}/"
    f"{HF_REPO}/"
    f"resolve/main/class_indices.json"
)

MODEL_PATH = "plant_disease_model.h5"
CLASS_INDICES_PATH = "class_indices.json"

# ============================================================
# DOWNLOAD FILES
# ============================================================
def download_if_missing(url, path):
    if not os.path.exists(path):
        with st.spinner(f"Downloading {path}... Please wait."):
            urllib.request.urlretrieve(url, path)

# ============================================================
# DISEASE INFORMATION
# ============================================================
DISEASE_INFO = {

    "Tomato_healthy": {
        "display_name": "Healthy Tomato",
        "description":
            "The tomato plant is healthy and shows no visible signs "
            "of disease or infection.",
        "first_aid":
            "No treatment required. Continue normal plant care.",
        "prevention":
            "Maintain proper watering, sunlight and nutrition.",
        "severity":
            "None"
    },

    "Tomato_Early_blight": {
        "display_name": "Early Blight",
        "description":
            "A fungal disease causing dark brown spots with "
            "yellow rings on leaves.",
        "first_aid":
            "1. Remove infected leaves.\n"
            "2. Apply copper fungicide.\n"
            "3. Avoid overhead watering.",
        "prevention":
            "Rotate crops and use resistant varieties.",
        "severity":
            "Moderate"
    },

    "Tomato_Late_blight": {
        "display_name": "Late Blight",
        "description":
            "A rapidly spreading disease that causes water-soaked "
            "lesions and browning of leaves.",
        "first_aid":
            "1. Remove infected plant parts.\n"
            "2. Apply fungicide.\n"
            "3. Improve drainage.",
        "prevention":
            "Use disease-free seeds and well-drained soil.",
        "severity":
            "High"
    },

    "Tomato_Leaf_Mold": {
        "display_name": "Leaf Mold",
        "description":
            "Caused by fungus under humid conditions. "
            "Produces yellow spots and grey mold.",
        "first_aid":
            "1. Remove infected leaves.\n"
            "2. Improve ventilation.\n"
            "3. Apply fungicide.",
        "prevention":
            "Reduce humidity and avoid overhead irrigation.",
        "severity":
            "Moderate"
    },

    "Tomato_Septoria_leaf_spot": {
        "display_name": "Septoria Leaf Spot",
        "description":
            "Small circular spots with dark borders on leaves.",
        "first_aid":
            "1. Remove infected leaves.\n"
            "2. Apply fungicide.\n"
            "3. Water at soil level.",
        "prevention":
            "Avoid wet foliage and rotate crops.",
        "severity":
            "Moderate"
    }
}

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model_and_classes():

    download_if_missing(MODEL_URL, MODEL_PATH)
    download_if_missing(CLASS_INDICES_URL,
                        CLASS_INDICES_PATH)

    model = keras.models.load_model(MODEL_PATH)

    with open(CLASS_INDICES_PATH, "r") as f:
        index_to_class = json.load(f)

    dummy = np.zeros(
        (1, IMG_SIZE, IMG_SIZE, 3),
        dtype=np.float32
    )
    _ = model(dummy)

    return model, index_to_class


model, index_to_class = load_model_and_classes()

class_names = [
    index_to_class[str(i)]
    for i in range(len(index_to_class))
]

short_names = [
    c.replace("Tomato_", "")
    for c in class_names
]

# ============================================================
# PREDICTION FUNCTION
# ============================================================
def predict_disease(pil_img):

    img = pil_img.resize(
        (IMG_SIZE, IMG_SIZE)
    )

    img_array = (
        np.array(img)
        .astype("float32")
        / 255.0
    )

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    predictions = model.predict(
        img_array,
        verbose=0
    )

    predicted_index = int(
        np.argmax(predictions[0])
    )

    confidence = float(
        predictions[0][predicted_index]
    ) * 100

    predicted_class = (
        index_to_class[str(predicted_index)]
    )

    return (
        predicted_class,
        confidence,
        predictions[0]
    )

# ============================================================
# HERO SECTION
# ============================================================
st.markdown("""
<div class="hero">
    <h1>🍅 Tomato Leaf Disease Detector</h1>
    <h3>AI-Powered Plant Health Analysis System</h3>
    <p>
        Upload a tomato leaf image and receive
        instant disease diagnosis,
        confidence score and treatment guidance.
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# UPLOAD SECTION
# ============================================================
col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown("### 📤 Upload Leaf Image")

    uploaded_file = st.file_uploader(
        "",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear tomato leaf image."
    )

# ============================================================
# PREDICTION UI
# ============================================================
if uploaded_file is not None:

    pil_img = Image.open(
        uploaded_file
    ).convert("RGB")

    with col_left:
        st.image(
            pil_img,
            caption="Uploaded Image",
            use_container_width=True
        )

    (
        predicted_class,
        confidence,
        all_probs
    ) = predict_disease(pil_img)

    disease_info = DISEASE_INFO.get(
        predicted_class,
        {}
    )

    with col_right:

        st.markdown(f"""
        <div class="info-card">
            <h2>
                🩺
                {disease_info.get(
                    'display_name',
                    predicted_class
                )}
            </h2>

            <h3>
                Confidence:
                {confidence:.1f}%
            </h3>
        </div>
        """,
        unsafe_allow_html=True)

        severity = disease_info.get(
            "severity",
            "Unknown"
        )

        if "High" in severity:
            st.error(
                f"🚨 Severity: {severity}"
            )
        elif "Moderate" in severity:
            st.warning(
                f"⚠️ Severity: {severity}"
            )
        else:
            st.success(
                f"✅ Severity: {severity}"
            )

        tab1, tab2, tab3 = st.tabs(
            [
                "📖 Description",
                "💊 Treatment",
                "🛡 Prevention"
            ]
        )

        with tab1:
            st.write(
                disease_info.get(
                    "description",
                    "-"
                )
            )

        with tab2:
            st.write(
                disease_info.get(
                    "first_aid",
                    "-"
                )
            )

        with tab3:
            st.write(
                disease_info.get(
                    "prevention",
                    "-"
                )
            )

    st.divider()

    st.subheader(
        "📊 Model Confidence"
    )

    df = pd.DataFrame({
        "Disease":
            short_names,
        "Confidence":
            [
                float(p) * 100
                for p in all_probs
            ]
    })

    fig = px.bar(
        df,
        x="Disease",
        y="Confidence",
        text="Confidence",
        title="Prediction Confidence by Class"
    )

    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside"
    )

    fig.update_layout(
        yaxis_title="Confidence (%)",
        xaxis_title="Disease Classes"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

else:

    st.markdown("""
    <div class="info-card"
         style="text-align:center;">
        <h2>
            🍅 Welcome to Plant AI
        </h2>

        <p>
            Upload a clear image of
            a tomato leaf to begin.
        </p>

        <p>
            Supported formats:
            JPG, JPEG, PNG
        </p>
    </div>
    """, unsafe_allow_html=True)

st.divider()

st.caption(
    "Built with TensorFlow + Streamlit | "
    "CNN trained on PlantVillage Dataset"
)
```
