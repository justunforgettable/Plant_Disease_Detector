import streamlit as st
import numpy as np
from tensorflow import keras
from PIL import Image
import json
import os
import urllib.request
import pandas as pd
import plotly.express as px
import time


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Plant AI - Tomato Disease Detector",
    page_icon="🍅",
    layout="wide"
)


# ============================================================
# MODERN UI CSS
# ============================================================

st.markdown("""
<style>

/* Main background */
.stApp {
    background: linear-gradient(
        135deg,
        #fff0f6 0%,
        #f0fff4 50%,
        #e8f5e9 100%
    );
}


/* Remove default padding */
.block-container {
    padding-top: 2rem;
}


/* Hero animation */
.hero {

    background:
    linear-gradient(
        135deg,
        rgba(255,105,180,0.85),
        rgba(46,125,50,0.85)
    );

    padding: 40px;

    border-radius: 30px;

    color:white;

    text-align:center;

    box-shadow:
    0 15px 40px rgba(0,0,0,0.18);

    animation:
    fadeIn 1.5s ease-in-out;

}


/* Glass cards */

.glass-card {

    background:
    rgba(255,255,255,0.35);

    backdrop-filter:
    blur(15px);

    border-radius:
    25px;

    padding:
    25px;

    border:
    1px solid rgba(255,255,255,0.5);

    box-shadow:
    0 10px 30px rgba(0,0,0,0.12);

}


/* Feature cards */

.feature-card {

    background:
    rgba(255,255,255,0.45);

    backdrop-filter:
    blur(10px);

    padding:
    18px;

    border-radius:
    20px;

    text-align:center;

    box-shadow:
    0 8px 25px rgba(0,0,0,0.1);

}


/* Upload box */

[data-testid="stFileUploader"] {

    background:
    rgba(255,255,255,0.45);

    border-radius:
    20px;

    padding:
    10px;

}


/* Animation */

@keyframes fadeIn {

from {

opacity:0;

transform:
translateY(20px);

}


to {

opacity:1;

transform:
translateY(0);

}

}


/* Hide streamlit footer */

footer {
visibility:hidden;
}

/* Information section styling */

.stTabs [data-baseweb="tab"] {
    font-size: 18px;
    font-weight: 600;
}


.info-text {
    font-size: 17px;
    line-height: 1.7;
    padding: 10px;
}


</style>

""",
unsafe_allow_html=True)



# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🍅 Plant AI")

    st.write(
    """
    **Deep Learning Plant Disease Detector**

    Dataset:
    PlantVillage

    Model:
    CNN

    Framework:
    TensorFlow
    """
    )


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
# DOWNLOAD MODEL
# ============================================================


def download_if_missing(url,path):

    if not os.path.exists(path):

        with st.spinner(
            f"Loading {path}..."
        ):

            urllib.request.urlretrieve(
                url,
                path
            )



# -------------------------------
# Disease Information
# -------------------------------

st.divider()


tab1, tab2, tab3 = st.tabs(
    [
        "📖 Description",
        "💊 Treatment",
        "🛡 Prevention"
    ]
)


with tab1:

    st.markdown(
        f"""
        <div class="info-text">
        <h3>📖 Disease Description</h3>
        {info.get("description", "-")}
        </div>
        """,
        unsafe_allow_html=True
    )


with tab2:

    st.markdown(
        f"""
        <div class="info-text">
        <h3>💊 Recommended Treatment</h3>
        {info.get("first_aid", "-")}
        </div>
        """,
        unsafe_allow_html=True
    )


with tab3:

    st.markdown(
        f"""
        <div class="info-text">
        <h3>🛡 Prevention Methods</h3>
        {info.get("prevention", "-")}
        </div>
        """,
        unsafe_allow_html=True
    )
# ============================================================
# LOAD MODEL + CLASS INDEX
# ============================================================

@st.cache_resource
def load_model_and_classes():

    download_if_missing(
        MODEL_URL,
        MODEL_PATH
    )

    download_if_missing(
        CLASS_INDICES_URL,
        CLASS_INDICES_PATH
    )


    model = keras.models.load_model(
        MODEL_PATH
    )


    with open(CLASS_INDICES_PATH,"r") as f:
        index_to_class = json.load(f)


    # Build model once
    dummy = np.zeros(
        (1, IMG_SIZE, IMG_SIZE, 3),
        dtype=np.float32
    )

    model(dummy)


    return model,index_to_class



model,index_to_class = load_model_and_classes()



class_names = [
    index_to_class[str(i)]
    for i in range(len(index_to_class))
]


short_names = [
    x.replace(
        "Tomato_",
        ""
    )
    for x in class_names
]



# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_disease(image):

    img = image.resize(
        (IMG_SIZE,IMG_SIZE)
    )


    img_array = (
        np.array(img)
        .astype("float32")
        /
        255.0
    )


    img_array = np.expand_dims(
        img_array,
        axis=0
    )


    prediction = model.predict(
        img_array,
        verbose=0
    )


    index = int(
        np.argmax(prediction[0])
    )


    confidence = (
        float(prediction[0][index])
        *
        100
    )


    disease = index_to_class[
        str(index)
    ]


    return (
        disease,
        confidence,
        prediction[0]
    )

# ============================================================
# COMPACT HEADER
# ============================================================

st.markdown("""
<div class="hero-small">

<h1>🍅 Plant AI</h1>

<p>
Tomato Leaf Disease Detection using Deep Learning
</p>

</div>
""",
unsafe_allow_html=True)



st.markdown("""
<style>

.hero-small {

background:
linear-gradient(
135deg,
#ec407a,
#43a047
);

padding:20px;

border-radius:20px;

color:white;

text-align:center;

margin-bottom:25px;

box-shadow:
0 8px 25px rgba(0,0,0,0.15);

}


.result-card {

background:
rgba(255,255,255,0.55);

backdrop-filter:
blur(15px);

padding:30px;

border-radius:25px;

box-shadow:
0 10px 30px rgba(0,0,0,0.15);

}


.confidence {

font-size:35px;

font-weight:700;

color:#2e7d32;

}


</style>
""",
unsafe_allow_html=True)


# ============================================================
# UPLOAD SECTION
# ============================================================

left,right = st.columns(
    [1,1]
)


with left:

    st.subheader(
        "📤 Upload Tomato Leaf"
    )


    uploaded_file = st.file_uploader(
        "Choose image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )


# ============================================================
# RESULT SECTION
# ============================================================

if uploaded_file:

    image = Image.open(
        uploaded_file
    ).convert("RGB")


    with left:

        st.image(
            image,
            caption="Uploaded Leaf",
            use_container_width=True
        )


    disease, confidence, probabilities = predict_disease(
        image
    )


    info = DISEASE_INFO.get(
        disease,
        {}
    )


    # -------------------------------
    # Prediction Card
    # -------------------------------

    with right:

        st.markdown(
        f"""
        <div class="result-card">

        <h2>
        🩺 {info.get("display_name", disease)}
        </h2>


        <p class="confidence">
        {confidence:.2f}%
        </p>


        <h4>
        Model Confidence
        </h4>


        </div>
        """,
        unsafe_allow_html=True
        )


        severity = info.get(
            "severity",
            "Unknown"
        )


        if severity == "High":

            st.error(
                "🚨 High Severity - Immediate Action Required"
            )

        elif severity == "Moderate":

            st.warning(
                "⚠️ Moderate Severity"
            )

        else:

            st.success(
                "✅ Healthy Plant"
            )



    # -------------------------------
    # Disease Information
    # -------------------------------

    st.divider()


    tab1, tab2, tab3 = st.tabs(
        [
            "📖 Description",
            "💊 Treatment",
            "🛡 Prevention"
        ]
    )


    with tab1:

    st.markdown(
        f"""
        <div class="info-text">
        {info.get("description", "-")}
        </div>
        """,
        unsafe_allow_html=True
    )


with tab2:

    st.markdown(
        f"""
        <div class="info-text">
        {info.get("first_aid", "-")}
        </div>
        """,
        unsafe_allow_html=True
    )


with tab3:

    st.markdown(
        f"""
        <div class="info-text">
        {info.get("prevention", "-")}
        </div>
        """,
        unsafe_allow_html=True
    )



    # -------------------------------
    # Confidence Chart
    # -------------------------------

    st.divider()


    st.subheader(
        "📊 Prediction Probability"
    )


    df = pd.DataFrame(

        {
            "Disease": short_names,

            "Confidence":
            [
                round(float(x)*100,2)
                for x in probabilities
            ]
        }

    )


    fig = px.bar(

        df,

        x="Disease",

        y="Confidence",

        text="Confidence",

        color="Confidence",

        title="Model Confidence for Each Class"

    )


    fig.update_traces(

        texttemplate="%{text}%",

        textposition="outside"

    )


    fig.update_layout(

        yaxis_title="Confidence (%)",

        xaxis_title="Disease Classes",

        height=450

    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


else:

    st.markdown(
    """
    <div class="glass-card"
    style="text-align:center">

    <h2>
    🍅 Ready for Analysis
    </h2>


    <p>
    Upload a tomato leaf image and AI will detect
    possible diseases with confidence score.
    </p>


    <p>
    Supported formats:
    JPG • JPEG • PNG
    </p>


    </div>
    """,
    unsafe_allow_html=True
    )

st.divider()


st.caption(
    "Built with TensorFlow + Streamlit | PlantVillage Dataset"
)
