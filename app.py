# ============================================================
# PLANT AI - TOMATO DISEASE DETECTOR
# Streamlit Cloud Stable Version
# ============================================================


# ============================================================
# TensorFlow Stability Settings
# MUST BE BEFORE TENSORFLOW IMPORT
# ============================================================

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"



# ============================================================
# IMPORTS
# ============================================================

import streamlit as st
import numpy as np
from tensorflow import keras
from PIL import Image
import json
import urllib.request
import pandas as pd
import plotly.express as px



# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(

    page_title="Plant AI - Tomato Disease Detector",

    page_icon="🍅",

    layout="wide"

)




# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(

"""
<style>


/* Main background */

.stApp {

background:

linear-gradient(

135deg,

#fff0f6,

#f0fff4,

#e8f5e9

);

}



/* Hero Header */

.hero-small {


background:

linear-gradient(

135deg,

#ec407a,

#43a047

);


padding:25px;


border-radius:25px;


color:white;


text-align:center;


box-shadow:

0 8px 25px rgba(0,0,0,0.15);


}



.hero-small h1 {

font-size:42px;

margin-bottom:8px;

}



.hero-small h3 {

font-size:22px;

}




/* Upload Box */

[data-testid="stFileUploader"] {


background:

rgba(255,255,255,0.55);


border-radius:20px;


padding:15px;


}




/* Result Card */


.result-card {


background:

rgba(255,255,255,0.65);


border-radius:25px;


padding:25px;


text-align:center;


box-shadow:

0 8px 25px rgba(0,0,0,0.12);


}




.result-card h2 {


font-size:28px;


color:#2e7d32;


}



.confidence {


font-size:36px;


font-weight:800;


color:#1b5e20;


}




/* Information Cards */


.info-text {


background:

rgba(255,255,255,0.65);


border-radius:20px;


padding:25px;


font-size:17px;


line-height:1.7;


box-shadow:

0 5px 20px rgba(0,0,0,0.08);


}




.info-text h3 {


font-size:24px;


color:#2e7d32;


}




/* Landing Card */


.glass-card {


background:

rgba(255,255,255,0.65);


border-radius:25px;


padding:30px;


text-align:center;


box-shadow:

0 8px 25px rgba(0,0,0,0.10);


}




footer {

visibility:hidden;

}


</style>

""",

unsafe_allow_html=True

)






# ============================================================
# SIDEBAR
# ============================================================


with st.sidebar:


    st.title("🍅 Plant AI")


    st.write(

"""
### Deep Learning Plant Disease Detector


Dataset:
PlantVillage


Model:
CNN


Framework:
TensorFlow + Streamlit

"""

    )





# ============================================================
# CONSTANTS
# ============================================================


IMG_SIZE = 128



MODEL_URL = (

"https://huggingface.co/justunforgettable/"

"Plant_Disease_Detector/"

"resolve/main/plant_disease_model.h5"

)



CLASS_URL = (

"https://huggingface.co/justunforgettable/"

"Plant_Disease_Detector/"

"resolve/main/class_indices.json"

)




MODEL_PATH = "plant_disease_model.h5"


CLASS_PATH = "class_indices.json"






# ============================================================
# DOWNLOAD FILE FUNCTION
# ============================================================


def download_file(url, path):


    if not os.path.exists(path):


        try:


            with st.spinner(

                f"Loading {path}..."

            ):


                urllib.request.urlretrieve(

                    url,

                    path

                )



        except Exception as e:


            st.error(

                f"Unable to download file: {e}"

            )


# ============================================================
# DISEASE INFORMATION DATABASE
# ============================================================


DISEASE_INFO = {


"Tomato_healthy":
{

"display_name":
"Healthy Tomato 🍃",

"description":
"The tomato plant appears healthy without visible symptoms of disease.",

"first_aid":
"No treatment required. Continue proper watering, sunlight and nutrition.",

"prevention":
"Maintain plant hygiene, proper irrigation and good air circulation.",

"severity":
"None"

},



"Tomato_Early_blight":
{

"display_name":
"Early Blight 🍂",

"description":
"Fungal disease causing dark brown spots with yellow rings on tomato leaves.",

"first_aid":
"""
• Remove infected leaves
• Apply copper based fungicide
• Avoid watering leaves directly
""",

"prevention":
"""
• Practice crop rotation
• Maintain proper airflow
• Use resistant varieties
""",

"severity":
"Moderate"

},



"Tomato_Late_blight":
{

"display_name":
"Late Blight ⚠️",

"description":
"Serious fungal disease causing water soaked lesions and rapid leaf damage.",

"first_aid":
"""
• Remove infected plant parts
• Apply recommended fungicide
• Improve drainage
""",

"prevention":
"""
• Use healthy seeds
• Avoid excess moisture
• Maintain plant spacing
""",

"severity":
"High"

},



"Tomato_Leaf_Mold":
{

"display_name":
"Leaf Mold 🍃",

"description":
"Fungal infection usually occurring in humid conditions and affecting leaves.",

"first_aid":
"""
• Remove affected leaves
• Improve ventilation
• Apply suitable fungicide
""",

"prevention":
"""
• Reduce humidity
• Avoid overhead irrigation
• Maintain spacing
""",

"severity":
"Moderate"

},



"Tomato_Septoria_leaf_spot":
{

"display_name":
"Septoria Leaf Spot 🍁",

"description":
"Small circular spots with dark borders appearing on tomato leaves.",

"first_aid":
"""
• Remove infected leaves
• Apply fungicide
• Keep foliage dry
""",

"prevention":
"""
• Crop rotation
• Proper irrigation
• Avoid wet leaves
""",

"severity":
"Moderate"

}

}






# ============================================================
# TEXT FORMATTER FOR HTML
# ============================================================


def clean_text(text):


    return text.replace(

        "\n",

        "<br>"

    )







# ============================================================
# LOAD MODEL AND CLASSES
# ============================================================



@st.cache_resource

def load_model_and_classes():



    download_file(

        MODEL_URL,

        MODEL_PATH

    )



    download_file(

        CLASS_URL,

        CLASS_PATH

    )




    try:


        model = keras.models.load_model(

            MODEL_PATH,

            compile=False

        )



    except Exception as e:


        st.error(

            f"Model loading failed: {e}"

        )


        st.stop()





    with open(

        CLASS_PATH,

        "r"

    ) as file:


        index_to_class = json.load(file)




    return model, index_to_class







model, index_to_class = load_model_and_classes()






# ============================================================
# CLASS NAMES
# ============================================================


class_names = [


    index_to_class[str(i)]

    for i in range(

        len(index_to_class)

    )

]




short_names = [


    name.replace(

        "Tomato_",

        ""

    )


    for name in class_names

]







# ============================================================
# PREDICTION FUNCTION
# ============================================================


def predict_disease(image):


    image = image.resize(

        (

            IMG_SIZE,

            IMG_SIZE

        )

    )



    img_array = np.array(

        image

    ).astype(

        "float32"

    ) / 255.0




    img_array = np.expand_dims(

        img_array,

        axis=0

    )





    prediction = model.predict(

        img_array,

        verbose=0

    )





    index = int(

        np.argmax(

            prediction[0]

        )

    )





    confidence = float(

        prediction[0][index]

    ) * 100





    disease = index_to_class[

        str(index)

    ]





    return (

        disease,

        confidence,

        prediction[0]

    )


# ============================================================
# HEADER
# ============================================================


st.markdown(

"""
<div class="hero-small">

<h1>
🍅 Plant AI
</h1>

<h3>
Tomato Leaf Disease Detection System
</h3>

<p>
Powered by Deep Learning + CNN
</p>

</div>
""",

unsafe_allow_html=True

)





# ============================================================
# IMAGE UPLOAD AREA
# ============================================================


left, right = st.columns(

    [1,1]

)





with left:


    st.subheader(

        "📤 Upload Tomato Leaf"

    )



    uploaded_file = st.file_uploader(

        "Choose Image",

        type=[

            "jpg",

            "jpeg",

            "png"

        ],

        help=

        "Upload a clear tomato leaf image"

    )







# ============================================================
# AFTER IMAGE UPLOAD
# ============================================================


if uploaded_file:



    image = Image.open(

        uploaded_file

    ).convert(

        "RGB"

    )





    with left:


        st.image(

            image,

            caption=

            "Uploaded Leaf",

            width=350

        )






    disease, confidence, probabilities = predict_disease(

        image

    )





    info = DISEASE_INFO.get(

        disease,

        {}

    )






    # ========================================================
    # RESULT CARD
    # ========================================================


    with right:



        st.markdown(

        f"""

        <div class="result-card">


        <h2>

        🩺 {info.get(

            "display_name",

            disease

        )}

        </h2>



        <div class="confidence">

        {confidence:.2f}%

        </div>



        <p>

        AI Confidence Score

        </p>



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

                "🚨 High Severity Disease"

            )



        elif severity == "Moderate":


            st.warning(

                "⚠️ Moderate Severity Disease"

            )



        else:


            st.success(

                "✅ Plant is Healthy"

            )








    # ========================================================
    # INFORMATION TABS
    # ========================================================


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


        <h3>

        📖 Disease Description

        </h3>



        {info.get(

            "description",

            "-"

        )}



        </div>

        """,

        unsafe_allow_html=True

        )





    with tab2:


        st.markdown(

        f"""

        <div class="info-text">


        <h3>

        💊 Treatment / First Aid

        </h3>



        {clean_text(

            info.get(

                "first_aid",

                "-"

            )

        )}



        </div>


        """,

        unsafe_allow_html=True

        )






    with tab3:


        st.markdown(

        f"""

        <div class="info-text">


        <h3>

        🛡 Prevention Methods

        </h3>



        {clean_text(

            info.get(

                "prevention",

                "-"

            )

        )}



        </div>


        """,

        unsafe_allow_html=True

        )







    # ========================================================
    # PROBABILITY CHART
    # ========================================================


    st.divider()



    st.subheader(

        "📊 Prediction Probability"

    )





    df = pd.DataFrame(

        {

            "Disease":

            short_names,


            "Confidence":

            [

                round(

                    float(x)*100,

                    2

                )

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

        title=

        "Confidence Across Disease Classes"

    )






    fig.update_traces(

        texttemplate="%{text}%",

        textposition="outside"

    )





    fig.update_layout(

        height=450,

        yaxis_title="Confidence (%)",

        xaxis_title="Disease"

    )






    st.plotly_chart(

        fig,

        use_container_width=True

    )







# ============================================================
# LANDING PAGE
# ============================================================


else:



    st.markdown(

    """

    <div class="glass-card">


    <h3>

    🍅 Ready for Analysis

    </h3>



    <p>

    Upload a tomato leaf image and AI will

    identify possible diseases.

    </p>



    <p>

    🧠 CNN Prediction

    |

    📊 Confidence Score

    |

    💊 Treatment Guidance

    </p>



    <p>

    Supported Formats:

    JPG • JPEG • PNG

    </p>



    </div>

    """,

    unsafe_allow_html=True

    )







# ============================================================
# FOOTER
# ============================================================


st.divider()



st.caption(

"Built with TensorFlow + Streamlit | PlantVillage Dataset"

)
            st.stop()

