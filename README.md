# 🍅 Plant AI — Tomato Leaf Disease Detection System

<div align="center">

Deep learning web app that detects tomato leaf diseases from images and gives confidence scores, treatment tips, and prevention guidance.

🚀 **Live Demo:** https://plantdiseasedetector-92zkewmvliwudz7jmy3uwp.streamlit.app/

</div>

---

## 📌 Overview

A custom CNN trained on a 5-class subset of the PlantVillage dataset, served via a Streamlit app. The model is hosted on the **Hugging Face Hub** and downloaded at runtime (the `.h5` file is ~97 MB, too large for GitHub).

## ✨ Features

- Upload a tomato leaf image → get predicted disease + confidence %
- Disease description, severity, first-aid, and prevention info
- Interactive confidence chart across all 5 classes (Plotly)
- Custom pastel-themed Streamlit UI

## 📂 Dataset & Classes

Subset of [PlantVillage](https://www.kaggle.com/datasets/emmarex/plantdisease) — 5 tomato classes: **Healthy, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot** (5,776 train / 1,447 val images, 80/20 split).

## 🏗️ Model

Custom CNN (3× Conv2D + BatchNorm + MaxPooling blocks → Dense(256) → Dropout) — 128×128×3 input, ~8.48M parameters, TensorFlow/Keras.

| Metric | Score |
|--------|-------|
| Validation Accuracy | **76.2%** |
| Validation Loss | 0.816 |

## 🔍 Explainability

Grad-CAM implemented in the training notebook (`conv2d_2` layer, `GradientTape`-based due to the `Sequential` wrapper blocking standard access) — not yet integrated into the deployed app.

## 🛠️ Tech Stack

Python · TensorFlow · Keras · Streamlit · Pandas · Plotly · Pillow · NumPy · Hugging Face Hub


## 🔮 Future Enhancements

- Integrate Grad-CAM into the live app
- Real-time camera detection
- Support for more crops
- Mobile app version

## 👨‍💻 Author

**Nahid Kausar** — B.Tech CSE

---

<div align="center">⭐ Star this repo if it helped you!</div>
