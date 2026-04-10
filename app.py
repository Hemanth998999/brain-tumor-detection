import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import (
    Input, Dense, Dropout,
    GlobalAveragePooling2D,
    BatchNormalization
)
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image


# -----------------------------------
# Page configuration
# -----------------------------------
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="centered"
)


# -----------------------------------
# Load model only once
# -----------------------------------
@st.cache_resource
def load_brain_model():
    base_model = EfficientNetB4(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False

    inputs = Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(4, activation="softmax")(x)

    model = Model(inputs, outputs)

    # Load weights from current .keras file
    model.load_weights("brain_tumor_best.keras")
    return model


model = load_brain_model()

classes = ['glioma', 'meningioma', 'notumor', 'pituitary']


# -----------------------------------
# UI
# -----------------------------------
st.title("🧠 Brain Tumor Detection")
st.write("Upload an MRI scan image to detect the tumor type.")

uploaded_file = st.file_uploader(
    "Choose MRI image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")

    st.image(
        img,
        caption="Uploaded MRI Scan",
        use_container_width=True
    )

    # Preprocessing
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    pred = model.predict(img_array)
    class_idx = np.argmax(pred)
    confidence = np.max(pred) * 100

    # Main result
    st.subheader("📋 Prediction Result")
    st.success(f"🧠 Tumor Type: **{classes[class_idx]}**")
    st.info(f"📊 Confidence: **{confidence:.2f}%**")

    # All probabilities
    st.subheader("📈 Class Probabilities")
    for i, cls in enumerate(classes):
        prob = pred[0][i] * 100
        st.write(f"**{cls}**: {prob:.2f}%")
        st.progress(float(pred[0][i]))