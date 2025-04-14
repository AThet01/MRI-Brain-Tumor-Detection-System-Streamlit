import streamlit as st
import tensorflow as tf
import numpy as np
import os
from PIL import Image
import gdown

# === Google Drive model download setup ===
MODEL_FILENAME = "model.h5"
MODEL_FILE_ID = "17e5HUzhUciAvQ9X30_u9CQKhdgn2ecHv"  # Replace with your actual file ID
MODEL_PATH = os.path.join(os.path.dirname(__file__), MODEL_FILENAME)

# Download model if not already present
if not os.path.exists(MODEL_PATH):
    st.warning("Model not found locally. Downloading from Google Drive...")
    url = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)

# Load pre-trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Class labels
tumor_classes = ['Pituitary', 'Glioma', 'No tumor', 'Meningioma']

st.markdown(
    """
    <h1 style='text-align: center; color: #003366;'>MRI Brain Tumor Detection System</h1>
    <p style='text-align: center;'>Upload an MRI image to detect if there is a tumor and its type.</p>
    """, unsafe_allow_html=True
)

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Image", use_column_width=True)

    IMAGE_SIZE = 128
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    image = np.array(image) / 255.0

    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)

    image = np.expand_dims(image, axis=0)

    with st.spinner("Analyzing MRI scan..."):
        prediction = model.predict(image)
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        predicted_class = tumor_classes[predicted_class_index]
        confidence = np.max(prediction) * 100

    st.markdown(
        f"""
        <div style="border: 2px solid #ddd; padding: 15px; border-radius: 10px; background-color: #f9f9f9; text-align: center;">
            <h2 style="color: green;">Tumor Type: {predicted_class}</h2>
            <p style="font-size: 18px;"><strong>Confidence:</strong> {confidence:.2f}%</p>
        </div>
        """, unsafe_allow_html=True
    )
