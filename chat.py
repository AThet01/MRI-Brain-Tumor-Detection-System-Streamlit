import streamlit as st
import tensorflow as tf
import numpy as np
import os
from PIL import Image

# Load model with dynamic path
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.h5")
if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Ensure model.h5 is in the same directory.")

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
