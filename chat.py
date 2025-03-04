import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load pre-trained model
MODEL_PATH = "C:\ML Projects\Brain Tumor (streamlit)\model.h5"  # Ensure the model is in the same directory or provide the correct path
model = tf.keras.models.load_model(MODEL_PATH)

# Class labels
tumor_classes = ['Pituitary', 'Glioma', 'No tumor', 'Meningioma']

# Streamlit UI
st.markdown(
    """
    <h1 style='text-align: center; color: #003366;'>MRI Brain Tumor Detection System</h1>
    <p style='text-align: center;'>Upload an MRI image to detect if there is a tumor and its type.</p>
    """, unsafe_allow_html=True
)

# File Upload
uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Image", use_column_width=True)
    
    # Resize image to the size expected by the model (128x128)
    IMAGE_SIZE = 128  # Same size as Flask app
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))  # Resize to 128x128 for the model
    image = np.array(image) / 255.0  # Normalize the image to [0, 1] (same as Flask)

    # If model expects 3 channels, make sure the image has 3 channels (RGB)
    if len(image.shape) == 2:  # If grayscale, convert to RGB
        image = np.stack([image] * 3, axis=-1)
    
    # Expand dimensions to match the model input (batch size, height, width, channels)
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Perform Prediction
    with st.spinner("Analyzing MRI scan..."):
        prediction = model.predict(image)
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        predicted_class = tumor_classes[predicted_class_index]
        confidence = np.max(prediction) * 100
    
    # Display Results
    st.markdown(
        f"""
        <div style="border: 2px solid #ddd; padding: 15px; border-radius: 10px; background-color: #f9f9f9; text-align: center;">
            <h2 style="color: green;">Tumor Type: {predicted_class}</h2>
            <p style="font-size: 18px;"><strong>Confidence:</strong> {confidence:.2f}%</p>
        </div>
        """, unsafe_allow_html=True
    )
