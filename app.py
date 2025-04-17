import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

model = load_model('ip_classifier_model.h5')

st.title('AI for Intellectual Property Awareness')
st.write('Upload a file to classify the type of Intellectual Property')

uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    
    # Preprocessing and prediction (adapt it to your model's requirements)
    image = image.resize((224, 224))  # Resize to your model's input size
    image_array = np.array(image) / 255.0  # Normalize if required
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    prediction = model.predict(image_array)

    # Display prediction (adapt this to your model's output format)
    st.write(f"Prediction: {prediction}")
