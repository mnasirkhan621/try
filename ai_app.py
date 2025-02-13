



import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from PIL import Image
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import cv2

# Load AI Text Detection Model
text_model = tf.keras.models.load_model("ai_text_detector.keras")

# Load Tokenizer & Label Encoder
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Load AI Image Detection Model
image_model = tf.keras.models.load_model("trained_densenet_model_of_Nasir_on_full_dataset_of_1_to_5.keras")

# Streamlit UI
st.title("AI-Generated Image & Text Detector")

# Image Upload Section
st.header("Upload an Image")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess Image
    image = np.array(image)
    image = cv2.resize(image, (224, 224))  # Resize to match model input size
    image = image / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)

    # Predict
    image_prediction = image_model.predict(image)
    image_result = "AI-Generated" if image_prediction[0] > 0.5 else "Real"

    st.write(f"### **Prediction: {image_result} Image**")

# Text Input Section
st.header("Paste Text for Detection")
text_input = st.text_area("Enter text here:")

if text_input:
    # Preprocess Text
    text_sequence = tokenizer.texts_to_sequences([text_input])
    padded_text = pad_sequences(text_sequence, maxlen=200, padding="post", truncating="post")

    # Predict
    text_prediction = text_model.predict(padded_text)
    text_result = "AI-Generated" if text_prediction[0] > 0.5 else "Human-Written"

    st.write(f"### **Prediction: {text_result} Text**")

# Run Instructions
st.write("### How to Run?")
st.code("streamlit run ai_image_text_detector.py", language="bash")
