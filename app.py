import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from utils import preprocess_image, postprocess_mask

st.title("🛣️ Road Segmentation with Deep Learning")

model = tf.keras.models.load_model("road_mapper.h5", compile=False)

uploaded_file = st.file_uploader("Upload a road image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    preprocessed_image = preprocess_image(image)
    
    # 🔍 Add this block for debugging
    prediction = model.predict(preprocessed_image)
    st.write("Prediction shape:", prediction.shape)
    st.write("Prediction max/min:", prediction.max(), prediction.min())
    
    # Continue with post-processing and displaying the output
    mask = postprocess_mask(prediction)
    st.image(mask, caption="Predicted Mask", use_column_width=True)
