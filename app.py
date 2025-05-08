import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from utils import preprocess_image, postprocess_mask

st.title("🛣️ Road Segmentation with Deep Learning")

model = tf.keras.models.load_model("road_mapper.h5", compile=False)

uploaded_file = st.file_uploader("Upload a road image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", use_column_width=True)

    if st.button("Segment Road"):
        with st.spinner("Segmenting..."):
            preprocessed = preprocess_image(image)
            prediction = model.predict(preprocessed)
            mask_image = postprocess_mask(prediction)
        
        st.image(mask_image, caption="Predicted Road Mask", use_column_width=True)
