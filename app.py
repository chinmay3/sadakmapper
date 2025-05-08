import streamlit as st
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
import matplotlib.pyplot as plt

# Load the best model saved during training
model = load_model('save_best.keras', custom_objects={'soft_dice_loss': soft_dice_loss, 'iou_coef': iou_coef})

# Streamlit app UI
st.title("Satellite Image Segmentation Model")

st.markdown("""
    This app predicts the segmentation mask for satellite images using a deep learning model.
    Upload a satellite image, and it will output the segmented image.
""")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Load the image and display it
    img = image.load_img(uploaded_file, target_size=(128, 128))  # Resize to match model input size
    st.image(img, caption='Uploaded Image.', use_column_width=True)

    # Preprocess the image
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image

    # Make the prediction
    prediction = model.predict(img_array)

    # Display the raw output (before thresholding)
    st.subheader("Prediction (Raw Output):")
    fig, ax = plt.subplots()
    ax.imshow(prediction[0], cmap='jet')  # Use 'jet' for better visualization
    ax.set_title("Prediction")
    st.pyplot(fig)

    # Apply thresholding for binary segmentation (optional)
    thresholded_prediction = (prediction > 0.5).astype(np.uint8)

    # Display the thresholded output
    st.subheader("Thresholded Prediction:")
    fig, ax = plt.subplots()
    ax.imshow(thresholded_prediction[0], cmap='gray')  # Use grayscale for thresholded mask
    ax.set_title("Thresholded Prediction")
    st.pyplot(fig)
