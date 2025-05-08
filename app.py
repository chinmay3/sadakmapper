import os
# CRITICAL: Set this environment variable BEFORE importing TensorFlow or Keras
# This tells TensorFlow 2.16+ to use the Keras 2 API via the tf-keras package
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image # Updated import
from tensorflow.keras.models import load_model    # Updated import
from tensorflow.keras import backend as K         # Import Keras backend
import matplotlib.pyplot as plt

# --- Define Custom Functions BEFORE they are used ---

# Dice coefficient (used for segmentation performance measurement)
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# Soft Dice loss (used for segmentation loss function)
def soft_dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

# Intersection over Union (IoU) coefficient (used for evaluating segmentation accuracy)
def iou_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])  # axis for spatial dimensions
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3]) - intersection # Added axis to K.sum
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou

# --- Load the model AFTER custom functions are defined ---
# Ensure 'save_best.keras' is in the same directory as app.py or provide the correct path
try:
    model = load_model(
        'save_best.keras',
        custom_objects={'soft_dice_loss': soft_dice_loss, 'iou_coef': iou_coef}
    )
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.error("Please ensure 'save_best.keras' is in the correct location and the custom objects are defined correctly.")
    st.stop() # Stop execution if model fails to load


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
    try:
        prediction = model.predict(img_array)

        # Display the raw output (before thresholding)
        st.subheader("Prediction (Raw Output):")
        fig, ax = plt.subplots()
        # Assuming prediction is (1, height, width, 1) or (1, height, width)
        # Squeeze to remove single dimensions for imshow if necessary
        ax.imshow(np.squeeze(prediction[0]), cmap='jet')  # Use 'jet' for better visualization
        ax.set_title("Prediction")
        st.pyplot(fig)

        # Apply thresholding for binary segmentation (optional)
        thresholded_prediction = (prediction > 0.5).astype(np.uint8)

        # Display the thresholded output
        st.subheader("Thresholded Prediction:")
        fig_thresh, ax_thresh = plt.subplots()
        ax_thresh.imshow(np.squeeze(thresholded_prediction[0]), cmap='gray')  # Use grayscale for thresholded mask
        ax_thresh.set_title("Thresholded Prediction")
        st.pyplot(fig_thresh)

    except Exception as e:
        st.error(f"Error during prediction or display: {e}")
