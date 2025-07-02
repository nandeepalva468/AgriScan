import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt

# Define class names
class_names = ['miner', 'nodisease', 'phoma', 'rust']

@st.cache_resource
def load_model_from_file():
    """
    Load the trained model from the file.
    """
    model = load_model('coffee_model11.h5')
    return model

def preprocess_image(image):
    """
    Preprocess the uploaded image for prediction.
    Resize, normalize, and add batch dimension.
    """
    img_array = np.array(image)
    img_array = tf.image.resize(img_array, (256, 256))  # Resize to match model input
    img_array = img_array / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def predict_image(model, image):
    """
    Predict the class of the uploaded image using the model.
    """
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_class_index]
    return predicted_class, predictions[0]

def main():
    """
    Main function to run the Streamlit app.
    """
    st.title("☕ Coffee Leaf Disease Detection App")
    st.write("Upload an image of a coffee leaf to detect the disease.")

    # Load the model
    model = load_model_from_file()

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Load and display the image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Predict button
        if st.button("Predict!"):
            predicted_class, prediction_scores = predict_image(model, image)
            st.write(f"### Predicted Class: {predicted_class}")
            st.write("### Prediction Scores:")
            for i, score in enumerate(prediction_scores):
                st.write(f"{class_names[i]}: {score:.4f}")

if __name__ == "__main__":
    main()
