import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import requests

# Define class names
class_names = ['miner', 'nodisease', 'phoma', 'rust']

@st.cache_resource
def load_model():
    """
    Load the trained model.
    """
    model = tf.keras.models.load_model("odel1_20.h5")  # Provide the correct path to your model
    return model

def predict(model, image, class_names):
    """
    Predict the class of the uploaded image.
    """
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.image.resize(img_array, (256, 256))
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]), 2)

    return predicted_class, confidence

def send_telegram_message(bot_token, chat_id, message):
    """
    Send a message to a Telegram chat.
    """
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    response = requests.post(url, data=payload)
    if response.status_code == 200:
        st.success("Message sent to Telegram successfully!")
    else:
        st.error(f"Failed to send message. Error: {response.text}")

def main():
    """
    Main function to run the Streamlit app.
    """
    st.title("☕ Coffee Leaf Disease Detection")
    st.write("Upload an image of a coffee leaf to predict its class and confidence level.")

    # Load the model
    model = load_model()

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Open and display the image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Predict button
        if st.button("Predict"):
            predicted_class, confidence = predict(model, image, class_names)

            # Display results
            st.write(f"### Predicted Class: {predicted_class}")
            st.write(f"### Confidence: {confidence}%")

            # Send Telegram message
            bot_token = "7852441374:AAEf2VpFsQ9Ln33eDIgFzUvkZ9YgE-rEiD8"  # Replace with your bot token
            chat_id = "1329699944"           # Replace with your chat ID
            message = f"Predicted Class: {predicted_class}\nConfidence: {confidence}%"
            send_telegram_message(bot_token, chat_id, message)

            # Display the image with the result using Matplotlib
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(image)
            ax.set_title(f"Actual: Uploaded Image\nPredicted: {predicted_class}\nConfidence: {confidence}%")
            ax.axis("off")
            st.pyplot(fig)

if __name__ == "__main__":
    main()
