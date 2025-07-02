import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn 
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp 
import rasterio
import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

# Define your model class
class ChangeNet(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder = model.encoder  # model encoder
        self.decoder = model.decoder  # model decoder
        self.head = model.segmentation_head  # segmentation head for generation of change mask

    def forward(self, x1, x2):
        enc1 = self.encoder(x1)  # get latent features of image1
        enc2 = self.encoder(x2)  # get latent features of image2
        encoder_out = []
        for i in range(len(enc1)):
            encoder_out.append(torch.add(enc1[i], enc2[i]))  # Add the latent features and append them to a list
        decoder_out = self.decoder(*encoder_out)  # Pass the latent features through a decoder
        out = self.head(decoder_out)  # Pass the decoder output through the segmentation head to generate change mask
        return out

# Load your model
model = smp.Unet('resnet34', encoder_depth=3, decoder_channels=(64, 64, 16))
change_model = ChangeNet(model)
change_model.load_state_dict(torch.load('my_model.pt', map_location=torch.device('cpu')))
change_model.eval()

# Define a function to preprocess the uploaded images
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),  # Resize the image to the input size of your model
        transforms.ToTensor(),           # Convert PIL image to PyTorch tensor
    ])
    return transform(image).unsqueeze(0)

# Define a function to make predictions
def predict(image1, image2):
    with torch.no_grad():
        # Preprocess the uploaded images
        image1 = preprocess_image(image1)
        image2 = preprocess_image(image2)
        
        # Generate prediction
        pred_mask = (torch.sigmoid(change_model(image1, image2)).squeeze()).cpu().numpy()
        
    return pred_mask

# Define your Streamlit app
def main():
    st.title("Remote Sensing Image Seen Classification")

    # Upload images
    uploaded_image1 = st.file_uploader("Upload Image 1", type=["jpg", "png"])
    uploaded_image2 = st.file_uploader("Upload Image 2", type=["jpg", "png"])

    if uploaded_image1 is not None and uploaded_image2 is not None:
        # Display uploaded images
        st.image(uploaded_image1, caption='Uploaded Image 1', use_column_width=True)
        st.image(uploaded_image2, caption='Uploaded Image 2', use_column_width=True)

        # Convert uploaded images to PIL Image
        image1 = Image.open(uploaded_image1)
        image2 = Image.open(uploaded_image2)

        # Add predict button
        if st.button('Predict'):
            # Make prediction
            pred_mask = predict(image1, image2)

            # Convert the predicted mask to RGB
            img_rgb = plt.cm.viridis(pred_mask)[:, :, :3]
            plt.imsave('AGB_image.png', img_rgb)
            # Define custom colormap
            

            # Assuming pred_mask is your predicted mask array with values representing different classes

            # Define custom colormap
            colors = ['green', 'blue', 'red']  # Colors for oxygen, carbon, and affected area respectively
            cmap = ListedColormap(colors)

            # Create RGB image using the custom colormap
            img_rgb = cmap(pred_mask)

            # Plot the image
            plt.imshow(img_rgb)
            plt.colorbar(label='Legend')
            plt.xticks([])  # Remove x ticks
            plt.yticks([])  # Remove y ticks

            # Custom tick labels for color bar
            tick_labels = ['Oxygen', 'Carbon', 'Affected Area']
            cbar = plt.colorbar(ticks=[0.33, 0.67, 1])
            cbar.ax.set_yticklabels(tick_labels)

            # Save the image as a .png file
            plt.savefig('predicted_mask_with_legend.png', bbox_inches='tight')
            plt.show()



            # Display prediction
            st.subheader("Predicted Mask")
            st.image(img_rgb, caption='Predicted Mask', use_column_width=True, clamp=True)
            #cv2.imwrite('AGB_image.png',img_rgb, clamp=True)
            agb_image_path = "AGB_image.png"

            with rasterio.open(agb_image_path) as src:
                red_band = src.read(1)  # Red band (1-indexed)
                nir_band = src.read(3)  # Near-infrared band (1-indexed)

            # Calculate vegetation indices
            # NDVI
            ndvi = (nir_band - red_band) / (nir_band + red_band)
            ndvi = np.mean(ndvi)
            print("NDVI values:\n", ndvi)

            # TVI
            tvi = np.sqrt((nir_band - red_band) / (nir_band + red_band + 0.5))
            tvi = np.mean(tvi)
            print("TVI values:\n", tvi)

            # SAVI
            L = 0.5  # L is the soil adjustment factor
            savi = ((nir_band - red_band) / (nir_band + red_band + L)) * (1 + L)
            savi = np.mean(savi)
            print("SAVI values:\n", savi)

            # RDVI
            rdvi = (nir_band - red_band) / np.sqrt(nir_band + red_band)
            rdvi = np.mean(rdvi)
            print("RDVI values:\n", rdvi)

            # MSR
            msr = nir_band / red_band
            msr = np.mean(msr)
            print("MSR values:\n", msr)

            st.write("NDVI :", ndvi)
            st.write("TVI :", tvi)
            st.write("SAVI :", savi)
            st.write("RDVI :", rdvi)
            st.write("MSR :", msr)


if __name__ == "__main__":
    main()
