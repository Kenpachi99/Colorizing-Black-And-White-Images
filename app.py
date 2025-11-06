import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from skimage.color import rgb2lab, lab2rgb
import torchvision.transforms as transforms
import os
import gdown

# Import custom modules
from models_utils import Discriminator, init_weights, init_model

# Set page config
st.set_page_config(
    page_title="Image Colorization",
    page_icon="🎨",
    layout="centered"
)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def build_generator(n_input=1, n_output=2, size=256):
    """Build the generator model using fastai's DynamicUnet"""
    try:
        from fastai.vision.learner import create_body
        from torchvision.models.resnet import resnet34
        from fastai.vision.models.unet import DynamicUnet

        backbone = create_body(resnet34, pretrained=True, n_in=n_input, cut=-2)
        G_net = DynamicUnet(backbone, n_output, (size, size)).to(device)
        return G_net
    except Exception as e:
        st.error(f"Error building generator: {str(e)}")
        return None

@st.cache_resource
def load_model():
    """Load the pre-trained generator model"""
    model_path = "generator.pt"

    # Check if model exists locally
    if not os.path.exists(model_path):
        st.warning("⚠️ Model weights not found locally.")

        # Try to download from Google Drive (you'll need to upload your trained model)
        st.info("Attempting to download pre-trained model...")

        # Placeholder - Replace with your actual Google Drive link
        # Example: https://drive.google.com/uc?id=YOUR_FILE_ID
        gdrive_url = None

        if gdrive_url:
            try:
                gdown.download(gdrive_url, model_path, quiet=False)
                st.success("Model downloaded successfully!")
            except Exception as e:
                st.error(f"Failed to download model: {str(e)}")
                st.info("""
                **To use this app, you need to:**
                1. Train the model using the provided notebook (ImColor.ipynb)
                2. Save the trained model as 'generator.pt'
                3. Upload it to Google Drive and make it shareable
                4. Update the `gdrive_url` variable in app.py with your model's link

                OR

                Place your trained 'generator.pt' file in the project root directory.
                """)
                return None
        else:
            st.error("""
            **Model not found!**

            Please follow these steps:
            1. Train the model using ImColor.ipynb notebook
            2. Save the model weights as 'generator.pt'
            3. Either:
               - Place the file in this directory, OR
               - Upload to Google Drive, get the shareable link, and update the app.py file

            For quick testing, you can train the model for just 1-2 epochs.
            """)
            return None

    # Build and load the model
    G_net = build_generator(n_input=1, n_output=2, size=256)
    if G_net is None:
        return None

    try:
        G_net.load_state_dict(torch.load(model_path, map_location=device))
        G_net.eval()
        return G_net
    except Exception as e:
        st.error(f"Error loading model weights: {str(e)}")
        return None

def process_image(image, model):
    """Process the input image and colorize it"""
    img_size = 256

    # Resize and convert to numpy array
    transform = transforms.Resize((img_size, img_size), Image.BICUBIC)
    img_resized = transform(image)
    img_np = np.array(img_resized)

    # Convert to LAB color space
    img_lab = rgb2lab(img_np).astype("float32")
    img_lab = transforms.ToTensor()(img_lab)

    # Normalize L channel
    L = img_lab[[0], ...] / 50. - 1.
    L = L.unsqueeze(0).to(device)

    # Generate AB channels
    with torch.no_grad():
        ab_pred = model(L)

    # Denormalize and convert back to RGB
    L = (L.squeeze(0).cpu() + 1.) * 50.
    ab = ab_pred.squeeze(0).cpu() * 110.

    # Combine L and ab channels
    Lab = torch.cat([L, ab], dim=0).permute(1, 2, 0).numpy()

    # Convert to RGB
    rgb_img = lab2rgb(Lab)

    # Clip values to valid range
    rgb_img = np.clip(rgb_img, 0, 1)

    return rgb_img

def main():
    st.title("🎨 Black & White Image Colorization")
    st.markdown("""
    Upload a black and white image and watch it come to life with colors!

    This app uses a **GAN-based deep learning model** (U-Net Generator with ResNet34 backbone)
    trained to automatically colorize grayscale images.
    """)

    # Load the model
    with st.spinner("Loading model..."):
        model = load_model()

    if model is None:
        st.stop()

    st.success("✅ Model loaded successfully!")

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a black & white image...",
        type=['png', 'jpg', 'jpeg', 'bmp']
    )

    if uploaded_file is not None:
        # Load and display original image
        image = Image.open(uploaded_file).convert('RGB')

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)

        # Process button
        if st.button("🎨 Colorize Image", type="primary"):
            with st.spinner("Colorizing your image..."):
                try:
                    # Process the image
                    colorized = process_image(image, model)

                    # Display colorized image
                    with col2:
                        st.subheader("Colorized Image")
                        st.image(colorized, use_container_width=True)

                    # Download button
                    st.success("✨ Colorization complete!")

                    # Convert to PIL Image for download
                    colorized_pil = Image.fromarray((colorized * 255).astype(np.uint8))

                    # Save to bytes
                    from io import BytesIO
                    buf = BytesIO()
                    colorized_pil.save(buf, format='PNG')
                    byte_im = buf.getvalue()

                    st.download_button(
                        label="📥 Download Colorized Image",
                        data=byte_im,
                        file_name="colorized_image.png",
                        mime="image/png"
                    )

                except Exception as e:
                    st.error(f"Error during colorization: {str(e)}")
                    st.exception(e)

    # Sidebar with information
    with st.sidebar:
        st.header("About")
        st.markdown("""
        ### How it works
        1. Upload a grayscale or black & white image
        2. Click the "Colorize Image" button
        3. Download your colorized image!

        ### Model Architecture
        - **Generator**: U-Net with ResNet34 backbone
        - **Training**: GAN-based approach with PatchGAN discriminator
        - **Color Space**: LAB color space for better colorization

        ### Tips
        - Works best with natural images (landscapes, portraits)
        - Images are resized to 256x256 for processing
        - Upload clear, high-contrast B&W images for best results

        ---

        **Made with ❤️ using PyTorch and Streamlit**
        """)

        st.header("Sample Images")
        st.markdown("""
        Try colorizing:
        - Old black & white photographs
        - Grayscale portraits
        - Historical images
        - Line drawings (may vary in quality)
        """)

if __name__ == "__main__":
    main()
