# 🎨 Colorizing Black and White Images

An AI-powered web application that automatically colorizes black and white images using a GAN-based deep learning model.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-FF4B4B.svg)

## 🌟 Features

- **Upload & Colorize**: Simple drag-and-drop interface for black & white images
- **Deep Learning**: GAN-based model with U-Net Generator and ResNet34 backbone
- **Instant Results**: Get colorized images in seconds
- **Download**: Save your colorized images directly from the browser

## 🚀 Quick Start

### Option 1: Use Pre-trained Model (Recommended)

1. **Train the model** (if you haven't already):
   - Open `ImColor.ipynb` in Google Colab or Jupyter
   - Run all cells to train the model
   - The model will be saved as `generator.pt`

2. **Run locally**:
   ```bash
   # Install dependencies
   pip install -r requirements.txt

   # Run the app
   streamlit run app.py
   ```

3. **Open in browser**: The app will automatically open at `http://localhost:8501`

### Option 2: Deploy on Streamlit Cloud

1. **Fork this repository** to your GitHub account

2. **Train and upload your model**:
   - Train the model using `ImColor.ipynb`
   - Upload `generator.pt` to Google Drive
   - Make it publicly accessible (Anyone with the link can view)
   - Get the shareable link and extract the FILE_ID
   - Update `app.py` line with your Google Drive link:
     ```python
     gdrive_url = "https://drive.google.com/uc?id=YOUR_FILE_ID"
     ```

3. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository and branch
   - Set main file path: `app.py`
   - Click "Deploy"!

4. **Your app is live!** 🎉

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- Streamlit
- FastAI
- Other dependencies in `requirements.txt`

## 🏗️ Model Architecture

### Generator
- **Base**: U-Net with ResNet34 backbone
- **Input**: Grayscale image (L channel in LAB color space)
- **Output**: AB color channels
- **Training**: Pre-trained with 10k COCO dataset images

### Discriminator
- **Architecture**: PatchGAN
- **Purpose**: Distinguishes real vs. generated colorizations
- **Training**: Adversarial training with Generator

## 📊 Training Details

The model uses:
- **Dataset**: COCO 2017 (10,000 images: 8k train, 2k validation)
- **Loss Functions**:
  - GAN loss (Binary Cross Entropy)
  - L1 loss (weighted by λ=100)
- **Optimizer**: Adam (lr=0.0004, β1=0.5, β2=0.999)
- **Training**: 5 epochs with batch size 16

## 🎯 How to Use the App

1. **Upload Image**: Click "Browse files" or drag & drop a B&W image
2. **Click Colorize**: Press the "Colorize Image" button
3. **View Results**: See original and colorized images side by side
4. **Download**: Save your colorized image

## 📁 Project Structure

```
Colorizing-Black-And-White-Images/
├── app.py                  # Streamlit web application
├── models_utils.py         # Model architectures (Generator, Discriminator)
├── utils.py                # Utility functions for training
├── ImColor.ipynb           # Training notebook
├── requirements.txt        # Python dependencies
├── .streamlit/
│   └── config.toml        # Streamlit configuration
├── img/                   # Sample images
└── README.md              # This file
```

## 🔧 Deployment on Streamlit Cloud - Step by Step

### Step 1: Prepare Your Model

```bash
# Option A: Train your own model
# 1. Open ImColor.ipynb
# 2. Run all cells
# 3. Model saves as generator.pt

# Option B: Use a pre-trained model
# Download from Google Drive or other source
```

### Step 2: Upload Model to Google Drive

1. Go to [Google Drive](https://drive.google.com)
2. Upload `generator.pt`
3. Right-click → Share → Change to "Anyone with the link"
4. Copy the link (format: `https://drive.google.com/file/d/FILE_ID/view?usp=sharing`)
5. Extract the `FILE_ID` from the URL

### Step 3: Update app.py

```python
# In app.py, find this line (around line 48):
gdrive_url = None

# Replace with:
gdrive_url = "https://drive.google.com/uc?id=YOUR_FILE_ID"
```

### Step 4: Push to GitHub

```bash
git add .
git commit -m "Add model download link"
git push origin main
```

### Step 5: Deploy

1. Visit [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Repository: `your-username/Colorizing-Black-And-White-Images`
4. Branch: `main` (or your branch name)
5. Main file path: `app.py`
6. Click "Deploy"

Wait 2-5 minutes for deployment. Your app will be live at:
`https://your-username-colorizing-black-and-white-images.streamlit.app`

## 💡 Tips for Best Results

- Use high-contrast black & white images
- Natural photographs work best (landscapes, portraits)
- Historical photos give interesting results
- Line drawings may have varying quality

## 🐛 Troubleshooting

### Model not loading?
- Ensure `generator.pt` is in the correct location
- Check Google Drive link is public and FILE_ID is correct
- Verify file size is under GitHub's limits (use Git LFS for large files)

### Out of memory errors?
- Streamlit Cloud has memory limits (~1GB)
- Ensure model file is optimized
- Consider using CPU-only version

### Slow colorization?
- First run may be slower (downloading model)
- Subsequent runs are cached and faster

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **PyTorch** and **FastAI** for deep learning frameworks
- **Streamlit** for the amazing web app framework
- **COCO Dataset** for training images

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**Made with ❤️ using PyTorch, FastAI, and Streamlit**
