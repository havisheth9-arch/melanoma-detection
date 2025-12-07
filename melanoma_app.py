import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import timm
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

# ------------------------------
# 1. Load model
# ------------------------------
MODEL_PATH = "efficientnet_b0_best.pth"

@st.cache_resource
def load_model():
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=7)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

# Class labels:
CLASSES = [
    "Actinic keratoses",
    "Basal cell carcinoma",
    "Benign keratosis",
    "Dermatofibroma",
    "Melanoma",
    "Melanocytic nevus",
    "Vascular lesion"
]

# ------------------------------
# 2. Image transforms
# ------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ------------------------------
# 3. Streamlit UI
# ------------------------------
st.set_page_config(page_title="Melanoma Classifier", layout="wide")
st.title("🔬 Melanoma Detection with Grad-CAM")
st.write("Upload a dermoscopic image to classify and visualize the important regions.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Show original image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    input_tensor = transform(img).unsqueeze(0)

    # ------------------------------
    # 4. Prediction
    # ------------------------------
    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1)[0]

    top_class = torch.argmax(probs).item()
    confidence = probs[top_class].item() * 100

    st.subheader("Prediction")
    st.write(f"**{CLASSES[top_class]}** ({confidence:.2f}% confidence)")

    # ------------------------------
    # 5. Grad-CAM
    # ------------------------------
    target_layer = model.conv_head  # EfficientNet last conv layer

    cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=False)

    rgb_img = np.float32(img.resize((224, 224))) / 255
    input_tensor_cam = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    grayscale_cam = cam(input_tensor=input_tensor_cam, targets=None)[0]
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    st.subheader("Grad-CAM Heatmap")
    st.image(visualization, use_column_width=True)

    # ------------------------------
    # 6. Probability Bar Chart
    # ------------------------------
    st.subheader("Class Probabilities")
    for label, p in zip(CLASSES, probs):
        st.write(f"{label}: **{p.item()*100:.2f}%**")
