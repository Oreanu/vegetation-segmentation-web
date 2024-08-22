import streamlit as st
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
from transformers import SegformerForSemanticSegmentation, AutoImageProcessor
import torchvision.transforms as T

# Function to load the PSPNet model from an uploaded file
@st.cache_resource
def load_pspnet_model(uploaded_model, num_classes=25, encoder_name='resnet101'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = smp.PSPNet(
        encoder_name=encoder_name,
        encoder_weights=None,  # Not loading any pre-trained weights
        classes=num_classes,
        activation=None  # Use None for logits
    )
    
    state_dict = torch.load(uploaded_model, map_location=device)

    # Remove 'model.' prefix from state_dict keys if they exist
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('model.'):
            new_key = key[len('model.'):]
        else:
            new_key = key
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    return model, device

# Function to load the Segformer model from an uploaded file
@st.cache_resource
def load_segformer_model(uploaded_model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = SegformerForSemanticSegmentation.from_pretrained(
        'nvidia/segformer-b3-finetuned-ade-512-512',
        num_labels=25,
        ignore_mismatched_sizes=True
    )
    
    state_dict = torch.load(uploaded_model, map_location=device)
    
    # Remove 'model.' prefix if it exists
    if any(key.startswith('model.') for key in state_dict.keys()):
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace('model.', '')
            new_state_dict[new_key] = value
        state_dict = new_state_dict
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, device

# Load the processor for Segformer
@st.cache_resource
def load_processor():
    return AutoImageProcessor.from_pretrained('nvidia/segformer-b3-finetuned-ade-512-512', do_rescale=False)

# Function to process and visualize the results using PSPNet
def visualize_pspnet(image, model, device):
    # Convert the image to RGB if it has an alpha channel
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    transform = T.Compose([
        T.Resize([368, 368]),  # Resizing image to the required input size
        T.ToTensor(),  # Converting the image to a tensor
    ])
    
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        outputs = torch.nn.functional.interpolate(outputs, size=(image.size[1], image.size[0]), mode='bilinear', align_corners=False)
        preds = outputs.argmax(dim=1)

    pred_mask = preds[0].cpu().numpy()
    return pred_mask

# Function to process and visualize the results using Segformer
def visualize_segformer(image, model, processor, device):
    # Convert the image to RGB if it has an alpha channel
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    inputs = processor(images=image, return_tensors="pt", do_rescale=False)
    inputs = inputs['pixel_values'].to(device)

    with torch.no_grad():
        outputs = model(inputs)
        logits = outputs.logits
        preds = logits.argmax(dim=1)

    pred_mask = preds[0].cpu().numpy()
    return pred_mask

# Function to create an overlay image
def create_overlay(image, mask, alpha=0.5):
    # Convert the image to RGB if it has an alpha channel
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    
    image_np = np.array(image).astype(float)
    
    # Ensure the mask is in uint8 format
    mask_uint8 = mask.astype(np.uint8)
    
    # Resize the mask to match the image size
    mask_resized = Image.fromarray(mask_uint8).resize((image_np.shape[1], image_np.shape[0]), Image.NEAREST)
    mask_resized_np = np.array(mask_resized)

    # Apply the mask using a colormap (viridis in this case)
    cmap = plt.get_cmap('viridis')
    mask_colored = cmap(mask_resized_np / mask_resized_np.max())[:, :, :3] * 255.0
    mask_rgb = mask_colored.astype(np.uint8)
    
    # Ensure image_np has 3 channels
    if image_np.shape[2] == 4:
        image_np = image_np[:, :, :3]  # Remove the alpha channel

    overlay = (1 - alpha) * image_np + alpha * mask_rgb
    overlay = overlay.astype(np.uint8)
    return Image.fromarray(overlay)

# Streamlit UI
st.title("PSPNet vs Segformer Image Segmentation")

# Upload the models
uploaded_pspnet_model_file = st.file_uploader("Upload your PSPNet model (pspnet_final.pt)", type=["pt"])
uploaded_segformer_model_file = st.file_uploader("Upload your Segformer model (segformer_final.pt)", type=["pt"])

if uploaded_pspnet_model_file and uploaded_segformer_model_file:
    # Load the models and processor
    pspnet_model, pspnet_device = load_pspnet_model(uploaded_pspnet_model_file)
    segformer_model, segformer_device = load_segformer_model(uploaded_segformer_model_file)
    processor = load_processor()

    # Upload image
    uploaded_image_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if uploaded_image_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_image_file)

        # Run segmentation using PSPNet
        st.write("Running segmentation with PSPNet...")
        pspnet_pred_mask = visualize_pspnet(image, pspnet_model, pspnet_device)

        # Run segmentation using Segformer
        st.write("Running segmentation with Segformer...")
        segformer_pred_mask = visualize_segformer(image, segformer_model, processor, segformer_device)

        # Create overlay images
        pspnet_overlay_image = create_overlay(image, pspnet_pred_mask)
        segformer_overlay_image = create_overlay(image, segformer_pred_mask)

        # Layout: Original Image, PSPNet Overlay, Segformer Overlay
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(image, caption="Original Image", use_column_width=True)
        with col2:
            st.image(pspnet_overlay_image, caption="PSPNet Overlay Image", use_column_width=True)
        with col3:
            st.image(segformer_overlay_image, caption="Segformer Overlay Image", use_column_width=True)
        
        st.write("Segmentation completed with both models.")
