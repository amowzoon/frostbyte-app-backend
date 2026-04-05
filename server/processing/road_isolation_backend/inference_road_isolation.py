import torch
import torch.nn.functional as F
import numpy as np
import sys
import os
from PIL import Image
sys.path.append("../MobileSam")
from mobile_sam import sam_model_registry
from processing.road_isolation_backend.segmentation_head import SegmentationHead

def road_filter_inference(checkpoint_path, image, device='cuda', target_size=(1024, 1024)):
    """
    Run road segmentation on a single image and return the mask and overlay.

    Args:
        checkpoint_path (str): Path to the model checkpoint (.pth)
        image (PIL.Image.Image or np.ndarray): Input RGB image
        device (str): 'cuda' or 'cpu'
        target_size (tuple): (height, width) for resizing before inference

    Returns:
        mask_np (np.ndarray): 2D binary mask (0=background, 1=road/sidewalk)
        overlay (np.ndarray): RGB image with green overlay for roads
    """

    # Convert PIL to numpy if needed
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Ensure RGB
    if image.shape[-1] != 3:
        raise ValueError("Input image must have 3 channels (RGB)")

    # Resize image to target size
    from torchvision import transforms
    resize = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(target_size),
        transforms.ToTensor()
    ])
    img_tensor = resize(image).unsqueeze(0).to(device)  # shape: (1, 3, H, W)

    # Load model
    model_type = "vit_t"
    mobile_sam = sam_model_registry[model_type](checkpoint=None)
    encoder = mobile_sam.image_encoder.to(device)
    seg_head = SegmentationHead(in_channels=256, num_classes=2).to(device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Ensure float32 weights
    for key in checkpoint['encoder_state_dict']:
        checkpoint['encoder_state_dict'][key] = checkpoint['encoder_state_dict'][key].float()
    for key in checkpoint['seg_head_state_dict']:
        checkpoint['seg_head_state_dict'][key] = checkpoint['seg_head_state_dict'][key].float()

    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    seg_head.load_state_dict(checkpoint['seg_head_state_dict'])

    encoder.eval()
    seg_head.eval()

    # Run inference
    with torch.no_grad():
        embeddings = encoder(img_tensor)
        logits = seg_head(embeddings)
        logits = F.interpolate(logits, size=img_tensor.shape[-2:], mode='bilinear', align_corners=False)
        pred_mask = torch.argmax(logits, dim=1)  # shape: (1, H, W)
    
    # Convert mask to numpy, squeeze batch dimension
    mask_np = pred_mask.squeeze(0).cpu().numpy().astype(np.uint8)  # shape: (H, W)

    # Resize mask back to original image size if needed
    if mask_np.shape != image.shape[:2]:
        import cv2
        mask_np = cv2.resize(mask_np, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Create overlay
    colored_mask = np.zeros_like(image)
    colored_mask[mask_np == 1] = [0, 255, 0]  # green overlay
    overlay = (0.6 * image + 0.4 * colored_mask).astype(np.uint8)

    return mask_np, overlay
