# inference.py
import torch
import torch.nn.functional as F

import json
import urllib.request
import cv2
import numpy as np
from PIL import Image
from transformers import SegformerImageProcessorFast, SegformerForSemanticSegmentation

id2label = json.load(urllib.request.urlopen(
    "https://huggingface.co/datasets/EPFL-ECEO/coralscapes/resolve/main/id2label.json"))
label2color = json.load(urllib.request.urlopen(
    "https://huggingface.co/datasets/EPFL-ECEO/coralscapes/resolve/main/label2color.json"))

# Load model from HF (swap this with your own if you want)
HF_MODEL_ID = "EPFL-ECEO/segformer-b2-finetuned-coralscapes-1024-1024"

def create_segmentation_overlay(pred, id2label, label2color, image, alpha=0.25):
    """
    Colorizes the segmentation prediction and creates an overlay image.

    Args:
        pred: The segmentation prediction (numpy array).
        id2label: Dictionary mapping class IDs to labels.
        label2color: Dictionary mapping labels to colors.
        image: The original PIL Image.

    Returns:
        A PIL Image representing the overlay of the original image and the colorized segmentation mask.
    """
    H, W = pred.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)

    # Get unique class IDs present in the prediction
    unique_classes = np.unique(pred)

    # Create a mapping from class ID to color
    id2color = {int(id): label2color[label] for id, label in id2label.items()}

    # Define a default color for unknown classes (e.g., black)
    default_color = [0, 0, 0]

    # Iterate through unique class IDs and colorize the image
    for class_id in unique_classes:
        # Get the color for the current class ID, use default_color if not found
        rgb_c = id2color.get(int(class_id), default_color)
        # Assign the color to the pixels with the current class ID
        rgb[pred == class_id] = rgb_c

    mask_rgb = Image.fromarray(rgb)

    # 4) Alpha overlay
    overlay = Image.blend(image.convert("RGBA"), mask_rgb.convert("RGBA"), alpha=alpha)

    return overlay

def resize_image(image, target_size=1024):
    """
    Used to resize the image such that the smaller side equals 1024
    """
    h_img, w_img = image.size
    if h_img < w_img:
        new_h, new_w = target_size, int(w_img * (target_size / h_img))
    else:
        new_h, new_w  = int(h_img * (target_size / w_img)), target_size
    resized_img = image.resize((new_h, new_w))
    return resized_img

class CoralSegModel:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = SegformerImageProcessorFast.from_pretrained(HF_MODEL_ID)

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            HF_MODEL_ID,
            dtype=torch.bfloat16
        ).to(self.device)

        self.model.eval()

    @torch.inference_mode()
    def segment_image(self, image, preprocessor, model, crop_size = (1024, 1024), num_classes = 40, batch_size=4) -> np.ndarray:
        """
        Batched sliding window inference for improved GPU utilization.
        """
        h_crop, w_crop = crop_size

        img = torch.Tensor(np.array(resize_image(image, target_size=1024)).transpose(2, 0, 1)).unsqueeze(0)
        img = img.to(self.device, torch.bfloat16)
        _, _, h_img, w_img = img.size()

        h_grids = int(np.round(3/2*h_img/h_crop)) if h_img > h_crop else 1
        w_grids = int(np.round(3/2*w_img/w_crop)) if w_img > w_crop else 1

        h_stride = int((h_img - h_crop + h_grids -1)/(h_grids -1)) if h_grids > 1 else h_crop
        w_stride = int((w_img - w_crop + w_grids -1)/(w_grids -1)) if w_grids > 1 else w_crop

        preds = img.new_zeros((1, num_classes, h_img, w_img))
        count_mat = img.new_zeros((1, 1, h_img, w_img))

        # Collect all crops and their coordinates
        crops = []
        coords = []
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                
                crop_img = img[:, :, y1:y2, x1:x2]
                crops.append(crop_img)
                coords.append((x1, x2, y1, y2))
        
        # Process crops in batches
        for i in range(0, len(crops), batch_size):
            batch_crops = crops[i:i+batch_size]
            batch_coords = coords[i:i+batch_size]
            
            # Stack crops into a batch
            batch_tensor = torch.cat(batch_crops, dim=0)
            
            if preprocessor:
                inputs = preprocessor(batch_tensor, return_tensors="pt", device=self.device)
                inputs["pixel_values"] = inputs["pixel_values"].to(self.device, torch.bfloat16)
            else:
                inputs = {"pixel_values": batch_tensor}
            
            outputs = model(**inputs)
            
            # Process each output in the batch
            for j, (x1, x2, y1, y2) in enumerate(batch_coords):
                resized_logits = F.interpolate(
                    outputs.logits[j].unsqueeze(dim=0), 
                    size=(y2-y1, x2-x1), 
                    mode="bilinear", 
                    align_corners=False
                )
                preds[:, :, y1:y2, x1:x2] += resized_logits
                count_mat[:, :, y1:y2, x1:x2] += 1

        assert (count_mat == 0).sum() == 0
        preds = preds / count_mat
        preds = preds.argmax(dim=1)
        preds = F.interpolate(preds.unsqueeze(0).type(torch.uint8), size=image.size[::-1], mode='nearest')
        label_pred = preds.squeeze().cpu().numpy()
        return label_pred

    @torch.inference_mode()
    def predict_map_and_overlay(self, frame_bgr: np.ndarray):
        """
        Returns:
          pred_map:  HxW (uint8/int) with class indices in [0..C-1]
          overlay:   HxWx3 RGB uint8 (blended color mask over original)
          rgb:       HxWx3 RGB uint8 original frame (for AnnotatedImage base)
        """
        rgb = frame_bgr

        pil = Image.fromarray(rgb)
        pred = self.segment_image(pil, self.processor, self.model)
        overlay_rgb = create_segmentation_overlay(pred, id2label, label2color, pil, 0.45)
        
        return pred, overlay_rgb, rgb
