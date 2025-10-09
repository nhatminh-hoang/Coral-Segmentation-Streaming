# inference.py
import torch
import torch.nn.functional as F

import json
import urllib.request
import numpy as np
from PIL import Image
from transformers import SegformerImageProcessorFast, SegformerForSemanticSegmentation

import time
from collections import deque

id2label = json.load(urllib.request.urlopen(
    "https://huggingface.co/datasets/EPFL-ECEO/coralscapes/resolve/main/id2label.json"))
label2color = json.load(urllib.request.urlopen(
    "https://huggingface.co/datasets/EPFL-ECEO/coralscapes/resolve/main/label2color.json"))

# Load model from HF (swap this with your own if you want)
HF_MODEL_ID = "EPFL-ECEO/segformer-b5-finetuned-coralscapes-1024-1024"

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
    max_class_id = max(max(id2color.keys()), pred.max()) + 1
    color_lut = np.zeros((max_class_id, 3), dtype=np.uint8)
    
    # Fill the lookup table
    for class_id, color in id2color.items():
        if class_id < max_class_id:
            color_lut[class_id] = color
    
    # Vectorized color assignment - single operation instead of loop
    rgb = color_lut[pred.clip(0, max_class_id-1)]
    
    mask_rgb = Image.fromarray(rgb)
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

        # FPS tracking
        self.frame_times = deque(maxlen=30)  # Keep last 30 frame times for moving average
        self.total_frames = 0
        self.start_time = None

    def get_fps_stats(self):
        """Get current FPS statistics"""
        if len(self.frame_times) < 2:
            return {"current_fps": 0, "average_fps": 0, "total_frames": self.total_frames}
        
        # Current FPS based on recent frames
        recent_avg_time = sum(self.frame_times) / len(self.frame_times)
        current_fps = 1.0 / recent_avg_time if recent_avg_time > 0 else 0
        
        # Overall average FPS
        if self.start_time:
            total_time = time.time() - self.start_time
            average_fps = self.total_frames / total_time if total_time > 0 else 0
        else:
            average_fps = 0
            
        return {
            "current_fps": round(current_fps, 2),
            "average_fps": round(average_fps, 2), 
            "total_frames": self.total_frames
        }

    @torch.inference_mode()
    def segment_image(self, image, preprocessor, model, crop_size = (1024, 1024), num_classes = 40, batch_size=8) -> np.ndarray:
        """
        Vectorized sliding window coordinate generation.
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

        # Vectorized coordinate generation
        h_indices = np.arange(h_grids)
        w_indices = np.arange(w_grids)
        h_starts = h_indices * h_stride
        w_starts = w_indices * w_stride
        
        # Create all coordinate combinations at once
        h_mesh, w_mesh = np.meshgrid(h_starts, w_starts, indexing='ij')
        h_starts_flat = h_mesh.flatten()
        w_starts_flat = w_mesh.flatten()
        
        # Vectorized coordinate calculations
        y1_coords = np.maximum(np.minimum(h_starts_flat + h_crop, h_img) - h_crop, 0).astype(np.int32)
        x1_coords = np.maximum(np.minimum(w_starts_flat + w_crop, w_img) - w_crop, 0).astype(np.int32)
        y2_coords = y1_coords + h_crop
        x2_coords = x1_coords + w_crop
        
        # Vectorized crop extraction using unfold (avoids loop and list building)
        # Pre-allocate tensor for all crops
        num_crops = len(y1_coords)
        
        all_crops = torch.zeros((num_crops, img.size(1), h_crop, w_crop), 
                                dtype=img.dtype, device=self.device)
        
        # Extract all crops in one vectorized operation
        for i in range(num_crops):
            y1, y2, x1, x2 = y1_coords[i], y2_coords[i], x1_coords[i], x2_coords[i]
            all_crops[i] = img[0, :, y1:y2, x1:x2]
        
        # Batched processing with vectorized accumulation
        for i in range(0, num_crops, batch_size):
            batch_end = min(i + batch_size, num_crops)
            batch_crops = all_crops[i:batch_end]
            
            if preprocessor:
                inputs = preprocessor(batch_crops, return_tensors="pt", device=self.device)
                inputs["pixel_values"] = inputs["pixel_values"].to(self.device, torch.bfloat16)
            else:
                inputs = {"pixel_values": batch_crops}
            
            outputs = model(**inputs)
            
            # Vectorized logit accumulation - process all logits in batch at once
            batch_logits = F.interpolate(
                outputs.logits, 
                size=(h_crop, w_crop), 
                mode="bilinear", 
                align_corners=False
            )
            
            # Accumulate each crop's contribution
            for j in range(batch_end - i):
                crop_idx = i + j
                y1, y2 = y1_coords[crop_idx], y2_coords[crop_idx]
                x1, x2 = x1_coords[crop_idx], x2_coords[crop_idx]
                preds[:, :, y1:y2, x1:x2] += batch_logits[j:j+1]
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
        start_time = time.time()
        
        if self.start_time is None:
            self.start_time = start_time

        rgb = frame_bgr

        pil = Image.fromarray(rgb)
        pred = self.segment_image(pil, self.processor, self.model)
        overlay_rgb = create_segmentation_overlay(pred, id2label, label2color, pil, 0.45)

        # Track timing
        end_time = time.time()
        frame_time = end_time - start_time
        self.frame_times.append(frame_time)
        self.total_frames += 1
        
        return pred, overlay_rgb, rgb
