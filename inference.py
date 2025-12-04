import os

import torch
import torch.nn.functional as F

import time
from collections import deque

import cv2
import numpy as np
from PIL import Image
from transformers import SegformerImageProcessorFast, SegformerForSemanticSegmentation

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: onnxruntime not installed. ONNX inference will not be available.")

id2label   = {
    '1': 'seagrass',
    '2': 'trash',
    '3': 'other coral dead',
    '4': 'other coral bleached',
    '5': 'sand',
    '6': 'other coral alive',
    '7': 'human',
    '8': 'transect tools',
    '9': 'fish',
    '10': 'algae covered substrate',
    '11': 'other animal',
    '12': 'unknown hard substrate',
    '13': 'background',
    '14': 'dark',
    '15': 'transect line',
    '16': 'massive/meandering bleached',
    '17': 'massive/meandering alive',
    '18': 'rubble',
    '19': 'branching bleached',
    '20': 'branching dead',
    '21': 'millepora',
    '22': 'branching alive',
    '23': 'massive/meandering dead',
    '24': 'clam',
    '25': 'acropora alive',
    '26': 'sea cucumber',
    '27': 'turbinaria',
    '28': 'table acropora alive',
    '29': 'sponge',
    '30': 'anemone',
    '31': 'pocillopora alive',
    '32': 'table acropora dead',
    '33': 'meandering bleached',
    '34': 'stylophora alive',
    '35': 'sea urchin',
    '36': 'meandering alive',
    '37': 'meandering dead',
    '38': 'crown of thorn',
    '39': 'dead clam'
 }

label2color= {'human': [255, 0, 0], 'background': [29, 162, 216], 'fish': [255, 255, 0], 'sand': [194, 178, 128], 'rubble': [161, 153, 128], 'unknown hard substrate': [125, 125, 125], 'algae covered substrate': [125, 163, 125], 'dark': [31, 31, 31], 'branching bleached': [252, 231, 240], 'branching dead': [123, 50, 86], 'branching alive': [226, 91, 157], 'stylophora alive': [255, 111, 194], 'pocillopora alive': [255, 146, 150], 'acropora alive': [236, 128, 255], 'table acropora alive': [189, 119, 255], 'table acropora dead': [85, 53, 116], 'millepora': [244, 150, 115], 'turbinaria': [228, 255, 119], 'other coral bleached': [250, 224, 225], 'other coral dead': [114, 60, 61], 'other coral alive': [224, 118, 119], 'massive/meandering alive': [236, 150, 21], 'massive/meandering dead': [134, 86, 18], 'massive/meandering bleached': [255, 248, 228], 'meandering alive': [230, 193, 0], 'meandering dead': [119, 100, 14], 'meandering bleached': [251, 243, 216], 'transect line': [0, 255, 0], 'transect tools': [8, 205, 12], 'sea urchin': [0, 142, 255], 'sea cucumber': [0, 231, 255], 'anemone': [0, 255, 189], 'sponge': [240, 80, 80], 'clam': [189, 255, 234], 'other animal': [0, 255, 255], 'trash': [255, 0, 134], 'seagrass': [125, 222, 125], 'crown of thorn': [179, 245, 234], 'dead clam': [89, 155, 134]}                        # {'seagrass':[R,G,B],...}

# Vietnamese translations for labels
label2vietnamese = {
    'seagrass': 'Cỏ biển',
    'trash': 'Rác thải',
    'other coral dead': 'San hô khác (chết)',
    'other coral bleached': 'San hô khác (bị tẩy trắng)',
    'sand': 'Cát',
    'other coral alive': 'San hô khác (sống)',
    'human': 'Con người',
    'transect tools': 'Dụng cụ khảo sát',
    'fish': 'Cá',
    'algae covered substrate': 'Nền phủ tảo',
    'other animal': 'Động vật khác',
    'unknown hard substrate': 'Nền cứng không xác định',
    'background': 'Nền',
    'dark': 'Vùng tối',
    'transect line': 'Dây khảo sát',
    'massive/meandering bleached': 'San hô dạng khối / mê cung (bị tẩy trắng)',
    'massive/meandering alive': 'San hô dạng khối / mê cung (sống)',
    'rubble': 'Đá vụn',
    'branching bleached': 'San hô dạng phân nhánh (bị tẩy trắng)',
    'branching dead': 'San hô dạng phân nhánh (chết)',
    'millepora': 'San hô lửa',
    'branching alive': 'San hô dạng phân nhánh (sống)',
    'massive/meandering dead': 'San hô dạng khối / mê cung (chết)',
    'clam': 'Nghêu/Sò',
    'acropora alive': 'San hô Acropora (sống)',
    'sea cucumber': 'Hải sâm',
    'turbinaria': 'San hô mũ (Turbinaria)',
    'table acropora alive': 'San hô bàn Acropora (sống)',
    'sponge': 'Bọt biển',
    'anemone': 'Hải quỳ',
    'pocillopora alive': 'San hô Pocillopora (sống)',
    'table acropora dead': 'San hô bàn Acropora (chết)',
    'meandering bleached': 'San hô dạng mê cung (tẩy trắng)',
    'stylophora alive': 'San hô Stylophora (sống)',
    'sea urchin': 'Nhím biển',
    'meandering alive': 'San hô dạng mê cung (sống)',
    'meandering dead': 'San hô dạng mê cung (chết)',
    'crown of thorn': 'Sao biển gai (Crown of Thorns)',
    'dead clam': 'Nghêu/Sò (chết)'
}

# Load model from HF (swap this with your own if you want)
HF_MODEL_ID = "EPFL-ECEO/segformer-b2-finetuned-coralscapes-1024-1024"
model_version = "b2" if "b2" in HF_MODEL_ID else "b5"

# Precompute once: class-id -> RGB and packed LUT
ID2COLOR = {int(i): label2color[label] for i, label in id2label.items()}
COLOR_LUT = np.zeros((max(ID2COLOR.keys()) + 1, 3), dtype=np.uint8)
for k, v in ID2COLOR.items():
    COLOR_LUT[k] = v

def create_segmentation_overlay(pred, image, alpha=0.25):
    """
    Fast overlay using precomputed COLOR_LUT + OpenCV addWeighted.
    pred: (H, W) uint8 class map
    image: PIL Image (RGB)
    """
    # Safety clip and table-lookup to colors
    pred = np.clip(pred, 0, COLOR_LUT.shape[0] - 1)
    rgb_mask = COLOR_LUT[pred]  # (H, W, 3) uint8

    img_array = np.asarray(image, dtype=np.uint8)

    # cv2.addWeighted: (1-alpha)*img + alpha*mask
    blended = cv2.addWeighted(img_array, 1.0 - alpha, rgb_mask, alpha, 0.0)
    return Image.fromarray(blended)

    

def resize_image(image: Image.Image, target_size=1024):
    """
    Resize PIL image to target_size x target_size using aspect-ratio preserving
    resize with padding (letterbox). Returns a new PIL image.
    """
    resized_img = image.resize((target_size, target_size), Image.BILINEAR)
    return resized_img

class CoralSegModel:
    def __init__(self, device=None, use_onnx=False, onnx_path=None):
        """
        Initialize the Coral Segmentation Model.
        
        Args:
            device: Device to run on ('cuda' or 'cpu'). Auto-detected if None.
            use_onnx: If True, use ONNX runtime instead of PyTorch model.
            onnx_path: Path to ONNX model file. Required if use_onnx=True.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_onnx = use_onnx
        
        self.processor = SegformerImageProcessorFast.from_pretrained(HF_MODEL_ID)

        if onnx_path is None and use_onnx:
            onnx_path = f"model/segformer{model_version}/model.onnx"
            print(f"No ONNX path provided. Using default: {onnx_path}")

        if use_onnx:
            if not ONNX_AVAILABLE:
                raise RuntimeError("Install onnxruntime-gpu for GPU inference.")

            so = ort.SessionOptions()
            # Full graph optimizations + save optimized graph (optional)
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            # so.optimized_model_filepath = f"model/segformer{model_version}/optimized.onnx"

            # CUDA EP options; see docs for meanings
            cuda_opts = {
                "device_id": 0,
                "arena_extend_strategy": "kNextPowerOfTwo",
                "do_copy_in_default_stream": 1,
                "cudnn_conv_use_max_workspace": 1,
                "cudnn_conv_algo_search": "HEURISTIC",
                "use_tf32": 1,           # Ampere+ can benefit
                # "prefer_nhwc": 0,      # leave default unless model supports NHWC
            }

            providers = [
                ("CUDAExecutionProvider", cuda_opts),
                ("CPUExecutionProvider", {}),
            ]

            # Optional: try TensorRT first (requires TRT + compatible ORT build)
            try_trt = False  # set True if TensorRT is installed
            if try_trt:
                trt_opts = {
                    "trt_fp16_enable": True,
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": "trt_cache",
                    # "trt_int8_enable": True,  # if you have calibration
                }
                providers.insert(0, ("TensorrtExecutionProvider", trt_opts))

            self.onnx_session = ort.InferenceSession(
                onnx_path,
                sess_options=so,
                providers=[p[0] for p in providers],
                provider_options=[p[1] for p in providers],
            )

            # Flip this on if you want to use I/O binding (see segment_image)
            self._use_iobinding = False
            self.model = None

            print(f"ONNX Runtime session created with providers: {self.onnx_session.get_providers()}")

        else:
            # Load PyTorch model
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                HF_MODEL_ID,
                dtype=torch.bfloat16
            ).to(self.device)
            self.model.eval()
            self.onnx_session = None

        # FPS tracking
        self.frame_times = deque(maxlen=30)  # Keep last 30 frame times for moving average
        self.total_frames = 0
        self.start_time = None

    def reset_fps_stats(self):
        """Reset FPS statistics"""
        self.frame_times.clear()
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
    def segment_image(self, image: Image.Image, target_size=512) -> np.ndarray:
        """
        Fast single-pass inference. Uses NumPy everywhere for ONNX;
        uses torch only if self.use_onnx is False.
        Returns a (H, W) uint8 class map in original image size.
        """
        # Let the image processor do resizing/normalization
        if self.use_onnx:
            # Ask for NumPy outputs to avoid torch altogether
            inputs = self.processor(
                image,
                return_tensors="pt",
                do_resize=True,
                size={"height": target_size, "width": target_size},
                do_rescale=True,
                do_normalize=True,
                # device='cpu'  # ensure on CPU for ONNX
            )
            pixel_values = inputs["pixel_values"].cpu().numpy().astype(np.float32)  # (1,3,H,W)

            if getattr(self, "_use_iobinding", False):
                # Optional: I/O binding to keep buffers on device
                io = self.onnx_session.io_binding()
                input_name = self.onnx_session.get_inputs()[0].name
                inp = ort.OrtValue.ortvalue_from_numpy(pixel_values, "cuda", 0)
                io.bind_input(input_name, inp)
                for out in self.onnx_session.get_outputs():
                    io.bind_output(out.name, "cuda")
                self.onnx_session.run_with_iobinding(io)
                # copy outputs to CPU once
                onnx_outputs = io.copy_outputs_to_cpu()
                logits = onnx_outputs[0]  # (1,C,h,w) float32
            else:
                onnx_inputs = {self.onnx_session.get_inputs()[0].name: pixel_values}
                logits = self.onnx_session.run(None, onnx_inputs)[0]  # np.ndarray

            # Argmax on CPU (NumPy) then upsample to original size with nearest
            small = logits.argmax(axis=1)[0].astype(np.uint8)         # (h,w)
            pred = cv2.resize(
                small,
                image.size,  # (W, H)
                interpolation=cv2.INTER_NEAREST
            )
            return pred

        else:
            # PyTorch path
            inputs = self.processor(
                image,
                return_tensors="pt",
                do_resize=True,
                size={"height": target_size, "width": target_size},
                do_rescale=True,
                do_normalize=True,
            )
            pixel_values = inputs["pixel_values"].to(self.device)
            outputs = self.model(pixel_values=pixel_values)
            logits = outputs.logits  # [1, C, h, w]
            preds = logits.argmax(dim=1)[0].to("cpu").numpy().astype(np.uint8)
            pred = cv2.resize(preds, image.size, interpolation=cv2.INTER_NEAREST)
            return pred

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
        pred = self.segment_image(pil, target_size=480)
        overlay_rgb = create_segmentation_overlay(pred, pil, 0.45)

        # Track timing
        end_time = time.time()
        frame_time = end_time - start_time
        self.frame_times.append(frame_time)
        self.total_frames += 1
        
        return pred, overlay_rgb, rgb

if __name__ == "__main__":
    model = CoralSegModel(use_onnx=True)
    
    # Run benchmark
    for _ in range(100):
        img = np.random.randint(0, 255, (1080,1920,3), dtype=np.uint8)
        pred_map, overlay, rgb = model.predict_map_and_overlay(img)
        fps_stats = model.get_fps_stats()
        print(f"Processed frame {fps_stats['total_frames']} - Current FPS: {fps_stats['current_fps']}, Average FPS: {fps_stats['average_fps']}")