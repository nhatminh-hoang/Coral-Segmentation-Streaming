"""
Segmenter Worker Process
========================
Standalone script that continuously processes video into 60-second chunks.
Runs as a separate OS process for true CPU parallelism.
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

# Add parent directory to path for inference module
sys.path.insert(0, str(Path(__file__).parent))

from inference import CoralSegModel, id2label, label2color, COLOR_LUT

# ==============================
# CONFIG
# ==============================
VIDEO_DIR = Path("sample_videos")
CHUNK_DIR = Path("chunks")
ALLOWED_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm")
DEFAULT_SKIP = 3
MAX_CHUNKS = 5
TARGET_FPS = 15
CHUNK_DURATION_SEC = 60

# ==============================
# MODEL INITIALIZATION
# ==============================
print("🧠 Loading segmentation model...")
model = CoralSegModel(use_onnx=False)
print("✅ Model loaded.")

# ==============================
# HELPERS
# ==============================
def list_video_files(directory: Path):
    if not directory.exists():
        return []
    files = []
    for p in sorted(directory.iterdir()):
        if p.is_file() and p.suffix.lower() in ALLOWED_EXTS:
            files.append(p)
    return files

def rotate_chunks():
    """Keep only the MAX_CHUNKS latest chunks, delete older ones."""
    mp4_files = sorted(CHUNK_DIR.glob("chunk_*.mp4"), key=os.path.getmtime)
    while len(mp4_files) > MAX_CHUNKS:
        oldest = mp4_files.pop(0)
        pred_file = Path(str(oldest).replace(".mp4", "_preds.npy"))
        try:
            oldest.unlink()
            pred_file.unlink(missing_ok=True)
            print(f"🗑️ Rotated out old chunk: {oldest.name}")
        except Exception as e:
            print(f"Error rotating chunk: {e}")

def create_filtered_overlay(pred_map: np.ndarray, base_rgb: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Create overlay with all labels shown."""
    filtered_pred = np.clip(pred_map, 0, COLOR_LUT.shape[0] - 1)
    rgb_mask = COLOR_LUT[filtered_pred]
    blended = cv2.addWeighted(base_rgb, 1.0 - alpha, rgb_mask, alpha, 0.0)
    return blended

# ==============================
# MAIN PROCESSING LOOP
# ==============================
def process_one_chunk(source: str):
    """Process one 60-second video chunk."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"❌ Cannot open source: {source}")
        return False

    ts = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    video_p = CHUNK_DIR / f"chunk_{ts}.mp4"
    pred_p = CHUNK_DIR / f"chunk_{ts}_preds.npy"
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    
    all_preds = []
    frames_captured = 0
    target_frames = CHUNK_DURATION_SEC * TARGET_FPS
    
    print(f"🔨 Segmenting new chunk: {ts}...")
    start_time = time.time()
    
    idx = 0
    while frames_captured < target_frames:
        ok, frame = cap.read()
        if not ok:
            # Loop video if it ends before 60s
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        # Skip frames for target FPS
        if idx % DEFAULT_SKIP != 0:
            idx += 1
            continue
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pred_map, _, base_rgb = model.predict_map_and_overlay(frame_rgb)
        overlay_rgb = create_filtered_overlay(pred_map, base_rgb)
        
        if out is None:
            h, w = overlay_rgb.shape[:2]
            out = cv2.VideoWriter(str(video_p), fourcc, float(TARGET_FPS), (w, h))
        
        out.write(cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))
        
        # Downsample pred map for hover
        scale = 4
        pred_small = pred_map[::scale, ::scale].astype(np.uint8)
        all_preds.append(pred_small)
        
        frames_captured += 1
        idx += 1

    cap.release()
    if out: out.release()
    
    elapsed = time.time() - start_time
    
    if frames_captured > 0:
        np.save(pred_p, np.array(all_preds))
        print(f"✅ Chunk {ts} saved ({frames_captured} frames in {elapsed:.1f}s)")
        rotate_chunks()  # Rotate AFTER saving (so new chunk is counted)
        return True
    else:
        if video_p.exists(): video_p.unlink()
        if pred_p.exists(): pred_p.unlink()
        return False

def main():
    """Main entry point for segmenter worker."""
    CHUNK_DIR.mkdir(exist_ok=True)
    
    print("=" * 50)
    print("🎬 SEGMENTER WORKER STARTED")
    print(f"   Source: {VIDEO_DIR}")
    print(f"   Output: {CHUNK_DIR}")
    print(f"   Max Chunks: {MAX_CHUNKS}")
    print("=" * 50)
    
    while True:
        try:
            # Get video source
            videos = list_video_files(VIDEO_DIR)
            if not videos:
                print("⏳ No video files found. Waiting...")
                time.sleep(5)
                continue
            
            source = str(videos[0])
            process_one_chunk(source)
            
        except KeyboardInterrupt:
            print("\n🛑 Segmenter worker stopped by user.")
            break
        except Exception as e:
            print(f"❌ Segmenter error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()
