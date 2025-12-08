"""
Production Coral Segmentation Streaming Web App
==============================================
New FastAPI server for production deployment with:
- Real-time WebSocket streaming for segmentation
- JSON-based label statistics API
- Modern web UI at root
- Keeps Gradio compatibility at /gradio
"""

import os
import json
import asyncio
from io import BytesIO
from pathlib import Path
import time as time_module
from typing import List, Optional, Dict, Any, Set
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from inference import CoralSegModel, id2label, label2color, label2vietnamese

# ==============================
# CONFIG
# ==============================
VIDEO_DIR = Path("sample_videos")
DATA_DIR = Path("data")
STATIC_DIR = Path("static")
ALLOWED_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm")
DEFAULT_SKIP = 3

# ==============================
# MODEL INITIALIZATION
# ==============================
model = CoralSegModel(use_onnx=False)  # Use PyTorch backend

# ==============================
# HELPERS
# ==============================
def rgb_to_hex(rgb):
    r, g, b = [int(x) for x in rgb]
    return f"#{r:02X}{g:02X}{b:02X}"

def list_video_files(directory: Path):
    if not directory.exists():
        return []
    files = []
    for p in sorted(directory.iterdir()):
        if p.is_file() and p.suffix.lower() in ALLOWED_EXTS:
            files.append(p)
    return files

def _safe_read(cap):
    ok, frame = cap.read()
    return frame if ok and frame is not None else None

# ==============================
# TIME-SERIES DATA STORAGE
# ==============================
from datetime import datetime, timedelta

STATS_FILE = DATA_DIR / "label_stats.json"
SAVE_INTERVAL_FRAMES = 30  # Save stats every N processed frames

def load_time_series() -> Dict[str, Any]:
    """Load time-series statistics from JSON file"""
    if STATS_FILE.exists():
        with open(STATS_FILE, "r") as f:
            data = json.load(f)
            # Handle legacy format - convert if needed
            if "time_series" not in data:
                return {"time_series": []}
            return data
    return {"time_series": []}

def save_time_series(data: Dict[str, Any]):
    """Save time-series statistics to JSON file"""
    DATA_DIR.mkdir(exist_ok=True)
    with open(STATS_FILE, "w") as f:
        json.dump(data, f, indent=2)

def append_frame_stats(label_stats: Dict[str, int]):
    """Append current frame stats with timestamp to time-series data"""
    if not label_stats:
        return
    
    data = load_time_series()
    entry = {
        "timestamp": datetime.now().isoformat(),
        "labels": dict(label_stats)
    }
    data["time_series"].append(entry)
    
    # Optional: Prune old data (older than 6 months)
    cutoff = datetime.now() - timedelta(days=180)
    data["time_series"] = [
        e for e in data["time_series"]
        if datetime.fromisoformat(e["timestamp"]) > cutoff
    ]
    
    save_time_series(data)

def compute_period_mean(period: str) -> Dict[str, Any]:
    """
    Calculate mean pixel area per label for a given time period.
    
    Args:
        period: One of 'day', 'week', 'month', '3month', '6month'
    
    Returns:
        Dict with 'labels', 'values' (mean pixel areas), 'period', 'sample_count'
    """
    period_days = {
        "day": 1,
        "week": 7,
        "month": 30,
        "3month": 90,
        "6month": 180
    }
    
    days = period_days.get(period, 1)
    cutoff = datetime.now() - timedelta(days=days)
    
    data = load_time_series()
    time_series = data.get("time_series", [])
    
    # Filter by time period
    filtered = []
    for entry in time_series:
        try:
            entry_time = datetime.fromisoformat(entry["timestamp"])
            if entry_time > cutoff:
                filtered.append(entry)
        except (ValueError, KeyError):
            continue
    
    if not filtered:
        return {"labels": [], "values": [], "period": period, "sample_count": 0}
    
    # Aggregate: compute mean per label
    label_totals: Dict[str, List[int]] = defaultdict(list)
    for entry in filtered:
        for label, count in entry.get("labels", {}).items():
            label_totals[label].append(count)
    
    # Compute means
    labels = []
    values = []
    for label, counts in sorted(label_totals.items(), key=lambda x: sum(x[1])/len(x[1]), reverse=True):
        mean_val = sum(counts) / len(counts)
        labels.append(label)
        values.append(round(mean_val, 2))
    
    return {
        "labels": labels,
        "values": values,
        "period": period,
        "sample_count": len(filtered),
        "unit": "mean_pixels"
    }

# Build color map
color_map = {}
for id_str, en_label in id2label.items():
    rgb = label2color.get(en_label, [255, 255, 255])
    hex_color = rgb_to_hex(rgb)
    color_map[en_label] = hex_color

# ==============================
# FASTAPI APP
# ==============================
app = FastAPI(title="CoralScapes Production Monitor")

# CORS for local network access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ==============================
# STATE MANAGEMENT
# ==============================
# Coral label keywords for filtering
CORAL_KEYWORDS = {'coral', 'branching', 'massive', 'meandering', 'acropora', 
                  'table acropora', 'pocillopora', 'stylophora', 'millepora'}

def _is_coral_label(label: str) -> bool:
    """Check if a label is coral-related"""
    s = label.lower()
    return any(k in s for k in CORAL_KEYWORDS)

class StreamState:
    """Manages global streaming state"""
    def __init__(self):
        # Default to coral-only labels active
        self.active_labels: Set[str] = set(
            label for label in id2label.values() if _is_coral_label(label)
        )
        self.skip_frames: int = DEFAULT_SKIP
        self.language: str = "Tiếng Việt"  # Vietnamese default
        self.frame_stats: Dict[str, int] = defaultdict(int)  # Current frame label counts
        self.camera_url: Optional[str] = None  # Camera/stream URL (None = use sample videos)
        self.source_changed: bool = False  # Flag to signal source change to processor
        self.processed_frames: int = 0  # Counter for periodic stats saving
    
    def toggle_label(self, label: str) -> bool:
        """Toggle a label on/off. Returns new state."""
        if label in self.active_labels:
            self.active_labels.discard(label)
            return False
        else:
            self.active_labels.add(label)
            return True
    
    def set_label_active(self, label: str, active: bool):
        """Set a specific label's active state"""
        if active:
            self.active_labels.add(label)
        else:
            self.active_labels.discard(label)
    
    def update_frame_stats(self, pred_map: np.ndarray):
        """Update label statistics from current frame prediction"""
        self.frame_stats.clear()
        if pred_map is not None:
            unique, counts = np.unique(pred_map, return_counts=True)
            for class_id, count in zip(unique, counts):
                str_id = str(class_id)
                if str_id in id2label:
                    label = id2label[str_id]
                    self.frame_stats[label] = int(count)
        
        # Increment counter and save periodically
        self.processed_frames += 1
        if self.processed_frames % SAVE_INTERVAL_FRAMES == 0:
            append_frame_stats(dict(self.frame_stats))

stream_state = StreamState()

# ==============================
# FRAME CACHE (5-Frame Ring Buffer)
# ==============================
import base64
from dataclasses import dataclass, field
from threading import Lock

@dataclass
class CachedFrame:
    """Single cached frame with all data needed for clients"""
    image_b64: str  # Base64 encoded JPEG
    pred_map_b64: str  # Base64 encoded downsampled prediction map
    pred_shape: List[int]  # [height, width] of downsampled pred map
    pred_scale: int  # Downsample scale factor
    fps: float
    avg_fps: float
    total_frames: int
    visible_labels: List[Dict]
    timestamp: float = field(default_factory=time_module.time)

class FrameCache:
    """Thread-safe cache storing the 5 most recent processed frames"""
    
    def __init__(self, max_size: int = 5):
        self.max_size = max_size
        self._frames: List[CachedFrame] = []
        self._lock = Lock()
        self._frame_event = asyncio.Event()  # Signal new frame availability
    
    def push(self, frame: CachedFrame):
        """Add a new frame to the cache (thread-safe)"""
        with self._lock:
            self._frames.append(frame)
            if len(self._frames) > self.max_size:
                self._frames.pop(0)  # Remove oldest
    
    def get_latest(self) -> Optional[CachedFrame]:
        """Get the most recent cached frame (thread-safe)"""
        with self._lock:
            return self._frames[-1] if self._frames else None
    
    def get_all(self) -> List[CachedFrame]:
        """Get all cached frames (thread-safe)"""
        with self._lock:
            return list(self._frames)
    
    def notify_new_frame(self):
        """Signal that a new frame is available"""
        self._frame_event.set()
    
    async def wait_for_frame(self, timeout: float = 1.0):
        """Wait for a new frame signal with timeout"""
        try:
            await asyncio.wait_for(self._frame_event.wait(), timeout=timeout)
            self._frame_event.clear()
            return True
        except asyncio.TimeoutError:
            return False
    
    def clear(self):
        """Clear all cached frames"""
        with self._lock:
            self._frames.clear()

# Global frame cache
frame_cache = FrameCache(max_size=5)

# ==============================
# OVERLAY CREATION HELPER
# ==============================
def create_filtered_overlay(pred_map: np.ndarray, base_rgb: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Create overlay with only active labels shown"""
    from inference import COLOR_LUT
    
    # Create a filtered color map
    filtered_pred = pred_map.copy()
    
    # Get label to ID mapping
    label2id = {label: int(id_str) for id_str, label in id2label.items()}
    
    # Set inactive labels to background (class 0 or transparent)
    for label, class_id in label2id.items():
        if label not in stream_state.active_labels:
            filtered_pred[pred_map == class_id] = 0  # Set to background
    
    # Apply color lookup
    filtered_pred = np.clip(filtered_pred, 0, COLOR_LUT.shape[0] - 1)
    rgb_mask = COLOR_LUT[filtered_pred]
    
    # Blend with original
    blended = cv2.addWeighted(base_rgb, 1.0 - alpha, rgb_mask, alpha, 0.0)
    return blended

# ==============================
# BACKGROUND FRAME PROCESSOR
# ==============================
class FrameProcessor:
    """Background processor that processes exactly 1 frame at a time"""
    
    def __init__(self):
        self._running = False
        self._task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start the background processing task"""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._process_loop())
        print("🎬 Background frame processor started")
    
    async def stop(self):
        """Stop the background processing task"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        print("🛑 Background frame processor stopped")
    
    async def _process_loop(self):
        """Main processing loop - runs continuously"""
        while self._running:
            try:
                await self._process_source()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Frame processor error: {e}")
                await asyncio.sleep(2.0)
    
    async def _process_source(self):
        """Process video from current source (camera or sample videos)"""
        model.reset_fps_stats()
        stream_state.source_changed = False
        
        if stream_state.camera_url:
            # Use camera/stream URL
            cap = cv2.VideoCapture(stream_state.camera_url)
            if not cap.isOpened():
                print(f"Cannot open camera: {stream_state.camera_url}")
                await asyncio.sleep(2.0)
                return
            
            try:
                await self._process_video(cap, "Camera Stream")
            finally:
                cap.release()
        else:
            # Use sample videos
            files = list_video_files(VIDEO_DIR)
            
            if not files:
                print("No videos found in ./sample_videos")
                await asyncio.sleep(2.0)
                return
            
            for path in files:
                if stream_state.source_changed or not self._running:
                    break
                    
                cap = cv2.VideoCapture(str(path))
                if not cap.isOpened():
                    print(f"Cannot open: {path.name}")
                    continue
                
                try:
                    await self._process_video(cap, path.name)
                finally:
                    cap.release()
        
        await asyncio.sleep(1.0)
    
    async def _process_video(self, cap, source_name: str):
        """Process a single video source"""
        idx = 0
        
        while self._running and not stream_state.source_changed:
            frame = _safe_read(cap)
            if frame is None:
                break
            
            skip = stream_state.skip_frames
            if skip > 1 and (idx % skip) != 0:
                idx += 1
                # Small yield to prevent blocking
                if idx % 10 == 0:
                    await asyncio.sleep(0)
                continue
            
            # Process frame (this is the GPU-intensive part - only happens ONCE per frame)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pred_map, overlay_pil, base_rgb = model.predict_map_and_overlay(frame_rgb)
            
            # Update global frame statistics
            stream_state.update_frame_stats(pred_map)
            
            # Create filtered overlay based on active labels
            overlay_rgb = create_filtered_overlay(pred_map, base_rgb)
            
            # Encode as JPEG
            _, buffer = cv2.imencode('.jpg', cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR), 
                                     [cv2.IMWRITE_JPEG_QUALITY, 85])
            image_b64 = base64.b64encode(buffer).decode('utf-8')
            
            # Get FPS info
            fps_stats = model.get_fps_stats()
            
            # Get visible labels stats
            visible_stats = []
            for label, count in stream_state.frame_stats.items():
                if label in stream_state.active_labels:
                    visible_stats.append({
                        "label": label,
                        "count": count,
                        "color": color_map.get(label, "#FFF")
                    })
            visible_stats.sort(key=lambda x: x["count"], reverse=True)
            
            # Downsample prediction map for hover feature
            h, w = pred_map.shape
            scale = 4
            pred_small = pred_map[::scale, ::scale]
            pred_bytes = pred_small.astype(np.uint8).tobytes()
            pred_b64 = base64.b64encode(pred_bytes).decode('utf-8')
            
            # Create cached frame
            cached = CachedFrame(
                image_b64=image_b64,
                pred_map_b64=pred_b64,
                pred_shape=[pred_small.shape[0], pred_small.shape[1]],
                pred_scale=scale,
                fps=fps_stats["current_fps"],
                avg_fps=fps_stats["average_fps"],
                total_frames=fps_stats["total_frames"],
                visible_labels=visible_stats[:8]
            )
            
            # Push to cache and notify waiting clients
            frame_cache.push(cached)
            frame_cache.notify_new_frame()
            
            idx += 1
            
            # Yield to event loop to allow WebSocket handlers to run
            await asyncio.sleep(0)

# Global frame processor
frame_processor = FrameProcessor()

# ==============================
# API ROUTES
# ==============================
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the production web app"""
    return FileResponse(STATIC_DIR / "index.html")

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

@app.get("/api/labels")
async def get_labels():
    """Get all label metadata with colors and translations"""
    labels = []
    for id_str, en_label in id2label.items():
        labels.append({
            "id": int(id_str),
            "name": en_label,
            "name_vn": label2vietnamese.get(en_label, en_label),
            "color": color_map.get(en_label, "#FFFFFF"),
            "color_rgb": label2color.get(en_label, [255, 255, 255]),
            "active": en_label in stream_state.active_labels
        })
    return {"labels": labels}

@app.get("/api/stats/{period}")
async def get_stats(period: str):
    """Get mean pixel area statistics for a time period (day/week/month/3month/6month)"""
    return compute_period_mean(period)

@app.get("/api/timeseries")
async def get_timeseries(limit: int = Query(default=50), labels_limit: int = Query(default=5)):
    """
    Get recent time-series data for line chart visualization.
    
    Args:
        limit: Maximum number of data points to return (default 50)
        labels_limit: Maximum number of labels to include (default 5, top by total count)
    
    Returns:
        Dict with timestamps, datasets (one per label), and label colors
    """
    data = load_time_series()
    time_series = data.get("time_series", [])
    
    if not time_series:
        return {"timestamps": [], "datasets": [], "labels": []}
    
    # Get the last N entries
    recent = time_series[-limit:] if len(time_series) > limit else time_series
    
    # Collect all labels and count total occurrences
    label_totals: Dict[str, int] = defaultdict(int)
    for entry in recent:
        for label, count in entry.get("labels", {}).items():
            label_totals[label] += count
    
    # Get top labels (coral labels prioritized)
    sorted_labels = sorted(label_totals.items(), key=lambda x: x[1], reverse=True)
    top_labels = [label for label, _ in sorted_labels[:labels_limit]]
    
    # Build timestamps and datasets
    timestamps = []
    datasets = {label: [] for label in top_labels}
    
    for entry in recent:
        try:
            ts = datetime.fromisoformat(entry["timestamp"])
            timestamps.append(ts.strftime("%H:%M:%S"))
        except (ValueError, KeyError):
            timestamps.append("")
        
        entry_labels = entry.get("labels", {})
        for label in top_labels:
            datasets[label].append(entry_labels.get(label, 0))
    
    # Format as list of dataset objects for Chart.js
    formatted_datasets = []
    for label in top_labels:
        color = color_map.get(label, "#FFFFFF")
        formatted_datasets.append({
            "label": label,
            "label_vn": label2vietnamese.get(label, label),
            "data": datasets[label],
            "color": color
        })
    
    return {
        "timestamps": timestamps,
        "datasets": formatted_datasets,
        "labels": top_labels
    }

@app.get("/api/current_stats")
async def get_current_stats():
    """Get current frame label statistics"""
    stats = []
    for label, count in stream_state.frame_stats.items():
        stats.append({
            "label": label,
            "label_vn": label2vietnamese.get(label, label),
            "count": count,
            "color": color_map.get(label, "#FFFFFF"),
            "percentage": round(count / (sum(stream_state.frame_stats.values()) or 1) * 100, 2)
        })
    # Sort by count descending
    stats.sort(key=lambda x: x["count"], reverse=True)
    return {"stats": stats[:10]}  # Top 10 labels

@app.post("/api/toggle_label/{label_name}")
async def toggle_label(label_name: str):
    """Toggle a label's visibility in the stream"""
    new_state = stream_state.toggle_label(label_name)
    return {"label": label_name, "active": new_state}

@app.post("/api/set_labels")
async def set_labels(labels: Dict[str, bool]):
    """Set multiple labels active/inactive"""
    for label, active in labels.items():
        stream_state.set_label_active(label, active)
    return {"active_labels": list(stream_state.active_labels)}

@app.post("/api/settings")
async def update_settings(skip: int = Query(default=3), language: str = Query(default="English")):
    """Update streaming settings"""
    stream_state.skip_frames = skip
    stream_state.language = language
    return {"skip": skip, "language": language}

@app.get("/api/videos")
async def list_videos():
    """List available video files"""
    files = [p.name for p in list_video_files(VIDEO_DIR)]
    return {"videos": files}

@app.post("/api/camera_url")
async def set_camera_url(url: str = ""):
    """Set camera/stream URL for live streaming"""
    if url and url.strip():
        stream_state.camera_url = url.strip()
        stream_state.source_changed = True  # Signal processor to switch source
        return {"camera_url": stream_state.camera_url, "message": "Camera URL set"}
    else:
        stream_state.camera_url = None
        stream_state.source_changed = True
        return {"camera_url": None, "message": "Using sample videos"}

@app.get("/api/camera_url")
async def get_camera_url():
    """Get current camera URL"""
    return {"camera_url": stream_state.camera_url}

# ==============================
# STARTUP/SHUTDOWN EVENTS
# ==============================
@app.on_event("startup")
async def startup_event():
    """Start the background frame processor on app startup"""
    await frame_processor.start()

@app.on_event("shutdown")
async def shutdown_event():
    """Stop the background frame processor on app shutdown"""
    await frame_processor.stop()

# ==============================
# WEBSOCKET STREAMING (Cache-based)
# ==============================
@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time video streaming.
    
    Clients receive frames from the shared cache instead of triggering
    individual GPU processing. This ensures O(1) GPU load regardless
    of the number of connected clients.
    """
    await websocket.accept()
    print(f"✅ WebSocket client connected (total: active)")
    
    # Track the last frame timestamp sent to this client
    last_sent_timestamp = 0.0
    
    try:
        # Send latest cached frame immediately if available (instant join experience)
        latest = frame_cache.get_latest()
        if latest:
            await websocket.send_json({
                "type": "frame",
                "image": latest.image_b64,
                "fps": latest.fps,
                "avg_fps": latest.avg_fps,
                "total_frames": latest.total_frames,
                "visible_labels": latest.visible_labels,
                "pred_map": latest.pred_map_b64,
                "pred_shape": latest.pred_shape,
                "pred_scale": latest.pred_scale
            })
            last_sent_timestamp = latest.timestamp
        
        while True:
            # Handle incoming commands from client (non-blocking)
            try:
                data = await asyncio.wait_for(
                    websocket.receive_json(),
                    timeout=0.001
                )
                # Handle incoming commands
                if data.get("type") == "toggle_label":
                    label = data.get("label")
                    if label:
                        stream_state.toggle_label(label)
                elif data.get("type") == "set_skip":
                    stream_state.skip_frames = data.get("skip", DEFAULT_SKIP)
                elif data.get("type") == "set_camera_url":
                    stream_state.camera_url = data.get("url") or None
                    stream_state.source_changed = True
            except asyncio.TimeoutError:
                pass
            
            # Wait for new frame or timeout
            await frame_cache.wait_for_frame(timeout=0.1)
            
            # Get latest frame from cache
            latest = frame_cache.get_latest()
            if latest and latest.timestamp > last_sent_timestamp:
                await websocket.send_json({
                    "type": "frame",
                    "image": latest.image_b64,
                    "fps": latest.fps,
                    "avg_fps": latest.avg_fps,
                    "total_frames": latest.total_frames,
                    "visible_labels": latest.visible_labels,
                    "pred_map": latest.pred_map_b64,
                    "pred_shape": latest.pred_shape,
                    "pred_scale": latest.pred_scale
                })
                last_sent_timestamp = latest.timestamp
            
    except WebSocketDisconnect:
        print("❌ WebSocket client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")

# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
