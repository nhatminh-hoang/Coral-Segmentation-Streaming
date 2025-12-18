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
import re
import json
import base64
import asyncio
from threading import Lock
from io import BytesIO
from pathlib import Path
import time as time_module
from dataclasses import dataclass, field
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
# CHUNK MANAGEMENT
# ==============================
CHUNK_DIR = Path("chunks")
MAX_CHUNKS = 5

@dataclass
class ChunkInfo:
    timestamp: str  # Format: DD_MM_YYYY_HH_MM_SS
    video_path: Path
    pred_path: Path
    created_at: float = field(default_factory=time_module.time)

class ChunkManager:
    """Thread-safe manager for 1-minute video chunks and prediction data"""
    def __init__(self, storage_dir: Path = CHUNK_DIR):
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(exist_ok=True)
        self._chunks: List[ChunkInfo] = []
        self._lock = Lock()
        self._loop_mode = False
        self._load_existing_chunks()

    def _load_existing_chunks(self):
        """Scan directory for existing chunks and populate queue"""
        import re
        pattern = re.compile(r"chunk_(\d{2}_\d{2}_\d{4}_\d{2}_\d{2}_\d{2})\.mp4")
        found = []
        for p in self.storage_dir.glob("chunk_*.mp4"):
            match = pattern.search(p.name)
            if match:
                ts = match.group(1)
                pred_p = p.with_name(f"chunk_{ts}_preds.npy")
                if pred_p.exists():
                    found.append(ChunkInfo(ts, p, pred_p, p.stat().st_mtime))
        
        def parse_ts(ts):
            try: return datetime.strptime(ts, "%d_%m_%Y_%H_%M_%S")
            except: return datetime.min
        
        found.sort(key=lambda x: parse_ts(x.timestamp))
        self._chunks = found[-MAX_CHUNKS:]
        
        for p in self.storage_dir.glob("chunk_*"):
            ts_match = pattern.search(p.name) or re.search(r"chunk_(\d{2}_\d{2}_\d{4}_\d{2}_\d{2}_\d{2})_preds\.npy", p.name)
            if ts_match:
                ts = ts_match.group(1)
                if not any(c.timestamp == ts for c in self._chunks):
                    try: p.unlink()
                    except: pass

    def add_chunk(self, timestamp: str, video_path: Path, pred_path: Path):
        """Add a new chunk and maintain rotation (max 5)"""
        with self._lock:
            new_chunk = ChunkInfo(timestamp, video_path, pred_path)
            self._chunks.append(new_chunk)
            if len(self._chunks) > MAX_CHUNKS:
                old = self._chunks.pop(0)
                try:
                    old.video_path.unlink()
                    old.pred_path.unlink()
                except: pass
            self._loop_mode = False

    def get_latest(self) -> Optional[ChunkInfo]:
        with self._lock: return self._chunks[-1] if self._chunks else None

    def get_all(self) -> List[ChunkInfo]:
        with self._lock: return list(self._chunks)

    def is_loop_mode(self) -> bool:
        with self._lock: return self._loop_mode

    def set_loop_mode(self, enabled: bool):
        with self._lock: self._loop_mode = enabled

    def rescan_from_disk(self):
        """Rescan chunks directory to sync with actual files on disk"""
        with self._lock:
            pattern = re.compile(r"chunk_(\d{2}_\d{2}_\d{4}_\d{2}_\d{2}_\d{2})\.mp4")
            found = {}
            for p in self.storage_dir.glob("chunk_*.mp4"):
                m = pattern.search(p.name)
                if m:
                    ts = m.group(1)
                    pred_p = p.with_name(p.stem + "_preds.npy")
                    if pred_p.exists():
                        found[ts] = ChunkInfo(ts, p, pred_p)
            
            # Update chunks list to only include actually existing files
            self._chunks = [found[c.timestamp] for c in self._chunks if c.timestamp in found]
            # Add any new chunks not in our list
            for ts, chunk in sorted(found.items()):
                if not any(c.timestamp == ts for c in self._chunks):
                    self._chunks.append(chunk)
            # Sort by timestamp
            self._chunks.sort(key=lambda c: c.timestamp)
            # Keep only MAX_CHUNKS
            while len(self._chunks) > MAX_CHUNKS:
                self._chunks.pop(0)

chunk_manager = ChunkManager()

# ==============================
# BROADCAST STATE
# ==============================
class GlobalBroadcastState:
    """Manages the current global playback time and active chunk"""
    def __init__(self):
        self.current_chunk_ts: Optional[str] = None
        self.chunk_start_time: float = 0
        self.fps: float = 15.0  # Assumed FPS for broadcast
        self._lock = None # asyncio.Lock - init in ensure_init
        self.frame_count = 0
        self._cv = None # asyncio.Condition - init in ensure_init

    def _ensure_init(self):
        """Lazy initialization of async primitives to ensure they use the correct event loop"""
        if self._lock is None:
            self._lock = asyncio.Lock()
        if self._cv is None:
            self._cv = asyncio.Condition(self._lock)

    async def get_current_frame_idx(self) -> int:
        """Calculate current frame index based on global clock sync"""
        self._ensure_init()
        async with self._lock:
            if self.chunk_start_time == 0: return 0
            now = time_module.time()
            return int((now - self.chunk_start_time) * self.fps)

    async def notify_frame(self):
        self._ensure_init()
        async with self._cv:
            self.frame_count += 1
            self._cv.notify_all()

    async def wait_for_frame(self, last_count: int, timeout: float = 0.5) -> int:
        self._ensure_init()
        try:
            async with self._cv:
                await asyncio.wait_for(
                    self._cv.wait_for(lambda: self.frame_count > last_count),
                    timeout=timeout
                )
                return self.frame_count
        except asyncio.TimeoutError:
            return last_count

    async def update(self):
        """Update global state - called periodically"""
        self._ensure_init()
        async with self._lock:
            now = time_module.time()
            if self.current_chunk_ts is None or (now - self.chunk_start_time) >= 60:
                latest = chunk_manager.get_latest()
                if latest and (self.current_chunk_ts is None or latest.timestamp > self.current_chunk_ts):
                    self.current_chunk_ts = latest.timestamp
                    self.chunk_start_time = now
                    print(f"🔄 Broadcast transitioned to new chunk: {self.current_chunk_ts}")
                else:
                    # No new chunk found - enter/remain in loop mode
                    chunk_manager.set_loop_mode(True)
                    all_chunks = chunk_manager.get_all()
                    if all_chunks:
                        loop_pool = all_chunks[-3:]
                        try:
                            curr_idx = next(i for i, c in enumerate(loop_pool) if c.timestamp == self.current_chunk_ts)
                            next_idx = (curr_idx + 1) % len(loop_pool)
                        except StopIteration:
                            next_idx = 0
                        next_chunk = loop_pool[next_idx]
                        self.current_chunk_ts = next_chunk.timestamp
                        self.chunk_start_time = now
                        print(f"🔁 Looping to chunk: {self.current_chunk_ts}")

broadcast_state = GlobalBroadcastState()

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
# PARALLEL PROCESSOR (Broadcaster Only)
# ==============================
class ParallelProcessor:
    """Manages broadcasting from pre-generated chunks (segmentation is a separate process)"""
    def __init__(self):
        self._running = False
        self._tasks: List[asyncio.Task] = []
        self._broadcast_buffer: List[str] = []  # In-memory frames (base64 JPEG) for current chunk
        self._broadcast_preds: Optional[np.ndarray] = None
        self._buffer_lock = Lock()
        self._last_seen_chunk: Optional[str] = None

    async def start(self):
        if self._running: return
        self._running = True
        self._tasks = [
            asyncio.create_task(self._broadcaster_loop()),
            asyncio.create_task(self._chunk_watcher_loop())
        ]
        print("🎬 Broadcaster started (watching chunks directory)")

    async def stop(self):
        self._running = False
        for t in self._tasks:
            t.cancel()
        print("🛑 Broadcaster stopped")

    async def _chunk_watcher_loop(self):
        """Watch for new chunks in the directory"""
        while self._running:
            try:
                # Rescan directory to sync with actual files
                chunk_manager.rescan_from_disk()
                
                # Check for new chunks
                all_chunks = chunk_manager.get_all()
                if all_chunks:
                    latest = all_chunks[-1]
                    if latest.timestamp != self._last_seen_chunk:
                        print(f"🔄 New chunk detected: {latest.timestamp}")
                        self._last_seen_chunk = latest.timestamp
                        broadcast_state.current_chunk_ts = latest.timestamp
                        broadcast_state.chunk_start_time = time_module.time()
                        await self._load_chunk_to_buffer(latest.timestamp)
                await asyncio.sleep(2.0)  # Check every 2 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Chunk watcher error: {e}")
                await asyncio.sleep(5.0)

    async def _broadcaster_loop(self):
        """Task B: Syncs global playback and loads current chunk into memory at 15fps ticker"""
        prev_chunk_ts = None
        while self._running:
            try:
                # 1. Update global state (decide which chunk to play)
                await broadcast_state.update()
                
                # 2. If chunk changed, load it into memory
                curr_ts = broadcast_state.current_chunk_ts
                if curr_ts and curr_ts != prev_chunk_ts:
                    await self._load_chunk_to_buffer(curr_ts)
                    prev_chunk_ts = curr_ts
                
                # 3. Tight loop for roughly 1 second of 15fps ticking
                for _ in range(15):
                    if not self._running: break
                    await broadcast_state.notify_frame()
                    await asyncio.sleep(1/15.0)
            except Exception as e:
                print(f"Broadcaster error: {e}")
                await asyncio.sleep(1.0)

    async def _load_chunk_to_buffer(self, timestamp: str):
        """Load video frames and prediction maps into RAM for high-efficiency broadcast"""
        all_chunks = chunk_manager.get_all()
        chunk = next((c for c in all_chunks if c.timestamp == timestamp), None)
        if not chunk: return

        print(f"📥 Loading chunk {timestamp} to RAM...")
        
        # Load predictions
        try:
            preds = np.load(chunk.pred_path)
        except Exception as e:
            print(f"Error loading predictions for {timestamp}: {e}")
            return

        # Load video frames
        cap = cv2.VideoCapture(str(chunk.video_path))
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok: break
            
            # Encode as JPEG and base64
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            image_b64 = base64.b64encode(buffer).decode('utf-8')
            frames.append(image_b64)
            if len(frames) % 50 == 0: await asyncio.sleep(0) # Yield
        
        cap.release()
        
        with self._buffer_lock:
            self._broadcast_buffer = frames
            self._broadcast_preds = preds
        
        print(f"⚡ RAM buffer ready: {len(frames)} frames for chunk {timestamp}")

    async def get_current_frame_data(self) -> Optional[Dict]:
        """Get the current frame data from RAM buffer for shared broadcast"""
        with self._buffer_lock:
            if not self._broadcast_buffer: 
                print("DEBUG: Broadcast buffer empty")
                return None
            
            # Use global sync index
            idx = await broadcast_state.get_current_frame_idx()
            if idx >= len(self._broadcast_buffer):
                idx = len(self._broadcast_buffer) - 1
            if idx < 0: idx = 0
            
            image_b64 = self._broadcast_buffer[idx]
            
            # Handle missing predictions gracefully
            pred_b64 = ""
            pred_shape = [0, 0]
            visible_stats = []
            
            if self._broadcast_preds is not None and idx < len(self._broadcast_preds):
                pred_small = self._broadcast_preds[idx]
                
                # Update global stats for dashboards
                stream_state.update_frame_stats(pred_small)
                
                # Get visible labels stats for WebSocket
                for label, count in stream_state.frame_stats.items():
                    if label in stream_state.active_labels:
                        visible_stats.append({
                            "label": label,
                            "count": count,
                            "color": color_map.get(label, "#FFF")
                        })
                visible_stats.sort(key=lambda x: x["count"], reverse=True)
                
                # Hover map encoding
                pred_b64 = base64.b64encode(pred_small.tobytes()).decode('utf-8')
                pred_shape = [pred_small.shape[0], pred_small.shape[1]]
            
            return {
                "image": image_b64,
                "pred_map": pred_b64,
                "pred_shape": pred_shape,
                "pred_scale": 4,
                "fps": 15.0,
                "avg_fps": 15.0,
                "total_frames": idx,
                "visible_labels": visible_stats[:8]
            }

parallel_processor = ParallelProcessor()

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
    """Start the parallel processor on app startup"""
    await parallel_processor.start()

@app.on_event("shutdown")
async def shutdown_event():
    """Stop the parallel processor on app shutdown"""
    await parallel_processor.stop()

# ==============================
# WEBSOCKET STREAMING (Cache-based)
# ==============================
@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time video streaming.
    
    Clients receive frames from the shared RAM buffer managed by 
    the broadcaster. All clients are synchronized to the global clock.
    """
    await websocket.accept()
    print(f"✅ synchronized WebSocket client connected")
    
    # Track the last frame index sent to this client to avoid duplicates
    last_sent_idx = -1
    last_chunk_ts = None
    last_count = 0
    
    try:
        while True:
            # Handle incoming commands (non-blocking)
            try:
                data = await asyncio.wait_for(websocket.receive_json(), timeout=0.001)
                if data.get("type") == "toggle_label":
                    label = data.get("label")
                    if label: stream_state.toggle_label(label)
            except asyncio.TimeoutError:
                pass
            
            # Wait for global frame signal
            last_count = await broadcast_state.wait_for_frame(last_count, timeout=0.1)
            
            # Get data from RAM buffer
            curr_idx = await broadcast_state.get_current_frame_idx()
            curr_ts = broadcast_state.current_chunk_ts
            
            if curr_ts != last_chunk_ts or curr_idx != last_sent_idx:
                fdata = await parallel_processor.get_current_frame_data()
                if fdata:
                    if curr_idx % 15 == 0: print(f"DEBUG: Sending frame {curr_idx} for chunk {curr_ts}")
                    await websocket.send_json({
                        "type": "frame",
                        "image": fdata["image"],
                        "fps": fdata["fps"],
                        "avg_fps": fdata["avg_fps"],
                        "total_frames": fdata["total_frames"],
                        "visible_labels": fdata["visible_labels"],
                        "pred_map": fdata["pred_map"],
                        "pred_shape": fdata["pred_shape"],
                        "pred_scale": fdata["pred_scale"]
                    })
                    last_sent_idx = curr_idx
                    last_chunk_ts = curr_ts
            
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
