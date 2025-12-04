# app.py
import os
from io import BytesIO
from pathlib import Path
import time as time_module
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import gradio as gr
from fastapi import FastAPI, File, UploadFile, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse, RedirectResponse

# ===== your existing imports / model wrappers =====
from inference import CoralSegModel, id2label, label2color, label2vietnamese

# ==============================
# CONFIG & MODEL (unchanged)
# ==============================
VIDEO_DIR = Path("sample_videos")  # auto-scan this folder
ALLOWED_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm")
DEFAULT_SKIP = 3  # process every Nth frame by default

model = CoralSegModel(use_onnx=True)  # your local model wrapper

# Build color map for AnnotatedImage:
# map BOTH English and Vietnamese labels to the same RGB color
def rgb_to_hex(rgb):
    r, g, b = [int(x) for x in rgb]
    return f"#{r:02X}{g:02X}{b:02X}"

color_map = {}
for id_str, en_label in id2label.items():
    rgb = label2color.get(en_label, [255, 255, 255])  # [R,G,B]
    hex_color = rgb_to_hex(rgb)
    color_map[en_label] = hex_color
    if en_label in label2vietnamese:
        color_map[label2vietnamese[en_label]] = hex_color

# ==============================
# HELPERS (unchanged)
# ==============================
CORAL_KEYWORDS = {
    "coral",
    "branching",
    "massive", "meandering",
    "acropora", "table acropora",
    "pocillopora",
    "stylophora",
    "millepora",
}

def _is_coral_label(en_label: str) -> bool:
    s = en_label.lower()
    return any(k in s for k in CORAL_KEYWORDS)

def _all_label_choices(language: str) -> list[str]:
    if language == "Tiếng Việt":
        return [label2vietnamese.get(lbl, lbl) for lbl in id2label.values()]
    return list(id2label.values())

def _coral_default_values(language: str) -> list[str]:
    coral_en = [lbl for lbl in id2label.values() if _is_coral_label(lbl)]
    if language == "Tiếng Việt":
        return [label2vietnamese.get(lbl, lbl) for lbl in coral_en]
    return coral_en

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

def get_label(label_en: str, language="English"):
    if language == "Tiếng Việt" and label_en in label2vietnamese:
        return label2vietnamese[label_en]
    return label_en

def _load_font(size=40):
    try:
        return ImageFont.truetype("font/AndikaNewBasic-B.ttf", size=size)
    except Exception:
        return ImageFont.load_default()

def build_annotations(pred_map: np.ndarray, selected: list[str], language="English"):
    if pred_map is None or not selected:
        return []

    label2id_en = {label: int(id_str) for id_str, label in id2label.items()}
    vietnamese2id = {
        label2vietnamese[en]: int(id_str)
        for id_str, en in id2label.items()
        if en in label2vietnamese
    }

    font = _load_font(40)
    anns = []

    for label_name in selected:
        class_id = None
        original_en = None

        if label_name in label2id_en:
            class_id = label2id_en[label_name]
            original_en = label_name
        elif label_name in vietnamese2id:
            class_id = vietnamese2id[label_name]
            for en_label, vn_label in label2vietnamese.items():
                if vn_label == label_name:
                    original_en = en_label
                    break

        if class_id is None:
            continue

        mask = (pred_map == class_id).astype(np.uint8)
        if mask.mean() == 0:
            continue

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boundary_mask = np.zeros_like(mask, dtype=np.float32)
        cv2.drawContours(boundary_mask, contours, -1, 0.5, thickness=10)

        display_label = get_label(original_en, language)

        if contours:
            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                pil_img = Image.fromarray((boundary_mask * 255).astype(np.uint8))
                draw = ImageDraw.Draw(pil_img)

                bbox = draw.textbbox((0, 0), display_label, font=font, align="center")
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                box = [cX - tw // 2 - 5, cY - th // 2 - 5, cX + tw // 2 + 5, cY + th // 2 + 5]
                draw.rectangle(box, fill=255)
                draw.text((cX - tw // 2, cY - th // 1.5), display_label, fill=0, font=font, align="center")
                boundary_mask = (np.array(pil_img) / 255.0).astype(np.float32)

        anns.append((boundary_mask, display_label))

    return anns

def get_fps_info():
    stats = model.get_fps_stats()
    return f"Current: {stats['current_fps']} FPS | Average: {stats['average_fps']} FPS"

# ==============================
# STREAMING: AUTO PROCESS FOLDER (unchanged)
# ==============================
def auto_process_folder(skip: int):
    try:
        while True:
            model.reset_fps_stats()
            files = list_video_files(VIDEO_DIR)

            if not files:
                yield None, None, None, "No videos found in ./sample_videos"
                time_module.sleep(2.0)
                continue

            for path in files:
                cap = cv2.VideoCapture(str(path))
                if not cap.isOpened():
                    yield None, None, None, f"Cannot open: {path.name}"
                    continue

                idx = 0
                try:
                    while True:
                        frame = _safe_read(cap)
                        if frame is None:
                            break
                        if skip > 1 and (idx % skip) != 0:
                            idx += 1
                            continue

                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pred_map, overlay_rgb, base_rgb = model.predict_map_and_overlay(frame)
                        fps_info = get_fps_info()
                        yield overlay_rgb, pred_map, base_rgb, f"{fps_info}"
                        idx += 1
                finally:
                    cap.release()

            time_module.sleep(1.0)

    except GeneratorExit:
        return

# ==============================
# SNAPSHOT / TOGGLES (unchanged)
# ==============================
def make_snapshot(selected_labels, pred_map, base_rgb, language="English", alpha=0.25):
    if pred_map is None or base_rgb is None:
        return gr.update()
    if language == "Tiếng Việt":
        selected_labels = [label2vietnamese.get(label, label) for label in selected_labels]
    else:
        vn2en = {v: k for k, v in label2vietnamese.items()}
        selected_labels = [vn2en.get(label, label) for label in selected_labels]

    overlay = Image.fromarray(base_rgb)
    ann = build_annotations(pred_map, selected_labels or [], language)
    return (overlay, ann)

def update_toggles_lang(lang):
    return gr.CheckboxGroup(
        choices=_all_label_choices(lang),
        value=_coral_default_values(lang)
    )

def toggle_options(open_state: bool):
    new_state = not bool(open_state)
    return new_state, gr.Group(visible=new_state)

# ==============================
# UI (unchanged)
# ==============================
with gr.Blocks(title="CoralScapes Auto Segmentation") as demo:
    with gr.Row():
        with gr.Column(scale=3):
            pred_state = gr.State(None)
            base_state = gr.State(None)

            live_img   = gr.Image(label="Live segmented output", streaming=True)
            fps_box    = gr.Textbox(label="FPS Info", interactive=False)

            snap_btn   = gr.Button("📸 Snapshot (hover-able)")
            hover_img  = gr.AnnotatedImage(label="Snapshot (hover to see label)", color_map=color_map)

        with gr.Column(scale=1):
            options_open = gr.State(False)
            options_btn = gr.Button("⚙️ Tuỳ chọn", size="sm")

            with gr.Group(visible=False) as options_panel:
                skip = gr.Slider(1, 60, value=DEFAULT_SKIP, step=1, label="Xử lý mỗi N khung hình")
                language = gr.Radio(
                    choices=["English", "Tiếng Việt"],
                    value="Tiếng Việt",
                    label="Ngôn ngữ hiển thị / Display Language"
                )
                vi_choices_all = _all_label_choices("Tiếng Việt")
                vi_coral_default = _coral_default_values("Tiếng Việt")
                toggles = gr.CheckboxGroup(
                    choices=vi_choices_all, value=vi_coral_default,
                    label="Bật/tắt các lớp trong ảnh chụp",
                )

            options_btn.click(
                toggle_options,
                inputs=[options_open],
                outputs=[options_open, options_panel],
            )

    demo.load(
        auto_process_folder,
        inputs=[skip],
        outputs=[live_img, pred_state, base_state, fps_box],
        queue=True,
    )

    language.change(update_toggles_lang, inputs=[language], outputs=[toggles])
    snap_btn.click(make_snapshot, inputs=[toggles, pred_state, base_state, language], outputs=[hover_img])
    toggles.change(make_snapshot, inputs=[toggles, pred_state, base_state, language], outputs=[hover_img])
    language.change(make_snapshot, inputs=[toggles, pred_state, base_state, language], outputs=[hover_img])

# Enable Gradio queue before mounting
demo.queue(default_concurrency_limit=10, max_size=40)

# ==============================
# FASTAPI APP (new)
# ==============================
app = FastAPI(title="CoralScapes Segmentation Service")

# CORS so others on LAN can access (e.g., http://10.20.8.10:7860)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # narrow if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.get("/")
def root():
    # Redirect to the mounted Gradio UI
    return RedirectResponse(url="/app")

# --- Simple inference API: POST an image, get PNG overlay back ---
@app.post("/api/process_image")
async def process_image(file: UploadFile = File(...)):
    data = await file.read()
    img = Image.open(BytesIO(data)).convert("RGB")
    frame = np.array(img)
    pred_map, overlay_rgb, base_rgb = model.predict_map_and_overlay(frame)

    # Return overlay RGB as PNG bytes
    out = BytesIO()
    Image.fromarray(overlay_rgb).save(out, format="PNG")
    out.seek(0)
    return StreamingResponse(out, media_type="image/png")

# (Optional) list available local videos
@app.get("/api/videos")
def list_videos():
    files = [p.name for p in list_video_files(VIDEO_DIR)]
    return {"videos": files}

# ==============================
# MOUNT GRADIO UNDER /app
# ==============================
# Official pattern: mount Blocks onto FastAPI
# Docs: https://www.gradio.app/docs/gradio/mount_gradio_app
app = gr.mount_gradio_app(app, demo, path="/app")

# If running as a script: `uvicorn app:app --host 0.0.0.0 --port 7860`
# (Do not call demo.launch(); FastAPI + uvicorn serve the whole app.)
