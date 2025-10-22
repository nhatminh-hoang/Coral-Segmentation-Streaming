import os
os.environ["HF_HUB_OFFLINE"] = "1"

from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import gradio as gr
from pathlib import Path
import time as time_module

from inference import CoralSegModel, id2label, label2color, label2vietnamese

# ==============================
# CONFIG & MODEL
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
# HELPERS
# ==============================
CORAL_KEYWORDS = {
    "coral",           # catches 'other coral alive/dead/bleached'
    "branching",
    "massive", "meandering",
    "acropora", "table acropora",
    "pocillopora",
    "stylophora",
    "millepora",       # hydrocoral; included in Coralscapes taxonomy
}

def _is_coral_label(en_label: str) -> bool:
    s = en_label.lower()
    return any(k in s for k in CORAL_KEYWORDS)

def coral_label_choices(language: str = "English") -> list[str]:
    """Return only coral-related labels in the requested UI language."""
    en_coral = [lbl for lbl in id2label.values() if _is_coral_label(lbl)]
    if language == "Tiếng Việt":
        return [label2vietnamese.get(lbl, lbl) for lbl in en_coral]
    return en_coral

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
    """Map English label to selected language (Vietnamese if available)."""
    if language == "Tiếng Việt" and label_en in label2vietnamese:
        return label2vietnamese[label_en]
    return label_en

def _load_font(size=40):
    try:
        return ImageFont.truetype("font/AndikaNewBasic-B.ttf", size=size)
    except Exception:
        return ImageFont.load_default()

def build_annotations(pred_map: np.ndarray, selected: list[str], language="English"):
    """
    Return [(mask,label), ...] where mask is float32 0..1 HxW for gr.AnnotatedImage.
    Works with both English and Vietnamese label names in `selected`.
    """
    if pred_map is None or not selected:
        return []

    # English/Vietnamese -> class_id
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
            # recover English name for consistent display mapping
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

        # put label at centroid of the largest contour (if present)
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
                draw.rectangle(box, fill=255)  # white backing box
                draw.text((cX - tw // 2, cY - th // 1.5), display_label, fill=0, font=font, align="center")
                boundary_mask = (np.array(pil_img) / 255.0).astype(np.float32)

        anns.append((boundary_mask, display_label))

    return anns

def get_fps_info():
    """Get current FPS information as formatted string."""
    stats = model.get_fps_stats()
    return f"Current: {stats['current_fps']} FPS | Average: {stats['average_fps']} FPS"

# ==============================
# STREAMING: AUTO PROCESS FOLDER
# ==============================
def auto_process_folder(skip: int):
    """
    Infinite generator:
      - Rescans VIDEO_DIR each pass
      - Streams frames for every video found
      - If no videos, yields a status message and waits briefly
    Yields: overlay_rgb, pred_map, base_rgb, fps_info
    """
    try:
        while True:
            model.reset_fps_stats()
            files = list_video_files(VIDEO_DIR)

            if not files:
                # keep the stream alive while waiting for files to appear
                yield None, None, None, "No videos found in ./sample_videos"
                time_module.sleep(2.0)
                continue

            for path in files:
                cap = cv2.VideoCapture(str(path))
                if not cap.isOpened():
                    # surface the issue in the FPS box but keep going
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

            # small pause before rescanning the folder again
            time_module.sleep(1.0)

    except GeneratorExit:
        # Session ended; tidy exit
        return

# ==============================
# SNAPSHOT / TOGGLES
# ==============================
def make_snapshot(selected_labels, pred_map, base_rgb, language="English", alpha=0.25):
    if pred_map is None or base_rgb is None:
        return gr.update()
    # normalize selected label names to match the chosen display language
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
        choices=coral_label_choices(lang),
        value=coral_label_choices(lang)
    )

def toggle_options(open_state: bool):
    new_state = not bool(open_state)
    # Return updated state + make the whole group visible/hidden
    return new_state, gr.Group(visible=new_state)  # update container visibility (recommended pattern)

# ==============================
# UI
# ==============================
with gr.Blocks(title="CoralScapes Auto Segmentation") as demo:
    with gr.Row():
        with gr.Column(scale=3):
            pred_state = gr.State(None)  # last pred_map (HxW np.uint8)
            base_state = gr.State(None)  # last base_rgb (HxWx3 uint8)

            live_img   = gr.Image(label="Live segmented output", streaming=True)
            fps_box    = gr.Textbox(label="FPS Info", interactive=False)

            snap_btn   = gr.Button("📸 Snapshot (hover-able)")
            hover_img  = gr.AnnotatedImage(label="Snapshot (hover to see label)", color_map=color_map)

        with gr.Column(scale=1):
            # -- NEW: Options toggle button + hidden container
            options_open = gr.State(False)
            options_btn = gr.Button("⚙️ Tuỳ chọn", size="sm")  # "Options" in Vietnamese

            with gr.Group(visible=False) as options_panel:
                skip = gr.Slider(1, 60, value=DEFAULT_SKIP, step=1, label="Xử lý mỗi N khung hình")
                language = gr.Radio(
                    choices=["English", "Tiếng Việt"],
                    value="Tiếng Việt",  # default to Vietnamese
                    label="Ngôn ngữ hiển thị / Display Language"
                )
                # Initialize toggles directly in Vietnamese to avoid flash-of-English
                vi_choices = coral_label_choices("Tiếng Việt")
                toggles = gr.CheckboxGroup(
                    choices=vi_choices, value=vi_choices,
                    label="Bật/tắt các lớp trong ảnh chụp",
                )

            # Toggle button wiring
            options_btn.click(
                toggle_options,
                inputs=[options_open],
                outputs=[options_open, options_panel],
            )

    # Start automatically when the app loads (infinite streamer).
    demo.load(
        auto_process_folder,
        inputs=[skip],
        outputs=[live_img, pred_state, base_state, fps_box],
        queue=True,
    )

    # Keep toggles synced with language
    language.change(update_toggles_lang, inputs=[language], outputs=[toggles])

    # Snapshot wiring
    snap_btn.click(make_snapshot, inputs=[toggles, pred_state, base_state, language], outputs=[hover_img])
    toggles.change(make_snapshot, inputs=[toggles, pred_state, base_state, language], outputs=[hover_img])
    language.change(make_snapshot, inputs=[toggles, pred_state, base_state, language], outputs=[hover_img])

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=10, max_size=40)
    demo.launch(server_name="0.0.0.0", server_port=8080)
