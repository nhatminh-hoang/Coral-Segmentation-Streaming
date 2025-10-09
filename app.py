from PIL import Image
import cv2
import numpy as np
import gradio as gr

from inference import CoralSegModel, id2label, label2color, create_segmentation_overlay
model = CoralSegModel()

# ---- helpers ----
def _safe_read(cap):
    ok, frame = cap.read()
    return frame if ok and frame is not None else None

def get_fps_info():
    """Get current FPS information as formatted string"""
    stats = model.get_fps_stats()
    return f"Current: {stats['current_fps']} FPS | Average: {stats['average_fps']} FPS | Frames: {stats['total_frames']}"

def _safe_read(cap):
    ok, frame = cap.read()
    return frame if ok and frame is not None else None

def build_annotations(pred_map: np.ndarray, selected: list[str]) -> list[tuple[np.ndarray, str]]:
    """Return [(mask,label), ...] where mask is 0/1 float HxW for AnnotatedImage."""
    if pred_map is None or not selected:
        return []
    
    # Create reverse mapping: label_name -> class_id
    label2id = {label: int(id_str) for id_str, label in id2label.items()}
    
    anns = []
    for label_name in selected:
        if label_name not in label2id:
            continue  # Skip unknown labels
        
        class_id = label2id[label_name]  # Convert label name to class ID
        mask = (pred_map == class_id).astype(np.uint8)
        if mask.mean() > 0:
            # Find contours and create boundary mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            boundary_mask = np.zeros_like(mask, dtype=np.float32)
            cv2.drawContours(boundary_mask, contours, -1, 1.0, thickness=10)  # thickness=2 for visibility
            anns.append((boundary_mask, label_name))  # Use the label name for display
            
    return anns
    
# ==============================
# STREAMING EVENT FUNCTIONS
# ==============================
# IMPORTANT: make the event functions themselves generators.
# Also: include the States as outputs so we can update them every frame.
def remote_start(url: str, n: int, pred_state, base_state):
    if not url:
        return
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        return
    idx = 0
    try:
        while True:
            frame = _safe_read(cap)
            if frame is None:
                break
            if n > 1 and (idx % n) != 0:
                idx += 1
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            print(frame.shape)
            pred_map, overlay_rgb, base_rgb = model.predict_map_and_overlay(frame)
            fps_info = get_fps_info()
            # yield live image + updated States' *values*
            yield overlay_rgb, pred_map, base_rgb, fps_info
            idx += 1
    finally:
        cap.release()

def upload_start(video_file: str, n: int):
    if not video_file:
        return
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        return
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if n > 1 and (idx % n) != 0:
                idx += 1
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            print(frame.shape)
            pred_map, overlay_rgb, base_rgb = model.predict_map_and_overlay(frame)
            fps_info = get_fps_info()
            
            yield overlay_rgb, pred_map, base_rgb, fps_info
            idx += 1
    finally:
        cap.release()

# ==============================
# SNAPSHOT / TOGGLES (non-streaming)
# ==============================
# NOTE: When you pass gr.State as an input, you receive the *value*, not the wrapper.
def make_snapshot(selected_labels, pred_map, base_rgb, alpha=0.25):
    if pred_map is None or base_rgb is None:
        return gr.update()
    # rebuild overlay to match the live look
    # overlay = create_segmentation_overlay(pred_map, id2label, label2color, Image.fromarray(base_rgb), alpha=alpha)
    overlay = Image.fromarray(base_rgb)
    ann = build_annotations(pred_map, selected_labels or [])
    return (overlay, ann)  # (base_image, [(mask,label), ...])

# ==============================
# UI
# ==============================
with gr.Blocks(title="CoralScapes Streaming Segmentation") as demo:
    gr.Markdown("# CoralScapes Streaming Segmentation")
    gr.Markdown(
        "Left: **live stream** (fast). Right: **snapshot** with **hover labels** and **per-class toggles**."
    )

    with gr.Tab("Remote Stream (RTSP/HTTP)"):
        with gr.Row():
            with gr.Column(scale=2):
                
                # States start as None. We'll UPDATE them on every frame by returning them as outputs.
                pred_state_remote = gr.State(None)  # holds last pred_map (HxW np.uint8)
                base_state_remote = gr.State(None)  # holds last base_rgb (HxWx3 uint8)

                live_remote = gr.Image(label="Live segmented stream", streaming=True)
                fps_display_remote = gr.Textbox(label="FPS Info", interactive=False)

                start_btn = gr.Button("Start")

                snap_btn_remote = gr.Button("📸 Snapshot (hover-able)")
                hover_remote = gr.AnnotatedImage(label="Snapshot (hover to see label)")
                

            with gr.Column(scale=1):
                url  = gr.Textbox(label="Stream URL", placeholder="rtsp://user:pass@ip:port/…")
                skip = gr.Slider(1, 60, value=10, step=1, label="Process every Nth frame")

                toggles_remote = gr.CheckboxGroup(
                    choices=list(id2label.values()), value=list(id2label.values()),
                    label="Toggle classes in snapshot",
                )

            start_btn.click(
                remote_start,
                inputs=[url, skip, pred_state_remote, base_state_remote],
                outputs=[live_remote, pred_state_remote, base_state_remote, fps_display_remote],
                queue=True,
            )
            
            snap_btn_remote.click(
                make_snapshot,
                inputs=[toggles_remote, pred_state_remote, base_state_remote],
                outputs=[hover_remote],
            )
            toggles_remote.change(
                make_snapshot,
                inputs=[toggles_remote, pred_state_remote, base_state_remote],
                outputs=[hover_remote],
            )

    with gr.Tab("Upload Video"):
        with gr.Row():
            # Left column (now contains toggles, snapshot button, and live output)
            with gr.Column(scale=3):
                # States remain in the same column as live_upload
                pred_state_upload = gr.State(None)
                base_state_upload = gr.State(None)
                
                live_upload = gr.Image(label="Live segmented output", streaming=True)
                fps_display_upload = gr.Textbox(label="FPS Info", interactive=False)

                start_btn2 = gr.Button("Process")
                snap_btn_upload = gr.Button("📸 Snapshot (hover-able)")
                hover_upload = gr.AnnotatedImage(label="Snapshot (hover to see label)")
                
            # Right column (now contains video input and slider)
            with gr.Column(scale=1):
                vid_in = gr.Video(sources=["upload"], label="Input Video")
                skip2 = gr.Slider(1, 60, value=10, step=1, label="Process every Nth frame")

                toggles_upload = gr.CheckboxGroup(
                    choices=list(id2label.values()), value=list(id2label.values()),
                    label="Toggle classes in snapshot",
                )
                
        # Event handlers remain the same
        start_btn2.click(
            upload_start,
            inputs=[vid_in, skip2],
            outputs=[live_upload, pred_state_upload, base_state_upload, fps_display_upload],
            queue=True,
        )

        snap_btn_upload.click(
            make_snapshot,
            inputs=[toggles_upload, pred_state_upload, base_state_upload],
            outputs=[hover_upload],
        )
        
        toggles_upload.change(
            make_snapshot,
            inputs=[toggles_upload, pred_state_upload, base_state_upload],
            outputs=[hover_upload],
        )

if __name__ == "__main__":
    demo.launch(share=True)
