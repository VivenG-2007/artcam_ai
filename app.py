"""
AI Filter Platform – main application.

Features
--------
- Generate filter code via Groq + DeepSeek pipeline (keys from .env).
- Apply filter to a static image (explicit button, no passive processing).
- Live webcam stream processed frame-by-frame inside Gradio (no separate window).
- Video file upload → filter applied to every frame → playable output video.
- Post / Share panel revealed after a successful image apply.
"""

from __future__ import annotations

import base64
import os
import tempfile

import cv2
import gradio as gr
import numpy as np

import config  # loads .env
from ai_generator import AIFilterGenerator
from compiler import FilterCompiler
from database import FilterDatabase
from share_service import ShareService
from validator import FilterValidator

# ── Singletons ────────────────────────────────────────────────────────────────
validator = FilterValidator()
compiler  = FilterCompiler()
generator = AIFilterGenerator(validator=validator, compiler=compiler)
db        = FilterDatabase()
share_svc = ShareService()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


# ── Core helpers ──────────────────────────────────────────────────────────────

# State for smoothing the bounding box tracker
_tracker_state = {
    "bounds": None,
    "frame_count": 0,
}

def get_face_bounds(frame_bgr: np.ndarray) -> tuple | None:
    """Fast, smoothed body tracking using face detection."""
    global _tracker_state
    
    _tracker_state["frame_count"] += 1
    
    # To reduce lag, only run heavy Haar detection every 3rd frame
    # and run it on a tiny downscaled frame (e.g., width 320)
    h, w = frame_bgr.shape[:2]
    run_detection = (_tracker_state["frame_count"] % 3 == 1) or (_tracker_state["bounds"] is None)
    
    if run_detection:
        scale = 320.0 / w
        small = cv2.resize(frame_bgr, (0, 0), fx=scale, fy=scale)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4)
        
        if len(faces) > 0:
            # Scale coords back up to original frame size
            fx, fy, fw, fh = [int(v / scale) for v in faces[0]]
            
            # Expand to full body
            body_w = int(fw * 2.8)
            body_h = int(fh * 3.5)
            body_x = int(fx - (body_w - fw) / 2)
            body_y = int(fy - fh * 0.2)
            
            # Clamp to frame
            body_x = max(0, min(w - 10, body_x))
            body_y = max(0, min(h - 10, body_y))
            body_w = min(w - body_x, body_w)
            body_h = min(h - body_y, body_h)
            
            target_bounds = (body_x, body_y, body_w, body_h)
            
            # EMA Smoothing (Alpha = 0.4 for smooth, snappy tracking)
            if _tracker_state["bounds"] is None:
                _tracker_state["bounds"] = target_bounds
            else:
                curr = _tracker_state["bounds"]
                _tracker_state["bounds"] = tuple(
                    int(curr[i] * 0.6 + target_bounds[i] * 0.4) for i in range(4)
                )
    
    return _tracker_state["bounds"]

def validate_and_compile(code: str):
    """Validate → compile → smoke test. Returns (fn, message)."""
    code = code or ""
    if not code.strip():
        return None, "No filter code found. Generate or paste code first."
    ok, msg = validator.validate(code)
    if not ok:
        return None, f"Validation failed: {msg}"
    fn, err = compiler.compile_and_smoke_test(code)
    if fn is None:
        return None, f"Compile error: {err}"
    return fn, "ok"


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2RGB)


def rgb_to_bgr(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR)


# ── Business logic ────────────────────────────────────────────────────────────

def generate_filter(prompt: str):
    """Run the full AI pipeline. Keys loaded from env."""
    groq_key       = os.environ.get("GROQ_API_KEY", "")
    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")

    result = generator.generate(
        prompt=prompt,
        groq_api_key=groq_key,
        openrouter_api_key=openrouter_key,
    )
    try:
        db.save_generation_session(result.to_dict())
    except Exception:
        pass

    lines  = [f"{a.stage}: {'PASS' if a.success else 'FAIL'}" for a in result.attempts]
    status = result.message
    if lines:
        status += "\n" + "\n".join(lines)
    return result.code, status


def apply_to_image(code: str, image: np.ndarray):
    """Apply filter to a static uploaded image (numpy RGB → numpy RGB)."""
    if image is None:
        return None, "⚠️  Upload an image first.", gr.update(visible=False)

    fn, msg = validate_and_compile(code)
    if fn is None:
        return None, msg, gr.update(visible=False)

    try:
        frame_bgr = rgb_to_bgr(image)
        bounds = get_face_bounds(frame_bgr)
        try:
            result_bgr = fn(frame_bgr, landmarks=bounds)
        except TypeError:
            result_bgr = fn(frame_bgr)  # Fallback if AI forgot to add landmarks argument
            
        result_rgb = bgr_to_rgb(result_bgr)
        return result_rgb, "✅ Filter applied!", gr.update(visible=True)
    except Exception as exc:
        return None, f"❌ Execution error: {exc}", gr.update(visible=False)


def process_webcam_frame(frame: np.ndarray, code: str):
    """Called on every webcam frame by Gradio streaming.
    frame is numpy RGB (from browser); returns (numpy RGB, status_str)."""
    if frame is None:
        return None, "Waiting for camera…"

    fn, msg = validate_and_compile(code)
    if fn is None:
        # Draw "NO FILTER" overlay on the raw frame so user sees feed still works
        out = frame.copy()
        cv2.putText(out, "NO FILTER LOADED", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 255), 2)
        return out, f"⚠️ {msg}"
    try:
        frame_bgr = rgb_to_bgr(frame)
        bounds = get_face_bounds(frame_bgr)
        try:
            result_bgr = fn(frame_bgr, landmarks=bounds)
        except TypeError:
            result_bgr = fn(frame_bgr)
            
        result_rgb = bgr_to_rgb(result_bgr)
        # Draw a small "FILTER ON" badge so it's obvious the filter is running
        cv2.putText(result_rgb, "FILTER ACTIVE", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 120), 2)
        return result_rgb, "✅ Filter running live"
    except Exception as exc:
        return frame, f"❌ Frame error: {exc}"


def process_video(code: str, video_path: str):
    """Apply filter to every frame of an uploaded video. Returns output path."""
    if not video_path:
        return None, "⚠️  Upload a video first."

    fn, msg = validate_and_compile(code)
    if fn is None:
        return None, msg

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "❌ Could not open video file."

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = tempfile.mktemp(suffix=".mp4")
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    writer   = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    processed = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            try:
                bounds = get_face_bounds(frame)
                try:
                    out_frame = fn(frame, landmarks=bounds)
                except TypeError:
                    out_frame = fn(frame)
                writer.write(out_frame.astype(np.uint8))
            except Exception:
                writer.write(frame)   # write original on per-frame error
            processed += 1
    finally:
        cap.release()
        writer.release()

    return out_path, f"✅ Done – {processed}/{total} frames processed."


def post_filter(filter_name: str, code: str, image_output):
    if not (filter_name or "").strip():
        return "⚠️  Enter a name for this post."
    if not (code or "").strip():
        return "⚠️  No filter code to share."

    preview_b64 = ""
    if image_output is not None:
        try:
            frame_bgr = rgb_to_bgr(np.array(image_output, dtype=np.uint8))
            _, buf    = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
            preview_b64 = base64.b64encode(buf).decode()
        except Exception:
            pass

    _, msg = share_svc.share(filter_name.strip(), code, preview_b64)
    return msg


def save_filter(name: str, code: str):
    if not (name or "").strip():
        return "Enter a filter name.", refresh_filter_list()
    if not (code or "").strip():
        return "No code to save.", refresh_filter_list()
    ok, msg = validator.validate(code)
    if not ok:
        return f"Cannot save invalid code: {msg}", refresh_filter_list()
    try:
        db.save_filter(name.strip(), code)
    except Exception as exc:
        return f"Could not save: {exc}", refresh_filter_list()
    return f"✅ Filter '{name}' saved.", refresh_filter_list()


def delete_filter(name: str):
    if not name:
        return "Select a filter to delete.", refresh_filter_list()
    try:
        db.delete_filter(name)
    except Exception as exc:
        return f"Could not delete: {exc}", refresh_filter_list()
    return f"🗑️  Filter '{name}' deleted.", refresh_filter_list()


def load_filter(name: str):
    if not name:
        return "", "Select a filter."
    try:
        code = db.get_filter_code(name)
    except Exception as exc:
        return "", f"Could not load: {exc}"
    if code is None:
        return "", f"Filter '{name}' not found."
    return code, f"✅ Loaded '{name}'."


def refresh_filter_list():
    try:
        choices = db.list_filters()
    except Exception:
        choices = []
    return gr.Dropdown(choices=choices, label="Saved Filters", interactive=True)


# ── Styling ───────────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

:root {
    --bg:      #0a0a0f;
    --surface: #12121a;
    --border:  #2a2a3a;
    --accent:  #7c3aed;
    --accent2: #06b6d4;
    --accent3: #f59e0b;
    --text:    #e2e8f0;
    --muted:   #64748b;
}

body, .gradio-container {
    background: var(--bg) !important;
    font-family: 'Syne', sans-serif !important;
    color: var(--text) !important;
}
h1, h2, h3 { font-family: 'Syne', sans-serif !important; font-weight: 800 !important; }

.gr-button-primary {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    border: none !important; border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important; font-weight: 700 !important;
    letter-spacing: 0.05em !important; color: white !important;
    transition: opacity 0.2s !important;
}
.gr-button-primary:hover { opacity: 0.88 !important; }

.gr-button-secondary {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important; border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important; color: var(--text) !important;
}
.gr-textbox, .gr-code {
    background: var(--surface) !important; border: 1px solid var(--border) !important;
    border-radius: 8px !important; font-family: 'Space Mono', monospace !important;
    color: var(--text) !important;
}
label {
    font-family: 'Space Mono', monospace !important; color: var(--muted) !important;
    font-size: 0.75rem !important; letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}
.gr-panel, .gr-box {
    background: var(--surface) !important; border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}
.post-panel {
    border: 1px solid var(--accent3) !important; border-radius: 12px !important;
    padding: 1rem !important; background: rgba(245,158,11,0.07) !important;
    margin-top: 0.75rem !important;
}
footer { display: none !important; }
"""

HEADER_HTML = """
<div style="text-align:center;padding:2rem 0 1rem;background:linear-gradient(135deg,#0a0a0f,#12121a);">
  <h1 style="font-family:'Syne',sans-serif;font-size:2.5rem;font-weight:800;
             background:linear-gradient(90deg,#7c3aed,#06b6d4);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0;">
    AI Filter Platform
  </h1>
  <p style="font-family:'Space Mono',monospace;color:#64748b;font-size:0.85rem;
            margin-top:0.5rem;letter-spacing:0.1em;">
    GROQ DRAFT → VALIDATE → DEEPSEEK REPAIR → GROQ STRICT RETRY
  </p>
  <p style="font-family:'Space Mono',monospace;color:#10b981;font-size:0.75rem;margin-top:0.25rem;">
    ✅ API keys loaded from .env
  </p>
</div>
"""

TROUBLESHOOTING_MD = """
### Troubleshooting
- Make sure your `.env` has valid `GROQ_API_KEY` and `OPENROUTER_API_KEY`.
- MongoDB is optional – only needed for save/logging.
- For video, large files may take a while to process frame-by-frame.
"""


# ── UI ────────────────────────────────────────────────────────────────────────

def build_ui():
    with gr.Blocks(title="AI Filter Platform") as demo:
        gr.HTML(HEADER_HTML)

        with gr.Row():
            # ── Left: generate + code ──────────────────────────────────────
            with gr.Column(scale=1):
                gr.Markdown("### 🧠 Generate Filter")
                prompt_box = gr.Textbox(
                    label="Describe your filter",
                    placeholder='e.g. "cinematic teal-orange grade with soft bloom"',
                    lines=3,
                )
                gen_btn    = gr.Button("✨ Generate Filter", variant="primary")

                gr.Markdown("### 🖊️ Filter Code")
                code_box   = gr.Code(label="Generated Code (editable)", language="python", lines=18)
                status_box = gr.Textbox(label="Pipeline Status", interactive=False, lines=6)
                gr.Markdown(TROUBLESHOOTING_MD)

            # ── Right: apply / webcam / video ──────────────────────────────
            with gr.Column(scale=1):

                with gr.Tab("🖼️ Static Image"):
                    gr.Markdown("_Upload → click **Apply Filter**. Nothing runs automatically._")
                    image_input  = gr.Image(label="Upload Image", type="numpy", interactive=True)
                    apply_btn    = gr.Button("🎨 Apply Filter to Image", variant="primary")
                    image_output = gr.Image(label="Filtered Result", type="numpy", interactive=False)
                    apply_status = gr.Textbox(label="Status", interactive=False, lines=2)

                    with gr.Group(elem_classes=["post-panel"], visible=False) as post_panel:
                        gr.Markdown("### 📤 Post / Share Result")
                        post_name_box = gr.Textbox(label="Post Name", placeholder="cinematic_sunset")
                        post_btn      = gr.Button("🚀 Post & Share")
                        post_status   = gr.Textbox(label="Post Status", interactive=False, lines=2)

                with gr.Tab("📹 Video File"):
                    gr.Markdown("_Upload a video → click **Apply to Video**. Output is downloadable._")
                    video_input  = gr.Video(label="Upload Video")
                    video_btn    = gr.Button("🎬 Apply Filter to Video", variant="primary")
                    video_output = gr.Video(label="Filtered Video", interactive=False)
                    video_status = gr.Textbox(label="Status", interactive=False, lines=2)

                with gr.Tab("🎥 Live Webcam"):
                    gr.Markdown(
                        "_Generate a filter first, then open this tab. "
                        "Your webcam is processed **live** — no separate window. "
                        "Allow camera access in your browser when prompted._"
                    )
                    with gr.Row():
                        webcam_in = gr.Image(
                            label="📷 Camera Input",
                            sources=["webcam"],
                            streaming=True,
                            type="numpy",
                        )
                        webcam_out = gr.Image(
                            label="🎨 Filtered Output",
                            type="numpy",
                            interactive=False,
                        )
                    webcam_status = gr.Textbox(
                        label="Status", interactive=False, lines=1,
                        value="Waiting for camera feed…",
                    )

                gr.Markdown("### 💾 Save / Load")
                with gr.Row():
                    filter_name_box = gr.Textbox(label="Filter Name", placeholder="my_filter", scale=2)
                    save_btn        = gr.Button("Save", variant="primary", scale=1)

                saved_filters_dd = refresh_filter_list()
                with gr.Row():
                    load_btn   = gr.Button("Load",   variant="secondary")
                    delete_btn = gr.Button("Delete", variant="secondary")

                manage_status = gr.Textbox(label="Action Status", interactive=False, lines=2)

        # ── Events ────────────────────────────────────────────────────────────

        gen_btn.click(
            fn=generate_filter,
            inputs=[prompt_box],
            outputs=[code_box, status_box],
        )

        apply_btn.click(
            fn=apply_to_image,
            inputs=[code_box, image_input],
            outputs=[image_output, apply_status, post_panel],
        )

        post_btn.click(
            fn=post_filter,
            inputs=[post_name_box, code_box, image_output],
            outputs=[post_status],
        )

        video_btn.click(
            fn=process_video,
            inputs=[code_box, video_input],
            outputs=[video_output, video_status],
        )

        # Live webcam: webcam_in → filter → webcam_out  (never self→self)
        webcam_in.stream(
            fn=process_webcam_frame,
            inputs=[webcam_in, code_box],
            outputs=[webcam_out, webcam_status],
            stream_every=0.04,   # ~25 fps
            time_limit=3600,     # keep stream alive for 1 hour
        )

        save_btn.click(fn=save_filter,   inputs=[filter_name_box, code_box], outputs=[manage_status, saved_filters_dd])
        load_btn.click(fn=load_filter,   inputs=[saved_filters_dd],          outputs=[code_box, manage_status])
        delete_btn.click(fn=delete_filter, inputs=[saved_filters_dd],        outputs=[manage_status, saved_filters_dd])

    return demo


if __name__ == "__main__":
    # Note: Gradio 4/5/6 handles CSS kwargs differently, passing empty strings/dicts can help
    build_ui().launch(server_name="0.0.0.0", server_port=7860, share=False, css=CSS)
