"""
realtime.py
────────────────────────────────────────────────────────────────────────────────
Real-time webcam filter engine using OpenCV.

Runs the filter function on each captured frame in a background thread.
Targets 15–30 FPS with graceful error recovery per frame.
"""

import cv2
import numpy as np
import threading
import time
from typing import Callable, Optional


class RealtimeEngine:
    """Background thread that captures webcam frames and applies a filter."""

    TARGET_FPS   = 30
    FRAME_DELAY  = 1.0 / TARGET_FPS
    WINDOW_TITLE = "AI Filter – Live Preview  (press Q to quit)"

    def __init__(self) -> None:
        self._thread:   Optional[threading.Thread] = None
        self._running:  bool = False
        self._filter_fn: Optional[Callable] = None

    # ── Public API ────────────────────────────────────────────────────────────
    def start(self, filter_fn: Callable) -> None:
        """Start (or restart) webcam capture with the given filter function."""
        self.stop()                      # stop any existing session
        self._filter_fn = filter_fn
        self._running   = True
        self._thread    = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the capture loop to stop and wait for it to finish."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self._thread = None

    def update_filter(self, filter_fn: Callable) -> None:
        """Swap the active filter without restarting the camera."""
        self._filter_fn = filter_fn

    @property
    def is_running(self) -> bool:
        return self._running and (self._thread is not None) and self._thread.is_alive()

    # ── Internal loop ─────────────────────────────────────────────────────────
    def _loop(self) -> None:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[RealtimeEngine] ERROR: Cannot open webcam (device 0).")
            self._running = False
            return

        # Tune capture resolution for speed
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, self.TARGET_FPS)

        fps_counter = FPSCounter()

        while self._running:
            t_start = time.perf_counter()

            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            # Apply filter – catch per-frame errors so the stream never dies
            try:
                fn = self._filter_fn
                if fn is not None:
                    output = fn(frame)
                    # Ensure uint8 BGR output
                    if output is not None and isinstance(output, np.ndarray):
                        frame = output.astype(np.uint8)
            except Exception as e:
                _overlay_error(frame, str(e))

            # FPS overlay
            fps = fps_counter.tick()
            cv2.putText(
                frame, f"FPS: {fps:.1f}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
            )

            cv2.imshow(self.WINDOW_TITLE, frame)

            # Quit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # Frame-rate limiter
            elapsed = time.perf_counter() - t_start
            wait    = self.FRAME_DELAY - elapsed
            if wait > 0:
                time.sleep(wait)

        cap.release()
        cv2.destroyAllWindows()
        self._running = False


# ── Utilities ─────────────────────────────────────────────────────────────────
class FPSCounter:
    def __init__(self, window: int = 30) -> None:
        self._times: list = []
        self._window = window

    def tick(self) -> float:
        now = time.perf_counter()
        self._times.append(now)
        cutoff = now - 1.0
        self._times = [t for t in self._times if t > cutoff]
        return len(self._times)


def _overlay_error(frame: np.ndarray, msg: str) -> None:
    """Draw a red error message on the frame in-place."""
    cv2.putText(
        frame, f"ERR: {msg[:60]}",
        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
    )
