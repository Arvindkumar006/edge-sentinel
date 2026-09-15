import threading
import time
from typing import Optional, Tuple
import cv2
import numpy as np

class WebcamCapture:
    """
    Thread-safe webcam capture worker with automatic frame acquisition,
    graceful error handling, and reconnection support.
    """

    def __init__(
        self,
        device_index: int = 0,
        width: int = 640,
        height: int = 480,
        target_fps: int = 30,
        auto_reconnect: bool = True,
        reconnect_delay_sec: float = 2.0
    ):
        self.device_index = device_index
        self.width = width
        self.height = height
        self.target_fps = target_fps
        self.auto_reconnect = auto_reconnect
        self.reconnect_delay_sec = reconnect_delay_sec

        self._cap: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self._latest_frame: Optional[np.ndarray] = None
        self._is_opened = False
        self._error_message: Optional[str] = None
        self._fps_actual = 0.0
        self._frame_count = 0
        self._last_fps_time = time.perf_counter()

    def start(self) -> bool:
        """Starts the capture background thread."""
        if self._running:
            return True

        self._running = True
        self._open_camera()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True, name="WebcamCaptureThread")
        self._thread.start()
        return self._is_opened

    def stop(self) -> None:
        """Stops capture and releases camera resources."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._release_camera()

    def _open_camera(self) -> bool:
        """Attempts to open the OpenCV VideoCapture."""
        self._release_camera()
        try:
            # On Windows, cv2.CAP_DSHOW provides fast startup and stable DirectShow bindings
            cap = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)
            if not cap.isOpened():
                # Fallback to standard backend if DSHOW fails
                cap = cv2.VideoCapture(self.device_index)

            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                cap.set(cv2.CAP_PROP_FPS, self.target_fps)
                self._cap = cap
                self._is_opened = True
                self._error_message = None
                print(f"[WebcamCapture] Camera {self.device_index} opened successfully.")
                return True
            else:
                self._is_opened = False
                self._error_message = f"Cannot open camera index {self.device_index}. Device may be in use."
                print(f"[WebcamCapture] {self._error_message}")
                return False
        except Exception as e:
            self._is_opened = False
            self._error_message = f"Camera initialization error: {e}"
            print(f"[WebcamCapture] {self._error_message}")
            return False

    def _release_camera(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        self._is_opened = False

    def _capture_loop(self) -> None:
        """Continuous background thread pulling frames."""
        frame_interval = 1.0 / max(1, self.target_fps)

        while self._running:
            loop_start = time.perf_counter()

            if not self._is_opened or self._cap is None:
                if self.auto_reconnect:
                    time.sleep(self.reconnect_delay_sec)
                    if self._running:
                        self._open_camera()
                else:
                    time.sleep(0.5)
                continue

            try:
                ret, frame = self._cap.read()
                if ret and frame is not None:
                    with self._lock:
                        self._latest_frame = frame

                    # FPS counting
                    self._frame_count += 1
                    now = time.perf_counter()
                    if now - self._last_fps_time >= 1.0:
                        self._fps_actual = self._frame_count / (now - self._last_fps_time)
                        self._frame_count = 0
                        self._last_fps_time = now
                else:
                    self._is_opened = False
                    self._error_message = "Failed to grab frame from camera. Reconnecting..."
                    self._release_camera()
            except Exception as e:
                self._is_opened = False
                self._error_message = f"Frame capture exception: {e}"
                self._release_camera()

            elapsed = time.perf_counter() - loop_start
            sleep_time = max(0.001, frame_interval - elapsed)
            time.sleep(sleep_time)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Reads the most recent frame thread-safely.
        Returns (success, frame_copy).
        """
        with self._lock:
            if self._latest_frame is not None:
                return True, self._latest_frame.copy()
        return False, None

    @property
    def is_opened(self) -> bool:
        return self._is_opened

    @property
    def error_message(self) -> Optional[str]:
        return self._error_message

    @property
    def capture_fps(self) -> float:
        return self._fps_actual
