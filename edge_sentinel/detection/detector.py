import time
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from edge_sentinel.inference.base import BaseInferenceEngine

@dataclass
class DetectionResult:
    frame: np.ndarray
    detections: List[Dict[str, Any]]
    person_count: int
    inference_latency_ms: float
    total_fps: float
    backend_name: str
    device_name: str
    timestamp: float
    device_detections: List[Dict[str, Any]] = None

    def __post_init__(self):
        if self.device_detections is None:
            self.device_detections = []

class PersonDetector:
    """
    Coordinates inference execution, detection filtering, and performance tracking.
    Enforces privacy-preserving local rules: no frames leave this pipeline.
    """

    def __init__(
        self,
        engine: BaseInferenceEngine,
        confidence_threshold: float = 0.50,
        target_classes: Optional[List[int]] = None
    ):
        self.engine = engine
        self.confidence_threshold = confidence_threshold
        self.target_classes = target_classes if target_classes is not None else [0, 67]

        self._fps_window = []
        self._last_time = time.perf_counter()
        self._current_fps = 0.0

    def set_confidence_threshold(self, threshold: float) -> None:
        self.confidence_threshold = max(0.05, min(0.99, threshold))

    def process_frame(self, frame: np.ndarray) -> DetectionResult:
        """
        Executes person and device detection on a frame and calculates real-time metrics.
        """
        now = time.perf_counter()
        dt = now - self._last_time
        self._last_time = now

        # Update smoothed FPS
        if dt > 0:
            instant_fps = 1.0 / dt
            self._fps_window.append(instant_fps)
            if len(self._fps_window) > 15:
                self._fps_window.pop(0)
            self._current_fps = sum(self._fps_window) / len(self._fps_window)

        # Run inference through backend
        raw_detections, latency_ms = self.engine.predict(
            frame=frame,
            confidence_threshold=min(self.confidence_threshold, 0.35),
            target_classes=self.target_classes
        )

        # Filter strictly for person class (class_id == 0)
        person_detections = [
            d for d in raw_detections
            if d.get("class_id") == 0 and d.get("confidence", 0.0) >= self.confidence_threshold
        ]

        # Filter for potential screen-capture devices (class_id == 67: cell phone)
        device_detections = [
            d for d in raw_detections
            if d.get("class_id") == 67 and d.get("confidence", 0.0) >= max(0.25, self.confidence_threshold * 0.70)
        ]

        return DetectionResult(
            frame=frame,
            detections=person_detections,
            person_count=len(person_detections),
            inference_latency_ms=latency_ms,
            total_fps=self._current_fps,
            backend_name=self.engine.get_backend_name(),
            device_name=self.engine.get_device(),
            timestamp=now,
            device_detections=device_detections
        )
