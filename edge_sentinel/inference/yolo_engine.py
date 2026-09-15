import time
import os
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from edge_sentinel.inference.base import BaseInferenceEngine

class YOLOInferenceEngine(BaseInferenceEngine):
    """
    Lightweight YOLOv8 inference engine running locally on CPU/PyTorch.
    Conforms to BaseInferenceEngine for seamless future replacement with
    Qualcomm AI Hub / QNN / ONNX NPU backends.
    """

    def __init__(self, model_path: str = "models/yolov8n.pt", device: str = "cpu", input_size: int = 640, status: str = "ACTIVE"):
        self.model_path = model_path
        self.device = device
        self.input_size = input_size
        self._status = status
        self._model = None
        self._is_loaded = False
        self._class_names: Dict[int, str] = {}

    def load(self) -> bool:
        """Loads YOLOv8 model safely."""
        try:
            from ultralytics import YOLO
            if not os.path.exists(self.model_path):
                print(f"[YOLOEngine] Model path '{self.model_path}' not found. Downloading yolov8n.pt...")
                self._model = YOLO("yolov8n.pt")
                # Save locally if needed
                os.makedirs(os.path.dirname(self.model_path) or ".", exist_ok=True)
            else:
                self._model = YOLO(self.model_path)

            self._class_names = getattr(self._model, "names", {0: "person"})
            self._is_loaded = True
            print(f"[YOLOEngine] Loaded model from '{self.model_path}' on device '{self.device}'.")
            return True
        except Exception as e:
            print(f"[YOLOEngine] Failed to load model: {e}")
            self._is_loaded = False
            return False

    def predict(
        self,
        frame: np.ndarray,
        confidence_threshold: float = 0.5,
        target_classes: Optional[List[int]] = None
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Runs inference on frame and returns detections + actual measured latency in ms.
        """
        if not self._is_loaded or self._model is None:
            return [], 0.0

        if target_classes is None:
            target_classes = [0]  # Default to person only

        start_t = time.perf_counter()
        try:
            # Predict with Ultralytics YOLO
            results = self._model.predict(
                source=frame,
                conf=confidence_threshold,
                classes=target_classes,
                imgsz=self.input_size,
                device=self.device,
                verbose=False
            )
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0

            detections: List[Dict[str, Any]] = []
            if results and len(results) > 0:
                boxes = results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    xyxy = boxes.xyxy.cpu().numpy()
                    confs = boxes.conf.cpu().numpy()
                    cls_ids = boxes.cls.cpu().numpy().astype(int)

                    for i in range(len(xyxy)):
                        cls_id = int(cls_ids[i])
                        label = self._class_names.get(cls_id, "person")
                        detections.append({
                            "bbox": [float(c) for c in xyxy[i]],
                            "confidence": float(confs[i]),
                            "class_id": cls_id,
                            "label": label
                        })

            return detections, elapsed_ms
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            print(f"[YOLOEngine] Inference error: {e}")
            return [], elapsed_ms

    def get_backend_name(self) -> str:
        return "YOLOv8n (Ultralytics PyTorch Local)"

    def get_device(self) -> str:
        return f"{self.device.upper()} (Host System)"

    def get_backend_type(self) -> str:
        return "CPU"

    def get_accelerator_type(self) -> str:
        return "CPU"

    def get_status(self) -> str:
        return self._status

    def set_status(self, status: str) -> None:
        self._status = status

    def is_available(self) -> bool:
        return True

    def get_model_name(self) -> str:
        return "YOLOv8n"

    def get_input_resolution(self) -> Tuple[int, int]:
        return (self.input_size, self.input_size)
