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
        """Loads YOLOv8 model safely via Ultralytics PyTorch or ONNX Runtime CPU."""
        try:
            from ultralytics import YOLO
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(
                    f"[YOLOEngine] Model path '{self.model_path}' not found. "
                    "Edge Sentinel requires local model files and does not perform automatic network downloads."
                )
            self._model = YOLO(self.model_path)

            self._class_names = getattr(self._model, "names", {0: "person", 67: "cell phone"})
            self._is_loaded = True
            print(f"[YOLOEngine] Loaded model from '{self.model_path}' on device '{self.device}'.")
            return True
        except Exception as e:
            # Fallback for environments where PyTorch/Ultralytics is unavailable (e.g. Windows ARM64)
            try:
                import onnxruntime as ort
                onnx_candidates = [
                    "models/qualcomm/yolov8n.onnx",
                    self.model_path.replace(".pt", ".onnx"),
                    self.model_path
                ]
                onnx_path = next((p for p in onnx_candidates if os.path.exists(p) and p.endswith(".onnx")), None)
                if onnx_path:
                    session_options = ort.SessionOptions()
                    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                    self._session = ort.InferenceSession(onnx_path, sess_options=session_options, providers=["CPUExecutionProvider"])
                    self._active_provider = "CPUExecutionProvider"
                    self._class_names = {0: "person", 67: "cell phone"}
                    self._is_loaded = True
                    print(f"[YOLOEngine] Loaded ONNX model '{onnx_path}' on CPUExecutionProvider.")
                    return True
            except Exception:
                pass
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
        if not self._is_loaded or (self._model is None and getattr(self, "_session", None) is None):
            return [], 0.0

        if target_classes is None:
            target_classes = [0]  # Default to person only

        start_t = time.perf_counter()
        if hasattr(self, "_session") and self._session is not None:
            # Predict via ONNX Runtime CPUExecutionProvider
            try:
                from edge_sentinel.inference.snapdragon_engine import SnapdragonInferenceEngine
                input_tensor, orig_shape, ratio, dwdh = SnapdragonInferenceEngine._preprocess(self, frame)
                input_name = self._session.get_inputs()[0].name
                outputs = self._session.run(None, {input_name: input_tensor})
                detections = SnapdragonInferenceEngine._postprocess(
                    self,
                    output_tensor=outputs[0],
                    orig_shape=orig_shape,
                    ratio=ratio,
                    dwdh=dwdh,
                    confidence_threshold=confidence_threshold,
                    target_classes=target_classes
                )
                elapsed_ms = (time.perf_counter() - start_t) * 1000.0
                return detections, elapsed_ms
            except Exception as e:
                elapsed_ms = (time.perf_counter() - start_t) * 1000.0
                print(f"[YOLOEngine] ONNX CPU inference error: {e}")
                return [], elapsed_ms

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
        if getattr(self, "_session", None) is not None:
            return "YOLOv8n (ONNX Runtime CPU)"
        return "YOLOv8n (Ultralytics PyTorch Local)"

    def get_device(self) -> str:
        if getattr(self, "_session", None) is not None:
            return "CPUExecutionProvider (Host System)"
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
