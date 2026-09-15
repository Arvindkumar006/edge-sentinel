"""Qualcomm Snapdragon AI Acceleration Engine.

Provides an inference backend conforming to BaseInferenceEngine for Qualcomm
Snapdragon platforms (e.g. Snapdragon X Elite / Snapdragon 8cx Gen 3 Copilot+ PCs).

Deployment Architecture:
  YOLOv8n PyTorch
        ↓
  ONNX Export
        ↓
  QNN-compatible Conversion / Compilation
        ↓
  Qualcomm QNN model / artifact (.bin / .dlc / QNN-context ONNX)
        ↓
  QNN Execution Provider (with QnnHtp backend)
        ↓
  Verified Snapdragon NPU

Strict Truthfulness Rule:
  Exporting to ordinary ONNX does NOT constitute Snapdragon NPU execution.
  Only if QNNExecutionProvider is verified with active QnnHtp.dll backend
  and verified on Snapdragon hardware will accelerator report "NPU".
  Otherwise, it safely reports "CPU" or "UNKNOWN" and triggers fallback.
"""

import os
import platform
import time
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from edge_sentinel.inference.base import BaseInferenceEngine

# Standard COCO class names relevant to Edge Sentinel
COCO_CLASSES = {
    0: "person",
    67: "cell phone"
}

class SnapdragonInferenceEngine(BaseInferenceEngine):
    """
    Qualcomm Snapdragon Inference Engine supporting QNN Execution Provider
    and Qualcomm AI Hub compiled models.
    """

    def __init__(
        self,
        model_path: str = "models/qualcomm/yolov8n.onnx",
        qnn_backend_path: str = "QnnHtp.dll",
        input_size: int = 640,
        status: str = "ACTIVE"
    ):
        self.model_path = model_path
        self.qnn_backend_path = qnn_backend_path
        self.input_size = input_size
        self._status = status
        self._is_loaded = False
        self._session = None
        self._actual_accelerator = "UNKNOWN"
        self._active_provider = "NONE"
        self._diagnostic_reason = ""
        self._class_names = dict(COCO_CLASSES)

    def is_available(self) -> bool:
        """
        Verifies whether Qualcomm Snapdragon hardware and QNN Execution Provider
        are genuinely available on the host machine.
        """
        # 1. Check processor architecture: Snapdragon Windows machines run on ARM64
        machine_arch = platform.machine().lower()
        if "arm64" not in machine_arch and "aarch64" not in machine_arch:
            self._diagnostic_reason = (
                f"Host architecture is '{platform.machine()}' (AMD64/x86). "
                "Qualcomm Snapdragon NPU requires ARM64 Windows platform."
            )
            return False

        # 2. Check if onnxruntime is installed and exposes QNNExecutionProvider
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            if "QNNExecutionProvider" not in providers:
                self._diagnostic_reason = (
                    f"onnxruntime installed but QNNExecutionProvider missing. "
                    f"Available providers: {providers}. Requires onnxruntime-qnn."
                )
                return False
        except ImportError:
            self._diagnostic_reason = (
                "onnxruntime not installed. Requires onnxruntime-qnn on ARM64."
            )
            return False

        # 3. Check for compiled model artifact
        if not os.path.exists(self.model_path):
            self._diagnostic_reason = (
                f"Compiled Qualcomm model artifact '{self.model_path}' not found."
            )
            return False

        self._diagnostic_reason = "Snapdragon NPU environment verified."
        return True

    def load(self) -> bool:
        """
        Attempts to load the compiled model into QNN Execution Provider
        targeting Qualcomm Hexagon NPU.
        """
        if not self.is_available():
            print(f"[SnapdragonEngine] Unavailable: {self._diagnostic_reason}")
            self._is_loaded = False
            self._actual_accelerator = "UNKNOWN"
            return False

        try:
            import onnxruntime as ort

            # Configure QNN Execution Provider options for Snapdragon Hexagon NPU
            # backend_path: QnnHtp.dll directs execution to Hexagon Tensor Processor (NPU)
            qnn_options = {
                "backend_path": self.qnn_backend_path,
                "htp_performance_mode": "burst",
                "enable_htp_fp16_precision": "1"
            }

            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            print(f"[SnapdragonEngine] Attempting to load '{self.model_path}' with QNNExecutionProvider...")
            self._session = ort.InferenceSession(
                self.model_path,
                sess_options=session_options,
                providers=[("QNNExecutionProvider", qnn_options), "CPUExecutionProvider"]
            )

            current_providers = self._session.get_providers()
            if "QNNExecutionProvider" in current_providers:
                self._active_provider = "QNNExecutionProvider"
                self._actual_accelerator = "NPU"
                self._status = "ACTIVE"
                self._is_loaded = True
                print(f"[SnapdragonEngine] Successfully initialized on Qualcomm Hexagon NPU!")
                return True
            else:
                self._active_provider = current_providers[0] if current_providers else "CPU"
                self._actual_accelerator = "CPU"
                self._status = "FALLBACK"
                self._diagnostic_reason = "QNNExecutionProvider failed to initialize; runtime fell back to CPU."
                print(f"[SnapdragonEngine] {self._diagnostic_reason}")
                return False

        except Exception as e:
            self._diagnostic_reason = f"QNN initialization error: {e}"
            print(f"[SnapdragonEngine] Load error: {e}")
            self._is_loaded = False
            self._actual_accelerator = "UNKNOWN"
            return False

    def predict(
        self,
        frame: np.ndarray,
        confidence_threshold: float = 0.5,
        target_classes: Optional[List[int]] = None
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Runs inference via QNN and decodes output bounding boxes.
        Returns detections matching BaseInferenceEngine schema + latency in ms.
        """
        if not self._is_loaded or self._session is None:
            return [], 0.0

        if target_classes is None:
            target_classes = [0, 67]

        start_t = time.perf_counter()
        try:
            # 1. Preprocess frame
            input_tensor, orig_shape, ratio, dwdh = self._preprocess(frame)

            # 2. Run inference
            input_name = self._session.get_inputs()[0].name
            outputs = self._session.run(None, {input_name: input_tensor})
            output_tensor = outputs[0]  # Shape: (1, 84, 8400) or (1, num_boxes, 84)

            # 3. Postprocess and decode detections
            detections = self._postprocess(
                output_tensor=output_tensor,
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
            print(f"[SnapdragonEngine] Prediction error: {e}")
            raise RuntimeError(f"Snapdragon NPU inference failed: {e}")

    def _preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int], float, Tuple[int, int]]:
        """
        Letterbox preprocesses frame into (1, 3, 640, 640) float32 normalized tensor.
        """
        h0, w0 = frame.shape[:2]
        r = min(self.input_size / h0, self.input_size / w0)
        new_unpad = (int(round(w0 * r)), int(round(h0 * r)))
        dw, dh = self.input_size - new_unpad[0], self.input_size - new_unpad[1]
        dw /= 2
        dh /= 2

        # Pillow + NumPy letterbox preprocessing (100% native ARM64, zero OpenCV dependency)
        from PIL import Image
        top = int(round(dh - 0.1))
        left = int(round(dw - 0.1))
        pil_img = Image.fromarray(frame)
        if (w0, h0) != new_unpad:
            pil_img = pil_img.resize(new_unpad, Image.Resampling.BILINEAR)
        padded = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        padded[top:top + new_unpad[1], left:left + new_unpad[0]] = np.array(pil_img)

        # Convert HWC -> CHW, normalize 0.0 - 1.0
        tensor = padded.transpose((2, 0, 1)).astype(np.float32) / 255.0
        tensor = np.expand_dims(tensor, axis=0)
        return tensor, (h0, w0), r, (int(dw), int(dh))

    def _postprocess(
        self,
        output_tensor: np.ndarray,
        orig_shape: Tuple[int, int],
        ratio: float,
        dwdh: Tuple[int, int],
        confidence_threshold: float,
        target_classes: List[int]
    ) -> List[Dict[str, Any]]:
        """
        Decodes YOLOv8 raw output tensor into standard bounding boxes with NMS.
        Handles both (1, 84, 8400) and transposed (1, 8400, 84) output shapes.
        """
        predictions = np.squeeze(output_tensor)
        if predictions.shape[0] == 84:
            predictions = predictions.T  # -> (8400, 84)

        boxes = predictions[:, :4]  # [cx, cy, w, h]
        scores = predictions[:, 4:]  # (8400, 80)

        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)

        # Filter by confidence and target classes
        mask = (confidences >= confidence_threshold)
        if target_classes is not None:
            cls_mask = np.isin(class_ids, target_classes)
            mask = mask & cls_mask

        filtered_boxes = boxes[mask]
        filtered_confs = confidences[mask]
        filtered_cls = class_ids[mask]

        if len(filtered_boxes) == 0:
            return []

        # Convert [cx, cy, w, h] -> [x1, y1, x2, y2] in original frame coordinates
        dw, dh = dwdh
        x1 = (filtered_boxes[:, 0] - filtered_boxes[:, 2] / 2 - dw) / ratio
        y1 = (filtered_boxes[:, 1] - filtered_boxes[:, 3] / 2 - dh) / ratio
        x2 = (filtered_boxes[:, 0] + filtered_boxes[:, 2] / 2 - dw) / ratio
        y2 = (filtered_boxes[:, 1] + filtered_boxes[:, 3] / 2 - dh) / ratio

        # Clip to image boundary
        h0, w0 = orig_shape
        x1 = np.clip(x1, 0, w0)
        y1 = np.clip(y1, 0, h0)
        x2 = np.clip(x2, 0, w0)
        y2 = np.clip(y2, 0, h0)

        # Pure NumPy vectorized Non-Maximum Suppression (zero OpenCV dependency)
        indices = self._numpy_nms(x1, y1, x2, y2, filtered_confs, iou_threshold=0.45)

        detections = []
        for idx in indices:
            c_id = int(filtered_cls[idx])
            detections.append({
                "bbox": [float(x1[idx]), float(y1[idx]), float(x2[idx]), float(y2[idx])],
                "confidence": float(filtered_confs[idx]),
                "class_id": c_id,
                "label": self._class_names.get(c_id, f"class_{c_id}")
            })

        return detections

    @staticmethod
    def _numpy_nms(x1: np.ndarray, y1: np.ndarray, x2: np.ndarray, y2: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45) -> List[int]:
        """Pure NumPy vectorized Non-Maximum Suppression (NMS)."""
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(int(i))
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            inds = np.where(ovr <= iou_threshold)[0]
            order = order[inds + 1]
        return keep

    def get_backend_name(self) -> str:
        return "Qualcomm Snapdragon QNN (Hexagon NPU)"

    def get_device(self) -> str:
        if self._actual_accelerator == "NPU":
            return "Qualcomm Hexagon NPU"
        elif self._actual_accelerator == "CPU":
            return "CPU (Fallback)"
        return "Qualcomm Snapdragon (Offline/Unverified)"

    def get_backend_type(self) -> str:
        return "SNAPDRAGON"

    def get_accelerator_type(self) -> str:
        return self._actual_accelerator

    def get_status(self) -> str:
        return self._status

    def set_status(self, status: str) -> None:
        self._status = status

    def get_model_name(self) -> str:
        return "YOLOv8n"

    def get_input_resolution(self) -> Tuple[int, int]:
        return (self.input_size, self.input_size)

    def get_diagnostic_reason(self) -> str:
        return self._diagnostic_reason
