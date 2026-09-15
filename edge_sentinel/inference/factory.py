"""Inference Engine Factory and Backend Selector.

Manages backend selection between CPU, SNAPDRAGON, and AUTO modes,
providing transparent automatic fallback to CPU when Snapdragon
hardware or runtime is unavailable.
"""

from typing import Tuple, Optional
from edge_sentinel.inference.base import BaseInferenceEngine
from edge_sentinel.inference.yolo_engine import YOLOInferenceEngine
from edge_sentinel.inference.snapdragon_engine import SnapdragonInferenceEngine

def create_inference_engine(
    backend: str = "auto",
    model_path: str = "models/yolov8n.pt",
    qnn_model_path: str = "models/qualcomm/yolov8n.onnx",
    device: str = "cpu",
    input_size: int = 640
) -> Tuple[BaseInferenceEngine, str, Optional[str]]:
    """
    Instantiates the appropriate inference engine based on backend configuration.

    Selection modes:
      - 'auto': Attempts Snapdragon NPU if available; falls back to CPU if unavailable.
      - 'snapdragon': Requires Snapdragon; falls back to CPU with a warning if unavailable.
      - 'cpu': Explicitly runs on local CPU.

    Returns:
      Tuple of:
        - engine: BaseInferenceEngine instance
        - active_backend: 'CPU' or 'SNAPDRAGON'
        - fallback_notice: Optional string message if fallback occurred
    """
    mode = (backend or "auto").strip().lower()

    if mode == "cpu":
        print("[InferenceFactory] Configured backend: CPU. Instantiating YOLOInferenceEngine.")
        engine = YOLOInferenceEngine(model_path=model_path, device=device, input_size=input_size, status="ACTIVE")
        return engine, "CPU", None

    if mode in ("snapdragon", "auto"):
        snapdragon_engine = SnapdragonInferenceEngine(
            model_path=qnn_model_path,
            input_size=input_size
        )

        if snapdragon_engine.is_available():
            if snapdragon_engine.load():
                print("[InferenceFactory] Successfully initialized Qualcomm Snapdragon NPU backend.")
                return snapdragon_engine, "SNAPDRAGON", None
            else:
                reason = snapdragon_engine.get_diagnostic_reason() or "Snapdragon engine load failed"
        else:
            reason = snapdragon_engine.get_diagnostic_reason() or "Snapdragon hardware/runtime not available"

        # Fallback to CPU
        notice = f"Snapdragon backend unavailable ({reason}) — CPU fallback active"
        print(f"[InferenceFactory] {notice}")

        cpu_engine = YOLOInferenceEngine(
            model_path=model_path,
            device=device,
            input_size=input_size,
            status="FALLBACK"
        )
        return cpu_engine, "CPU", notice

    # Default fallback for unrecognized backend setting
    notice = f"Unrecognized backend '{backend}' — default CPU fallback active"
    print(f"[InferenceFactory] {notice}")
    engine = YOLOInferenceEngine(model_path=model_path, device=device, input_size=input_size, status="FALLBACK")
    return engine, "CPU", notice
