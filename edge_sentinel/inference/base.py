"""Inference Engine Abstraction Layer.

Defines the contract for model execution so that the underlying backend
(e.g., PyTorch CPU today, Qualcomm QNN / Qualcomm AI Hub NPU tomorrow)
can be swapped transparently without modifying the rest of the pipeline.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
import numpy as np

class BaseInferenceEngine(ABC):
    """Abstract base class for all computer vision inference engines."""

    @abstractmethod
    def load(self) -> bool:
        """Loads and initializes the model weights and runtime context."""
        pass

    @abstractmethod
    def predict(
        self,
        frame: np.ndarray,
        confidence_threshold: float = 0.5,
        target_classes: List[int] = None
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Runs inference on a single image/frame.

        Args:
            frame: BGR numpy array from camera (H, W, C)
            confidence_threshold: Minimum detection confidence (0.0 - 1.0)
            target_classes: Optional list of class IDs to filter (e.g. [0] for person)

        Returns:
            Tuple of:
              - List of detection dictionaries:
                [
                    {
                        "bbox": [x1, y1, x2, y2],  # coordinates as floats
                        "confidence": float,
                        "class_id": int,
                        "label": str
                    },
                    ...
                ]
              - Latency in milliseconds (measured actual inference time)
        """
        pass

    @abstractmethod
    def get_backend_name(self) -> str:
        """Returns the human-readable name of the active inference backend."""
        pass

    @abstractmethod
    def get_device(self) -> str:
        """Returns the execution target device (e.g., 'CPU', 'Qualcomm Hexagon NPU')."""
        pass

    @abstractmethod
    def get_backend_type(self) -> str:
        """Returns 'CPU' or 'SNAPDRAGON'."""
        pass

    @abstractmethod
    def get_accelerator_type(self) -> str:
        """Returns 'CPU', 'NPU', 'GPU', or 'UNKNOWN'."""
        pass

    @abstractmethod
    def get_status(self) -> str:
        """Returns 'ACTIVE' or 'FALLBACK'."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Returns True if the required runtime environment and hardware are available."""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Returns the model descriptor (e.g., 'YOLOv8n')."""
        pass

    @abstractmethod
    def get_input_resolution(self) -> Tuple[int, int]:
        """Returns the expected (width, height) input resolution tuple."""
        pass
