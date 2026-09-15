"""Tests for Phase 5: Qualcomm Snapdragon AI Acceleration & CPU Fallback.

Verifies:
1. CPU backend remains functional
2. Snapdragon backend interface conforms to BaseInferenceEngine
3. Backend selection works ('cpu', 'snapdragon', 'auto')
4. AUTO selects Snapdragon when genuinely available
5. AUTO falls back to CPU when unavailable
6. Detection schema is identical across backends
7. Person class (0) remains detected
8. Cell-phone class (67) remains detected
9. Backend telemetry is truthful (never claims NPU without verification)
10. Snapdragon failure triggers automatic CPU fallback without crashing
11. Model compatibility validator accurately measures IoU and confidence delta
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from edge_sentinel.config import AppConfig, ModelConfig, InferenceConfig
from edge_sentinel.inference.base import BaseInferenceEngine
from edge_sentinel.inference.yolo_engine import YOLOInferenceEngine
from edge_sentinel.inference.snapdragon_engine import SnapdragonInferenceEngine
from edge_sentinel.inference.factory import create_inference_engine
from edge_sentinel.inference.validator import ModelCompatibilityValidator, calculate_iou
from edge_sentinel.pipeline import SentinelPipeline

def test_1_cpu_backend_functional():
    """Verifies CPU backend loads, predicts, and implements BaseInferenceEngine."""
    engine = YOLOInferenceEngine()
    assert isinstance(engine, BaseInferenceEngine)
    assert engine.get_backend_type() == "CPU"
    assert engine.get_accelerator_type() == "CPU"
    assert engine.get_status() == "ACTIVE"
    assert engine.is_available() is True
    assert engine.get_model_name() == "YOLOv8n"
    assert engine.get_input_resolution() == (640, 640)

    # Test prediction on synthetic image
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections, lat_ms = engine.predict(dummy_frame)
    assert isinstance(detections, list)
    assert isinstance(lat_ms, float)

def test_2_snapdragon_backend_interface():
    """Verifies SnapdragonInferenceEngine adheres strictly to BaseInferenceEngine."""
    snap_engine = SnapdragonInferenceEngine()
    assert isinstance(snap_engine, BaseInferenceEngine)
    assert snap_engine.get_backend_type() == "SNAPDRAGON"
    assert snap_engine.get_model_name() == "YOLOv8n"
    assert snap_engine.get_input_resolution() == (640, 640)

    # On host AMD64 system without QNN, is_available must be False and give reasons
    is_avail = snap_engine.is_available()
    assert is_avail is False
    reason = snap_engine.get_diagnostic_reason()
    assert "AMD64" in reason or "x86" in reason or "not found" in reason or "missing" in reason

def test_3_backend_selection_cpu():
    """Verifies explicit 'cpu' config selects CPU backend without fallback."""
    config = AppConfig()
    config.inference.backend = "cpu"
    pipeline = SentinelPipeline(config)

    assert pipeline.active_backend == "CPU"
    assert pipeline.fallback_notice is None
    assert isinstance(pipeline.engine, YOLOInferenceEngine)
    assert pipeline.engine.get_status() == "ACTIVE"
    assert pipeline.engine.get_accelerator_type() == "CPU"

def test_4_auto_selects_snapdragon_when_available():
    """Verifies AUTO selects Snapdragon when NPU runtime is genuinely available."""
    with patch.object(SnapdragonInferenceEngine, "is_available", return_value=True):
        with patch.object(SnapdragonInferenceEngine, "load", return_value=True):
            engine, backend, fallback_notice = create_inference_engine(backend="auto")
            assert backend == "SNAPDRAGON"
            assert fallback_notice is None
            assert isinstance(engine, SnapdragonInferenceEngine)

def test_5_auto_falls_back_to_cpu_when_unavailable():
    """Verifies AUTO gracefully falls back to CPU when Snapdragon runtime is absent."""
    with patch.object(SnapdragonInferenceEngine, "is_available", return_value=False):
        engine, backend, fallback_notice = create_inference_engine(backend="auto")
        assert backend == "CPU"
        assert fallback_notice is not None
        assert "Snapdragon backend unavailable" in fallback_notice
        assert "CPU fallback active" in fallback_notice
        assert isinstance(engine, YOLOInferenceEngine)
        assert engine.get_status() == "FALLBACK"

def test_6_detection_schema_identical():
    """Verifies both CPU and Snapdragon mock outputs conform to exact same detection schema."""
    expected_keys = {"bbox", "confidence", "class_id", "label"}

    cpu_engine = YOLOInferenceEngine()
    # Mock Ultralytics predict output
    with patch.object(cpu_engine, "_model") as mock_model:
        mock_box = MagicMock()
        mock_box.__len__.return_value = 1
        mock_box.xyxy.cpu().numpy.return_value = np.array([[100.0, 150.0, 300.0, 450.0]])
        mock_box.conf.cpu().numpy.return_value = np.array([0.88])
        mock_box.cls.cpu().numpy.return_value = np.array([0])
        mock_result = MagicMock()
        mock_result.boxes = mock_box
        mock_model.predict.return_value = [mock_result]
        cpu_engine._is_loaded = True
        cpu_engine._class_names = {0: "person", 67: "cell phone"}

        cpu_dets, _ = cpu_engine.predict(np.zeros((480, 640, 3), dtype=np.uint8))
        assert len(cpu_dets) == 1
        assert set(cpu_dets[0].keys()) == expected_keys
        assert cpu_dets[0]["class_id"] == 0
        assert cpu_dets[0]["label"] == "person"

    # Verify Snapdragon output schema
    snap_engine = SnapdragonInferenceEngine()
    # Postprocess mock output tensor
    dummy_out = np.zeros((1, 84, 8400), dtype=np.float32)
    # Put a detection at index 0: cx=320, cy=240, w=200, h=300, person score=0.92
    dummy_out[0, 0, 0] = 320.0
    dummy_out[0, 1, 0] = 240.0
    dummy_out[0, 2, 0] = 200.0
    dummy_out[0, 3, 0] = 300.0
    dummy_out[0, 4, 0] = 0.92  # class 0: person

    snap_dets = snap_engine._postprocess(
        output_tensor=dummy_out,
        orig_shape=(480, 640),
        ratio=1.0,
        dwdh=(0, 0),
        confidence_threshold=0.5,
        target_classes=[0, 67]
    )
    assert len(snap_dets) == 1
    assert set(snap_dets[0].keys()) == expected_keys
    assert snap_dets[0]["class_id"] == 0
    assert snap_dets[0]["label"] == "person"

def test_7_person_and_phone_classes_preserved():
    """Verifies target classes [0, 67] for person and cell phone are preserved."""
    snap_engine = SnapdragonInferenceEngine()
    dummy_out = np.zeros((1, 84, 8400), dtype=np.float32)
    # Box 0: person (class 0)
    dummy_out[0, 0, 0] = 150.0
    dummy_out[0, 1, 0] = 150.0
    dummy_out[0, 2, 0] = 100.0
    dummy_out[0, 3, 0] = 100.0
    dummy_out[0, 4, 0] = 0.85

    # Box 1: cell phone (class 67 -> index 4 + 67 = 71)
    dummy_out[0, 0, 1] = 400.0
    dummy_out[0, 1, 1] = 300.0
    dummy_out[0, 2, 1] = 50.0
    dummy_out[0, 3, 1] = 80.0
    dummy_out[0, 71, 1] = 0.78

    dets = snap_engine._postprocess(
        output_tensor=dummy_out,
        orig_shape=(480, 640),
        ratio=1.0,
        dwdh=(0, 0),
        confidence_threshold=0.5,
        target_classes=[0, 67]
    )
    cls_ids = {d["class_id"] for d in dets}
    assert 0 in cls_ids
    assert 67 in cls_ids
    labels = {d["label"] for d in dets}
    assert "person" in labels
    assert "cell phone" in labels

def test_8_truthful_telemetry_never_fake_npu():
    """Verifies accelerator is never reported as NPU unless verified on hardware."""
    snap_engine = SnapdragonInferenceEngine()
    assert snap_engine.get_accelerator_type() != "NPU"

    # CPU engine never claims NPU
    cpu_engine = YOLOInferenceEngine()
    assert cpu_engine.get_accelerator_type() == "CPU"

    # Even if Snapdragon backend is selected in config on AMD64, pipeline falls back to CPU
    config = AppConfig()
    config.inference.backend = "snapdragon"
    pipeline = SentinelPipeline(config)
    output = pipeline.process_frame(np.zeros((480, 640, 3), dtype=np.uint8))

    assert output.accelerator_type != "NPU"
    assert output.backend_status == "FALLBACK"
    assert "CPU fallback" in output.fallback_notice

def test_9_snapdragon_runtime_failure_triggers_cpu_fallback():
    """Verifies that if a runtime error occurs during Snapdragon inference, it triggers CPU fallback."""
    config = AppConfig()
    pipeline = SentinelPipeline(config)
    # Simulate active Snapdragon engine that throws RuntimeError on predict
    failing_engine = MagicMock(spec=SnapdragonInferenceEngine)
    failing_engine.predict.side_effect = RuntimeError("Snapdragon NPU hardware timeout")
    failing_engine.get_backend_name.return_value = "Snapdragon QNN"
    failing_engine.get_device.return_value = "Hexagon NPU"
    failing_engine.get_backend_type.return_value = "SNAPDRAGON"
    failing_engine.get_accelerator_type.return_value = "NPU"
    failing_engine.get_status.return_value = "ACTIVE"
    failing_engine.get_model_name.return_value = "YOLOv8n"
    failing_engine.get_input_resolution.return_value = (640, 640)

    pipeline.detector.engine = failing_engine
    pipeline.active_backend = "SNAPDRAGON"

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Must NOT raise, but recover to CPU fallback!
    output = pipeline.process_frame(dummy_frame)

    assert pipeline.active_backend == "CPU"
    assert output.backend_status == "FALLBACK"
    assert output.accelerator_type == "CPU"
    assert "Snapdragon backend unavailable" in output.fallback_notice

def test_10_model_compatibility_validator():
    """Verifies ModelCompatibilityValidator accurately matches boxes and reports tolerances."""
    validator = ModelCompatibilityValidator(min_iou_threshold=0.70, max_conf_diff=0.15)

    cpu_dets = [
        {"bbox": [100.0, 100.0, 300.0, 400.0], "confidence": 0.88, "class_id": 0, "label": "person"},
        {"bbox": [450.0, 200.0, 500.0, 300.0], "confidence": 0.75, "class_id": 67, "label": "cell phone"}
    ]
    snap_dets = [
        {"bbox": [102.0, 98.0, 298.0, 402.0], "confidence": 0.86, "class_id": 0, "label": "person"},
        {"bbox": [448.0, 202.0, 502.0, 298.0], "confidence": 0.72, "class_id": 67, "label": "cell phone"}
    ]

    report = validator.compare_detections(cpu_dets, snap_dets)
    assert report["is_compatible"] is True
    assert report["matched_count"] == 2
    assert report["all_classes_matched"] is True
    assert report["average_iou"] > 0.85
    assert report["max_conf_diff"] < 0.05

def test_11_all_phases_1_to_4_remain_functional():
    """Verifies full end-to-end pipeline with Phase 1-4 capabilities remains operational."""
    config = AppConfig()
    pipeline = SentinelPipeline(config)
    assert pipeline.load() is True

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    output = pipeline.process_frame(frame)

    # Telemetry and contracts from Phase 1 through 5
    assert output.fps >= 0.0
    assert output.inference_latency_ms >= 0.0
    assert output.tracking_overhead_ms >= 0.0
    assert output.threat_overhead_ms >= 0.0
    assert output.protection_overhead_ms >= 0.0
    assert output.threat is not None
    assert output.context is not None
    assert output.protection is not None
    assert output.backend_type in ("CPU", "SNAPDRAGON")
    assert output.backend_status in ("ACTIVE", "FALLBACK")
    assert output.accelerator_type in ("CPU", "NPU", "GPU", "UNKNOWN")
