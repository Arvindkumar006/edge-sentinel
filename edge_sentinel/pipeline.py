import time
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import numpy as np

from edge_sentinel.config import AppConfig
from edge_sentinel.inference.base import BaseInferenceEngine
from edge_sentinel.inference.yolo_engine import YOLOInferenceEngine
from edge_sentinel.inference.factory import create_inference_engine
from edge_sentinel.detection.detector import PersonDetector
from edge_sentinel.tracking.tracker import PersonTracker, TrackedPerson
from edge_sentinel.context.schema import WorkspaceContext
from edge_sentinel.context.primary_user import PrimaryUserEstimator
from edge_sentinel.threat.schema import PrivacyZone, ThreatAssessment
from edge_sentinel.threat.engine import ThreatAssessmentEngine
from edge_sentinel.protection.schema import ProtectionDecision
from edge_sentinel.protection.manager import ProtectionManager

@dataclass
class PipelineOutput:
    """Consolidated immutable output of a single frame processing cycle."""
    frame: np.ndarray
    tracks: List[TrackedPerson]
    context: WorkspaceContext
    threat: ThreatAssessment
    inference_latency_ms: float
    tracking_overhead_ms: float
    threat_overhead_ms: float
    total_pipeline_latency_ms: float
    fps: float
    backend_name: str
    device_name: str
    timestamp: float
    protection: Optional[ProtectionDecision] = None
    protection_overhead_ms: float = 0.0
    device_detections: List[Dict[str, Any]] = None
    backend_type: str = "CPU"
    accelerator_type: str = "CPU"
    backend_status: str = "ACTIVE"
    fallback_notice: Optional[str] = None
    model_name: str = "YOLOv8n"
    input_resolution: Tuple[int, int] = (640, 640)

    def __post_init__(self):
        if self.device_detections is None:
            self.device_detections = []
        if self.protection is None:
            self.protection = ProtectionDecision()

class SentinelPipeline:
    """
    Coordinates the end-to-end processing pipeline:
    capture -> inference -> detection -> tracking -> context -> threat -> protection.
    Provides clean separation of concerns away from the UI.
    """

    def __init__(self, config: AppConfig):
        self.config = config

        # 1. Inference Engine via Factory (Qualcomm Snapdragon / CPU / Auto)
        backend_choice = getattr(config.inference, "backend", getattr(config.model, "backend", "auto"))
        self.engine, self.active_backend, self.fallback_notice = create_inference_engine(
            backend=backend_choice,
            model_path=config.model.model_path,
            qnn_model_path=config.model.qnn_model_path,
            device=config.model.device,
            input_size=config.model.input_size
        )

        # 2. Detection Coordinator
        self.detector = PersonDetector(
            engine=self.engine,
            confidence_threshold=config.detection.confidence_threshold,
            target_classes=config.model.target_classes
        )

        # 3. Person Tracker
        self.tracker = PersonTracker(
            max_missing_frames=config.tracking.max_missing_frames,
            iou_match_threshold=config.tracking.iou_match_threshold,
            dist_match_threshold=config.tracking.dist_match_threshold
        )

        # 4. Context & Primary User Estimator
        self.context_estimator = PrimaryUserEstimator(
            temp_absence_threshold_sec=config.presence.temp_absence_threshold_sec,
            absent_threshold_sec=config.presence.absent_threshold_sec,
            center_weight=config.presence.center_weight,
            proximity_weight=config.presence.proximity_weight,
            persistence_weight=config.presence.persistence_weight
        )

        # 5. Threat Assessment Engine (Phase 3 + Phase 4 Device Threat)
        privacy_zone = PrivacyZone(
            enabled=config.privacy_zone.enabled,
            x_min=config.privacy_zone.x_min,
            y_min=config.privacy_zone.y_min,
            x_max=config.privacy_zone.x_max,
            y_max=config.privacy_zone.y_max
        )
        self.threat_engine = ThreatAssessmentEngine(
            privacy_zone=privacy_zone,
            weight_secondary_detected=config.threat.weight_secondary_detected,
            weight_close_workstation=config.threat.weight_close_workstation,
            weight_persistent_presence=config.threat.weight_persistent_presence,
            weight_inside_privacy_zone=config.threat.weight_inside_privacy_zone,
            weight_approaching_privacy_zone=config.threat.weight_approaching_privacy_zone,
            weight_primary_absent=config.threat.weight_primary_absent,
            weight_multiple_secondary=config.threat.weight_multiple_secondary,
            proximity_alert_threshold=config.threat.proximity_alert_threshold,
            persistence_duration_threshold_sec=config.threat.persistence_duration_threshold_sec,
            approach_velocity_threshold=config.threat.approach_velocity_threshold,
            smoothing_factor=config.threat.smoothing_factor,
            deactivation_hold_sec=config.threat.deactivation_hold_sec,
            threshold_monitoring=config.threat.threshold_monitoring,
            threshold_warning=config.threat.threshold_warning,
            threshold_critical=config.threat.threshold_critical,
            device_threat_enabled=config.threat.device.enabled,
            weight_device_detected=config.threat.device.weight_device_detected,
            weight_device_inside_zone=config.threat.device.weight_device_inside_zone,
            weight_device_held_by_secondary=config.threat.device.weight_device_held_by_secondary,
            weight_device_primary_absent=config.threat.device.weight_device_primary_absent
        )

        # 6. Active Protection Layer (Phase 4)
        self.protection_manager = ProtectionManager(
            enabled=config.protection.enabled,
            warning_overlay_enabled=config.protection.warning_overlay_enabled,
            strong_overlay_enabled=config.protection.strong_overlay_enabled,
            lock_workstation_enabled=config.protection.lock_workstation_enabled,
            lock_require_user_absent=config.protection.lock_require_user_absent,
            lock_cooldown_sec=config.protection.lock_cooldown_sec,
            recovery_hold_sec=config.protection.recovery_hold_sec
        )

        self._fps_window = []
        self._last_time = time.perf_counter()
        self._current_fps = 0.0

    def load(self) -> bool:
        """Initializes underlying models."""
        return self.engine.load()

    def process_frame(self, frame: np.ndarray, timestamp: Optional[float] = None) -> PipelineOutput:
        """
        Executes a single end-to-end frame processing step:
        inference -> detection -> tracking -> context -> threat -> protection.
        """
        t0 = timestamp if timestamp is not None else time.perf_counter()

        # Update smoothed FPS
        dt = t0 - self._last_time
        self._last_time = t0
        if dt > 0:
            instant_fps = 1.0 / dt
            self._fps_window.append(instant_fps)
            if len(self._fps_window) > 15:
                self._fps_window.pop(0)
            self._current_fps = sum(self._fps_window) / len(self._fps_window)

        # Step 1: Inference & Detection (Person + Potential Screen-Capture Devices)
        try:
            det_result = self.detector.process_frame(frame)
        except Exception as e:
            print(f"[SentinelPipeline] Detection error on current engine: {e}")
            if self.active_backend == "SNAPDRAGON":
                self.switch_to_cpu_fallback(reason=f"Inference error: {e}")
                det_result = self.detector.process_frame(frame)
            else:
                raise e

        inf_latency_ms = det_result.inference_latency_ms

        # Step 2: Tracking (Person Detections)
        t_track_start = time.perf_counter()
        active_tracks = self.tracker.update(
            detections=det_result.detections,
            frame_shape=frame.shape[:2],
            timestamp=t0
        )
        tracking_overhead_ms = (time.perf_counter() - t_track_start) * 1000.0

        # Step 3: Context Estimation
        context = self.context_estimator.update(
            tracks=active_tracks,
            frame_shape=frame.shape[:2],
            timestamp=t0
        )

        # Step 4: Threat Assessment (Person + Device Signals)
        t_threat_start = time.perf_counter()
        threat = self.threat_engine.assess(
            context=context,
            tracks=active_tracks,
            frame_shape=frame.shape[:2],
            timestamp=t0,
            device_detections=det_result.device_detections
        )
        threat_overhead_ms = (time.perf_counter() - t_threat_start) * 1000.0

        # Step 5: Active Protection Evaluation (Phase 4)
        t_prot_start = time.perf_counter()
        protection = self.protection_manager.evaluate(
            threat=threat,
            context=context,
            timestamp=t0
        )
        protection_overhead_ms = (time.perf_counter() - t_prot_start) * 1000.0

        total_latency_ms = inf_latency_ms + tracking_overhead_ms + threat_overhead_ms + protection_overhead_ms

        return PipelineOutput(
            frame=frame,
            tracks=active_tracks,
            context=context,
            threat=threat,
            inference_latency_ms=inf_latency_ms,
            tracking_overhead_ms=tracking_overhead_ms,
            threat_overhead_ms=threat_overhead_ms,
            total_pipeline_latency_ms=total_latency_ms,
            fps=self._current_fps,
            backend_name=self.engine.get_backend_name(),
            device_name=self.engine.get_device(),
            timestamp=t0,
            protection=protection,
            protection_overhead_ms=protection_overhead_ms,
            device_detections=det_result.device_detections,
            backend_type=self.engine.get_backend_type(),
            accelerator_type=self.engine.get_accelerator_type(),
            backend_status=self.engine.get_status(),
            fallback_notice=self.fallback_notice,
            model_name=self.engine.get_model_name(),
            input_resolution=self.engine.get_input_resolution()
        )

    def switch_to_cpu_fallback(self, reason: str = "Runtime failure") -> None:
        """Seamlessly switches active backend to CPU fallback without crashing."""
        print(f"[SentinelPipeline] Switching to CPU fallback: {reason}")
        self.engine = YOLOInferenceEngine(
            model_path=self.config.model.model_path,
            device=self.config.model.device,
            input_size=self.config.model.input_size,
            status="FALLBACK"
        )
        self.engine.load()
        self.detector.engine = self.engine
        self.active_backend = "CPU"
        self.fallback_notice = f"Snapdragon backend unavailable ({reason}) — CPU fallback active"

    def set_confidence_threshold(self, threshold: float) -> None:
        self.detector.set_confidence_threshold(threshold)

    def reset_state(self) -> None:
        self.tracker.reset()
        self.context_estimator.reset()
        self.threat_engine.reset()
        self.protection_manager.reset()
