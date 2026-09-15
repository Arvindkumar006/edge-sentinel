import os
import yaml
from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class CameraConfig:
    device_index: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30
    auto_reconnect: bool = True
    reconnect_delay_sec: float = 2.0

@dataclass
class InferenceConfig:
    backend: str = "auto"  # 'auto', 'snapdragon', 'cpu'
    qnn_backend_path: str = "QnnHtp.dll"
    allow_fallback: bool = True

@dataclass
class ModelConfig:
    engine: str = "yolov8"
    model_path: str = "models/yolov8n.pt"
    qnn_model_path: str = "models/qualcomm/yolov8n.onnx"
    backend: str = "auto"  # 'auto', 'snapdragon', 'cpu'
    device: str = "cpu"
    input_size: int = 640
    target_classes: List[int] = field(default_factory=lambda: [0, 67])

@dataclass
class DetectionConfig:
    confidence_threshold: float = 0.50
    iou_threshold: float = 0.45
    max_detections: int = 10

@dataclass
class TrackingConfig:
    max_missing_frames: int = 20
    iou_match_threshold: float = 0.30
    dist_match_threshold: float = 120.0

@dataclass
class PresenceConfig:
    temp_absence_threshold_sec: float = 2.5
    absent_threshold_sec: float = 7.0
    center_weight: float = 0.40
    proximity_weight: float = 0.35
    persistence_weight: float = 0.25

@dataclass
class PrivacyZoneConfig:
    enabled: bool = True
    x_min: float = 0.25
    y_min: float = 0.15
    x_max: float = 0.75
    y_max: float = 0.90

@dataclass
class DeviceThreatConfig:
    enabled: bool = True
    weight_device_detected: float = 20.0
    weight_device_inside_zone: float = 15.0
    weight_device_held_by_secondary: float = 25.0
    weight_device_primary_absent: float = 20.0
    proximity_threshold: float = 0.015

@dataclass
class ThreatConfig:
    weight_secondary_detected: float = 25.0
    weight_close_workstation: float = 20.0
    weight_persistent_presence: float = 15.0
    weight_inside_privacy_zone: float = 15.0
    weight_approaching_privacy_zone: float = 15.0
    weight_primary_absent: float = 25.0
    weight_multiple_secondary: float = 10.0
    proximity_alert_threshold: float = 0.45
    persistence_duration_threshold_sec: float = 4.0
    approach_velocity_threshold: float = 15.0
    smoothing_factor: float = 0.40
    deactivation_hold_sec: float = 2.0
    threshold_monitoring: float = 30.0
    threshold_warning: float = 60.0
    threshold_critical: float = 80.0
    device: DeviceThreatConfig = field(default_factory=DeviceThreatConfig)

@dataclass
class ProtectionConfig:
    enabled: bool = True
    warning_overlay_enabled: bool = True
    strong_overlay_enabled: bool = True
    lock_workstation_enabled: bool = False  # Disabled by default
    lock_require_user_absent: bool = True
    lock_cooldown_sec: float = 60.0
    recovery_hold_sec: float = 3.0
    warning_alpha: float = 0.85
    strong_alpha: float = 0.95

@dataclass
class UIConfig:
    theme: str = "dark"
    window_title: str = "Edge Sentinel — Active Privacy Defense (Phase 4)"
    window_width: int = 1220
    window_height: int = 760
    display_fps: bool = True
    display_latency: bool = True
    display_privacy_badge: bool = True

@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    presence: PresenceConfig = field(default_factory=PresenceConfig)
    privacy_zone: PrivacyZoneConfig = field(default_factory=PrivacyZoneConfig)
    threat: ThreatConfig = field(default_factory=ThreatConfig)
    protection: ProtectionConfig = field(default_factory=ProtectionConfig)
    ui: UIConfig = field(default_factory=UIConfig)

def load_config(config_path: str = "config/config.yaml") -> AppConfig:
    """Loads configuration from YAML file with fallback to default parameters."""
    if not os.path.exists(config_path):
        return AppConfig()

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        cam_data = data.get("camera", {})
        inf_data = data.get("inference", {})
        model_data = data.get("model", {})
        det_data = data.get("detection", {})
        track_data = data.get("tracking", {})
        pres_data = data.get("presence", {})
        zone_data = data.get("privacy_zone", {})
        threat_data = data.get("threat", {})
        dev_data = threat_data.get("device", data.get("device_threat", {}))
        prot_data = data.get("protection", {})
        ui_data = data.get("ui", {})

        # Unified backend resolution: priority to inference.backend, then model.backend, default 'auto'
        backend_choice = inf_data.get("backend", model_data.get("backend", "auto"))

        return AppConfig(
            camera=CameraConfig(
                device_index=cam_data.get("device_index", 0),
                width=cam_data.get("width", 640),
                height=cam_data.get("height", 480),
                fps=cam_data.get("fps", 30),
                auto_reconnect=cam_data.get("auto_reconnect", True),
                reconnect_delay_sec=cam_data.get("reconnect_delay_sec", 2.0),
            ),
            inference=InferenceConfig(
                backend=backend_choice,
                qnn_backend_path=inf_data.get("qnn_backend_path", "QnnHtp.dll"),
                allow_fallback=inf_data.get("allow_fallback", True),
            ),
            model=ModelConfig(
                engine=model_data.get("engine", "yolov8"),
                model_path=model_data.get("model_path", "models/yolov8n.pt"),
                qnn_model_path=model_data.get("qnn_model_path", "models/qualcomm/yolov8n.onnx"),
                backend=backend_choice,
                device=model_data.get("device", "cpu"),
                input_size=model_data.get("input_size", 640),
                target_classes=model_data.get("target_classes", [0, 67]),
            ),
            detection=DetectionConfig(
                confidence_threshold=det_data.get("confidence_threshold", 0.50),
                iou_threshold=det_data.get("iou_threshold", 0.45),
                max_detections=det_data.get("max_detections", 10),
            ),
            tracking=TrackingConfig(
                max_missing_frames=track_data.get("max_missing_frames", 20),
                iou_match_threshold=track_data.get("iou_match_threshold", 0.30),
                dist_match_threshold=track_data.get("dist_match_threshold", 120.0),
            ),
            presence=PresenceConfig(
                temp_absence_threshold_sec=pres_data.get("temp_absence_threshold_sec", 2.5),
                absent_threshold_sec=pres_data.get("absent_threshold_sec", 7.0),
                center_weight=pres_data.get("center_weight", 0.40),
                proximity_weight=pres_data.get("proximity_weight", 0.35),
                persistence_weight=pres_data.get("persistence_weight", 0.25),
            ),
            privacy_zone=PrivacyZoneConfig(
                enabled=zone_data.get("enabled", True),
                x_min=zone_data.get("x_min", 0.25),
                y_min=zone_data.get("y_min", 0.15),
                x_max=zone_data.get("x_max", 0.75),
                y_max=zone_data.get("y_max", 0.90),
            ),
            threat=ThreatConfig(
                weight_secondary_detected=threat_data.get("weight_secondary_detected", 25.0),
                weight_close_workstation=threat_data.get("weight_close_workstation", 20.0),
                weight_persistent_presence=threat_data.get("weight_persistent_presence", 15.0),
                weight_inside_privacy_zone=threat_data.get("weight_inside_privacy_zone", 15.0),
                weight_approaching_privacy_zone=threat_data.get("weight_approaching_privacy_zone", 15.0),
                weight_primary_absent=threat_data.get("weight_primary_absent", 25.0),
                weight_multiple_secondary=threat_data.get("weight_multiple_secondary", 10.0),
                proximity_alert_threshold=threat_data.get("proximity_alert_threshold", 0.45),
                persistence_duration_threshold_sec=threat_data.get("persistence_duration_threshold_sec", 4.0),
                approach_velocity_threshold=threat_data.get("approach_velocity_threshold", 15.0),
                smoothing_factor=threat_data.get("smoothing_factor", 0.40),
                deactivation_hold_sec=threat_data.get("deactivation_hold_sec", 2.0),
                threshold_monitoring=threat_data.get("threshold_monitoring", 30.0),
                threshold_warning=threat_data.get("threshold_warning", 60.0),
                threshold_critical=threat_data.get("threshold_critical", 80.0),
                device=DeviceThreatConfig(
                    enabled=dev_data.get("enabled", True),
                    weight_device_detected=dev_data.get("weight_device_detected", 20.0),
                    weight_device_inside_zone=dev_data.get("weight_device_inside_zone", 15.0),
                    weight_device_held_by_secondary=dev_data.get("weight_device_held_by_secondary", 25.0),
                    weight_device_primary_absent=dev_data.get("weight_device_primary_absent", 20.0),
                    proximity_threshold=dev_data.get("proximity_threshold", 0.015),
                )
            ),
            protection=ProtectionConfig(
                enabled=prot_data.get("enabled", True),
                warning_overlay_enabled=prot_data.get("warning_overlay_enabled", True),
                strong_overlay_enabled=prot_data.get("strong_overlay_enabled", True),
                lock_workstation_enabled=prot_data.get("lock_workstation_enabled", False),
                lock_require_user_absent=prot_data.get("lock_require_user_absent", True),
                lock_cooldown_sec=prot_data.get("lock_cooldown_sec", 60.0),
                recovery_hold_sec=prot_data.get("recovery_hold_sec", 3.0),
                warning_alpha=prot_data.get("warning_alpha", 0.85),
                strong_alpha=prot_data.get("strong_alpha", 0.95),
            ),
            ui=UIConfig(
                theme=ui_data.get("theme", "dark"),
                window_title=ui_data.get("window_title", "Edge Sentinel — Active Privacy Defense (Phase 4)"),
                window_width=ui_data.get("window_width", 1220),
                window_height=ui_data.get("window_height", 760),
                display_fps=ui_data.get("display_fps", True),
                display_latency=ui_data.get("display_latency", True),
                display_privacy_badge=ui_data.get("display_privacy_badge", True),
            ),
        )
    except Exception as e:
        print(f"[Config] Error reading {config_path}: {e}. Using defaults.")
        return AppConfig()
