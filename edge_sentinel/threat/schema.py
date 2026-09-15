from enum import Enum
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

class ThreatLevel(str, Enum):
    SAFE = "SAFE"                 # 0 - 29
    MONITORING = "MONITORING"     # 30 - 59
    WARNING = "WARNING"           # 60 - 79
    CRITICAL = "CRITICAL"         # 80 - 100

@dataclass
class PrivacyZone:
    """Configurable workstation privacy zone in normalized [0.0, 1.0] coordinates."""
    enabled: bool = True
    x_min: float = 0.25
    y_min: float = 0.15
    x_max: float = 0.75
    y_max: float = 0.90

    def contains(self, point: Tuple[float, float], frame_shape: Tuple[int, int]) -> bool:
        """Checks if a pixel point (cx, cy) falls inside the normalized privacy zone."""
        if not self.enabled:
            return False
        h, w = frame_shape[:2]
        nx = point[0] / max(1.0, float(w))
        ny = point[1] / max(1.0, float(h))
        return (self.x_min <= nx <= self.x_max) and (self.y_min <= ny <= self.y_max)

    def get_pixel_rect(self, frame_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """Returns (x1, y1, x2, y2) in pixel coordinates."""
        h, w = frame_shape[:2]
        return (
            int(self.x_min * w),
            int(self.y_min * h),
            int(self.x_max * w),
            int(self.y_max * h)
        )

    def center_pixel(self, frame_shape: Tuple[int, int]) -> Tuple[float, float]:
        """Returns the pixel center of the privacy zone."""
        h, w = frame_shape[:2]
        cx = ((self.x_min + self.x_max) / 2.0) * w
        cy = ((self.y_min + self.y_max) / 2.0) * h
        return (cx, cy)

@dataclass
class ThreatAssessment:
    """Consolidated threat assessment output from ThreatAssessmentEngine."""
    score: float                         # 0 - 100 (smoothed / debounced)
    raw_score: float                     # 0 - 100 (instantaneous before smoothing)
    level: str                           # "SAFE", "MONITORING", "WARNING", "CRITICAL"
    confidence: float                    # 0.0 - 1.0
    reasons: List[str]                   # Explainable list of active triggers
    secondary_count: int                 # Count of secondary people
    duration: float                      # Longest secondary person duration (seconds)
    approaching_privacy_zone: bool       # True if secondary person is approaching zone
    approach_velocity: float             # Rate of approach (pixels/sec or normalized/sec)
    inside_privacy_zone: bool            # True if secondary person has breached zone
    timestamp: float = 0.0
    device_detected: bool = False        # True if cell phone/camera detected
    device_count: int = 0                # Count of detected devices
    device_inside_zone: bool = False     # True if device is inside privacy zone
    potential_screen_capture_risk: bool = False # True if device presents screen-capture threat

