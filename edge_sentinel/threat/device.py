import math
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field

from edge_sentinel.threat.schema import PrivacyZone
from edge_sentinel.tracking.tracker import TrackedPerson
from edge_sentinel.context.schema import WorkspaceContext

@dataclass
class DeviceThreatEvaluation:
    """Detailed evaluation result for phone/screen-capture device threats."""
    device_detected: bool = False
    device_count: int = 0
    device_inside_zone: bool = False
    potential_screen_capture_risk: bool = False
    is_held_by_secondary: bool = False
    is_near_workstation: bool = False
    score_contribution: float = 0.0
    reasons: List[str] = field(default_factory=list)
    devices: List[Dict[str, Any]] = field(default_factory=list)

class DeviceThreatDetector:
    """
    Evaluates detected potential screen-capture devices (e.g. smartphones).
    Combines spatial location, privacy zone intersection, proximity, and
    secondary person holding association to produce explainable risk metrics.

    Note on Camera Detection:
    Standard YOLOv8n (COCO) detects class 67 ("cell phone"). Standard COCO does
    not contain a distinct "camera" class. Devices are therefore classified
    as cell phones, and framed objectively as 'Potential screen-capture devices'
    without assuming recording intent.
    """

    def __init__(
        self,
        enabled: bool = True,
        weight_device_detected: float = 20.0,
        weight_device_inside_zone: float = 15.0,
        weight_device_held_by_secondary: float = 25.0,
        weight_device_primary_absent: float = 20.0,
        proximity_threshold: float = 0.015, # Normalized bbox area threshold
        holding_distance_px: float = 140.0
    ):
        self.enabled = enabled
        self.weight_device_detected = weight_device_detected
        self.weight_device_inside_zone = weight_device_inside_zone
        self.weight_device_held_by_secondary = weight_device_held_by_secondary
        self.weight_device_primary_absent = weight_device_primary_absent
        self.proximity_threshold = proximity_threshold
        self.holding_distance_px = holding_distance_px

    def evaluate(
        self,
        device_detections: List[Dict[str, Any]],
        context: WorkspaceContext,
        secondary_tracks: List[TrackedPerson],
        privacy_zone: PrivacyZone,
        frame_shape: Tuple[int, int]
    ) -> DeviceThreatEvaluation:
        if not self.enabled or not device_detections:
            return DeviceThreatEvaluation()

        frame_h, frame_w = frame_shape[:2]
        frame_area = float(frame_h * frame_w)

        device_count = len(device_detections)
        is_inside_zone = False
        is_near_workstation = False
        is_held_by_secondary = False
        score = 0.0
        reasons = []

        # Signal 1: Base device presence
        score += self.weight_device_detected
        reasons.append("Potential screen-capture device detected")

        for d in device_detections:
            bbox = d.get("bbox", [0, 0, 0, 0])
            bw = max(0.0, bbox[2] - bbox[0])
            bh = max(0.0, bbox[3] - bbox[1])
            area_ratio = (bw * bh) / max(1.0, frame_area)
            center = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)

            # Signal 2: Inside or near privacy zone
            if privacy_zone.contains(center, frame_shape):
                is_inside_zone = True

            # Proximity check
            if area_ratio >= self.proximity_threshold or bw >= 50.0 or bh >= 50.0:
                is_near_workstation = True

            # Signal 3: Secondary person holding / in close proximity to the device
            for st in secondary_tracks:
                # Check distance between device center and secondary track center or bbox
                dist = math.hypot(center[0] - st.center[0], center[1] - st.center[1])
                # Also check bbox overlap
                s_bbox = st.bbox
                overlap_x = max(0.0, min(bbox[2], s_bbox[2]) - max(bbox[0], s_bbox[0]))
                overlap_y = max(0.0, min(bbox[3], s_bbox[3]) - max(bbox[1], s_bbox[1]))
                has_overlap = (overlap_x > 0 and overlap_y > 0)

                if dist <= self.holding_distance_px or has_overlap:
                    is_held_by_secondary = True

        if is_near_workstation:
            reasons.append("Smartphone detected near workstation")

        if is_inside_zone:
            score += self.weight_device_inside_zone
            reasons.append("Potential screen-capture device inside workstation privacy zone")

        if is_held_by_secondary:
            score += self.weight_device_held_by_secondary
            reasons.append("Secondary person holding device near privacy zone")

        # Signal 4: Primary user absent + device present
        if not context.primary_user_present and (is_inside_zone or is_held_by_secondary or is_near_workstation):
            score += self.weight_device_primary_absent
            reasons.append("Screen-capture risk escalated while primary user absent")

        # Flag general screen-capture risk if near workstation, held by secondary, or in zone
        screen_capture_risk = is_inside_zone or is_held_by_secondary or (is_near_workstation and len(secondary_tracks) > 0)
        if screen_capture_risk:
            reasons.append("Potential screen-capture risk")

        return DeviceThreatEvaluation(
            device_detected=True,
            device_count=device_count,
            device_inside_zone=is_inside_zone,
            potential_screen_capture_risk=screen_capture_risk,
            is_held_by_secondary=is_held_by_secondary,
            is_near_workstation=is_near_workstation,
            score_contribution=score,
            reasons=reasons,
            devices=device_detections
        )
