import cv2
import numpy as np
from typing import List, Dict, Any, Optional
from edge_sentinel.tracking.tracker import TrackedPerson
from edge_sentinel.context.schema import WorkspaceContext, PresenceState
from edge_sentinel.threat.schema import ThreatAssessment, PrivacyZone, ThreatLevel

class SentinelVisualizer:
    """
    High-tech modern visualizer for Phase 3:
    - Distinguishes Primary User (Emerald Green) vs Secondary Person (Amber/Red)
    - Visualizes Workstation Privacy Zone (dashed/bracketed boundary with threat response)
    - Displays persistent Track IDs, confidence, presence duration, and proximity
    - Renders telemetry HUD with Threat Level, Threat Score, and latency breakdown
    """

    COLOR_PRIMARY = (80, 220, 120)     # Vibrant emerald green for primary user
    COLOR_SECONDARY = (0, 165, 255)    # Amber/orange for secondary person
    COLOR_SAFE = (80, 220, 120)        # Emerald green
    COLOR_MONITORING = (0, 210, 255)   # Cyber yellow/amber
    COLOR_WARNING = (0, 140, 255)      # Deep amber/orange
    COLOR_CRITICAL = (60, 60, 255)     # Crimson red
    COLOR_BG_DARK = (20, 20, 24)       # Modern matte dark background
    COLOR_WHITE = (250, 250, 250)
    COLOR_MUTED = (160, 160, 160)
    COLOR_CYAN = (235, 206, 0)

    COLOR_DEVICE = (255, 220, 0)       # Cyan for detected screen capture devices (BGR format)

    def __init__(
        self,
        show_fps: bool = True,
        show_latency: bool = True,
        show_privacy_badge: bool = True,
        privacy_zone: Optional[PrivacyZone] = None
    ):
        self.show_fps = show_fps
        self.show_latency = show_latency
        self.show_privacy_badge = show_privacy_badge
        self.privacy_zone = privacy_zone

    def render(
        self,
        frame: np.ndarray,
        tracks: List[TrackedPerson],
        context: WorkspaceContext,
        threat: Optional[ThreatAssessment],
        inference_latency_ms: float,
        tracking_overhead_ms: float,
        threat_overhead_ms: float,
        fps: float,
        device_name: str,
        protection: Optional[Any] = None,
        protection_overhead_ms: float = 0.0,
        device_detections: Optional[List[Dict[str, Any]]] = None
    ) -> np.ndarray:
        """Renders bounding boxes, privacy zone, device detections, badges, HUD, and status onto frame."""
        vis = frame.copy()
        h, w = vis.shape[:2]


        # 1. Draw Workstation Privacy Zone Boundary (if enabled)
        if self.privacy_zone is not None and self.privacy_zone.enabled:
            zx1, zy1, zx2, zy2 = self.privacy_zone.get_pixel_rect((h, w))
            
            # Determine zone color based on threat state
            if threat is not None and threat.inside_privacy_zone:
                zone_color = self.COLOR_CRITICAL if threat.level == ThreatLevel.CRITICAL.value else self.COLOR_WARNING
                zone_label = "PRIVACY ZONE [BREACH DETECTED]"
            elif threat is not None and threat.approaching_privacy_zone:
                zone_color = self.COLOR_MONITORING
                zone_label = f"PRIVACY ZONE [APPROACHING {threat.approach_velocity:.0f} px/s]"
            else:
                zone_color = (120, 100, 40)  # Subtle translucent slate/cyan
                zone_label = "WORKSTATION PRIVACY ZONE"

            # Draw subtle semi-transparent zone fill
            overlay = vis.copy()
            cv2.rectangle(overlay, (zx1, zy1), (zx2, zy2), zone_color, -1)
            cv2.addWeighted(overlay, 0.04, vis, 0.96, 0, vis)

            # Draw corner brackets for privacy zone boundary
            bracket_len = 24
            cv2.rectangle(vis, (zx1, zy1), (zx2, zy2), zone_color, 1, cv2.LINE_AA)
            # Corner accents
            thick = 2
            cv2.line(vis, (zx1, zy1), (zx1 + bracket_len, zy1), zone_color, thick)
            cv2.line(vis, (zx1, zy1), (zx1, zy1 + bracket_len), zone_color, thick)
            cv2.line(vis, (zx2, zy1), (zx2 - bracket_len, zy1), zone_color, thick)
            cv2.line(vis, (zx2, zy1), (zx2, zy1 + bracket_len), zone_color, thick)
            cv2.line(vis, (zx1, zy2), (zx1 + bracket_len, zy2), zone_color, thick)
            cv2.line(vis, (zx1, zy2), (zx1, zy2 - bracket_len), zone_color, thick)
            cv2.line(vis, (zx2, zy2), (zx2 - bracket_len, zy2), zone_color, thick)
            cv2.line(vis, (zx2, zy2), (zx2, zy2 - bracket_len), zone_color, thick)

            # Zone label badge
            (zw, zh), _ = cv2.getTextSize(zone_label, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)
            cv2.rectangle(vis, (zx1, zy1 - zh - 6), (zx1 + zw + 10, zy1), self.COLOR_BG_DARK, -1)
            cv2.rectangle(vis, (zx1, zy1 - zh - 6), (zx1 + zw + 10, zy1), zone_color, 1)
            cv2.putText(vis, zone_label, (zx1 + 5, zy1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.36, zone_color, 1, cv2.LINE_AA)

        primary_tid = context.primary_track_id

        # 2. Draw Bounding Boxes for all active tracks
        for track in tracks:
            is_primary = (track.track_id == primary_tid)
            box_color = self.COLOR_PRIMARY if is_primary else self.COLOR_SECONDARY

            x1, y1, x2, y2 = [int(v) for v in track.bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            # Box fill
            overlay = vis.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, -1)
            fill_alpha = 0.08 if is_primary else 0.12
            cv2.addWeighted(overlay, fill_alpha, vis, 1.0 - fill_alpha, 0, vis)

            # Box outline
            cv2.rectangle(vis, (x1, y1), (x2, y2), box_color, 1, cv2.LINE_AA)

            # Corner brackets
            corner_len = min(22, max(8, int((x2 - x1) * 0.15)))
            thick = 2
            cv2.line(vis, (x1, y1), (x1 + corner_len, y1), box_color, thick, cv2.LINE_AA)
            cv2.line(vis, (x1, y1), (x1, y1 + corner_len), box_color, thick, cv2.LINE_AA)
            cv2.line(vis, (x2, y1), (x2 - corner_len, y1), box_color, thick, cv2.LINE_AA)
            cv2.line(vis, (x2, y1), (x2, y1 + corner_len), box_color, thick, cv2.LINE_AA)
            cv2.line(vis, (x1, y2), (x1 + corner_len, y2), box_color, thick, cv2.LINE_AA)
            cv2.line(vis, (x1, y2), (x1, y2 - corner_len), box_color, thick, cv2.LINE_AA)
            cv2.line(vis, (x2, y2), (x2 - corner_len, y2), box_color, thick, cv2.LINE_AA)
            cv2.line(vis, (x2, y2), (x2, y2 - corner_len), box_color, thick, cv2.LINE_AA)

            # Label badge
            if is_primary:
                label_text = f"[ID: {track.track_id}] PRIMARY USER ({track.confidence * 100:.0f}%)"
                sub_text = f"PRESENCE: {track.duration_visible:.1f}s"
            else:
                label_text = f"[ID: {track.track_id}] SECONDARY ({track.confidence * 100:.0f}%)"
                sub_text = f"PROX: {track.proximity_score:.2f} | {track.duration_visible:.1f}s"

            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
            (stw, sth), _ = cv2.getTextSize(sub_text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            badge_w = max(tw, stw) + 14
            badge_h = th + sth + 12

            tag_y1 = max(0, y1 - badge_h - 4)
            tag_y2 = tag_y1 + badge_h
            tag_x2 = min(w, x1 + badge_w)

            cv2.rectangle(vis, (x1, tag_y1), (tag_x2, tag_y2), self.COLOR_BG_DARK, -1)
            cv2.rectangle(vis, (x1, tag_y1), (tag_x2, tag_y2), box_color, 1)
            cv2.putText(vis, label_text, (x1 + 6, tag_y1 + th + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.44, box_color, 1, cv2.LINE_AA)
            cv2.putText(vis, sub_text, (x1 + 6, tag_y1 + th + sth + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.COLOR_WHITE, 1, cv2.LINE_AA)

        # 3. Draw Device Detections (Cell Phone / Potential Screen-Capture Devices)
        if device_detections:
            for dev in device_detections:
                dbbox = dev.get("bbox", [0, 0, 0, 0])
                dx1, dy1, dx2, dy2 = [int(v) for v in dbbox]
                dx1, dy1 = max(0, dx1), max(0, dy1)
                dx2, dy2 = min(w - 1, dx2), min(h - 1, dy2)
                dconf = dev.get("confidence", 0.0)

                cv2.rectangle(vis, (dx1, dy1), (dx2, dy2), self.COLOR_DEVICE, 2, cv2.LINE_AA)
                dev_tag = f"DEVICE: CELL PHONE ({dconf * 100:.0f}%)"
                (dtw, dth), _ = cv2.getTextSize(dev_tag, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
                dtag_y1 = max(0, dy1 - dth - 6)
                cv2.rectangle(vis, (dx1, dtag_y1), (dx1 + dtw + 8, dy1), self.COLOR_BG_DARK, -1)
                cv2.rectangle(vis, (dx1, dtag_y1), (dx1 + dtw + 8, dy1), self.COLOR_DEVICE, 1)
                cv2.putText(vis, dev_tag, (dx1 + 4, dy1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.COLOR_DEVICE, 1, cv2.LINE_AA)

        # 4. Top Telemetry HUD
        hud_h = 44
        hud_bg = vis[0:hud_h, 0:w].copy()
        cv2.rectangle(vis, (0, 0), (w, hud_h), self.COLOR_BG_DARK, -1)
        cv2.addWeighted(vis[0:hud_h, 0:w], 0.85, hud_bg, 0.15, 0, vis[0:hud_h, 0:w])
        cv2.line(vis, (0, hud_h), (w, hud_h), (50, 50, 60), 1)

        total_lat_ms = inference_latency_ms + tracking_overhead_ms + threat_overhead_ms + protection_overhead_ms
        telemetry_str = (
            f"EDGE SENTINEL P4 | {fps:.1f} FPS | "
            f"Inf: {inference_latency_ms:.1f}ms | Trk: {tracking_overhead_ms:.2f}ms | Thr: {threat_overhead_ms:.2f}ms | "
            f"Prot: {protection_overhead_ms:.2f}ms | Tot: {total_lat_ms:.1f}ms"
        )
        cv2.putText(vis, telemetry_str, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.40, self.COLOR_WHITE, 1, cv2.LINE_AA)

        # Threat & Shield Badges on HUD (Top Right)
        if threat is not None:
            if threat.level == ThreatLevel.CRITICAL.value:
                t_col = self.COLOR_CRITICAL
            elif threat.level == ThreatLevel.WARNING.value:
                t_col = self.COLOR_WARNING
            elif threat.level == ThreatLevel.MONITORING.value:
                t_col = self.COLOR_MONITORING
            else:
                t_col = self.COLOR_SAFE

            shield_str = ""
            if protection is not None and getattr(protection, "shield_active", False):
                shield_str = " | SHIELD: ACTIVE"

            badge_str = f"THREAT: {threat.level} ({threat.score:.0f}/100){shield_str}"
            (bw, bh), _ = cv2.getTextSize(badge_str, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
            badge_x = w - bw - 14
            cv2.putText(vis, badge_str, (badge_x, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.40, t_col, 1, cv2.LINE_AA)
        elif self.show_privacy_badge:
            badge_str = "100% LOCAL ON-DEVICE"
            (bw, bh), _ = cv2.getTextSize(badge_str, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            cv2.putText(vis, badge_str, (w - bw - 14, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.COLOR_SAFE, 1, cv2.LINE_AA)

        # 5. Bottom Status Bar
        btm_h = 34
        cv2.rectangle(vis, (0, h - btm_h), (w, h), self.COLOR_BG_DARK, -1)

        if context.presence_state == PresenceState.ACTIVE.value:
            p_text = f"PRIMARY: ACTIVE (#{context.primary_track_id}, {context.primary_presence_duration:.1f}s)"
        elif context.presence_state == PresenceState.TEMPORARILY_ABSENT.value:
            p_text = f"PRIMARY: TEMPORARILY ABSENT ({context.user_absent_duration:.1f}s)"
        else:
            p_text = f"PRIMARY: ABSENT ({context.user_absent_duration:.1f}s)"

        if threat is not None and threat.reasons and threat.level != ThreatLevel.SAFE.value:
            top_reason = threat.reasons[0]
            status_str = f"{p_text} | ALERT: {top_reason}"
            status_color = self.COLOR_WARNING if threat.level == ThreatLevel.WARNING.value else (self.COLOR_CRITICAL if threat.level == ThreatLevel.CRITICAL.value else self.COLOR_MONITORING)
        else:
            status_str = f"{p_text} | WORKSPACE SECURE"
            status_color = self.COLOR_SAFE

        if threat is not None and getattr(threat, "potential_screen_capture_risk", False):
            status_str += " [SCREEN-CAPTURE RISK]"

        cv2.line(vis, (0, h - btm_h), (w, h - btm_h), status_color, 2)
        cv2.circle(vis, (16, h - int(btm_h / 2)), 5, status_color, -1, cv2.LINE_AA)
        cv2.putText(vis, status_str, (28, h - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.40, self.COLOR_WHITE, 1, cv2.LINE_AA)

        return vis

