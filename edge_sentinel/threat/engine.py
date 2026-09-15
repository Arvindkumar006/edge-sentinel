import time
import math
from typing import List, Tuple, Optional, Dict, Any

from edge_sentinel.context.schema import WorkspaceContext, PresenceState
from edge_sentinel.tracking.tracker import TrackedPerson
from edge_sentinel.threat.schema import ThreatLevel, PrivacyZone, ThreatAssessment
from edge_sentinel.threat.device import DeviceThreatDetector, DeviceThreatEvaluation

class ThreatAssessmentEngine:
    """
    Context-Aware Privacy Threat Engine.
    Evaluates multi-person spatial/temporal context against the workstation privacy zone.
    Scores threat (0-100), applies debouncing/hysteresis, and provides explainable reasons.
    """

    def __init__(
        self,
        privacy_zone: Optional[PrivacyZone] = None,
        weight_secondary_detected: float = 25.0,
        weight_close_workstation: float = 20.0,
        weight_persistent_presence: float = 15.0,
        weight_inside_privacy_zone: float = 15.0,
        weight_approaching_privacy_zone: float = 15.0,
        weight_primary_absent: float = 25.0,
        weight_multiple_secondary: float = 10.0,
        proximity_alert_threshold: float = 0.45,
        persistence_duration_threshold_sec: float = 4.0,
        approach_velocity_threshold: float = 15.0,
        smoothing_factor: float = 0.40,
        deactivation_hold_sec: float = 2.0,
        threshold_monitoring: float = 30.0,
        threshold_warning: float = 60.0,
        threshold_critical: float = 80.0,
        device_threat_enabled: bool = True,
        weight_device_detected: float = 20.0,
        weight_device_inside_zone: float = 15.0,
        weight_device_held_by_secondary: float = 25.0,
        weight_device_primary_absent: float = 20.0
    ):
        self.privacy_zone = privacy_zone if privacy_zone is not None else PrivacyZone()
        self.weight_secondary_detected = weight_secondary_detected
        self.weight_close_workstation = weight_close_workstation
        self.weight_persistent_presence = weight_persistent_presence
        self.weight_inside_privacy_zone = weight_inside_privacy_zone
        self.weight_approaching_privacy_zone = weight_approaching_privacy_zone
        self.weight_primary_absent = weight_primary_absent
        self.weight_multiple_secondary = weight_multiple_secondary

        self.proximity_alert_threshold = proximity_alert_threshold
        self.persistence_duration_threshold_sec = persistence_duration_threshold_sec
        self.approach_velocity_threshold = approach_velocity_threshold

        self.smoothing_factor = smoothing_factor
        self.deactivation_hold_sec = deactivation_hold_sec

        self.threshold_monitoring = threshold_monitoring
        self.threshold_warning = threshold_warning
        self.threshold_critical = threshold_critical

        # Device threat detector (Phase 4)
        self.device_detector = DeviceThreatDetector(
            enabled=device_threat_enabled,
            weight_device_detected=weight_device_detected,
            weight_device_inside_zone=weight_device_inside_zone,
            weight_device_held_by_secondary=weight_device_held_by_secondary,
            weight_device_primary_absent=weight_device_primary_absent
        )

        # Hysteresis state
        self._smoothed_score: float = 0.0
        self._current_level: ThreatLevel = ThreatLevel.SAFE
        self._last_high_threat_time: Optional[float] = None


    def reset(self) -> None:
        """Resets engine state."""
        self._smoothed_score = 0.0
        self._current_level = ThreatLevel.SAFE
        self._last_high_threat_time = None

    def assess(
        self,
        context: WorkspaceContext,
        tracks: List[TrackedPerson],
        frame_shape: Tuple[int, int],
        timestamp: Optional[float] = None,
        device_detections: Optional[List[Dict[str, Any]]] = None
    ) -> ThreatAssessment:
        """
        Evaluates current workspace context, tracks, and device detections to compute a threat assessment.
        """
        now = timestamp if timestamp is not None else time.perf_counter()
        frame_h, frame_w = frame_shape[:2]

        # Filter for secondary people
        primary_tid = context.primary_track_id
        secondary_tracks = [t for t in tracks if t.track_id != primary_tid]
        secondary_count = len(secondary_tracks)

        # Baseline: No secondary people detected
        if secondary_count == 0:
            raw_score = 0.0
            reasons = []
            max_duration = 0.0
            is_approaching = False
            max_approach_velocity = 0.0
            is_inside_zone = False
            confidence = 1.0
        else:
            raw_score = 0.0
            reasons = []
            max_duration = max(t.duration_visible for t in secondary_tracks)

            # Signal 1: Secondary person detected
            raw_score += self.weight_secondary_detected
            reasons.append(f"Secondary person detected in field of view ({secondary_count} person{'s' if secondary_count > 1 else ''})")

            # Signal 2: Close to workstation
            is_close = any(
                t.proximity_score >= self.proximity_alert_threshold
                for t in secondary_tracks
            )
            if not is_close and context.nearest_secondary_distance is not None:
                # Also consider distance from primary user
                if context.nearest_secondary_distance <= 0.35:
                    is_close = True

            if is_close:
                raw_score += self.weight_close_workstation
                reasons.append("Secondary person is close to workstation boundary")

            # Signal 3: Persistent presence
            if max_duration >= self.persistence_duration_threshold_sec:
                raw_score += self.weight_persistent_presence
                reasons.append(f"Persistent secondary presence ({max_duration:.1f}s continuous)")

            # Signal 4: Inside privacy zone
            is_inside_zone = any(
                self.privacy_zone.contains(t.center, frame_shape)
                for t in secondary_tracks
            )
            if is_inside_zone:
                raw_score += self.weight_inside_privacy_zone
                reasons.append("Secondary person entered workstation privacy zone")

            # Signal 5: Approaching privacy zone
            is_approaching = False
            max_approach_velocity = 0.0
            zone_center = self.privacy_zone.center_pixel(frame_shape)

            for t in secondary_tracks:
                approaching, vel = self._detect_approach(t, zone_center)
                if approaching:
                    is_approaching = True
                    max_approach_velocity = max(max_approach_velocity, vel)

            if is_approaching:
                raw_score += self.weight_approaching_privacy_zone
                reasons.append(f"Secondary person is moving toward privacy zone ({max_approach_velocity:.1f} px/s)")

            # Signal 6: Primary user absent + secondary person present
            if not context.primary_user_present:
                # High risk if primary is absent and secondary is near or inside zone
                if is_inside_zone or is_close or is_approaching or secondary_count > 0:
                    raw_score += self.weight_primary_absent
                    reasons.append(f"Primary user is absent while secondary person is active ({context.presence_state})")

            # Signal 7: Multiple secondary people
            if secondary_count >= 2:
                raw_score += self.weight_multiple_secondary
                reasons.append(f"Multiple secondary people detected ({secondary_count} onlookers)")

            # Average confidence of secondary detections
            confidence = sum(t.confidence for t in secondary_tracks) / float(secondary_count)

        # Device Threat Evaluation (Phase 4)
        dev_eval = self.device_detector.evaluate(
            device_detections=device_detections or [],
            context=context,
            secondary_tracks=secondary_tracks,
            privacy_zone=self.privacy_zone,
            frame_shape=frame_shape
        )
        if dev_eval.device_detected:
            raw_score += dev_eval.score_contribution
            reasons.extend(dev_eval.reasons)
            if secondary_count == 0 and dev_eval.devices:
                confidence = sum(d.get("confidence", 0.8) for d in dev_eval.devices) / float(len(dev_eval.devices))

        # Strictly clamp raw score to [0, 100]
        clamped_raw_score = max(0.0, min(100.0, raw_score))

        # Hysteresis: Score smoothing
        if self._smoothed_score == 0.0 and clamped_raw_score == 0.0:
            smoothed_score = 0.0
        else:
            smoothed_score = (
                self.smoothing_factor * clamped_raw_score +
                (1.0 - self.smoothing_factor) * self._smoothed_score
            )
        self._smoothed_score = smoothed_score

        # Determine target threat level from smoothed score
        target_level = self._score_to_level(smoothed_score)

        # Hysteresis: Debounced level transitions
        effective_level = self._apply_deactivation_hold(target_level, smoothed_score, now)

        # Ensure reasons are populated cleanly
        if not reasons:
            reasons = ["Workstation environment normal — no threats detected"]

        return ThreatAssessment(
            score=round(smoothed_score, 1),
            raw_score=round(clamped_raw_score, 1),
            level=effective_level.value,
            confidence=round(confidence, 2),
            reasons=reasons,
            secondary_count=secondary_count,
            duration=round(max_duration, 1),
            approaching_privacy_zone=is_approaching,
            approach_velocity=round(max_approach_velocity, 1),
            inside_privacy_zone=is_inside_zone,
            timestamp=now,
            device_detected=dev_eval.device_detected,
            device_count=dev_eval.device_count,
            device_inside_zone=dev_eval.device_inside_zone,
            potential_screen_capture_risk=dev_eval.potential_screen_capture_risk
        )


    def _detect_approach(
        self,
        track: TrackedPerson,
        zone_center: Tuple[float, float]
    ) -> Tuple[bool, float]:
        """
        Lightweight geometric/temporal approach estimation.
        Checks if centroid distance to privacy zone center has decreased over recent history.
        """
        history = getattr(track, "center_history", [])
        if len(history) < 3:
            return False, 0.0

        # Sample starting point (~0.5 - 1.0s ago) and latest point
        t_start, p_start = history[0][2], (history[0][0], history[0][1])
        t_end, p_end = history[-1][2], (history[-1][0], history[-1][1])
        dt = t_end - t_start

        if dt < 0.15:
            return False, 0.0

        dist_start = math.hypot(p_start[0] - zone_center[0], p_start[1] - zone_center[1])
        dist_end = math.hypot(p_end[0] - zone_center[0], p_end[1] - zone_center[1])

        delta_dist = dist_start - dist_end  # Positive if moving closer to zone center
        velocity = delta_dist / dt

        is_approaching = (delta_dist > 15.0 and velocity >= self.approach_velocity_threshold)
        return is_approaching, max(0.0, velocity)

    def _score_to_level(self, score: float) -> ThreatLevel:
        """Maps a 0-100 score to discrete threat levels."""
        if score >= self.threshold_critical:
            return ThreatLevel.CRITICAL
        elif score >= self.threshold_warning:
            return ThreatLevel.WARNING
        elif score >= self.threshold_monitoring:
            return ThreatLevel.MONITORING
        else:
            return ThreatLevel.SAFE

    def _apply_deactivation_hold(
        self,
        target_level: ThreatLevel,
        current_score: float,
        now: float
    ) -> ThreatLevel:
        """
        Prevents rapid jitter/alternation between threat levels.
        If threat steps down, holds current elevated level for deactivation_hold_sec.
        """
        order = {
            ThreatLevel.SAFE: 0,
            ThreatLevel.MONITORING: 1,
            ThreatLevel.WARNING: 2,
            ThreatLevel.CRITICAL: 3
        }

        current_rank = order[self._current_level]
        target_rank = order[target_level]

        if target_rank >= current_rank:
            # Immediate escalation
            self._current_level = target_level
            if target_rank > 0:
                self._last_high_threat_time = now
            return self._current_level

        # Stepping down: Check if hold time has elapsed
        if self._last_high_threat_time is not None:
            time_since_high = now - self._last_high_threat_time
            if time_since_high < self.deactivation_hold_sec:
                # Hold elevated level
                return self._current_level

        # Hold time elapsed: Allow de-escalation
        self._current_level = target_level
        return self._current_level
