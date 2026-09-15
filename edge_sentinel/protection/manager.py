import time
from typing import List, Optional, Tuple
from datetime import datetime

from edge_sentinel.threat.schema import ThreatLevel, ThreatAssessment
from edge_sentinel.context.schema import WorkspaceContext, PresenceState
from edge_sentinel.protection.schema import (
    ProtectionAction,
    ShieldState,
    ProtectionDecision,
    ProtectionEvent
)
from edge_sentinel.protection.overlay import ScreenProtectionOverlay
from edge_sentinel.protection.lock import WorkstationLocker

class ProtectionManager:
    """
    Coordinates active privacy defense:
    - Evaluates ThreatAssessment against response policy
    - Manages adaptive privacy shield overlay (WARNING / STRONG)
    - Enforces recovery hold & hysteresis to prevent shield flicker
    - Manages optional Windows workstation locking guardrails
    - Maintains an audit timeline of protection events
    """

    def __init__(
        self,
        enabled: bool = True,
        warning_overlay_enabled: bool = True,
        strong_overlay_enabled: bool = True,
        lock_workstation_enabled: bool = False,
        lock_require_user_absent: bool = True,
        lock_cooldown_sec: float = 60.0,
        recovery_hold_sec: float = 3.0,
        overlay: Optional[ScreenProtectionOverlay] = None,
        max_timeline_events: int = 20
    ):
        self.enabled = enabled
        self.warning_overlay_enabled = warning_overlay_enabled
        self.strong_overlay_enabled = strong_overlay_enabled
        self.recovery_hold_sec = recovery_hold_sec
        self.max_timeline_events = max_timeline_events

        # Windows Locker component
        self.locker = WorkstationLocker(
            enabled=lock_workstation_enabled,
            cooldown_sec=lock_cooldown_sec
        )
        self.lock_require_user_absent = lock_require_user_absent

        # Overlay UI component (optional, can be attached or headless)
        self.overlay = overlay

        # State tracking
        self.current_state = ShieldState.INACTIVE
        self.current_action = ProtectionAction.NONE
        self.current_overlay_mode = "NONE"
        self._last_high_threat_time: Optional[float] = None
        self._last_active_level: Optional[str] = None
        self._timeline: List[ProtectionEvent] = []

    @property
    def timeline(self) -> List[ProtectionEvent]:
        """Returns the list of recent protection events."""
        return list(self._timeline)

    def set_overlay(self, overlay: ScreenProtectionOverlay) -> None:
        self.overlay = overlay

    def evaluate(
        self,
        threat: ThreatAssessment,
        context: WorkspaceContext,
        timestamp: Optional[float] = None
    ) -> ProtectionDecision:
        """
        Evaluates current threat assessment and workspace context to produce a ProtectionDecision.
        """
        now = timestamp if timestamp is not None else time.perf_counter()

        if not self.enabled:
            self._deactivate_shield()
            return ProtectionDecision(
                shield_active=False,
                action=ProtectionAction.NONE,
                overlay_mode="NONE",
                lock_triggered=False,
                lock_eligible=False,
                reasons=["Protection disabled in configuration"],
                event_message="Protection disabled",
                timestamp=now
            )

        level = threat.level
        is_user_absent = (not context.primary_user_present) or (context.presence_state == PresenceState.ABSENT.value)

        # 1. Evaluate policy
        target_action = ProtectionAction.NONE
        target_mode = "NONE"
        reasons = list(threat.reasons)

        if level == ThreatLevel.CRITICAL.value:
            target_action = ProtectionAction.STRONG_OVERLAY
            target_mode = "STRONG" if self.strong_overlay_enabled else "NONE"
            self._last_high_threat_time = now
            self._last_active_level = level

        elif level == ThreatLevel.WARNING.value:
            target_action = ProtectionAction.WARNING_OVERLAY
            target_mode = "WARNING" if self.warning_overlay_enabled else "NONE"
            self._last_high_threat_time = now
            self._last_active_level = level

        elif level == ThreatLevel.MONITORING.value:
            # Subtle notice, no blocking
            target_action = ProtectionAction.MONITORING_NOTICE
            target_mode = "NONE"

        else: # SAFE
            target_action = ProtectionAction.NONE
            target_mode = "NONE"

        # 2. Hysteresis & Recovery Hold
        # If transitioning from an elevated shield state (WARNING/STRONG) towards a lower state,
        # hold the shield active until recovery_hold_sec has completely elapsed.
        effective_action = target_action
        effective_mode = target_mode
        shield_active = False

        if target_action in (ProtectionAction.WARNING_OVERLAY, ProtectionAction.STRONG_OVERLAY):
            # Immediate escalation
            shield_active = True
        else:
            # Check recovery hold
            if self._last_high_threat_time is not None:
                elapsed_since_high = now - self._last_high_threat_time
                if elapsed_since_high < self.recovery_hold_sec:
                    # Hold previous active shield level
                    shield_active = True
                    if self._last_active_level == ThreatLevel.CRITICAL.value and self.strong_overlay_enabled:
                        effective_action = ProtectionAction.STRONG_OVERLAY
                        effective_mode = "STRONG"
                    elif self.warning_overlay_enabled:
                        effective_action = ProtectionAction.WARNING_OVERLAY
                        effective_mode = "WARNING"
                    reasons.append(f"Holding shield during recovery grace period ({self.recovery_hold_sec - elapsed_since_high:.1f}s remaining)")
                else:
                    # Hold elapsed safely
                    self._last_high_threat_time = None
                    self._last_active_level = None
                    shield_active = False
            else:
                shield_active = False

        # 3. Workstation Lock Evaluation (Only at CRITICAL + primary user absent)
        is_critical = (level == ThreatLevel.CRITICAL.value) or (effective_action == ProtectionAction.STRONG_OVERLAY)
        lock_eligible = self.locker.is_eligible(
            is_critical=is_critical,
            is_user_absent=is_user_absent if self.lock_require_user_absent else True,
            now=now
        )
        lock_triggered = False

        if lock_eligible and self.locker.enabled:
            lock_triggered = self.locker.trigger_lock(
                is_critical=is_critical,
                is_user_absent=is_user_absent if self.lock_require_user_absent else True,
                now=now
            )
            if lock_triggered:
                effective_action = ProtectionAction.WORKSTATION_LOCK
                reasons.append("Critical threat while user absent — Windows Workstation Lock triggered")

        # 4. Sync with physical overlay (if attached)
        if self.overlay is not None:
            if shield_active and effective_mode in ("WARNING", "STRONG"):
                primary_reason = threat.reasons[0] if threat.reasons else "Privacy threat detected"
                self.overlay.show(effective_mode, reason_text=primary_reason)
            else:
                self.overlay.hide()

        # 5. Record timeline events on state transitions
        if effective_action != self.current_action:
            self._record_event(effective_action, reasons[0] if reasons else "Status updated", now)

        self.current_action = effective_action
        self.current_overlay_mode = effective_mode
        self.current_state = ShieldState.ACTIVE if shield_active else ShieldState.INACTIVE

        event_msg = self._format_event_message(effective_action, shield_active)

        return ProtectionDecision(
            shield_active=shield_active,
            action=effective_action,
            overlay_mode=effective_mode,
            lock_triggered=lock_triggered,
            lock_eligible=lock_eligible,
            reasons=reasons,
            event_message=event_msg,
            timestamp=now
        )

    def _record_event(self, action: ProtectionAction, reason: str, now: float) -> None:
        time_str = datetime.now().strftime("%H:%M:%S")
        evt = ProtectionEvent(
            timestamp=now,
            time_str=time_str,
            action=action.value,
            reason=reason
        )
        self._timeline.insert(0, evt)
        if len(self._timeline) > self.max_timeline_events:
            self._timeline.pop()

    def _format_event_message(self, action: ProtectionAction, shield_active: bool) -> str:
        if action == ProtectionAction.WORKSTATION_LOCK:
            return "🔒 Workstation Locked"
        elif action == ProtectionAction.STRONG_OVERLAY:
            return "🛡 Strong Privacy Shield Active"
        elif action == ProtectionAction.WARNING_OVERLAY:
            return "⚠ Privacy Warning Active"
        elif action == ProtectionAction.MONITORING_NOTICE:
            return "👁 Monitoring Observation"
        else:
            return "Workspace Secure"

    def _deactivate_shield(self) -> None:
        self.current_state = ShieldState.INACTIVE
        self.current_action = ProtectionAction.NONE
        self.current_overlay_mode = "NONE"
        if self.overlay is not None:
            self.overlay.hide()

    def reset(self) -> None:
        self._deactivate_shield()
        self._last_high_threat_time = None
        self._last_active_level = None
        self.locker.reset()
        self._timeline.clear()
