import pytest
import time
from typing import List, Dict, Any

from edge_sentinel.threat.schema import ThreatLevel, PrivacyZone, ThreatAssessment
from edge_sentinel.threat.engine import ThreatAssessmentEngine
from edge_sentinel.threat.device import DeviceThreatDetector
from edge_sentinel.context.schema import WorkspaceContext, PresenceState
from edge_sentinel.tracking.tracker import TrackedPerson
from edge_sentinel.protection.schema import ProtectionAction, ShieldState, ProtectionDecision
from edge_sentinel.protection.manager import ProtectionManager
from edge_sentinel.protection.lock import WorkstationLocker

@pytest.fixture
def base_context():
    return WorkspaceContext(
        primary_user_present=True,
        primary_track_id=1,
        presence_state=PresenceState.ACTIVE.value,
        people_count=1,
        secondary_people_count=0,
        nearest_secondary_distance=None,
        secondary_presence_duration=0.0,
        user_absent_duration=0.0,
        primary_presence_duration=10.0,
        active_track_ids=[1],
        timestamp=10.0
    )

@pytest.fixture
def absent_context():
    return WorkspaceContext(
        primary_user_present=False,
        primary_track_id=None,
        presence_state=PresenceState.ABSENT.value,
        people_count=1,
        secondary_people_count=1,
        nearest_secondary_distance=0.2,
        secondary_presence_duration=5.0,
        user_absent_duration=6.0,
        primary_presence_duration=0.0,
        active_track_ids=[2],
        timestamp=10.0
    )

@pytest.fixture
def frame_shape():
    return (480, 640)

@pytest.fixture
def privacy_zone():
    return PrivacyZone(enabled=True, x_min=0.25, y_min=0.15, x_max=0.75, y_max=0.90)

@pytest.fixture
def prot_manager():
    return ProtectionManager(
        enabled=True,
        warning_overlay_enabled=True,
        strong_overlay_enabled=True,
        lock_workstation_enabled=False, # Safe default
        lock_require_user_absent=True,
        lock_cooldown_sec=30.0,
        recovery_hold_sec=2.0
    )

# 1. SAFE -> no shield
def test_1_safe_no_shield(prot_manager, base_context):
    safe_threat = ThreatAssessment(
        score=0.0, raw_score=0.0, level=ThreatLevel.SAFE.value,
        confidence=1.0, reasons=["Normal"], secondary_count=0,
        duration=0.0, approaching_privacy_zone=False,
        approach_velocity=0.0, inside_privacy_zone=False
    )
    decision = prot_manager.evaluate(safe_threat, base_context, timestamp=1.0)
    assert decision.shield_active is False
    assert decision.action == ProtectionAction.NONE
    assert decision.overlay_mode == "NONE"
    assert prot_manager.current_state == ShieldState.INACTIVE

# 2. WARNING -> warning shield
def test_2_warning_shows_warning_shield(prot_manager, base_context):
    warning_threat = ThreatAssessment(
        score=65.0, raw_score=70.0, level=ThreatLevel.WARNING.value,
        confidence=0.9, reasons=["Secondary person close to workstation"],
        secondary_count=1, duration=5.0, approaching_privacy_zone=False,
        approach_velocity=0.0, inside_privacy_zone=False
    )
    decision = prot_manager.evaluate(warning_threat, base_context, timestamp=2.0)
    assert decision.shield_active is True
    assert decision.action == ProtectionAction.WARNING_OVERLAY
    assert decision.overlay_mode == "WARNING"
    assert prot_manager.current_state == ShieldState.ACTIVE

# 3. CRITICAL -> strong shield
def test_3_critical_shows_strong_shield(prot_manager, base_context):
    critical_threat = ThreatAssessment(
        score=85.0, raw_score=90.0, level=ThreatLevel.CRITICAL.value,
        confidence=0.95, reasons=["Primary absent and unauthorized person in zone"],
        secondary_count=1, duration=8.0, approaching_privacy_zone=False,
        approach_velocity=0.0, inside_privacy_zone=True
    )
    decision = prot_manager.evaluate(critical_threat, base_context, timestamp=3.0)
    assert decision.shield_active is True
    assert decision.action == ProtectionAction.STRONG_OVERLAY
    assert decision.overlay_mode == "STRONG"
    assert prot_manager.current_state == ShieldState.ACTIVE

# 4. threat recovery -> shield deactivation
def test_4_threat_recovery_deactivates_shield(prot_manager, base_context):
    # First elevate to WARNING at t=10.0
    warning_threat = ThreatAssessment(
        score=65.0, raw_score=65.0, level=ThreatLevel.WARNING.value,
        confidence=0.9, reasons=["Secondary person warning"],
        secondary_count=1, duration=5.0, approaching_privacy_zone=False,
        approach_velocity=0.0, inside_privacy_zone=False
    )
    prot_manager.evaluate(warning_threat, base_context, timestamp=10.0)

    # Safe threat occurs at t=10.5
    safe_threat = ThreatAssessment(
        score=0.0, raw_score=0.0, level=ThreatLevel.SAFE.value,
        confidence=1.0, reasons=["Normal"], secondary_count=0,
        duration=0.0, approaching_privacy_zone=False,
        approach_velocity=0.0, inside_privacy_zone=False
    )
    d_during_hold = prot_manager.evaluate(safe_threat, base_context, timestamp=10.5)
    assert d_during_hold.shield_active is True  # Held during recovery

    # Safe threat past recovery_hold_sec (hold is 2.0s, so t=12.5 is > 2.0s)
    d_after_hold = prot_manager.evaluate(safe_threat, base_context, timestamp=12.5)
    assert d_after_hold.shield_active is False
    assert d_after_hold.action == ProtectionAction.NONE
    assert prot_manager.current_state == ShieldState.INACTIVE

# 5. hysteresis prevents shield flicker
def test_5_hysteresis_prevents_shield_flicker(prot_manager, base_context):
    critical_threat = ThreatAssessment(
        score=85.0, raw_score=85.0, level=ThreatLevel.CRITICAL.value,
        confidence=0.95, reasons=["Critical threat"],
        secondary_count=1, duration=8.0, approaching_privacy_zone=False,
        approach_velocity=0.0, inside_privacy_zone=True
    )
    prot_manager.evaluate(critical_threat, base_context, timestamp=20.0)

    # Temporary 1-frame drop to SAFE at t=20.1 (drop of 100ms)
    safe_threat = ThreatAssessment(
        score=0.0, raw_score=0.0, level=ThreatLevel.SAFE.value,
        confidence=1.0, reasons=["Normal"], secondary_count=0,
        duration=0.0, approaching_privacy_zone=False,
        approach_velocity=0.0, inside_privacy_zone=False
    )
    d_drop = prot_manager.evaluate(safe_threat, base_context, timestamp=20.1)
    # Must NOT flicker to inactive
    assert d_drop.shield_active is True
    assert d_drop.action == ProtectionAction.STRONG_OVERLAY

    # Threat returns at t=20.2
    d_return = prot_manager.evaluate(critical_threat, base_context, timestamp=20.2)
    assert d_return.shield_active is True
    assert d_return.action == ProtectionAction.STRONG_OVERLAY

# 6. CRITICAL + user absent -> lock eligibility
def test_6_critical_user_absent_lock_eligibility(prot_manager, absent_context, base_context):
    critical_threat = ThreatAssessment(
        score=95.0, raw_score=100.0, level=ThreatLevel.CRITICAL.value,
        confidence=0.95, reasons=["Primary absent and intruder near screen"],
        secondary_count=1, duration=8.0, approaching_privacy_zone=False,
        approach_velocity=0.0, inside_privacy_zone=True
    )
    # When user is present, lock should NOT be eligible
    d_present = prot_manager.evaluate(critical_threat, base_context, timestamp=30.0)
    assert d_present.lock_eligible is False

    # When user is ABSENT and threat is CRITICAL, lock IS eligible
    d_absent = prot_manager.evaluate(critical_threat, absent_context, timestamp=30.1)
    assert d_absent.lock_eligible is True
    assert d_absent.shield_active is True
    # Lock is disabled in config by default, so lock_triggered is False
    assert d_absent.lock_triggered is False

# 7. lock cannot repeatedly trigger
def test_7_lock_cannot_repeatedly_trigger():
    locker = WorkstationLocker(enabled=True, cooldown_sec=60.0, dry_run=True)
    # First trigger should succeed
    res1 = locker.trigger_lock(is_critical=True, is_user_absent=True, now=100.0)
    assert res1 is True
    assert locker.lock_count == 1
    assert locker.was_triggered is True

    # Immediate second call during cooldown (at t=105.0s, only 5s elapsed)
    assert locker.is_eligible(is_critical=True, is_user_absent=True, now=105.0) is False
    res2 = locker.trigger_lock(is_critical=True, is_user_absent=True, now=105.0)
    assert res2 is False
    assert locker.lock_count == 1

    # After cooldown has elapsed (t=165.0s, > 60s elapsed)
    assert locker.is_eligible(is_critical=True, is_user_absent=True, now=165.0) is True
    res3 = locker.trigger_lock(is_critical=True, is_user_absent=True, now=165.0)
    assert res3 is True
    assert locker.lock_count == 2

# 8. phone detection parsing
def test_8_phone_detection():
    raw_detections = [
        {"class_id": 0, "confidence": 0.88, "bbox": [100.0, 100.0, 250.0, 400.0]},
        {"class_id": 67, "confidence": 0.75, "bbox": [300.0, 200.0, 360.0, 280.0]}, # cell phone
        {"class_id": 15, "confidence": 0.60, "bbox": [50.0, 50.0, 80.0, 80.0]}      # cat / other
    ]
    # Filter for device class
    phone_dets = [d for d in raw_detections if d.get("class_id") == 67]
    assert len(phone_dets) == 1
    assert phone_dets[0]["class_id"] == 67
    assert phone_dets[0]["confidence"] == 0.75

# 9. device near privacy zone -> increased risk
def test_9_device_near_privacy_zone_increased_risk(base_context, frame_shape, privacy_zone):
    detector = DeviceThreatDetector(enabled=True)
    # Phone inside privacy zone (320, 240 is inside [160..480, 72..432])
    device_det = [{"class_id": 67, "confidence": 0.85, "bbox": [300.0, 200.0, 340.0, 280.0]}]
    sec_track = TrackedPerson(
        track_id=2, bbox=[280.0, 150.0, 380.0, 420.0], confidence=0.90,
        center=(330.0, 285.0), proximity_score=0.5, time_first_seen=0.0,
        time_last_seen=5.0, duration_visible=5.0
    )
    eval_result = detector.evaluate(
        device_detections=device_det,
        context=base_context,
        secondary_tracks=[sec_track],
        privacy_zone=privacy_zone,
        frame_shape=frame_shape
    )
    assert eval_result.device_detected is True
    assert eval_result.device_inside_zone is True
    assert eval_result.is_held_by_secondary is True
    assert eval_result.potential_screen_capture_risk is True
    assert eval_result.score_contribution >= 50.0
    assert any("Potential screen-capture risk" in r for r in eval_result.reasons)
    assert any("Secondary person holding device near privacy zone" in r for r in eval_result.reasons)

# 10. phone far away -> no unnecessary critical alert
def test_10_phone_far_away_no_unnecessary_critical(base_context, frame_shape, privacy_zone):
    detector = DeviceThreatDetector(enabled=True)
    # Small phone far in corner (x=590, y=30), outside privacy zone, no secondary person
    device_det = [{"class_id": 67, "confidence": 0.60, "bbox": [580.0, 20.0, 600.0, 40.0]}]
    eval_result = detector.evaluate(
        device_detections=device_det,
        context=base_context,
        secondary_tracks=[], # No secondary person
        privacy_zone=privacy_zone,
        frame_shape=frame_shape
    )
    assert eval_result.device_detected is True
    assert eval_result.device_inside_zone is False
    assert eval_result.is_held_by_secondary is False
    assert eval_result.score_contribution <= 20.0 # Low contribution, does not breach CRITICAL

# 11. potential screen-capture explanation
def test_11_potential_screen_capture_explanation(base_context, frame_shape, privacy_zone):
    detector = DeviceThreatDetector(enabled=True)
    device_det = [{"class_id": 67, "confidence": 0.80, "bbox": [300.0, 200.0, 360.0, 290.0]}]
    eval_result = detector.evaluate(
        device_detections=device_det,
        context=base_context,
        secondary_tracks=[],
        privacy_zone=privacy_zone,
        frame_shape=frame_shape
    )
    # Check that explanation uses proper terminology and never claims recording intent
    all_reasons = " ".join(eval_result.reasons)
    assert "Potential screen-capture" in all_reasons
    assert "recording" not in all_reasons.lower()
    assert "person is recording" not in all_reasons.lower()

# 12. existing Phase 1–3 pipeline integration remains valid
def test_12_existing_phases_remain_passing(base_context, frame_shape, privacy_zone):
    engine = ThreatAssessmentEngine(
        privacy_zone=privacy_zone,
        smoothing_factor=1.0, # Instantaneous for unit testing
        deactivation_hold_sec=0.0
    )
    primary_track = TrackedPerson(
        track_id=1, bbox=[200.0, 100.0, 440.0, 440.0], confidence=0.95,
        center=(320.0, 270.0), proximity_score=0.8, time_first_seen=0.0,
        time_last_seen=10.0, duration_visible=10.0
    )
    # When assessed with no devices and single user, threat is SAFE
    ass = engine.assess(base_context, [primary_track], frame_shape, timestamp=10.0)
    assert ass.score == 0.0
    assert ass.level == ThreatLevel.SAFE.value
    assert ass.device_detected is False
    assert ass.potential_screen_capture_risk is False
