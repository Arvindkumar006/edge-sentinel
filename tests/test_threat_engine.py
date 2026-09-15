import pytest
import time
from edge_sentinel.context.schema import WorkspaceContext, PresenceState
from edge_sentinel.tracking.tracker import TrackedPerson
from edge_sentinel.threat.schema import ThreatLevel, PrivacyZone, ThreatAssessment
from edge_sentinel.threat.engine import ThreatAssessmentEngine

@pytest.fixture
def base_context():
    """Returns a baseline context with primary user present."""
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
def frame_shape():
    return (480, 640)

@pytest.fixture
def threat_engine():
    # smoothing_factor=1.0 disables smoothing in unit tests so we test deterministic instantaneous scoring
    zone = PrivacyZone(enabled=True, x_min=0.25, y_min=0.15, x_max=0.75, y_max=0.90)
    return ThreatAssessmentEngine(
        privacy_zone=zone,
        weight_secondary_detected=25.0,
        weight_close_workstation=20.0,
        weight_persistent_presence=15.0,
        weight_inside_privacy_zone=15.0,
        weight_approaching_privacy_zone=15.0,
        weight_primary_absent=25.0,
        weight_multiple_secondary=10.0,
        proximity_alert_threshold=0.45,
        persistence_duration_threshold_sec=4.0,
        approach_velocity_threshold=15.0,
        smoothing_factor=1.0,  # instantaneous for testing
        deactivation_hold_sec=0.0
    )


def test_1_no_secondary_person_safe(threat_engine, base_context, frame_shape):
    """Test 1: When only primary user is present, threat is SAFE and score is 0."""
    primary_track = TrackedPerson(
        track_id=1,
        bbox=[200.0, 100.0, 440.0, 440.0],
        confidence=0.95,
        center=(320.0, 270.0),
        proximity_score=0.8,
        time_first_seen=0.0,
        time_last_seen=10.0,
        duration_visible=10.0
    )
    assessment = threat_engine.assess(base_context, [primary_track], frame_shape, timestamp=10.0)
    assert assessment.score == 0.0
    assert assessment.level == ThreatLevel.SAFE.value
    assert assessment.secondary_count == 0
    assert assessment.inside_privacy_zone is False


def test_2_brief_passerby_low_risk(threat_engine, base_context, frame_shape):
    """Test 2: A brief passerby outside privacy zone results in low risk (SAFE, score=25)."""
    primary_track = TrackedPerson(
        track_id=1, bbox=[200.0, 100.0, 440.0, 440.0], confidence=0.95,
        center=(320.0, 270.0), proximity_score=0.8, time_first_seen=0.0,
        time_last_seen=10.0, duration_visible=10.0
    )
    # Brief passerby: duration=0.8s (< 4s), corner position (outside zone), small proximity (0.2)
    secondary_track = TrackedPerson(
        track_id=2, bbox=[540.0, 50.0, 620.0, 200.0], confidence=0.85,
        center=(580.0, 125.0), proximity_score=0.20, time_first_seen=9.2,
        time_last_seen=10.0, duration_visible=0.8
    )
    ctx = base_context
    ctx.people_count = 2
    ctx.secondary_people_count = 1
    ctx.nearest_secondary_distance = 0.65

    assessment = threat_engine.assess(ctx, [primary_track, secondary_track], frame_shape, timestamp=10.0)
    assert assessment.raw_score == 25.0  # Only secondary detected weight (+25)
    assert assessment.level == ThreatLevel.SAFE.value
    assert assessment.secondary_count == 1
    assert assessment.inside_privacy_zone is False


def test_3_persistent_secondary_increased_risk(threat_engine, base_context, frame_shape):
    """Test 3: Lingering secondary person (> 4s) adds persistent presence weight (+15 -> score=40 MONITORING)."""
    primary_track = TrackedPerson(
        track_id=1, bbox=[200.0, 100.0, 440.0, 440.0], confidence=0.95,
        center=(320.0, 270.0), proximity_score=0.8, time_first_seen=0.0,
        time_last_seen=10.0, duration_visible=10.0
    )
    secondary_track = TrackedPerson(
        track_id=2, bbox=[540.0, 50.0, 620.0, 200.0], confidence=0.85,
        center=(580.0, 125.0), proximity_score=0.20, time_first_seen=4.0,
        time_last_seen=10.0, duration_visible=6.0  # >= 4.0s
    )
    ctx = base_context
    ctx.people_count = 2
    ctx.secondary_people_count = 1
    ctx.nearest_secondary_distance = 0.65

    assessment = threat_engine.assess(ctx, [primary_track, secondary_track], frame_shape, timestamp=10.0)
    # 25 (detected) + 15 (persistent) = 40
    assert assessment.raw_score == 40.0
    assert assessment.level == ThreatLevel.MONITORING.value
    assert any("Persistent secondary presence" in r for r in assessment.reasons)


def test_4_nearby_secondary_warning(threat_engine, base_context, frame_shape):
    """Test 4: Persistent person standing close to workstation escalates to WARNING (score >= 60)."""
    primary_track = TrackedPerson(
        track_id=1, bbox=[200.0, 100.0, 440.0, 440.0], confidence=0.95,
        center=(320.0, 270.0), proximity_score=0.8, time_first_seen=0.0,
        time_last_seen=10.0, duration_visible=10.0
    )
    # Close proximity (0.50 >= 0.45) + persistent (5.0s >= 4.0s)
    secondary_track = TrackedPerson(
        track_id=2, bbox=[440.0, 80.0, 600.0, 420.0], confidence=0.90,
        center=(520.0, 250.0), proximity_score=0.55, time_first_seen=5.0,
        time_last_seen=10.0, duration_visible=5.0
    )
    ctx = base_context
    ctx.people_count = 2
    ctx.secondary_people_count = 1
    ctx.nearest_secondary_distance = 0.30

    assessment = threat_engine.assess(ctx, [primary_track, secondary_track], frame_shape, timestamp=10.0)
    # 25 (detected) + 20 (close) + 15 (persistent) = 60 -> WARNING
    assert assessment.raw_score == 60.0
    assert assessment.level == ThreatLevel.WARNING.value
    assert any("close to workstation" in r for r in assessment.reasons)


def test_5_inside_privacy_zone_increased_risk(threat_engine, base_context, frame_shape):
    """Test 5: Secondary person inside privacy zone triggers +15 privacy zone breach points."""
    primary_track = TrackedPerson(
        track_id=1, bbox=[200.0, 100.0, 440.0, 440.0], confidence=0.95,
        center=(320.0, 270.0), proximity_score=0.8, time_first_seen=0.0,
        time_last_seen=10.0, duration_visible=10.0
    )
    # Inside privacy zone: center (380, 240) in (0.25*640=160 to 0.75*640=480, 0.15*480=72 to 0.90*480=432)
    secondary_track = TrackedPerson(
        track_id=2, bbox=[320.0, 100.0, 440.0, 380.0], confidence=0.90,
        center=(380.0, 240.0), proximity_score=0.30, time_first_seen=9.0,
        time_last_seen=10.0, duration_visible=1.0  # brief, not persistent
    )
    ctx = base_context
    ctx.people_count = 2
    ctx.secondary_people_count = 1
    ctx.nearest_secondary_distance = 0.20

    assessment = threat_engine.assess(ctx, [primary_track, secondary_track], frame_shape, timestamp=10.0)
    # 25 (detected) + 20 (close) + 15 (inside zone) = 60
    assert assessment.inside_privacy_zone is True
    assert assessment.raw_score == 60.0
    assert any("entered workstation privacy zone" in r for r in assessment.reasons)


def test_6_approaching_privacy_zone(threat_engine, base_context, frame_shape):
    """Test 6: Secondary person moving toward privacy zone center triggers approach velocity signal."""
    primary_track = TrackedPerson(
        track_id=1, bbox=[200.0, 100.0, 440.0, 440.0], confidence=0.95,
        center=(320.0, 270.0), proximity_score=0.8, time_first_seen=0.0,
        time_last_seen=10.0, duration_visible=10.0
    )
    # Privacy zone center is (320, 252). Secondary person was at (580, 252) at t=9.0, moved to (490, 252) at t=10.0
    # delta_dist = 90 px in 1.0s = 90 px/s (>= 15 px/s threshold)
    secondary_track = TrackedPerson(
        track_id=2, bbox=[450.0, 120.0, 530.0, 380.0], confidence=0.88,
        center=(490.0, 252.0), proximity_score=0.30, time_first_seen=9.0,
        time_last_seen=10.0, duration_visible=1.0,
        center_history=[
            (580.0, 252.0, 9.0),
            (535.0, 252.0, 9.5),
            (490.0, 252.0, 10.0)
        ]
    )
    ctx = base_context
    ctx.people_count = 2
    ctx.secondary_people_count = 1

    assessment = threat_engine.assess(ctx, [primary_track, secondary_track], frame_shape, timestamp=10.0)
    assert assessment.approaching_privacy_zone is True
    assert assessment.approach_velocity >= 15.0
    assert any("moving toward privacy zone" in r for r in assessment.reasons)


def test_7_primary_user_absent_plus_secondary_high_risk(threat_engine, frame_shape):
    """Test 7: Primary user absent while secondary person is near/in zone escalates score (+25)."""
    # Primary user absent context
    ctx = WorkspaceContext(
        primary_user_present=False,
        primary_track_id=None,
        presence_state=PresenceState.ABSENT.value,
        people_count=1,
        secondary_people_count=1,
        nearest_secondary_distance=0.15,
        secondary_presence_duration=5.0,
        user_absent_duration=8.0,
        primary_presence_duration=0.0,
        active_track_ids=[2],
        timestamp=10.0
    )

    secondary_in_zone = TrackedPerson(
        track_id=2, bbox=[220.0, 100.0, 420.0, 400.0], confidence=0.92,
        center=(320.0, 250.0), proximity_score=0.60, time_first_seen=5.0,
        time_last_seen=10.0, duration_visible=5.0
    )

    assessment = threat_engine.assess(ctx, [secondary_in_zone], frame_shape, timestamp=10.0)
    # 25 (detected) + 20 (close) + 15 (persistent) + 15 (in zone) + 25 (primary absent) = 100 -> CRITICAL
    assert assessment.raw_score == 100.0
    assert assessment.level == ThreatLevel.CRITICAL.value
    assert any("Primary user is absent" in r for r in assessment.reasons)


def test_8_multiple_secondary_people(threat_engine, base_context, frame_shape):
    """Test 8: Multiple secondary people adds +10 multi-onlooker points."""
    primary = TrackedPerson(
        track_id=1, bbox=[200.0, 100.0, 440.0, 440.0], confidence=0.95,
        center=(320.0, 270.0), proximity_score=0.8, time_first_seen=0.0,
        time_last_seen=10.0, duration_visible=10.0
    )
    sec1 = TrackedPerson(
        track_id=2, bbox=[50.0, 80.0, 160.0, 300.0], confidence=0.85,
        center=(105.0, 190.0), proximity_score=0.20, time_first_seen=9.0,
        time_last_seen=10.0, duration_visible=1.0
    )
    sec2 = TrackedPerson(
        track_id=3, bbox=[520.0, 80.0, 630.0, 300.0], confidence=0.85,
        center=(575.0, 190.0), proximity_score=0.20, time_first_seen=9.0,
        time_last_seen=10.0, duration_visible=1.0
    )
    ctx = base_context
    ctx.people_count = 3
    ctx.secondary_people_count = 2

    assessment = threat_engine.assess(ctx, [primary, sec1, sec2], frame_shape, timestamp=10.0)
    # 25 (detected) + 10 (multiple secondary) = 35 -> MONITORING
    assert assessment.raw_score == 35.0
    assert assessment.level == ThreatLevel.MONITORING.value
    assert any("Multiple secondary people detected" in r for r in assessment.reasons)


def test_9_score_capped_at_100(threat_engine, frame_shape):
    """Test 9: Combined signals totaling > 100 are strictly capped at 100."""
    ctx = WorkspaceContext(
        primary_user_present=False, primary_track_id=None,
        presence_state=PresenceState.ABSENT.value, people_count=2,
        secondary_people_count=2, nearest_secondary_distance=0.1,
        secondary_presence_duration=10.0, user_absent_duration=10.0,
        primary_presence_duration=0.0, active_track_ids=[2, 3],
        timestamp=10.0
    )
    # All signals active: 25 + 20 + 15 + 15 + 15 + 25 + 10 = 125 -> capped to 100
    sec = TrackedPerson(
        track_id=2, bbox=[250.0, 100.0, 420.0, 400.0], confidence=0.92,
        center=(335.0, 250.0), proximity_score=0.60, time_first_seen=0.0,
        time_last_seen=10.0, duration_visible=10.0,
        center_history=[(400.0, 250.0, 8.0), (335.0, 250.0, 10.0)]
    )
    assessment = threat_engine.assess(ctx, [sec], frame_shape, timestamp=10.0)
    assert assessment.raw_score == 100.0
    assert assessment.score == 100.0


def test_10_threat_level_thresholds(threat_engine):
    """Test 10: Correct mapping from scores to ThreatLevel enum."""
    assert threat_engine._score_to_level(0.0) == ThreatLevel.SAFE
    assert threat_engine._score_to_level(29.9) == ThreatLevel.SAFE
    assert threat_engine._score_to_level(30.0) == ThreatLevel.MONITORING
    assert threat_engine._score_to_level(59.9) == ThreatLevel.MONITORING
    assert threat_engine._score_to_level(60.0) == ThreatLevel.WARNING
    assert threat_engine._score_to_level(79.9) == ThreatLevel.WARNING
    assert threat_engine._score_to_level(80.0) == ThreatLevel.CRITICAL
    assert threat_engine._score_to_level(100.0) == ThreatLevel.CRITICAL


def test_11_hysteresis_and_debouncing(frame_shape, base_context):
    """Test 11: Deactivation hold and smoothing prevents single-frame alternating jitter."""
    zone = PrivacyZone(enabled=True)
    engine = ThreatAssessmentEngine(
        privacy_zone=zone,
        smoothing_factor=0.5,
        deactivation_hold_sec=3.0
    )

    primary = TrackedPerson(
        track_id=1, bbox=[200.0, 100.0, 440.0, 440.0], confidence=0.95,
        center=(320.0, 270.0), proximity_score=0.8, time_first_seen=0.0,
        time_last_seen=10.0, duration_visible=10.0
    )
    # High threat person
    sec_high = TrackedPerson(
        track_id=2, bbox=[300.0, 100.0, 460.0, 420.0], confidence=0.92,
        center=(380.0, 260.0), proximity_score=0.60, time_first_seen=2.0,
        time_last_seen=10.0, duration_visible=8.0
    )
    ctx = base_context
    ctx.people_count = 2
    ctx.secondary_people_count = 1

    # Elevate threat to WARNING/CRITICAL at t=10.0
    for _ in range(5):
        ass1 = engine.assess(ctx, [primary, sec_high], frame_shape, timestamp=10.0)
    assert ass1.level in (ThreatLevel.WARNING.value, ThreatLevel.CRITICAL.value)

    # Next frame at t=11.0: secondary person drops out of view (1 noisy frame)
    # Since deactivation_hold_sec = 3.0s, the elevated level should be HELD
    ass2 = engine.assess(base_context, [primary], frame_shape, timestamp=11.0)
    assert ass2.level in (ThreatLevel.WARNING.value, ThreatLevel.CRITICAL.value)  # Held by hysteresis

    # At t=14.5 (> 3.0s hold): threat level safely de-escalates to SAFE
    ass3 = engine.assess(base_context, [primary], frame_shape, timestamp=14.5)
    assert ass3.level == ThreatLevel.SAFE.value


def test_12_correct_risk_reason_generation(threat_engine, base_context, frame_shape):
    """Test 12: Assessment produces explainable reasons matching active triggers."""
    primary = TrackedPerson(
        track_id=1, bbox=[200.0, 100.0, 440.0, 440.0], confidence=0.95,
        center=(320.0, 270.0), proximity_score=0.8, time_first_seen=0.0,
        time_last_seen=10.0, duration_visible=10.0
    )
    sec = TrackedPerson(
        track_id=2, bbox=[300.0, 100.0, 460.0, 420.0], confidence=0.92,
        center=(380.0, 260.0), proximity_score=0.55, time_first_seen=4.0,
        time_last_seen=10.0, duration_visible=6.0
    )
    ctx = base_context
    ctx.people_count = 2
    ctx.secondary_people_count = 1

    ass = threat_engine.assess(ctx, [primary, sec], frame_shape, timestamp=10.0)
    assert len(ass.reasons) >= 3
    # Verify specific factual strings
    assert any("Secondary person detected" in r for r in ass.reasons)
    assert any("close to workstation" in r for r in ass.reasons)
    assert any("Persistent secondary presence" in r for r in ass.reasons)
    assert any("entered workstation privacy zone" in r for r in ass.reasons)
