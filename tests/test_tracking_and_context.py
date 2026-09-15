import pytest
import time
from edge_sentinel.tracking.tracker import PersonTracker, TrackedPerson
from edge_sentinel.context.schema import PresenceState, WorkspaceContext
from edge_sentinel.context.primary_user import PrimaryUserEstimator

def test_stable_track_ids():
    """Test that a moving person retains the same track_id across consecutive frames."""
    tracker = PersonTracker(max_missing_frames=10, iou_match_threshold=0.3)
    frame_shape = (480, 640)

    # Frame 1: Person at (200, 100, 400, 400)
    dets_f1 = [{"bbox": [200.0, 100.0, 400.0, 400.0], "confidence": 0.90}]
    tracks_f1 = tracker.update(dets_f1, frame_shape, timestamp=1.0)
    assert len(tracks_f1) == 1
    original_id = tracks_f1[0].track_id

    # Frame 2: Person shifts slightly to (205, 102, 405, 402)
    dets_f2 = [{"bbox": [205.0, 102.0, 405.0, 402.0], "confidence": 0.91}]
    tracks_f2 = tracker.update(dets_f2, frame_shape, timestamp=1.1)
    assert len(tracks_f2) == 1
    assert tracks_f2[0].track_id == original_id

    # Frame 3: Person shifts further to (215, 105, 415, 405)
    dets_f3 = [{"bbox": [215.0, 105.0, 415.0, 405.0], "confidence": 0.89}]
    tracks_f3 = tracker.update(dets_f3, frame_shape, timestamp=1.2)
    assert len(tracks_f3) == 1
    assert tracks_f3[0].track_id == original_id
    assert tracks_f3[0].hits == 3


def test_person_entering_and_leaving():
    """Test track birth upon entry and expiration after max_missing_frames."""
    tracker = PersonTracker(max_missing_frames=5)
    frame_shape = (480, 640)

    # Person 1 appears
    tracks = tracker.update([{"bbox": [100.0, 100.0, 250.0, 400.0], "confidence": 0.88}], frame_shape, timestamp=1.0)
    assert len(tracks) == 1
    p1_id = tracks[0].track_id

    # Person 2 enters the scene
    tracks = tracker.update([
        {"bbox": [100.0, 100.0, 250.0, 400.0], "confidence": 0.88},
        {"bbox": [400.0, 120.0, 550.0, 420.0], "confidence": 0.85}
    ], frame_shape, timestamp=1.1)
    assert len(tracks) == 2
    assert tracker.active_track_count == 2
    p2_id = [t.track_id for t in tracks if t.track_id != p1_id][0]
    assert p2_id != p1_id

    # Person 2 leaves (only Person 1 detected for 6 frames)
    for f in range(6):
        tracks = tracker.update([{"bbox": [100.0, 100.0, 250.0, 400.0], "confidence": 0.88}], frame_shape, timestamp=1.2 + f * 0.1)

    # Person 2 should now be completely purged from active tracks
    assert len(tracks) == 1
    assert tracks[0].track_id == p1_id
    assert tracker.active_track_count == 1
    assert tracker.get_track(p2_id) is None


def test_primary_user_selection():
    """Test that the centered, larger candidate is elected primary user over background onlooker."""
    estimator = PrimaryUserEstimator(
        temp_absence_threshold_sec=2.0,
        absent_threshold_sec=5.0,
        center_weight=0.5,
        proximity_weight=0.3,
        persistence_weight=0.2
    )
    frame_shape = (480, 640)

    # Candidate A: Near the center (320, 240), large bbox (workstation primary user)
    track_center = TrackedPerson(
        track_id=1,
        bbox=[160.0, 80.0, 480.0, 440.0],
        confidence=0.92,
        center=(320.0, 260.0),
        proximity_score=0.85,
        time_first_seen=1.0,
        time_last_seen=2.0,
        duration_visible=1.0
    )

    # Candidate B: In the corner (550, 80), small bbox (passerby/onlooker)
    track_corner = TrackedPerson(
        track_id=2,
        bbox=[520.0, 60.0, 600.0, 240.0],
        confidence=0.80,
        center=(560.0, 150.0),
        proximity_score=0.20,
        time_first_seen=1.0,
        time_last_seen=2.0,
        duration_visible=1.0
    )

    ctx = estimator.update([track_center, track_corner], frame_shape, timestamp=2.0)
    assert ctx.primary_user_present is True
    assert ctx.primary_track_id == 1
    assert ctx.presence_state == PresenceState.ACTIVE.value
    assert ctx.secondary_people_count == 1


def test_temporary_absence():
    """Test state machine transition from ACTIVE -> TEMPORARILY_ABSENT upon brief disappearance."""
    estimator = PrimaryUserEstimator(
        temp_absence_threshold_sec=2.0,
        absent_threshold_sec=6.0
    )
    frame_shape = (480, 640)

    track_user = TrackedPerson(
        track_id=1,
        bbox=[200.0, 100.0, 440.0, 440.0],
        confidence=0.90,
        center=(320.0, 270.0),
        proximity_score=0.8,
        time_first_seen=10.0,
        time_last_seen=12.0,
        duration_visible=2.0
    )

    # Step 1: User is present
    ctx1 = estimator.update([track_user], frame_shape, timestamp=12.0)
    assert ctx1.presence_state == PresenceState.ACTIVE.value
    assert ctx1.primary_user_present is True

    # Step 2: User briefly steps out of frame (1.0s elapsed < 2.0s threshold)
    ctx2 = estimator.update([], frame_shape, timestamp=13.0)
    assert ctx2.presence_state == PresenceState.TEMPORARILY_ABSENT.value
    assert ctx2.primary_user_present is False
    assert ctx2.user_absent_duration == pytest.approx(1.0, rel=0.1)

    # Step 3: User returns before absent_threshold_sec triggers
    ctx3 = estimator.update([track_user], frame_shape, timestamp=13.5)
    assert ctx3.presence_state == PresenceState.ACTIVE.value
    assert ctx3.primary_user_present is True


def test_prolonged_absence():
    """Test state machine transition from TEMPORARILY_ABSENT -> ABSENT when absent > threshold."""
    estimator = PrimaryUserEstimator(
        temp_absence_threshold_sec=2.0,
        absent_threshold_sec=5.0
    )
    frame_shape = (480, 640)

    track_user = TrackedPerson(
        track_id=1,
        bbox=[200.0, 100.0, 440.0, 440.0],
        confidence=0.90,
        center=(320.0, 270.0),
        proximity_score=0.8,
        time_first_seen=1.0,
        time_last_seen=5.0,
        duration_visible=4.0
    )

    # User active at t=5.0
    estimator.update([track_user], frame_shape, timestamp=5.0)

    # User missing at t=7.0 (2.0s elapsed -> TEMPORARILY_ABSENT)
    ctx_temp = estimator.update([], frame_shape, timestamp=7.0)
    assert ctx_temp.presence_state == PresenceState.TEMPORARILY_ABSENT.value

    # User missing at t=11.0 (6.0s elapsed > 5.0s absent threshold -> ABSENT)
    ctx_absent = estimator.update([], frame_shape, timestamp=11.0)
    assert ctx_absent.presence_state == PresenceState.ABSENT.value
    assert ctx_absent.primary_user_present is False
    assert ctx_absent.primary_track_id is None
    assert ctx_absent.user_absent_duration >= 5.0


def test_multiple_people():
    """Test workspace context calculations with multiple simultaneous people."""
    estimator = PrimaryUserEstimator()
    frame_shape = (480, 640)

    t1 = TrackedPerson(
        track_id=1,
        bbox=[180.0, 100.0, 420.0, 440.0],
        confidence=0.93,
        center=(300.0, 270.0),
        proximity_score=0.85,
        time_first_seen=1.0,
        time_last_seen=5.0,
        duration_visible=4.0
    )

    t2 = TrackedPerson(
        track_id=2,
        bbox=[480.0, 120.0, 620.0, 400.0],
        confidence=0.85,
        center=(550.0, 260.0),
        proximity_score=0.45,
        time_first_seen=3.0,
        time_last_seen=5.0,
        duration_visible=2.0
    )

    ctx = estimator.update([t1, t2], frame_shape, timestamp=5.0)
    assert ctx.people_count == 2
    assert ctx.secondary_people_count == 1
    assert ctx.primary_track_id == 1
    assert ctx.nearest_secondary_distance is not None
    assert ctx.nearest_secondary_distance > 0.0
    assert ctx.secondary_presence_duration == 2.0


def test_two_person_replacement_and_reassignment():
    """TEST H: Primary leaves, secondary remains. Verify absence and reassignment without face recognition."""
    estimator = PrimaryUserEstimator(
        temp_absence_threshold_sec=2.0,
        absent_threshold_sec=5.0,
        center_weight=0.5,
        proximity_weight=0.3,
        persistence_weight=0.2
    )
    frame_shape = (480, 640)

    # Initial state: User 1 (Center) and User 2 (Corner)
    u1 = TrackedPerson(
        track_id=1,
        bbox=[180.0, 100.0, 420.0, 440.0],
        confidence=0.92,
        center=(320.0, 270.0),
        proximity_score=0.85,
        time_first_seen=1.0,
        time_last_seen=5.0,
        duration_visible=4.0
    )
    u2 = TrackedPerson(
        track_id=2,
        bbox=[480.0, 120.0, 620.0, 400.0],
        confidence=0.85,
        center=(550.0, 260.0),
        proximity_score=0.45,
        time_first_seen=3.0,
        time_last_seen=5.0,
        duration_visible=2.0
    )

    ctx_both = estimator.update([u1, u2], frame_shape, timestamp=5.0)
    assert ctx_both.primary_track_id == 1
    assert ctx_both.secondary_people_count == 1

    # User 1 leaves, User 2 moves into workstation center
    u2_centered = TrackedPerson(
        track_id=2,
        bbox=[200.0, 100.0, 440.0, 440.0],
        confidence=0.91,
        center=(320.0, 270.0),
        proximity_score=0.82,
        time_first_seen=3.0,
        time_last_seen=8.0,
        duration_visible=5.0
    )

    # During brief absence of u1 (t=8.0, 3s after t=5.0), u1 is temporarily absent
    # Since u1 is locked but absent, and absent_threshold has not expired:
    # u2 is still secondary until u1 is declared ABSENT
    ctx_interim = estimator.update([u2_centered], frame_shape, timestamp=8.0)
    assert ctx_interim.presence_state == PresenceState.TEMPORARILY_ABSENT.value

    # At t=11.0 (6s after u1 was last seen > 5.0s absent_threshold_sec):
    # u1 is declared ABSENT, clearing primary lock.
    ctx_expired = estimator.update([], frame_shape, timestamp=10.5)
    assert ctx_expired.presence_state == PresenceState.ABSENT.value

    # Now u2_centered is elected as the new primary user purely via spatial heuristics!
    ctx_reassigned = estimator.update([u2_centered], frame_shape, timestamp=11.0)
    assert ctx_reassigned.presence_state == PresenceState.ACTIVE.value
    assert ctx_reassigned.primary_track_id == 2
    assert ctx_reassigned.secondary_people_count == 0

