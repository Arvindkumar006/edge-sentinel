import time
import math
from typing import List, Optional, Tuple
from edge_sentinel.tracking.tracker import TrackedPerson
from edge_sentinel.context.schema import PresenceState, WorkspaceContext

class PrimaryUserEstimator:
    """
    Estimates primary user presence via spatial and temporal signals (no facial recognition).
    Manages the workstation presence state machine:
    ACTIVE -> TEMPORARILY_ABSENT -> ABSENT -> ACTIVE.
    """

    def __init__(
        self,
        temp_absence_threshold_sec: float = 2.5,
        absent_threshold_sec: float = 7.0,
        center_weight: float = 0.40,
        proximity_weight: float = 0.35,
        persistence_weight: float = 0.25
    ):
        self.temp_absence_threshold_sec = temp_absence_threshold_sec
        self.absent_threshold_sec = absent_threshold_sec
        self.center_weight = center_weight
        self.proximity_weight = proximity_weight
        self.persistence_weight = persistence_weight

        # State machine variables
        self._current_state = PresenceState.ABSENT
        self._primary_track_id: Optional[int] = None
        self._last_primary_seen_time: Optional[float] = None
        self._primary_first_seen_time: Optional[float] = None
        self._absence_start_time: Optional[float] = None

    def reset(self) -> None:
        """Resets presence state machine."""
        self._current_state = PresenceState.ABSENT
        self._primary_track_id = None
        self._last_primary_seen_time = None
        self._primary_first_seen_time = None
        self._absence_start_time = None

    def update(
        self,
        tracks: List[TrackedPerson],
        frame_shape: Tuple[int, int],
        timestamp: Optional[float] = None
    ) -> WorkspaceContext:
        """
        Updates the presence state machine given the active tracks in the current frame.
        """
        now = timestamp if timestamp is not None else time.perf_counter()
        frame_h, frame_w = frame_shape[:2]
        frame_center = (frame_w / 2.0, frame_h / 2.0)
        max_dist_from_center = math.hypot(frame_w / 2.0, frame_h / 2.0)

        # 1. Identify primary user candidate
        candidate_primary = self._select_primary_track(tracks, frame_center, max_dist_from_center)

        # 2. Update state machine
        if candidate_primary is not None:
            self._primary_track_id = candidate_primary.track_id
            self._last_primary_seen_time = now
            if self._primary_first_seen_time is None or self._current_state == PresenceState.ABSENT:
                self._primary_first_seen_time = candidate_primary.time_first_seen

            self._absence_start_time = None
            self._current_state = PresenceState.ACTIVE
            user_absent_dur = 0.0
            primary_presence_dur = max(0.0, now - self._primary_first_seen_time)
        else:
            # Primary user is NOT visible in this frame
            if self._absence_start_time is None:
                self._absence_start_time = self._last_primary_seen_time if self._last_primary_seen_time is not None else now

            user_absent_dur = max(0.0, now - self._absence_start_time)
            primary_presence_dur = 0.0

            if user_absent_dur >= self.absent_threshold_sec:
                self._current_state = PresenceState.ABSENT
                self._primary_track_id = None
                self._primary_first_seen_time = None
            else:
                self._current_state = PresenceState.TEMPORARILY_ABSENT

        user_present = (self._current_state == PresenceState.ACTIVE)

        # 3. Analyze secondary people
        secondary_tracks = [t for t in tracks if t.track_id != self._primary_track_id]
        secondary_count = len(secondary_tracks)

        nearest_secondary_dist: Optional[float] = None
        secondary_duration = 0.0

        if secondary_tracks:
            secondary_duration = max(t.duration_visible for t in secondary_tracks)

            if candidate_primary is not None:
                dists = [
                    math.hypot(t.center[0] - candidate_primary.center[0], t.center[1] - candidate_primary.center[1])
                    for t in secondary_tracks
                ]
                nearest_secondary_dist = min(dists) / max(1.0, max_dist_from_center)
            else:
                dists = [
                    math.hypot(t.center[0] - frame_center[0], t.center[1] - frame_center[1])
                    for t in secondary_tracks
                ]
                nearest_secondary_dist = min(dists) / max(1.0, max_dist_from_center)

        active_tids = [t.track_id for t in tracks]

        return WorkspaceContext(
            primary_user_present=user_present,
            primary_track_id=self._primary_track_id if user_present else None,
            presence_state=self._current_state.value,
            people_count=len(tracks),
            secondary_people_count=secondary_count,
            nearest_secondary_distance=nearest_secondary_dist,
            secondary_presence_duration=secondary_duration,
            user_absent_duration=user_absent_dur,
            primary_presence_duration=primary_presence_dur,
            active_track_ids=active_tids,
            timestamp=now
        )

    def _select_primary_track(
        self,
        tracks: List[TrackedPerson],
        frame_center: Tuple[float, float],
        max_dist: float
    ) -> Optional[TrackedPerson]:
        """
        Heuristic spatial/temporal scoring to select the primary workstation user.
        """
        if not tracks:
            return None

        # If previous primary track is established:
        if self._primary_track_id is not None:
            for t in tracks:
                if t.track_id == self._primary_track_id:
                    return t
            # If primary user is missing but we have not reached full ABSENT state,
            # do not immediately reassign primary status to a secondary person.
            if self._current_state != PresenceState.ABSENT:
                return None

        best_score = -1.0
        best_track: Optional[TrackedPerson] = None

        for t in tracks:
            # Centeredness signal
            dist_to_center = math.hypot(t.center[0] - frame_center[0], t.center[1] - frame_center[1])
            center_score = max(0.0, 1.0 - (dist_to_center / max(1.0, max_dist)))

            # Proximity signal
            proximity_score = min(1.0, t.proximity_score)

            # Persistence signal
            persistence_score = min(1.0, t.duration_visible / 5.0)

            total_score = (
                self.center_weight * center_score +
                self.proximity_weight * proximity_score +
                self.persistence_weight * persistence_score
            )

            if total_score > best_score:
                best_score = total_score
                best_track = t

        return best_track

    @property
    def current_state(self) -> PresenceState:
        return self._current_state
