import time
import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

@dataclass
class TrackedPerson:
    """Represents a tracked person across consecutive frames."""
    track_id: int
    bbox: List[float]               # [x1, y1, x2, y2]
    confidence: float
    center: Tuple[float, float]     # (cx, cy)
    proximity_score: float          # 0.0 to 1.0 (normalized bounding box scale)
    time_first_seen: float
    time_last_seen: float
    duration_visible: float
    missing_frames: int = 0
    hits: int = 1
    center_history: List[Tuple[float, float, float]] = field(default_factory=list)

    @property
    def width(self) -> float:
        return max(1.0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return max(1.0, self.bbox[3] - self.bbox[1])

    @property
    def area(self) -> float:
        return self.width * self.height


def compute_iou(boxA: List[float], boxB: List[float]) -> float:
    """Computes Intersection over Union between two boxes [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h

    areaA = max(1.0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    areaB = max(1.0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    union_area = areaA + areaB - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def compute_centroid_distance(centerA: Tuple[float, float], centerB: Tuple[float, float]) -> float:
    """Computes Euclidean distance between two 2D points."""
    return math.hypot(centerA[0] - centerB[0], centerA[1] - centerB[1])


class PersonTracker:
    """
    Lightweight, robust IoU and centroid association tracker for people.
    Maintains persistent track IDs, longevity metrics, and proximity estimates.
    """

    def __init__(
        self,
        max_missing_frames: int = 20,
        iou_match_threshold: float = 0.30,
        dist_match_threshold: float = 120.0
    ):
        self.max_missing_frames = max_missing_frames
        self.iou_match_threshold = iou_match_threshold
        self.dist_match_threshold = dist_match_threshold

        self._next_track_id = 1
        self._active_tracks: Dict[int, TrackedPerson] = {}

    def reset(self) -> None:
        """Resets all track history."""
        self._next_track_id = 1
        self._active_tracks.clear()

    def update(
        self,
        detections: List[Dict[str, Any]],
        frame_shape: Tuple[int, int],
        timestamp: Optional[float] = None
    ) -> List[TrackedPerson]:
        """
        Associates incoming detections with existing tracks.

        Args:
            detections: List of detection dicts with 'bbox' and 'confidence'
            frame_shape: (height, width) of camera frame for normalization
            timestamp: Current frame timestamp in seconds (default time.perf_counter())

        Returns:
            List of currently visible TrackedPerson instances in this frame.
        """
        now = timestamp if timestamp is not None else time.perf_counter()
        frame_h, frame_w = frame_shape[:2]
        frame_area = max(1.0, float(frame_w * frame_h))

        # Parse candidate detections
        candidates = []
        for det in detections:
            bbox = [float(v) for v in det.get("bbox", [0, 0, 0, 0])]
            conf = float(det.get("confidence", 0.0))
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            box_area = max(1.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
            proximity = min(1.0, math.sqrt(box_area / frame_area))  # 0.0 to ~1.0 scale
            candidates.append({
                "bbox": bbox,
                "confidence": conf,
                "center": (cx, cy),
                "proximity": proximity
            })

        track_ids = list(self._active_tracks.keys())
        matched_tracks = set()
        matched_candidates = set()

        if track_ids and candidates:
            # 1. First Pass: Match by IoU (greedy assignment from highest IoU)
            cost_matrix = []
            for t_idx, tid in enumerate(track_ids):
                t = self._active_tracks[tid]
                row = []
                for c_idx, c in enumerate(candidates):
                    iou = compute_iou(t.bbox, c["bbox"])
                    row.append(iou)
                cost_matrix.append(row)

            # Assign matches where IoU >= iou_match_threshold
            pairs = []
            for t_idx in range(len(track_ids)):
                for c_idx in range(len(candidates)):
                    score = cost_matrix[t_idx][c_idx]
                    if score >= self.iou_match_threshold:
                        pairs.append((score, t_idx, c_idx))

            # Sort descending by IoU
            pairs.sort(key=lambda x: x[0], reverse=True)
            for score, t_idx, c_idx in pairs:
                tid = track_ids[t_idx]
                if tid not in matched_tracks and c_idx not in matched_candidates:
                    matched_tracks.add(tid)
                    matched_candidates.add(c_idx)
                    self._update_track(tid, candidates[c_idx], now)

            # 2. Second Pass: Centroid distance matching for remaining unmatched detections
            remaining_tids = [tid for tid in track_ids if tid not in matched_tracks]
            remaining_c_indices = [idx for idx in range(len(candidates)) if idx not in matched_candidates]

            dist_pairs = []
            for tid in remaining_tids:
                t = self._active_tracks[tid]
                for c_idx in remaining_c_indices:
                    dist = compute_centroid_distance(t.center, candidates[c_idx]["center"])
                    if dist <= self.dist_match_threshold:
                        dist_pairs.append((dist, tid, c_idx))

            dist_pairs.sort(key=lambda x: x[0])  # ascending distance
            for dist, tid, c_idx in dist_pairs:
                if tid not in matched_tracks and c_idx not in matched_candidates:
                    matched_tracks.add(tid)
                    matched_candidates.add(c_idx)
                    self._update_track(tid, candidates[c_idx], now)

        # 3. Handle unmatched detections: create new tracks
        for c_idx, cand in enumerate(candidates):
            if c_idx not in matched_candidates:
                new_id = self._next_track_id
                self._next_track_id += 1
                self._active_tracks[new_id] = TrackedPerson(
                    track_id=new_id,
                    bbox=cand["bbox"],
                    confidence=cand["confidence"],
                    center=cand["center"],
                    proximity_score=cand["proximity"],
                    time_first_seen=now,
                    time_last_seen=now,
                    duration_visible=0.0,
                    missing_frames=0,
                    hits=1,
                    center_history=[(cand["center"][0], cand["center"][1], now)]
                )
                matched_tracks.add(new_id)

        # 4. Handle unmatched tracks: increment missing frames or expire
        expired_ids = []
        for tid, track in self._active_tracks.items():
            if tid not in matched_tracks:
                track.missing_frames += 1
                if track.missing_frames > self.max_missing_frames:
                    expired_ids.append(tid)

        for tid in expired_ids:
            del self._active_tracks[tid]

        # Return only tracks that were actively detected in this frame
        visible_tracks = [
            track for tid, track in self._active_tracks.items()
            if track.missing_frames == 0
        ]
        # Sort by track_id for determinism
        visible_tracks.sort(key=lambda x: x.track_id)
        return visible_tracks

    def _update_track(self, tid: int, cand: Dict[str, Any], now: float) -> None:
        """Updates an existing track with new detection data."""
        track = self._active_tracks[tid]
        # Exponential smoothing for center to reduce jitter
        smooth_cx = 0.75 * cand["center"][0] + 0.25 * track.center[0]
        smooth_cy = 0.75 * cand["center"][1] + 0.25 * track.center[1]

        track.bbox = cand["bbox"]
        track.confidence = cand["confidence"]
        track.center = (smooth_cx, smooth_cy)
        track.proximity_score = cand["proximity"]
        track.time_last_seen = now
        track.duration_visible = max(0.0, now - track.time_first_seen)
        track.missing_frames = 0
        track.hits += 1
        track.center_history.append((smooth_cx, smooth_cy, now))
        if len(track.center_history) > 30:
            track.center_history.pop(0)

    @property
    def active_track_count(self) -> int:
        return len(self._active_tracks)

    def get_track(self, track_id: int) -> Optional[TrackedPerson]:
        return self._active_tracks.get(track_id)
